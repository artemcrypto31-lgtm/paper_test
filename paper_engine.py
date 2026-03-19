import asyncio
import math
import sqlite3
import os
import signal
import sys
import numpy as np
import telebot
import threading
import queue
import logging
import pandas as pd
import ccxt
import time
import warnings
from telebot import types
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore', category=FutureWarning)

# =====================================================================
# ЛОГИРОВАНИЕ
# =====================================================================
def _setup_logging():
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)
    from logging.handlers import RotatingFileHandler
    fh = RotatingFileHandler('fortress.log', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    fh.setFormatter(fmt)
    root.addHandler(fh)

_setup_logging()
log = logging.getLogger('fortress')

# =====================================================================
# КОНФИГУРАЦИЯ & КОНСТАНТЫ
# =====================================================================
load_dotenv()
bot     = telebot.TeleBot(os.getenv('BOT_TOKEN'))
USER_ID = int(os.getenv('USER_ID'))

STARTING_BALANCE  = 1000.0
MIN_VOLUME_24H    = 120_000_000.0
MIN_NATR          = 1.5       
ADX_1H_THRESHOLD  = 30        
ADX_15M_THRESHOLD = 20        
RISK_PER_TRADE    = 0.01      
MONITOR_INTERVAL  = 60        

# --- НОВЫЕ ЖЕСТКИЕ ПАРАМЕТРЫ РИСК-МЕНЕДЖМЕНТА ---
MAX_STOP_PERCENT = 5.0  # Максимально допустимый размер стопа в % (Отбраковка)
MIN_RR = 2.0            # Минимальное соотношение Risk:Reward
MAX_ATR_TP = 3.5        # Максимальная дистанция Тейка в ATR (защита от нереалистичных целей)
LOOKBACK_SWING = 8      # Свечей для поиска технического экстремума (Swing High/Low)
# ------------------------------------------------

GRID_1H_MINUTES   = 60
GRID_15M_MINUTES  = 15

FEE_RATE = 0.0005 

BLACKLIST = {'LINK/USDT:USDT', 'SOL/USDT:USDT', 'LTC/USDT:USDT', 'XMR/USDT:USDT', 'USDC/USDT:USDT'}

BTC_CORR_TIMEFRAME = '4h'
BTC_CORR_LIMIT     = 100
BTC_CORR_MAX       = 0.8

exchange = ccxt.binance({'options': {'defaultType': 'future'}, 'enableRateLimit': True})
thread_pool = ThreadPoolExecutor(max_workers=5)

# =====================================================================
# ИНДИКАТОРЫ & SMART MONEY
# =====================================================================
def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def _adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    atr14    = _atr(h, l, c, n)
    up       = h.diff()
    down     = -l.diff()
    dm_plus  = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=h.index)
    dm_minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=h.index)
    dip  = dm_plus.ewm(alpha=1/n, adjust=False).mean()  / atr14 * 100
    dim  = dm_minus.ewm(alpha=1/n, adjust=False).mean() / atr14 * 100
    dx   = ((dip - dim).abs() / (dip + dim).replace(0, np.nan) * 100).fillna(0)
    return dx.ewm(alpha=1/n, adjust=False).mean()

def is_liquidity_sweep(df: pd.DataFrame, side: str):
    for i in range(-3, 0):
        curr_l, curr_h, curr_c = df['l'].iloc[i], df['h'].iloc[i], df['c'].iloc[i]
        prev_l = df['l'].iloc[-18:i].min() if i < -1 else df['l'].iloc[-18:-1].min()
        prev_h = df['h'].iloc[-18:i].max() if i < -1 else df['h'].iloc[-18:-1].max()
        if side == 'long':
            if curr_l < prev_l and curr_c > prev_l: return True
        else:
            if curr_h > prev_h and curr_c < prev_h: return True
    return False

# =====================================================================
# БАЗА ДАННЫХ
# =====================================================================
_db_queue = queue.Queue()
DB_PATH   = 'paper_trading.db'

def db_worker():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('PRAGMA journal_mode=WAL')
    while True:
        try:
            task = _db_queue.get()
            if task is None: break
            queries, result_event, result_holder = task
            try:
                conn.execute('BEGIN TRANSACTION')
                last_id = None
                for sql, params in queries:
                    cur = conn.execute(sql, params)
                    last_id = cur.lastrowid
                conn.commit()
                if result_holder is not None: result_holder['lastrowid'] = last_id
            except Exception as e:
                conn.rollback()
                log.error(f'[DB] Транзакция отменена: {e}', exc_info=True)
            finally:
                if result_event: result_event.set()
        except Exception as e:
            log.error(f'[DB] Ошибка воркера: {e}', exc_info=True)

def _write_db_transaction(queries: list, sync=False):
    event = threading.Event() if sync else None
    holder = {} if sync else None
    _db_queue.put((queries, event, holder))
    if sync:
        event.wait()
        return holder.get('lastrowid')

_read_conn_local = threading.local()

def _read_db(sql: str, params: tuple = (), fetchone: bool = False):
    if not hasattr(_read_conn_local, 'conn') or _read_conn_local.conn is None:
        _read_conn_local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _read_conn_local.conn.execute('PRAGMA journal_mode=WAL')
    try:
        cur = _read_conn_local.conn.execute(sql, params)
        return cur.fetchone() if fetchone else cur.fetchall()
    except Exception:
        try: _read_conn_local.conn.close()
        except: pass
        _read_conn_local.conn = None
        raise

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('PRAGMA journal_mode=WAL')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS wallet (balance REAL)')
    if not c.execute('SELECT balance FROM wallet').fetchone():
        c.execute('INSERT INTO wallet VALUES (?)', (STARTING_BALANCE,))

    c.execute('''CREATE TABLE IF NOT EXISTS active_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT UNIQUE, side TEXT,
        entry_price REAL, size REAL, sl REAL, tp REAL, is_trailing INTEGER DEFAULT 0,
        trailing_sl REAL DEFAULT NULL, partial_price REAL DEFAULT NULL,
        entry_time TEXT, strategy TEXT
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS trade_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, side TEXT,
        entry_price REAL, exit_price REAL, size REAL, gross_pnl REAL, net_pnl REAL, fee REAL,
        result TEXT, entry_time TEXT, exit_time TEXT, strategy TEXT
    )''')
    conn.commit()
    conn.close()

# =====================================================================
# ЯДРО АНАЛИЗА
# =====================================================================
def fetch_btc_closes() -> pd.Series | None:
    try:
        df = pd.DataFrame(exchange.fetch_ohlcv('BTC/USDT:USDT', BTC_CORR_TIMEFRAME, limit=BTC_CORR_LIMIT + 5), columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        return df['c'].reset_index(drop=True).iloc[-BTC_CORR_LIMIT:]
    except: return None

def is_correlated_with_btc(symbol: str, btc_closes: pd.Series) -> bool:
    try:
        df = pd.DataFrame(exchange.fetch_ohlcv(symbol, BTC_CORR_TIMEFRAME, limit=BTC_CORR_LIMIT + 5), columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        closes = df['c'].reset_index(drop=True).iloc[-BTC_CORR_LIMIT:]
        if len(closes) < 10 or len(btc_closes) < 10: return False
        sym_ret = closes.pct_change().dropna()
        btc_ret = btc_closes.pct_change().dropna()
        n = min(len(sym_ret), len(btc_ret))
        if sym_ret.iloc[-n:].std() == 0 or btc_ret.iloc[-n:].std() == 0: return False
        correlation = float(np.corrcoef(sym_ret.iloc[-n:].values, btc_ret.iloc[-n:].values)[0, 1])
        return abs(correlation) >= BTC_CORR_MAX
    except: return False

def analyze_1h(symbol: str, btc_closes: pd.Series | None = None):
    if btc_closes is not None and is_correlated_with_btc(symbol, btc_closes): return None

    df4h = pd.DataFrame(exchange.fetch_ohlcv(symbol, '4h', limit=200), columns=['ts', 'o', 'h', 'l', 'c', 'v'])
    ema50_4h = _ema(df4h['c'], 50).iloc[-2]
    if pd.isna(ema50_4h): return None
    above_ema50 = df4h['c'].iloc[-2] > ema50_4h

    try:
        adx_val = _adx(df4h['h'], df4h['l'], df4h['c'], 14).iloc[-2]
        if not pd.isna(adx_val) and adx_val < ADX_1H_THRESHOLD: return None
    except: pass

    df1h = pd.DataFrame(exchange.fetch_ohlcv(symbol, '1h', limit=60), columns=['ts', 'o', 'h', 'l', 'c', 'v'])
    c = df1h['c']; h = df1h['h']; l = df1h['l']; v = df1h['v']
    atr_val = _atr(h, l, c, 14).iloc[-2]
    e12 = _ema(c, 12).iloc[-2]; e26 = _ema(c, 26).iloc[-2]

    if pd.isna(atr_val) or pd.isna(e12) or pd.isna(e26) or atr_val <= 0: return None
    if (atr_val / c.iloc[-2]) * 100 < MIN_NATR: return None

    if above_ema50 and e12 > e26: 
        side = 'long'
        extremum = l.iloc[-LOOKBACK_SWING-1:-1].min() # Ищем локальное дно
    elif not above_ema50 and e12 < e26: 
        side = 'short'
        extremum = h.iloc[-LOOKBACK_SWING-1:-1].max() # Ищем локальный хай
    else: return None

    vol_ma = v.rolling(20).mean()
    if sum([v.iloc[-2] > v.iloc[-3] > v.iloc[-4], 
            (c.iloc[-2] > c.iloc[-3]) if side=='long' else (c.iloc[-2] < c.iloc[-3]), 
            v.iloc[-2] > vol_ma.iloc[-2] * 1.5]) < 2: return None

    return {'symbol': symbol, 'price': c.iloc[-2], 'atr': atr_val, 'side': side, 'strategy': 'ema_1h', 'extremum': extremum}

def analyze_15m(symbol: str, btc_closes: pd.Series | None = None):
    if btc_closes is not None and is_correlated_with_btc(symbol, btc_closes): return None

    df4h = pd.DataFrame(exchange.fetch_ohlcv(symbol, '4h', limit=100), columns=['ts', 'o', 'h', 'l', 'c', 'v'])
    try:
        adx_val = _adx(df4h['h'], df4h['l'], df4h['c'], 14).iloc[-2]
        if not pd.isna(adx_val) and adx_val < ADX_15M_THRESHOLD: return None
    except: pass

    df15m = pd.DataFrame(exchange.fetch_ohlcv(symbol, '15m', limit=60), columns=['ts', 'o', 'h', 'l', 'c', 'v'])
    c = df15m['c']; h = df15m['h']; l = df15m['l']
    
    atr_val = _atr(h, l, c, 14).iloc[-1]
    e12_15 = _ema(c, 12).iloc[-1]
    e26_15 = _ema(c, 26).iloc[-1]

    if pd.isna(atr_val) or pd.isna(e12_15) or pd.isna(e26_15) or atr_val <= 0: return None
    if (atr_val / c.iloc[-1]) * 100 < (MIN_NATR * 0.5): return None

    if e12_15 > e26_15:
        trend_side = 'long'
        extremum = l.iloc[-LOOKBACK_SWING:].min()
    else:
        trend_side = 'short'
        extremum = h.iloc[-LOOKBACK_SWING:].max()

    if not is_liquidity_sweep(df15m, trend_side):
        return None

    return {'symbol': symbol, 'price': c.iloc[-1], 'atr': atr_val, 'side': trend_side, 'strategy': 'smart_15m', 'extremum': extremum}

def open_trade(setup: dict) -> bool:
    symbol = setup['symbol']
    price = setup['price']
    atr = setup['atr']
    extremum = setup['extremum']
    side = setup['side']

    if _read_db('SELECT 1 FROM active_trades WHERE symbol=?', (symbol,), fetchone=True):
        return False

    # --- МАТЕМАТИЧЕСКИЙ ФИЛЬТР (ЯДРО ЗАЩИТЫ) ---
    if side == 'long':
        sl = extremum - (atr * 1.5)
        sl_dist = price - sl
    else:
        sl = extremum + (atr * 1.5)
        sl_dist = sl - price

    # Защита от кривых экстремумов (если цена уже ушла за него)
    if sl_dist <= price * 0.001: 
        return False

    risk_pct = (sl_dist / price) * 100
    
    # ФИЛЬТР 1: Отбраковка огромных стопов
    if risk_pct > MAX_STOP_PERCENT:
        log.info(f'[{symbol}] Сделка отклонена: Стоп-лосс {risk_pct:.2f}% превышает лимит {MAX_STOP_PERCENT}%')
        return False

    tp_dist = sl_dist * MIN_RR
    
    # ФИЛЬТР 2: Отбраковка нереалистичных тейков
    if tp_dist > (atr * MAX_ATR_TP):
        log.info(f'[{symbol}] Сделка отклонена: Тейк требует нереалистичного движения (>{MAX_ATR_TP} ATR)')
        return False

    tp = price + tp_dist if side == 'long' else price - tp_dist
    # ---------------------------------------------

    balance = _read_db('SELECT balance FROM wallet', fetchone=True)[0]
    trades  = _read_db('SELECT symbol,side,entry_price,size FROM active_trades')
    unrealized = 0.0
    for sym, sd, ep, sz in trades:
        try:
            p = exchange.fetch_ticker(sym)['last']
            unrealized += (p - ep) * sz if sd == 'long' else (ep - p) * sz
        except: pass
    
    equity = balance + unrealized
    size = (equity * RISK_PER_TRADE) / sl_dist

    try:
        # Пишем в базу РАССЧИТАННЫЙ tp вместо NULL
        _write_db_transaction([
            ('INSERT INTO active_trades (symbol,side,entry_price,size,sl,tp,entry_time,strategy) VALUES (?,?,?,?,?,?,?,?)',
             (symbol, side, price, size, sl, tp, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), setup['strategy']))
        ], sync=True)
    except Exception as e:
        log.error(f'Ошибка входа {symbol}: {e}')
        return False

    icon = '⏱️' if '15m' in setup['strategy'] else '🕰️'
    bot.send_message(USER_ID, 
        f'{icon} *ВХОД ({setup["strategy"]}): {symbol}*\n'
        f'Тип: *{side.upper()}* | Цена: `{price:.6g}`\n'
        f'Риск сделки: `{risk_pct:.2f}%` от точки входа\n'
        f'🔴 SL: `{sl:.6g}` | 🎯 TP1: `{tp:.6g}`', 
        parse_mode='Markdown')
    return True

def close_trade(trade: tuple, exit_price: float, reason: str):
    (tid, symbol, side, entry_price, size, sl, tp, is_trailing, trailing_sl, partial_price, entry_time, strategy) = trade

    gross_pnl = (exit_price - entry_price) * size if side == 'long' else (entry_price - exit_price) * size
    fee_entry = entry_price * size * FEE_RATE
    fee_exit  = exit_price * size * FEE_RATE
    total_fee = fee_entry + fee_exit
    net_pnl   = gross_pnl - total_fee
    result = 'WIN' if net_pnl > 0 else ('LOSS' if net_pnl < 0 else 'BREAKEVEN')

    queries = [
        ('UPDATE wallet SET balance = balance + ?', (net_pnl,)),
        ('DELETE FROM active_trades WHERE id=?', (tid,)),
        ('INSERT INTO trade_log (symbol,side,entry_price,exit_price,size,gross_pnl,net_pnl,fee,result,entry_time,exit_time,strategy) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
         (symbol, side, entry_price, exit_price, size, gross_pnl, net_pnl, total_fee, result, entry_time, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), strategy))
    ]
    _write_db_transaction(queries, sync=True)

    new_balance = _read_db('SELECT balance FROM wallet', fetchone=True)[0]
    icon = '🏆' if result == 'WIN' else '💀'
    
    # Защита от сбоев Telegram API (экранирование)
    safe_strategy = strategy.replace('_', '\\_')
    
    msg_text = (f'{icon} *ЗАКРЫТО: {symbol}* ({safe_strategy})\n'
                f'{reason} | {side.upper()} → `{exit_price:.6g}`\n'
                f'Чистый PnL: `{net_pnl:+.2f}` USDT (Комиссия: `{total_fee:.2f}`)\n'
                f'Баланс: `{new_balance:.2f}` USDT')
    
    try:
        bot.send_message(USER_ID, msg_text, parse_mode='Markdown')
    except Exception as e:
        log.error(f'[UI Error] Сбой отправки Markdown для {symbol}: {e}')
        # Отправка сырого текста, если Markdown выдал ошибку
        clean_text = msg_text.replace('*', '').replace('`', '').replace('\\_', '_')
        bot.send_message(USER_ID, clean_text)

# =====================================================================
# АСИНХРОННЫЕ ВОРКЕРЫ
# =====================================================================
def seconds_until_next_grid(interval_minutes: int) -> float:
    now = datetime.now()
    minutes = now.minute + now.second / 60.0
    next_grid = math.ceil(minutes / interval_minutes) * interval_minutes
    wait_min = next_grid - minutes
    if wait_min < 0.1: wait_min += interval_minutes
    return max(wait_min * 60, 5.0)

async def monitor_trades():
    while True:
        try:
            trades = _read_db('SELECT id,symbol,side,entry_price,size,sl,tp,is_trailing,trailing_sl,partial_price,entry_time,strategy FROM active_trades')
            for trade in trades:
                (tid, symbol, side, entry_price, size, sl, tp, is_trailing, trailing_sl, partial_price, entry_time, strategy) = trade
                try:
                    ticker = await asyncio.to_thread(exchange.fetch_ticker, symbol)
                    price = ticker['last']
                    candles = await asyncio.to_thread(exchange.fetch_ohlcv, symbol, '1m', limit=2)
                    candle_low, candle_high = candles[-2][3], candles[-2][2]

                    if side == 'long':
                        sl_hit = price <= sl or candle_low <= sl
                        tp1_hit = price >= tp or candle_high >= tp # ИСПОЛЬЗУЕМ TP ИЗ БАЗЫ
                        
                        if not is_trailing:
                            if sl_hit: close_trade(trade, min(price, candle_low), '🔴 STOP-LOSS')
                            elif tp1_hit:
                                close_price = max(price, candle_high)
                                partial_size = size * 0.5
                                net_pnl = ((close_price - entry_price) * partial_size) - ((entry_price * partial_size * FEE_RATE) + (close_price * partial_size * FEE_RATE))
                                queries = [
                                    ('UPDATE wallet SET balance = balance + ?', (net_pnl,)),
                                    ('UPDATE active_trades SET size=?,is_trailing=1,trailing_sl=?,partial_price=? WHERE id=?', (partial_size, entry_price, close_price, tid))
                                ]
                                _write_db_transaction(queries)
                                await asyncio.to_thread(bot.send_message, USER_ID, f'🎯 *50% ТЕЙК: {symbol}* | PnL: `+{net_pnl:.2f}`', parse_mode='Markdown')
                        else:
                            tf = '15m' if '15m' in strategy else '1h'
                            df_tr = pd.DataFrame(await asyncio.to_thread(exchange.fetch_ohlcv, symbol, tf, limit=20), columns=['ts', 'o', 'h', 'l', 'c', 'v'])
                            atr = _atr(df_tr['h'], df_tr['l'], df_tr['c'], 14).iloc[-1]
                            new_tsl = max(trailing_sl, price - atr * 1.0)
                            
                            if price <= trailing_sl or candle_low <= trailing_sl:
                                close_trade(trade, min(price, candle_low), '📍 ТРЕЙЛИНГ СТОП')
                            elif new_tsl > trailing_sl:
                                _write_db_transaction([('UPDATE active_trades SET trailing_sl=? WHERE id=?', (new_tsl,))])

                    else: 
                        sl_hit = price >= sl or candle_high >= sl
                        tp1_hit = price <= tp or candle_low <= tp # ИСПОЛЬЗУЕМ TP ИЗ БАЗЫ

                        if not is_trailing:
                            if sl_hit: close_trade(trade, max(price, candle_high), '🔴 STOP-LOSS')
                            elif tp1_hit:
                                close_price = min(price, candle_low)
                                partial_size = size * 0.5
                                net_pnl = ((entry_price - close_price) * partial_size) - ((entry_price * partial_size * FEE_RATE) + (close_price * partial_size * FEE_RATE))
                                queries = [
                                    ('UPDATE wallet SET balance = balance + ?', (net_pnl,)),
                                    ('UPDATE active_trades SET size=?,is_trailing=1,trailing_sl=?,partial_price=? WHERE id=?', (partial_size, close_price, close_price, tid))
                                ]
                                _write_db_transaction(queries)
                                await asyncio.to_thread(bot.send_message, USER_ID, f'🎯 *50% ТЕЙК: {symbol}* | PnL: `+{net_pnl:.2f}`', parse_mode='Markdown')
                        else:
                            tf = '15m' if '15m' in strategy else '1h'
                            df_tr = pd.DataFrame(await asyncio.to_thread(exchange.fetch_ohlcv, symbol, tf, limit=20), columns=['ts', 'o', 'h', 'l', 'c', 'v'])
                            atr = _atr(df_tr['h'], df_tr['l'], df_tr['c'], 14).iloc[-1]
                            new_tsl = min(trailing_sl, price + atr * 1.0)
                            
                            if price >= trailing_sl or candle_high >= trailing_sl:
                                close_trade(trade, max(price, candle_high), '📍 ТРЕЙЛИНГ СТОП')
                            elif new_tsl < trailing_sl:
                                _write_db_transaction([('UPDATE active_trades SET trailing_sl=? WHERE id=?', (new_tsl,))])
                except Exception as e:
                    log.error(f'[Monitor] Ошибка {symbol}: {e}')
        except Exception as e:
            log.error(f'[Monitor] Критическая ошибка: {e}', exc_info=True)
        await asyncio.sleep(MONITOR_INTERVAL)

async def hunter_1h():
    await asyncio.sleep(seconds_until_next_grid(GRID_1H_MINUTES))
    while True:
        try:
            log.info('[Scanner] Запуск 1H сканера')
            tickers = await asyncio.to_thread(exchange.fetch_tickers)
            candidates = [s for s, d in tickers.items() if '/USDT:USDT' in s and s not in BLACKLIST and (d.get('quoteVolume') or 0) >= MIN_VOLUME_24H]
            btc_closes = await asyncio.to_thread(fetch_btc_closes)

            for sym in candidates:
                try:
                    setup = await asyncio.to_thread(analyze_1h, sym, btc_closes)
                    if setup: open_trade(setup)
                except: pass
                await asyncio.sleep(0.2)
        except Exception as e: log.error(f'[1H Scanner] Ошибка: {e}')
        await asyncio.sleep(seconds_until_next_grid(GRID_1H_MINUTES))

async def hunter_15m():
    await asyncio.sleep(seconds_until_next_grid(GRID_15M_MINUTES))
    while True:
        try:
            log.info('[Scanner] Запуск 15m сканера (SMART)')
            tickers = await asyncio.to_thread(exchange.fetch_tickers)
            candidates = [s for s, d in tickers.items() if '/USDT:USDT' in s and s not in BLACKLIST and (d.get('quoteVolume') or 0) >= MIN_VOLUME_24H]
            btc_closes = await asyncio.to_thread(fetch_btc_closes)

            for sym in candidates:
                try:
                    setup = await asyncio.to_thread(analyze_15m, sym, btc_closes)
                    if setup: open_trade(setup)
                except: pass
                await asyncio.sleep(0.2)
        except Exception as e: log.error(f'[15m Scanner] Ошибка: {e}')
        await asyncio.sleep(seconds_until_next_grid(GRID_15M_MINUTES))

# =====================================================================
# TELEGRAM БОТ (Команды и Интерфейс)
# =====================================================================
def main_keyboard():
    m = types.ReplyKeyboardMarkup(resize_keyboard=True)
    m.row('💰 БАЛАНС', '⚔️ СДЕЛКИ')
    m.row('📈 СТАТИСТИКА', '🚪 ЗАКРЫТЬ ВСЕ')
    return m

@bot.message_handler(commands=['start'])
def cmd_start(msg):
    bot.send_message(USER_ID, '🛡️ *Fortress Dual-Core*\nТест: 1H (Тренд) против 15m (Smart Money).', parse_mode='Markdown', reply_markup=main_keyboard())

@bot.message_handler(func=lambda m: m.text == '💰 БАЛАНС')
def cmd_balance(msg):
    def _run():
        balance = _read_db('SELECT balance FROM wallet', fetchone=True)[0]
        unrealized = 0.0
        for sym, side, ep, sz in _read_db('SELECT symbol,side,entry_price,size FROM active_trades'):
            try:
                p = exchange.fetch_ticker(sym)['last']
                raw_pnl = (p - ep) * sz if side == 'long' else (ep - p) * sz
                unrealized += raw_pnl - (ep * sz * FEE_RATE) - (p * sz * FEE_RATE)
            except: pass
        bot.send_message(USER_ID, f'🏦 БАЛАНС: `{balance:.2f}`\nОткрытый PnL (с ком.): `{unrealized:+.2f}`\n💰 Equity: `{balance + unrealized:.2f}`', parse_mode='Markdown')
    thread_pool.submit(_run)

@bot.message_handler(func=lambda m: m.text == '⚔️ СДЕЛКИ')
def cmd_trades(msg):
    def _run():
        try:
            trades = _read_db('SELECT id,symbol,side,entry_price,sl,tp,is_trailing,trailing_sl,entry_time,size,strategy FROM active_trades')
            if not trades:
                bot.send_message(USER_ID, '⚔️ Активных сделок нет.')
                return
                
            text = '⚔️ *АКТИВНЫЕ СДЕЛКИ:*\n\n'
            markup = types.InlineKeyboardMarkup(row_width=2)
            buttons = []
            
            for tid, sym, side, ep, sl, tp, is_tr, tsl, et, sz, strategy in trades:
                t = et[5:16] if et else '—'
                icon = '⏱️' if '15m' in strategy else '🕰️'
                safe_strategy = strategy.replace('_', '\\_')
                
                try:
                    cur_price = exchange.fetch_ticker(sym)['last']
                    net_pnl = ((cur_price - ep) * sz if side == 'long' else (ep - cur_price) * sz) - ((ep * sz * FEE_RATE) + (cur_price * sz * FEE_RATE))
                    pnl_str = f'{"🟢" if net_pnl >= 0 else "🔴"} PnL: `{net_pnl:+.2f}`\n  📥 Вход: `{ep:.6g}` | Текущая: `{cur_price:.6g}`'
                except: pnl_str = f'📥 Вход: `{ep:.6g}` | Текущая: `ошибка API`'

                if is_tr:
                    mode = f'\n  📍 T-SL: `{tsl:.6g}`\n  🔴 SL: `{sl:.6g}`'
                else:
                    mode = f'\n  🎯 TP1: `{tp:.6g}`\n  🔴 SL: `{sl:.6g}`'
                    
                text += f'{icon} *{sym}* ({safe_strategy}) | {side.upper()} | {t}\n  {pnl_str}{mode}\n\n'
                
                clean_sym = sym.replace('/USDT:USDT', '')
                buttons.append(types.InlineKeyboardButton(f'✖️ {clean_sym}', callback_data=f'kill_{tid}'))
                
            markup.add(*buttons)
            
            try: bot.send_message(USER_ID, text, parse_mode='Markdown', reply_markup=markup)
            except:
                clean_text = text.replace('*', '').replace('`', '').replace('\\_', '_')
                bot.send_message(USER_ID, clean_text, reply_markup=markup)
        except Exception as e:
            log.error(f'[UI Error] Ошибка кнопки сделок: {e}', exc_info=True)
    thread_pool.submit(_run)

@bot.callback_query_handler(func=lambda call: call.data.startswith('kill_'))
def callback_kill_trade(call):
    trade_id = int(call.data.split('_')[1])
    bot.answer_callback_query(call.id, "Отправлен сигнал на закрытие...")
    
    def _run():
        trade = _read_db('SELECT id,symbol,side,entry_price,size,sl,tp,is_trailing,trailing_sl,partial_price,entry_time,strategy FROM active_trades WHERE id=?', (trade_id,), fetchone=True)
        if trade:
            try:
                close_trade(trade, exchange.fetch_ticker(trade[1])['last'], '🎯 Точечное ручное закрытие')
                bot.edit_message_reply_markup(chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=None)
            except Exception as e:
                log.error(f'[Manual Kill Error] {trade[1]}: {e}')
    thread_pool.submit(_run)

@bot.message_handler(func=lambda m: m.text == '📈 СТАТИСТИКА')
def cmd_stats(msg):
    def _run():
        rows = _read_db('SELECT net_pnl, result, strategy FROM trade_log')
        if not rows:
            bot.send_message(USER_ID, '📈 Закрытых сделок пока нет.')
            return
        
        def calc_stats(data):
            if not data: return "Нет данных"
            total = len(data)
            wins = sum(1 for r in data if r[1] == 'WIN')
            pnl = sum(r[0] for r in data)
            gp = sum(r[0] for r in data if r[0] > 0)
            gl = abs(sum(r[0] for r in data if r[0] < 0))
            pf = gp / gl if gl > 0 else (gp if gp > 0 else 0)
            return f'Сделок: `{total}` | WR: `{wins/total*100:.1f}%` | PF: `{pf:.2f}`\nPnL: `{pnl:+.2f}` USDT'

        stats_1h = calc_stats([r for r in rows if '1h' in r[2]])
        stats_15m = calc_stats([r for r in rows if '15m' in r[2]])
        overall_pnl = sum(r[0] for r in rows)

        bot.send_message(USER_ID, 
            f'📈 *A/B ТЕСТИРОВАНИЕ (SMART)*\n\n'
            f'🕰️ *Стратегия 1H (Тренд)*\n{stats_1h}\n\n'
            f'⏱️ *Стратегия 15m (Smart)*\n{stats_15m}\n\n'
            f'───────────────\n'
            f'🏆 Итоговый PnL: `{overall_pnl:+.2f}` USDT', 
            parse_mode='Markdown')
    thread_pool.submit(_run)

@bot.message_handler(func=lambda m: m.text == '🚪 ЗАКРЫТЬ ВСЕ')
def cmd_close_all(msg):
    markup = types.InlineKeyboardMarkup()
    btn_yes = types.InlineKeyboardButton('⚠️ ДА, ЗАКРЫТЬ ВСЕ', callback_data='confirm_close_all')
    btn_no = types.InlineKeyboardButton('❌ Отмена', callback_data='cancel_close_all')
    markup.add(btn_yes, btn_no)
    
    bot.send_message(
        USER_ID, 
        'Вы уверены, что хотите закрыть **ВСЕ** активные сделки по текущей рыночной цене?', 
        parse_mode='Markdown', 
        reply_markup=markup
    )

@bot.callback_query_handler(func=lambda call: call.data in ['confirm_close_all', 'cancel_close_all'])
def callback_close_all(call):
    bot.answer_callback_query(call.id) 
    
    if call.data == 'cancel_close_all':
        bot.edit_message_text(
            'Действие отменено. Сделки продолжают работу.', 
            chat_id=call.message.chat.id, 
            message_id=call.message.message_id
        )
        return

    if call.data == 'confirm_close_all':
        bot.edit_message_text(
            '⏳ Выполняю экстренное закрытие всех позиций...', 
            chat_id=call.message.chat.id, 
            message_id=call.message.message_id
        )
        
        def _run():
            trades = _read_db('SELECT id,symbol,side,entry_price,size,sl,tp,is_trailing,trailing_sl,partial_price,entry_time,strategy FROM active_trades')
            if not trades:
                bot.send_message(USER_ID, 'Нет активных сделок для закрытия.')
                return
                
            closed_count = 0
            for t in trades:
                try: 
                    close_trade(t, exchange.fetch_ticker(t[1])['last'], '🚪 Ручное закрытие')
                    closed_count += 1
                except Exception as e: 
                    log.error(f'[Manual Close Error] {t[1]}: {e}')
                    
            bot.send_message(USER_ID, f'✅ Успешно закрыто позиций: `{closed_count}`.', parse_mode='Markdown')
            
        thread_pool.submit(_run)

# =====================================================================
# ЗАПУСК
# =====================================================================
async def main_async():
    await asyncio.gather(
        hunter_1h(),
        hunter_15m(),
        monitor_trades()
    )

if __name__ == '__main__':
    init_db()
    log.info('🛡️ Fortress Dual-Core (Smart) запущен')
    threading.Thread(target=db_worker, daemon=True).start()
    threading.Thread(target=lambda: asyncio.run(main_async()), daemon=True).start()
    while True:
        try: bot.infinity_polling(timeout=90)
        except: time.sleep(5)