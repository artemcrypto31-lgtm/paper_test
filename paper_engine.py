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
MIN_NATR          = 1.0
RISK_PER_TRADE    = 0.01      
ATR_MULT_SL       = 2.0       
MONITOR_INTERVAL  = 60        

# Сетки сканирования
GRID_1H_MINUTES   = 60
GRID_15M_MINUTES  = 15

# Реальная комиссия (Binance Futures Taker)
FEE_RATE = 0.0005 

BLACKLIST = {'LINK/USDT:USDT', 'SOL/USDT:USDT', 'LTC/USDT:USDT', 'XMR/USDT:USDT', 'USDC/USDT:USDT'}

BTC_CORR_TIMEFRAME = '4h'
BTC_CORR_LIMIT     = 100
BTC_CORR_MAX       = 0.8

exchange = ccxt.binance({'options': {'defaultType': 'future'}, 'enableRateLimit': True})
thread_pool = ThreadPoolExecutor(max_workers=5)

# =====================================================================
# ИНДИКАТОРЫ
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

# =====================================================================
# БАЗА ДАННЫХ (Атомарные транзакции)
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
    """Стратегия 1: Глобальный вход на 1H свечах"""
    if btc_closes is not None and is_correlated_with_btc(symbol, btc_closes): return None

    df4h = pd.DataFrame(exchange.fetch_ohlcv(symbol, '4h', limit=200), columns=['ts', 'o', 'h', 'l', 'c', 'v'])
    ema50_4h = _ema(df4h['c'], 50).iloc[-2]
    if pd.isna(ema50_4h): return None
    above_ema50 = df4h['c'].iloc[-2] > ema50_4h

    try:
        adx_val = _adx(df4h['h'], df4h['l'], df4h['c'], 14).iloc[-2]
        if not pd.isna(adx_val) and adx_val < 25: return None
    except: pass

    df1h = pd.DataFrame(exchange.fetch_ohlcv(symbol, '1h', limit=60), columns=['ts', 'o', 'h', 'l', 'c', 'v'])
    c = df1h['c']; h = df1h['h']; l = df1h['l']; v = df1h['v']
    atr_val = _atr(h, l, c, 14).iloc[-2]
    e12 = _ema(c, 12).iloc[-2]; e26 = _ema(c, 26).iloc[-2]

    if pd.isna(atr_val) or pd.isna(e12) or pd.isna(e26) or atr_val <= 0: return None
    if (atr_val / c.iloc[-2]) * 100 < MIN_NATR: return None

    if above_ema50 and e12 > e26: side = 'long'
    elif not above_ema50 and e12 < e26: side = 'short'
    else: return None

    vol_ma = v.rolling(20).mean()
    if sum([v.iloc[-2] > v.iloc[-3] > v.iloc[-4], 
            (c.iloc[-2] > c.iloc[-3]) if side=='long' else (c.iloc[-2] < c.iloc[-3]), 
            v.iloc[-2] > vol_ma.iloc[-2] * 1.5]) < 2: return None

    return {'symbol': symbol, 'price': c.iloc[-2], 'atr': atr_val, 'side': side, 'strategy': 'ema_1h'}

def analyze_15m(symbol: str, btc_closes: pd.Series | None = None):
    """Стратегия 2: Снайперский вход на 15m с подтверждением старших ТФ"""
    if btc_closes is not None and is_correlated_with_btc(symbol, btc_closes): return None

    df4h = pd.DataFrame(exchange.fetch_ohlcv(symbol, '4h', limit=100), columns=['ts', 'o', 'h', 'l', 'c', 'v'])
    ema50_4h = _ema(df4h['c'], 50).iloc[-2]
    if pd.isna(ema50_4h): return None
    above_ema50_4h = df4h['c'].iloc[-2] > ema50_4h

    df1h = pd.DataFrame(exchange.fetch_ohlcv(symbol, '1h', limit=60), columns=['ts', 'o', 'h', 'l', 'c', 'v'])
    e12_1h = _ema(df1h['c'], 12).iloc[-2]; e26_1h = _ema(df1h['c'], 26).iloc[-2]
    if pd.isna(e12_1h) or pd.isna(e26_1h): return None
    bull_1h = e12_1h > e26_1h

    df15m = pd.DataFrame(exchange.fetch_ohlcv(symbol, '15m', limit=60), columns=['ts', 'o', 'h', 'l', 'c', 'v'])
    c = df15m['c']; h = df15m['h']; l = df15m['l']; v = df15m['v']
    atr_val = _atr(h, l, c, 14).iloc[-2]
    e12_15 = _ema(c, 12).iloc[-2]; e26_15 = _ema(c, 26).iloc[-2]

    if pd.isna(atr_val) or pd.isna(e12_15) or pd.isna(e26_15) or atr_val <= 0: return None
    if (atr_val / c.iloc[-2]) * 100 < (MIN_NATR * 0.5): return None

    if above_ema50_4h and bull_1h and e12_15 > e26_15: side = 'long'
    elif not above_ema50_4h and not bull_1h and e12_15 < e26_15: side = 'short'
    else: return None

    vol_ma = v.rolling(20).mean()
    if sum([v.iloc[-2] > v.iloc[-3] > v.iloc[-4], 
            (c.iloc[-2] > c.iloc[-3]) if side=='long' else (c.iloc[-2] < c.iloc[-3]), 
            v.iloc[-2] > vol_ma.iloc[-2] * 1.5]) < 2: return None

    return {'symbol': symbol, 'price': c.iloc[-2], 'atr': atr_val, 'side': side, 'strategy': 'ema_multi_15m'}

def open_trade(setup: dict) -> bool:
    symbol = setup['symbol']
    if _read_db('SELECT 1 FROM active_trades WHERE symbol=?', (symbol,), fetchone=True):
        return False

    balance = _read_db('SELECT balance FROM wallet', fetchone=True)[0]
    trades  = _read_db('SELECT symbol,side,entry_price,size FROM active_trades')
    unrealized = 0.0
    for sym, sd, ep, sz in trades:
        try:
            p = exchange.fetch_ticker(sym)['last']
            unrealized += (p - ep) * sz if sd == 'long' else (ep - p) * sz
        except: pass
    
    equity = balance + unrealized
    sl_dist = setup['atr'] * ATR_MULT_SL
    if sl_dist < setup['price'] * 0.001: return False
    
    size = (equity * RISK_PER_TRADE) / sl_dist
    sl = setup['price'] - sl_dist if setup['side'] == 'long' else setup['price'] + sl_dist

    try:
        _write_db_transaction([
            ('INSERT INTO active_trades (symbol,side,entry_price,size,sl,tp,entry_time,strategy) VALUES (?,?,?,?,?,NULL,?,?)',
             (symbol, setup['side'], setup['price'], size, sl, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), setup['strategy']))
        ], sync=True)
    except Exception as e:
        log.error(f'Ошибка входа {symbol}: {e}')
        return False

    tp1 = setup['price'] - sl_dist * 2.5 if setup['side'] == 'short' else setup['price'] + sl_dist * 2.5
    icon = '⏱️' if setup['strategy'] == 'ema_multi_15m' else '🕰️'
    bot.send_message(USER_ID, 
        f'{icon} *ВХОД ({setup["strategy"]}): {symbol}*\n'
        f'Тип: *{setup["side"].upper()}* | Цена: `{setup["price"]:.6g}`\n'
        f'🔴 SL: `{sl:.6g}` | 🎯 TP1: `{tp1:.6g}`', 
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
    bot.send_message(USER_ID,
        f'{icon} *ЗАКРЫТО: {symbol}* ({strategy})\n{reason} | {side.upper()} → `{exit_price:.6g}`\n'
        f'Чистый PnL: `{net_pnl:+.2f}` USDT (Комиссия: `{total_fee:.2f}`)\nБаланс: `{new_balance:.2f}` USDT',
        parse_mode='Markdown')

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
                        tp1 = entry_price + (entry_price - sl) * 2.5
                        tp1_hit = price >= tp1 or candle_high >= tp1
                        
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
                            tf = '15m' if strategy == 'ema_multi_15m' else '1h'
                            df_tr = pd.DataFrame(await asyncio.to_thread(exchange.fetch_ohlcv, symbol, tf, limit=20), columns=['ts', 'o', 'h', 'l', 'c', 'v'])
                            atr = _atr(df_tr['h'], df_tr['l'], df_tr['c'], 14).iloc[-1]
                            new_tsl = max(trailing_sl, price - atr * 1.0)
                            
                            if price <= trailing_sl or candle_low <= trailing_sl:
                                close_trade(trade, min(price, candle_low), '📍 ТРЕЙЛИНГ СТОП')
                            elif new_tsl > trailing_sl:
                                _write_db_transaction([('UPDATE active_trades SET trailing_sl=? WHERE id=?', (new_tsl,))])

                    else: 
                        sl_hit = price >= sl or candle_high >= sl
                        tp1 = entry_price - (sl - entry_price) * 2.5
                        tp1_hit = price <= tp1 or candle_low <= tp1

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
                            tf = '15m' if strategy == 'ema_multi_15m' else '1h'
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
            log.info('[Scanner] Запуск 15m сканера')
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
# TELEGRAM БОТ (Команды)
# =====================================================================
def main_keyboard():
    m = types.ReplyKeyboardMarkup(resize_keyboard=True)
    m.row('💰 БАЛАНС', '⚔️ СДЕЛКИ')
    m.row('📈 СТАТИСТИКА', '🚪 ЗАКРЫТЬ ВСЕ')
    return m

@bot.message_handler(commands=['start'])
def cmd_start(msg):
    bot.send_message(USER_ID, '🛡️ *Fortress Dual-Core*\nA/B Тест: 1H против 15m. Изоляция позиций.', parse_mode='Markdown', reply_markup=main_keyboard())

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
            trades = _read_db(
                'SELECT symbol,side,entry_price,sl,is_trailing,trailing_sl,entry_time,size,strategy '
                'FROM active_trades'
            )
            if not trades:
                bot.send_message(USER_ID, '⚔️ Активных сделок нет.')
                return
                
            text = '⚔️ *АКТИВНЫЕ СДЕЛКИ:*\n\n'
            for sym, side, ep, sl, is_tr, tsl, et, sz, strategy in trades:
                t = et[5:16] if et else '—'
                icon = '⏱️' if '15m' in strategy else '🕰️'
                
                # Экранируем подчеркивания в названии стратегии для Markdown
                safe_strategy = strategy.replace('_', '\\_')
                
                try:
                    cur_price = exchange.fetch_ticker(sym)['last']
                    net_pnl = ((cur_price - ep) * sz if side == 'long' else (ep - cur_price) * sz) - ((ep * sz * FEE_RATE) + (cur_price * sz * FEE_RATE))
                    pnl_str = f'{"🟢" if net_pnl >= 0 else "🔴"} PnL: `{net_pnl:+.2f}` | Цена: `{cur_price:.6g}`\n  '
                except: 
                    pnl_str = ''

                if is_tr:
                    mode = f'📍 T-SL: `{tsl:.6g}`\n  🔴 SL: `{sl:.6g}`'
                else:
                    tp1 = ep + (ep - sl) * 2.5 if side == 'long' else ep - (sl - ep) * 2.5
                    mode = f'🎯 TP1: `{tp1:.6g}`\n  🔴 SL: `{sl:.6g}`'
                    
                text += f'{icon} *{sym}* ({safe_strategy}) | {side.upper()} | {t}\n  {pnl_str}{mode}\n\n'
            
            # Финальная защита от падения Telegram API
            try:
                bot.send_message(USER_ID, text, parse_mode='Markdown')
            except:
                # Если Markdown всё равно не пролез, шлем простым текстом
                clean_text = text.replace('*', '').replace('`', '').replace('\\_', '_')
                bot.send_message(USER_ID, clean_text)
                
        except Exception as e:
            log.error(f'[UI Error] Ошибка кнопки сделок: {e}', exc_info=True)
            
    thread_pool.submit(_run)

@bot.message_handler(func=lambda m: m.text == '📈 СТАТИСТИКА')
def cmd_stats(msg):
    def _run():
        rows = _read_db('SELECT net_pnl, result, strategy FROM trade_log')
        if not rows:
            bot.send_message(USER_ID, 'Сделок нет.')
            return
        
        def calc_stats(data):
            if not data: return "Нет данных"
            total = len(data)
            wins = sum(1 for r in data if r[1] == 'WIN')
            pnl = sum(r[0] for r in data)
            return f'Сделок: `{total}` | WR: `{wins/total*100:.1f}%`\nPnL: `{pnl:+.2f}` USDT'

        stats_1h = calc_stats([r for r in rows if r[2] == 'ema_1h'])
        stats_15m = calc_stats([r for r in rows if r[2] == 'ema_multi_15m'])
        overall_pnl = sum(r[0] for r in rows)

        bot.send_message(USER_ID, 
            f'📈 *A/B ТЕСТИРОВАНИЕ*\n\n'
            f'🕰️ *Стратегия 1H*\n{stats_1h}\n\n'
            f'⏱️ *Стратегия 15m*\n{stats_15m}\n\n'
            f'───────────────\n'
            f'🏆 Итоговый PnL: `{overall_pnl:+.2f}` USDT', 
            parse_mode='Markdown')
    thread_pool.submit(_run)

@bot.message_handler(func=lambda m: m.text == '🚪 ЗАКРЫТЬ ВСЕ')
def cmd_close_all(msg):
    bot.send_message(USER_ID, 'Закрываю...')
    def _run():
        trades = _read_db('SELECT id,symbol,side,entry_price,size,sl,tp,is_trailing,trailing_sl,partial_price,entry_time,strategy FROM active_trades')
        for t in trades:
            try: close_trade(t, exchange.fetch_ticker(t[1])['last'], '🚪 Ручное закрытие')
            except: pass
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
    log.info('🛡️ Fortress Dual-Core запущен')
    threading.Thread(target=db_worker, daemon=True).start()
    threading.Thread(target=lambda: asyncio.run(main_async()), daemon=True).start()
    while True:
        try: bot.infinity_polling(timeout=90)
        except: time.sleep(5)