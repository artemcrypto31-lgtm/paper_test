"""
Fortress Paper Trading Bot V3.1
=================================
- Автоматическое открытие виртуальных сделок (LONG)
- Мониторинг SL/TP каждые 60 секунд
- Частичное закрытие 50% на TP1 + трейлинг стоп
- Журнал всех сделок в SQLite
- Полная статистика (WinRate, PnL, Drawdown, и др.)
- Бэктест по тикеру с тремя стратегиями
- Радар пампа — раннее обнаружение разгона
- Готов к переходу на реальную торговлю (один флаг LIVE_TRADING=True)
"""

import asyncio
import sqlite3
import os
import telebot
import threading
import pandas as pd
import pandas_ta as ta
import ccxt
import time
from telebot import types
from datetime import datetime
from dotenv import load_dotenv
from backtesting import Backtest, Strategy
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# =====================================================================
# КОНФИГУРАЦИЯ
# =====================================================================
load_dotenv()
bot = telebot.TeleBot(os.getenv('BOT_TOKEN'))
USER_ID = int(os.getenv('USER_ID'))

STARTING_BALANCE   = 1000.0
MIN_VOLUME_24H     = 120_000_000.0
MIN_NATR           = 1.0
RISK_PER_TRADE     = 0.01       # 1% баланса на сделку
REWARD_RATIO       = 2.5        # TP = SL * 2.5
ATR_MULT_SL        = 2.0        # SL = ATR * 2.0
SCAN_INTERVAL_SEC  = 900        # Интервал сканирования (секунды)
MONITOR_INTERVAL   = 60         # Интервал проверки SL/TP (секунды)
RADAR_INTERVAL     = 7200       # Радар каждые 2 часа (в секундах)

BLACKLIST = [
    'LINK/USDT:USDT',
    'SOL/USDT:USDT',
    'LTC/USDT:USDT',
    'XMR/USDT:USDT',
    'SEI/USDT:USDT',
    'OPN/USDT:USDT',
    'FIO/USDT:USDT',
]

# ─── ПЕРЕКЛЮЧАТЕЛЬ РЕАЛЬНОЙ ТОРГОВЛИ ────────────────────────────────
LIVE_TRADING        = False
EXCHANGE_API_KEY    = os.getenv('EXCHANGE_API_KEY', '')
EXCHANGE_API_SECRET = os.getenv('EXCHANGE_API_SECRET', '')
# ────────────────────────────────────────────────────────────────────

# =====================================================================
# БАЗА ДАННЫХ
# =====================================================================
def init_db():
    conn = sqlite3.connect('paper_trading.db')
    c = conn.cursor()

    c.execute('CREATE TABLE IF NOT EXISTS wallet (balance REAL)')
    if not c.execute('SELECT balance FROM wallet').fetchone():
        c.execute('INSERT INTO wallet VALUES (?)', (STARTING_BALANCE,))

    c.execute('''CREATE TABLE IF NOT EXISTS active_trades (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol        TEXT,
        side          TEXT,
        entry_price   REAL,
        size          REAL,
        sl            REAL,
        tp            REAL,
        entry_time    TEXT,
        entry_balance REAL
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS trade_log (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol        TEXT,
        side          TEXT,
        entry_price   REAL,
        exit_price    REAL,
        size          REAL,
        pnl           REAL,
        result        TEXT,
        entry_time    TEXT,
        exit_time     TEXT,
        balance_after REAL
    )''')
    conn.commit()

    cols = [row[1] for row in conn.execute('PRAGMA table_info(active_trades)').fetchall()]
    if 'entry_balance' not in cols:
        conn.execute('ALTER TABLE active_trades ADD COLUMN entry_balance REAL')
        conn.commit()
        print('[DB] Миграция: добавлен entry_balance')

    conn.close()


def get_db():
    return sqlite3.connect('paper_trading.db', check_same_thread=False)

# =====================================================================
# БИРЖА
# =====================================================================
def get_exchange(live=False):
    params = {'options': {'defaultType': 'future'}}
    if live:
        params['apiKey'] = EXCHANGE_API_KEY
        params['secret'] = EXCHANGE_API_SECRET
    return ccxt.binance(params)

# =====================================================================
# КЛАВИАТУРА
# =====================================================================
def main_keyboard():
    m = types.ReplyKeyboardMarkup(resize_keyboard=True)
    m.row('💰 БАЛАНС', '⚔️ СДЕЛКИ')
    m.row('📈 СТАТИСТИКА', '📊 БЭКТЕСТ')
    m.row('🔍 РАДАР', '📡 СТАТУС')
    return m

# =====================================================================
# СТРАТЕГИЯ 1: EMA Cross + Volume Filter
# =====================================================================
class FortressBT(Strategy):
    def init(self):
        c = pd.Series(self.data.Close)
        h = pd.Series(self.data.High)
        l = pd.Series(self.data.Low)
        v = pd.Series(self.data.Volume)

        self.ema12  = self.I(ta.ema, c, 12)
        self.ema26  = self.I(ta.ema, c, 26)
        self.atr    = self.I(ta.atr, h, l, c, 14)
        self.vol_ma = self.I(ta.sma, v, 20)

    def next(self):
        price = self.data.Close[-1]
        v     = self.data.Volume

        trend_ok      = self.ema12[-1] > self.ema26[-1]
        vol_growing   = v[-1] > v[-2] > v[-3]
        price_growing = self.data.Close[-1] > self.data.Close[-2] > self.data.Close[-3]
        vol_spike     = v[-1] > self.vol_ma[-1] * 1.5

        signals = sum([vol_growing, price_growing, vol_spike])

        if trend_ok and signals >= 2 and not self.position:
            sl  = price - self.atr[-1] * ATR_MULT_SL
            tp2 = price + (price - sl) * 6.0
            self.buy(sl=sl, tp=tp2)

        elif self.position and self.trades:
            trade      = self.trades[-1]
            entry      = trade.entry_price
            sl_current = trade.sl
            atr_now    = self.atr[-1]
            tp1        = entry + (entry - sl_current) * 2.5

            if price >= tp1:
                new_sl = max(sl_current, entry)
                new_sl = max(new_sl, price - atr_now * 1.0)
                trade.sl = new_sl

# =====================================================================
# СТРАТЕГИЯ 2: Momentum Breakout
# =====================================================================
class BreakoutBT(Strategy):
    def init(self):
        h = pd.Series(self.data.High)
        l = pd.Series(self.data.Low)
        c = pd.Series(self.data.Close)
        v = pd.Series(self.data.Volume)

        self.atr     = self.I(ta.atr, h, l, c, 14)
        self.rsi     = self.I(ta.rsi, c, 14)
        self.ema50   = self.I(ta.ema, c, 50)
        self.vol_ma  = self.I(ta.sma, v, 20)
        self.highest = self.I(lambda x: pd.Series(x).shift(1).rolling(20).max().values, c)

    def next(self):
        price     = self.data.Close[-1]
        trend_ok  = price > self.ema50[-1]
        breakout  = price > self.highest[-1]
        rsi_ok    = 45 < self.rsi[-1] < 72
        volume_ok = self.data.Volume[-1] > self.vol_ma[-1] * 1.3

        if trend_ok and breakout and rsi_ok and volume_ok and not self.position:
            sl = price - self.atr[-1] * 1.5
            tp = price + abs(price - sl) * 3.5
            self.buy(sl=sl, tp=tp)

# =====================================================================
# СТРАТЕГИЯ 3: Volatility Squeeze
# =====================================================================
class SqueezeBT(Strategy):
    def init(self):
        h = pd.Series(self.data.High)
        l = pd.Series(self.data.Low)
        c = pd.Series(self.data.Close)
        v = pd.Series(self.data.Volume)

        self.atr    = self.I(ta.atr, h, l, c, 14)
        self.rsi    = self.I(ta.rsi, c, 14)
        self.vol_ma = self.I(ta.sma, v, 20)

        bb = ta.bbands(c, length=20, std=2.0)
        self._bb_lower = bb.iloc[:, 0].values
        self._bb_mid   = bb.iloc[:, 1].values
        self._bb_upper = bb.iloc[:, 2].values

    def next(self):
        i = len(self.data.Close) - 1
        if i < 6:
            return

        price    = self.data.Close[-1]
        mid_now  = self._bb_mid[i]
        mid_prev = self._bb_mid[i - 5]

        if mid_now == 0 or mid_prev == 0:
            return

        width_now  = (self._bb_upper[i] - self._bb_lower[i]) / mid_now
        width_prev = (self._bb_upper[i - 5] - self._bb_lower[i - 5]) / mid_prev

        squeeze   = width_now < width_prev * 0.85
        breakout  = price > self._bb_upper[i]
        volume_ok = self.data.Volume[-1] > self.vol_ma[-1] * 2.5
        rsi_ok    = 55 < self.rsi[-1] < 80

        if squeeze and breakout and volume_ok and rsi_ok and not self.position:
            sl = price - self.atr[-1] * 2.0
            tp = price + abs(price - sl) * 4.0
            self.buy(sl=sl, tp=tp)

# =====================================================================
# АНАЛИЗ СИМВОЛА
# =====================================================================
def analyze_symbol(exchange, symbol):
    try:
        df4h = pd.DataFrame(
            exchange.fetch_ohlcv(symbol, '4h', limit=60),
            columns=['ts','o','h','l','c','v']
        )
        ema50_4h = ta.ema(df4h['c'], 50).iloc[-1]
        if df4h['c'].iloc[-1] < ema50_4h:
            return None

        df = pd.DataFrame(
            exchange.fetch_ohlcv(symbol, '1h', limit=60),
            columns=['ts','o','h','l','c','v']
        )
        c = df['c']
        h = df['h']
        l = df['l']
        v = df['v']

        atr  = ta.atr(h, l, c, 14).iloc[-1]
        natr = (atr / c.iloc[-1]) * 100
        if natr < MIN_NATR:
            return None

        vol_ma = v.rolling(20).mean()
        ema12  = ta.ema(c, 12)
        ema26  = ta.ema(c, 26)

        trend_ok = ema12.iloc[-1] > ema26.iloc[-1]
        if not trend_ok:
            return None

        vol_growing   = v.iloc[-1] > v.iloc[-2] > v.iloc[-3]
        price_growing = c.iloc[-1] > c.iloc[-2] > c.iloc[-3]
        vol_spike     = v.iloc[-1] > vol_ma.iloc[-1] * 1.5

        bear_bars = df[df['c'] < df['o']].tail(20)
        if len(bear_bars) > 0:
            big_bear_open = bear_bars.loc[bear_bars['v'].idxmax(), 'o']
            context_ok = c.iloc[-1] > big_bear_open
        else:
            context_ok = True

        signals = sum([vol_growing, price_growing, vol_spike, context_ok])
        if signals < 3:
            return None

        return {'symbol': symbol, 'price': c.iloc[-1], 'atr': atr}

    except Exception:
        pass
    return None

# =====================================================================
# РАДАР ПАМПА
# =====================================================================
def analyze_radar(exchange, symbol):
    try:
        df = pd.DataFrame(
            exchange.fetch_ohlcv(symbol, '1h', limit=24),
            columns=['ts','o','h','l','c','v']
        )
        c = df['c']
        v = df['v']
        h = df['h']
        l = df['l']

        atr  = ta.atr(h, l, c, 14).iloc[-1]
        natr = (atr / c.iloc[-1]) * 100
        if natr < 1.5:
            return None

        vol_ma          = v.rolling(10).mean()
        vol_growing     = all(v.iloc[-i] > v.iloc[-i-1] for i in range(1, 4))
        price_growing   = all(c.iloc[-i] > c.iloc[-i-1] for i in range(1, 4))
        vol_spike       = v.iloc[-1] > vol_ma.iloc[-1] * 1.8
        price_change_6h = (c.iloc[-1] / c.iloc[-7] - 1) * 100
        vol_acceleration = v.iloc[-1] / max(vol_ma.iloc[-1], 0.0001)

        signals = sum([
            vol_growing,
            price_growing,
            vol_spike,
            price_change_6h > 5.0
        ])

        if signals >= 3:
            return {
                'symbol':        symbol,
                'price':         c.iloc[-1],
                'change_6h':     price_change_6h,
                'vol_x':         vol_acceleration,
                'natr':          natr,
                'signals':       signals,
                'vol_growing':   vol_growing,
                'price_growing': price_growing,
                'vol_spike':     vol_spike,
            }
    except Exception:
        pass
    return None


async def pump_radar():
    exchange = get_exchange()
    alerted  = set()

    while True:
        await asyncio.sleep(RADAR_INTERVAL)
        try:
            print('[Radar] Сканирование радара...')
            tickers    = exchange.fetch_tickers()
            candidates = [
                s for s, d in tickers.items()
                if '/USDT:USDT' in s and (d.get('quoteVolume') or 0) >= MIN_VOLUME_24H
            ]

            found = []
            for sym in candidates:
                result = analyze_radar(exchange, sym)
                if result and sym not in alerted:
                    found.append(result)
                    alerted.add(sym)

            if len(alerted) > 100:
                alerted.clear()

            if not found:
                print('[Radar] Подозрительных монет не найдено')
                continue

            found.sort(key=lambda x: (x['signals'], x['change_6h']), reverse=True)

            text  = f'🔍 *РАДАР ПАМПА* | {datetime.now().strftime("%H:%M")}\n'
            text += f'Найдено подозрительных монет: {len(found)}\n\n'

            for r in found:
                flags = []
                if r['vol_growing']:   flags.append('📈 объём растёт')
                if r['price_growing']: flags.append('🕯 свечи вверх')
                if r['vol_spike']:     flags.append('⚡ всплеск объёма')
                if r['change_6h'] > 5: flags.append(f'🚀 +{r["change_6h"]:.1f}% за 6ч')

                text += (f'⚡ *{r["symbol"]}*\n'
                         f'  Цена: `{r["price"]:.6g}` | NATR: `{r["natr"]:.1f}%`\n'
                         f'  Объём x`{r["vol_x"]:.1f}` от среднего\n'
                         f'  {" | ".join(flags)}\n'
                         f'  Сигналов: `{r["signals"]}/4`\n\n')

            if len(text) > 4000:
                text = text[:4000] + '\n... и другие'

            bot.send_message(USER_ID, text, parse_mode='Markdown')
            print(f'[Radar] Отправлен алерт: {len(found)} монет')

        except Exception as e:
            print(f'[Radar] Ошибка: {e}')

# =====================================================================
# ОТКРЫТИЕ СДЕЛКИ
# =====================================================================
def open_trade(symbol, price, atr_val):
    conn = get_db()
    try:
        if conn.execute('SELECT 1 FROM active_trades WHERE symbol=?', (symbol,)).fetchone():
            return False

        try:
            exchange     = get_exchange()
            candles      = exchange.fetch_ohlcv(symbol, '1h', limit=3)
            price_1h_ago = candles[-2][4]
            change_1h    = (price / price_1h_ago - 1) * 100
            if change_1h > 5.0:
                print(f'[Scanner] Пропуск {symbol}: уже вырос на {change_1h:.1f}% за час')
                return False
        except Exception:
            pass

        balance = conn.execute('SELECT balance FROM wallet').fetchone()[0]
        sl_dist = atr_val * ATR_MULT_SL
        sl      = price - sl_dist
        tp      = price + sl_dist * REWARD_RATIO
        size    = (balance * RISK_PER_TRADE) / sl_dist

        conn.execute(
            'INSERT INTO active_trades (symbol,side,entry_price,size,sl,tp,entry_time,entry_balance) '
            'VALUES (?,?,?,?,?,?,?,?)',
            (symbol, 'long', price, size, sl, tp,
             datetime.now().strftime('%Y-%m-%d %H:%M:%S'), balance)
        )
        conn.commit()

        bot.send_message(USER_ID,
            f'🚀 *АВТОВХОД: {symbol}*\n'
            f'Тип: *LONG*\n'
            f'Цена входа: `{price:.6g}`\n'
            f'🔴 SL: `{sl:.6g}`\n'
            f'🟢 TP: `{tp:.6g}`\n'
            f'📦 Объём: `{size:.4f}` units\n'
            f'💵 Риск: `{balance * RISK_PER_TRADE:.2f}` USDT',
            parse_mode='Markdown'
        )
        return True
    finally:
        conn.close()

# =====================================================================
# ЗАКРЫТИЕ СДЕЛКИ
# =====================================================================
def close_trade(trade, exit_price, reason):
    tid, symbol, side, entry_price, size, sl, tp, entry_time, entry_balance = trade
    pnl    = (exit_price - entry_price) * size
    result = 'WIN' if pnl > 0 else ('LOSS' if pnl < 0 else 'BREAKEVEN')

    conn = get_db()
    try:
        balance     = conn.execute('SELECT balance FROM wallet').fetchone()[0]
        new_balance = balance + pnl

        conn.execute('UPDATE wallet SET balance=?', (new_balance,))
        conn.execute('DELETE FROM active_trades WHERE id=?', (tid,))
        conn.execute(
            'INSERT INTO trade_log (symbol,side,entry_price,exit_price,size,pnl,result,'
            'entry_time,exit_time,balance_after) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (symbol, side, entry_price, exit_price, size, pnl, result,
             entry_time, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), new_balance)
        )
        conn.commit()

        icon = '🏆' if result == 'WIN' else '💀'
        bot.send_message(USER_ID,
            f'{icon} *ЗАКРЫТО: {symbol}* ({reason})\n'
            f'Выход: `{exit_price:.6g}`\n'
            f'PnL: `{"+".join(["", f"{pnl:.2f}"]) if pnl >= 0 else f"{pnl:.2f}"}` USDT\n'
            f'Баланс: `{new_balance:.2f}` USDT',
            parse_mode='Markdown'
        )
    finally:
        conn.close()

# =====================================================================
# МОНИТОРИНГ SL/TP
# =====================================================================
async def monitor_trades():
    exchange = get_exchange()
    while True:
        try:
            conn   = get_db()
            trades = conn.execute(
                'SELECT id,symbol,side,entry_price,size,sl,tp,entry_time,entry_balance '
                'FROM active_trades'
            ).fetchall()
            conn.close()

            for trade in trades:
                tid, symbol, side, entry_price, size, sl, tp, entry_time, entry_balance = trade
                try:
                    ticker      = exchange.fetch_ticker(symbol)
                    price       = ticker['last']
                    candles     = exchange.fetch_ohlcv(symbol, '1m', limit=2)
                    last_candle = candles[-2]
                    candle_low  = last_candle[3]
                    candle_high = last_candle[2]

                    df  = pd.DataFrame(
                        exchange.fetch_ohlcv(symbol, '15m', limit=20),
                        columns=['ts','o','h','l','c','v']
                    )
                    atr = ta.atr(df['h'], df['l'], df['c'], 14).iloc[-1]

                    sl_hit   = price <= sl or candle_low <= sl
                    tp1      = entry_price + (entry_price - sl) * 2.5
                    tp1_hit  = price >= tp1 or candle_high >= tp1

                    partial_done = tp < 0
                    trailing_sl  = abs(tp) if partial_done else None

                    if not partial_done:
                        if sl_hit:
                            close_price = min(price, candle_low)
                            close_trade(trade, close_price, '🔴 STOP-LOSS')

                        elif tp1_hit:
                            close_price  = max(price, candle_high)
                            partial_size = size * 0.5
                            pnl_partial  = (close_price - entry_price) * partial_size

                            conn2       = get_db()
                            balance     = conn2.execute('SELECT balance FROM wallet').fetchone()[0]
                            new_balance = balance + pnl_partial

                            new_trailing_sl = entry_price
                            conn2.execute('UPDATE wallet SET balance=?', (new_balance,))
                            conn2.execute(
                                'UPDATE active_trades SET size=?, tp=? WHERE id=?',
                                (partial_size, -new_trailing_sl, tid)
                            )
                            conn2.commit()
                            conn2.close()

                            bot.send_message(USER_ID,
                                f'🎯 *ЧАСТИЧНОЕ ЗАКРЫТИЕ: {symbol}*\n'
                                f'Закрыто 50% по `{close_price:.6g}`\n'
                                f'PnL частичный: `+{pnl_partial:.2f}` USDT\n'
                                f'SL перенесён в безубыток: `{entry_price:.6g}`\n'
                                f'Трейлинг активирован 🎯\n'
                                f'Баланс: `{new_balance:.2f}` USDT',
                                parse_mode='Markdown'
                            )

                    else:
                        new_trailing_sl = max(trailing_sl, price - atr * 1.0)

                        if price <= trailing_sl or candle_low <= trailing_sl:
                            close_price = min(price, candle_low)
                            close_trade(trade, close_price, '📍 ТРЕЙЛИНГ СТОП')

                        elif new_trailing_sl > trailing_sl:
                            conn2 = get_db()
                            conn2.execute(
                                'UPDATE active_trades SET tp=? WHERE id=?',
                                (-new_trailing_sl, tid)
                            )
                            conn2.commit()
                            conn2.close()

                except Exception:
                    pass

        except Exception as e:
            print(f'[Monitor] Ошибка: {e}')
        await asyncio.sleep(MONITOR_INTERVAL)

# =====================================================================
# СКАНЕР РЫНКА
# =====================================================================
async def signal_hunter():
    exchange = get_exchange()
    while True:
        try:
            bot.send_message(USER_ID, '🔎 *Сканирование рынка...*', parse_mode='Markdown')

            tickers = None
            for attempt in range(3):
                try:
                    tickers = exchange.fetch_tickers()
                    break
                except Exception as e:
                    print(f'[Scanner] Попытка {attempt+1}/3 не удалась: {e}')
                    await asyncio.sleep(30)

            if tickers is None:
                print('[Scanner] Все попытки исчерпаны, пропускаем цикл')
                await asyncio.sleep(SCAN_INTERVAL_SEC)
                continue

            candidates = [
                s for s, d in tickers.items()
                if '/USDT:USDT' in s and (d.get('quoteVolume') or 0) >= MIN_VOLUME_24H
            ]

            found = 0
            total = len(candidates)
            for i, sym in enumerate(candidates, 1):
                if sym in BLACKLIST:
                    continue
                print(f'[{i}/{total}] {sym}     ', end='\r')
                setup = analyze_symbol(exchange, sym)
                if setup and open_trade(setup['symbol'], setup['price'], setup['atr']):
                    found += 1
                await asyncio.sleep(0.3)

            print()
            bot.send_message(USER_ID,
                f'🏁 *Скан завершён*\nОткрыто новых сделок: {found}',
                parse_mode='Markdown'
            )
        except Exception as e:
            print(f'[Scanner] Ошибка: {e}')
            await asyncio.sleep(60)
        await asyncio.sleep(SCAN_INTERVAL_SEC)

# =====================================================================
# TELEGRAM HANDLERS
# =====================================================================
@bot.message_handler(commands=['start'])
def cmd_start(msg):
    if msg.from_user.id != USER_ID: return
    bot.send_message(USER_ID,
        '🛡️ *Fortress V3.1 активен*\n\nВиртуальная торговля запущена.\n'
        'Все сделки ведутся в бумажном режиме.\n\n'
        'Команды кнопок:\n'
        '💰 Баланс | ⚔️ Сделки | 📈 Статистика | 📊 Бэктест\n\n'
        'Бэктест: отправь `📊 БЭКТЕСТ` и введи тикер (например *LINK*)',
        parse_mode='Markdown', reply_markup=main_keyboard()
    )

@bot.message_handler(func=lambda m: m.text == '💰 БАЛАНС')
def cmd_balance(msg):
    if msg.from_user.id != USER_ID: return
    conn    = get_db()
    balance = conn.execute('SELECT balance FROM wallet').fetchone()[0]
    trades  = conn.execute('SELECT symbol,entry_price,size FROM active_trades').fetchall()
    conn.close()

    unrealized = 0.0
    exc = get_exchange()
    for sym, ep, sz in trades:
        try:
            unrealized += (exc.fetch_ticker(sym)['last'] - ep) * sz
        except Exception:
            pass

    equity = balance + unrealized
    bot.send_message(USER_ID,
        f'🏦 *БАЛАНС*\n'
        f'Доступно: `{balance:.2f}` USDT\n'
        f'Открытый PnL: `{unrealized:+.2f}` USDT\n'
        f'💰 *Equity: {equity:.2f} USDT*\n'
        f'📈 Старт: `{STARTING_BALANCE:.2f}` USDT  |  '
        f'{"🟢" if equity >= STARTING_BALANCE else "🔴"} {((equity / STARTING_BALANCE - 1) * 100):+.2f}%',
        parse_mode='Markdown'
    )

@bot.message_handler(func=lambda m: m.text == '⚔️ СДЕЛКИ')
def cmd_trades(msg):
    if msg.from_user.id != USER_ID: return
    conn   = get_db()
    trades = conn.execute(
        'SELECT symbol,side,entry_price,sl,tp,entry_time FROM active_trades'
    ).fetchall()
    conn.close()

    if not trades:
        bot.send_message(USER_ID, '⚔️ Активных сделок нет.')
        return

    text = '⚔️ *АКТИВНЫЕ СДЕЛКИ:*\n\n'
    for sym, side, ep, sl, tp, et in trades:
        time_str = et[11:16] if (et and len(et) >= 16) else '—'
        if tp < 0:
            mode_str = f'📍 Трейлинг SL: `{abs(tp):.6g}`'
        else:
            tp1 = ep + (ep - sl) * 2.5
            mode_str = f'🎯 TP1: `{tp1:.6g}` | 🟢 TP2: трейлинг'
        text += (f'• *{sym}* | {side.upper()}\n'
                 f'  Вход: `{ep:.6g}` ({time_str})\n'
                 f'  🔴 SL: `{sl:.6g}`\n'
                 f'  {mode_str}\n\n')
    bot.send_message(USER_ID, text, parse_mode='Markdown')

@bot.message_handler(func=lambda m: m.text == '📈 СТАТИСТИКА')
def cmd_stats(msg):
    if msg.from_user.id != USER_ID: return
    conn = get_db()
    rows = conn.execute(
        'SELECT pnl,result,balance_after,entry_time,exit_time FROM trade_log ORDER BY id'
    ).fetchall()
    active_count = conn.execute('SELECT COUNT(*) FROM active_trades').fetchone()[0]
    conn.close()

    if not rows:
        bot.send_message(USER_ID, '📈 Закрытых сделок пока нет.')
        return

    total   = len(rows)
    wins    = sum(1 for r in rows if r[1] == 'WIN')
    losses  = sum(1 for r in rows if r[1] == 'LOSS')
    pnl_sum = sum(r[0] for r in rows)
    gross_p = sum(r[0] for r in rows if r[0] > 0)
    gross_l = abs(sum(r[0] for r in rows if r[0] < 0))
    pf      = (gross_p / gross_l) if gross_l else float('inf')
    wr      = (wins / total * 100) if total else 0

    balances = [STARTING_BALANCE] + [r[2] for r in rows]
    peak, max_dd = balances[0], 0.0
    for b in balances:
        if b > peak: peak = b
        dd = (peak - b) / peak * 100
        if dd > max_dd: max_dd = dd

    avg_win  = gross_p / wins   if wins   else 0
    avg_loss = gross_l / losses if losses else 0

    text = (
        f'📈 *СТАТИСТИКА ТОРГОВЛИ*\n\n'
        f'Всего сделок: `{total}` (актив: {active_count})\n'
        f'✅ Побед: `{wins}` | ❌ Поражений: `{losses}`\n'
        f'WinRate: `{wr:.1f}%`\n'
        f'Profit Factor: `{pf:.2f}`\n\n'
        f'Общий PnL: `{pnl_sum:+.2f}` USDT\n'
        f'Средняя победа: `+{avg_win:.2f}` USDT\n'
        f'Среднее поражение: `-{avg_loss:.2f}` USDT\n'
        f'Макс. просадка: `{max_dd:.1f}%`\n\n'
        f'Старт: `{STARTING_BALANCE:.2f}` | '
        f'Баланс: `{balances[-1]:.2f}` USDT\n'
        f'Результат: {"🟢" if pnl_sum >= 0 else "🔴"} '
        f'`{((balances[-1] / STARTING_BALANCE - 1) * 100):+.2f}%`'
    )
    bot.send_message(USER_ID, text, parse_mode='Markdown')

@bot.message_handler(func=lambda m: m.text == '📊 БЭКТЕСТ')
def cmd_bt_init(msg):
    if msg.from_user.id != USER_ID: return
    m = bot.send_message(USER_ID,
        '📊 Введи тикер для сравнения стратегий\n(например *LINK*, *SOL*, *BTC*):',
        parse_mode='Markdown'
    )
    bot.register_next_step_handler(m, cmd_bt_run)

def cmd_bt_run(msg):
    ticker = msg.text.strip().upper()
    symbol = f'{ticker}/USDT:USDT'
    bot.send_message(USER_ID, f'⏳ Тестирую три стратегии на {symbol}...')
    try:
        exchange = get_exchange()
        bars     = exchange.fetch_ohlcv(symbol, '1h', limit=1000)
        if len(bars) < 100:
            bot.send_message(USER_ID, '❌ Недостаточно данных. Проверь тикер.')
            return

        df = pd.DataFrame(bars, columns=['OpenTime','Open','High','Low','Close','Volume'])
        df['OpenTime'] = pd.to_datetime(df['OpenTime'], unit='ms')
        df.set_index('OpenTime', inplace=True)

        def run_bt(strategy_class):
            bt    = Backtest(df, strategy_class, cash=1000,
                             commission=0.0004, finalize_trades=True)
            stats = bt.run()
            total = int(stats['# Trades'])
            return {
                'trades': total,
                'wr':     float(stats['Win Rate [%]']) if total > 0 else 0.0,
                'ret':    float(stats['Return [%]']),
                'dd':     float(stats['Max. Drawdown [%]']),
                'pf':     float(stats.get('Profit Factor') or 0),
            }

        s1 = run_bt(FortressBT)
        s2 = run_bt(BreakoutBT)
        s3 = run_bt(SqueezeBT)

        def fmt(s):
            return (f"Сделок: `{s['trades']}`\n"
                    f"WinRate: `{s['wr']:.1f}%`\n"
                    f"Доходность: `{s['ret']:+.2f}%`\n"
                    f"Макс. просадка: `{s['dd']:.2f}%`\n"
                    f"Profit Factor: `{s['pf']:.2f}`")

        best = max([('1️⃣ EMA Cross', s1['pf']),
                    ('2️⃣ Breakout',  s2['pf']),
                    ('3️⃣ Squeeze',   s3['pf'])],
                   key=lambda x: x[1])
        winner = f'🏆 Лучше по PF: *{best[0]}*'

        bot.send_message(USER_ID,
            f'📊 *СРАВНЕНИЕ СТРАТЕГИЙ: {ticker}USDT.P*\n'
            f'Период: 1ч × 1000 свечей (~41 день)\n\n'
            f'─────────────────────\n'
            f'1️⃣ *EMA Cross (текущая)*\n{fmt(s1)}\n\n'
            f'─────────────────────\n'
            f'2️⃣ *Momentum Breakout*\n{fmt(s2)}\n\n'
            f'─────────────────────\n'
            f'3️⃣ *Volatility Squeeze*\n{fmt(s3)}\n\n'
            f'─────────────────────\n'
            f'{winner}',
            parse_mode='Markdown'
        )

    except Exception as e:
        bot.send_message(USER_ID,
            f'❌ Ошибка бэктеста `{ticker}`\n`{str(e)[:300]}`',
            parse_mode='Markdown'
        )

@bot.message_handler(func=lambda m: m.text == '📡 СТАТУС')
def cmd_status(msg):
    if msg.from_user.id != USER_ID: return
    conn    = get_db()
    active  = conn.execute('SELECT COUNT(*) FROM active_trades').fetchone()[0]
    closed  = conn.execute('SELECT COUNT(*) FROM trade_log').fetchone()[0]
    conn.close()
    mode = '🔴 РЕАЛЬНАЯ' if LIVE_TRADING else '🟡 БУМАЖНАЯ'
    bot.send_message(USER_ID,
        f'📡 *СТАТУС СИСТЕМЫ*\n\n'
        f'Режим: {mode}\n'
        f'Активных сделок: `{active}`\n'
        f'Закрытых сделок: `{closed}`\n'
        f'Скан каждые: `{SCAN_INTERVAL_SEC // 60}` мин\n'
        f'Монитор SL/TP: каждые `{MONITOR_INTERVAL}` сек\n'
        f'Время сервера: `{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}`',
        parse_mode='Markdown'
    )

@bot.message_handler(func=lambda m: m.text == '🔍 РАДАР')
def cmd_radar_manual(msg):
    if msg.from_user.id != USER_ID: return
    bot.send_message(USER_ID, '🔍 Запускаю радар вручную...')
    try:
        exchange   = get_exchange()
        tickers    = exchange.fetch_tickers()
        candidates = [
            s for s, d in tickers.items()
            if '/USDT:USDT' in s and (d.get('quoteVolume') or 0) >= MIN_VOLUME_24H
        ]
        found = []
        total = len(candidates)
        for i, sym in enumerate(candidates, 1):
            print(f'[Radar Manual] [{i}/{total}] {sym}     ', end='\r')
            result = analyze_radar(exchange, sym)
            if result:
                found.append(result)
        print()

        if not found:
            bot.send_message(USER_ID, '🔍 Радар: подозрительных монет не найдено.')
            return

        found.sort(key=lambda x: (x['signals'], x['change_6h']), reverse=True)

        text  = '🔍 *РАДАР ПАМПА* (ручной)\n'
        text += f'Найдено: {len(found)} монет\n\n'

        for r in found:
            flags = []
            if r['vol_growing']:   flags.append('📈 объём растёт')
            if r['price_growing']: flags.append('🕯 свечи вверх')
            if r['vol_spike']:     flags.append('⚡ всплеск объёма')
            if r['change_6h'] > 5: flags.append(f'🚀 +{r["change_6h"]:.1f}% за 6ч')

            text += (f'⚡ *{r["symbol"]}*\n'
                     f'  Цена: `{r["price"]:.6g}` | NATR: `{r["natr"]:.1f}%`\n'
                     f'  Объём x`{r["vol_x"]:.1f}` от среднего\n'
                     f'  {" | ".join(flags)}\n'
                     f'  Сигналов: `{r["signals"]}/4`\n\n')

        if len(text) > 4000:
            text = text[:4000] + '\n... и другие'

        bot.send_message(USER_ID, text, parse_mode='Markdown')

    except Exception as e:
        bot.send_message(USER_ID, f'❌ Ошибка радара: {str(e)[:200]}')

# =====================================================================
# ТОЧКА ВХОДА
# =====================================================================
async def main_async():
    await asyncio.gather(
        signal_hunter(),
        monitor_trades(),
        pump_radar()
    )

if __name__ == '__main__':
    init_db()
    print('🛡️  Fortress Paper Trading V3.1')
    print(f'   Режим: {"РЕАЛЬНАЯ ТОРГОВЛЯ" if LIVE_TRADING else "Бумажная торговля"}')
    print(f'   Баланс старта: {STARTING_BALANCE} USDT')

    threading.Thread(
        target=lambda: asyncio.run(main_async()),
        daemon=True
    ).start()

    while True:
        try:
            print('📡 Бот слушает Telegram...')
            bot.infinity_polling(timeout=90, long_polling_timeout=90)
        except Exception as e:
            print(f'Сетевой сбой: {e}. Перезапуск через 5 сек...')
            time.sleep(5)