"""
Fortress Paper Trading Bot V4.5
=================================
Изменения по сравнению с V4.4:
  - Фильтр корреляции с BTC (4H, 50 свечей, порог 0.8)
    Монеты с корреляцией > 0.8 отсеиваются в обоих сканерах
  - BTC данные загружаются один раз перед сканом (не на каждую монету)
"""

import asyncio
import math
import sqlite3
import os
import numpy as np
import telebot
import threading
import queue
import pandas as pd
import ccxt
import time
import warnings
from telebot import types
from datetime import datetime
from dotenv import load_dotenv
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

warnings.filterwarnings('ignore', category=FutureWarning)

# =====================================================================
# ИНДИКАТОРЫ — чистый pandas/numpy (pandas 3.x совместимо, без pandas_ta)
# =====================================================================
def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()

def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    """Возвращает (macd_line, signal_line) как pd.Series."""
    ml = _ema(series, fast) - _ema(series, slow)
    sl = _ema(ml, signal)
    return ml, sl

def _bbands(series: pd.Series, n: int = 20, std: float = 2.0):
    """Возвращает (lower, mid, upper) как pd.Series."""
    mid   = _sma(series, n)
    sigma = series.rolling(n).std()
    return mid - std * sigma, mid, mid + std * sigma

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
# КОНФИГУРАЦИЯ
# =====================================================================
load_dotenv()
bot     = telebot.TeleBot(os.getenv('BOT_TOKEN'))
USER_ID = int(os.getenv('USER_ID'))

STARTING_BALANCE  = 1000.0
MIN_VOLUME_24H    = 120_000_000.0
MIN_NATR          = 1.0
RISK_PER_TRADE    = 0.01      # 1% баланса на сделку
REWARD_RATIO      = 2.5       # TP = SL × 2.5  (для EMA Cross)
ATR_MULT_SL       = 2.0       # SL = ATR × 2.0 (для EMA Cross)
MONITOR_INTERVAL  = 60        # Проверка сделок каждые 60 сек
RADAR_INTERVAL    = 7200      # Радар каждые 2 часа

# Лимиты открытых сделок
MAX_EMA_TRADES    = 5         # Макс. одновременных EMA сделок
MAX_MACD_TRADES   = 5         # Макс. одновременных MACD сделок

# MACD параметры SL/TP (ATR-based, RR = 1:2.5)
MACD_ATR_SL       = 1.5       # SL = ATR × 1.5
MACD_ATR_TP       = 3.75      # TP = ATR × 3.75

# Фильтр корреляции с BTC
BTC_CORR_TIMEFRAME = '4h'     # таймфрейм для расчёта корреляции
BTC_CORR_LIMIT     = 50       # количество свечей
BTC_CORR_MAX       = 0.8      # порог: монеты выше отсеиваются

# Сетка запусков сканеров
EMA_GRID_MINUTES  = 15        # EMA Cross: каждые 15 мин по сетке
MACD_GRID_MINUTES = 30        # MACD:      каждые 30 мин по сетке

BLACKLIST = {
    'LINK/USDT:USDT',
    'SOL/USDT:USDT',
    'LTC/USDT:USDT',
    'XMR/USDT:USDT',
    'SEI/USDT:USDT',
    'OPN/USDT:USDT',
    'FIO/USDT:USDT',
}

LIVE_TRADING        = False
EXCHANGE_API_KEY    = os.getenv('EXCHANGE_API_KEY', '')
EXCHANGE_API_SECRET = os.getenv('EXCHANGE_API_SECRET', '')

# =====================================================================
# ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР БИРЖИ
# =====================================================================
def _make_exchange():
    params = {'options': {'defaultType': 'future'}, 'enableRateLimit': True}
    if LIVE_TRADING:
        params['apiKey'] = EXCHANGE_API_KEY
        params['secret'] = EXCHANGE_API_SECRET
    return ccxt.binance(params)

exchange = _make_exchange()

# =====================================================================
# УТИЛИТА: ОЖИДАНИЕ ДО СЛЕДУЮЩЕЙ ТОЧКИ СЕТКИ
# =====================================================================
def seconds_until_next_grid(interval_minutes: int) -> float:
    """
    Возвращает секунды до следующей точки сетки.
    interval_minutes=15 -> 11:00, 11:15, 11:30 ...
    interval_minutes=30 -> 11:00, 11:30, 12:00 ...
    """
    now       = datetime.now()
    minutes   = now.minute + now.second / 60.0
    next_grid = math.ceil(minutes / interval_minutes) * interval_minutes
    wait_min  = next_grid - minutes
    if wait_min < 0.1:
        wait_min += interval_minutes
    return max(wait_min * 60, 5.0)

# =====================================================================
# ФИЛЬТР КОРРЕЛЯЦИИ С BTC
# =====================================================================
def fetch_btc_closes() -> pd.Series | None:
    """Загружает закрытия BTC/USDT на 4H для расчёта корреляции. Вызывается один раз перед сканом."""
    try:
        df = pd.DataFrame(
            exchange.fetch_ohlcv('BTC/USDT:USDT', BTC_CORR_TIMEFRAME, limit=BTC_CORR_LIMIT + 5),
            columns=['ts', 'o', 'h', 'l', 'c', 'v']
        )
        return df['c'].reset_index(drop=True).iloc[-BTC_CORR_LIMIT:]
    except Exception as e:
        print(f'[Corr] Не удалось загрузить BTC: {e}')
        return None


def is_correlated_with_btc(symbol: str, btc_closes: pd.Series) -> bool:
    """
    Возвращает True если монета слишком коррелирует с BTC (|corr| >= BTC_CORR_MAX).
    Корреляция считается по доходностям (pct_change) — более точный метод.
    При ошибке — возвращает False (не блокируем монету если нет данных).
    """
    try:
        df = pd.DataFrame(
            exchange.fetch_ohlcv(symbol, BTC_CORR_TIMEFRAME, limit=BTC_CORR_LIMIT + 5),
            columns=['ts', 'o', 'h', 'l', 'c', 'v']
        )
        closes = df['c'].reset_index(drop=True).iloc[-BTC_CORR_LIMIT:]
        if len(closes) < 10 or len(btc_closes) < 10:
            return False
        sym_ret = closes.pct_change().dropna()
        btc_ret = btc_closes.pct_change().dropna()
        n = min(len(sym_ret), len(btc_ret))
        correlation = float(np.corrcoef(sym_ret.iloc[-n:].values, btc_ret.iloc[-n:].values)[0, 1])
        if abs(correlation) >= BTC_CORR_MAX:
            print(f'[Corr] {symbol}: отсев — корреляция BTC {correlation:.2f}')
            return True
        return False
    except Exception:
        return False

# =====================================================================
# ПОТОКОБЕЗОПАСНАЯ ОЧЕРЕДЬ ДЛЯ SQLite
# =====================================================================
_db_queue = queue.Queue()
DB_PATH   = 'paper_trading.db'


def db_worker():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('PRAGMA journal_mode=WAL')
    while True:
        try:
            task = _db_queue.get()
            if task is None:
                break
            sql, params, result_event, result_holder = task
            try:
                cur = conn.execute(sql, params)
                conn.commit()
                if result_holder is not None:
                    result_holder['lastrowid'] = cur.lastrowid
            except Exception as e:
                print(f'[DB Worker] Ошибка: {e} | SQL: {sql}')
            finally:
                if result_event:
                    result_event.set()
        except Exception as e:
            print(f'[DB Worker] Критическая ошибка: {e}')


def _write_db(sql: str, params: tuple = ()):
    _db_queue.put((sql, params, None, None))


def _write_db_sync(sql: str, params: tuple = ()):
    event  = threading.Event()
    holder = {}
    _db_queue.put((sql, params, event, holder))
    event.wait()
    return holder.get('lastrowid')


def _read_db(sql: str, params: tuple = (), fetchone: bool = False):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute('PRAGMA journal_mode=WAL')
    try:
        cur = conn.execute(sql, params)
        return cur.fetchone() if fetchone else cur.fetchall()
    finally:
        conn.close()

# =====================================================================
# ИНИЦИАЛИЗАЦИЯ БАЗЫ ДАННЫХ
# =====================================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('PRAGMA journal_mode=WAL')
    c = conn.cursor()

    c.execute('CREATE TABLE IF NOT EXISTS wallet (balance REAL)')
    if not c.execute('SELECT balance FROM wallet').fetchone():
        c.execute('INSERT INTO wallet VALUES (?)', (STARTING_BALANCE,))

    c.execute('''CREATE TABLE IF NOT EXISTS active_trades (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol        TEXT UNIQUE,
        side          TEXT,
        entry_price   REAL,
        size          REAL,
        sl            REAL,
        tp            REAL,
        is_trailing   INTEGER DEFAULT 0,
        trailing_sl   REAL    DEFAULT NULL,
        partial_price REAL    DEFAULT NULL,
        entry_time    TEXT,
        entry_balance REAL,
        scanner       TEXT    DEFAULT 'ema',
        macd_active   INTEGER DEFAULT 0
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
        balance_after REAL,
        scanner       TEXT DEFAULT 'ema'
    )''')

    conn.commit()

    # Миграция
    existing_at = [r[1] for r in conn.execute('PRAGMA table_info(active_trades)').fetchall()]
    existing_tl = [r[1] for r in conn.execute('PRAGMA table_info(trade_log)').fetchall()]

    for col, sql in {
        'entry_balance': 'ALTER TABLE active_trades ADD COLUMN entry_balance REAL',
        'is_trailing':   'ALTER TABLE active_trades ADD COLUMN is_trailing INTEGER DEFAULT 0',
        'trailing_sl':   'ALTER TABLE active_trades ADD COLUMN trailing_sl REAL DEFAULT NULL',
        'partial_price': 'ALTER TABLE active_trades ADD COLUMN partial_price REAL DEFAULT NULL',
        'scanner':       "ALTER TABLE active_trades ADD COLUMN scanner TEXT DEFAULT 'ema'",
        'macd_active':   'ALTER TABLE active_trades ADD COLUMN macd_active INTEGER DEFAULT 0',
    }.items():
        if col not in existing_at:
            try:
                conn.execute(sql); conn.commit()
                print(f'[DB] +{col} в active_trades')
            except Exception as e:
                print(f'[DB] Миграция {col}: {e}')

    if 'scanner' not in existing_tl:
        try:
            conn.execute("ALTER TABLE trade_log ADD COLUMN scanner TEXT DEFAULT 'ema'")
            conn.commit()
            print('[DB] +scanner в trade_log')
        except Exception as e:
            print(f'[DB] Миграция scanner в trade_log: {e}')

    conn.close()

# =====================================================================
# КЛАВИАТУРА
# =====================================================================
def main_keyboard():
    m = types.ReplyKeyboardMarkup(resize_keyboard=True)
    m.row('💰 БАЛАНС', '⚔️ СДЕЛКИ')
    m.row('📈 СТАТИСТИКА', '📊 БЭКТЕСТ')
    m.row('🔍 РАДАР', '📡 СТАТУС')
    m.row('👀 ВОТЧ-ЛИСТ', '🧠 АНАЛИЗ МОНЕТЫ')
    m.row('🚪 ЗАКРЫТЬ ВСЕ СДЕЛКИ')
    return m

# =====================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ БЭКТЕСТА (чистый numpy, без NaN)
# =====================================================================
def _bt_ema(arr: np.ndarray, n: int) -> np.ndarray:
    """EMA без NaN — первые значения заполняются SMA."""
    result = np.empty(len(arr))
    result[0] = arr[0]
    k = 2.0 / (n + 1)
    for i in range(1, len(arr)):
        result[i] = arr[i] * k + result[i - 1] * (1 - k)
    return result

def _bt_sma(arr: np.ndarray, n: int) -> np.ndarray:
    """SMA без NaN — первые значения равны первому доступному SMA."""
    result = np.empty(len(arr))
    cumsum = np.cumsum(arr)
    for i in range(len(arr)):
        if i < n:
            result[i] = cumsum[i] / (i + 1)
        else:
            result[i] = (cumsum[i] - cumsum[i - n]) / n
    return result

def _bt_rsi(arr: np.ndarray, n: int = 14) -> np.ndarray:
    """RSI без NaN."""
    result = np.full(len(arr), 50.0)
    delta = np.diff(arr, prepend=arr[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = _bt_sma(gain, n)
    avg_loss = _bt_sma(loss, n)
    for i in range(len(arr)):
        if avg_loss[i] == 0:
            result[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            result[i] = 100.0 - 100.0 / (1.0 + rs)
    return result

def _bt_atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 14) -> np.ndarray:
    """ATR без NaN."""
    tr = np.empty(len(h))
    tr[0] = h[0] - l[0]
    for i in range(1, len(h)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    return _bt_sma(tr, n)

# =====================================================================
# СТРАТЕГИЯ 1: EMA Cross
# =====================================================================
class FortressBT(Strategy):
    def init(self):
        self.ema12  = self.I(_bt_ema, self.data.Close, 12,  name='EMA12')
        self.ema26  = self.I(_bt_ema, self.data.Close, 26,  name='EMA26')
        self.atr    = self.I(_bt_atr, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        self.vol_ma = self.I(_bt_sma, self.data.Volume, 20, name='VolMA')

    def next(self):
        if len(self.data.Close) < 30:
            return
        price = self.data.Close[-1]
        v     = self.data.Volume
        trend_ok      = self.ema12[-1] > self.ema26[-1]
        vol_growing   = v[-1] > v[-2] > v[-3]
        price_growing = self.data.Close[-1] > self.data.Close[-2] > self.data.Close[-3]
        vol_spike     = v[-1] > self.vol_ma[-1] * 1.5
        signals       = sum([vol_growing, price_growing, vol_spike])
        if trend_ok and signals >= 2 and not self.position:
            sl  = price - self.atr[-1] * ATR_MULT_SL
            if sl <= 0 or sl >= price:
                return
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
        self.atr     = self.I(_bt_atr, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        self.rsi     = self.I(_bt_rsi, self.data.Close, 14, name='RSI')
        self.ema50   = self.I(_bt_ema, self.data.Close, 50, name='EMA50')
        self.vol_ma  = self.I(_bt_sma, self.data.Volume, 20, name='VolMA')
        self.highest = self.I(
            lambda x: pd.Series(x).shift(1).rolling(20).max().bfill().ffill().values,
            self.data.Close, name='Highest'
        )

    def next(self):
        if len(self.data.Close) < 55:
            return
        price     = self.data.Close[-1]
        trend_ok  = price > self.ema50[-1]
        breakout  = price > self.highest[-1]
        rsi_ok    = 45 < self.rsi[-1] < 72
        volume_ok = self.data.Volume[-1] > self.vol_ma[-1] * 1.3
        if trend_ok and breakout and rsi_ok and volume_ok and not self.position:
            sl = price - self.atr[-1] * 1.5
            if sl <= 0 or sl >= price:
                return
            tp = price + abs(price - sl) * 3.5
            self.buy(sl=sl, tp=tp)

# =====================================================================
# СТРАТЕГИЯ 3: Volatility Squeeze
# =====================================================================
class SqueezeBT(Strategy):
    def init(self):
        self.atr    = self.I(_bt_atr, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        self.rsi    = self.I(_bt_rsi, self.data.Close, 14, name='RSI')
        self.vol_ma = self.I(_bt_sma, self.data.Volume, 20, name='VolMA')
        c = pd.Series(self.data.Close)
        bb_lower, bb_mid, bb_upper = _bbands(c, n=20, std=2.0)
        self._bb_lower = bb_lower.bfill().ffill().values
        self._bb_mid   = bb_mid.bfill().ffill().values
        self._bb_upper = bb_upper.bfill().ffill().values

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
        width_prev = (self._bb_upper[i-5] - self._bb_lower[i-5]) / mid_prev
        squeeze   = width_now < width_prev * 0.85
        breakout  = price > self._bb_upper[i]
        volume_ok = self.data.Volume[-1] > self.vol_ma[-1] * 2.5
        rsi_ok    = 55 < self.rsi[-1] < 80
        if squeeze and breakout and volume_ok and rsi_ok and not self.position:
            sl = price - self.atr[-1] * 2.0
            if sl <= 0 or sl >= price:
                return
            tp = price + abs(price - sl) * 4.0
            self.buy(sl=sl, tp=tp)

# =====================================================================
# СТРАТЕГИЯ 4: Impulse Bar
# =====================================================================
class ImpulseBT(Strategy):
    def init(self):
        pass

    def next(self):
        if len(self.data.Close) < 55:
            return
        opens  = pd.Series(self.data.Open)
        highs  = pd.Series(self.data.High)
        lows   = pd.Series(self.data.Low)
        closes = pd.Series(self.data.Close)
        bodies   = (closes - opens).abs()
        avg_body = bodies.iloc[-22:-2].mean()
        if avg_body == 0:
            return
        big_idx  = bodies.iloc[-22:-2].idxmax()
        big_body = bodies.iloc[big_idx]
        if big_body < avg_body * 3.0:
            return
        bar_age = len(closes) - 1 - big_idx
        if bar_age > 10:
            return
        big_open  = opens.iloc[big_idx]
        big_close = closes.iloc[big_idx]
        big_high  = highs.iloc[big_idx]
        big_low   = lows.iloc[big_idx]
        bar_range = big_high - big_low
        if bar_range == 0:
            return
        is_bull = big_close > big_open
        if is_bull:
            if (big_close - big_low) / bar_range < 0.85:
                return
            zone_bottom = big_open
            zone_top    = (big_open + big_close) / 2
        else:
            if (big_high - big_close) / bar_range < 0.85:
                return
            zone_bottom = (big_open + big_close) / 2
            zone_top    = big_open
        price = self.data.Close[-1]
        if is_bull and not self.position:
            if zone_bottom <= price <= zone_top:
                sl_dist = price - big_low
                if sl_dist <= 0:
                    return
                self.buy(sl=big_low, tp=price + sl_dist * 2.5)
        elif not is_bull and not self.position:
            if zone_bottom <= price <= zone_top:
                sl_dist = big_high - price
                if sl_dist <= 0:
                    return
                self.sell(sl=big_high, tp=price - sl_dist * 2.5)

# =====================================================================
# СТРАТЕГИЯ 5: MACD Reversal (бэктест — LONG + SHORT)
# =====================================================================
class MACDReversalBT(Strategy):
    macd_fast   = 12
    macd_slow   = 26
    macd_signal = 9

    def init(self):
        close      = pd.Series(self.data.Close)
        ml_s, sl_s = _macd(close, self.macd_fast, self.macd_slow, self.macd_signal)
        ml_vals    = ml_s.fillna(0).values.copy()
        sl_vals    = sl_s.fillna(0).values.copy()
        self.ml     = self.I(lambda: ml_vals.copy(), name='MACD')
        self.sl_ind = self.I(lambda: sl_vals.copy(), name='Signal')

    def next(self):
        if len(self.data.Close) < self.macd_slow + self.macd_signal + 5:
            return
        if crossover(self.ml, self.sl_ind):
            if self.position.is_short:
                self.position.close()
            if not self.position:
                self.buy()
        elif crossover(self.sl_ind, self.ml):
            if self.position.is_long:
                self.position.close()
            if not self.position:
                self.sell()

# =====================================================================
# АНАЛИЗ СИМВОЛА — EMA Cross
# =====================================================================
# =====================================================================
def analyze_symbol(symbol: str, btc_closes: pd.Series | None = None):
    """
    Фильтры:
      - Объём 24ч > 120M USDT
      - NATR > 1.0%
      - Корреляция с BTC < 0.8 (монета живёт своей жизнью)
      - Цена выше EMA50 на 4H
      - ADX > 25 на 4H (есть тренд, не боковик)
      - EMA12 > EMA26 на 1H
      - 3 из 4: vol_growing, price_growing, vol_spike×1.5, context_ok
      - Фильтр импульсного бара
      - Все решения на iloc[-2] (закрытая свеча)
    """
    # ── Фильтр корреляции с BTC ───────────────────────────────────────
    if btc_closes is not None and is_correlated_with_btc(symbol, btc_closes):
        return None

    # ── 4H: тренд + ADX фильтр ────────────────────────────────────────
    df4h = pd.DataFrame(
        exchange.fetch_ohlcv(symbol, '4h', limit=200),
        columns=['ts', 'o', 'h', 'l', 'c', 'v']
    )
    ema50_4h = _ema(df4h['c'], 50).iloc[-2]
    if pd.isna(ema50_4h) or df4h['c'].iloc[-2] < ema50_4h:
        return None

    # ADX фильтр — торгуем только в тренде (ADX > 25)
    # Ручная реализация ADX (совместима с pandas >= 2.0)
    try:
        h4 = df4h['h']; l4 = df4h['l']; c4 = df4h['c']
        tr  = pd.concat([h4 - l4, (h4 - c4.shift()).abs(), (l4 - c4.shift()).abs()], axis=1).max(axis=1)
        dm_plus  = ((h4 - h4.shift()) > (l4.shift() - l4)).astype(float) * (h4 - h4.shift()).clip(lower=0)
        dm_minus = ((l4.shift() - l4) > (h4 - h4.shift())).astype(float) * (l4.shift() - l4).clip(lower=0)
        n = 14
        atr14   = tr.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
        dip     = dm_plus.ewm(alpha=1/n,  min_periods=n, adjust=False).mean() / atr14 * 100
        dim     = dm_minus.ewm(alpha=1/n, min_periods=n, adjust=False).mean() / atr14 * 100
        dx      = ((dip - dim).abs() / (dip + dim).replace(0, np.nan) * 100).fillna(0)
        adx_val = dx.ewm(alpha=1/n, min_periods=n, adjust=False).mean().iloc[-2]
        if not pd.isna(adx_val) and adx_val < 25:
            print(f'[EMA] {symbol}: отсев — ADX {adx_val:.1f} < 25 (боковик)')
            return None
    except Exception:
        pass  # Если ADX не считается — пропускаем фильтр

    # ── 1H: индикаторы ───────────────────────────────────────────────
    df = pd.DataFrame(
        exchange.fetch_ohlcv(symbol, '1h', limit=60),
        columns=['ts', 'o', 'h', 'l', 'c', 'v']
    )
    c = df['c']; h = df['h']; l = df['l']; v = df['v']

    atr_s = _atr(h, l, c, 14)
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)

    atr_val = atr_s.iloc[-2]
    e12_val = ema12.iloc[-2]
    e26_val = ema26.iloc[-2]

    if pd.isna(atr_val) or atr_val <= 0: return None
    if pd.isna(e12_val):                  return None
    if pd.isna(e26_val):                  return None
    if (atr_val / c.iloc[-2]) * 100 < MIN_NATR:              return None
    if e12_val <= e26_val:                                    return None

    vol_ma        = v.rolling(20).mean()
    vol_growing   = v.iloc[-2] > v.iloc[-3] > v.iloc[-4]
    price_growing = c.iloc[-2] > c.iloc[-3] > c.iloc[-4]
    vol_spike     = v.iloc[-2] > vol_ma.iloc[-2] * 1.5

    bear_bars  = df[df['c'] < df['o']].iloc[:-1]
    context_ok = (c.iloc[-2] > bear_bars.loc[bear_bars['v'].idxmax(), 'o']
                  if len(bear_bars) > 0 else True)

    if sum([vol_growing, price_growing, vol_spike, context_ok]) < 3:
        return None

    # ── Фильтр импульсного бара ───────────────────────────────────────
    bodies   = (df['c'] - df['o']).abs()
    avg_body = bodies.iloc[-22:-2].mean()
    if avg_body > 0:
        big_idx  = bodies.iloc[-22:-2].idxmax()
        big_body = bodies.iloc[big_idx]
        bar_age  = len(df) - 1 - big_idx
        if big_body >= avg_body * 3.0 and bar_age <= 10:
            big_open  = df['o'].iloc[big_idx]
            big_close = df['c'].iloc[big_idx]
            midpoint  = (big_open + big_close) / 2
            if big_close > big_open:
                in_zone = big_open <= c.iloc[-2] <= midpoint
            else:
                in_zone = midpoint <= c.iloc[-2] <= big_open
            if not in_zone:
                return None

    return {'symbol': symbol, 'price': c.iloc[-2], 'atr': atr_val}

# =====================================================================
# АНАЛИЗ СИМВОЛА — MACD 30м
# =====================================================================
def analyze_symbol_macd(symbol: str, btc_closes: pd.Series | None = None):
    """
    Фильтры:
      - Объём 24ч > 120M USDT (проверяется до вызова)
      - NATR > 1.0%
      - Корреляция с BTC < 0.8 (монета живёт своей жизнью)
      - MACD пересекает Signal на 1ч (последняя завершённая свеча)
    Возвращает {'symbol', 'price', 'side', 'atr'} или None
    """
    # ── Фильтр корреляции с BTC ───────────────────────────────────────
    if btc_closes is not None and is_correlated_with_btc(symbol, btc_closes):
        return None

    df = pd.DataFrame(
        exchange.fetch_ohlcv(symbol, '1h', limit=100),
        columns=['ts', 'o', 'h', 'l', 'c', 'v']
    )
    c = df['c']; h = df['h']; l = df['l']

    atr_val = _atr(h, l, c, 14).iloc[-2]
    if pd.isna(atr_val) or atr_val <= 0:
        return None
    if (atr_val / c.iloc[-2]) * 100 < MIN_NATR:
        return None

    ml, sl = _macd(c, fast=12, slow=26, signal=9)

    if any(pd.isna(ml.iloc[i]) or pd.isna(sl.iloc[i]) for i in [-2, -3]):
        return None

    bull_cross = ml.iloc[-3] < sl.iloc[-3] and ml.iloc[-2] > sl.iloc[-2]
    bear_cross = ml.iloc[-3] > sl.iloc[-3] and ml.iloc[-2] < sl.iloc[-2]

    if bull_cross:
        return {'symbol': symbol, 'price': c.iloc[-2], 'side': 'long', 'atr': atr_val}
    if bear_cross:
        return {'symbol': symbol, 'price': c.iloc[-2], 'side': 'short', 'atr': atr_val}
    return None

# =====================================================================
# РАДАР ПАМПА
# =====================================================================
def run_radar_scan() -> list:
    tickers    = exchange.fetch_tickers()
    candidates = [
        s for s, d in tickers.items()
        if '/USDT:USDT' in s
        and (d.get('quoteVolume') or 0) >= MIN_VOLUME_24H
    ]
    found = []
    for i, sym in enumerate(candidates, 1):
        print(f'[Radar] [{i}/{len(candidates)}] {sym}     ', end='\r')
        try:
            r = _analyze_radar_symbol(sym)
            if r:
                found.append(r)
        except (ccxt.NetworkError, ccxt.ExchangeError):
            pass
        except Exception as e:
            print(f'[Radar] {sym}: {e}')
    print()
    found.sort(key=lambda x: (x['signals'], x['change_6h']), reverse=True)
    return found


def _analyze_radar_symbol(symbol: str):
    df = pd.DataFrame(
        exchange.fetch_ohlcv(symbol, '1h', limit=24),
        columns=['ts', 'o', 'h', 'l', 'c', 'v']
    )
    c = df['c']; v = df['v']; h = df['h']; l = df['l']
    atr  = _atr(h, l, c, 14).iloc[-1]
    natr = (atr / c.iloc[-1]) * 100
    if natr < 1.5:
        return None
    vol_ma          = v.rolling(10).mean()
    vol_growing     = all(v.iloc[-i] > v.iloc[-i-1] for i in range(1, 4))
    price_growing   = all(c.iloc[-i] > c.iloc[-i-1] for i in range(1, 4))
    vol_spike       = v.iloc[-1] > vol_ma.iloc[-1] * 1.8
    change_6h       = (c.iloc[-1] / c.iloc[-7] - 1) * 100
    vol_x           = v.iloc[-1] / max(vol_ma.iloc[-1], 0.0001)
    signals         = sum([vol_growing, price_growing, vol_spike, change_6h > 5.0])
    if signals < 3:
        return None
    return {
        'symbol': symbol, 'price': c.iloc[-1], 'change_6h': change_6h,
        'vol_x': vol_x, 'natr': natr, 'signals': signals,
        'vol_growing': vol_growing, 'price_growing': price_growing,
        'vol_spike': vol_spike,
    }


def _format_radar_results(found: list, title: str) -> str:
    text = f'🔍 *{title}* | {datetime.now().strftime("%H:%M")}\nНайдено: {len(found)}\n\n'
    for r in found:
        flags = []
        if r['vol_growing']:    flags.append('📈 объём растёт')
        if r['price_growing']:  flags.append('🕯 свечи вверх')
        if r['vol_spike']:      flags.append('⚡ всплеск объёма')
        if r['change_6h'] > 5: flags.append(f'🚀 +{r["change_6h"]:.1f}%')
        text += (
            f'⚡ *{r["symbol"]}*\n'
            f'  `{r["price"]:.6g}` | NATR `{r["natr"]:.1f}%` | '
            f'Vol x`{r["vol_x"]:.1f}`\n'
            f'  {" | ".join(flags)} | Сигналов `{r["signals"]}/4`\n\n'
        )
    return text[:4000]

# =====================================================================
# ОТКРЫТИЕ СДЕЛКИ — EMA Cross
# =====================================================================
def open_trade_ema(symbol: str, price: float, atr_val: float) -> bool:
    if _read_db('SELECT 1 FROM active_trades WHERE symbol=?',
                (symbol,), fetchone=True):
        return False

    try:
        candles = exchange.fetch_ohlcv(symbol, '1h', limit=3)
        change  = (price / candles[-2][4] - 1) * 100
        if change > 5.0:
            print(f'[EMA] Пропуск {symbol}: +{change:.1f}% за час')
            return False
    except (ccxt.NetworkError, ccxt.ExchangeError):
        pass

    # Лимит открытых EMA сделок
    ema_count = _read_db(
        'SELECT COUNT(*) FROM active_trades WHERE macd_active=0',
        fetchone=True)[0]
    if ema_count >= MAX_EMA_TRADES:
        print(f'[EMA] Лимит {MAX_EMA_TRADES} сделок достигнут — пропуск {symbol}')
        return False

    balance = _read_db('SELECT balance FROM wallet', fetchone=True)[0]
    trades  = _read_db('SELECT symbol,side,entry_price,size FROM active_trades')
    unrealized = 0.0
    for sym, sd, ep, sz in trades:
        try:
            p = exchange.fetch_ticker(sym)['last']
            unrealized += (p - ep) * sz if sd == 'long' else (ep - p) * sz
        except Exception:
            pass
    equity  = balance + unrealized
    sl_dist = atr_val * ATR_MULT_SL
    sl      = price - sl_dist
    tp      = price + sl_dist * REWARD_RATIO
    size    = (equity * RISK_PER_TRADE) / sl_dist

    try:
        _write_db_sync(
            'INSERT INTO active_trades '
            '(symbol,side,entry_price,size,sl,tp,'
            ' is_trailing,trailing_sl,partial_price,'
            ' entry_time,entry_balance,scanner,macd_active) '
            'VALUES (?,?,?,?,?,?,0,NULL,NULL,?,?,?,0)',
            (symbol, 'long', price, size, sl, tp,
             datetime.now().strftime('%Y-%m-%d %H:%M:%S'), balance, 'ema')
        )
    except Exception as e:
        print(f'[EMA] Не удалось открыть {symbol}: {e}')
        return False

    bot.send_message(USER_ID,
        f'🚀 *EMA ВХОД: {symbol}*\n'
        f'Тип: *LONG* | Цена: `{price:.6g}`\n'
        f'🔴 SL: `{sl:.6g}`\n'
        f'🎯 TP1: `{price + sl_dist * 2.5:.6g}` → трейлинг\n'
        f'💵 Риск: `{balance * RISK_PER_TRADE:.2f}` USDT',
        parse_mode='Markdown'
    )
    return True

# =====================================================================
# ОТКРЫТИЕ СДЕЛКИ — MACD
# =====================================================================
def open_trade_macd(symbol: str, price: float, side: str, atr_val: float) -> bool:
    if _read_db('SELECT 1 FROM active_trades WHERE symbol=?',
                (symbol,), fetchone=True):
        return False

    # Лимит открытых MACD сделок
    macd_count = _read_db(
        'SELECT COUNT(*) FROM active_trades WHERE macd_active=1',
        fetchone=True)[0]
    if macd_count >= MAX_MACD_TRADES:
        print(f'[MACD] Лимит {MAX_MACD_TRADES} сделок достигнут — пропуск {symbol}')
        return False

    balance = _read_db('SELECT balance FROM wallet', fetchone=True)[0]
    trades  = _read_db('SELECT symbol,side,entry_price,size FROM active_trades')
    unrealized = 0.0
    for sym, sd, ep, sz in trades:
        try:
            p = exchange.fetch_ticker(sym)['last']
            unrealized += (p - ep) * sz if sd == 'long' else (ep - p) * sz
        except Exception:
            pass
    equity = balance + unrealized

    # SL и TP на основе ATR (RR = 1:2.5)
    sl_dist = atr_val * MACD_ATR_SL
    tp_dist = atr_val * MACD_ATR_TP
    if side == 'long':
        sl = price - sl_dist
        tp = price + tp_dist
    else:
        sl = price + sl_dist
        tp = price - tp_dist

    # Размер позиции: риск 1% от equity / дистанция SL
    size = (equity * RISK_PER_TRADE) / sl_dist

    try:
        _write_db_sync(
            'INSERT INTO active_trades '
            '(symbol,side,entry_price,size,sl,tp,'
            ' is_trailing,trailing_sl,partial_price,'
            ' entry_time,entry_balance,scanner,macd_active) '
            'VALUES (?,?,?,?,?,?,0,NULL,NULL,?,?,?,1)',
            (symbol, side, price, size, sl, tp,
             datetime.now().strftime('%Y-%m-%d %H:%M:%S'), balance, 'macd')
        )
    except Exception as e:
        print(f'[MACD] Не удалось открыть {symbol}: {e}')
        return False

    icon = '🟢' if side == 'long' else '🔴'
    bot.send_message(USER_ID,
        f'{icon} *MACD ВХОД: {symbol}*\n'
        f'Тип: *{side.upper()}* | Цена: `{price:.6g}`\n'
        f'🔴 SL: `{sl:.6g}` | 🎯 TP: `{tp:.6g}`\n'
        f'📏 R:R: `1 : 2.5` | Риск: `{equity * RISK_PER_TRADE:.2f}` USDT',
        parse_mode='Markdown'
    )
    return True

# =====================================================================
# ЗАКРЫТИЕ СДЕЛКИ
# =====================================================================
def close_trade(trade: tuple, exit_price: float, reason: str):
    (tid, symbol, side, entry_price, size, sl, tp,
     is_trailing, trailing_sl, partial_price,
     entry_time, entry_balance, scanner, macd_active) = trade

    pnl = (exit_price - entry_price) * size if side == 'long' \
          else (entry_price - exit_price) * size

    result = 'WIN' if pnl > 0 else ('LOSS' if pnl < 0 else 'BREAKEVEN')

    # Атомарное обновление баланса — нет race condition
    _write_db('UPDATE wallet SET balance = balance + ?', (pnl,))
    _write_db('DELETE FROM active_trades WHERE id=?', (tid,))

    # Читаем итоговый баланс уже после атомарного обновления
    new_balance = _read_db('SELECT balance FROM wallet', fetchone=True)[0]

    _write_db(
        'INSERT INTO trade_log '
        '(symbol,side,entry_price,exit_price,size,pnl,result,'
        ' entry_time,exit_time,balance_after,scanner) '
        'VALUES (?,?,?,?,?,?,?,?,?,?,?)',
        (symbol, side, entry_price, exit_price, size, pnl, result,
         entry_time, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
         new_balance, scanner)
    )

    icon = '🏆' if result == 'WIN' else '💀'
    lbl  = '📊 EMA' if scanner == 'ema' else '📉 MACD'
    bot.send_message(USER_ID,
        f'{icon} *ЗАКРЫТО: {symbol}* ({lbl})\n'
        f'{reason} | {side.upper()} → `{exit_price:.6g}`\n'
        f'PnL: `{pnl:+.2f}` USDT | Баланс: `{new_balance:.2f}` USDT',
        parse_mode='Markdown'
    )

# =====================================================================
# МОНИТОРИНГ СДЕЛОК (только EMA Cross)
# =====================================================================
async def monitor_trades():
    while True:
        try:
            trades = _read_db(
                'SELECT id,symbol,side,entry_price,size,sl,tp,'
                '       is_trailing,trailing_sl,partial_price,'
                '       entry_time,entry_balance,scanner,macd_active '
                'FROM active_trades'
            )
            for trade in trades:
                (tid, symbol, side, entry_price, size, sl, tp,
                 is_trailing, trailing_sl, partial_price,
                 entry_time, entry_balance, scanner, macd_active) = trade

                if macd_active:
                    # MACD сделки — мониторим SL и TP как обычные сделки
                    try:
                        cur_price   = exchange.fetch_ticker(symbol)['last']
                        candles_1m  = exchange.fetch_ohlcv(symbol, '1m', limit=2)
                        c_low       = candles_1m[-2][3]
                        c_high      = candles_1m[-2][2]

                        if side == 'long':
                            sl_hit = cur_price <= sl or c_low  <= sl
                            tp_hit = cur_price >= tp or c_high >= tp
                            if sl_hit:
                                close_trade(trade, min(cur_price, c_low), '🔴 STOP-LOSS')
                            elif tp_hit:
                                close_trade(trade, max(cur_price, c_high), '🎯 TAKE-PROFIT')
                        else:  # short
                            sl_hit = cur_price >= sl or c_high >= sl
                            tp_hit = cur_price <= tp or c_low  <= tp
                            if sl_hit:
                                close_trade(trade, max(cur_price, c_high), '🔴 STOP-LOSS')
                            elif tp_hit:
                                close_trade(trade, min(cur_price, c_low), '🎯 TAKE-PROFIT')
                    except (ccxt.NetworkError, ccxt.ExchangeError):
                        pass
                    continue

                try:
                    ticker      = exchange.fetch_ticker(symbol)
                    price       = ticker['last']
                    candles     = exchange.fetch_ohlcv(symbol, '1m', limit=2)
                    candle_low  = candles[-2][3]
                    candle_high = candles[-2][2]

                    df_15m = pd.DataFrame(
                        exchange.fetch_ohlcv(symbol, '15m', limit=20),
                        columns=['ts', 'o', 'h', 'l', 'c', 'v']
                    )
                    atr = _atr(df_15m['h'], df_15m['l'], df_15m['c'], 14).iloc[-1]

                    sl_hit  = price <= sl or candle_low  <= sl
                    tp1     = entry_price + (entry_price - sl) * 2.5
                    tp1_hit = price >= tp1 or candle_high >= tp1

                    if not is_trailing:
                        if sl_hit:
                            close_trade(trade, min(price, candle_low),
                                        '🔴 STOP-LOSS')
                        elif tp1_hit:
                            close_price  = max(price, candle_high)
                            partial_size = size * 0.5
                            pnl_partial  = (close_price - entry_price) * partial_size
                            # Атомарное обновление — нет race condition
                            _write_db('UPDATE wallet SET balance = balance + ?',
                                      (pnl_partial,))
                            _write_db(
                                'UPDATE active_trades '
                                'SET size=?,is_trailing=1,'
                                '    trailing_sl=?,partial_price=? '
                                'WHERE id=?',
                                (partial_size, entry_price, close_price, tid)
                            )
                            bot.send_message(USER_ID,
                                f'🎯 *ЧАСТИЧНОЕ ЗАКРЫТИЕ: {symbol}*\n'
                                f'50% по `{close_price:.6g}` | '
                                f'PnL: `+{pnl_partial:.2f}` USDT\n'
                                f'SL → безубыток | Трейлинг активирован',
                                parse_mode='Markdown'
                            )
                    else:
                        new_tsl = max(trailing_sl, price - atr * 1.0)
                        if price <= trailing_sl or candle_low <= trailing_sl:
                            close_trade(trade, min(price, candle_low),
                                        '📍 ТРЕЙЛИНГ СТОП')
                        elif new_tsl > trailing_sl:
                            _write_db(
                                'UPDATE active_trades SET trailing_sl=? WHERE id=?',
                                (new_tsl, tid)
                            )

                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    print(f'[Monitor] Сеть {symbol}: {e}')
                except Exception as e:
                    print(f'[Monitor] Ошибка {symbol}: {e}')

        except Exception as e:
            print(f'[Monitor] Критическая ошибка: {e}')

        await asyncio.sleep(MONITOR_INTERVAL)

# =====================================================================
# EMA CROSS СКАНЕР — по сетке каждые 15 мин
# =====================================================================
async def signal_hunter():
    wait = seconds_until_next_grid(EMA_GRID_MINUTES)
    print(f'[EMA] Первый скан через {wait/60:.1f} мин')
    await asyncio.sleep(wait)

    while True:
        try:
            now = datetime.now().strftime('%H:%M')
            bot.send_message(USER_ID,
                f'🔎 *EMA Cross скан* | {now}', parse_mode='Markdown')

            tickers = None
            for attempt in range(3):
                try:
                    tickers = exchange.fetch_tickers(); break
                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    print(f'[EMA] Попытка {attempt+1}/3: {e}')
                    await asyncio.sleep(30)

            if tickers is None:
                await asyncio.sleep(seconds_until_next_grid(EMA_GRID_MINUTES))
                continue

            candidates = [
                s for s, d in tickers.items()
                if '/USDT:USDT' in s and s not in BLACKLIST
                and (d.get('quoteVolume') or 0) >= MIN_VOLUME_24H
            ]

            # Загружаем BTC один раз для всего скана
            btc_closes = fetch_btc_closes()
            if btc_closes is None:
                print('[EMA] BTC данные недоступны — фильтр корреляции отключён')

            found = 0
            for i, sym in enumerate(candidates, 1):
                print(f'[EMA] [{i}/{len(candidates)}] {sym}     ', end='\r')
                try:
                    setup = analyze_symbol(sym, btc_closes)
                    if setup and open_trade_ema(
                            setup['symbol'], setup['price'], setup['atr']):
                        found += 1
                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    print(f'[EMA] Сеть {sym}: {e}')
                except Exception as e:
                    print(f'[EMA] Ошибка {sym}: {e}')
                await asyncio.sleep(0.3)

            print()
            bot.send_message(USER_ID,
                f'🏁 *EMA скан завершён* | Открыто: {found}',
                parse_mode='Markdown')

        except Exception as e:
            print(f'[EMA Scanner] Критическая ошибка: {e}')

        await asyncio.sleep(seconds_until_next_grid(EMA_GRID_MINUTES))

# =====================================================================
# MACD СКАНЕР — по сетке каждые 30 мин
# =====================================================================
async def macd_hunter():
    wait = seconds_until_next_grid(MACD_GRID_MINUTES)
    print(f'[MACD] Первый скан через {wait/60:.1f} мин')
    await asyncio.sleep(wait)

    while True:
        try:
            now = datetime.now().strftime('%H:%M')
            bot.send_message(USER_ID,
                f'📉 *MACD 30м скан* | {now}', parse_mode='Markdown')

            tickers = None
            for attempt in range(3):
                try:
                    tickers = exchange.fetch_tickers(); break
                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    print(f'[MACD] Попытка {attempt+1}/3: {e}')
                    await asyncio.sleep(30)

            if tickers is None:
                await asyncio.sleep(seconds_until_next_grid(MACD_GRID_MINUTES))
                continue

            candidates = [
                s for s, d in tickers.items()
                if '/USDT:USDT' in s and s not in BLACKLIST
                and (d.get('quoteVolume') or 0) >= MIN_VOLUME_24H
            ]

            # Загружаем BTC один раз для всего скана
            btc_closes = fetch_btc_closes()
            if btc_closes is None:
                print('[MACD] BTC данные недоступны — фильтр корреляции отключён')

            # ── Шаг 1: проверяем выходы из открытых MACD сделок ──────
            macd_trades = _read_db(
                'SELECT id,symbol,side,entry_price,size,sl,tp,'
                '       is_trailing,trailing_sl,partial_price,'
                '       entry_time,entry_balance,scanner,macd_active '
                'FROM active_trades WHERE macd_active=1'
            )
            closed = 0
            for trade in macd_trades:
                sym = trade[1]; side = trade[2]
                try:
                    # При развороте корреляцию не проверяем — уже в позиции
                    setup = analyze_symbol_macd(sym)
                    if setup and setup['side'] != side:
                        close_trade(trade, setup['price'], '🔄 MACD разворот')
                        closed += 1
                        open_trade_macd(sym, setup['price'], setup['side'], setup['atr'])
                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    print(f'[MACD Monitor] Сеть {sym}: {e}')
                except Exception as e:
                    print(f'[MACD Monitor] Ошибка {sym}: {e}')

            # ── Шаг 2: ищем новые входы ───────────────────────────────
            found_long = 0; found_short = 0
            for i, sym in enumerate(candidates, 1):
                print(f'[MACD] [{i}/{len(candidates)}] {sym}     ', end='\r')
                try:
                    setup = analyze_symbol_macd(sym, btc_closes)
                    if setup:
                        if open_trade_macd(sym, setup['price'], setup['side'], setup['atr']):
                            if setup['side'] == 'long':
                                found_long += 1
                            else:
                                found_short += 1
                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    print(f'[MACD] Сеть {sym}: {e}')
                except Exception as e:
                    print(f'[MACD] Ошибка {sym}: {e}')
                await asyncio.sleep(0.3)

            print()
            bot.send_message(USER_ID,
                f'🏁 *MACD скан завершён* | {now}\n'
                f'🟢 Long: {found_long} | 🔴 Short: {found_short} | '
                f'🔄 Закрыто: {closed}',
                parse_mode='Markdown')

        except Exception as e:
            print(f'[MACD Scanner] Критическая ошибка: {e}')

        await asyncio.sleep(seconds_until_next_grid(MACD_GRID_MINUTES))

# =====================================================================
# ВОТЧ-ЛИСТ — топ-5 монет близких к сигналу
# =====================================================================
WATCHLIST_TOP = 5

def _score_symbol(symbol: str) -> dict | None:
    """
    Считает скор готовности монеты к входу.
    Возвращает dict со скором и деталями или None при ошибке.
    Максимум: 16 очков (EMA 10 + MACD 6)
    """
    try:
        score    = 0
        details  = []
        direction = None  # 'long' или 'short'

        # ── EMA Cross скор (макс 10) ──────────────────────────────────
        df4h = pd.DataFrame(
            exchange.fetch_ohlcv(symbol, '4h', limit=200),
            columns=['ts', 'o', 'h', 'l', 'c', 'v']
        )
        ema50_4h = _ema(df4h['c'], 50).iloc[-2]

        df = pd.DataFrame(
            exchange.fetch_ohlcv(symbol, '1h', limit=60),
            columns=['ts', 'o', 'h', 'l', 'c', 'v']
        )
        c = df['c']; h = df['h']; l = df['l']; v = df['v']

        atr_val = _atr(h, l, c, 14).iloc[-2]
        e12_val = _ema(c, 12).iloc[-2]
        e26_val = _ema(c, 26).iloc[-2]
        price   = c.iloc[-2]

        # NATR
        natr = (atr_val / price * 100) if (not pd.isna(atr_val) and atr_val) else 0
        if natr >= MIN_NATR:
            score += 2
        else:
            details.append('низкая волатильность')

        # 4H тренд
        if not pd.isna(ema50_4h):
            if price > ema50_4h:
                score += 2
                direction = 'long'
            else:
                details.append('↓EMA50 4H')

        # EMA12 vs EMA26
        if not pd.isna(e12_val) and not pd.isna(e26_val) and e26_val != 0:
            diff_pct = abs(e12_val - e26_val) / e26_val * 100
            if e12_val > e26_val:
                score += 2
                direction = 'long'
                if diff_pct < 0.5:
                    details.append(f'EMA почти пересекаются ({diff_pct:.2f}%)')
            else:
                if diff_pct < 1.0:
                    score += 1
                    details.append(f'EMA близко к пересечению ({diff_pct:.2f}%)')

        # Доп. условия объёма и цены
        vol_ma = v.rolling(20).mean()
        if v.iloc[-2] > v.iloc[-3] > v.iloc[-4]:
            score += 1
        if c.iloc[-2] > c.iloc[-3] > c.iloc[-4]:
            score += 1
        if v.iloc[-2] > vol_ma.iloc[-2] * 1.5:
            score += 1
        if v.iloc[-2] > vol_ma.iloc[-2] * 1.2:
            score += 1

        # ── MACD 30м скор (макс 6) ────────────────────────────────────
        df30 = pd.DataFrame(
            exchange.fetch_ohlcv(symbol, '30m', limit=60),
            columns=['ts', 'o', 'h', 'l', 'c', 'v']
        )
        ml, sl = _macd(df30['c'], fast=12, slow=26, signal=9)

        if not pd.isna(ml.iloc[-2]) and not pd.isna(sl.iloc[-2]):
            macd_diff = ml.iloc[-2] - sl.iloc[-2]
            macd_diff_prev = ml.iloc[-3] - sl.iloc[-3]

            # Уже пересеклось
            if ml.iloc[-2] > sl.iloc[-2]:
                score += 3
                direction = direction or 'long'
                details.append('MACD ✅ бычье пересечение')
            elif ml.iloc[-2] < sl.iloc[-2]:
                score += 2
                direction = direction or 'short'

            # Сближаются?
            if abs(macd_diff) < abs(macd_diff_prev) * 0.5:
                score += 2
                details.append('MACD сближается ⚡')
            elif abs(macd_diff) < abs(macd_diff_prev) * 0.8:
                score += 1
                details.append('MACD почти пересечение')

        # Итог
        if score < 5:
            return None  # слишком слабый — не включать в вотч-лист

        # Цена входа — текущая цена (реальная)
        try:
            current_price = exchange.fetch_ticker(symbol)['last']
        except Exception:
            current_price = price

        return {
            'symbol':    symbol,
            'score':     score,
            'max_score': 16,
            'direction': direction or 'long',
            'price':     current_price,
            'natr':      natr,
            'details':   details,
        }

    except Exception:
        return None


def run_watchlist_scan(candidates: list) -> list:
    """
    Прогоняет кандидатов через скоринг и возвращает топ-5.
    Исключает монеты у которых уже открыта сделка.
    """
    active_symbols = {
        r[0] for r in _read_db('SELECT symbol FROM active_trades')
    }
    results = []
    total   = len(candidates)

    for i, sym in enumerate(candidates, 1):
        print(f'[Watch] [{i}/{total}] {sym}     ', end='\r')
        if sym in active_symbols:
            continue
        try:
            r = _score_symbol(sym)
            if r:
                results.append(r)
        except (ccxt.NetworkError, ccxt.ExchangeError):
            pass
        except Exception:
            pass

    print()
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:WATCHLIST_TOP]


def _format_watchlist(found: list) -> str:
    """Форматирует вотч-лист для Telegram."""
    now  = datetime.now().strftime('%H:%M')
    text = f'👀 *ВОТЧ-ЛИСТ* | {now}\nТоп-{WATCHLIST_TOP} монет близких к сигналу\n\n'

    medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
    for i, r in enumerate(found):
        pct      = r['score'] / r['max_score'] * 100
        dir_icon = '🟢 LONG' if r['direction'] == 'long' else '🔴 SHORT'
        bar      = '█' * int(pct / 10) + '░' * (10 - int(pct / 10))

        detail_str = ' | '.join(r['details']) if r['details'] else 'близко к сигналу'

        text += (
            f'{medals[i]} *{r["symbol"]}*\n'
            f'  `{bar}` {pct:.0f}%\n'
            f'  Направление: {dir_icon}\n'
            f'  Цена входа ≈ `{r["price"]:.6g}`\n'
            f'  NATR: `{r["natr"]:.1f}%`\n'
            f'  {detail_str}\n\n'
        )

    text += '⚠️ _Это зона наблюдения — не сигнал к действию_'
    return text[:4000]


# =====================================================================
# АВТО-ВОТЧ-ЛИСТ — раз в час ровно
# =====================================================================
WATCHLIST_INTERVAL = 3600  # каждые 60 минут

async def watchlist_hunter():
    # Ждём до следующего часа (00 минут)
    now     = __import__('datetime').datetime.now()
    wait    = (60 - now.minute) * 60 - now.second
    if wait < 10:
        wait += 3600
    print(f'[Watch] Первый скан через {wait/60:.1f} мин')
    await asyncio.sleep(wait)

    while True:
        try:
            tickers = exchange.fetch_tickers()
            candidates = [
                s for s, d in tickers.items()
                if '/USDT:USDT' in s and s not in BLACKLIST
                and (d.get('quoteVolume') or 0) >= MIN_VOLUME_24H
            ]
            watchlist = run_watchlist_scan(candidates)
            if watchlist:
                bot.send_message(USER_ID,
                    _format_watchlist(watchlist),
                    parse_mode='Markdown')
            else:
                print('[Watch] Вотч-лист: монет не найдено')
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            print(f'[Watch] Сеть: {e}')
        except Exception as e:
            print(f'[Watch] Ошибка: {e}')

        await asyncio.sleep(WATCHLIST_INTERVAL)

# =====================================================================
# АВТО-РАДАР
# =====================================================================
async def pump_radar():
    alerted = set()
    while True:
        await asyncio.sleep(RADAR_INTERVAL)
        try:
            found = [r for r in run_radar_scan()
                     if r['symbol'] not in alerted]
            for r in found:
                alerted.add(r['symbol'])
            if len(alerted) > 100:
                alerted.clear()
            if found:
                bot.send_message(USER_ID,
                    _format_radar_results(found, 'РАДАР ПАМПА (авто)'),
                    parse_mode='Markdown')
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            print(f'[Radar] Сеть: {e}')
        except Exception as e:
            print(f'[Radar] Ошибка: {e}')

# =====================================================================
# TELEGRAM HANDLERS
# =====================================================================
@bot.message_handler(commands=['start'])
def cmd_start(msg):
    if msg.from_user.id != USER_ID:
        return
    bot.send_message(USER_ID,
        '🛡️ *Fortress V4.5*\n\n'
        '📊 EMA Cross: каждые 15 мин по сетке\n'
        '📉 MACD 1ч: каждые 30 мин по сетке\n'
        'Статистика общая + раздельная по сканерам',
        parse_mode='Markdown', reply_markup=main_keyboard()
    )

@bot.message_handler(func=lambda m: m.text == '💰 БАЛАНС')
def cmd_balance(msg):
    if msg.from_user.id != USER_ID:
        return
    balance = _read_db('SELECT balance FROM wallet', fetchone=True)[0]
    trades  = _read_db('SELECT symbol,side,entry_price,size FROM active_trades')
    unrealized = 0.0
    for sym, side, ep, sz in trades:
        try:
            p = exchange.fetch_ticker(sym)['last']
            unrealized += (p - ep) * sz if side == 'long' else (ep - p) * sz
        except (ccxt.NetworkError, ccxt.ExchangeError):
            pass
    equity = balance + unrealized
    bot.send_message(USER_ID,
        f'🏦 *БАЛАНС*\n'
        f'Доступно: `{balance:.2f}` USDT\n'
        f'Открытый PnL: `{unrealized:+.2f}` USDT\n'
        f'💰 *Equity: {equity:.2f} USDT*\n'
        f'{"🟢" if equity >= STARTING_BALANCE else "🔴"} '
        f'{((equity / STARTING_BALANCE - 1) * 100):+.2f}%',
        parse_mode='Markdown'
    )

@bot.message_handler(func=lambda m: m.text == '⚔️ СДЕЛКИ')
def cmd_trades(msg):
    if msg.from_user.id != USER_ID:
        return
    trades = _read_db(
        'SELECT symbol,side,entry_price,sl,tp,'
        '       is_trailing,trailing_sl,entry_time,scanner,macd_active,size '
        'FROM active_trades'
    )
    if not trades:
        bot.send_message(USER_ID, '⚔️ Активных сделок нет.')
        return
    text = '⚔️ *АКТИВНЫЕ СДЕЛКИ:*\n\n'
    for sym, side, ep, sl, tp, is_tr, tsl, et, scn, macd_a, sz in trades:
        # Дата и время открытия
        if et and len(et) >= 16:
            dt_part = et[5:10]   # MM-DD
            tm_part = et[11:16]  # HH:MM
            t = f'{dt_part} {tm_part}'
        else:
            t = '—'
        icon = '📊' if scn == 'ema' else '📉'

        # Текущий PnL
        try:
            cur_price = exchange.fetch_ticker(sym)['last']
            cur_pnl   = (cur_price - ep) * sz if side == 'long' \
                        else (ep - cur_price) * sz
            pnl_emoji = '🟢' if cur_pnl >= 0 else '🔴'
            pnl_str   = f'{pnl_emoji} PnL: `{cur_pnl:+.2f}` USDT | Цена: `{cur_price:.6g}`\n  '
        except Exception:
            pnl_str = ''

        if macd_a:
            if sl and tp:
                mode = (f'🔴 SL: `{sl:.6g}` | 🎯 TP: `{tp:.6g}`\n  '
                        f'📏 R:R: `1 : 2.5`')
            else:
                mode = '🔄 Выход по MACD сигналу'
        elif is_tr:
            mode = (f'📍 Трейлинг SL: `{tsl:.6g}`\n  '
                    f'🔴 SL: `{sl:.6g}`')
        else:
            tp1 = ep + (ep - sl) * 2.5
            mode = (f'🎯 TP1: `{tp1:.6g}`\n  '
                    f'🔴 SL: `{sl:.6g}`\n  '
                    f'📏 R:R: `1 : 2.5`')
        text += f'{icon} *{sym}* | {side.upper()} | {t}\n  {pnl_str}{mode}\n\n'

    try:
        bot.send_message(USER_ID, text, parse_mode='Markdown')
    except Exception:
        clean = text.replace('*', '').replace('`', '').replace('_', '')
        bot.send_message(USER_ID, clean)

@bot.message_handler(func=lambda m: m.text == '📈 СТАТИСТИКА')
def cmd_stats(msg):
    if msg.from_user.id != USER_ID:
        return
    rows = _read_db(
        'SELECT pnl,result,balance_after,scanner FROM trade_log ORDER BY id'
    )
    active = _read_db('SELECT COUNT(*) FROM active_trades', fetchone=True)[0]
    if not rows:
        bot.send_message(USER_ID, '📈 Закрытых сделок пока нет.')
        return

    def calc(data):
        if not data:
            return None
        total  = len(data)
        wins   = sum(1 for r in data if r[1] == 'WIN')
        losses = sum(1 for r in data if r[1] == 'LOSS')
        gp     = sum(r[0] for r in data if r[0] > 0)
        gl     = abs(sum(r[0] for r in data if r[0] < 0))
        return {
            'total': total, 'wins': wins, 'losses': losses,
            'pnl':   sum(r[0] for r in data),
            'pf':    gp / gl if gl else float('inf'),
            'wr':    wins / total * 100,
            'aw':    gp / wins   if wins   else 0,
            'al':    gl / losses if losses else 0,
        }

    all_r  = [(r[0], r[1], r[2]) for r in rows]
    ema_r  = [(r[0], r[1], r[2]) for r in rows if r[3] == 'ema']
    macd_r = [(r[0], r[1], r[2]) for r in rows if r[3] == 'macd']

    st_a = calc(all_r)
    st_e = calc(ema_r)
    st_m = calc(macd_r)

    balances = [STARTING_BALANCE] + [r[2] for r in all_r]
    peak = max_dd = 0.0
    peak = balances[0]
    for b in balances:
        if b > peak: peak = b
        dd = (peak - b) / peak * 100
        if dd > max_dd: max_dd = dd

    def fmt_block(label, s):
        if not s:
            return f'{label}: нет данных\n'
        return (
            f'{label}\n'
            f'  Сделок `{s["total"]}` | WR `{s["wr"]:.1f}%` | PF `{s["pf"]:.2f}`\n'
            f'  PnL `{s["pnl"]:+.2f}` | Победа `+{s["aw"]:.2f}` | Потеря `-{s["al"]:.2f}`\n'
        )

    bot.send_message(USER_ID,
        f'📈 *СТАТИСТИКА* | Активных: `{active}`\n\n'
        f'─────────────────────\n'
        f'{fmt_block("📊 EMA Cross", st_e)}'
        f'─────────────────────\n'
        f'{fmt_block("📉 MACD 30м", st_m)}'
        f'─────────────────────\n'
        f'{fmt_block("🏆 ИТОГО", st_a)}'
        f'Макс. просадка: `{max_dd:.1f}%`\n'
        f'Баланс: `{balances[-1]:.2f}` USDT | '
        f'{"🟢" if st_a["pnl"] >= 0 else "🔴"} '
        f'`{((balances[-1] / STARTING_BALANCE - 1) * 100):+.2f}%`',
        parse_mode='Markdown'
    )

@bot.message_handler(func=lambda m: m.text == '📊 БЭКТЕСТ')
def cmd_bt_init(msg):
    if msg.from_user.id != USER_ID:
        return
    m = bot.send_message(USER_ID,
        '📊 Введи тикер (например *VVV*, *ARC*, *POWER*):',
        parse_mode='Markdown')
    bot.register_next_step_handler(m, cmd_bt_run)


def cmd_bt_run(msg):
    ticker = msg.text.strip().upper()
    symbol = f'{ticker}/USDT:USDT'
    bot.send_message(USER_ID, f'⏳ Тестирую 6 стратегий на {symbol}...')
    threading.Thread(
        target=_run_backtest_thread, args=(ticker, symbol), daemon=True
    ).start()


def _run_backtest_thread(ticker: str, symbol: str):
    try:
        bars = exchange.fetch_ohlcv(symbol, '1h', limit=1000)
        if len(bars) < 100:
            bot.send_message(USER_ID, '❌ Недостаточно данных.')
            return

        def make_df(b):
            d = pd.DataFrame(
                b, columns=['OpenTime','Open','High','Low','Close','Volume'])
            d['OpenTime'] = pd.to_datetime(d['OpenTime'], unit='ms')
            d.set_index('OpenTime', inplace=True)
            return d

        df   = make_df(bars)
        df5  = make_df(exchange.fetch_ohlcv(symbol, '5m',  limit=1000))
        df30 = make_df(exchange.fetch_ohlcv(symbol, '30m', limit=1000))

        def run_bt(cls, data):
            bt    = Backtest(data, cls, cash=1000,
                             commission=0.0004, finalize_trades=True)
            stats = bt.run()
            n     = int(stats['# Trades'])
            return {
                'trades': n,
                'wr':  float(stats['Win Rate [%]']) if n > 0 else 0.0,
                'ret': float(stats['Return [%]']),
                'dd':  float(stats['Max. Drawdown [%]']),
                'pf':  float(stats.get('Profit Factor') or 0),
            }

        s1 = run_bt(FortressBT,     df)
        s2 = run_bt(BreakoutBT,     df)
        s3 = run_bt(SqueezeBT,      df)
        s4 = run_bt(ImpulseBT,      df)
        s5 = run_bt(MACDReversalBT, df5)
        s6 = run_bt(MACDReversalBT, df30)

        def fmt(s):
            return (
                f"Сделок: `{s['trades']}`\n"
                f"WinRate: `{s['wr']:.1f}%`\n"
                f"Доходность: `{s['ret']:+.2f}%`\n"
                f"Макс. просадка: `{s['dd']:.2f}%`\n"
                f"Profit Factor: `{s['pf']:.2f}`"
            )

        best = max(
            [('1️⃣ EMA Cross',   s1['pf']),
             ('2️⃣ Breakout',    s2['pf']),
             ('3️⃣ Squeeze',     s3['pf']),
             ('4️⃣ Impulse Bar', s4['pf']),
             ('5️⃣ MACD 5м',     s5['pf']),
             ('6️⃣ MACD 30м',    s6['pf'])],
            key=lambda x: x[1]
        )

        bot.send_message(USER_ID,
            f'📊 *{ticker}USDT.P*\n'
            f'1ч×1000 | 5м×1000 | 30м×1000\n\n'
            f'─────────────────────\n'
            f'1️⃣ *EMA Cross (1ч)*\n{fmt(s1)}\n\n'
            f'─────────────────────\n'
            f'2️⃣ *Momentum Breakout (1ч)*\n{fmt(s2)}\n\n'
            f'─────────────────────\n'
            f'3️⃣ *Volatility Squeeze (1ч)*\n{fmt(s3)}\n\n'
            f'─────────────────────\n'
            f'4️⃣ *Impulse Bar (1ч)*\n{fmt(s4)}\n\n'
            f'─────────────────────\n'
            f'5️⃣ *MACD Reversal (5м)*\n{fmt(s5)}\n\n'
            f'─────────────────────\n'
            f'6️⃣ *MACD Reversal (30м)*\n{fmt(s6)}\n\n'
            f'─────────────────────\n'
            f'🏆 Лучше по PF: *{best[0]}*',
            parse_mode='Markdown'
        )

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        bot.send_message(USER_ID, f'❌ Сетевая ошибка: `{e}`',
                         parse_mode='Markdown')
    except Exception as e:
        bot.send_message(USER_ID,
            f'❌ Ошибка бэктеста `{ticker}`\n`{str(e)[:300]}`',
            parse_mode='Markdown')

@bot.message_handler(func=lambda m: m.text == '📡 СТАТУС')
def cmd_status(msg):
    if msg.from_user.id != USER_ID:
        return
    ae = _read_db("SELECT COUNT(*) FROM active_trades WHERE scanner='ema'",
                  fetchone=True)[0]
    am = _read_db("SELECT COUNT(*) FROM active_trades WHERE scanner='macd'",
                  fetchone=True)[0]
    cl = _read_db('SELECT COUNT(*) FROM trade_log', fetchone=True)[0]
    ne = seconds_until_next_grid(EMA_GRID_MINUTES)
    nm = seconds_until_next_grid(MACD_GRID_MINUTES)
    bot.send_message(USER_ID,
        f'📡 *СТАТУС* | {"🔴 РЕАЛ" if LIVE_TRADING else "🟡 БУМАГА"}\n\n'
        f'📊 EMA Cross | Активных: `{ae}`\n'
        f'  Следующий скан: `{ne/60:.1f}` мин\n\n'
        f'📉 MACD 30м | Активных: `{am}`\n'
        f'  Следующий скан: `{nm/60:.1f}` мин\n\n'
        f'Закрытых сделок: `{cl}`\n'
        f'Время: `{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}`',
        parse_mode='Markdown'
    )

@bot.message_handler(func=lambda m: m.text == '🔍 РАДАР')
def cmd_radar_manual(msg):
    if msg.from_user.id != USER_ID:
        return
    bot.send_message(USER_ID, '🔍 Запускаю радар...')
    threading.Thread(target=_run_radar_thread, daemon=True).start()


def _run_radar_thread():
    try:
        found = run_radar_scan()
        if not found:
            bot.send_message(USER_ID, '🔍 Радар: ничего не найдено.')
            return
        bot.send_message(USER_ID,
            _format_radar_results(found, 'РАДАР ПАМПА (ручной)'),
            parse_mode='Markdown')
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        bot.send_message(USER_ID, f'❌ Сетевая ошибка радара: `{e}`',
                         parse_mode='Markdown')
    except Exception as e:
        bot.send_message(USER_ID, f'❌ Ошибка: `{str(e)[:200]}`',
                         parse_mode='Markdown')


@bot.message_handler(func=lambda m: m.text == '👀 ВОТЧ-ЛИСТ')
def cmd_watchlist(msg):
    if msg.from_user.id != USER_ID:
        return
    bot.send_message(USER_ID,
        f'👀 Сканирую топ-{WATCHLIST_TOP} монет близких к сигналу...')
    threading.Thread(target=_run_watchlist_thread, daemon=True).start()


def _run_watchlist_thread():
    try:
        tickers = exchange.fetch_tickers()
        candidates = [
            s for s, d in tickers.items()
            if '/USDT:USDT' in s and s not in BLACKLIST
            and (d.get('quoteVolume') or 0) >= MIN_VOLUME_24H
        ]
        watchlist = run_watchlist_scan(candidates)
        if not watchlist:
            bot.send_message(USER_ID,
                '👀 Вотч-лист: подходящих монет не найдено.')
            return
        bot.send_message(USER_ID,
            _format_watchlist(watchlist),
            parse_mode='Markdown')
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        bot.send_message(USER_ID, f'❌ Сетевая ошибка: `{e}`',
                         parse_mode='Markdown')
    except Exception as e:
        bot.send_message(USER_ID, f'❌ Ошибка вотч-листа: `{str(e)[:200]}`',
                         parse_mode='Markdown')


# =====================================================================
# ГЛУБОКИЙ АНАЛИЗ МОНЕТЫ
# =====================================================================
def _deep_analyze(symbol: str) -> str:
    ticker_data = exchange.fetch_ticker(symbol)
    price       = ticker_data['last']
    volume_24h  = ticker_data.get('quoteVolume') or 0
    change_24h  = ticker_data.get('percentage')  or 0

    results = {}
    for tf in ['15m', '1h', '4h', '1d']:
        try:
            df = pd.DataFrame(
                exchange.fetch_ohlcv(symbol, tf, limit=100),
                columns=['ts', 'o', 'h', 'l', 'c', 'v']
            )
            c = df['c']; h = df['h']; l = df['l']

            e12   = _ema(c, 12).iloc[-2]
            e26   = _ema(c, 26).iloc[-2]
            e50   = _ema(c, 50).iloc[-2]
            rsi_v = _rsi(c, 14).iloc[-2]
            atr_v = _atr(h, l, c, 14).iloc[-2]
            ml, sl_macd = _macd(c, 12, 26, 9)

            trend    = '🟢 бычий' if (not pd.isna(e12) and not pd.isna(e26) and e12 > e26) else '🔴 медвежий'
            vs_ema50 = ('↑EMA50' if price > e50 else '↓EMA50') if not pd.isna(e50) else ''

            if not pd.isna(rsi_v):
                if rsi_v > 70:   rsi_zone = '🔴 перекуплен'
                elif rsi_v < 30: rsi_zone = '🟢 перепродан'
                else:            rsi_zone = '🟡 нейтрал'
            else:
                rsi_zone = ''

            macd_sig = ''
            if not pd.isna(ml.iloc[-2]) and not pd.isna(sl_macd.iloc[-2]):
                if ml.iloc[-3] < sl_macd.iloc[-3] and ml.iloc[-2] > sl_macd.iloc[-2]:
                    macd_sig = 'бычье пересечение ⚡'
                elif ml.iloc[-3] > sl_macd.iloc[-3] and ml.iloc[-2] < sl_macd.iloc[-2]:
                    macd_sig = 'медвежье пересечение ⚡'
                elif ml.iloc[-2] > sl_macd.iloc[-2]:
                    macd_sig = 'выше сигнала'
                else:
                    macd_sig = 'ниже сигнала'

            natr = (atr_v / price * 100) if (not pd.isna(atr_v) and atr_v) else 0
            results[tf] = {
                'trend': trend, 'vs_ema50': vs_ema50,
                'rsi': rsi_v, 'rsi_zone': rsi_zone,
                'macd': macd_sig, 'natr': natr,
            }
        except Exception:
            results[tf] = None

    # Уровни по 1д
    support = resistance = None
    try:
        df1d = pd.DataFrame(
            exchange.fetch_ohlcv(symbol, '1d', limit=30),
            columns=['ts', 'o', 'h', 'l', 'c', 'v']
        )
        lows  = df1d['l'].iloc[:-1]
        highs = df1d['h'].iloc[:-1]
        below = lows[lows < price]
        above = highs[highs > price]
        support    = below.max() if len(below) > 0 else None
        resistance = above.min() if len(above) > 0 else None
    except Exception:
        pass

    # Наши стратегии
    try:
        ema_signal = '✅ сетап есть' if analyze_symbol(symbol) else '❌ нет сигнала'
    except Exception:
        ema_signal = '⚠️ ошибка'
    try:
        m = analyze_symbol_macd(symbol)
        macd_signal = f'✅ {m["side"].upper()}' if m else '❌ нет сигнала'
    except Exception:
        macd_signal = '⚠️ ошибка'

    # Итоговый скор
    bull = bear = 0.0
    weights = {'15m': 1, '1h': 2, '4h': 3, '1d': 2}
    for tf, r in results.items():
        if not r:
            continue
        w = weights.get(tf, 1)
        if '🟢' in r['trend']:
            bull += w
        else:
            bear += w
        if r['rsi']:
            if r['rsi'] > 50: bull += w * 0.5
            else:              bear += w * 0.5

    total = bull + bear
    bull_pct = (bull / total * 100) if total > 0 else 50
    if bull_pct >= 70:
        score_str = f'🟢 БЫЧИЙ {bull_pct:.0f}%'
    elif bull_pct <= 30:
        score_str = f'🔴 МЕДВЕЖИЙ {100-bull_pct:.0f}%'
    else:
        score_str = '🟡 НЕЙТРАЛЬНЫЙ'

    tf_names = {'15m': '15м', '1h': '1ч ', '4h': '4ч ', '1d': '1д '}
    text = (
        f'🧠 *АНАЛИЗ: {symbol}*\n'
        f'Цена: `{price:.6g}` | 24ч: `{change_24h:+.2f}%`\n'
        f'Объём 24ч: `{volume_24h/1_000_000:.1f}M` USDT\n\n'
        f'📊 *ТЕХНИЧЕСКИЙ АНАЛИЗ*\n'
    )
    for tf in ['15m', '1h', '4h', '1d']:
        r = results.get(tf)
        if not r:
            text += f'`{tf_names[tf]}` — нет данных\n'
            continue
        rsi_str = f'RSI `{r["rsi"]:.0f}` {r["rsi_zone"]}' if r['rsi'] else ''
        text += (
            f'`{tf_names[tf]}` {r["trend"]} {r["vs_ema50"]}\n'
            f'  {rsi_str} | NATR `{r["natr"]:.1f}%`\n'
            f'  MACD: {r["macd"]}\n'
        )

    text += '\n📍 *УРОВНИ*\n'
    if support:
        text += f'Поддержка:     `{support:.6g}` (-{(price-support)/price*100:.1f}%)\n'
    if resistance:
        text += f'Сопротивление: `{resistance:.6g}` (+{(resistance-price)/price*100:.1f}%)\n'

    text += (
        f'\n🤖 *НАШИ СТРАТЕГИИ*\n'
        f'EMA Cross 1ч: {ema_signal}\n'
        f'MACD 30м:     {macd_signal}\n'
        f'\n─────────────────────\n'
        f'📈 *СКОР: {score_str}*\n'
        f'─────────────────────\n'
        f'⚠️ _Анализ текущего момента — не прогноз_'
    )
    return text[:4000]


@bot.message_handler(func=lambda m: m.text == '🧠 АНАЛИЗ МОНЕТЫ')
def cmd_analyze_init(msg):
    if msg.from_user.id != USER_ID:
        return
    m = bot.send_message(USER_ID,
        '🧠 Введи тикер для анализа\n(например *BTC*, *ETH*, *DOGE*):',
        parse_mode='Markdown'
    )
    bot.register_next_step_handler(m, cmd_analyze_run)


def cmd_analyze_run(msg):
    ticker = msg.text.strip().upper()
    symbol = f'{ticker}/USDT:USDT'
    bot.send_message(USER_ID, f'⏳ Анализирую {symbol} на 4 таймфреймах...')
    threading.Thread(
        target=_run_analyze_thread, args=(symbol,), daemon=True
    ).start()


def _run_analyze_thread(symbol: str):
    try:
        text = _deep_analyze(symbol)
        bot.send_message(USER_ID, text, parse_mode='Markdown')
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        bot.send_message(USER_ID, f'❌ Сетевая ошибка: `{e}`',
                         parse_mode='Markdown')
    except Exception as e:
        bot.send_message(USER_ID,
            f'❌ Ошибка анализа: `{str(e)[:200]}`',
            parse_mode='Markdown')

# =====================================================================
# ЗАКРЫТИЕ ВСЕХ СДЕЛОК ВРУЧНУЮ
# =====================================================================
@bot.message_handler(func=lambda m: m.text == '🚪 ЗАКРЫТЬ ВСЕ СДЕЛКИ')
def cmd_close_all_init(msg):
    if msg.from_user.id != USER_ID:
        return
    trades = _read_db('SELECT COUNT(*) FROM active_trades', fetchone=True)[0]
    if not trades:
        bot.send_message(USER_ID, '⚔️ Активных сделок нет.')
        return
    m = bot.send_message(USER_ID,
        f'⚠️ *Закрыть ВСЕ {trades} сделок по текущей цене?*\n\n'
        f'Напиши *ДА* для подтверждения:',
        parse_mode='Markdown')
    bot.register_next_step_handler(m, cmd_close_all_confirm)


def cmd_close_all_confirm(msg):
    if msg.text.strip().upper() != 'ДА':
        bot.send_message(USER_ID, '❌ Отменено.')
        return
    bot.send_message(USER_ID, '⏳ Закрываю все сделки...')
    threading.Thread(target=_close_all_thread, daemon=True).start()


def _close_all_thread():
    trades = _read_db(
        'SELECT id,symbol,side,entry_price,size,sl,tp,'
        '       is_trailing,trailing_sl,partial_price,'
        '       entry_time,entry_balance,scanner,macd_active '
        'FROM active_trades'
    )
    if not trades:
        bot.send_message(USER_ID, '⚔️ Сделок нет.')
        return
    closed = 0
    for trade in trades:
        sym = trade[1]
        try:
            price = exchange.fetch_ticker(sym)['last']
            close_trade(trade, price, '🚪 Ручное закрытие')
            closed += 1
        except Exception as e:
            bot.send_message(USER_ID, f'❌ Ошибка закрытия {sym}: `{e}`',
                             parse_mode='Markdown')
    balance = _read_db('SELECT balance FROM wallet', fetchone=True)[0]
    bot.send_message(USER_ID,
        f'✅ *Закрыто сделок: {closed}*\n'
        f'💰 Баланс: `{balance:.2f}` USDT',
        parse_mode='Markdown')


# =====================================================================
# ТОЧКА ВХОДА
# =====================================================================
async def _guarded(coro_fn, name: str):
    """Оборачивает корутину — при падении уведомляет и перезапускает."""
    while True:
        try:
            await coro_fn()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            err_msg = str(e)[:300]
            print(f'[{name}] Авария: {err_msg} — перезапуск через 10 сек')
            try:
                bot.send_message(USER_ID,
                    f'🚨 *{name} УПАЛ*\n`{err_msg}`\n'
                    f'Автоперезапуск через 10 сек...',
                    parse_mode='Markdown')
            except Exception:
                pass
            await asyncio.sleep(10)


async def main_async():
    await asyncio.gather(
        _guarded(signal_hunter,    'EMA Scanner'),
        _guarded(macd_hunter,      'MACD Scanner'),
        _guarded(monitor_trades,   'Monitor'),
        _guarded(pump_radar,       'Radar'),
        _guarded(watchlist_hunter, 'Watchlist'),
    )


if __name__ == '__main__':
    init_db()

    print('🛡️  Fortress Paper Trading V4.5')
    print(f'   Режим: {"РЕАЛЬНАЯ ТОРГОВЛЯ" if LIVE_TRADING else "Бумажная торговля"}')
    print(f'   EMA Grid:  каждые {EMA_GRID_MINUTES} мин')
    print(f'   MACD Grid: каждые {MACD_GRID_MINUTES} мин')

    db_thread = threading.Thread(target=db_worker, daemon=True)
    db_thread.start()
    print('   DB Worker: запущен')

    threading.Thread(
        target=lambda: asyncio.run(main_async()),
        daemon=True
    ).start()
    print('   Async loop: запущен')

    while True:
        try:
            print('📡 Бот слушает Telegram...')
            bot.infinity_polling(timeout=90, long_polling_timeout=90)
        except Exception as e:
            print(f'Сетевой сбой: {e}. Перезапуск через 5 сек...')
            time.sleep(5)
