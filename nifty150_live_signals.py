"""
Nifty150 Live Signal Notifier — SuperTrend + ADX DI/DI-
=========================================================
Standalone script — copy this single file to any machine with internet
access and Python 3.9+, install requirements, and run it. It polls
15-min bars during NSE market hours and pops a desktop notification the
moment either strategy's signal FIRES on any Nifty150 stock, or on the
NIFTY 50 / BANK NIFTY / NEXT 50 / MIDCAP 150 / SMALLCAP 250 / NIFTY 500 indices (not on every bar — only on the
state change).

Strategies (locked configs from prior backtests):
  - SuperTrend ATR(14) x 3.0                    -> BUY on trend flip -1 -> 1
                                                   SELL on trend flip 1 -> -1
  - ADX DI+/DI- [Gu5], skip-open-bar             -> BUY on condition -> 1 or 0.5
                                                   SELL on condition -> -1 or -0.5
The BUY side is the backtested long-only config. SELL alerts are short-entry
signals from the same indicators — they were NOT part of those backtests.

Install:
    pip install yfinance pandas numpy numba curl_cffi

Run (leave running in a terminal / as a background service):
    python nifty150_live_signals.py

Or a single scan, for a scheduler such as GitHub Actions
(.github/workflows/live-signals.yml runs it every 15 min in market hours):
    python nifty150_live_signals.py --once

Or one scan right now, ignoring market hours (for trying it out):
    python nifty150_live_signals.py --test

Notes:
  - Data source is yfinance (free, ~15-min delayed) — not for split-second
    execution, fine for swing/intraday signal alerts.
  - Market hours gate assumes IST (Asia/Kolkata) and NSE's 09:15-15:30
    session, Mon-Fri. It sleeps outside those hours and wakes near open.
  - State is kept in signal_state.json next to this script, so restarting
    the script does not re-fire already-seen signals.
  - Universe = Nifty 500, fetched from NSE's official
    constituent CSVs at the start of every scan, so index reshuffles (about
    every 6 months) are picked up automatically. If NSE can't be reached,
    the built-in NIFTY150 list below (snapshot as of 2026-10-07) is used;
    the scan log notes when the live list differs from it.
  - Popups are a custom always-on-top borderless window pinned to the
    TOP-RIGHT corner of the primary screen (tkinter, stdlib — no extra
    install needed). Requires a desktop/GUI session; if none is available
    (e.g. a headless server), it falls back to console-only output.
  - Telegram: signals are also sent to a Telegram chat when a bot token and
    chat id are configured — via env vars TELEGRAM_TOKEN and
    TELEGRAM_CHAT_ID, or a telegram_config.json next to this script:
        {"bot_token": "123456:ABC...", "chat_id": "123456789"}
    Create the bot with @BotFather, send it any message, then get your chat
    id from https://api.telegram.org/bot<TOKEN>/getUpdates. Verify with:
        python nifty150_live_signals.py --test-telegram
    If neither is configured, Telegram is skipped (popup + console only).
    Optional: TELEGRAM_MODE = all (default: every signal), gated (only
    signals whose 1-hour SuperTrend agrees — the stage gate), or confirmed (double
    confirmations + index signals); TELEGRAM_UNTIL=HH:MM stops alerts after
    that time. The dashboard always shows every signal, with gate badges.

Rate-limit avoidance (yfinance / Yahoo chart API is aggressive about 429s):
  1. One shared curl_cffi session impersonating a real browser TLS/HTTP
     fingerprint — plain `requests` (yfinance's default) gets flagged and
     429'd much faster than a browser-like client. Falls back to default
     yfinance session if curl_cffi isn't installed (just less robust).
  2. Requests are paced, not fired back-to-back: the 150-stock universe is
     spread evenly across the poll interval (~15 min / 150 ≈ 6s apart),
     plus random jitter — this alone avoids bursting.
  3. Per-symbol retry with exponential backoff + jitter on rate-limit /
     network errors (up to RETRY_MAX attempts), so a transient 429 doesn't
     drop that stock for the whole cycle.
  4. A global circuit breaker: if consecutive rate-limit hits cross
     CIRCUIT_BREAKER_THRESHOLD, the whole scan pauses for COOLDOWN_SECONDS
     before resuming, instead of hammering a block that won't clear.
  5. Watchlist prioritization: stocks within WATCHLIST_PCT of their
     SuperTrend flip line (per last cycle's close) are scanned first, so the
     stocks most likely to fire are checked before the rest of the sweep.
     Every stock is still scanned every cycle.
"""
import warnings; warnings.filterwarnings("ignore")

import base64
import gzip
import html as html_lib
import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from numba import njit

try:
    import tkinter as tk
    # CI runners (e.g. GitHub Actions sets CI=true) have no display to pop up on
    DESKTOP_NOTIFY = not os.environ.get("CI")
except ImportError:
    DESKTOP_NOTIFY = False
    print("[warn] tkinter not available — falling back to console-only alerts.")

try:
    from curl_cffi import requests as curl_requests

    # libcurl drops the colon from Windows drive-letter paths that reach it via
    # CURL_CA_BUNDLE ("C:\foo" -> "C\foo", error 77) — e.g. behind a corporate
    # proxy. Passing the bundle to `verify=` with forward slashes avoids that.
    ca_bundle = os.environ.get("CURL_CA_BUNDLE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if ca_bundle:
        ca_bundle = ca_bundle.replace("\\", "/")
        SESSION = curl_requests.Session(impersonate="chrome", verify=ca_bundle)
        print(f"[info] using curl_cffi with explicit CA bundle: {ca_bundle}")
    else:
        SESSION = curl_requests.Session(impersonate="chrome")
        print("[info] using curl_cffi browser-impersonation session (rate-limit resistant)")
except ImportError:
    SESSION = None
    print("[warn] curl_cffi not installed — using yfinance's default session, "
          "more prone to 429s. Run: pip install curl_cffi")

def _load_telegram_config():
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    cfg_file = Path(__file__).parent / "telegram_config.json"
    if not (token and chat_id) and cfg_file.exists():
        cfg = json.loads(cfg_file.read_text())
        token = token or cfg.get("bot_token")
        chat_id = chat_id or cfg.get("chat_id")
    if token and chat_id:
        print("[info] Telegram alerts enabled")
        return token, str(chat_id)
    print("[info] Telegram not configured — set TELEGRAM_TOKEN/TELEGRAM_CHAT_ID "
          "or create telegram_config.json to enable")
    return None, None


TELEGRAM_TOKEN, TELEGRAM_CHAT_ID = _load_telegram_config()

TOAST_MARGIN = 20      # px from screen edge
TOAST_WIDTH = 340
TOAST_HEIGHT = 110
TOAST_DURATION_MS = 8000

RETRY_MAX = 4
RETRY_BASE_DELAY = 5        # seconds, doubles each retry + jitter
CIRCUIT_BREAKER_THRESHOLD = 8  # consecutive rate-limit errors across symbols
COOLDOWN_SECONDS = 180

# % distance from SuperTrend flip line to be scanned first. On 15-min bars the
# band hugs price: at 2% ~147/150 stocks qualified (no prioritization, and the
# tight spacing burst the whole universe); 0.5% picks the nearest ~30.
WATCHLIST_PCT = 0.5
WATCHLIST_DELAY = 2   # seconds between watchlist symbols (still paced, just tight)

IST = ZoneInfo("Asia/Kolkata")
STATE_FILE = Path(__file__).parent / "signal_state.json"
POLL_SECONDS = 15 * 60  # re-scan every 15 min, matching the bar size
FETCH_PERIOD = "40d"    # 15-min history per fetch: 10 sessions replayed + ~30 of warm-up (EMA200 settles; checked vs 60d)

ATR_PERIOD = 14
MULTIPLIER = 3.0
DI_LEN, SIG_LEN, HL_RANGE, HL_TREND = 14, 14, 20, 35

NIFTY150 = sorted(set([
    "ABB","ADANIENSOL","ADANIENT","ADANIGREEN","ADANIPORTS","ADANIPOWER",
    "AMBUJACEM","APLAPOLLO","APOLLOHOSP","ASHOKLEY","ASIANPAINT","AUBANK",
    "AUROPHARMA","AXISBANK","BAJAJ-AUTO","BAJAJFINSV","BAJAJHLDNG",
    "BAJFINANCE","BANKBARODA","BEL","BHARATFORG","BHARTIARTL","BHEL",
    "BOSCHLTD","BPCL","BRITANNIA","BSE","CANBK","CGPOWER","CHOLAFIN","CIPLA",
    "COALINDIA","CUMMINSIND","DABUR","DIVISLAB","DIXON","DLF","DMART",
    "DRREDDY","EICHERMOT","ENRIN","ETERNAL","FEDERALBNK","FORTIS","GAIL",
    "GLENMARK","GMRAIRPORT","GODREJCP","GODREJPROP","GRASIM","GVT&D","HAL",
    "HAVELLS","HCLTECH","HDFCAMC","HDFCBANK","HDFCLIFE","HEROMOTOCO",
    "HINDALCO","HINDPETRO","HINDUNILVR","HINDZINC","HYUNDAI","ICICIBANK",
    "ICICIGI","IDEA","IDFCFIRSTB","INDHOTEL","INDIGO","INDUSINDBK",
    "INDUSTOWER","INFY","IOC","IRFC","ITC","JINDALSTEL","JIOFIN","JSWENERGY",
    "JSWSTEEL","KOTAKBANK","LAURUSLABS","LT","LTM","LUPIN","M&M","MANKIND",
    "MARICO","MARUTI","MAXHEALTH","MAZDOCK","MCX","MFSL","MOTHERSON",
    "MUTHOOTFIN","NATIONALUM","NAUKRI","NESTLEIND","NHPC","NMDC","NTPC",
    "NYKAA","OIL","ONGC","PAYTM","PERSISTENT","PFC","PHOENIXLTD",
    "PIDILITIND","PNB","POLICYBZR","POLYCAB","POWERGRID","POWERINDIA",
    "PRESTIGE","RECLTD","RELIANCE","SBILIFE","SBIN","SHRIRAMFIN","SIEMENS",
    "SOLARINDS","SRF","SUNPHARMA","SUZLON","SWIGGY","TATACAP","TATACONSUM",
    "TATAPOWER","TATASTEEL","TCS","TECHM","TIINDIA","TITAN","TMCV","TMPV",
    "TORNTPHARM","TRENT","TVSMOTOR","ULTRACEMCO","UNIONBANK","UNITDSPR",
    "UPL","VAML","VBL","VEDL","VMM","WAAREEENER","WIPRO","YESBANK",
    "ZYDUSLIFE",
]))

# The universe is refreshed from NSE's published constituent CSVs (with
# NIFTY150 above as the fallback), so index reshuffles are picked up without
# editing this file. The universe is the Nifty 500.
NSE_INDEX_URLS = {
    "Nifty500": ("https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv", 500),
}
UNIVERSE_FALLBACK = Path(__file__).parent / "nifty500_list.csv"   # saved copy of the NSE list
NIFTY100_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv"
NIFTY100_FALLBACK = Path(__file__).parent / "nifty100_list.csv"
FOCUS_FROM, FOCUS_TO = "09:45", "14:30"   # Focus filter: signal candles from 9:45 AM to before 2:30 PM

# The indices themselves, scanned with the same strategies after the stocks.
INDEX_TICKERS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "NIFTY NEXT 50": "^NSMIDCP",
    "NIFTY MIDCAP 150": "NIFTYMIDCAP150.NS",
    "NIFTY SMALLCAP 250": "NIFTYSMLCAP250.NS",
    "NIFTY 500": "^CRSLDX",
}


SECTORS = {}  # symbol -> NSE industry, filled by load_universe()


def load_nifty100():
    """Nifty 100 constituents (live from NSE, else the saved copy) — the
    dashboard's Focus filter shows only these stocks plus the indices."""
    from io import StringIO
    try:
        text = (SESSION.get(NIFTY100_URL, timeout=15).text if SESSION else
                urllib.request.urlopen(urllib.request.Request(NIFTY100_URL, headers={"User-Agent": "Mozilla/5.0"}),
                                       timeout=15).read().decode())
        got = sorted(pd.read_csv(StringIO(text))["Symbol"].str.strip())
        if abs(len(got) - 100) <= 5:
            return got
    except Exception as e:  # noqa: BLE001
        print(f"[warn] Nifty 100 list fetch failed ({e}); using the saved copy")
    if NIFTY100_FALLBACK.exists():
        return sorted(pd.read_csv(NIFTY100_FALLBACK)["Symbol"].str.strip())
    return []


def load_universe():
    """Live Nifty 500 constituents from NSE, else the saved copy
    (nifty500_list.csv), else the old NIFTY150 snapshot. The CSV must parse to
    roughly its expected size, so an error page or a half-loaded file can't
    silently shrink the universe. Also fills SECTORS from the Industry column.
    NSE placeholder rows (symbols starting "DUMMY") are skipped."""
    from io import StringIO
    try:
        symbols = set()
        for name, (url, expected) in NSE_INDEX_URLS.items():
            if SESSION:
                text = SESSION.get(url, timeout=15).text
            else:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                text = urllib.request.urlopen(req, timeout=15).read().decode()
            table = pd.read_csv(StringIO(text))
            table = table[~table["Symbol"].str.strip().str.startswith("DUMMY")]
            got = set(table["Symbol"].str.strip())
            if "Industry" in table:
                SECTORS.update(zip(table["Symbol"].str.strip(), table["Industry"].astype(str).str.strip()))
            if abs(len(got) - expected) > 5:
                raise ValueError(f"{name} list has {len(got)} symbols, expected ~{expected}")
            symbols |= got
        return sorted(symbols)
    except Exception as e:
        if UNIVERSE_FALLBACK.exists():
            table = pd.read_csv(UNIVERSE_FALLBACK)
            table = table[~table["Symbol"].str.strip().str.startswith("DUMMY")]
            SECTORS.update(zip(table["Symbol"].str.strip(), table["Industry"].astype(str).str.strip()))
            print(f"[warn] live NSE constituent fetch failed ({e}); using the saved Nifty 500 list")
            return sorted(table["Symbol"].str.strip())
        print(f"[warn] live NSE constituent fetch failed ({e}); using built-in list")
        return NIFTY150


# ─── Indicators ────────────────────────────────────────────────────────────

@njit
def _atr_rma(high, low, close, period):
    n = len(close)
    atr = np.empty(n)
    atr[0] = high[0] - low[0]
    alpha = 1.0 / period
    for i in range(1, n):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr[i] = alpha * tr + (1.0 - alpha) * atr[i-1]
    return atr


@njit
def _supertrend(close, high, low, atr, mult):
    n = len(close)
    trend = np.ones(n, dtype=np.int8)
    upper = np.empty(n)
    lower = np.empty(n)
    for i in range(n):
        mid = (high[i] + low[i]) * 0.5
        upper[i] = mid + mult * atr[i]
        lower[i] = mid - mult * atr[i]
    for i in range(1, n):
        upper[i] = upper[i] if close[i-1] > upper[i-1] else min(upper[i], upper[i-1])
        lower[i] = lower[i] if close[i-1] < lower[i-1] else max(lower[i], lower[i-1])
        if trend[i-1] == -1 and close[i] > upper[i-1]:
            trend[i] = 1
        elif trend[i-1] == 1 and close[i] < lower[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
    return trend, upper, lower


@njit
def _calc_di_sig(high, low, close, di_len, sig_len):
    n = len(close)
    up = np.zeros(n); down = np.zeros(n); tr = np.zeros(n)
    for i in range(1, n):
        up[i] = high[i] - high[i-1]; down[i] = low[i-1] - low[i]
    plus_dm = np.zeros(n); minus_dm = np.zeros(n)
    for i in range(1, n):
        if up[i] > down[i] and up[i] > 0: plus_dm[i] = up[i]
        if down[i] > up[i] and down[i] > 0: minus_dm[i] = down[i]
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

    def rma(x, length):
        r = np.empty(len(x)); r[0] = x[0]; alpha = 1.0 / length
        for i in range(1, len(x)): r[i] = alpha * x[i] + (1 - alpha) * r[i-1]
        return r

    atr_di = rma(tr, di_len); plus_rma = rma(plus_dm, di_len); minus_rma = rma(minus_dm, di_len)
    di_plus = np.empty(n); di_minus = np.empty(n)
    last_p, last_m = 0.0, 0.0
    for i in range(n):
        if atr_di[i] != 0:
            p = 100.0 * plus_rma[i] / atr_di[i]; m = 100.0 * minus_rma[i] / atr_di[i]
            if not np.isnan(p): last_p = p
            if not np.isnan(m): last_m = m
        di_plus[i] = last_p; di_minus[i] = last_m
    dx = np.empty(n)
    for i in range(n):
        s = di_plus[i] + di_minus[i]
        dx[i] = abs(di_plus[i] - di_minus[i]) / (s if s != 0 else 1.0)
    sig = 100.0 * rma(dx, sig_len)
    return di_plus, di_minus, sig


@njit
def _build_condition(di_plus, di_minus, sig, hl_range_lvl, hl_trend_lvl, is_open_bar):
    n = len(sig); condition = np.zeros(n)
    for i in range(1, n):
        hlr = sig[i] <= hl_range_lvl; hlr_p = sig[i-1] <= hl_range_lvl
        di_up = di_plus[i] >= di_minus[i]; di_up_p = di_plus[i-1] >= di_minus[i-1]
        di_dn = di_minus[i] > di_plus[i]; di_dn_p = di_minus[i-1] > di_plus[i-1]
        di_upup = di_plus[i] >= hl_trend_lvl; di_dndn = di_minus[i] > hl_trend_lvl
        sig_up = sig[i] > sig[i-1]
        entry_long = (not hlr and di_up and sig_up and not di_up_p) or \
                     (not hlr and di_up and sig_up and sig[i] > hl_range_lvl and hlr_p)
        entry_short = (not hlr and di_dn and sig_up and not di_dn_p) or \
                      (not hlr and di_dn and sig_up and sig[i] > hl_range_lvl and hlr_p)
        entry_long_str = not hlr and di_up and sig_up and di_upup
        entry_short_str = not hlr and di_dn and sig_up and di_dndn
        if is_open_bar[i]:
            entry_long = False; entry_short = False; entry_long_str = False; entry_short_str = False
        cross = (di_plus[i-1] - di_minus[i-1]) * (di_plus[i] - di_minus[i]) < 0
        exit_long = (cross and di_up_p) or (hlr and not hlr_p)
        exit_short = (cross and di_dn_p) or (hlr and not hlr_p)
        prev = condition[i-1]
        if prev != 1 and entry_long_str: condition[i] = 1
        elif prev != -1 and entry_short_str: condition[i] = -1
        elif prev != 0.5 and entry_long: condition[i] = 0.5
        elif prev != -0.5 and entry_short: condition[i] = -0.5
        elif prev != 0 and exit_long: condition[i] = 0
        elif prev != 0 and exit_short: condition[i] = 0
        else: condition[i] = prev
    return condition


# ─── Data + state ──────────────────────────────────────────────────────────

class RateLimited(Exception):
    pass


BARS_CACHE_FILE = Path(__file__).parent / "bars_cache.pkl"
BARS_CACHE = None  # {yf_ticker: DataFrame}, loaded on first use, saved after each scan


def load_bars_cache():
    global BARS_CACHE
    if BARS_CACHE is None:
        BARS_CACHE = {}
        if BARS_CACHE_FILE.exists():
            try:
                BARS_CACHE = pd.read_pickle(BARS_CACHE_FILE)
            except Exception as e:
                print(f"[warn] bar cache unreadable ({e}); starting fresh")
    return BARS_CACHE


def save_bars_cache():
    if BARS_CACHE is not None:
        pd.to_pickle(BARS_CACHE, BARS_CACHE_FILE)


def fetch_15m(symbol, yf_ticker=None):
    """15-min bars for the last ~15 sessions. Downloads the full FETCH_PERIOD
    only on the first fetch of the day (which also picks up any split/bonus
    adjustment Yahoo made to older bars); later scans that day download just
    today's bars and merge them into the cached history. `yf_ticker`
    overrides the "<symbol>.NS" lookup (e.g. "^NSEI" for an index)."""
    yf_ticker = yf_ticker or f"{symbol}.NS"
    cache = load_bars_cache()
    cached = cache.get(yf_ticker)
    today = datetime.now(IST).date()
    fresh_today = cached is not None and len(cached) and cached.attrs.get("full_fetch") == str(today)
    df = _download(symbol, yf_ticker, "1d" if fresh_today else FETCH_PERIOD)
    if fresh_today:
        if df is None or df.empty:
            return cached
        df = pd.concat([cached, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    elif df is None:
        return cached
    keep = sorted(set(df.index.date))[-40:]
    df = df[np.isin(df.index.date, keep)].copy()
    df.attrs["full_fetch"] = str(today)  # set last: attrs don't reliably survive concat/slicing
    cache[yf_ticker] = df
    return df


HTF_HISTORY = "60d"  # hourly history behind the 1-hour trend gate


def _bucket(index, minutes):
    """Session-aligned bucket start for each timestamp: NSE opens 9:15, so 2h
    candles are 9:15 / 11:15 / 13:15 / 15:15 and 1h candles start at :15."""
    mins = index.hour * 60 + index.minute - 555
    return index.normalize() + pd.to_timedelta(555 + (mins // minutes) * minutes, unit="m")


def _ohlc(df, key):
    g = df.groupby(key)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last()})
    return out, g


def htf_trends(symbol, yf_ticker, df15):
    """SuperTrend direction on 1-hour candles, for the stage gate.
    Built from 60 days of hourly bars (downloaded once a day, cached) plus
    today's 15-min bars, so the gates update intraday without extra downloads.
    Returns {"1h": (close_ns, trend)} where close_ns
    is when each candle closed (epoch ns) — callers only use candles that had
    closed by the time in question, so nothing looks ahead."""
    cache = load_bars_cache()
    key = f"{yf_ticker}|1h"
    today = datetime.now(IST).date()
    hist = cache.get(key)
    if hist is None or hist.attrs.get("full_fetch") != str(today):
        got = _download(symbol, yf_ticker, HTF_HISTORY, interval="1h")
        if got is not None:
            hist = got.copy()
            hist.attrs["full_fetch"] = str(today)
            cache[key] = hist
    if hist is None or df15 is None or df15.empty:
        return None
    last_day = df15.index[-1].date()
    today15 = df15[df15.index.date == last_day]
    h1_today, _ = _ohlc(today15, _bucket(today15.index, 60))
    h1 = pd.concat([hist[hist.index.date < last_day][["open", "high", "low", "close"]], h1_today]).sort_index()
    out = {}
    for name, minutes in (("1h", 60),):
        key_ix = (_bucket(h1.index, minutes) if minutes else h1.index.normalize() + pd.Timedelta(minutes=555))
        bars, g = _ohlc(h1, key_ix)
        if len(bars) < 20:
            return None
        last_start = g.apply(lambda x: x.index[-1])
        session_end = last_start.dt.normalize() + pd.Timedelta(hours=15, minutes=30)
        closes = (last_start + pd.Timedelta(hours=1)).where(last_start + pd.Timedelta(hours=1) <= session_end, session_end)
        h, l, c = (bars[k].to_numpy(np.float64) for k in ("high", "low", "close"))
        tr, _, _ = _supertrend(c, h, l, _atr_rma(h, l, c, ATR_PERIOD), MULTIPLIER)
        out[name] = (closes.astype("int64").to_numpy(), tr.astype(np.int8))
    return out


DAILY_HISTORY = "5y"  # daily candles behind the golden-cross trend (long, so EMA200 has settled)
FRESH_CROSS_DAYS = 10
DAILY_CROSS_DAYS = 10   # sessions of daily EMA crosses listed on the Live tab


def daily_history(symbol, yf_ticker):
    """Daily candles (DAILY_HISTORY), downloaded once a day and cached."""
    cache = load_bars_cache()
    key = f"{yf_ticker}|1d"
    today = datetime.now(IST).date()
    hist = cache.get(key)
    if hist is None or hist.attrs.get("full_fetch") != str(today):
        got = _download(symbol, yf_ticker, DAILY_HISTORY, interval="1d")
        if got is not None:
            hist = got.copy()
            hist.attrs["full_fetch"] = str(today)
            cache[key] = hist
    return hist


def daily_ema_trend(symbol, yf_ticker, df15):
    """Daily EMA 50 / 100 / 200 trend, as on the TradingView "EMA 50, 100 and
    200" indicator: gc = +1 golden (EMA50 above EMA200) / -1 death, with the date
    of the last cross and sessions since; ec = early-cross state (EMA50 vs EMA100);
    ema200_dist = % price is above/below EMA200. Five years of daily candles are
    downloaded once a day (cached); today's close is the latest 15-min close."""
    hist = daily_history(symbol, yf_ticker)
    if hist is None or df15 is None or df15.empty:
        return {}
    last_day = df15.index[-1].date()
    closes = hist[hist.index.date < last_day]["close"]
    days = list(closes.index.date) + [last_day]
    c = pd.Series(list(closes.to_numpy()) + [float(df15["close"].iloc[-1])])
    if len(c) < 220:
        return {}
    e50, e100, e200 = (c.ewm(span=n, adjust=False).mean().to_numpy() for n in (50, 100, 200))

    def last_cross(fast, slow):
        side = np.sign(fast - slow)
        flips = np.nonzero(side[1:] != side[:-1])[0]
        k = int(flips[-1]) + 1 if len(flips) else 0
        return int(side[-1]), str(days[k]), int(len(side) - 1 - k)

    gc, gc_date, gc_days = last_cross(e50, e200)
    ec, ec_date, ec_days = last_cross(e50, e100)
    # Every daily cross since EMA200 settled (first 200 sessions skipped):
    # dx = the last DAILY_CROSS_DAYS sessions' crosses for the dashboard / alerts
    # ("f" = formed on today's still-open candle, so not final until 3:30 PM);
    # dxs = [kind 0 = 50×100 / 1 = 50×200, dir, % move in its favour until the
    # opposite cross (or to now), sessions held, still open] for the 5-year record.
    px = c.to_numpy()
    now = datetime.now(IST)
    last_open = last_day == now.date() and now.strftime("%H:%M") < "15:30"   # today's daily candle not closed yet
    dx, dxs = [], []
    for kind, slow in ((0, e100), (1, e200)):
        side = np.sign(e50 - slow)
        flips = [k for k in np.nonzero(side[1:] != side[:-1])[0] + 1 if k >= 200 and side[k] != 0]
        for j, k in enumerate(flips):
            d = int(side[k])
            end = flips[j + 1] if j + 1 < len(flips) else len(px) - 1
            mv = (px[end] / px[k] - 1) * 100 * d
            dxs.append([kind, d, round(float(mv), 2), int(end - k), int(j + 1 == len(flips))])
            if k >= len(px) - DAILY_CROSS_DAYS:
                dx.append({"k": "EMA_EARLY" if kind == 0 else "EMA_GOLDEN", "d": d, "date": str(days[k]),
                           "px": round(float(px[k]), 2), "f": int(k == len(px) - 1 and last_open)})
    return {"gc": gc, "gc_date": gc_date, "gc_days": gc_days, "ec": ec, "ec_date": ec_date,
            "ec_days": ec_days, "ema200_dist": round(float((c.iloc[-1] / e200[-1] - 1) * 100), 2),
            "dx": dx, "dxs": dxs}


def daily_cross_alerts(name, prev, cur):
    """Alerts for daily EMA crosses that became final (the candle closed) since
    the last scan. The first time a symbol is seen nothing fires — only the
    newest cross date is remembered — so old crosses never alert."""
    done = [x for x in cur.get("dx", []) if not x["f"]]
    newest = max((x["date"] for x in done), default=prev.get("dx_seen", ""))
    seen = prev.get("dx_seen")
    cur["dx_seen"] = max(newest, seen or "")
    if seen is None:
        return []
    return [(x["k"], name, "BUY" if x["d"] == 1 else "SELL", x["px"], x["date"]) for x in done if x["date"] > seen]


MOM_KEY = "__MOMENTUM__"
MOM_TOP, MOM_LOOKBACK, MOM_SKIP = 15, 12, 1
MOM_COST = 0.281   # % per delivery round trip (as momentum_backtest.py), charged on turnover
MOM_BENCH = {"NIFTY 500": "^CRSLDX", "Momentum ETF": "MOMENTUM50.NS"}  # Nifty500 Momentum 50 ETF


def _daily_frame(df):
    d = df.copy()
    d.index = d.index.tz_localize(None).normalize()
    return d[~d.index.duplicated(keep="last")]


def momentum_scores(daily, month):
    """12-1 momentum for the portfolio held in `month` (YYYY-MM): return from the
    close of month-14 to the close of month-2 — ranked at month-1's last close,
    skipping the most recent month. Same rule as momentum_backtest.py."""
    p = pd.Period(month, "M")
    end_p, start_p = p - 1 - MOM_SKIP, p - 1 - MOM_SKIP - MOM_LOOKBACK
    out = {}
    for sym, d in daily.items():
        me = d["close"].groupby(d.index.to_period("M")).last()
        if end_p in me.index and start_p in me.index and me[start_p] > 0:
            out[sym] = float(me[end_p] / me[start_p] - 1)
    return out


def _first_open(d, month):
    """(date, open) of the first session in `month`, or None."""
    if d is None or d.empty:
        return None
    rows = d[d.index.to_period("M") == pd.Period(month, "M")]
    return (str(rows.index[0].date()), float(rows["open"].iloc[0])) if len(rows) else None


def update_momentum(state, universe):
    """Paper-track the monthly momentum portfolio (top MOM_TOP by 12-1 momentum,
    equal weight, bought at the month's first open). On the first scan of a new
    month the previous month is closed at this month's first open, costs are
    charged on turnover, and the new portfolio is formed. Benchmarks over the same
    months: NIFTY 200, a listed Nifty200 Momentum 30 ETF, and an equal-weight
    portfolio of the whole universe. Stores a ready-to-render view in state."""
    cache = load_bars_cache()
    daily = {s: _daily_frame(cache[f"{s}.NS|1d"]) for s in universe if cache.get(f"{s}.NS|1d") is not None}
    if len(daily) < MOM_TOP * 3:
        return
    bench_daily, bench_now = {}, {}
    for name, ticker in MOM_BENCH.items():
        h = daily_history(name, ticker)
        if h is not None:
            bench_daily[name] = _daily_frame(h)
        try:
            b15 = fetch_15m(name, ticker)
            if b15 is not None and len(b15):
                bench_now[name] = float(b15["close"].iloc[-1])
        except Exception as e:  # noqa: BLE001
            print(f"[warn] momentum benchmark {name}: {e}")
    rec = state.setdefault(MOM_KEY, {"months": {}})
    months = rec["months"]
    month = f"{datetime.now(IST):%Y-%m}"

    def price_now(sym):
        info = state.get(sym) or {}
        if info.get("last_close"):
            return float(info["last_close"])
        return float(daily[sym]["close"].iloc[-1]) if sym in daily else None

    if month not in months:
        entries = {s: fo for s, d in daily.items() if (fo := _first_open(d, month))}
        scores = {k: v for k, v in momentum_scores(daily, month).items() if k in entries}
        if len(scores) >= MOM_TOP * 2:
            top = sorted(scores, key=scores.get, reverse=True)[:MOM_TOP]
            prev_key = max(months) if months else None
            if prev_key and not months[prev_key].get("closed"):
                pm = months[prev_key]
                gross = float(np.mean([(entries[s][1] if s in entries else (price_now(s) or pm["entry"][s]))
                                       / pm["entry"][s] - 1 for s in pm["holdings"]]))
                ew = [entries[s][1] / px - 1 for s, px in pm["ew_entry"].items() if s in entries and px]
                bench = {}
                for name, px in pm.get("bench_entry", {}).items():
                    fo = _first_open(bench_daily.get(name), month)
                    if px and fo:
                        bench[name] = fo[1] / px - 1
                pm.update(closed=True, exit_date=next(iter(entries.values()))[0],
                          gross=gross, net=gross - pm["cost"], ew=float(np.mean(ew)) if ew else None, bench=bench)
            prev_hold = set(months[prev_key]["holdings"]) if prev_key else set()
            turnover = len(set(top) - prev_hold) / MOM_TOP if prev_hold else 1.0
            months[month] = {
                "start": next(iter(entries.values()))[0], "holdings": top,
                "entry": {s: entries[s][1] for s in top}, "scores": {s: scores[s] for s in top},
                "turnover": turnover, "cost": (turnover if prev_hold else 0.5) * MOM_COST / 100,
                "ew_entry": {s: v[1] for s, v in entries.items()},
                "bench_entry": {n: fo[1] for n in MOM_BENCH if (fo := _first_open(bench_daily.get(n), month))},
                "closed": False}
            if not rec.get("since"):
                rec["since"] = month
    cur = months.get(month)
    if not cur:
        return
    for name in MOM_BENCH:           # benchmarks added later get this month's entry too
        if name not in cur.setdefault("bench_entry", {}):
            fo = _first_open(bench_daily.get(name), month)
            if fo:
                cur["bench_entry"][name] = fo[1]
    # ── live view
    rows = []
    for rank, sym in enumerate(sorted(cur["holdings"], key=lambda x: -cur["scores"][x]), 1):
        now = price_now(sym)
        rows.append({"sym": sym, "rank": rank, "score": cur["scores"][sym], "entry": cur["entry"][sym],
                     "now": now, "mtd": (now / cur["entry"][sym] - 1) if now else None})
    mtd = float(np.mean([r["mtd"] for r in rows if r["mtd"] is not None])) - cur["cost"]
    ew_now = [price_now(s) / px - 1 for s, px in cur["ew_entry"].items() if px and price_now(s)]
    bench_mtd = {n: bench_now[n] / px - 1 for n, px in cur.get("bench_entry", {}).items() if n in bench_now and px}
    nxt = (pd.Period(month, "M") + 1).strftime("%Y-%m")
    nscores = momentum_scores(daily, nxt)
    ranking = sorted(nscores, key=nscores.get, reverse=True)
    closed = [m for m in sorted(months) if months[m].get("closed")]

    def cum(key):
        vals = [months[m]["net"] if key == "net" else (months[m]["ew"] if key == "ew" else months[m]["bench"].get(key))
                for m in closed]
        live = {"net": mtd, "ew": float(np.mean(ew_now)) if ew_now else None}.get(key, bench_mtd.get(key))
        vals = [v for v in vals + [live] if v is not None]
        return float(np.prod([1 + v for v in vals]) - 1) if vals else None

    rec["view"] = {
        "month": month, "start": cur["start"], "since": rec.get("since"), "rows": rows, "mtd": mtd,
        "ew_mtd": float(np.mean(ew_now)) if ew_now else None, "bench_mtd": bench_mtd,
        "turnover": cur["turnover"], "next_month": nxt, "next_top": ranking[:MOM_TOP],
        "next_in": [s for s in ranking[:MOM_TOP] if s not in cur["holdings"]],
        "next_out": [s for s in cur["holdings"] if s not in ranking[:MOM_TOP]],
        "ranking": [(s, nscores[s]) for s in ranking[:60]],
        "history": [{"month": m, "net": months[m]["net"], "ew": months[m].get("ew"), "bench": months[m].get("bench", {}),
                     "turnover": months[m]["turnover"]} for m in closed],
        "cum": {"portfolio": cum("net"), "ew": cum("ew"), **{n: cum(n) for n in MOM_BENCH}},
        "prev_holdings": months[closed[-1]]["holdings"] if closed else [],
    }


def htf_at(htf, frame, ts):
    """Higher-timeframe trend (1 / -1, 0 if unknown) as of `ts` (tz-aware
    Timestamp), from the last candle of that frame that had closed by then."""
    if not htf or frame not in htf:
        return 0
    closes, tr = htf[frame]
    k = int(np.searchsorted(closes, ts.value, side="right")) - 1
    return int(tr[k]) if k >= 0 else 0


def _download(symbol, yf_ticker, period, interval="15m"):
    """One Yahoo download with exponential-backoff retry on rate limiting."""
    last_err = None
    for attempt in range(RETRY_MAX):
        try:
            ticker = yf.Ticker(yf_ticker, session=SESSION) if SESSION else yf.Ticker(yf_ticker)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                return None
            df = df.rename(columns=str.lower)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(IST)
            if interval != "1d":  # daily candles are stamped at midnight
                df = df.between_time("09:15", "15:30")
            return df.dropna(subset=["open", "close"])[["open", "high", "low", "close", "volume"]]
        except Exception as e:
            msg = str(e).lower()
            is_rate_limit = "rate" in msg or "429" in msg or "too many requests" in msg
            last_err = e
            if is_rate_limit:
                if attempt == RETRY_MAX - 1:
                    raise RateLimited(str(e))
                delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 3)
                print(f"[rate-limit] {symbol}: retry {attempt+1}/{RETRY_MAX} in {delay:.0f}s")
                time.sleep(delay)
            else:
                raise
    raise last_err


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def show_toast(title, message):
    """Borderless always-on-top popup pinned to the top-right corner."""
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    try:
        root.attributes("-alpha", 0.95)
    except tk.TclError:
        pass

    screen_w = root.winfo_screenwidth()
    x = screen_w - TOAST_WIDTH - TOAST_MARGIN
    y = TOAST_MARGIN
    root.geometry(f"{TOAST_WIDTH}x{TOAST_HEIGHT}+{x}+{y}")

    frame = tk.Frame(root, bg="#1e1e1e", padx=14, pady=12)
    frame.pack(fill="both", expand=True)
    title_color = "#f87171" if "SELL" in title else "#4ade80"
    tk.Label(frame, text=title, bg="#1e1e1e", fg=title_color,
              font=("Segoe UI", 11, "bold"), anchor="w", justify="left",
              wraplength=TOAST_WIDTH - 28).pack(fill="x")
    tk.Label(frame, text=message, bg="#1e1e1e", fg="#f0f0f0",
              font=("Segoe UI", 9), anchor="w", justify="left",
              wraplength=TOAST_WIDTH - 28).pack(fill="x", pady=(6, 0))

    root.after(TOAST_DURATION_MS, root.destroy)
    root.bind("<Button-1>", lambda e: root.destroy())
    root.mainloop()


TELEGRAM_MAX_CHARS = 3900  # Telegram's limit is 4096 per message


def send_telegram(text):
    """Send HTML-formatted text via the Telegram Bot API (stdlib only), split
    into several messages on line boundaries if it's too long. Returns True if
    every part was delivered."""
    if not TELEGRAM_TOKEN:
        return False
    parts, cur = [], ""
    for line in text.split("\n"):
        if cur and len(cur) + len(line) + 1 > TELEGRAM_MAX_CHARS:
            parts.append(cur)
            cur = ""
        cur = f"{cur}\n{line}" if cur else line
    parts.append(cur)
    return all(_send_telegram_part(part) for part in parts if part.strip())


def _send_telegram_part(text):
    data = urllib.parse.urlencode({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": "true",
    }).encode()
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, data=data, timeout=15) as resp:
                if json.loads(resp.read()).get("ok"):
                    return True
        except Exception as e:
            # don't print the URL — it contains the bot token
            print(f"[warn] Telegram send failed (attempt {attempt+1}/3): {type(e).__name__}: "
                  f"{str(e).replace(TELEGRAM_TOKEN, '***')}")
        time.sleep(2 * (attempt + 1))
    return False


def format_alert(fired, confirmed=(), html=True):
    """One scan's signals as a single message: 🟢 BUY and 🔴 SELL sections,
    one line per signal (ticker · price · strategy), then any new double
    confirmations. No date/time — Telegram timestamps every message."""
    b = (lambda x: f"<b>{html_lib.escape(x)}</b>") if html else (lambda x: x)

    def px(name, price):  # indices are points, not rupees
        return f"{price:,.2f}" if name in INDEX_TICKERS else f"₹{price:,.2f}"

    def line(sig):
        strategy, name, label, price, _ = sig
        strat = STRATEGY_NAMES.get(strategy, strategy)
        strong = " · Strong" if label.endswith("STRONG") else ""
        return f"{b(name)}  {px(name, price)}  ·  {strat}{strong}"

    sections = []
    for emoji, title, side in (("🟢", "BUY", False), ("🔴", "SELL", True)):
        rows = sorted((f for f in fired if ("SELL" in f[2]) == side), key=lambda f: (f[1], f[0]))
        if rows:
            sections.append(f"{emoji} {b(title)}\n" + "\n".join(line(f) for f in rows))
    if confirmed:
        sections.append(f"⭐ {b('Double confirmation')}\n" + "\n".join(
            f"{'🟢' if d == 'buy' else '🔴'} {b(name)}  {px(name, price)}  ·  SuperTrend + ADX DI {d.upper()}"
            for name, d, price in sorted(confirmed)))
    return "\n\n".join(sections)


STRATEGY_NAMES = {"SuperTrend": "SuperTrend", "ADX_DI": "ADX DI", "EMA_EARLY": "EMA 50×100 (daily)",
                  "EMA_GOLDEN": "EMA 50×200 (daily)", "CONFIRMED": "Both"}

TELEGRAM_MODE = os.environ.get("TELEGRAM_MODE", "all").strip().lower() or "all"
TELEGRAM_UNTIL = os.environ.get("TELEGRAM_UNTIL", "").strip()  # e.g. "14:00": no alerts after that
# Which names may alert on Telegram: comma-separated, or "all". Default: the NIFTY 50 index only.
TELEGRAM_SYMBOLS = os.environ.get("TELEGRAM_SYMBOLS", "").strip() or "NIFTY 50"


def telegram_filter(fired, confirmed, now=None, gate_flags=None):
    """Apply the Telegram mode (the dashboard always shows everything):
      all       — every signal + double confirmations (default)
      gated     — only signals whose 1-hour trend agrees, plus double
                  confirmations that pass the 1-hour gate
      confirmed — double confirmations + index signals only
    after limiting to TELEGRAM_SYMBOLS (default "NIFTY 50"; "all" = every
    stock and index), and TELEGRAM_UNTIL, a cut-off time after which nothing
    is sent."""
    now = now or datetime.now(IST)
    if TELEGRAM_SYMBOLS.lower() != "all":
        allowed = {x.strip().upper() for x in TELEGRAM_SYMBOLS.split(",") if x.strip()}
        fired = [f for f in fired if f[1].upper() in allowed]
        confirmed = [c for c in confirmed if c[0].upper() in allowed]
    if TELEGRAM_UNTIL:
        try:
            hh, mm = map(int, TELEGRAM_UNTIL.split(":"))
            if (now.hour, now.minute) > (hh, mm):
                return [], []
        except ValueError:
            print(f"[warn] TELEGRAM_UNTIL={TELEGRAM_UNTIL!r} isn't HH:MM — ignoring it")
    if TELEGRAM_MODE == "confirmed":
        return [f for f in fired if f[1] in INDEX_TICKERS], list(confirmed)
    if TELEGRAM_MODE == "gated" and gate_flags:
        ok = lambda name, label: all(gate_flags(name, label).values())  # noqa: E731
        return ([f for f in fired if ok(f[1], f[2])],
                [c for c in confirmed if ok(c[0], "SELL" if c[1] == "sell" else "BUY")])
    return list(fired), list(confirmed)


def notify_signals(fired, confirmed=(), gate_flags=None):
    """Send one scan's signals: Telegram digest, console, desktop popup."""
    if not fired and not confirmed:
        return
    n_sell = sum("SELL" in f[2] for f in fired)
    if os.environ.get("CI"):
        # GitHub Actions logs are public for a public repo — keep signals out of them
        print(f"[signal] {len(fired)} fired (details hidden from CI logs — see Telegram / dashboard)")
    else:
        print("\n" + format_alert(fired, confirmed, html=False) + "\n")
    tg_fired, tg_conf = telegram_filter(fired, confirmed, gate_flags=gate_flags)
    if tg_fired or tg_conf:
        send_telegram(format_alert(tg_fired, tg_conf))  # before the popup, which blocks
    if DESKTOP_NOTIFY:
        try:
            body = format_alert(fired, confirmed, html=False).splitlines()
            more = f"\n… and {len(body) - 8} more lines" if len(body) > 8 else ""
            show_toast(f"{len(fired) - n_sell} BUY · {n_sell} SELL", "\n".join(body[:8]) + more)
        except Exception as e:
            print(f"[warn] desktop popup failed: {e}")


# ─── Dashboard ──────────────────────────────────────────────────────────────

DASHBOARD_FILE = Path(__file__).parent / "dashboard.html"
SIGNAL_LOG_KEY = "__SIGNALS__"   # live fired signals, kept in the state file
SIGNAL_LOG_MAX = 300
SCAN_META_KEY = "__SCAN__"       # last scan time / coverage
ADX_LABELS = {1.0: "BUY_STRONG", 0.5: "BUY", 0.0: "—", -0.5: "SELL", -1.0: "SELL_STRONG"}
SESSION_BARS = 25                # 15-min bars per NSE session (9:15–3:30)

DASHBOARD_CSS = """
:root{--bg:#f5f6f8;--panel:#fff;--fg:#0f172a;--muted:#64748b;--faint:#94a3b8;--line:#e2e8f0;
--head:#f8fafc;--hover:#f1f5f9;--up:#059669;--up-bg:#ecfdf5;--up-bd:#a7f3d0;--dn:#dc2626;--dn-bg:#fef2f2;
--dn-bd:#fecaca;--accent:#4f46e5;--accent-bg:#eef2ff;--hl:#fffbeb;--warn:#b45309;--warn-bg:#fffbeb;
--shadow:0 1px 2px rgba(15,23,42,.04),0 1px 3px rgba(15,23,42,.06)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;DARK}}
:root[data-theme="dark"]{color-scheme:dark;DARK}
:root[data-theme="light"]{color-scheme:light}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 Inter,-apple-system,BlinkMacSystemFont,
"Segoe UI",Roboto,sans-serif;font-feature-settings:"tnum" 1,"cv11" 1;-webkit-font-smoothing:antialiased}
.wrap{max-width:1680px;margin:0 auto;padding:0 clamp(14px,2vw,32px)}
/* top bar */
.top{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--bg) 88%,transparent);
backdrop-filter:saturate(1.4) blur(10px);border-bottom:1px solid var(--line)}
.bar1{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 0 10px}
.brand{display:flex;align-items:center;gap:10px;min-width:0}
.logo{width:30px;height:30px;border-radius:8px;background:var(--fg);color:var(--bg);display:grid;
place-items:center;font-weight:800;font-size:13px;letter-spacing:-.02em;flex:none}
.brand h1{font-size:16px;margin:0;letter-spacing:-.01em;white-space:nowrap}
.brand .sub{font-size:12px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tools-r{display:flex;align-items:center;gap:8px;flex:none}
.status{display:inline-flex;align-items:center;gap:7px;height:30px;padding:0 11px;border-radius:999px;
font-size:12px;font-weight:600;border:1px solid var(--line);background:var(--panel);white-space:nowrap}
.status i{width:7px;height:7px;border-radius:50%;background:var(--faint)}
.status.live i{background:var(--up);box-shadow:0 0 0 3px color-mix(in srgb,var(--up) 25%,transparent);
animation:pulse 2s infinite}
.status.old{color:var(--warn);background:var(--warn-bg);border-color:transparent}.status.old i{background:var(--warn)}
@keyframes pulse{50%{opacity:.4}}
.iconbtn{width:30px;height:30px;display:grid;place-items:center;border-radius:8px;border:1px solid var(--line);
background:var(--panel);color:var(--fg);cursor:pointer;padding:0}.iconbtn svg{width:15px;height:15px}
.theme .sun{display:none}:root[data-theme="dark"] .theme .sun{display:block}
:root[data-theme="dark"] .theme .moon{display:none}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .theme .sun{display:block}
:root:not([data-theme="light"]) .theme .moon{display:none}}
.tabs{display:flex;gap:4px;overflow-x:auto;scrollbar-width:none;margin-bottom:-1px}
.tabs::-webkit-scrollbar{display:none}
.tab{appearance:none;border:0;background:none;color:var(--muted);font:inherit;font-weight:600;font-size:13px;
padding:9px 12px 11px;border-bottom:2px solid transparent;cursor:pointer;white-space:nowrap}
.tab:hover{color:var(--fg)}.tab.on{color:var(--fg);border-bottom-color:var(--fg)}
.tab .n{margin-left:6px;font-size:11px;font-weight:600;color:var(--muted);background:var(--hover);
padding:1px 6px;border-radius:999px}
/* layout */
main{padding:20px 0 40px}.view{display:none}.view.on{display:block}
section{margin-bottom:26px}
.sh{display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin:0 0 10px;flex-wrap:wrap}
.sh h2{font-size:15px;margin:0;letter-spacing:-.01em}.sh .meta{font-size:12px;color:var(--muted)}
.note{font-size:12px;color:var(--muted);margin:-4px 0 12px;max-width:820px}
.note.below{margin:14px 0 0;max-width:1100px;line-height:1.6}
details.note>summary{cursor:pointer;font-weight:600;width:max-content;max-width:100%}
details.note[open]>summary{margin-bottom:4px}
details.more{margin-top:26px}
details.more>summary{cursor:pointer;font-size:14px;font-weight:700;padding:11px 14px;border:1px solid var(--line);
border-radius:10px;background:var(--panel)}
details.more[open]>summary{margin-bottom:4px}
.fbtn{height:30px;padding:0 12px;border-radius:999px;border:1px solid var(--line);background:var(--panel);color:var(--muted);
font:inherit;font-size:12px;font-weight:700;cursor:pointer}
.fbtn.on{border-color:var(--accent);color:var(--accent)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow)}
/* index strip */
.idx{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:22px}
.ix{padding:11px 13px;min-width:0}
.ix .nm{font-size:11px;font-weight:700;color:var(--muted);letter-spacing:.03em;white-space:nowrap;
overflow:hidden;text-overflow:ellipsis}
.ix .px{font-size:17px;font-weight:700;letter-spacing:-.01em;margin:3px 0 1px}
.ix .ft{display:flex;align-items:center;justify-content:space-between;gap:6px;font-size:12px}
.ix .tags{display:flex;gap:4px;margin-top:7px}
/* kpis */
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:26px}
.kpis.k3{grid-template-columns:repeat(3,1fr);margin-top:-16px}
.kpis.k5{grid-template-columns:repeat(5,1fr)}
.kpi{padding:13px 15px}.kpi .k{font-size:12px;color:var(--muted);font-weight:500}
.kpi .v{font-size:24px;font-weight:700;letter-spacing:-.02em;margin:2px 0}
.kpi .s{font-size:12px;color:var(--muted)}
.split{display:flex;height:4px;border-radius:2px;overflow:hidden;background:var(--line);margin-top:9px}
.split b{background:var(--up)}.split i{background:var(--dn)}
/* tags */
.tag{display:inline-flex;align-items:center;height:20px;padding:0 7px;border-radius:5px;font-size:11px;
font-weight:700;letter-spacing:.02em;white-space:nowrap;border:1px solid transparent}
.tag.up{color:var(--up);background:var(--up-bg);border-color:var(--up-bd)}
.tag.dn{color:var(--dn);background:var(--dn-bg);border-color:var(--dn-bd)}
.tag.mute{color:var(--muted);background:var(--hover)}
.tag.ok{color:var(--accent);background:var(--accent-bg)}
.up{color:var(--up)}.dn{color:var(--dn)}.mu{color:var(--muted)}.fa{color:var(--faint)}
/* toolbars */
.tb{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px}
.seg{display:inline-flex;padding:2px;border:1px solid var(--line);border-radius:9px;background:var(--panel)}
.seg button{appearance:none;border:0;background:none;color:var(--muted);font:inherit;font-size:12px;
font-weight:600;padding:5px 10px;border-radius:7px;cursor:pointer;white-space:nowrap}
.seg button.on{background:var(--fg);color:var(--bg)}
.tb input[type=search]{flex:1 1 180px;min-width:140px;height:32px;padding:0 11px;border:1px solid var(--line);
border-radius:9px;background:var(--panel);color:var(--fg);font:inherit;font-size:13px}
.tb input:focus{outline:2px solid var(--accent);outline-offset:-1px}
.sel{height:32px;padding:0 30px 0 11px;border:1px solid var(--line);border-radius:9px;background:var(--panel);
color:var(--fg);font:inherit;font-size:13px;font-weight:600;cursor:pointer}
.sel:focus{outline:2px solid var(--accent);outline-offset:-1px}
.sel.dl{padding:0 12px;background-image:none;appearance:none}.sel.dl:hover{border-color:var(--accent)}
.chk{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;color:var(--muted);
cursor:pointer;user-select:none}.chk input{accent-color:var(--accent)}
.daysum{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12px;color:var(--muted);margin:2px 0 10px}
.daysum b{color:var(--fg);font-weight:600}
/* tables */
.tw{overflow:auto;max-height:72vh;border-radius:12px}
table{border-collapse:separate;border-spacing:0;width:100%;font-size:13px}
th{position:sticky;top:0;z-index:1;background:var(--head);color:var(--muted);font-size:11px;font-weight:600;
text-transform:uppercase;letter-spacing:.04em;text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);
white-space:nowrap;cursor:default;user-select:none}
th[data-k]{cursor:pointer}th[data-k]:hover{color:var(--fg)}
th.sorted::after{content:" ↑"}th.sorted.desc::after{content:" ↓"}
td{padding:9px 12px;border-bottom:1px solid var(--line);white-space:nowrap;vertical-align:middle}
tbody tr:last-child td{border-bottom:0}tbody tr:hover td{background:var(--hover)}
tr.fresh td{background:var(--hl)}
.r{text-align:right}.sym{font-weight:600}.sm{font-size:11px;color:var(--faint)}
.empty{padding:28px 16px;text-align:center;color:var(--muted);font-size:13px}
/* accuracy */
.wr{display:flex;align-items:center;gap:8px;min-width:120px}
.wr .tr{flex:1;height:6px;border-radius:3px;background:var(--line);overflow:hidden;min-width:50px}
.wr .tr b{display:block;height:100%;background:var(--up)}.wr .tr b.lo{background:var(--dn)}
tr.grp td{background:var(--head);font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;
letter-spacing:.04em;padding:7px 12px}
a.tv{color:inherit;text-decoration:none}a.tv:hover{text-decoration:underline;color:var(--accent)}
.star{appearance:none;border:0;background:none;color:var(--faint);cursor:pointer;font-size:15px;padding:0 6px 0 0;line-height:1}
.star.on{color:#f59e0b}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}.grid2 section{margin:0}
tr.secrow{cursor:pointer}tr.secrow.on td{background:var(--accent-bg)}
#eq svg{display:block;width:100%;height:220px}#eq .lg{display:flex;gap:16px;font-size:12px;color:var(--muted);margin-bottom:6px}
#eq .lg i{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}
@media (max-width:980px){.grid2{grid-template-columns:1fr}.kpis.k5{grid-template-columns:repeat(3,1fr)}}
.foot{font-size:12px;color:var(--faint);padding:14px 0 0;border-top:1px solid var(--line);line-height:1.6}
@media (max-width:1100px){.hide-md{display:none}}
@media (max-width:980px){.idx{grid-template-columns:repeat(3,1fr)}}
@media (max-width:700px){.wrap{padding:0 14px}.brand .sub{display:none}
.idx{display:flex;overflow-x:auto;scroll-snap-type:x mandatory;margin:0 -14px 20px;padding:0 14px 4px}
.idx .ix{flex:0 0 46%;scroll-snap-align:start}.kpis{grid-template-columns:repeat(2,1fr)}
.kpis.k5{grid-template-columns:repeat(2,1fr)}
.kpis.k3{grid-template-columns:1fr}
.hide-sm{display:none}td,th{padding:8px 9px}.status .lbl{display:none}.status{padding:0 9px}}
""".replace("DARK", """--bg:#000;--panel:#000;--fg:#f1f5f9;--muted:#8a94a6;--faint:#5b6577;--line:#1c1f26;
--head:#07080a;--hover:#0d0f13;--up:#34d399;--up-bg:#04170f;--up-bd:#0b3a26;--dn:#f87171;--dn-bg:#1c0607;
--dn-bd:#4a1416;--accent:#818cf8;--accent-bg:#0e0f24;--hl:#141005;--warn:#fbbf24;--warn-bg:#191204;--shadow:none""")

DASHBOARD_JS = """
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const esc=x=>String(x).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
/* theme */
$('#theme').onclick=()=>{const r=document.documentElement;
 const dark=r.dataset.theme?r.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;
 r.dataset.theme=dark?'light':'dark';try{localStorage.setItem('theme',r.dataset.theme)}catch(e){}};
/* status */
(()=>{const t=new Date(document.body.dataset.scanned),st=$('#status'),m=(Date.now()-t)/6e4;if(isNaN(m))return;
 const ago=m<1?'just now':m<120?Math.round(m)+' min ago':m<2880?Math.round(m/60)+' h ago':Math.round(m/1440)+' days ago';
 if(m<=25){st.classList.add('live');st.querySelector('.lbl').textContent='Live · '+ago;}
 else{st.classList.add('old');st.querySelector('.lbl').textContent='Updated '+ago;
 st.title='Scans run Mon–Fri 9:15 AM – 3:30 PM IST';}})();
/* tabs */
function show(id){$$('.tab').forEach(t=>t.classList.toggle('on',t.dataset.v===id));
 $$('.view').forEach(v=>v.classList.toggle('on',v.id==='v-'+id));try{localStorage.setItem('tab',id)}catch(e){}}
$$('.tab').forEach(t=>t.onclick=()=>{show(t.dataset.v);history.replaceState(null,'','#'+t.dataset.v)});
const ok=v=>v&&$('#v-'+v);let first=location.hash.slice(1);
if(!ok(first)){try{first=localStorage.getItem('tab')}catch(e){}}show(ok(first)?first:'live');
/* favourites (per device) */
let FAV=new Set();try{FAV=new Set(JSON.parse(localStorage.getItem('favs')||'[]'))}catch(e){}
const saveFav=()=>{try{localStorage.setItem('favs',JSON.stringify([...FAV]))}catch(e){}};
function paintStars(){$$('.star').forEach(b=>{const on=FAV.has(b.dataset.fav);b.classList.toggle('on',on);b.textContent=on?'★':'☆'})}
document.addEventListener('click',ev=>{const b=ev.target.closest('.star');if(!b)return;const k=b.dataset.fav;
 FAV.has(k)?FAV.delete(k):FAV.add(k);saveFav();paintStars();if(typeof renderPast==='function')renderPast();stk&&stk.run()});
const tvUrl=n=>'https://www.tradingview.com/chart/?interval=15&symbol='+encodeURIComponent('NSE:'+(TVI[n]||n.replace(/-/g,'_')));
const tvA=n=>'<a class="tv sym" href="'+tvUrl(n)+'" target="_blank" rel="noopener" title="Open '+esc(n)+' on TradingView (15 min)">'+esc(n)+'</a>';
/* segmented controls + filters */
function seg(el,fn){const bs=$$('button',el);bs.forEach(b=>b.onclick=()=>{bs.forEach(x=>x.classList.remove('on'));
 b.classList.add('on');fn(b.dataset.f)})}
function filterer(tbodySel,countSel){const rows=$$(tbodySel+' tr[data-sym]'),f={};
 const run=()=>{let n=0;for(const r of rows){const d=r.dataset;let ok=true;
  for(const[k,v]of Object.entries(f)){if(v===''||v==='all')continue;
   if(k==='q'){if(!d.sym.includes(v))ok=false}else if(k==='conf'){if(v&&d.conf!=='1')ok=false}
   else if(k==='focus'){ok=ok&&inFocus(d.sym,d.t||'12:00')}
   else if(k==='view'){ok=ok&&VIEWS[v](d)}else if(d[k]!==v)ok=false}
  r.style.display=ok?'':'none';n+=ok}
  if(countSel)$(countSel).textContent=n+' shown';const e=$(tbodySel+' .nores');if(e)e.style.display=n?'none':'';};
 return {set(k,v){f[k]=v;run()},run}}
const VIEWS={all:()=>true,today:d=>d.today==='1'||FAV.has(d.sym),golden:d=>d.gc==='1',death:d=>d.gc==='-1',fresh:d=>+d.gcd<=10,fav:d=>FAV.has(d.sym),stale:d=>d.stale==='1',aligned:d=>(d.st==='1'&&+d.adx>0)||(d.st==='-1'&&+d.adx<0),near:d=>+d.dist<=NEAR,
 long:d=>d.st==='1',short:d=>d.st==='-1',abuy:d=>+d.adx>0,asell:d=>+d.adx<0};
/* Focus: Nifty 100 stocks + indices, signal candles 9:45 AM – 2:30 PM (switch in the header) */
let FOCUS=true;try{FOCUS=localStorage.getItem('focus')!=='0'}catch(e){}
const inFocus=(n,t)=>!FOCUS||(N100.has(n)&&t>=FOCUS_WIN[0]&&t<FOCUS_WIN[1]);
const live=filterer('#t-live tbody','#c-live'),conf=filterer('#t-conf tbody','#c-conf'),dxf=filterer('#t-dx tbody','#c-dx');
seg($('#f-live-dir'),v=>live.set('dir',v));
/* shared archive (ARCH/NAMES/PX/PDAYS come from the data script).
   ARCH row: [time, name#, label, strategy 0 ST/1 ADX/2 CONFIRMED, price, exit, bars, open, max gain, max drawdown] */
const SN=['SuperTrend','ADX_DI','CONFIRMED'];
const SNAME={SuperTrend:'SuperTrend',ADX_DI:'ADX DI',CONFIRMED:'Both'};
const isSig=r=>r[3]!==2;
const CONF=new Set(ARCH.filter(r=>r[3]===2).map(r=>r[0].slice(0,10)+'|'+r[1]));
const mvOf=r=>{const px=r[7]?PX[r[1]]:r[5];if(!px||!r[4])return null;const m=(px/r[4]-1)*100;return r[2].includes('SELL')?-m:m};
/* past signals, drawn per selected day:
   [time, name, label, strategy, price, exit, move, bars, open, confirmed, max gain, max drawdown, 1H gate] */
const PAST=ARCH.filter(r=>isSig(r)&&PDAYS.has(r[0].slice(0,10))).reverse().map(r=>[r[0],NAMES[r[1]],r[2],SN[r[3]],r[4],
 r[7]?null:r[5],mvOf(r),r[6],r[7],CONF.has(r[0].slice(0,10)+'|'+r[1])?1:0,r[8],r[9],r[12]]);
const PT=$('#t-past tbody'),pf={day:'',dir:'all',strat:'all',conf:false,sym:'',sess:'',str:'all',fav:false};let pk='t',pa=false;
const fp=v=>v==null?'—':v.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
const tg=l=>'<span class="tag '+(l.includes('SELL')?'dn':'up')+'">'+l.replace('_',' ')+'</span>';
const pc=v=>v==null?'<span class="fa">—</span>':Math.abs(v)<0.005?'<span class="mu">0.00%</span>':
 '<span class="'+(v>0?'up':'dn')+'">'+(v>0?'+':'')+v.toFixed(2)+'%</span>';
const t12=t=>{let[h,m]=t.slice(11,16).split(':').map(Number);const ap=h>=12?'PM':'AM';return(h%12||12)+':'+String(m).padStart(2,'0')+' '+ap};
const MON=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const dshort=t=>+t.slice(8,10)+' '+MON[+t.slice(5,7)-1];
const bt=(b,o)=>b==null?'—':o?(b===0?'<span class="tag ok">new</span>':'<span class="mu">'+b+'+</span>'):b;
let PROWS=[];
function renderPast(){const rows=PAST.filter(r=>r[0].startsWith(pf.day)&&inFocus(r[1],r[0].slice(11,16))&&(pf.dir==='all'||(pf.dir==='sell')===r[2].includes('SELL'))
 &&(pf.strat==='all'||r[3]===pf.strat)&&(!pf.conf||r[9])&&(!pf.sym||r[1]===pf.sym)&&(!pf.fav||FAV.has(r[1]))
 &&(pf.str==='all'||(pf.str==='strong')===r[2].endsWith('STRONG'))
 &&(!pf.sess||(r[0].slice(11,16)>=pf.sess.slice(0,5)&&r[0].slice(11,16)<pf.sess.slice(6))));
 const i={t:0,sym:1,mv:6,bars:7,mfe:10,mae:11}[pk];rows.sort((a,b)=>{const x=a[i],y=b[i];
  const v=typeof x==='string'?x.localeCompare(y):(x??-999)-(y??-999);return pa?v:-v});
 PT.innerHTML=rows.map(r=>'<tr><td class="mu">'+(pf.day?'':dshort(r[0])+', ')+t12(r[0])+'</td><td>'+tvA(r[1])+(r[9]?' <span class="tag ok">✓</span>':'')+
  '</td><td>'+tg(r[2])+'</td><td class="mu hide-md">'+SNAME[r[3]]+'</td><td class="r hide-sm">'+fp(r[4])+
  '</td><td class="r hide-md">'+(r[8]?'<span class="fa">open</span>':fp(r[5]))+'</td><td class="r">'+pc(r[6])+'</td><td class="r">'+pc(r[10])+'</td><td class="r hide-sm">'+pc(r[11])+
  '</td><td class="r">'+bt(r[7],r[8])+'</td></tr>').join('')
  ||'<tr><td colspan="11" class="empty">No signals match these filters.</td></tr>';
 PROWS=rows;$('#c-past').textContent=rows.length+' shown';pastSummary(rows);}
/* summary line for exactly the rows shown (follows every filter) */
function pastSummary(rows){const cl=rows.filter(r=>!r[8]&&r[6]!=null),b=rows.filter(r=>!r[2].includes('SELL')).length;
 const conf=new Set(rows.filter(r=>r[9]).map(r=>r[0].slice(0,10)+'|'+r[1])).size;
 const win=cl.length?cl.filter(r=>r[6]>0).length/cl.length*100:null,avg=cl.length?cl.reduce((x,r)=>x+r[6],0)/cl.length:null;
 const sp=(h)=>'<span>'+h+'</span>';
 $('#past-sum').innerHTML=rows.length?sp('<b>'+rows.length+'</b> signals')+'<span class="up">'+b+' buy</span><span class="dn">'+(rows.length-b)+' sell</span>'+
  sp('<b>'+conf+'</b> double confirmations')+(cl.length?sp('median <b>'+num(med(cl.map(r=>r[7])))+'</b> bars to flip')+
  sp('win rate <b>'+num(win,0,'%')+'</b> ('+cl.length+' closed)')+sp('avg move <b>'+pc(avg)+'</b>')+
  sp('median max gain <b>'+pc(med(cl.map(r=>r[10]).filter(v=>v!=null)))+'</b> · max drawdown <b>'+pc(med(cl.map(r=>r[11]).filter(v=>v!=null)))+'</b>'):''):''}
/* Excel download: a CSV (UTF-8 with BOM, so Excel opens it with ₹/× intact) */
function dlSignals(rows,name){const LF=String.fromCharCode(10),CRLF=String.fromCharCode(13,10),BOM=String.fromCharCode(65279);
 const q=v=>{v=v==null?'':String(v);return v.includes(',')||v.includes('"')||v.includes(LF)?'"'+v.split('"').join('""')+'"':v};
 const n=v=>v==null?'':(Math.abs(v)<0.005?0:+v).toFixed(2);
 const out=[['Date','Time','Symbol','Signal','Strategy','Entry price','Exit price','Status','Move %','Max gain %','Max drawdown %','Bars to flip','Double confirmation day']];
 for(const r of rows)out.push([r[0].slice(0,10),t12(r[0]),r[1],r[2].replace('_',' '),SNAME[r[3]],n(r[4]),r[8]?'':n(r[5]),r[8]?'Open':'Closed',
  n(r[6]),n(r[10]),n(r[11]),r[7]??'',r[9]?'Yes':'No']);
 const blob=new Blob([BOM+out.map(x=>x.map(q).join(',')).join(CRLF)],{type:'text/csv;charset=utf-8'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;document.body.appendChild(a);a.click();
 setTimeout(()=>{URL.revokeObjectURL(a.href);a.remove()},500)}
const ALLSIG=ARCH.filter(r=>isSig(r)).map(r=>[r[0],NAMES[r[1]],r[2],SN[r[3]],r[4],r[7]?null:r[5],mvOf(r),r[6],r[7],
 CONF.has(r[0].slice(0,10)+'|'+r[1])?1:0,r[8],r[9]]).sort((a,b)=>b[0].localeCompare(a[0]));
$('#dl-shown').onclick=()=>dlSignals(PROWS,'signals_'+(pf.day||'all-days')+'.csv');
$('#dl-all').onclick=()=>dlSignals(ALLSIG,'all_signals_'+(ALLSIG.length?ALLSIG[ALLSIG.length-1][0].slice(0,10)+'_to_'+ALLSIG[0][0].slice(0,10):'none')+'.csv');
$$('#t-past th[data-k]').forEach(th=>th.onclick=()=>{const k=th.dataset.k;pa=k===pk?!pa:th.dataset.asc==='1';pk=k;
 $$('#t-past th').forEach(x=>x.classList.remove('sorted','desc'));th.classList.add('sorted');if(!pa)th.classList.add('desc');renderPast()});
const daySel=$('#f-past-day');
daySel.onchange=()=>{pf.day=daySel.value;renderPast()};
seg($('#f-past-dir'),v=>{pf.dir=v;renderPast()});seg($('#f-past-strat'),v=>{pf.strat=v;renderPast()});
$('#f-past-conf').onchange=e=>{pf.conf=e.target.checked;renderPast()};
$('#f-past-sym').onchange=e=>{pf.sym=e.target.value;renderPast()};
$('#f-past-sess').onchange=e=>{pf.sess=e.target.value;renderPast()};
seg($('#f-past-str'),v=>{pf.str=v;renderPast()});
$('#f-past-fav').onchange=e=>{pf.fav=e.target.checked;renderPast()};
daySel.onchange();
/* accuracy: overall or for one stock / index */
function med(a){if(!a.length)return null;const s=[...a].sort((x,y)=>x-y),m=s.length>>1;return s.length%2?s[m]:(s[m-1]+s[m])/2}
function stats(rows){const c=[],o=[],b=[],g=[],d=[];for(const r of rows){const m=mvOf(r);if(m==null)continue;
 if(r[7])o.push(m);else{c.push(m);if(r[6]!=null)b.push(r[6]);if(r[8]!=null){g.push(r[8]);d.push(r[9])}}}
 return{n:rows.length,closed:c.length,open:o.length,win:c.length?c.filter(x=>x>0).length/c.length*100:null,bars:med(b),
  move:med(c),avg:c.length?c.reduce((x,y)=>x+y,0)/c.length:null,open_move:med(o),mfe:med(g),mae:med(d),
  all_avg:(c.length+o.length)?c.concat(o).reduce((x,y)=>x+y,0)/(c.length+o.length):null,
  open_share:(c.length+o.length)?o.length/(c.length+o.length):0}}
const kpi=(k,v,s)=>'<div class="panel kpi"><div class="k">'+k+'</div><div class="v">'+v+'</div><div class="s">'+s+'</div></div>';
const big=v=>v==null?'—':Math.abs(v)<0.005?'<span class="mu">0.00%</span>':'<span class="'+(v>0?'up':'dn')+'">'+(v>0?'+':'')+v.toFixed(2)+'%</span>';
function num(v,d=0,suf=''){return v==null?'—':v.toFixed(d)+suf}
function accRow(tag,rows){const a=stats(rows);
 const warn=a.open_share>0.3?' <span class="tag mute" title="'+Math.round(a.open_share*100)+'% of these signals are still open. Closed-only figures (win rate, median move) are biased towards quick reversals — use Avg incl. open.">⚠ '+Math.round(a.open_share*100)+'% open</span>':'';
 const win=a.win==null?'<span class="fa">—</span>':'<div class="wr"><span style="min-width:38px">'+a.win.toFixed(0)+
  '%</span><span class="tr"><b class="'+(a.win<50?'lo':'')+'" style="width:'+a.win.toFixed(0)+'%"></b></span></div>';
 return '<tr><td>'+tg(tag)+'</td><td class="r">'+a.n+'</td><td class="r hide-sm">'+a.closed+'</td><td>'+win+'</td><td class="r">'+
  num(a.bars)+'</td><td class="r">'+pc(a.move)+'</td><td class="r">'+pc(a.mfe)+'</td><td class="r">'+pc(a.mae)+
  '</td><td class="r hide-sm">'+pc(a.avg)+'</td><td class="r"><b>'+pc(a.all_avg)+'</b></td><td class="r hide-sm">'+a.open+' · '+pc(a.open_move)+warn+'</td></tr>'}
function renderAcc(sym){const ni=sym?NAMES.indexOf(sym):-1;
 const rows=ARCH.filter(r=>(!sym||r[1]===ni)&&inFocus(NAMES[r[1]],r[0].slice(11,16)));
 const sig=rows.filter(r=>isSig(r)),cf=rows.filter(r=>r[3]===2),all=stats(sig),con=stats(cf);
 const days=[...new Set(sig.map(r=>r[0].slice(0,10)))].sort();
 $('#acc-span').textContent=days.length?dshort(days[0])+' – '+dshort(days[days.length-1])+' · '+days.length+' sessions':'no signals';
 $('#acc-what').textContent=(sym?'Accuracy for '+sym+' only':FOCUS?'Focus: Nifty 100 + indices, 9:45 AM – 2:30 PM':'All '+NAMES.length+' stocks & indices, all day');
 $('#acc-kpis').innerHTML=kpi('Signals tracked',all.n,all.closed+' closed · '+all.open+' open')+
  kpi('Win rate (closed)',num(all.win,0,'%'),'moved in the signal’s favour by its flip')+
  kpi('Median bars to flip',num(all.bars),'≈ '+(all.bars==null?'—':(all.bars*15/60).toFixed(1)+' h')+' of trading')+
  kpi('Confirmed setups win rate',num(con.win,0,'%'),con.closed+' closed · median '+pc(con.move));
 $('#acc-k3').innerHTML=[['Median move at flip','move','where price was when the strategy flipped'],
  ['Median max gain','mfe','furthest in the signal’s favour before the flip'],
  ['Median max drawdown','mae','furthest against the signal before the flip']].map(([t,k,h])=>kpi(t,big(all[k]),h+' · confirmed '+pc(con[k]))).join('');
 const st=sig.filter(r=>r[3]===0),ad=sig.filter(r=>r[3]===1),
  L=(a,l)=>a.filter(r=>r[2]===l),grp=t=>'<tr class="grp"><td colspan="11">'+t+'</td></tr>';
 $('#acc-body').innerHTML=grp('SuperTrend')+accRow('BUY',L(st,'BUY'))+accRow('SELL',L(st,'SELL'))+accRow('ALL',st)+
  grp('ADX DI')+accRow('BUY',L(ad,'BUY'))+accRow('BUY_STRONG',L(ad,'BUY_STRONG'))+accRow('SELL',L(ad,'SELL'))+
  accRow('SELL_STRONG',L(ad,'SELL_STRONG'))+accRow('ALL',ad)+
  grp('Double confirmation — SuperTrend + ADX DI (enter on 2nd signal, exit on 1st flip)')+
  accRow('CONFIRMED BUY',L(cf,'BUY'))+accRow('CONFIRMED SELL',L(cf,'SELL'));
 accRows=rows;renderSim(rows);renderHours(sig);$('#lb-wrap').style.display=sym?'none':'';if(!sym)renderLeaders(sig);
 const w=$('#acc-log-wrap');w.style.display=sym?'':'none';if(!sym)return;
 const log=[...rows].reverse();$('#acc-log-title').textContent=sym+' — every signal';$('#acc-log-count').textContent=log.length+' signals';
 $('#acc-log').innerHTML=log.map(r=>'<tr><td class="mu">'+dshort(r[0])+', '+t12(r[0])+'</td><td>'+tg(r[3]===2?'CONFIRMED '+r[2]:r[2])+
  '</td><td class="mu hide-sm">'+SNAME[SN[r[3]]]+'</td><td class="r hide-sm">'+fp(r[4])+
  '</td><td class="r hide-md">'+(r[7]?'<span class="fa">open</span>':fp(r[5]))+'</td><td class="r">'+pc(mvOf(r))+'</td><td class="r">'+
  pc(r[8])+'</td><td class="r hide-sm">'+pc(r[9])+'</td><td class="r">'+bt(r[6],r[7])+'</td></tr>').join('')||
  '<tr><td colspan="9" class="empty">No signals for this one yet.</td></tr>'}
/* exit-rule simulator: ARCH r[10]/r[11] = bars until price first moved SIM_LEVELS[i]% in / against the
   signal's favour (0 = never). Same-candle touches count as the stop (cautious). */
const simT=$('#sim-t'),simS=$('#sim-s');
[simT,simS].forEach((el,k)=>{el.innerHTML='<option value="-1">None</option>'+SIM_LEVELS.map((v,i)=>'<option value="'+i+'">'+v+'%</option>').join('');});
simT.value='1';simS.value='-1';
function simOne(r,ti,si){const tb=ti<0?0:(r[10][ti]||0),sb=si<0?0:(r[11][si]||0);
 if(tb&&(!sb||tb<sb))return[SIM_LEVELS[ti],'t'];if(sb)return[-SIM_LEVELS[si],'s'];
 if(r[7])return[null,'o'];return[mvOf(r),'f']}
function simRow(label,rows,ti,si){let res=[],t=0,st=0,f=0;const base=[];
 for(const r of rows){const[v,k]=simOne(r,ti,si);if(v==null)continue;res.push(v);k==='t'?t++:k==='s'?st++:f++;}
 for(const r of rows){if(!r[7]){const m=mvOf(r);if(m!=null)base.push(m)}}
 const avg=res.length?res.reduce((a,b)=>a+b,0)/res.length:null,bavg=base.length?base.reduce((a,b)=>a+b,0)/base.length:null;
 const win=res.length?res.filter(x=>x>0).length/res.length*100:null;
 const wr=win==null?'<span class="fa">—</span>':'<div class="wr"><span style="min-width:38px">'+win.toFixed(0)+'%</span><span class="tr"><b class="'+(win<50?'lo':'')+'" style="width:'+win.toFixed(0)+'%"></b></span></div>';
 const d=(avg!=null&&bavg!=null)?avg-bavg:null;
 return '<tr><td>'+label+'</td><td class="r">'+res.length+'</td><td>'+wr+'</td><td class="r">'+pc(avg)+'</td><td class="r">'+pc(med(res))+
  '</td><td class="r hide-sm">'+t+'</td><td class="r hide-sm">'+st+'</td><td class="r hide-sm">'+f+'</td><td class="r">'+(d==null?'—':pc(d)+' <span class="sm">per trade</span>')+'</td></tr>'}
function renderSim(rows){const ti=+simT.value,si=+simS.value,sig=rows.filter(r=>isSig(r));
 $('#sim-note').textContent=ti<0&&si<0?'No target or stop: every trade exits at the flip (same as the table above).':'';
 $('#sim-body').innerHTML=simRow('All signals',sig,ti,si)+simRow('SuperTrend',sig.filter(r=>r[3]===0),ti,si)+
  simRow('ADX DI',sig.filter(r=>r[3]===1),ti,si)+simRow('Double confirmation',rows.filter(r=>r[3]===2),ti,si);
 renderEquity(sig,ti,si)}
function renderEquity(sig,ti,si){const closed=sig.filter(r=>!r[7]).sort((a,b)=>a[0]<b[0]?-1:1);
 const base=[],rule=[];let a=0,b=0;for(const r of closed){const m=mvOf(r);if(m==null)continue;a+=m;base.push(a);
  const[v]=simOne(r,ti,si);b+=v==null?0:v;rule.push(b)}
 const el=$('#eq');if(base.length<2){el.innerHTML='<div class="empty">Not enough closed signals yet.</div>';return}
 const all=base.concat(rule),lo=Math.min(0,...all),hi=Math.max(0,...all),W=1000,H=220,P=8;
 const X=i=>P+i*(W-2*P)/(base.length-1),Y=v=>H-P-(v-lo)*(H-2*P)/((hi-lo)||1);
 const line=(arr,c)=>'<polyline fill="none" stroke="'+c+'" stroke-width="2" vector-effect="non-scaling-stroke" points="'+arr.map((v,i)=>X(i).toFixed(1)+','+Y(v).toFixed(1)).join(' ')+'"/>';
 const showRule=!(ti<0&&si<0);
 el.innerHTML='<div class="lg"><span><i style="background:var(--faint)"></i>Exit at flip: '+pc(a)+'</span>'+
  (showRule?'<span><i style="background:var(--accent)"></i>With simulator rule: '+pc(b)+'</span>':'')+'<span>'+base.length+' trades</span></div>'+
  '<svg viewBox="0 0 '+W+' '+H+'" preserveAspectRatio="none"><line x1="0" x2="'+W+'" y1="'+Y(0)+'" y2="'+Y(0)+'" stroke="var(--line)" stroke-dasharray="4 4" vector-effect="non-scaling-stroke"/>'+
  line(base,'var(--faint)')+(showRule?line(rule,'var(--accent)'):'')+'</svg>'}
function renderHours(sig){const h={};for(const r of sig){const k=r[0].slice(11,13);(h[k]=h[k]||[]).push(r)}
 $('#acc-hour').innerHTML=Object.keys(h).sort().map(k=>{const a=stats(h[k]),hr=+k;
  const lbl=(hr%12||12)+(hr<12?' AM':' PM');
  const wr=a.win==null?'—':'<div class="wr"><span style="min-width:38px">'+a.win.toFixed(0)+'%</span><span class="tr"><b class="'+(a.win<50?'lo':'')+'" style="width:'+a.win.toFixed(0)+'%"></b></span></div>';
  return '<tr><td>'+lbl+'</td><td class="r">'+a.n+'</td><td>'+wr+'</td><td class="r">'+pc(a.avg)+'</td><td class="r">'+pc(a.mfe)+'</td></tr>'}).join('')}
function renderLeaders(sig){const g={};for(const r of sig)(g[r[1]]=g[r[1]]||[]).push(r);
 const rows=Object.entries(g).map(([k,v])=>[NAMES[k],stats(v)]).filter(([,a])=>a.closed>=5).sort((x,y)=>y[1].avg-x[1].avg);
 const row=([n,a])=>'<tr><td>'+tvA(n)+'</td><td class="r">'+a.closed+'</td><td>'+(a.win==null?'—':a.win.toFixed(0)+'%')+'</td><td class="r">'+pc(a.avg)+'</td><td class="r hide-sm">'+pc(a.mfe)+'</td></tr>';
 const top=rows.slice(0,8),bot=rows.slice(-8).filter(x=>!top.includes(x));
 $('#acc-lb').innerHTML=rows.length?'<tr class="grp"><td colspan="5">Best</td></tr>'+top.map(row).join('')+
  (bot.length?'<tr class="grp"><td colspan="5">Worst</td></tr>'+bot.map(row).join(''):''):'<tr><td colspan="5" class="empty">Not enough closed signals yet.</td></tr>'}
let accRows=ARCH;
[simT,simS].forEach(el=>el.onchange=()=>renderSim(accRows));
const accSel=$('#f-acc-sym');
accSel.onchange=()=>{renderAcc(accSel.value);try{localStorage.setItem('accsym',accSel.value)}catch(e){}};
try{const v=localStorage.getItem('accsym');if(v&&[...accSel.options].some(o=>o.value===v))accSel.value=v}catch(e){}
renderAcc(accSel.value);
const stk=filterer('#t-stocks tbody','#c-stocks');seg($('#f-stk-view'),v=>stk.set('view',v));stk.set('view','today');
const secSel=$('#f-stk-sector');secSel.onchange=()=>{stk.set('sector',secSel.value);
 $$('tr.secrow').forEach(r=>r.classList.toggle('on',r.dataset.sec===secSel.value))};
$$('tr.secrow').forEach(r=>r.onclick=()=>{secSel.value=secSel.value===r.dataset.sec?'':r.dataset.sec;secSel.onchange();
 $('#f-stk-q').scrollIntoView({behavior:'smooth',block:'center'})});
paintStars();
/* Focus switch: re-filter every view */
const fb=$('#focus');
function applyFocus(){fb.classList.toggle('on',FOCUS);fb.textContent=FOCUS?'Focus':'All signals';
 for(const f of[live,conf,dxf])f.set('focus',FOCUS?'1':'');
 $('#dx-n100').style.display=FOCUS?'':'none';$('#dx-all').style.display=FOCUS?'none':'';
 renderPast();renderAcc(accSel.value)}
fb.onclick=()=>{FOCUS=!FOCUS;try{localStorage.setItem('focus',FOCUS?'1':'0')}catch(e){}applyFocus()};
applyFocus();
$('#f-stk-q').oninput=e=>stk.set('q',e.target.value.trim().toUpperCase());stk.run();
/* sortable tables */
$$('table[data-sort]').forEach(tbl=>{const tb=$('tbody',tbl);let key=tbl.dataset.sort,asc=true;
 $$('th[data-k]',tbl).forEach(th=>th.onclick=()=>{const k=th.dataset.k;asc=k===key?!asc:th.dataset.asc==='1';key=k;
  $$('th',tbl).forEach(x=>x.classList.remove('sorted','desc'));th.classList.add('sorted');if(!asc)th.classList.add('desc');
  const rows=$$('tr[data-sym]',tb);rows.sort((a,b)=>{const x=a.dataset[k],y=b.dataset[k];
   const v=isNaN(+x)||isNaN(+y)?x.localeCompare(y):(+x)-(+y);return asc?v:-v});rows.forEach(r=>tb.appendChild(r))})});
"""


# ── formatting helpers ─────────────────────────────────────────────────────

def _fmt_price(p):
    return f"{p:,.2f}" if isinstance(p, (int, float)) else "—"


def _ist(ts):
    """'2026-10-09 15:15:00' or a bar timestamp with +05:30 -> naive IST datetime."""
    return datetime.fromisoformat(str(ts)[:19])


def fmt_time(ts):
    """12-hour IST clock time, e.g. '3:15 PM'."""
    try:
        return _ist(ts).strftime("%I:%M %p").lstrip("0")
    except ValueError:
        return str(ts)


def fmt_day(ts):
    """e.g. 'Fri, 9 Oct'."""
    try:
        d = _ist(ts)
        return f"{d:%a}, {d.day} {d:%b}"
    except ValueError:
        return str(ts)


def fmt_stamp(ts, year=False):
    """e.g. '9 Oct, 3:15 PM' (or '9 Oct 2026, 3:15 PM')."""
    try:
        d = _ist(ts)
        return f"{d.day} {d:%b}{f' {d.year}' if year else ''}, {fmt_time(ts)}"
    except ValueError:
        return str(ts)


def _direction(label):
    return "sell" if "SELL" in label else "buy"


TV_INDEX = {"NIFTY 50": "NIFTY", "BANK NIFTY": "BANKNIFTY", "NIFTY NEXT 50": "NIFTYJR",
            "NIFTY MIDCAP 150": "NIFTYMIDCAP150", "NIFTY SMALLCAP 250": "NIFTYSMLCAP250",
            "NIFTY 500": "CNX500"}


def tv_url(name):
    """TradingView 15-min chart for a stock or index ("-" becomes "_", "&" stays)."""
    sym = TV_INDEX.get(name, name.replace("-", "_"))
    return "https://www.tradingview.com/chart/?interval=15&symbol=" + urllib.parse.quote("NSE:" + sym)


def _tv(name, cls="sym"):
    e = html_lib.escape
    return (f'<a class="tv {cls}" href="{e(tv_url(name))}" target="_blank" rel="noopener" '
            f'title="Open {e(name)} on TradingView (15 min)">{e(name)}</a>')


def _pct(v, signed=True):
    if v is None:
        return '<span class="fa">—</span>'
    if abs(v) < 0.005:
        return '<span class="mu">0.00%</span>'
    cls = "up" if v > 0 else "dn"
    return f'<span class="{cls}">{v:+.2f}%</span>' if signed else f'<span class="{cls}">{v:.2f}%</span>'


def _tag(label, kind=None):
    kind = kind or ("dn" if "SELL" in label else "up")
    return f'<span class="tag {kind}">{html_lib.escape(label.replace("_", " "))}</span>'


def _st_tag(st):
    return {1: _tag("LONG", "up"), -1: _tag("SHORT", "dn")}.get(st, '<span class="fa">—</span>')


def _adx_tag(adx):
    label = ADX_LABELS.get(adx, "—")
    return _tag(label) if adx else '<span class="fa">—</span>'


def _gc_tag(info, short=False):
    """Golden / death cross badge with sessions since the cross."""
    gc = info.get("gc")
    if gc not in (1, -1):
        return '<span class="fa">—</span>'
    days = info.get("gc_days")
    fresh = days is not None and days <= FRESH_CROSS_DAYS
    label = ("GOLDEN" if gc == 1 else "DEATH") + ("" if short else " CROSS")
    since = info.get("gc_date", "")
    title = (f'Daily EMA50 {"above" if gc == 1 else "below"} EMA200 since {fmt_day(since)} ({days} sessions)'
             if since else "")
    return (f'<span class="tag {"up" if gc == 1 else "dn"}" title="{html_lib.escape(title)}">{label}</span>'
            f'<span class="sm"> {days}d{" · new" if fresh else ""}</span>')


def _tf_tag(v):
    return {1: _tag("UP", "up"), -1: _tag("DOWN", "dn")}.get(v, '<span class="fa">—</span>')


def _strat_name(strategy):
    return STRATEGY_NAMES.get(strategy, strategy)


def _bars_txt(bars, still_open):
    if bars is None:
        return '<span class="fa">—</span>'
    if still_open:
        return '<span class="tag ok">new</span>' if bars == 0 else f'<span class="mu">{bars}+</span>'
    return str(bars)


# ── analytics ──────────────────────────────────────────────────────────────

def _current_price(state, name):
    info = state.get(name) or state.get(INDEX_STATE_PREFIX + name) or {}
    return info.get("last_close")


def outcome(sig, now_px):
    """(move in the signal's favour %, closed?) — at the bar where its strategy
    flipped if it has, else from signal price to `now_px`."""
    end = sig.get("exit_price") if not sig.get("open", True) else now_px
    if not end or not sig.get("price"):
        return None, not sig.get("open", True)
    move = (end / sig["price"] - 1) * 100.0
    return (move if _direction(sig["label"]) == "buy" else -move), not sig.get("open", True)


def find_confluence(signals, days=None):
    """Stocks where SuperTrend and ADX_DI both fired the same direction on the
    same trading day (the latest signal of each strategy that day counts).
    Returns {day: [entry, ...]} newest day first; an entry's "time"/"price"
    are those of the second (confirming) signal."""
    latest = {}
    for s in sorted(signals, key=lambda x: x["time"]):
        latest.setdefault((s["time"][:10], s["name"]), {})[s["strategy"]] = s
    by_day = {}
    for (day, name), strat in latest.items():
        st, adx = strat.get("SuperTrend"), strat.get("ADX_DI")
        if st and adx and _direction(st["label"]) == _direction(adx["label"]):
            last = max(st, adx, key=lambda x: x["time"])
            by_day.setdefault(day, []).append({"name": name, "dir": _direction(st["label"]),
                                               "time": last["time"], "price": last["price"],
                                               "st": st, "adx": adx})
    for entries in by_day.values():
        entries.sort(key=lambda x: x["time"], reverse=True)
    days_sorted = sorted(by_day, reverse=True)
    return {d: by_day[d] for d in (days_sorted[:days] if days else days_sorted)}


def confirmed_trades(archive):
    """Double confirmations as trades (enter on the 2nd signal, exit on the 1st
    flip) — built bar-exactly in bar_signals() as strategy "CONFIRMED"."""
    return [h for h in archive if h["strategy"] == "CONFIRMED"]


def strategy_signals(archive):
    """Archive minus the derived CONFIRMED trades: the raw strategy signals."""
    return [h for h in archive if h["strategy"] != "CONFIRMED"]


def _match_replay(archive, name, strategies, direction, day, upto=None):
    """Latest replayed signal for a stock on a day (to attach max gain/drawdown
    to a live alert or a live double confirmation)."""
    hits = [h for h in archive if h["name"] == name and h["strategy"] in strategies
            and _direction(h["label"]) == direction and h["time"][:10] == day
            and (upto is None or h["time"] <= upto)]
    return max(hits, key=lambda h: h["time"]) if hits else None


def accuracy_stats(signals, state):
    n = len(signals)
    closed, live = [], []
    bars = []
    mfe, mae = [], []
    for s in signals:
        mv, is_closed = outcome(s, _current_price(state, s["name"]))
        if mv is None:
            continue
        (closed if is_closed else live).append(mv)
        if is_closed and s.get("bars") is not None:
            bars.append(s["bars"])
        if is_closed and s.get("mfe") is not None:
            mfe.append(s["mfe"])
            mae.append(s["mae"])
    med = lambda xs: float(np.median(xs)) if xs else None  # noqa: E731
    return {"n": n, "closed": len(closed), "open": len(live), "mfe": med(mfe), "mae": med(mae),
            "win": (sum(m > 0 for m in closed) / len(closed) * 100) if closed else None,
            "bars": med(bars), "move": med(closed), "avg": float(np.mean(closed)) if closed else None,
            "open_move": med(live)}


# ── page ───────────────────────────────────────────────────────────────────

def _momentum_view_html(state, sectors):
    """Momentum tab: paper-tracked monthly top-15 portfolio, benchmarks, next
    month's list, ranking and month-by-month history."""
    e = html_lib.escape
    v = (state.get(MOM_KEY) or {}).get("view")
    if not v:
        return '<div class="panel empty">The momentum portfolio forms on the first scan with 13+ months of daily data.</div>'
    mon = lambda m: pd.Period(m, "M").strftime("%b %Y")  # noqa: E731
    pct = lambda x: _pct(x * 100) if x is not None else '<span class="fa">—</span>'  # noqa: E731
    bm = v.get("bench_mtd", {})
    cum = v.get("cum", {})
    since = v.get("since") or v["month"]
    kpis = (
        f'<div class="panel kpi"><div class="k">This month ({e(mon(v["month"]))})</div><div class="v">{pct(v["mtd"])}</div>'
        f'<div class="s">NIFTY 500 {pct(bm.get("NIFTY 500"))} · momentum ETF {pct(bm.get("Momentum ETF"))}</div></div>'
        f'<div class="panel kpi"><div class="k">Paper return since {e(mon(since))}</div><div class="v">{pct(cum.get("portfolio"))}</div>'
        f'<div class="s">NIFTY 500 {pct(cum.get("NIFTY 500"))} · ETF {pct(cum.get("Momentum ETF"))}</div></div>'
        f'<div class="panel kpi"><div class="k">Equal weight, all stocks</div><div class="v">{pct(v.get("ew_mtd"))}</div>'
        f'<div class="s">this month · since {e(mon(since))} {pct(cum.get("ew"))}</div></div>'
        f'<div class="panel kpi"><div class="k">Next rebalance ({e(mon(v["next_month"]))})</div>'
        f'<div class="v">{len(v["next_in"])} in · {len(v["next_out"])} out</div>'
        f'<div class="s">at the first session’s open · already determined by data</div></div>')
    leaving = set(v["next_out"])
    hold = "".join(
        f'<tr data-sym="{e(r["sym"])}" data-mtd="{r["mtd"] if r["mtd"] is not None else -9}" data-score="{r["score"]}">'
        f'<td class="r mu">{r["rank"]}</td><td>{_tv(r["sym"])}</td><td class="mu hide-md">{e(sectors.get(r["sym"], "—"))}</td>'
        f'<td class="r">{_pct(r["score"] * 100)}</td><td class="r">{_fmt_price(r["entry"])}</td>'
        f'<td class="r">{_fmt_price(r["now"])}</td><td class="r">{pct(r["mtd"])}</td>'
        f'<td>{_gc_tag(state.get(r["sym"], {}), short=True)}</td>'
        f'<td>{_tag("LEAVES", "dn") if r["sym"] in leaving else _tag("STAYS", "mute")}</td></tr>'
        for r in v["rows"])
    nxt_in = set(v["next_in"])
    ranking = "".join(
        f'<tr data-sym="{e(sym)}"><td class="r mu">{k}</td><td>{_tv(sym)}{" " + _tag("NEW", "ok") if sym in nxt_in else ""}</td>'
        f'<td class="mu hide-md">{e(sectors.get(sym, "—"))}</td><td class="r">{_pct(sc * 100)}</td>'
        f'<td>{_gc_tag(state.get(sym, {}), short=True)}</td></tr>'
        for k, (sym, sc) in enumerate(v["ranking"][:30], 1))
    hist = "".join(
        f'<tr><td>{e(mon(h["month"]))}</td><td class="r">{pct(h["net"])}</td>'
        f'<td class="r">{pct(h["bench"].get("NIFTY 500", h["bench"].get("NIFTY 200")))}</td><td class="r">{pct(h["bench"].get("Momentum ETF"))}</td>'
        f'<td class="r">{pct(h.get("ew"))}</td><td class="r">{h["turnover"] * 100:.0f}%</td></tr>' for h in reversed(v["history"]))
    chips = lambda xs, kind: " ".join(f'<span class="tag {kind}">{e(x)}</span>' for x in xs) or '<span class="fa">none</span>'  # noqa: E731

    def tbl(head, rows, empty):
        return (f'<div class="panel tw"><table><thead><tr>{head}</tr></thead><tbody>'
                f'{rows or f"<tr><td colspan=9 class=empty>{empty}</td></tr>"}</tbody></table></div>')

    return f"""<div class="sh"><h2>Momentum portfolio — paper tracking</h2><span class="meta">top {MOM_TOP} by 12-1 momentum · equal weight · rebalanced monthly</span></div>
<div class="kpis">{kpis}</div>
<section><div class="sh"><h2>Holdings — {e(mon(v["month"]))}</h2><span class="meta">bought at the open on {e(fmt_day(v["start"]))} · turnover {v["turnover"] * 100:.0f}% · costs included</span></div>
{tbl('<th class="r">#</th><th>Stock</th><th class="hide-md">Sector</th><th class="r">12-1 momentum</th><th class="r">Entry</th><th class="r">Now</th><th class="r">Month to date</th><th>Daily trend</th><th>Next month</th>', hold, "")}</section>
<div class="grid2"><section><div class="sh"><h2>Next month — {e(mon(v["next_month"]))}</h2><span class="meta">ranking ends a month back, so this list is already fixed</span></div>
<div class="panel" style="padding:14px 16px;font-size:13px;line-height:2"><b>Coming in:</b> {chips(v["next_in"], "up")}<br>
<b>Going out:</b> {chips(v["next_out"], "dn")}</div>
<div style="margin-top:12px">{tbl('<th class="r">#</th><th>Stock</th><th class="hide-md">Sector</th><th class="r">12-1 momentum</th><th>Daily trend</th>', ranking, "")}</div></section>
<section><div class="sh"><h2>Month by month</h2><span class="meta">paper results after costs</span></div>
{tbl('<th>Month</th><th class="r">Portfolio</th><th class="r">NIFTY 500</th><th class="r">Momentum ETF</th><th class="r">Equal weight</th><th class="r">Turnover</th>', hist, "First month in progress — results appear here as each month closes.")}</section></div>
<details class="note below"><summary>Rules and caveats</summary><b>Rules</b> (same as momentum_backtest.py): at each month-end, rank the stocks by their return over the past
12 months skipping the latest month; hold the top {MOM_TOP} in equal weight from the next month's first open; charge delivery costs
(~{MOM_COST:.2f}% per round trip) on every stock swapped. Universe: the Nifty 500 (October 2026 was formed from the earlier
150-stock list). Benchmarks: NIFTY 500, a listed Nifty500 Momentum 50 ETF (NSE's real momentum index, after fund costs) and an
equal-weight portfolio of all the stocks. <b>Read with care:</b> the historical backtests (momentum_report.md,
momentum_report_n500.md) are inflated by survivorship bias — over Aug 2022 – Sep 2026 a real Nifty200 Momentum 30 ETF beat the
Nifty 200 by about 2.7 points a year, against ~19–23 in the backtests. This tab is a forward paper record from {e(mon(since))}, so it carries no such bias.
Paper tracking only — not a recommendation; speak to a SEBI-registered adviser before investing.</details>"""


def write_dashboard(state):
    """Render dashboard.html from the state file — a static page with four tabs:
    Live (indices, today's confirmations and signals), Past signals (day-wise
    replay of the last HISTORY_DAYS sessions), Accuracy (win rate, median bars
    and median move per signal type over the archive) and Stocks."""
    e = html_lib.escape
    meta = state.get(SCAN_META_KEY, {})
    scan_time = meta.get("time")
    universe = meta.get("universe") or NIFTY150
    stocks = {s: state[s] for s in universe if s in state}
    log = state.get(SIGNAL_LOG_KEY, [])
    archive = [h for h in state.get(ARCHIVE_KEY, []) if not h["strategy"].startswith("EMA_")]
    n100 = meta.get("n100") or (sorted(pd.read_csv(NIFTY100_FALLBACK)["Symbol"].str.strip())
                                if NIFTY100_FALLBACK.exists() else [])

    def hhmm(sig):
        """Candle time (HH:MM) of a live signal, for the Focus time window."""
        return str(sig.get("bar") or sig["time"])[11:16]

    # ── index strip
    ix_html = []
    for name in INDEX_TICKERS:
        info = state.get(INDEX_STATE_PREFIX + name)
        if not info:
            continue
        dist = info.get("distance_pct")
        dist_txt = e(f"{dist:.2f}% to flip") if dist is not None else ""
        ix_html.append(
            f'<div class="panel ix"><div class="nm" title="{e(name)}">{_tv(name, "")}</div>'
            f'<div class="px">{_fmt_price(info.get("last_close"))}</div>'
            f'<div class="ft">{_pct(info.get("day_chg_pct"))}'
            f'<span class="sm">{dist_txt}</span></div>'
            f'<div class="tags">{_st_tag(info.get("st_trend"))}{_adx_tag(info.get("adx_cond"))}</div>'
            f'<div class="tags">{_gc_tag(info, short=True)}</div></div>')

    # ── live: today's signals + confirmations
    latest_day = log[-1]["time"][:10] if log else None
    today = [s for s in log if s["time"][:10] == latest_day][::-1]
    n_buy = sum(_direction(s["label"]) == "buy" for s in today)
    conf_today = find_confluence(log).get(latest_day, []) if latest_day else []
    st_long = sum(v.get("st_trend") == 1 for v in stocks.values())
    st_short = sum(v.get("st_trend") == -1 for v in stocks.values())
    adx_buy = sum((v.get("adx_cond") or 0) > 0 for v in stocks.values())
    adx_sell = sum((v.get("adx_cond") or 0) < 0 for v in stocks.values())

    def split(u, d):
        t = (u + d) or 1
        return f'<div class="split"><b style="width:{u / t * 100:.1f}%"></b><i style="width:{d / t * 100:.1f}%"></i></div>'

    def now_move(s):
        mv, _ = outcome({**s, "open": True}, _current_price(state, s["name"]))
        return mv

    n_golden = sum(v.get("gc") == 1 for v in stocks.values())
    n_death = sum(v.get("gc") == -1 for v in stocks.values())
    n_fresh = sum(1 for v in stocks.values() if v.get("gc") in (1, -1)
                  and v.get("gc_days") is not None and v["gc_days"] <= FRESH_CROSS_DAYS)  # 0 = crossed today
    kpis = (
        f'<div class="panel kpi"><div class="k">Signals {"today" if latest_day else ""}</div>'
        f'<div class="v">{len(today)}</div><div class="s"><span class="up">{n_buy} buy</span> · '
        f'<span class="dn">{len(today) - n_buy} sell</span></div>'
        f'{split(n_buy, len(today) - n_buy)}</div>'
        f'<div class="panel kpi"><div class="k">Double confirmations</div><div class="v">{len(conf_today)}</div>'
        f'<div class="s"><span class="up">{sum(c["dir"] == "buy" for c in conf_today)} buy</span> · '
        f'<span class="dn">{sum(c["dir"] == "sell" for c in conf_today)} sell</span></div></div>'
        f'<div class="panel kpi"><div class="k">SuperTrend</div><div class="v">{st_long}'
        f'<span class="mu" style="font-size:14px"> / {st_short}</span></div><div class="s">long / short</div>'
        f'{split(st_long, st_short)}</div>'
        f'<div class="panel kpi"><div class="k">Golden cross (daily)</div><div class="v">{n_golden}'
        f'<span class="mu" style="font-size:14px"> / {n_death}</span></div><div class="s">golden / death · '
        f'{n_fresh} crossed in the last {FRESH_CROSS_DAYS} sessions</div>{split(n_golden, n_death)}</div>'
        f'<div class="panel kpi"><div class="k">ADX DI</div><div class="v">{adx_buy}'
        f'<span class="mu" style="font-size:14px"> / {adx_sell}</span></div><div class="s">buy / sell state</div>'
        f'{split(adx_buy, adx_sell)}</div>')


    def excursion_cells(h):
        if not h:
            return '<td class="r"><span class="fa">—</span></td><td class="r hide-sm"><span class="fa">—</span></td>'
        return f'<td class="r">{_pct(h.get("mfe"))}</td><td class="r hide-sm">{_pct(h.get("mae"))}</td>'

    conf_rows = "".join(
        f'<tr data-sym="{e(c["name"])}" data-t="{hhmm(c["st"] if c["st"]["time"] > c["adx"]["time"] else c["adx"])}"><td>{_tv(c["name"])}</td>'
        f'<td>{_tag("CONFIRMED " + c["dir"].upper())}</td>'
        f'<td>{_tag(c["st"]["label"])} <span class="mu">{e(fmt_time(c["st"]["time"]))}</span></td>'
        f'<td>{_tag(c["adx"]["label"])} <span class="mu">{e(fmt_time(c["adx"]["time"]))}</span></td>'
        f'<td class="r">{_fmt_price(c["price"])}</td><td class="r">{_fmt_price(_current_price(state, c["name"]))}</td>'
        f'<td class="r">{_pct(now_move({"label": c["dir"].upper(), "price": c["price"], "name": c["name"]}))}</td>'
        f'{excursion_cells(_match_replay(archive, c["name"], ("CONFIRMED",), c["dir"], latest_day))}</tr>'
        for c in conf_today)
    live_rows = "".join(
        f'<tr data-sym="{e(s["name"])}" data-dir="{_direction(s["label"])}" '
        f'data-t="{"12:00" if s["strategy"].startswith("EMA_") else hhmm(s)}" '
        f'{" class=fresh" if s["time"] == scan_time else ""}>'
        f'<td class="mu">{e(fmt_time(s["time"]))}</td><td>{_tv(s["name"])}</td><td>{_tag(s["label"])}</td>'
        f'<td class="mu hide-sm">{_strat_name(s["strategy"])}</td><td class="r">{_fmt_price(s["price"])}</td>'
        f'<td class="r">{_pct(now_move(s))}</td>'
        f'{excursion_cells(_match_replay(archive, s["name"], (s["strategy"],), _direction(s["label"]), s["time"][:10], s["time"]))}</tr>'
        for s in today)

    # ── daily EMA crosses (last DAILY_CROSS_DAYS sessions), newest first
    crosses = []
    for key, info in state.items():
        if not isinstance(info, dict) or not info.get("dx"):
            continue
        nm = info.get("name") or key
        for x in info["dx"]:
            crosses.append((x, nm, _current_price(state, nm)))
    crosses.sort(key=lambda t: (t[0]["date"], t[1]), reverse=True)

    def cross_row(x, nm, now_px):
        mv = (now_px / x["px"] - 1) * 100 * x["d"] if now_px and x["px"] else None
        kind = "EMA 50×200" if x["k"] == "EMA_GOLDEN" else "EMA 50×100"
        what = (("GOLDEN" if x["d"] == 1 else "DEATH") if x["k"] == "EMA_GOLDEN" else ("BUY" if x["d"] == 1 else "SELL"))
        forming = ' <span class="tag mute" title="Today\'s candle is still open — final at 3:30 PM">forming</span>' if x["f"] else ""
        return (f'<tr data-sym="{e(nm)}" data-t="12:00" data-dir="{"buy" if x["d"] == 1 else "sell"}"><td class="mu">{e(fmt_day(x["date"]))}</td>'
                f'<td>{_tv(nm)}</td><td>{_tag(what, "up" if x["d"] == 1 else "dn")}{forming}</td><td class="mu hide-sm">{kind}</td>'
                f'<td class="r">{_fmt_price(x["px"])}</td><td class="r">{_fmt_price(now_px)}</td><td class="r">{_pct(mv)}</td></tr>')
    cross_rows = "".join(cross_row(*t) for t in crosses)

    # 5-year record of the daily crosses, for the Accuracy tab (all stocks / Nifty 100)
    def cross_record(names):
        groups = {}
        for nm in names:
            info = state.get(nm) or state.get(INDEX_STATE_PREFIX + nm) or {}
            for kind, d, mv, held, still_open in info.get("dxs", []):
                groups.setdefault((kind, d), []).append((mv, held, still_open))
        out = []
        for kind, title in ((0, "EMA 50×100 — Early Cross"), (1, "EMA 50×200 — Golden / Death Cross")):
            out.append(f'<tr class="grp"><td colspan="7">{title} (hold until the opposite cross)</td></tr>')
            for d in (1, -1):
                g = groups.get((kind, d), [])
                closed = [m for m, _, o in g if not o]
                lbl = ("GOLDEN" if d == 1 else "DEATH") if kind else ("BUY" if d == 1 else "SELL")
                if not closed:
                    out.append(f'<tr><td>{_tag(lbl, "up" if d == 1 else "dn")}</td><td class="r" colspan="6"><span class="fa">—</span></td></tr>')
                    continue
                win = sum(m > 0 for m in closed) / len(closed) * 100
                srt = sorted(closed)
                medv = srt[len(srt) // 2] if len(srt) % 2 else (srt[len(srt) // 2 - 1] + srt[len(srt) // 2]) / 2
                held = sorted(h for _, h, o in g if not o)
                out.append(f'<tr><td>{_tag(lbl, "up" if d == 1 else "dn")}</td><td class="r">{len(g)}</td>'
                           f'<td class="r">{len(g) - len(closed)}</td>'
                           f'<td><div class="wr"><span style="min-width:38px">{win:.0f}%</span><span class="tr"><b class="{"lo" if win < 50 else ""}" style="width:{win:.0f}%"></b></span></div></td>'
                           f'<td class="r">{_pct(sum(closed) / len(closed))}</td><td class="r">{_pct(medv)}</td>'
                           f'<td class="r">{held[len(held) // 2]}</td></tr>')
        return "".join(out)
    rec_all = cross_record(list(stocks) + list(INDEX_TICKERS))
    rec_n100 = cross_record([n for n in n100 if n in stocks] + list(INDEX_TICKERS))

    # ── past signals (replayed, last HISTORY_DAYS sessions)
    days_all = sorted({h["time"][:10] for h in archive}, reverse=True)
    past_days = days_all[:HISTORY_DAYS]
    past = sorted((h for h in strategy_signals(archive) if h["time"][:10] in past_days),
                  key=lambda h: h["time"], reverse=True)
    conf_keys = {(h["time"][:10], h["name"]) for h in confirmed_trades(archive)}
    # The whole archive goes to the page once as compact JSON; Past signals and
    # Accuracy (overall or per stock/index) are drawn from it in the browser.
    # Row: [time, name#, label, strategy (0 ST / 1 ADX / 2 CONFIRMED), price,
    #       exit price, bars, open, max gain, max drawdown]
    strat_code = {"SuperTrend": 0, "ADX_DI": 1, "CONFIRMED": 2}
    names = sorted({h["name"] for h in archive})
    name_ix = {n: i for i, n in enumerate(names)}
    arch_rows = [[h["time"], name_ix[h["name"]], h["label"], strat_code[h["strategy"]], round(h["price"], 2),
                  round(h["exit_price"], 2) if h.get("exit_price") else None, h.get("bars"),
                  1 if h.get("open", True) else 0, h.get("mfe"), h.get("mae"),
                  h.get("hf") or [], h.get("ha") or [], int(bool(h.get("gh")))]
                 for h in sorted(archive, key=lambda x: x["time"])]
    prices = {name_ix[n]: _current_price(state, n) for n in names}
    # The archive and the 500-row stock table are the bulk of the page; both go in
    # gzip-compressed (base64) and are unpacked in the browser (DecompressionStream).
    def gz(text):
        return base64.b64encode(gzip.compress(text.encode("utf-8"), 9)).decode("ascii")

    data_js = ("const NAMES=" + json.dumps(names) + ";const ARCH_GZ=\"" + gz(json.dumps(arch_rows, separators=(",", ":"))) + "\""
               + ";const PX=" + json.dumps(prices, separators=(",", ":"))
               + ";const PDAYS=new Set(" + json.dumps(past_days) + ");const N100=new Set(" + json.dumps(n100 + list(INDEX_TICKERS))
               + ");const FOCUS_WIN=" + json.dumps([FOCUS_FROM, FOCUS_TO]) + ";const SIM_LEVELS="
               + json.dumps(list(SIM_LEVELS)) + ";const TVI=" + json.dumps(TV_INDEX)
               + ";").replace("</", "<\\/")
    per_day = {d: sum(1 for h in past if h["time"][:10] == d) for d in past_days}
    day_opts = "".join(f'<option value="{d}">{e(fmt_day(d))} · {per_day[d]} signals</option>' for d in past_days)
    if past_days:
        day_opts += f'<option value="">All {len(past_days)} days · {len(past)} signals</option>'
    past_counts = {}
    for h in past:
        past_counts[h["name"]] = past_counts.get(h["name"], 0) + 1

    def sym_options(cnt, all_label):
        ix = "".join(f'<option value="{e(n)}">{e(n)} · {cnt[n]}</option>' for n in INDEX_TICKERS if n in cnt)
        st = "".join(f'<option value="{e(n)}">{e(n)} · {cnt[n]}</option>' for n in sorted(cnt) if n not in INDEX_TICKERS)
        return (f'<option value="">{all_label} · {sum(cnt.values())}</option>'
                + (f'<optgroup label="Indices">{ix}</optgroup>' if ix else "")
                + (f'<optgroup label="Stocks">{st}</optgroup>' if st else ""))
    past_sym_opts = sym_options(past_counts, "All stocks &amp; indices")

    # ── accuracy: symbol picker (stats themselves are computed in the browser)
    span = (f"{fmt_day(days_all[-1])} – {fmt_day(days_all[0])} · {len(days_all)} sessions"
            if days_all else "no data yet")
    counts = {}
    for h in strategy_signals(archive):
        counts[h["name"]] = counts.get(h["name"], 0) + 1
    idx_opts = "".join(f'<option value="{e(n)}">{e(n)} · {counts[n]}</option>' for n in INDEX_TICKERS if n in counts)
    stk_opts = "".join(f'<option value="{e(n)}">{e(n)} · {counts[n]}</option>'
                       for n in sorted(counts) if n not in INDEX_TICKERS)
    sym_opts = (f'<option value="">All stocks &amp; indices · {sum(counts.values())}</option>'
                + (f'<optgroup label="Indices">{idx_opts}</optgroup>' if idx_opts else "")
                + (f'<optgroup label="Stocks">{stk_opts}</optgroup>' if stk_opts else ""))

    # ── stocks
    days_all_latest = max((h["time"][:10] for h in archive), default=None)
    last_sig = {}
    for h in sorted(strategy_signals(archive), key=lambda x: x["time"]):
        last_sig[h["name"]] = h
    for s in log:
        if s["name"] not in last_sig or s["time"] > last_sig[s["name"]]["time"]:
            last_sig[s["name"]] = s
    fresh = {s["name"] for s in log if s["time"] == scan_time}
    # "Setups": a double confirmation on the latest day, or a daily EMA cross in the last DAILY_CROSS_DAYS sessions
    sig_today = ({c["name"] for c in conf_today}
                 | {h["name"] for h in archive if h["strategy"] == "CONFIRMED" and h["time"][:10] == days_all_latest}
                 | {sym for sym, v in stocks.items() if v.get("dx")})
    sectors = meta.get("sectors", {})
    newest_bar = max((v.get("last_ts", "") for v in stocks.values()), default="")
    stale = {sym for sym, v in stocks.items() if v.get("last_ts", "") < newest_bar}
    stock_rows = []
    for sym in sorted(stocks, key=lambda x: (stocks[x].get("distance_pct") is None, stocks[x].get("distance_pct") or 0)):
        info = stocks[sym]
        st, adx, dist, chg = (info.get("st_trend"), info.get("adx_cond") or 0.0,
                              info.get("distance_pct"), info.get("day_chg_pct"))
        ls = last_sig.get(sym)
        dist_s = f"{dist:.2f}%" if dist is not None else "—"
        mv = outcome({**ls, "open": True}, info.get("last_close"))[0] if ls else None
        max_txt = (f' · max {ls["mfe"]:+.2f}% / {ls["mae"]:+.2f}%' if ls and ls.get("mfe") is not None else "")
        sector = sectors.get(sym, "—")
        stale_tag = (f' <span class="tag mute" title="Latest candle is {e(fmt_stamp(info.get("last_ts", "")))} — '
                     f'Yahoo hasn\'t published newer data for this stock">stale</span>' if sym in stale else "")
        stock_rows.append(
            f'<tr data-sym="{e(sym)}" data-sector="{e(sector)}" data-stale="{int(sym in stale)}" data-today="{int(sym in sig_today)}" '
            f'data-st="{st}" data-adx="{adx}" data-dist="{dist if dist is not None else 999}" '
            f'data-chg="{chg if chg is not None else 0}" data-px="{info.get("last_close") or 0}" '
            f'data-mv="{mv if mv is not None else -999}" '
            f'data-gc="{info.get("gc", 0)}" data-gcd="{info.get("gc_days", 999) if info.get("gc") else 999}" '
            f'data-gck="{info.get("gc", 0) * 10000 - (info.get("gc_days") or 0)}"'
            f'{" class=fresh" if sym in fresh else ""}>'
            f'<td><button class="star" data-fav="{e(sym)}" aria-label="Favourite">☆</button>{_tv(sym)}{stale_tag}</td>'
            f'<td class="mu hide-md">{e(sector)}</td><td class="r">{_fmt_price(info.get("last_close"))}</td>'
            f'<td class="r">{_pct(chg)}</td><td>{_st_tag(st)}</td>'
            f'<td>{_adx_tag(adx)}</td><td>{_gc_tag(info)}</td>'
            f'<td class="r">{dist_s}</td>'
            f'<td>{(_tag(ls["label"]) + " " + _pct(mv) + "<div class=sm>" + e(fmt_stamp(ls["time"])) + max_txt + "</div>") if ls else "<span class=fa>—</span>"}</td></tr>')

    by_sector = {}
    for sym, v in stocks.items():
        by_sector.setdefault(sectors.get(sym, "Other"), []).append(v)
    sector_rows = []
    for sec, vs in sorted(by_sector.items(), key=lambda kv: -sum(v.get("st_trend") == 1 for v in kv[1]) / len(kv[1])):
        n = len(vs)
        lng = sum(v.get("st_trend") == 1 for v in vs)
        ab = sum((v.get("adx_cond") or 0) > 0 for v in vs)
        asl = sum((v.get("adx_cond") or 0) < 0 for v in vs)
        chgs = [v["day_chg_pct"] for v in vs if v.get("day_chg_pct") is not None]
        gold = sum(v.get("gc") == 1 for v in vs)
        sector_rows.append(
            f'<tr data-sym="{e(sec)}" data-sec="{e(sec)}" class="secrow"><td class="sym">{e(sec)}</td><td class="r">{n}</td>'
            f'<td><div class="wr"><span style="min-width:38px">{lng / n * 100:.0f}%</span><span class="tr">'
            f'<b class="{"lo" if lng / n < 0.5 else ""}" style="width:{lng / n * 100:.0f}%"></b></span></div></td>'
            f'<td class="r"><span class="up">{ab}</span> / <span class="dn">{asl}</span></td>'
            f'<td class="r">{gold}/{n} <span class="sm">golden</span></td>'
            f'<td class="r">{_pct(sum(chgs) / len(chgs) if chgs else None)}</td></tr>')
    sector_opts = '<option value="">All sectors</option>' + "".join(
        f'<option value="{e(sec)}">{e(sec)} · {len(vs)}</option>' for sec, vs in sorted(by_sector.items()))

    def table(tid, head, rows, empty, sort=None):
        return (f'<div class="panel tw"><table id="{tid}"{f" data-sort={sort}" if sort else ""}><thead><tr>{head}</tr>'
                f'</thead><tbody>{rows}<tr class="nores" style="display:{"none" if rows else ""}">'
                f'<td colspan="12" class="empty">{empty}</td></tr></tbody></table></div>')

    scanned_iso = f"{scan_time.replace(' ', 'T')}+05:30" if scan_time else ""
    stale_txt = f' · {len(stale)} stale' if stale else ""
    sub = (f'{meta.get("checked")}/{meta.get("total")} stocks{stale_txt} · {len(INDEX_TICKERS)} indices · '
           f'{e(fmt_stamp(scan_time, year=True))} IST' if scan_time else "Waiting for the first full scan")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta http-equiv="refresh" content="300"><meta name="color-scheme" content="light dark">
<script>try{{const t=localStorage.getItem('theme');if(t==='light'||t==='dark')
document.documentElement.dataset.theme=t;}}catch(e){{}}</script>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<title>Technical Dashboard</title><style>{DASHBOARD_CSS}</style></head>
<body data-scanned="{e(scanned_iso)}">
<div class="top"><div class="wrap">
<div class="bar1"><div class="brand"><div class="logo"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l5-5 4 4 8-9"/><path d="M15 7h5v5"/></svg></div><div style="min-width:0"><h1>Technical Dashboard</h1>
<div class="sub">SuperTrend ({ATR_PERIOD}×{MULTIPLIER:g}) + ADX DI · {sub}</div></div></div>
<div class="tools-r"><button class="fbtn on" id="focus" title="Focus: only Nifty 100 stocks + indices, and only signals from 9:45 AM to 2:30 PM. Click to show everything.">Focus</button>
<span class="status" id="status"><i></i><span class="lbl">Waiting</span></span>
<button class="iconbtn theme" id="theme" title="Light / dark" aria-label="Switch light or dark theme">
<svg class="moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>
<svg class="sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
</button></div></div>
<nav class="tabs"><button class="tab" data-v="live">Live</button>
<button class="tab" data-v="past">Past signals</button>
<button class="tab" data-v="accuracy">Accuracy</button>
<button class="tab" data-v="momentum">Momentum</button>
<button class="tab" data-v="stocks">Stocks<span class="n">{len(stocks)}</span></button></nav>
</div></div>

<main class="wrap">
<div class="view" id="v-live">
<div class="idx">{"".join(ix_html) or '<div class="panel empty" style="grid-column:1/-1">No index data yet.</div>'}</div>
<div class="kpis k5">{kpis}</div>
<section><div class="sh"><h2>Double confirmations</h2><span class="meta" id="c-conf">{e(fmt_day(latest_day)) if latest_day else ""}</span></div>
<p class="note">SuperTrend and ADX DI gave the same signal on the same day — the latest signal of each counts.</p>
{table("t-conf", '<th>Stock</th><th>Setup</th><th>SuperTrend</th><th>ADX DI</th><th class="r">At signal</th><th class="r">Now</th><th class="r">Move</th><th class="r">Max gain</th><th class="r hide-sm">Max drawdown</th>',
       conf_rows, "No double confirmations today yet.")}</section>
<section><div class="sh"><h2>Daily EMA crosses</h2><span class="meta" id="c-dx">last {DAILY_CROSS_DAYS} sessions</span></div>
<p class="note">EMA50 crossing EMA100 (early cross) or EMA200 (golden / death cross) on daily candles — the TradingView "EMA 50, 100 and 200" indicator.</p>
{table("t-dx", '<th>Day</th><th>Stock</th><th>Cross</th><th class="hide-sm">EMAs</th><th class="r">At cross</th><th class="r">Now</th><th class="r">Move</th>',
       cross_rows, "No daily EMA crosses in this window.")}</section>
<details class="more"><summary>All signals today <span class="mu" id="c-live"></span></summary>
<div class="tb" style="margin:12px 0"><div class="seg" id="f-live-dir"><button class="on" data-f="all">All</button>
<button data-f="buy">Buy</button><button data-f="sell">Sell</button></div></div>
{table("t-live", '<th>Time</th><th>Stock</th><th>Signal</th><th class="hide-sm">Strategy</th><th class="r">Price</th><th class="r">Move</th><th class="r">Max gain</th><th class="r hide-sm">Max drawdown</th>',
       live_rows, "No signals yet today — they appear here as each scan finds them.")}</details>
</div>

<div class="view" id="v-past">
<div class="sh"><h2>Past signals</h2><span class="meta" id="c-past"></span></div>
<div class="tb"><select class="sel" id="f-past-day" aria-label="Day">{day_opts or '<option value="">No data yet</option>'}</select>
<div class="seg" id="f-past-dir"><button class="on" data-f="all">All</button><button data-f="buy">Buy</button><button data-f="sell">Sell</button></div>
<div class="seg" id="f-past-strat"><button class="on" data-f="all">All</button><button data-f="SuperTrend">SuperTrend</button><button data-f="ADX_DI">ADX DI</button></div>
<select class="sel" id="f-past-sess" aria-label="Time of day"><option value="">All day</option>
<option value="09:15-11:00">9:15 – 11 AM</option><option value="11:00-14:00">11 AM – 2 PM</option>
<option value="14:00-15:30">2 – 3:30 PM</option></select>
<div class="seg" id="f-past-str"><button class="on" data-f="all">Any strength</button><button data-f="strong">Strong</button><button data-f="normal">Not strong</button></div>
<label class="chk"><input type="checkbox" id="f-past-conf"> Confirmed only</label>
<label class="chk"><input type="checkbox" id="f-past-fav"> ★ only</label>
<select class="sel" id="f-past-sym" aria-label="Stock or index">{past_sym_opts}</select>
<button class="sel dl" id="dl-shown" title="Download the signals shown below (current filters) as a spreadsheet">⬇ Excel (shown)</button>
<button class="sel dl" id="dl-all" title="Download every signal in the archive as a spreadsheet">⬇ Excel (all signals)</button></div>
<div class="daysum" id="past-sum"></div>
{table("t-past", '<th class="sorted desc" data-k="t" data-asc="0">Time</th><th data-k="sym" data-asc="1">Stock</th><th>Signal</th><th class="hide-md">Strategy</th><th class="r hide-sm">Price</th><th class="r hide-md">Exit</th><th class="r" data-k="mv" data-asc="0">Move</th><th class="r" data-k="mfe" data-asc="0">Max gain</th><th class="r hide-sm" data-k="mae" data-asc="1">Max drawdown</th><th class="r" data-k="bars" data-asc="1">Bars</th>',
       "", "No signals match these filters.")}
<details class="note below"><summary>How to read this table</summary>Every signal from the last {HISTORY_DAYS} sessions, replayed bar by bar from 15-minute data. Time = the
15-min candle the signal formed on (labelled by its start, like TradingView); <b>Move</b> = price change in the signal's favour
up to its flip (or to now if still open); <b>Max gain</b> / <b>Max drawdown</b> = furthest price went in the signal's favour /
against it before the flip; <b>Bars</b> = 15-min bars until that strategy flipped (SuperTrend reversed / ADX left that side).
✓ = part of a double confirmation that day. <b>Focus</b> (top right) limits this to Nifty 100 stocks + indices and
signals from 9:45 AM to 2:30 PM; switch it off to see everything.</details>
</div>

<div class="view" id="v-accuracy">
<div class="sh"><h2>Accuracy tracker</h2><span class="meta" id="acc-span">{e(span)}</span></div>
<div class="tb"><select class="sel" id="f-acc-sym" aria-label="Stock or index">{sym_opts}</select>
<span class="mu" id="acc-what" style="font-size:12px"></span></div>
<div class="kpis" id="acc-kpis"></div>
<div class="kpis k3" id="acc-k3"></div>
<div class="panel tw"><table><thead><tr><th>Signal</th><th class="r">Signals</th><th class="r hide-sm">Closed</th>
<th>Win rate</th><th class="r">Median bars</th><th class="r">Median move</th><th class="r">Median max gain</th>
<th class="r">Median max drawdown</th><th class="r hide-sm">Avg move</th>
<th class="r" title="Closed signals at their exit plus open signals at today's price">Avg incl. open</th>
<th class="r hide-sm">Open · now</th></tr></thead><tbody id="acc-body"></tbody></table></div>
<section style="margin-top:26px"><div class="sh"><h2>Daily EMA crosses — 5-year record</h2>
<span class="meta">every cross since 2021 on daily candles, held until the opposite cross · before costs</span></div>
<div class="panel tw"><table><thead><tr><th>Cross</th><th class="r">Crosses</th><th class="r">Still open</th><th>Win rate</th>
<th class="r">Avg move</th><th class="r">Median move</th><th class="r">Median sessions held</th></tr></thead>
<tbody id="dx-n100">{rec_n100}</tbody><tbody id="dx-all" style="display:none">{rec_all}</tbody></table></div>
<p class="note below">Today's Nifty 500 / 100 members only, so stocks that dropped out of the index are missing (survivorship bias
flatters these numbers). Open crosses aren't in the win rate. A delivery round trip costs ~{MOM_COST:.2f}%.</p></section>
<details class="more"><summary>More analysis — exit simulator, equity curve, time of day, stock leaderboard</summary>
<section style="margin-top:18px"><div class="sh"><h2>Exit-rule simulator</h2>
<span class="meta">what if you exited at a profit target or stop-loss instead of waiting for the flip?</span></div>
<div class="tb"><label class="chk">Target <select class="sel" id="sim-t"></select></label>
<label class="chk">Stop-loss <select class="sel" id="sim-s"></select></label>
<span class="mu" style="font-size:12px" id="sim-note"></span></div>
<div class="panel tw"><table><thead><tr><th>Signals</th><th class="r">Trades</th><th>Win rate</th><th class="r">Avg / trade</th>
<th class="r">Median</th><th class="r hide-sm">Hit target</th><th class="r hide-sm">Hit stop</th><th class="r hide-sm">Exited at flip</th>
<th class="r">vs waiting for flip</th></tr></thead><tbody id="sim-body"></tbody></table></div></section>
<section style="margin-top:26px"><div class="sh"><h2>Equity curve</h2>
<span class="meta">cumulative % if every closed signal were taken with equal size, in time order</span></div>
<div class="panel" style="padding:14px 16px"><div id="eq"></div></div></section>
<div class="grid2" style="margin-top:26px"><section><div class="sh"><h2>By time of day</h2>
<span class="meta">signal candle hour</span></div><div class="panel tw"><table><thead><tr><th>Hour</th><th class="r">Signals</th>
<th>Win rate</th><th class="r">Avg move</th><th class="r">Median max gain</th></tr></thead><tbody id="acc-hour"></tbody></table></div></section>
<section id="lb-wrap"><div class="sh"><h2>Stock leaderboard</h2><span class="meta">min 5 closed signals · by avg move</span></div>
<div class="panel tw"><table><thead><tr><th>Stock</th><th class="r">Closed</th><th>Win rate</th><th class="r">Avg move</th>
<th class="r hide-sm">Median max gain</th></tr></thead><tbody id="acc-lb"></tbody></table></div></section></div></details>
<section id="acc-log-wrap" style="display:none;margin-top:26px"><div class="sh"><h2 id="acc-log-title">Signals</h2>
<span class="meta" id="acc-log-count"></span></div>
<div class="panel tw"><table><thead><tr><th>When</th><th>Signal</th><th class="hide-sm">Strategy</th>
<th class="r hide-sm">Price</th><th class="r hide-md">Exit</th><th class="r">Move</th><th class="r">Max gain</th>
<th class="r hide-sm">Max drawdown</th><th class="r">Bars</th></tr></thead><tbody id="acc-log"></tbody></table></div></section>
<details class="note below"><summary>How these numbers are worked out</summary>A signal <b>wins</b> if price moved in its favour (up after a buy, down after a sell) by the bar where that
strategy flipped. <b>Median bars</b> = 15-min bars until the flip; <b>Median move</b> = typical % move in the signal's favour at
the flip; <b>Max gain</b> / <b>Max drawdown</b> = the furthest price went in the signal's favour / against it
(candle highs and lows) before the flip. Open signals aren't in the win rate or median move — <b>Avg incl. open</b> adds them at today's price, and ⚠ flags rows
where many signals are still open, where closed-only figures understate them. Builds up to {ARCHIVE_KEEP_DAYS} days of history.
<b>Simulator</b>: a trade exits at the target or stop, whichever price reaches first (if both are touched in the same 15-min candle
it assumes the stop — the cautious choice); otherwise at the flip. Open signals that haven't hit either level aren't counted.
Ignores brokerage, taxes and slippage. <b>Focus</b> (top right) applies here too.</details>
</div>

<div class="view" id="v-momentum">
{_momentum_view_html(state, meta.get("sectors", {}))}
</div>

<div class="view" id="v-stocks">
<section><div class="sh"><h2>Sector breadth</h2><span class="meta">tap a sector to filter the stocks below</span></div>
{table("t-sectors", '<th>Sector</th><th class="r">Stocks</th><th>SuperTrend long</th><th class="r">ADX buy / sell</th><th class="r">Golden cross</th><th class="r">Avg day</th>',
       "".join(sector_rows), "No sector data yet.")}</section>
<div class="sh"><h2>All stocks</h2><span class="meta" id="c-stocks"></span></div>
<div class="tb"><input type="search" id="f-stk-q" placeholder="Search stock…" autocomplete="off">
<select class="sel" id="f-stk-sector" aria-label="Sector">{sector_opts}</select>
<div class="seg" id="f-stk-view"><button class="on" data-f="today" title="Double confirmation on the latest day or a daily EMA cross in the last {DAILY_CROSS_DAYS} sessions, plus your starred stocks">Setups + ★</button><button data-f="all">All {len(stocks)}</button><button data-f="fav" title="Your starred stocks">★</button>
<button data-f="golden" title="Daily EMA50 above EMA200">Golden cross</button><button data-f="death" title="Daily EMA50 below EMA200">Death cross</button>
<button data-f="fresh" title="Golden or death cross in the last 10 sessions">New cross</button><button data-f="aligned" title="SuperTrend and ADX DI point the same way">Aligned</button>
<button data-f="near" title="Within {WATCHLIST_PCT:g}% of a SuperTrend flip">Near flip</button><button data-f="long">ST long</button>
<button data-f="short">ST short</button><button data-f="abuy">ADX buy</button><button data-f="asell">ADX sell</button>
<button data-f="stale" title="Latest candle older than the rest">Stale</button></div></div>
{table("t-stocks", '<th data-k="sym" data-asc="1">Stock</th><th class="hide-md" data-k="sector" data-asc="1">Sector</th><th class="r" data-k="px" data-asc="0">Price</th><th class="r" data-k="chg" data-asc="0">Day</th><th data-k="st" data-asc="0">SuperTrend</th><th data-k="adx" data-asc="0">ADX DI</th><th data-k="gck" data-asc="0" title="Daily EMA50 vs EMA200 — golden / death cross, sessions since">Daily trend</th><th class="r sorted" data-k="dist" data-asc="1">To flip</th><th data-k="mv" data-asc="0">Last signal</th>',
       "", "No stocks match.", "dist")}
</div>

<div class="foot">All times IST. Data: Yahoo Finance (~15 min delayed); this page refreshes every 5 minutes.
Golden / death cross = daily EMA50 above / below EMA200 (the TradingView "EMA 50, 100 and 200" indicator) — a
trend read, not a trade signal: in backtests (ema_cross_report.md) trading the crosses lagged buy-and-hold.
SELL signals and the accuracy figures are not from the original backtests. Not investment advice.</div>
</main>
<script>{data_js}</script>
<script>const STOCKS_GZ="{gz("".join(stock_rows))}";</script>
<script>const NEAR={WATCHLIST_PCT};
(async()=>{{
const gunzip=async b64=>{{const bin=Uint8Array.from(atob(b64),c=>c.charCodeAt(0));
 return await new Response(new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'))).text()}};
let ARCH=[];
try{{ARCH=JSON.parse(await gunzip(ARCH_GZ));
 document.querySelector('#t-stocks tbody').insertAdjacentHTML('afterbegin',await gunzip(STOCKS_GZ));}}
catch(err){{document.querySelector('main').insertAdjacentHTML('afterbegin','<div class="panel empty">This browser could not unpack the dashboard data — please update it.</div>');}}
{DASHBOARD_JS}
}})();</script></body></html>"""
    DASHBOARD_FILE.write_text(page, encoding="utf-8")


# ─── Market hours gate ──────────────────────────────────────────────────────

def seconds_until_next_check():
    now = datetime.now(IST)
    if now.weekday() >= 5:  # Sat/Sun
        days_ahead = 7 - now.weekday()
        nxt = (now + timedelta(days=days_ahead)).replace(hour=9, minute=15, second=0, microsecond=0)
        return (nxt - now).total_seconds()
    open_t = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if now < open_t:
        return (open_t - now).total_seconds()
    if now > close_t:
        nxt = open_t + timedelta(days=(3 if now.weekday() == 4 else 1))
        return (nxt - now).total_seconds()
    return 0  # market is open now


# ─── Scan ───────────────────────────────────────────────────────────────────

PACE_FRACTION = 0.85  # fraction of the poll window spent spreading requests
ONCE_PACE_SECONDS = 0.3  # --once mode: 500 stocks must fit a 15-min cycle
ONCE_LATEST_RUN = (16, 15)  # --once: allow late/delayed runs to catch the final bar
MIN_SCAN_GAP = 8 * 60       # --once: a run this soon after the last scan is a duplicate trigger


INDEX_STATE_PREFIX = "__INDEX__"  # state keys for indices, kept apart from stocks


def evaluate(name, df, prev, htf=None):
    """Run the strategies (SuperTrend, ADX DI) on one bar series. Returns (new_state, fired) where
    fired lists the signals that changed since `prev` (last cycle's state).
    `htf` (from htf_trends) adds the 1-hour trend gate."""
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    last_ts = str(df.index[-1])
    last_close = float(c[-1])

    atr = _atr_rma(h, l, c, ATR_PERIOD)
    trend, st_upper, st_lower = _supertrend(c, h, l, atr, MULTIPLIER)
    st_trend = int(trend[-1])
    # how far price is from the band it must cross to flip — drives the watchlist
    active_band = float(st_lower[-1] if st_trend == 1 else st_upper[-1])
    distance_pct = abs(last_close - active_band) / last_close * 100.0 if last_close else None

    is_open = np.array([t.hour == 9 and t.minute == 15 for t in df.index.time])
    di_plus, di_minus, sig = _calc_di_sig(h, l, c, DI_LEN, SIG_LEN)
    condition = _build_condition(di_plus, di_minus, sig, HL_RANGE, HL_TREND, is_open)
    adx_cond = float(condition[-1])

    fired = []
    prev_st = prev.get("st_trend")
    prev_adx = prev.get("adx_cond")
    if prev_st is not None and prev_st == -1 and st_trend == 1:
        fired.append(("SuperTrend", name, "BUY", last_close, last_ts))
    if prev_st is not None and prev_st == 1 and st_trend == -1:
        fired.append(("SuperTrend", name, "SELL", last_close, last_ts))
    if prev_adx is not None and prev_adx not in (1.0, 0.5) and adx_cond in (1.0, 0.5):
        label = "BUY_STRONG" if adx_cond == 1.0 else "BUY"
        fired.append(("ADX_DI", name, label, last_close, last_ts))
    if prev_adx is not None and prev_adx not in (-1.0, -0.5) and adx_cond in (-1.0, -0.5):
        label = "SELL_STRONG" if adx_cond == -1.0 else "SELL"
        fired.append(("ADX_DI", name, label, last_close, last_ts))

    # % change vs. the previous session's last close (for the dashboard)
    dates = df.index.date
    prev_session = c[dates < dates[-1]]
    day_chg_pct = (float((last_close / prev_session[-1] - 1) * 100.0)
                   if len(prev_session) and prev_session[-1] else None)

    new_state = {"st_trend": st_trend, "adx_cond": adx_cond, "last_ts": last_ts,
                 "last_close": last_close, "distance_pct": distance_pct,
                 "day_chg_pct": day_chg_pct, "name": name,
                 "tf1h": htf_at(htf, "1h", df.index[-1] + pd.Timedelta(minutes=15)),
                 "history": bar_signals(name, df, c, trend, condition, htf),
                 "history_from": str(df.index[np.isin(dates, sorted(set(dates))[-REPLAY_DAYS:])][0])}
    return new_state, fired


HISTORY_DAYS = 10        # trading days selectable in the dashboard's "Past signals" tab
SIM_LEVELS = (0.5, 1.0, 1.5, 2.0, 3.0)  # % targets / stops the exit simulator can test
REPLAY_DAYS = 10         # trading days replayed per scan (of the ~15 fetched; the rest is warm-up)
ARCHIVE_KEY = "__ARCHIVE__"
ARCHIVE_KEEP_DAYS = 30   # calendar days of replayed signals kept for the accuracy tracker


def bar_signals(name, df, close, trend, condition, htf=None):
    """Every signal the strategies gave, bar by bar, over the last REPLAY_DAYS
    trading days in `df` — the same state changes the live scan alerts on, but
    replayed on completed bars. "time" is the 15-min candle the signal formed
    on, labelled by its start like TradingView (the 3:15 PM candle closes the
    session at 3:30); "price" is that candle's close.
    "bars" is how many 15-min bars the signal lasted before that strategy
    turned (SuperTrend flipped back / ADX left that side); "open" means it
    hasn't turned yet and "bars" counts the bars so far. "exit_price" /
    "exit_time" are the close and candle time of the bar where it turned.
    "mfe" / "mae" are the maximum move in the signal's favour / against it
    (from candle highs and lows) between the signal and its flip, or so far.

    Double confirmations are added as strategy "CONFIRMED" entries: the latest
    SuperTrend and ADX DI signal of a day agree in direction and the first is
    still active when the second fires; the trade enters on the second signal
    and exits when the first of the two flips."""
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    st_side = trend.astype(np.int8)
    adx_side = np.sign(condition).astype(np.int8)
    last = len(close) - 1

    def flip_index(side, i):
        changed = np.nonzero(side[i + 1:] != side[i])[0]
        return (i + 1 + int(changed[0]), False) if len(changed) else (last, True)

    def candle(j):
        return df.index[j].strftime("%Y-%m-%d %H:%M:%S")

    def first_hits(i, end, buy):
        """For each SIM_LEVELS % move: bars after the signal until price first
        reached it in the signal's favour (hf) / against it (ha); 0 = never."""
        p = close[i]
        up = (high[i + 1:end + 1] / p - 1) * 100.0
        dn = (1 - low[i + 1:end + 1] / p) * 100.0
        fav, adv = (up, dn) if buy else (dn, up)

        def hits(x):
            out = []
            for lvl in SIM_LEVELS:
                j = np.nonzero(x >= lvl)[0]
                out.append(int(j[0]) + 1 if len(j) else 0)
            return out
        return hits(fav), hits(adv)

    def gates(i, buy):
        """Stage gate at the signal: did the 1-hour trend (last candle
        closed by this candle's close) point the same way?"""
        t = df.index[i] + pd.Timedelta(minutes=15)
        want = 1 if buy else -1
        return {"gh": htf_at(htf, "1h", t) == want}

    def entry(i, end, still_open, strategy, label):
        buy = "SELL" not in label
        if end > i:
            hi = (high[i + 1:end + 1].max() / close[i] - 1) * 100.0
            lo = (low[i + 1:end + 1].min() / close[i] - 1) * 100.0
            mfe, mae = (hi, lo) if buy else (-lo, -hi)
            hf, ha = first_hits(i, end, buy)
        else:
            mfe = mae = 0.0
            hf = ha = [0] * len(SIM_LEVELS)
        return {"time": candle(i), "strategy": strategy, "name": name, "label": label,
                "price": float(close[i]), "bar": str(df.index[i]),
                "bars": int(end - i), "open": still_open,
                "exit_price": None if still_open else float(close[end]),
                "exit_time": None if still_open else candle(end),
                "mfe": round(float(max(mfe, 0.0)), 3), "mae": round(float(min(mae, 0.0)), 3),
                "hf": hf, "ha": ha, **gates(i, buy), "_i": int(i), "_end": int(end)}

    dates = df.index.date
    days = sorted(set(dates))[-REPLAY_DAYS:]
    out = []
    for i in np.nonzero(np.isin(dates, days))[0]:
        if i == 0:
            continue
        labels = []
        if trend[i - 1] == -1 and trend[i] == 1:
            labels.append(("SuperTrend", "BUY"))
        elif trend[i - 1] == 1 and trend[i] == -1:
            labels.append(("SuperTrend", "SELL"))
        cur, prv = condition[i], condition[i - 1]
        if cur in (1.0, 0.5) and prv not in (1.0, 0.5):
            labels.append(("ADX_DI", "BUY_STRONG" if cur == 1.0 else "BUY"))
        elif cur in (-1.0, -0.5) and prv not in (-1.0, -0.5):
            labels.append(("ADX_DI", "SELL_STRONG" if cur == -1.0 else "SELL"))
        sides = {"SuperTrend": st_side, "ADX_DI": adx_side}
        for strat, label in labels:
            end, still_open = flip_index(sides[strat], i)
            out.append(entry(i, end, still_open, strat, label))

    latest = {}
    for sig in out:  # chronological, so the last one per day/strategy wins
        latest.setdefault(sig["time"][:10], {})[sig["strategy"]] = sig
    for day, by in latest.items():
        st, adx = by.get("SuperTrend"), by.get("ADX_DI")
        if not (st and adx) or ("SELL" in st["label"]) != ("SELL" in adx["label"]):
            continue
        first, second = sorted((st, adx), key=lambda x: x["_i"])
        if not first["open"] and first["_end"] <= second["_i"]:
            continue  # first had already flipped — not active together
        closed_ends = [x["_end"] for x in (first, second) if not x["open"]]
        end, still_open = (min(closed_ends), False) if closed_ends else (last, True)
        out.append(entry(second["_i"], end, still_open, "CONFIRMED",
                         "SELL" if "SELL" in st["label"] else "BUY"))
    for sig in out:
        del sig["_i"], sig["_end"]
    return out


def archive_history(state, new_state):
    """Move a symbol's freshly replayed signals into the rolling archive. The
    replay window is authoritative: that symbol's archived signals inside it are
    replaced (so a signal on a still-forming bar that disappears is dropped and
    open signals get their exit filled in); older ones are kept up to
    ARCHIVE_KEEP_DAYS for the accuracy tracker."""
    history = new_state.pop("history", [])
    since = new_state.pop("history_from", None)
    if since is None:
        return
    name = history[0]["name"] if history else new_state.get("name")
    cutoff = (datetime.now(IST) - timedelta(days=ARCHIVE_KEEP_DAYS)).strftime("%Y-%m-%d")
    archive = [h for h in state.get(ARCHIVE_KEY, [])
               if h["time"][:10] >= cutoff and not (h["name"] == name and h["bar"] >= since)
               and not h["strategy"].startswith("EMA_")]   # 15-min EMA crosses retired (now daily)
    state[ARCHIVE_KEY] = archive + history


def build_scan_order(universe, state):
    """Return (watchlist, rest): symbols within WATCHLIST_PCT of their last-
    known SuperTrend flip line, nearest first, then everyone else. Symbols
    with no prior distance (first run, or skipped last cycle) go in `rest`."""
    watchlist, rest = [], []
    for symbol in universe:
        dist = state.get(symbol, {}).get("distance_pct")
        if dist is not None and dist <= WATCHLIST_PCT:
            watchlist.append((dist, symbol))
        else:
            rest.append(symbol)
    watchlist.sort()
    return [s for _, s in watchlist], rest


def scan_indices(state):
    """Same strategies on the index series themselves (see INDEX_TICKERS)."""
    fired = []
    for name, yf_ticker in INDEX_TICKERS.items():
        key = INDEX_STATE_PREFIX + name
        try:
            df = fetch_15m(name, yf_ticker)
            if df is None or len(df) < 60:
                continue
            prev = state.get(key, {})
            state[key], index_fired = evaluate(name, df, prev, htf_trends(name, yf_ticker, df))
            state[key].update(daily_ema_trend(name, yf_ticker, df))
            index_fired += daily_cross_alerts(name, prev, state[key])
            archive_history(state, state[key])
            fired.extend(index_fired)
        except Exception as e:
            print(f"[warn] index {name}: {e}")
    return fired


def scan_once(per_symbol_delay=None):
    """Scan the universe with requests paced across the poll window, so the
    150-symbol sweep never bursts. Symbols sitting close to a SuperTrend flip
    (per last cycle's distance_pct) are scanned first so a live flip is picked
    up early; the rest are spaced ~POLL_SECONDS * PACE_FRACTION / len(rest)
    apart (with jitter). Passing per_symbol_delay (--once mode) uses that one
    spacing for everything. A circuit breaker pauses the whole scan if rate
    limiting keeps recurring. The indices are scanned last."""
    state = load_state()
    universe = load_universe()
    fired = []
    checked = 0
    consecutive_rate_limits = 0

    watchlist, rest = build_scan_order(universe, state)
    if watchlist:
        print(f"[watchlist] {len(watchlist)} symbol(s) near a SuperTrend flip — scanning them first")
    scan_order = watchlist + rest

    if per_symbol_delay is None:
        rest_budget = max(POLL_SECONDS * PACE_FRACTION - len(watchlist) * WATCHLIST_DELAY, 0)
        rest_delay = rest_budget / len(rest) if rest else 0
        watch_delay = WATCHLIST_DELAY
    else:
        rest_delay = watch_delay = per_symbol_delay

    for idx, symbol in enumerate(scan_order):
        try:
            df = fetch_15m(symbol)
            consecutive_rate_limits = 0

            if df is None or len(df) < 60:
                continue

            prev = state.get(symbol, {})
            state[symbol], symbol_fired = evaluate(symbol, df, prev, htf_trends(symbol, f"{symbol}.NS", df))
            state[symbol].update(daily_ema_trend(symbol, f"{symbol}.NS", df))
            symbol_fired += daily_cross_alerts(symbol, prev, state[symbol])
            archive_history(state, state[symbol])
            fired.extend(symbol_fired)
            checked += 1

        except RateLimited:
            consecutive_rate_limits += 1
            print(f"[warn] {symbol}: still rate-limited after {RETRY_MAX} retries, skipping this cycle")
            if consecutive_rate_limits >= CIRCUIT_BREAKER_THRESHOLD:
                print(f"[circuit-breaker] {consecutive_rate_limits} consecutive rate-limit hits — "
                      f"cooling down {COOLDOWN_SECONDS}s before resuming")
                time.sleep(COOLDOWN_SECONDS)
                consecutive_rate_limits = 0

        except Exception as e:
            print(f"[warn] {symbol}: {e}")

        if idx < len(scan_order) - 1:
            delay = watch_delay if idx < len(watchlist) else rest_delay
            time.sleep(delay * random.uniform(0.7, 1.3))

    fired.extend(scan_indices(state))
    try:
        update_momentum(state, universe)
    except Exception as e:  # noqa: BLE001 — never let the paper tracker break alerts
        print(f"[warn] momentum update failed: {e}")

    now_str = f"{datetime.now(IST):%Y-%m-%d %H:%M:%S}"
    log = state.get(SIGNAL_LOG_KEY, [])
    def gate_flags(name, label):
        info = state.get(name) or state.get(INDEX_STATE_PREFIX + name) or {}
        want = -1 if "SELL" in label else 1
        return {"gh": info.get("tf1h") == want}
    log.extend({"time": now_str, "strategy": strategy, "name": name, "label": label,
                "price": price, "bar": ts, **gate_flags(name, label)}
               for strategy, name, label, price, ts in fired)
    state[SIGNAL_LOG_KEY] = log[-SIGNAL_LOG_MAX:]
    sectors = {**state.get(SCAN_META_KEY, {}).get("sectors", {}), **SECTORS}
    n100 = load_nifty100() or state.get(SCAN_META_KEY, {}).get("n100", [])
    state[SCAN_META_KEY] = {"time": now_str, "checked": checked, "total": len(universe), "sectors": sectors,
                            "universe": universe, "n100": n100}

    save_state(state)
    save_bars_cache()
    write_dashboard(state)
    print(f"[{now_str}] checked {checked}/{len(universe)} stocks "
          f"+ {len(INDEX_TICKERS)} indices, {len(fired)} new signal(s)")

    # double confirmations completed by this scan's signals
    confirmed = [(c["name"], c["dir"], c["price"])
                 for c in find_confluence(state[SIGNAL_LOG_KEY]).get(now_str[:10], [])
                 if c["time"] == now_str]
    notify_signals(fired, confirmed, gate_flags)


def once_skip_reason(now, last_scan=None, probe=None):
    """Why a scheduled (--once) run should do nothing, or None to scan:
    a second trigger within MIN_SCAN_GAP of the last scan (the external timer
    and GitHub's own schedule both fire), or no NIFTY 50 candle dated today
    (an NSE holiday, or the first candle isn't published yet)."""
    if last_scan is None and STATE_FILE.exists():
        last_scan = load_state().get(SCAN_META_KEY, {}).get("time")
    if last_scan and (now.replace(tzinfo=None) - datetime.fromisoformat(last_scan)).total_seconds() < MIN_SCAN_GAP:
        return f"duplicate trigger (last scan {last_scan[11:16]})"
    if probe is None:
        probe = _download("NIFTY 50", INDEX_TICKERS["NIFTY 50"], "1d")
    if probe is None or probe.empty or probe.index[-1].date() != now.date():
        return "no market data for today (holiday, or first candle not published yet)"
    return None


def main():
    print(f"Nifty150 live signal notifier — SuperTrend({ATR_PERIOD}x{MULTIPLIER}) + ADX DI/DI-")
    print(f"Universe: {len(NIFTY150)} stocks  |  Poll: every {POLL_SECONDS//60} min  |  State: {STATE_FILE}")
    while True:
        wait = seconds_until_next_check()
        if wait > 0:
            resume_at = datetime.now(IST) + timedelta(seconds=wait)
            print(f"Market closed — sleeping until {resume_at:%Y-%m-%d %H:%M} IST")
            time.sleep(min(wait, 3600))  # wake hourly to re-check, handles long waits safely
            continue
        t0 = time.time()
        scan_once()
        elapsed = time.time() - t0
        remaining = POLL_SECONDS - elapsed
        if remaining > 0:
            time.sleep(remaining)


if __name__ == "__main__":
    if "--test-telegram" in sys.argv:
        if not TELEGRAM_TOKEN:
            sys.exit("Telegram is not configured — see the Telegram note at the top of this file.")
        ok = send_telegram("✅ <b>Technical Dashboard</b> — Telegram alerts are working.\n"
                           "Signals will arrive like this:\n\n" + format_alert(
                               [("SuperTrend", "RELIANCE", "BUY", 1207.70, ""),
                                ("ADX_DI", "RELIANCE", "BUY_STRONG", 1207.70, ""),
                                ("SuperTrend", "M&M", "SELL", 2792.00, "")],
                               [("RELIANCE", "buy", 1207.70)]))
        sys.exit(0 if ok else "Telegram test message failed — check the token and chat id.")
    if "--once" in sys.argv:
        # Single scan for schedulers (GitHub Actions) — the cron decides timing,
        # this gate just drops runs that GitHub delayed past the session.
        now = datetime.now(IST)
        if now.weekday() >= 5 or not ((9, 15) <= (now.hour, now.minute) <= ONCE_LATEST_RUN):
            print(f"[{now:%Y-%m-%d %H:%M} IST] outside market hours — skipping scan")
            if STATE_FILE.exists():  # still (re)publish the page from the last scan
                write_dashboard(load_state())
            sys.exit(0)
        reason = once_skip_reason(now)
        if reason:
            print(f"[{now:%Y-%m-%d %H:%M} IST] {reason} — skipping scan")
            if STATE_FILE.exists() and "duplicate" not in reason:
                write_dashboard(load_state())
            sys.exit(0)
        scan_once(float(os.environ.get("SCAN_PACE_SECONDS", ONCE_PACE_SECONDS)))
        sys.exit(0)
    if "--warm" in sys.argv:
        # Pre-market (scheduled ~8:30 AM IST): the slow once-a-day downloads —
        # 15 days of 15-min bars, 60 days hourly, 5 years daily — for every
        # symbol, so market-hours scans only fetch today's bars. No signals,
        # no alerts, no dashboard; just fills the bar cache.
        uni = load_universe()
        names = [(s, f"{s}.NS") for s in uni] + list(INDEX_TICKERS.items()) + list(MOM_BENCH.items())
        t0 = time.time()
        for k, (name, yt) in enumerate(names, 1):
            try:
                df = fetch_15m(name, yt)
                if df is not None:
                    htf_trends(name, yt, df)
                daily_history(name, yt)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] warm {name}: {e}")
            if k % 50 == 0:
                print(f"[warm] {k}/{len(names)} ({time.time() - t0:.0f}s)")
                save_bars_cache()
            time.sleep(float(os.environ.get("SCAN_PACE_SECONDS", ONCE_PACE_SECONDS)))
        save_bars_cache()
        print(f"[warm] cached {len(names)} symbols in {time.time() - t0:.0f}s")
        sys.exit(0)
    if "--test" in sys.argv:
        print("[--test] Ignoring market-hours gate, running one scan against latest available bars.")
        scan_once(float(os.environ.get("SCAN_PACE_SECONDS", ONCE_PACE_SECONDS)))
        sys.exit(0)
    main()
