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
  - Universe = Nifty 100 + Nifty Midcap 50, fetched from NSE's official
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
FETCH_PERIOD = "15d"    # 15-min history per fetch: 10 sessions replayed + ~5 of indicator warm-up

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
# editing this file. Nifty 100 and Midcap 50 don't overlap -> 150 names.
NSE_INDEX_URLS = {
    "Nifty100": ("https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv", 100),
    "Midcap50": ("https://nsearchives.nseindia.com/content/indices/ind_niftymidcap50list.csv", 50),
}

# The indices themselves, scanned with the same strategies after the stocks.
INDEX_TICKERS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "NIFTY NEXT 50": "^NSMIDCP",
    "NIFTY MIDCAP 150": "NIFTYMIDCAP150.NS",
    "NIFTY SMALLCAP 250": "NIFTYSMLCAP250.NS",
    "NIFTY 500": "^CRSLDX",
}


def load_universe():
    """Live Nifty 100 + Midcap 50 constituents from NSE, else the NIFTY150
    snapshot. Each CSV must parse to roughly its expected size, so an error
    page or a half-loaded file can't silently shrink the universe."""
    from io import StringIO
    try:
        symbols = set()
        for name, (url, expected) in NSE_INDEX_URLS.items():
            if SESSION:
                text = SESSION.get(url, timeout=15).text
            else:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                text = urllib.request.urlopen(req, timeout=15).read().decode()
            got = set(pd.read_csv(StringIO(text))["Symbol"].str.strip())
            if abs(len(got) - expected) > 5:
                raise ValueError(f"{name} list has {len(got)} symbols, expected ~{expected}")
            symbols |= got
        added, dropped = symbols - set(NIFTY150), set(NIFTY150) - symbols
        if added or dropped:
            print(f"[info] NSE constituents changed vs. built-in list — "
                  f"added {sorted(added)}, dropped {sorted(dropped)}")
        return sorted(symbols)
    except Exception as e:
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
    keep = sorted(set(df.index.date))[-15:]
    df = df[np.isin(df.index.date, keep)].copy()
    df.attrs["full_fetch"] = str(today)  # set last: attrs don't reliably survive concat/slicing
    cache[yf_ticker] = df
    return df


def _download(symbol, yf_ticker, period):
    """One Yahoo download with exponential-backoff retry on rate limiting."""
    last_err = None
    for attempt in range(RETRY_MAX):
        try:
            ticker = yf.Ticker(yf_ticker, session=SESSION) if SESSION else yf.Ticker(yf_ticker)
            df = ticker.history(period=period, interval="15m")
            if df.empty:
                return None
            df = df.rename(columns=str.lower)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(IST)
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
        strat = "SuperTrend" if strategy == "SuperTrend" else "ADX DI"
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


def notify_signals(fired, confirmed=()):
    """Send one scan's signals: Telegram digest, console, desktop popup."""
    if not fired and not confirmed:
        return
    n_sell = sum("SELL" in f[2] for f in fired)
    if os.environ.get("CI"):
        # GitHub Actions logs are public for a public repo — keep signals out of them
        print(f"[signal] {len(fired)} fired (details hidden from CI logs — see Telegram / dashboard)")
    else:
        print("\n" + format_alert(fired, confirmed, html=False) + "\n")
    send_telegram(format_alert(fired, confirmed))  # before the popup, which blocks
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
.foot{font-size:12px;color:var(--faint);padding:14px 0 0;border-top:1px solid var(--line);line-height:1.6}
@media (max-width:1100px){.hide-md{display:none}}
@media (max-width:980px){.idx{grid-template-columns:repeat(3,1fr)}}
@media (max-width:700px){.wrap{padding:0 14px}.brand .sub{display:none}
.idx{display:flex;overflow-x:auto;scroll-snap-type:x mandatory;margin:0 -14px 20px;padding:0 14px 4px}
.idx .ix{flex:0 0 46%;scroll-snap-align:start}.kpis{grid-template-columns:repeat(2,1fr)}
.kpis.k3{grid-template-columns:1fr}
.hide-sm{display:none}td,th{padding:8px 9px}.status .lbl{display:none}.status{padding:0 9px}}
""".replace("DARK", """--bg:#000;--panel:#000;--fg:#f1f5f9;--muted:#8a94a6;--faint:#5b6577;--line:#1c1f26;
--head:#07080a;--hover:#0d0f13;--up:#34d399;--up-bg:#04170f;--up-bd:#0b3a26;--dn:#f87171;--dn-bg:#1c0607;
--dn-bd:#4a1416;--accent:#818cf8;--accent-bg:#0e0f24;--hl:#141005;--warn:#fbbf24;--warn-bg:#191204;--shadow:none""")

DASHBOARD_JS = """
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
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
/* segmented controls + filters */
function seg(el,fn){const bs=$$('button',el);bs.forEach(b=>b.onclick=()=>{bs.forEach(x=>x.classList.remove('on'));
 b.classList.add('on');fn(b.dataset.f)})}
function filterer(tbodySel,countSel){const rows=$$(tbodySel+' tr[data-sym]'),f={};
 const run=()=>{let n=0;for(const r of rows){const d=r.dataset;let ok=true;
  for(const[k,v]of Object.entries(f)){if(v===''||v==='all')continue;
   if(k==='q'){if(!d.sym.includes(v))ok=false}else if(k==='conf'){if(v&&d.conf!=='1')ok=false}
   else if(k==='view'){ok=ok&&VIEWS[v](d)}else if(d[k]!==v)ok=false}
  r.style.display=ok?'':'none';n+=ok}
  if(countSel)$(countSel).textContent=n+' shown';const e=$(tbodySel+' .nores');if(e)e.style.display=n?'none':'';};
 return {set(k,v){f[k]=v;run()},run}}
const VIEWS={all:()=>true,aligned:d=>(d.st==='1'&&+d.adx>0)||(d.st==='-1'&&+d.adx<0),near:d=>+d.dist<=NEAR,
 long:d=>d.st==='1',short:d=>d.st==='-1',abuy:d=>+d.adx>0,asell:d=>+d.adx<0};
const live=filterer('#t-live tbody');seg($('#f-live-dir'),v=>live.set('dir',v));
/* shared archive (ARCH/NAMES/PX/PDAYS come from the data script).
   ARCH row: [time, name#, label, strategy 0 ST/1 ADX/2 CONFIRMED, price, exit, bars, open, max gain, max drawdown] */
const SN=['SuperTrend','ADX_DI','CONFIRMED'];
const CONF=new Set(ARCH.filter(r=>r[3]===2).map(r=>r[0].slice(0,10)+'|'+r[1]));
const mvOf=r=>{const px=r[7]?PX[r[1]]:r[5];if(!px||!r[4])return null;const m=(px/r[4]-1)*100;return r[2].includes('SELL')?-m:m};
/* past signals, drawn per selected day:
   [time, name, label, strategy, price, exit, move, bars, open, confirmed, max gain, max drawdown] */
const PAST=ARCH.filter(r=>r[3]<2&&PDAYS.has(r[0].slice(0,10))).reverse().map(r=>[r[0],NAMES[r[1]],r[2],SN[r[3]],r[4],
 r[7]?null:r[5],mvOf(r),r[6],r[7],CONF.has(r[0].slice(0,10)+'|'+r[1])?1:0,r[8],r[9]]);
const PT=$('#t-past tbody'),pf={day:'',dir:'all',strat:'all',conf:false,sym:''};let pk='t',pa=false;
const esc=x=>String(x).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fp=v=>v==null?'—':v.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
const tg=l=>'<span class="tag '+(l.includes('SELL')?'dn':'up')+'">'+l.replace('_',' ')+'</span>';
const pc=v=>v==null?'<span class="fa">—</span>':Math.abs(v)<0.005?'<span class="mu">0.00%</span>':
 '<span class="'+(v>0?'up':'dn')+'">'+(v>0?'+':'')+v.toFixed(2)+'%</span>';
const t12=t=>{let[h,m]=t.slice(11,16).split(':').map(Number);const ap=h>=12?'PM':'AM';return(h%12||12)+':'+String(m).padStart(2,'0')+' '+ap};
const MON=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const dshort=t=>+t.slice(8,10)+' '+MON[+t.slice(5,7)-1];
const bt=(b,o)=>b==null?'—':o?(b===0?'<span class="tag ok">new</span>':'<span class="mu">'+b+'+</span>'):b;
function renderPast(){const rows=PAST.filter(r=>r[0].startsWith(pf.day)&&(pf.dir==='all'||(pf.dir==='sell')===r[2].includes('SELL'))
 &&(pf.strat==='all'||r[3]===pf.strat)&&(!pf.conf||r[9])&&(!pf.sym||r[1]===pf.sym));
 const i={t:0,sym:1,mv:6,bars:7,mfe:10,mae:11}[pk];rows.sort((a,b)=>{const x=a[i],y=b[i];
  const v=typeof x==='string'?x.localeCompare(y):(x??-999)-(y??-999);return pa?v:-v});
 PT.innerHTML=rows.map(r=>'<tr><td class="mu">'+(pf.day?'':dshort(r[0])+', ')+t12(r[0])+'</td><td class="sym">'+esc(r[1])+(r[9]?' <span class="tag ok">✓</span>':'')+
  '</td><td>'+tg(r[2])+'</td><td class="mu hide-md">'+(r[3]==='SuperTrend'?'SuperTrend':'ADX DI')+'</td><td class="r hide-sm">'+fp(r[4])+
  '</td><td class="r hide-md">'+(r[8]?'<span class="fa">open</span>':fp(r[5]))+'</td><td class="r">'+pc(r[6])+'</td><td class="r">'+pc(r[10])+'</td><td class="r hide-sm">'+pc(r[11])+
  '</td><td class="r">'+bt(r[7],r[8])+'</td></tr>').join('')
  ||'<tr><td colspan="10" class="empty">No signals match these filters.</td></tr>';
 $('#c-past').textContent=rows.length+' shown';}
$$('#t-past th[data-k]').forEach(th=>th.onclick=()=>{const k=th.dataset.k;pa=k===pk?!pa:th.dataset.asc==='1';pk=k;
 $$('#t-past th').forEach(x=>x.classList.remove('sorted','desc'));th.classList.add('sorted');if(!pa)th.classList.add('desc');renderPast()});
const daySel=$('#f-past-day');
daySel.onchange=()=>{pf.day=daySel.value;$$('.daysum').forEach(x=>x.style.display=x.dataset.day===pf.day?'':'none');renderPast()};
seg($('#f-past-dir'),v=>{pf.dir=v;renderPast()});seg($('#f-past-strat'),v=>{pf.strat=v;renderPast()});
$('#f-past-conf').onchange=e=>{pf.conf=e.target.checked;renderPast()};
$('#f-past-sym').onchange=e=>{pf.sym=e.target.value;renderPast()};
daySel.onchange();
/* accuracy: overall or for one stock / index */
const med=a=>{if(!a.length)return null;const s=[...a].sort((x,y)=>x-y),m=s.length>>1;return s.length%2?s[m]:(s[m-1]+s[m])/2};
function stats(rows){const c=[],o=[],b=[],g=[],d=[];for(const r of rows){const m=mvOf(r);if(m==null)continue;
 if(r[7])o.push(m);else{c.push(m);if(r[6]!=null)b.push(r[6]);if(r[8]!=null){g.push(r[8]);d.push(r[9])}}}
 return{n:rows.length,closed:c.length,open:o.length,win:c.length?c.filter(x=>x>0).length/c.length*100:null,bars:med(b),
  move:med(c),avg:c.length?c.reduce((x,y)=>x+y,0)/c.length:null,open_move:med(o),mfe:med(g),mae:med(d)}}
const kpi=(k,v,s)=>'<div class="panel kpi"><div class="k">'+k+'</div><div class="v">'+v+'</div><div class="s">'+s+'</div></div>';
const big=v=>v==null?'—':Math.abs(v)<0.005?'<span class="mu">0.00%</span>':'<span class="'+(v>0?'up':'dn')+'">'+(v>0?'+':'')+v.toFixed(2)+'%</span>';
const num=(v,d=0,suf='')=>v==null?'—':v.toFixed(d)+suf;
function accRow(tag,rows){const a=stats(rows);
 const win=a.win==null?'<span class="fa">—</span>':'<div class="wr"><span style="min-width:38px">'+a.win.toFixed(0)+
  '%</span><span class="tr"><b class="'+(a.win<50?'lo':'')+'" style="width:'+a.win.toFixed(0)+'%"></b></span></div>';
 return '<tr><td>'+tg(tag)+'</td><td class="r">'+a.n+'</td><td class="r hide-sm">'+a.closed+'</td><td>'+win+'</td><td class="r">'+
  num(a.bars)+'</td><td class="r">'+pc(a.move)+'</td><td class="r">'+pc(a.mfe)+'</td><td class="r">'+pc(a.mae)+
  '</td><td class="r hide-sm">'+pc(a.avg)+'</td><td class="r hide-sm">'+a.open+' · '+pc(a.open_move)+'</td></tr>'}
function renderAcc(sym){const ni=sym?NAMES.indexOf(sym):-1;const rows=sym?ARCH.filter(r=>r[1]===ni):ARCH;
 const sig=rows.filter(r=>r[3]<2),cf=rows.filter(r=>r[3]===2),all=stats(sig),con=stats(cf);
 const days=[...new Set(sig.map(r=>r[0].slice(0,10)))].sort();
 $('#acc-span').textContent=days.length?dshort(days[0])+' – '+dshort(days[days.length-1])+' · '+days.length+' sessions':'no signals';
 $('#acc-what').textContent=sym?'Accuracy for '+sym+' only':'All '+NAMES.length+' stocks & indices';
 $('#acc-kpis').innerHTML=kpi('Signals tracked',all.n,all.closed+' closed · '+all.open+' open')+
  kpi('Win rate (closed)',num(all.win,0,'%'),'moved in the signal’s favour by its flip')+
  kpi('Median bars to flip',num(all.bars),'≈ '+(all.bars==null?'—':(all.bars*15/60).toFixed(1)+' h')+' of trading')+
  kpi('Confirmed setups win rate',num(con.win,0,'%'),con.closed+' closed · median '+pc(con.move));
 $('#acc-k3').innerHTML=[['Median move at flip','move','where price was when the strategy flipped'],
  ['Median max gain','mfe','furthest in the signal’s favour before the flip'],
  ['Median max drawdown','mae','furthest against the signal before the flip']].map(([t,k,h])=>kpi(t,big(all[k]),h+' · confirmed '+pc(con[k]))).join('');
 const st=sig.filter(r=>r[3]===0),ad=sig.filter(r=>r[3]===1),L=(a,l)=>a.filter(r=>r[2]===l),grp=t=>'<tr class="grp"><td colspan="10">'+t+'</td></tr>';
 $('#acc-body').innerHTML=grp('SuperTrend')+accRow('BUY',L(st,'BUY'))+accRow('SELL',L(st,'SELL'))+accRow('ALL',st)+
  grp('ADX DI')+accRow('BUY',L(ad,'BUY'))+accRow('BUY_STRONG',L(ad,'BUY_STRONG'))+accRow('SELL',L(ad,'SELL'))+
  accRow('SELL_STRONG',L(ad,'SELL_STRONG'))+accRow('ALL',ad)+grp('Double confirmation (enter on 2nd signal, exit on 1st flip)')+
  accRow('CONFIRMED BUY',L(cf,'BUY'))+accRow('CONFIRMED SELL',L(cf,'SELL'));
 const w=$('#acc-log-wrap');w.style.display=sym?'':'none';if(!sym)return;
 const log=[...rows].reverse();$('#acc-log-title').textContent=sym+' — every signal';$('#acc-log-count').textContent=log.length+' signals';
 $('#acc-log').innerHTML=log.map(r=>'<tr><td class="mu">'+dshort(r[0])+', '+t12(r[0])+'</td><td>'+tg(r[3]===2?'CONFIRMED '+r[2]:r[2])+
  '</td><td class="mu hide-sm">'+(r[3]===0?'SuperTrend':r[3]===1?'ADX DI':'Both')+'</td><td class="r hide-sm">'+fp(r[4])+
  '</td><td class="r hide-md">'+(r[7]?'<span class="fa">open</span>':fp(r[5]))+'</td><td class="r">'+pc(mvOf(r))+'</td><td class="r">'+
  pc(r[8])+'</td><td class="r hide-sm">'+pc(r[9])+'</td><td class="r">'+bt(r[6],r[7])+'</td></tr>').join('')||
  '<tr><td colspan="9" class="empty">No signals for this one yet.</td></tr>'}
const accSel=$('#f-acc-sym');accSel.onchange=()=>{renderAcc(accSel.value);try{localStorage.setItem('accsym',accSel.value)}catch(e){}};
try{const v=localStorage.getItem('accsym');if(v&&[...accSel.options].some(o=>o.value===v))accSel.value=v}catch(e){}
renderAcc(accSel.value);
const stk=filterer('#t-stocks tbody','#c-stocks');seg($('#f-stk-view'),v=>stk.set('view',v));
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


def _strat_name(strategy):
    return "SuperTrend" if strategy == "SuperTrend" else "ADX DI"


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
    archive = state.get(ARCHIVE_KEY, [])

    # ── index strip
    ix_html = []
    for name in INDEX_TICKERS:
        info = state.get(INDEX_STATE_PREFIX + name)
        if not info:
            continue
        dist = info.get("distance_pct")
        dist_txt = e(f"{dist:.2f}% to flip") if dist is not None else ""
        ix_html.append(
            f'<div class="panel ix"><div class="nm" title="{e(name)}">{e(name)}</div>'
            f'<div class="px">{_fmt_price(info.get("last_close"))}</div>'
            f'<div class="ft">{_pct(info.get("day_chg_pct"))}'
            f'<span class="sm">{dist_txt}</span></div>'
            f'<div class="tags">{_st_tag(info.get("st_trend"))}{_adx_tag(info.get("adx_cond"))}</div></div>')

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

    kpis = (
        f'<div class="panel kpi"><div class="k">Signals {"today" if latest_day else ""}</div>'
        f'<div class="v">{len(today)}</div><div class="s"><span class="up">{n_buy} buy</span> · '
        f'<span class="dn">{len(today) - n_buy} sell</span></div>{split(n_buy, len(today) - n_buy)}</div>'
        f'<div class="panel kpi"><div class="k">Double confirmations</div><div class="v">{len(conf_today)}</div>'
        f'<div class="s"><span class="up">{sum(c["dir"] == "buy" for c in conf_today)} buy</span> · '
        f'<span class="dn">{sum(c["dir"] == "sell" for c in conf_today)} sell</span></div></div>'
        f'<div class="panel kpi"><div class="k">SuperTrend</div><div class="v">{st_long}'
        f'<span class="mu" style="font-size:14px"> / {st_short}</span></div><div class="s">long / short</div>'
        f'{split(st_long, st_short)}</div>'
        f'<div class="panel kpi"><div class="k">ADX DI</div><div class="v">{adx_buy}'
        f'<span class="mu" style="font-size:14px"> / {adx_sell}</span></div><div class="s">buy / sell state</div>'
        f'{split(adx_buy, adx_sell)}</div>')

    def now_move(s):
        mv, _ = outcome({**s, "open": True}, _current_price(state, s["name"]))
        return mv

    def excursion_cells(h):
        if not h:
            return '<td class="r"><span class="fa">—</span></td><td class="r hide-sm"><span class="fa">—</span></td>'
        return f'<td class="r">{_pct(h.get("mfe"))}</td><td class="r hide-sm">{_pct(h.get("mae"))}</td>'

    conf_rows = "".join(
        f'<tr data-sym="{e(c["name"])}"><td class="sym">{e(c["name"])}</td>'
        f'<td>{_tag("CONFIRMED " + c["dir"].upper())}</td>'
        f'<td>{_tag(c["st"]["label"])} <span class="mu">{e(fmt_time(c["st"]["time"]))}</span></td>'
        f'<td>{_tag(c["adx"]["label"])} <span class="mu">{e(fmt_time(c["adx"]["time"]))}</span></td>'
        f'<td class="r">{_fmt_price(c["price"])}</td><td class="r">{_fmt_price(_current_price(state, c["name"]))}</td>'
        f'<td class="r">{_pct(now_move({"label": c["dir"].upper(), "price": c["price"], "name": c["name"]}))}</td>'
        f'{excursion_cells(_match_replay(archive, c["name"], ("CONFIRMED",), c["dir"], latest_day))}</tr>'
        for c in conf_today)
    live_rows = "".join(
        f'<tr data-sym="{e(s["name"])}" data-dir="{_direction(s["label"])}"'
        f'{" class=fresh" if s["time"] == scan_time else ""}>'
        f'<td class="mu">{e(fmt_time(s["time"]))}</td><td class="sym">{e(s["name"])}</td><td>{_tag(s["label"])}</td>'
        f'<td class="mu hide-sm">{_strat_name(s["strategy"])}</td><td class="r">{_fmt_price(s["price"])}</td>'
        f'<td class="r">{_pct(now_move(s))}</td>'
        f'{excursion_cells(_match_replay(archive, s["name"], (s["strategy"],), _direction(s["label"]), s["time"][:10], s["time"]))}</tr>'
        for s in today)

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
                  1 if h.get("open", True) else 0, h.get("mfe"), h.get("mae")]
                 for h in sorted(archive, key=lambda x: x["time"])]
    prices = {name_ix[n]: _current_price(state, n) for n in names}
    data_js = ("const NAMES=" + json.dumps(names) + ";const ARCH=" + json.dumps(arch_rows, separators=(",", ":"))
               + ";const PX=" + json.dumps(prices, separators=(",", ":"))
               + ";const PDAYS=new Set(" + json.dumps(past_days) + ");").replace("</", "<\\/")
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
    day_sums = []
    for d in past_days:
        ds = [h for h in past if h["time"][:10] == d]
        b = sum(_direction(h["label"]) == "buy" for h in ds)
        acc = accuracy_stats(ds, state)
        parts = [f"<span><b>{len(ds)}</b> signals</span>", f'<span class="up">{b} buy</span>',
                 f'<span class="dn">{len(ds) - b} sell</span>',
                 f"<span><b>{sum(1 for k in conf_keys if k[0] == d)}</b> double confirmations</span>"]
        if acc["bars"] is not None:
            parts.append(f"<span>median <b>{acc['bars']:.0f}</b> bars to flip</span>")
        if acc["win"] is not None:
            parts.append(f"<span>win rate <b>{acc['win']:.0f}%</b> ({acc['closed']} closed)</span>")
        if acc["mfe"] is not None:
            parts.append(f"<span>median max gain <b class=up>{acc['mfe']:+.2f}%</b> · "
                         f"max drawdown <b class=dn>{acc['mae']:+.2f}%</b></span>")
        day_sums.append(f'<div class="daysum" data-day="{d}">{"".join(parts)}</div>')

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
    last_sig = {}
    for h in sorted(strategy_signals(archive), key=lambda x: x["time"]):
        last_sig[h["name"]] = h
    for s in log:
        if s["name"] not in last_sig or s["time"] > last_sig[s["name"]]["time"]:
            last_sig[s["name"]] = s
    fresh = {s["name"] for s in log if s["time"] == scan_time}
    stock_rows = []
    for sym in sorted(stocks, key=lambda x: (stocks[x].get("distance_pct") is None, stocks[x].get("distance_pct") or 0)):
        info = stocks[sym]
        st, adx, dist, chg = (info.get("st_trend"), info.get("adx_cond") or 0.0,
                              info.get("distance_pct"), info.get("day_chg_pct"))
        ls = last_sig.get(sym)
        dist_s = f"{dist:.2f}%" if dist is not None else "—"
        mv = outcome({**ls, "open": True}, info.get("last_close"))[0] if ls else None
        max_txt = (f' · max {ls["mfe"]:+.2f}% / {ls["mae"]:+.2f}%' if ls and ls.get("mfe") is not None else "")
        stock_rows.append(
            f'<tr data-sym="{e(sym)}" data-st="{st}" data-adx="{adx}" data-dist="{dist if dist is not None else 999}" '
            f'data-chg="{chg if chg is not None else 0}" data-px="{info.get("last_close") or 0}" '
            f'data-mv="{mv if mv is not None else -999}"{" class=fresh" if sym in fresh else ""}>'
            f'<td class="sym">{e(sym)}</td><td class="r">{_fmt_price(info.get("last_close"))}</td>'
            f'<td class="r">{_pct(chg)}</td><td>{_st_tag(st)}</td><td>{_adx_tag(adx)}</td>'
            f'<td class="r">{dist_s}</td>'
            f'<td>{(_tag(ls["label"]) + " " + _pct(mv) + "<div class=sm>" + e(fmt_stamp(ls["time"])) + max_txt + "</div>") if ls else "<span class=fa>—</span>"}</td></tr>')

    def table(tid, head, rows, empty, sort=None):
        return (f'<div class="panel tw"><table id="{tid}"{f" data-sort={sort}" if sort else ""}><thead><tr>{head}</tr>'
                f'</thead><tbody>{rows}<tr class="nores" style="display:{"none" if rows else ""}">'
                f'<td colspan="12" class="empty">{empty}</td></tr></tbody></table></div>')

    scanned_iso = f"{scan_time.replace(' ', 'T')}+05:30" if scan_time else ""
    sub = (f'{meta.get("checked")}/{meta.get("total")} stocks · {len(INDEX_TICKERS)} indices · '
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
<div class="tools-r"><span class="status" id="status"><i></i><span class="lbl">Waiting</span></span>
<button class="iconbtn theme" id="theme" title="Light / dark" aria-label="Switch light or dark theme">
<svg class="moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>
<svg class="sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
</button></div></div>
<nav class="tabs"><button class="tab" data-v="live">Live<span class="n">{len(today)}</span></button>
<button class="tab" data-v="past">Past signals<span class="n">{len(past)}</span></button>
<button class="tab" data-v="accuracy">Accuracy</button>
<button class="tab" data-v="stocks">Stocks<span class="n">{len(stocks)}</span></button></nav>
</div></div>

<main class="wrap">
<div class="view" id="v-live">
<div class="idx">{"".join(ix_html) or '<div class="panel empty" style="grid-column:1/-1">No index data yet.</div>'}</div>
<div class="kpis">{kpis}</div>
<section><div class="sh"><h2>Double confirmations</h2><span class="meta">{e(fmt_day(latest_day)) if latest_day else ""}</span></div>
<p class="note">SuperTrend and ADX DI gave the same signal on the same day — the latest signal of each counts.</p>
{table("t-conf", '<th>Stock</th><th>Setup</th><th>SuperTrend</th><th>ADX DI</th><th class="r">At signal</th><th class="r">Now</th><th class="r">Move</th><th class="r">Max gain</th><th class="r hide-sm">Max drawdown</th>',
       conf_rows, "No double confirmations today yet.")}</section>
<section><div class="sh"><h2>Today's signals</h2><div class="seg" id="f-live-dir"><button class="on" data-f="all">All</button>
<button data-f="buy">Buy</button><button data-f="sell">Sell</button></div></div>
{table("t-live", '<th>Time</th><th>Stock</th><th>Signal</th><th class="hide-sm">Strategy</th><th class="r">Price</th><th class="r">Move</th><th class="r">Max gain</th><th class="r hide-sm">Max drawdown</th>',
       live_rows, "No signals yet today — they appear here as each scan finds them.")}</section>
</div>

<div class="view" id="v-past">
<div class="sh"><h2>Past signals</h2><span class="meta" id="c-past"></span></div>
<div class="tb"><select class="sel" id="f-past-day" aria-label="Day">{day_opts or '<option value="">No data yet</option>'}</select>
<div class="seg" id="f-past-dir"><button class="on" data-f="all">All</button><button data-f="buy">Buy</button><button data-f="sell">Sell</button></div>
<div class="seg" id="f-past-strat"><button class="on" data-f="all">Both</button><button data-f="SuperTrend">SuperTrend</button><button data-f="ADX_DI">ADX DI</button></div>
<label class="chk"><input type="checkbox" id="f-past-conf"> Confirmed only</label>
<select class="sel" id="f-past-sym" aria-label="Stock or index">{past_sym_opts}</select></div>
{"".join(day_sums)}
{table("t-past", '<th class="sorted desc" data-k="t" data-asc="0">Time</th><th data-k="sym" data-asc="1">Stock</th><th>Signal</th><th class="hide-md">Strategy</th><th class="r hide-sm">Price</th><th class="r hide-md">Exit</th><th class="r" data-k="mv" data-asc="0">Move</th><th class="r" data-k="mfe" data-asc="0">Max gain</th><th class="r hide-sm" data-k="mae" data-asc="1">Max drawdown</th><th class="r" data-k="bars" data-asc="1">Bars</th>',
       "", "No signals match these filters.")}
<p class="note below">Every signal from the last {HISTORY_DAYS} sessions, replayed bar by bar from 15-minute data. Time = the
15-min candle the signal formed on (labelled by its start, like TradingView); <b>Move</b> = price change in the signal's favour
up to its flip (or to now if still open); <b>Max gain</b> / <b>Max drawdown</b> = furthest price went in the signal's favour /
against it before the flip; <b>Bars</b> = 15-min bars until that strategy flipped (SuperTrend reversed / ADX left that side).
✓ = part of a double confirmation that day.</p>
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
<th class="r hide-sm">Open · now</th></tr></thead><tbody id="acc-body"></tbody></table></div>
<section id="acc-log-wrap" style="display:none;margin-top:26px"><div class="sh"><h2 id="acc-log-title">Signals</h2>
<span class="meta" id="acc-log-count"></span></div>
<div class="panel tw"><table><thead><tr><th>When</th><th>Signal</th><th class="hide-sm">Strategy</th>
<th class="r hide-sm">Price</th><th class="r hide-md">Exit</th><th class="r">Move</th><th class="r">Max gain</th>
<th class="r hide-sm">Max drawdown</th><th class="r">Bars</th></tr></thead><tbody id="acc-log"></tbody></table></div></section>
<p class="note below">A signal <b>wins</b> if price moved in its favour (up after a buy, down after a sell) by the bar where that
strategy flipped. <b>Median bars</b> = 15-min bars until the flip; <b>Median move</b> = typical % move in the signal's favour at
the flip; <b>Max gain</b> / <b>Max drawdown</b> = the furthest price went in the signal's favour / against it
(candle highs and lows) before the flip. Open signals aren't scored — their current move is shown separately. Builds up to {ARCHIVE_KEEP_DAYS} days of history.</p>
</div>

<div class="view" id="v-stocks">
<div class="sh"><h2>All stocks</h2><span class="meta" id="c-stocks"></span></div>
<div class="tb"><input type="search" id="f-stk-q" placeholder="Search stock…" autocomplete="off">
<div class="seg" id="f-stk-view"><button class="on" data-f="all">All</button><button data-f="aligned" title="SuperTrend and ADX DI point the same way">Aligned</button>
<button data-f="near" title="Within {WATCHLIST_PCT:g}% of a SuperTrend flip">Near flip</button><button data-f="long">ST long</button>
<button data-f="short">ST short</button><button data-f="abuy">ADX buy</button><button data-f="asell">ADX sell</button></div></div>
{table("t-stocks", '<th data-k="sym" data-asc="1">Stock</th><th class="r" data-k="px" data-asc="0">Price</th><th class="r" data-k="chg" data-asc="0">Day</th><th data-k="st" data-asc="0">SuperTrend</th><th data-k="adx" data-asc="0">ADX DI</th><th class="r sorted" data-k="dist" data-asc="1">To flip</th><th data-k="mv" data-asc="0">Last signal</th>',
       "".join(stock_rows), "No stocks match.", "dist")}
</div>

<div class="foot">All times IST. Data: Yahoo Finance (~15 min delayed); this page refreshes every 5 minutes.
SELL signals and the accuracy figures are not from the original backtests. Not investment advice.</div>
</main>
<script>{data_js}</script>
<script>const NEAR={WATCHLIST_PCT};{DASHBOARD_JS}</script></body></html>"""
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
ONCE_PACE_SECONDS = 1.0  # --once mode: CI minutes are billed, so scan faster
ONCE_LATEST_RUN = (16, 15)  # --once: allow late/delayed runs to catch the final bar


INDEX_STATE_PREFIX = "__INDEX__"  # state keys for indices, kept apart from stocks


def evaluate(name, df, prev):
    """Run both strategies on one bar series. Returns (new_state, fired) where
    fired lists the signals that changed since `prev` (last cycle's state)."""
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
                 "history": bar_signals(name, df, c, trend, condition),
                 "history_from": str(df.index[np.isin(dates, sorted(set(dates))[-REPLAY_DAYS:])][0])}
    return new_state, fired


HISTORY_DAYS = 10        # trading days selectable in the dashboard's "Past signals" tab
REPLAY_DAYS = 10         # trading days replayed per scan (of the ~15 fetched; the rest is warm-up)
ARCHIVE_KEY = "__ARCHIVE__"
ARCHIVE_KEEP_DAYS = 30   # calendar days of replayed signals kept for the accuracy tracker


def bar_signals(name, df, close, trend, condition):
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

    def entry(i, end, still_open, strategy, label):
        buy = "SELL" not in label
        if end > i:
            hi = (high[i + 1:end + 1].max() / close[i] - 1) * 100.0
            lo = (low[i + 1:end + 1].min() / close[i] - 1) * 100.0
            mfe, mae = (hi, lo) if buy else (-lo, -hi)
        else:
            mfe = mae = 0.0
        return {"time": candle(i), "strategy": strategy, "name": name, "label": label,
                "price": float(close[i]), "bar": str(df.index[i]),
                "bars": int(end - i), "open": still_open,
                "exit_price": None if still_open else float(close[end]),
                "exit_time": None if still_open else candle(end),
                "mfe": round(float(max(mfe, 0.0)), 3), "mae": round(float(min(mae, 0.0)), 3),
                "_i": int(i), "_end": int(end)}

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
        for strat, label in labels:
            end, still_open = flip_index(st_side if strat == "SuperTrend" else adx_side, i)
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
               if h["time"][:10] >= cutoff and not (h["name"] == name and h["bar"] >= since)]
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
            state[key], index_fired = evaluate(name, df, state.get(key, {}))
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

            state[symbol], symbol_fired = evaluate(symbol, df, state.get(symbol, {}))
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

    now_str = f"{datetime.now(IST):%Y-%m-%d %H:%M:%S}"
    log = state.get(SIGNAL_LOG_KEY, [])
    log.extend({"time": now_str, "strategy": strategy, "name": name, "label": label,
                "price": price, "bar": ts} for strategy, name, label, price, ts in fired)
    state[SIGNAL_LOG_KEY] = log[-SIGNAL_LOG_MAX:]
    state[SCAN_META_KEY] = {"time": now_str, "checked": checked, "total": len(universe),
                            "universe": universe}

    save_state(state)
    save_bars_cache()
    write_dashboard(state)
    print(f"[{now_str}] checked {checked}/{len(universe)} stocks "
          f"+ {len(INDEX_TICKERS)} indices, {len(fired)} new signal(s)")

    # double confirmations completed by this scan's signals
    confirmed = [(c["name"], c["dir"], c["price"])
                 for c in find_confluence(state[SIGNAL_LOG_KEY]).get(now_str[:10], [])
                 if c["time"] == now_str]
    notify_signals(fired, confirmed)


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
        scan_once(float(os.environ.get("SCAN_PACE_SECONDS", ONCE_PACE_SECONDS)))
        sys.exit(0)
    if "--test" in sys.argv:
        print("[--test] Ignoring market-hours gate, running one scan against latest available bars.")
        scan_once(float(os.environ.get("SCAN_PACE_SECONDS", ONCE_PACE_SECONDS)))
        sys.exit(0)
    main()
