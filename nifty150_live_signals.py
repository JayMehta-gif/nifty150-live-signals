"""
Nifty150 Live Signal Notifier — SuperTrend + ADX DI/DI-
=========================================================
Standalone script — copy this single file to any machine with internet
access and Python 3.9+, install requirements, and run it. It polls
15-min bars during NSE market hours and pops a desktop notification the
moment either strategy's signal FIRES on any Nifty150 stock, or on the
NIFTY 50 / NIFTY 100 indices themselves (not on every bar — only on the
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
    "NIFTY 100": "^CNX100",
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


def fetch_15m(symbol, yf_ticker=None):
    """Fetch 15-min bars with exponential-backoff retry on rate limiting.
    `yf_ticker` overrides the "<symbol>.NS" lookup (e.g. "^NSEI" for an index)."""
    yf_ticker = yf_ticker or f"{symbol}.NS"
    last_err = None
    for attempt in range(RETRY_MAX):
        try:
            ticker = yf.Ticker(yf_ticker, session=SESSION) if SESSION else yf.Ticker(yf_ticker)
            df = ticker.history(period="10d", interval="15m")
            if df.empty:
                return None
            df = df.rename(columns=str.lower)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(IST)
            df = df.between_time("09:15", "15:30")
            return df.dropna(subset=["open", "close"])
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


def send_telegram(title, message):
    """Send via the Telegram Bot API (stdlib only). Returns True on success."""
    if not TELEGRAM_TOKEN:
        return False
    data = urllib.parse.urlencode({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"{title}\n{message}",
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


def notify(title, message):
    print(f"\n*** {title} ***\n{message}\n")
    send_telegram(title, message)  # before the popup, which blocks for TOAST_DURATION_MS
    if DESKTOP_NOTIFY:
        try:
            show_toast(title, message)
        except Exception as e:
            print(f"[warn] desktop popup failed: {e}")


# ─── Dashboard ──────────────────────────────────────────────────────────────

DASHBOARD_FILE = Path(__file__).parent / "dashboard.html"
SIGNAL_LOG_KEY = "__SIGNALS__"   # recent fired signals, kept in the state file
SIGNAL_LOG_MAX = 300
SCAN_META_KEY = "__SCAN__"       # last scan time / coverage
ADX_LABELS = {1.0: "BUY_STRONG", 0.5: "BUY", 0.0: "—", -0.5: "SELL", -1.0: "SELL_STRONG"}
CONFLUENCE_DAYS = 5              # trading days of "both strategies agree" history shown

DASHBOARD_CSS = """
:root{--bg:#f4f5f7;--card:#fff;--fg:#111827;--muted:#6b7280;--line:#e5e7eb;--soft:#f9fafb;
--up:#047857;--up-bg:#d1fae5;--up-line:#10b981;--dn:#b91c1c;--dn-bg:#fee2e2;--dn-line:#ef4444;
--hl:#fef9c3;--accent:#2563eb;--warn:#92400e;--warn-bg:#fef3c7;--shadow:0 1px 2px rgba(0,0,0,.05)}
@media (prefers-color-scheme:dark){:root{--bg:#0b0d12;--card:#141820;--fg:#e5e7eb;--muted:#9ca3af;
--line:#262b36;--soft:#10141b;--up:#34d399;--up-bg:#0f2e23;--up-line:#10b981;--dn:#f87171;
--dn-bg:#341416;--dn-line:#ef4444;--hl:#2f2a10;--accent:#60a5fa;--warn:#fcd34d;--warn-bg:#2b230b;
--shadow:none}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
font-variant-numeric:tabular-nums;-webkit-font-smoothing:antialiased}
main{max-width:1080px;margin:0 auto;padding:20px 16px 32px}
header{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-end;gap:10px}
h1{font-size:20px;margin:0;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:12px}
.status{display:inline-flex;align-items:center;gap:6px;padding:5px 10px;border-radius:999px;
font-size:12px;font-weight:600;background:var(--card);border:1px solid var(--line)}
.status i{width:8px;height:8px;border-radius:50%;background:var(--up-line)}
.status.live i{animation:pulse 2s infinite}.status.old{color:var(--warn);background:var(--warn-bg);border-color:transparent}
.status.old i{background:var(--warn)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
section{margin-top:26px}
h2{display:flex;align-items:center;gap:8px;font-size:15px;margin:0 0 10px}
h2 .count{font-size:12px;color:var(--muted);font-weight:500}
.hint{color:var(--muted);font-size:12px;margin:-6px 0 10px}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:16px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;box-shadow:var(--shadow)}
.tile .k{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.tile .v{font-size:24px;font-weight:700;margin:2px 0}
.tile .s{font-size:12px;color:var(--muted)}
.bar{display:flex;height:6px;border-radius:3px;overflow:hidden;background:var(--line);margin-top:8px}
.bar .u{background:var(--up-line)}.bar .d{background:var(--dn-line)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--line);
border-radius:12px;padding:12px 14px;box-shadow:var(--shadow)}
.card.up{border-left-color:var(--up-line)}.card.dn{border-left-color:var(--dn-line)}
.card .top{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.card .name{font-weight:700;font-size:15px}
.card .px{font-size:22px;font-weight:700;margin:2px 0 6px}
.card .row{display:flex;justify-content:space-between;align-items:center;font-size:12px;
color:var(--muted);padding:3px 0}
.card .row>span:first-child{white-space:nowrap}.card .row>span:last-child{text-align:right}
.cards.wide{grid-template-columns:repeat(auto-fill,minmax(290px,1fr))}
.card.fresh{background:linear-gradient(var(--hl),var(--hl)) padding-box}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700;
letter-spacing:.02em;white-space:nowrap}
.pill.up{color:var(--up);background:var(--up-bg)}.pill.dn{color:var(--dn);background:var(--dn-bg)}
.pill.big{font-size:12px;padding:3px 10px}
.flat{color:var(--muted)}
.chg.up{color:var(--up)}.chg.dn{color:var(--dn)}
.empty{background:var(--card);border:1px dashed var(--line);border-radius:12px;padding:16px;
color:var(--muted);text-align:center}
details{margin-top:10px}summary{cursor:pointer;color:var(--accent);font-size:13px}
details .day{font-size:12px;color:var(--muted);margin:12px 0 6px;font-weight:600}
.list{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:var(--shadow)}
.sig{display:grid;grid-template-columns:48px minmax(90px,150px) 110px 1fr auto;gap:10px;
align-items:center;padding:9px 14px;border-bottom:1px solid var(--line)}
.sig:last-child{border-bottom:0}.sig .t{color:var(--muted);font-size:12px}
.sig .n{font-weight:600;overflow:hidden;text-overflow:ellipsis}.sig .pill{justify-self:start}.sig .p{text-align:right}
.sig.fresh{background:var(--hl)}
.tools{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
.tools input{flex:1 1 180px;padding:8px 12px;border:1px solid var(--line);border-radius:10px;
background:var(--card);color:var(--fg);font:inherit}
.tools input:focus{outline:2px solid var(--accent);outline-offset:-1px}
.chip{padding:6px 12px;border:1px solid var(--line);border-radius:999px;background:var(--card);
color:var(--fg);font:inherit;font-size:12px;cursor:pointer}
.chip.on{background:var(--fg);color:var(--bg);border-color:var(--fg)}
.tw{overflow:auto;max-height:70vh;background:var(--card);border:1px solid var(--line);
border-radius:12px;box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%}
th,td{padding:8px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);cursor:pointer;
user-select:none;position:sticky;top:0;background:var(--soft);z-index:1}
th[data-k]:hover{color:var(--fg)}th.sorted::after{content:" ↑"}th.sorted.desc::after{content:" ↓"}
.num{text-align:right}
tbody tr:hover td{background:var(--soft)}
tr.fresh td{background:var(--hl)}
tr:last-child td{border-bottom:0}
.foot{color:var(--muted);font-size:12px;margin-top:22px;line-height:1.6}
@media (max-width:760px){.tiles{grid-template-columns:repeat(2,1fr)}}
@media (max-width:600px){main{padding:14px 12px 28px}.hide-sm{display:none}
.sig{grid-template-columns:40px 1fr auto auto}.sig .st{display:none}
th,td{padding:8px 9px}}
"""

DASHBOARD_JS = """
const scanned=new Date(document.body.dataset.scanned),st=document.getElementById('status');
const ageMin=(Date.now()-scanned)/60000;
if(!isNaN(ageMin)){if(ageMin<=25){st.classList.add('live');st.lastChild.textContent=' Live · updated '+
 (ageMin<1?'just now':Math.round(ageMin)+' min ago');}
 else{st.classList.add('old');st.lastChild.textContent=' Last scan '+(ageMin<120?Math.round(ageMin)+
 ' min':ageMin<2880?Math.round(ageMin/60)+' h':Math.round(ageMin/1440)+' days')+' ago';
 st.title='Normal outside market hours (Mon–Fri 09:15–15:30 IST); otherwise a scheduled run may be late.';}}
function chips(group,fn){const cs=[...document.querySelectorAll('[data-group='+group+'] .chip')];
 cs.forEach(c=>c.onclick=()=>{cs.forEach(x=>x.classList.remove('on'));c.classList.add('on');fn(c.dataset.f);});}
const sigs=[...document.querySelectorAll('#signals .sig')];
chips('sig',f=>sigs.forEach(r=>r.style.display=(f==='all'||r.dataset.dir===f)?'':'none'));
const rows=[...document.querySelectorAll('#stocks tbody tr')];let filter='all',query='';
function apply(){let n=0;for(const r of rows){const d=r.dataset;
 const ok=(filter==='all'||(filter==='long'&&d.st==='1')||(filter==='short'&&d.st==='-1')||
 (filter==='abuy'&&+d.adx>0)||(filter==='asell'&&+d.adx<0)||(filter==='near'&&+d.dist<=NEAR)||
 (filter==='both'&&((d.st==='1'&&+d.adx>0)||(d.st==='-1'&&+d.adx<0))))&&d.sym.includes(query);
 r.style.display=ok?'':'none';n+=ok;}document.getElementById('shown').textContent=n+' shown';}
chips('tbl',f=>{filter=f;apply();});
document.getElementById('q').oninput=e=>{query=e.target.value.trim().toUpperCase();apply();};
let sortKey='dist',asc=true;const tb=document.querySelector('#stocks tbody');
document.querySelectorAll('#stocks th[data-k]').forEach(th=>th.onclick=()=>{
 const k=th.dataset.k;asc=(k===sortKey)?!asc:(k==='sym'||k==='dist');sortKey=k;
 document.querySelectorAll('#stocks th').forEach(x=>x.classList.remove('sorted','desc'));
 th.classList.add('sorted');if(!asc)th.classList.add('desc');
 rows.sort((a,b)=>{const x=a.dataset[k],y=b.dataset[k];
  const v=(k==='sym')?x.localeCompare(y):(+x)-(+y);return asc?v:-v;});
 rows.forEach(r=>tb.appendChild(r));});
apply();
"""


def _fmt_price(p):
    return f"{p:,.2f}" if isinstance(p, (int, float)) else "—"


def _fmt_chg(chg):
    if chg is None:
        return '<span class="flat">—</span>'
    cls = "up" if chg > 0 else ("dn" if chg < 0 else "flat")
    return f'<span class="chg {cls}">{chg:+.2f}%</span>'


def _st_pill(st):
    if st == 1:
        return '<span class="pill up">LONG</span>'
    if st == -1:
        return '<span class="pill dn">SHORT</span>'
    return '<span class="flat">—</span>'


def _adx_pill(adx):
    label = ADX_LABELS.get(adx, "—")
    cls = "up" if adx and adx > 0 else ("dn" if adx and adx < 0 else "")
    return f'<span class="pill {cls}">{label}</span>' if cls else '<span class="flat">—</span>'


def _direction(label):
    return "sell" if "SELL" in label else "buy"


def _signal_pill(label, big=False):
    return (f'<span class="pill {"dn" if "SELL" in label else "up"}{" big" if big else ""}">'
            f'{html_lib.escape(label)}</span>')


def find_confluence(log):
    """Stocks where SuperTrend and ADX_DI both fired the same direction on the
    same trading day. Uses each strategy's latest signal that day, so a buy
    later reversed by a sell doesn't count. Returns {day: [entry, ...]},
    newest day first; each entry's "time" is when the second signal landed."""
    latest = {}  # (day, name) -> {strategy: signal}
    for s in log:
        latest.setdefault((s["time"][:10], s["name"]), {})[s["strategy"]] = s
    by_day = {}
    for (day, name), strat in latest.items():
        st, adx = strat.get("SuperTrend"), strat.get("ADX_DI")
        if st and adx and _direction(st["label"]) == _direction(adx["label"]):
            last = max(st, adx, key=lambda s: s["time"])
            by_day.setdefault(day, []).append({
                "name": name, "dir": _direction(st["label"]), "time": last["time"],
                "price": last["price"], "st": st, "adx": adx})
    for entries in by_day.values():
        entries.sort(key=lambda x: x["time"], reverse=True)
    return dict(sorted(by_day.items(), reverse=True)[:CONFLUENCE_DAYS])


def _confluence_card(c, state, scan_time):
    e = html_lib.escape
    info = state.get(c["name"]) or state.get(INDEX_STATE_PREFIX + c["name"]) or {}
    cls = "up" if c["dir"] == "buy" else "dn"
    fresh = " fresh" if c["time"] == scan_time else ""
    return (f'<div class="card {cls}{fresh}"><div class="top"><span class="name">{e(c["name"])}</span>'
            f'<span class="pill {cls} big">BOTH {c["dir"].upper()}</span></div>'
            f'<div class="px">{_fmt_price(info.get("last_close", c["price"]))}'
            f' <span style="font-size:13px">{_fmt_chg(info.get("day_chg_pct"))}</span></div>'
            f'<div class="row"><span>SuperTrend</span><span>{_signal_pill(c["st"]["label"])} '
            f'{e(c["st"]["time"][11:16])} @ {_fmt_price(c["st"]["price"])}</span></div>'
            f'<div class="row"><span>ADX DI</span><span>{_signal_pill(c["adx"]["label"])} '
            f'{e(c["adx"]["time"][11:16])} @ {_fmt_price(c["adx"]["price"])}</span></div></div>')


def write_dashboard(state):
    """Render dashboard.html from the state file — a static page, no server:
    summary tiles, index cards, stocks where both strategies agree, the
    day's signal timeline, and a filterable/sortable table of every stock."""
    e = html_lib.escape
    meta = state.get(SCAN_META_KEY, {})
    scan_time = meta.get("time")
    universe = meta.get("universe") or NIFTY150
    stocks = {s: state[s] for s in universe if s in state}
    log = state.get(SIGNAL_LOG_KEY, [])

    # summary tiles
    latest_day = log[-1]["time"][:10] if log else None
    today = [s for s in log if s["time"][:10] == latest_day][::-1]
    n_buy = sum(_direction(s["label"]) == "buy" for s in today)
    st_long = sum(v.get("st_trend") == 1 for v in stocks.values())
    st_short = sum(v.get("st_trend") == -1 for v in stocks.values())
    adx_buy = sum((v.get("adx_cond") or 0) > 0 for v in stocks.values())
    adx_sell = sum((v.get("adx_cond") or 0) < 0 for v in stocks.values())
    confluence = find_confluence(log)
    conf_today = confluence.get(latest_day, []) if latest_day else []

    def bar(u, d):
        tot = (u + d) or 1
        return (f'<div class="bar"><span class="u" style="width:{u / tot * 100:.1f}%"></span>'
                f'<span class="d" style="width:{d / tot * 100:.1f}%"></span></div>')

    tiles = (
        f'<div class="tile"><div class="k">Signals {"today" if latest_day else ""}</div>'
        f'<div class="v">{len(today)}</div><div class="s">{n_buy} buy · {len(today) - n_buy} sell</div>'
        f'{bar(n_buy, len(today) - n_buy)}</div>'
        f'<div class="tile"><div class="k">Both agree</div><div class="v">{len(conf_today)}</div>'
        f'<div class="s">{sum(c["dir"] == "buy" for c in conf_today)} buy · '
        f'{sum(c["dir"] == "sell" for c in conf_today)} sell</div></div>'
        f'<div class="tile"><div class="k">SuperTrend</div><div class="v">{st_long}<span class="s"> / '
        f'{st_short}</span></div><div class="s">long / short</div>{bar(st_long, st_short)}</div>'
        f'<div class="tile"><div class="k">ADX DI</div><div class="v">{adx_buy}<span class="s"> / '
        f'{adx_sell}</span></div><div class="s">buy / sell state</div>{bar(adx_buy, adx_sell)}</div>')

    # indices
    cards = []
    for name in INDEX_TICKERS:
        info = state.get(INDEX_STATE_PREFIX + name)
        if not info:
            continue
        dist = info.get("distance_pct")
        cls = "up" if info.get("st_trend") == 1 else ("dn" if info.get("st_trend") == -1 else "")
        cards.append(
            f'<div class="card {cls}"><div class="top"><span class="name">{e(name)}</span>'
            f'{_fmt_chg(info.get("day_chg_pct"))}</div>'
            f'<div class="px">{_fmt_price(info.get("last_close"))}</div>'
            f'<div class="row"><span>SuperTrend</span>{_st_pill(info.get("st_trend"))}</div>'
            f'<div class="row"><span>ADX DI</span>{_adx_pill(info.get("adx_cond"))}</div>'
            f'<div class="row"><span>To flip</span><span>'
            f'{f"{dist:.2f}%" if dist is not None else "—"}</span></div></div>')

    # both strategies agree
    if conf_today:
        conf_html = '<div class="cards wide">' + "".join(
            _confluence_card(c, state, scan_time) for c in conf_today) + "</div>"
    else:
        conf_html = ('<div class="empty">No stock has fired the same direction on both '
                     f'strategies {"on " + e(latest_day) if latest_day else "yet"}.</div>')
    earlier = [(d, cs) for d, cs in confluence.items() if d != latest_day]
    if earlier:
        conf_html += (f'<details><summary>Earlier days ({sum(len(cs) for _, cs in earlier)})</summary>'
                      + "".join(f'<div class="day">{e(d)}</div><div class="cards wide">'
                                + "".join(_confluence_card(c, state, scan_time) for c in cs)
                                + "</div>" for d, cs in earlier) + "</details>")

    # signal timeline
    sig_rows = "".join(
        f'<div class="sig{" fresh" if s["time"] == scan_time else ""}" data-dir="{_direction(s["label"])}">'
        f'<span class="t">{e(s["time"][11:16])}</span><span class="n">{e(s["name"])}</span>'
        f'{_signal_pill(s["label"])}<span class="flat st">{e(s["strategy"])}</span>'
        f'<span class="p">{_fmt_price(s["price"])}</span></div>' for s in today)

    # stock table, nearest-to-flip first
    def nearest_first(sym):
        dist = stocks[sym].get("distance_pct")
        return (dist is None, dist or 0.0)

    fresh = {s["name"] for s in log if s["time"] == scan_time}
    trs = []
    for sym in sorted(stocks, key=nearest_first):
        info = stocks[sym]
        st, adx = info.get("st_trend"), info.get("adx_cond") or 0.0
        dist, chg = info.get("distance_pct"), info.get("day_chg_pct")
        trs.append(
            f'<tr{" class=fresh" if sym in fresh else ""} data-sym="{e(sym)}" data-st="{st}" '
            f'data-adx="{adx}" data-dist="{dist if dist is not None else 999}" '
            f'data-chg="{chg if chg is not None else 0}" data-px="{info.get("last_close") or 0}">'
            f'<td><b>{e(sym)}</b></td><td class="num">{_fmt_chg(chg)}</td>'
            f'<td>{_st_pill(st)}</td><td>{_adx_pill(adx)}</td>'
            f'<td class="num">{_fmt_price(info.get("last_close"))}</td>'
            f'<td class="num">{f"{dist:.2f}%" if dist is not None else "—"}</td>'
            f'<td class="hide-sm flat">{e(str(info.get("last_ts", ""))[5:16])}</td></tr>')

    scanned_iso = f"{scan_time.replace(' ', 'T')}+05:30" if scan_time else ""
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="300">
<meta name="color-scheme" content="light dark">
<title>Nifty150 Live Signals</title><style>{DASHBOARD_CSS}</style></head>
<body data-scanned="{e(scanned_iso)}"><main>
<header><div><h1>Nifty150 Live Signals</h1>
<div class="sub">SuperTrend ({ATR_PERIOD}×{MULTIPLIER:g}) + ADX DI · Nifty 100 + Midcap 50 ·
{meta.get("checked", "—")}/{meta.get("total", "—")} stocks · last scan {e(scan_time or "—")} IST</div></div>
<span class="status" id="status"><i></i> Waiting for first scan</span></header>
<div class="tiles">{tiles}</div>
<section><h2>Both strategies agree <span class="count">{e(latest_day or "")}</span></h2>
<p class="hint">SuperTrend and ADX DI both fired the same direction on the same day
(latest signal of each counts).</p>{conf_html}</section>
<section><h2>Indices</h2>
<div class="cards">{"".join(cards) or '<div class="empty">No index data yet.</div>'}</div></section>
<section id="signals"><h2>Signals <span class="count">{e(latest_day or "")}</span></h2>
<div class="tools" data-group="sig"><button class="chip on" data-f="all">All</button>
<button class="chip" data-f="buy">Buy</button><button class="chip" data-f="sell">Sell</button></div>
<div class="list">{sig_rows or '<div class="empty" style="border:0">No signals yet.</div>'}</div></section>
<section><h2>All stocks <span class="count" id="shown"></span></h2>
<div class="tools" data-group="tbl"><input id="q" placeholder="Search symbol…" autocomplete="off">
<button class="chip on" data-f="all">All</button><button class="chip" data-f="both">Both agree now</button>
<button class="chip" data-f="near">Near flip</button>
<button class="chip" data-f="long">ST long</button><button class="chip" data-f="short">ST short</button>
<button class="chip" data-f="abuy">ADX buy</button><button class="chip" data-f="asell">ADX sell</button></div>
<div class="tw"><table id="stocks"><thead><tr>
<th data-k="sym">Stock</th><th class="num" data-k="chg">Chg</th><th data-k="st">SuperTrend</th>
<th data-k="adx">ADX DI</th><th class="num" data-k="px">Price</th>
<th class="num sorted" data-k="dist">To flip</th><th class="hide-sm">Bar (IST)</th></tr></thead>
<tbody>{"".join(trs)}</tbody></table></div></section>
<div class="foot">Highlighted = fired in the latest scan. "To flip" = distance from price to the
SuperTrend band it must cross. Data: Yahoo Finance, ~15 min delayed; page refreshes every 5 min.
SELL signals were not part of the backtests. Not investment advice.</div>
</main><script>const NEAR={WATCHLIST_PCT};{DASHBOARD_JS}</script></body></html>"""
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
    day_chg_pct = ((last_close / prev_session[-1] - 1) * 100.0
                   if len(prev_session) and prev_session[-1] else None)

    new_state = {"st_trend": st_trend, "adx_cond": adx_cond, "last_ts": last_ts,
                 "last_close": last_close, "distance_pct": distance_pct,
                 "day_chg_pct": day_chg_pct}
    return new_state, fired


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
    """Same strategies on the index series themselves (NIFTY 50 / NIFTY 100)."""
    fired = []
    for name, yf_ticker in INDEX_TICKERS.items():
        key = INDEX_STATE_PREFIX + name
        try:
            df = fetch_15m(name, yf_ticker)
            if df is None or len(df) < 60:
                continue
            state[key], index_fired = evaluate(name, df, state.get(key, {}))
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
    write_dashboard(state)
    print(f"[{now_str}] checked {checked}/{len(universe)} stocks "
          f"+ {len(INDEX_TICKERS)} indices, {len(fired)} new signal(s)")

    for strategy, symbol, label, price, ts in fired:
        notify(f"{strategy} {label} — {symbol}",
               f"{symbol} @ {price:.2f}  ({strategy}, bar {ts})")


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
        ok = send_telegram("Nifty150 notifier — test", "Telegram alerts are working.")
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
