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

DASHBOARD_CSS = """
:root{--bg:#f6f7f9;--card:#fff;--fg:#14171c;--muted:#626a75;--line:#e3e6ea;
--up:#0f8a4a;--up-bg:#e3f5ea;--dn:#c8323a;--dn-bg:#fbe7e8;--hl:#fff6d6;--accent:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#0e1116;--card:#161a21;--fg:#e7e9ec;
--muted:#9aa3ae;--line:#272c35;--up:#4ade80;--up-bg:#12301f;--dn:#f87171;--dn-bg:#3a1719;
--hl:#3a3212;--accent:#60a5fa}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 -apple-system,"Segoe UI",Roboto,Arial,sans-serif}
main{max-width:1000px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:22px 0 8px}
.meta{color:var(--muted);font-size:12px;margin-top:4px}
.stale{display:none;margin-top:8px;padding:8px 10px;border-radius:8px;background:var(--hl);font-size:12px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.card .name{font-weight:600}.card .px{font-size:20px;font-weight:600;margin:4px 0}
.pill{display:inline-block;padding:1px 7px;border-radius:999px;font-size:11px;font-weight:600;white-space:nowrap}
.up{color:var(--up);background:var(--up-bg)}.dn{color:var(--dn);background:var(--dn-bg)}
.flat{color:var(--muted)}
.sig{display:flex;gap:10px;align-items:baseline;padding:8px 12px;border-bottom:1px solid var(--line)}
.sig:last-child{border-bottom:0}.sig .t{color:var(--muted);font-size:12px;min-width:44px}
.sig .n{font-weight:600}.sig .p{margin-left:auto;font-variant-numeric:tabular-nums}
.list{background:var(--card);border:1px solid var(--line);border-radius:10px}
.empty{padding:12px;color:var(--muted)}
.tools{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}
.tools input{flex:1 1 160px;padding:7px 10px;border:1px solid var(--line);border-radius:8px;
background:var(--card);color:var(--fg);font:inherit}
.chip{padding:6px 10px;border:1px solid var(--line);border-radius:999px;background:var(--card);
color:var(--fg);font:inherit;font-size:12px;cursor:pointer}
.chip.on{border-color:var(--accent);color:var(--accent)}
.tw{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{font-size:12px;color:var(--muted);cursor:pointer;user-select:none;position:sticky;top:0;background:var(--card)}
th.num,td.num{text-align:right}
tr.fresh td{background:var(--hl)}
tr:last-child td{border-bottom:0}
.foot{color:var(--muted);font-size:12px;margin:18px 0 8px}
@media (max-width:600px){.hide-sm{display:none}main{padding:12px}}
"""

DASHBOARD_JS = """
const scanned=new Date(document.body.dataset.scanned);
const ageMin=(Date.now()-scanned)/60000;
if(ageMin>25){const s=document.getElementById('stale');s.style.display='block';
 s.textContent='Last scan was '+(ageMin<120?Math.round(ageMin)+' min':Math.round(ageMin/60)+' h')+
 ' ago — normal outside market hours (Mon–Fri 09:15–15:30 IST); otherwise the scheduled run may be late.';}
const rows=[...document.querySelectorAll('#stocks tbody tr')];
let filter='all',query='';
function apply(){for(const r of rows){const d=r.dataset;
 const ok=(filter==='all'||(filter==='long'&&d.st==='1')||(filter==='short'&&d.st==='-1')||
 (filter==='abuy'&&+d.adx>0)||(filter==='asell'&&+d.adx<0)||(filter==='near'&&+d.dist<=NEAR))
 &&d.sym.includes(query);r.style.display=ok?'':'none';}}
document.querySelectorAll('.chip').forEach(c=>c.onclick=()=>{
 document.querySelectorAll('.chip').forEach(x=>x.classList.remove('on'));c.classList.add('on');
 filter=c.dataset.f;apply();});
document.getElementById('q').oninput=e=>{query=e.target.value.trim().toUpperCase();apply();};
let sortKey='dist',asc=true;
document.querySelectorAll('#stocks th').forEach(th=>th.onclick=()=>{
 const k=th.dataset.k;if(!k)return;asc=(k===sortKey)?!asc:true;sortKey=k;
 const tb=document.querySelector('#stocks tbody');
 rows.sort((a,b)=>{const x=a.dataset[k],y=b.dataset[k];
  const v=(k==='sym')?x.localeCompare(y):(+x)-(+y);return asc?v:-v;});
 rows.forEach(r=>tb.appendChild(r));});
"""


def _fmt_price(p):
    return f"{p:,.2f}" if isinstance(p, (int, float)) else "—"


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


def _signal_pill(label):
    return f'<span class="pill {"dn" if "SELL" in label else "up"}">{html_lib.escape(label)}</span>'


def write_dashboard(state):
    """Render dashboard.html from the state file: index cards, today's signals,
    and a filterable/sortable table of every stock. Static page, no server."""
    e = html_lib.escape
    meta = state.get(SCAN_META_KEY, {})
    scan_time = meta.get("time")
    universe = meta.get("universe") or NIFTY150

    cards = []
    for name in INDEX_TICKERS:
        info = state.get(INDEX_STATE_PREFIX + name)
        if not info:
            continue
        dist = info.get("distance_pct")
        cards.append(
            f'<div class="card"><div class="name">{e(name)}</div>'
            f'<div class="px">{_fmt_price(info.get("last_close"))}</div>'
            f'SuperTrend {_st_pill(info.get("st_trend"))} &nbsp; ADX {_adx_pill(info.get("adx_cond"))}'
            f'<div class="meta">{f"{dist:.2f}% to flip · " if dist is not None else ""}'
            f'bar {e(str(info.get("last_ts", ""))[5:16])}</div></div>')

    log = state.get(SIGNAL_LOG_KEY, [])
    latest_day = log[-1]["time"][:10] if log else None
    today = [s for s in log if s["time"][:10] == latest_day][::-1]
    fresh = {s["name"] for s in log if s["time"] == scan_time}
    sig_rows = "".join(
        f'<div class="sig"><span class="t">{e(s["time"][11:16])}</span>'
        f'<span class="n">{e(s["name"])}</span>{_signal_pill(s["label"])}'
        f'<span class="flat">{e(s["strategy"])}</span>'
        f'<span class="p">{_fmt_price(s["price"])}</span></div>' for s in today)
    sig_title = f"Signals — {latest_day}" if latest_day else "Signals"

    def nearest_first(sym):
        dist = state.get(sym, {}).get("distance_pct")
        return (dist is None, dist or 0.0)

    trs = []
    for sym in sorted(universe, key=nearest_first):
        info = state.get(sym)
        if not info:
            continue
        st, adx, dist = info.get("st_trend"), info.get("adx_cond", 0.0), info.get("distance_pct")
        trs.append(
            f'<tr{" class=fresh" if sym in fresh else ""} data-sym="{e(sym)}" data-st="{st}" '
            f'data-adx="{adx}" data-dist="{dist if dist is not None else 999}" '
            f'data-px="{info.get("last_close") or 0}">'
            f'<td><b>{e(sym)}</b></td><td>{_st_pill(st)}</td><td>{_adx_pill(adx)}</td>'
            f'<td class="num">{_fmt_price(info.get("last_close"))}</td>'
            f'<td class="num">{f"{dist:.2f}%" if dist is not None else "—"}</td>'
            f'<td class="hide-sm flat">{e(str(info.get("last_ts", ""))[5:16])}</td></tr>')

    scanned_iso = f"{scan_time.replace(' ', 'T')}+05:30" if scan_time else ""
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="300">
<title>Nifty150 Live Signals</title><style>{DASHBOARD_CSS}</style></head>
<body data-scanned="{e(scanned_iso)}"><main>
<h1>Nifty150 Live Signals</h1>
<div class="meta">SuperTrend({ATR_PERIOD}×{MULTIPLIER:g}) + ADX DI · last scan {e(scan_time or "—")} IST ·
{meta.get("checked", "—")}/{meta.get("total", "—")} stocks · updates every 15 min in market hours</div>
<div class="stale" id="stale"></div>
<h2>Indices</h2><div class="cards">{"".join(cards) or '<div class="empty">No index data yet.</div>'}</div>
<h2>{e(sig_title)}</h2>
<div class="list">{sig_rows or '<div class="empty">No signals yet.</div>'}</div>
<h2>Stocks</h2>
<div class="tools"><input id="q" placeholder="Search symbol" autocomplete="off">
<button class="chip on" data-f="all">All</button><button class="chip" data-f="near">Near flip</button>
<button class="chip" data-f="long">ST long</button><button class="chip" data-f="short">ST short</button>
<button class="chip" data-f="abuy">ADX buy</button><button class="chip" data-f="asell">ADX sell</button></div>
<div class="tw"><table id="stocks"><thead><tr>
<th data-k="sym">Stock</th><th data-k="st">SuperTrend</th><th data-k="adx">ADX DI</th>
<th class="num" data-k="px">Price</th><th class="num" data-k="dist">To flip</th>
<th class="hide-sm">Bar (IST)</th></tr></thead>
<tbody>{"".join(trs)}</tbody></table></div>
<div class="foot">Sorted nearest-to-flip first; tap a column to sort. Highlighted rows fired in the
latest scan. Data: Yahoo Finance, ~15 min delayed. SELL signals were not part of the backtests.
Not investment advice.</div>
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

    new_state = {"st_trend": st_trend, "adx_cond": adx_cond, "last_ts": last_ts,
                 "last_close": last_close, "distance_pct": distance_pct}
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
