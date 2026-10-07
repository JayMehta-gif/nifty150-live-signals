"""
Nifty150 Live Signal Notifier — SuperTrend + ADX DI/DI-
=========================================================
Standalone script — copy this single file to any machine with internet
access and Python 3.9+, install requirements, and run it. It polls
15-min bars during NSE market hours and pops a desktop notification the
moment either strategy's signal FIRES on any Nifty150 stock (not on every
bar — only on the state change).

Strategies (locked configs from prior backtests):
  - SuperTrend ATR(14) x 3.0, long-only         -> BUY on trend flip -1 -> 1
  - ADX DI+/DI- [Gu5], long-only, skip-open-bar  -> BUY on condition -> 1 or 0.5

Install:
    pip install yfinance pandas numpy numba curl_cffi

Run (leave running in a terminal / as a background service):
    python nifty150_live_signals.py

Or a single scan, for a scheduler such as GitHub Actions
(.github/workflows/live-signals.yml runs it every 15 min in market hours):
    python nifty150_live_signals.py --once

Notes:
  - Data source is yfinance (free, ~15-min delayed) — not for split-second
    execution, fine for swing/intraday signal alerts.
  - Market hours gate assumes IST (Asia/Kolkata) and NSE's 09:15-15:30
    session, Mon-Fri. It sleeps outside those hours and wakes near open.
  - State is kept in signal_state.json next to this script, so restarting
    the script does not re-fire already-seen signals.
  - Nifty150 list below (Nifty 100 + Nifty Midcap 50, from NSE's official
    constituent CSVs as of 2026-10-07) is a point-in-time snapshot; NSE
    reshuffles index constituents roughly every 6 months — update the
    NIFTY150 list periodically from the official NSE index sheets
    (nsearchives.nseindia.com/content/indices/ind_nifty100list.csv and
    ind_niftymidcap50list.csv).
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
"""
import warnings; warnings.filterwarnings("ignore")

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
    return trend


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


def fetch_15m(symbol):
    """Fetch 15-min bars with exponential-backoff retry on rate limiting."""
    last_err = None
    for attempt in range(RETRY_MAX):
        try:
            ticker = yf.Ticker(f"{symbol}.NS", session=SESSION) if SESSION else yf.Ticker(f"{symbol}.NS")
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
    tk.Label(frame, text=title, bg="#1e1e1e", fg="#4ade80",
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


def scan_once(per_symbol_delay=None):
    """Scan the universe with requests paced across the poll window, so the
    150-symbol sweep never bursts — each symbol fetch is spaced ~POLL_SECONDS
    * PACE_FRACTION / len(NIFTY150) apart (with jitter), and a circuit
    breaker pauses the whole scan if rate limiting keeps recurring."""
    state = load_state()
    fired = []
    checked = 0
    consecutive_rate_limits = 0

    if per_symbol_delay is None:
        per_symbol_delay = (POLL_SECONDS * PACE_FRACTION) / len(NIFTY150)

    for idx, symbol in enumerate(NIFTY150):
        try:
            df = fetch_15m(symbol)
            consecutive_rate_limits = 0

            if df is None or len(df) < 60:
                continue

            h = df["high"].to_numpy(dtype=np.float64)
            l = df["low"].to_numpy(dtype=np.float64)
            c = df["close"].to_numpy(dtype=np.float64)
            last_ts = str(df.index[-1])
            last_close = float(c[-1])

            atr = _atr_rma(h, l, c, ATR_PERIOD)
            trend = _supertrend(c, h, l, atr, MULTIPLIER)
            st_trend = int(trend[-1])

            is_open = np.array([t.hour == 9 and t.minute == 15 for t in df.index.time])
            di_plus, di_minus, sig = _calc_di_sig(h, l, c, DI_LEN, SIG_LEN)
            condition = _build_condition(di_plus, di_minus, sig, HL_RANGE, HL_TREND, is_open)
            adx_cond = float(condition[-1])

            checked += 1
            prev = state.get(symbol, {})
            prev_st = prev.get("st_trend")
            prev_adx = prev.get("adx_cond")

            if prev_st is not None and prev_st == -1 and st_trend == 1:
                fired.append(("SuperTrend", symbol, "BUY", last_close, last_ts))
            if prev_adx is not None and prev_adx not in (1.0, 0.5) and adx_cond in (1.0, 0.5):
                label = "BUY_STRONG" if adx_cond == 1.0 else "BUY"
                fired.append(("ADX_DI", symbol, label, last_close, last_ts))

            state[symbol] = {"st_trend": st_trend, "adx_cond": adx_cond,
                              "last_ts": last_ts, "last_close": last_close}

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

        if idx < len(NIFTY150) - 1:
            time.sleep(per_symbol_delay * random.uniform(0.7, 1.3))

    save_state(state)
    print(f"[{datetime.now(IST):%Y-%m-%d %H:%M:%S}] checked {checked}/{len(NIFTY150)} stocks, "
          f"{len(fired)} new signal(s)")

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
            sys.exit(0)
        scan_once(float(os.environ.get("SCAN_PACE_SECONDS", ONCE_PACE_SECONDS)))
        sys.exit(0)
    main()
