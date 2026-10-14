"""
Backtester for the live-signal strategies, with realistic Indian intraday costs.
================================================================================
Answers one question honestly: would a fixed, rules-based use of these
signals have made money after costs, on data it was not designed on?

Two tests:
  1. 15-min test  — the live system's own timeframe. Yahoo only serves ~60 days
                    of 15-min bars, so this covers ~40 sessions.
  2. 1-hour test  — the same rules on hourly candles over ~2 years (bull, bear
                    and sideways phases), as a robustness check.

The NARROW STRATEGY (fixed before testing — see NARROW below):
  * universe   : the 150 stocks, intraday cash (MIS), long and short
  * setup      : double confirmation — SuperTrend and ADX DI both signal the same
                 direction on the same day, the first still active when the
                 second fires — AND both stage gates agree (2-hour + daily
                 SuperTrend; for the 1-hour test, daily only)
  * timing     : signal candle must close by 11:00 AM; entry at the NEXT
                 candle's open (no look-ahead)
  * exit       : +1% target or -1% stop (same-candle touch of both = stop),
                 else at the 3:00 PM candle's close (3:15 PM), before square-off
  * size       : one trade per stock per day, Rs 1,00,000 notional per trade
  * costs      : brokerage Rs 20/order (or 0.03%), STT 0.025% on sell, NSE txn
                 0.00297%, SEBI fee, GST 18%, stamp 0.003% on buy, plus 0.02%
                 slippage per side  (~0.12% per round trip in total)

Out-of-sample discipline: the strategy was shaped by looking at sessions from
SEEN_FROM onwards; only sessions BEFORE that count as the unseen test.

Usage:
    python backtest.py                 # both tests, writes backtest_report.md + backtest_trades.csv
    python backtest.py --skip-hourly   # 15-min test only
"""
import warnings; warnings.filterwarnings("ignore")

import argparse
import statistics as S
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

import nifty150_live_signals as live

SEEN_FROM = "2026-09-25"     # first session used while designing the strategy
WARMUP_SESSIONS = 5          # indicators need history before the first trade
NOTIONAL = 100_000           # Rs per trade
SLIPPAGE = 0.0002            # per side
OUT = Path(__file__).parent


# ─── strategy definition ────────────────────────────────────────────────────

@dataclass(frozen=True)
class Rules:
    name: str
    setup: str = "confirmed"        # "confirmed" | "raw" (every single-strategy signal)
    gates: str = "both"             # "both" | "2h" | "1d" | "none"
    entry_by: str = "11:00"         # signal candle must close by this time ("" = any time)
    side: str = "both"              # "both" | "long" | "short"
    target: float = 1.0             # % (0 = none)
    stop: float = 1.0               # % (0 = none)


NARROW = Rules("NARROW (pre-registered)")
VARIANTS = [
    replace(NARROW, name="… long only", side="long"),
    replace(NARROW, name="… short only", side="short"),
    replace(NARROW, name="… no time filter", entry_by=""),
    replace(NARROW, name="… no stage gates", gates="none"),
    replace(NARROW, name="… 2H gate only", gates="2h"),
    replace(NARROW, name="… target 0.5% / stop 0.5%", target=0.5, stop=0.5),
    replace(NARROW, name="… target 1.5% / stop 1%", target=1.5),
    replace(NARROW, name="… no target/stop (exit 3:15 PM)", target=0, stop=0),
    Rules("BASELINE: every raw signal", setup="raw", gates="none", entry_by=""),
]


# ─── costs ──────────────────────────────────────────────────────────────────

def round_trip_cost_pct(notional=NOTIONAL):
    """Indian intraday equity (MIS) round trip, as % of notional."""
    brokerage = 2 * min(20.0, 0.0003 * notional)
    stt = 0.00025 * notional                      # sell side
    txn = 0.0000297 * 2 * notional                # NSE transaction charge
    sebi = 0.000001 * 2 * notional
    gst = 0.18 * (brokerage + txn + sebi)
    stamp = 0.00003 * notional                    # buy side
    slip = SLIPPAGE * 2 * notional
    return (brokerage + stt + txn + sebi + gst + stamp + slip) / notional * 100


COST_PCT = round_trip_cost_pct()


# ─── data ───────────────────────────────────────────────────────────────────

def download(yf_ticker, period, interval):
    tk = live.yf.Ticker(yf_ticker, session=live.SESSION) if live.SESSION else live.yf.Ticker(yf_ticker)
    for attempt in range(4):
        try:
            df = tk.history(period=period, interval=interval)
            if df.empty:
                return None
            df = df.rename(columns=str.lower)
            df.index = df.index.tz_convert(live.IST)
            return df.between_time("09:15", "15:30").dropna(subset=["open", "close"])[
                ["open", "high", "low", "close"]]
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                print(f"[warn] {yf_ticker} {interval}: {e}")
                return None
            time.sleep(5 * (attempt + 1))


def htf_series(h1, minutes):
    """Session-aligned higher-timeframe candles from hourly bars, with the time
    each candle closed and its SuperTrend direction (minutes=None -> daily)."""
    key = live._bucket(h1.index, minutes) if minutes else h1.index.normalize() + pd.Timedelta(minutes=555)
    bars, g = live._ohlc(h1, key)
    last_start = g.apply(lambda x: x.index[-1])
    end = last_start.dt.normalize() + pd.Timedelta(hours=15, minutes=30)
    closes = (last_start + pd.Timedelta(hours=1)).where(last_start + pd.Timedelta(hours=1) <= end, end)
    h, l, c = (bars[k].to_numpy(np.float64) for k in ("high", "low", "close"))
    tr, _, _ = live._supertrend(c, h, l, live._atr_rma(h, l, c, live.ATR_PERIOD), live.MULTIPLIER)
    return closes.astype("int64").to_numpy(), tr.astype(np.int8)


def trend_at(series, t_ns):
    closes, tr = series
    k = np.searchsorted(closes, t_ns, side="right") - 1
    return np.where(k >= 0, tr[np.clip(k, 0, None)], 0)


# ─── signal + trade engine ──────────────────────────────────────────────────

def prepare(df, bar_minutes, h1):
    """Indicator states per bar, and the 2H / 1D trend known at each bar's close."""
    h, l, c = (df[k].to_numpy(np.float64) for k in ("high", "low", "close"))
    trend, _, _ = live._supertrend(c, h, l, live._atr_rma(h, l, c, live.ATR_PERIOD), live.MULTIPLIER)
    is_open = np.array([t.hour == 9 and t.minute == 15 for t in df.index.time])
    dp, dm, sig = live._calc_di_sig(h, l, c, live.DI_LEN, live.SIG_LEN)
    cond = live._build_condition(dp, dm, sig, live.HL_RANGE, live.HL_TREND, is_open)
    close_ns = (df.index + pd.Timedelta(minutes=bar_minutes)).asi8
    g2 = trend_at(htf_series(h1, 120), close_ns) if h1 is not None else np.zeros(len(df), int)
    g1 = trend_at(htf_series(h1, None), close_ns) if h1 is not None else np.zeros(len(df), int)
    return trend.astype(int), np.sign(cond).astype(int), cond, g2, g1


def triggers(df, trend, adx_side, cond, rules):
    """Bars where the rules' setup fires: (bar index, direction +1/-1)."""
    out = []
    dates = df.index.date
    st_last = adx_last = None       # (direction, bar) of today's latest signal
    fired_today = set()
    for i in range(1, len(df)):
        if dates[i] != dates[i - 1]:
            st_last = adx_last = None
            fired_today = set()
        st_sig = trend[i] if trend[i] != trend[i - 1] else 0
        adx_sig = 0
        if cond[i] in (1.0, 0.5) and cond[i - 1] not in (1.0, 0.5):
            adx_sig = 1
        elif cond[i] in (-1.0, -0.5) and cond[i - 1] not in (-1.0, -0.5):
            adx_sig = -1
        if rules.setup == "raw":
            for d in {st_sig, adx_sig} - {0}:
                out.append((i, d))
            continue
        if st_sig:
            st_last = (st_sig, i)
        if adx_sig:
            adx_last = (adx_sig, i)
        if (st_sig or adx_sig) and st_last and adx_last and st_last[0] == adx_last[0]:
            d = st_last[0]
            # both still pointing that way now (the earlier one hasn't flipped back)
            if trend[i] == d and adx_side[i] == d and d not in fired_today:
                fired_today.add(d)
                out.append((i, d))
    return out


def simulate(df, bar_minutes, trig, g2, g1, rules, last_entry_bar_start):
    """Turn triggers into trades: next-candle-open entry, target/stop/time exit."""
    o, h, l, c = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    times = df.index
    dates = times.date
    sessions = sorted(set(dates))
    first_tradable = sessions[WARMUP_SESSIONS] if len(sessions) > WARMUP_SESSIONS else None
    cutoff = None
    if rules.entry_by:
        hh, mm = map(int, rules.entry_by.split(":"))
        cutoff = hh * 60 + mm - bar_minutes          # latest signal-candle START
    trades, traded = [], set()
    for i, d in trig:
        if first_tradable is None or dates[i] < first_tradable:
            continue
        if rules.side == "long" and d < 0 or rules.side == "short" and d > 0:
            continue
        if rules.gates in ("both", "2h") and g2[i] != d:
            continue
        if rules.gates in ("both", "1d") and g1[i] != d:
            continue
        start_min = times[i].hour * 60 + times[i].minute
        if cutoff is not None and start_min > cutoff:
            continue
        j = i + 1
        if j >= len(df) or dates[j] != dates[i] or (dates[i], df.attrs.get("sym")) in traded:
            continue
        if times[j].hour * 60 + times[j].minute > last_entry_bar_start:
            continue
        traded.add((dates[i], df.attrs.get("sym")))
        entry = o[j]
        tgt = entry * (1 + d * rules.target / 100) if rules.target else None
        stp = entry * (1 - d * rules.stop / 100) if rules.stop else None
        exit_px, why, k = None, "time", j
        while k < len(df) and dates[k] == dates[i]:
            hit_t = tgt is not None and (h[k] >= tgt if d > 0 else l[k] <= tgt)
            hit_s = stp is not None and (l[k] <= stp if d > 0 else h[k] >= stp)
            if hit_s:
                exit_px, why = stp, "stop"
                break
            if hit_t:
                exit_px, why = tgt, "target"
                break
            if times[k].hour * 60 + times[k].minute >= last_entry_bar_start:   # square-off candle
                exit_px = c[k]
                break
            k += 1
        if exit_px is None:
            exit_px = c[k - 1]
        gross = (exit_px / entry - 1) * 100 * d
        trades.append({"sym": df.attrs.get("sym"), "date": str(dates[i]), "signal_time": str(times[i])[11:16],
                       "side": "LONG" if d > 0 else "SHORT", "entry": round(entry, 2), "exit": round(exit_px, 2),
                       "exit_reason": why, "gross_pct": gross, "net_pct": gross - COST_PCT})
    return trades


# ─── reporting ──────────────────────────────────────────────────────────────

def summarise(trades, sessions):
    if not trades:
        return {"trades": 0}
    net = [t["net_pct"] for t in trades]
    gross = [t["gross_pct"] for t in trades]
    wins = [x for x in net if x > 0]
    losses = [x for x in net if x <= 0]
    eq = np.cumsum([x for _, x in sorted((t["date"] + t["signal_time"], t["net_pct"]) for t in trades)])
    dd = float(np.min(eq - np.maximum.accumulate(eq))) if len(eq) else 0.0
    by_day = {}
    for t in trades:
        by_day[t["date"]] = by_day.get(t["date"], 0) + t["net_pct"]
    return {"trades": len(trades), "per_day": len(trades) / max(sessions, 1),
            "win": len(wins) / len(net) * 100, "gross": S.mean(gross), "net": S.mean(net),
            "median": S.median(net), "pf": (sum(wins) / -sum(losses)) if losses and sum(losses) else float("inf"),
            "total": sum(net), "rupees": sum(net) / 100 * NOTIONAL, "max_dd": dd,
            "green_days": sum(v > 0 for v in by_day.values()) / len(by_day) * 100,
            "targets": sum(t["exit_reason"] == "target" for t in trades),
            "stops": sum(t["exit_reason"] == "stop" for t in trades)}


def fmt_row(label, s):
    if not s.get("trades"):
        return f"| {label} | 0 | | | | | | | | |"
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return (f"| {label} | {s['trades']} | {s['per_day']:.1f} | {s['win']:.0f}% | {s['gross']:+.2f}% | "
            f"**{s['net']:+.2f}%** | {pf} | {s['total']:+.1f}% (₹{s['rupees']:,.0f}) | {s['max_dd']:.1f}% | "
            f"{s['green_days']:.0f}% |")


HEAD = ("| Rules | Trades | /session | Win | Avg gross | **Avg net** | Profit factor | Total net (₹ at ₹1L/trade) "
        "| Max drawdown | Green days |\n|---|---|---|---|---|---|---|---|---|---|")


def run_test(label, bar_interval, period, bar_minutes, last_bar_start, h1_period, gates_available, symbols,
             entry_by=None):
    """entry_by overrides the rules' cut-off for this bar size: on hourly candles
    "signal candle closes by 11:00" would allow only the 9:15 candle, where ADX
    DI can never fire, so the hourly test uses the 10:15 candle (closes 11:15)."""
    print(f"\n=== {label}: downloading {len(symbols)} symbols ({bar_interval}, {period}) ===")
    data = {}
    for n, sym in enumerate(symbols, 1):
        df = download(f"{sym}.NS", period, bar_interval)
        h1 = download(f"{sym}.NS", h1_period, "1h") if h1_period else None
        if df is not None and len(df) > 200:
            df.attrs["sym"] = sym
            data[sym] = (df, h1)
        if n % 25 == 0:
            print(f"  {n}/{len(symbols)}")
        time.sleep(0.4)
    all_dates = sorted({d for df, _ in data.values() for d in set(df.index.date)})
    tradable = all_dates[WARMUP_SESSIONS:]
    unseen = [d for d in tradable if str(d) < SEEN_FROM]
    seen = [d for d in tradable if str(d) >= SEEN_FROM]
    prepared = {sym: prepare(df, bar_minutes, h1) for sym, (df, h1) in data.items()}  # once per symbol
    results = {}
    for rules in [NARROW] + VARIANTS:
        rr = rules if gates_available else replace(rules, gates={"both": "1d", "2h": "none"}.get(rules.gates, rules.gates))
        if entry_by is not None and rr.entry_by:
            rr = replace(rr, entry_by=entry_by)
        trades = []
        for sym, (df, h1) in data.items():
            trend, adx_side, cond, g2, g1 = prepared[sym]
            trades += simulate(df, bar_minutes, triggers(df, trend, adx_side, cond, rr), g2, g1, rr, last_bar_start)
        results[rules.name] = trades
    return data, all_dates, unseen, seen, results


NL = "\n"


def run_15m(symbols):
    lines = []
    data, dates, unseen, seen, res = run_test("15-min test", "15m", "60d", 15, 15 * 60, "730d", True, symbols)
    lines.append(f"## 1. 15-minute bars (the live system) — {dates[WARMUP_SESSIONS]} → {dates[-1]}{NL}")
    lines.append(f"{len(data)} stocks. Unseen period: **{unseen[0] if unseen else '—'} → {unseen[-1] if unseen else '—'} "
                 f"({len(unseen)} sessions)**, never looked at while designing. Seen period: {len(seen)} sessions.{NL}")
    rows, csv = [], []
    for name, trades in res.items():
        u = [t for t in trades if t["date"] < SEEN_FROM]
        s = [t for t in trades if t["date"] >= SEEN_FROM]
        rows.append((name, summarise(u, len(unseen)), summarise(s, len(seen))))
        if name == NARROW.name:
            csv = trades
    lines.append("### Unseen sessions (the real test)" + NL + HEAD)
    lines += [fmt_row(n, u) for n, u, _ in rows]
    lines.append(NL + "### Seen sessions (used while designing — expect these to look better)" + NL + HEAD)
    lines += [fmt_row(n, s) for n, _, s in rows]
    pd.DataFrame(csv).to_csv(OUT / "backtest_trades.csv", index=False)
    return lines


def run_hourly(symbols):
    lines = []
    data_h, dates_h, _, _, res_h = run_test("1-hour test", "1h", "730d", 60, 14 * 60 + 15, "730d", True, symbols,
                                            entry_by="11:15")
    lines.append(f"{NL}## 2. Robustness: same rules on 1-hour candles — {dates_h[WARMUP_SESSIONS]} → {dates_h[-1]}{NL}")
    lines.append("Hourly signals differ from 15-min ones, so this tests whether the *idea* holds across ~2 years "
                 "of bull, bear and sideways markets — not the exact live signals. Signal candle must close by "
                 "11:15 AM (the 10:15 candle — ADX DI never fires on the opening candle), entry at the next "
                 "candle's open, exit by the 2:15 PM candle's close." + NL)
    n_h = len(dates_h) - WARMUP_SESSIONS
    lines.append(HEAD)
    for name, trades in res_h.items():
        lines.append(fmt_row(name, summarise(trades, n_h)))

    def half_of(t):
        return t["date"][:4] + ("-H1" if t["date"][5:7] <= "06" else "-H2")

    for title, key in (("NARROW", NARROW.name), ("Short only", "… short only")):
        lines.append(f"{NL}### {title} by half-year (1-hour test){NL}"
                     f"| Period | Trades | Win | Avg net | Total net |{NL}|---|---|---|---|---|")
        nt = res_h[key]
        for half in sorted({half_of(t) for t in nt}):
            ss = summarise([t for t in nt if half_of(t) == half], 1)
            lines.append(f"| {half} | {ss['trades']} | {ss['win']:.0f}% | {ss['net']:+.2f}% | {ss['total']:+.1f}% |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-hourly", action="store_true")
    ap.add_argument("--hourly-only", action="store_true", help="re-run section 2, keep section 1 of the last report")
    args = ap.parse_args()
    symbols = live.load_universe()
    if args.hourly_only:
        prev = (OUT / "backtest_report.md").read_text(encoding="utf-8")
        lines = [prev.split(NL + "## 2.")[0].rstrip(NL)]
    else:
        lines = ["# Backtest report" + NL,
                 f"Costs: **{COST_PCT:.3f}% per round trip** (brokerage, STT, exchange/SEBI fees, GST, stamp, "
                 f"0.02% slippage per side). Notional ₹{NOTIONAL:,} per trade. 'Avg net' is per trade after costs.{NL}"]
        lines += run_15m(symbols)
    if not args.skip_hourly:
        lines += run_hourly(symbols)
    (OUT / "backtest_report.md").write_text(NL.join(lines) + NL, encoding="utf-8")
    print(NL.join(lines))


if __name__ == "__main__":
    main()


