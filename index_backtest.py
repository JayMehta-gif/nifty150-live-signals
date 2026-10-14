"""
NIFTY 50 index backtest (traded through NIFTY futures), 2015 → 2026.
=====================================================================
Runs the same engine as backtest.py on one long 15-minute NIFTY 50 history
(a CSV export: date,open,high,low,close,volume), so the result covers bull,
bear, crash and sideways markets instead of 60 days.

* The 2-hour and daily stage gates are built from the same 15-min file
  (session-aligned candles; only candles closed before the signal are used).
* Sessions with fewer than 24 candles (Diwali muhurat, the 2021 NSE outage)
  are not traded; candles outside 9:15–3:30 are dropped.
* Costs model NIFTY futures, today's rates applied to every year:
  brokerage Rs 20/order, STT 0.02% on sell, NSE txn 0.00173%, SEBI fee,
  stamp 0.002% on buy, GST 18%, plus 0.01% slippage per side
  (~0.05% per round trip). Futures trade at a small basis to the index; that
  basis barely moves within a day, so intraday % moves are taken from the index.

Discipline:
  1. NARROW, exactly as fixed in backtest.py, is reported over every year.
  2. The variants are compared ONLY on 2015–2020; the best one is then tested,
     unchanged, on 2021–2026. Picking the best of many on all 12 years would be
     data-mining.

Usage:
    python index_backtest.py "path/to/NIFTY 50_15minute.csv"
"""
import warnings; warnings.filterwarnings("ignore")

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import nifty150_live_signals as live

TRAIN_END = "2020-12-31"
LOT_POINTS_NOTIONAL = 22_000 * 75     # ~ one NIFTY futures lot, for the cost estimate
SLIPPAGE = 0.0001                     # per side


def futures_cost_pct(notional=LOT_POINTS_NOTIONAL):
    brokerage = 2 * 20.0
    stt = 0.0002 * notional                       # sell side
    txn = 0.0000173 * 2 * notional
    sebi = 0.000001 * 2 * notional
    stamp = 0.00002 * notional                    # buy side
    gst = 0.18 * (brokerage + txn + sebi)
    slip = SLIPPAGE * 2 * notional
    return (brokerage + stt + txn + sebi + stamp + gst + slip) / notional * 100


def load(path):
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df.index = df.index.tz_localize(live.IST)
    df = df.between_time("09:15", "15:15")[["open", "high", "low", "close"]]
    counts = df.groupby(df.index.date).size()
    short = set(counts[counts < 24].index)
    df.attrs["sym"] = "NIFTY 50"
    return df, short


def htf_from_15m(df, minutes):
    """2-hour (minutes=120) or daily (None) SuperTrend from the 15-min bars."""
    key = live._bucket(df.index, minutes) if minutes else df.index.normalize() + pd.Timedelta(minutes=555)
    bars, g = live._ohlc(df, key)
    last_start = g.apply(lambda x: x.index[-1])
    end = last_start.dt.normalize() + pd.Timedelta(hours=15, minutes=30)
    closes = (last_start + pd.Timedelta(minutes=15)).where(last_start + pd.Timedelta(minutes=15) <= end, end)
    h, l, c = (bars[k].to_numpy(np.float64) for k in ("high", "low", "close"))
    tr, _, _ = live._supertrend(c, h, l, live._atr_rma(h, l, c, live.ATR_PERIOD), live.MULTIPLIER)
    return closes.astype("int64").to_numpy(), tr.astype(np.int8)


def prepare(df):
    h, l, c = (df[k].to_numpy(np.float64) for k in ("high", "low", "close"))
    trend, _, _ = live._supertrend(c, h, l, live._atr_rma(h, l, c, live.ATR_PERIOD), live.MULTIPLIER)
    is_open = np.array([t.hour == 9 and t.minute == 15 for t in df.index.time])
    dp, dm, sig = live._calc_di_sig(h, l, c, live.DI_LEN, live.SIG_LEN)
    cond = live._build_condition(dp, dm, sig, live.HL_RANGE, live.HL_TREND, is_open)
    close_ns = (df.index + pd.Timedelta(minutes=15)).asi8
    g2 = bt.trend_at(htf_from_15m(df, 120), close_ns)
    g1 = bt.trend_at(htf_from_15m(df, None), close_ns)
    return trend.astype(int), np.sign(cond).astype(int), cond, g2, g1


INDEX_VARIANTS = [
    bt.NARROW,
    *bt.VARIANTS,
    # index-sized exits: a 1% intraday move in NIFTY is large, so test smaller ones too
    replace(bt.NARROW, name="… target 0.5% / stop 0.3%", target=0.5, stop=0.3),
    replace(bt.NARROW, name="… target 0.3% / stop 0.3%", target=0.3, stop=0.3),
    replace(bt.NARROW, name="… SuperTrend-only signals, gated", setup="raw", target=0.5, stop=0.3),
]


def run(df, short, prepared, rules):
    trend, adx_side, cond, g2, g1 = prepared
    trades = bt.simulate(df, 15, bt.triggers(df, trend, adx_side, cond, rules), g2, g1, rules, 15 * 60)
    return [t for t in trades if pd.Timestamp(t["date"]).date() not in short]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Mehta\Downloads\NIFTY 50_15minute (1).csv"
    bt.COST_PCT = futures_cost_pct()
    df, short = load(path)
    prepared = prepare(df)
    sessions = sorted(set(df.index.date) - short)
    train = [d for d in sessions if str(d) <= TRAIN_END]
    test = [d for d in sessions if str(d) > TRAIN_END]
    NL = "\n"
    lines = ["# NIFTY 50 index backtest (via NIFTY futures)" + NL,
             f"{sessions[0]} → {sessions[-1]} · {len(sessions)} sessions · costs **{bt.COST_PCT:.3f}% per round trip** "
             f"(NIFTY futures: brokerage, STT, NSE/SEBI fees, stamp, GST, 0.01% slippage per side). "
             f"₹ figures are per 1 lot (~₹{LOT_POINTS_NOTIONAL:,} notional).{NL}"]

    results = {r.name: run(df, short, prepared, r) for r in INDEX_VARIANTS}

    def rupees(s):
        return s["total"] / 100 * LOT_POINTS_NOTIONAL if s.get("trades") else 0

    head = bt.HEAD.replace("(₹ at ₹1L/trade)", "(% · ₹ per lot)")

    def row(label, trades, n):
        s = bt.summarise(trades, n)
        if s.get("trades"):
            s = {**s, "rupees": rupees(s)}
        return bt.fmt_row(label, s)

    lines.append("## 1. Every variant — 2015–2020 (selection period)" + NL + head)
    for name, trades in results.items():
        lines.append(row(name, [t for t in trades if t["date"] <= TRAIN_END], len(train)))
    scored = {n: bt.summarise([t for t in tr if t["date"] <= TRAIN_END], len(train)) for n, tr in results.items()}
    eligible = {n: s for n, s in scored.items() if s.get("trades", 0) >= 50}
    best = max(eligible, key=lambda n: eligible[n]["net"]) if eligible else None
    lines.append(f"{NL}Best on 2015–2020 (min 50 trades, by avg net per trade): **{best}**{NL}")

    lines.append("## 2. Out-of-sample — 2021–2026, untouched" + NL + head)
    for name in [bt.NARROW.name] + ([best] if best and best != bt.NARROW.name else []):
        lines.append(row(name, [t for t in results[name] if t["date"] > TRAIN_END], len(test)))
    if best != "BASELINE: every raw signal":
        lines.append(row("BASELINE: every raw signal", [t for t in results["BASELINE: every raw signal"]
                                                         if t["date"] > TRAIN_END], len(test)))

    lines.append(NL + "## 3. Year by year" + NL + "| Year | " + " | ".join(
        f"{n} (trades · win · avg net · total)" for n in [bt.NARROW.name] + ([best] if best and best != bt.NARROW.name else [])) +
        " |" + NL + "|---|" + "---|" * (1 + (best != bt.NARROW.name)))
    for y in sorted({t["date"][:4] for tr in results.values() for t in tr}):
        cells = []
        for n in [bt.NARROW.name] + ([best] if best and best != bt.NARROW.name else []):
            ys = bt.summarise([t for t in results[n] if t["date"][:4] == y], 1)
            cells.append(f"{ys['trades']} · {ys['win']:.0f}% · {ys['net']:+.2f}% · {ys['total']:+.1f}%" if ys.get("trades") else "—")
        lines.append(f"| {y} | " + " | ".join(cells) + " |")

    out = Path(__file__).parent / "index_backtest_report.md"
    out.write_text(NL.join(lines) + NL, encoding="utf-8")
    pd.DataFrame(results[best or bt.NARROW.name]).to_csv(Path(__file__).parent / "index_backtest_trades.csv", index=False)
    print(NL.join(lines))


if __name__ == "__main__":
    main()
