"""
Backtest of the TradingView "EMA 50, 100 and 200" indicator's two signals.
=========================================================================
  Early Cross  — EMA50 crosses EMA100   (up = buy, down = sell)
  Golden Cross — EMA50 crosses EMA200   (up = golden cross / buy, down = death cross / sell)

Traded as the indicator implies: enter at the next candle's open after a cross,
hold until the opposite cross of the same pair (exit at the next open); intraday
trades are also closed at the 3:00 PM candle's close (3:15). Long and short on
NIFTY futures; long only for stocks held overnight (cash stocks can't be shorted
overnight). Costs and data as in ema_backtest.py; 2021 → 2026 is the test period
for the comparison with buy-and-hold.

Usage:  python ema_cross_backtest.py   (needs ema_data_cache.pkl from ema_backtest.py)
"""
import warnings; warnings.filterwarnings("ignore")

import statistics as S
from pathlib import Path

import numpy as np
import pandas as pd

import ema_backtest as eb
import index_backtest as ib

OUT = Path(__file__).parent
TEST = "2021-01-01"
NL = "\n"


def simulate_cross(df, slow, cost_pct, intraday, side="both", roll_every=0, warmup=220):
    o, c = df["open"].to_numpy(np.float64), df["close"].to_numpy(np.float64)
    s = pd.Series(c)
    fast_e = s.ewm(span=50, adjust=False).mean().to_numpy()
    slow_e = s.ewm(span=slow, adjust=False).mean().to_numpy()
    above = np.sign(fast_e - slow_e)
    dates, mins = df.index.date, df.index.hour * 60 + df.index.minute
    trades, pos, pending = [], None, 0
    for i in range(warmup, len(df)):
        new_day = dates[i] != dates[i - 1]
        if pending and (not intraday or not new_day):
            if pos and pos["d"] != pending:                         # opposite cross -> exit
                pos.update(exit=o[i], xi=i)
                trades.append(pos)
                pos = None
            want = pending if side == "both" or (side == "long") == (pending > 0) else 0
            if want and pos is None and not (intraday and mins[i] > 14 * 60 + 30):
                pos = {"d": want, "entry": o[i], "ei": i, "date": str(dates[i])}
        pending = 0
        if pos and intraday and mins[i] >= 15 * 60:
            pos.update(exit=c[i], xi=i)
            trades.append(pos)
            pos = None
        if above[i] != 0 and above[i] != above[i - 1] and above[i - 1] != 0:
            pending = int(above[i])                                 # cross on this candle
    out = []
    for p in trades:
        gross = (p["exit"] / p["entry"] - 1) * 100 * p["d"]
        held = p["xi"] - p["ei"]
        rolls = held // roll_every if roll_every else 0
        out.append({"date": p["date"], "gross": gross, "net": gross - cost_pct * (1 + rolls), "bars": held,
                    "time": "", "d": p["d"]})
    return out


def line(label, tr, unit):
    if not tr:
        return f"| {label} | 0 | | | | | |"
    st = eb.stats(tr)
    return (f"| {label} | {st['n']} | {st['win']:.0f}% | {st['avg_win']:+.2f}% / {st['avg_loss']:+.2f}% | "
            f"{st['gross']:+.2f}% | **{st['net']:+.2f}%** | {st['bars']:.0f} {unit} |")


HEAD = ("| Signal | Trades | Win | Avg win / avg loss | Avg gross | **Avg net** | Median hold |" + NL +
        "|---|---|---|---|---|---|---|")


def main():
    lines = ["# TradingView 'EMA 50, 100 and 200' — Early Cross & Golden Cross backtest" + NL,
             f"Costs per round trip: NIFTY futures {eb.COST_FUTURES:.3f}% (+rollover every 21 sessions held) · "
             f"stocks intraday {eb.COST_INTRADAY_STOCK:.3f}% · stocks delivery {eb.COST_DELIVERY:.3f}%." + NL]
    df, short = ib.load(r"C:\Users\Mehta\Downloads\NIFTY 50_15minute (1).csv")
    df = df[~np.isin(df.index.date, list(short))]
    nd = eb.daily_from_15m(df)
    store = pd.read_pickle(OUT / "ema_data_cache.pkl")
    sigs = (("Early Cross (EMA50 × EMA100)", 100), ("Golden/Death Cross (EMA50 × EMA200)", 200))

    lines.append("## NIFTY 50 futures — intraday 15-min" + NL + "| Period | " + HEAD[2:].split(NL)[0][2:] + NL + HEAD.split(NL)[1] + "|---|")
    lines[-1] = "## NIFTY 50 futures — intraday 15-min" + NL + HEAD
    for name, slow in sigs:
        tr = simulate_cross(df, slow, eb.COST_FUTURES, True)
        lines.append(line(f"{name} · 2015–2020", [t for t in tr if t["date"] < TEST], "bars"))
        lines.append(line(f"{name} · 2021–2026", [t for t in tr if t["date"] >= TEST], "bars"))

    lines.append(NL + "## NIFTY 50 futures — daily (swing)" + NL + HEAD)
    nd_test = nd[nd.index >= pd.Timestamp(TEST, tz=eb.live.IST)]
    bh = nd_test["close"].iloc[-1] / nd_test["open"].iloc[0]
    comp = []
    for name, slow in sigs:
        for side in ("both", "long"):
            tr = simulate_cross(nd, slow, eb.COST_FUTURES, False, side=side, roll_every=21)
            lab = f"{name}{' · long only' if side == 'long' else ' · long & short'}"
            lines.append(line(f"{lab} · 2015–2020", [t for t in tr if t["date"] < TEST], "sessions"))
            tt = [t for t in tr if t["date"] >= TEST]
            lines.append(line(f"{lab} · 2021–2026", tt, "sessions"))
            comp.append((lab, np.prod([1 + t["net"] / 100 for t in tt]) if tt else 1.0))
    lines.append(NL + f"2021 → 2026 compounded: " + " · ".join(f"{l}: ×{m:.2f}" for l, m in comp) +
                 f" · **buy & hold NIFTY: ×{bh:.2f}**" + NL)

    lines.append("## 150 stocks — intraday 15-min (~30 sessions)" + NL + HEAD)
    for name, slow in sigs:
        tr = []
        for (sym, per, iv), d in store.items():
            if iv == "15m" and d is not None and len(d) > 400:
                tr += simulate_cross(d, slow, eb.COST_INTRADAY_STOCK, True)
        lines.append(line(name, tr, "bars"))

    lines.append(NL + "## 150 stocks — daily swing, long only, 2021 → 2026 vs buy & hold (equal weight)" + NL +
                 "| Signal | Stocks | Strategy (equal weight) | Buy & hold same stocks | Median stock: strategy / B&H | "
                 "Stocks where strategy beat B&H |" + NL + "|---|---|---|---|---|---|")
    for name, slow in sigs:
        s_m, b_m = [], []
        for (sym, per, iv), d in store.items():
            if iv != "1d" or d is None or len(d) < 600:
                continue
            tr = [t for t in simulate_cross(d, slow, eb.COST_DELIVERY, False, side="long") if t["date"] >= TEST]
            s_m.append(np.prod([1 + t["net"] / 100 for t in tr]) if tr else 1.0)
            d2 = d[d.index >= pd.Timestamp(TEST, tz=eb.live.IST)]
            b_m.append(d2["close"].iloc[-1] / d2["open"].iloc[0] * (1 - eb.COST_DELIVERY / 100))
        s_m, b_m = np.array(s_m), np.array(b_m)
        cagr = lambda m: (m ** (1 / 5.8) - 1) * 100  # noqa: E731
        lines.append(f"| {name} | {len(s_m)} | ×{s_m.mean():.2f} ({cagr(s_m.mean()):.1f}%/yr) | "
                     f"×{b_m.mean():.2f} ({cagr(b_m.mean()):.1f}%/yr) | ×{np.median(s_m):.2f} / ×{np.median(b_m):.2f} | "
                     f"{np.mean(s_m > b_m) * 100:.0f}% |")
    lines.append(NL + "Stock buy-and-hold carries survivorship bias (today's index members), but the comparison is like "
                 "for like — same stocks, same period, costs included.")
    (OUT / "ema_cross_report.md").write_text(NL.join(lines) + NL, encoding="utf-8")
    print(NL.join(lines))


if __name__ == "__main__":
    main()
