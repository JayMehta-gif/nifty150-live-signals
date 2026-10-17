"""
EMA 50 / 100 / 200 strategy backtest — intraday (15-min) and swing (daily).
============================================================================
Rules fixed before testing (no data was used to design them):

  Stack     bullish when EMA50 > EMA100 > EMA200, bearish when EMA50 < EMA100 < EMA200
  Signals   STACK    — the stack has just become fully aligned (bull or bear)
            PULLBACK — stack aligned and price closes back across EMA50 in the
                       trend's direction after being on the other side
  Entry     next candle's open (never the signal candle's own close)
  Exit      price closes across EMA100 against the trade -> out at the next open;
            intraday trades are also closed at the 3:00 PM candle's close (3:15)
  Position  one at a time per instrument; long and short (stocks swing: long only,
            since cash stocks can't be held short overnight)

Tests and costs
  1. Intraday 15-min  NIFTY futures, 2015–2026 (CSV)     ~0.05%/round trip
                      150 stocks, last ~60 days (Yahoo)  ~0.12%/round trip (MIS)
  2. Swing daily      NIFTY futures, 2015–2026 (CSV)     ~0.05%/round trip + one
                      extra round trip per 21 sessions held (monthly rollover)
                      150 stocks, 10 years, long only     ~0.27%/round trip (delivery)
  Benchmarks          buy-and-hold NIFTY / buy-and-hold each stock over the same span

Discipline: variants are compared on 2015–2020 only; the best is then shown on
2021–2026 untouched. "Win" = trade closed with a profit after costs.

Usage:  python ema_backtest.py ["path/to/NIFTY 50_15minute.csv"]
"""
import warnings; warnings.filterwarnings("ignore")

import statistics as S
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import index_backtest as ib
import nifty150_live_signals as live

OUT = Path(__file__).parent
TRAIN_END = "2020-12-31"
NL = "\n"

COST_INTRADAY_STOCK = bt.round_trip_cost_pct()         # ~0.122
COST_FUTURES = ib.futures_cost_pct()                    # ~0.049


def delivery_cost_pct(notional=100_000):
    """Indian delivery (CNC) round trip: STT 0.1% both sides, stamp 0.015% buy,
    NSE txn, SEBI, DP charge ~Rs 15.93 on sell, GST, 0.02% slippage per side."""
    stt = 0.001 * 2 * notional
    stamp = 0.00015 * notional
    txn = 0.0000297 * 2 * notional
    sebi = 0.000001 * 2 * notional
    dp = 15.93
    gst = 0.18 * (txn + sebi + dp)
    slip = 0.0002 * 2 * notional
    return (stt + stamp + txn + sebi + dp + gst + slip) / notional * 100


COST_DELIVERY = delivery_cost_pct()


@dataclass(frozen=True)
class Rules:
    name: str
    signal: str = "pullback"       # "pullback" | "stack"
    side: str = "both"             # "both" | "long" | "short"
    exit_ema: int = 100            # close across this EMA against the trade = exit
    target: float = 0.0            # % (0 = none) — intraday variants only
    stop: float = 0.0


PRIMARY = Rules("PULLBACK to EMA50, exit below/above EMA100")
VARIANTS = [
    Rules("STACK alignment, exit EMA100", signal="stack"),
    replace(PRIMARY, name="PULLBACK, long only", side="long"),
    replace(PRIMARY, name="PULLBACK, short only", side="short"),
    replace(PRIMARY, name="PULLBACK, exit EMA50", exit_ema=50),
    replace(PRIMARY, name="PULLBACK, exit EMA200", exit_ema=200),
    Rules("STACK alignment, exit EMA200", signal="stack", exit_ema=200),
]
INTRADAY_EXTRA = [
    replace(PRIMARY, name="PULLBACK + 0.5% target / 0.5% stop", target=0.5, stop=0.5),
    replace(PRIMARY, name="PULLBACK + 1% target / 1% stop", target=1.0, stop=1.0),
]


def emas(c):
    s = pd.Series(c)
    return {n: s.ewm(span=n, adjust=False).mean().to_numpy() for n in (50, 100, 200)}


def simulate(df, rules, cost_pct, intraday, sym="", roll_every=0, warmup=220):
    """Bar-by-bar: signal on bar i -> enter at open of i+1; exit when a close
    crosses the exit EMA against the trade (out at the next open), on target /
    stop (intraday variants; same-candle touch of both = stop), or at the
    3:00 PM candle's close (intraday)."""
    o, h, l, c = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    e = emas(c)
    stack = np.where((e[50] > e[100]) & (e[100] > e[200]), 1, np.where((e[50] < e[100]) & (e[100] < e[200]), -1, 0))
    times = df.index
    dates = times.date
    mins = times.hour * 60 + times.minute
    trades, pos = [], None
    pending = None                                    # (direction) to enter at next open
    exit_next = False
    for i in range(warmup, len(df)):
        new_day = i > 0 and dates[i] != dates[i - 1]
        # ── execute orders queued at the previous close
        if pos and exit_next and (not intraday or not new_day):
            pos["exit"], pos["why"], pos["xi"] = o[i], "ema", i
            trades.append(pos)
            pos, exit_next = None, False
        exit_next = False
        if pending and pos is None and (not intraday or not new_day):
            pos = {"d": pending, "entry": o[i], "ei": i, "date": str(dates[i]), "t": str(times[i])[11:16]}
        pending = None
        # ── manage open position on this bar
        if pos:
            d = pos["d"]
            if rules.target or rules.stop:
                tgt = pos["entry"] * (1 + d * rules.target / 100) if rules.target else None
                stp = pos["entry"] * (1 - d * rules.stop / 100) if rules.stop else None
                if stp is not None and (l[i] <= stp if d > 0 else h[i] >= stp):
                    pos.update(exit=stp, why="stop", xi=i); trades.append(pos); pos = None
                elif tgt is not None and (h[i] >= tgt if d > 0 else l[i] <= tgt):
                    pos.update(exit=tgt, why="target", xi=i); trades.append(pos); pos = None
            if pos and intraday and mins[i] >= 15 * 60:
                pos.update(exit=c[i], why="time", xi=i); trades.append(pos); pos = None
            if pos and d * (c[i] - e[rules.exit_ema][i]) < 0:
                exit_next = True
        # ── new signal (only when flat; intraday: not after 2:30 PM)
        if pos is None and not exit_next and stack[i] != 0:
            d = int(stack[i])
            if rules.side == "long" and d < 0 or rules.side == "short" and d > 0:
                continue
            if intraday and mins[i] > 14 * 60 + 30:
                continue
            if rules.signal == "stack":
                fire = stack[i - 1] != d
            else:
                fire = d * (c[i - 1] - e[50][i - 1]) <= 0 < d * (c[i] - e[50][i])
            if fire:
                pending = d
    out = []
    for p in trades:
        gross = (p["exit"] / p["entry"] - 1) * 100 * p["d"]
        held = p["xi"] - p["ei"]
        rolls = held // roll_every if roll_every else 0
        net = gross - cost_pct * (1 + rolls)
        out.append({"sym": sym, "date": p["date"], "time": p["t"], "side": "LONG" if p["d"] > 0 else "SHORT",
                    "entry": round(p["entry"], 2), "exit": round(p["exit"], 2), "why": p["why"],
                    "bars": int(held), "gross": gross, "net": net})
    return out


def stats(tr, sessions=None, days_held_unit=""):
    if not tr:
        return None
    net = [t["net"] for t in tr]
    wins = [x for x in net if x > 0]
    losses = [x for x in net if x <= 0]
    eq = np.cumsum([t["net"] for t in sorted(tr, key=lambda t: (t["date"], t["time"]))])
    return {"n": len(tr), "win": len(wins) / len(net) * 100, "gross": S.mean(t["gross"] for t in tr),
            "net": S.mean(net), "med": S.median(net), "pf": sum(wins) / -sum(losses) if losses and sum(losses) else float("inf"),
            "total": float(eq[-1]), "dd": float(np.min(eq - np.maximum.accumulate(eq))),
            "bars": S.median(t["bars"] for t in tr), "avg_win": S.mean(wins) if wins else 0,
            "avg_loss": S.mean(losses) if losses else 0}


HEAD = ("| Rules | Trades | Win | Avg win / avg loss | Avg gross | **Avg net** | Profit factor | Total net | "
        "Max drawdown | Median hold |" + NL + "|---|---|---|---|---|---|---|---|---|---|")


def row(name, s, unit):
    if not s:
        return f"| {name} | 0 | | | | | | | | |"
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return (f"| {name} | {s['n']} | {s['win']:.0f}% | {s['avg_win']:+.2f}% / {s['avg_loss']:+.2f}% | {s['gross']:+.2f}% | "
            f"**{s['net']:+.2f}%** | {pf} | {s['total']:+.1f}% | {s['dd']:.1f}% | {s['bars']:.0f} {unit} |")


def section(title, results, unit, split=True, note=""):
    lines = [NL + f"## {title}" + NL + (note + NL if note else "")]
    if split:
        lines.append("### 2015–2020 (variants compared here)" + NL + HEAD)
        tr_s = {n: stats([t for t in tr if t["date"] <= TRAIN_END]) for n, tr in results.items()}
        lines += [row(n, s, unit) for n, s in tr_s.items()]
        ok = {n: s for n, s in tr_s.items() if s and s["n"] >= 30}
        best = max(ok, key=lambda n: ok[n]["net"]) if ok else None
        lines.append(f"{NL}Best on 2015–2020: **{best}** → shown untouched on 2021–2026 below." + NL)
        lines.append("### 2021–2026 (out of sample)" + NL + HEAD)
        for n in dict.fromkeys([PRIMARY.name, best]):
            if n:
                lines.append(row(n, stats([t for t in results[n] if t["date"] > TRAIN_END]), unit))
    else:
        lines.append(HEAD)
        lines += [row(n, stats(tr), unit) for n, tr in results.items()]
    return lines


def by_year(trades, title):
    lines = [NL + f"### {title} by year" + NL + "| Year | Trades | Win | Avg net | Total net |" + NL + "|---|---|---|---|---|"]
    for y in sorted({t["date"][:4] for t in trades}):
        s = stats([t for t in trades if t["date"][:4] == y])
        lines.append(f"| {y} | {s['n']} | {s['win']:.0f}% | {s['net']:+.2f}% | {s['total']:+.1f}% |")
    return lines


CACHE = OUT / "ema_data_cache.pkl"


def download_cached(sym, period, interval, store):
    """Yahoo download, memoised in a pickle so re-runs don't re-download.
    Daily candles are stamped at midnight, so only intraday data is clipped to
    the 9:15–3:30 session (bt.download would drop every daily candle)."""
    key = (sym, period, interval)
    if key in store:
        return store[key]
    df = None
    tk = live.yf.Ticker(f"{sym}.NS", session=live.SESSION) if live.SESSION else live.yf.Ticker(f"{sym}.NS")
    for attempt in range(4):
        try:
            got = tk.history(period=period, interval=interval)
            if not got.empty:
                got = got.rename(columns=str.lower)
                got.index = got.index.tz_convert(live.IST)
                if interval != "1d":
                    got = got.between_time("09:15", "15:30")
                df = got.dropna(subset=["open", "close"])[["open", "high", "low", "close"]]
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                print(f"[warn] {sym} {interval}: {e}")
            time.sleep(5 * (attempt + 1))
    store[key] = df
    return df


def daily_from_15m(df):
    key = df.index.normalize()
    g = df.groupby(key)
    d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last()})
    return d


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Mehta\Downloads\NIFTY 50_15minute (1).csv"
    nifty15, short = ib.load(csv)
    nifty15 = nifty15[~np.isin(nifty15.index.date, list(short))]
    lines = ["# EMA 50 / 100 / 200 strategy — backtest" + NL,
             f"Costs per round trip: NIFTY futures **{COST_FUTURES:.3f}%** (+1 per 21 sessions held, rollover) · "
             f"stocks intraday **{COST_INTRADAY_STOCK:.3f}%** · stocks delivery **{COST_DELIVERY:.3f}%**. "
             "All returns are % of the position, after costs. 'Win' = profitable after costs." + NL]

    # ── 1a. NIFTY intraday
    res = {r.name: simulate(nifty15, r, COST_FUTURES, True, "NIFTY 50") for r in [PRIMARY] + VARIANTS + INTRADAY_EXTRA}
    lines += section("1a. Intraday 15-min — NIFTY 50 futures (2015–2026)", res, "bars (15 min)")
    lines += by_year(res[PRIMARY.name], "PULLBACK (primary), NIFTY intraday")

    # ── 2a. NIFTY swing (daily, from the same file)
    nd = daily_from_15m(nifty15)
    res_d = {r.name: simulate(nd, r, COST_FUTURES, False, "NIFTY 50", roll_every=21) for r in [PRIMARY] + VARIANTS}
    bh = (nd["close"].iloc[-1] / nd["close"].iloc[220] - 1) * 100
    bh_test = (nd["close"].iloc[-1] / nd.loc[nd.index > pd.Timestamp(TRAIN_END, tz=live.IST), "close"].iloc[0] - 1) * 100
    lines += section("2a. Swing, daily candles — NIFTY 50 futures (2015–2026)", res_d, "sessions",
                     note=f"Benchmark — buy and hold NIFTY over the same span: **{bh:+.1f}%** "
                          f"(2021–2026 alone: **{bh_test:+.1f}%**). 'Total net' is the sum of trade returns (no compounding).")
    lines += by_year(res_d[PRIMARY.name], "PULLBACK (primary), NIFTY daily swing")
    long_only = res_d["PULLBACK, long only"]
    lines.append(f"{NL}Time in market, long-only pullback: "
                 f"{sum(t['bars'] for t in long_only) / max(len(nd) - 220, 1) * 100:.0f}% of sessions.")

    # ── stocks: 15-min (60 days) and daily (10 years)
    symbols = live.load_universe()
    print(f"downloading {len(symbols)} stocks (15m 60d + 1d 10y)…")
    intr, swing, bh_stock = {}, {}, []
    rules_i = [PRIMARY] + VARIANTS + INTRADAY_EXTRA
    rules_s = [r for r in [PRIMARY] + VARIANTS if r.side != "short"]
    rules_s = [replace(r, side="long") if r.side == "both" else r for r in rules_s]
    store = pd.read_pickle(CACHE) if CACHE.exists() else {}
    for k, sym in enumerate(symbols, 1):
        fresh = (sym, "60d", "15m") not in store
        d15 = download_cached(sym, "60d", "15m", store)
        d1 = download_cached(sym, "10y", "1d", store)
        if d15 is not None and len(d15) > 400:
            for r in rules_i:
                intr.setdefault(r.name, []).extend(simulate(d15, r, COST_INTRADAY_STOCK, True, sym))
        if d1 is not None and len(d1) > 400:
            for r in rules_s:
                swing.setdefault(r.name, []).extend(simulate(d1, r, COST_DELIVERY, False, sym))
            bh_stock.append((d1["close"].iloc[-1] / d1["close"].iloc[220] - 1) * 100)
        if k % 25 == 0:
            print(f"  {k}/{len(symbols)}")
            pd.to_pickle(store, CACHE)
        if fresh:
            time.sleep(0.3)
    pd.to_pickle(store, CACHE)
    if not bh_stock:
        raise SystemExit("no daily stock data downloaded — check the connection / Yahoo")

    lines += section("1b. Intraday 15-min — 150 stocks (last ~60 days)", intr, "bars (15 min)", split=False,
                     note="Only ~60 days of 15-min data exist on Yahoo, and EMA200 needs ~9 sessions of warm-up, "
                          "so this covers ~30 sessions. New strategy — none of it was used to design the rules.")
    lines += section("2b. Swing, daily candles — 150 stocks, long only (10 years)", swing, "sessions",
                     note=f"Benchmark — median buy-and-hold return per stock over the same span: "
                          f"**{S.median(bh_stock):+.0f}%** (each stock held the whole time, one delivery round trip).")
    lines += by_year(swing[replace(PRIMARY, side='long').name if False else "PULLBACK, long only"],
                     "PULLBACK long only, 150 stocks daily swing")

    (OUT / "ema_backtest_report.md").write_text(NL.join(lines) + NL, encoding="utf-8")
    print(NL.join(lines))


if __name__ == "__main__":
    main()
