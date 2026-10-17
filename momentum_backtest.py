"""
Monthly momentum-ranking backtest on the 150-stock universe.
=============================================================
Rules fixed before testing:

  Signal     at each month's last close, rank stocks by 12-1 momentum: the
             return over the past 12 months, skipping the most recent month
  Portfolio  hold the top 15, equal weight
  Trade      at the open of the month's first session (no look-ahead);
             positions are reset to equal weight every month
  Costs      Indian delivery round trip (~0.28%: STT 0.1% each side, stamp, fees,
             DP charge, GST, 0.02% slippage per side) charged on the fraction of
             the portfolio replaced each month

Variants (compared on 2017–2020 only; the best is then shown on 2021–2026):
  6-month momentum (6-1), top 10, top 30, NSE-style risk-adjusted momentum
  (average of 6- and 12-month return divided by volatility), quarterly rebalance.

Benchmarks
  * Equal weight, all stocks of the same universe, rebalanced monthly — carries
    the same survivorship bias, so "momentum minus this" isolates the effect of
    ranking. THIS is the fair comparison.
  * NIFTY 50 buy and hold (from the 15-min CSV) — for context only.

SURVIVORSHIP BIAS: the universe is today's Nifty 100 + Midcap 50, i.e. stocks
that did well enough to be in it now. That flatters every long-only result and
may flatter momentum in particular. Treat absolute returns as upper bounds.

Usage:  python momentum_backtest.py   (needs ema_data_cache.pkl from ema_backtest.py)
"""
import warnings; warnings.filterwarnings("ignore")

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

import ema_backtest as eb
import index_backtest as ib

OUT = Path(__file__).parent
TRAIN_END = "2020-12-31"
NL = "\n"
COST = eb.COST_DELIVERY / 100          # round trip, as a fraction
RF = 0.065                             # risk-free rate for Sharpe (approx. Indian T-bill)


@dataclass(frozen=True)
class Rules:
    name: str
    lookback: int = 12          # months
    skip: int = 1               # most recent months skipped
    top: int = 15
    score: str = "return"       # "return" | "riskadj"
    every: int = 1              # rebalance every N months


PRIMARY = Rules("12-1 momentum, top 15, monthly")
VARIANTS = [
    replace(PRIMARY, name="6-1 momentum, top 15", lookback=6),
    replace(PRIMARY, name="12-1 momentum, top 10", top=10),
    replace(PRIMARY, name="12-1 momentum, top 30", top=30),
    replace(PRIMARY, name="Risk-adjusted (NSE-style), top 15", score="riskadj"),
    replace(PRIMARY, name="12-1 momentum, top 15, quarterly", every=3),
]


def load_universe(symbols=None):
    """Daily closes/opens for the universe from the shared download cache. With
    `symbols` (e.g. the Nifty 500 list) any missing ones are downloaded first."""
    path = OUT / "ema_data_cache.pkl"
    store = pd.read_pickle(path) if path.exists() else {}
    if symbols:
        missing = [s for s in symbols if (s, "10y", "1d") not in store]
        for k, sym in enumerate(missing, 1):
            eb.download_cached(sym, "10y", "1d", store)
            if k % 50 == 0:
                print(f"  downloaded {k}/{len(missing)}")
                pd.to_pickle(store, path)
            eb.time.sleep(0.25)
        pd.to_pickle(store, path)
    closes, opens = {}, {}
    for (sym, period, iv), d in store.items():
        if symbols and sym not in symbols:
            continue
        if iv == "1d" and d is not None and len(d) > 300:
            idx = d.index.tz_localize(None).normalize()
            closes[sym] = pd.Series(d["close"].to_numpy(), index=idx)
            opens[sym] = pd.Series(d["open"].to_numpy(), index=idx)
    return pd.DataFrame(closes).sort_index(), pd.DataFrame(opens).sort_index()


def month_points(close):
    """Last session of each month (signal) and first session of the next (trade)."""
    m = close.index.to_period("M")
    last = close.groupby(m).apply(lambda x: x.index[-1])
    first = close.groupby(m).apply(lambda x: x.index[0])
    return list(last.values), list(first.values)


def scores(close, t, rules):
    """Momentum score per stock at signal date t (NaN if not enough history)."""
    hist = close.loc[:t]
    month_ends = hist.groupby(hist.index.to_period("M")).tail(1)
    if len(month_ends) < rules.lookback + 2:
        return pd.Series(dtype=float)
    end = month_ends.iloc[-1 - rules.skip]
    if rules.score == "return":
        start = month_ends.iloc[-1 - rules.skip - rules.lookback]
        return (end / start - 1).dropna()
    # NSE-style: mean of 6- and 12-month return, divided by annualised daily volatility (1 year)
    r6 = end / month_ends.iloc[-1 - rules.skip - 6] - 1
    r12 = end / month_ends.iloc[-1 - rules.skip - 12] - 1
    vol = hist.iloc[-252:].pct_change().std() * np.sqrt(252)
    return ((r6 + r12) / 2 / vol).dropna()


def backtest(close, opn, rules):
    """Monthly portfolio returns (open-to-open), net of costs on turnover."""
    sig_dates, trade_dates = month_points(close)
    rows, held = [], []
    for k in range(len(sig_dates) - 1):
        t_sig, t_in, t_out = sig_dates[k], trade_dates[k + 1] if k + 1 < len(trade_dates) else None, None
        if t_in is None or k + 2 >= len(trade_dates):
            break
        t_out = trade_dates[k + 2]
        if k % rules.every == 0 or not held:
            sc = scores(close, t_sig, rules)
            sc = sc[opn.loc[t_in, sc.index].notna() & opn.loc[t_out, sc.index].notna()]
            if len(sc) < rules.top * 2:
                continue
            new = list(sc.sort_values(ascending=False).index[:rules.top])
        else:
            new = [s for s in held if pd.notna(opn.loc[t_out, s])]
        turnover = (len(set(new) - set(held)) / len(new)) if held else 1.0
        ret = (opn.loc[t_out, new] / opn.loc[t_in, new] - 1).mean()
        cost = turnover * COST if held else COST / 2          # first month: buy side only
        rows.append({"month": pd.Timestamp(t_in).strftime("%Y-%m"), "ret": ret - cost, "gross": ret,
                     "turnover": turnover, "holdings": " ".join(new)})
        held = new
    return pd.DataFrame(rows)


def equal_weight(close, opn, months):
    """Same universe, all stocks with data, equal weight, monthly — the fair benchmark."""
    _, trade_dates = month_points(close)
    by_month = {pd.Timestamp(t).strftime("%Y-%m"): k for k, t in enumerate(trade_dates)}
    out = []
    for m in months:
        k = by_month[m]
        t_in, t_out = trade_dates[k], trade_dates[k + 1]
        ok = opn.loc[t_in].notna() & opn.loc[t_out].notna()
        out.append((opn.loc[t_out, ok] / opn.loc[t_in, ok] - 1).mean() - COST * 0.05)   # ~5% monthly drift rebalance
    return pd.Series(out, index=months)


def nifty_monthly(months):
    df, short = ib.load(r"C:\Users\Mehta\Downloads\NIFTY 50_15minute (1).csv")
    d = eb.daily_from_15m(df)
    d.index = d.index.tz_localize(None)
    first = d.groupby(d.index.to_period("M"))["open"].first()
    first.index = first.index.strftime("%Y-%m")
    r = first.shift(-1) / first - 1
    return r.reindex(months)


def yahoo_monthly(ticker, months):
    """Monthly open-to-open returns of an index / ETF from Yahoo (for benchmarks)."""
    tk = eb.live.yf.Ticker(ticker, session=eb.live.SESSION)
    d = tk.history(period="10y", interval="1d")
    if d.empty:
        return pd.Series(index=months, dtype=float)
    d.index = d.index.tz_localize(None)
    first = d.groupby(d.index.to_period("M"))["Open"].first()
    first.index = first.index.strftime("%Y-%m")
    return (first.shift(-1) / first - 1).reindex(months)


def perf(r):
    r = pd.Series(r).dropna()
    if r.empty:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / 12
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(12)
    dd = (eq / eq.cummax() - 1).min()
    return {"cagr": cagr * 100, "vol": vol * 100, "sharpe": (cagr - RF) / vol if vol else np.nan,
            "dd": dd * 100, "mult": eq.iloc[-1], "months": len(r)}


def fmt(label, p, extra=""):
    if not p:
        return f"| {label} | — | | | | | {extra} |"
    return (f"| {label} | ×{p['mult']:.2f} | **{p['cagr']:+.1f}%** | {p['vol']:.1f}% | {p['sharpe']:.2f} | "
            f"{p['dd']:.1f}% | {extra} |")


HEAD = ("| Portfolio | Growth of ₹1 | **CAGR** | Volatility | Sharpe | Max drawdown (monthly) | vs equal-weight universe |"
        + NL + "|---|---|---|---|---|---|---|")


def main():
    import sys
    n500 = "--n500" in sys.argv
    symbols = None
    if n500:
        symbols = set(pd.read_csv(OUT / "nifty500_list.csv")["Symbol"].str.strip())
    close, opn = load_universe(symbols)
    print(f"{close.shape[1]} stocks, {close.index[0].date()} → {close.index[-1].date()}")
    results = {r.name: backtest(close, opn, r) for r in [PRIMARY] + VARIANTS}
    months = list(results[PRIMARY.name]["month"])
    ew = equal_weight(close, opn, months)
    nifty = nifty_monthly(months)
    n500_idx = yahoo_monthly("^CRSLDX", months)

    def split(series_by_month, lo, hi):
        s = pd.Series(series_by_month)
        return s[(s.index >= lo) & (s.index <= hi)]

    uni_name = "Nifty 500 (today's members)" if n500 else "Nifty 100 + Midcap 50 (today's members)"
    lines = [f"# Momentum-ranking backtest — {uni_name}, monthly rebalance" + NL,
             f"{close.shape[1]} stocks with daily data, portfolios {months[0]} → {months[-1]} ({len(months)} months). "
             f"Delivery costs {eb.COST_DELIVERY:.2f}% per round trip on turnover. Returns are open-to-open, monthly." + NL,
             "> **Survivorship bias:** the universe is *today's* index members, which flatters every long-only line below "
             "(including the equal-weight benchmark). The fair test is momentum **minus** the equal-weight universe." + NL]

    for title, lo, hi in (("2017–2020 (variants compared here)", "2000-01", "2020-12"),
                          ("2021–2026 (out of sample)", "2021-01", "2099-12")):
        lines.append(f"## {title}" + NL + HEAD)
        ew_p = perf(split(dict(zip(months, ew)), lo, hi))
        for name, df in results.items():
            r = split(dict(zip(df["month"], df["ret"])), lo, hi)
            p = perf(r)
            excess = (p["cagr"] - ew_p["cagr"]) if p and ew_p else None
            turn = df[(df["month"] >= lo) & (df["month"] <= hi)]["turnover"].mean() * 12
            lines.append(fmt(name, p, f"{excess:+.1f} pts/yr · turnover {turn:.0f}x/yr" if excess is not None else ""))
        lines.append(fmt("Equal weight, all stocks (fair benchmark)", ew_p, "—"))
        lines.append(fmt("NIFTY 50 buy & hold (context)", perf(split(dict(zip(months, nifty)), lo, hi)), ""))
        lines.append(fmt("NIFTY 500 index (context)", perf(split(dict(zip(months, n500_idx)), lo, hi)), ""))
        lines.append("")

    # pick best on train, show out of sample + year by year
    train = {n: perf(split(dict(zip(d["month"], d["ret"])), "2000-01", "2020-12")) for n, d in results.items()}
    best = max(train, key=lambda n: train[n]["cagr"] - 0)  # highest train CAGR
    lines.append(f"Best on 2017–2020: **{best}**." + NL)

    lines.append("## Year by year — primary rule vs equal-weight universe vs NIFTY" + NL +
                 "| Year | 12-1 top 15 | Equal-weight universe | Momentum − equal weight | NIFTY 50 |" + NL + "|---|---|---|---|---|")
    prim = pd.Series(dict(zip(results[PRIMARY.name]["month"], results[PRIMARY.name]["ret"])))
    ews, nfs = pd.Series(dict(zip(months, ew))), pd.Series(dict(zip(months, nifty)))
    for y in sorted({m[:4] for m in months}):
        sel = [m for m in months if m.startswith(y)]
        a, b, c = ((1 + s[sel]).prod() - 1 for s in (prim, ews, nfs))
        lines.append(f"| {y}{' (part)' if len(sel) < 12 else ''} | {a * 100:+.1f}% | {b * 100:+.1f}% | "
                     f"**{(a - b) * 100:+.1f} pts** | {c * 100:+.1f}% |")
    beat = (prim - ews).dropna()
    lines.append(NL + f"Momentum beat the equal-weight universe in **{(beat > 0).mean() * 100:.0f}%** of months "
                 f"(average monthly difference {beat.mean() * 100:+.2f} pts).")

    # survivorship-free reality check: a real Nifty200 Momentum 30 ETF vs the Nifty 200 index
    etf, n200 = yahoo_monthly("MOMENTUM.NS", months), yahoo_monthly("^CNX200", months)
    live_months = [m for m in months if pd.notna(etf.get(m)) and pd.notna(n200.get(m))]
    if live_months:
        pe, pn = perf(etf[live_months]), perf(n200[live_months])
        pm, pw = perf(prim[live_months]), perf(ews[live_months])
        lines.append(NL + f"## Reality check without survivorship bias ({live_months[0]} → {live_months[-1]})" + NL +
                     "A listed **Nifty200 Momentum 30 ETF** (MOMENTUM.NS) holds NSE's real momentum index — point-in-time "
                     "members, after fund costs. Its gap over the Nifty 200 is what momentum actually delivered; compare "
                     "it with the backtest's gap over its (survivor) universe in the same months." + NL +
                     "| Same months | CAGR | Gap |" + NL + "|---|---|---|" + NL +
                     f"| Real momentum ETF | {pe['cagr']:+.1f}% | **{pe['cagr'] - pn['cagr']:+.1f} pts/yr** vs Nifty 200 ({pn['cagr']:+.1f}%) |" + NL +
                     f"| Backtest 12-1 top 15 | {pm['cagr']:+.1f}% | **{pm['cagr'] - pw['cagr']:+.1f} pts/yr** vs its equal-weight universe ({pw['cagr']:+.1f}%) |")

    last = results[PRIMARY.name].iloc[-1]
    lines.append(NL + f"## Latest portfolio (12-1 top 15, for {last['month']})" + NL + ", ".join(last["holdings"].split()))
    tag = "_n500" if n500 else ""
    (OUT / f"momentum_report{tag}.md").write_text(NL.join(lines) + NL, encoding="utf-8")
    results[PRIMARY.name].to_csv(OUT / f"momentum_monthly{tag}.csv", index=False)
    print(NL.join(lines))


if __name__ == "__main__":
    main()
