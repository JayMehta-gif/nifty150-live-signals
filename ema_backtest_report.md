# EMA 50 / 100 / 200 strategy — backtest

Costs per round trip: NIFTY futures **0.049%** (+1 per 21 sessions held, rollover) · stocks intraday **0.122%** · stocks delivery **0.281%**. All returns are % of the position, after costs. 'Win' = profitable after costs.


## 1a. Intraday 15-min — NIFTY 50 futures (2015–2026)

### 2015–2020 (variants compared here)
| Rules | Trades | Win | Avg win / avg loss | Avg gross | **Avg net** | Profit factor | Total net | Max drawdown | Median hold |
|---|---|---|---|---|---|---|---|---|---|
| PULLBACK to EMA50, exit below/above EMA100 | 500 | 42% | +0.45% / -0.33% | +0.05% | **-0.00%** | 0.99 | -1.2% | -16.6% | 8 bars (15 min) |
| STACK alignment, exit EMA100 | 203 | 51% | +0.39% / -0.38% | +0.06% | **+0.01%** | 1.05 | +2.0% | -6.6% | 11 bars (15 min) |
| PULLBACK, long only | 333 | 42% | +0.32% / -0.29% | +0.02% | **-0.03%** | 0.80 | -11.2% | -16.8% | 8 bars (15 min) |
| PULLBACK, short only | 167 | 41% | +0.72% / -0.39% | +0.11% | **+0.06%** | 1.26 | +10.0% | -6.4% | 10 bars (15 min) |
| PULLBACK, exit EMA50 | 689 | 28% | +0.46% / -0.20% | +0.04% | **-0.01%** | 0.91 | -8.9% | -20.0% | 3 bars (15 min) |
| PULLBACK, exit EMA200 | 462 | 47% | +0.44% / -0.41% | +0.04% | **-0.01%** | 0.96 | -4.2% | -17.3% | 12 bars (15 min) |
| STACK alignment, exit EMA200 | 202 | 51% | +0.39% / -0.40% | +0.05% | **+0.00%** | 1.01 | +0.4% | -8.0% | 12 bars (15 min) |
| PULLBACK + 0.5% target / 0.5% stop | 514 | 44% | +0.33% / -0.31% | +0.02% | **-0.03%** | 0.83 | -15.0% | -19.7% | 5 bars (15 min) |
| PULLBACK + 1% target / 1% stop | 500 | 42% | +0.42% / -0.33% | +0.03% | **-0.02%** | 0.91 | -8.4% | -15.1% | 8 bars (15 min) |

Best on 2015–2020: **PULLBACK, short only** → shown untouched on 2021–2026 below.

### 2021–2026 (out of sample)
| Rules | Trades | Win | Avg win / avg loss | Avg gross | **Avg net** | Profit factor | Total net | Max drawdown | Median hold |
|---|---|---|---|---|---|---|---|---|---|
| PULLBACK to EMA50, exit below/above EMA100 | 509 | 38% | +0.37% / -0.32% | -0.01% | **-0.06%** | 0.71 | -29.4% | -31.5% | 9 bars (15 min) |
| PULLBACK, short only | 224 | 42% | +0.44% / -0.37% | +0.02% | **-0.03%** | 0.86 | -6.9% | -13.1% | 9 bars (15 min) |

### PULLBACK (primary), NIFTY intraday by year
| Year | Trades | Win | Avg net | Total net |
|---|---|---|---|---|
| 2015 | 73 | 41% | -0.02% | -1.5% |
| 2016 | 76 | 33% | -0.09% | -6.8% |
| 2017 | 94 | 36% | -0.05% | -4.6% |
| 2018 | 90 | 52% | +0.05% | +4.1% |
| 2019 | 92 | 36% | -0.05% | -4.6% |
| 2020 | 75 | 52% | +0.16% | +12.3% |
| 2021 | 81 | 38% | -0.10% | -8.1% |
| 2022 | 83 | 51% | +0.04% | +2.9% |
| 2023 | 87 | 36% | -0.06% | -5.1% |
| 2024 | 96 | 47% | -0.02% | -1.6% |
| 2025 | 94 | 28% | -0.12% | -11.3% |
| 2026 | 68 | 29% | -0.09% | -6.3% |

## 2a. Swing, daily candles — NIFTY 50 futures (2015–2026)
Benchmark — buy and hold NIFTY over the same span: **+185.1%** (2021–2026 alone: **+60.0%**). 'Total net' is the sum of trade returns (no compounding).

### 2015–2020 (variants compared here)
| Rules | Trades | Win | Avg win / avg loss | Avg gross | **Avg net** | Profit factor | Total net | Max drawdown | Median hold |
|---|---|---|---|---|---|---|---|---|---|
| PULLBACK to EMA50, exit below/above EMA100 | 13 | 54% | +9.29% / -3.17% | +3.70% | **+3.54%** | 3.42 | +46.0% | -14.0% | 31 sessions |
| STACK alignment, exit EMA100 | 7 | 43% | +7.67% / -7.11% | -0.58% | **-0.78%** | 0.81 | -5.4% | -27.6% | 48 sessions |
| PULLBACK, long only | 11 | 55% | +10.42% / -2.46% | +4.73% | **+4.57%** | 5.09 | +50.2% | -7.3% | 31 sessions |
| PULLBACK, short only | 2 | 50% | +2.52% / -6.72% | -1.98% | **-2.10%** | 0.38 | -4.2% | -6.7% | 36 sessions |
| PULLBACK, exit EMA50 | 25 | 40% | +4.17% / -0.89% | +1.21% | **+1.13%** | 3.11 | +28.3% | -4.8% | 6 sessions |
| PULLBACK, exit EMA200 | 9 | 44% | +13.16% / -5.55% | +3.02% | **+2.77%** | 1.90 | +24.9% | -20.9% | 68 sessions |
| STACK alignment, exit EMA200 | 7 | 43% | +6.72% / -8.92% | -1.97% | **-2.21%** | 0.57 | -15.5% | -32.6% | 69 sessions |

Best on 2015–2020: **None** → shown untouched on 2021–2026 below.

### 2021–2026 (out of sample)
| Rules | Trades | Win | Avg win / avg loss | Avg gross | **Avg net** | Profit factor | Total net | Max drawdown | Median hold |
|---|---|---|---|---|---|---|---|---|---|
| PULLBACK to EMA50, exit below/above EMA100 | 26 | 19% | +4.65% / -1.79% | -0.47% | **-0.55%** | 0.62 | -14.4% | -15.8% | 6 sessions |

### PULLBACK (primary), NIFTY daily swing by year
| Year | Trades | Win | Avg net | Total net |
|---|---|---|---|---|
| 2016 | 2 | 50% | -0.31% | -0.6% |
| 2017 | 1 | 100% | +5.41% | +5.4% |
| 2018 | 3 | 67% | +0.45% | +1.3% |
| 2019 | 3 | 67% | +0.87% | +2.6% |
| 2020 | 4 | 25% | +9.32% | +37.3% |
| 2022 | 8 | 12% | -1.57% | -12.5% |
| 2023 | 4 | 25% | +1.81% | +7.2% |
| 2024 | 2 | 50% | +2.93% | +5.9% |
| 2025 | 4 | 25% | -1.98% | -7.9% |
| 2026 | 8 | 12% | -0.88% | -7.0% |

Time in market, long-only pullback: 42% of sessions.

## 1b. Intraday 15-min — 150 stocks (last ~60 days)
Only ~60 days of 15-min data exist on Yahoo, and EMA200 needs ~9 sessions of warm-up, so this covers ~30 sessions. New strategy — none of it was used to design the rules.

| Rules | Trades | Win | Avg win / avg loss | Avg gross | **Avg net** | Profit factor | Total net | Max drawdown | Median hold |
|---|---|---|---|---|---|---|---|---|---|
| PULLBACK to EMA50, exit below/above EMA100 | 3577 | 36% | +0.69% / -0.58% | +0.00% | **-0.12%** | 0.68 | -427.5% | -428.2% | 10 bars (15 min) |
| STACK alignment, exit EMA100 | 1017 | 43% | +0.67% / -0.66% | +0.03% | **-0.09%** | 0.76 | -94.4% | -106.4% | 11 bars (15 min) |
| PULLBACK, long only | 1164 | 29% | +0.73% / -0.57% | -0.08% | **-0.20%** | 0.51 | -236.0% | -235.9% | 8 bars (15 min) |
| PULLBACK, short only | 2413 | 40% | +0.68% / -0.58% | +0.04% | **-0.08%** | 0.77 | -191.5% | -201.3% | 11 bars (15 min) |
| PULLBACK, exit EMA50 | 4959 | 25% | +0.71% / -0.39% | +0.01% | **-0.11%** | 0.61 | -570.1% | -569.9% | 3 bars (15 min) |
| PULLBACK, exit EMA200 | 3189 | 41% | +0.69% / -0.69% | -0.00% | **-0.12%** | 0.69 | -395.7% | -396.5% | 15 bars (15 min) |
| STACK alignment, exit EMA200 | 1016 | 44% | +0.67% / -0.69% | +0.03% | **-0.09%** | 0.77 | -91.7% | -105.1% | 11 bars (15 min) |
| PULLBACK + 0.5% target / 0.5% stop | 4058 | 43% | +0.35% / -0.49% | -0.01% | **-0.13%** | 0.54 | -521.3% | -521.7% | 3 bars (15 min) |
| PULLBACK + 1% target / 1% stop | 3666 | 38% | +0.59% / -0.57% | -0.01% | **-0.13%** | 0.63 | -471.2% | -472.2% | 7 bars (15 min) |

## 2b. Swing, daily candles — 150 stocks, long only (10 years)
Benchmark — median buy-and-hold return per stock over the same span: **+260%** (each stock held the whole time, one delivery round trip).

### 2015–2020 (variants compared here)
| Rules | Trades | Win | Avg win / avg loss | Avg gross | **Avg net** | Profit factor | Total net | Max drawdown | Median hold |
|---|---|---|---|---|---|---|---|---|---|
| PULLBACK to EMA50, exit below/above EMA100 | 870 | 28% | +27.43% / -3.71% | +5.41% | **+5.13%** | 2.93 | +4464.3% | -326.4% | 13 sessions |
| STACK alignment, exit EMA100 | 368 | 34% | +53.62% / -6.03% | +14.51% | **+14.23%** | 4.57 | +5237.5% | -535.7% | 36 sessions |
| PULLBACK, long only | 870 | 28% | +27.43% / -3.71% | +5.41% | **+5.13%** | 2.93 | +4464.3% | -326.4% | 13 sessions |
| PULLBACK, exit EMA50 | 1665 | 25% | +13.93% / -2.37% | +2.03% | **+1.75%** | 1.99 | +2911.2% | -290.3% | 4 sessions |
| PULLBACK, exit EMA200 | 485 | 33% | +53.51% / -5.71% | +13.98% | **+13.70%** | 4.57 | +6645.6% | -425.5% | 39 sessions |
| STACK alignment, exit EMA200 | 366 | 38% | +94.09% / -7.18% | +31.28% | **+31.00%** | 7.93 | +11346.6% | -368.1% | 58 sessions |

Best on 2015–2020: **STACK alignment, exit EMA200** → shown untouched on 2021–2026 below.

### 2021–2026 (out of sample)
| Rules | Trades | Win | Avg win / avg loss | Avg gross | **Avg net** | Profit factor | Total net | Max drawdown | Median hold |
|---|---|---|---|---|---|---|---|---|---|
| PULLBACK to EMA50, exit below/above EMA100 | 2071 | 29% | +26.21% / -3.76% | +5.17% | **+4.89%** | 2.83 | +10134.5% | -590.1% | 13 sessions |
| STACK alignment, exit EMA200 | 615 | 36% | +54.25% / -6.84% | +15.30% | **+15.01%** | 4.42 | +9233.6% | -616.0% | 52 sessions |

### PULLBACK long only, 150 stocks daily swing by year
| Year | Trades | Win | Avg net | Total net |
|---|---|---|---|---|
| 2017 | 161 | 30% | +1.82% | +292.7% |
| 2018 | 270 | 19% | -0.83% | -224.2% |
| 2019 | 262 | 29% | +1.32% | +345.6% |
| 2020 | 177 | 41% | +22.88% | +4050.2% |
| 2021 | 407 | 35% | +5.26% | +2141.5% |
| 2022 | 359 | 26% | +2.83% | +1015.0% |
| 2023 | 353 | 39% | +17.93% | +6331.1% |
| 2024 | 370 | 27% | +2.00% | +741.4% |
| 2025 | 333 | 24% | +0.67% | +224.5% |
| 2026 | 249 | 17% | -1.28% | -318.9% |

## 3. Fair comparison — does the daily EMA strategy beat simply holding? (2021 → Oct 2026)
Section 2b's per-trade averages are inflated by **survivorship bias** (today's index members are the stocks that rose) and its
"Total net" sums overlapping trades. Here each stock gets an equal slice of capital; the strategy compounds its own trades
in that slice, compared with buying and holding the same stock (costs included, idle cash earns nothing).

| Daily swing, 2021 → 2026 | Strategy | Buy and hold same stocks |
|---|---|---|
| PULLBACK long, exit EMA100 — 142 stocks, equal weight | ×1.82 (CAGR 10.9%) | **×4.56 (CAGR 29.9%)** |
| … median stock | ×1.37 | ×3.13 |
| … stocks where strategy beat buy & hold | 13% | |
| STACK long, exit EMA200 — 142 stocks, equal weight | ×1.64 (CAGR 9.0%) | **×4.56 (CAGR 29.9%)** |
| NIFTY 50 (futures), PULLBACK long only — 19 trades | ×0.99 | **×1.60** |

Buy-and-hold figures for the stocks carry the same survivorship bias, so treat their level with caution — but the
comparison is like for like: on the same stocks, staying invested beat timing entries and exits with EMA 50/100/200.
