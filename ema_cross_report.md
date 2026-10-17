# TradingView 'EMA 50, 100 and 200' — Early Cross & Golden Cross backtest

Costs per round trip: NIFTY futures 0.049% (+rollover every 21 sessions held) · stocks intraday 0.122% · stocks delivery 0.281%.

## NIFTY 50 futures — intraday 15-min
| Signal | Trades | Win | Avg win / avg loss | Avg gross | **Avg net** | Median hold |
|---|---|---|---|---|---|---|
| Early Cross (EMA50 × EMA100) · 2015–2020 | 252 | 51% | +0.42% / -0.39% | +0.07% | **+0.02%** | 13 bars |
| Early Cross (EMA50 × EMA100) · 2021–2026 | 237 | 51% | +0.35% / -0.37% | +0.05% | **+0.00%** | 13 bars |
| Golden/Death Cross (EMA50 × EMA200) · 2015–2020 | 179 | 57% | +0.38% / -0.48% | +0.06% | **+0.01%** | 13 bars |
| Golden/Death Cross (EMA50 × EMA200) · 2021–2026 | 164 | 46% | +0.38% / -0.36% | +0.03% | **-0.02%** | 13 bars |

## NIFTY 50 futures — daily (swing)
| Signal | Trades | Win | Avg win / avg loss | Avg gross | **Avg net** | Median hold |
|---|---|---|---|---|---|---|
| Early Cross (EMA50 × EMA100) · long & short · 2015–2020 | 11 | 36% | +16.25% / -6.35% | +2.21% | **+1.87%** | 88 sessions |
| Early Cross (EMA50 × EMA100) · long & short · 2021–2026 | 8 | 12% | +27.74% / -5.06% | -0.65% | **-0.96%** | 78 sessions |
| Early Cross (EMA50 × EMA100) · long only · 2015–2020 | 6 | 67% | +16.25% / -7.25% | +8.91% | **+8.42%** | 125 sessions |
| Early Cross (EMA50 × EMA100) · long only · 2021–2026 | 4 | 25% | +27.74% / -3.16% | +5.04% | **+4.57%** | 174 sessions |
| Golden/Death Cross (EMA50 × EMA200) · long & short · 2015–2020 | 11 | 18% | +32.95% / -6.85% | +0.72% | **+0.38%** | 99 sessions |
| Golden/Death Cross (EMA50 × EMA200) · long & short · 2021–2026 | 6 | 17% | +28.22% / -5.79% | +0.27% | **-0.12%** | 105 sessions |
| Golden/Death Cross (EMA50 × EMA200) · long only · 2015–2020 | 6 | 33% | +32.95% / -6.18% | +7.40% | **+6.86%** | 157 sessions |
| Golden/Death Cross (EMA50 × EMA200) · long only · 2021–2026 | 3 | 33% | +28.22% / -4.30% | +7.21% | **+6.54%** | 216 sessions |

2021 → 2026 compounded: Early Cross (EMA50 × EMA100) · long & short: ×0.88 · Early Cross (EMA50 × EMA100) · long only: ×1.16 · Golden/Death Cross (EMA50 × EMA200) · long & short: ×0.95 · Golden/Death Cross (EMA50 × EMA200) · long only: ×1.17 · **buy & hold NIFTY: ×1.60**

## 150 stocks — intraday 15-min (~30 sessions)
| Signal | Trades | Win | Avg win / avg loss | Avg gross | **Avg net** | Median hold |
|---|---|---|---|---|---|---|
| Early Cross (EMA50 × EMA100) | 1361 | 42% | +0.71% / -0.71% | +0.01% | **-0.11%** | 13 bars |
| Golden/Death Cross (EMA50 × EMA200) | 958 | 41% | +0.70% / -0.67% | +0.01% | **-0.11%** | 13 bars |

## 150 stocks — daily swing, long only, 2021 → 2026 vs buy & hold (equal weight)
| Signal | Stocks | Strategy (equal weight) | Buy & hold same stocks | Median stock: strategy / B&H | Stocks where strategy beat B&H |
|---|---|---|---|---|---|
| Early Cross (EMA50 × EMA100) | 142 | ×2.00 (12.7%/yr) | ×4.56 (29.9%/yr) | ×1.45 / ×3.13 | 13% |
| Golden/Death Cross (EMA50 × EMA200) | 142 | ×1.56 (8.0%/yr) | ×4.56 (29.9%/yr) | ×1.23 / ×3.13 | 9% |

Stock buy-and-hold carries survivorship bias (today's index members), but the comparison is like for like — same stocks, same period, costs included.
