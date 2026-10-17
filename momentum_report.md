# Momentum-ranking backtest — Nifty 100 + Midcap 50 (today's members), monthly rebalance

147 stocks with daily data, portfolios 2017-12 → 2026-09 (106 months). Delivery costs 0.28% per round trip on turnover. Returns are open-to-open, monthly.

> **Survivorship bias:** the universe is *today's* index members, which flatters every long-only line below (including the equal-weight benchmark). The fair test is momentum **minus** the equal-weight universe.

## 2017–2020 (variants compared here)
| Portfolio | Growth of ₹1 | **CAGR** | Volatility | Sharpe | Max drawdown (monthly) | vs equal-weight universe |
|---|---|---|---|---|---|---|
| 12-1 momentum, top 15, monthly | ×1.85 | **+22.0%** | 25.9% | 0.60 | -30.1% | +9.5 pts/yr · turnover 4x/yr |
| 6-1 momentum, top 15 | ×2.48 | **+28.8%** | 25.3% | 0.88 | -28.3% | +16.3 pts/yr · turnover 4x/yr |
| 12-1 momentum, top 10 | ×2.04 | **+26.0%** | 26.4% | 0.74 | -27.6% | +13.6 pts/yr · turnover 4x/yr |
| 12-1 momentum, top 30 | ×1.58 | **+16.0%** | 24.7% | 0.38 | -29.8% | +3.5 pts/yr · turnover 2x/yr |
| Risk-adjusted (NSE-style), top 15 | ×1.94 | **+23.9%** | 25.1% | 0.69 | -25.9% | +11.5 pts/yr · turnover 4x/yr |
| 12-1 momentum, top 15, quarterly | ×1.66 | **+17.8%** | 24.1% | 0.47 | -28.2% | +5.3 pts/yr · turnover 2x/yr |
| Equal weight, all stocks (fair benchmark) | ×1.44 | **+12.5%** | 25.3% | 0.24 | -30.0% | — |
| NIFTY 50 buy & hold (context) | ×1.36 | **+10.6%** | 21.7% | 0.19 | -29.7% |  |
| NIFTY 500 index (context) | ×1.27 | **+8.0%** | 22.4% | 0.07 | -30.4% |  |

## 2021–2026 (out of sample)
| Portfolio | Growth of ₹1 | **CAGR** | Volatility | Sharpe | Max drawdown (monthly) | vs equal-weight universe |
|---|---|---|---|---|---|---|
| 12-1 momentum, top 15, monthly | ×8.35 | **+44.6%** | 22.8% | 1.67 | -19.1% | +17.5 pts/yr · turnover 3x/yr |
| 6-1 momentum, top 15 | ×4.36 | **+29.2%** | 23.0% | 0.99 | -19.9% | +2.1 pts/yr · turnover 4x/yr |
| 12-1 momentum, top 10 | ×9.03 | **+46.6%** | 25.5% | 1.57 | -21.3% | +19.5 pts/yr · turnover 3x/yr |
| 12-1 momentum, top 30 | ×5.20 | **+33.2%** | 20.1% | 1.33 | -22.5% | +6.1 pts/yr · turnover 3x/yr |
| Risk-adjusted (NSE-style), top 15 | ×5.20 | **+33.2%** | 21.9% | 1.22 | -22.2% | +6.1 pts/yr · turnover 3x/yr |
| 12-1 momentum, top 15, quarterly | ×9.95 | **+49.1%** | 23.2% | 1.84 | -20.8% | +22.0 pts/yr · turnover 1x/yr |
| Equal weight, all stocks (fair benchmark) | ×3.97 | **+27.1%** | 16.0% | 1.29 | -19.2% | — |
| NIFTY 50 buy & hold (context) | ×1.61 | **+8.6%** | 13.1% | 0.16 | -14.4% |  |
| NIFTY 500 index (context) | ×1.89 | **+11.7%** | 14.0% | 0.37 | -17.8% |  |

Best on 2017–2020: **6-1 momentum, top 15**.

## Year by year — primary rule vs equal-weight universe vs NIFTY
| Year | 12-1 top 15 | Equal-weight universe | Momentum − equal weight | NIFTY 50 |
|---|---|---|---|---|
| 2017 (part) | +7.2% | +4.4% | **+2.8 pts** | +2.6% |
| 2018 | -2.0% | -5.1% | **+3.1 pts** | +3.3% |
| 2019 | +11.2% | +9.6% | **+1.6 pts** | +12.1% |
| 2020 | +58.0% | +32.2% | **+25.8 pts** | +14.7% |
| 2021 | +70.9% | +54.8% | **+16.1 pts** | +24.2% |
| 2022 | +26.3% | +12.7% | **+13.6 pts** | +4.3% |
| 2023 | +83.6% | +54.0% | **+29.7 pts** | +19.8% |
| 2024 | +80.2% | +30.2% | **+50.0 pts** | +8.8% |
| 2025 | +20.5% | +14.8% | **+5.7 pts** | +10.7% |
| 2026 (part) | -3.1% | -1.1% | **-2.0 pts** | -13.9% |

Momentum beat the equal-weight universe in **64%** of months (average monthly difference +1.03 pts).

## Reality check without survivorship bias (2022-08 → 2026-09)
A listed **Nifty200 Momentum 30 ETF** (MOMENTUM.NS) holds NSE's real momentum index — point-in-time members, after fund costs. Its gap over the Nifty 200 is what momentum actually delivered; compare it with the backtest's gap over its (survivor) universe in the same months.
| Same months | CAGR | Gap |
|---|---|---|
| Real momentum ETF | +11.8% | **+2.7 pts/yr** vs Nifty 200 (+9.1%) |
| Backtest 12-1 top 15 | +43.7% | **+18.9 pts/yr** vs its equal-weight universe (+24.8%) |

## Latest portfolio (12-1 top 15, for 2026-09)
LAURUSLABS, ADANIENSOL, NATIONALUM, BHARATFORG, IDEA, ADANIPOWER, FEDERALBNK, MCX, BHEL, SHRIRAMFIN, POWERINDIA, NYKAA, GVT&D, CUMMINSIND, MOTHERSON
