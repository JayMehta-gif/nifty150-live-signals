# Momentum-ranking backtest — Nifty 500 (today's members), monthly rebalance

473 stocks with daily data, portfolios 2017-12 → 2026-09 (106 months). Delivery costs 0.28% per round trip on turnover. Returns are open-to-open, monthly.

> **Survivorship bias:** the universe is *today's* index members, which flatters every long-only line below (including the equal-weight benchmark). The fair test is momentum **minus** the equal-weight universe.

## 2017–2020 (variants compared here)
| Portfolio | Growth of ₹1 | **CAGR** | Volatility | Sharpe | Max drawdown (monthly) | vs equal-weight universe |
|---|---|---|---|---|---|---|
| 12-1 momentum, top 15, monthly | ×1.76 | **+20.0%** | 30.3% | 0.45 | -33.8% | +11.9 pts/yr · turnover 4x/yr |
| 6-1 momentum, top 15 | ×3.65 | **+43.6%** | 29.3% | 1.26 | -25.0% | +35.5 pts/yr · turnover 5x/yr |
| 12-1 momentum, top 10 | ×2.04 | **+26.0%** | 33.0% | 0.59 | -30.0% | +17.9 pts/yr · turnover 4x/yr |
| 12-1 momentum, top 30 | ×1.72 | **+19.2%** | 27.6% | 0.46 | -30.6% | +11.1 pts/yr · turnover 4x/yr |
| Risk-adjusted (NSE-style), top 15 | ×1.78 | **+20.6%** | 26.6% | 0.53 | -28.4% | +12.6 pts/yr · turnover 4x/yr |
| 12-1 momentum, top 15, quarterly | ×1.93 | **+23.9%** | 29.8% | 0.58 | -28.7% | +15.8 pts/yr · turnover 2x/yr |
| Equal weight, all stocks (fair benchmark) | ×1.27 | **+8.1%** | 28.4% | 0.06 | -39.7% | — |
| NIFTY 50 buy & hold (context) | ×1.36 | **+10.6%** | 21.7% | 0.19 | -29.7% |  |
| NIFTY 500 index (context) | ×1.27 | **+8.0%** | 22.4% | 0.07 | -30.4% |  |

## 2021–2026 (out of sample)
| Portfolio | Growth of ₹1 | **CAGR** | Volatility | Sharpe | Max drawdown (monthly) | vs equal-weight universe |
|---|---|---|---|---|---|---|
| 12-1 momentum, top 15, monthly | ×13.91 | **+58.1%** | 28.8% | 1.79 | -26.1% | +27.3 pts/yr · turnover 4x/yr |
| 6-1 momentum, top 15 | ×11.67 | **+53.3%** | 30.2% | 1.55 | -28.7% | +22.6 pts/yr · turnover 5x/yr |
| 12-1 momentum, top 10 | ×12.49 | **+55.1%** | 31.5% | 1.54 | -31.8% | +24.4 pts/yr · turnover 4x/yr |
| 12-1 momentum, top 30 | ×12.77 | **+55.7%** | 26.3% | 1.87 | -29.4% | +25.0 pts/yr · turnover 3x/yr |
| Risk-adjusted (NSE-style), top 15 | ×12.22 | **+54.5%** | 29.8% | 1.61 | -27.4% | +23.8 pts/yr · turnover 4x/yr |
| 12-1 momentum, top 15, quarterly | ×12.30 | **+54.7%** | 28.1% | 1.71 | -24.9% | +24.0 pts/yr · turnover 2x/yr |
| Equal weight, all stocks (fair benchmark) | ×4.67 | **+30.8%** | 18.0% | 1.35 | -20.7% | — |
| NIFTY 50 buy & hold (context) | ×1.61 | **+8.6%** | 13.1% | 0.16 | -14.4% |  |
| NIFTY 500 index (context) | ×1.89 | **+11.7%** | 14.0% | 0.37 | -17.8% |  |

Best on 2017–2020: **6-1 momentum, top 15**.

## Year by year — primary rule vs equal-weight universe vs NIFTY
| Year | 12-1 top 15 | Equal-weight universe | Momentum − equal weight | NIFTY 50 |
|---|---|---|---|---|
| 2017 (part) | +7.5% | +5.3% | **+2.2 pts** | +2.6% |
| 2018 | -11.1% | -13.4% | **+2.3 pts** | +3.3% |
| 2019 | +7.1% | +0.4% | **+6.6 pts** | +12.1% |
| 2020 | +71.6% | +38.7% | **+32.9 pts** | +14.7% |
| 2021 | +167.9% | +66.4% | **+101.5 pts** | +24.2% |
| 2022 | -3.9% | +13.6% | **-17.5 pts** | +4.3% |
| 2023 | +110.3% | +62.9% | **+47.5 pts** | +19.8% |
| 2024 | +60.1% | +36.2% | **+23.9 pts** | +8.8% |
| 2025 | +12.9% | +7.1% | **+5.9 pts** | +10.7% |
| 2026 (part) | +42.0% | +4.0% | **+38.0 pts** | -13.9% |

Momentum beat the equal-weight universe in **58%** of months (average monthly difference +1.52 pts).

## Reality check without survivorship bias (2022-08 → 2026-09)
A listed **Nifty200 Momentum 30 ETF** (MOMENTUM.NS) holds NSE's real momentum index — point-in-time members, after fund costs. Its gap over the Nifty 200 is what momentum actually delivered; compare it with the backtest's gap over its (survivor) universe in the same months.
| Same months | CAGR | Gap |
|---|---|---|
| Real momentum ETF | +11.8% | **+2.7 pts/yr** vs Nifty 200 (+9.1%) |
| Backtest 12-1 top 15 | +50.7% | **+22.7 pts/yr** vs its equal-weight universe (+28.0%) |

## Latest portfolio (12-1 top 15, for 2026-09)
CUPID, STLTECH, MTARTECH, ATHERENERG, THANGAMAYL, HFCL, SANSERA, KIRLOSENG, ACUTAAS, NETWEB, TDPOWERSYS, LAURUSLABS, ADANIENSOL, HINDCOPPER, AETHER
