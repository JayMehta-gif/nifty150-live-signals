# Backtest report

Costs: **0.122% per round trip** (brokerage, STT, exchange/SEBI fees, GST, stamp, 0.02% slippage per side). Notional ₹100,000 per trade. 'Avg net' is per trade after costs.

## 1. 15-minute bars (the live system) — 2026-07-28 → 2026-10-14

150 stocks. Unseen period: **2026-07-28 → 2026-09-24 (42 sessions)**, never looked at while designing. Seen period: 13 sessions.

### Unseen sessions (the real test)
| Rules | Trades | /session | Win | Avg gross | **Avg net** | Profit factor | Total net (₹ at ₹1L/trade) | Max drawdown | Green days |
|---|---|---|---|---|---|---|---|---|---|
| NARROW (pre-registered) | 259 | 6.2 | 47% | +0.04% | **-0.09%** | 0.79 | -22.1% (₹-22,082) | -32.8% | 37% |
| … long only | 124 | 3.0 | 41% | -0.03% | **-0.15%** | 0.67 | -18.6% (₹-18,623) | -24.0% | 29% |
| … short only | 135 | 3.2 | 52% | +0.10% | **-0.03%** | 0.93 | -3.5% (₹-3,460) | -13.4% | 48% |
| … no time filter | 513 | 12.2 | 44% | +0.04% | **-0.08%** | 0.76 | -42.3% (₹-42,252) | -58.2% | 31% |
| … no stage gates | 824 | 19.6 | 46% | +0.01% | **-0.11%** | 0.74 | -90.2% (₹-90,154) | -111.7% | 26% |
| … 2H gate only | 392 | 9.3 | 48% | +0.04% | **-0.08%** | 0.80 | -31.9% (₹-31,859) | -44.8% | 40% |
| … target 0.5% / stop 0.5% | 259 | 6.2 | 46% | -0.04% | **-0.16%** | 0.50 | -41.4% (₹-41,450) | -42.2% | 24% |
| … target 1.5% / stop 1% | 259 | 6.2 | 44% | +0.06% | **-0.06%** | 0.86 | -15.2% (₹-15,226) | -35.0% | 41% |
| … no target/stop (exit 3:15 PM) | 259 | 6.2 | 46% | +0.08% | **-0.04%** | 0.92 | -11.0% (₹-10,973) | -41.0% | 44% |
| BASELINE: every raw signal | 4574 | 108.9 | 43% | -0.01% | **-0.13%** | 0.66 | -607.7% (₹-607,713) | -650.4% | 17% |

### Seen sessions (used while designing — expect these to look better)
| Rules | Trades | /session | Win | Avg gross | **Avg net** | Profit factor | Total net (₹ at ₹1L/trade) | Max drawdown | Green days |
|---|---|---|---|---|---|---|---|---|---|
| NARROW (pre-registered) | 88 | 6.8 | 62% | +0.25% | **+0.13%** | 1.41 | +11.5% (₹11,493) | -4.6% | 54% |
| … long only | 12 | 0.9 | 50% | +0.03% | **-0.09%** | 0.75 | -1.1% (₹-1,057) | -1.2% | 38% |
| … short only | 76 | 5.8 | 64% | +0.29% | **+0.17%** | 1.53 | +12.5% (₹12,550) | -3.4% | 73% |
| … no time filter | 185 | 14.2 | 49% | +0.09% | **-0.03%** | 0.90 | -6.2% (₹-6,249) | -16.3% | 31% |
| … no stage gates | 242 | 18.6 | 57% | +0.16% | **+0.03%** | 1.08 | +7.9% (₹7,939) | -19.0% | 46% |
| … 2H gate only | 122 | 9.4 | 60% | +0.22% | **+0.10%** | 1.29 | +12.5% (₹12,472) | -9.1% | 54% |
| … target 0.5% / stop 0.5% | 88 | 6.8 | 59% | +0.10% | **-0.02%** | 0.91 | -1.9% (₹-1,882) | -3.7% | 46% |
| … target 1.5% / stop 1% | 88 | 6.8 | 57% | +0.26% | **+0.13%** | 1.37 | +11.7% (₹11,674) | -4.6% | 54% |
| … no target/stop (exit 3:15 PM) | 88 | 6.8 | 62% | +0.34% | **+0.22%** | 1.68 | +18.9% (₹18,933) | -6.1% | 46% |
| BASELINE: every raw signal | 1465 | 112.7 | 49% | +0.07% | **-0.05%** | 0.87 | -76.1% (₹-76,148) | -112.4% | 31% |

## 2. Robustness: same rules on 1-hour candles — 2023-11-07 → 2026-10-14

Hourly signals differ from 15-min ones, so this tests whether the *idea* holds across ~2 years of bull, bear and sideways markets — not the exact live signals. Signal candle must close by 11:15 AM (the 10:15 candle — ADX DI never fires on the opening candle), entry at the next candle's open, exit by the 2:15 PM candle's close.

| Rules | Trades | /session | Win | Avg gross | **Avg net** | Profit factor | Total net (₹ at ₹1L/trade) | Max drawdown | Green days |
|---|---|---|---|---|---|---|---|---|---|
| NARROW (pre-registered) | 838 | 1.2 | 47% | +0.01% | **-0.11%** | 0.75 | -94.0% (₹-93,997) | -116.1% | 43% |
| … long only | 486 | 0.7 | 46% | +0.00% | **-0.12%** | 0.72 | -58.7% (₹-58,743) | -75.3% | 41% |
| … short only | 352 | 0.5 | 47% | +0.02% | **-0.10%** | 0.78 | -35.3% (₹-35,254) | -43.3% | 46% |
| … no time filter | 1747 | 2.4 | 45% | +0.02% | **-0.10%** | 0.73 | -177.9% (₹-177,917) | -197.0% | 41% |
| … no stage gates | 2151 | 3.0 | 46% | -0.00% | **-0.13%** | 0.72 | -269.7% (₹-269,696) | -286.4% | 41% |
| … 2H gate only | 1714 | 2.4 | 46% | +0.00% | **-0.12%** | 0.73 | -209.4% (₹-209,412) | -228.0% | 42% |
| … target 0.5% / stop 0.5% | 838 | 1.2 | 44% | -0.05% | **-0.17%** | 0.49 | -142.9% (₹-142,905) | -144.8% | 40% |
| … target 1.5% / stop 1% | 838 | 1.2 | 44% | +0.04% | **-0.09%** | 0.81 | -72.2% (₹-72,220) | -99.6% | 42% |
| … no target/stop (exit 3:15 PM) | 838 | 1.2 | 46% | +0.02% | **-0.11%** | 0.80 | -88.5% (₹-88,451) | -115.4% | 45% |
| BASELINE: every raw signal | 27144 | 37.5 | 45% | +0.01% | **-0.11%** | 0.72 | -3100.2% (₹-3,100,163) | -3122.3% | 26% |

### NARROW by half-year (1-hour test)
| Period | Trades | Win | Avg net | Total net |
|---|---|---|---|---|
| 2023-H2 | 52 | 56% | +0.13% | +7.0% |
| 2024-H1 | 118 | 48% | -0.13% | -15.5% |
| 2024-H2 | 129 | 47% | -0.12% | -15.1% |
| 2025-H1 | 162 | 53% | +0.01% | +2.1% |
| 2025-H2 | 127 | 44% | -0.15% | -18.9% |
| 2026-H1 | 171 | 40% | -0.24% | -41.3% |
| 2026-H2 | 79 | 41% | -0.16% | -12.4% |

### Short only by half-year (1-hour test)
| Period | Trades | Win | Avg net | Total net |
|---|---|---|---|---|
| 2024-H1 | 36 | 36% | -0.39% | -14.2% |
| 2024-H2 | 58 | 52% | -0.06% | -3.4% |
| 2025-H1 | 77 | 56% | +0.11% | +8.5% |
| 2025-H2 | 52 | 50% | -0.12% | -6.0% |
| 2026-H1 | 95 | 40% | -0.24% | -22.5% |
| 2026-H2 | 34 | 50% | +0.07% | +2.2% |
