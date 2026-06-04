# Strategy evaluation — methods A–D

OOS period: 2021-10-21 → 2022-06-29
Vol method: ewma_close  σ_tgt=10%  max_lev=10.0

## Section 1: Method comparison (Yang-Zhang vol)

| Metric | A Benchmark | B model_confidence | B all_or_nothing | B ncdf |
| ---|---|---|---|--- |
| Ann return | +8.62% | +2.76% | +4.90% | +2.70% |
| Ann vol | 6.10% | 1.63% | 2.87% | 1.60% |
| Sharpe | 1.412 | 1.688 | 1.707 | 1.693 |
| t-stat | 1.19 | 1.43 | 1.44 | 1.43 |
| SR 95% CI low | -1.559 | -1.239 | -1.211 | -1.226 |
| SR 95% CI high | 3.750 | 3.846 | 3.892 | 3.856 |
| PSR(SR*=0) | 0.861 | 0.908 | 0.911 | 0.909 |
| Max DD | -3.42% | -0.82% | -1.41% | -0.80% |
| Total cost (bps) | 416.1 | 106.3 | 187.9 | 103.8 |
| Annual turnover | 48.28 | 12.33 | 21.80 | 12.05 |
| Breakeven ½-spread | 17.8 bps | 22.3 bps | 22.5 bps | 22.4 bps |
| Events taken | 100% | 54% | 54% | 54% |

## Notes
- All Sharpe ratios are annualised (×√252).
- Bootstrap CI: Politis-Romano stationary block bootstrap (n_boot=2000).
- Transaction costs: 2bps half-spread + 10bps×|Δw| Grinold-Kahn impact.
- Vol targeting: Yang-Zhang(20-bar) annualised σ̂, σ_tgt=10%, max_lev=10×.
- C-VSN+LSTM / D-TFT: trained on pre-BOUNDARY OOF data; best checkpoint by val Sharpe.

## Section 2: Vol estimator comparison (methods A and B-aon)

| Metric | A Benchmark / Yang-Zhang (default) | A Benchmark / EWMA(60) | A Benchmark / GARCH(1,1) | A Benchmark / GJR-GARCH (equity only) | B all_or_nothing / Yang-Zhang (default) | B all_or_nothing / EWMA(60) | B all_or_nothing / GARCH(1,1) | B all_or_nothing / GJR-GARCH (equity only) |
| ---|---|---|---|---|---|---|---|--- |
| Ann return | +7.93% | +8.62% | +7.56% | +7.49% | +3.92% | +4.90% | +3.99% | +3.82% |
| Ann vol | 5.50% | 6.10% | 5.84% | 5.72% | 2.59% | 2.87% | 2.76% | 2.65% |
| Sharpe | 1.441 | 1.412 | 1.295 | 1.311 | 1.514 | 1.707 | 1.446 | 1.440 |
| t-stat | 1.22 | 1.19 | 1.09 | 1.11 | 1.28 | 1.44 | 1.22 | 1.22 |
| Max DD | -3.12% | -3.42% | -3.31% | -3.21% | -1.29% | -1.41% | -1.43% | -1.33% |

### Vol estimators
- **Yang-Zhang**: bias-minimised OHLC estimator, window=20 bars.
- **EWMA(60)**: exponentially weighted close-to-close returns, span=60.
- **GARCH(1,1)**: refitted every 21 bars on up to 2000 bars of history.
- **GJR-GARCH**: asymmetric GARCH for equity instruments (es1s, nq1s, fesx1s);   GARCH(1,1) for commodity instruments.