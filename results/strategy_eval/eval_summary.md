# Strategy evaluation — OOS results

OOS period: 2021-10-21 → 2022-06-29
Vol: ewma_close(span=60)  σ_tgt=10%  max_lev=10.0

## Lecture conventions (StrategyWeights slides 38–43)
- Returns: simple  r_t = (P_t − P_{t-1}) / P_{t-1}
- EWMA vol: λ=2/(span+1), exact lecture recurrence, ×√252, floor=2%
- Weight: w_t,k = ŷ_t,k × σ_tgt / σ̂_t,k
- Lag: w_t earns r_{t+1} (position at close of t, return from t→t+1)
- Aggregate: R^port = (1/K) Σ_k w_t,k r_{t+1,k}  (K=11, flat=cash)
- Costs: 2bps half-spread + 10bps×|Δw| Grinold-Kahn

## Results

| Metric | A Benchmark | B all_or_nothing | B SOPS |
| ---|---|---|--- |
| Ann return | +8.62% | +4.65% | +4.65% |
| Ann vol | 6.10% | 2.88% | 2.88% |
| Sharpe | 1.412 | 1.614 | 1.614 |
| t-stat | 1.19 | 1.36 | 1.36 |
| SR 95% CI low | -1.559 | -1.286 | -1.286 |
| SR 95% CI high | 3.750 | 3.821 | 3.821 |
| PSR(SR*=0) | 0.861 | 0.899 | 0.899 |
| Max DD | -3.42% | -1.49% | -1.49% |
| Total cost (bps) | 416.1 | 188.4 | 188.4 |
| Annual turnover | 48.28 | 21.86 | 21.86 |
| Breakeven ½-spread | 17.8 bps | 21.3 bps | 21.3 bps |
| Events taken | 100% | 54% | 54% |