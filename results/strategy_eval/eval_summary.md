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
| Ann return | +8.62% | +4.90% | +4.90% |
| Ann vol | 6.10% | 2.87% | 2.87% |
| Sharpe | 1.412 | 1.707 | 1.707 |
| t-stat | 1.19 | 1.44 | 1.44 |
| SR 95% CI low | -1.559 | -1.211 | -1.211 |
| SR 95% CI high | 3.750 | 3.892 | 3.892 |
| PSR(SR*=0) | 0.861 | 0.911 | 0.911 |
| Max DD | -3.42% | -1.41% | -1.41% |
| Total cost (bps) | 416.1 | 187.9 | 187.9 |
| Annual turnover | 48.28 | 21.80 | 21.80 |
| Breakeven ½-spread | 17.8 bps | 22.5 bps | 22.5 bps |
| Events taken | 100% | 54% | 54% |