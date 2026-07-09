# sm/scope-router — focused energy-pooling experiment

## TL;DR

**`cl_only` (individual crude model) beats `energy_all` on cl1s by +12pp AUC**, robust across models, with non-overlapping 1-SE confidence bands. **Dropping NG from the energy pool does not help** any of the other energy instruments. **Recommendation: for the final integration, route cl1s to an individual model and keep the remaining energy instruments pooled in `energy_all` (or `energy_cl_ho` for ho1s per Harry's PM-refresh).**

## The question

Does dropping `ng1s` from energy training improve validated AUC, and does `cl1s` benefit from individual training?

Four scopes tested:

| scope          | members                              |
|---             |---                                   |
| `energy_all`   | cl1s + ho1s + rb1s + ng1s (baseline) |
| `energy_ex_ng` | cl1s + ho1s + rb1s (drop NG)         |
| `cl_only`      | cl1s                                 |
| `ng_separate`  | ng1s                                 |

## Method

- Forked from `origin/Sreeram_experimental@99bea79`, no modification of the pipeline.
- Script: `src/stml/experimental/make_scope_router.py` (~150 lines). Imports and reuses `champion_pipeline._evaluate_candidate` — identical CPCV(6,2) + per-instrument embargo + sample-weighted per-fold AUC as `make_baseline.py`.
- In-process extension of `POOL_MEMBERS` only; no commit modification of `champion_pipeline.py`.
- Features: pre-computed `data/sreeram_experimental_features.parquet` (4886 events × 80 cols), restricted to modelling window `t_signal ≤ 2021-10-06` (3466 events).
- Default variant: `without_bbg` (Sreeram's baseline_xgb_per_class.csv shows without_bbg slightly beats with_bbg on all 3 classes, so we don't want to confound the scope question).
- Models: `lightgbm` (energy class winner per baseline_xgb_per_class.csv) and `xgboost` for robustness.

## Results (ledger excerpt)

`results/sm_scope_router/scope_router_ledger.csv` — full table written by the script. Headline rows:

| scope         | inst   | model    | n_modelling | n_oos | mean_auc   | sem    | lower_CI_1SE |
|---            |---     |---       |---:         |---:   |---:        |---:    |---:          |
| energy_all    | cl1s   | lightgbm | 804         | 1364  | **0.5651** | 0.0276 | 0.5375       |
| energy_all    | cl1s   | xgboost  | 804         | 1364  | 0.5581     | 0.0324 | 0.5257       |
| energy_all    | ho1s   | lightgbm | 804         | 169   | 0.5249     | 0.0456 | 0.4793       |
| energy_all    | rb1s   | lightgbm | 804         | 2071  | 0.5186     | 0.0189 | 0.4998       |
| energy_all    | ng1s   | lightgbm | 804         | 148   | 0.5734     | 0.0533 | 0.5201       |
| energy_ex_ng  | cl1s   | lightgbm | 774         | 1368  | 0.5546     | 0.0280 | 0.5266       |
| energy_ex_ng  | ho1s   | lightgbm | 774         | 169   | 0.5166     | 0.0513 | 0.4653       |
| energy_ex_ng  | rb1s   | lightgbm | 774         | 2075  | 0.5165     | 0.0161 | 0.5004       |
| **`cl_only`** | **cl1s** | **lightgbm** | **294** | **1372** | **0.6839** | **0.0225** | **0.6614** |
| **`cl_only`** | **cl1s** | **xgboost**  | **294** | **1372** | **0.6809** | **0.0234** | **0.6575** |
| ng_separate   | ng1s   | —        | 30          | 0     | SKIPPED    | —      | n_pool < 250 |

## Findings

### 1. CL1S BENEFITS MASSIVELY FROM INDIVIDUAL TRAINING — +12pp, robust across models

| route for cl1s | AUC (lightgbm)   | AUC (xgboost)    |
|---             |---:              |---:              |
| `energy_all`   | 0.5651           | 0.5581           |
| `energy_ex_ng` | 0.5546           | 0.5665           |
| **`cl_only`**  | **0.6839**       | **0.6809**       |

- Gap vs `energy_all`: **+11.88pp** (lightgbm) / **+12.28pp** (xgboost).
- 1-SE bands do NOT overlap: cl_only lower_CI = 0.6614 vs energy_all upper_CI (cl1s, lightgbm) = 0.5927.
- This is consistent with Harry's `champions_summary.csv` (cl1s/lightgbm@cl1s 0.671) and the prior `baseline_per_instrument.csv` (cl1s pooled 0.583).
- **Mechanism likely**: pooling crude with refined products (ho1s, rb1s, ng1s) dilutes the cl1s-specific signal. The 294 crude-only events carry sharper labels than the diluted 804-event pool.

### 2. DROPPING NG FROM THE ENERGY POOL DOES NOT HELP

| target  | `energy_all` lightgbm | `energy_ex_ng` lightgbm | Δ                |
|---      |---:                   |---:                     |---:              |
| cl1s    | 0.5651                | 0.5546                  | **−1.05pp**      |
| ho1s    | 0.5249                | 0.5166                  | −0.83pp          |
| rb1s    | 0.5186                | 0.5165                  | −0.21pp          |

- Removing NG slightly HURTS cl1s, ho1s, and rb1s (small, mostly within 1 SE).
- Asymmetric: NG itself benefits from being IN the pool (AUC 0.5734 in `energy_all`, vs `ng_separate` skipped for < MIN_INDIVIDUAL_EVENTS=250).
- The "NG contaminates energy training" hypothesis is REJECTED at this evidence level.

### 3. NG1S IS THIN — `ng_separate` IS NOT VIABLE

- Only 30 modelling events on `ng1s` alone (post 2021-10-06 cut), well below `MIN_INDIVIDUAL_EVENTS = 250`.
- Skipped automatically by `champion_pipeline._evaluate_candidate` guard.
- ng1s must remain pooled (best home: `energy_all`, AUC 0.5734).

### 4. ROBUSTNESS CHECK — MODEL-AGNOSTIC FINDING

- cl_only's headline finding holds under BOTH lightgbm (0.6839) and xgboost (0.6809). Different model families converge.
- Notably xgboost is BADLY MIS-CALIBRATED on ho1s in `energy_all` (AUC 0.3615 — strongly inverted!) and noisy on ng1s (0.4769). This confirms Sreeram's earlier finding that lightgbm is the energy class winner; xgboost is not a substitute for general energy modelling.
- For cl1s individual specifically, BOTH models give the same +12pp gap conclusion.

## Recommendation for final integration

**Per-instrument routing:**

| instrument | recommended route | model                       | rationale                                |
|---         |---                |---                          |---                                       |
| cl1s       | `cl_only`         | lightgbm or xgboost         | +12pp AUC vs pool; robust across models  |
| ho1s       | `energy_all` or `energy_cl_ho` | lightgbm (NOT xgboost) | xgboost inverted; lightgbm pool wins     |
| rb1s       | `energy_all`      | lightgbm                    | dropping NG doesn't help                 |
| ng1s       | `energy_all`      | lightgbm                    | too thin for individual; pool gives best |

This aligns with Sreeram's existing `INSTRUMENT_REGIMES` for ho1s, rb1s, ng1s, and PROMOTES `cl_only` over `energy_cl_ho` / `energy_all` for cl1s based on the +12pp evidence.

**Estimated class-level uplift if cl_only adopted for cl1s**: weighted by `n_oos`, the energy class champion mean would move from ≈ 0.553 (cl1s+ho1s+rb1s pooled) to ≈ 0.595 (cl1s individual). Same direction as Harry's pre-existing finding.

## Limitations and follow-ups

1. **One feature variant (`without_bbg`) only.** Should re-run under `with_bbg` for full ablation. Quick — ~14 s.
2. **Two models tested.** Should also try `random_forest` and `elasticnet_logistic` for symmetry with the existing roster.
3. **`MIN_INDIVIDUAL_EVENTS = 250`** is an inherited threshold from `champion_pipeline.py`. cl1s passes (294 events); ng1s does not (30 events).
4. **No `cl_ho` variant tested** (cl1s + ho1s pool, Harry's "energy_cl_ho"). Worth a single additional run.
5. **No `harry_router` test** — per-instrument champion routing using `INSTRUMENT_REGIMES`. Already implemented as `champion_pipeline.run_champion_for_instrument` — can be invoked directly via `make_champions.py`.
6. **Statistical significance is per-fold mean ± SEM**, not bootstrap CI on a held-out window. The 1-SE non-overlap is suggestive but a 95% CI would be stronger.
7. **No PBO / Deflated-Sharpe applied** here. The scope-router framework is configuration-space, not hyperparameter-space, so PBO is less directly applicable, but should be re-run on the final integrated config.

## Reproduce

```bash
# From a fresh clone:
git worktree add ../stml-wt/sm-scope-router -b sm/scope-router origin/Sreeram_experimental
cd ../stml-wt/sm-scope-router
uv sync --reinstall-package stml

# Headline run (single model):
uv run python -m stml.experimental.make_scope_router \
    --variant without_bbg --models lightgbm

# Robustness (lightgbm + xgboost):
uv run python -m stml.experimental.make_scope_router \
    --variant without_bbg --models lightgbm xgboost
```

Output: `results/sm_scope_router/scope_router_ledger.csv`. Total runtime: ~14 s for single model, ~25 s for both.

Branch tip: `sm/scope-router` from `origin/Sreeram_experimental@99bea79`.
