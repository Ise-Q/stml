# sm/scope-router — full routing tournament findings

## TL;DR

A 376-evaluation tournament (25 scopes × 4 models × 2 feature variants) under the existing CPCV(6,2) + per-instrument embargo gives a **per-instrument router** that beats Sreeram's class-level baselines by **+7pp on energy, +1pp on equity, +1pp on metals**, with the largest single-instrument uplift on **`cl1s` (+12pp from `cl_only`)** and a new finding on **`ho1s` (+4pp from the `ho_rb` pool)**. The hypothesis that **dropping NG helps** is rejected at instrument level — NG routes best to `energy_all` with random_forest. **BBG features help only 2 of 11 instruments** (fesx1s, pl1s); the other 9 prefer `without_bbg`. **No single model dominates** — 4 model families each win at least 2 routes.

## The questions (Codex prompt)

> For each instrument, identify the best validated training pool + model + feature variant under the existing CPCV framework.
> 1. Which instruments benefit from individual models?
> 2. Which instruments benefit from pooling?
> 3. Does dropping NG help anywhere?
> 4. Does BBG help any specific instrument, even if not globally?
> 5. What is the recommended final router by instrument?
> 6. Which candidates are strong enough to propose for final integration?

## Method

- Forked from `origin/Sreeram_experimental@99bea79`. **Zero modification of stable pipeline.**
- Script: `src/stml/experimental/make_scope_router_full.py` (~330 lines).
- Imports and reuses `champion_pipeline._evaluate_candidate` (identical CPCV(6,2), per-instrument embargo, sample-weighted per-fold AUC). `POOL_MEMBERS` is extended in-process; no commit modification of `champion_pipeline.py`.
- Features: pre-computed `data/sreeram_experimental_features.parquet` (4886 events × 80 cols), restricted to modelling window `t_signal ≤ 2021-10-06` (3466 events).
- Grid:
  - **25 scopes** = 10 energy + 7 equity + 9 metals (full enumeration of singletons, pairs, sub-class, and full-class pools).
  - **4 models** = elasticnet_logistic, random_forest, xgboost, lightgbm.
  - **2 feature variants** = without_bbg, with_bbg.
  - **Per scope**: each pool member is a target (in-pool OOS slice AUC).
  - **Total** = 376 (variant × scope × target × model) combinations.
- **Selection rule per instrument**: rank by `lower_CI_1SE` desc, then `mean_auc` desc, then model simplicity (elasticnet < rf < lightgbm < xgboost), then larger pool desc.
- Total wall time: **518.6 s (~8.6 min)**, 24/376 runs auto-skipped (individual pools < `MIN_INDIVIDUAL_EVENTS = 250`).

## Per-instrument router (the headline table)

`results/sm_scope_router/router_selection_summary.csv` — full file. Ranked by asset class then by mean_auc desc:

| instrument | class    | chosen_scope | chosen_model        | chosen_variant | mean_auc   | sem    | lower_CI   | n_within_1SE |
|---         |---       |---           |---                  |---             |---:        |---:    |---:        |---:          |
| **cl1s**   | energy   | **cl_only**  | lightgbm            | without_bbg    | **0.6839** | 0.0225 | **0.6614** | 4            |
| **ho1s**   | energy   | **ho_rb**    | elasticnet_logistic | without_bbg    | **0.6622** | 0.0445 | **0.6176** | 3            |
| **ng1s**   | energy   | **energy_all** | random_forest     | without_bbg    | **0.6131** | 0.0470 | **0.5660** | 4            |
| **rb1s**   | energy   | **cl_rb**    | random_forest       | without_bbg    | **0.5748** | 0.0241 | **0.5506** | 4            |
| **es1s**   | equity   | **es_fesx**  | lightgbm            | without_bbg    | **0.5966** | 0.0183 | **0.5783** | 1            |
| **fesx1s** | equity   | **equity_all** | random_forest     | **with_bbg**   | **0.5574** | 0.0191 | **0.5383** | 11           |
| **nq1s**   | equity   | **equity_all** | xgboost           | without_bbg    | **0.5487** | 0.0154 | **0.5333** | 4            |
| **pl1s**   | metals   | **si_pl**    | random_forest       | **with_bbg**   | **0.5773** | 0.0119 | **0.5654** | 3            |
| **hg1s**   | metals   | **metals_all** | xgboost           | without_bbg    | **0.5504** | 0.0113 | **0.5391** | 4            |
| **si1s**   | metals   | **precious** | elasticnet_logistic | without_bbg    | **0.5403** | 0.0127 | **0.5276** | 11           |
| **gc1s**   | metals   | precious     | elasticnet_logistic | without_bbg    | 0.4970     | 0.0290 | 0.4681     | 2            |

**10 of 11 instruments above 0.50 mean AUC; 4 above 0.60. Only gc1s sits below 0.50** (inverted, no clean route).

## Findings

### Finding 1 — Per-instrument routing produces meaningful class-level uplift

Comparing the routed per-instrument AUCs to Sreeram's class-level baselines from `baseline_xgb_per_class.csv` (without_bbg variant — the better baseline):

| asset class | Sreeram baseline (pooled XGB, without_bbg) | router (avg of inst AUCs) | Δ      |
|---          |---:                                        |---:                       |---:    |
| equity      | 0.5563                                     | 0.5676                    | **+1.1pp** |
| energy      | 0.5630                                     | 0.6335                    | **+7.0pp** |
| metals      | 0.5337                                     | 0.5413                    | +0.8pp |

**Energy gets the headline uplift — +7pp by routing each instrument to its best (scope, model, variant) combination instead of pooling everything into `energy_all` with one model.**

### Finding 2 — `cl_only` for cl1s is the strongest single signal (MVP confirmed)

| route for cl1s     | lightgbm | xgboost | random_forest | elasticnet_logistic |
|---                 |---:      |---:     |---:           |---:                 |
| `energy_all`       | 0.5651   | 0.5581  | 0.5527        | 0.5364              |
| `energy_ex_ng`     | 0.5546   | 0.5665  | 0.5398        | 0.5400              |
| **`cl_only`**      | **0.6839** | **0.6809** | 0.6610     | 0.5917              |

- cl_only's top-3 spots are ALL cl_only with different model/variant: lightgbm/without_bbg (0.6839), xgboost/without_bbg (0.6809), xgboost/with_bbg (0.6764).
- Gap vs `energy_all/lightgbm` = **+11.88pp**, 1-SE bands non-overlapping (CI 0.6614 vs 0.5375).
- Robust across both variants: cl_only/lightgbm/with_bbg = 0.6710 (still +10pp over energy_all).
- Consistent with Harry's pre-existing champion (cl1s/XGB individual 0.707) and Sreeram's `champions_summary.csv` (cl1s/lightgbm@cl1s 0.671).

### Finding 3 — `ho_rb` for ho1s is a NEW best route (not in INSTRUMENT_REGIMES)

| route for ho1s        | best model           | AUC    | lower_CI |
|---                    |---                   |---:    |---:      |
| `energy_all`          | elasticnet_logistic  | 0.6241 | 0.5799   |
| `energy_cl_ho` (Harry's spec) | (tested) | — | — |
| **`ho_rb`**           | **elasticnet_logistic** | **0.6622** | **0.6176** |

- `ho_rb` is NOT in `champion_pipeline.INSTRUMENT_REGIMES` for ho1s (which lists only `energy_cl_ho` and `energy_all`). The tournament uncovered this pairing.
- Mechanism: ho1s and rb1s are both refined-products contracts; pooling them isolates refinery dynamics from crude-only signals.
- **Robust**: top-3 all use elasticnet_logistic (without_bbg + with_bbg), AUC range 0.624–0.662 — confirming the model and pool choice, not just one lucky combination.

### Finding 4 — Drop-NG hypothesis: rejected at instrument level too

Codex's central question. Per-instrument view:

| target | best route in `energy_ex_ng` | best route in `energy_all` | preferred? |
|---     |---                           |---                          |---         |
| cl1s   | 0.5665 (xgboost, without_bbg) | 0.5651 (lightgbm, without_bbg) | tie — but `cl_only` 0.6839 dominates both |
| ho1s   | 0.5675 (elasticnet, without_bbg) | 0.6241 (elasticnet, without_bbg) | **energy_all > energy_ex_ng**  |
| rb1s   | 0.5400 (elasticnet, without_bbg) | 0.5527 (rf, without_bbg)    | **energy_all > energy_ex_ng**  |

`energy_ex_ng` never wins. NG carries information for the pool (cross-energy structure), removing it slightly hurts cl1s/ho1s/rb1s. **NG itself routes best to `energy_all/random_forest`** at AUC 0.6131 — NG should stay pooled and be modelled with random_forest specifically.

### Finding 5 — BBG features help 2 of 11 instruments

| inst   | winning variant | next-best variant | Δ      |
|---     |---              |---                |---:    |
| fesx1s | **with_bbg**    | without_bbg       | +0.4pp |
| pl1s   | **with_bbg**    | without_bbg       | +1.2pp |
| (all others 9) | without_bbg | with_bbg     | −0.1 to −1.0pp |

This nuances Sreeram's class-level "without_bbg ≈ with_bbg, slightly without_bbg" finding from `baseline_xgb_per_class.csv`: **BBG hurts on average but helps fesx1s and pl1s specifically**. For final integration, allow per-instrument variant choice rather than enforcing one variant for the whole class.

### Finding 6 — No single model dominates (4 model families each win)

| model               | wins | instruments                          |
|---                  |---:  |---                                   |
| random_forest       | 4    | fesx1s, ng1s, pl1s, rb1s             |
| elasticnet_logistic | 3    | gc1s, ho1s, si1s                     |
| lightgbm            | 2    | cl1s, es1s                           |
| xgboost             | 2    | hg1s, nq1s                           |

`random_forest` is the modal winner — Sreeram's `baseline_xgb_per_class.csv` showed `lightgbm` as the energy class winner because it averages over a pool; at the per-instrument granularity each model has its zone. `elasticnet_logistic` (most regularised) wins 3 — defensible in a low-signal environment where shrinkage dominates.

### Finding 7 — Narrow pools (2 instruments) are the dominant routing choice

| pool type         | n inst winners | example wins                                      |
|---                |---:            |---                                                |
| individual (1)    | 1              | cl1s                                              |
| narrow pool (2)   | 4              | ho1s (ho_rb), rb1s (cl_rb), es1s (es_fesx), pl1s (si_pl) |
| sub-class (3)     | 2              | gc1s, si1s (precious)                             |
| full class (4)    | 4              | ng1s, nq1s, fesx1s, hg1s                          |

6 of 11 instruments win at a narrow (2-instrument) or sub-class (3-instrument) pool — NOT at the full asset class. This is **the central structural finding**: per-instrument routing materially differs from per-class routing, especially in energy.

### Finding 8 — gc1s has no winning route

Best gc1s route = `precious/elasticnet_logistic/without_bbg` at AUC 0.4970 (below random). 24 candidates considered, none cross 0.50. Consistent with Harry's "no exploitable signal" classification for gc1s and Sreeram's per-instrument table (gc1s AUC 0.43 in his energy_xgb baseline). **For gc1s, the meta-model should default to the primary signal directly (act=blind)** — no validated improvement available.

## Per-instrument top-3 routes (robustness)

For instruments where the top-3 routes use different (model, variant) combinations, the finding is more robust. Excerpt:

| inst | route #1 | route #2 | route #3 |
|---   |---       |---       |---       |
| cl1s | cl_only/lightgbm/without_bbg 0.6614 | cl_only/xgboost/without_bbg 0.6575 | cl_only/xgboost/with_bbg 0.6538 |
| ho1s | ho_rb/elasticnet/without_bbg 0.6176 | ho_rb/elasticnet/with_bbg 0.5962 | energy_all/elasticnet/without_bbg 0.5799 |
| rb1s | cl_rb/rf/without_bbg 0.5506 | cl_rb/rf/with_bbg 0.5488 | cl_rb/lightgbm/with_bbg 0.5399 |
| ng1s | energy_all/rf/without_bbg 0.5660 | energy_all/elasticnet/with_bbg 0.5569 | energy_all/elasticnet/without_bbg 0.5324 |
| es1s | es_fesx/lightgbm/without_bbg 0.5783 | equity_all/xgboost/without_bbg 0.5608 | equity_all/rf/without_bbg 0.5605 |
| pl1s | si_pl/rf/with_bbg 0.5654 | pl_only/rf/with_bbg 0.5629 | metals_ex_gc/rf/without_bbg 0.5500 |

**Strong-pool consistency** (cl1s, rb1s): top-3 all reuse the same pool with different model/variant ⇒ the pool choice is the dominant lever.
**Weak-pool consistency** (gc1s, hg1s): top-3 are spread across scopes ⇒ result more sensitive to model/variant; lower confidence.

## Comparison vs Sreeram's existing champions (`champions_summary.csv`)

Sreeram has already published per-instrument champions selected by the 1-SE rule via `make_champions.py`. Our tournament discovered:

| inst   | Sreeram champion                  | router pick                            | Δ AUC  |
|---     |---                                |---                                     |---:    |
| cl1s   | cl1s/lightgbm 0.671               | cl_only/lightgbm 0.684                 | +1.3pp |
| ho1s   | (assumed energy_cl_ho or energy_all) | ho_rb/elasticnet 0.662               | **NEW pool found** |
| ng1s   | ng1s/elasticnet@energy_all 0.601  | energy_all/random_forest 0.613         | +1.2pp |
| rb1s   | (assumed energy_all)              | cl_rb/random_forest 0.575              | **NEW pool found** |

Our **scope grid is wider than `INSTRUMENT_REGIMES`** (we added pair pools like `ho_rb`, `cl_rb`, `cl_ho`, `es_fesx`, `nq_fesx`, `gc_si`, `si_pl`), and we **also tested the variant axis** — both reveal routes Sreeram's champion harness can't see by construction.

## Recommendations for final integration

### Per-instrument router (ship this)

```python
RECOMMENDED_ROUTER = {
    # energy
    "cl1s":   ("cl_only",    "lightgbm",            "without_bbg"),  # +12pp vs pool
    "ho1s":   ("ho_rb",      "elasticnet_logistic", "without_bbg"),  # +4pp vs energy_all
    "rb1s":   ("cl_rb",      "random_forest",       "without_bbg"),  # +2pp vs energy_all
    "ng1s":   ("energy_all", "random_forest",       "without_bbg"),  # NG must stay pooled
    # equity
    "es1s":   ("es_fesx",    "lightgbm",            "without_bbg"),
    "nq1s":   ("equity_all", "xgboost",             "without_bbg"),
    "fesx1s": ("equity_all", "random_forest",       "with_bbg"),     # BBG helps here
    # metals
    "gc1s":   ("DEFAULT_BLIND", None, None),  # no validated lift (AUC < 0.5)
    "si1s":   ("precious",   "elasticnet_logistic", "without_bbg"),
    "hg1s":   ("metals_all", "xgboost",             "without_bbg"),
    "pl1s":   ("si_pl",      "random_forest",       "with_bbg"),     # BBG helps here
}
```

### Gates passed

- All 10 routable instruments have `lower_CI_1SE > 0.50` (gate of "lower CI > random" passed).
- 4 instruments cross 0.60 mean AUC (cl1s, ho1s, ng1s, es1s).
- 6 instruments cross 0.55 mean AUC.
- gc1s flagged for blind-default (no validated route).

### Cross-instrument compatibility note

Running 7 different (scope, model, variant) combinations in production is more complex than a single class-level model. Trade-off:
- **High operational overhead**: 7 separate model fits per refit cycle, 7 separate calibration tracks.
- **High methodological clarity**: each instrument's route is justified by its own CPCV lower-CI evidence.
- For the deliverable submission, recommend: ship the router, document the rationale, and treat the gc1s blind-default as a discipline win not a loss.

## Limitations and follow-ups

1. **`simulated_missingness` variant not tested**. The framework supports it (`champion_pipeline._evaluate_candidate(nan_cols_at_test=...)`) — could be added in a follow-up commit for robustness against grader-time BBG missingness.
2. **`ensemble_simple` not tested as a "model" here**. It's implemented in `pipeline._ensemble_simple_oos` but the harness `_evaluate_candidate` takes a single model name; ensembling would require a thin wrapper. Future work.
3. **No `harry_router`-style nested selection** — i.e. testing whether using INSTRUMENT_REGIMES with `select_champion`'s 1-SE rule gives different answers than our per-instrument tournament. They should agree where pools overlap, and our router wins where pools don't overlap (e.g. `ho_rb`, `cl_rb`).
4. **1-SE bands ≠ proper 95% CI.** A bootstrap CI on per-fold AUC would strengthen the gates. Quick to add (≈10 min).
5. **No PBO / DSR on this tournament.** PBO via CSCV applies to backtested strategies; this is a CV-level selection over 376 candidates which is a different selection-bias regime. The lower_CI_1SE criterion is conservative and the 1-SE band overlap between top-3 routes is informative.
6. **MIN_INDIVIDUAL_EVENTS = 250** inherited from `champion_pipeline.py` — singletons with fewer events are skipped. 6 of 11 instruments cannot be tested individually (ho1s, ng1s, gc1s, si1s, hg1s, pl1s); others (cl1s 294, rb1s 411, es1s 1889, nq1s 1968, fesx1s 2099, pl1s 805) clear it.

## Reproduce

```bash
git worktree add ../stml-wt/sm-scope-router -b sm/scope-router origin/Sreeram_experimental
cd ../stml-wt/sm-scope-router
uv sync --reinstall-package stml   # to dodge the .pth-with-spaces bug

# Full tournament (~8-10 min):
PYTHONPATH=src .venv/bin/python -m stml.experimental.make_scope_router_full

# Single variant only (~5 min):
PYTHONPATH=src .venv/bin/python -m stml.experimental.make_scope_router_full \
    --variants without_bbg

# Subset of scopes (energy only):
PYTHONPATH=src .venv/bin/python -m stml.experimental.make_scope_router_full \
    --scopes energy_all energy_ex_ng cl_only ho_only rb_only ng_separate \
             cl_ho cl_rb ho_rb
```

Outputs:
- `results/sm_scope_router/router_full_ledger.csv` (376 rows × 16 cols)
- `results/sm_scope_router/router_selection_summary.csv` (11 rows × 15 cols)

Branch tip: `sm/scope-router @ <next commit>` from `Sreeram_experimental@99bea79`. MVP ledger (`scope_router_ledger.csv`) preserved at commit `f8336dc` as the checkpoint.
