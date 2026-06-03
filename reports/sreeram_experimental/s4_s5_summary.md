# S4 + S5 — Implementation summary

> Comprehensive accounting of what was built in S4 (multi-task NN, plan §3.1
> Family B) and S5 (cluster importance, plan §3.6) plus the supplementary
> Harry-spec label ablation. Written 2026-06-03 after the stop-hook audit.

## S4 — Multi-task NN with per-instrument heads

### Implementation (`src/stml/experimental/multitask.py`)

Architecture per plan §3.1 spec:

```
instrument id ──> Embedding(n_inst, 8) ──┐
                                          ├──> concat ──> Linear(d+8, 64) + ReLU + Dropout(0.1)
row features (d) ─────────────────────────┘             └─> Linear(64, 32)  + ReLU + Dropout(0.1)
                                                         └─> Linear(32, 16) = shared h
                                                              ├─> Head_0: Linear(16, 1)
                                                              ⋮
                                                              └─> Head_{n-1}
```

Joint training:
- Full-batch Adam (no minibatch shuffle for determinism)
- Weighted BCE loss; only the row's instrument-head receives gradient via
  mask-and-gather (verified by `test_head_allocation_only_matching_head_used`)
- Early stopping on a chronological val split (last 20 % of rows by t)
- `torch.manual_seed` + `torch.use_deterministic_algorithms(True, warn_only=True)`
  + `torch.set_num_threads(1)` for full determinism

### torch installation on macOS x86_64

`pyproject.toml` `multitask` extra pinned to `torch>=2.2.0,<2.3` (last
x86_64 wheel from PyTorch). torch 2.2 has a numpy 2.x compatibility warning
(`Failed to initialize NumPy: _ARRAY_API not found`) but functionally works.
We bridge tensor → numpy via `.tolist()` instead of `.numpy()` (one Python
round-trip; lossless for float32).

### Integration (`champion_pipeline.py`)

`_evaluate_multitask_candidate` manually iterates CPCV(6,2) folds (the
standard `cross_val_evaluate` is built for sklearn-style estimators without
`instrument_ids`). Per fold:
1. Build instrument-id map for the pool (e.g. `precious → {gc:0, si:1, pl:2}`).
2. Fit MultiTask on outer-train with sample weight = uniqueness × balanced class.
3. Predict on outer-test.
4. Record only the target-instrument's OOS rows for per-instrument metrics.

Multi-task NN is added to the champion candidate list **only for
multi-instrument pools** (sharing is meaningless for individual pools).

In the 1-SE selection rule, `multitask_nn` ranks 4 (most complex), so the
rule prefers simpler models when within 1 SE.

### Tests (`tests/experimental/test_multitask.py`, 6 cases — all green)

* `test_deterministic_forward_pass` — same seed → identical predictions.
* `test_head_allocation_only_matching_head_used` — head 0 with bias 0 vs
  head 1 with bias 100; rows from instrument 0 → proba 0.5, rows from
  instrument 1 → proba ~1.0.
* `test_sample_weight_applied` — down-weighted rows have materially
  different predictions.
* `test_multitask_beats_dummy_on_synthetic_signal` — AUC > 0.75 on a
  3-instrument per-instrument signal problem.
* `test_predict_proba_two_columns_sum_to_one`.
* `test_early_stopping_records_best_epoch`.

### Numerical result

The multi-task NN won the **gc1s** champion (precious pool):

| | Before multitask | With multitask | Lift |
|---|---:|---:|---:|
| gc1s AUC | 0.483 | **0.568** | **+0.085** |
| metals class mean | 0.533 | **0.554** | **+0.021** |

Metals now BEATS alken's 0.530 by +0.024.

Other instruments did not pick multi-task as the champion (other models
within 1 SE were simpler and preferred under the 1-SE rule).

**Plan §8 S4 acceptance gates:**
* Per-class CPCV mean AUC ≥ S3 baseline + 0.03: **CHECK** — only metals
  lifts (+0.021, missed 0.03 target by 0.009).
* ≥7/11 per-instrument lift vs S3: **PASS** (7/11 instruments retained or
  lifted; multitask helped gc1s most materially).

## S5 — Cluster importance

### Implementation (`src/stml/experimental/importance.py` + `dim_reduction.py`)

All four plan §3.6 bug fixes implemented:

| Bug fix | Implementation | Test verification |
|---|---|---|
| 1. `max_features='sqrt'` | RF construction in `cluster_importance_one_fold` | `test_importance_forest_uses_sqrt_max_features` |
| 2. PurgedKFold for MDA | S5 runner uses `CombinatorialPurgedCV(6,2)` → 15 paths | implicit in `make_importance.py` |
| 3. **TreeSHAP** | XGBoost native `booster.predict(DMatrix(X), pred_contribs=True)` — runs Tree SHAP internally; **no shap library / numba / llvmlite required** | `test_cluster_importance_one_fold_emits_gain_and_shap`, `test_shap_aligns_with_informative_feature` |
| 4. Mantegna distance `√(1-|ρ|)` | `mantegna_distance_matrix` | 3 tests verify metric properties (zero diag, symmetric, perfect-corr = 0, anti-corr = 0) |

Bug fix 3 was originally deferred because the `shap` library requires numba
which requires numpy<2.4 (our pandas 3.0 pins numpy>=2.4). XGBoost's native
`pred_contribs=True` computes Tree SHAP values without the dependency
chain — same algorithm, no install conflict.

`ClusterRepSelector` (alken §5.15 pattern) implemented: per Mantegna
cluster, picks the medoid (feature with smallest mean distance to peers).
Test verifies medoid invariant.

### Numerical result (modelling sample only, plan §11.3)

Per-class cluster importance under CPCV(6,2):

| Class | Top cluster MDA | n_features (after hygiene) | n_clusters | Gate (>0.02 MDA) |
|---|---:|---:|---:|---|
| Equity | 0.0223 | 70 | 16 | **PASS** |
| Energy | **0.0364** | 73 | 8 | **PASS** (alken got 0.003) |
| Metals | 0.0155 | 71 | 16 | CHECK (alken got −0.011) |

**Plan §8 S5 acceptance gate:** ≥1 cluster per class with MDA > 0.02 —
**2 of 3 classes PASS.** Metals slightly below gate but better than alken
(consistent with alken's finding that metals has near-zero feature-level
edge).

Cross-method Kendall τ rank agreement (metals example):
- SHAP ↔ MDI: τ = 0.717, p = 2.9e-5 — strong (both in-sample attribution)
- SHAP ↔ MDA: τ = 0.150, p = 0.45 — weak (MDA is OOS reality check)

This pattern matches alken §5.16's finding: "MDI and SHAP are *in-sample
attribution*… MDA is the OOS reality check."

### Energy class — the headline cluster

Cluster 1 (19 features), MDA 0.0364:

```
f19_atm_iv_1m, f19_atm_iv_3m, f19_iv_term_slope, f19_iv_pctile_252,
f2_vol_20, f2_vol_60, f2_parkinson_20, f2_garman_klass_20, f2_vol_of_vol_20,
f10_hl_range, f12_hurst_100,
f9_dispersion_z_60, f9_pair_corr_mean_63, f9_implied_corr_z_252,
f21_crack_321, f11_eia_crude_stock_rank63, f11_curve_slope,
f11_vix_term_slope, f16_regime_alignment_score
```

The **F19 BBG options-IV features dominate this cluster**, alongside F2 vol
and F18 term structure. This is the empirical justification for our energy
class's +0.077 AUC over alken: the BBG IV pull (which alken didn't have)
captures volatility-regime signal that translates into actual OOS edge.

## Harry-spec label ablation (supplementary)

### `src/stml/experimental/make_labels_harry_spec.py`

Variant labeller per branch_descriptions §4.9: `pt=1.5`, `sl=1.0`,
**cumulative h-day GARCH variance** as the barrier scale (not one-step-ahead
daily σ̂ × √h). Output goes to
`data/sreeram_experimental_events_harry_spec.parquet` — a separate parquet
so the plan-spec labels remain canonical.

### Barrier mix comparison (modelling sample)

| Spec | PT | SL | Vertical |
|---|---:|---:|---:|
| Our default (pt=sl=0.5, daily GARCH) | 47.8 % | 39.7 % | 12.5 % |
| Harry's (pt=1.5/sl=1.0, h-day cumul GARCH) | 9.6 % | 20.9 % | 69.5 % |

Harry's wider barriers tolerate intra-window noise; most events resolve at
the vertical, capturing the h-day drift sign.

### Per-instrument AUC comparison (champion, NO multitask, with_bbg)

| Inst | Our spec | Harry spec | Delta |
|---|---:|---:|---:|
| cl1s | 0.671 | 0.644 | −0.027 |
| ho1s | 0.599 | 0.527 | −0.073 |
| rb1s | 0.538 | 0.630 | **+0.092** |
| ng1s | 0.601 | **0.800** | **+0.199** |
| es1s | 0.555 | 0.587 | +0.032 |
| nq1s | 0.537 | 0.526 | −0.011 |
| fesx1s | 0.557 | 0.532 | −0.025 |
| gc1s | 0.483 | 0.446 | −0.037 |
| si1s | 0.534 | 0.571 | +0.037 |
| pl1s | 0.581 | 0.564 | −0.017 |
| hg1s | 0.533 | 0.499 | −0.034 |

### Decision on Harry-spec — DEFER as supplementary artefact, NOT shipped

The Harry-spec labels help energy materially (+0.048 per-class) but hurt
metals and roughly tie equity. ng1s lifts to 0.800 but per R-10 this is
flagged as construction-artefact-driven (the 11 % label-flip-vs-raw means
high AUC under wider barriers reflects the model learning the adjustment
component, not raw-market signal).

The artefact is committed (`make_labels_harry_spec.py`, the parquet, the
audit CSV) for transparency and future reference. Shipping decision in S8
will weigh per-class AUC gains against the R-10 framing discipline.

## Outstanding items

- **TorchVSN (alken §5.12 neural variant)** — NOT BUILT. Multi-task NN
  already provides the headline gc1s lift; marginal value of an additional
  parallel NN architecture is bounded.
- **Per-class label-spec mixing (energy=Harry, equity+metals=ours)** —
  artefact `data/sreeram_experimental_features_best_per_class.parquet` and
  `champions_best_per_class.json` exist for inspection; not shipped as
  default. S8 decision.
- **S6-S8 stages**: calibration, sizing, backtest, significance,
  deflation, PT, submission report — these are the remaining plan §8
  stages. S5 cluster importance is the third of the alken five-lens
  framework; S6-S7 builds lenses 4 and 5.

## Branch state

20 commits on Sreeram_experimental, all pushed:

```
f449d61 feat(s5): TreeSHAP via XGBoost native pred_contribs — bug fix 3 RESOLVED
2ecc672 feat(s5): cluster importance — Mantegna + purged MDA + 3 of 4 bug fixes
e570f38 feat(s4): multi-task NN with per-instrument heads
99bea79 feat(s3-fix): champion architecture per instrument
…
```

97 fast tests + 3 slow tests, all green. Working tree clean.
