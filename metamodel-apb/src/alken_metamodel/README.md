# `alken_metamodel` — package module map

The meta-labelling metamodel package. Entry point: `python -m alken_metamodel.emit` →
`emit.main()` (see the [subproject README](../../README.md) for quickstart and the external I/O
boundary).

**Fold-safety contract.** Every feature is built by recomputing `stml`'s *causal* feature functions
on each CV fold's train slice; fitted blocks (regime / HMM / reducer / calibrator) are fit on
fold-train only. The frozen `results/feature_matrix.parquet` is never read — enforced by a guard
test. See [`../../docs/methodology.md`](../../docs/methodology.md) for the full rationale.

Modules are listed below in **pipeline data-flow order**, not alphabetically.

## Data & features

| Module | Purpose |
| :---: | :---: |
| `features.py` | Per-instrument feature assembly for the meta-labelling metamodel (Stage 2). |
| `regime.py` | Regime features (Stage 2, commitment #8 / nlr-cw §4) — online EWMA-HMM + stml static F3/F17. |
| `macro.py` | Point-in-time-lagged macro block for §1/§3 (S1.7). |
| `volatility.py` | OHLC range-based volatility estimators: Garman-Klass, Parkinson, Rogers-Satchell. |
| `triple_barrier.py` | Triple-barrier meta-labelling (López de Prado 2018, Ch.3-4) — `t1`, uniqueness weights. |

## Modelling & validation

| Module | Purpose |
| :---: | :---: |
| `cross_validation.py` | Leakage-safe cross-validation: purged k-fold, CPCV, nested CPCV. |
| `models.py` | Act/skip model roster for the meta-labelling horse-race (Stage 2) — elastic-net, XGBoost, LightGBM. |
| `neural.py` | Neural act/skip variants for the §3 horse-race (Stage 2 enrichment) — torch-MLP, torch-VSN. |
| `dim_reduction.py` | Dimensionality reduction for the NN horse-race (EX.2) — cluster-representative medoids. |
| `evaluation.py` | Out-of-sample evaluation harness for the meta-labelling horse-race (Stage 2, §5). |
| `calibration.py` | Probability calibration for EX.4 (reliability curve, ECE, Platt, isotonic). |
| `cluster_importance.py` | Cluster-level feature importance for §4 (MDI + purged MDA + cluster SHAP). |
| `signal_analysis.py` | Primary-signal characterisation for EX.5 (the metamodel's ceiling). |

## Strategy / §6 (bonus)

| Module | Purpose |
| :---: | :---: |
| `sizing.py` | Position sizing: fractional Kelly + volatility targeting (Section 6 bonus). |
| `cost_model.py` | Transaction-cost model for the §6 backtest (S6.7; Grinold–Kahn half-spread + impact). |
| `backtest.py` | Strategy backtest for the §6 bonus track (Carver 2015 vol-targeting context, nlr-cw §7). |
| `deflation.py` | Backtest deflation for the §6 deployment gate (S6.8) — DSR / MinBTL / CSCV-PBO. |
| `significance.py` | Significance-first inference for §6 (S6.14, LR-6) — Sharpe t-stat, bootstrap CI, PSR. |

## Orchestration & infrastructure

| Module | Purpose |
| :---: | :---: |
| `pipeline.py` | End-to-end orchestration: one meta-labelling metamodel per asset class (Stage 5). |
| `emit.py` | Deterministic CSV emitters, §6 strategy sizing, and the CLI entry point (Stage 5). |
| `experiment_log.py` | Unified experiment log (XT.2): one deterministic row per run. |
| `seeding.py` | Global determinism control (`random` / `numpy` / `torch` / `tensorflow` / `PYTHONHASHSEED`). |
| `_env.py` | Pin OpenMP/BLAS threading before native libraries initialise (also fixes a macOS libomp segfault). |
| `__init__.py` | Alken meta-labelling metamodel (T3.03 coursework) — imports `_env` first. |

## `_vendor/` — verbatim sts-ml scripts

Copied byte-identical from the sts-ml course archive so the subproject runs cold; the **four
Stage-3 bug fixes** are applied on top in a separate commit so each is visible in the diff
(provenance + patch log in [`_vendor/__init__.py`](_vendor/__init__.py)).

| Module | Role | Stage-3 fixes |
| :---: | :---: | :---: |
| `vsn.py` | Keras Variable Selection Network (GLU / GRN / VSN) — the off-path KerasVSN comparison | — |
| `cluster_feature_importance.py` | `OptimalClusterer` + Mantegna distance + cluster MDI/MDA | #2 injected `PurgedKFold`; #4 Mantegna `√(1−\|ρ\|)` |
| `trend_scanning.py` | `tValLinR`, `trend_labels` — reused as a **backward feature**, never the label | — |
| `regression_metrics.py` | MAE / MSE / RMSE / MAPE / R² report | — |

Bug fixes **#1** (`max_features='auto'` → `'sqrt'`) and **#3** (real cluster SHAP via `TreeExplainer`)
live in `cluster_importance.py` (the §4 module), not in `_vendor/`.
