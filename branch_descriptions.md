# Branch Descriptions — `Sreeram`, `Harry`, and `model/alken-metamodel`

> A thorough catalogue of what's on each branch: data, labels, features, models, CV machinery, evaluation, importance analysis, outputs, and open issues. Written as a single source of truth for the team‑synthesis decision (one branch, several, or a combined deliverable).
>
> Author: Sreeram (read‑only audit of all three branches, 2026‑06‑02). The Harry branch was inspected via a temporary worktree at `/tmp/stml-harry/` and the `model/alken-metamodel` branch via `/tmp/stml-alken/`, so the working tree on `Sreeram` was never touched. The `main` branch is the shared spine all three feature branches build on; it is described first because they all re‑use its loaders and missing‑data handling without modification. The `model/alken-metamodel` branch additionally adds a consolidated feature library that itself extends the shared spine — see [§5.2](#52-the-shared-feature-library-stmlmetamodel).

---

## Table of Contents

1. [Assignment recap](#1-assignment-recap)
2. [Shared spine (`main`)](#2-shared-spine-main)
3. [Branch `Sreeram` — full pipeline v0 → v5](#3-branch-sreeram--full-pipeline-v0--v5)
   1. [Layout](#31-layout)
   2. [Stage 1a — Triple‑barrier labels](#32-stage-1a--triple-barrier-labels-srcstmllabelingpy)
   3. [Stage 1b — Purged CV and embargo](#33-stage-1b--purged-cv-and-embargo-srcstmlcvpy)
   4. [Stage 2 — Features G1–G8](#34-stage-2--features-g1g8-srcstmlfeaturespy)
   5. [Stage 3 — HMM/GMM regimes](#35-stage-3--hmmgmm-regimes-srcstmlregimespy)
   6. [Stage 3 — Model families](#36-stage-3--model-families-srcstmlmodelspy)
   7. [Stage 4 — Cluster‑level feature importance](#37-stage-4--cluster-level-feature-importance-srcstmlimportancepy)
   8. [Stage 5 — Evaluation](#38-stage-5--evaluation-srcstmlevaluationpy)
   9. [Pipelines: `pipeline.py`, `experiments.py`](#39-pipelines-pipelinepy-experimentspy)
   10. [Forensic diagnostics (`_diag1` → `_diag7`)](#310-forensic-diagnostics-_diag1--_diag7)
   11. [v3 — Commodity‑only XGBoost](#311-v3--commodity-only-xgboost-srcstmlbest_strategypy)
   12. [v4 — Stacked conditional ensemble](#312-v4--stacked-conditional-ensemble-srcstmlv4py)
   13. [v5 — Principled rebuild](#313-v5--principled-rebuild-srcstmlv5py)
   14. [Strategy track (`strategy.py`)](#314-strategy-track-srcstmlstrategypy)
   15. [Tests](#315-tests-tests)
   16. [Output inventory](#316-output-inventory)
   17. [Open issues on Sreeram](#317-open-issues-on-sreeram)
4. [Branch `Harry` — independent audit + reconciled pipeline](#4-branch-harry--independent-audit--reconciled-pipeline)
   1. [Branch philosophy](#41-branch-philosophy-reportsharry00-contextmd)
   2. [Step 1 — Signal‑direction audit](#42-step-1--signal-direction-audit-srcstmlharrysignal_auditpy)
   3. [Step 2 — Triple‑barrier labels (t+1 entry)](#43-step-2--triple-barrier-labels-t1-entry-srcstmlharrylabelspy)
   4. [Step 3a‑3g — Feature pack](#44-step-3a3g--feature-pack-srcstmlharryfeatures)
   5. [Step 3.5b — Macro features (M1–M6)](#45-step-35b--macro-features-m1m6-srcstmlharryfeaturesmacro_featurespy)
   6. [Alternate data (Bloomberg‑style)](#46-alternate-data-bloomberg-style-dataalternate_data_cleanedcsv)
   7. [`new_work` — HMM volatility (Core #1)](#47-new_work--hmm-volatility-core-1-srcstmlnew_workhmm_volpy)
   8. [`new_work` — HMM macro (Core #2)](#48-new_work--hmm-macro-core-2-srcstmlnew_workhmm_macropy)
   9. [`new_work` — GARCH triple‑barrier](#49-new_work--garch-triple-barrier-srcstmlnew_worktriple_barrierpy)
   10. [`new_work` — CPCV barrier search](#410-new_work--cpcv-barrier-search-srcstmlnew_workcpcv_searchpy)
   11. [`new_work` — Model comparison harness](#411-new_work--model-comparison-harness-srcstmlnew_workmodel_comparisonpy)
   12. [`new_work` — Cluster importance](#412-new_work--cluster-importance-srcstmlnew_workfeature_importancepy)
   13. [`new_work` — Champion importance](#413-new_work--champion-importance-srcstmlnew_workchampion_importancepy)
   14. [`new_work` — Reconciliation](#414-new_work--reconciliation-srcstmlnew_workreconciliationpy)
   15. [Notebooks](#415-notebooks-notebooksharry-srcstmlnew_work)
   16. [Tests](#416-tests-testsharry)
   17. [Output inventory](#417-output-inventory)
   18. [Open issues on Harry](#418-open-issues-on-harry)
   19. [2026‑06‑02 PM methodology refresh — clean split + 1SE + all‑11 champion importance](#419-the-2026-06-02-pm-methodology-refresh--clean-trainsplit-test-split--1se-selection--all-11-champion-importance)
5. [Branch `model/alken-metamodel` — methodology‑first per‑class metamodels](#5-branch-modelalken-metamodel--methodology-first-per-class-metamodels)
   1. [Branch shape — two coexisting projects](#51-branch-shape--two-coexisting-projects)
   2. [The shared feature library (`stml.metamodel`)](#52-the-shared-feature-library-stmlmetamodel)
   3. [The nested subproject (`metamodel-apb/`) — Stage 0 scaffold](#53-the-nested-subproject-metamodel-apb--stage-0-scaffold)
   4. [Stage 1 — Volatility (`volatility.py`)](#54-stage-1--volatility-volatilitypy)
   5. [Stage 1 — Triple‑barrier labels (`triple_barrier.py`)](#55-stage-1--triple-barrier-labels-triple_barrierpy)
   6. [Stage 1 — Cross‑validation (`cross_validation.py`)](#56-stage-1--cross-validation-cross_validationpy)
   7. [Stage 1 — Sizing (`sizing.py`)](#57-stage-1--sizing-sizingpy)
   8. [Stage 1.7 — PIT‑lagged macro block (`macro.py`)](#58-stage-17--pit-lagged-macro-block-macropy)
   9. [Stage 2 — Per‑instrument feature assembly (`features.py`)](#59-stage-2--per-instrument-feature-assembly-featurespy)
   10. [Stage 2 — Online EWMA HMM regimes (`regime.py`)](#510-stage-2--online-ewma-hmm-regimes-regimepy)
   11. [Stage 2 — Model roster (`models.py`)](#511-stage-2--model-roster-modelspy)
   12. [Stage 2 — Neural variants (`neural.py`)](#512-stage-2--neural-variants-neuralpy)
   13. [Stage 2 — Evaluation harness (`evaluation.py`)](#513-stage-2--evaluation-harness-evaluationpy)
   14. [Stage 2/EX.4 — Calibration (`calibration.py`)](#514-stage-2ex4--calibration-calibrationpy)
   15. [Stage 2/EX.2 — Dimensionality reduction (`dim_reduction.py`)](#515-stage-2ex2--dimensionality-reduction-dim_reductionpy)
   16. [Stage 3/4 — Cluster importance (`cluster_importance.py`)](#516-stage-34--cluster-importance-cluster_importancepy)
   17. [Stage 5 — End‑to‑end pipeline (`pipeline.py`)](#517-stage-5--end-to-end-pipeline-pipelinepy)
   18. [Stage 5 — Deterministic emit (`emit.py`)](#518-stage-5--deterministic-emit-emitpy)
   19. [Stage 6 — Backtest (`backtest.py`, `cost_model.py`)](#519-stage-6--backtest-backtestpy-cost_modelpy)
   20. [Stage 6 — Significance and deflation (`significance.py`, `deflation.py`)](#520-stage-6--significance-and-deflation-significancepy-deflationpy)
   21. [Stage 5/6 — Signal analysis (`signal_analysis.py`)](#521-stage-56--signal-analysis-signal_analysispy)
   22. [Determinism + leakage discipline (`seeding.py`, `_env.py`, `_vendor/`)](#522-determinism--leakage-discipline-seedingpy-_envpy-_vendor)
   23. [Experiments (EX.1, EX.3, EX.4, EX.5, S3, S4, S6.7, S6.8, X.8)](#523-experiments-ex1-ex3-ex4-ex5-s3-s4-s67-s68-x8)
   24. [The five passes (pass 1 → pass 5)](#524-the-five-passes-pass-1--pass-5)
   25. [Shipped configuration and the deliverable](#525-shipped-configuration-and-the-deliverable)
   26. [Numerical results](#526-numerical-results)
   27. [Tests](#527-tests)
   28. [Documentation and the academic report](#528-documentation-and-the-academic-report)
   29. [Open issues on alken-metamodel](#529-open-issues-on-alken-metamodel)
6. [Side‑by‑side comparison — three branches](#6-side-by-side-comparison--three-branches)
7. [Decisions still to make](#7-decisions-still-to-make)

---

## 1. Assignment recap

Imperial **BUSI70575** coursework. Build a meta‑model that predicts the probability the primary signal is worth trading under a triple‑barrier exit, for 11 instruments across 3 asset classes:

| Class | Tickers |
|---|---|
| Equity index futures | `es1s` (S&P 500), `nq1s` (Nasdaq 100), `fesx1s` (Euro Stoxx 50) |
| Energy | `cl1s` (WTI), `ho1s` (Heating Oil), `rb1s` (RBOB Gasoline), `ng1s` (Natural Gas) |
| Metals | `gc1s` (Gold), `si1s` (Silver), `hg1s` (Copper), `pl1s` (Platinum) |

Marking (100 marks + 10 bonus):

| Section | Marks |
|---|---:|
| Feature engineering | 20 |
| Labeling (triple‑barrier) | 20 |
| Model development & comparison (≥3 models from linear / tree / NN families) | 30 |
| Cluster‑level feature importance | 10 |
| Model evaluation (OOS, per‑instrument, baseline) | 20 |
| Strategy construction (optional) | +10 |

**Score is on methodology, not performance.** Best research wins an Alken interview.

**Data:** `data/ohlcv_data.csv` (daily OHLCV from 1990; ES 1997, FESX 1998, NQ 1999) and `data/primary_signals.csv` (signal in {‑1, 0, +1}, **2020‑01‑03 → 2022‑06‑30**, 645 dates). Hidden test = **H2‑2022** (Jul→Dec 2022); the grader reruns the code on it.

**Deliverables:** end‑to‑end runnable code; `predictions.csv` covering H1‑2022 in `(date, instrument, prediction ∈ [0, 1])` rows; optional `strategy_weights.csv` in `(date, instrument, weight)` rows. Deadline **2026‑06‑04**.

**Strategy constraint (clarified by user):** max **10 % annualised volatility**. (No other constraints.)

**Late discovery:** teams are allowed to use additional data beyond the assignment CSVs as long as it's correctly causal and inside the training window. Sreeram's branch does **not** use the extra data; Harry's branch already does — see [§4.6](#46-alternate-data-bloomberg-style-dataalternate_data_cleanedcsv).

---

## 2. Shared spine (`main`)

Both branches inherit a small, opinionated shared foundation that lives on `main` and is read‑only on both feature branches.

### Repository layout (`main` head, also present on both branches)

```
stml/
├── data/
│   ├── ohlcv_data.csv          # raw inputs, read‑only
│   ├── primary_signals.csv
│   └── meta/                   # generated NA diagnostics
├── refs/
│   ├── missing-holidays.md     # research note (a‑priori)
│   ├── project-instructions.md
│   └── programming-session-sol/ Solution_Programming_Session_{1..8}.ipynb
├── reports/
│   └── missing-data-report.md  # NA findings ↔ the research note
├── src/stml/
│   ├── __init__.py
│   ├── io.py                   # load_data / load_clean_data / load_returns_panel
│   └── na_checks.py            # calendars + missing‑data classifier
├── notebooks/
│   └── agent_eda.ipynb
├── pyproject.toml              # uv project, src layout, editable install
├── uv.lock
└── .gitattributes              # nbstripout filter for *.ipynb
```

### `src/stml/io.py`

Three loaders, each auto‑locating the repo root from any depth:

* `load_data() → (ohlcv_long, signals_wide)` — raw CSVs sorted by `(instrument, date)` and by `date`.
* `load_clean_data() → (ohlcv_clean, signals)` — applies the NA policy: drops 3 calendar‑impossible Sunday `2005‑05‑08` rows on `gc1s/hg1s/si1s`; **keeps** 765 zero‑volume weekday settles (they carry valid OHLC, only volume wasn't recorded); never reindexes to a calendar grid or forward‑fills.
* `load_returns_panel(kind="log") → wide date × instrument panel` — returns computed on each instrument's **own dense series** then pivoted. Structural NaNs (pre‑inception or other‑venue holidays) are preserved and must not be filled.

Instrument universe constant: `INSTRUMENTS = ['cl1s','es1s','fesx1s','gc1s','hg1s','ho1s','ng1s','nq1s','pl1s','rb1s','si1s']`.

### `src/stml/na_checks.py`

Self‑contained holiday calendars for **NYMEX/COMEX**, **CME equity**, and **Eurex** spanning **1990–2022**. Two complementary diagnostics:

1. Per‑instrument calendar comparison — expected sessions = business days − full holidays. Missing expected sessions are classified `early_close` or `unexplained`.
2. Cross‑sectional presence matrix — `global` (all venues closed), `exchange_specific` (one venue), `mixed`.

Headline counts (from `reports/missing-data-report.md`):

* **339 dated trading halts** — 86 global, 197 US‑futures‑only (FESX trades through), 54 Eurex‑only (US trades through), 2 CME‑equity‑only.
* **7 residual unexplained single‑instrument gaps** — each annotated with a likely cause.
* **0 multi‑instrument vendor outages**, **0 NaN OHLC**, **0 high<low**, **0 duplicate rows**.

Outputs in `data/meta/`:

```
missing_holidays_metadata.csv   339 rows — venue scope + affected instruments
other_missing_metadata.csv      61 rows  — early closes (54) + glitches (7)
missing_dates_classified.csv   400 rows
missing_dates_per_instrument.csv 240 rows
summary_per_instrument.csv      11 rows — inception, present, missing
anomalous_rows.csv             768 rows — 3 weekend + 765 zero‑vol
unexplained_missing.csv          7 rows
```

Public utility functions (used by both branches):

| Function | Use |
|---|---|
| `clean_long(df)` | cleaning policy |
| `build_missing_holiday_metadata(df)` | scope classifier |
| `detect_anomalous_rows(df)` | weekend / bad OHLC / zero vol predicate (the zero‑vol mask Harry's `microstructure_fixed` vendors verbatim) |
| `native_returns(long, kind)` / `wide_returns(rets)` | per‑instrument returns → wide pivot |
| `rolling_vol_panel(rets, window)` | per‑instrument rolling vol, then aligned |
| `corr_max_info(W, min_periods)` | pairwise‑complete correlation, repaired to PSD |
| `rolling_pair_corr(W, a, b, window)` | rolling correlation on the pair's intersection calendar |
| `cov_ledoit_wolf(...)` | shrinkage covariance |

### `pyproject.toml`

`stml` is a proper installable package (src layout, editable via `uv sync`). Pinned deps include `numpy`, `pandas`, `scikit-learn`, `xgboost`, `hmmlearn`, `torch>=2.2,<2.3` (CPU index), `statsmodels`, `seaborn`, `matplotlib`. Dev extras: `jupyterlab`, `nbstripout`, `pytest`, `ruff`. Optional extras group `calendars` for `pandas-market-calendars` / `exchange-calendars` / `holidays` (only used as a cross‑check; the hand‑coded tables are authoritative).

### Workflow conventions

Per the README on `main`: never commit to `main` directly; long‑lived personal branch `dev/<initials>`; PRs to `main` at group checkpoints; `notebooks/<initials>/` and `results/<initials>/` are per‑person; shared notebooks via short‑lived `shared/<desc>` branches; nbstripout auto‑strips notebook outputs; never hand‑edit `uv.lock`.

---

## 3. Branch `Sreeram` — full pipeline v0 → v5

> 32 commits ahead of `main`; ~30k LOC (incl. data). Six features stacked into a single coherent story: **stages 1‑3 baseline → stages 4‑5 v2 → forensic v3 commodity‑only → v4 stacked → v5 principled**. All in `src/stml/` (single namespace) and `results/sreeram/`. Build notes in `docs/build/01-…12-…md` provide the rationale per stage.

### 3.1 Layout

```
stml/  (Sreeram‑only files marked •)
├── data/                               unchanged
├── docs/build/                       • 13 markdown files documenting each stage
├── notebooks/agent_eda.ipynb           unchanged (root of notebooks/)
├── notebooks/sreeram/                • empty (personal scratch lives in results/sreeram/)
├── refs/                              unchanged
├── reports/                           unchanged (missing‑data‑report.md)
├── results/sreeram/                  • 21 files — diag scripts, predictions, weights, feature snapshot
├── src/stml/
│   ├── __init__.py                     unchanged
│   ├── io.py                           unchanged
│   ├── na_checks.py                    unchanged
│   ├── labeling.py                   • Stage 1a
│   ├── cv.py                         • Stage 1b
│   ├── features.py                   • Stages 2‑3 (G1–G8)
│   ├── regimes.py                    • Stage 3 (HMM/GMM)
│   ├── models.py                     • Stage 3 (LogReg, XGBoost, MLP, VSN stub)
│   ├── importance.py                 • Stage 4
│   ├── evaluation.py                 • Stage 5
│   ├── pipeline.py                   • v0/v1 orchestrator
│   ├── experiments.py                • Stage 4‑5 master (predictions_v2)
│   ├── best_of.py                    • Per‑instrument CV‑selection (deprecated by best_strategy)
│   ├── best_strategy.py              • v3 commodity‑only
│   ├── v4.py                         • v4 stacked
│   ├── v5.py                         • v5 principled
│   └── strategy.py                   • Strategy track (Stage 6)
├── tests/
│   ├── test_labeling.py              • 19 unit tests
│   ├── test_cv.py                    • 13 unit tests
│   └── test_regimes.py               • 12 unit tests (incl. HMM causality)
├── pyproject.toml                      unchanged
└── uv.lock                           • adds xgboost / hmmlearn / etc.
```

### 3.2 Stage 1a — Triple‑barrier labels (`src/stml/labeling.py`)

Standard López de Prado AFML Ch.3‑4. Public surface:

```python
get_daily_vol(close, span=100, min_periods=20)             → causal EWMA std of log returns
extract_signal_events(signals, instruments=None,
                      include_flat=False)                  → long DataFrame (t, instrument, side)
apply_triple_barrier_one(close, t_event, side, sigma_at_t,
                         h, pt_mult=1.0, sl_mult=1.0)      → (t1, barrier_hit, signed_ret)
get_meta_labels(ohlcv_long, signals,
                h=10, pt_mult=1.0, sl_mult=1.0,
                vol_span=100, vol_min_periods=20,
                instruments=None, price_col='close')       → events frame with label
get_uniqueness_weights(events, normalize=True)             → AFML Ch.4 per‑instrument
get_fixed_horizon_labels(...)                              → rejected baseline (Lecture 1 critique)
label_summary(events_labeled)                              → per‑instrument balance + barrier mix
```

**Conventions:**

* Daily volatility via `pandas .ewm(span, adjust=False).std()` on log returns — strictly causal.
* Barriers in **log‑return** units (not price): `upper = +pt_mult · σ_t · √h`, `lower = −sl_mult · σ_t · √h`.
* First‑touch logic on **close prices** in the strict window `(t, t+h]` — the event bar at `t` is excluded.
* Label = 1 if signed log return at `t1` > 0, else 0. Vertical hit breaks by sign of the realised return (canonical meta‑labeling).
* Symmetric default `pt_mult = sl_mult = 1.0`. Horizon **h=10 trading days**.
* **Per‑instrument concurrency** for uniqueness weights — two events on different instruments don't share the path. Calendar‑time concurrency is what purging in CV handles, separately.
* Events with NaN sigma (insufficient history) are kept but labelled at vertical with NaN return → dropped at the end.

> ⚠ **Entry convention.** Sreeram's resolver scans `close.iloc[idx:idx+h+1]` where `idx = close.index.get_loc(t_event)`. The window includes `t_event` at position 0, and the return is `log(close[t1] / close[t_event])`. That means the **return between the close of `t` and the close of `t+1` is inside the held window**, even though the signal at `t` is observed at the close of `t` (information at the same bar). Harry's labeller uses entry at `t+1` — see [§4.3](#43-step-2--triple-barrier-labels-t1-entry-srcstmlharrylabelspy) for the disagreement.

Default config: **`h=10, pt=1.0, sl=1.0, vol_span=100`**.

### 3.3 Stage 1b — Purged CV and embargo (`src/stml/cv.py`)

`PurgedKFold` — sklearn‑compatible (`get_n_splits` / `split`). Events sorted by `t`, divided into `n_splits` contiguous chunks; for each fold, training events whose `[t, t1]` overlaps the test block's `[test_start, test_end]` are purged, and training events whose `t` falls within `embargo_td` after the test block are also dropped. Embargo can be given as a `pd.Timedelta` (e.g. `pd.Timedelta(days=h)`) or as a fraction of total date span (AFML convention).

`walk_forward_splits(t, boundaries, embargo_td=None)` — yields `(train_pos, test_pos, b_lo, b_hi)` with expanding train and consecutive non‑overlapping test windows. Wrapped by `WalkForwardSplitter`.

`split_by_boundary(t, boundary, embargo_td=None) → (train_pos, predict_pos)` — single split used by the master pipeline.

`assert_no_leakage(train, test, t, t1, embargo_td=None)` — defensive assertion used in tests; raises `AssertionError` with descriptive offender info.

### 3.4 Stage 2 — Features G1–G8 (`src/stml/features.py`)

~75 features split into 8 economic groups, all causal, registered in `FEATURE_GROUPS` so cluster importance can compare data‑driven clusters to declared groups.

| Group | Features (representative) | Computation |
|---|---|---|
| **G1 vol** | `vol_5d / 21d / 63d`, `ewma_vol_50`, `vol_ratio_5_63`, `vol_of_vol_63`, `semivol_21d`, `parkinson_vol_21d`, `garman_klass_vol_21d`, `rogers_satchell_vol_21d` plus `z_*` expanding z‑scores | annualised rolling stds; range‑based estimators on OHLC |
| **G2 trend** | `mom_5/21/63d`, `ma_dist_21/63d` (log‑gap in σ units), `ma21_slope`, `trend_tval_10/21/42d` (backward t‑stat from Programming Session 1) | momentum sums, distance from MA, regression t‑stat |
| **G3 mean‑rev** | `autocorr_21d`, `efficiency_ratio_21d` (Kaufman), `variance_ratio_5d_21w`, `hurst_100d` (rolling rescaled‑range) | bounded indicators, kept **raw** (not z‑scored) |
| **G4 microstructure** | `volume_z_63d`, `volume_trend_21d`, `oi_trend_21d`, `amihud_illiq_21d`, `hl_range_21d` | per‑instrument rolling stats; **no zero‑volume mask** (this is what Harry's `microstructure_fixed` corrects) |
| **G5 signal** | per‑instrument: `side_signal`, `signal_run_len`, `days_since_flip`. cross‑sectional: `net_signal_equity/energy/metals` (signed mean signal within class) | groupby‑cumulative run length |
| **G6 regime** | (lives in `regimes.py`, not `features.py`) | see Stage 3 |
| **G7 calendar** | `month_sin/cos`, `dow_sin/cos` | cyclical encoding |
| **G8 cross‑section** (v4 addition) | `cross_sec_mom_rank_21d`, `cross_sec_vol_rank_21d` (within asset class), `corr_to_sector_63d` (own returns vs sector‑mean ex‑self), `avg_cross_asset_corr_63d` (crisis indicator), `signal_breadth_full`, `signal_consensus_pct`, `trend_persistence`, `vol_clustering_21d`, `recent_shock_z` | sector groupby + cross‑sectional rank + EWMA pairwise corr |

**Standardisation policy:** scale‑dependent features (vol, momentum, MA distance) get `_expanding_zscore(s, min_periods=60)` per instrument; bounded features (autocorr, efficiency ratio, variance ratio, Hurst, trend t‑val) stay raw. Expanding z‑score is what `_diag2` later identifies as the major source of train/OOS drift (see §3.10).

`compute_features(ohlcv, events, signals, include_groups=("G1","G2","G3","G4","G5","G7"), zscore_min_periods=60)` is the master function — returns one row per event id (same index as `events`).

Asset‑class mapping is exported at module level:

```python
ASSET_CLASSES = {
    "es1s":"equity",  "nq1s":"equity",  "fesx1s":"equity",
    "cl1s":"energy",  "ho1s":"energy",  "rb1s":"energy", "ng1s":"energy",
    "gc1s":"metals",  "si1s":"metals",  "hg1s":"metals", "pl1s":"metals",
}
```

### 3.5 Stage 3 — HMM/GMM regimes (`src/stml/regimes.py`)

The Lecture 3 showpiece — and the most causality‑sensitive module on this branch.

* **HMM** — per instrument, Gaussian HMM with `covariance_type="full"`, `n_components=3`, fit via Baum‑Welch on a 2‑D observation `(daily log return, 21d annualised vol)`. **Trained only on data with `date < boundary`** (`boundary = 2022‑01‑01` for submission; rerun moves to 2022‑07‑01). Requires ≥200 training observations; otherwise returns an empty/index‑only DataFrame.
* **Causal filtered posteriors** — `causal_filtered_probs(hmm, X)` is **hand‑rolled** in log‑domain:

  ```
  log α[0] = log π + log emis[0]
  log α[t] = log emis[t] + logsumexp_k'( log α[t-1, k'] + log A[k', k] )
  ```

  then normalised. Explicitly avoids `hmmlearn.predict_proba` which returns smoothed posteriors `P(state | X_{0..T})` and would leak future information.
* **State ordering** by ascending mean realised vol (column 1 of obs) so the posterior columns are `hmm_state_lo / mid / hi` and `hmm_state_argmax` is consistent across runs and instruments. This solves the HMM's label‑switching ambiguity.
* **GMM** — per instrument, Gaussian Mixture with 3 components on a 3‑D snapshot `(21d vol, 21d momentum, 21d autocorr)`. Same `date < boundary` training, then `predict_proba` on the whole series. Components reordered by training‑set mean vol → `gmm_cluster_lo / mid / hi`, `gmm_cluster_argmax`.

`compute_regime_features(ohlcv, events, boundary, n_states=3, n_components=3) → DataFrame` indexed by `events.index`, with 8 columns (4 HMM + 4 GMM). The dedicated test suite `tests/test_regimes.py` includes a **`test_filtered_no_peeking`** invariant — `filtered[t]` computed on `X[:t+1]` must equal `filtered[t]` computed on `X[:T]` for any `T > t`. Also `test_filtered_differs_from_smoothed` so a refactor that accidentally falls back to `predict_proba` would fail.

### 3.6 Stage 3 — Model families (`src/stml/models.py`)

Three model families per the rubric, all sharing the same `fit(X, y, t, t1, sample_weight=None) / predict_proba(X)` interface so they plug interchangeably into the pipeline.

| Class | Family | Pipeline | Tuning |
|---|---|---|---|
| **`ElasticNetLogReg`** | linear | StandardScaler → LogReg(`penalty='elasticnet'`, `solver='saga'`, `class_weight='balanced'`) → isotonic calibration via `CalibratedClassifierCV` | RandomizedSearchCV over `C` (log grid −3..2) × `l1_ratio ∈ {0, 0.25, 0.5, 0.75, 1.0}`, `n_iter=20`, inner `PurgedKFold(5)`, scored by `neg_log_loss` |
| **`XGBoostMeta`** | tree | XGBClassifier(`binary:logistic`, `tree_method='hist'`, `scale_pos_weight` from class balance) → isotonic calibration | RandomizedSearchCV over depth, LR, n_estimators, subsample, colsample_bytree, reg_alpha/lambda, min_child_weight, `n_iter=30` |
| **`MlpMeta`** | NN | StandardScaler → MLPClassifier(`early_stopping=True`, `validation_fraction=0.1`) → isotonic calibration | hidden sizes ∈ {(50,30), (80,40), (100,), (64,32,16)}, alpha ∈ {1e‑4..1e‑1}, lr_init ∈ {1e‑4..5e‑3}, activations relu/tanh |
| **`VsnMeta`** | NN | Variable Selection Network in PyTorch — shared scalar embedding (`d_e=16`), per‑feature GRN, softmax variable‑selection over feature attention, final GRN+linear head | early stopping on a purged held‑out fold |

Every model exposes `best_params_`, `feature_names_`, `calibrator_` (a `CalibratedClassifierCV` fitted on a **purged** held‑out fold so calibration is also leakage‑free). XGBoost exposes `feature_importance('gain')` returning a normalised Series; VSN exposes mean attention weights over the training set; MLP uses sklearn permutation importance with AUC scoring.

> ⚠ **VSN is dead code on Apple Silicon.** Module comments (~line 614) note `torch 2.2.x` was compiled against numpy 1.x and the project's `numpy>=2.4` violates the ABI. `MlpMeta` is the actual NN slot used end‑to‑end.

### 3.7 Stage 4 — Cluster‑level feature importance (`src/stml/importance.py`)

* `feature_distance_matrix(X)` — pairwise distance `1 − |Spearman ρ|`, NaN‑guarded for constant columns.
* `cluster_features(X, n_clusters=None, max_k=12, linkage_method='average')` — hierarchical clustering on the distance matrix; `pick_optimal_k` uses silhouette on the precomputed distance to choose K when not supplied.
* `clustered_mdi(feature_importance: Series, feature_to_cluster: dict) → DataFrame[mdi, mdi_share]` — sums per‑feature MDI within each cluster.
* `clustered_mda(model, X, y, feature_to_cluster, scoring='neg_log_loss', n_repeats=5) → DataFrame[mda, std, rank]` — permutes **all features in a cluster together with the same row permutation**, so intra‑cluster correlation structure is preserved and the cluster's marginal contribution is isolated. Two scorers: `neg_log_loss` (higher = better; importance = increase in log‑loss when permuted) or `roc_auc` (importance = AUC drop).
* `cluster_economic_overlap(feature_to_cluster, feature_groups) → cross‑tab` of data‑driven clusters vs declared G1–G8 groups.

### 3.8 Stage 5 — Evaluation (`src/stml/evaluation.py`)

* `classification_report(y_true, y_score, threshold=0.5, sample_weight=None)` — `n`, `label_1_share`, accuracy, precision, recall, F1, AUC, average precision, log‑loss, Brier.
* `per_instrument_breakdown(events, y_true, y_score, threshold=0.5)` — same metrics grouped by instrument; handles single‑class instruments by reporting partial metrics.
* `confusion_matrix_df`, `threshold_sweep` (default 51‑point grid), `baseline_compare` (vs always‑predict‑1 blind primary).
* `calibration_table(y_true, y_score, n_bins=10)` — decile reliability with `calibration_gap = mean_pred − actual_pos`.
* `optimal_threshold(metric='f1' | 'precision' | 'youden')`.
* `regime_conditional_performance(y, p, regime_state, threshold=0.5)` — AUC/F1/LL grouped by HMM `argmax` state.
* `filtered_strategy_metrics(y, p, side, ret, threshold)` — `n`, hit rate, mean ret (bp), pseudo‑annualised Sharpe `(mean/std) × √(252/h)`. **Rough proxy** for the full strategy, not the same as `strategy.py`.

### 3.9 Pipelines: `pipeline.py`, `experiments.py`

**`pipeline.py`** — the v0 / v1 baseline. `PipelineConfig` carries every knob (`h`, `pt_mult`, `sl_mult`, `vol_span`, `train_predict_boundary=2022‑01‑01`, `predict_end=2022‑07‑01`, `embargo_days=10`, inner `n_splits=5`, `n_iter=20`, `random_state=42`, feature groups, `include_regimes=True`, `output_dir`, `predictions_filename`, optional `instruments`, `model_name ∈ {logreg, xgboost, mlp}`). `run_pipeline()` returns a `PipelineResult` with: labels, features, weights, train/predict positional indices, fitted model, predictions DataFrame, in‑sample and OOS reports, per‑instrument breakdown, confusion at 0.5, threshold curve, blind baseline. Writes the deliverable CSV: one row per `(date, instrument)` in the predict window with `prediction = 0.0` for signal‑0 rows, `0.5` (neutral) for NaN‑feature rows, model probability otherwise.

**`experiments.py`** — Stage 4 + Stage 5 master.

```python
ASSET_CLASSES = {...}     # same as features.py

_Data dataclass: ohlcv, signals, events_all, events_lab,
                 X_all, X_lab, y_lab, t_lab, t1_lab, w_lab,
                 side_lab, ret_lab, regimes_lab,
                 boundary, predict_end

_build_data(boundary, predict_end, h=10, pt_mult=1.0, sl_mult=1.0, vol_span=100)
run_all_models(boundary=2022‑01‑01, predict_end=2022‑07‑01,
               embargo_days=10, n_iter=15, n_splits_inner=5,
               random_state=42) → {data, models, results, tr_pos, oos_pos, embargo}
per_sector_ablation(...) → pooled vs per‑sector XGBoost, AUC comparison
build_v2_artifacts(...) → trains all 3 families, per‑sector ablation,
                          cluster importance (MDI on XGBoost + MDA cross‑check),
                          MLP permutation importance, deep evaluation
                          (calibration, optimal threshold global + per‑instrument,
                          regime‑conditional perf, confusion, threshold sweep,
                          baseline, filtered‑strategy metrics).
                          Writes predictions_v2.csv (best model by OOS log‑loss).
```

### 3.10 Forensic diagnostics (`_diag1` → `_diag7`)

Scripts in `results/sreeram/` that document the v2 → v3 → v4 narrative. They are throwaway investigative code, not part of the production path, but the v2 → v3 finding (commodity‑only training) is the load‑bearing pivot of the whole branch.

**`_diag1.py` — Single‑feature OOS AUC**

For each feature, compute OOS AUC as a univariate classifier (`max(auc, 1‑auc)` for direction agnosticism). Best single feature: `mom_21d` at **0.598**. Top‑10 mean: **0.572**. 13 features have AUC > 0.55. *The ceiling on what's learnable from any single feature is ~0.60; the combined model at v2 is **below 0.5**, meaning the model is **destroying** information rather than combining it.*

**`_diag2.py` — Distribution drift (KS test, train vs OOS)**

Top KS shifts: `month_sin` 0.518 (trivially shifted), then **z‑scored vol/momentum features at KS 0.22–0.29**. Diagnosis: the expanding‑window z‑score uses 1990‑present statistics, but 2020‑22 is materially higher vol. A "z=+1" in training and "z=+2" in test are not comparable.

Same script tries 6 fixes:

| Fix | OOS AUC |
|---|---:|
| Baseline pooled XGBoost (v2) | 0.494 |
| Drop z‑scored features | 0.467 |
| **Drop equity instruments from TRAINING** | **0.578** |
| Recency‑weight training (decay 0.5/yr) | 0.512 |
| Ensemble (LogReg + XGB, arith mean) | 0.515 |
| Per‑instrument models | 0.453 |
| Boundary at 2022‑04‑01 (3 fewer months OOS) | 0.589 |

**`_diag3.py` — Per‑instrument label balance flips**

Headline:

| Instrument | Train label_1 share | OOS label_1 share | Flip |
|---|---:|---:|---:|
| `ng1s` | 0.72 | 0.41 | **−0.31** |
| `gc1s` | 0.65 | 0.41 | **−0.24** |
| `cl1s` | 0.66 | 0.78 | +0.12 |
| `nq1s` | 0.60 | 0.59 | 0 |

The primary signal itself collapses on `ng1s` and `gc1s` in H1‑2022 — there is genuine signal‑level regime break, not just model fragility.

Then tries: recency weighting decay ∈ {0.3, 0.5, 0.7}, per‑instrument XGB (≥80 events), per‑sector XGB, hybrid (per‑sector commodities + pooled equity), CV‑selected best per instrument, equity‑only model. CV‑selected per‑instrument: AUC **0.441** vs pooled 0.494 — **CV picks the most overfit model**. Inner‑CV AUCs of 0.91–0.99 collapse to OOS AUCs of 0.30–0.68 on the same instruments → textbook regime overfit (all CV folds share the 2020‑21 regime).

**`_diag4.py` — `best_of` per‑instrument selection**

Runs `stml.best_of.build_best_of` which trains pooled + per‑sector + per‑instrument variants of XGBoost, selects per instrument by inner purged CV AUC, predicts on OOS. Confirms 0.441 aggregate AUC. Writes `predictions_v3.csv` from this pick. (Subsequently overwritten by `_diag6.py`.)

**`_diag5.py` — final fixes**

Trains 6 model variants on commodity events: pooled XGB, pooled XGB + recency 0.3, commodity XGB, commodity XGB + recency 0.3, metals‑sector XGB, energy‑sector XGB. Tests 13 ensemble/hybrid configurations including geometric mean, arithmetic mean, equity‑abstain (predict label_1_share for equity), equity‑zero (predict 0 for equity). Best in this batch: **`avg(commod_xgb, commod_xgb_rec)`** at AUC 0.557 vs pooled baseline 0.494 (+5.5 pp).

**`_diag6.py` — final v3 build**

Trains `m_commod` and `m_commod_r` (XGBoost on commodity events, with and without recency decay 0.3) + LogReg commodity baseline. Best ensemble = arithmetic mean of the two XGBs. Writes the final `predictions_v3.csv` (1408 rows, H1‑2022).

**`_diag7.py` — feature selection + regularisation experiments**

* Inner‑CV single‑feature AUC → keep features with CV AUC > 0.54 (~13 features).
* Commodity XGB with selected features only.
* Commodity XGB with aggressive regularisation (`max_depth=3`, `reg_alpha=1`).
* Random Forest on commodity (different bias profile — Breiman bagging vs boosting).
* Ultra‑regularised LogReg on commodity.
* Ensembles of all four.

Best: `avg(selected, regularized)` at AUC ~0.561. Confirms v3 commodity‑only is the structural fix, not over‑tuning.

**`_diag_strategy.py` — strategy hyperparameter sweep**

Backtests 8 strategy configurations on top of `predictions_v4.csv`. Configs sweep `threshold ∈ {0.40..0.55}`, `target_vol ∈ {0.10..0.20}`, `max_per_instrument ∈ {0.30..0.50}`, `gross_cap ∈ {2.0..3.0}`. Picks by Sharpe. Best: `threshold=0.40, target_vol=0.15, target_portfolio_vol=0.15, max_per_instrument=0.40, gross_cap=2.5, net_cap=2.0` → Sharpe 3.91, CAGR 0.763, ann_vol 0.181, MDD −3.6 %, avg holding 4.0 days, turnover/day 0.21. Writes `strategy_weights.csv`.

> ⚠ The chosen target_vol of 0.15 and realised ann_vol of 0.181 both **exceed** the 10 % portfolio‑vol cap clarified by the user. The committed `strategy_weights.csv` would breach the cap as currently configured.

**`_v5_stress.py` — multi‑boundary stress test of v5**

Runs `run_v5` at three boundaries — 2021‑10‑01 (test = 2021‑Q4), 2022‑01‑01 (test = H1‑2022, our submission), 2022‑04‑01 (test = Q2‑2022, val spans regime break). For each: VAL AUC, TEST AUC, bootstrap 95 % CI, Brier. Also computes Pearson correlation of v5 predictions between the 2022‑01 and 2022‑04 boundaries on overlapping test events.

### 3.11 v3 — Commodity‑only XGBoost (`src/stml/best_strategy.py`)

The simplest fix from the diagnostics, productionised. `train_commodity_only_model` filters `tr_pos` to non‑equity events and fits a single `XGBoostMeta`. Optional `recency_decay` multiplies the uniqueness weights by `decay ** years_before_boundary` (and renormalises). `run_best_strategy(boundary=2022‑01‑01, predict_end=2022‑07‑01, embargo_days=10, n_iter=15)` produces `predictions_v3.csv`.

**Why it works (per `docs/build/10‑forensic‑improvements.md`):** equity 2020‑21 was COVID melt‑up; equity 2022 was Fed‑pivot bear. Feature → label mapping inverted. Training on commodities, the model **abstains** on equity OOS rows because the feature distribution looks unfamiliar (≈ 0.5 prob, AUC ≈ 0.5) instead of being actively wrong (AUC 0.30–0.45). 7/10 instruments improve; `gc1s` 0.53 → 0.81, `ng1s` 0.45 → 0.70 are huge.

`best_of.py` (the CV‑selected version) is also still in the module tree but is **rejected** by diag5/6 — `write_predictions_v3` is explicitly `NotImplementedError`.

### 3.12 v4 — Stacked conditional ensemble (`src/stml/v4.py`)

8 level‑0 base models all trained on commodity events only:

| # | Model | Rationale |
|---|---|---|
| M1 | XGB h=10 full features | Workhorse — non‑linear interactions over the full causal feature set |
| M2 | XGB h=10 + recency 0.3 | Recent events better represent the test regime |
| M3 | XGB h=5 | Different label semantics — scalp‑style bets |
| M4 | XGB h=15 | Different label semantics — position‑trade‑style bets |
| M5 | ElasticNet LogReg h=10 | Linear baseline; regularises tree overfit |
| M6 | Random Forest h=10 (`max_depth=5`) | Different bias profile (bagging vs boosting) |
| M7 | XGB long‑only commodity | Long bets have different drift / tail structure |
| M8 | XGB short‑only commodity | Short bets are asymmetric (squeezes, harder vol) |

**Stack meta‑learner:** LogReg over **out‑of‑fold** base predictions (purged K‑fold, no leakage). Stack coefficients land at:

```
M2_xgb_recency = +1.27
M6_rf_h10      = +0.84
M1_xgb_h10     = +0.58
M4_xgb_h15     =  0.00
M8_xgb_short   = −0.18
M3_xgb_h5      = −0.47
M7_xgb_long    = −1.27
M5_lr_h10      = −1.43
```

The stack **inverts** M5/M7/M3 — their predictions are anti‑predictive in this period, so contributing them with a negative sign improves the stack. A clean stacking finding.

**Per‑instrument isotonic calibration** is fit on the training‑period stack output, applied to OOS.

Adds **G8 cross‑sectional features** (9 new features — total 75): `cross_sec_mom_rank_21d`, `cross_sec_vol_rank_21d`, `corr_to_sector_63d`, `avg_cross_asset_corr_63d` (crisis indicator), `signal_breadth_full`, `signal_consensus_pct`, `trend_persistence`, `vol_clustering_21d`, `recent_shock_z`.

Headline OOS H1‑2022 (n=1002):

| Model | AUC | F1 | Brier | LogLoss |
|---|---:|---:|---:|---:|
| M1 XGB h=10 | 0.556 | 0.652 | 0.249 | 0.695 |
| M2 XGB h=10 recency | **0.574** | 0.659 | 0.246 | 0.689 |
| M3 XGB h=5 | 0.482 | 0.616 | 0.265 | 0.732 |
| M4 XGB h=15 | 0.505 | 0.598 | 0.260 | 0.722 |
| M5 LogReg | 0.504 | 0.704 | 0.249 | 0.694 |
| M6 RF | 0.490 | 0.694 | 0.252 | 0.701 |
| M7 long‑only | 0.541 | 0.690 | 0.252 | 0.700 |
| M8 short‑only | 0.511 | 0.661 | 0.254 | 0.708 |
| STACK raw | 0.542 | 0.540 | 0.249 | — |
| **STACK per‑instrument calibrated** | **0.562** | 0.637 | 0.245 | — |

Per‑instrument vs v3:

* `fesx1s` 0.36 → 0.63 (+0.27)
* `nq1s` 0.49 → 0.65 (+0.16)
* `pl1s` 0.51 → 0.64 (+0.13)
* `cl1s` 0.61 → 0.52 (−0.09)
* `es1s` 0.61 → 0.38 (−0.23) — note: this is the OOS slice through the stacked + calibrated model

Net: 6 of 10 improve, 4 regress, aggregate +1.3 pp AUC over v3.

### 3.13 v5 — Principled rebuild (`src/stml/v5.py`)

The author audited their own work and concluded **everything from v3 onward was selection‑on‑test**:

* "Dropped equity from training because H1‑2022 OOS was bad on equity."
* "Tuned `recency_decay=0.3` because that gave best H1‑2022 OOS."
* "Picked the stacked structure because it beat baselines on H1‑2022."
* "Designed/kept G8 features after checking they helped H1‑2022 OOS."

Each decision used H1‑2022 information to make architectural choices. The grader tests on **H2‑2022** (different regime), so the chain doesn't transfer.

v5 enforces 8 explicit principles, **each stress‑tested before adoption**:

| # | Principle | Implementation |
|---|---|---|
| P1 | **Strict 3‑way TRAIN / VAL / TEST.** All decisions on TRAIN+VAL, TEST touched once. | `val_months=6` before boundary; TRAIN before VAL minus embargo |
| P2 | **Robust scaling** | Winsorize features at 1 %/99 % on TRAIN, then StandardScaler fit on TRAIN — replaces the leaky expanding z‑score |
| P3 | **Simple‑average ensemble** | ElasticNet + tightened XGBoost + RF. Equal weights (variance reduction without bias). No learned stack weights. |
| P4 | **Global calibration via CV** | Try identity / Platt / isotonic; pick lowest 5‑fold CV log‑loss on VAL. **No per‑instrument calibration** (overfits ~100‑event samples) |
| P5 | **Shrinkage toward 0.5** | `final = α · pred + (1 − α) · 0.5`, α chosen on VAL by min log‑loss. Search `α ∈ {0.50, 0.60, 0.70, 0.80, 0.90, 1.00}` |
| P6 | **Walk‑forward stability** | Repeat the whole pipeline at several earlier boundaries, check VAL prediction correlation |
| P7 | **Bootstrap CIs** | Stratified bootstrap, n=200, on TEST AUC |
| P8 | **All instruments, all features** | No OOS‑driven filtering |

For our submission boundary `2022‑01‑01`:

* TRAIN: 2020‑01 → 2021‑06‑21 (18 months) minus embargo
* VAL: 2021‑07 → 2022‑01 (6 months)
* TEST: 2022‑01 → 2022‑07 (H1‑2022)

Headline TEST result:

```
AUC (uncalibrated)     0.471
AUC (calibrated)       0.471  (CV picks identity = no calibration)
AUC (shrunk α=0.90)    0.471
Bootstrap CI           [0.431, 0.499]  ← INCLUDES 0.5
Brier                  0.255
F1 @ 0.5               0.620
```

**Under strict methodology, the 95 % CI on H1‑2022 includes 0.5 — indistinguishable from random.** v4's 0.562 is therefore an overestimate driven by selection‑on‑test.

**Stress test (from `_v5_stress.py`):**

| Boundary | TEST window | AUC | 95 % CI | Regime relationship |
|---|---|---:|---|---|
| 2021‑10‑01 | 2021‑Q4 | **0.596** | [0.549, 0.641] | VAL=2021‑H1, TEST=2021‑Q4 — same regime |
| 2022‑01‑01 | H1‑2022 | 0.467 | [0.431, 0.499] | VAL=2021‑H2, TEST=H1‑2022 — **regime break** |
| 2022‑04‑01 | Q2‑2022 | 0.470 | [0.417, 0.523] | VAL spans the break |

**Pattern: when VAL and TEST share a regime, AUC ~0.60; when they straddle a break, AUC ~0.47.** Cross‑boundary prediction Pearson correlation between 2022‑01 and 2022‑04 = **0.997** — the model itself is stable, the world shifted.

**Forecast for the rerun:** grader's rerun puts VAL = H1‑22 and TEST = H2‑22, **both in the same Fed‑pivot regime**. Same scenario as the 2021‑10 case → expected H2‑22 AUC ≈ 0.55–0.60. That's the only honest argument for expected rerun performance.

### 3.14 Strategy track (`src/stml/strategy.py`)

```python
@dataclass StrategyConfig:
    threshold = 0.55
    target_vol = 0.10                    # per‑instrument target vol contribution
    target_portfolio_vol = 0.10
    max_per_instrument = 0.30
    gross_cap = 2.0                       # |w1|+|w2|+... ≤ 2.0 (200 % gross)
    net_cap = 1.0
    risk_free_rate = 0.0
    vol_lookback = 21
    vol_ann_factor = 252.0
    cov_min_periods = 252
```

Pipeline:

1. **Raw conviction:** `conv = max(0, (p − threshold) / (1 − threshold))` (linear ramp from threshold to 1).
2. **Position sign:** `raw_w[t, i] = sign(signal[t, i]) · conv(p[t, i])`.
3. **Per‑instrument vol target:** `w[t, i] = raw_w · target_vol / forecast_vol[t, i]` (21d realised vol, annualised).
4. **Portfolio vol cap:** scale all weights uniformly if `√(wᵀ Σ w) > target_portfolio_vol`. Σ = 1y historical covariance (annualised), fallback to identity if insufficient data.
5. **Position cap:** `|w| ≤ max_per_instrument`.
6. **Gross/net caps:** scale down if `Σ|w| > gross_cap` or `|Σw| > net_cap`.

`backtest(predictions, signals, ohlcv, cfg)` returns weights, daily portfolio log returns, equity curve, forward returns frame, metrics dict (`CAGR`, `ann_return`, `ann_vol`, `Sharpe`, `Sortino`, `MDD`, `avg_holding_days`, `turnover_per_day`, `n_days`, `n_positions_avg`).

`blind_baseline_strategy` replaces the threshold with 0 and conviction with constant 1.0 — take every signal at fixed size — for apples‑to‑apples comparison under the same vol‑targeting/caps.

`write_strategy_weights(weights, output_path)` writes the assignment‑format CSV: `(date, instrument, weight)` rows.

**v4 strategy backtest** (committed `strategy_weights.csv`, config `threshold=0.40, target_vol=0.15, target_portfolio_vol=0.15, max_per_instrument=0.40, gross_cap=2.5, net_cap=2.0`):

| Metric | Meta | Blind baseline |
|---|---:|---:|
| CAGR | 0.763 | 1.362 |
| Ann. vol | 0.181 | 0.302 |
| Sharpe | 3.91 | 4.22 |
| Sortino | 6.25 | 7.31 |
| MDD | −3.6 % | −6.0 % |
| Avg holding (days) | 4.0 | 5.8 |
| Turnover/day | 0.21 | 0.66 |
| Avg # positions | 7.4 | 7.7 |

The build doc reads it honestly: the blind baseline has a higher Sharpe **in this specific period** because H1‑22 was unusually kind to the primary trend signal; the meta‑strategy has half the drawdown and lower turnover.

### 3.15 Tests (`tests/`)

**`test_labeling.py`** — 19 unit tests:

* `TestGetDailyVol` — causality (vol at `t` invariant under future‑data append), flat prices → zero vol, NaN prefix before `min_periods`, requires sorted index.
* `TestApplyTripleBarrierOne` — long hits PT on rising, long hits SL on falling, short hits PT on falling, vertical with flat, earliest‑touch wins, event at end of data → NaN ret, NaN sigma falls back to vertical, invalid side raises.
* `TestGetMetaLabels` — schema, label = sign(ret) invariant, trending‑up panel produces high long label share, t1 ≥ t, signal=0 extraction.
* `TestUniquenessWeights` — disjoint events → weight 1, fully overlapping → 0.5 raw + 1.0 normalised, concurrency is per‑instrument, post‑normalisation mean = 1.
* `TestFixedHorizonLabels` — matches naive computation on hand‑computed up‑trend.

**`test_cv.py`** — 13 unit tests covering `PurgedKFold` (split shape, test sets disjoint and cover, no‑leakage invariant, embargo enforced, mismatched index raises, `n_splits ≥ 2`, `n_samples > n_splits`), `walk_forward_splits` (monotonic expanding train, consecutive disjoint test windows, pre‑test embargo), `split_by_boundary` (exact partition, embargo pulls back train), `assert_no_leakage` (flags overlap, passes disjoint).

**`test_regimes.py`** — 12 unit tests:

* `TestCausalFilteredProbs` — `filtered[t]` on `X[:1000]` equals `filtered[t]` on `X[:1400]` (the **headline causality invariant**), rows sum to 1, filtered ≠ smoothed (would notice a refactor that fell back to `predict_proba`).
* `TestHmmFeatures` — shape and column names, states are vol‑ordered (high‑vol regime activates `hmm_state_hi`/argmax=K‑1 disproportionately), early vs late boundary produce different posteriors in overlap, too‑little‑data returns empty.
* `TestGmmFeatures` — shape, row probs sum to 1.
* `TestComputeRegimeFeatures` — one row per event.

> ⚠ **Tests don't run on this machine.** The `.venv` was built for x86_64 (older intel hardware) but the current machine is arm64 Apple Silicon, so `numpy._core._multiarray_umath` fails to load. A clean `rm -rf .venv && uv sync` would fix it. Tests are intact; this is purely an environment mismatch.

### 3.16 Output inventory

All in `results/sreeram/`:

| File | Size / shape | Content |
|---|---|---|
| `predictions_v0.csv` | 1408 rows | LogReg baseline (h=10, all features incl. G6 regimes), H1‑22 |
| `predictions_v1.csv` | 1408 rows | LogReg with full feature set, H1‑22 |
| `predictions_v1_alt_boundary.csv` | 704 rows | LogReg at boundary=2022‑04‑01 (Q2 only) |
| `predictions_v2.csv` | 1408 rows | Stage‑4 master output, best model by OOS log‑loss |
| `predictions_v2_xgb.csv` | 1408 rows | XGBoost slot of the Stage‑4 master |
| `predictions_v3.csv` | 1408 rows | Commodity‑only XGB ensemble (`avg(commod_xgb, commod_xgb_rec)`) |
| `predictions_v4.csv` | 1408 rows | Stacked ensemble (best AUC) |
| `predictions_v5.csv` | **704 rows** ⚠ | Principled v5; window mismatch — only covers Q2‑22 because the file was overwritten by `_v5_stress.py`'s last config (boundary=2022‑04‑01) |
| `strategy_weights.csv` | 1408 rows | v4‑derived weights at target_vol=0.15 (breaches 10 % cap) |
| `strategy_weights_v4.csv` | 1408 rows | v4 alt config |
| `feature_matrix_snapshot.csv` | 4984 rows × 84 cols | Full labeled events 2020‑01 → 2022‑06, all features + labels + period (`train_2020_2021` / `test_H1_2022`) |
| `_diag*.py`, `_v5_stress.py`, `_diag_strategy.py` | 9 scripts | Diagnostic narrative |

OOS prediction stats:

| File | non‑zero n | mean | std | min | max |
|---|---:|---:|---:|---:|---:|
| `predictions_v0.csv` | 1011 | 0.563 | 0.072 | 0.379 | 0.756 |
| `predictions_v1.csv` | 1011 | 0.562 | 0.077 | 0.348 | 0.716 |
| `predictions_v2.csv` | 1011 | 0.565 | 0.064 | 0.378 | **0.911** |
| `predictions_v2_xgb.csv` | 1011 | 0.559 | 0.052 | 0.321 | 0.732 |
| `predictions_v3.csv` | 1011 | 0.569 | 0.072 | 0.358 | **0.956** |
| `predictions_v4.csv` | 1011 | 0.548 | 0.087 | 0.342 | 0.769 |
| `predictions_v5.csv` | 503 | 0.551 | 0.056 | 0.397 | 0.642 |
| `predictions_v1_alt_boundary.csv` | 503 | 0.563 | 0.054 | 0.390 | 0.847 |

Strategy weights stats:

| File | non‑zero rows | mean | mean abs | min | max |
|---|---:|---:|---:|---:|---:|
| `strategy_weights.csv` | 946 | 0.029 | 0.086 | −0.280 | 0.400 |
| `strategy_weights_v4.csv` | 438 | 0.009 | 0.015 | −0.138 | 0.156 |

Per‑instrument abs‑mean weight in `strategy_weights.csv`: cl1s 0.132, si1s 0.134, es1s 0.124, hg1s 0.125, rb1s 0.110, gc1s 0.081, fesx1s 0.078, nq1s 0.067, pl1s 0.067, ng1s 0.021, ho1s 0.004.

### 3.17 Open issues on Sreeram

1. **`predictions_v5.csv` is stale and wrong‑window.** Last written by `_v5_stress.py`'s third config (boundary=2022‑04‑01). Only covers Q2‑22 (704 rows). For v5 to be a valid H1‑22 submission, `run_v5(V5Config(boundary=2022‑01‑01))` needs to be rerun cleanly.
2. **Strategy weights breach the 10 % vol cap.** Both `strategy_weights*.csv` were tuned at `target_portfolio_vol=0.15` and realise `ann_vol=0.181`. Submitting on the strategy track requires regenerating at `target_portfolio_vol=0.10` and re‑checking.
3. **No submission identity decided.** v4 is the highest‑AUC artifact; v5 is the principled artifact. No marker file or doc says which to submit.
4. **VSN PyTorch module is dead code on Apple Silicon** — numpy/torch ABI mismatch. `MlpMeta` is the actual NN slot.
5. **`venv` is x86_64**; tests can't run on Apple Silicon without a `uv sync` rebuild.
6. **Several decisions inside v3/v4 are flagged as selection‑on‑test by v5's own doc.** If those are kept in the writeup, the AUC numbers from v3/v4 should be cited with that caveat.
7. **The `bloomberg/` directory under `data/` is empty.** Sreeram's branch never touched the alternate data; Harry's branch already has it (see §4.6).

---

## 4. Branch `Harry` — independent audit + reconciled pipeline + (2026‑06‑02 PM) clean‑split champion importance

> 53 commits ahead of `main` as of 2026‑06‑02 PM (was 49 in the morning); ~246k LOC including binary outputs (CSVs, PNGs, parquet caches). Two coexisting subpackages — `src/stml/harry/` for the audit‑driven causal foundation, `src/stml/new_work/` for the production stage 4‑5 pipeline. 239 tests under `tests/harry/`. Reports in `reports/harry/00-…03-6‑…md`. Outputs in `results/harry/` and (predominantly) `src/stml/new_work/outputs/`. Nothing in `notebooks/sreeram/` is touched on this branch; isolation is strict.
>
> **Update window 2026‑06‑02 PM (4 new commits, ~380 files changed, +118k / −165k lines):** Harry shipped a major methodology refinement that **changes the champion list and significantly tightens the leakage controls**. Section §4.19 documents the diff fully. The headline:
>
> * A new module `split_config.py` introduces a **global train / test split at 2021‑10‑06** (the 70th percentile of 4 764 pooled events) with a **10‑trading‑day embargo to 2021‑10‑20**. The held‑out 30 % is SEALED and never read during importance or model selection.
> * `model_comparison.py` was rewritten: the inner expanding‑window 75/25 split is replaced by **purged inner k‑fold (k=4)** and the hyperparameter pick uses the **1‑standard‑error rule** (smallest within 1 SE of the best mean — strongly regularised). `SKIP_INDIVIDUAL_GROUPS = {"ng1s"}`, `NO_MLP_GROUPS = {"cl1s", "gc1s"}`.
> * `champion_importance.py` was rewritten end‑to‑end (978 lines diff). It now runs importance for **all 11 instruments' champions** (not just the four signal‑bearing ones), splits per **estimator family** — tree (MDA + MDI + SHAP) vs logistic (MDA + |coef|) — and adds **PC1‑PC3 within‑cluster PCA** on the top 3 clusters per instrument.
> * **Three new per‑asset‑class notebooks** replace the two pre‑split full‑sample notebooks: `equity_importance.ipynb`, `energy_importance.ipynb`, `metals_importance.ipynb`. The old `feature_importance_analysis_{energy,metals}.ipynb` and the old `reconciliation_report.md` were deleted as superseded.
> * **The champion list changed.** Previously (pre‑split reconciliation): cl1s · es1s · ho1s · rb1s (4 signal‑bearing). Now (post‑split, purged k‑fold + 1SE): **cl1s · nq1s · fesx1s · pl1s · hg1s (5 signal‑bearing)**. Only `cl1s` survived both methodologies; es1s, ho1s, rb1s fell out; nq1s, fesx1s, pl1s, hg1s emerged.
>
> The pre‑update content in §4.1‑§4.18 is preserved as it still describes the bulk of the branch (signal audit, labels, feature pack, macro, the original new_work HMM/GARCH/CPCV machinery, the original reconciliation). §4.19 is the diff.

### 4.1 Branch philosophy (`reports/harry/00‑context.md`)

> "Sreeram's pipeline is built for trend‑following. signal‑deep‑dive's characterization says the signal is counter‑trend. Both pieces of work are technically rigorous in isolation; together they are inconsistent."

Harry's role is the **synthesis + creativity layer**: a Step‑1 audit that resolves the contradiction with one independent measurement, then features no other branch ships, plus a properly purged model‑comparison harness and cluster importance.

**Hard constraints (written into the context doc):**

1. Absolute branch isolation — writes confined to `src/stml/harry/`, `notebooks/harry/`, `tests/harry/`, `reports/harry/`, `results/harry/`, plus `pyproject.toml` / `uv.lock`.
2. Reuse via **copy** with citation, not import — the two work branches stay self‑consistent in their own worlds.
3. `random_state=42` everywhere; every CV split seeded; every model fit seeded.
4. **Truncation‑invariance** for every feature: `feature(panel[:t+1]).iloc[t] == feature(panel).iloc[t]` for any later `t`. Enforced by a single parametrised test (`tests/harry/test_causality.py`) that auto‑discovers feature registrations.
5. **Next‑day execution: `PnL_t = s_t · r_{t+1}`.** Labels are evaluated over `[t+1, t+1+h]`, not `[t, t+h]`. This is the load‑bearing convention difference vs Sreeram.

`SETUP.md` notes: wavelet module needs `uv sync --extra harry-features` (PyWavelets). `tda.py` (persistent homology / ripser) was scoped out — C++ build deps. Harry's pipeline (Step 4) uses sklearn's `HistGradientBoostingClassifier` and a re‑implemented VSN in PyTorch ≥ 2.12 instead of Sreeram's XGBoost/torch stack — though in practice the production code in `new_work` uses XGBoost again.

### 4.2 Step 1 — Signal‑direction audit (`src/stml/harry/signal_audit.py`)

Per‑instrument scalar statistics over the full released window 2020‑01‑03 → 2022‑06‑30:

| Statistic | Definition |
|---|---|
| `corr(s_t, r_{t-k})` for `k ∈ {-5,-3,-1,0,1,3,5,10,20}` | k > 0 = trailing return; k = 0 = contemporaneous; k < 0 = forward |
| `mean(s_t · r_{t+1})` | next‑day PnL |
| `mean(s_t · cumret_{t+1..t+h})` | h=10 forward PnL |
| `hit_rate_h = P(s_t · cumret_{t+1..t+h} > 0 \| s_t ≠ 0)` | participating‑signal hit rate |

**Moving‑block bootstrap** (Künsch 1989) for 95 % CIs: block size 20 trading days ≈ p90 signal run length, `n_boot=1000`, seeded `42 + i` per instrument so each row reproduces independently. Block bootstrap preserves the autocorrelation in both the piecewise‑constant signal and the returns; an i.i.d. bootstrap would massively understate CI widths.

Tagging: `tag` (canonical, on `mean_trail_corr = mean(corr_trail_{1,5,10,20})`), `tag_trail_1`, `tag_trail_h10` — each thresholded at ±0.05 into `trend` / `mean_reverting` / `mixed`.

`audit_stability` re‑runs the audit on the first and second halves of the released window (split at the median signal date), flags sign flips where both halves' 95 % CIs exclude zero.

CLI: `python -m stml.harry.signal_audit --out results/harry/signal_direction.csv --stability-out results/harry/signal_direction_stability.csv --h 10 --n-boot 1000 --block-size 20 --seed 42`.

**Headline findings (`reports/harry/01‑signal‑direction.md`):**

| inst | class | n bets | `corr_trail_1` | `corr_fwd_1` | mean_pnl_h | hit_rate_h | canonical tag |
|---|---|---:|---:|---:|---:|---:|---|
| es1s | equity | 575 | **−0.161** | **+0.103** | +0.0046 | 0.604 | mixed |
| nq1s | equity | 604 | −0.053 | **+0.114** | +0.0088 | 0.599 | mixed |
| fesx1s | equity | 637 | **−0.193** | +0.047 | +0.0019 | 0.523 | **mean_reverting** |
| cl1s | energy | 422 | +0.003 | **+0.121** | **+0.0238** | **0.706** | mixed |
| ho1s | energy | 63 | −0.114 | +0.036 | +0.0027 | 0.667 | mixed |
| rb1s | energy | 628 | −0.103 | **+0.065** | −0.0091 | 0.503 | mixed |
| ng1s | energy | 124 | −0.040 | +0.093 | +0.0068 | 0.600 | mixed |
| gc1s | metals | 168 | −0.087 | **+0.129** | +0.0017 | 0.605 | mixed |
| si1s | metals | 578 | **−0.143** | +0.057 | −0.0010 | 0.562 | **mean_reverting** |
| hg1s | metals | 628 | −0.109 | **+0.122** | +0.0030 | 0.540 | **mean_reverting** |
| pl1s | metals | 557 | **−0.147** | +0.071 | +0.0014 | 0.524 | **mean_reverting** |

**Bold** in corr columns = CI excludes zero. Conclusions:

1. **Next‑day PnL convention confirmed:** `corr(s_t, r_{t+1}) > 0` for **all 11**; CI strictly above zero for 6 (es, nq, cl, rb, gc, hg).
2. **Counter‑trend at lag 1** for **10 of 11** instruments (only `cl1s` ≈ 0). Strongest at horizon 1; decays monotonically: `corr_trail_1` mean −0.107, `corr_trail_20` mean −0.011.
3. **`cl1s` is structurally different.** `corr_trail_1` ≈ 0; `corr_fwd_1` = +0.121 (highest); h=10 hit rate 0.706 (highest); mean PnL +0.0238 (highest). signal‑deep‑dive's `best_construction_lag` for cl1s independently lands at −5 with positive loading.
4. **`ng1s` is single‑direction.** 100 % short — every non‑zero signal is −1.
5. **`ho1s` has only 63 bets.** Thinnest signal.
6. **Half‑window stability** — `results/harry/signal_direction_stability.csv` re‑runs on the two halves; **zero sign flips** at the 95 % CI level across 88 (instrument, metric) rows. The largest divergence is `fesx1s` `corr_trail_1` strengthening from −0.158 → −0.255 (same sign).

**Reconciliation with signal‑deep‑dive's "10/11 mean‑reverting" headline:**

| classifier | trend | mean_reverting | mixed | what it uses |
|---|---:|---:|---:|---|
| `tag_trail_1` | 0 | **9** | 2 | sign of `corr_trail_1` only |
| `tag_trail_h10` | 2 | 2 | 7 | sign of `corr_trail_10` only |
| `tag` (canonical) | 0 | 4 | 7 | mean of `corr_trail_{1,5,10,20}` |

signal‑deep‑dive's 10/11 claim is **trail_1 only**. At h=10 (Harry's barrier horizon) the structure is much weaker — that's why labels stay symmetric and the model is allowed to learn its own bias.

Tests: 21 unit tests in `test_signal_audit.py` (14 cited in the build doc, file actually contains 21).

### 4.3 Step 2 — Triple‑barrier labels (t+1 entry) (`src/stml/harry/labels.py`)

Three load‑bearing differences vs Sreeram's labeller:

**Decision 1 — Entry at `t+1`, not `t`.**

```
entry_close = close at bar t+1   (Sreeram uses bar t)
window      = close[t+1 .. t+1+h]
barrier widths (log‑return units, signed in bet direction):
    pt_width = pt_mult · σ_t · √h
    sl_width = sl_mult · σ_t · √h
for u in (t+2, …, t+1+h):
    signed_dist = side · log(close[u] / entry_close)
    if signed_dist ≥ pt_width: PT touch
    if signed_dist ≤ -sl_width: SL touch
t1   = u_first_touch or t+1+h
ret  = side · log(close[t1] / entry_close)
label = 1 if ret > 0 else 0
```

The build doc walks through a hand‑computed 5‑row example where the same input produces **opposite labels** under the two conventions (Sreeram's: SL touch with ret = −0.105, label 0; Harry's: PT touch with ret = +0.201, label 1). The case is committed as `test_off_by_one_fix_changes_label_5_row_ohlc`.

**Decision 2 — Symmetric default, asymmetric API.** `TripleBarrierConfig(h=10, pt_mult=1.0, sl_mult=1.0, vol_span=100)`. Audit's 0 trend / 4 mean_reverting / 7 mixed canonical tags are too equivocal to bake bias into the label. `test_asymmetric_barriers_change_label_distribution` constructs three events with injected sigma and verifies wider PT → fewer label=1 (the direction the original Step‑2 spec example had reversed).

**Decision 3 — Trading‑day concurrency.** Uniqueness on each instrument's `close.dropna().index` (no calendar grid). Implemented via diff/cumsum on the bar position array — O(n_events + n_bars). Calendar weekends inserted by reindexing don't inflate or shrink concurrency. `test_trading_day_concurrency_invariant_to_weekend_padding` builds the panel on business days alone and on a calendar grid with NaN‑padded weekends and verifies identical `uniqueness_weight`, `t_end`, `label`, `ret`.

**Output schema (`results/harry/events.csv`):**

```
instrument         lowercased ticker
t_signal           date of the primary signal (close of t)
t_start            entry date = t_signal + 1 trading day
t_end              resolution date (first PT/SL touch or vertical)
side               +1 (long) or -1 (short)
ret                side * (log(close[t_end]) - log(close[t_start]))
label              1 if ret > 0 else 0
uniqueness_weight  AFML Ch.4 per-instrument, in [0, 1]
sigma              σ_t at the signal date (the barrier scale)
```

`persist_events.py` writes BOTH `events.csv` AND `events.meta.json` carrying SHA256 of the CSV bytes + a deterministic config hash + per‑instrument label balance + generation timestamp. `test_events_consistency.py` reads only the persisted files and validates drift.

**Per‑instrument numbers on the real released window (committed `events.meta.json`):**

```
        n_events  n_long  n_short  n_label_1  n_label_0
cl1s         411     375       36        287        124        pos=0.698
es1s         564     446      118        326        238        pos=0.578
fesx1s       626     286      340        323        303        pos=0.516
gc1s         161     129       32         99         62        pos=0.615
hg1s         617     298      319        317        300        pos=0.514
ho1s          63      53       10         41         22        pos=0.651
ng1s         120       0      120         67         53        pos=0.558
nq1s         593     397      196        354        239        pos=0.597
pl1s         547     411      136        282        265        pos=0.516
rb1s         617     358      259        315        302        pos=0.510
si1s         567     297      270        307        260        pos=0.541
TOTAL       4886
```

* `cl1s` leads at 0.698 label‑1 (matches audit's 0.71 forward hit rate).
* `ng1s` is 0 long / 120 short — single‑direction (independently flagged by signal‑deep‑dive's characterization).
* `ho1s` has only 63 events.
* Sum of uniqueness weights ≈ 644 → effective sample size 13 % of 4886 (heavy overlap because h=10 exceeds typical inter‑signal gaps).

Tests: 19 unit tests in `test_labels.py` covering config validation, EWMA causality (truncation invariance), the off‑by‑one fix, asymmetric barriers (label rate direction and per‑event monotonicity), uniqueness (zero/one bounds, disjoint = 1, fully overlapping = 0.5), trading‑day concurrency invariance, touch semantics (vertical, first PT/SL wins, short side, invalid inputs), output schema/dtype.

### 4.4 Step 3a‑3g — Feature pack (`src/stml/harry/features/`)

7 submodules, 22 public feature functions; every function registers itself in a module‑level `CAUSALITY_REGISTRATIONS` list that `tests/harry/test_causality.py` auto‑discovers and parametrises three universal tests over: truncation invariance, shape preservation, no NaN/Inf past warmup. Adding a feature → automatic harness inclusion, no test file edits required.

`np.random.default_rng(seed=42)` everywhere; per‑row derived seed `seed × 1_000_003 + t` for any bootstrap so truncated‑input output equals full‑input output exactly.

| File | Public functions | Warmup | What it captures | Citations |
|---|---|---:|---|---|
| **`signal_trajectory.py`** | `signal_run_length` (0), `time_since_last_flip` (0), `signal_entropy_20d` (19, Shannon entropy of {‑1,0,+1} PMF), `signal_flip_rate_60d` (60), `signal_cum_pnl_20d` (19) | 0–60 | The signal's **own structure** — the only non‑price labelled input | AFML Ch.3, Programming Session 4 |
| **`conditional_risk.py`** | `expected_hit_time` (252), `prob_timeout` (252, both **bootstrap first‑passage MC**), `path_tortuosity_20d` (19, `Σ\|r\|/\|Σr\|`), `realized_semi_vol_ratio` (19, RMS+/RMS−) | 19–252 | Conditional distribution of barrier resolution; path shape | Cont & Tankov (2003), Markowitz (1959) |
| **`information_theoretic.py`** | `rolling_mutual_information_252d` (251, 5‑quantile binned MI in nats), `transfer_entropy_vol_to_signal_acc` (126, Schreiber lag‑1 TE via 4 Shannon entropies) | 99–251 | Non‑linear dependence beyond Pearson correlation | Shannon (1948), Schreiber (2000), Cover & Thomas Ch.2 |
| **`microstructure_fixed.py`** | `amihud_illiquidity` (19, masks zero‑volume rows), `rolls_effective_spread` (21), `kyles_lambda` (19, mean `\|r\|/√volume`), `overnight_gap` (1, takes explicit `close.shift(1)`) | 1–21 | Liquidity / microstructure noise — **with zero‑volume mask correctly applied (Sreeram's G4 doesn't mask, propagates Inf)** | Amihud (2002), Roll (1984), Kyle (1985), Hasbrouck (2009) |
| **`cross_asset.py`** | `distance_to_lead_lag_centroid` (126 default), `asset_class_dispersion_z` (63), `ewma_implied_corr_z` (252, EWMA halflife=20) | 63–252 | How this instrument sits in the panel; crisis correlation indicator | Pollet & Wilson (2010), AFML Ch.25 |
| **`wavelet.py`** | `mra_energy_bands` returns a 5‑column DataFrame (251, db4 wavelet, 5 detail levels, `mode="periodization"`) | 251 | Energy at ~daily/weekly/bi‑weekly/monthly/quarterly scales | Percival & Walden (2000), Gencay et al. (2002) |
| **`concept_drift.py`** | `regime_alignment_score` (180, rolling LogReg discriminator P(row is "recent" vs train‑era)) | varies | Quantification of Sreeram's v5 regime‑break critique as a per‑row feature | Sugiyama & Kawanabe (2012), AFML Ch.7 |

Total: **22 public functions** → **25 produced feature columns** (the wavelet returns 5 columns, so 20 scalars + 5 wavelet).

The wavelet module raises a clean `ImportError` pointing to `SETUP.md` if PyWavelets isn't installed, so a clone without the optional extra still loads the rest cleanly.

**Causality harness:** 22 features × 3 properties = 66 parametrised checks. Synthetic input panels (`_synth_single_instrument`, `_synth_returns_panel`) defined once; **adapters** map the synthetic columns to each feature's actual signature. The harness verifies at three sample t values (100, 200, 400). The doc claims `pytest tests/harry/` shows **239 tests passing**.

Unit‑test counts per family:

```
test_signal_trajectory.py        13
test_conditional_risk.py         12
test_information_theoretic.py    11
test_microstructure_fixed.py      9
test_cross_asset.py              12
test_wavelet.py                   6
test_concept_drift.py            10
test_macro_features.py           42
test_labels.py                   19
test_signal_audit.py             21
test_events_consistency.py        5
test_causality.py                 3 (universal harness × N registrations)
```

### 4.5 Step 3.5b — Macro features (M1–M6) (`src/stml/harry/features/macro_features.py`)

Six groups of external macro features, all with the universal causality contract, sourced from the alternate data panel (see §4.6). 31 produced features in total. `MACRO_INSTRUMENT_TARGETS` dict documents which instruments each feature primarily targets.

| Group | Features | Source columns | Warmup |
|---|---|---|---:|
| **M1 vol / term structure** | `vix_level_z`, `vix_5d_change`, `vix_term_slope` (VIX3M − VIX), `move_z`, `move_vix_ratio`, `skew_z` | `VIX`, `VIX3M`, `MOVE`, `CBOE_SKEW` | 252 |
| **M2 rates / curve** | `us_2s10s_slope` (10Y_UST − 2Y_UST), `ust_10y_5d_change`, `bund_10y_5d_change`, `ust_bund_spread`, `real_yield_10y` (TIPS10Y z), `breakeven_10y` (BE10Y z), `be_5d_change` | `10Y_UST`, `2Y_UST`, `10Y_BUND`, `TIPS10Y`, `BE10Y` | 252 |
| **M3 credit** | `hy_oas_z`, `hy_oas_5d_change`, `ig_oas_z`, `hy_ig_ratio` | `HY_OAS`, `IG_OAS` | 252 |
| **M4 FX / dollar** | `dxy_z`, `dxy_5d_change`, `eurusd_5d_change` | `DXY`, `EURUSD` | 252 |
| **M5 commodity fundamentals** | `crude_stock_surprise`, `dist_stock_surprise`, `gasoline_stock_surprise`, `ng_stock_surprise` (all **release‑count windows**: `(Q_t − mean(prev n)) / std(prev n)` on the release sub‑series, not calendar days; default `n=5`), `copper_stock_z`, `baltic_dry_z`, `baltic_5d_change` | `EIA_CRUDE_STOCK`, `EIA_DIST_STOCK`, `EIA_GASOLINE_STOCK`, `EIA_NG_STOCK`, `LME_COPPER_STOCK`, `BAL_DRY_INDEX` | ~30 (EIA), 252 (z) |
| **M6 macro growth** | `ism_pmi_level`, `ism_pmi_3m_change`, `china_pmi_level`, `global_pmi_breadth` (fraction of {US ISM, China PMI} > 50; range {0, 0.5, 1.0}) | `US_ISM_MFG_PMI`, `CHINA_PMI_MFG` | 63 |

EIA surprise causality: `_eia_surprise` shifts the release sub‑series by one (so rolling stats at release `k` use releases `[k−n, …, k−1]`), then reindexes to the daily index with `ffill`. Both operations are causal. Macro‑cadence sanity assertion in `_macro_sanity_check`: EIA cols should have `zero_diff_frac > 0.70` (weekly release cadence), PMI > 0.90 (monthly), early z‑score rows must be NaN.

42 unit tests in `test_macro_features.py` (per build doc and confirmed by grep).

### 4.6 Alternate data (Bloomberg‑style) (`data/alternate_data_cleaned.csv`)

**This is the key piece you were trying to add to Sreeram — it's already on Harry's branch.** Source: `DATA_SYS_PASTED.xlsx` cleaned through `cleaning_alternate_data.ipynb`. 21 series + `Date`, 8479 rows, **1990‑01‑02 → 2022‑06‑30**, fully causal (no future leakage), inside the assignment training window.

Columns:

```
Date
10Y_BUND, 10Y_UST, 2Y_UST, TIPS10Y, BE10Y          (rates / curve)
HY_OAS, IG_OAS                                       (credit)
VIX, VIX3M, MOVE, CBOE_SKEW                          (volatility)
DXY, EURUSD                                          (FX)
EIA_CRUDE_STOCK, EIA_DIST_STOCK, EIA_GASOLINE_STOCK,
EIA_NG_STOCK, LME_COPPER_STOCK, BAL_DRY_INDEX        (commodity fundamentals)
CHINA_PMI_MFG, US_ISM_MFG_PMI                        (macro growth)
```

Engineered features are also pre‑computed and stored as `data/meta/macro_features.csv` (8478 rows × 32 columns: `Date` + 31 `f11_*` features, matching the M1–M6 catalog above).

Coverage caveats embedded in the data:

* `BE10Y` and `TIPS10Y` start later in the 1990s (TIPS began 1997).
* `EIA_DIST_STOCK`, `EIA_GASOLINE_STOCK`, `EIA_NG_STOCK` start later (weekly EIA releases for these specifics began in the late 1990s/early 2000s).
* `CBOE_SKEW`, `MOVE` available from 1990; `VIX3M` from later (CBOE 3‑month index started in 2007).
* `CHINA_PMI_MFG` from 2005‑ish.

The HMM macro module (§4.8) deliberately keeps the pre‑sample to the period where the relevant subset of columns is non‑null.

### 4.7 `new_work` — HMM volatility (Core #1) (`src/stml/new_work/hmm_vol.py`)

Per‑instrument 3‑state Gaussian HMM on **Garman‑Klass log realised vol**:

```
σ²_GK = 0.5 · (ln H/L)² − (2 ln 2 − 1) · (ln C/O)²
observation = log(max(σ²_GK, 1e‑10))   1‑D
```

GK uses the full intraday range so estimation variance is ~half of close‑to‑close at the same sample size.

**Causality protocol:**

1. **Pre‑sample** = all dates before `cutoff = primary_signals.date.min() = 2020‑01‑03`.
2. Standardise the observation using **pre‑sample mean/std only**.
3. Fit `GaussianHMM(n_components=3, covariance_type='diag', n_iter=300, tol=1e‑5)` on the pre‑sample with **20 random restarts**, pick the model with the highest log‑likelihood.
4. **Sort states ascending by mean log‑vol** → state 0 = calm, state 2 = turbulent. Reorders `startprob_`, `transmat_`, `means_`, `_covars_` consistently.
5. **Freeze** the parameters.
6. Run a **hand‑rolled scaled forward algorithm** (`filter_forward(model, X)`) — log‑domain with `logsumexp` normalisation per step — on the full sequence (pre‑sample + metamodel window). Explicitly **not** `hmmlearn.predict_proba` (smoothed). Returns one row per (date, instrument), filtered posteriors P(state | X_{0..t}).

Features (per date, per instrument):

| Feature | Definition |
|---|---|
| `hmm_vol_p0_calm` | filtered P(state 0) |
| `hmm_vol_p2_turbulent` | filtered P(state 2) (`p1` omitted because they sum to 1) |
| `hmm_vol_next_turbulent` | one‑step forecast = `α_t · Q[:, 2]` |
| `hmm_vol_entropy` | Shannon entropy `H(α_t)` — regime ambiguity scalar |

Output: `src/stml/new_work/features_hmm_vol.csv`. 645 dates × 11 instruments = 7095 rows.

### 4.8 `new_work` — HMM macro (Core #2) (`src/stml/new_work/hmm_macro.py`)

**Global** 2/3‑state HMM on a 4‑D macro observation:

```
f11_hy_oas_z         (credit stress ↑)
f11_vix_level_z      (equity fear ↑)
f11_us_2s10s_slope   (recession risk ↓)
f11_dxy_5d_change    (USD flight to safety ↑)
```

Standardised on pre‑sample, scaler frozen, same forward‑filter protocol.

**M selection** = BIC + forward‑chaining time‑series CV, evaluated for `M ∈ {2, 3}`:

* BIC = −2·ℓ̂_total + k·log(n), k = (M−1) + M(M−1) + 2Md.
* Forward‑chain TS‑CV: pre‑sample split into 6 folds; for fold i, train on `[0:split_i]`, validate on `[split_i:split_{i+1}]`; metric = mean per‑observation held‑out log‑lik.
* Selection rule: pick M where BIC and CV agree; on disagreement tie‑break to fewer states (parsimony + longer expected dwell).

**State ordering** by ascending composite risk score `mean_VIX_z + mean_HY_OAS_z` → state 0 = risk‑on, state M−1 = risk‑off / stress.

Features (global, broadcast to every (date, instrument)):

| Feature | Definition |
|---|---|
| `hmm_macro_p0`, `hmm_macro_p1` (only if M=3) | filtered P(state s) for s < M−1 (drop redundant last) |
| `hmm_macro_next_riskoff` | `α_t · Q[:, M−1]` |
| `hmm_macro_entropy` | Shannon entropy |

Output: `src/stml/new_work/features_hmm_macro.csv`. Same 7095 rows.

### 4.9 `new_work` — GARCH triple‑barrier (`src/stml/new_work/triple_barrier.py`)

A second labelling implementation, this time with **GARCH(1,1)** σ:

```python
sigma_garch(close, h, refit=21, min_obs=500, max_window=2000)
```

Causal expanding‑window GARCH(1,1) (zero mean, Normal innovations) fit via `arch.arch_model`. Refits every 21 trading days; in between, σ is forward‑filled. h‑day variance is summed across the h forecast steps. Returns scaled ×100 internally for numerical stability, then unscaled.

`label_signals_fixed(ohlcv, signals, h=10, pt_mult=1.5, sl_mult=1.0, min_ret=0.0, garch_refit=21, garch_min_obs=500, garch_max_window=2000)` — **asymmetric default `pt=1.5, sl=1.0`** (not symmetric like Harry's other labeller). Output columns:

```
date, instrument, side, t1, ret, bin, trgt, h, pt_mult, sl_mult,
sigma_method ('garch'), avg_uniqueness
```

Note: `bin` (1 if `ret > min_ret`) replaces the `label` name from `harry/labels.py`. Same uniqueness convention (per‑instrument bar index, diff/cumsum trick).

Persisted as `data/meta/triple_barrier_labels_fixed.csv` (4895 rows, header `date,instrument,side,t1,ret,bin,trgt,h`). Two sibling files: `triple_barrier_labels.csv` (4895 rows, very similar to `_fixed`) and `triple_barrier_labels_optimised.csv` (4854 rows, **CPCV‑search winning config** — pos rates near 0.48–0.65, distinctly lower than `_fixed`).

### 4.10 `new_work` — CPCV barrier search (`src/stml/new_work/cpcv_search.py`)

**Combinatorial Purged K‑fold (AFML Ch.12)** grid search over (h, pt_mult, sl_mult).

* Grid: `H_GRID = (1, 2, 3, 4, 5, 7, 10, 15, 20)`, `PT_GRID = SL_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)`. 324 configs.
* `IN_SAMPLE_FRAC = 0.70` — search uses only the first 70 % of each instrument's signal dates. Terminal 30 % is **never touched**.
* `N_GROUPS=6, K=2` → C(6,2) = **15 test paths**. `EMBARGO=0.01` (1 % of total date span after each test block).
* `BALANCE_LAMBDA=0.5`. Objective = `mean(AUC across folds) − 0.5 · |class_balance − 0.5|` — penalises configs that make the label trivially predictable by majority class.
* `MIN_IS_EVENTS=40` to skip degenerate configs.
* Thin instruments (`ho1s`, `ng1s`) are **pooled with their asset class** rather than searched separately.

The 7 placeholder features used inside the search (documented as a "minimal self‑contained" set that will be replaced by `stml.harry.features` for production):

```
mom_5d, mom_20d, vol_20d, vol_60d, ret_z_60d, side, trgt
```

Inside each fold, train a depth‑4 RandomForest with `min_samples_leaf=20`, `class_weight='balanced'`, `random_state=42`, sample weights = `avg_uniqueness × |ret|`.

`compute_pbo(fold_aucs)` runs **CSCV** (Bailey et al. 2016): for each of the 15 paths as held‑out, compute IS performance on the remaining 14, rank the IS‑best config OOS. PBO = fraction of held‑outs where the IS‑best ranks below median OOS. High PBO (> 0.5) → IS‑best is likely a random winner.

`label_signals_optimised(ohlcv, signals, best_row)` produces `data/meta/triple_barrier_labels_optimised.csv` using the winning config.

`check_plateau(results, best_row)` reports whether near‑best configs perform within noise — important for "is the winner sharp or flat" interpretation in the report.

### 4.11 `new_work` — Model comparison harness (`src/stml/new_work/model_comparison.py`)

The 30‑mark "Model Development and Comparison" deliverable, properly.

**Groups (13):** 10 individual instruments (excludes `ho1s` from individual — too thin) plus 3 pooled groups:

```
energy_all    = [cl1s, ho1s, rb1s, ng1s]
energy_cl_ho  = [cl1s, ho1s]
precious      = [gc1s, si1s, pl1s]
```

For pooled groups, **one‑hot instrument dummies** (`inst_*`) are added as features after hygiene so the model absorbs per‑name base‑rate and level differences.

**`INSTRUMENT_REGIMES` — competing regimes per instrument** (used in the selection table):

```python
INSTRUMENT_REGIMES = {
    "es1s":   ["es1s"],
    "nq1s":   ["nq1s"],
    "fesx1s": ["fesx1s"],
    "cl1s":   ["cl1s", "energy_all", "energy_cl_ho"],
    "ho1s":   ["energy_all", "energy_cl_ho"],
    "rb1s":   ["rb1s", "energy_all"],
    "ng1s":   ["ng1s", "energy_all"],
    "gc1s":   ["gc1s", "precious"],
    "si1s":   ["si1s", "precious"],
    "pl1s":   ["pl1s", "precious"],
    "hg1s":   ["hg1s"],
}
```

**4 model families:**

| Family | Implementation | Tuning grid |
|---|---|---|
| `logistic` | StandardScaler → LogReg(`penalty='elasticnet'`, `solver='saga'`, `class_weight=balanced`) | `C ∈ {0.01, 0.1, 1.0}` × `l1_ratio ∈ {0.0, 0.5, 1.0}` |
| `rf` | RandomForest(`max_features='sqrt'`, `class_weight=balanced`, sample weights = `avg_uniqueness`) | `max_depth ∈ {2, 4, 6}` × `min_samples_leaf ∈ {10, 20}` |
| `xgb` | XGBoost (subsample 0.8, colsample 0.8, `min_child_weight=5`, `reg_lambda=2`, `reg_alpha=0.1`, `gamma=0.1`, `scale_pos_weight = n_neg / n_pos`, early stopping rounds 20) | `max_depth ∈ {3, 4}` × `lr ∈ {0.01, 0.05}` × n_estimators via early stopping |
| `mlp` | StandardScaler → MLPClassifier (Adam, `early_stopping=True`, `validation_fraction=0.2`) | hidden ∈ {(64,), (64, 32)} × `alpha ∈ {0.001, 0.01}` |

Nested tuning: outer CPCV (`n_groups=6, k=2`, embargo 1 %) → inner expanding‑window 75/25 split on each training fold. SEED = 42 throughout. AUC is the inner‑split selection metric.

**Outputs per (group, model):**

```
{group}/{model}/oos_predictions.csv          one row per OOS event with date, instrument, y_true, y_score, fold
{group}/{model}/per_instrument_metrics.csv   AUC mean/std, logloss, brier per instrument
{group}/{model}/metrics.csv                  pooled AUC mean/std across folds
{group}/{model}/calibration.png              calibration curve
{group}/{model}/hyperparams.json             chosen hyperparams per fold
```

Plus aggregate tables:

```
_cache/{group}_events.parquet               cached feature matrix per group
master_results.csv                          77 rows: every (group, model, instrument)
selection_table.csv                         first‑pass best per instrument
selection_table_v2.csv                      RECONCILED champions (see §4.14)
reconciliation_report.md                    the writeup
```

**Cached event‑matrix shapes:**

```
cl1s_events.parquet        405 events × 101 features
es1s_events.parquet        547 × 101
fesx1s_events.parquet      ~ 580 × 101
gc1s_events.parquet        ~ 144 × 101
hg1s_events.parquet        ~ 600 × 101
ng1s_events.parquet        ~ 108 × 101
nq1s_events.parquet        ~ 565 × 101
pl1s_events.parquet        ~ 528 × 101
rb1s_events.parquet        ~ 600 × 101
si1s_events.parquet        ~ 545 × 101
energy_all_events.parquet     1186 × 104 (incl. 3 inst_* dummies)
energy_cl_ho_events.parquet    466 × 102
precious_events.parquet       1247 × 103
```

Feature column prefixes used: `f1_*` (counter‑trend), `f2_*` (vol/dispersion), `f4_*` (PCA latent on F1+F2+F6+F7+F10 pre‑sample), `f5_*` (signal‑derived), `f6_*` (momentum contrast), `f7_*` (microstructure), `f8_*` (calendar), `f10_*` (price action), `f11_*` (macro M1–M6), `f12_*` (wavelet bands), `f13_*` (conditional risk), `f15_*` (cross‑asset), `hmm_vol_*`, `hmm_macro_*`, `inst_*` (pooled dummies). Total 101–104 features after hygiene.

### 4.12 `new_work` — Cluster importance (`src/stml/new_work/feature_importance.py`)

Section 4 of the rubric (10 marks). Pipeline:

1. **Hygiene** (`apply_hygiene`):
   * Drop columns with > 30 % NaN at event dates (`col_nan_threshold=0.30`).
   * Drop event rows with any remaining NaN feature (warmup trim).
   * Drop near‑zero‑variance columns (`NZV_THRESHOLD=1e‑6`).
   * Dedupe Spearman near‑perfect twins (`TWIN_THRESHOLD=0.99`, keep first‑encountered).
   * Logged step by step.
   * `_macro_sanity_check`: warns if EIA macro cols update too frequently (zero‑diff frac < 0.70), PMI < 0.90, or rolling z‑scores have non‑NaN values in the 1990‑91 warmup.

2. **Partition** (`assign_groups`):
   * **Correlation‑clusterable** prefixes: `f1_, f2_, f6_, f7_, f10_, f11_, f12_, f13_, f15_, hmm_vol_, hmm_macro_`.
   * **Hand‑assigned** prefixes: `f4_` → `F4_latent`, `f5_` → `F5_signal`, `f8_` → `F8_calendar`, `inst_` → `F_instrument`.

3. **Clustering** (`select_k`, `get_cluster_labels`):
   * Distance = `sqrt(1 − |Spearman ρ|)` on the corr‑cluster block.
   * **Ward linkage** via `scipy.cluster.hierarchy.linkage`.
   * K selected by maximising **silhouette** over `K ∈ range(3, 16)`. CH (Calinski‑Harabasz) and DB (Davies‑Bouldin) reported alongside.
   * `cluster_representatives` picks one feature per cluster: the member with smallest mean Spearman distance to clustermates (used for the dim‑reduction sanity check).
   * `build_cluster_map` combines: correlation clusters get auto‑labels `C{cid}_{dominant_F_prefix}` (with `_lowfreq_macro` suffix when 100 % `f11_*`), plus hand‑assigned groups by prefix.

4. **Importance inside CPCV** (`run_cpcv_importance`):
   * Adaptive CPCV params (`_adaptive_cpcv_params`): for thin instruments, `n_groups` drops from 6 to 3 and `min_samples_leaf` from 20 to 5, so a depth‑4 forest doesn't degenerate to single‑leaf trees on ~57 events.
   * Per fold:
     * Fit shared RF (depth 4, leaf 20, class_weight balanced, sample weights = uniqueness).
     * Compute base AUC.
     * **MDI** (`feature_importances_`) — flagged in the writeup as train‑set biased toward high‑cardinality features.
     * **PFI** — per feature, permute, measure AUC drop.
     * **Clustered MDA** — per cluster, permute the **whole block of cluster members with the same row permutation**, measure AUC drop. Headline metric.
     * **Tree SHAP** — `feature_perturbation='tree_path_dependent'`, `check_additivity=False` (works with `class_weight='balanced'` and SHAP 0.46+). Handles both legacy list output and modern ndarray (n, p, k) shapes.
   * Aggregate mean ± std across CPCV paths.

5. **Cross‑method rank agreement** — Kendall τ between MDA/MDI/SHAP cluster rankings.

6. **Dim‑reduction sanity** — confirm AUC holds with one representative per cluster vs all features.

7. **Within‑cluster breakdown** — for the top‑K significant clusters, rank members by mean|SHAP|, plus PC1 variance explained + top loadings.

Per‑instrument outputs in `outputs/{inst}/` and `outputs/importance/{inst}/`:

```
cluster_membership.csv        feature → cluster id mapping
cluster_k_metrics.csv         silhouette / CH / DB per K
clustered_mda_full.csv        per‑cluster MDA mean/std + rank
per_feature_importance.csv    PFI per feature
global_shap_summary.csv       per‑feature mean|SHAP|
rank_agreement.csv            Kendall τ matrix
cluster_crosscheck_table.csv  data‑clusters vs declared F‑groups
within_cluster_*.csv          top‑K cluster breakdowns
findings_note.txt             prose summary
dendrogram.png                hierarchical dendrogram
corr_problem_cluster.png      (where present)
cluster_three_metrics.png     K selection chart
clustered_mda_chart.png       bar chart
global_shap_chart.png         summary plot
*.png                         within‑cluster breakdowns
```

Three notebooks present the analysis:

```
src/stml/new_work/feature_importance_analysis.ipynb         (main: equity / es1s, nq1s, fesx1s)
src/stml/new_work/feature_importance_analysis_energy.ipynb  (cl1s, ho1s, rb1s, ng1s)
src/stml/new_work/feature_importance_analysis_metals.ipynb  (gc1s, si1s, hg1s, pl1s)
```

A revised "load‑only" notebook was committed last (commit `c8f4ce6 harry: replace importance notebook — load-only presentation of champion results`) so the notebook reads the persisted importance CSVs and presents them rather than recomputing — that's the one in the build.

### 4.13 `new_work` — Champion importance (`src/stml/new_work/champion_importance.py`)

After reconciliation (§4.14) identifies 4 signal‑bearing champions, this module re‑runs the cluster importance using the **actual champion model class** (XGB or RF) rather than the generic RF used in `feature_importance.py`. Same CPCV / hygiene / clustering steps; SHAP via TreeSHAP with `feature_perturbation='tree_path_dependent'`. Within‑cluster breakdown for the top 3 significant clusters per champion.

Per‑champion outputs in `outputs/importance/{cl1s, es1s, ho1s, rb1s}/`:

```
findings_note.txt                       prose summary (see §4.17 for numbers)
clustered_mda_full.csv                  per‑cluster MDA mean/std
clustered_mda_chart.png                 bar chart
cluster_membership.csv                  feature → cluster id
cluster_k_metrics.csv                   silhouette/CH/DB by K
cluster_crosscheck_table.csv            clusters × F‑groups
global_shap_summary.csv                 per‑feature SHAP
global_shap_chart.png                   summary plot
rank_agreement.csv                      Kendall τ MDA‑MDI / MDA‑SHAP / MDI‑SHAP
dendrogram.png
within_cluster_{cluster_id}_f{N}*.csv   top‑3 cluster breakdowns
within_cluster_{cluster_id}_f{N}*.png
```

### 4.14 `new_work` — Reconciliation (`src/stml/new_work/reconciliation.py`)

The sharpest methodological move on this branch.

**Problem:** the master_results CSV reports **pooled AUC** across all 15 CPCV paths' OOS — which understates performance on imbalanced folds. The fair metric is **per‑fold mean ± std AUC**.

**Pipeline:**

1. **`per_fold_stats(oos_df, instrument)`** — recompute mean and std of AUC per (group, model, instrument) using each fold's slice. Folds with single‑class slices are skipped. `wide_ci` flagged when fewer than 8 valid folds (thin instruments).

2. **Signal vs no‑signal classification:** `lower_ci = per_fold_mean − per_fold_std > 0.50` → "signal".

3. **STD‑based tie detection:** within an instrument's competing regimes, any (group, model) within 1σ of the best AUC is "tied".

4. **Calibration of all tied candidates:** for each fold, inner 60/40 time split of training → simplified fixed‑hyperparam inner model (`_inner_model_predict`) → predictions on the last 40 % become calibration samples → fit Platt + isotonic calibrators → apply to stored raw OOS scores → pick the lowest Brier (raw fallback if both calibrators worsen Brier). Dominant calibration method across folds reported as `none` / `sigmoid` / `isotonic`.

5. **Final champion:** lowest calibrated Brier → tiebreak by model simplicity (`logistic < rf < xgb < mlp`).

**Reconciled champion table (`selection_table_v2.csv`):**

| Instrument | Signal? | Champion | Per‑fold AUC ± std | Lower CI | Brier raw → cal | Calib method | Notes |
|---|---|---|---:|---:|---|---|---|
| **cl1s** | **YES** | `cl1s` / **XGB** | **0.707 ± 0.130** | 0.577 | 0.218 → 0.196 | none | 11 within 1σ |
| **es1s** | **YES** | `es1s` / **RF** | **0.605 ± 0.069** | 0.536 | 0.246 → 0.243 | none | 4 within 1σ |
| **ho1s** | **YES** | `energy_cl_ho` / **RF** | **0.634 ± 0.155** | 0.620 | 0.214 → 0.202 | none | 6 within 1σ; thin (63 events) |
| **rb1s** | **YES** | `energy_all` / **XGB** | **0.629 ± 0.118** | 0.514 | 0.249 → 0.247 | none | 6 within 1σ |
| fesx1s | no | — | 0.498 ± 0.048 | 0.450 | — | — | |
| gc1s | no | — | 0.543 ± 0.121 | 0.423 | — | — | |
| hg1s | no | — | 0.530 ± 0.064 | 0.467 | — | — | |
| ng1s | no | — | 0.590 ± 0.143 | 0.448 | — | — | **borderline** — softer 0.48 threshold would include |
| nq1s | no | — | 0.542 ± 0.108 | 0.434 | — | — | |
| pl1s | no | — | 0.541 ± 0.052 | 0.489 | — | — | |
| si1s | no | — | 0.499 ± 0.067 | 0.432 | — | — | |

**4 of 11 instruments produce a usable meta‑model under the strict lower‑CI > 0.50 criterion.** The remaining 7 are honestly classified no‑signal.

**Pooling verdict** (from `reconciliation_report.md` §5):

* `cl1s`: individual 0.707 vs pool 0.692 — within noise.
* `es1s`: individual only.
* `ho1s`: no individual (too thin) — pooling is the only option.
* `rb1s`: individual 0.617 vs pool 0.632 — within noise.

### 4.15 Notebooks (`notebooks/harry/`, `src/stml/new_work/`)

**`notebooks/harry/`:**

* `03-features-sanity.ipynb` — 36 cells (24 code, 12 md). Step 3 final feature sanity check (per commit history). Plots per‑feature distributions, warmup behaviour, correlation matrices.
* `04-hmm-vol-turbulence.ipynb` — 23 cells (13 code, 10 md). Core #1 walkthrough. Per‑instrument HMM fits, posterior visualisations, state interpretation.
* `05-hmm-macro-riskoff.ipynb` — 25 cells (15 code, 10 md). Core #2 walkthrough. M selection plots, posterior visualisations, alignment with NBER recession dating.

**`src/stml/new_work/`:**

* `feature_importance_analysis.ipynb` — equity instruments (`es1s`, `nq1s`, `fesx1s`).
* `feature_importance_analysis_energy.ipynb` — `cl1s`, `ho1s`, `rb1s`, `ng1s`.
* `feature_importance_analysis_metals.ipynb` — `gc1s`, `si1s`, `hg1s`, `pl1s`.

Per the final commit message, these became **load‑only presentations** that read the persisted importance CSVs/PNGs rather than recomputing.

### 4.16 Tests (`tests/harry/`)

**239 tests in total** per the build doc. Counts per file (from grep `def test_`):

```
test_signal_audit.py             21
test_labels.py                   19
test_signal_trajectory.py        13
test_conditional_risk.py         12
test_cross_asset.py              12
test_information_theoretic.py    11
test_concept_drift.py            10
test_microstructure_fixed.py      9
test_wavelet.py                   6
test_events_consistency.py        5
test_macro_features.py           42
test_causality.py                 3   (× registrations = ~66 parametrised)
TOTAL (parametrised expanded)  ≈ 239
```

`conftest.py` injects `src/` into `sys.path` so the editable install isn't strictly needed for testing. `test_causality.py` is the universal harness: 3 properties × every registered feature, on a fixed synthetic panel at sample t values 100, 200, 400. Adding a feature with a `CAUSALITY_REGISTRATIONS` entry auto‑enrolls it.

### 4.17 Output inventory

**`results/harry/`:**

```
events.csv                          4886 rows  (the canonical labelled events)
events.meta.json                    SHA256 + per‑instrument label balance + config hash
signal_direction.csv               11 rows × ~30 cols (audit headline table)
signal_direction_stability.csv     88 rows (audit on full + halves)
```

**`src/stml/new_work/`:**

```
features_hmm_vol.csv               645 × 11 = 7095 rows (per‑inst HMM filtered posteriors)
features_hmm_macro.csv             same shape (global macro HMM broadcast)
```

**`data/meta/` (Harry additions):**

```
macro_features.csv                 8478 × 32 (Date + 31 f11_*)
triple_barrier_labels.csv          4895 rows  (initial)
triple_barrier_labels_fixed.csv    4895 rows  (h=10, pt=1.5, sl=1.0, GARCH)
triple_barrier_labels_optimised.csv 4854 rows (CPCV‑search winning config; pos_rate ~0.48‑0.65)
```

**`src/stml/new_work/outputs/`:**

```
model_comparison/
  master_results.csv               77 rows (every group × model × instrument)
  selection_table.csv              first‑pass champion per instrument
  selection_table_v2.csv           reconciled champions
  reconciliation_report.md         the full writeup
  {13 groups} / {4 models} /       4 model files per (group, model): metrics, per‑instrument,
                                    oos_predictions, hyperparams.json, calibration.png
  _cache/{group}_events.parquet    cached event feature matrices

importance/
  cl1s/, es1s/, ho1s/, rb1s/       champion importance — findings_note.txt + ~12 files each

{individual instrument folders}    earlier per‑instrument importance from feature_importance.py
warmup_diagnosis.md                investigative note (no code changes)
main_findings.csv                  equity importance summary
main_findings_energy.csv           energy importance summary
k_selection*.png                   K selection charts
```

**Champion importance findings (`outputs/importance/{inst}/findings_note.txt`):**

**cl1s / XGB (AUC 0.707):**

Significant clusters (mean > 1σ):

| Cluster | MDA mean ± std | Dominant F‑group |
|---|---|---|
| **C4_f2** | **+0.133 ± 0.122** | f2 (80 % pure) |
| **C12_f11** | **+0.085 ± 0.074** | f11 (62 % mixed) |

Top within‑cluster (mean|SHAP|):

* `C4_f2 (PC1 = 53.5 %)`: `f2_vol_60` 0.518, `hmm_vol_entropy` 0.036, `f2_ret_kurt_60` 0.025, `f2_vol_of_vol_20` 0.023.
* `C12_f11 (PC1 = 30.1 %)`: `f11_move_z` 0.274, `f11_gasoline_stock_surprise` 0.068, `f1_bb_bandwidth_20` 0.051, `f11_baltic_dry_z` 0.026.

Cross‑method rank agreement: **MDA–MDI τ = +0.35 disagree**, **MDA–SHAP τ = +0.25 disagree**, **MDI–SHAP τ = +0.80 agree**. The disagreement is itself informative: tree splits (MDI/SHAP) reward different features than the marginal contribution (MDA).

**es1s / RF (AUC 0.605):**

Top within‑cluster (significant clusters from the file, summarised):

* `F5_signal (PC1 = 44.5 %)`: `f5_trailing_run_length` SHAP 0.034, `f5_long_bias_20` 0.008, `f5_signal` 0.003, `f5_participation_20` 0.002.
* `C8_f11 (PC1 = 44.7 %)`: `f11_hy_ig_ratio` 0.010, `f11_move_vix_ratio` 0.008, `f11_ust_bund_spread` 0.007.
* `C2_f11 (PC1 = 64.9 %)`: `f11_breakeven_10y` 0.007, `f11_copper_stock_z` 0.003.

MDI‑SHAP τ = +0.94 (agree).

**ho1s / energy_cl_ho / RF (AUC 0.634 ± 0.155):**

**No cluster is 1σ‑significant.** With only 63 events many CPCV folds drop to single‑class. The doc flags it as "indicative" — high AUC point estimate is real, but the importance picture is too noisy to interpret.

Notable directional reads anyway: `C13_f11` +0.066, `F4_latent` +0.023, `C10_f11_lowfreq_macro` +0.022, `F5_signal` +0.021 (all swamped by std).

**rb1s / energy_all / XGB (AUC 0.629):**

Significant clusters: **F5_signal alone (+0.113 ± 0.082)** — the cleanest "trajectory‑driven" champion. All other clusters within 1σ of zero.

**Per‑group main findings (`main_findings.csv`, `main_findings_energy.csv`):**

```
main_findings.csv (equity)
inst    full_AUC  n_clusters  reduced_AUC  top_cluster              MDA top  bottom        bottom_MDA  notes
es1s    0.557     14          0.577        F5_signal                +0.066   C2_f1         −0.021
nq1s    0.442     15          0.390        C12_f1                   +0.035   C6_f7         −0.022
fesx1s  0.509     18          0.488        C13_f11_lowfreq_macro    +0.020   C10_f12       −0.010    ⚠ print‑day signal

main_findings_energy.csv
inst    full   reduced  n_clust  top_cluster              MDA       MDI_top  SHAP_top  Fg       note
cl1s    0.593  0.587    18       C4_f2                    +0.037    0.144    0         f2
ho1s    0.500  0.500    17       C1_f11_lowfreq_macro     +0.000    0.000    0         f11      ⚠
rb1s    0.530  0.642    18       F5_signal                +0.055    0.088    0         mixed
ng1s    0.531  0.597    18       C9_f7                    +0.088    0.100    0         f7
```

The "print‑day signal" flag on `fesx1s` notes that one of its macro features changes only on calendar release days; if a cluster led by that feature shows up in the importance table, it might be a calendar artifact rather than a real signal.

### 4.18 Open issues on Harry

1. **No consolidated `predictions.csv` deliverable.** The branch outputs `oos_predictions.csv` per (group, model) but no single `(date, instrument, prediction)` CSV in the assignment format. To submit Harry alone you'd need to stitch the four champions' predictions together and pad with `0.5` (or 0.0?) for the 7 no‑signal instruments.
2. **7 instruments have no signal under strict criterion.** fesx1s/nq1s/si1s/gc1s/hg1s/pl1s/ng1s are honestly classified random. That's correct methodology but it constrains the strategy track.
3. **No strategy module.** Harry's original plan (Step 5) was Kelly‑fractional + vol‑targeted sizing. Not shipped. So no `strategy_weights.csv`.
4. **No team‑synthesis memo.** `99-team-synthesis-memo.md` is referenced in `00-context.md` as the planned Step 6 but is not committed. The audit/labels/features writeups are complete; the higher‑level "how Harry/Sreeram/signal‑deep‑dive reconcile" memo isn't.
5. **VSN deferred.** `SETUP.md` says the planned PyTorch VSN didn't materialise. `MlpMeta` is the NN slot. Same as Sreeram.
6. **TDA module deferred.** Ripser C++ build issues. Doesn't matter for the rubric (7 feature families are already plenty).
7. **`triple_barrier_labels_optimised.csv` exists but isn't consumed downstream.** `feature_importance.py` and `model_comparison.py` both read `triple_barrier_labels_fixed.csv` (h=10, pt=1.5, sl=1.0). The CPCV search winning config never reaches the production champions. The CPCV pipeline is a methodological showpiece but not load‑bearing.
8. **Pre‑sample dependency.** Both HMMs (vol and macro) are fit on pre‑2020 history and frozen. That's leakage‑safe but **non‑adaptive on the grader's rerun** — the boundary moves to 2022‑07‑01 but the HMMs' pre‑sample stays the same. The HMM posteriors will continue to use the same trained parameters on the new H2‑22 window. Fine if the pre‑2020 distribution still represents 2022 dynamics; exactly the assumption Sreeram's v5 stress test invalidated.
9. **GARCH labels asymmetric, Harry's harry/labels.py symmetric.** Two coexisting label conventions on the same branch — the production champions use the asymmetric GARCH (`_fixed`), while the audit‑justifying writeup uses the symmetric EWMA. Both are documented, but a future reader has to read carefully to keep them straight.
10. **`hg1s`'s per‑instrument MDA shows `C4_f2`'s `f2_vol_60` dominates at SHAP 0.518** (huge relative to other features) — worth a sanity check that this isn't a single‑feature near‑sufficient model dressed up as multi‑feature.

### 4.19 The 2026‑06‑02 PM methodology refresh — clean train/test split + 1SE selection + all‑11 champion importance

Four new commits on the afternoon of 2026‑06‑02 (`197952a`, `eb98fa3`, `9e96728`, `1e3242a`) landed a substantial methodology refresh. This subsection documents what changed, why it changed, and the new numbers it produced. The pre‑refresh content above (§4.1‑§4.18) is preserved because the audit, label, feature, macro, HMM, GARCH triple‑barrier and CPCV‑barrier‑search machinery is unchanged. What changed is downstream: the inner CV used for hyperparameter tuning, the train/test split used for importance, and the per‑instrument champion list.

#### 4.19.1 New module — `src/stml/new_work/split_config.py`

A 28‑line module that becomes the single source of truth for the global train / test split. From its docstring:

> "Global cut: 2021‑10‑06 (70th percentile of 4,764 pooled events across 11 instruments). Embargo end: 2021‑10‑20 (cut + 10 trading days). Train: date ≤ GLOBAL_CUT and t1 < GLOBAL_CUT (purge label‑window crossers). Test: date > EMBARGO_END (SEALED — do not read during training)."

Public API:

```python
GLOBAL_CUT  = pd.Timestamp("2021-10-06")   # 70th percentile of pooled events
EMBARGO_END = pd.Timestamp("2021-10-20")   # cut + 10 trading days

def apply_train_mask(events_df) -> pd.DataFrame:
    """Train rows: date <= cut AND t1 < cut (purge boundary-crossing events)."""

def apply_test_mask(events_df) -> pd.DataFrame:
    """Test rows: date > embargo_end. SEALED — do not call during training."""
```

Both functions are overridable via environment variables `STML_TRAIN_CUT` and `STML_EMBARGO_END`. The split is at the **70th percentile** of pooled events (~70/30 split with embargo) rather than at a calendar boundary — this gives each champion enough train rows even on thin instruments while keeping the test set representative.

Why this matters: under the previous architecture, model selection used CPCV across the full released window. Champion *importance* was then computed on the same full window. That conflated selection and importance attribution on the same data. The new architecture cleanly separates:

* **Selection** — purged inner k‑fold + 1SE on the train slice.
* **Importance** — CPCV (15 paths) on the train slice only.
* **OOS evaluation** — the SEALED 30 % held out for the final read.

#### 4.19.2 `model_comparison.py` — purged inner k‑fold + 1SE rule

The rewritten model‑comparison harness applies `split_config.apply_train_mask` inside `assemble_group` (so every cached group's event matrix is train‑only). The model_comparison cache parquet files were regenerated on train‑only data — the on‑disk parquets in `_cache/` are now smaller than the pre‑refresh ones.

The inner CV that picks hyperparameters changed from an expanding 75/25 split to a **purged inner k‑fold (k=4)**:

```python
class _PurgedKFold:
    """Sequential purged k-fold for inner hyperparameter tuning.

    Splits events (sorted by date) into k consecutive groups and yields
    (train_idx, val_idx) integer-position arrays with the same purge +
    embargo logic as CombinatorialPurgedKFold."""

    def __init__(self, k: int = 4, embargo: float = CPCV_EMBARGO):
        # k=4 → four 25 % windows, each used as inner val once
        # purge train rows whose t1 crosses into val
        # embargo train rows starting within (embargo · total_days) after val end
```

And the hyperparameter pick switched from "highest mean AUC" to the **1‑standard‑error rule**:

```python
def _select_1se(config_scores, regularisation_key) -> dict | None:
    """1SE rule: return the most-regularised config within 1SE of the best mean.

    SE = std(inner-fold AUCs) / √N  on the max-mean config.
    regularisation_key(cfg_dict) → sortable key, ascending = more regularised."""
```

For logistic: tie‑break is `(C, -l1_ratio)` — smaller C wins (more L2 penalty), higher l1_ratio wins (more sparsity). For RF: tie‑break is `(max_depth, -min_samples_leaf)` — shallower wins, larger leaf size wins. For XGB and MLP: analogous bias toward less capacity.

Two new exclusion sets are declared at module level:

```python
SKIP_INDIVIDUAL_GROUPS: frozenset[str] = frozenset({"ng1s"})       # too thin
NO_MLP_GROUPS:          frozenset[str] = frozenset({"cl1s", "gc1s"})  # MLP excluded
INSTRUMENT_REGIMES["ng1s"] = ["energy_all"]                        # was ["ng1s", "energy_all"]
```

ng1s no longer competes as an individual group (only via `energy_all` pooling). cl1s and gc1s exclude MLP from the horse race entirely.

The **purge guard** in the inner fold is the standard AFML rule: drop train events whose label window `[date, t1]` crosses into the validation window, plus a forward embargo equal to `embargo_frac · total_days`.

#### 4.19.3 `champion_importance.py` — full rewrite covering all 11 instruments

The 978‑line diff (546 new lines, 432 removed) rebuilds the champion importance analysis around two parallel branches keyed on the champion estimator family:

**Tree branch** (RF / XGB champions):

* Clustered **MDA** — joint permutation across cluster members, N = 10 repeats, scored on the target instrument's test slice when the champion comes from a pooled group.
* Clustered **MDI** — sum of `feature_importances_` within cluster (flagged as a train‑set statistic).
* Group **SHAP** — sum of mean‑|SHAP| within cluster via `TreeExplainer(feature_perturbation="tree_path_dependent")`.
* **Rank agreement** — Kendall τ across MDA / MDI / SHAP rankings.

**Logistic branch** (elastic‑net champions):

* Clustered **MDA** — same joint permutation, applied to scaled inputs (model‑agnostic).
* **Cluster Coef** — sum of `|standardised coefficient|` within cluster, averaged across CPCV folds (replaces SHAP — TreeSHAP is undefined for linear models).
* **Rank agreement** — Kendall τ between MDA and Coef rankings.

Both branches:

* Use the same Mantegna distance `√(1 − |Spearman ρ|)`, Ward linkage, silhouette‑K clustering on the train slice's continuous features.
* Add a **PC1‑PC3 within‑cluster PCA** decomposition for the top 3 clusters per instrument, with variance explained and top loadings.
* Run CPCV with `n_groups=6, k=2, embargo=0.01` (15 paths).

The new docstring header explicitly enumerates all 11 champions with their signal status — the entire 11‑instrument coverage is the new scope (vs the old 4‑champion focused scope).

#### 4.19.4 The new champion table — what changed

Reading `src/stml/new_work/outputs/model_comparison/selection_table.csv` (the post‑refresh table):

| Instrument | Best group | Best model | AUC | Lower CI | Signal? |
|---|---|---|---:|---:|---|
| **es1s** | es1s | RF | 0.516 | 0.404 | **NO** |
| **nq1s** | nq1s | **XGB** | **0.689** | **0.616** | **YES** |
| **fesx1s** | fesx1s | **logistic** | **0.579** | **0.519** | **YES** |
| **cl1s** | cl1s | **XGB** | **0.675** | **0.536** | **YES** |
| ho1s | energy_all | logistic | 0.800 | 0.496 | NO (close call) |
| rb1s | energy_all | logistic | 0.551 | 0.457 | **NO** |
| ng1s | energy_all | RF | 0.477 | 0.218 | NO |
| gc1s | precious | XGB | 0.478 | 0.377 | NO |
| si1s | si1s | XGB | 0.515 | 0.439 | NO |
| **pl1s** | pl1s | **logistic** | **0.608** | **0.527** | **YES** |
| **hg1s** | hg1s | **RF** | **0.604** | **0.562** | **YES** |

For comparison, the pre‑refresh champions (still in `selection_table_v2.csv`, kept as historical reference but **no longer the canonical pick**):

| Instrument | Champion | AUC | Lower CI | Signal? |
|---|---|---:|---:|---|
| cl1s | cl1s/xgb | 0.707 | 0.577 | YES (still) |
| es1s | es1s/rf | 0.605 | 0.536 | YES → **NO** |
| ho1s | energy_cl_ho/rf | 0.634 | 0.620 | YES → **NO** (close) |
| rb1s | energy_all/xgb | 0.629 | 0.514 | YES → **NO** |

**Diff in plain English:**

* **Survived both:** `cl1s` (the highest‑conviction signal across both methodologies — clearly the cleanest case).
* **Newly signal‑bearing:** `nq1s` (xgb, 0.689 with strong CI 0.616), `fesx1s` (logistic, 0.579 / 0.519), `pl1s` (logistic, 0.608 / 0.527), `hg1s` (rf, 0.604 / 0.562). All four were classified "no signal" under the old reconciliation; the new clean‑split + 1SE methodology rescues them.
* **Lost their signal flag:** `es1s` (was 0.605, now 0.516 — drops below the threshold), `ho1s` (was 0.634, now 0.800 mean AUC but the CI collapses to 0.496 — clearly a small‑sample artefact: only 170 training rows survive the train mask), `rb1s` (was 0.629, now 0.551).

The **net is 5 signal‑bearing instruments** (was 4), with a substantially different composition. cl1s is the only stable survivor; all other previously‑signal‑bearing instruments dropped out and four new ones emerged.

#### 4.19.5 What this means for interpretation

The refresh is a methodologically more defensible setup — a sealed test set, no contamination of the cv used for selection with the data used for importance — so the new champion list is the one to trust if Harry's pipeline is what we draw from. But it also means **importance findings from the pre‑refresh champion notes are partially obsolete**. Specifically the per‑instrument `findings_note.txt` files in `outputs/importance/{cl1s,es1s,ho1s,rb1s}/` that we documented in §4.17 are still on the branch but reflect the OLD champion selection. The new per‑asset‑class notebooks (`equity_importance.ipynb` etc.) are the authoritative interpretation source.

#### 4.19.6 New per‑asset‑class notebooks

`src/stml/new_work/equity_importance.ipynb` (40 cells, ~795 lines), `energy_importance.ipynb` and `metals_importance.ipynb` each run a uniform 5‑step methodology:

1. **Feature clusters** — Spearman distance, Ward linkage, silhouette‑K, hand‑assigned groups (F4 latent, F5 signal, F8 calendar, F_instrument dummies).
2. **Cluster‑level importance** — clustered MDA chart, MDI / SHAP / Coef cross‑check, Kendall τ rank agreement.
3. **Within‑cluster breakdown** — top 3 clusters by MDA → members ranked by mean|SHAP| (tree) or mean|coef| (logistic), PC1‑PC3 variance explained, top loadings.
4. **Global per‑feature view** — `global_shap_summary.csv` + chart (tree) or `global_coef_summary.csv` + chart (logistic).
5. **Cross‑asset summary** — at the end of each notebook, a one‑page synthesis of how the three instruments in the asset class compare.

Equity opens with:

> *"Champions from `selection_table.csv` — purged inner k‑fold + 1SE, global cut **2021‑10‑06**. All importance analysis uses **train data only** (post‑split). The held‑out 30 % (post‑2021‑10‑20) is **sealed**."*

The notebooks render `outputs/importance/{instrument}/*.png` and tabulate the corresponding CSVs. They are the authoritative read of the new champion analysis.

#### 4.19.7 Deletions and superseded artefacts

The cleanup commit `1e3242a` removed:

* `src/stml/new_work/feature_importance_analysis_energy.ipynb` — 651 lines deleted. Reason: "ran on full sample and are no longer valid".
* `src/stml/new_work/feature_importance_analysis_metals.ipynb` — 621 lines deleted, same reason.
* `src/stml/new_work/outputs/k_selection_energy.png` — stale.
* `src/stml/new_work/outputs/main_findings_energy.csv` — 5 rows.
* `src/stml/new_work/outputs/main_findings_top5_energy.png` — stale.

The `feature_importance_analysis.ipynb` (the equity/main one) was kept and updated with a **5‑step methodology cell** added at the top. Older `outputs/importance/{instrument}/findings_note.txt` were also rewritten by `champion_importance.py` to reflect the new champion if applicable.

The `selection_table_v2.csv` (the old reconciled selection with calibrated Brier) is **kept** on the branch as historical reference but is no longer the canonical champion source — that's now `selection_table.csv`.

The `reconciliation_report.md` was removed (no longer applicable under the new selection scheme — there's no need to do calibration‑based tiebreak when the inner CV + 1SE already picks the most parsimonious model).

#### 4.19.8 Net effect on the branch's posture

Before the refresh, Harry's branch sold **methodological rigour through reconciliation** (calibration on tied candidates, simplicity tiebreak). After the refresh, it sells **methodological rigour through clean separation** (sealed test set, purged inner k‑fold, 1SE for parsimony). The new posture is closer to alken's discipline and easier to defend in a quant interview: there is a clear hold‑out, hyperparameters are not selected to maximise an already‑optimistic AUC, and the importance attribution is on the train slice only.

**Output inventory delta (compared to §4.17):**

* Added: `src/stml/new_work/split_config.py`, `equity_importance.ipynb`, `energy_importance.ipynb`, `metals_importance.ipynb`, per‑instrument `global_coef_summary.csv` / `global_coef_chart.png` for logistic champions, new within‑cluster CSV/PNG pairs for the new champion list.
* Removed: `reconciliation_report.md`, `feature_importance_analysis_{energy,metals}.ipynb`, the old pre‑split `outputs/{instrument}/*.{csv,png}` flat files at the asset‑class level.
* Renamed semantics: `selection_table.csv` is now the canonical champion source (was a first‑pass; `selection_table_v2.csv` had been the canonical reconciled version).

**Open issues introduced by the refresh:**

* The `selection_table_v2.csv` and the per‑instrument `findings_note.txt` files written before the refresh are inconsistent with the new champion list. A future agent must be told to read `selection_table.csv` and the three asset‑class notebooks, not the older artefacts.
* `ho1s` has the highest mean AUC of any instrument (0.800) but a CI that drops to 0.496 — the result is statistically thin (170 training rows after the train mask). The "energy_all" pool is what carries it; on its own ho1s is too sparse.
* The 1SE rule is biased toward strong regularisation. This is methodologically defensible (it's the standard "Occam's razor" tiebreak) but it can drop hyperparameter combinations that genuinely fit on this specific data. The trade‑off is documented in the model_comparison docstring.

#### 4.19.9 What this section means for the new pipeline we are about to build

Take Harry's `split_config.py` pattern verbatim — a sealed test set with a configurable cut and embargo. Take the purged inner k‑fold + 1SE selection rule for hyperparameter tuning. Take the train‑only‑for‑importance discipline. Do **not** take the specific champion choices (we'll do our own selection on our own feature set), nor the `INSTRUMENT_REGIMES` mapping (we'll likely model per asset class with multi‑task NN heads, which has different pooling semantics).

---

## 5. Branch `model/alken-metamodel` — methodology‑first per‑class metamodels

> 74 commits ahead of `main`; ~unknown LOC but the most disciplined commit narrative of the three branches (every commit message stage‑tagged: `feat(s1.7):`, `feat(ex6,s512):`, `docs(pass5):` …). Built as **two coexisting projects on one branch**: a shared feature library at the repo root, plus a **nested uv subproject** at `metamodel-apb/` carrying the modelling pipeline, experiments, methodology document, action‑item tracker, literature‑review notes, and a submission‑ready Harvard‑referenced academic report. The whole branch is engineered around the brief's "methodology, not performance" stance — the headline result is reported as an **honest negative** ("insufficient evidence of a deployable edge") corroborated by five independent statistical lenses.

### 5.1 Branch shape — two coexisting projects

```
stml/  (alken‑metamodel branch; marker • = new on this branch vs main)
├── data/
│   ├── ohlcv_data.csv                  unchanged
│   ├── primary_signals.csv             unchanged
│   ├── additional_data.xlsx          • 22 macro series workbook (F11 inputs)
│   ├── features/                     • one CSV per family — f1_..f17_*.csv
│   └── meta/                           unchanged + provenance entries
├── notebooks/                          shared notebooks + jay/ subfolder
├── refs/
│   ├── triple_barrier_guide.md       • López de Prado triple‑barrier reference
│   └── ...                             unchanged
├── reports/
│   ├── feature-catalog.md            • catalog of every produced feature column
│   ├── README.md                     • how to read the artifacts
│   └── missing-data-report.md          unchanged
├── results/
│   ├── feature_matrix.{csv,parquet}  • 4984 rows × 175 features (no label column)
│   ├── feature_redundancy.{csv,json} • Spearman + hierarchical cluster map
│   ├── instrument_scope.json         • per‑instrument scope incl. embargo_p90
│   ├── feature_matrix_provenance.json• fe_train_end + partition counts + seed
│   └── README.md
├── src/stml/
│   ├── __init__.py                     unchanged
│   ├── io.py                           unchanged shared loader
│   ├── na_checks.py                    unchanged shared NA infra
│   └── metamodel/                    • 13 modules — the shared feature library
│       ├── build_features.py
│       ├── catalog.py                  FeatureSpec registry + catalog renderer
│       ├── features.py                 F1/F2/F5/F6/F7/F8/F10 core E‑class
│       ├── features_ext.py             F2‑RS/F5‑adds/F7‑adds/F12/F13/F15 + add_z_twins
│       ├── regime_features.py          F3 (GMM + Markov‑switching, fitted)
│       ├── regime_features_hmm.py      F17 (3‑state Gaussian HMM, fitted)
│       ├── drift_features.py           F16 (rolling discriminator)
│       ├── latent.py                   F4 (StandardScaler + PCA + KMeans + AE)
│       ├── xsection.py                 F9 (cross‑sectional + cross‑asset)
│       ├── macro_features.py           F11 (PIT‑lagged macro, fitted)
│       ├── scope.py                    InstrumentScope registry
│       ├── splits.py                   Chronological train/val/test + embargo
│       └── pipeline.py                 FeaturePipeline.fit().transform()
├── tests/                            • 13 files — feature layer leakage/coverage/determinism
├── pyproject.toml                      shared base (top level)
├── uv.lock
├── README.md                         • the feature‑base README (~26 KB)
└── metamodel-apb/                    • THE NESTED MODELLING SUBPROJECT
    ├── CLAUDE.md                       Claude Code guidance for the subproject
    ├── pyproject.toml                  alken_metamodel uv subproject
    ├── uv.lock
    ├── src/alken_metamodel/            25 modules — the modelling pipeline
    ├── tests/                          25 test files (~215 unit tests)
    ├── experiments/                    10 EX.* / S* probe scripts
    ├── docs/
    │   ├── methodology.md              631‑line methodology doc
    │   └── plans/
    │       ├── 2026-05-30-metamodel-build.md      (pass 1)
    │       ├── 2026-05-30-pass2.md
    │       ├── 2026-05-30-pass3.md
    │       ├── 2026-05-30-pass4.md
    │       └── 2026-05-30-pass5.md
    └── reports/
        ├── T3_03_Alken_Metamodel_Report.md         Submission‑ready Harvard ref (~220 lines, 42 refs)
        ├── CW_Breakdown.md                         The internal coursework breakdown
        ├── STML_ActionItem_Tracker.md              Chronological PM‑1..PM‑12 log of every decision
        └── research/
            ├── LR-1.md … LR-9.md                   9 literature‑review notes
            └── nlr-cw-v1.md                        Master literature review (60 refs)
```

Build the feature matrix (top level): `uv run python -m stml.metamodel.build_features`.

Run the modelling pipeline (nested): `uv run --directory metamodel-apb python -m alken_metamodel.emit --asset-classes equity energy metals`.

The hard rule (`metamodel-apb/CLAUDE.md`): **never consume `results/feature_matrix.parquet`** — the matrix freezes fitted stats at one global `fe_train_end` and would leak into in‑sample folds before that date. The modelling pipeline imports the stml feature *functions* and recomputes them inside each CV fold on the fold‑train slice; a guard test `test_no_metamodel_module_reads_frozen_parquet` enforces this.

### 5.2 The shared feature library (`stml.metamodel`)

A consolidation of every teammate's feature work into **175 columns across 17 families** (F1‑F17, no F14), tidy‑long keyed by `(date, instrument)` over **4984 non‑zero‑signal trade days**. The matrix carries `partition` (train/val/test) and `fe_train_end_date` provenance columns and **no label column** — downstream branches attach their own labels.

**Two leakage classes:**
- **E (engineered)** — no fit; proven causal by **right‑edge truncation‑invariance** (the value at `t` is identical on `data[:t+1]` and `data[:T]`).
- **TF (fitted)** — F3, F4, F11 macro z‑scorer, F16, F17 — fit on the FE‑train partition only (`< 2021-07-01`) and applied causally with frozen parameters.

**Standardisation:** every scale‑dependent E‑class column ships a parallel `z_<col>` twin via `add_z_twins` — a per‑instrument **causal expanding‑window z‑score** (`expanding(min_periods=60)`) that bakes in no train/test cutoff. Bounded / already‑normalised columns (ratios, probabilities, t‑statistics, sin/cos, Hurst, wavelet energies) get no twin. There are 24 z‑twin columns in the produced matrix.

**Per‑family catalogue (representative columns):**

| Family | Provenance | Representative columns |
|---|---|---|
| **F1** counter‑trend | signal‑deep‑dive | `f1_mr_score_{10,20,40}`, `f1_dist_ma_sigma_{10,20,40}`, `f1_ret_reversal_{10,20,40}`, `f1_hilo_pos_{10,20,40}`, `f1_rsi_14`, `f1_bb_pctb_20`, `f1_bb_bandwidth_20` |
| **F2** vol / dispersion | signal‑deep‑dive + Sreeram | `f2_vol_{10,20,60}` (annualised), `f2_vol_ratio_20_60`, `f2_vol_pctile_20`, `f2_vol_of_vol_20`, `f2_parkinson_20`, `f2_garman_klass_20`, `f2_atr_14`, `f2_ret_skew_60`, `f2_ret_kurt_60`, `f2_rogers_satchell_20` |
| **F3** regime posteriors (TF) | signal‑deep‑dive | `f3_gmm_prob_highvol`, `f3_markov_prob_highvol`, `f3_markov_switch_prob`, `f3_regime_dwell` |
| **F4** latent (TF) | signal‑deep‑dive | `f4_pc1..f4_pc4`, `f4_cluster_id`, `f4_cluster_dist`, `f4_ae_code1..f4_ae_code4`, `f4_ae_recon_err` |
| **F5** signal‑derived | signal‑deep‑dive + Harry | `f5_signal`, `f5_abs_signal`, `f5_trailing_run_length`, `f5_days_since_flip`, `f5_days_since_nonzero`, `f5_participation_{20,60}`, `f5_long_bias_20`, `f5_sign_agree_mr`, `f5_signal_entropy_20`, `f5_flip_rate_60` |
| **F6** momentum | signal‑deep‑dive | `f6_ts_momentum_{20,60}`, `f6_ma_cross_20_60`, `f6_macd_12_26`, `f6_macd_hist_12_26_9`, `f6_adx_14`, `f6_donchian_pos_20` |
| **F7** microstructure | signal‑deep‑dive + Harry | `f7_volume_z_20`, `f7_volume_trend_20`, `f7_oi_{level,change,z_20,price_div_20}`, `f7_amihud_20`, `f7_rolls_spread_20`, `f7_kyles_lambda_20`, `f7_overnight_gap` |
| **F8** calendar | signal‑deep‑dive | `f8_dow_sin/cos`, `f8_month_sin/cos` |
| **F9** cross‑section / cross‑asset | signal‑deep‑dive + Harry | `f9_xsect_rank`, `f9_xsection_universe_size`, `f9_pair_corr_mean`, `f9_dist_lead_lag_centroid`, `f9_asset_class_dispersion_z`, `f9_ewma_implied_corr_z` |
| **F10** price action | signal‑deep‑dive | `f10_hl_range`, `f10_hl_range_mean_20`, `f10_oto_ret`, `f10_oto_ret_mean_20` |
| **F11** macro (TF) | signal‑deep‑dive | 45 cols across 12 series × {level, chg5, chg20} + 3 spreads — `f11_vix_*`, `f11_move_*`, `f11_dxy_*`, `f11_10y_ust_*`, `f11_2y_ust_*`, `f11_hy_oas_*`, `f11_be10y_*`, `f11_tips10y_*`, `f11_eia_crude_stock_*`, `f11_eia_ng_stock_*`, `f11_us_ism_mfg_pmi_*`, `f11_china_pmi_mfg_*`, plus `f11_spread_{vix_term, curve_slope, credit_diff}_*` |
| **F12** path structure | Sreeram | `f12_autocorr_21`, `f12_efficiency_ratio_21`, `f12_variance_ratio_5_21`, `f12_trend_tval_{10,21,42}`, `f12_hurst_100`, `f12_ma21_slope` |
| **F13** wavelet | Harry | `f13_mra_energy_{d1..d5}` (multi‑resolution analysis energy at five detail bands) |
| **F15** conditional risk | Harry | `f15_expected_hit_time`, `f15_prob_timeout`, `f15_path_tortuosity_20`, `f15_realized_semi_vol_ratio_20` |
| **F16** concept‑drift (TF) | Harry | `f16_regime_alignment_score` (rolling logistic discriminator: `P(today looks recent vs FE‑train era)`) |
| **F17** HMM (TF) | Sreeram | `f17_hmm_state_lo`, `f17_hmm_state_mid`, `f17_hmm_state_hi`, `f17_hmm_state_argmax` (3‑state Gaussian HMM, causal forward filter) |
| **z‑twins** | Sreeram conv. | 24 `z_*` columns: `z_f2_vol_{10,20,60}`, `z_f2_vol_of_vol_20`, `z_f2_{parkinson,garman_klass,atr_14,rogers_satchell}_20`, `z_f6_{macd_12_26, macd_hist_12_26_9}`, `z_f7_{amihud, oi_level, oi_change, kyles_lambda, rolls_spread, overnight_gap}`, `z_f10_{hl_range, hl_range_mean_20, oto_ret, oto_ret_mean_20}`, `z_f12_ma21_slope`, `z_f15_{expected_hit_time, path_tortuosity_20, realized_semi_vol_ratio_20}` |

**Module map (in `src/stml/metamodel/`):**

| Module | Role |
|---|---|
| `features.py` | core E‑class assembly (F1, F2, F5, F6, F7, F8, F10) |
| `features_ext.py` | extended E‑class (F2‑RS, F5‑adds, F7‑adds, F12, F13, F15), plus `expanding_zscore`, `Z_TWIN_COLUMNS`, `add_z_twins` |
| `regime_features.py` | F3 GMM + Markov‑switching, `fit_regime` / `transform_regime` |
| `regime_features_hmm.py` | F17 3‑state Gaussian HMM, `fit_hmm` / `transform_hmm` (requires `hmmlearn`) |
| `drift_features.py` | F16 — `regime_alignment_score` rolling logistic discriminator |
| `latent.py` | F4 — `StandardScaler + PCA(4) + KMeans + deterministic shallow autoencoder` on the class‑pooled FE‑train block |
| `xsection.py` | F9 — per‑day rank, universe size, mean rolling pair‑correlation to peers, lead‑lag centroid distance, within‑class dispersion z, EWMA implied‑correlation z |
| `macro_features.py` | F11 — ingests `data/additional_data.xlsx`, applies per‑class **point‑in‑time publication lags**, derives 12 series + 3 spreads (45 cols), FE‑train‑freezes a z‑score |
| `scope.py` | per‑instrument `InstrumentScope` registry: fitting scope, `n_eff_gate`, low_power flag, `embargo_p90` |
| `splits.py` | chronological train/val/test split + embargo + `n_eff` helpers (vendored from the removed replication layer) |
| `catalog.py` | `FeatureSpec` registry — documents every produced column (incl. z‑twins), renders `reports/feature-catalog.md`, asserts exact 1:1 column coverage |
| `pipeline.py` | `FeaturePipeline.fit(...).transform(...)` orchestrates every family into one tidy‑long matrix |
| `build_features.py` | CLI: loads data, runs the pipeline over all 11 instruments, persists matrix + per‑family CSVs + redundancy + scope + provenance + catalog |

**Canonical chronological split** (the `partition` column, never shuffled):

| split | date range | count | role |
|---|---|---:|---|
| `train` | 2020‑01‑03 → 2021‑07‑01 | 387 dates / 2925 rows | TF‑family fit window |
| `val` | 2021‑07‑02 → 2021‑12‑30 | 129 dates / 1041 rows | downstream validation |
| `test` | 2021‑12‑31 → 2022‑06‑30 | 129 dates / 1018 rows | downstream final confirmation |

OHLCV from before 2020 is kept only for feature warm‑up; matrix rows are the non‑zero‑signal trade days from 2020 onward.

**Per‑instrument scope** (`results/instrument_scope.json`) — the critical metadata the nested subproject consumes for the per‑instrument embargo:

| inst | class | `n_eff_gate` | `low_power` | **`embargo_p90`** (trading days) |
|---|---|---:|---|---:|
| `es1s` | EQ | 35 | false | 10 |
| `nq1s` | EQ | 20 | false | 8 |
| `fesx1s` | EQ | 25 | false | 9 |
| `cl1s` | EN | 9 | **true** | 14 |
| `ho1s` | EN | 9 | **true** | **26** |
| `rb1s` | EN | 13 | false | 19 |
| `ng1s` | EN | **2** | **true** | **33** |
| `gc1s` | ME | 11 | false | 12 |
| `si1s` | ME | 19 | false | 7 |
| `hg1s` | ME | 29 | false | 11 |
| `pl1s` | ME | 26 | false | 8 |

`ng1s` at `n_eff_gate=2` (only two effective post‑embargo signal runs) explains why its EX.5 information coefficient is undefined.

### 5.3 The nested subproject (`metamodel-apb/`) — Stage 0 scaffold

Commit `2091cb1 feat(metamodel-apb): scaffold nested uv subproject (Stage 0)` set this up as a separate uv project that imports stml as an editable path dependency. `CLAUDE.md` codifies the non‑negotiable rules:

1. **Leakage / fold‑safety.** Reuse stml's *causal* feature **functions** (`stml.metamodel.features.assemble_engineered`, `f2_vol_dispersion`, `regime.fit_*` / `regime.transform_*`) **recomputed inside each CV fold** on the fold‑train slice; fitted regime/HMM/latent blocks are re‑fit on fold‑train only. **Never consume `../results/feature_matrix.parquet`.**
2. **Triple‑barrier labels overlap** → standard k‑fold is invalid. Track `t1` (first‑touch) per label; **purge + embargo on `t1` everywhere**, including cluster MDA.
3. **Lock the feature set before final OOS.** No data‑snooping on Jan–Jun 2022 (it rehearses the hidden Jul–Dec 2022 half).
4. **Determinism.** Call `alken_metamodel.seeding.set_seeds()` at every entry point. Scalers/estimators fit on **train only**; every lag/rolling feature **shifted**. `emit` sorts rows, pins column order, fixes float format → byte‑identical re‑emit. **The prediction window is config‑driven, never hardcoded to Jan–Jun 2022** (the grader swaps in the hidden half).
5. **Data is read‑only.** Released data lives in `../data/` (loaded via `stml.io.load_clean_data()`); never edit it.
6. **Branch discipline.** Never commit to `main` (stml convention); work on a model branch, open a draft PR.

Subproject layout (25 modules):

```
metamodel-apb/src/alken_metamodel/
├── __init__.py
├── _env.py                      single-thread native kernels (fixes macOS libomp segfault)
├── _vendor/
│   ├── cluster_feature_importance.py   verbatim sts-ml + the four §4 bug fixes
│   ├── regression_metrics.py           verbatim sts-ml
│   ├── trend_scanning.py               verbatim sts-ml (tValLinR)
│   └── vsn.py                          verbatim PS6 FinalModel
├── seeding.py                   sync random / numpy / torch / tensorflow / PYTHONHASHSEED
├── volatility.py                Stage 1 — GK / Parkinson / RS estimators
├── triple_barrier.py            Stage 1 — meta-labels + t1 + uniqueness
├── cross_validation.py          Stage 1 — PurgedKFold / CPCV / nested CPCV
├── sizing.py                    Stage 1 — fractional Kelly + vol target + κ helpers
├── macro.py                     Stage 1.7 — PIT-lagged macro block
├── features.py                  Stage 2 — per-instrument feature adapter
├── regime.py                    Stage 2 — online EWMA HMM + stml static blocks
├── models.py                    Stage 2 — logistic / XGB / LightGBM
├── neural.py                    Stage 2 — TorchMLP / TorchVSN / KerasVSN / ReducedEstimator
├── evaluation.py                Stage 2 — sample-weighted purged OOS harness
├── calibration.py               Stage 2 — reliability / ECE / Platt / Isotonic
├── dim_reduction.py             Stage 2 — ClusterRepSelector / PCAReducer / AutoencoderReducer
├── cluster_importance.py        Stage 3 — Mantegna + cluster MDI + purged MDA + SHAP
├── pipeline.py                  Stage 5 — run_asset_class + PipelineConfig + nested_cpcv harness
├── emit.py                      Stage 5 — deterministic CSV writer + CLI entry point
├── backtest.py                  Stage 6 — barrier-exact + cost-aware + performance metrics
├── cost_model.py                Stage 6 — Grinold-Kahn cost model
├── significance.py              Stage 6 — t-stat / studentised bootstrap / Lo CI / MinTRL
├── deflation.py                 Stage 6 — DSR ladder + CSCV-PBO + MinBTL
├── signal_analysis.py           Stage 5/EX.5 — PT / TM / H-M proxy / IC / IR
└── experiment_log.py            deterministic per-class one-row CSV log
```

### 5.4 Stage 1 — Volatility (`volatility.py`)

Three OHLC range‑based volatility estimators (Garman & Klass 1980; Parkinson 1980; Rogers & Satchell 1991). All annualisable, all clamped to non‑negative variance.

```
Parkinson:        σ² = (ln(H/L))² / (4·ln 2)
Garman–Klass:     σ² = 0.5 · (ln(H/L))²  −  (2·ln 2 − 1) · (ln(C/O))²
Rogers–Satchell:  σ² = ln(H/C)·ln(H/O)  +  ln(L/C)·ln(L/O)
```

**Garman–Klass is preferred** for energy and rates futures because the open‑close term captures overnight gaps from scheduled releases (EIA reports, auctions). Parkinson under‑estimates vol on gap days; Rogers–Satchell is drift‑independent and corroborates GK. These wrap the same closed forms used in stml's `f2_vol_dispersion` but expose them as standalone unit‑tested σ̂ sources so the labelling module is self‑contained.

`get_daily_vol(close, span=100, min_periods=20)` style not present here — the triple‑barrier σ̂ instead reads the de‑annualised `f2_vol_20 / √252` produced by the feature stack. Cite Korkusuz, Kambouroudis & McMillan 2023 (nlr‑cw §A1).

### 5.5 Stage 1 — Triple‑barrier labels (`triple_barrier.py`)

Standard LdP AFML Ch.3‑4. Public surface:

```python
add_vertical_barrier(close, t_events, max_holding)        → vertical-barrier expiry per event
apply_pt_sl_on_t1(close, events, pt_sl)                   → first PT/SL touch times per event
get_events(close, t_events, pt_sl, trgt, side, t1)        → assemble + resolve to t1 = first touch
get_bins(events, close)                                   → realised ret + label {0, 1}
get_num_co_events(close_index, t1)                        → concurrency on the trading-day index
average_uniqueness(t1, num_co_events)                     → AFML Ch.4 uniqueness weights
triple_barrier_labels(close, signal, target, pt_sl, …)    → events + labels + weights, one row per event
```

**Conventions:**

- **Vol‑adaptive symmetric barriers** `±k·σ̂ₜ` where `σ̂ₜ` is the de‑annualised GK daily vol (`f2_vol_20 / √252`). Fixed‑% thresholds ignore heteroskedasticity and make the label distribution pro‑cyclical.
- **Vertical barrier** `T_max` (default `max_holding = 10` bars).
- **Symmetric default** `pt_sl = (1.0, 1.0)`.
- **Timeout labelled by sign of expiry P&L** — `bin = (ret > 0).astype(float)` — the canonical meta‑labelling convention.
- **Side‑adjusted return** `ret = (close[t1] / close[entry] − 1) · side`.
- **Average uniqueness** = mean of `1 / concurrency` over the label's span (AFML Ch.4), down‑weights overlapping labels. Verified exactly on disjoint (→ 1.0) and fully‑overlapping (→ 0.5) toy cases in `tests/test_triple_barrier.py`.

Entry is at the close of the signal day (single‑day entry, no t+1 convention like Harry's). The output frame has columns `side, t1, ret, bin, weight`, one row per non‑zero signal day.

### 5.6 Stage 1 — Cross‑validation (`cross_validation.py`)

Three splitters plus the **per‑instrument embargo** machinery (S2.6, pass 4 addition).

**`embargo_size(n, pct=0.01)`** — `⌈pct·n⌉` bars; the published 1% AFML default, unchanged when no per‑instrument map is provided.

**`PurgedKFold(n_splits, t1, pct_embargo, instruments=None, embargo_days=None)`** — sklearn‑compatible. Test folds are contiguous chunks of `range(len(X))`; the `_purge_train` helper drops training events whose `[t, t1]` span intersects any test block, plus a forward embargo.

**`CombinatorialPurgedCV(n_groups, n_test_groups, t1, pct_embargo, …)`** — N groups, k test groups → `C(N, k)` purged splits. Defaults `N=6, k=2` → **15 paths**. Used as the *selection* splitter in the shipped default config (not just a diagnostic).

**`nested_cpcv(X, t1, outer_groups=6, outer_test_groups=2, inner_groups=5, inner_test_groups=1, …)`** — yields `(outer_train, outer_test, inner_cv)`. Inner CPCV is built only from outer‑train rows so tuning never touches the outer test fold. Implemented and unit‑tested as the principled selection‑bias‑aware evaluator; a meaningful real‑data run is ~15×5×5 fits per class and is flagged as **deferred**.

**Per‑instrument embargo (S2.6, pass 4).** When `embargo_days` map + aligned `instruments` Series are supplied, `_purge_train` switches paths:

* The instrument‑agnostic **span‑overlap purge** still covers cross‑instrument label concurrency: drop train events whose `[t, t1]` intersects the test block's `[b_t0, b_t1]`.
* The forward **embargo** is applied **per instrument** — for each instrument `tk` in the test block, find its own max `t1`, then advance `embargo_days[tk]` trading days on that instrument's *own* date axis (computed by `_instrument_date_axes`), and drop train events of the same instrument starting within that window.

That is the correct treatment for a pooled panel where a flat `⌈1%·T⌉` embargo would under‑cover the most persistent instruments (ng1s 33d, ho1s 26d). A known‑answer test asserts each instrument's purged forward window equals its `embargo_p90` and that zero train/val `t1` overlap survives.

### 5.7 Stage 1 — Sizing (`sizing.py`)

Position sizing module supporting fractional Kelly, vol targeting, the smooth‑taper sizing variant (S6.15), and the Baker‑McHale per‑instrument shrinkage (EX.6). Module‑level constants:

```
KAPPA              = 0.25   fractional-Kelly multiplier (MZB 1992)
CONFIDENCE_FLOOR   = 0.55   minimum p̂ for a non-zero bet
TAPER_WIDTH        = 0.05   half-width of the smooth ramp around the floor
TARGET_VOL         = 0.25   25% annualised vol target (Carver 2015)
MAX_LEVERAGE       = 5.0
```

Public functions:

| Function | Role |
|---|---|
| `confidence_taper(p, floor, width)` | C¹‑continuous smoothstep replacing the hard floor — 0 below `floor − width`, 1 above `floor + width`, exactly 0.5 at the floor. The growth‑optimal calibrated shape from LR‑7. |
| `kappa_baker_mchale(edge, resid_var)` | Per‑instrument shrinkage `κᵢ = eᵢ² / (eᵢ² + σᵢ²)` — shrinks size hardest where p̂ is least reliable. Bounded in [0, 1]. |
| `cer_improves(candidate, baseline, min_gain)` | The S6.15 CER gate — adopt a sizing change iff it strictly raises OOS certainty‑equivalent by more than `min_gain`. |
| `kelly_fraction(p, b, d)` | Asymmetric‑payoff Kelly `f* = (p·b − (1−p)·d) / (b·d)`. |
| `fractional_kelly(p, b, d, kappa, floor, cap, taper_width)` | `κ · f*` gated by the confidence floor and clipped to `[0, cap]`. Hard floor when `taper_width is None`; smooth taper otherwise. |
| `vol_target_leverage(realised_vol, target_vol, max_leverage)` | `target_vol / realised_vol`, clipped to `[0, max_leverage]`. |
| `position_weight(side, p, b, d, realised_vol, …)` | Signed position weight = `side · fractional_kelly · vol_target_leverage`. Positive = long, negative = short, zero when p̂ < floor. |

The shipped deliverable uses **flat `κ = 0.25`** with the **hard floor** — both the smooth taper and the Baker‑McHale κᵢ refinements were tested in pass 5 and **rejected** by the EX.6 paired bootstrap gate (see §5.20).

### 5.8 Stage 1.7 — PIT‑lagged macro block (`macro.py`)

Loads `data/additional_data.xlsx` — **22 macro series** carrying observation dates only — and turns them into a publication‑lag‑safe feature block for the modelling pipeline (separate from F11 in the shared library).

The series and their conservative **publication lags** (calendar days):

| Series | Lag | Reason |
|---|---:|---|
| `VIX`, `VIX3M`, `MOVE`, `DXY`, `CBOE_SKEW`, `2Y_UST`, `10Y_UST`, `10Y_BUND`, `EURUSD`, `BAL_DRY_INDEX`, `HY_OAS`, `IG_OAS`, `TIPS10Y`, `BE10Y`, `LME_COPPER_STOCK` | **+1** | one‑day implementation lag (close known EOD, traded next day) |
| `EIA_CRUDE_STOCK`, `EIA_DIST_STOCK`, `EIA_GASOLINE_STOCK`, `EIA_NG_STOCK` | **+5** | weekly EIA Petroleum/Gas Status Report ~5 days after the reference week |
| `US_ISM_MFG_PMI`, `CHINA_PMI_MFG`, `GERMANY_PMI_MFG` | **+30** | monthly PMI published the following month |

`pit_align(series, trade_dates, lag_days)` shifts the observation index forward by `lag_days` (observation → availability), then forward‑fills onto the trade calendar. Trade date `t` sees only the most recent value released on or before `t` — **point‑in‑time correct and truncation‑invariant**. A `pit_align` release‑deferral test and block‑level truncation‑invariance test pin the behaviour.

`macro_features(trade_dates)` assembles 16 derived theory‑of‑storage / cross‑asset features:

| Group | Features |
|---|---|
| Vol / term | `macro_vix_level`, `macro_vix_term_slope` (= VIX3M − VIX), `macro_move` |
| FX | `macro_dxy_chg20` (20‑day log change) |
| Credit | `macro_hy_oas`, `macro_ig_oas`, `macro_credit_slope` (= HY − IG) |
| Real rates | `macro_real_rate` (TIPS10Y), `macro_breakeven` (BE10Y) |
| EIA inventories | `macro_eia_{crude, dist, gasoline, ng}_chg` (20‑day log change of the lag‑aligned inventory series) |
| Growth | `macro_china_pmi`, `macro_us_pmi` (deviation from the 50 expansion line), `macro_copper_stock_chg` |

**Flagged as NOT derivable** in the methodology: per‑instrument calendar/basis spreads — OHLCV ships only the front‑month `*1s` contract (no second maturity), so this is correctly omitted rather than fabricated.

### 5.9 Stage 2 — Per‑instrument feature assembly (`features.py`)

The adapter that wires the stml E‑class feature **functions** (proven causal by right‑edge truncation invariance) into the meta‑model, plus a **backward trend‑scanning feature**.

The fold‑safety contract (the crux of the leakage discipline):

> Every `assemble_*` is **stateless**, so the leakage concern is purely the input window: compute on each instrument's **full fixed‑start history** and then **right‑slice the OUTPUT** to the fold dates (`right_slice`). Never left‑truncate the input — it would break (a) rolling warm‑up, (b) the start‑sensitive expanding z‑twins, and (c) `f15`'s positional‑seed bootstrap (whose truncation‑invariance is right‑edge only). Enforced as a property test: `feature[t]` is identical whether computed on `data[:t+1]` or `data[:T]`.

Public surface:

```python
_segment_tval(y)                                     OLS slope t-stat, closed-form (~100x faster than statsmodels)
backward_trend_feature(close, span=(5, 25), cap=20)  the trend-scanning feature
assemble_instrument_features(ohlcv_inst, signal,
                              trend_span, drift_train_end, drift_seed)
right_slice(features, end)                           enforce fold boundary on OUTPUT only
filter_signal_days(features, signal_inst)            keep only non-zero signal trade days
attach_instrument(features, instrument)              re-attach the instrument key
daily_barrier_sigma(features)                        de-annualise f2_vol_20 → daily σ̂ for triple barrier
```

The **backward trend feature** (correctness note from methodology §1):

> Trend scanning is mandated as a *feature*, never the label. We reuse the trend‑scanning algorithm with `look_forward=False`, but compute the OLS slope t‑statistic in closed form (`_segment_tval`, validated equal to the vendored `tValLinR` to 1e‑9) and apply a **deterministic ±20 cap** instead of `trend_labels`' global‑variance cap — that global cap depends on the whole series and is itself a right‑edge truncation leak.

Output columns: `trend_tval_back`, `trend_sign_back`, `trend_window_back`.

When `drift_train_end` is set, the F16 concept‑drift column (`f16_regime_alignment_score`) is appended — stml's causal covariate‑shift discriminator that scores how "recent" each row looks vs the pre‑`drift_train_end` FE‑train era. Causal rolling refit (per‑refit seed keyed on positional refit index → right‑edge truncation‑invariant). `None` (default) omits it.

### 5.10 Stage 2 — Online EWMA HMM regimes (`regime.py`)

Two regime blocks per instrument, both causal:

**(1) Online EWMA 2‑state Gaussian HMM** (net‑new, the literature‑faithful answer to commitment #8). A forward‑filtered HMM on daily log returns whose emission means/variances are re‑estimated *online* via a forgetting factor `lam` (EWMA of responsibility‑weighted sufficient statistics), with a fixed persistent transition prior. Because every parameter at `t` is a recursion over observations `≤ t`, **the feature is causal / right‑edge truncation‑invariant by construction** — no batch fit, no fit/transform split, hence **no per‑fold CPCV seam artefact** (concatenating non‑contiguous CPCV train groups would fabricate fake 1‑step transitions in a batch HMM). The transition matrix is a fixed *persistent* prior (the "penalising jumps" half of Nystrup–Madsen–Lindström 2017); the means/variances are the time‑varying part. Cite Hamilton 1989; Nystrup, Madsen & Lindström 2017; Ang & Bekaert 2002.

Hyperparameters (nlr‑cw §4 defaults, trivially overridable):

```
EWMA_LAMBDA    = 0.94    forgetting factor for the online emission update
PERSISTENCE    = 0.97    fixed self-transition prior (persistent regimes)
HMM_WARMUP     = 60      bars used to seed the emissions; emitted as NaN
VAR_FLOOR      = 1e-12   variance floor for the Gaussian density
HI_INIT_SCALE  = 2.5     high-vol state initial σ = HI_INIT_SCALE · warm-up σ
```

Emitted columns:

| Column | Definition |
|---|---|
| `ewma_hmm_prob_highvol` | filtered `P(high‑vol state)` at each bar |
| `ewma_hmm_state` | hard 0/1 state (probability ≥ 0.5) |
| `ewma_hmm_var_hi` | time‑varying high‑state variance |
| `ewma_hmm_var_lo` | time‑varying low‑state variance |
| `ewma_hmm_switch_prob` | one‑step change in the high‑vol probability |

**(2) stml static blocks** (supplementary). `static_regime_features(ohlcv_inst, fit_end, seed)` fits stml's F3 GMM + Markov‑switching (`fit_regime` / `transform_regime`) and F17 3‑state Gaussian HMM (`fit_hmm` / `transform_hmm`) on a **contiguous prefix** `(ret, vol).index <= fit_end`, then causally transforms on the full history. Contiguity avoids the seam hazard above. Returns the stml `f3_*` and `f17_*` columns.

`assemble_regime_features(ohlcv_inst, fit_end, seed, lam)` is the master that runs both blocks and concatenates on the price calendar.

### 5.11 Stage 2 — Model roster (`models.py`)

The tree/linear core of the §3 horse‑race behind one uniform `MetaClassifier` interface: `fit(X, y, sample_weight) / predict_act_proba(X)`.

**Design choices (against PS4/PS5/PS6 dossier):**

- **One weighting channel.** PS4/5/6 ship no class‑ or sample‑weighting at all. Meta‑labels are both *overlapping* (need LdP Ch.4 uniqueness weights) and *imbalanced* (~30‑40% positive). Both are folded into a single `sample_weight` passed identically to every estimator's `fit` **and** into every OOS metric (`balanced_sample_weight(y, base=uniqueness)` composes uniqueness × inverse‑class‑frequency, rescales each class to `n / n_classes`).
- **XGBoost** uses the PS5 cell‑43 config (binary:logistic). **LightGBM** mirrors that regularised configuration.
- **Determinism.** Single‑threaded tree building, `deterministic=True, force_row_wise=True` on LightGBM, `n_jobs=1` everywhere → byte‑stable re‑fit.
- **NaN policy per‑model.** Trees consume NaN natively; the logistic path imputes with a train‑fitted median (`SimpleImputer(strategy="median")`) and then standardises.

Three constructors:

| Constructor | Estimator |
|---|---|
| `make_elasticnet_logistic(seed=42, l1_ratio=0.5, C=1.0, max_iter=5000)` | `LogisticRegression(solver="saga", l1_ratio=0.5, C=1.0, ...)` — sklearn ≥1.8 drives elastic‑net via `l1_ratio` |
| `make_xgb(seed=42)` | `XGBClassifier(n_estimators=100, max_depth=4, lr=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, n_jobs=1)` |
| `make_lightgbm(seed=42)` | `LGBMClassifier(n_estimators=200, num_leaves=15, lr=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, min_child_samples=5, deterministic=True, force_row_wise=True)` |
| `tree_linear_roster(seed=42)` | dict of all three, keyed by name |

The shipped pipeline assembles its roster from `tree_linear_roster` for the tree/linear slot plus the torch neural variants from `neural.py` for the NN slot.

### 5.12 Stage 2 — Neural variants (`neural.py`)

Three neural classifiers (lazy imports keep torch/tensorflow out of the module load path — this is what caused the earlier libomp segfault):

**`TorchMLP`** — small torch MLP, BCEWithLogitsLoss, **full‑batch Adam** (no minibatch shuffle → deterministic), median‑impute + standardise inputs.

```
hidden = (64, 32), epochs = 150, lr = 1e-3, dropout = 0.1, weight_decay = 1e-4
```

**`TorchVSN`** — Variable Selection Network ported to torch (Lim et al. 2021). One Linear projector + one Gated Residual Network *per feature*, softmax variable‑selection weights for interpretability, weighted BCE, full‑batch.

```
embedding_dim = 16, hidden_dim = 8, output_dim = 14, epochs = 150, lr = 1e-3
```

Build pattern:

```
for each feature i:
    embed_i = Linear(1 → emb)
    grn_i   = GRN(emb → hidden → out_dim)
flatten → GRN → Linear → softmax → variable selection weights
weighted sum across features → head Linear
```

The VSN's one‑GRN‑per‑feature cost is intractable on the full ~111 pooled features, which is why the shipped path runs it on **cluster‑representative‑reduced** features (one medoid per Mantegna cluster — see §5.15).

**`KerasVSN`** — vendored PS6 `FinalModel` reused via TensorFlow GradientTape with per‑sample binary cross‑entropy. **Off‑path:** TF op‑determinism is best‑effort, so this variant is kept as a documented comparison only, not selectable by the grader's rerun (the grader's hidden‑half re‑run could otherwise pick a non‑reproducible model).

**`ReducedEstimator(estimator, reducer)`** wraps a reducer + estimator together so cluster‑representative selection is fitted on each fold's **train rows only** (fold‑safe). The reduced basis never sees the validation fold.

**Rosters:**

| Function | Estimators |
|---|---|
| `neural_roster(seed)` | torch_mlp, torch_vsn, keras_vsn |
| `default_roster(seed, reduce=True, max_clusters=12)` | tree/linear (full features) + torch_mlp + torch_vsn on cluster‑reduced features. **The shipped deliverable roster.** |
| `full_roster(seed)` | tree/linear + neural (incl. off‑path KerasVSN) |

### 5.13 Stage 2 — Evaluation harness (`evaluation.py`)

Adapts PS5's `evaluate_model` (cell 72) — a fixed‑0.5‑threshold dict of accuracy/precision/recall/F1/AUC — into the meta‑model's needs.

- **Sample‑weighted** metrics. Triple‑barrier labels overlap, so every score is weighted by the LdP uniqueness weight (passed identically to `fit` and to the metric functions) — otherwise concurrent labels double‑count in the OOS estimate.
- **Calibration** metrics (Brier, log‑loss, average‑precision) alongside ranking AUC: downstream Kelly sizing consumes the probability itself.
- **Tunable decision threshold** (PS5 was 0.5‑only).
- **Degenerate‑fold safety.** Under purging a fold can be single‑class; ranking metrics then return NaN (documented) rather than crashing.

Public functions:

| Function | Returns |
|---|---|
| `evaluate_predictions(y_true, proba, sample_weight, threshold)` | dict: `n, threshold, accuracy, precision, recall, f1, brier, auc, avg_precision, log_loss` |
| `evaluate_oos(model, x_test, y_test, …)` | same dict on a held‑out OOS slice |
| `always_act_baseline(y_true, …)` | the blind‑primary baseline (constant `P(act) = 1`) — AUC = NaN by definition |
| `threshold_sweep(y_true, proba, thresholds)` | precision/recall/F1/accuracy across decision thresholds (§5 sweep) |
| `oos_predictions(make_model, X, y, cv, sample_weight)` | purged‑CV OOS `P(act)` for every row (NaN where never in a test fold) |
| `cross_val_evaluate(make_model, X, y, cv, sample_weight, threshold)` | one row per fold; refits fresh per purged fold and scores the held‑out OOS slice |

### 5.14 Stage 2/EX.4 — Calibration (`calibration.py`)

Probability calibration tools — measure ECE on a binned reliability curve, then apply Platt scaling or isotonic regression.

| Class / function | Role |
|---|---|
| `reliability_curve(y_true, proba, n_bins=10)` | per‑bin mean predicted prob vs realised positive frequency |
| `expected_calibration_error(y_true, proba, n_bins=10)` | weighted gap from the diagonal |
| `PlattCalibrator()` | 1‑D `LogisticRegression` mapping raw scores → calibrated probabilities |
| `IsotonicCalibrator()` | free‑form monotone score → probability map, clipped out of range |

Platt is monotone, so **AUC is unchanged** after calibration — the act/skip ranking is preserved. A unit‑tested invariant. Only Brier/ECE and the stake move.

### 5.15 Stage 2/EX.2 — Dimensionality reduction (`dim_reduction.py`)

The VSN builds one GRN per feature, so the 100+ pooled features are intractable at CV scale. EX.2 reduces three ways; **`ClusterRepSelector` is the promoted reducer (S3.7)**.

**`ClusterRepSelector(seed, max_clusters=10)`** — one feature per Mantegna cluster (the **medoid** — the feature with minimum total Spearman distance to its cluster‑mates). Deterministic, unsupervised, interpretable, reuses the §4 clustering verbatim. Drops zero‑variance columns before clustering (constant features would poison the distance matrix).

**`PCAReducer(var_threshold=0.95, n_components, seed)`** — variance‑threshold PCA on standardised, median‑imputed features. Dense, not interpretable.

**`AutoencoderReducer(latent_dim=8, epochs=200, lr=1e-3, seed)`** — small deterministic torch autoencoder (encoder + decoder, full‑batch). Non‑linear, over‑fit‑prone at this N. Built for the EX.2 comparison only; cluster‑rep is promoted.

Every reducer is **fit on train only** (basis/selection frozen at fit time, re‑applied to OOS rows), median‑imputes leading NaNs, and reduces strictly below the input dimension.

### 5.16 Stage 3/4 — Cluster importance (`cluster_importance.py`)

The §4 deliverable (10 marks). Mantegna distance → PCA → optimal‑K K‑means → cluster MDI + **purged cluster MDA** + cluster SHAP. Carries the **four required bug fixes** to PS2/sts‑ml/PS4, each visible in the diff:

| # | Bug | Fix | Where |
|---|---|---|---|
| 1 | `max_features='auto'` (PS4 grid; removed in sklearn ≥1.3) | `'sqrt'` | `cluster_forest()` for MDI/SHAP |
| 2 | `KFold(shuffle=True)` for MDA (leaks across overlapping labels) | injected `PurgedKFold` | vendored `calculate_cluster_importance_pfi` |
| 3 | no real SHAP in PS2/sts‑ml (MDI + PFI only) | `cluster_shap` via `shap.TreeExplainer`, summing member \|SHAP\| (the §4 contribution) | `cluster_importance.py` |
| 4 | Spearman distance `1 − |ρ|` (non‑metric) | **Mantegna** `√(1 − |ρ|)` (Mantegna 1999) | vendored `compute_spearman_distance_matrix` |

`cluster_feature_importance(X, y, t1, seed, n_splits, pct_embargo, max_clusters) → (table, clusters)` returns a per‑cluster table with `mdi, mda, shap` columns. `X` must be index‑aligned to `t1` so the purged MDA splits leak‑free.

The interpretive lesson from the methodology and the build‑doc:

> **MDI and SHAP are *in‑sample attribution*** — they always split 100% of a fitted model's importance across the clusters, so a high MDI/SHAP says only *which* features the model leaned on in‑sample, not that those features carry OOS edge. **Cluster permutation MDA is the OOS reality check, and it is near‑zero across the board** (every cluster `|MDA| < 0.02` bar one) — exactly what a ≈0.5‑edge problem should look like, and a useful negative result.

### 5.17 Stage 5 — End‑to‑end pipeline (`pipeline.py`)

`PipelineConfig` is a frozen dataclass carrying every knob. Defaults:

```
fe_train_end             = 2021-07-01
modelling_end            = 2021-12-31
predict_start            = 2022-01-01
predict_end              = 2022-06-30
pt_sl                    = (1.0, 1.0)
max_holding              = 10
n_splits                 = 5
pct_embargo              = 0.01            (uniform fallback embargo)
per_instrument_embargo   = False           (S2.6 — emit CLI sets True)
seed                     = 42
use_regime               = True
use_macro                = False           (emit CLI sets True)
use_drift                = False           (S1.8-b; emit CLI sets True for F16)
roster                   = "tree_linear"   (emit CLI sets "default")
cv_scheme                = "purged"        (emit CLI sets "cpcv")
cpcv_groups              = 6
cpcv_test_groups         = 2
```

`run_asset_class(ohlcv, signals, asset_class, config) → AssetClassResult` is the per‑class orchestrator. Asset‑class members (from `stml.metamodel.scope.ASSET_CLASS_MAP`):

```
equity  → es1s, nq1s, fesx1s
energy  → cl1s, ho1s, rb1s, ng1s
metals  → gc1s, si1s, hg1s, pl1s
```

The pipeline steps:

1. **`build_class_panel`** — per‑instrument modelling table joining causal features + EWMA HMM + static regimes + (optional) PIT macro + F16 + triple‑barrier labels on non‑zero‑signal trade days, pooled into one frame with instrument one‑hot dummies. Keeps the event‑date index (duplicated across instruments) so the purged CV can purge concurrent cross‑instrument labels by their `t1` spans.
2. **`select_model`** — horse‑race the roster by mean OOS AUC under `config.cv_scheme`. The shipped path uses **CPCV‑as‑selection** (15‑path mean AUC), with per‑instrument embargo when enabled.
3. **`fit_oos_calibrator`** — fit a Platt calibrator on purged‑OOS predictions of the **modelling sample only** (`dates <= modelling_end`, strictly `< predict_start` → cannot leak). Falls back to identity when single‑class or too thin.
4. **`per_instrument_diagnostics`** — per‑instrument purged‑OOS AUC/precision for the winning estimator, so a strong pooled number can't hide a weak member.
5. **Refit** the winner on the full modelling sample.
6. **Predict** `P(act)` on `[predict_start, predict_end]`. Calibrate via the Platt map; both raw and calibrated columns retained.
7. Return `AssetClassResult(asset_class, predictions, best_model, cv_scores, n_modelling, diagnostics, calibrator, oos_brier, oos_precision)`.

**`nested_cpcv_select_and_evaluate(X, y, t1, sample_weight, config, instruments)`** — the headline nested‑CPCV harness. Each outer fold runs the full inner horse‑race on the outer‑train rows ONLY (no tuning leakage — the inner splits are built from `t1.iloc[outer_train]`), refits the winner, scores it on the held‑out outer fold. The returned per‑outer‑fold distribution is the selection‑bias‑aware OOS estimate. **Implemented and unit‑tested; a meaningful real‑data run is flagged as deferred** (15 outer × 5 inner × 5 estimators ≈ 375 fits per class).

**`load_embargo_days()` (lru_cache=1)** — reads `results/instrument_scope.json` (located via stml's repo‑root resolver), extracts per‑instrument `embargo_p90` (trading days). Reading the file introduces no leakage; it's metadata, not features. Lets the CV purge a per‑instrument forward window rather than a flat 1%.

### 5.18 Stage 5 — Deterministic emit (`emit.py`)

CLI entry point and deterministic CSV writer. Two deliverables (brief, Deliverables):

- `metamodel_predictions.csv` — `(date, instrument, prediction)` with calibrated `P(act) ∈ [0, 1]`
- `strategy_weights.csv` — `(date, instrument, weight)` sized on the calibrated `p̂`

Determinism contract:

- Rows sorted by `(date, instrument)`
- Pinned column order: `["date", "instrument", "prediction"]` / `["date", "instrument", "weight"]`
- ISO date strings (`%Y-%m-%d`)
- `FLOAT_FORMAT = "%.10f"`
- `lineterminator="\n"`
- A re‑emit produces **byte‑identical** output. **Verified on the pass‑4 shipped path**: two independent full `emit` runs produced byte‑for‑byte identical `metamodel_predictions.csv`, `metamodel_predictions_raw.csv` and `strategy_weights.csv` (1011 rows each; canonical == calibrated).

Also emits:

- `metamodel_predictions_calibrated.csv` — byte‑identical alias of the calibrated deliverable
- `metamodel_predictions_raw.csv` — uncalibrated `p̂` retained for the §3 Brier/ECE story
- `coverage_caveat.csv` — per‑instrument `n_oos_rows`, `ic`, `ic_undefined`, `thin` flag (`thin = n_oos_rows < 60 OR ic_undefined`)
- `experiment_log.csv` — one row per asset class (deterministic, rebuilt per emit)

CLI flags (defaults):

```
--asset-classes equity energy metals
--predict-start 2022-01-01
--predict-end   2022-06-30
--outdir        outputs
--roster        default                    (tree/linear + reduced torch NNs)
--cv-scheme     cpcv                       (CPCV-as-selection, 15 paths)
--no-macro                                 (default off → macro ON)
--no-per-instrument-embargo                (default on → per-inst ON)
--no-drift                                 (default on → F16 ON)
```

The **prediction window is config‑driven**, never hardcoded — the grader swaps in the hidden Jul–Dec 2022 half by changing one CLI flag.

`strategy_weights(predictions, config)` computes the position weight per row by calling `position_weight(side, p, b, d, realised_vol)` from `sizing.py`. A non‑finite ann_vol yields a flat zero weight. `build_deliverables` ties it all together: runs each asset‑class metamodel, assembles raw + calibrated prediction frames, and sizes Kelly on the calibrated `p̂` (pass‑3 S6.11).

### 5.19 Stage 6 — Backtest (`backtest.py`, `cost_model.py`)

**`backtest.py`** carries both backtests in one module:

**Simple fixed‑horizon** (`backtest_strategy`, older variant kept for the S6.9 holding‑model comparison):

```python
positions = build_position_panel(weights, returns_panel, max_holding=10)   # ffill weight max_holding days
daily     = strategy_returns(positions, returns_panel)                     # ∑ᵢ posᵢ · forward_retᵢ
metrics   = performance_metrics(daily)
```

**Barrier‑exact + cost‑aware** (`barrier_backtest`, the §6 deliverable):

```python
positions = build_barrier_position_panel(meta, returns_panel)              # each label held over [date, t1), overlaps NETTED
gross     = strategy_returns(positions, returns_panel)
costs     = transaction_costs(positions, half_spread_bps=2.0, impact_bps=10.0, impact_exponent=1.0)
net       = gross - costs
```

The barrier‑exact mode exits on the **actual triple‑barrier first‑touch `t1`** — not a fixed `max_holding` — and **nets overlapping labels** on the same instrument (LdP convention). The Grinold–Kahn cost model (`cost_model.py`) charges a **half‑spread** (2 bps/side, conservative for liquid front‑month futures) plus a **market‑impact** term (10 bps · `|Δw|^exponent`, linear by default; impact `=2` gives the convex penalty).

**`performance_metrics(returns, ann=252)`** returns `n, total_return, cagr, ann_vol, sharpe, sortino, max_drawdown`. **Sortino convention** (pass‑5 fix):

> The downside deviation uses the **full‑T denominator with target 0** (Sortino & Price 1994) — positive returns contribute zero to the semivariance and the divisor is the *whole* sample length, not the count of negative observations. We deliberately avoid the common mis‑implementation that takes `std()` over the negative returns only (a smaller, biased divisor that inflates Sortino); a known‑value test (`tests/test_backtest.py`) pins the full‑T form.

**`certainty_equivalent(returns, risk_aversion=5.0)`** — mean‑variance CER `E[r] − ½·γ·Var[r]` (DeMiguel et al. 2009). Used by EX.6 and S6.15 as the utility‑aware sizing gate.

`barrier_backtest` returns a report dict combining standard performance metrics with **`turnover` (annualised one‑way notional)**, **`avg_holding_period` (mean busday span `[date, t1)`)**, the gross vs net split, the undiscounted arithmetic Σ‑cost (`total_cost`), and the **compounded cost drag** (`cost_drag_compounded = gross_total − net_total`) — S6.10 reconciliation: `net = gross − costs` holds daily, but the report's totals are compounded while `total_cost` is an arithmetic Σ, so `gross − total_cost ≠ net` by the compounding interaction.

### 5.20 Stage 6 — Significance and deflation (`significance.py`, `deflation.py`)

**`significance.py`** — **the PRIMARY §6 inference** (S6.14, LR‑6). Reports per‑period Sharpe statistics in *assumption‑strength order*:

| Statistic | Function |
|---|---|
| 1. **t‑stat = SR·√n** | `t_statistic(returns)` — no annualisation needed; an honest first read |
| 2. **Studentised stationary block‑bootstrap CI** (PRIMARY) | `stationary_bootstrap_sharpe_ci(returns, alpha=0.05, reps=2000, seed=42)` — block length from Politis‑White's `optimal_block_length` (data‑driven, deterministic); each replicate is studentised by the Lo (2002) analytic SE (`std_err_func`), so `arch` doesn't fall back to a nested bootstrap. Seeded → byte‑stable. |
| 3. Lo / Opdyke analytic band | `sharpe_ci_analytic(returns, alpha=0.05)` — `SR ± z_{1−α/2}·σ(SR̂)` (parametric cross‑check) |
| 4. MinTRL | `min_track_record_length(sr, sr_benchmark=0, skew, kurt, prob=0.95)` — `1 + (1 − skew·SR + ((kurt−1)/4)·SR²)·(z / (SR − SR*))²` |
| 5. Ljung–Box(10) | `ljung_box_test(returns, lags=10)` — the IID gate **before** any √252 annualisation. PSR / DSR / CSCV‑PBO live in `deflation.py` and are **demoted to corroboration**. |

**`stationary_bootstrap_cer_diff_ci`** — the EX.6 paired CER‑difference bootstrap. `r_alt` and `r_base` are the same strategy rescaled (e.g. flat‑κ vs per‑instrument κᵢ), so they are highly correlated: a single block‑index draw is applied to **both** series (independent resampling would inflate the band and bias the adopt/revert call toward "revert"). The functional is `CER(r) = E[r] − ½·λ·Var[r]`; each replicate is studentised by the delta‑method (influence‑function) SE of the paired difference, `ψ(r) = (r−μ) − ½·λ·[(r−μ)² − σ²]`. When the paired difference is deterministic the CI collapses to the point estimate.

**`deflation.py`** — backtest deflation for the §6 deployment gate (S6.8). A single reported Sharpe is selected from N configurations during model selection, so it overstates skill and ignores non‑normality.

| Function | Definition |
|---|---|
| `sharpe_ratio(returns, ddof=1)` | per‑period mean / std |
| `sharpe_std(sr, n, skew, kurt)` | Mertens 2002 / Lo 2002 SE under non‑normality: `Var = (1 − skew·SR + ((kurt−1)/4)·SR²) / (n−1)` |
| `expected_max_sharpe(n_trials, trials_std=1)` | `SR0 = trials_std · [(1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e))]` (Bailey & LdP 2014; `γ` = Euler‑Mascheroni) |
| `probabilistic_sharpe_ratio(sr, sr_benchmark, n, skew, kurt)` | `PSR(SR*) = Φ((SR − SR*) / σ(SR̂))` |
| `deflated_sharpe_ratio(returns, n_trials, trials_sharpe_std)` | PSR vs `SR0 = E[max of N trials]` |
| `min_backtest_length(n_trials, target_sharpe)` | `MinBTL = (E[max of N standard trials])² / target_Sharpe²` |
| `probability_of_backtest_overfitting(matrix, n_blocks=16)` | **CSCV PBO** (Bailey, Borwein, LdP, Zhu 2017). Splits T observations into `n_blocks` contiguous blocks; for each `C(S, S/2)` choice of IS blocks the IS‑best trial's relative OOS rank gives a logit λ, and `PBO = P(λ < 0)`. For `n_blocks=16`, `C(16,8) = 12,870` — corrects the long‑propagated "12,780" typo. |
| `effective_n_trials(perf_matrix, seed, max_clusters)` | ONC effective trial count: cluster the trial‑correlation matrix (Mantegna distance) and count clusters. Correlated trials collapse, so `N_eff ≤ N_raw` — the optimistic end of the DSR range. Reuses the §4 `make_clusters`. |

**DSR is reported as a ladder over N**: N_eff → N_raw → 2·N_raw → 4·N_raw, the upper rungs reflecting the implicit feature‑selection search (cluster‑rep reducer + F16 re‑open) that N_raw under‑counts.

### 5.21 Stage 5/6 — Signal analysis (`signal_analysis.py`)

Primary‑signal characterisation for EX.5 (sets the meta‑model's ceiling) and the directional‑timing tests for §5/§6 (S5.10).

| Function | Definition |
|---|---|
| `signal_hit_rate(signal, forward_returns)` | fraction of non‑zero signal days whose sign matches the forward return |
| `signal_turnover(signal)` | mean `|Δsignal| / 2` per period |
| `information_coefficient(signal, forward_returns, method="spearman")` | **Spearman** rank correlation between the signal and the forward return — the IC |
| `information_ratio(ic, breadth)` | **Grinold's Fundamental Law: `IR = IC·√breadth`** (Grinold 1989) |
| **`pesaran_timmermann(realised, predicted)`** | **THE PRIMARY directional test** (LR‑8). Base‑rate‑aware: `S = (P̂ − P̂*) / √(var P̂ − var P̂*) → N(0, 1)` under no‑skill null. **A constant call in a trending market scores S = 0** rather than spurious positive. Returns `(stat, one_sided_p_value)`. Undefined when all directional calls coincide. Cite PT 1992 pp. 461‑465. |
| `treynor_mazuy(market, portfolio)` | **TM convexity** (HBR 1966, cited only for the quadratic spec — not authority on the artefact). Regresses `r_p = α + β·r_m + γ·r_m² + ε`; `γ` is the timing signature. |
| `henriksson_merton(real, pred)` | **Base‑rate‑*sensitive* proxy** (the relabel pass 4 made). Reports `(hit_rate, z_stat, p_value)` for the naive `z = (p̂ − 0.5) / √(0.25/P)`. Documented caveat: not canonical H‑M — neither the parametric regression form nor the conditional non‑parametric form. Biased *toward* "skill" in any trending window; reported only as a complement. A proxy biased toward skill that still shows none is the conservative reading. |

### 5.22 Determinism + leakage discipline (`seeding.py`, `_env.py`, `_vendor/`)

**`seeding.py`** — `set_seeds(seed=42)` synchronises `random`, `numpy`, `torch`, `tensorflow`, `PYTHONHASHSEED`. Called at every entry point — the determinism contract.

**`_env.py`** — single‑thread native kernels (BLAS, OpenMP). **Also fixes a macOS libomp segfault** that occurred when both `xgboost` and `torch` were imported into the same process.

**`_vendor/`** — verbatim sts‑ml scripts, kept readable for the §4 bug‑fix audit:

| File | Source | Modifications |
|---|---|---|
| `cluster_feature_importance.py` | sts‑ml | The four §4 bug fixes applied (see §5.16) |
| `regression_metrics.py` | sts‑ml | verbatim |
| `trend_scanning.py` | sts‑ml | `tValLinR` validated equal to `_segment_tval` to 1e‑9 |
| `vsn.py` | PS6 | `FinalModel` Variable Selection Network |

### 5.23 Experiments (EX.1, EX.3, EX.4, EX.5, S3, S4, S6.7, S6.8, X.8)

10 scripts under `experiments/`, all writing to `experiments/results/*.md` (gitignored). **None feed the locked config** — they are reproducible diagnostics, not part of the deliverable pipeline.

| Script | Purpose |
|---|---|
| **`_common.py`** | Shared helpers — builds the real pooled modelling panel per asset class on the released window. |
| **`ex1_edge_decomposition.py`** | Per asset class on modelling sample: (a) primary‑signal barrier hit rate = meta‑label `pos_rate`, (b) best model's **15‑path CPCV AUC distribution and `paths > 0.5` count**. |
| **`ex3_barrier_surface.py`** | Barrier `(pt_sl, max_holding)` sensitivity surface; diagnostic only, never feeds the locked config. |
| **`ex4_calibration.py`** | Reliability curves + ECE before/after Platt + isotonic on a leakage‑safe 70/30 in‑time split. |
| **`ex5_signal_characterisation.py`** | Per‑instrument hit rate, turnover, IC, IR, Grinold's Fundamental Law `IR = IC·√BR`. Sets the meta‑model ceiling. |
| **`s3_calibration_selected.py`** | Pass‑3 calibration on the *selected* models (not LightGBM‑as‑proxy). ECE Energy 0.140→0.100, Equity 0.055→0.027, Metals 0.210→0.001. |
| **`s4_cluster_importance.py`** | Pass‑4 cluster importance on the real F16‑expanded matrix. |
| **`s6_barrier_backtest.py`** | The 449‑line headline runner. Refits the shipped default‑path model per class, predicts the OOS window, sizes with calibrated Kelly, runs both backtests, computes utility (PT/TM/H‑M proxy + CER), the significance block, the S6.15 CER gate, the **EX.6 leakage‑safe κᵢ gate**, and the **S5.12 standardise‑then‑repool TM check**. Persists `s6_net_returns.csv` so §6 inference is reproducible from an artifact. |
| **`s6_deflation_gate.py`** | DSR ladder + CSCV‑PBO + MinBTL — **demoted to corroboration** of S6.14. |
| **`x8_feature_counts.py`** | Measures (not asserts) per‑class feature counts: Energy 141 / Equity 140 / Metals 141 columns pooled; cluster matrix 108/107/108 after dropping zero‑variance and regime/macro blocks. |

### 5.24 The five passes (pass 1 → pass 5)

This branch's distinctive trait: the methodology was developed across **five sequential passes**, each with its own plan file under `docs/plans/`. The `STML_ActionItem_Tracker.md` logs every decision chronologically as PM‑1 through PM‑12.

| Pass | Plan file | Theme |
|---|---|---|
| **Pass 1** | `2026-05-30-metamodel-build.md` | Initial Stage‑0 → Stage‑6 pipeline build — triple_barrier, CV, sizing, vol, features, regime, models, evaluation, NaN policy, pipeline, emit, cluster_importance, per‑instrument OOS, neural, VSN, backtest, methodology draft. |
| **Pass 2** | `2026-05-30-pass2.md` | **Honesty pass** on two write‑up over‑claims (advisor review); reframes the §6 narrative; promotes the torch NN family + CPCV‑as‑selection + macro to the shipped default path. |
| **Pass 3** | `2026-05-30-pass3.md` | **Deflation gate, per‑class calibration, utility metrics, honest §6.** EX.4 calibration spec, S3.9 ship Platt, S6.8 DSR/PBO/MinBTL ladder, S6.10 accounting reconciliation, S6.11 calibrated‑deliverable contract. |
| **Pass 4** | `2026-05-30-pass4.md` | **F16 re‑open + per‑instrument embargo + significance‑first §6.** S2.6 per‑instrument `embargo_p90`, S1.8 F16 concept‑drift added under the same per‑fold causal discipline, S6.14 promotes significance over deflation, S5.10 relabels H‑M as a proxy and elevates **Pesaran–Timmermann** to PRIMARY. |
| **Pass 5** | `2026-05-30-pass5.md` | **Pre‑submission polish.** EX.6 leakage‑safe κᵢ (REVERT), S5.12 standardise‑then‑repool TM (COLLAPSE proves artefact), Sortino fix to full‑T form, Kang‑Kim 2025 citation deleted as non‑existent, the academic report `T3_03_Alken_Metamodel_Report.md` produced. |

Each pass plan opens with a numbered action list; each tracker entry records the **independently verified** numbers (e.g. PM‑9: "t = SR·√n = 0.932 ✓ (claimed 0.932); ann Sharpe 1.312 ✓; PSR(0) 0.823 ✓; MinTRL 399d ✓; Ljung‑Box(10) p = 0.010 ✓"). Adoption decisions (e.g. EX.6 κᵢ REVERT, smooth taper REVERT) are explicitly logged with their rationale.

### 5.25 Shipped configuration and the deliverable

Default emit configuration (`emit.py main()`):

```python
PipelineConfig(
    predict_start            = 2022-01-01,
    predict_end              = 2022-06-30,
    fe_train_end             = 2021-07-01,
    modelling_end            = 2021-12-31,
    pt_sl                    = (1.0, 1.0),
    max_holding              = 10,
    n_splits                 = 5,
    pct_embargo              = 0.01,
    per_instrument_embargo   = True,   # S2.6 — pass 4
    use_regime               = True,
    use_macro                = True,   # S1.7 — pass 3
    use_drift                = True,   # S1.8-b F16 — pass 4
    roster                   = "default",   # tree/linear + reduced torch NNs
    cv_scheme                = "cpcv",      # CPCV-as-selection, 15 paths
    cpcv_groups              = 6,
    cpcv_test_groups         = 2,
    seed                     = 42,
)
```

**Selected models (CPCV winners, 15‑path mean OOS AUC):**

| Class | Selected | 15‑path mean AUC |
|---|---|---:|
| **Equity** | **XGBoost** | **0.579** |
| **Energy** | **torch‑MLP** (a neural variant won) | 0.525 |
| **Metals** | **elastic‑net logistic** | 0.530 |

Sizing: **flat `κ = 0.25`** fractional Kelly, hard floor `p̂ ≥ 0.55`, vol target 25% — both the smooth‑taper variant and the leakage‑safe Baker‑McHale κᵢ were rejected by the EX.6/S6.15 CER gates.

Per‑class Platt calibration fit on purged modelling‑OOF preds (strictly `<= modelling_end`, before `predict_start` → cannot leak). Platt is monotone so **AUC is unchanged** by construction (unit‑tested invariant); only Brier/ECE and the stake move.

**Deliverable files** (`outputs/`, gitignored):

```
metamodel_predictions.csv               1011 rows × 3 cols, calibrated
metamodel_predictions_calibrated.csv    1011 rows × 3 cols, byte-identical alias
metamodel_predictions_raw.csv           1011 rows × 3 cols, uncalibrated
strategy_weights.csv                    1011 rows × 3 cols, vol-targeted + Kelly with calibrated p̂
coverage_caveat.csv                     11 rows — per-instrument n_oos_rows, ic, ic_undefined, thin
experiment_log.csv                      3 rows — one per asset class, deterministic
```

**Byte‑identical re‑emit verified** across two independent runs on the pass‑4 shipped path. Canonical `metamodel_predictions.csv` equals the calibrated file (resolves S6.11 — the deliverable *is* calibrated).

### 5.26 Numerical results

**Classification (per‑instrument purged‑OOS AUC):**

| Class | Model | Per‑instrument AUC (n labels) |
|---|---|---|
| Equity | XGBoost | es1s **0.60** (457) · nq1s **0.61** (482) · fesx1s **0.59** (510) |
| Metals | logistic | hg1s 0.58 (504) · pl1s 0.50 (453) · si1s 0.50 (462) · gc1s 0.41 (138) |
| Energy | torch‑MLP | cl1s 0.55 (334) · rb1s 0.50 (504) · ho1s 0.43 (61) · ng1s 0.35 (68) |

**Thin‑coverage flag (`n_oos_rows < 60` OR undefined IC):** **ho1s (2 rows), gc1s (30), ng1s (56)**.

**CPCV path robustness (EX.1):**

| Class | Best | Mean CPCV AUC | Paths > 0.5 | Reading |
|---|---|---:|---|---|
| **Equity** | XGBoost | **0.572** | **15 / 15** | edge **robust** |
| **Metals** | logistic | 0.524 | 13 / 15 | marginal but positive |
| **Energy** | LightGBM | 0.493 | 6 / 15 | **no reliable edge** |

**Cluster MDA (§4, on the real F16‑expanded matrix):**

| Class | clusters | top cluster MDA | top cluster SHAP | near‑zero MDA |
|---|---|---|---|---|
| Equity | 3 | 0.025 (only materially positive cluster) | 0.43 | 2 / 3 |
| Energy | 3 | 0.003 | 0.40 | 3 / 3 |
| Metals | 2 | −0.011 | 0.71 | 2 / 2 |

**Calibration (EX.4 / S3.9, leakage‑safe in‑time 70/30 on the *selected* models):**

| Class | Selected | ECE raw → Platt | Brier raw → Platt | AUC (raw = Platt by Platt monotonicity) |
|---|---|---|---|---:|
| Energy | torch‑MLP | 0.140 → **0.100** | 0.254 → 0.244 | 0.541 |
| Equity | XGBoost | 0.055 → **0.027** | 0.247 → 0.242 | 0.607 |
| Metals | logistic | 0.210 → **0.001** | 0.313 → **0.247** | 0.532 |

**Strategy backtest (§6, barrier‑exact + cost‑aware, calibrated sizing, n = 127 OOS days):**

| Book | Model | Sharpe | Sortino | Ann. vol | Max DD | Turnover/yr | Hold (d) | Gross → Net |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **All 11** | — | **1.31** | 1.99 | 7.5 % | −2.6 % | 53.7 | 2.8 | +8.3 % → **+4.9 %** |
| Energy | torch‑MLP | 1.86 | 3.22 | 2.9 % | −1.0 % | 7.8 | 3.0 | +3.2 % → +2.7 % |
| Equity | XGBoost | 0.86 | 1.37 | 5.2 % | −2.0 % | 22.6 | 2.4 | +3.6 % → +2.2 % |
| Metals | logistic | 0.00 | 0.01 | 3.8 % | −2.8 % | 23.3 | 3.0 | +1.3 % → −0.0 % |

**S6.14 significance — THE PRIMARY §6 RESULT (pooled all‑11):**

| Statistic | Value | Reading |
|---|---|---|
| Sharpe (per‑period); **t = SR·√n** | SR 0.083; **t = 0.93** (n = 127) | **NOT SIGNIFICANT** at 5%, before any deflation |
| **Studentised stationary block‑bootstrap 95% CI (PRIMARY)** | per‑period **[−0.04, +0.19]**; ann × √252 [−0.59, 3.07] | **CONTAINS 0** — the width *is* the finding |
| Lo / Opdyke analytic band | per‑period [−0.09, +0.26] | parametric cross‑check, also straddles 0 |
| PSR(0); **MinTRL** | 0.82 (< 0.95); **399 days** vs 127 available | track ~3× too short to certify |
| Ljung–Box(10) | p = 0.010 | returns are serially correlated → √252 annualisation *overstates*; read the per‑period CI |

**S6.8 deflation (corroboration):**

| Book | Net Sharpe | DSR ladder (N_eff → N_raw → 2·N_raw → 4·N_raw) | CSCV‑PBO | MinBTL vs ~0.5y OOS |
|---|---:|---|---:|---|
| Energy | 1.86 | [0.86 → 0.77 → 0.70 → 0.64] (N 2→20) | 0.29 | [0.27 → 1.42] y |
| Equity | 0.86 | [0.57 → 0.51 → 0.43 → 0.37] (N 3→20) | 0.46 | [0.73 → 1.42] y |
| Metals | 0.00 | [0.35 → 0.19 → 0.12 → 0.08] (N 2→20) | 0.12 | [0.27 → 1.42] y |
| **All‑11** | **1.31** | **[0.61 → 0.34 → 0.26 → 0.20]** (N 3→60) | **0.35** | **[0.73 → 3.14] y** |

**Even the optimistic N_eff end stays below 0.95** everywhere (pooled 0.61); DSR only falls as N rises.

**Pesaran‑Timmermann directional skill (§5, S5.10):**

| Book | PT stat (p) — *primary* | TM γ (t) | H‑M proxy hit (z) | CER/day (γ = 5) |
|---|---|---|---|---|
| Energy | +0.24 (0.41) | +0.81 (1.34) | 0.510 (0.29) | +0.000203 |
| Equity | −0.32 (0.63) | **−4.51 (−2.01)** | 0.447 (−1.38) | +0.000151 |
| Metals | −2.17 (0.99) | −2.05 (−1.10) | 0.408 (−3.06) | −0.000014 |
| **All‑11** | **−2.31 (0.99)** | +1.18 (2.55) | 0.449 (−2.57) | +0.000335 |

**S5.11/S5.12 — the Treynor‑Mazuy artefact resolution.** The pooled TM γ = +1.18 (t = 2.55, nominally significant) **would** read as positive convexity timing skill. Two‑literature synthesis:

1. **Aggregation bias** (Robinson 1950; Zellner 1962; Blyth 1972; Pesaran & Smith 1995) — pooling sleeves of heterogeneous return scale yields inconsistent, sign‑reversing coefficients. Per‑sleeve γ here are Equity −4.51 (sig), Energy +0.81 (ns), Metals −2.05 (ns) — decisively rejecting coefficient homogeneity.
2. **Artificial timing** (Jagannathan‑Korajczyk 1986) — option‑like / convex payoffs produce *artificial* market‑timing ability where no genuine timing exists. A protective barrier‑exit rule is exactly such a convex, option‑isomorphic payoff (Henriksson‑Merton 1981; Glosten‑Jagannathan 1994; Fung‑Hsieh 2001).

**S5.12 in‑data proof.** Re‑estimate pooled TM after **vol‑targeting each sleeve to a common scale** (divide both market return and signed PnL by that sleeve's return std — the *same* factor, preserving `pnl = side · ret`):

> **Pooled γ collapses from +1.18 to −0.0031 (t = −0.14, p = 0.89)**, against a trade‑count‑weighted average of per‑sleeve γ of **−1.835**. Once the sleeves share a common scale the manufactured convexity disappears — **direct in‑sample proof that the +1.18 was scale‑aggregation, not timing.**

**S6.15 / EX.6 — sizing follow‑up (REVERT, deliverable unchanged):**

- Smooth taper replacing the hard floor: CER `0.000335 → 0.000353` (+5%). **Immaterial** under the >10%‑of‑baseline margin → **REVERT**.
- Per‑instrument Baker‑McHale κᵢ estimated on **modelling‑sample OOF** (leakage‑safe): CER `0.000335 → 0.000644` (point **+92 %**). But **paired studentised bootstrap CI = [−0.000599, +0.001136] contains 0** → **REVERT**.
- The OOS‑estimated κᵢ variant (CER → 0.000715) is a **circular diagnostic** (look‑ahead) — rejected.
- **Decision: flat κ = 0.25, hard floor; deliverable weights byte‑identical to pass‑4.**

**The five‑lens convergence (S6.12)** — one honest negative, not five unlucky ones:

| Lens | Pass‑5 result | Verdict |
|---|---|---|
| §3/§5 OOS AUC | ≈ 0.50 (0.57 / 0.54 / 0.53) | no ranking skill |
| §4 cluster MDA (OOS) | \|MDA\| < 0.02 across clusters | no feature carries OOS edge |
| §6.14 significance | t = 0.93; bootstrap 95% CI contains 0 | Sharpe not distinguishable from 0 |
| §6.8 deflation | pooled DSR [0.61 → 0.20]; PBO 0.35 | fails the selection‑bias gate |
| §5 timing (Pesaran‑Timmermann) | pooled −2.31 (p = 0.99) | no positive directional timing |

> This convergence is *predicted*, not coincidental. The named primary is short‑horizon mean‑reversion (`f1_mr_score_20`), and **Grinold's Fundamental Law `IR = IC·√BR`** *[PROVEN]* makes the null structural: with the primary's IC ≈ 0, the achievable information ratio is ≈ 0 **regardless of breadth or sizing**.

### 5.27 Tests

**Top‑level (feature library, `tests/`):** 13 files, ~120+ tests:

| File | Tests | Covers |
|---|---:|---|
| `test_build_determinism.py` | 3 | byte‑identical rebuild + autoencoder reproducibility within 1e‑10 |
| `test_drift_features.py` | 4 | F16 right‑edge truncation invariance + rolling refit determinism |
| `test_features_catalog.py` | 12 | catalog 1:1 ↔ produced columns + schema |
| `test_features_ext_leakage.py` | 5 | F2‑RS / F5‑adds / F7‑adds / F12 / F13 / F15 truncation invariance |
| `test_features_leakage.py` | 15 | F1 / F2 / F5 / F6 / F7 / F8 / F10 truncation invariance |
| `test_features_provenance.py` | 9 | provenance JSON schema and values |
| `test_features_scope.py` | 12 | `InstrumentScope` registry consistency |
| `test_latent.py` | 10 | F4 deterministic PCA + KMeans + autoencoder |
| `test_macro_features.py` | 18 | PIT alignment + publication‑lag correctness + block‑level truncation invariance + EIA release semantics |
| `test_regime_features.py` | 9 | F3 GMM + Markov filter causality |
| `test_regimes_hmm.py` | 4 | F17 HMM state ordering + causal forward filter |
| `test_splits.py` | 18 | chronological train/val/test + embargo + `n_eff` |
| `test_xsection.py` | 17 | F9 per‑day rank + pair correlation + cross‑asset z |

**Nested subproject (`metamodel-apb/tests/`):** 25 files, ~215 tests:

| File | Tests | Covers |
|---|---:|---|
| `test_volatility.py` | 6 | GK/Parkinson/RS closed forms; non‑negative; rolling agreement |
| `test_triple_barrier.py` | 9 | first‑touch, vol‑adaptive barriers, uniqueness on disjoint (= 1) / fully overlapping (= 0.5) |
| `test_cross_validation.py` | 6 | PurgedKFold + CPCV correctness, embargo enforcement |
| `test_cross_validation_embargo.py` | 3 | **per‑instrument `embargo_p90`** on each instrument's own date axis |
| `test_sizing.py` | 11 | Kelly fraction + vol target + Baker‑McHale κᵢ monotonicity + smooth taper continuity |
| `test_features.py` | 13 | **`test_right_edge_truncation_invariance`** (the property test) + `test_no_metamodel_module_reads_frozen_parquet` (the leakage guard) + backward trend t‑val agreement with `tValLinR` |
| `test_regime.py` | 9 | EWMA HMM truncation invariance; state ordering; warm‑up NaN |
| `test_macro.py` | 5 | PIT alignment + release deferral + block‑level truncation invariance |
| `test_models.py` | 6 | `balanced_sample_weight` properties + estimator interface |
| `test_neural.py` | 11 | TorchMLP / TorchVSN byte determinism + ReducedEstimator fold safety |
| `test_evaluation.py` | 8 | sample‑weighted metrics + single‑class fold NaN handling |
| `test_cluster_importance.py` | 5 | Mantegna distance + noise‑cluster ≈ 0 sanity + the four bug fixes |
| `test_dim_reduction.py` | 8 | medoid selection, PCA variance threshold, autoencoder full‑batch determinism |
| `test_calibration.py` | 4 | Platt AUC invariance + isotonic monotonicity + ECE properties |
| `test_pipeline.py` | 19 | per‑class orchestration + `run_asset_class` + nested CPCV + Platt deliverable contract |
| `test_pipeline_embargo.py` | 3 | per‑instrument embargo threaded through PurgedKFold + CPCV + nested CPCV |
| `test_load_path.py` | 1 | both deliverable and §6 backtest ingest via `stml.io.load_clean_data()` |
| `test_backtest.py` | 13 | **`test_sortino_full_t_form`** (the known‑value test pinning the Sortino‑Price 1994 form) + barrier‑exact accounting + reconciliation identity |
| `test_cost_model.py` | 5 | Grinold‑Kahn half‑spread + impact + turnover correctness |
| `test_significance.py` | 9 | t‑stat + Lo CI + studentised stationary bootstrap + Ljung‑Box + MinTRL |
| `test_deflation.py` | 23 | PSR + DSR + Bailey‑LdP expected‑max + CSCV PBO + ONC N_eff + `C(16, 8) = 12,870` |
| `test_signal_analysis.py` | 13 | PT base‑rate invariance + TM convexity sign + H‑M proxy + IC + IR |
| `test_ex6_s512.py` | 6 | leakage‑safe κᵢ window guard + paired CER‑diff bootstrap + S5.12 sign‑collapse |
| `test_experiment_log.py` | 3 | deterministic per‑class row order |

Pass‑5 docstring asserts **210 passed** in the nested suite. **RED‑first TDD** throughout: every new feature has a test written before the code.

### 5.28 Documentation and the academic report

The documentation layer is unique to this branch — six distinct artifacts beyond the in‑code docstrings.

| Artifact | Purpose |
|---|---|
| `docs/methodology.md` (631 lines) | The full methodology: scope, architecture, sections 0–6 + bonus, commitments → modules → citations, determinism and leakage discipline, honest limitations. Heavily cross‑referenced with the literature review. |
| `docs/plans/2026-05-30-{metamodel-build,pass2,pass3,pass4,pass5}.md` | One plan per pass, each with a numbered action list and adoption rationale. |
| `reports/STML_ActionItem_Tracker.md` (378 lines) | The chronological log of PM‑1 through PM‑12 evaluation entries — each entry independently verifies the numerical claims of its pass and records adoption / rejection / revert decisions. |
| `reports/CW_Breakdown.md` (200 lines) | The internal coursework breakdown — what each rubric section requires + module mapping. |
| `reports/research/LR-1.md … LR-9.md + nlr-cw-v1.md` | **9 literature‑review notes (LR‑1 … LR‑9)** investigating specific claims (Grinold's Fundamental Law, Bailey‑LdP DSR/MinBTL, Carver vol targeting, Jagannathan‑Korajczyk artificial timing, etc.) and a master `nlr-cw-v1.md` with 60 citations across the 8 commitments. |
| `reports/T3_03_Alken_Metamodel_Report.md` (~220 lines, **42 Harvard refs**) | **The submission‑ready academic report.** Title: "A Meta‑Labelling Metamodel for a Multi‑Asset Futures Universe: An Honest‑Negative Evaluation". §1‑9 narrative (intro / data + features / labelling / models + validation / feature importance / evaluation incl. TM artefact resolution / strategy + significance / discussion / conclusion). References use Imperial Harvard style. |

The academic report's executive summary lays out the central finding:

> The central finding is an **honest negative**. On the six‑month out‑of‑sample window the pooled net Sharpe ratio of 1.31 is **not statistically distinguishable from zero**: the raw *t*‑statistic is 0.93 (n = 127), and the primary inference — a studentised stationary block‑bootstrap 95% confidence interval — straddles zero. This conclusion is corroborated by five mutually independent diagnostics: out‑of‑sample AUC ≈ 0.50, near‑zero cluster mean‑decrease‑in‑accuracy, a deflated‑Sharpe ladder that fails the 0.95 threshold at every trial count, a negative Pesaran‑Timmermann directional‑accuracy statistic, and the insignificant Sharpe itself. Two subtle statistical traps were identified and resolved: a positive *pooled* Treynor‑Mazuy coefficient shown to be a scale‑aggregation artefact (reproduced and dissolved in‑sample), and a circular out‑of‑sample‑estimated position‑sizing shrinkage rejected in favour of a leakage‑safe variant that, in turn, fails a bootstrap materiality test. The appropriate scholarly conclusion is **insufficient evidence of a deployable edge**, not a demonstrated failure: the positive Sharpe is attributable to the convex barrier‑exit mechanism, volatility targeting and diversification rather than to act/skip skill.

### 5.29 Open issues on alken-metamodel

From the methodology's own Limitations section and the action tracker:

1. **Nested CPCV real‑data run is deferred.** `nested_cpcv_select_and_evaluate` is implemented + unit‑tested but a meaningful run is ~15 × 5 × 5 = 375 fits per class. Flagged in the methodology as the remaining selection‑side gap.
2. **Macro vintages.** `additional_data.xlsx` ships observation dates and is **publication‑lag (PIT) aligned** so no *timing* look‑ahead, but the workbook ships **revised / final values, not real‑time vintages** — so a revision look‑ahead (e.g. a later‑restated TIPS10Y/BE10Y print) is not excluded. A full ALFRED real‑time‑vintage reconciliation is the remaining macro gap.
3. **§6 constraint set** is a literature‑default stub (the 20 May constraints doc absent from the repo). Only the barrier‑exact holding model, cost model and deflation gate were upgraded.
4. **Autoencoder reducer** built for the EX.2 comparison only; cluster‑rep is the promoted reducer.
5. **Keras‑VSN** kept off the selectable path — TF op‑determinism is best‑effort, and a non‑reproducible model could be picked by the grader's hidden‑half re‑run.
6. **ho1s (2 OOS rows), gc1s (30), ng1s (56)** rest on thin coverage — flagged in `coverage_caveat.csv`; per‑instrument numbers small‑sample noise.
7. **Equity instruments start late** (es1s 1997, fesx1s 1998, nq1s 1999) — thin pre‑2020 history for fitted features.
8. **Three earlier‑draft citation issues already corrected in pass 5**: non‑existent "Kang & Kim (2025)" (a conflation of Fu/Kang/Hong/Kim 2024) removed from LR‑1; Sortino implementation moved to the full‑T Sortino‑Price (1994) form; Ang‑Bekaert Table‑2 correlation *values* not quoted (sourced to Guidolin‑Timmermann); Carver 2015 Ch.9 page marked pending print‑edition.

---

## 6. Side‑by‑side comparison — three branches

| Aspect | Sreeram | Harry | `model/alken-metamodel` |
|---|---|---|---|
| **Branch posture** | Build broad pipeline first, audit and iterate (v0 → v5). Forensically discovered the equity regime break and selection‑on‑test trap. | Audit first, pick a side, build under strict invariants. Methodology set before any modelling. | Methodology‑first, literature‑backed (60 refs across 9 LR notes + nlr‑cw‑v1.md); the project develops across 5 sequential passes with chronological PM‑1..PM‑12 evaluation log. |
| **Files added vs main** | ~30 k LOC including data | ~246 k LOC including binary outputs | shared feature library at repo root + nested uv subproject `metamodel-apb/`. The most disciplined commit message convention (every commit stage‑tagged: `feat(s1.7):`, `feat(ex6,s512):`, `docs(pass5):`). |
| **Architecture / namespace** | Flat — everything in `src/stml/` next to `io.py` / `na_checks.py`. | Two subpackages on one branch: `src/stml/harry/` for foundation, `src/stml/new_work/` for production. | **Two coexisting projects**: top‑level `src/stml/metamodel/` shared feature library (175 features) + nested `metamodel-apb/src/alken_metamodel/` modelling pipeline. Explicit `CLAUDE.md` forbids consuming the frozen parquet. |
| **PnL convention** | Implicit, label window `[t, t+h]` (same‑bar return inside window) | Explicit, `PnL_t = s_t · r_{t+1}`, label window `[t+1, t+1+h]` | Implicit, label window `[t, t+h]`. Triple‑barrier σ̂ ties to de‑annualised `f2_vol_20 / √252`. |
| **Labeller default** | Symmetric `pt=sl=1.0`, EWMA σ (span 100), h=10 | Two labellers coexist: `harry/labels.py` (symmetric, EWMA) and `new_work/triple_barrier.py` (asymmetric `pt=1.5, sl=1.0`, **GARCH(1,1)**). Production uses GARCH. | Symmetric `pt_sl=(1.0, 1.0)`, **vol‑adaptive Garman‑Klass σ̂**, `max_holding=10`. Vertical barrier labelled by sign of expiry PnL. |
| **Causality enforcement** | Per‑module tests (esp. HMM filtered ≠ smoothed) | Universal harness over a registry — 22 features × 3 properties auto‑parametrised | **Right‑edge truncation‑invariance property tests** on every feature function (`test_right_edge_truncation_invariance`) **plus** `test_no_metamodel_module_reads_frozen_parquet` (the leakage guard against consuming the frozen feature matrix). |
| **Feature count** | ~75 (G1–G8, after v4) | ~25 columns from harry/features + 31 macro + 8 HMM + ~50 ported signal‑deep‑dive `f1‑f15` ≈ 100+ features | **175 in the shared library** (E + TF + z‑twins). **140‑141 pooled per class** in the nested pipeline (core 115 + F16 + EWMA‑HMM 5 + PIT‑macro 16 + instrument one‑hots 3‑4). Cluster matrix ~108. |
| **Cross‑sectional features** | G8 added in v4 | F15 family + asset‑class membership + dispersion/correlation z‑scores | **F9** family — per‑day rank, universe size, pair‑corr mean, lead‑lag centroid, asset‑class dispersion z, EWMA implied‑corr z. |
| **Macro / Bloomberg data** | None (Sreeram's `data/bloomberg/` is empty) | 21 series + 31 engineered features, fully causal, 1990‑2022 | **22 series in `additional_data.xlsx`** + **F11 in the shared library (45 cols)** + **PIT‑lagged macro block in the nested pipeline** with conservative publication lags (daily +1d, EIA weekly +5d, PMI monthly +30d). |
| **HMM** | 3‑state per‑instrument on (return, vol), filtered forward (hand‑rolled `causal_filtered_probs`), trained on `date < boundary` | Core #1 (per‑inst, 3‑state on log‑GK vol) + Core #2 (global, 2/3‑state on 4‑D macro). Both trained on pre‑2020 only and **frozen** | **Online EWMA 2‑state Gaussian HMM** (causal/fit‑free; recursive emission update via EWMA of responsibility‑weighted sufficient stats, fixed persistent transition prior) **PLUS** stml static F3 GMM + Markov‑switching + F17 3‑state HMM on a contiguous `fe_train_end` prefix. The EWMA variant has **no per‑fold CPCV seam artefact**. |
| **GMM** | Yes — 3 components on (vol, mom, autocorr) | Not used | Yes — F3 block uses GMM + Markov‑switching on `(ret, vol)`, fit on contiguous prefix. |
| **CV machinery** | `PurgedKFold(n_splits=5)` + embargo (`pd.Timedelta(days=10)`) | CPCV (`n_groups=6, k=2, embargo=0.01`) — 15 OOS paths per config; adaptive params for thin instruments | **CPCV‑as‑SELECTION** (`n_groups=6, k=2` → 15 paths) **+ nested CPCV implemented and unit‑tested** (real‑data run deferred) **+ per‑instrument embargo using each instrument's own `embargo_p90` on its own date axis** (S2.6, pass 4). ng1s 33d, ho1s 26d, rb1s 19d. |
| **Model families** | LogReg (elasticnet), XGBoost, MLP. VSN exists but doesn't run on Apple Silicon. | logistic, RF, XGBoost, MLP — 4 families with inner expanding‑window 75/25 tuning | **5 estimators in the shipped roster**: elastic‑net logistic + XGBoost + LightGBM on full features, **torch‑MLP + torch‑VSN on cluster‑rep‑reduced features**. Keras‑VSN exists but is **off‑path** (TF op‑determinism is best‑effort). |
| **Calibration** | Isotonic via `CalibratedClassifierCV` with purged CV, fit during model construction | Post‑hoc per‑fold leakage‑safe Platt + isotonic → pick best by Brier | **Per‑class Platt** fit on the *selected* model's purged modelling‑OOF preds (strictly `< predict_start` → cannot leak). **Platt monotonicity → AUC unchanged** is a unit‑tested invariant. Deliverable ships calibrated; raw retained for the §3 Brier story. |
| **Per‑sector / pooled** | Per‑sector ablation; v3 trains on commodity only | 3 pooled groups (`energy_all`, `energy_cl_ho`, `precious`) + 10 individual groups; pooled groups get instrument dummies | **Three asset‑class metamodels** (Equity / Energy / Metals) by design — separate models, separate Platt calibrators, separate diagnostics. Instrument one‑hot dummies inside each class. |
| **Cluster importance** | Hierarchical clustering on `1 − |Spearman|` with silhouette K. Clustered MDI + MDA (shared row permutation). Cross‑tab vs declared G1‑G8. | Ward linkage on `√(1 − |ρ|)`. K by silhouette + CH + DB. MDA + MDI + Tree SHAP. Cross‑method Kendall τ. Within‑cluster mean\|SHAP\| + PC1. Adaptive CPCV for thin instruments. | **Mantegna `√(1 − |ρ|)` distance** + PCA + optimal‑K KMeans + cluster MDI + **purged MDA** + cluster SHAP (TreeExplainer). Carries the **four required bug fixes** to PS2/sts‑ml/PS4 (max_features='sqrt', injected PurgedKFold for MDA, real SHAP added, Mantegna metric). |
| **Sizing** | Kelly‑adjacent conviction ramp + vol target. v4 weights breach 10% portfolio‑vol cap as committed. | Not implemented. | **Fractional Kelly κ = 0.25** + hard floor p̂ ≥ 0.55 + 25% vol target. Smooth taper + Baker‑McHale κᵢ implemented and **tested**, both REVERTED by the EX.6 paired CER‑difference bootstrap gate (point gain immaterial / CI contains 0). |
| **Backtest** | Conventional vol‑targeted with conviction ramp; in‑sample backtested. | None. | **Barrier‑exact** (exit on actual triple‑barrier `t1`, overlapping labels **netted**) **+ cost‑aware** (Grinold‑Kahn half‑spread + impact, 2 bps / 10 bps). Reports gross/net split, turnover, holding period. **Sortino‑Price 1994 full‑T form** (pinned by known‑value test). S6.10 accounting reconciliation closes the compounding identity. |
| **Significance** | Bootstrap CI in v5 (200 reps), pre‑promoted; stress test reports 95% CI [0.43, 0.50] on H1‑22 contains 0.5. | None. | **THE PRIMARY §6 result.** t‑stat = SR·√n; **studentised stationary block‑bootstrap CI** (Politis‑White data‑driven block length, Lo SE, 2000 reps, seeded → deterministic); Lo/Opdyke analytic band; PSR(0); MinTRL; Ljung‑Box(10) gate before any √252 annualisation. Pooled t = 0.93 (NOT significant); CI per‑period [−0.04, +0.19] CONTAINS 0. |
| **Deflation** | Not implemented. | Not implemented. | **DSR ladder over N_eff → 4·N_raw** + CSCV‑PBO (n_blocks = 16, **C(16,8) = 12,870** — corrects the "12,780" typo) + MinBTL. Demoted to corroboration of S6.14 in passes 4‑5. |
| **Timing skill** | Not measured. | Not measured. | **Pesaran‑Timmermann (PRIMARY)** + Treynor‑Mazuy (corroboration) + Henriksson‑Merton (proxy, explicitly labelled as base‑rate‑sensitive). Pooled PT = −2.31 (p = 0.99) — no positive directional timing. **S5.11/S5.12 resolve the +1.18 pooled TM as a scale‑aggregation artefact, with in‑data proof**: standardising sleeves collapses pooled γ from +1.18 to −0.0031 (t = −0.14). |
| **Champion selection** | "Best by OOS log‑loss" within `experiments.py`. v3 picks "drop equity from training" by hand. | **Two methodologies coexist** (see §4.19). Original (`selection_table_v2.csv`): per‑fold mean ± std AUC → signal flag → STD‑based ties → Platt calibration → tiebreak by simplicity → 4 champions (cl1s, es1s, ho1s, rb1s). 2026‑06‑02 PM refresh (`selection_table.csv` — the new canonical pick): purged inner k‑fold (k=4) + **1SE rule** on a clean global train slice (cut 2021‑10‑06, embargo to 2021‑10‑20) → 5 champions: **cl1s · nq1s · fesx1s · pl1s · hg1s**. Only cl1s survived both. | **CPCV‑selection by 15‑path mean OOS AUC**. Equity → XGBoost (0.579), Energy → torch‑MLP (0.525, neural variant won), Metals → elastic‑net logistic (0.530). |
| **Honest accounting** | v5 doc admits v3/v4 were selection‑on‑test. Stress test confirms the model is stable; the world moved. | 7 of 11 instruments classified no‑signal. ho1s flagged "indicative only" — high AUC on 63 events. | **Five‑lens convergence:** AUC ≈ 0.5, cluster MDA ≈ 0, t = 0.93 Sharpe, DSR fails, PT negative. Pesaran‑Timmermann is the scale‑invariant primary; TM convexity is the barrier‑exit option‑isomorphism (Jagannathan‑Korajczyk 1986), not timing. **Insufficient evidence of a deployable edge** — not a proven failure. |
| **OOS AUC story** | v0 0.51 → v2 0.49 → v3 0.55 → v4 0.56 → v5 0.47 (CI [0.43, 0.50] includes 0.5) | **Pre‑refresh (old reconciliation):** cl1s 0.707 · es1s 0.605 · ho1s 0.634 · rb1s 0.629; others 0.50‑0.59 with lower CI < 0.5. **Post‑refresh (2026‑06‑02 PM clean‑split):** **nq1s 0.689 · cl1s 0.675 · pl1s 0.608 · hg1s 0.604 · fesx1s 0.579** are signal‑bearing (lower CI ≥ 0.52); es1s 0.516, rb1s 0.551, si1s 0.515, ho1s 0.800 mean but CI 0.496, ng1s 0.477, gc1s 0.478 do not clear the bar. | Equity 0.60‑0.61, Energy cl1s 0.55 / rb1s 0.50 / ho1s 0.43 / ng1s 0.35, Metals hg1s 0.58 / pl1s 0.50 / si1s 0.50 / gc1s 0.41. **Mean ≈ 0.50 overall — the expected gradeable result on a near‑zero‑IC primary.** |
| **Strategy** | Implemented (`strategy.py`). Realised ann_vol 0.181 breaches 10% cap as committed. | Not implemented. | **Implemented and shipped.** Pooled net Sharpe 1.31 at 7.5% ann vol (clears 10% constraint). Sortino 1.99. Net total return +4.9%. **But t = 0.93 → NOT statistically distinguishable from zero**, by the project's own primary inference. |
| **Predictions deliverable** | `predictions_v0` – `v4` (1408 rows each), `v5` (704 rows — wrong window). | Per‑(group, model) `oos_predictions.csv`. No consolidated `predictions.csv` in the deliverable format. | **`metamodel_predictions.csv`** (calibrated, 1011 rows), `_calibrated.csv` (byte‑identical alias), `_raw.csv`, `strategy_weights.csv` (1011 rows), `coverage_caveat.csv`, `experiment_log.csv`. **Byte‑identical re‑emit verified** across two independent runs. |
| **Tests** | 44 unit tests (`test_labeling.py` 19, `test_cv.py` 13, `test_regimes.py` 12). Can't run on this venv (arm64 / x86_64 mismatch). | ~239 unit tests + universal causality harness. | **~335 unit tests** — ~120 in the top‑level feature library + **~215 in the nested subproject** (210 passing per the pass‑5 doc). RED‑first TDD throughout. Includes **byte‑identical rebuild test, frozen‑parquet guard test, Sortino‑Price known‑value test**, and the `test_right_edge_truncation_invariance` property test. |
| **Documentation** | 13 `docs/build/*.md` files; the v3 → v5 narrative is the writeup template. | 5 `reports/harry/*.md` files + `SETUP.md` + `reconciliation_report.md` + `warmup_diagnosis.md` + `outputs/importance/*/findings_note.txt`. | **Most extensive of the three:** `methodology.md` (631 lines) + 5 pass plans + `STML_ActionItem_Tracker.md` (PM‑1..PM‑12 chronological evaluation log) + `CW_Breakdown.md` + **9 literature‑review notes (LR‑1..9)** + `nlr-cw-v1.md` (60 refs) + **`T3_03_Alken_Metamodel_Report.md`** (submission‑ready Harvard‑referenced academic report, 42 refs, "honest negative" framing). |
| **Implicit alpha prior** | Trend‑following (G2 cluster prominent in‑sample) | Counter‑trend audit conclusion, but labels symmetric so model can learn its own bias. | The named primary is identified as **short‑horizon mean‑reversion** (`f1_mr_score_20`) and the methodology explicitly invokes **Grinold's Fundamental Law `IR = IC·√BR`** as the structural reason the null is predicted: with IC ≈ 0, no act/skip filter can manufacture skill. |
| **Risk to rerun** | v5's stress test says: expected H2‑22 AUC ≈ 0.55‑0.60 (same regime). | HMMs frozen on pre‑2020 → no adaptation to the H1‑22 distribution shift. | **Prediction window is config‑driven** (CLI flag `--predict-start / --predict-end`); the grader swaps in the hidden H2‑22 half by changing one value. The leakage discipline (recompute features per fold, never consume frozen parquet) means a rerun is mechanically clean. |
| **Methodology highlights** | Hand‑rolled `causal_filtered_probs`; cluster MDA with shared‑permutation; per‑instrument calibration in v4; v5 principle list. | t+1 entry off‑by‑one fix; block‑bootstrap signal audit; release‑count macro surprises; CPCV barrier search with CSCV PBO; reconciliation pipeline. **2026‑06‑02 PM refresh adds:** clean train/test global split (cut 2021‑10‑06, embargo to 2021‑10‑20) in `split_config.py`; **purged inner k‑fold (k=4) + 1SE selection** in `model_comparison.py`; champion importance covering **all 11 instruments** with logistic branch (MDA + \|coef\|) vs tree branch (MDA + MDI + SHAP); PC1‑PC3 within‑cluster PCA; three per‑asset‑class notebooks (`equity_importance.ipynb` / `energy_importance.ipynb` / `metals_importance.ipynb`). | **Pesaran‑Timmermann as primary directional test** + the Jagannathan‑Korajczyk artificial‑timing resolution of the pooled TM artefact with in‑data standardise‑then‑repool proof; **leakage‑safe Baker‑McHale κᵢ rejected by paired bootstrap** (a refinement that flatters but cannot survive its own CI); **DSR ladder over N_eff → 4·N_raw**; per‑instrument `embargo_p90`; online EWMA HMM with no CPCV seam artefact; 5 passes of advisor‑review honesty refinement. |
| **Performance highlights** | v4 AUC 0.562, v4 strategy Sharpe 3.91 (but at 18.1% ann vol). | **Pre‑refresh:** cl1s 0.707 ± 0.130 (lower CI 0.577); rb1s 0.629 with F5_signal cluster significant; ho1s 0.634 (63 events). **Post‑refresh (2026‑06‑02 PM):** nq1s 0.689 / CI 0.616 (strongest new), cl1s 0.675 / 0.536 (only one to survive both methodologies), hg1s 0.604 / 0.562 (RF champion with strong CI), pl1s 0.608 / 0.527 (logistic), fesx1s 0.579 / 0.519. | **Headline framed as honest negative.** Pooled net Sharpe 1.31; **t = 0.93 NOT significant**; bootstrap 95% CI [−0.04, +0.19] CONTAINS 0; PSR(0) = 0.82; MinTRL = 399 d vs 127 available; CSCV PBO = 0.35; PT = −2.31 (p = 0.99). All five lenses agree. |

---

## 7. Decisions still to make

The three branches cover the rubric from genuinely different angles. Each is internally coherent; the team‑synthesis decision is about which artifact (or which combination) we submit.

### Per‑section comparison

* **Feature engineering (20 marks).** alken‑metamodel ships the most consolidated catalog (175 columns documented in `reports/feature-catalog.md`, 1:1 catalog ↔ produced column assertion) and is the only branch with the full PIT‑lagged macro block + F16 concept‑drift + cluster‑representative reducer integrated end‑to‑end. Harry's macro + microstructure_fixed + wavelet + concept_drift + signal_trajectory + conditional_risk + information_theoretic is the more *creative* per‑family pack, but ~95% of those families are already wired into alken's library. Sreeram's G1‑G8 is a competent technical‑indicator + regime catalog and has the strongest documentation of the v3→v5 trade‑offs.
* **Labeling (20 marks).** Harry's t+1 entry is the most defensible labeling convention with a worked example showing the same input produces opposite labels under the two conventions. alken uses single‑day entry but with vol‑adaptive symmetric barriers tied to Garman‑Klass σ̂, uniqueness weights verified on disjoint and fully‑overlapping toy cases, and the **per‑instrument `embargo_p90`** purge that no other branch has. Sreeram is the same convention but without the per‑instrument refinement.
* **Models (30 marks).** alken runs the cleanest harness: 5 estimators (elastic‑net / XGB / LightGBM / torch‑MLP / torch‑VSN), CPCV‑as‑selection, nested CPCV implemented, **per‑class** model selection (a neural variant won Energy at 15‑path AUC 0.525). Harry runs 4 families × 13 groups with proper inner tuning. Sreeram runs 3 families and adds the v4 stacked ensemble (architecturally creative) plus v5 simple‑average ensemble.
* **Cluster importance (10 marks).** alken's `cluster_importance.py` carries the **four required bug fixes** (max_features='sqrt', PurgedKFold for MDA, real SHAP via TreeExplainer, Mantegna metric) — the only branch where they are systematically applied with each fix called out in the diff. Harry's analysis is broader (Ward linkage, silhouette+CH+DB, cross‑method Kendall τ, within‑cluster mean|SHAP|+PC1, champion‑specific re‑runs). Sreeram's is competent but smaller.
* **Evaluation (20 marks).** alken is the methodology gold standard for this section: per‑instrument before aggregate, sample‑weighted, threshold‑aware, **Pesaran‑Timmermann directional test as primary**, calibration with monotonicity invariant, deflation ladder, paired CER bootstrap, Sortino‑Price full‑T form. Sreeram has a deeper evaluation suite of its own (calibration, regime‑conditional, filtered‑strategy metrics) plus the v5 stress test as the strongest single piece of "critical analysis" on Sreeram. Harry has honest "no‑signal" exclusions for 7 instruments and the reconciliation/calibration writeup.
* **Strategy (+10).** alken is the only branch with a strategy that **passes** the 10% vol cap (7.5% ann vol pooled), barrier‑exact + cost‑aware, with explicit significance testing showing it is *not statistically distinguishable from zero*. Sreeram's strategy realises 18.1% ann vol → would breach. Harry didn't implement one.

### Three open decisions

These are the decisions for the team‑synthesis memo — not for this document.

1. **Submit which?**
   * **alken‑metamodel** alone — most rigorous, honest negative, byte‑identical reproducible deliverable, the academic report and methodology are already submission‑ready. Best methodology score; "honest negative" framing is exactly what the rubric rewards.
   * **Sreeram v5** alone — principled rebuild, low headline AUC, easy to defend in front of a quant reviewer; has predictions for all 11 instruments but the `predictions_v5.csv` window is currently wrong (only Q2‑22) and needs a clean re‑run.
   * **Sreeram v4** alone — highest paper AUC (0.562) but v5 itself documents that v3/v4 are selection‑on‑test.
   * **Harry** alone — champions for 4 of 11 instruments only; requires writing a small wrapper to produce the consolidated `predictions.csv` in deliverable format and padding the 7 no‑signal instruments.
   * **Ensemble** — simple mean over Sreeram v5 + alken predictions on overlapping rows, or a weighted average if calibration warrants. The two share H1‑2022 windows and the same 11‑instrument universe.
2. **Macro data on Sreeram?** alken already has `additional_data.xlsx` + `f11_macro_context.csv` + the PIT‑lagged macro block. If we submit Sreeram, copy these over and add a Sreeram macro layer; if we submit alken, it's already there. Harry also has macro features but the data file (`data/alternate_data_cleaned.csv`) is different from alken's. Decide which macro dataset is canonical for the team submission.
3. **Strategy track:** alken's strategy already clears the 10% vol cap and is the cleanest candidate, **but** its primary inference says t = 0.93 → not distinguishable from zero. Submitting a strategy with a documented "insufficient evidence of edge" is methodologically defensible and grade‑maximising under "methodology, not performance"; submitting it as if it works would fail an honesty review. Sreeram's `strategy_weights.csv` would need regenerating at `target_portfolio_vol = 0.10` (likely halving position sizes and dropping Sharpe). Harry has no strategy.
