# Source Code Guide

This directory contains the implementation for the data loading, signal
characterization, replication search, validation gates, and artifact rendering.
The main package is `stml`.

## Main Entry Points

`stml.io`
: Loads raw and cleaned project data.

`stml.na_checks`
: Cleans and diagnoses the futures OHLCV panel, including holiday and missing
data handling.

`stml.replication.run_characterize`
: Produces `reports/signal-characterization.md` and
`results/jj/thresholds.json`.

`stml.replication.run_replicate`
: Runs the family x cell replication search, writes per-family reports, writes
`reports/replication-summary.md`, updates the ledger, and writes
`results/jj/top_candidates.json`.

`stml.metamodel.build_features`
: Builds the leakage-safe feature matrix for the triple-barrier metamodel and
writes `results/feature_matrix.{parquet,csv}`,
`results/feature_redundancy.{json,csv}`, `results/instrument_scope.json`,
`results/feature_matrix_provenance.json`, and `reports/feature-catalog.md`.

## Replication Module Map

`replication/align.py`
: Aligns released signals with next-day returns. The PnL convention is
`PnL_t = signal_t * return_{t+1}`.

`replication/archetypes.py`
: Defines the signal families. Each family maps OHLCV to a candidate
`{-1, 0, +1}` replica signal using trailing, look-ahead-free features.

`replication/baselines.py`
: Defines naive baseline predictors: flat, majority, stratified random, and
persistence.

`replication/characterize.py`
: Computes signal diagnostics such as lead/lag, alpha type, regime behavior,
base-rate drift, and cross-asset structure.

`replication/gates.py`
: Implements G1-G4 validation gates.

`replication/ledger.py`
: Persists and renders the trial audit trail.

`replication/metrics.py`
: Implements classification and ordinal replication metrics.

`replication/nav.py`
: Computes cumulative log-NAV and NAV discrepancy metrics, including increment
correlation.

`replication/search.py`
: Implements train-only parameter search using Optuna TPE above the `n_eff`
floor and exhaustive grid below it.

`replication/splits.py`
: Defines chronological train/validation/test splits, embargo logic, and
effective sample size.

## Metamodel Module Map (Feature Engineering)

The `stml.metamodel` package builds the feature-engineering layer for the
triple-barrier metamodel. Every feature is computed at a non-zero-signal trade
day using only information `<= t`; fitted models are trained on the
feature-engineering train partition (FE-train, ending `2021-07-01`) and applied
causally. Engineered features prove causality by truncation-invariance; fitted
features prove it by a separate fit-provenance assertion.

`metamodel/features.py`
: Engineered, no-fit, trailing-only features (F1 counter-trend, F2
vol/dispersion, F5 signal-derived, F6 momentum-contrast, F7 microstructure, F8
calendar). Mirrors `replication.archetypes` score formulas and reuses
`na_checks` returns/vol.

`metamodel/regime_features.py`
: F3 regime posteriors. Fits a Gaussian mixture on `(return, vol)` and a 2-state
Markov-switching model on FE-train, then emits **filtered** (one-sided, causal)
per-day high-vol probabilities. Does not import `replication.characterize`
(that path is smoothed and signal-era-fit); it re-implements a causal path.

`metamodel/latent.py`
: F4 unsupervised structure. Fits a `StandardScaler`, `PCA(4)`, `KMeans`, and a
deterministic shallow dense autoencoder on the class-pooled FE-train block, then
transforms each instrument's series. Records autoencoder-vs-PCA(k=4)
reconstruction MSE per class.

`metamodel/xsection.py`
: F9 cross-sectional features: per-day rank over the finite-score subset,
universe size, and mean rolling pair-correlation to asset-class peers
(expected-negative).

`metamodel/macro_features.py`
: F11 cross-asset macro context (TF). Ingests the `data/additional_data.xlsx`
paired-column workbook, recovers each series' native cadence by a stamp grid
(daily / Friday-`W-FRI` / month-end-`ME`), applies per-class point-in-time
publication lags (daily = 0, EIA = +6 calendar days, PMI = +1 business day),
as-of-merges onto the trade dates, derives level + two momentum horizons per
series and per spread (12 series + 3 spreads = 45 columns), and FE-train-freezes
a z-score (`MacroBundle`). Broadcast globally to all 11 instruments.

`metamodel/scope.py`
: The D5 `InstrumentScope` registry: per-instrument fitting scope, `n_eff_gate`,
low-power flag, and embargo width. Persists `results/instrument_scope.json`.

`metamodel/catalog.py`
: The `FeatureSpec` registry documenting every produced column; renders
`reports/feature-catalog.md` and asserts 1:1 column coverage.

`metamodel/pipeline.py`
: `FeaturePipeline.fit(...).transform(...)` orchestrates the families into one
tidy-long matrix, attaches `partition` + `fe_train_end_date` provenance, and
restricts rows to non-zero-signal trade days.

`metamodel/build_features.py`
: CLI that loads data, runs the pipeline over all 11 instruments, and persists
the matrix, redundancy map, scope registry, provenance, and catalog.

## End-To-End Flow

The core flow is:

```text
load_clean_data
  -> chronological_split
  -> signal characterization
  -> train-only threshold calibration
  -> family parameter search on train
  -> validation gates G1-G4
  -> one final held-out test confirmation
  -> reports and results artifacts
```

The canonical split is chronological and never shuffled:

| split | date range | use |
|---|---|---|
| train | `2020-01-03` to `2021-07-01` | search objective and threshold calibration |
| validation | `2021-07-02` to `2021-12-30` | G1-G4 gates |
| test | `2021-12-31` to `2022-06-30` | final confirmation only |

The replication runner keeps OHLCV from `2019-01-01` onward for feature warm-up.
It does not use pre-2020 labelled target signals, because the released signal
labels begin in 2020.

## Key Concepts In Code

`target signal`
: Released primary signal for an instrument.

`replica signal`
: Candidate signal generated by a family from OHLCV.

`family` or `archetype`
: Rule template with a parameter grid and generation function.

`cell`
: Search/evaluation unit, usually an instrument. `pool:energy` is the special
pooled cell for low-power energy instruments.

`n_eff`
: Effective sample size, implemented in `replication/splits.py`. It counts
constant signal runs rather than daily rows.

`embargoed_val`
: Validation positions remaining after dropping rows at both validation edges
to reduce leakage from signal runs that straddle split boundaries.

`FLOOR`
: Minimum post-embargo validation `n_eff` for standalone treatment. Current
value is `10`.

`TPE`
: Optuna's Tree-structured Parzen Estimator sampler, used in
`replication/search.py` when `n_eff >= FLOOR`.

`budget`
: Number of TPE trials per `(family, cell)`. The default is `64`.

`grid`
: Exhaustive Cartesian product of a family's parameter axes. Used for low
`n_eff` cells and as the reference for search coverage.

## Parameter Search

`replication/search.py` selects parameters on train only.

The objective is:

```text
composite_skill = 0.5 * (kappa + ordinal_skill_vs_flat)
discrepancy = 1 - composite_skill
```

Lower discrepancy is better. Validation metrics are recorded for gates and
reports, but they do not choose the search winner.

Search tier:

```text
if post_embargo_n_eff >= 10:
    use seeded Optuna TPE with budget trials
else:
    use exhaustive deterministic grid
```

## Gate Logic

`replication/gates.py` evaluates selected candidates.

G1: beats baseline plus multiplicity
: Validation `kappa` and `ordinal_skill` must exceed train-only chance cutoffs
plus a margin that grows with the number of configurations tried.

G2: drift-aware generalization
: Validation skill must be positive and at least `0.5 * train_skill`, where
skill is measured against each split's own majority baseline.

G3: parameter plateau
: Validation composite skill is recomputed for `+/-1 step` neighbors along
numeric/ordinal parameter axes. The worst neighbor must clear the cutoff and the
neighbor spread must be below `0.15`.

G4: multi-metric consistency
: Validation `kappa`, `ordinal_skill`, and NAV increment correlation must all be
positive.

The final `passed` flag is the logical AND of G1, G2, G3, and G4.

## Metrics

`kappa`
: Chance-corrected agreement between target and replica.

`ordinal_skill`
: Chance-corrected skill that treats a full sign flip as worse than moving
between flat and active.

`MCC`
: Matthews correlation coefficient, used as an imbalance-resistant diagnostic.

`macro_f1`
: Mean F1 over classes present in the target signal.

`balanced_acc`
: Mean recall over target classes.

`NAV increment correlation`
: Correlation between target and replica per-day PnL increments under the
next-day return convention.

## Generated Artifacts

The replication code writes:

- `reports/signal-characterization.md`
- `results/jj/thresholds.json`
- `reports/<family>.md`
- `reports/replication-summary.md`
- `reports/replication-ledger.md`
- `results/jj/ledger.json`
- `results/jj/top_candidates.json`

The metamodel feature layer (`stml.metamodel.build_features`) additionally
writes:

- `results/feature_matrix.parquet` and `results/feature_matrix.csv`
- `data/macro_features_engineered.parquet` and `data/macro_features_engineered.csv`
  (the standalone F11 cross-asset macro dataset, row-aligned to the matrix)
- `results/feature_redundancy.json` and `results/feature_redundancy.csv`
- `results/instrument_scope.json`
- `results/feature_matrix_provenance.json`
- `reports/feature-catalog.md`

The README files in `reports/`, `results/`, and `results/jj/` explain how to
read those artifacts from a user perspective.
