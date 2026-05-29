# Source Code Guide

This directory contains the implementation for **data loading** and the
**feature-engineering layer**. The package is `stml`. There is no modelling or
signal-replication code here — this branch is the shared feature base everyone
branches off from.

## Main Entry Points

`stml.io`
: Loads raw and cleaned project data — `load_data()`, `load_clean_data()`,
`load_returns_panel()`. The single sanctioned way to load the panel.

`stml.na_checks`
: Cleans and diagnoses the futures OHLCV panel (calendars, holiday + missing-data
handling, returns/vol/correlation helpers).

`stml.metamodel.build_features`
: Builds the leakage-safe, standardized feature matrix and writes
`results/feature_matrix.{parquet,csv}`, one CSV per family under
`data/features/`, `results/feature_redundancy.{json,csv}`,
`results/instrument_scope.json`, `results/feature_matrix_provenance.json`,
and `reports/feature-catalog.md`.

## Feature layer — contract

Every feature is computed at a `(date, instrument)` non-zero-signal trade day
using only information `<= t`. Two leakage classes:

- **E (engineered)** — no fit; causal by *truncation-invariance* (value at `t`
  identical on `data[:t+1]` and `data[:T]`).
- **TF (fitted)** — fit on the FE-train partition only (ending `2021-07-01`) and
  applied causally with frozen parameters (F3 regimes, F4 latent, F11 macro
  z-scorer, F16 drift discriminator, F17 HMM).

Standardization: every scale-dependent E-class column also gets a `z_<col>` twin
— a per-instrument causal expanding-window z-score (split-agnostic). The matrix
carries **no label column**; downstream branches attach their own.

## Metamodel Module Map

`metamodel/features.py`
: Core engineered E-class families (F1 counter-trend, F2 vol/dispersion, F5
signal-derived, F6 momentum-contrast, F7 microstructure, F8 calendar, F10
price-action). Trailing-only; reuses `na_checks` returns/vol.

`metamodel/features_ext.py`
: Extended E-class families folded in from teammate branches — F2 Rogers-Satchell
vol, F5 signal entropy/flip-rate, F7 Roll spread / Kyle's λ / overnight gap, **F12**
mean-reversion / path-structure & trend-quality (Hurst, variance ratio,
efficiency ratio, autocorrelation, trend t-values, MA slope), **F13** wavelet
multiscale energy, **F15** conditional-risk / first-passage (seeded bootstrap).
Also holds `expanding_zscore`, the `Z_TWIN_COLUMNS` set, and `add_z_twins`.

`metamodel/regime_features.py`
: F3 regime posteriors. Fits a Gaussian mixture on `(return, vol)` and a 2-state
Markov-switching model on FE-train, then emits **filtered** (one-sided, causal)
high-vol probabilities.

`metamodel/regime_features_hmm.py`
: **F17** HMM regime posteriors. Fits a 3-state Gaussian HMM on FE-train
`(ret, vol)` and emits filtered (forward-only, causal) lo/mid/hi state posteriors
+ argmax. Requires `hmmlearn` (`uv sync --group features-extra`).

`metamodel/drift_features.py`
: **F16** concept-drift / regime-alignment. A rolling logistic discriminator of
FE-train-era vs recent feature rows; `P(today looks recent)` is the score.

`metamodel/latent.py`
: F4 unsupervised structure. `StandardScaler` + `PCA(4)` + `KMeans` + a
deterministic shallow autoencoder on the class-pooled FE-train block.

`metamodel/xsection.py`
: F9 cross-sectional + cross-asset positioning: per-day rank, universe size, mean
rolling pair-correlation to peers, plus the lead-lag centroid distance,
within-class dispersion z, and EWMA implied-correlation z (from Harry).

`metamodel/macro_features.py`
: F11 cross-asset macro context (TF). Ingests `data/additional_data.xlsx`, applies
per-class point-in-time publication lags, derives level + momentum columns
(12 series + 3 spreads = 45 cols), and FE-train-freezes a z-score.

`metamodel/scope.py`
: The per-instrument `InstrumentScope` registry (fitting scope, `n_eff_gate`,
low-power flag, embargo width). Persists `results/instrument_scope.json`.

`metamodel/splits.py`
: Chronological train/val/test split + embargo + `n_eff` helpers. Vendored from
the (removed) replication layer so the feature layer is self-contained.

`metamodel/catalog.py`
: The `FeatureSpec` registry documenting every produced column (incl. z-twins);
renders `reports/feature-catalog.md` and asserts exact 1:1 column coverage.

`metamodel/pipeline.py`
: `FeaturePipeline.fit(...).transform(...)` orchestrates every family into one
tidy-long matrix, attaches `partition` + `fe_train_end_date` provenance, and
restricts rows to non-zero-signal trade days.

`metamodel/build_features.py`
: CLI that loads data, runs the pipeline over all 11 instruments, and persists the
matrix, per-family CSVs, redundancy map, scope registry, provenance, and catalog.

## End-To-End Flow

```text
load_clean_data
  -> FeaturePipeline.fit   (fit TF families on FE-train only, ending 2021-07-01)
  -> FeaturePipeline.transform   (apply causally; assemble tidy-long matrix)
  -> assert_coverage   (catalog 1:1 guard)
  -> persist matrix + per-family CSVs + redundancy + scope + provenance + catalog
```

The canonical chronological split (`partition` column), never shuffled:

| split | date range | use |
|---|---|---|
| train | `2020-01-03` to `2021-07-01` | TF-family fit window (FE-train) |
| val   | `2021-07-02` to `2021-12-30` | downstream validation |
| test  | `2021-12-31` to `2022-06-30` | downstream final confirmation |

OHLCV from before 2020 is kept only for feature warm-up; the matrix rows are the
non-zero-signal released trade-days (2020 onward).

## Generated Artifacts

`stml.metamodel.build_features` writes:

- `results/feature_matrix.parquet` and `results/feature_matrix.csv`
- `data/features/<family>.csv` (one CSV per family, keyed by `(date, instrument)`;
  the F11 macro dataset is `f11_macro_context.csv`)
- `results/feature_redundancy.json` and `.csv`
- `results/instrument_scope.json`
- `results/feature_matrix_provenance.json`
- `reports/feature-catalog.md`

The README files in `reports/` and `results/` explain how to read those artifacts.
