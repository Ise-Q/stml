# Results Artifacts

This directory holds the machine-readable artifacts of the shared
**feature-engineering base**, produced by `stml.metamodel.build_features`:

- **root-level files** — the feature matrix and its companion redundancy /
  scope / provenance maps;
- `<initials>/` — personal figures / tables / intermediate outputs (created on
  your own branch, not shared until a checkpoint).

The standardized feature data split **one CSV per family** lives under
[`../data/features/`](../data/features) (keyed by `(date, instrument)`), next to
the raw inputs.

The graded per-column documentation lives in
[`reports/feature-catalog.md`](../reports/feature-catalog.md).

## How to regenerate

```bash
uv sync --group features-extra
uv run python -m stml.metamodel.build_features                      # all 11 instruments
uv run python -m stml.metamodel.build_features --instruments si1s   # smoke subset
```

The build is deterministic: rebuilding reproduces the matrix frame-for-frame
(values, dtypes, and column order), with the autoencoder-derived columns equal to
within `1e-10`.

## Files

`feature_matrix.parquet` / `feature_matrix.csv`
: The tidy-long feature matrix. Parquet is canonical; CSV is the convenience
copy. One row per **non-zero-signal trade day** per instrument over the released
window (`2020-01-03` to `2022-06-30`): **4,984 rows** across the 11 instruments,
**179 columns** = 4 metadata + **175 feature columns**. There is **no label
column** — each downstream branch attaches its own.

`data/features/<family>.csv` (under `data/`, not here)
: The matrix split by feature family — `f1_counter_trend.csv` …
`f17_hmm_regimes.csv`. Each is keyed by `(date, instrument)` and carries that
family's raw columns plus their `z_` twins. The union of the family files'
feature columns equals the master matrix's feature columns. The **F11
cross-asset macro** dataset is `data/features/f11_macro_context.csv` (the 45
`f11_*` columns; no separate standalone macro export).

`feature_redundancy.json` / `feature_redundancy.csv`
: The feature redundancy map: a pairwise-complete correlation matrix
(`na_checks.corr_max_info`) over the 175 feature columns, a SciPy hierarchical
clustering of `1 - |corr|`, and per feature its highest-`|corr|` partner. Use it
to spot near-duplicate features (raw columns and their `z_` twins cluster
together by construction, as do the globally-broadcast F11 macro columns).

`instrument_scope.json`
: The per-instrument `InstrumentScope` registry — fitting-scope decisions and
power statistics. See below.

`feature_matrix_provenance.json`
: The FE-train boundary, split boundaries, and a summary of every fitted
artifact's training window. See below.

## feature_matrix layout

Metadata columns (always first): `date`, `instrument`, `partition`
(`train`/`val`/`test` from the chronological split of the released signal
calendar; train ends `2021-07-01`), and `fe_train_end_date` (constant
`2021-07-01`, also in `df.attrs`).

The 175 feature columns are grouped by family (full per-column documentation in
[`reports/feature-catalog.md`](../reports/feature-catalog.md)):

| Family | What it captures | Provenance |
|---|---|---|
| F1 counter-trend | distance-from-MA z-scores (C1 highest-value), RSI, Bollinger | signal-deep-dive |
| F2 vol / dispersion | rolling / Parkinson / Garman-Klass / **Rogers-Satchell** vol, ATR, skew/kurtosis | sdd + Sreeram |
| F3 regime | filtered GMM + Markov high-vol posteriors (TF) | signal-deep-dive |
| F4 latent | PCA(4), KMeans, dense-autoencoder code + recon error (TF) | signal-deep-dive |
| F5 signal-derived | trailing run-length / days-since-flip, participation, **entropy / flip-rate** | sdd + Harry |
| F6 momentum-contrast | vol-scaled momentum, MA-cross / MACD, ADX, Donchian | signal-deep-dive |
| F7 microstructure | volume/OI z & trend, Amihud, **Roll spread / Kyle's λ / overnight gap** | sdd + Harry |
| F8 calendar | day-of-week and month sin/cos | signal-deep-dive |
| F9 cross-section | rank, universe size, pair-corr, **lead-lag / dispersion-z / implied-corr-z** | sdd + Harry |
| F10 OHLC price-action | high-low log range, open-to-open return | signal-deep-dive |
| F11 macro context | PIT-lagged + FE-train-z-scored cross-asset macro (TF, global broadcast) | signal-deep-dive |
| F12 path-structure | Hurst, variance ratio, efficiency ratio, autocorr, trend t-values, MA slope | Sreeram |
| F13 wavelet | multiscale (db4) MRA energy fractions d1–d5 | Harry |
| F15 conditional risk | bootstrap expected hit time, timeout prob, tortuosity, semi-vol ratio | Harry |
| F16 concept-drift | rolling FE-train-vs-recent discriminator alignment score (TF) | Harry |
| F17 HMM regime | filtered 3-state Gaussian-HMM lo/mid/hi posteriors + argmax (TF) | Sreeram |

Leakage-class totals: **E = 108, TF = 65, LI = 2** (`f2_vol_20`,
`f5_trailing_run_length`). **E** = engineered, proven causal by
truncation-invariance; **TF** = fitted on FE-train only, proven causal by a
fit-provenance assertion; **LI** = the label-interface subset earmarked for a
downstream triple-barrier label.

**Standardization.** Every scale-dependent E-class column has a `z_<col>` twin: a
per-instrument causal expanding-window z-score (`expanding(min_periods=60)`),
split-agnostic. Bounded / already-normalized columns get no twin.

## instrument_scope.json

A JSON object keyed by ticker. Each entry records:

`asset_class`
: `EQ` {es1s, nq1s, fesx1s} / `EN` {cl1s, ho1s, rb1s, ng1s} / `ME` {gc1s, si1s, hg1s, pl1s}.

`fit_scope_regime`
: Always `per_instrument` — regime models fit on the instrument's own dense
daily-return series.

`fit_scope_latent`
: Always `pooled_within_class` — latent models fit on the asset class's pooled
FE-train engineered rows.

`n_eff_gate`
: Post-embargo validation effective sample size (number of constant-signal runs),
the reporting / low-power statistic.

`low_power`
: `True` when `n_eff_gate < FLOOR` (10): `cl1s`, `ho1s`, `ng1s`.

`embargo_p90`
: The full-period 90th-percentile constant-signal run length — the purge/embargo a
downstream cross-validation should apply at each split boundary.

## feature_matrix_provenance.json

The structured handoff record:

```text
fe_train_end_date  : 2021-07-01
split boundaries   : train / val / test date ranges of the released window
regime artifacts   : per-instrument train_index summary (min/max date + count)
latent artifacts   : per-class train_index count
macro artifacts    : F11 train_index bounds, per-class publication-lag config,
                     n_macro_features (45), kept-series list
matrix shape       : n rows, n feature columns
seed               : the RNG seed used for the deterministic build
```

It lets a downstream model verify mechanically that every fitted feature was
trained only on data `≤ 2021-07-01`.

## How these files relate to the reports and source

- [`reports/feature-catalog.md`](../reports/feature-catalog.md) documents every
  feature column and records the autoencoder-vs-PCA(k=4) reconstruction comparison.
- [`src/README.md`](../src/README.md) describes the `stml.metamodel` package that
  produces these artifacts.
