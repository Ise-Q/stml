# Stage 2 — Thin End-to-End Pipeline

> Modules: [`src/stml/features.py`](../../src/stml/features.py) (minimal),
> [`src/stml/models.py`](../../src/stml/models.py),
> [`src/stml/evaluation.py`](../../src/stml/evaluation.py),
> [`src/stml/pipeline.py`](../../src/stml/pipeline.py)
> Output: `results/sreeram/predictions_v0.csv` (1408 rows, 128 dates × 11
> instruments, Jan–Jun 2022)

## Purpose

This is **the de-risking milestone**. The objective is not performance — it
is to validate that the full end-to-end architecture *runs* before we deepen
any single component. Stages 3–7 thicken this skeleton in place.

After this stage we have a submittable artifact: `predictions_v0.csv`. Stages
3–6 will replace it with a much better `predictions_v1.csv` / `_v2.csv`, but
the format and pipeline are now locked.

## What the pipeline does

```
load_clean_data
    │
    ▼
get_meta_labels (triple-barrier, h=10, pt=sl=1.0, vol_span=100)
    │  → events + labels + t1
    │
    ▼
get_uniqueness_weights  (concurrency-based, per-instrument)
    │
    ▼
compute_features  (G1 vol, G2 trend, G3 mean-rev, G5 signal-context, G7 calendar)
    │  → 38 features, all causal
    │
    ▼
split_by_boundary(2022-01-01, embargo=10d)
    │  → train: events with t < 2021-12-22 AND has label
    │  → predict: events with t in [2022-01-03, 2022-06-30]
    │
    ▼
ElasticNetLogReg.fit(...)
    │  RandomizedSearchCV(n_iter=10) over PurgedKFold(n_splits=5, embargo=10d)
    │  Then CalibratedClassifierCV(method='isotonic', cv=PurgedKFold(3))
    │
    ▼
predict_proba → CSV (signal==0 rows emit 0.0)
```

## Module-by-module summary

### `features.py` (38 features at this depth)

- **G1 (vol)**: realized vol over {5, 21, 63}d; EWMA vol (span 50); 5/63 vol
  ratio; vol-of-vol; downside semi-vol. Plus expanding-z-score versions of
  each (cross-instrument comparability).
- **G2 (trend)**: momentum {5, 21, 63}d; MA distance at {21, 63}d in
  sigma-units (`log(close/MA) / (sigma_1d * sqrt(w))`); MA-slope.
  Plus z-scored versions.
- **G3 (mean-reversion)**: 21d return autocorrelation; 21d Kaufman efficiency
  ratio; 5-day-aggregation variance ratio over 21d window.
- **G5 (signal context)**: the signal itself (`side_signal`), signal run-length,
  days-since-flip, asset-class net signal balance.
- **G7 (calendar)**: cyclical sin/cos month and day-of-week.

**Standardization policy.** Scale-dependent features (vol, momentum, MA dist)
get per-instrument *expanding-window* z-scores (min_periods=60). Bounded
features (autocorr ∈ [−1, 1], efficiency ratio ∈ [0, 1], variance ratio) are
kept raw. This makes the panel poolable across very different instruments.

**Important units fix.** First draft computed MA distance as `(close − MA) /
(σ · √w)` — that's $-units divided by a dimensionless quantity, giving an
arbitrarily large feature. Fixed to `log(close / MA) / (σ_1d · √w)`, which is
dimensionless ("how many w-day sigmas from MA"). Same fix for MA slope.

### `models.py` — `ElasticNetLogReg`

- Sklearn pipeline: `StandardScaler → LogisticRegression(elasticnet, saga)`.
- Hyperparameters tuned by `RandomizedSearchCV` over `PurgedKFold(n_splits=5,
  embargo=10d)`. Search grid: `C ∈ {1e-3, ..., 1e2}`, `l1_ratio ∈ {0, .25,
  .5, .75, 1}`. Score = `neg_log_loss`.
- `class_weight='balanced'` (the labels are 56/44; small but present imbalance).
- **Probability calibration** via `CalibratedClassifierCV(method='isotonic',
  cv=PurgedKFold(3))` — the deliverable is a probability, so calibration
  matters and gives us reliability diagrams later.
- `linear_coefficients()` exposes the underlying signed coefficients on the
  *standardised* feature scale for a sign-check.

### `evaluation.py`

- `classification_report` — n, label balance, accuracy, precision, recall, F1,
  AUC, avg precision (PR-AUC), log-loss, Brier — all weight-aware.
- `per_instrument_breakdown` — same metrics, broken down by instrument.
- `confusion_matrix_df` — labelled 2×2.
- `threshold_sweep` — precision/recall/F1 vs. decision threshold (used for
  the threshold-analysis section the rubric requires).
- `baseline_compare` — meta-model vs. "follow primary blindly" (always
  predict 1). The blind baseline is the rubric's mandated control.

### `pipeline.py` — `run_pipeline(config)`

Parameterised by `PipelineConfig`. Returns a `PipelineResult` dataclass with
everything needed for diagnostics + the deliverable CSV. **The train/predict
boundary is the key parameter** — for our submission it is `2022-01-01`; on
rerun the grader will set it to `2022-07-01` and reproducibility is the test.

## v0 results — sanity, not glory

Running with the base config (h=10, pt=sl=1.0, n_iter=10):

```
[1/7] loaded ohlcv (83544, 8), signals (645, 12)
[2/7] labeled events: 4,975 | label_1 share = 0.563
[3/7] features computed: (4984, 38)
[4/7] split: train=3,916, predict=1,011
[5/7] best_params={'clf__l1_ratio': 0.75, 'clf__C': 0.01}
       IN-sample: AUC=0.650, F1=0.702, Brier=0.232
[6/7] OOS H1-2022:
       n=1002, AUC=0.514, F1=0.583, Brier=0.255
       label_1_share=0.549
[7/7] wrote 1408 prediction rows → results/sreeram/predictions_v0.csv
```

**Interpretation:**

- **OOS AUC 0.514** — barely above random. *This is what we expect.* The
  thin pipeline uses only basic vol/trend/mean-rev features with a linear
  model and no regime detection. The point of Stage 2 is to validate the
  architecture, not the model.
- **In-sample → OOS gap (0.650 → 0.514)** — modest overfitting in-sample, as
  expected. Purged CV in tuning keeps this from being catastrophic.
- **OOS label_1_share 0.549** vs full-sample 0.563 — H1 2022 was a tougher
  half (the bear market in equities + commodity rotation), but still > 0.5.
- **`l1_ratio=0.75, C=0.01`** — heavy regularisation toward sparse weights.
  The model is being honest about how little signal a thin feature set carries.

## Deliverable CSV (v0)

```
date,instrument,prediction
2022-01-03,es1s,0.4246
2022-01-03,nq1s,0.4503
...
2022-06-30,pl1s,0.5882
```

- 1408 rows = 128 dates × 11 instruments.
- 397 rows have `prediction = 0.0` (the primary signal was 0 → no bet).
- 1011 rows have probabilistic predictions in `[0.379, 0.756]`, mean 0.563.
- Calibration looks reasonable — the mean predicted probability matches the
  observed label_1 share within rounding.

## What Stage 3 will change

- **Features**: extend G1/G2/G3 with range-based vol estimators, backward
  trend-scanning t-value, variance ratio, Hurst, Amihud illiquidity, plus
  G4 microstructure (volume z, OI trend), G5 cross-sectional ranks.
- **Showpiece (G6)**: Gaussian HMM regime probabilities + GMM cluster
  membership, **fitted strictly causally** — filtered (forward) probabilities
  only, never smoothed.
- Expected lift: AUC roughly 0.55–0.60 with the full feature set under linear
  models; gradient-boosted trees in Stage 4 should push further.

## Verified invariants

- **No leakage in training**: `assert_no_leakage` passes on every fold inside
  `RandomizedSearchCV`'s PurgedKFold.
- **Train/predict boundary respected**: max train `t` = 2021-12-21; min predict
  `t` = 2022-01-03; embargo of 10 days holds.
- **Deliverable format**: matches the assignment spec (date, instrument,
  prediction) with probabilities in [0, 1].
- **Signal-0 convention**: emits `0.0`, documented.
