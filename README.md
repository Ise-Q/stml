# BUSI70575 — Meta-Model on a Primary Trading Signal

A meta-model that takes the supplied primary signal `s ∈ {−1, 0, +1}` for
**11 futures across three asset classes** and predicts, for each non-zero
signal, the **probability that following the trade is profitable** under a
triple-barrier exit. The optional strategy-construction track converts the
calibrated probabilities into vol-targeted position weights.

The submission is organised so that the narrative + charts live in a single
notebook (`submission.ipynb`) and the two deliverable CSVs live under
`outputs/`. Every numerical claim made in the notebook reproduces from the
CSV artefacts under `results/submission/`.

---

## 1. Start here

```
submission.ipynb                     ← run top-to-bottom; one section per brief section
outputs/metamodel_predictions.csv    ← REQUIRED deliverable (H1 2022)
outputs/strategy_weights.csv         ← Optional (bonus track) deliverable (H1 2022)
```

Both CSVs are in the brief's exact format:

| File | Schema | Rows |
|---|---|---|
| `metamodel_predictions.csv` | `date,instrument,prediction` | 1,408 = 128 trading days × 11 instruments |
| `strategy_weights.csv`      | `date,instrument,weight`     | 1,408 |

Convention: `prediction = 0.0` on signal-zero days (no trade taken). Unresolved
barrier events at the window edge fill `0.5` (model abstains). `weight = 0.0`
on signal-zero days. Predictions are in `[0, 1]`; weights are vol-targeted to
`σ_tgt = 10 %` with a defensive hard clamp at `±10`.

---

## 2. Install

Tested on **Python 3.10, 3.11, and 3.12** (Linux + macOS).

### Option A — `uv` (recommended; reads `uv.lock`)
```bash
uv sync                          # base
uv sync --group features-extra   # HMM regimes (F17) + wavelet (F13)
uv sync --extra multitask        # multi-task neural net + NN-strategy track
uv sync --extra importance       # SHAP for cluster-level importance
```

### Option B — plain `pip`
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install hmmlearn pywavelets   # for the HMM + wavelet feature families
pip install "torch>=2.0,<2.3"     # required to retrain the multi-task NN (§3)
pip install shap                  # only if regenerating cluster importance
```

The notebook executes with the base install (it reads cached artefacts; no
torch needed to render charts). The `multitask` extra (or `pip install torch`)
is required to **retrain** the multi-task NN family and re-run the NN-family
column of the comparison table; without it the cached NN AUCs are still
displayed but the family-comparison run will skip the NN evaluator.

---

## 3. Reproduce the H1 2022 deliverable CSVs

> H1 2022 is the released sealed-test window the model was scored on. For the
> hidden H2 2022 window, jump to §4 — it has additional steps before the
> pipeline can be re-run.

**Fastest path (seconds, reads the cached per-event predictions):**

```bash
python scripts/build_submission_deliverables.py
```

This is a deterministic CSV stitcher: it reads the pipeline's per-event
prediction cache (`results/submission/oos_events_with_predictions.csv`) and
emits the full `(date × instrument)` grid for the H1 2022 window
(`outputs/metamodel_predictions.csv`, `outputs/strategy_weights.csv`). Two
consecutive runs produce identical bytes (md5-verified). It does **not** run
model inference, so it cannot score a window the pipeline has not already
scored — use the full-pipeline path below to change the inference window.

**Full pipeline from raw inputs (≈ 60 min on CPU):**

```bash
python -m stml.experimental.make_labels          # loads triple_barrier_labels.csv → events.parquet
python -m stml.experimental.make_features        # 16 feature families → features.parquet
python -m stml.experimental.make_scope           # per-instrument purge/embargo map
python -m stml.experimental.make_baseline        # per-class baseline AUCs
python -m stml.experimental.make_champions       # CPCV(6,2) + 1-SE champion per instrument
python -m stml.experimental.make_importance      # cluster MDA + MDI + SHAP
python -m stml.experimental.make_importance_deep # SHAP + within-cluster recursion
python -m stml.experimental.make_deliverables    # refits champion on train+val, scores test partition → outputs/*.csv
python -m stml.experimental.make_significance    # PSR / MinTRL / DSR / PT
```

> The PIT-aligned Bloomberg parquets the pipeline consumes ship pre-built
> under `data/bloomberg/cleaned/` (committed). The raw vendor pulls that
> produced them are gitignored, so `stml.experimental.bloomberg_ingest` is
> **not** part of the reproduction chain on a clean clone — it is only used
> internally when refreshing the cleaned parquets from a fresh Bloomberg
> pull.

---

## 4. H2 2022 rerun (hidden test window)

The brief states the held-out H2 2022 window is the hidden test set. The
submission ships **the H2 2022 supporting data the model needs** (Bloomberg
macro + EIA via `data/OOS_additional_data.xlsx`) and **a labels-regeneration
script** that re-creates the triple-barrier labels for the extended window
using each instrument's locked geometry. The full procedure runs end-to-end
once the OHLCV and primary-signal CSVs are swapped.

### 4a. Required input files (replace these two)

The grader supplies two input CSVs extended through Dec 2022. Drop them in at
the same paths and in the **same long/wide schemas** as the released files:

| File | Schema | Notes |
|---|---|---|
| `data/ohlcv_data.csv` | Long: `date, instrument, open, high, low, close, volume, open_interest` | The released file covers 1990-01-02 → 2022-06-30 in long format. Extend it with H2 2022 rows for all 11 instruments. `volume` and `open_interest` are used by F7 microstructure features — leave them populated if available. |
| `data/primary_signals.csv` | Wide: `date, es1s, nq1s, fesx1s, cl1s, ho1s, rb1s, ng1s, gc1s, si1s, hg1s, pl1s` (one column per instrument; values `∈ {−1, 0, +1}`) | Extend with H2 2022 rows. Order of columns is fixed by the released file. |

**Everything else stays where it is.** The triple-barrier labels CSV, the
cleaned Bloomberg parquets, and the Bloomberg OOS workbook all already ship
in this repo or get regenerated by the steps below.

### 4b. Step-by-step procedure

The whole sequence runs in ≈ 60 min on a single CPU core. The labels and
Bloomberg-extension scripts are one-shot and idempotent — they detect when
H2 data is already in place and exit early.

**Step 1 — Extend the cleaned Bloomberg panels (~1 second):**

```bash
python scripts/extend_bloomberg_for_h2.py
```

Reads `data/OOS_additional_data.xlsx` and appends Jul–Dec 2022 rows to the
cleaned Bloomberg parquets the model consumes:

| Bloomberg family | Used by model? | Source for H2 2022 |
|---|---|---|
| Macro (21 series — VIX, MOVE, DXY, UST/Bund/TIPS yields, OAS, PMIs, etc.) | Yes (**F11**) | `OOS_additional_data.xlsx` (real values) |
| EIA weekly crude change | Yes (**F22**) | Derived from `EIA_CRUDE_STOCK` in the OOS workbook |
| EIA release-day binary flag | Yes (**F22**) | Wednesdays in the H2 calendar |
| Futures term structure (F18) | **No — dropped on parsimony grounds** (§4 of notebook) | n/a |
| Options implied vol (F19) | **No — dropped on parsimony grounds** (§4 of notebook) | n/a |

**Step 2 — Regenerate the triple-barrier labels including H2 events
(~5 seconds):**

```bash
python scripts/relabel_for_h2.py
```

The shipped `data/triple_barrier_labels.csv` covers only the released window
because the per-instrument barrier-geometry grid search ran on H1 only. This
script keeps the released-window rows byte-identical and **appends new
`partition = "test"` rows for every H2 2022 non-zero-signal event**, using
each instrument's locked `(pt, sl, h)` and the same σ definition as the
shipped CSV (verified: 50/50 round-trip match against shipped H1 test rows;
σ matches to 4 decimal places). Idempotent — running it again is a no-op.

**Step 3 — Re-run the full pipeline from §3** (uses the same commands; the
prediction window is driven by the `partition = "test"` rows in the labels
CSV, which now includes H2):

```bash
python -m stml.experimental.make_labels
python -m stml.experimental.make_features
python -m stml.experimental.make_scope
python -m stml.experimental.make_baseline
python -m stml.experimental.make_champions
python -m stml.experimental.make_importance
python -m stml.experimental.make_importance_deep
python -m stml.experimental.make_deliverables
python -m stml.experimental.make_significance
```

`make_features` extends the feature matrix to H2 automatically because the
input panels (OHLCV + Bloomberg) now run through Dec 2022. `make_deliverables`
refits each instrument's champion on the `train + val` partitions (still H1
2020 → H2 2021 — the H2 2022 window is sealed from fitting) and predicts on
every event with `partition = "test"`, which after step 2 includes both H1
and H2 2022.

**Step 4 — Slice the H2 portion of the deliverable** (only if the H1 portion
is not also wanted in the same file):

```bash
python scripts/build_submission_deliverables.py \
    --start 2022-07-01 --end 2022-12-31
```

This re-emits the two output CSVs against the cached per-event predictions
restricted to the H2 grid. The H1 output produced by step 3's
`make_deliverables` is left untouched.

### 4c. What changes from the H1 run, and what doesn't

* **What changes.** OHLCV + signals, cleaned Bloomberg parquets, triple-barrier
  labels CSV (H1 portion untouched, H2 rows appended), features parquet,
  feature drift audit, per-event prediction cache, the two output CSVs.
* **What doesn't change.** Champion selection (CPCV runs on `train + val`,
  which is purely H1 / earlier); per-instrument calibration (Platt fits on
  CPCV OOF, again `train + val` only); the bootstrap `p*` thresholds; the
  per-instrument geometry `(pt, sl, h)`; the sizing-policy parameters
  `(a, c)`; the cost-model assumptions; the random seed.

### 4d. Sanity-checks for the H2 rerun

After step 3 completes, expect:

* `wc -l outputs/metamodel_predictions.csv` ≈ 1,408 + (`H2 trading days` × 11)
  rows + 1 header (H1 stays; H2 appends).
* `data/triple_barrier_labels.csv` partition counts should show `test` larger
  than the original 951 (it picks up the H2 events).
* `results/submission/oos_events_with_predictions.csv` has the new H2 events
  with non-`0.5` predictions wherever the model has signal coverage.

---

## 5. Section map — how the brief's requirements are met

Each row maps a brief requirement to where it is implemented, the code that
produces it, and the cached result artefact a reader can open directly.

| Brief section | What the brief asks for | Notebook section | Code | Result artefact (cached, reproducible) |
|---|---|---|---|---|
| Feature Engineering | A documented feature set across diverse families | §1 | `src/stml/experimental/features/`, `make_features.py` | 94 registered features across 16 families; engineered (E) vs fitted-on-train (TF) split rendered live by §1 chart |
| Triple-Barrier Labeling | Triple-barrier meta-labels with documented geometry | §2 + §2.1 | `data/triple_barrier_labels.csv`, `make_labels.py` | Per-instrument `(pt, sl, h)` summary; **theoretical-ceiling (oracle) PnL** raw-vs-adjusted chart per instrument under each instrument's chosen geometry |
| Model Development & Comparison — **including neural networks**, **with hyperparameter tuning**, with a clear comparison and winner | **Five families** (Elastic-net Logistic, Random Forest, LightGBM, XGBoost, **Multi-task Neural Network**) evaluated under CPCV(6, 2) + 1-SE rule | §3 | `champion_pipeline.py`, `make_champions.py`, `models.py`, `multitask.py`, `scripts/tune_multitask_nn.py` | `champions_summary.csv` (1-SE winner per instrument), `champions_per_pool_per_model.csv` (full candidate matrix), `multitask_nn_tuning.csv` (4-config NN grid result on `equity_all`) |
| Cluster-Level Feature Importance | Per-asset-class cluster MDA + MDI + SHAP with significance + rank agreement | §4 | `make_importance.py`, `make_importance_deep.py` | `results/submission/importance/{equity,energy,metals}/clustered_importance.csv`, `deep_summary.csv`, `pruned_vs_full_auc.csv` |
| Model Evaluation | Per-instrument classification table + dual confusion matrix + calibration | §5 | `evaluation.py`, `make_baseline.py`, `calibration.py` | `baseline_per_instrument.csv`, `threshold_summary.csv`, calibration reliability diagram in §5 |
| Strategy Construction (bonus track) | Vol-targeted strategy with cost model + significance battery | §6 | `sizing.py`, `backtest.py`, `make_deliverables.py`, `significance.py` | `backtest_metrics.csv`, `strategy_variant_comparison.csv`, `significance_summary.csv`, `deflation_ladder.csv` |

> **Neural-network coverage (brief §3) — explicit note.** The multi-task NN
> (MLP with per-instrument heads on a shared encoder) is implemented, trained
> with early stopping on a 20 % inner-val slice, evaluated under the *same*
> CPCV(6, 2) protocol as the linear and tree families, and appears in the
> family comparison in §3. A 4-config hyperparameter grid
> (`hidden_widths`, `dropout`, `lr`) is run on the `equity_all` pool by
> `scripts/tune_multitask_nn.py` and persisted to
> `results/submission/multitask_nn_tuning.csv`. The CPCV champion picks the
> NN for `cl1s` and `si1s`; the shipped deliverable substitutes the 1-SE
> `elasticnet_logistic` alternative on those names (logged at deliverable
> build time) so the surface is sklearn-only. The NN is therefore **fully
> evidenced as a candidate family in the comparison**, with a documented
> deployment-time substitution rationale — the brief's NN requirement is
> about *development and comparison*, not about which model ships.

---

## 6. Repository layout

```
submission.ipynb                Notebook — one section per brief section, runs top-to-bottom
README.md                       This file
scripts/
└── build_submission_deliverables.py   Deterministic CSV emitter

data/
├── ohlcv_data.csv              Raw OHLCV (released window; replace for H2 2022)
├── primary_signals.csv         Raw primary signals (replace for H2 2022)
├── triple_barrier_labels.csv   Per-instrument-geometry meta-labels (canonical)
├── OOS_additional_data.xlsx          Raw H2 2022 macro source
├── bloomberg/cleaned/                PIT-aligned cleaned BBG parquets (released window)
└── meta/                              Generated missingness diagnostics

outputs/
├── metamodel_predictions.csv   REQUIRED deliverable (H1 2022, 1,408 rows)
└── strategy_weights.csv        Optional deliverable (H1 2022, 1,408 rows)

src/stml/                       Pipeline package
├── io.py                       load_data / load_clean_data / load_returns_panel
├── na_checks.py                Calendar + missing-data diagnostics
├── metamodel/                  Shared feature-engineering base (F1–F17 catalogue)
└── experimental/               Modelling pipeline (this submission)
    ├── make_*.py               Stage runners (S1 labels → S7 significance)
    ├── champion_pipeline.py    CPCV(6,2) + 1-SE champion selector
    ├── sizing.py               Six position-sizing methods (Madmoun S3 slides 33–34)
    ├── backtest.py             Barrier-exact backtest + cost model
    ├── significance.py         PSR / MinTRL / DSR / PT
    └── calibration.py          Platt + isotonic per-instrument calibration

results/                        All numerical evidence cited in the notebook
├── backtest_metrics.csv        Strategy headline metrics
├── champions_summary.csv       Per-instrument champion (CPCV AUC + 1-SE lower CI)
├── baseline_per_instrument.csv Per-instrument classification metrics
├── threshold_summary.csv       Per-instrument bootstrap p* thresholds
├── significance_summary.csv    t-stat / bootstrap CI / PSR / MinTRL
├── deflation_ladder.csv        DSR at multiple N_eff values
├── strategy_variant_comparison.csv   Sizing variants head-to-head
├── importance/{equity,energy,metals}/  Per-class cluster MDA + SHAP
└── bbg_missingness_ablation.csv      AUC with vs. without Bloomberg features

tests/experimental/             150+ tests covering leakage, determinism, schema
```

---

## 7. Reproducibility contract

- **Random seed.** All randomness gated on `random_state=42`. Verified by
  `tests/experimental/test_scaffold.py`.
- **Byte-deterministic emit.** Two consecutive runs of
  `build_submission_deliverables.py` produce identical bytes. Verified by
  `tests/experimental/test_s6.py`.
- **Causal feature contract.** Every E-class feature is causal by
  truncation-invariance; every TF-class fit lives on the FE-train block only
  (`date ≤ 2021-07-01`) and is applied with frozen parameters. Verified by
  `tests/experimental/test_features.py` and
  `tests/experimental/test_methodology_guards.py`.
- **Partition discipline.** The labels CSV ships with an authoritative
  `partition` column (`train` 60 % / `val` 21 % / `test` 19 %); the pipeline
  honours it everywhere. The H1 2022 deliverable is the `test` partition.
- **Determinism on H2 2022 rerun.** The H2 2022 macro context CSV is shipped
  pre-computed; the pipeline does not need to re-fit anything at rerun time
  beyond predicting on the extended axis.
