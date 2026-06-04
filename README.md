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
uv sync --extra multitask        # multi-task neural net (optional)
uv sync --extra importance       # SHAP for cluster-level importance
```

### Option B — plain `pip`
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install hmmlearn pywavelets   # for the HMM + wavelet feature families
pip install "torch>=2.0,<2.3"     # only if running the multi-task NN
pip install shap                  # only if regenerating cluster importance
```

The notebook will execute with just the base install. The optional extras are
only needed to **regenerate** the cached results from raw inputs.

---

## 3. Reproduce the deliverable CSVs

Fastest path (reads the pipeline's cached per-event predictions; runs in seconds):

```bash
python scripts/build_submission_deliverables.py
```

Byte-deterministic re-emit verified: two consecutive runs produce identical
CSVs. For the hidden H2 2022 window:

```bash
python scripts/build_submission_deliverables.py \
    --start 2022-07-01 --end 2022-12-31
```

To regenerate the full pipeline from raw inputs (≈ 60 min on CPU):

```bash
python -m stml.experimental.make_labels         # triple-barrier meta-labels
python -m stml.experimental.bloomberg_ingest    # PIT-align macro panel
python -m stml.experimental.make_features       # 18 feature families
python -m stml.experimental.make_scope          # per-instrument embargo
python -m stml.experimental.make_baseline       # per-class baseline AUCs
python -m stml.experimental.make_champions      # CPCV(6,2) + 1-SE champion per instrument
python -m stml.experimental.make_importance     # cluster MDA + MDI + SHAP
python -m stml.experimental.make_deliverables   # writes outputs/*.csv
python -m stml.experimental.make_significance   # PSR / MinTRL / DSR / PT
```

---

## 4. H2 2022 rerun

The brief states the held-out H2 2022 window is the hidden test set.

1. Replace `data/ohlcv_data.csv` and `data/primary_signals.csv` with versions
   extended through Dec 2022.
2. Two macro inputs for the hidden window are shipped:
   - `data/features/f11_macro_context_oos.csv` — z-scored daily macro features
     (45 columns, Jul–Dec 2022, all 11 instruments)
   - `data/OOS_additional_data.xlsx` — raw H2 2022 macro source (22 series)
3. Re-run the pipeline. The shipped model was selected for robustness to
   missing Bloomberg-augmented features at inference time; the per-class
   AUC delta with vs. without Bloomberg is documented in
   `results/bbg_missingness_ablation.csv`.
4. Re-emit the deliverables:
   ```bash
   python scripts/build_submission_deliverables.py \
       --start 2022-07-01 --end 2022-12-31
   ```

---

## 5. Brief coverage

| Brief section | Marks | Notebook section | Code | Result artefacts |
|---|---:|---|---|---|
| Feature Engineering | 20 | §1 | `src/stml/experimental/features/`, `make_features.py` | 105 features over 18 families |
| Triple-Barrier Labeling | 20 | §2 | `data/triple_barrier_labels.csv`, `make_labels.py` | Per-instrument geometry summary |
| Model Development & Comparison | 30 | §3 | `champion_pipeline.py`, `make_champions.py`, `models.py` | `champions_summary.csv`, `champions_per_pool_per_model.csv` |
| Cluster-Level Feature Importance | 10 | §4 | `make_importance_deep.py` | `results/importance/{equity,energy,metals}/*` |
| Model Evaluation | 20 | §5 | `evaluation.py`, `make_baseline.py` | `baseline_per_instrument.csv`, `threshold_summary.csv` |
| Strategy Construction (+10 bonus) | 10 | §6 | `sizing.py`, `backtest.py`, `make_deliverables.py` | `backtest_metrics.csv`, `strategy_variant_comparison.csv`, `significance_summary.csv` |

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
├── features/
│   └── f11_macro_context_oos.csv     H2 2022 macro features (Jul–Dec)
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
    ├── features/               18 feature families
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
