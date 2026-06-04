# BUSI70575 Submission — Sreeram team (`Sreeram_experimental`)

A meta-model on top of the released primary trading signal for 11 futures across
Equity / Energy / Metals. Per the brief: triple-barrier meta-labels, CPCV(6, 2)
champion selection with the 1-SE rule, cluster-level feature importance, and an
optional vol-targeted strategy on top of the calibrated probabilities.

The deliverable CSVs are shipped in `outputs/`; the modelling pipeline is in
`src/stml/experimental/`; the per-rubric write-up is in `reports/sreeram_experimental/`.

---

## 1. Deliverables (where the grader looks first)

```
outputs/metamodel_predictions.csv   ← Required deliverable. H1 2022, full grid.
outputs/strategy_weights.csv        ← Optional bonus track. H1 2022, full grid.
```

Both files are in the brief's exact format:

| File | Columns | Rows | Notes |
|---|---|---|---|
| `metamodel_predictions.csv` | `date, instrument, prediction` | **1,408** = 128 trading days × 11 instruments | `prediction ∈ [0, 1]`. `signal == 0` rows are emitted with `prediction = 0.0` (no trade taken). Unresolved barrier events at the window edge are filled with `0.5` (model abstains). |
| `strategy_weights.csv` | `date, instrument, weight` | 1,408 | Vol-targeted per Madmoun §40 (`σ_tgt = 10 %`). `weight = 0.0` on signal-zero days. |

To re-emit both files from the pipeline's cached events:

```bash
uv run python scripts/build_submission_deliverables.py
# H2 2022 rerun:
uv run python scripts/build_submission_deliverables.py --start 2022-07-01 --end 2022-12-31
```

Byte-deterministic (run twice → identical bytes). The script reads:
- `data/primary_signals.csv` — for the (date, instrument) grid
- `results/sreeram_experimental/oos_events_with_predictions.csv` — for the model's calibrated probabilities and vol-targeted weights

---

## 2. Install

We tested on **Python 3.10, 3.11, and 3.12** (Linux + macOS). The bleeding-edge
pins that were in the original `pyproject.toml` have been relaxed so a grader
on a stable distro can install in one command.

### Option A — `uv` (recommended; reads `uv.lock`)
```bash
uv sync                          # base
uv sync --group features-extra   # adds hmmlearn + pywavelets for F13/F17
uv sync --extra multitask        # adds torch for multi-task NN (S4)
uv sync --extra importance       # adds shap for cluster importance (S5)
```

### Option B — `pip` (fallback; no uv)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install hmmlearn pywavelets   # for F13 + F17
pip install "torch>=2.0,<2.3"     # only if running the multi-task NN
pip install shap                  # only if regenerating cluster importance
```

---

## 3. End-to-end run

The submission is a **set of Python files** (per the brief's "notebook or set of
Python files" clause). Stages are runnable as `python -m stml.experimental.<module>`:

```bash
# Stage 1 — labels (triple-barrier with per-instrument geometry from data/triple_barrier_labels.csv)
uv run python -m stml.experimental.make_labels

# Stage 2 — features
uv run python -m stml.experimental.bloomberg_ingest   # ingests data/bloomberg/raw/
uv run python -m stml.experimental.make_features      # 105 features + drift filter
uv run python -m stml.experimental.make_scope         # per-instrument embargo widths

# Stage 3 — modelling
uv run python -m stml.experimental.make_baseline      # per-class baseline AUC
uv run python -m stml.experimental.make_champions     # CPCV(6,2) + 1-SE champion per instrument

# Stage 5 — cluster-level importance
uv run python -m stml.experimental.make_importance    # MDA + MDI + SHAP, per asset class

# Stage 6 — calibration + sizing + backtest + emit deliverables
uv run python -m stml.experimental.make_deliverables  # writes outputs/*.csv

# Stage 7 — significance + deflation (PSR, MinTRL, DSR, PT)
uv run python -m stml.experimental.make_significance
```

The pipeline takes **~60 minutes** end-to-end on CPU (per `plan.md` §12).
A faster path for the grader who only needs to reproduce the deliverable CSVs
from the cached event predictions:

```bash
uv run python scripts/build_submission_deliverables.py
```

This runs in seconds because it reads the committed
`results/sreeram_experimental/oos_events_with_predictions.csv`.

---

## 4. H2 2022 rerun

The brief says the grader will rerun the code on the hidden H2 2022 window.
The release window covers `2020-01-03 → 2022-06-30`; H2 2022 is the held-out
period. To reproduce the deliverables on H2 2022:

1. Replace `data/ohlcv_data.csv` and `data/primary_signals.csv` with the
   versions extended through Dec 2022.
2. The cleaned Bloomberg parquets under `data/bloomberg/cleaned/` only cover
   the released window. For H2 2022 we ship two macro-context inputs:
   - `data/features/f11_macro_context_oos.csv` — Part I-schema z-scored macro features (45 columns, daily, Jul-Dec 2022)
   - `data/OOS_additional_data.xlsx` — raw H2 2022 macro series (22 Bloomberg-style columns)
3. Re-run the pipeline. For instruments / models whose champion has a
   Bloomberg-dependent feature, the inference path falls back to NaN-tolerant
   imputation (see `results/sreeram_experimental/bbg_missingness_ablation.csv`
   for the documented per-class AUC delta with vs. without BBG).
4. Re-emit the deliverable CSVs:
   ```bash
   uv run python scripts/build_submission_deliverables.py \
       --start 2022-07-01 --end 2022-12-31
   ```

**Documented limitation (plan §13 R-11):** the model is trained on a feature
set that includes Bloomberg-augmented columns (futures term structure, options
IV, EIA inventory) which extend through the H1 2022 release boundary only.
The shipped model was selected to be robust to BBG missingness at inference
time; the ablation CSV records the AUC drop on each asset class when the BBG
columns are zeroed-out at test time.

---

## 5. Rubric coverage

| Rubric § | Marks | Where it lives | Key artifact |
|---|---:|---|---|
| 1 Feature Engineering | 20 | `src/stml/experimental/features/`, `src/stml/experimental/make_features.py`, `reports/feature-catalog.md` | 105 features across 18 families, including Bloomberg-augmented |
| 2 Triple-Barrier Labeling | 20 | `data/triple_barrier_labels.csv`, `results/sreeram_experimental/jay_geometry_summary.csv` | Per-instrument `(pt, sl, h)` geometry picked by adjusted-Sharpe grid search |
| 3 Model Development & Comparison | 30 | `src/stml/experimental/champion_pipeline.py`, `make_champions.py`, `models.py`, `multitask.py` | Linear + RF + XGBoost + LightGBM + multi-task NN, CPCV(6,2) + 1-SE rule |
| 4 Cluster-Level Feature Importance | 10 | `src/stml/experimental/make_importance_deep.py`, `notebooks/sreeram_experimental/{energy,equity,metals}_importance.ipynb` | MDA + MDI + SHAP at cluster level, per asset class |
| 5 Model Evaluation | 20 | `results/sreeram_experimental/baseline_per_instrument.csv`, `champions_summary.csv`, `threshold_summary.csv`, `significance_summary.csv` | Per-instrument precision/recall/F1/AUC + primary-blind baseline + threshold sweep + PSR/MinTRL/DSR |
| Bonus Strategy Construction | +10 | `src/stml/experimental/make_deliverables.py`, `reports/sreeram_experimental/strategy_construction.md` | Vol-targeted SOPS + 4 NN variants + Grinold-Kahn costs |

### Key results (single source of truth: `reports/sreeram_experimental/final_report.md`)
- Annualised net Sharpe **3.20** on the H1 2022 OOS slice (vol 5.4 %, max DD −2.3 %)
- t = 2.70, bootstrap 95 % CI excludes 0, PSR(0) = 0.998, MinTRL = 61 (have 179)
- 7 of 11 instruments above AUC 0.55; per-class champion AUCs in `champions_summary.csv`
- DSR at N_eff = 2 → 0.996; at 4 · N_raw = 480 → 0.976

---

## 6. Repository layout

```
data/
├── ohlcv_data.csv                  raw OHLCV (released window; grader replaces for H2 2022)
├── primary_signals.csv             raw signals (released window; grader replaces)
├── triple_barrier_labels.csv       per-instrument-geometry meta-labels (canonical)
├── features/f11_macro_context_oos.csv   H2 2022 macro features (45 cols, daily)
├── OOS_additional_data.xlsx        raw H2 2022 macro source (22 Bloomberg-style series)
├── bloomberg/cleaned/              PIT-aligned cleaned BBG parquets (H1 2022 only)
└── meta/                           generated missingness diagnostics

outputs/
├── metamodel_predictions.csv       REQUIRED DELIVERABLE (1,408 rows, H1 2022)
└── strategy_weights.csv            OPTIONAL DELIVERABLE (1,408 rows, H1 2022)

results/sreeram_experimental/
├── oos_events_with_predictions.csv      H1 2022 events with calibrated probas + weights
├── oof_calibrated_predictions.csv       Training-period OOF probas (for SOPS fit)
├── champions_summary.csv                Per-instrument champion (pool, model, AUC, 1-SE-lower CI)
├── baseline_per_instrument.csv          Precision/recall/F1/AUC vs primary-blind baseline
├── threshold_summary.csv                Per-instrument bootstrap p* thresholds
├── significance_summary.csv             Strategy t-stat / bootstrap CI / PSR / MinTRL
├── deflation_ladder.csv                 DSR at multiple N_eff values
├── importance/{equity,energy,metals}/   Per-class cluster MDA + SHAP + dendrograms
└── strategy_variant_comparison.csv      SOPS vs 4 NN backbones head-to-head

reports/sreeram_experimental/
├── final_report.md                 Single-doc rubric-aligned summary
├── plan.md                         Golden-record build plan (every decision documented)
├── strategy_construction.md        Madmoun slide-by-slide mapping for the strategy track
├── s4_s5_summary.md                Multi-task NN + cluster importance write-up
├── action_tracker.md               Per-session build log
└── bloomberg_{pull_list,validation_report}.md   BBG ingestion documentation

src/stml/experimental/
├── make_*.py                       Stage runners (S1–S7)
├── champion_pipeline.py            CPCV + 1-SE champion selector
├── sizing.py                       6 Madmoun sizing functions + SOPS
├── backtest.py / cost_model.py     Barrier-exact backtest with Grinold-Kahn costs
├── significance.py                 PSR / MinTRL / DSR / PT
├── calibration.py                  Platt + isotonic
├── volatility.py                   Causal EWMA / Yang-Zhang / Parkinson / Garman-Klass
├── threshold.py                    Bootstrap p* = L/(G+L) from train returns
└── features/                       105 features across 18 families
```

---

## 7. Reproducibility contract

- All randomness gated on `random_state=42`. Verified by `tests/experimental/test_scaffold.py`.
- CSV emit is **byte-deterministic**: two consecutive runs of `make_deliverables`
  produce identical `outputs/*.csv`. Verified by `tests/experimental/test_s6.py`.
- All TF-class features are fit on the FE-train block only (≤ `2021-07-01`) and
  frozen. Verified by `tests/experimental/test_features.py` and
  `tests/experimental/test_methodology_guards.py`.
- 126+ tests in `tests/experimental/` (run `pytest tests/experimental/ -q`).
- The labels CSV ships with a `partition` column (`train` / `val` / `test`)
  and the pipeline honours it. The `test` partition (2021-12-31 → 2022-06-29)
  is the H1 2022 OOS slice that produces the deliverable.

---

## 8. Contact

For any question about this submission: see `reports/sreeram_experimental/final_report.md`
or the per-stage `action_tracker.md` entries.
