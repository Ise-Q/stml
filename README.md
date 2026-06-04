# STML Competition Submission

This repository is set up for grading through one notebook:

- `submission.ipynb` - run this top-to-bottom.
- `src/stml/` - helper package imported by the notebook.
- `data/` - raw released data and committed inputs used by the notebook.
- `src/stml/new_work/outputs/` - committed model outputs used in load mode.
- `results/strategy_eval/` - committed strategy evaluation outputs.

The notebook runs in **LOAD mode** by default (`FORCE_RECOMPUTE = False`), so the expensive CPCV/model
search artifacts are loaded from the committed outputs. This is intentional for grading reproducibility.

## Notebook Structure

- **Sections 0-5:** setup, EDA, macro data, HMM/regime features, full F1-F17 feature engineering, and leakage discipline.
- **Section 6:** triple-barrier labels, sample weights, CPCV protocol, and barrier-geometry justification.
- **Sections 7-9:** model comparison across logistic/RF/XGBoost/MLP, final variant lock, and saved hyperparameters.
- **Section 10:** cluster-level feature importance and weight-vector extraction.
- **Section 11:** model evaluation: precision, recall, F1, AUC, confusion matrix, threshold sweep, per-instrument breakdown, and blind-primary baseline.
- **Section 12:** optional strategy construction and primary-vs-metamodel comparison.
- **Section 13:** final summary.

## How To Run

Recommended:

```bash
uv sync --group features-extra
uv run jupyter nbconvert --to notebook --execute submission.ipynb --output submission.executed.ipynb --ExecutePreprocessor.timeout=7200
```

Or open `submission.ipynb` in Jupyter/JupyterLab and choose **Run All**.

Plain pip fallback:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
jupyter nbconvert --to notebook --execute submission.ipynb --output submission.executed.ipynb --ExecutePreprocessor.timeout=7200
```

Expected runtime in load mode is a few minutes. Running the notebook regenerates
`results/feature_matrix.parquet`; this is a normal generated artifact.

## Prediction File

The metamodel predictions are here:

```text
src/stml/new_work/outputs/metamodel_predictions.csv
```

Important columns:

- `date` - trade signal date.
- `instrument` - instrument ticker.
- `calibrated_proba` - final probability prediction, i.e. `P(trade is profitable)`.
- `raw_proba` - uncalibrated model probability.
- `bin` - realised triple-barrier label for the released out-of-sample evaluation window.

### Brief-format deliverables (repo root)

Two ready-to-grade CSVs in the exact brief schema, covering the **first half of 2022 (Jan–Jun)**:

- `predictions.csv` — `date,instrument,prediction` (944 events). `prediction` is the calibrated
  `P(trade is profitable)` (the `calibrated_proba` column), one row per non-zero primary-signal event.
- `strategy_weights.csv` — `date,instrument,weight` (strategy track). Per-instrument vol-targeted
  portfolio weights under the selected **SOPS** sizer, scaled by `1/K` (K = 11) so the portfolio
  return is `Σ_k weight · next-day return`; flat instrument-days are `0`.

The training out-of-fold probabilities used for calibration/strategy work are in
`data/oof_meta_probabilities.csv`.
