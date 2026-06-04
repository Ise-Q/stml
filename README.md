# Systematic Trading Strategies with ML — Meta-Model Submission

**Imperial College London × Alken Asset Management.** A *meta-model* that takes the provided primary
trading signal `s ∈ {−1, 0, +1}` for 11 futures and predicts, for each non-zero signal, the
**probability that following the trade is profitable** under a triple-barrier exit.

This submission covers **Part I — the feature-engineering foundation** end-to-end: it regenerates the
full F1–F17 feature matrix **from raw OHLCV + signals**, including for the hidden Jul–Dec 2022 window.
(Labelling, model fitting and position-sizing are Part II.)

---

## What's in this folder

```
submission.ipynb                       the one notebook — run this top-to-bottom
src/stml/                              the util package (imported as `import stml`)
data/
  ohlcv_data.csv                       raw OHLCV            ← REPLACE with your through-Dec-2022 file
  primary_signals.csv                  raw signals         ← REPLACE with your through-Dec-2022 file
  additional_data.xlsx                 F11 macro workbook (released window)
  OOS_additional_data.xlsx             F11 macro source for Jul–Dec 2022 (provenance)
  features/f11_macro_context_oos.csv   our EXTERNAL features for Jul–Dec 2022 (the shipped CSV)
results/                              regenerated artifacts land here (feature_matrix.parquet, …)
requirements.txt                     pinned deps (pip fallback)
pyproject.toml / uv.lock             project + locked deps (uv)
tests/                               test suite (pytest)
```

> The util folder is `src/stml/` — kept as an installable package so every notebook import resolves and
> `stml.io` can auto-locate `data/` from the repo root.

---

## How to run

### Option A — with `uv` (recommended)
```bash
uv sync --group features-extra
uv run jupyter lab          # open submission.ipynb, then Run All
# …or headless:
uv run jupyter nbconvert --to notebook --execute submission.ipynb
```

### Option B — with plain `pip` (Python 3.12)
```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # makes `import stml` resolve
jupyter lab                        # open submission.ipynb, then Run All
```

`requirements.txt` already includes the two feature dependencies **hmmlearn** (F17 regimes) and
**PyWavelets** (F13), and points at the **CPU** build of PyTorch — no GPU needed.

---

## Re-running on the hidden Jul–Dec 2022 data

1. Replace `data/ohlcv_data.csv` and `data/primary_signals.csv` with your versions **extended through
   Dec 2022** (same columns, just more rows). Keep the long price history — trailing features warm up on
   it causally.
2. Run `submission.ipynb` top-to-bottom.

The notebook **auto-detects** the extension. Section 0 then:
- fits every learned feature family on the **FE-train block only (≤ 2021-07-01)** — the boundary is
  pinned by *date*, so extending the axis never moves it;
- regenerates all engineered features for the new rows, tagged **`partition == "oos"`**;
- splices our shipped external macro features (`data/features/f11_macro_context_oos.csv`, z-scored with
  the same frozen FE-train statistics) onto those `oos` rows — so you do **not** need any macro source
  for the hidden window;
- writes the full matrix to `results/feature_matrix.parquet`.

The shipped external CSV covers Jul 1 – Dec 30 2022; any `oos` row outside that span keeps
median-imputable NaNs for F11 rather than failing.

---

## Verifying

```bash
uv run pytest        # or: pytest
```
`tests/test_oos_partition.py` checks that the date-pinned split reproduces the released-window
boundaries and that extending the axis tags new rows `oos` without moving any released partition. The
released-window rebuild reproduces the committed feature matrix to ~1e-10 (determinism, seed = 42).

---

## Notes
- **Leakage discipline.** Fitted (TF) families are fit on `≤ 2021-07-01` and frozen; engineered (E)
  families are causal by truncation-invariance; structural NaNs are never forward-filled. Details in
  `submission.ipynb` §4–§5.
- **Python 3.12** required (`requires-python = ">=3.12"`).
- The full per-column feature documentation is in `reports/feature-catalog.md`.
