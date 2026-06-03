# BUSI70575 Coursework — Metamodel for a Multi-Asset Primary Trading Signal

A meta-labelling model that predicts, for each daily primary signal, the
probability that following the trade would be profitable under a
triple-barrier exit rule. Eleven futures instruments across three asset
classes are covered.

## Universe

| Asset class           | Instruments                                                          |
| --------------------- | -------------------------------------------------------------------- |
| Equity index futures  | `es1s` (S&P 500), `nq1s` (Nasdaq 100), `fesx1s` (Euro Stoxx 50)      |
| Energy                | `cl1s` (WTI), `ho1s` (Heating Oil), `rb1s` (Gasoline), `ng1s` (Nat Gas) |
| Metals                | `gc1s` (Gold), `si1s` (Silver), `hg1s` (Copper), `pl1s` (Platinum)   |

## Methodology

The pipeline follows the structure prescribed by the coursework brief:

1. **Feature engineering** — a feature set derived from daily OHLCV:
   technical indicators, latent-variable summaries (GMM / HMM regime
   probabilities), and other unsupervised features. Each feature is
   documented with what it is intended to capture.
2. **Triple-barrier labelling** — applied per instrument with justified
   barrier widths and a time-limit, following López de Prado (2018).
3. **Model development** — three families with hyperparameter tuning under
   purged cross-validation:
    - regularised logistic regression (linear),
    - gradient-boosted trees (tree-based),
    - a neural network (sequential / variable-selection architecture).
4. **Cluster-level feature importance** — correlated features are clustered,
   then MDI, MDA and SHAP are computed at the cluster level so the
   discussion is about feature *groups* rather than individual columns.
5. **Out-of-sample evaluation** — a clean held-out period (carved out of the
   training window) with precision, recall, F1, AUC, confusion matrices,
   decision-threshold analysis, a per-instrument breakdown, and a side-by-side
   comparison against a baseline that follows the primary signal blindly.
6. **Strategy construction** (competition track) — position sizing on top
   of the calibrated metamodel probabilities, targeting 10% annualised
   volatility.

## Data

Two CSV files, downloadable from Insendi under *Coursework*:

| File                  | Contents                                                                 |
| --------------------- | ------------------------------------------------------------------------ |
| `ohlcv_data.csv`      | Daily OHLCV history for all eleven instruments, one row per `(instrument, date)`. History starts in 1990 for most instruments (ES1S from 1997, FESX1S from 1998, NQ1S from 1999). |
| `primary_signals.csv` | Daily primary-model signals from January 2020, one row per `date`, one column per instrument, values in `{-1, 0, +1}`. |

Both files belong in `data/`. The released window covers up to **30 June 2022**;
the final six months (July–December 2022) are a hidden test set used by the
markers. See [data/README.md](data/README.md) for column-level detail.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── data/                     # OHLCV + primary signals (not tracked)
├── src/metamodel/            # Library code
├── notebooks/                # End-to-end Jupyter notebooks
├── reports/                  # Figures, tables, written analysis
└── outputs/                  # Generated deliverables (not tracked)
```

## Reproducing the pipeline

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/
```

Run the notebooks in order. The end-to-end run writes both deliverables into
`outputs/`.

## Deliverables

### Required — metamodel predictions

`outputs/predictions.csv`, covering January–June 2022:

```csv
date,instrument,prediction
2022-01-03,cl1s,0.74
2022-01-03,es1s,0.51
```

`prediction` is the probability in `[0, 1]` that the primary signal is worth
taking on that day, for that instrument.

### Optional — strategy weights

`outputs/weights.csv`, also covering January–June 2022:

```csv
date,instrument,weight
2022-01-03,cl1s,0.18
2022-01-03,es1s,-0.05
```

`weight` is the signed position weight (positive = long, negative = short).
The portfolio targets 10% annualised volatility.

## Module

BUSI70575 — *Systematic Trading Strategies with Machine Learning Algorithms*.
Coursework competition, weighted 50% of the final grade. Submission deadline:
4 June 2026.
