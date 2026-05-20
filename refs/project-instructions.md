# Coursework: Competition — Reference

> Systematic Trading Strategies with Machine Learning Algorithms
> Alken Asset Management coursework. **50% of final grade.**
> Contact: h.madmoun@ic.ac.uk

## Overview

Build a **metamodel** on top of a primary trading signal that is provided for **11 instruments across three asset classes**. For each primary signal, the metamodel predicts the **probability that following it would be profitable** under a triple-barrier exit rule.

The metamodel takes the primary signal plus your engineered features and outputs a **probability in [0, 1]** that the bet is worth taking.

This is a **group project** in teams of **5 students** (26 groups total). Marked out of **100** (+10 bonus possible for the optional competition track; bonus is capped so the final mark does not exceed 100).

## Key Information

| Item | Detail |
|---|---|
| Deadline | June 4, 2026 |
| Team size | 5 students (26 groups) |
| Weight | 50% of final grade |
| Marked out of | 100 (with +10 bonus for the optional competition track) |

## The Universe

Daily primary-model signals (-1, 0, +1) are provided for the following 11 instruments. You must cover **at least one full asset class**; covering more (up to all 11 instruments) is optional.

**Equity Index Futures**

| Ticker | Index |
|---|---|
| ES1S | S&P 500 |
| NQ1S | Nasdaq 100 |
| FESX1S | Euro Stoxx 50 |

**Energy**

| Ticker | Commodity |
|---|---|
| CL1S | WTI Crude Oil |
| HO1S | Heating Oil |
| RB1S | RBOB Gasoline |
| NG1S | Natural Gas |

**Metals**

| Ticker | Metal |
|---|---|
| GC1S | Gold |
| SI1S | Silver |
| HG1S | Copper |
| PL1S | Platinum |

## Task Description

Build a metamodel for each instrument you cover. The pipeline is:

1. Feature engineering from OHLCV (and anything else you can derive)
2. Labeling via the **triple-barrier method**, as taught in the course
3. Training and comparing several ML models with hyperparameter tuning
4. Feature importance analysis at the cluster level
5. Evaluation on a clean out-of-sample period
6. (Optional) Strategy construction on top of the metamodel probabilities

## Marking Scheme

| Section | Marks |
|---|---|
| Feature Engineering | 20 |
| Labeling (Triple-Barrier Method) | 20 |
| Model Development and Comparison | 30 |
| Feature Importance Analysis (Cluster-Level) | 10 |
| Model Evaluation | 20 |
| **Total** | **100** |
| Optional: Strategy Construction (Competition) | +10 bonus |

The bonus is capped so the final mark does not exceed 100.

### 1. Feature Engineering (20 marks)

Build a rich feature set drawing on techniques covered in the course:

- Technical indicators
- Latent variable models (GMM, HMM)
- Any of the unsupervised learning methods discussed
- Anything else you can justify

**Be as creative as possible.** Document what each feature is meant to capture.

### 2. Labeling: Triple-Barrier Method (20 marks)

Apply the triple-barrier method as taught in the course. You must **justify your choice of barrier widths and time-limit**.

### 3. Model Development and Comparison (30 marks)

At least **three models with hyperparameter tuning**, drawn from across the three families:

- **Linear models** (e.g. Logistic Regression with regularization)
- **Tree-based models** (e.g. Random Forest, XGBoost, LightGBM)
- **Neural networks** (e.g. Variable Selection Network or Sequential Neural Networks)

Present a clear comparison: which model wins, on which metric, and why you think so.

### 4. Feature Importance Analysis: Cluster-Level (10 marks)

Beyond per-feature importance, compute importance at the **cluster level**:

- Cluster correlated features together
- Apply MDI, MDA, or SHAP at the cluster level
- Discuss which feature groups drive your metamodel

### 5. Model Evaluation (20 marks)

Evaluate on an out-of-sample period that you carve out cleanly from the training period.

- Classification metrics: precision, recall, F1, AUC
- Confusion matrix and decision-threshold analysis
- **Per-instrument breakdown** (the metamodel may help on some instruments and not others — say so)
- Comparison against a baseline that follows the primary signal blindly

### Optional: Strategy Construction, Competition Track (10 bonus marks)

For groups that want to compete: use the metamodel probabilities to build a **position-sizing strategy** on top of the primary signal, either on a single asset class or on the full 11-instrument universe.

Full constraints (position limits, gross/net exposure, rebalancing rules, target volatility) are released on **Wednesday 20 May**.

Backtest metrics to report:

- CAGR
- Annualised volatility
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Average holding period
- Turnover

## Dataset

Two CSV files are available on **Insendi under Coursework**.

### `ohlcv_data.csv`

Daily OHLCV history for all 11 instruments. One row per (instrument, date).

| Column | Description |
|---|---|
| date | Trading date (YYYY-MM-DD) |
| instrument | Lowercase ticker (e.g. cl1s, es1s, gc1s) |
| open, high, low, close | Continuous-contract prices |
| volume | Daily volume |
| open_interest | Daily open interest |

History starts in 1990 for most instruments. Equity Index futures start later: **ES1S in 1997, FESX1S in 1998, NQ1S in 1999.**

### `primary_signals.csv`

Daily primary-model signals from January 2020 onwards. One row per date, one column per instrument.

| Column | Description |
|---|---|
| date | Trading date (YYYY-MM-DD) |
| es1s, nq1s, …, pl1s | Primary signal in {-1, 0, +1} |

Signal convention:

- **+1**: the primary model wants to go **long** that day
- **-1**: the primary model wants to go **short** that day
- **0**: no position taken by the primary model

**Important.** The released data covers up to **30 June 2022**. The final **6 months** (July to December 2022) are held out and will be used as a hidden test set to evaluate your final submission.

## Evaluation

You will be judged on:

1. Quality and creativity of your feature engineering
2. Rigour of your labeling and validation protocol
3. Appropriateness of your model selection and comparison
4. Critical analysis of your results
5. Code quality, reproducibility, and documentation

**The score is focused entirely on methodology, not on performance.** You can score a high mark even if your metamodel does not beat the primary signal.

The best submission, judged on **quality of research** rather than performance, will be **presented to the research team at Alken Asset Management, with an interview for an internship** at the end of it.

## Getting Started

- **OHLCV data** — download `ohlcv_data.csv` from the Coursework folder on Insendi.
- **Primary signals** — download `primary_signals.csv` from the Coursework folder on Insendi.
- **Programming sessions** — for implementation guidance, refer to all programming sessions and optional programming sessions of the course.

## Submission Rules

- **Group size**: 5 students per group, 26 groups in total
- **One submission per group**: a single combined submission
- **Documentation**: code must be clean, well-documented and reproducible; your notebook should run end-to-end and produce the deliverable CSV
- **Academic integrity**: all work must be original. Plagiarism will result in zero marks and potential disciplinary action.

## Deliverables

### 1. Code

A Jupyter notebook or a set of Python files that runs end-to-end and produces the deliverable files below. **Clean, well-documented code is part of the mark.**

### 2. Required: Metamodel Predictions

A CSV file covering the **first half of 2022** (January to June). Your code will be rerun on the hidden second half of 2022 for the final test.

Format: one row per (date, instrument, prediction).

```
date,instrument,prediction
2022-01-03,cl1s,0.74
2022-01-03,es1s,0.51
...
```

- `date`: trading date (YYYY-MM-DD)
- `instrument`: lowercase ticker
- `prediction`: probability in [0, 1] that the primary signal is worth taking

### 3. Optional: Strategy Weights

For groups competing on the strategy track, an additional CSV covering the first half of 2022:

```
date,instrument,weight
2022-01-03,cl1s,0.18
2022-01-03,es1s,-0.05
...
```

- `weight`: signed position weight (positive = long, negative = short)

Constraints on the weights will be specified on **20 May**.

## Contact

For any question regarding the coursework: **h.madmoun@ic.ac.uk**

Good luck and have fun!
