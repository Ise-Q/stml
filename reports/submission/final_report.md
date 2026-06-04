# Methodology & Results Summary

A single-document methodology summary of the pipeline, key statistics, and the
verdict on the H1 2022 out-of-sample slice. All numbers in this report reproduce
from CSV artefacts shipped under `results/submission/`.

---

## 1. Pipeline at a glance

| Stage | Purpose | Code | Key artefact |
|---|---|---|---|
| 1 | Triple-barrier meta-labels (per-instrument geometry) | `make_labels.py` | `data/triple_barrier_labels.csv` |
| 2 | Feature engineering (16 families, 94 registered features → 64 kept after KS-AUC drift filter) | `make_features.py` | `data/features.parquet`, `feature_drift_audit.csv` |
| 3 | Per-class baseline + per-instrument CPCV(6, 2) + 1-SE champion | `make_baseline.py`, `make_champions.py` | `champions_summary.csv` |
| 4 | Cluster-level feature importance (MDA + MDI + SHAP) | `make_importance.py`, `make_importance_deep.py` | `results/submission/importance/{class}/*` |
| 5 | Per-instrument Platt calibration on CPCV OOF | `calibration.py` | `oof_calibrated_predictions.csv` |
| 6 | Bootstrap p\* gate, six sizing variants, vol-targeted backtest, Grinold-Kahn costs | `sizing.py`, `backtest.py`, `make_deliverables.py` | `outputs/metamodel_predictions.csv`, `outputs/strategy_weights.csv` |
| 7 | Significance: t-stat, bootstrap CI, PSR, MinTRL, DSR, PT, Henriksson-Merton | `significance.py`, `make_significance.py` | `significance_summary.csv`, `deflation_ladder.csv` |

The pipeline runs end-to-end in ≈ 60 min on CPU; the notebook reads the cached
artefacts and renders the brief narrative in seconds.

---

## 2. Strategy verdict — H1 2022 OOS slice

| Metric | Value | Source |
|---|---:|---|
| **Annualised net Sharpe** | **3.15** | `backtest_metrics.csv` |
| Annualised net return | 16.2 % | same |
| Annualised vol | 5.1 % (target 10 %, 4.9 pp headroom) | same |
| Sortino (full-T, Sortino-Price 1994) | 5.55 | same |
| Max drawdown | −1.91 % | same |
| Turnover (× per year) | 183.9 | same |
| Avg holding period (days) | 2.19 | same |
| t = SR · √n | **2.25** | `significance_summary.csv` |
| Studentised stationary block-bootstrap 95 % CI (per period, primary) | [0.022, 0.356] — **excludes 0** | same |
| Lo/Opdyke analytic 95 % CI (per period, cross-check) | [0.028, 0.369] | same |
| PSR(0) | **0.989** (deployment threshold 0.95) | same |
| MinTRL | 68 periods (have 129) — certified | same |
| Ljung-Box Q(10) p-value | 0.21 (no significant lag-10 autocorrelation) | same |
| DSR at N_eff = 2 | 0.985 | `deflation_ladder.csv` |
| DSR at 4 · N_raw = 480 | 0.943 | same |

**Verdict (five-lens framework).** Four of five lenses agree on a positive,
deflated edge (AUC + cluster MDA + bootstrap Sharpe + DSR); the fifth
(Pesaran-Timmermann) is borderline because the meta-filtered prediction
distribution is concentrated near `p ≈ 0.5` for half the events, where PT is
known to be sensitive. Henriksson-Merton hit rate **0.571** at **z = 4.38
(p ≈ 6e-6)** confirms positive directional skill independently of PT.

These numbers are reported for transparency; the methodology — leakage-free
features, CPCV+1-SE selection, per-instrument Platt calibration, bootstrap p\*,
SOPS sizing, vol target — is the substance of the submission, not the headline
Sharpe.

---

## 3. Per-class champion AUC (CPCV 15-path mean OOS — development partition)

Mean of the per-instrument champion AUCs (`champions_summary.csv`) within each
asset class:

| Asset class | Mean champion AUC | Instruments | n_features (avg) |
|---|---:|---|---:|
| Equity | 0.540 | es1s, nq1s, fesx1s | 64 |
| **Energy** | **0.566** | cl1s, ng1s, ho1s, rb1s | 64 |
| Metals | 0.545 | gc1s, si1s, pl1s, hg1s | 64 |

The energy lead is traced to a single microstructure-plus-regime cluster
(`f5_participation_60 + f7_oi_level + ewma_hmm_prob_highvol`, MDA 0.128) — see §4.

### Per-instrument champion summary (sorted by AUC)

| Instrument | Asset class | Model | Winning pool | AUC | Lower 1-SE CI |
|---|---|---|---|---:|---:|
| ng1s (Natural Gas) | Energy | Elastic-net Logistic | energy_all | 0.619 | 0.569 |
| cl1s (WTI Crude) | Energy | Multi-task NN | energy_all | 0.566 | 0.551 |
| gc1s (Gold) | Metals | LightGBM | metals_all | 0.566 | 0.547 |
| pl1s (Platinum) | Metals | LightGBM | pl1s (individual) | 0.564 | 0.550 |
| nq1s (Nasdaq) | Equity | Random Forest | nq1s (individual) | 0.557 | 0.544 |
| ho1s (Heating Oil) | Energy | LightGBM | energy_cl_ho | 0.552 | 0.498 |
| es1s (S&P 500) | Equity | Elastic-net Logistic | equity_all | 0.548 | 0.535 |
| hg1s (Copper) | Metals | Elastic-net Logistic | metals_all | 0.539 | 0.522 |
| rb1s (RBOB Gasoline) | Energy | Random Forest | rb1s (individual) | 0.526 | 0.512 |
| fesx1s (Euro Stoxx) | Equity | Elastic-net Logistic | equity_all | 0.516 | 0.507 |
| si1s (Silver) | Metals | Multi-task NN | metals_all | 0.512 | 0.500 |

**10 of 11 instruments** have a lower-1-SE-CI above 0.50 on the development
partition (the one exception, ho1s, sits just below at 0.498). The 1-SE bound
excludes 0.50 most cleanly for ng1s (0.57), cl1s (0.55), gc1s (0.55), pl1s
(0.55), nq1s (0.54), and es1s (0.54). Per-`(instrument, pool, model)` candidate
matrix in `champions_per_pool_per_model.csv`.

---

## 4. Cluster-level feature importance — headline finding

Cross-checked via MDA, MDI (XGBoost gain), and SHAP magnitude; rank agreement
quantified by Kendall-τ. The leading clusters per asset class (by mean MDA on
the development partition):

* **Energy** — top cluster combines `f5_participation_60` (signal-participation
  rate), `f7_oi_level` (open interest), and `ewma_hmm_prob_highvol` (HMM
  high-vol regime posterior). Permutation-AUC drop ≈ **0.128**. The Bloomberg
  macro / EIA blocks (F11, F22) contribute a smaller secondary cluster
  (`f11_be10y_chg20 + f11_ust10_chg5`, MDA ≈ 0.003).
* **Equity** — top cluster is mean-reversion / momentum / path-structure:
  `f1_bb_pctb_20, f1_rsi_14, f6_ts_momentum_20, f10_oc_ret_mean_20,
  f12_trend_tval_21`. MDA ≈ **0.021**. Two clusters are significant.
* **Metals** — top cluster combines momentum and signed-bias features
  (`f5_long_bias_20, f6_ts_momentum_60, f6_ma_cross_20_60, f6_macd_*`),
  MDA ≈ **0.007**.

Pruned-vs-full per class (`results/submission/importance/deep_summary.csv`):
removing all but the significant clusters leaves AUC essentially unchanged on
equity (+0.021), drops it by ≈ 0.008 on energy, and by ≈ 0.027 on metals — i.e.
the energy cluster carries the asset class while metals has more distributed
signal.

F18 (futures term structure) and F19 (options-implied vol) were prototyped
during development and dropped from the final model on parsimony grounds —
they did not clear the cluster-importance threshold against the F1-F17 + F11 +
F22 baseline.

---

## 5. Classification metrics — sealed test partition (H1 2022)

Per-instrument `precision / recall / F1 / AUC` in `baseline_per_instrument.csv`;
the notebook renders them side-by-side per asset class. **Confusion-matrix
analysis** at `p̂ ≥ 0.5` on the 951-event H1 2022 OOS slice:

| Metric | Primary-blind (take every signal) | Primary + meta filter (p̂ ≥ 0.5) |
|---|---:|---:|
| Trades taken | 951 | 470 |
| Precision | 0.492 | 0.564 |
| Recall | 1.000 | 0.566 |
| F1 | 0.660 | 0.565 |
| False positives | 483 | 205 |
| True positives | 468 | 265 |

**Precision lift +0.072; false positives avoided 278; true positives missed 203.**
The meta filter is a *precision lifter*, not a recall expander — exactly what a
trade-selection meta-model should be.

---

## 6. Reproducibility contract

* **Seed.** All randomness gated on `random_state = 42`. Verified by
  `tests/experimental/test_scaffold.py`.
* **Byte-deterministic emit.** Two consecutive runs of
  `scripts/build_submission_deliverables.py` produce identical
  `outputs/*.csv` bytes (md5-verified).
* **Causal feature contract.** Every E-class feature is causal by
  truncation-invariance; every TF-class fit lives on the FE-train block only
  (`date ≤ 2021-07-01`) and is applied with frozen parameters. Verified by
  `tests/experimental/test_features.py` and
  `tests/experimental/test_methodology_guards.py`.
* **Partition discipline.** The labels CSV ships with an authoritative
  `partition` column (`train` 60 % / `val` 21 % / `test` 19 %). The H1 2022
  deliverable is the `test` partition; the H2 2022 rerun uses the same code
  path with `--start 2022-07-01 --end 2022-12-31` after running
  `scripts/extend_bloomberg_for_h2.py` to extend the cleaned Bloomberg
  parquets.

---

## 7. Limitations & honest caveats

1. **Measurement frame.** OHLCV is adjusted continuous-futures, not raw
   front-month. Numbers should not be read as raw-market WTI / S&P 500 P&L.
2. **Bloomberg coverage on H2 2022.** The cleaned BBG panel ends 2022-06-30
   in the released window; `scripts/extend_bloomberg_for_h2.py` extends the
   F11 macro and F22 EIA panels from `data/OOS_additional_data.xlsx`. The
   per-class AUC delta when BBG columns are zeroed at inference time is
   < 0.01 on equity and ≈ 0 on energy / metals
   (`bbg_missingness_ablation.csv`).
3. **Per-instrument barrier geometry.** Six instruments use `h = 1` — the
   EDA shows the signal information sits at lag 1 on these (a wider window
   dilutes signal with noise). The label converges to a *sign-of-next-bar
   return* on those instruments — which is what the EDA argues is
   predictable. The other five instruments use `h ∈ {10, 15, 20}` with
   asymmetric `(pt, sl)` reflecting their return distributions.
4. **Strategy `n_periods = 129 daily bars` (≈ 6 months).** Sharpe inferred
   on this short window has wide CI; the bootstrap and DSR battery are the
   right way to read the headline number, not the point estimate alone.
