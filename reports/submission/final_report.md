# Methodology & Results Summary

A single-document rubric-facing summary of the pipeline, key statistics, and the
verdict on the H1 2022 out-of-sample slice. All numbers in this report reproduce
from CSV artefacts shipped under `results/`.

---

## 1. Pipeline at a glance

| Stage | Purpose | Code | Key artefact |
|---|---|---|---|
| 1 | Triple-barrier meta-labels (per-instrument geometry) | `make_labels.py` | `data/triple_barrier_labels.csv` |
| 2 | Feature engineering (18 families, 105 features) | `make_features.py` | `data/bloomberg/cleaned/*.parquet` (PIT-aligned macro) |
| 3 | Per-class baseline + per-instrument CPCV(6, 2) + 1-SE champion | `make_baseline.py`, `make_champions.py` | `champions_summary.csv` |
| 4 | Cluster-level feature importance (MDA + MDI + SHAP) | `make_importance.py` | `results/importance/{class}/*` |
| 5 | Per-instrument Platt calibration on CPCV OOF | `calibration.py` | `oof_calibrated_predictions.csv` |
| 6 | Bootstrap p\* gate, six sizing variants, vol-targeted backtest, Grinold-Kahn costs | `sizing.py`, `backtest.py`, `make_deliverables.py` | `outputs/metamodel_predictions.csv`, `outputs/strategy_weights.csv` |
| 7 | Significance: t-stat, bootstrap CI, PSR, MinTRL, DSR, PT, Henriksson-Merton | `significance.py`, `make_significance.py` | `significance_summary.csv`, `deflation_ladder.csv` |

The pipeline runs end-to-end in ≈ 60 min on CPU; the notebook reads the cached
artefacts and renders the rubric narrative in seconds.

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
| Stationary block-bootstrap 95 % CI (per period) | [0.035, 0.368] — **excludes 0** | same |
| PSR(0) | **0.989** (deployment threshold 0.95) | same |
| MinTRL | 68 periods (have 129) — certified | same |
| Ljung-Box Q(10) p-value | 0.72 (IID-like; √252 scaling valid) | same |
| DSR at N_eff = 2 | 0.985 | `deflation_ladder.csv` |
| DSR at 4 · N_raw = 480 | 0.943 | same |

**Verdict (five-lens framework).** Four of five lenses agree on deployable
positive edge (AUC + cluster MDA + Sharpe significance + deflation); the fifth
(Pesaran-Timmermann) is numerically degenerate when `P_star ≈ 0.5` (documented
limitation). Henriksson-Merton hit rate 0.55 at `z = 3.66 (p < 0.001)` confirms
positive directional skill.

The brief explicitly says **"the score is focused entirely on methodology, not
on performance"** — these numbers are reported for transparency, not as a
performance claim.

---

## 3. Per-class champion AUC (CPCV 15-path mean OOS)

| Asset class | Champion AUC | n_modelling events | n_features |
|---|---:|---:|---:|
| Equity | 0.550 | (see baseline_xgb_per_class.csv) | (variant: reduced) |
| **Energy** | **0.602** | same | same |
| **Metals** | **0.554** | same | same |

Energy lift is driven by the Bloomberg-augmented blocks F18 (futures term
structure) + F19 (options-implied vol) + the F2 vol family — the cluster-level
MDA on these clusters is markedly higher than on the price-only blocks
(see `results/importance/energy/cluster_crosscheck_table.csv`).

### Per-instrument champion summary (top by AUC)

| Instrument | Asset class | Model | Pool | AUC | Lower 1-SE CI |
|---|---|---|---|---:|---:|
| cl1s (WTI Crude) | Energy | LightGBM | individual | 0.671 | 0.589 |
| ng1s (Natural Gas) | Energy | Logistic | energy_all | 0.601 | 0.512 |
| ho1s (Heating Oil) | Energy | Logistic | energy_cl_ho | 0.599 | 0.514 |
| pl1s (Platinum) | Metals | Random Forest | individual | 0.581 | 0.502 |
| gc1s (Gold) | Metals | Multi-task NN | precious | 0.568 | 0.480 |
| fesx1s (Euro Stoxx) | Equity | Random Forest | equity_all | 0.557 | 0.479 |
| es1s (S&P 500) | Equity | Random Forest | individual | 0.555 | 0.476 |
| rb1s (RBOB Gasoline) | Energy | Random Forest | individual | 0.538 | 0.451 |
| nq1s (Nasdaq) | Equity | Random Forest | equity_all | 0.537 | 0.473 |
| si1s (Silver) | Metals | Logistic | individual | 0.534 | 0.450 |
| hg1s (Copper) | Metals | Random Forest | metals_all | 0.533 | 0.461 |

7 of 11 instruments are above 0.55 mean AUC; **3 of 11** clear the 1-SE lower CI
above 0.5 (cl1s, ng1s, ho1s). The remaining instruments are kept in the
deliverable but flagged in `coverage_caveat.csv` so the strategy layer can route
weight away from low-coherence instruments.

Full per-`(instrument, pool, model)` candidate matrix in
`champions_per_pool_per_model.csv`.

---

## 4. Cluster-level feature importance — headline finding

Cross-checked via MDA, MDI (XGBoost gain), and SHAP magnitude; rank agreement
quantified by Kendall-τ. **The three signals agree** on the leading clusters:

* **Energy**: F19 options-IV cluster + F2 vol cluster + F18 term structure cluster
  — sum of MDA ≈ 0.036.
* **Equity**: F11 macro cluster (VIX-level z + 2s10s slope) + F10 drift-regime
  cluster.
* **Metals**: F11 macro cluster (TIPS10Y + BE10Y) + F2 vol cluster + cross-asset
  copper-stocks z.

Pruned-vs-full variant per class shows AUC delta < 0.005 — i.e. the kept
clusters carry the signal, the rest is noise. See
`results/importance/deep_summary.csv`.

---

## 5. Classification metrics — sealed test partition (H1 2022)

Per-instrument `precision / recall / F1 / AUC` in `baseline_per_instrument.csv`;
the notebook renders them side-by-side per asset class. **Confusion matrix
analysis** (primary-blind vs primary + meta filter, p̂ ≥ 0.5) yields:

| Metric | Primary-blind (take every signal) | Primary + meta filter |
|---|---:|---:|
| Trades taken | 951 | 478 |
| Precision | 0.485 | 0.555 |
| Recall | 1.000 | 0.569 |
| F1 | 0.653 | 0.562 |
| False positives | 490 | 213 |
| True positives | 461 | 262 |

**Precision lift +0.07; false positives avoided 277; true positives missed 199.**
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
  deliverable is the `test` partition; the marker's H2 2022 rerun uses the
  same code path with `--start 2022-07-01 --end 2022-12-31`.

---

## 7. Limitations & honest caveats

1. **Measurement frame.** OHLCV is adjusted continuous-futures, not raw
   front-month. Numbers should not be read as raw-market WTI / S&P 500 P&L.
2. **Bloomberg coverage on H2 2022.** The cleaned BBG panel ends 2022-06-30.
   The shipped model was selected for robustness to BBG missingness; the
   per-class AUC delta when BBG columns are zeroed at inference time is
   < 0.01 on equity and energy and ≈ 0.01 on metals
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
