# Meta-Model Results — Triple-Barrier Meta-Labeling, CPCV, Calibration & Cluster Importance

> Notebook: [`notebooks/jay/metamodel.ipynb`](../../notebooks/jay/metamodel.ipynb) (8 sections, executed end-to-end)
> Model package: [`src/stml/model/`](../../src/stml/model/) — `dataset · labels · cv · barrier_search · optuna_objective · calibration · trees · mlp · vsn · linear · importance · evaluate`
> Feature layer: [`src/stml/metamodel/`](../../src/stml/metamodel/) → [`results/feature_matrix.parquet`](../../results/) · catalog: [`reports/feature-catalog.md`](../feature-catalog.md)
> Deliverable: [`results/predictions_h1_2022.csv`](../../results/predictions_h1_2022.csv) · cross-family table: [`results/metamodel_cross_family_metrics.csv`](../../results/metamodel_cross_family_metrics.csv)
> Course refs: [`refs/meta-model-guide.pdf`](../../refs/meta-model-guide.pdf) (Madmoun, Parts 1–4); AFML Ch. 3 (triple-barrier), Ch. 4 (uniqueness), Ch. 7 (purged/combinatorial CV)

---

## 0. What this report is

This is the write-up of **my meta-model run** (Jay), compiled directly from the executed
output cells of `metamodel.ipynb`. The deliverable is the standard course metamodel: given
the provided primary signal `s ∈ {−1,0,+1}` for 11 futures, estimate for each **non-zero**
signal the probability in `[0,1]` that *taking the trade is profitable* under a triple-barrier
exit, and emit `date,instrument,prediction`.

**Grading is methodology, not performance** (CLAUDE.md §1). Every section is laid out as
**what was run → results → interpretation → leakage/mitigation note**, and the load-bearing
results I argue for are the *relative family comparison* and the *precision-lift vs the blind
primary* — **not** any single AUC.

> **Note on this revision.** An earlier run selected a *degenerate* barrier geometry (`pt=0`, which
> **disables** the profit-taking barrier and collapses the label toward sign-of-forward-return). The
> barrier search has been fixed to refuse disabled barriers, the grid now spans four horizons, and
> the geometry below (`pt=1, sl=2, h=15`) is a valid triple barrier. **The corrected, honest result
> is materially weaker than that earlier (inflated) run** — see §2 and the conclusions. That is the
> point: the apparent edge was largely a labeling artifact.

**Cross-cutting discipline applied throughout** (proved in code/tests, not asserted):

- **Leakage.** *Fitted* features — the **TF** (transform-fitted) class: PCA / GMM / Markov / HMM /
  scalers, families F3/F4/F11/F16/F17 — have their parameters **estimated only on data ≤ 2021-07-01**
  (the train/val boundary) and are then applied forward with those parameters **frozen**, so even the
  val/test feature *columns* were produced without seeing val/test data. (The **E**, engineered, class
  needs no freeze — it is causal by construction.) The `test` partition (Jan–Jun 2022) is our own
  self-held holdout, opened **exactly once** via the `release_test` tripwire. The hidden **H2-2022 set
  is not in our data at all** — the same pipeline regenerates the deliverable CSV for it, unchanged,
  when the graders re-run our code. CPCV, hyper-parameter selection, calibration, and the decision
  threshold are all fit on **dev only**.
- **Purge + per-instrument embargo** on every split (triple-barrier labels span multiple bars).
- **Sample-uniqueness weights recomputed per fold** (overlapping labels are not iid).
- **Seed 42** for all modelling. (The *feature-matrix provenance* file records `seed: 0` — that is the
  separate FE-build seed, not the modelling seed.)
- **Objective = AUC / an economic metric, never accuracy.**

---

## Section 0 — Setup & leakage-safe harness

The harness loads the frozen feature matrix and attaches each row's `bar_pos` (its index on the
instrument's own trading calendar — the axis CV and the labeler count `h` along).

| Quantity | Value |
|---|---|
| Feature matrix | **4 984 rows × 180 cols** (175 feature columns) |
| Dev slice (train+val, ≤ 2021-12-30) | **3 966 rows / 516 dates** |
| Partition row counts | train 2 925 · val 1 041 · test 1 018 |
| FE-train boundary | **2021-07-01** (asserted; build aborts on drift) |
| Primary signals | (645 dates × 11 instruments) |

**Optional-capability probe** (so P2 arms degrade *visibly*, never via a silent cap):
`lightgbm=True`, `nn_seq=False`, `feature_tracks=False` (LSTM/TFT and PCA-Track-B are documented
deferrals in §6/§7). A **dry run** of the whole CPCV stack on one instrument (`si1s`) returns
**453/453 OOF coverage** — the harness is wired correctly before the full sweep.

---

## Section 1 — Exploratory Data Analysis

*(These signal/feature statistics are independent of the barrier choice and are unchanged from the
earlier run.)*

### 1.1 The signal is imbalanced and persistent

| instrument | p(s=−1) | p(s=0) | p(s=+1) | lag-1 autocorr | mean run-len | nonzero frac |
|---|---|---|---|---|---|---|
| cl1s | 0.056 | 0.346 | 0.598 | 0.791 | 8.6 | 0.654 |
| es1s | 0.184 | 0.109 | 0.707 | 0.584 | 5.1 | 0.891 |
| fesx1s | 0.544 | 0.012 | 0.443 | 0.509 | 4.0 | 0.988 |
| gc1s | 0.050 | 0.740 | 0.211 | 0.619 | 2.9 | 0.260 |
| hg1s | 0.495 | 0.026 | 0.479 | 0.647 | 5.2 | 0.974 |
| ho1s | 0.016 | 0.902 | 0.082 | 0.534 | 2.2 | 0.098 |
| ng1s | 0.192 | 0.808 | **0.000** | 0.760 | 5.2 | 0.192 |
| nq1s | 0.313 | 0.064 | 0.623 | 0.556 | 4.4 | 0.936 |
| pl1s | 0.211 | 0.136 | 0.653 | 0.663 | 5.4 | 0.864 |
| rb1s | 0.405 | 0.026 | 0.569 | 0.769 | 7.8 | 0.974 |
| si1s | 0.422 | 0.104 | 0.474 | 0.600 | 4.0 | 0.896 |

**Interpretation.** Piecewise-constant with **long runs** (lag-1 autocorr 0.51–0.79), so neighbouring
rows are **not independent** — the motivation for purge/embargo and uniqueness weights. `ng1s` is
**short-only**; `ho1s` is almost always flat (thin name).

### 1.2 The signal predicts the *next* bar (counter-trend)

| instrument | corr(s, r_fwd) | mean pnl (bp) | hit-rate | n |
|---|---|---|---|---|
| es1s | 0.129 | 17.94 | 0.052 | 6 134 |
| nq1s | 0.130 | 24.22 | 0.055 | 5 718 |
| fesx1s | 0.052 | 8.12 | 0.056 | 6 029 |
| cl1s | 0.078 | 65.98 | 0.032 | 7 813 |
| ho1s | 0.009 | 43.54 | 0.006 | 7 463 |
| rb1s | 0.072 | 29.55 | 0.041 | 8 012 |
| ng1s | **NaN** | 68.76 | 0.009 | 7 455 |
| gc1s | 0.178 | 28.68 | 0.014 | 7 565 |
| si1s | 0.055 | 13.07 | 0.037 | 7 964 |
| hg1s | 0.115 | 17.59 | 0.042 | 8 014 |
| pl1s | 0.068 | 14.70 | 0.037 | 7 941 |

`corr(s_t, r_{t+1}) > 0` on essentially every instrument → the signal predicts the **next** bar (enter
at **t+1**); the `ng1s` NaN is expected (constant side → undefined correlation). Counter-trend /
mean-reversion read.

### 1.3 Feature structure

The redundancy map clusters the 175 features by `d=√(½(1−ρ))`: **106 clusters from 175 features**.
Family coverage / missingness inspected (structural NaNs from closed venues are never forward-filled;
trees handle them natively, the others median-impute on train rows only).

> **Takeaway → motivates Section 3.** Imbalanced + persistent signal + multi-bar labels ⇒ neighbouring
> observations share realised future ⇒ a plain k-fold leaks → **purged + embargoed combinatorial CV
> (CPCV)**.

---

## Section 2 — Triple-Barrier Labeling & Meta-Labels

The primary signal supplies the **side**; the meta-label answers *was taking this bet profitable?*
Each event's forward path is scanned in **trading-bar order** and the **first** barrier touched wins
(PT → 1, SL → 0, vertical → labeled by return sign). Barriers are σ-scaled with `σ = f2_vol_20`
**de-annualised by √252**.

### 2.1 Barrier-geometry selection — active barriers only, four horizons

Each `(pt, sl, h)` is scored by the **mean purged-CV AUC of a fixed shallow XGB** (a cheap constant
probe, so we measure *label learnability* — not joint barrier+model overfit), then picked on a
**robust plateau** (itself + grid neighbours at the same `h`). **Both barriers must be active**
(`pt>0` *and* `sl>0`): `search_barriers` now refuses a disabled barrier, because `pt=0` makes the
upper barrier `+∞` (`labels.py:118`), collapsing the label to ≈ sign-of-forward-return — the *most
learnable* and so artificially highest-AUC target (the trap the earlier run fell into).

| pt | sl | h | n | minority | pos_rate | AUC | AUC std | plateau |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.5 | 5 | 3924 | 0.445 | 0.555 | 0.5424 | 0.019 | 0.5503 |
| 1.0 | 1.0 | 5 | 3924 | 0.448 | 0.552 | 0.5582 | 0.016 | 0.5539 |
| 1.5 | 1.5 | 5 | 3924 | 0.444 | 0.556 | 0.5610 | 0.022 | 0.5555 |
| 2.0 | 2.0 | 5 | 3924 | 0.441 | 0.559 | 0.5524 | 0.030 | 0.5567 |
| 1.0 | 1.0 | 10 | 3880 | 0.452 | 0.549 | 0.5452 | **0.006** | 0.5448 |
| 2.0 | 2.0 | 10 | 3880 | 0.439 | 0.561 | 0.5435 | 0.058 | 0.5478 |
| **1.0** | **2.0** | **15** | 3841 | 0.346 | 0.654 | **0.5685** | 0.049 | **0.5619** ← chosen |
| 1.5 | 1.5 | 15 | 3841 | 0.445 | 0.555 | 0.5553 | 0.051 | 0.5481 |
| 2.0 | 2.0 | 20 | 3801 | 0.448 | 0.553 | 0.5565 | 0.082 | 0.5570 |
| 1.0 | 2.0 | 20 | 3801 | 0.343 | 0.657 | 0.5551 | 0.044 | 0.5563 |

*(10 of 24 rows shown; full grid in cell 25. No config was `skipped` — the grid is all-active; the
guard is a safety net.)* **Chosen (robust plateau): `pt=1.0, sl=2.0, h=15`** — plateau 0.5619.

**The single most important caveat.** The AUC differences across configs are **~0.01–0.02**, but the
per-config noise (`auc_std`) is **0.006–0.08** — *the noise is bigger than the signal*. No horizon or
geometry is statistically distinguishable. So the geometry is chosen on **validity + robustness**, not
the AUC peak; chasing the peak is exactly what produced the earlier degenerate `(0,2)` selection.

**Honest flag on `(1,2,15)`.** It is **asymmetric** — profit-take at 1σ, stop at 2σ. The near profit
target is hit more easily, so the class balance tilts to **~65% positive** (`pos_rate 0.654`,
`minority 0.346`). That is a deliberate risk/reward choice (take profit early, give the downside room),
but note it pre-supposes an asymmetry, and the high base rate leaves a *blind* primary already ~65%
precise — i.e. little room for a meta-filter to add value (borne out in §8).

### 2.2 Vertical-barrier variant — now non-degenerate

Labeling time-outs by `sign(ret)` vs `0`: `pos_rate 0.654` vs `0.639`. The two are now **close** —
unlike the old `(0,2)` geometry where `vertical_zero=True` collapsed to `pos_rate 0.000`. That gap
closing confirms the barrier is a *real* triple barrier (the profit barrier actually fires). We keep
the `sign(ret)` default.

### 2.3 Uniqueness

**Effective N ≈ 865 of 3 841 raw events (22.5% average uniqueness)** — higher than the old run's
13.3% because the close 1σ profit target resolves many events quickly, shortening the `[t0,t1]` label
spans and reducing overlap. Weights are applied everywhere.

---

## Section 3 — Validation Framework (CPCV)

| Parameter | Value |
|---|---|
| `(N, k)` | **(6, 2)** → **15 models, 5 backtest paths** |
| Purge / embargo | `h=15` bars / per-instrument `embargo_p90` |

**Logistic baseline under CPCV: mean path-AUC = 0.528 ± 0.008** over 5 paths. Design matrix
(3 841 × 114), label-1 rate **0.654**. The tight ±0.008 spread is the quantity Sections 4–6 optimise
and Section 8 reports; a single walk-forward path could not rank configs this finely.

---

## Section 4 — Regression Family (penalised logistic) + calibration

| Quantity | Value |
|---|---|
| Best config | **L1**, `C ≈ 21.4` (weak regularisation) |
| Mean path-AUC (± std) | 0.5298 (± 0.0110) |
| OOF AUC | **0.5360** |
| Brier (raw → calibrated) | 0.3198 → 0.2228 (isotonic) |
| Non-zero coefficients | **109 / 114** |

**Top surviving coefficients** (|coef|, standardised):

| feature | \|coef\| |
|---|---|
| f11_be10y_chg20 | 1.481 |
| f11_be10y_level | 1.187 |
| f11_tips10y_chg20 | 1.145 |
| f11_10y_ust_chg20 | 0.856 |
| f13_mra_energy_d1 | 0.769 |
| f11_spread_curve_slope_level | 0.628 |
| f11_spread_curve_slope_chg20 | 0.609 |
| f13_mra_energy_d3 | 0.600 |
| f11_10y_ust_level | 0.586 |
| f12_ma21_slope | 0.548 |

**Interpretation → mitigation.** Under the longer `h=15` horizon the linear model leans on the
**cross-asset macro / rates block (F11 — breakevens, TIPS, USTs, curve slope)** and the
wavelet-energy features (F13), rather than the signal's own short-horizon state. That is economically
sensible — a 3-week hold is exposed to the rates/macro regime — but the raw Brier (0.32) shows the
uncalibrated logistic is **badly mis-scaled** here (it over-predicts the 65% base rate), corrected to
0.223 by isotonic before any probability is used. OOF AUC 0.536 is barely above chance.

---

## Section 5 — Tree Family (RF / XGBoost / LightGBM / AdaBoost) + calibration

| model | OOF AUC | AP | Brier raw → cal | calib |
|---|---|---|---|---|
| **XGB** | **0.5445** | **0.7037** | 0.2520 → 0.2233 | isotonic |
| RF | 0.5421 | 0.6758 | 0.2317 → 0.2239 | isotonic |
| LGBM | 0.5362 | 0.6787 | 0.2499 → 0.2246 | isotonic |
| ADA | 0.5204 | 0.6701 | 0.2402 → 0.2252 | isotonic |

**XGB is the within-family champion** (OOF AUC 0.5445). The high AP (~0.70) just reflects the 65%
positive base rate (AP ≈ base rate for a weak ranker), so AP is *not* evidence of skill here — AUC is
the honest read, and it is ~0.54. Boosting's raw probabilities are over-confident (Brier 0.252) and
RF under-confident; both corrected on OOF by isotonic (≈0.224). Tuned configs already pull the
regularisers (shallow depth, `min_child_weight`, `subsample`, `reg_*`).

---

## Section 6 — Neural-Net Family (MLP + VSN) + calibration

| model | OOF AUC | Brier raw → cal |
|---|---|---|
| **MLP** | **0.5526** | 0.2368 → 0.2226 (isotonic) |
| VSN | 0.5019 | 0.2412 → 0.2253 (isotonic) |

The tabular **MLP is the highest-AUC family overall (0.5526)** — it becomes the cross-family champion
(§7–§8). The **VSN sits at chance (0.502)**; on this thin, high-base-rate panel its softmax feature
gates have little to latch onto. **LSTM/TFT** are documented P2 deferrals (`nn_seq.py` unbuilt;
`neuralforecast` uninstalled) — not silent caps. Mitigation: Dropout + BatchNorm + early stopping +
weight decay + capped epochs.

> **Champion caveat:** MLP (0.5526), XGB (0.5445) and the logistic (0.536) are all within ~0.02 AUC of
> each other and within noise of 0.50. "Champion" here means *least-bad on dev OOF*, not *good*.

---

## Section 7 — Cluster-Level Feature Importance (10-mark section)

Importance is decomposed over the **redundancy clusters** (not individual features, so collinear groups
don't split their signal), on the champion's columns. Overall champion = **MLP (OOF AUC 0.5526)**. Of
its 114 model features: **99 map to clusters + 15 one-hot dummies**; **7 representative-pruned clusters
are reported, never silently zeroed** (`[5, 10, 17, 27, 28, 41, 55]`). Clustered MDI/SHAP use a tree
(XGB) since the MLP is not tree-based; clustered MDA is model-agnostic on the champion.

**Single-feature importance** — each feature's standalone OOF AUC (light logistic probe):

| feature | SFI AUC |
|---|---|
| f5_signal | 0.5645 |
| f12_ma21_slope | 0.5563 |
| f17_hmm_state_lo | 0.5510 |
| f1_ret_reversal_40 | 0.5477 |
| f12_trend_tval_42 | 0.5465 |
| f17_hmm_state_argmax | 0.5439 |
| f9_xsect_rank | 0.5420 |
| f10_oto_ret_mean_20 | 0.5408 |

**Noise-injection sanity check (the required guard): PASS.** **88 real features outrank every injected
noise column** (of 114); max noise ΔAUC = **0.0004**. Economically-motivated features (the signal
state, counter-trend `f1_*`, MA-slope/trend `f12_*`, regime `f17_*`, cross-sectional rank `f9`) sit at
the top; deliberate noise sinks to the bottom. **Iterative cluster-drop** (→ Feature Track A) shows
dropping low-importance clusters does not degrade dev AUC. **Track B (PCA)** is a documented P2
deferral.

---

## Section 8 — Cross-Family Evaluation & Deliverable

The held-out **test** partition is opened **exactly once** (`release_test`, `final_confirmation=True`):
**888 labeled test events, label-1 share 0.695**. EV threshold `p* = L/(G+L)` with bootstrapped
`r_G = 0.0327`, `r_L = −0.0465` ⇒ **`p* = 0.587`**.

### 8.1 Cross-family comparison (OOF vs held-out test)

| family | model | OOF AUC | OOF F1 | OOF Brier | **test AUC** | test precision | test recall | test F1 |
|---|---|---|---|---|---|---|---|---|
| linear | LogReg | 0.5486 | 0.7872 | 0.2228 | 0.4827 | 0.6948 | 1.000 | 0.8199 |
| tree | XGB | 0.5533 | 0.7908 | 0.2233 | 0.4980 | 0.6948 | 1.000 | 0.8199 |
| **neural** | **MLP** | **0.5623** | 0.7890 | 0.2226 | 0.4906 | 0.6934 | 0.8687 | 0.7712 |

*(OOF AUCs here are recomputed on the **calibrated** OOF, so they differ by a hair from the raw §4–§6
numbers.)* **Test AUC is ≈ 0.48–0.50 for all three families — at or below chance.** At `p*=0.587` the
linear/tree models gate *nothing* on test (recall 1.000, precision = the 0.695 base rate); only the
MLP gates a little (recall 0.869). **The meta-model does not generalise to the 2022 holdout** —
honestly stated, small sample (888 events) + 2022 regime shift.

### 8.2 Scope ladder (champion, CPCV)

| scope | n | AUC |
|---|---|---|
| pooled | 3 841 | 0.5526 |
| class: EN (energy) | 924 | 0.5277 |
| class: EQ (equity) | 1 406 | 0.5409 |
| class: ME (metals) | 1 511 | 0.5220 |

All classes cluster at **0.52–0.54** — uniformly weak (unlike the old run where metals stood out as
near-chance; now *everything* is near-chance).

### 8.3 Adding-zeros — *does the meta-filter beat the blind primary?*

Gate the calibrated champion (MLP) at `p*`:

| strategy | precision | recall |
|---|---|---|
| primary alone (blind) | 0.654 | 1.000 |
| **primary + meta** (`p̂ ≥ p*`) | **0.661** | 0.979 |

- **Precision lift: +0.007** · 65 false positives avoided · 52 true positives missed.
- **NAV: +20.33 → +21.32** · **Sharpe: 1.65 → 1.77**.

**This is the headline honest result.** Under a *valid* triple barrier the meta-filter's edge is
**marginal** (+0.7pp precision, a small Sharpe bump) — a far cry from the earlier degenerate run's
+4.7pp. With a 65% positive base rate the blind primary is already mostly right, so there is little for
the filter to improve. The NAV/Sharpe still tick up (it removes a few genuine losers), but the effect
is small and does **not** survive to the test set (§8.1).

### 8.4 Per-instrument vs blind primary (holdout view)

| instrument | n | base-rate | prec. primary | prec. meta | prec. lift | NAV primary | NAV meta | Sharpe primary | Sharpe meta |
|---|---|---|---|---|---|---|---|---|---|
| cl1s | 318 | 0.720 | 0.720 | 0.719 | −0.001 | 7.077 | 7.049 | 4.917 | 4.905 |
| es1s | 445 | 0.685 | 0.685 | 0.691 | +0.006 | 1.586 | 1.624 | 1.952 | 2.017 |
| fesx1s | 495 | 0.640 | 0.640 | 0.657 | **+0.016** | 1.614 | 1.857 | 1.707 | 2.056 |
| gc1s | 138 | 0.725 | 0.725 | 0.725 | 0.000 | 0.809 | 0.809 | 3.808 | 3.808 |
| hg1s | 488 | 0.635 | 0.635 | 0.658 | **+0.023** | 1.443 | 1.805 | 1.579 | 2.172 |
| ho1s | 50 | 0.800 | 0.800 | 0.800 | 0.000 | 1.159 | 1.159 | 7.954 | 7.954 |
| ng1s | 68 | 0.779 | 0.779 | 0.779 | 0.000 | 2.179 | 2.179 | 5.929 | 5.929 |
| nq1s | 466 | 0.655 | 0.655 | 0.659 | +0.004 | 1.709 | 1.700 | 1.784 | 1.805 |
| pl1s | 437 | 0.677 | 0.677 | 0.677 | 0.000 | 2.785 | 2.785 | 2.084 | 2.084 |
| rb1s | 488 | 0.580 | 0.580 | 0.579 | −0.001 | −0.109 | 0.278 | −0.042 | 0.113 |
| si1s | 448 | 0.612 | 0.612 | 0.612 | +0.001 | 0.077 | 0.077 | 0.053 | 0.051 |

**Interpretation.** The filter is **mostly inert** — it passes nearly everything (the thin/high-base
names `gc1s`, `ho1s`, `ng1s`, `pl1s` are untouched). The only meaningful gains are `hg1s` (+0.023) and
`fesx1s` (+0.016); `rb1s` flips from a small loss to a small gain. No instrument is materially hurt.

### 8.5 Deliverable CSV

| field | value |
|---|---|
| rows | **1 408** = 128 dates × 11 instruments |
| schema | `date, instrument, prediction` |
| prediction range | [0.000, 1.000] |
| non-zero predictions | 1 010 (zero-signal days → prediction 0) |
| champion shipped | **MLP**, isotonic-calibrated |

The champion (MLP, fit on all dev) is applied to the held-out test features, isotonic-calibrated, then
melted onto the full signal grid. **The identical code path reruns on the hidden H2-2022 by swapping
only the date window.** *Note:* isotonic calibration on a small OOF set is a coarse step function, so
many predictions collapse to identical plateau values (e.g. several names at 0.606 on 2022-01-03) and
the range touches exactly {0, 1}; Platt would give smoother probabilities if smoothness matters
downstream (e.g. for sizing).

---

## Conclusions & limitations

- **The corrected result is honest and weak — and that is the finding.** Fixing the degenerate `pt=0`
  label dropped OOF AUC from ~0.58–0.60 to **~0.52–0.55** and the precision-lift-vs-blind-primary from
  +4.7pp to **+0.7pp**. The earlier "edge" was largely an artifact of a label that was ≈
  sign-of-forward-return. Under a valid triple barrier the meta-model barely beats following the
  primary blindly, and **does not generalise to the 2022 test set** (test AUC ≈ 0.48–0.50).
- **Why so little to gain.** The chosen `(1,2,15)` geometry has a **65% positive base rate**, so the
  blind primary is already mostly profitable — the meta-filter has little headroom. A symmetric
  geometry (≈50% base rate) would give the filter more to do; that is a defensible alternative to
  revisit.
- **Methodology over performance** (the graded axis): leakage-disciplined end-to-end (TF features
  frozen 2021-07-01, CPCV purge+embargo per instrument, uniqueness weights per fold, calibration and
  `p*` on dev OOF only, test opened once); barriers picked on a **robust plateau of *valid* geometries**
  with the disabled-barrier trap now guarded in code; importance done at cluster level with a passing
  noise-injection sanity check.
- **Deflation caveat.** Optuna budgets (10/4/3 trials) and NN epochs are capped for the deadline; with
  AUC differences inside the `auc_std` noise, **no single AUC is load-bearing** — the relative family
  ordering and the (small) precision lift are the real results.
- **Out of scope (P2 / future).** LSTM/TFT sequence NNs, PCA feature Track B, larger CPCV `(8,2)`,
  probability *sizing* (guide Part 4), and a symmetric-barrier comparison — all documented deferrals.

---

## For the submission decision

The notebook ships the **calibrated MLP** (highest dev OOF AUC, 0.553–0.562). For the team's "choose one
model" decision I'd present it this way:

1. **All three families are statistically tied** (OOF AUC: MLP 0.553, XGB 0.544, LogReg 0.536 — within
   the path-AUC noise) and **none beats chance on the 2022 holdout**. So the choice is *not* about
   performance.
2. **If performance-on-paper is the tiebreak → MLP** (what the CSV currently uses). **If
   interpretability/defensibility is the tiebreak → XGB**: it gives native MDI/SHAP for the importance
   section, calibrates cleanly, and is within 0.01 AUC of the MLP — a more *showable* champion for a
   methodology-graded submission.
3. **The case to make is the methodology, not the number:** we caught and fixed a labeling trap that
   was silently inflating the result, we validate under CPCV with full leakage discipline, and we
   **report honestly that the meta-model's edge is marginal and does not hold out-of-sample.** Per the
   brief ("a high mark is achievable even if the metamodel never beats the primary signal"), that
   transparency is the submission's strength.
