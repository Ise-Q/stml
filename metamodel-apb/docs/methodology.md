# Alken Meta-Labelling Metamodel — Methodology (Sections 1–6 + bonus)

**Course:** T3.03 Systematic Trading Strategies with ML — Alken team challenge.
**Scope:** a secondary *act/skip* classifier over the provided primary signal, across **11
futures instruments in 3 asset-class metamodels** (Equity / Energy / Metals).
**Grading stance:** *methodology, not performance* (`../refs/project-instructions.md`). Every
choice below is justified against the literature review `../reports/apb/nlr-cw-v1.md` (8
commitments, 60 references), the feature set is locked before the final OOS window, and the
results section reports honestly where the metamodel **does not** beat the blind-primary baseline.

> **Reproduce:** `uv run --directory metamodel-apb python -m alken_metamodel.emit
> --asset-classes equity energy metals`. Deterministic (seeds fixed; single-thread native
> kernels); the prediction window is config-driven so the grader can swap in the hidden
> Jul–Dec 2022 half. `uv run --directory metamodel-apb pytest` → full test suite.

## Scope — the shipped default path (pass 2)

Every number in this document comes from the **default `emit` path**, which pass 2 promoted to the
full methodology stack. Per asset class:

- a **five-estimator horse-race** — elastic-net logistic, XGBoost, LightGBM, **torch-MLP and
  torch-VSN** (the byte-deterministic NN variants) — where the two neural variants run on a
  **cluster-representative-reduced** feature set (one medoid per Mantegna §4 cluster) so the VSN's
  one-GRN-per-feature cost is tractable;
- selected by **Combinatorial Purged CV (15 paths)** — CPCV is now the *selection* splitter, not
  just a diagnostic — by mean OOS AUC;
- trained on the pooled causal feature matrix **including the EWMA-HMM + static regime blocks
  (`use_regime=True`) and the PIT-lagged macro block (`use_macro=True`)**, then refit and used to
  predict the config window.

Triple-barrier labelling, uniqueness weights, fractional-Kelly + vol-target sizing, the §4 cluster
importance (now run on the real matrix), and the barrier-exact §6 backtest are all in this path.

**Determinism reconciliation (why Keras-VSN is the one variant left off-path).** TensorFlow op-
determinism is best-effort, so a selectable Keras-VSN could be picked by the grader's hidden-half
re-run yet fail to reproduce — violating the determinism contract. It is therefore kept as an
**off-path documented comparison** (`roster="full"`); the torch variants are byte-stable and
selectable. The **autoencoder reducer** is likewise built for the EX.2 comparison only;
cluster-representative selection is the promoted reducer (deterministic, interpretable, tied to §4).

> Selected models this run (CPCV selection): **Equity → XGBoost** (0.579), **Energy → torch-MLP**
> (0.525 — a neural variant won), **Metals → elastic-net logistic** (0.530).

---

## 0. Architecture (one-directional data flow)

```
load_clean_data()                              stml.io          OHLCV (long) + signals (wide), read-only
  → per-instrument causal features             features.py      stml E-class stack + backward-trend feature
  → regime features                            regime.py        online EWMA 2-state HMM + stml static GMM/Markov/HMM
  → triple-barrier meta-labels (signal≠0 days) triple_barrier.py vol-adaptive ±k·σ̂ₜ + vertical T_max, record t1, uniqueness w
  → pool the class (+instrument-id one-hot)    pipeline.py      event-date index kept (cross-instrument purge)
  → purged-CV horse-race → SELECT              models.py + cross_validation.py + evaluation.py
  → refit on the locked modelling sample, predict P(act) on the config window
  → metamodel_predictions.csv                  emit.py          deterministic, byte-identical re-emit
  → fractional-Kelly + vol-target sizing       sizing.py        signed by primary side
  → strategy_weights.csv  → backtest           emit.py + backtest.py
cluster_importance.py  (§4 diagnostics, off the critical emit path)
```

**Leakage is the central design constraint.** We reuse stml's *causal feature functions*
recomputed per instrument, never the frozen `results/feature_matrix.parquet` (which freezes
fitted stats at one global `fe_train_end` and would leak into in-sample folds). Every stml
`assemble_*` is stateless and proven causal by **right-edge truncation-invariance**: a feature's
value at `t` is identical computed on `data[:t+1]` or on the full series — this is encoded as a
property test (`tests/test_features.py::test_right_edge_truncation_invariance`).

---

## 1. Feature engineering (§1, 20 marks) — `features.py`, `regime.py`

The pooled feature matrix (**measured per class, not the stale "124"** — X.8): the full shipped
pooled panel is **Energy 140 / Equity 139 / Metals 140 columns** (core engineered ≈115 + EWMA-HMM
regime 5 + PIT-macro 16 + instrument one-hots 3–4); the §4 clustering matrix, which drops the
regime/macro blocks and zero-variance columns on the modelling slice, is **108 / 107 / 108**. Both
are measured by `experiments/x8_feature_counts.py`. The layers:

| Block | Columns (examples) | What it captures | Module / source |
|---|---|---|---|
| F1 counter-trend | `f1_mr_score_{10,20,40}`, `f1_rsi_14`, `f1_bb_pctb_20` | mean-reversion pressure | stml `features.py` |
| F2 vol/dispersion | `f2_vol_20` (annualised, LI), `f2_garman_klass_20`, `f2_parkinson_20` | heteroskedastic risk; **GK** range vol (Garman–Klass 1980; Korkusuz 2023, nlr-cw §A1) | stml + `volatility.py` |
| F2-RS | `f2_rogers_satchell_20` | drift-independent range vol | stml `features_ext.py` |
| F5 signal-derived | `f5_trailing_run_length`, `f5_participation_20`, `f5_signal_entropy_20` | primary-signal trajectory (LI handoff) | stml |
| F6 momentum | `f6_ts_momentum_{20,60}`, `f6_macd_12_26`, `f6_adx_14` | trend strength | stml |
| F7 microstructure | `f7_oi_change`, `f7_amihud_20`, `f7_kyles_lambda_20` | liquidity / OI flow | stml |
| F8 calendar | `f8_dow_sin/cos`, `f8_month_sin/cos` | seasonality | stml |
| F10 price-action | `f10_hl_range`, `f10_oto_ret` | intraday range / gaps | stml |
| F12/F13/F15 | `f12_hurst_100`, `f13_mra_energy_d1..d5`, `f15_prob_timeout` | path structure, wavelet energy, conditional first-passage risk | stml `features_ext.py` |
| **Trend feature** | `trend_tval_back`, `trend_sign_back`, `trend_window_back` | **backward** trend-scanning *as a feature, not the label* (L1; nlr-cw §3) | `features.py` (reuses vendored `tValLinR`) |
| z-twins | `z_<col>` for 24 scale-dependent cols | per-instrument causal expanding-window standardisation | stml `add_z_twins` |
| Regime | see §1-regime below | volatility-state context | `regime.py` |

**Backward trend feature (correctness note).** Trend scanning is mandated as a *feature*, never
the label. We reuse the trend-scanning algorithm with `look_forward=False`, but compute the OLS
slope t-statistic in closed form (`_segment_tval`, validated equal to the vendored `tValLinR` to
1e-9) and apply a **deterministic ±20 cap** instead of `trend_labels`' global-variance cap — that
global cap depends on the whole series and is itself a right-edge truncation leak.

**Regime features (commitment #8, nlr-cw §4 — Nystrup-Madsen-Lindström 2017).** stml ships only
*static* regime models (F3 = 2-regime GMM + Markov-switching; F17 = 3-state Gaussian HMM with a
causal forward filter). To honour the EWMA *time-varying* commitment we **built** an **online
EWMA 2-state Gaussian HMM** (`regime.py`): a forward-filtered HMM on daily log returns whose
emission means/variances are re-estimated online via a forgetting factor (EWMA of
responsibility-weighted sufficient statistics), with a fixed persistent transition prior (the
"penalising jumps" half of Nystrup). Because every parameter at `t` is a recursion over
observations `≤ t`, it is **causal/fit-free** — no batch fit, no fit/transform split, hence no
per-fold CPCV seam artifact (concatenating non-contiguous CPCV train groups would fabricate fake
1-step transitions in a batch HMM). stml's static blocks are reused as supplementary features.
Emitted columns: `ewma_hmm_prob_highvol`, `ewma_hmm_state`, `ewma_hmm_var_hi/lo`,
`ewma_hmm_switch_prob`, plus `f3_*` and `f17_*`.

**PIT-lagged macro block (`additional_data.xlsx`, `macro.py` — now in the shipped path).** The
workbook ships 22 mixed-frequency macro series carrying **observation dates only**, so a naive
join would let a trade-day feature read an unreleased number. Each series is assigned a
conservative **publication lag** (daily market series +1 day; EIA weekly inventories +5; PMIs
+30), its observation index is shifted observation→availability, and it is forward-filled onto the
trade calendar — so feature `t` uses only data **released ≤ t** (point-in-time correct, proven by
a `pit_align` release-deferral test and block-level **truncation-invariance**). The 16 derived
drivers: Energy → EIA crude/distillate/gasoline/NG inventory changes; Metals → gold↔real-rates
(`TIPS10Y`,`BE10Y`), `DXY`, copper↔`CHINA_PMI` + `LME_COPPER_STOCK` change; cross-asset → VIX
term-slope (`VIX3M−VIX`), `MOVE`, credit `HY/IG_OAS` + spread. **Flagged as NOT derivable:**
per-instrument calendar/basis spreads — OHLCV ships only the front-month `*1s` contract (no second
maturity), so this is correctly omitted rather than fabricated.

---

## 2. Labelling (§2, 20 marks) — `triple_barrier.py`

Meta-labels in {0,1} ("act"/"skip") are assigned **only on the non-zero-signal trade days** via
the triple-barrier method (López de Prado 2018 Ch.3, nlr-cw §1):

- **Vol-adaptive symmetric barriers** `±k·σ̂ₜ` where `σ̂ₜ` is the de-annualised Garman–Klass-based
  daily vol (`f2_vol_20 / √252`) — fixed-% thresholds ignore the heteroskedasticity of returns and
  make the label distribution pro-cyclical. **Vertical barrier** `T_max` (default 10 bars) bounds
  the horizon.
- The label is the sign of the side-adjusted P&L at the **first** barrier touched; **`t1`
  (first-touch time) is recorded for every label** and drives purge/embargo everywhere.
- **Sample-uniqueness weights** from label concurrency (LdP Ch.4) down-weight labels whose
  horizons overlap — triple-barrier labels are not iid. Verified exactly (disjoint → 1.0;
  fully-overlapping → 0.75/0.50) in `tests/test_triple_barrier.py`.

Per-instrument class balance ranges ~50–69% positive; see the §5 per-instrument table.

---

## 3. Models (§3, 30 marks) — `models.py`, `neural.py`, `cross_validation.py`, `evaluation.py`

A horse-race behind one uniform `MetaClassifier` interface so the comparison is apples-to-apples
(Gu-Kelly-Xiu 2020; Krauss et al. 2017; IKM 2020 small-data restraint, nlr-cw §2). The **shipped
default roster is five estimators**; a sixth (Keras-VSN) is an off-path determinism-safe
comparison (see *Scope*):

1. **Elastic-net logistic** (saga; median-impute + standardise),
2. **XGBoost** (PS5 config), **3. LightGBM** — on the full feature set,
4. **torch-MLP**, **5. torch-VSN** (byte-deterministic VSN port with softmax feature-selection
   weights) — on the **cluster-representative-reduced** feature set (EX.2: one medoid per Mantegna
   cluster), which makes the VSN's one-GRN-per-feature architecture tractable at CV scale.
6. **Keras-VSN** (vendored PS6 `FinalModel`) — off-path, TF op-determinism best-effort.

Reduction is **fold-safe**: the reducer is wrapped with its estimator so the evaluation harness
fits the medoid selection on each fold's train rows only — the reduced basis never sees the
validation fold.

**One weighting channel.** PS4/5/6 ship no weighting; meta-labels are both *overlapping* (need
uniqueness weights) and *imbalanced* (~30–40% positive). Both are folded into a single
`sample_weight = uniqueness × inverse-class-frequency` passed identically to every estimator's
`fit` *and* into every OOS metric.

**Validation (commitment #3, nlr-cw §6 — LdP Ch.7/12; Bailey 2014; Harvey-Liu-Zhu 2016).**
`cross_validation.py` provides **PurgedKFold + embargo ⌈0.01·T⌉**, **CPCV (N=6, k=2 → 15 paths)**,
and **nested CPCV**. In the shipped run, **model selection uses CPCV** (the 15-path mean OOS AUC);
a real-data check confirmed **all five estimators yield 15/15 *finite* CPCV paths on Energy**, so
selection is not degenerating to a couple of lucky paths. **Nested CPCV**
(`nested_cpcv_select_and_evaluate`: inner CPCV horse-races the roster, outer CPCV scores the
winner) is **implemented and unit-tested** as the selection-bias-aware evaluator, but a **full
real-data nested run is not executed here** — a meaningful one runs the inner horse-race over all
five estimators per outer fold (~15×5×5 fits) and is flagged as deferred (alongside the S6.5 stub
and X.7). Selection is by mean OOS AUC; **calibration** is reported alongside because Kelly sizing
consumes the
probability itself (Gramegna-Giudici 2021, nlr-cw §2 — see the calibration subsection). The pooled
matrix keeps the event-date index so concurrent **cross-instrument** labels are purged by `t1`.

**Selected models (CPCV selection, real data):** Equity → **XGBoost** (15-path mean AUC 0.579),
Energy → **torch-MLP** (0.525 — a neural variant won its class), Metals → **elastic-net logistic**
(0.530). The torch NN family is now genuinely competitive, not a synthetic-only appendix.

**CPCV path robustness (EX.1, modelling sample).** The fraction of the 15 purged combinatorial
paths beating 0.5 is the discriminating signal for *where* the edge is real:

| Class | best model | mean CPCV AUC | paths > 0.5 | reading |
|---|---|---|---|---|
| Equity | XGBoost | **0.572** | **15 / 15** | edge is **robust** |
| Metals | logistic | 0.524 | 13 / 15 | marginal-but-positive |
| Energy | LightGBM | 0.493 | 6 / 15 | **no reliable edge** |

Equity's edge survives every combinatorial path; Energy's does not (6/15 ≈ coin-flip),
corroborating the §4 cluster importance and the deflated-Sharpe / multiple-testing caveat
(Bailey 2014; Harvey-Liu-Zhu 2016).

**Calibration is now shipped (S3.9, `calibration.py`).** Because Kelly sizes on p̂ directly, a model
that *ranks* act/skip adequately still mis-sizes if p̂ is uncalibrated. Pass 3 fits **one Platt map
per asset class** on the *selected* model's purged-OOS modelling predictions (strictly before
`predict_start`, so it cannot leak), and applies it to **both** the deliverable probabilities **and**
the Kelly stake. Platt is monotone, so **AUC is unchanged** (the act/skip ranking is preserved — a
unit-tested invariant); only Brier/ECE and the stake move. On a leakage-safe in-time 70/30 split of
the **selected** models (not LightGBM-as-proxy as in pass-2's EX.4):

| Class | Selected model | ECE raw → Platt | Brier raw → Platt | AUC (raw = Platt) |
|---|---|---|---|---|
| Energy | torch-MLP | 0.140 → **0.100** | 0.254 → 0.244 | 0.541 |
| Equity | XGBoost | 0.055 → **0.027** | 0.247 → 0.242 | 0.607 |
| Metals | logistic | 0.210 → **0.001** | 0.313 → **0.247** | 0.532 |

The raw probabilities are materially miscalibrated (Energy's torch-MLP worst at ECE 0.140 — NNs
need post-hoc calibration more than trees, Gramegna-Giudici 2021); Platt cuts ECE 1.4–200× with the
AUC untouched. The **deliverable ships the calibrated probabilities** (`metamodel_predictions.csv`),
with the raw file retained (`metamodel_predictions_raw.csv`) for this before/after; the
`experiment_log.csv` now records the calibrated class-level Brier (Equity 0.249, Energy 0.263,
Metals 0.344) and precision. **Note:** calibration *does* change §6 — it shifts which positions clear
the p̂≥0.55 Kelly floor, which moves the per-book Sharpes (Metals especially; see §6).

**XGBoost benchmark (LR.4).** The horse-race is the NN-vs-tree benchmark itself: torch-MLP and
torch-VSN compete directly against tuned XGBoost and LightGBM under identical CPCV. Equity and
Metals select tree/linear; **only Energy selects a neural variant** — so the NN family is neither
rubber-stamped nor excluded, and the comparison is reported, not assumed.

**CPCV small-N variance (documented cost).** The 15 combinatorial paths give short per-path test
folds, so a single path's AUC is noisy; selection is therefore on the **15-path mean**, and the
path-count-above-0.5 (below) is read as the robustness signal rather than any one path. The variance
is the price of the combinatorial check — accepted deliberately over a single train/test split.

---

## 4. Feature importance (§4, 10 marks) — `cluster_importance.py`

Substitution effects make per-feature importance unreliable under correlation (LdP 2020 Ch.6), so
importance is scored **per cluster**: features are clustered by **Mantegna distance** → PCA →
optimal-K K-means, then scored by **cluster MDI + purged cluster MDA + cluster SHAP**. This module
carries the **four required bug fixes**, each visible in the diff:

| # | Bug | Fix | Where |
|---|---|---|---|
| 1 | `max_features='auto'` (PS4 grid; removed in sklearn ≥1.3) | `'sqrt'` | `cluster_importance.py` MDI/SHAP forest |
| 2 | `KFold(shuffle=True)` for MDA (leaks across overlapping labels) | injected **PurgedKFold** | vendored `calculate_cluster_importance_pfi` |
| 3 | no real SHAP in PS2/sts-ml (MDI+PFI only) | **cluster SHAP** via `TreeExplainer`, summing member \|SHAP\| (the §4 contribution) | `cluster_importance.py` |
| 4 | Spearman distance `1−\|ρ\|` (non-metric) | **Mantegna** `√(1−\|ρ\|)` (Mantegna 1999) | vendored `compute_spearman_distance_matrix` |

**Run on the real matrix (S4.7/S4.8, re-executed).** Clustering each class's modelling matrix
(median-imputed, zero-variance dropped → **Energy 108 / Equity 107 / Metals 108** feature columns)
and scoring every cluster:

| Class | clusters | top cluster MDA | top cluster SHAP | near-zero-MDA clusters |
|---|---|---|---|---|
| Equity | 3 | **0.031** | 0.43 | 2 / 3 |
| Energy | 3 | 0.011 | 0.40 | 3 / 3 |
| Metals | 2 | −0.004 | 0.71 | 2 / 2 |

The honest reading (S4.8): **MDI and SHAP are in-sample attribution** — they always split 100% of a
fitted model's importance across the clusters, so a high MDI/SHAP says only *which* features the
model leaned on in-sample, not that those features carry OOS edge. **Cluster permutation MDA is the
OOS reality check, and it is near-zero across the board** (every cluster `|MDA| < 0.02` bar one) —
exactly what a ≈0.5-edge problem should look like, and a useful negative result. So the Metals SHAP
of 0.71 concentrated in one cluster must **not** be read as edge: its MDA is −0.004 (negative). The
**only materially positive cluster MDA is Equity's 0.031**, matching its 15/15 CPCV robustness;
Energy/Metals clusters add nothing under permutation. The MDI/SHAP-vs-MDA divergence *is* the §4
lesson — no single importance method is sufficient, and only the permutation (OOS) view is honest
about edge. The synthetic noise-cluster≈0 sanity test still holds.

---

## 5. Evaluation (§5, 20 marks) — `evaluation.py`, real OOS results

Metrics are **sample-weighted**, **threshold-aware**, and computed **per-instrument before any
aggregate** (so a strong pooled number can't hide a weak member). Single-class purged folds yield
NaN ranking metrics rather than crashing. The baseline is **blind-primary** (act on every signal).

**Per-instrument purged-OOS AUC (shipped default path, 11-instrument fan-out):**

| Class | Model | Per-instrument AUC (n labels) | vs blind-primary |
|---|---|---|---|
| Equity | XGBoost | es1s 0.60 (457) · nq1s 0.61 (482) · fesx1s 0.59 (510) | **beats** |
| Metals | logistic | hg1s 0.58 (504) · pl1s 0.50 (453) · si1s 0.50 (462) · gc1s 0.41 (138) | mixed |
| Energy | torch-MLP | cl1s 0.55 (334) · rb1s 0.50 (504) · ho1s 0.43 (61) · ng1s 0.35 (68) | mixed |

**OOS coverage caveat (S5.7 → widened S5.9).** All 11 instruments emit (no abstention), but the
thin-coverage flag is now `n_oos_rows < 60` **OR** an undefined information coefficient — which
flags **three** names, not just one: **ho1s = 2 rows** (and IC undefined — Spearman on 2 points is
meaningless), **gc1s = 30**, **ng1s = 56** (IC defined at 0.19 but on only 56 rows); the other eight
carry 88–127 rows. `coverage_caveat.csv` now records `n_oos_rows`, the per-instrument OOS `ic`,
`ic_undefined`, and `thin`, so the deliverable's thin support is explicit rather than implied
uniform. The per-instrument AUCs for ho1s/ng1s (≈60 *training* labels too) are small-sample noise.

**Primary-signal context (EX.5).** Characterising the *provided* signal sets the metamodel's
ceiling: directional hit-rates run **0.52–0.69** (gc1s strongest at 0.66, IC 0.21; rb1s weakest at
0.53), turnover 0.02–0.23 flips/day, with several names better in the low-vol regime. The base
signal is already decent, so the secondary act/skip filter has little headroom.

**Utility-aware evaluation (S5.8 — beyond AUC).** AUC says nothing about whether *acting* adds
economic value, so we add a **Henriksson–Merton (1981) market-timing test** on the OOS acted trades
(does the sized directional call time the realised move over `[date, t1)`?) and the **mean-variance
certainty-equivalent** of the strategy returns:

| Book | H–M hit | H–M z (p) | CER/day (γ=5) | reading |
|---|---|---|---|---|
| Energy | 0.510 | 0.28 (0.39) | +0.000186 | no timing skill (≈coin-flip) |
| Equity | 0.430 | −1.72 (0.96) | +0.000247 | **below-½** timing |
| Metals | 0.423 | −2.63 (1.00) | −0.000033 | **no positive** timing (most negative) |
| All-11 | 0.452 | −2.43 (0.99) | +0.000396 | no timing skill |

This is the sharpest honest-negative in the build: the acted trades show **no positive directional
timing** — hit rates sit *at or below* ½ everywhere (Metals the most negative, though on the same
thin ~6-month OOS we caution against over-reading, so this is read as "no skill", not a tradeable
anti-signal). The §6 barrier-exact Sharpes are therefore **not** evidence of directional skill; the
positive P&L comes from the exit asymmetry, vol-targeting and diversification (§6), not from calling
the move right. It substantiates
Equity-works / Energy-fails as **expected** — Equity is the only book with both AUC ≈ 0.60 and 15/15
CPCV robustness, yet even there H–M timing is sub-½ on the thin H1-2022 OOS, so no skill is claimed.

**Honest reading.** Classification-wise the metamodel adds clear value only on Equity (all three
names AUC ≈ 0.60, consistent with 15/15 CPCV paths); Metals and Energy are mixed and dragged by
small-sample names (gc1s 138, ho1s 61, ng1s 68 labels). A pooled AUC would have hidden both. Mean
OOS AUC ≈ 0.5 overall is the expected, gradeable result — meta-labelling on a decent primary signal
is genuinely hard (the identical harness scores AUC > 0.9 on separable synthetic data, so it
detects signal when it exists). A deflated-Sharpe / multiple-testing caveat applies (Harvey-Liu-Zhu
t>3, nlr-cw §6).

---

## 6. Strategy (§6, +10 bonus) — `sizing.py`, `backtest.py`

Position weight = **fractional Kelly** `κ·f*` (κ=0.25, floor p̂≥0.55) × **vol-target leverage**
(25% annualised), signed by the primary side (Kelly 1956; MacLean-Ziemba-Blazenko 1992;
Carver 2015, nlr-cw §7). The constraint set defaults to lit-review values behind a clearly-marked
stub (the 20 May constraints doc is not in the repo).

> **Headline (the strategy is NOT claimed to work).** The §6 Sharpe is selected from the
> horse-race, so before it can be read at all it must be deflated for selection bias. It **does not
> clear** the deployment gate (S6.8): the pooled net Sharpe of 1.55 carries a **Deflated Sharpe
> Ratio of only 0.39–0.49** (≪ 0.95) and a **Minimum Backtest Length of 2.1–3.1 years** against a
> ~0.5-year OOS window. Combined with AUC ≈ 0.5 (§3/§5) and **sub-½ Henriksson–Merton timing**
> (§5.8), the honest conclusion is that **no deployable edge is demonstrated**; the positive Sharpe
> is an artefact of the exit convention, vol-targeting and diversification, not act/skip skill.

**Barrier-exact, cost-aware OOS backtest, Jan–Jun 2022 (S6.7, calibrated sizing).** The position
exits on the **actual triple-barrier first-touch `t1`** (not a fixed `max_holding`), overlapping
labels are **netted**, a Grinold–Kahn cost model (half-spread + impact) is charged, and the stake is
sized on the **calibrated** p̂ (§3.9). Net of costs:

| Book | Model | Sharpe | Sortino | Ann. vol | Max DD | Turnover/yr | Hold (d) | Gross→Net |
|---|---|---|---|---|---|---|---|---|
| **All 11** | — | **1.55** | 2.35 | 7.3% | −1.9% | 51.6 | 2.8 | +9.0% → **+5.7%** |
| Energy | torch-MLP | 1.75 | 3.01 | 2.8% | −0.9% | 8.2 | 2.9 | +3.0% → +2.5% |
| Equity | XGBoost | 1.36 | 2.06 | 5.0% | −1.5% | 18.4 | 2.4 | +4.6% → +3.5% |
| Metals | logistic | −0.11 | −0.15 | 3.9% | −3.1% | 25.0 | 2.9 | +1.2% → −0.3% |

These differ from pass-2's raw-sized numbers because **calibration moved the sizing**: shrinking
over-confident p̂ toward the base rate drops several positions below the 0.55 Kelly floor, halving
turnover (131 → 52/yr) and vol (20% → 7%), lifting Energy (1.48 → 1.75) but pushing **Metals
negative** (0.31 → −0.11) — an honest, reportable consequence of sizing on calibrated probabilities.

**S6.8 — deployment deflation gate (the decisive computation).** On the calibrated net returns,
deflating against the trial universe (DSR/MinBTL: Bailey–López de Prado 2014; PBO via CSCV:
Bailey-Borwein-LdP-Zhu 2017), reported as a **range over N ∈ [N_eff → N_raw]** because a single
backtest cannot pin the effective trial count (N_eff from ONC-clustering the trial-return matrix):

| Book | net Sharpe | DSR [N_eff→N_raw] | CSCV-PBO | MinBTL vs OOS≈0.5y | clears 0.95? |
|---|---|---|---|---|---|
| Energy | 1.75 | [0.85 → 0.77] (N 2→5) | 0.37 | [0.27 → 1.42]y | **no** |
| Equity | 1.36 | [0.77 → 0.64] (N 2→5) | 0.50 | [0.27 → 1.42]y | **no** |
| Metals | −0.11 | [0.31 → 0.15] (N 2→5) | 0.06 | [0.27 → 1.42]y | **no** |
| All-11 | 1.55 | **[0.49 → 0.39]** (N 8→15) | 0.39 | **[2.13 → 3.14]y** | **no** |

Even the *optimistic* end of every range stays below 0.95; the pooled MinBTL (2–3 years) dwarfs the
half-year OOS. This is the **expected, correct** honest-negative — the gate was built to report the
truth, not tuned to clear it.

**S6.9 — the holding model is load-bearing (the methodology finding).** Identical positions, models,
features and calibration — *only the exit convention differs*:

| Book | simple `max_holding=10` | barrier-exact (actual `t1`) |
|---|---|---|
| Energy | +2.22 | +1.75 |
| Equity | **−0.19** | **+1.36** |
| Metals | −0.28 | −0.11 |

The exit convention alone flips **Equity's sign** (−0.19 → +1.36) and dominates the per-book
ranking; this is attributed to the **exit mechanism** (winners ride to the profit barrier, losers
are cut at the stop), **not** classification skill — consistent with AUC ≈ 0.5 and the sub-½ H–M
timing. Gated by S6.8, it is a *finding about backtest construction*, not a performance claim.

**S6.10 — accounting reconciliation.** `net = gross − costs` holds daily, but the report's
`gross_total_return`/`net_total_return` are **compounded** (∏) while `total_cost` is an **arithmetic
Σ**, so `gross − total_cost ≠ net` by the compounding interaction (pooled: gross +9.0%, Σ-cost
3.0%, net +5.7%). The reconciliation field **`cost_drag_compounded` = gross_total − net_total**
(pooled **3.26%**) closes the identity exactly; `total_cost` (3.04%) is retained as the undiscounted
sum. Turnover (one-way annualised notional, 51.6) and holding (trade-based busday, 2.8d) are on
**different bases**, so `hold ≈ 252/turnover` is not expected to hold — both are reported, not
forced into a false identity.

---

## Commitments → modules → citations (Definition of Done)

| # | Commitment | Module | nlr-cw / primary citation |
|---|---|---|---|
| 1 | Meta-labelling act/skip filter | `triple_barrier.py`, `pipeline.py` | §1 — LdP 2018 Ch.3; Joubert 2022 |
| 2 | Vol-adaptive ±k·σ̂ₜ + vertical T_max | `triple_barrier.py` | §1 — LdP 2018 Ch.3 |
| 3 | Purged CV + embargo + CPCV + nested | `cross_validation.py` | §6 — LdP Ch.7/12; Bailey 2014; Harvey-Liu-Zhu 2016 |
| 4 | Garman–Klass vol (+Parkinson check) | `volatility.py` | §A1 — Garman-Klass 1980; Korkusuz 2023 |
| 5 | Multi-family horse-race | `models.py`, `neural.py` | §2 — Gu-Kelly-Xiu 2020; Krauss 2017; IKM 2020 |
| 6 | Cluster MDI+MDA+**SHAP**, Mantegna | `cluster_importance.py` | §5 — LdP 2020 Ch.6; Lundberg 2020; Mantegna 1999 |
| 7 | Fractional Kelly + vol-target | `sizing.py`, `backtest.py` | §7 — Kelly 1956; MacLean et al. 1992; Carver 2015 |
| 8 | 2-state HMM, **EWMA time-varying** | `regime.py` | §4 — Hamilton 1989; Nystrup et al. 2017; Ang-Timmermann 2012 |

**Cross-cutting:** uniqueness weights (LdP Ch.4); per-class Platt calibration of the deliverable +
Kelly stake (Gramegna-Giudici 2021); single sample-weight channel for imbalance + uniqueness;
**deflation gate** — Deflated Sharpe + MinBTL (Bailey-López de Prado 2014) + CSCV-PBO
(Bailey-Borwein-LdP-Zhu 2017) — so §6 is reported deflated, never as "the strategy works"
(`deflation.py`).

---

## Determinism & leakage discipline (how each is enforced + tested)

- **Determinism:** seeds fixed across `random`/`numpy`/`torch`/`tensorflow`/`PYTHONHASHSEED`
  (`seeding.py`); single-thread native kernels via `_env.py` (also fixes a macOS libomp
  segfault); CSV emitter sorts rows, pins columns, fixes float format → **byte-identical re-emit**.
  Verified on the **shipped default path** (torch NN family + CPCV selection + macro): two
  independent full `emit` runs produced **byte-for-byte identical** `metamodel_predictions.csv` and
  `strategy_weights.csv` (1011 rows each). TensorFlow op-determinism is best-effort, which is
  exactly why **Keras-VSN is kept off the selectable path** — the grader's hidden-half re-run would
  otherwise risk selecting a non-reproducible model.
- **No leakage:** right-edge truncation-invariance of the whole feature stack
  (`test_features.py`); zero train/test `t1`-overlap after purge, embargo = ⌈0.01·T⌉, CPCV = 15
  paths (`test_cross_validation.py`); fitted regime blocks fit on a contiguous prefix (never
  non-contiguous CPCV groups); the frozen `feature_matrix.parquet` is never consumed.
- **Config-driven window:** the prediction window is a `PipelineConfig` field, never hardcoded to
  Jan–Jun 2022 — the grader swaps in the hidden Jul–Dec 2022 half by changing one config value.

## Limitations (honest)

- **No deployable edge is demonstrated (the headline).** The §6 strategy **does not clear the
  deflation gate** (S6.8): DSR 0.39–0.49 pooled (≪ 0.95), MinBTL 2.1–3.1y against a ~0.5y OOS. This
  is consistent with mean OOS AUC ≈ 0.5, near-zero cluster MDA (§4), and **sub-½ Henriksson–Merton
  timing** (§5.8) — the positive Sharpe is diversification + vol-targeting + the barrier exit, not
  act/skip skill. Only Equity is even classification-robust (15/15 CPCV paths), and even it shows no
  directional timing on the thin H1-2022 OOS.
- **Calibration is now shipped (S3.9), not deferred:** per-class Platt fit train-only on the
  selected models cuts held-out ECE 1.4–200× with AUC unchanged, and the deliverable ships the
  calibrated p̂. The remaining selection-side gap is the **nested-CPCV real-data run**, still
  deferred (a meaningful one is ~15×5×5 fits per class) alongside the S6.5 stub.
- ho1s (2 OOS rows, IC undefined) / ng1s (56) / gc1s (30) rest on thin coverage — all three now
  flagged (`< 60` rows or undefined IC) in `coverage_caveat.csv`; their per-instrument numbers are
  small-sample noise.
- The §6 **constraint set** remains a lit-review-default **stub** (the 20 May constraints doc is
  absent); only the barrier-exact *holding model*, cost model and deflation gate were upgraded.
- The **autoencoder reducer** was built for the EX.2 comparison only; cluster-rep is promoted.
- **Macro vintages (X.9):** the `additional_data.xlsx` series carry observation dates and are
  **publication-lag (PIT) aligned** so no *timing* look-ahead, but the workbook ships **revised/final
  values, not real-time vintages** — so a revision look-ahead (e.g. a later-restated TIPS10Y/BE10Y
  print) is not excluded. A full ALFRED real-time-vintage reconciliation is the remaining macro gap;
  the conservative publication lags (and the macro block's small, near-zero §4 MDA) bound its impact.
- **Write-up citations (X.7), corrected per the pass-3 directive:** the Japan/UK regime correlations
  are attributed to **Guidolin–Timmermann (St. Louis Fed WP 2005-034)**, with the **Ang–Bekaert
  Wald-test caveat (p = 0.156)** noted; the Kelly/vol-target convention cites **Carver 2015 Ch.9
  p.146**; the MZB fractional-Kelly figures keep ≈50%/≈75% and the unsupported "56%" is removed. The
  one residual is confirming the Carver page against the print edition before academic submission.
- Equity instruments start late (es1s 1997, fesx1s 1998, nq1s 1999) — thin pre-2020 history for
  fitted features.
