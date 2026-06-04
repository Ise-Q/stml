# Alken Meta-Labelling Metamodel — Methodology (Sections 1–6 + bonus)

**Course:** T3.03 Systematic Trading Strategies with ML — Alken team challenge.
**Scope:** a secondary *act/skip* classifier over the provided primary signal, across **11
futures instruments in 3 asset-class metamodels** (Equity / Energy / Metals).
**Grading stance:** *methodology, not performance* (`../refs/project-instructions.md`). Every
choice below is justified against the literature review `../reports/apb/nlr-cw-v1.md` (8
commitments, 60 references), the feature set is locked before the final OOS window, and the
results section reports honestly where the metamodel **does not** beat the blind-primary baseline.

> **Reproduce:** `uv run --project metamodel-apb python -m alken_metamodel.emit
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

> Selected models this run (CPCV selection): **Equity → XGBoost** (0.5897), **Energy → torch-MLP**
> (0.5233 — a neural variant won, knife-edge over xgboost 0.5212), **Metals → elastic-net logistic** (0.5324).

---

## 0. Architecture (one-directional data flow)

```text
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
recomputed per instrument, **never** the frozen `results/feature_matrix.parquet` (which freezes
fitted stats at one global `fe_train_end=2021-07-01` and would leak into in-sample folds before
that date — the exact failure this build avoids). Every stml `assemble_*` is stateless and proven
causal by **right-edge truncation-invariance**: a feature's value at `t` is identical computed on
`data[:t+1]` or on the full series — encoded as a property test
(`tests/test_features.py::test_right_edge_truncation_invariance`). This per-fold-recompute discipline
is a deliberate differentiator from the shared base, and the pass-4 feature re-open obeys it: the
added **F16 concept-drift** family is recomputed per fold (a causal rolling discriminator, never read
from the parquet), and a `test_no_metamodel_module_reads_frozen_parquet` guard asserts no module
consumes the frozen matrix.

---

## 1. Feature engineering (§1, 20 marks) — `features.py`, `regime.py`

The pooled feature matrix (**measured per class, not asserted** — X.8): the full pass-4 pooled panel
is **Energy 141 / Equity 140 / Metals 141 columns** (core engineered 115 — incl. F12/F13/F15 and the
F3/F17 static-regime cols — **+ the F16 concept-drift col (pass-4)** + EWMA-HMM regime 5 + PIT-macro
16 + instrument one-hots 3–4), i.e. +1 over the pass-3 counts (140/139/140); the §4 clustering matrix,
which drops the regime/macro blocks and zero-variance columns on the modelling slice, is ~108/107/108.
Both are measured by `experiments/x8_feature_counts.py`. The layers:

| Block | Columns (examples) | What it captures | Module / source |
| :---: | :---: | :---: | :---: |
| F1 counter-trend | `f1_mr_score_{10,20,40}`, `f1_rsi_14`, `f1_bb_pctb_20` | mean-reversion pressure | stml `features.py` |
| F2 vol/dispersion | `f2_vol_20` (annualised, LI), `f2_garman_klass_20`, `f2_parkinson_20` | heteroskedastic risk; **GK** range vol (Garman–Klass 1980; Korkusuz 2023, nlr-cw §A1) | stml + `volatility.py` |
| F2-RS | `f2_rogers_satchell_20` | drift-independent range vol | stml `features_ext.py` |
| F5 signal-derived | `f5_trailing_run_length`, `f5_participation_20`, `f5_signal_entropy_20` | primary-signal trajectory (LI handoff) | stml |
| F6 momentum | `f6_ts_momentum_{20,60}`, `f6_macd_12_26`, `f6_adx_14` | trend strength | stml |
| F7 microstructure | `f7_oi_change`, `f7_amihud_20`, `f7_kyles_lambda_20` | liquidity / OI flow | stml |
| F8 calendar | `f8_dow_sin/cos`, `f8_month_sin/cos` | seasonality | stml |
| F10 price-action | `f10_hl_range`, `f10_oto_ret` | intraday range / gaps | stml |
| F12/F13/F15 | `f12_hurst_100`, `f13_mra_energy_d1..d5`, `f15_prob_timeout` | path structure, wavelet energy, conditional first-passage risk | stml `features_ext.py` |
| **F16 concept-drift** (pass-4) | `f16_regime_alignment_score` | covariate-shift discriminator — does today's feature row look like the FE-train era or the recent past? (causal rolling logistic; Sugiyama–Kawanabe 2012) | stml `drift_features.py` |
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

- **Vol-adaptive barriers** `±k·σ̂ₜ` where `σ̂ₜ` is a de-annualised *realised* daily vol (a rolling
  20-day std of log returns, `f2_vol_20 / √252`) — **not**, despite an earlier draft's wording, a
  Garman–Klass estimate (Garman–Klass enters only as a *feature*, §1). Fixed-% thresholds ignore the
  heteroskedasticity of returns and make the label distribution pro-cyclical. **Vertical barrier**
  `T_max` (default 10 bars) bounds the horizon.
- **Per-class barrier geometry (EX.5, modelling-window economic-Sharpe selected).** The geometry is
  set per asset class rather than a single global symmetric barrier. The **default and Metals** path
  uses the de-annualised realised-vol-20 above with symmetric `(1,1)` multiples and a 10-day vertical
  barrier. **Equity** carries a 50-day rolling-volatility scale with an asymmetric `(2,1)`
  profit-to-stop ratio and a 10-day horizon; **Energy** a 20-day EWMA volatility with a tight
  `(0.5,0.25)` ratio and a vol-scaled horizon (base 10, clipped to `[2,40]`). The EX.5 sweep ranked
  candidate geometries by their downstream **net Sharpe** — not label accuracy — strictly on the
  modelling window (`≤ 2021-12-31`), so the choice never touches the OOS window and the selection
  statistic sits inside the leakage firewall (best modelling net Sharpe vs always-act floor: equity
  0.513 vs 0.099, energy 0.279 vs 0.084, metals 0.159 vs 0.107). Three honesty caveats: the sweep was
  one-factor-at-a-time and XGBoost-scored, so the per-class triples were never jointly validated under
  the deployed torch roster; only Equity's and Energy's geometries survive an XGBoost→LightGBM
  robustness swap, so **Metals is omitted from EX.5** (it flips negative under LightGBM) and falls back
  to the default symmetric barrier; and adopting these geometries is a wiring decision at the emit
  boundary, not a demonstrated improvement — §6 reports that under the torch roster Energy's geometry
  does not transfer OOS (it abstains entirely).
- The label is the sign of the side-adjusted P&L at the **first** barrier touched; **`t1`
  (first-touch time) is recorded for every label** and drives purge/embargo everywhere.
- **Per-instrument embargo (S2.6, pass-4).** The purged CV now embargoes each instrument by its own
  `embargo_p90` run-length (in trading days, from `instrument_scope.json`: ng1s 33d, ho1s 26d,
  rb1s 19d, cl1s 14d, … si1s 7d) advanced on that instrument's own date axis, replacing the uniform
  ⌈1%·T⌉ bars — correct for a pooled panel where a flat position-count covers only ~embargo/K days
  per instrument. The instrument-agnostic span-overlap purge still covers cross-instrument label
  concurrency; a known-answer test asserts each instrument's purged forward window equals its
  `embargo_p90` and that zero train/val `t1` overlap survives the larger embargo.
- **Sample-uniqueness weights** from label concurrency (LdP Ch.4) down-weight labels whose
  horizons overlap — triple-barrier labels are not iid. Verified exactly (disjoint → 1.0;
  fully-overlapping → 0.75/0.50) in `tests/test_triple_barrier.py`.

Per-instrument class balance ranges ~50–69% positive; see the §5 per-instrument table.

---

## 3. Models (§3, 30 marks) — `models.py`, `neural.py`, `cross_validation.py`, `evaluation.py`

A horse-race behind one uniform `MetaClassifier` interface so the comparison is apples-to-apples
(Gu-Kelly-Xiu 2020; Krauss et al. 2017; IKM 2020 small-data restraint, nlr-cw §2). The **shipped
default roster is five estimators** (1–5); a sixth (Keras-VSN) is an off-path determinism-safe
comparison (see *Scope*). Estimators 1–3 run on the full feature set; the two neural variants
(4–5) run on the **cluster-representative-reduced** feature set (EX.2: one medoid per Mantegna
cluster), which makes the VSN's one-GRN-per-feature architecture tractable at CV scale:

1. **Elastic-net logistic** (saga; median-impute + standardise),
2. **XGBoost** (PS5 config),
3. **LightGBM**,
4. **torch-MLP**,
5. **torch-VSN** (byte-deterministic VSN port with softmax feature-selection weights),
6. **Keras-VSN** (vendored PS6 `FinalModel`) — off-path, TF op-determinism best-effort.

Reduction is **fold-safe**: the reducer is wrapped with its estimator so the evaluation harness
fits the medoid selection on each fold's train rows only — the reduced basis never sees the
validation fold.

**One weighting channel.** PS4/5/6 ship no weighting; meta-labels are both *overlapping* (need
uniqueness weights) and *imbalanced* (~30–40% positive). Both are folded into a single
`sample_weight = uniqueness × inverse-class-frequency` passed identically to every estimator's
`fit` *and* into every OOS metric.

**Validation (commitment #3, nlr-cw §6 — LdP Ch.7/12; Bailey 2014; Harvey-Liu-Zhu 2016).**
`cross_validation.py` provides **PurgedKFold + per-instrument embargo** (S2.6: each instrument's
`embargo_p90` on its own date axis, replacing the uniform ⌈0.01·T⌉), **CPCV (N=6, k=2 → 15 paths)**,
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

**Selected models (CPCV selection, real data):** Equity → **XGBoost** (15-path mean AUC 0.5897),
Energy → **torch-MLP** (0.5233 — a neural variant won its class, knife-edge over xgboost 0.5212),
Metals → **elastic-net logistic** (0.5324). The torch NN family is now genuinely competitive, not a
synthetic-only appendix.

**CPCV path robustness (EX.1, modelling sample).** The fraction of the 15 purged combinatorial
paths beating 0.5 is the discriminating signal for *where* the edge is real:

| Class | best model | mean CPCV AUC | paths > 0.5 | reading |
| :---: | :---: | :---: | :---: | :---: |
| Equity | LightGBM | **0.5923** | **15 / 15** | edge is **robust** |
| Metals | elasticnet | 0.5241 | 13 / 15 | marginal-but-positive |
| Energy | XGBoost | 0.5266 | 12 / 15 | majority-but **within-sample only** |

This is a *within-sample* count on a stripped tree-and-linear roster (regime/macro off), so the
nominal Energy winner is XGBoost rather than the deployed torch-MLP — the path count, not the model
identity, is the object of interest. Equity's edge survives every combinatorial path; Metals and
Energy clear 0.5 on a majority of paths *here too*, but this in-sample count sits alongside
per-instrument **OOS** AUCs that fall below the no-skill line for Energy (§5) and a uniformly
negative significance and timing verdict (§5/§6) — it anticipates Equity's relative strength, not a
transferable Energy edge, corroborating the §4 cluster importance and the deflated-Sharpe /
multiple-testing caveat (Bailey 2014; Harvey-Liu-Zhu 2016).

**Calibration is now shipped (S3.9, `calibration.py`).** Because Kelly sizes on p̂ directly, a model
that *ranks* act/skip adequately still mis-sizes if p̂ is uncalibrated. Pass 3 fits **one Platt map
per asset class** on the *selected* model's purged-OOS modelling predictions (strictly before
`predict_start`, so it cannot leak), and applies it to **both** the deliverable probabilities **and**
the Kelly stake. Platt is monotone, so **AUC is unchanged** (the act/skip ranking is preserved — a
unit-tested invariant); only Brier/ECE and the stake move. On a leakage-safe in-time 70/30 split of
the **selected** models (not LightGBM-as-proxy as in pass-2's EX.4):

| Class | s3 selected model | ECE raw → Platt | Brier raw → Platt | AUC (raw = Platt) |
| :---: | :---: | :---: | :---: | :---: |
| Energy | XGBoost* | 0.1053 → **0.0481** | 0.254 → 0.244 | 0.541 |
| Equity | XGBoost | 0.0615 → **0.0173** | 0.247 → 0.242 | 0.607 |
| Metals | logistic | 0.2096 → **0.0012** | 0.313 → **0.247** | 0.532 |

\* The s3 calibration probe re-selects on its own 70% calib split → names **XGBoost** for Energy,
versus emit's full-sample deployment of **torch-MLP**; cite emit's deployed model as canonical.

The raw probabilities are materially miscalibrated (Metals worst at ECE 0.2096 — Gramegna-Giudici
2021); Platt cuts ECE sharply with the AUC untouched. The **deliverable ships the calibrated
probabilities** (`metamodel_predictions.csv`), with the raw file retained
(`metamodel_predictions_raw.csv`) for this before/after; the `experiment_log.csv` now records the
calibrated class-level Brier (Equity 0.2484, Energy 0.2597, Metals 0.3406) and precision (Equity
0.5549, Energy 0.5466, Metals 0.5754). **Note:** calibration *does* change §6 — it shifts which
positions clear the p̂≥0.55 Kelly floor, which moves the per-book Sharpes (Metals especially; see §6).

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
| :---: | :---: | :---: | :---: |
| 1 | `max_features='auto'` (PS4 grid; removed in sklearn ≥1.3) | `'sqrt'` | `cluster_importance.py` MDI/SHAP forest |
| 2 | `KFold(shuffle=True)` for MDA (leaks across overlapping labels) | injected **PurgedKFold** | vendored `calculate_cluster_importance_pfi` |
| 3 | no real SHAP in PS2/sts-ml (MDI+PFI only) | **cluster SHAP** via `TreeExplainer`, summing member \|SHAP\| (the §4 contribution) | `cluster_importance.py` |
| 4 | Spearman distance `1−\|ρ\|` (non-metric) | **Mantegna** `√(1−\|ρ\|)` (Mantegna 1999) | vendored `compute_spearman_distance_matrix` |

**Run on the real matrix (S4.7/S4.8, re-executed).** Clustering each class's modelling matrix
(median-imputed, zero-variance dropped → **Energy 108 / Equity 107 / Metals 108** feature columns)
and scoring every cluster:

| Class | clusters | top cluster MDA | top cluster SHAP | near-zero-MDA clusters |
| :---: | :---: | :---: | :---: | :---: |
| Equity | 3 | **0.0191** | 0.43 | 2 / 3 |
| Energy | 3 | −0.0045 | 0.40 | 3 / 3 |
| Metals | 2 | −0.0113 | 0.7134 | 2 / 2 |

The honest reading (S4.8): **MDI and SHAP are in-sample attribution** — they always split 100% of a
fitted model's importance across the clusters, so a high MDI/SHAP says only *which* features the
model leaned on in-sample, not that those features carry OOS edge. **Cluster permutation MDA is the
OOS reality check, and it is near-zero across the board** (every cluster `|MDA| < 0.02`, no exception
now) — exactly what a ≈0.5-edge problem should look like, and a useful negative result. So a high
cluster SHAP must **not** be read as edge. **Re-checked under the per-class barrier (S4.8):** the
picture is unchanged — Energy top-cluster MDA −0.0045, Metals −0.0113 (all clusters noise), and the
**largest cluster MDA is Equity's 0.0191** — still marginal and the only positive one, matching its
15/15 CPCV robustness; F16 lands in a noise cluster and adds nothing under permutation. The MDI/SHAP-vs-MDA
divergence *is* the §4 lesson — only the permutation (OOS) view is honest about edge. The synthetic
noise-cluster≈0 sanity test still holds.

---

## 5. Evaluation (§5, 20 marks) — `evaluation.py`, real OOS results

Metrics are **sample-weighted**, **threshold-aware**, and computed **per-instrument before any
aggregate** (so a strong pooled number can't hide a weak member). Single-class purged folds yield
NaN ranking metrics rather than crashing. The baseline is **blind-primary** (act on every signal).

**Per-instrument purged-OOS AUC (shipped default path, 11-instrument fan-out):**

| Class | Model | Per-instrument AUC (n labels) | vs blind-primary |
| :---: | :---: | :---: | :---: |
| Equity | XGBoost | es1s 0.61 (457) · nq1s 0.62 (482) · fesx1s 0.63 (510) | **beats** |
| Metals | logistic | hg1s 0.57 (504) · pl1s 0.49 (453) · si1s 0.51 (462) · gc1s 0.45 (138) | mixed |
| Energy | torch-MLP | cl1s 0.47 (334) · rb1s 0.50 (504) · ho1s 0.43 (61) · ng1s 0.42 (68) | **below no-skill** |

**OOS coverage caveat (S5.7 → widened S5.9).** All 11 instruments emit (no abstention), but the
thin-coverage flag is now `n_oos_rows < 60` **OR** an undefined information coefficient — which
flags **three** names, not just one: **ho1s = 2 rows** (and IC undefined — Spearman on 2 points is
meaningless), **gc1s = 30**, **ng1s = 56** (IC defined at 0.19 but on only 56 rows); the other eight
carry 88–127 rows. `coverage_caveat.csv` now records `n_oos_rows`, the per-instrument OOS `ic`,
`ic_undefined`, and `thin`, so the deliverable's thin support is explicit rather than implied
uniform. The per-instrument AUCs for ho1s/ng1s (≈60 *training* labels too) are small-sample noise.

**Primary-signal context (EX.5).** Characterising the *provided* signal sets the metamodel's
ceiling: directional hit-rates run **0.51–0.69** (gc1s strongest at 0.66, IC 0.2111; rb1s weakest),
turnover 0.02–0.23 flips/day, with several names better in the low-vol regime. The base signal is
already decent, so the secondary act/skip filter has little headroom. (This primary-signal IC — rank
correlation of the *signal* with the subsequent return, on the modelling window — must not be
confused with the metamodel's own OOS IC, the rank correlation of the *calibrated probability* with
the subsequent return on the 2022 window; for gc1s the two even differ in sign, the metamodel IC
being −0.23.)

**Utility-aware evaluation (S5.10 — beyond AUC).** AUC says nothing about whether *acting* adds
economic value, so we test market timing on the OOS acted trades. The **primary** test is
**Pesaran–Timmermann (1992)** — a non-parametric directional-accuracy statistic that *conditions on
the directional base rates*, so a constant call in a trending market scores 0 rather than a spurious
positive. **Treynor–Mazuy (1966)** γ (signed-PnL convexity in the realised move) corroborates, and
the older hit-rate **Henriksson–Merton sign test is reported as a base-rate-SENSITIVE proxy only**
(the implemented `henriksson_merton` is the plain hit-rate-vs-½ form — neither the parametric
regression nor the conditional non-parametric H–M — so it over-reads "skill" in a trending window;
a hand-worked toy case confirms PT≈0 where the proxy shows z>0). With the mean-variance
certainty-equivalent:

| Book | **PT stat (p)** *(primary)* | TM γ (t) |
| :---: | :---: | :---: |
| Energy | *abstains (no acted trades)* | *abstains* |
| Equity | +0.12 | +0.31 (insignificant) |
| Metals | −2.17 | −2.05 (insignificant) |
| All-11 | **−1.80 (p ≈ 0.96)** | **−0.51 (t = −0.35)** |

**No positive directional skill is demonstrated.** The load-bearing result is the directional test:
**Pesaran–Timmermann — the base-rate-aware primary — is negative or insignificant in every book**
(pooled −1.80, p ≈ 0.96; Energy abstains entirely under its barrier so contributes no acted trades),
and the §6.14 pooled Sharpe is itself insignificant (t = 0.34). A proxy *biased toward* skill still
showing none makes the negative stronger, not weaker. **Honest caveat on TM:** the corroborating
convexity test does not even *print* positive here — the *pooled* Treynor–Mazuy γ is **−0.51
(t = −0.35)**, insignificant and of a piece with the negative PT. That the same coefficient printed
**positive and nominally significant (+1.18, t = 2.55) under the OLD global barrier** — where it could
be mistaken for timing — is itself the cautionary point: the pooled coefficient is an unreliable scale
aggregate **in either sign**. With PT negative and the Sharpe insignificant, whatever convexity the
pooled regression detects is the signature of **the stop/barrier exit asymmetry (winners ride, losers
are cut), not market timing**; it is not a tradeable signal. (A measurement note: our TM regresses
signed-PnL on each trade's *own* realised return rather than an external market benchmark, so it is a
convexity diagnostic, not canonical TM.) The **full resolution** — a scale-aggregation artefact in
either sign, compounded by mechanical exit convexity, with PT as the scale-invariant primary — is
S5.11, and its in-data proof is S5.12. Cite Henriksson–Merton 1981 (pp. 513–533), Treynor–Mazuy 1966,
Pesaran–Timmermann 1992.

**S5.11 — the pooled-TM convexity is an aggregation artefact in either sign, not timing (LR-9).** The
*pooled* Treynor–Mazuy coefficient is **−0.51 (t = −0.35)** here, insignificant — it does not even
print positive. That the same coefficient printed **positive and nominally significant (γ = +1.18,
t = 2.55) under the OLD global barrier**, where it could be mistaken for timing skill, is itself the
cautionary point: the pooled coefficient is an unreliable scale aggregate whichever sign it takes.
First, pooling sub-portfolios of heterogeneous return scale, volatility and beta into one
quadratic-timing regression is known to yield inconsistent, sign-reversing coefficients — a
regression instance of Simpson's paradox / aggregation bias (Blyth 1972; Robinson 1950;
Pesaran & Smith 1995; Zellner 1962) — and here the per-sleeve coefficients (Equity γ = +0.31 and
Metals γ = −2.05, both insignificant, while **Energy abstains entirely** under its barrier and
contributes no acted trades) reject coefficient homogeneity, and a trade-count-weighted average of the
sleeve γ is **−1.40, well away from the pooled −0.51 — the pooled magnitude is a scale aggregate of
heterogeneous sleeves, not a shared coefficient**. Second, the convexity the pooled regression
detects when it detects any is the mechanical, option-like convexity of the barrier-exact exit, not
directional forecasting: **Jagannathan & Korajczyk (1986)** show that holding option-like or levered
payoffs produces *artificial* market-timing ability where none exists, and a protective,
big-move-capturing stop/barrier rule is exactly such a convex, option-isomorphic payoff
(Henriksson & Merton 1981; Glosten & Jagannathan 1994; Fung & Hsieh 2001). Market-timing skill is
therefore assessed primarily by the **Pesaran & Timmermann (1992, pp. 461–465)** directional-accuracy
test, which — being a sign / contingency-table statistic rather than a magnitude regression — is
invariant to the scale heterogeneity that drives the pooled-γ artefact; its result (pooled −1.80,
p ≈ 0.96) confirms the absence of timing skill, consistent with the disaggregated TM evidence.
(Treynor & Mazuy 1966 is a practitioner *HBR* article cited only for the quadratic specification, not
as authority on the artefact; and PT is undefined when all directional calls coincide — its power
depends on the up/down balance of the realised series.)

**S5.12 — the artefact reproduced and dissolved in our own data (standardise-then-repool, LR-9 rec #5).**
LR-9's cheap in-data check converts the cited theory into direct evidence. Re-estimating the pooled TM
regression after **vol-targeting each sleeve to a common scale** — dividing each sleeve's realised
market return *and* its signed PnL by that sleeve's return-standard-deviation (the *same* factor, so
the per-trade `pnl = side · return` identity is preserved) — **moves the pooled γ from −0.51 to
−0.0277 (t = −0.93, p = 0.354)**, collapsing to an insignificant near-zero. Once the sleeves share a common scale, whatever pooled convexity the
unstandardised regression reported — **positive under the OLD barrier, negative here** — dissolves
into the same scale-driven near-zero: direct in-sample proof that the pooled TM magnitude is
scale-aggregation, not timing. (Diagnostic only — `experiments/s6_barrier_backtest.py`, written to the
gitignored `experiments/results/`; the deliverable CSVs are untouched.)

**Honest reading.** Classification-wise the metamodel adds clear value only on Equity (all three
names AUC ≈ 0.62, consistent with 15/15 CPCV paths); Metals and Energy are mixed — Energy's
better-covered names now sit *below* the no-skill line (cl1s 0.47, ng1s 0.42) — and dragged by
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

> **Headline (no edge is demonstrated; the strategy is NOT claimed to work).** Before any
> selection-bias deflation, the more basic question is whether the §6 Sharpe is even
> distinguishable from zero on the ~128-day OOS. **It is not:** the pooled net Sharpe of **0.48**
> gives **t = SR·√n = 0.34** (n = 127, not significant at 5%), and the primary inference — a
> **studentised stationary block-bootstrap 95% CI of [−0.09, +0.15] (per-period) — contains 0**
> (S6.14). PSR(0) = 0.635 (< 0.95) and MinTRL ≈ 2,883 days vs 127 available agree. *Only then* does the
> deflation gate (S6.8) **corroborate**: pooled DSR [0.26 → 0.08] (≪ 0.95), PBO 0.36. Combined with
> AUC ≈ 0.5 (§3/§5), near-zero cluster MDA (§4) and **no positive directional timing**
> (Pesaran–Timmermann negative everywhere, §5), the honest conclusion is **insufficient evidence of
> a deployable edge** — the positive Sharpe is the barrier exit + vol-targeting + diversification,
> not act/skip skill. The per-class barrier and stricter per-instrument embargo did **not** change
> this: IC still ≈ 0 — which *strengthens* the Fundamental-Law null (§6.12). Most pointedly, under its
> committed per-class barrier **Energy takes zero OOS positions** (its calibrated p̂ sits in a narrow
> no-signal band below the act-floor), so the metamodel correctly **abstains** rather than
> manufacturing a book — the sharpest illustration of the negative.

**Barrier-exact, cost-aware OOS backtest, Jan–Jun 2022 (S6.7, calibrated sizing).** The position
exits on the **actual triple-barrier first-touch `t1`** (not a fixed `max_holding`), overlapping
labels are **netted**, a Grinold–Kahn cost model (half-spread + impact) is charged, and the stake is
sized on the **calibrated** p̂ (§3.9). Net of costs:

| Book | Model | Sharpe | Sortino | Ann. vol | Max DD | Gross→Net |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All 11** | — | **0.48** | 0.78 | 11.0% | −4.5% | +5.36% → **+2.38%** |
| Energy | torch-MLP | **0.0000** *(no trades — abstains)* | — | — | — | — |
| Equity | XGBoost | 0.51 | — | — | — | — |
| Metals | logistic | 0.00 | — | — | — | — |

> **Sortino convention.** The downside deviation uses the **full-T denominator with target 0**
> (Sortino & Price 1994) — positive returns contribute zero to the semivariance and the divisor is
> the *whole* sample length, not the count of negative observations. We deliberately avoid the
> common mis-implementation that takes `std()` over the negative returns only (a smaller, biased
> divisor that inflates Sortino); a known-value test (`tests/test_backtest.py`) pins the full-T form.

These are the **per-class-barrier** numbers (per-instrument embargo + the F16 re-open + calibrated
sizing). **Energy now takes zero OOS positions** under its committed `ewma20/(0.5,0.25)/vol-scaled`
barrier: its calibrated p̂ collapses into a near-constant no-signal band **[0.519, 0.522]** (AUC ≈
0.52), entirely below the 0.55 Kelly act-floor, so the book **abstains** (Sharpe → 0.0000, no trades)
in both `emit` and the §6 backtest. This is not "edge destroyed" — Energy's old shipped-barrier 1.86
Sharpe had *already* failed every test (insignificant t, fails deflation, negative/insignificant PT);
it was noise that happened to print positive over 127 days. Under the modelling-committed barrier the
same no-signal book resolves to a correctly-cautious abstention — **fragility consistent with no
edge**, and we deliberately do **not** re-pick the barrier off this OOS result (that would be the
snoop the report condemns; the config was committed on the modelling window). Calibration still
**moves the sizing** (shrinking over-confident p̂ toward the base rate now keeps ≈ **62%** of bets
below the 0.55 Kelly floor at zero weight, up from ≈ 36% under the old global barrier, holding pooled
vol to ~11%); the per-instrument floor concentration is quantified in §6.13.

**S6.14 — significance-first inference (the PRIMARY §6 result, LR-6).** A selected Sharpe must be
deflated, but the prior question is whether it clears zero at all on a ~128-day OOS. Reported in
assumption-strength order (`significance.py`, all per-period unless annualised):

| Statistic | Pooled (all-11) value | Reading |
| :---: | :---: | :---: |
| Sharpe + **t-stat** | SR/day 0.0303, **t = 0.34** (n=127) | **not significant** at 5%, before any deflation |
| **Studentised stationary block-bootstrap 95% CI** *(primary)* | per-period **[−0.09, +0.15]** | **contains 0** — the width *is* the finding |
| Lo/Opdyke analytic band | per-period [−0.14, +0.20] | parametric cross-check, also straddles 0 |
| PSR(0); **MinTRL** | 0.635 (< 0.95); **≈ 2,883 days** vs 127 available | track record >20× too short to certify |
| Ljung–Box(10) | p = 0.000 | returns are **serially correlated** → the √252 annualisation *overstates*; read the per-period CI |

The block length is Politis–White's `optimal_block_length` (data-driven); the bootstrap is
studentised by the Lo (2002) analytic SE and seeded (deterministic). The CI straddling zero is the
honest headline: **insufficient evidence**, not a demonstrated failure (cite Lo 2002, Opdyke 2007,
Bailey & López de Prado 2012/2014 *journal* versions — not Mertens' unpublished note; no
peer-reviewed DSR-at-T≈128 Monte-Carlo exists, so the deflation numbers below are corroboration, not
load-bearing point statistics).

**S6.8 — deflation gate (now demoted to corroboration).** On the same net returns, deflating for
selection bias (DSR/MinBTL: Bailey–López de Prado 2014; PBO via CSCV: Bailey-Borwein-LdP-Zhu 2017,
C(16,8) = **12,870** combinations — the canonical "12,780" is a propagated typo). DSR is reported as
a **ladder over N**: N_eff (ONC-clustered trials, optimistic) → N_raw (the roster horse-race) → 2·,
4·N_raw, the upper rungs reflecting the implicit feature-selection search (the data-driven
cluster-rep reducer + the F16 re-open) that N_raw under-counts:

| Book | net Sharpe | DSR ladder (N∈[6,15,30,60]) | CSCV-PBO | MinBTL vs OOS≈0.5y |
| :---: | :---: | :---: | :---: | :---: |
| Energy | **0.0000** *(abstains — no trades)* | — | — | — |
| Equity | 0.51 | — | — | — |
| Metals | 0.00 | — | — | — |
| **All-11** | 0.48 | **[0.260 → 0.159 → 0.109 → 0.075]** | 0.36 | **[1.69 → 3.14]y** |

Even the *optimistic* low-N end stays well below 0.95 (pooled 0.26), and DSR only falls as the trial
count N rises — so the gate fails a fortiori under the honest, higher count (the cluster-rep reducer,
the F16 re-open *and* the per-class barrier sweep all add selection the ladder does not enumerate).
The pooled MinBTL (up to 3.1y) dwarfs the half-year OOS. PBO ≈ 0.36 (pooled) is the noise-dominated
mid-range expected of a no-edge selection. **This corroborates, but does not lead, the S6.14
verdict.** (Per-book deflation is not separately regenerated under the per-class barrier — Energy
abstains; the pooled rung is the load-bearing figure.)

**S6.9 — the holding model is load-bearing (the methodology finding).** Identical positions, models,
features and calibration — *only the exit convention differs*:

| Book | simple `max_holding=10` | barrier-exact (actual `t1`) |
| :---: | :---: | :---: |
| Energy | 0.00 *(abstains)* | 0.00 *(abstains)* |
| Equity | −0.34 | +0.51 |
| Metals | −0.21 | +0.00 |

The exit convention alone moves the books that trade (Metals −0.21 → 0.00; Equity −0.34 → +0.51, a
sign flip), while Energy is zero under both conventions because it takes no positions; the ordering is
driven by the **exit mechanism** (winners ride to the profit barrier, losers are cut at the stop),
**not** classification skill — consistent with AUC ≈ 0.5 and no positive directional timing (§5).
Gated by S6.14/S6.8, it is a *finding about backtest construction*, not a performance claim.

**S6.10 — accounting reconciliation.** `net = gross − costs` holds daily, but the report's
`gross_total_return`/`net_total_return` are **compounded** (∏) while `total_cost` is an **arithmetic
Σ**, so `gross − total_cost ≠ net` by the compounding interaction (pooled: gross **+5.36%**, net
**+2.38%**). The reconciliation field **`cost_drag_compounded` = gross_total − net_total**
(pooled **2.98%**) closes the identity exactly; `total_cost` (3.04%) is retained as the undiscounted
sum. Turnover (one-way annualised notional, 51.6) and holding (trade-based busday, 2.8d) are on
**different bases**, so `hold ≈ 252/turnover` is not expected to hold — both are reported, not
forced into a false identity.

**S6.12 — the convergence (one honest negative, not five unlucky ones).** Five independent lenses
agree the meta-model adds no exploitable act/skip edge on this primary signal:

| Lens | Per-class-barrier result | Verdict |
| :---: | :---: | :---: |
| §3/§5 OOS AUC | ≈ 0.50 (0.59 / 0.52 / 0.53; energy per-inst below random) | no ranking skill |
| §4 cluster MDA (OOS) | \|MDA\| < 0.02 across clusters | no feature carries OOS edge |
| §6.14 significance | t = 0.34; bootstrap 95% CI contains 0 | Sharpe not distinguishable from 0 |
| §6.8 deflation | pooled DSR [0.26 → 0.08]; PBO 0.36 | fails the selection-bias gate |
| §5 timing (Pesaran–Timmermann) | pooled −1.80 (p ≈ 0.96) | no positive directional timing |

This convergence is *predicted*, not coincidental. The named primary is **short-horizon
mean-reversion** (`f1_mr_score_20`), and **Grinold's Fundamental Law `IR = IC·√BR`** *[PROVEN]* makes
the null structural: with the primary's IC ≈ 0, the achievable information ratio is ≈ 0 **regardless
of breadth or sizing** — a secondary act/skip filter cannot manufacture skill the primary lacks
(`/aqms-python ir_from_skill`). The H–M / TM / PT sign rules *[PROVEN]* turn that into a falsifiable
timing test the data fails to reject in the no-skill direction. The one thing we *assume* is López de
Prado's exploitable-skill precondition *[ASSUMED PREMISE]* (meta-labelling helps only when the
primary already has filterable skill); the "AUC ceiling for a mean-reversion primary" is an
*[EMPIRICAL]* heuristic, not a theorem. **The pass-4 expansion tightens the null:** F16 added under
the same per-fold causal discipline and a stricter per-instrument embargo, and IC/AUC are still ≈ 0 —
more features, no more edge, exactly what IR = IC·√BR predicts.

**S6.13 — calibration × floor concentration (a sizing caveat, not an edge).** Calibrated p̂ tops out
at **0.670** (on Equity), so roughly **62%** of bets sit below the 0.55 Kelly floor at zero weight, up
from a little over a third (≈ 36%) under the old global barrier. The floor-pass rate is wildly uneven
*across books*: **Metals** clears it on about three-quarters of bets (≈ 25% zero-weight) and
**Equity** on under a third (≈ 71% zero-weight), while **Energy** clears it on **none** — its
calibrated p̂, compressed by a no-signal model into the narrow band **[0.519, 0.522]** just above ½,
never reaches the floor, so the entire book **abstains** (100% zero-weight). A sleeve sized to zero by
a correctly-cautious floor is the sharpest illustration of the negative: the metamodel, asked whether
to act on Energy, declines on every day. Ties to the thin-coverage flags (S5.9) and the `n_eff` scope.

**S6.15 — CER-gated sizing (LR-7; flat κ retained).** Two refinements were evaluated against OOS
certainty-equivalent (`/empirical-finance certainty_equivalent`, γ = 5, baseline CER **0.000090**):
(a) a **smooth taper** replacing the hard p ≥ 0.55 floor (continuous at the old floor, a fixed
function of p̂); (b) per-instrument **Baker–McHale κᵢ = eᵢ²/(eᵢ²+σᵢ²)**. The κᵢ variant's earlier
OOS-estimated form was rejected as **circular** (its κᵢ read off the OOS window itself); the
leakage-safe redo, estimating κᵢ on the modelling sample only, posts a very large *point* gain
(0.000090 → 0.000593, **+561.5%**) that clears the materiality bar — but its **paired studentised
block-bootstrap 95% CI on the CER difference is [−0.000780, +0.001814] and contains 0**, so the gain
is indistinguishable from noise at this sample length (the more so where an entire sleeve abstains) →
**revert**. The smooth taper *does*, on its leakage-safe form, now clear the CER gate (0.000090 →
**0.000275**) — but that advantage is measured **on the OOS window itself**, so adopting it would
re-introduce exactly the look-ahead the locked-before-OOS sizing exists to prevent. The gate flip is
therefore reported as a **diagnostic, not acted on**: the shipped weights **retain flat κ = 0.25 / the
hard floor** and the byte-identical deliverable is unchanged. Carver's 25% vol-target halving for
negative-skew sleeves was checked. The honest default: a point estimate that flatters but cannot
survive its own confidence interval — or its own leakage firewall — is not adopted.

---

## Commitments → modules → citations (Definition of Done)

| # | Commitment | Module | nlr-cw / primary citation |
| :---: | :---: | :---: | :---: |
| 1 | Meta-labelling act/skip filter | `triple_barrier.py`, `pipeline.py` | §1 — LdP 2018 Ch.3; Joubert 2022 |
| 2 | Per-class vol-adaptive ±k·σ̂ₜ (realised-vol / rolling-std / EWMA) + vertical T_max | `triple_barrier.py` | §1 — LdP 2018 Ch.3 |
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
  Verified on the **pass-4 shipped path** (torch NN family + CPCV selection + macro + per-instrument
  embargo + F16): two independent full `emit` runs produced **byte-for-byte identical**
  `metamodel_predictions.csv`, `metamodel_predictions_raw.csv` and `strategy_weights.csv` (1011 rows
  each; canonical == calibrated). TensorFlow op-determinism is best-effort, which is exactly why
  **Keras-VSN is kept off the selectable path** — the grader's hidden-half re-run would otherwise
  risk selecting a non-reproducible model.
- **No leakage:** right-edge truncation-invariance of the whole feature stack incl. the F16
  concept-drift family (`test_features.py`); zero train/test `t1`-overlap after purge, with the
  **per-instrument `embargo_p90`** advanced on each instrument's own date axis (S2.6,
  `test_cross_validation_embargo.py`), CPCV = 15 paths (`test_cross_validation.py`); fitted regime
  blocks (incl. F17) fit on a contiguous prefix (never non-contiguous CPCV groups); the frozen
  `feature_matrix.parquet` is never consumed — asserted by `test_no_metamodel_module_reads_frozen_parquet`.
- **Config-driven window:** the prediction window is a `PipelineConfig` field, never hardcoded to
  Jan–Jun 2022 — the grader swaps in the hidden Jul–Dec 2022 half by changing one config value.
- **Data loading conforms to the base (X.11):** both the deliverable (`emit.py` `main()`) and the §6
  barrier backtest ingest OHLCV + signals through the base's **`stml.io.load_clean_data()`** — i.e.
  the authoritative `missing-data-report.md` policy: **keep the 765 zero-volume settle rows, drop
  only the 3 Sunday 2005-05-08 rows, and never forward-fill structural NaNs**. No bespoke zero-volume
  drop or ffill is introduced, so the metamodel sees byte-for-byte the shared base's cleaned panel —
  ruling out a silent data-cleaning inconsistency with the rest of the cohort.

## Limitations (honest)

- **No deployable edge is demonstrated — "insufficient evidence", not a proven failure (the
  headline, LR-6).** The pooled §6 Sharpe is **not statistically distinguishable from zero**: t = 0.34
  on n = 127, and the primary studentised block-bootstrap 95% CI **[−0.09, +0.15] contains 0**
  (S6.14). The deflation gate *corroborates* (pooled DSR [0.26 → 0.08], PBO 0.36, MinBTL up to 3.1y
  vs ~0.5y OOS), and so do AUC ≈ 0.5, near-zero cluster MDA (§4), and **no positive directional
  timing** (Pesaran–Timmermann negative everywhere, §5). The positive Sharpe is diversification +
  vol-targeting + the barrier exit, not act/skip skill. **One honest wrinkle (resolved — S5.11/S5.12):**
  the *pooled* Treynor–Mazuy γ here does not even print positive (−0.51, t = −0.35) — and the +1.18,
  t=2.55 it printed under the OLD global barrier was, equally, a scale-aggregation artefact **in either
  sign**, not timing skill; PT, the scale-invariant directional test, shows none. **Where the per-class
  barrier leaves a sleeve with no signal — Energy — the metamodel correctly abstains** (zero OOS
  positions) rather than manufacturing one. The per-class barrier + per-instrument embargo left the
  verdict unchanged: IC still ≈ 0, which strengthens the Fundamental-Law null rather than weakening it.
- **Calibration is now shipped (S3.9), not deferred:** per-class Platt fit train-only on the
  selected models cuts held-out ECE sharply (Metals most of all) with AUC unchanged, and the
  deliverable ships the calibrated p̂. The remaining selection-side gap is the **nested-CPCV real-data
  run**, still deferred (a meaningful one is ~15×5×5 fits per class) alongside the S6.5 stub.
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
  two literature-note flags are resolved honestly: the **Ang–Bekaert Table-2 correlation *values* are
  not quoted** here (the Japan/UK figures are sourced to Guidolin–Timmermann; Ang–Bekaert is cited
  only for the p = 0.156 Wald caveat), and the **Carver 2015 Ch.9 page is marked pending print-edition
  confirmation** rather than asserted precisely. The non-existent **"Kang & Kim (2025)"** anchor (a
  conflation of Fu, Kang, Hong & Kim 2024 — a GA/pairs-trading study, not meta-labelling) has been
  removed from the LR-1 citation set; it never appeared in this methodology.
- Equity instruments start late (es1s 1997, fesx1s 1998, nq1s 1999) — thin pre-2020 history for
  fitted features.
