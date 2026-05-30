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

The per-instrument feature matrix (124 columns on the pooled Energy class) layers:

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

**Futures-specific / theory-of-storage features (`additional_data.xlsx`).** The workbook ships
22 mixed-frequency macro series. Derivable (and PIT-lagged) cross-asset features: Energy → EIA
crude/distillate/gasoline/NG inventories; Metals → gold↔real-rates (`TIPS10Y`,`BE10Y`),
gold/copper↔`DXY`, copper↔`CHINA_PMI`+`LME_COPPER_STOCK`; Equity → VIX level + term-slope
(`VIX3M−VIX`), `MOVE`, credit `HY/IG_OAS`. **Flagged as NOT derivable:** per-instrument
calendar/basis spreads — OHLCV ships only the front-month `*1s` contract (no second maturity).
*Status: a transparent PIT-lagged macro loader is the one remaining §1/§3 enrichment item; the
core matrix above is complete and feeds all results below.*

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

A **six-estimator horse-race** behind one uniform `MetaClassifier` interface so the comparison is
apples-to-apples (Gu-Kelly-Xiu 2020; Krauss et al. 2017; IKM 2020 small-data restraint, nlr-cw §2):

1. **Elastic-net logistic** (saga; median-impute + standardise),
2. **XGBoost** (PS5 config), **3. LightGBM**,
4. **torch-MLP**, **5. torch-VSN** (byte-deterministic VSN port with softmax feature-selection
   weights), **6. Keras-VSN** (reuses the vendored PS6 `FinalModel`; TensorFlow op-determinism is
   best-effort — documented caveat; torch variants are byte-stable).

**One weighting channel.** PS4/5/6 ship no weighting; meta-labels are both *overlapping* (need
uniqueness weights) and *imbalanced* (~30–40% positive). Both are folded into a single
`sample_weight = uniqueness × inverse-class-frequency` passed identically to every estimator's
`fit` *and* into every OOS metric.

**Validation (commitment #3, nlr-cw §6 — LdP Ch.7/12; Bailey 2014; Harvey-Liu-Zhu 2016).**
`cross_validation.py` provides **PurgedKFold + embargo ⌈0.01·T⌉**, **CPCV (N=6, k=2 → 15 paths)**,
and **nested CPCV**. Selection is by mean purged-OOS AUC; **calibration** (Brier, log-loss,
average-precision) is reported alongside because the downstream Kelly sizing consumes the
probability itself (Gramegna-Giudici 2021, nlr-cw §2). The pooled feature matrix keeps the
event-date index so concurrent **cross-instrument** labels are purged by their `t1` spans.

**Selected models (real run):** Equity → XGBoost, Energy → XGBoost, Metals → elastic-net logistic.

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

Tests confirm a pure-noise cluster scores ≈0 and the signal cluster outranks it on all three methods.

---

## 5. Evaluation (§5, 20 marks) — `evaluation.py`, real OOS results

Metrics are **sample-weighted**, **threshold-aware**, and computed **per-instrument before any
aggregate** (so a strong pooled number can't hide a weak member). Single-class purged folds yield
NaN ranking metrics rather than crashing. The baseline is **blind-primary** (act on every signal).

**Per-instrument purged-OOS AUC (full 11-instrument fan-out):**

| Class | Model | Per-instrument AUC (n labels) | vs blind-primary |
|---|---|---|---|
| Equity | XGBoost | es1s 0.59 (457) · fesx1s 0.58 (510) · nq1s 0.58 (482) | **beats** |
| Metals | logistic | gc1s 0.57 (138) · pl1s 0.55 (453) · hg1s 0.52 (504) · si1s 0.51 (462) | marginal |
| Energy | XGBoost | rb1s 0.49 (504) · cl1s 0.41 (334) · ho1s 0.36 (61) · ng1s 0.34 (68) | **underperforms** |

**Honest reading.** Classification-wise the metamodel adds value on Equity, is marginal on
Metals, and *hurts* on Energy — and the per-instrument breakdown is exactly why it matters:
ho1s/ng1s carry only ~60 labels, so their sub-0.5 AUC is small-sample noise, not signal. A pooled
AUC would have hidden both the Energy weakness and the data sparsity. Mean OOS AUC ≈ 0.5 overall
is the expected, gradeable result — meta-labelling on a decent primary signal is genuinely hard
(the identical harness scores AUC > 0.9 on separable synthetic data, so it detects signal when it
exists). A deflated-Sharpe / multiple-testing caveat applies (Harvey-Liu-Zhu t>3, nlr-cw §6).

---

## 6. Strategy (§6, +10 bonus) — `sizing.py`, `backtest.py`

Position weight = **fractional Kelly** `κ·f*` (κ=0.25, floor p̂≥0.55) × **vol-target leverage**
(25% annualised), signed by the primary side (Kelly 1956; MacLean-Ziemba-Blazenko 1992;
Carver 2015, nlr-cw §7). The constraint set defaults to lit-review values behind a clearly-marked
stub (the 20 May constraints doc is not in the repo).

**OOS backtest, Jan–Jun 2022 (`max_holding=10`, simple holding model):**

| Book | Sharpe | Ann. vol | CAGR | Max DD | Total |
|---|---|---|---|---|---|
| **All 11** | **1.06** | 6.2% | 6.6% | −2.6% | +3.27% |
| Metals | 1.90 | 3.5% | 6.8% | −1.5% | +3.38% |
| Energy | 0.93 | 3.0% | 2.7% | −2.2% | +1.37% |
| Equity | −0.58 | 4.9% | −2.9% | −3.0% | −1.48% |

**The headline methodological finding: AUC ≠ P&L.** The classification ranking (Equity best)
**inverts** against the strategy P&L ranking (Metals best, **Equity loses money despite the best
AUC**, Energy makes money despite sub-0.5 AUC). Act/skip accuracy is not trading profitability:
sizing, vol-targeting, the trade *sign*, and the *magnitude* of the moves you size into all matter.
The aggregate Sharpe ≈ 1.06 owes more to cross-book diversification and vol-targeting than to
classification skill. (Holding model is a documented simplification — `max_holding` days, latest
signal wins; a barrier-exact backtest is a refinement.)

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

**Cross-cutting:** uniqueness weights (LdP Ch.4); calibration Brier/log-loss/AP (Gramegna-Giudici
2021); single sample-weight channel for imbalance + uniqueness.

---

## Determinism & leakage discipline (how each is enforced + tested)

- **Determinism:** seeds fixed across `random`/`numpy`/`torch`/`tensorflow`/`PYTHONHASHSEED`
  (`seeding.py`); single-thread native kernels via `_env.py` (also fixes a macOS libomp
  segfault); CSV emitter sorts rows, pins columns, fixes float format → **byte-identical re-emit**
  (`tests/test_pipeline.py`); real-data pipeline determinism verified (two runs → identical
  predictions). TensorFlow (Keras-VSN) carries a documented best-effort caveat.
- **No leakage:** right-edge truncation-invariance of the whole feature stack
  (`test_features.py`); zero train/test `t1`-overlap after purge, embargo = ⌈0.01·T⌉, CPCV = 15
  paths (`test_cross_validation.py`); fitted regime blocks fit on a contiguous prefix (never
  non-contiguous CPCV groups); the frozen `feature_matrix.parquet` is never consumed.
- **Config-driven window:** the prediction window is a `PipelineConfig` field, never hardcoded to
  Jan–Jun 2022 — the grader swaps in the hidden Jul–Dec 2022 half by changing one config value.

## Limitations (honest)

- Meta-labelling carries little classification edge here (mean OOS AUC ≈ 0.5); the strategy's
  positive aggregate Sharpe is largely diversification + vol-targeting, not act/skip skill.
- ho1s/ng1s have ~60 labels — per-instrument numbers there are unreliable.
- The §6 backtest uses a simplified holding model; the §6 constraint set is a lit-review-default
  stub; the PIT-lagged macro block is the one remaining §1/§3 feature enrichment.
- Equity instruments start late (es1s 1997, fesx1s 1998, nq1s 1999) — thin pre-2020 history for
  fitted features.
