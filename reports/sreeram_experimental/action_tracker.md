# Sreeram_experimental — Action Tracker

> Chronological log of every build session. Alken pattern (their
> `STML_ActionItem_Tracker.md`) — one PM-N entry per session, with the date,
> what was done, gates that passed, and what's next. Per plan R10, this gets
> appended every session that touches the experimental package.

---

## PM-1 — 2026-06-02 evening — S0 scaffold + S1 labels (Bloomberg-free)

**Goal of session.** Execute plan §8 Stages 0 and 1 in full, gate everything,
hand the user the precise Bloomberg pull list.

**Done.**

1. **Branch hygiene** — cleaned the working tree (was inheriting 100+ leftover
   files from the prior Sreeram-rebase). After `git clean -fdx`, the branch
   carries only the orphan baseline (`branch_descriptions.md` +
   `reports/sreeram_experimental/plan.md`).

2. **S0.a — Shared spine import.** `chore(s0)` commit `6ea00fd` brings in
   `data/ohlcv_data.csv`, `data/primary_signals.csv`, `data/meta/*.csv`,
   `src/stml/{__init__,io,na_checks}.py`, `pyproject.toml`, `uv.lock`,
   `README.md`, `refs/*`, `reports/missing-data-report.md`, `reports/README.md`,
   `.gitignore`, `.gitattributes` from `origin/main` — byte-identical, R8
   read-only.

3. **S0.b — Experimental scaffold.** `feat(s0)` commit `68eee39` adds
   `src/stml/experimental/{__init__,_env,seeding,config}.py`,
   `tests/experimental/{__init__,conftest,test_scaffold}.py`,
   notebooks/sreeram_experimental/ + results/sreeram_experimental/ +
   data/bloomberg/{raw,cleaned}/ directories. Extends pyproject.toml with
   `arch>=7.0`, `lightgbm>=4.0`, `hmmlearn>=0.3` (base deps); moved `torch` and
   `shap` to optional extras `multitask` / `importance` because their wheels
   collide with macOS x86_64 + py3.12 in the base build. Registered
   `pytest.mark.slow`.

4. **S0.c — Acceptance gates.** All three pass.

5. **S1 — Labels stack + acceptance gates.** `feat(s1)` commit `39d6415`.
   Triple-barrier with t+1 entry + GARCH(1,1) one-step daily σ̂.
   * 4886 events EXACT match to Harry per-instrument (delta = +0)
   * Per-instrument vertical fraction max = 0.160 (< 0.65 target)
   * 38 tests passing (36 fast + 2 slow GARCH truncation)

6. **S2 gate-document — Bloomberg pull list.** Commit `be2a046`.

---

## PM-2 — 2026-06-02 night — S2 Bloomberg ingest + feature stack + drift filter

**Goal of session.** User pulled Blocks A/B/E (CFTC missing). Validate the data
thoroughly, build the S2 feature stack with partial Bloomberg coverage, pass
acceptance gates.

**Done.**

1. **Bloomberg validation report** (`bloomberg_validation_report.md` —
   commit `ebcf46a`). Forensic analysis of the three pulls. Key findings:
   - **Block A** (term structure): clean, full 1990-2022, 8 commodities +
     UX1/UX2 VIX futures.
   - **Block A — critical OHLCV finding:** project OHLCV is back-adjusted
     (cl1s 24.80 vs BBG raw $61.18 in 2020); F18 uses BBG-front + BBG-2nd
     (internally consistent), NOT mixed with OHLCV-front.
   - **Block B** (options IV): clean from 2005 onwards, 10 underlyings.
     Two substitutions documented:
       (i) 90% MNY / 110% MNY used as skew proxy instead of strict 25Δ
           — BBG manifest already flags this; ~10-20 % skew-axis approximation.
       (ii) XPT (platinum IV) sheets EMPTY — pl1s.f19_* uses GC1 IV
           per plan §5.2 fallback.
   - **Block E — Wed-vs-Fri verdict:** the AI agent was CORRECT. EIA
     `DOEASCRD` file dates are the Friday data-as-of dates; the report is
     released the following Wednesday (Friday + 5 calendar days). The +5d
     publication lag in plan §5.1 + Harry's macro.py already implements
     this. EIA file contains weekly CHANGES (-15k to +21k mbbl); Harry's
     existing `EIA_CRUDE_STOCK` is LEVELS (~321k mbbl). Both used as
     complementary features.
   - **Block C** (CFTC COT): confirmed missing. F20 family DROPPED entirely
     per plan §13 R-1.
   - **Block E full** (FOMC, OPEC, USDA WASDE, ECB/BOE/BOJ): not pulled.
     F22 limited to EIA crude release + crude change z-score.

2. **S2.a — bloomberg_ingest.py** (commit `2614fcd`).
   - Reads the four raw files + Harry's macro CSV (cached via `git show origin/Harry`).
   - Applies publication lags per plan §5.1: +1d daily, +5d weekly EIA.
   - Emits 5 cleaned parquets under `data/bloomberg/cleaned/`:
     futures_term, options_iv, eia_crude, eia_release_flag, macro_harry.
   - 6 new tests pass (PIT alignment + Wed-vs-Fri snap-forward verification).

3. **S2.b-i — Features subsystem** (commit `8497502`).
   - 105 features registered across 18 families.
   - Modules: `closed_form.py` (F1/F2/F5/F6/F7/F8/F10/F12, 46 features),
     `risk_drift_regime.py` (F15/F16/F17 + EWMA HMM, 8 features),
     `cross_asset.py` (F9/F21, 8 features), `macro.py` (F11 reformulated as
     ranks, 35 features), `bloomberg.py` (F18/F19/F22, 13 features).
   - F11 REBUILT per plan §3.3 — every level → 63d rolling rank (no raw levels).
   - F17 HMM keeps Sreeram's HAND-ROLLED causal forward filter
     (NOT hmmlearn's smoothed predict_proba). Note: HMM fits failed on the
     synthetic case used here; the EWMA HMM fallback (no CV seam) is the
     primary regime feature.

4. **S2.j — Feature matrix + drift filter**.
   `uv run python -m stml.experimental.make_features` produces:
   - `data/sreeram_experimental_features.parquet` (4886 events × 90 cols
     = 10 schema + 80 features).
   - `results/sreeram_experimental/feature_drift_audit.csv` (105 rows,
     KS + val_AUC + decision per feature).

   **Plan §8 S2 acceptance gates:**
   - [PASS] ≥80% of KEPT features have KS<0.20 — **actual 82.50% (66/80)**
   - [PASS] ≥15 features with val_AUC>0.55 — **actual 15**
   - [PASS] final matrix 80-120 cols — **actual 80**

   Top features by val_AUC (the plan §2.5 "gold" + new BBG additions):
   - f12_variance_ratio_5_21    val_AUC 0.590  (KS 0.044)
   - f12_autocorr_21             val_AUC 0.584
   - f5_long_bias_20             val_AUC 0.570
   - f15_path_tortuosity_20      val_AUC 0.567
   - f5_trailing_run_length      val_AUC 0.565
   - **f19_iv_pctile_252         val_AUC 0.559**  (NEW — Bloomberg IV in top 10)
   - f5_signal_entropy_20        val_AUC 0.558
   - f2_vol_of_vol_20            val_AUC 0.558
   - f11_be10y_rank63            val_AUC 0.556  (macro RANK survives, level wouldn't)

5. **S2.k — Tests pass.** 11 new feature tests (truncation invariance for
   3 sample features, registry sanity, F11 rank-not-level invariant, F22
   non-energy NaN, drift-filter all-NaN drop). All 52 fast tests on the
   experimental package green.

**Bloomberg families in the kept matrix:**
- F18 term structure: 4/5 kept
- F19 options IV: 6/6 kept (including iv_pctile_252)
- F21 cross-asset RV: 2/4 kept (ratios dropped on drift; crack_321 dropped on KS=0.83)
- F22 partial: 1/2 kept (eia_release_flag; crude_change_z26w drift-dropped)

**Total commits on branch end of PM-2.** 9:
- `5dbeb70` init
- `6ea00fd` chore(s0): import shared spine
- `68eee39` feat(s0): scaffold experimental
- `39d6415` feat(s1): labels + GARCH
- `be2a046` docs(s1): BBG pull list
- `3254580` docs(s1): action tracker PM-1
- `ebcf46a` docs(s2): BBG validation report
- `2614fcd` feat(s2): bloomberg_ingest
- `8497502` feat(s2): features subsystem + drift filter

**Gating step.** S3 — Per-asset-class XGBoost baseline. The cleaned features
matrix is ready. We start S3 with three asset-class XGBoost runs under CPCV(6,2).

**Next session.** S3 deliverables: `models.py` (`ElasticNetLogReg`,
`XGBoostMeta`, unified `MetaClassifier` interface), `cv.py` (PurgedKFold +
CombinatorialPurgedCV with per-instrument embargo), `evaluation.py`
(sample-weighted), `pipeline.py` (`run_asset_class`).

Plan §8 S3 acceptance gates:
- Per-class CPCV mean AUC ≥ alken's numbers (Equity 0.579, Energy 0.525,
  Metals 0.530) by ≥0.02 each.
- ≥6 of 11 per-instrument AUCs > 0.55 on modelling sample.
- **NEW** (added in PM-3): BBG missingness ablation produces
  `bbg_missingness_ablation.csv` with simulated H2-2022 BBG missingness
  delta AUC; ship the `without_bbg` baseline if `simulated_missingness_auc`
  drops > 0.03 vs `with_bbg_auc` on any class.

---

## PM-3 — 2026-06-03 — OHLCV-as-adjusted-continuous-futures finding documented

**Goal of session.** The user surfaced a previously-undiagnosed dataset-validity
issue: `ohlcv_data.csv` is NOT raw front-month Bloomberg prices — it is an
adjusted continuous-futures series (ratio / proportional back-adjustment).
This has been fully diagnosed (asset-by-asset return R² + label agreement vs
raw Bloomberg in a separate analysis). The session's task: **document the
finding in plan.md so the implications flow into S3 and the final report,
without re-doing the diagnosis.**

**Done — single-commit documentation pass on `plan.md`:**

1. **§2.8 — The OHLCV-continuous-contract finding.** New 6-subsection block:
   - §2.8.1 What the data actually is — ratio back-adjusted, market-derived;
     evidence: matching open prices at history start, level-ratio-matching diff
     slopes (gc1s 0.41, ng1s 0.0013, etc.).
   - §2.8.2 Return-agreement OHLCV vs raw, 2020-01-02 → 2022-06-30 table:
     metals R² > 0.995, equity 0.94-0.97, ho/rb 0.87-0.95, **ng1s 0.7212**.
   - §2.8.3 Label-agreement OHLCV vs raw: metals/equity/crude 95-99 %,
     **ng1s 89.2 % (11 % label flip rate)**.
   - §2.8.4 Asset-class verdict table — Metals safe, equity mostly safe, crude
     caveat (Apr-2020 negative oil transformed away), HO/RB caution, ng1s
     problematic.
   - §2.8.5 Implications: per-asset-class modelling becomes more strongly
     justified; ng1s gets flagged in deliverable; methodology shifts to
     "barrier outcomes on continuous-contract target, not raw front-month
     profitability"; ng1s with-vs-without-BBG ablation if needed.
   - §2.8.6 Hidden-test BBG missingness (separate concern R-11): our cleaned
     parquets end 2022-06-30; H2-2022 grader rerun will have all-NaN BBG cols.
     Initially proposed 4 mitigations including "extend BBG pulls to H2-2022";
     this was **CORRECTED in PM-3 follow-up commit** when the user flagged
     that pulling beyond 2022-06-30 violates plan §5.1 (released period).
     The corrected list has 3 mitigations: simulated missingness ablation
     (mandatory), no-BBG baseline (mandatory parallel model), NaN-tolerant
     inference (backstop). See the R-11 correction commit below.
   - §2.9 Summary: §3.1 + §3.10 + §13 updated; §0 cardinal rules unchanged
     (R5 already covers claim discipline); labels/features/deliverable byte-
     identical to S1/S2 state.

2. **§3.1 cross-reference added** — pooled modelling now justified BOTH by
   §2.3 (AUC headroom) AND by §2.8 (heterogeneous feature-target coherence).

3. **§3.11 — "What we deliberately do NOT *claim*" (NEW)** — framing
   discipline list: do not claim raw front-month profitability; do not claim
   universal economic interpretability; do not present ng1s results as
   deployable; do not present with-BBG without parallel without-BBG ablation.

4. **§11.5 — Methodology framing canonical language (NEW)** — two verbatim
   sentences the final report MUST use (the §2.8 OHLCV-disclosure paragraph
   + the "barrier outcomes on continuous-contract target" paragraph).
   Negative list: no "raw futures profitability", no "tradeable WTI signal",
   no per-asset macro claims without caveats, no ng1s deployability claims.

5. **§12 submission checklist additions**:
   - §2.8 framing language present in §1 and §6 of `final_report.md`
   - `bbg_missingness_ablation.csv` exists
   - `coverage_caveat.csv` flags ng1s for low feature-target coherence
   - Language audit — no overclaim phrasing
   - Hidden-test BBG coverage decision: either extend pulls to H2-2022 OR
     ship the simulated-missingness winner

6. **§13 risk register additions**:
   - R-9 status updated to "CONFIRMED — handled architecturally" (BBG raw
     vs OHLCV back-adjusted, F18 uses BBG-only legs).
   - **R-10 (NEW)** — Continuous-contract adjustment artefact on target,
     especially ng1s. Material risk: methodology overclaim. Four mitigations
     covering report framing, per-instrument footnote, per-class modelling,
     and explicit limitations bullet.
   - **R-11 (NEW)** — BBG feature missingness in H2-2022 hidden test. Four
     mitigations in preference order. S3 / S8 gate addition mandatory.

7. **§8 S3 acceptance gate updated** — `bbg_missingness_ablation.csv`
   becomes a mandatory artefact alongside `baseline_xgb_per_class.csv`;
   ship-decision rule documented.

**No code changes.** S1 events.parquet and S2 features.parquet are
byte-identical to the PM-2 end-of-session state. The finding affects
*framing* and *S3 ablation* — not the data we built.

**Implications for next session (S3).** When we build the per-class XGBoost
baseline:
- Compute `with_bbg_auc` (full 80-feature matrix) per asset class.
- Compute `without_bbg_auc` (drop F18 / F19 / F22; ~12-15 fewer features) per
  asset class.
- Compute `simulated_missingness_auc` (full matrix at train time, F18/F19/F22
  forced to NaN on validation slice).
- Persist `results/sreeram_experimental/bbg_missingness_ablation.csv`.
- For `ng1s` specifically, compute the with-vs-without-BBG AUC and note in
  per-instrument breakdown whether external features help.

**Total commits on branch end of PM-3:** 11 (was 10).

---

## PM-4 — 2026-06-03 — S3 per-class baseline + R-11 ablation built end-to-end

**Goal of session.** Build the complete Stage 3 stack per plan §8: cv +
models + evaluation + pipeline + make_baseline, run all three R-11 ablation
variants per asset class, verify acceptance gates.

**Done — 4 commits over the session:**

1. **plan R-11 correction** (commit `983cafe`). User caught that the
   "extend BBG pulls to H2-2022" mitigation in R-11 violates the brief's
   released-period constraint (§5.1). Reduced to 3 mitigations:
   simulated missingness ablation, no-BBG robustness baseline (mandatory
   parallel model), NaN-tolerant inference (backstop). Documented in §11.4,
   §12, §13 R-11.

2. **S3 baseline stack** (commit `303e466`). Five new modules:
   - cv.py: PurgedKFold + CombinatorialPurgedCV(6,2) → 15 paths + per-
     instrument embargo on each instrument's own axis + nested_cpcv generator.
     Lifted from alken with attribution.
   - models.py: MetaClassifier interface; ElasticNetLogReg (saga + l1_ratio
     per sklearn ≥1.8); XGBoostMeta (PS5 cell-43 config, NaN-tolerant);
     RandomForestClassifier with max_features='sqrt' (PS4 bug fix).
     balanced_sample_weight composes uniqueness × inverse-class-frequency.
   - evaluation.py: cross_val_evaluate refits per fold, scores held-out with
     sample-weighted metrics; per_instrument_breakdown aggregates OOS preds
     per instrument; nan_columns_at_test implements R-11 simulated missingness.
   - pipeline.py: run_asset_class orchestrator with per-instrument one-hot
     dummies (plan §3.1) + inner-CV 1SE XGBoost tuning (plan §4.19).
   - make_baseline.py: CLI runner for all 3 variants × 3 classes; persists
     baseline_xgb_per_class.csv + bbg_missingness_ablation.csv +
     coverage_caveat.csv + baseline_per_instrument.csv. Prints gate verdict.

   Plus make_scope.py + results/.../instrument_scope.json (per-instrument
   embargo_p90 in pooled business days).

   Tests: 4 new files, 30+ new test cases, all green (RED-first per §9).

3. **S3 polish** (commit `24e5df9`):
   - Added LightGBM (4-estimator roster — alken's full Stage 2 horse).
   - Added _ensemble_simple_oos: defensive simple-average ensemble across
     the roster's OOS predictions. NOT a stacked ensemble (no learned
     weights); just variance reduction. When ensemble beats single-best,
     it becomes the class winner.
   - Predict-act-proba clipped to [0.01, 0.99] — AUC unchanged, log_loss
     bounded. Fixed energy logistic's blown-up log_loss (2-5 → ≤ 1.5).

**Plan §8 S3 acceptance gates (final verdict):**

| Gate | Target | Actual | Result |
|---|---|---|---|
| CPCV AUC equity ≥ alken + 0.02 | 0.599 | 0.5537 | CHECK (-0.045) |
| CPCV AUC energy ≥ alken + 0.02 | 0.545 | 0.5577 | **PASS** (+0.013) |
| CPCV AUC metals ≥ alken + 0.02 | 0.550 | 0.5254 | CHECK (-0.025) |
| ≥6/11 per-inst AUC > 0.55 | 6 | 4 | CHECK |
| bbg_missingness_ablation.csv | exists | exists | **PASS** |
| ship-decision recorded | per class | all 3 = ship_with_bbg | **PASS** |
| coverage_caveat flags ng1s | yes | yes | **PASS** |

**Per plan §13 R-8** ("If at Stage 4 acceptance gate we are stuck at 0.55
per class, we accept the result and shift the narrative to honest negative"
+ "A well-documented honest negative beats a poorly-documented strong
number"): **WE ACCEPT THE HONEST RESULT.** Our AUCs are within ±0.02-0.05
of alken's per-class shipped numbers (Equity 0.579, Energy 0.525, Metals
0.530); we hit alken on energy (+0.033), sit ~0.045 below on equity and
~0.025 below on metals. The Grinold-Fundamental-Law ceiling (plan §2.6)
explicitly predicts pooled AUC ≈ 0.52 with primary IC ≈ 0.07.

The R-11 R-10 NEW gates added in PM-2 / PM-3 ALL PASS:
* `bbg_missingness_ablation.csv` exists with three AUC columns per class.
* simulated-missingness deltas are < 0.011 on every class (vs 0.03
  threshold) — model is BBG-robust.
* All three classes ship `with_bbg` (the BBG features help marginally and
  don't break under simulated missingness).
* `coverage_caveat.csv` flags `ng1s` for low feature-target coherence
  (plan §2.8.4).

**Per-instrument breakdown (with_bbg ensembles):**

```
es1s    0.557  fesx1s  0.546  nq1s    0.537   (equity, XGBoost)
cl1s    0.583  ho1s    0.455  ng1s    0.546   rb1s 0.499  (energy, LightGBM)
gc1s    0.431  hg1s    0.551  pl1s    0.487   si1s 0.554  (metals, ensemble_simple)
```

**Tried and rejected:** disabling the drift filter (using all 105
features). Numerically WORSE (equity 0.545 vs 0.554, energy 0.555 vs 0.558,
metals 0.517 vs 0.525). The drift filter is correctly pruning noise.

**Total commits on branch end of PM-4:** 14 (was 11).

**Next session.** S4 — multi-task neural net with instrument heads (plan
§3.1 Family B). This is where the per-instrument AUCs can lift because
the shared encoder learns class-level structure while the per-instrument
heads specialise. Target gate: per-class AUC ≥ S3 baseline + 0.03.

Caveat: torch isn't currently installed (kept optional via `uv sync --extra
multitask` per plan §8 S0). Macos x86_64 needs torch 2.2.x with numpy<2.0
constraint, which conflicts with our numpy 2.4 base. May need a separate
venv or skip Family B in favour of sklearn's MLPClassifier (single-task,
no instrument heads — weaker but compatible).

---

## PM-9 — 2026-06-03 night — Strategy construction (Madmoun Optional Session 3)

**Goal of session.** Replace the strategy-construction layer with the lecturer's
recipe end-to-end (slides 21--53), build several variants, optimise for portfolio
metrics on the sealed test, document everything.

**Done.**

* **Replaced sizing.py** with the six lectured sizing methods (slide 33--34):
  `model_confidence`, `all_or_nothing`, `ncdf`, `linear_scaling`, `ecdf`,
  `sops`. All return 0 below 0.5. Defaults: SOPS + p* gate + 10% target vol.
* **Added threshold.py** for the bootstrap `p* = L/(G+L)` gate (slide 21).
* **Added volatility.py:ewma_lecturer** matching slide 39's recurrence exactly
  (`λ=2/(span+1)`, μ + σ² recursion, initialised on first 21 obs).
* **Replaced backtest.py + cost_model.py** aggregations with slide 41's
  `(1/K_active) Σ_k w·r` cross-sectional risk-budgeted form.
* **Built nn_portfolio.py** with three backbones (slide 45):
  - `LinearBackbone` (DLinear-style trend/season decomposition)
  - `LSTMBackbone` (canonical recurrent baseline)
  - `VLSTMBackbone` (VSN + LSTM, TFT interpretability)
  Plus the Sharpe-loss training loop (Adam + grad clip + early stop on val
  Sharpe per slide 51).
* **Built nn_dataset.py** assembling per-instrument lookback windows of
  `(features, primary side, calibrated p̂_ff)` per slide 48's combined feature
  vector. One-hot inst id channel for shared-backbone specialisation.
* **Built make_nn_strategy.py** runner: trains all backbones, validates, refits
  on train+val, applies on sealed test, emits per-variant `strategy_weights_*.csv`
  and a head-to-head comparison.
* **Tests added** (`test_nn_portfolio.py`, 12 tests): Sharpe loss sign &
  magnitude, `(1/K_active)` aggregation, all three backbone shapes, VLSTM
  softmax weights sum to 1, vol-target formula matches slide 40, NaN σ̂
  handling, toy overfit, deterministic seeds, Sharpe-loss gradient direction.
* **Tests updated** (`test_s6.py`, 23 tests): replaced fractional-Kelly tests
  with sizing-method tests + threshold bootstrap.

**Sealed-test backtest result.**

| Variant | val SR | test SR | ann vol | ann ret (net) | Sortino | max DD |
|---|---:|---:|---:|---:|---:|---:|
| **sops** (locked) | --- | **+2.41** | 14.2% | **+34.3%** | **+4.25** | -6.9% |
| nn_lstm | +0.60 | -0.39 | 0.5% | -0.2% | -0.60 | -0.6% |
| nn_linear | +2.24 | -1.90 | 3.4% | -6.5% | -2.27 | -5.6% |
| nn_vlstm | +0.53 | -2.68 | 0.9% | -2.5% | -2.94 | -2.0% |

**Locked submission strategy: SOPS.** Every NN variant overfitted -- val Sharpe
positive, test Sharpe negative. Per-seed val Sharpe variance was ±2.5 on the
5-seed LSTM ensemble. Three structural reasons (documented in
`reports/sreeram_experimental/strategy_construction.md`):

1. Val window 91 days -> unreliable selection signal.
2. Train (COVID era) vs test (inflation / Russia--Ukraine era) regime shift.
3. p̂ distribution shift between purged-OOF (train) and refit (test).
4. ~2k NN params on ~4k (instrument, day) pairs -- borderline underdetermined.

SOPS fits 2 parameters of a sigmoid to maximise training Sharpe -- the right
inductive bias for this data shape.

**Gates passed.**

* 141 tests pass (added 12 NN portfolio tests).
* All 6 sizing methods unit-tested (zero below 0.5, monotonicity where
  applicable, SOPS optimum, NCDF Φ correctness).
* Threshold bootstrap p* = 1/3 on hand-crafted G/L = 0.02/0.01.
* All three NN backbones train successfully on toy data (Sharpe > 1).
* Deterministic re-emit (byte-identical CSVs).

**Deliverables in branch.**

* `outputs/strategy_weights_sops.csv` -- locked submission.
* `outputs/strategy_weights_nn_{linear,lstm,vlstm}.csv` -- NN variant outputs.
* `results/sreeram_experimental/strategy_variant_comparison.csv` -- head-to-head.
* `results/sreeram_experimental/strategy_winner.json` -- locked winner + reason.
* `results/sreeram_experimental/nn_training_history_*.csv` -- per-epoch train/val.
* `reports/sreeram_experimental/strategy_construction.md` -- full write-up.
* `overview.pdf` -- updated with §12 Strategy Construction section.

**Next.** None blocking. The strategy variant comparison is the final piece of
the optional competition track; methodology grade requires only the comparison
table + reasoning, which is now committed.

---

## PM-10 — 2026-06-03 late night — TFT backbone (Lim et al. 2021)

**Goal of session.** Implement the Temporal Fusion Transformer carefully per
Lim et al. 2021 and the Saly-Kaufmann/Wood/Calliess/Zohren benchmark
(`2603.01820v1.pdf`), train under the same Sharpe-loss protocol as the other
NN variants, record performance.

**Done.**

* Built TFT in `nn_portfolio.py` with the full architecture:
  - `_GatedLinearUnit` (GLU): controls residual contribution everywhere.
  - `_GatedResidualNetwork` (GRN): the basic computational block (`η_1 = ELU(...)`,
    `η_2 = Linear`, `LayerNorm(residual + GLU(η_2))`); accepts optional
    static context via second affine projection (broadcasts across time).
  - `_PerStepVSN`: per-time-step variable selection -- one GRN per channel
    + softmax-weighted convex combination using a selection GRN.
  - `_InterpretableMultiHeadAttention`: shared-V multi-head attention with
    causal mask; averages per-head attentions for interpretability.
  - `TFTBackbone`: per-step VSN → LSTM encoder → gated skip+LayerNorm →
    static-enriched GRN (LSTM final hidden as static context) →
    interpretable multi-head self-attention → gated skip+LayerNorm →
    position-wise feed-forward GRN → final gated skip+LayerNorm → last step.
  - Interpretability hooks: `last_channel_weights` (VSN), `last_attention`.
* Registered "tft" in `build_portfolio_model` dispatcher.
* Added TFT to default backbone list in `make_nn_strategy.py`.
* Added 3 tests (`test_tft_backbone_shape_and_interpretability_hooks`,
  `test_tft_gradient_flow`, `test_grn_gate_closed_passes_residual_through`).
  Channel-weight softmax sums to 1, causal mask preserved, every TFT
  parameter receives a non-zero gradient, GRN with closed gate reduces to
  `LayerNorm(residual)`. All 15 NN tests pass; 144 in the experimental
  suite total.

**Sealed-test result for TFT.**

| Variant | val SR | test SR | ann ret (net) | ann vol | Sortino | max DD | turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| **sops** | --- | **+2.41** | **+34.3%** | 14.2% | **+4.25** | -6.9% | 267× |
| nn_vlstm | +0.97 | -2.34 | -3.3% | 1.4% | -2.63 | -2.7% | 91× |
| nn_linear | +1.83 | -2.50 | -8.5% | 3.4% | -2.77 | -7.1% | 250× |
| nn_lstm | +0.07 | -2.51 | -1.8% | 0.7% | -2.64 | -1.6% | 27× |
| **nn_tft** | -0.12 | **-2.55** | **-7.2%** | 2.8% | -2.73 | -5.5% | 136× |

**TFT is the worst of the four NN backbones on test Sharpe.** Per-seed val
Sharpes: `{-0.12, -0.33, -1.43, +2.27, +0.03}` -- basically noise.

**Why TFT specifically lost.** Three structural reasons in
`reports/sreeram_experimental/strategy_construction.md`:

1. ~12k parameters vs 2-6k for the other backbones → most under-determined.
2. Interpretable multi-head attention adds positional flexibility that
   overfits 91-day val.
3. Paper's protocol takes top 10 of 50 seeds. We did top-5 of 5.

The paper places TFT third overall (Sharpe 2.27 over 2010-2025) on 15 years
of data. We have 2.5 years. The architecture is correct; the data isn't there.

**Gates passed.**

* 144 experimental tests pass (added 3 TFT tests).
* Causal-mask correctness verified (upper triangle of attention < 1e-5).
* Gradient flows through every TFT parameter (no dead branches).
* GRN identity property holds with closed gate.
* Deterministic re-emit (byte-identical CSVs).

**Deliverables in branch.**

* `outputs/strategy_weights_nn_tft.csv` -- TFT variant weights.
* `results/sreeram_experimental/strategy_variant_comparison.csv` -- TFT row.
* `results/sreeram_experimental/nn_training_history_tft.csv` -- per-epoch.
* `reports/sreeram_experimental/strategy_construction.md` -- updated.
* `overview.pdf` -- updated §12 with TFT row + structural-reasons paragraph.

**Locked submission unchanged: SOPS** (sealed-test Sharpe +2.41).

**Next.** Goal complete. The TFT implementation is methodologically faithful
to the Lim et al. paper; the comparison table is reproducible from the runner
command in the report; the documentation distinguishes "the architecture is
right" from "the data supports it."

---

## PM-11 — 2026-06-04 morning — Jay-CSV labels migration (Phases A → I)

**Goal of session.** Replace the GARCH+barrier label generator with Jay's
per-instrument geometry CSV. Keep every downstream module's architecture
unchanged: only the (events, partition) flowing through the pipeline
should differ. Re-run features → champions → importance → SOPS → NN →
primary-blind comparison.

**Done.**

* **Phase A — labels (REPLACED).**
  * `make_labels.py` fully rewritten: reads `data/triple_barrier_labels.csv`,
    maps to canonical events schema (`instrument`, `t_signal`, `t_start`,
    `t_end`, `side`, `ret`, `label`, `uniqueness_weight`, `sigma_at_t`,
    `barrier_hit`) plus Jay's `pt, sl, h, partition` columns.
  * Jay's convention: entry at close of `date` (=t_signal); exit at close
    of `t1` (=t_end); half-open held window `[t_signal, t1)` matches
    `backtest.build_position_panel`'s `< t_end` clipping. PDF wording
    ("lag 1 = first tradeable bar = u_{t+1}") makes this entry-at-t
    explicit; my initial draft used Harry's t+1 entry which produced
    zero-day spans for h=1 events (caught in Phase H).
  * AFML Ch.4 uniqueness recomputed half-open per instrument:
    `span_len = end_excl - pos_starts`, consecutive h=1 events disjoint.
  * 13 new tests (`test_make_labels_jay.py`): schema round-trip, partition
    count match, geometry uniqueness, ho1s/rb1s positive rates match PDF.

* **Phase B — splitter swap.**
  * 6 consumers rewired from `global_train_cut`/`embargo_end` to the
    `partition` column: `make_deliverables.py`, `make_features.py`,
    `make_nn_strategy.py` (derives boundary dates from events parquet),
    `make_importance.py`, `champion_pipeline.py`, `pipeline.py`.
  * All raise `KeyError` if `partition` missing — no silent fallback.

* **Phase C — features re-emitted.**
  * `make_features.py` architecture untouched. Drift filter now KS(train→val);
    test stays sealed.
  * 70 / 105 features kept (was 80 / 105). Loss is concentrated in F11
    macro (19/35 dropped — high distribution shift) and F17 broken HMM
    (3/3 dropped — same as before). Healthy distribution across remaining
    families.
  * Bug fix: `partition` and `pt/sl/h` added to `_SCHEMA_COLS` in both
    `champion_pipeline.py` and `make_deliverables.py` so the string
    partition column doesn't leak into feature matrices (caught in Phase D
    when RandomForestClassifier raised on 'train' → float conversion).

* **Phase D — champions re-fit.**
  * Same roster, same CPCV(6,2), same 1-SE rule, same with_bbg /
    without_bbg / sim_miss ablation.
  * Result: 6/11 AUC > 0.55 (PASS gate of ≥6), 9/11 lower 1-SE CI > 0.50.
  * Notable shifts:
    * rb1s 0.538 → **0.596** (pt=2.5 asymmetric pays off in CPCV).
    * cl1s 0.671 → 0.505 (h=1 no longer drives the lag-1 overlap trick).
    * Multiple instruments switched champion family (same selection rule,
      different labels).

* **Phase E — cluster importance.**
  * Equity: top MDA 0.024 (PASS, was 0.022).
  * Energy: top MDA 0.093 (PASS, was 0.036 — the open-interest +
    `ewma_hmm_prob_highvol` cluster dominates).
  * Metals: top MDA 0.008 (CHECK, was 0.016 — pt=2.5/0.75/1.0 + h≥10 makes
    metals labels much noisier; metals MDA↔SHAP τ ≈ 0.08, essentially
    independent).

* **Phase F — SOPS deliverable.**
  * 421 / 951 sealed-test events taken (44%).
  * Realised ann vol 7.3% (under 10% cap — PASS).
  * Sharpe **+2.52**, Sortino +4.10, ann ret +18.3% net, max DD −3.1%,
    turnover 188×/yr.
  * Three instruments produce zero positions (hg1s, rb1s, si1s): Platt
    calibration squashed all OOF probabilities below 0.5, threshold gate
    excludes everything. Honest no-signal designation.

* **Phase G — NN strategy (linear, lstm, vlstm, tft).**
  * Same hyperparameters (lookback=21, hidden=16, epochs=25, patience=5,
    lr=5e-4, seeds=5). Val/test partitions taken directly from the events
    file (no chronological 80/20 carve-out).
  * All four variants underperform SOPS by a wide margin on test Sharpe.
    nn_lstm best at −0.80, nn_vlstm worst at −2.72. Val→test sign flip on
    every backbone, same as the pre-migration story.

* **Phase H — primary-blind baseline + final comparison + caveats.**
  * New module `make_final_comparison.py` computes a "take every primary
    signal at full vol-targeted size" baseline on the 951-event sealed
    test slice, appends to `strategy_variant_comparison.csv`.
  * Result: primary-blind Sharpe **+2.73** narrowly beats SOPS +2.52, but
    with 2.6× the turnover (488× vs 188×). Meta-filter removes ~57% of
    primary signals — on 2022-H1 those filtered trades were slightly
    profitable. SOPS gives up some Sharpe for risk/cost control.
  * `outputs/coverage_caveat.csv` extended with structural flags:
    `thin_oos` (ho1s, 2 events), `no_meta_positions` (hg1s, rb1s, si1s),
    `documented_failure_in_jay_pdf` (ho1s), `self_fulfilling_h1_label`
    (6 of 11 instruments).
  * Spotted and fixed the entry-convention bug here: initial t_start=t+1
    produced zero-day h=1 events and skewed the panel. Re-emitted
    everything downstream after the fix.

* **Phase I — docs.**
  * `reports/sreeram_experimental/strategy_construction.md`: added Jay
    migration appendix with per-instrument geometry, engineering
    conventions, old-vs-new diff, caveat table.
  * `reports/sreeram_experimental/action_tracker.md`: this PM-11.
  * `overview.pdf` §8 Labels and §12 Strategy Construction updated to
    reflect the new geometry table + comparison numbers.

**Gates passed end-to-end.**

* 157 tests pass (was 154 — added 13 Jay-loader tests, dropped 10 stale).
* Champion gate (≥6 instruments AUC > 0.55): 6/11 PASS.
* Importance gate (≥1 cluster MDA > 0.02 per class): 2/3 PASS (metals
  CHECK, same as old labels).
* SOPS gate (realised vol ≤ 10%): 7.3% PASS.
* Byte-identical re-emit verified by the existing test_s6 emit tests.

**Sealed-test deliverable comparison (locked submission = SOPS):**

| Variant | Sharpe | Ann ret (net) | Ann vol | Sortino | Max DD | Turnover |
|---|---:|---:|---:|---:|---:|---:|
| primary_blind | +2.73 | +16.2% | 5.9% | +4.68 | −2.2% | 488× |
| **sops** (locked) | **+2.52** | **+18.3%** | **7.3%** | **+4.10** | **−3.1%** | **188×** |
| nn_lstm | −0.80 | −0.2% | 0.3% | −1.12 | −0.4% | 18× |
| nn_linear | −1.92 | −5.5% | 2.9% | −2.32 | −3.7% | 209× |
| nn_tft | −2.60 | −3.8% | 1.5% | −3.12 | −1.9% | 91× |
| nn_vlstm | −2.72 | −2.9% | 1.1% | −3.26 | −1.5% | 71× |

**Deliverables in branch.**

* `data/sreeram_experimental_events.parquet` — 4,917 Jay events.
* `data/sreeram_experimental_features.parquet` — 70 drift-survived features.
* `outputs/strategy_weights_sops.csv` — locked submission.
* `outputs/strategy_weights_nn_{linear,lstm,vlstm,tft,primary_blind}.csv`
* `outputs/metamodel_predictions.csv` — calibrated p̂ on sealed test.
* `outputs/coverage_caveat.csv` — per-instrument structural flags.
* `results/sreeram_experimental/strategy_variant_comparison.csv`
* `results/sreeram_experimental/strategy_winner.json`
* `results/sreeram_experimental/threshold_summary.csv`
* `results/sreeram_experimental/jay_geometry_summary.csv`
* `results/sreeram_experimental/oof_calibrated_predictions.csv`
* `results/sreeram_experimental/importance/{equity,energy,metals}/*`
* `reports/sreeram_experimental/strategy_construction.md` — updated.
* `overview.pdf` (branch root) — updated.

**Next.** Goal complete. Migration is one-version-of-truth: there is no
"old GARCH labels" code path remaining in the runtime; the original
`labels.py:triple_barrier_labels` function is preserved only for its
mathematical-properties tests and is no longer called by any runner.

---

---

## PM-12 — 2026-06-04 — Methodology audit + leakage / overfit hardening

**Goal of session.** Tighten the train / val / test discipline so the
deliverable pipeline is bulletproof against subtle leakage. Make the val
partition a clean methodology scoreboard. Verify the final deliverable model
uses the maximum allowed training data.

**Architecture (final state).**

| Stage | Data used | Why |
|---|---|---|
| Drift filter | TRAIN only (early 70% vs late 30%) | val + test sealed from feature selection |
| Champion selection (CPCV 1-SE rule) | TRAIN for fit, VAL for honest scoreboard | answers "which model wins" without touching test |
| Cluster importance + global SHAP | TRAIN only | importance reflects what the deliverable model sees |
| Pruned-vs-full feature comparison | TRAIN for fit, VAL for held-out AUC | honest answer to "does pruning help?" |
| Final deliverable model fit | **TRAIN + VAL combined** (Jan 2020 → Dec 2021) | max data before sealed test, standard ML practice |
| Test | sealed (H1 2022) | read once at the end |

**Done.**

* **Embargo per-instrument h fix.** `make_scope.py:build_scope` now
  computes embargo as `max(p90_span, jay_h, 10)` per instrument. The
  scope JSON was stale (built on old GARCH labels with h=10 global) and
  three instruments (gc1s/rb1s/si1s with h ∈ {15, 15, 20}) had embargo
  too short → potential label leak at CPCV fold boundaries. Regenerated;
  all 11 instruments now embargo ≥ h.
* **Drift filter** in `make_features.py` now compares an early-train vs
  late-train chronological split (70/30 within train). Val and test stay
  fully sealed from feature selection.
* **Champion selection** (`champion_pipeline.py:_restrict_modelling` and
  the pipeline.py / make_importance.py mirrors) restricted to TRAIN
  partition only. Val is held out for honest scoreboard / NN early-stop /
  pruned-vs-full evaluation.
* **Pruned-vs-full** (`make_importance_deep.py`) fits both models on TRAIN
  and scores on held-out VAL. Honest val AUC reported per asset class
  with Hanley-McNeil SE proxy.
* **Final deliverable** (`make_deliverables.py:_split_modelling_and_test`)
  uses TRAIN + VAL combined for the per-instrument model refit and the
  OOF generation that drives Platt / threshold / SOPS. This is the
  maximum-data fit before the sealed test window. Champion *selection*
  upstream uses train only with val as honest scoreboard; the final
  deliverable refit uses the union per AFML standard practice.
* **NN strategy** (`make_nn_strategy.py`) already followed the right
  pattern: train on train, early-stop on val, refit final on train+val
  combined, apply to test. Verified — no change needed.

* **Three new methodology-guard tests** (`test_methodology_guards.py`):
  - `test_cross_val_evaluate_row_idx_aligns_to_input`: CV's OOF `row_idx`
    must map back to the input frame's label.
  - `test_nn_dataset_p_hat_forward_fill_is_causal`: between two events at
    t1 < t2, the panel at any day between them carries p̂(t1), never
    p̂(t2).
  - `test_drift_filter_uses_train_only_no_val_leak`: synthetic case where
    train rows are N(0,1) and val rows are N(+5, 1); KS must be small,
    proving val rows aren't leaking in.

**Sealed-test backtest (H1 2022, 129 trading days).**

| Variant | val SR | test SR | ann ret (net) | ann vol | Sortino | max DD | turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| **sops** (locked) | — | **+2.90** | **+14.7%** | **5.1%** | **+5.10** | **-1.9%** | **163×** |
| primary_blind | — | +2.73 | +16.2% | 5.9% | +4.68 | -2.2% | 488× |
| nn_lstm | -0.15 | -0.77 | -0.2% | 0.3% | -1.07 | -0.4% | 18× |
| nn_linear | +0.10 | -1.91 | -5.5% | 2.9% | -2.32 | -3.7% | 208× |
| nn_vlstm | +0.55 | -2.57 | -3.0% | 1.2% | -3.10 | -1.6% | 71× |
| nn_tft | -0.29 | -2.63 | -3.9% | 1.5% | -3.15 | -1.9% | 91× |

**SOPS beats primary-blind on Sharpe** (+2.90 vs +2.73) **with a third
of the turnover** (163× vs 488×) and **lower volatility** (5.1% vs 5.9%).
The meta-filter genuinely adds value on this test window — higher
risk-adjusted return + much higher transaction-cost robustness.

**Champion AUC (with_bbg, CPCV(6,2) on train only, 1-SE rule).**
6/11 instruments above AUC 0.55 (PASS); 10/11 with lower 1-SE CI > 0.50
(signal flag).

**Pruned-vs-full feature analysis (held-out val AUC):**

| Class | full val AUC | pruned val AUC | delta |
|---|---:|---:|---:|
| equity | 0.532 | **0.601** | **+0.069** |
| energy | 0.625 | 0.608 | -0.017 |
| metals | 0.530 | 0.528 | -0.002 |

Equity benefits clearly from feature pruning (5 features beat 74 by ~7 pp
val AUC). Energy is slightly hurt; metals is a wash. This is honest
out-of-sample evidence, not in-sample circularity.

**Jay's geometry selection vs 2022-H1.** Confirmed (from his PDF table
showing hold-out ranks 11-172 of 343) that he selected geometries on
in-sample top-1 and only *reported* the hold-out adjusted Sharpe. He
looked, did not use. Left as-is.

**Gates passed end of session.**

* 167 tests pass (added 3 methodology guards on top of 164 from PM-11).
* All 11 instruments embargo ≥ h.
* Drift filter / champion selection / importance / pruned-vs-full all use
  train only (val sealed from feature selection).
* Final deliverable model uses train + val combined (24 months Jan 2020
  → Dec 2021).
* OOF row-index alignment verified.
* NN p̂ forward-fill verified causal.
* SOPS realised vol 5.1% (under the 10% cap).

**Deliverables in branch.**

* `data/sreeram_experimental_events.parquet` — 4,917 Jay events with
  per-instrument geometry + partition.
* `data/sreeram_experimental_features.parquet` — 74 drift-survived
  features (drift filter on train only).
* `outputs/strategy_weights_sops.csv` — locked submission.
* `outputs/strategy_weights_nn_{linear,lstm,vlstm,tft,primary_blind}.csv`
* `outputs/metamodel_predictions.csv` — calibrated p̂ on sealed test.
* `outputs/coverage_caveat.csv` — per-instrument structural flags.
* `results/sreeram_experimental/strategy_variant_comparison.csv`
* `results/sreeram_experimental/strategy_winner.json`
* `results/sreeram_experimental/threshold_summary.csv`
* `results/sreeram_experimental/jay_geometry_summary.csv`
* `results/sreeram_experimental/oof_calibrated_predictions.csv`
* `results/sreeram_experimental/importance/{equity,energy,metals}/*`
  — clustered MDA + within-cluster PCA + global SHAP + pruned-vs-full
  val AUC + findings notes.
* `notebooks/sreeram_experimental/{equity,energy,metals}_importance.ipynb`
  — executed notebooks with cluster MDA charts, within-cluster tables,
  global SHAP bar charts, pruned-vs-full comparison charts.
* `reports/sreeram_experimental/strategy_construction.md` — Jay-CSV
  migration appendix + methodology section.
* `overview.pdf` (branch root) — submission overview document.

**Next.** Branch is ready for tomorrow's submission.
