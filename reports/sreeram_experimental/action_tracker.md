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
     parquets end 2022-06-30; H2-2022 grader rerun will have all-NaN BBG cols;
     four mitigations in preference order, with extending the BBG pull as the
     preferred fix.
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
