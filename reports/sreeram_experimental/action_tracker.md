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
