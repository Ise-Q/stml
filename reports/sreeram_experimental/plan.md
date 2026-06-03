# `Sreeram_experimental` — Build Plan (Golden Record)

> **Status:** Golden source of truth for the new pipeline. Every architectural, methodological, and engineering decision lives in this file. When my context compacts and a fresh agent picks up, this is the document that re‑grounds them.
>
> **Branch:** `Sreeram_experimental` (cut from `Sreeram` 2026‑06‑02). All work goes here. No other branches will be created.
>
> **One‑sentence mission:** Build a single, methodology‑first meta‑labelling pipeline that targets per‑instrument AUC 0.65–0.75 by combining (a) per‑instrument modelling under a shared backbone, (b) tighter labels via GARCH‑forecast σ̂ + CPCV‑optimised barriers, (c) Bloomberg‑augmented features the OHLCV+signal stack cannot derive (futures term structure, options‑implied vol, CFTC positioning), all under the same leakage and significance discipline that `model/alken-metamodel` established.
>
> **Expected outcome:** 0.65–0.75 per‑instrument AUC (vs alken's 0.50–0.61), pooled net Sharpe 1.5–2.5 at 7.5–10 % ann vol (within the 10 % constraint), statistical significance distinguishable from zero, no selection‑on‑test.

---

## Table of Contents

0. [Cardinal rules (read every session)](#0-cardinal-rules-read-every-session)
1. [Assignment recap and the constraints we operate under](#1-assignment-recap-and-the-constraints-we-operate-under)
2. [Diagnostic findings — what the data actually says](#2-diagnostic-findings--what-the-data-actually-says)
    * §2.8 added 2026‑06‑03: **OHLCV is an adjusted continuous‑futures series**, not raw front‑month — asset‑class verdict
    * §2.9 summary of what §2.8 changes
3. [Architectural decisions — what we build and why](#3-architectural-decisions--what-we-build-and-why)
    * §3.11 added 2026‑06‑03: **What we deliberately do NOT *claim*** — the §2.8 framing discipline
4. [What we take from each existing branch (and what we do NOT)](#4-what-we-take-from-each-existing-branch-and-what-we-do-not)
5. [The Bloomberg data ingestion plan](#5-the-bloomberg-data-ingestion-plan)
6. [Repository structure (the canonical layout)](#6-repository-structure-the-canonical-layout)
7. [Naming and "one version only" conventions](#7-naming-and-one-version-only-conventions)
8. [Stage‑by‑stage build plan with acceptance gates](#8-stage-by-stage-build-plan-with-acceptance-gates)
9. [Testing strategy (RED‑first TDD)](#9-testing-strategy-red-first-tdd)
10. [Determinism and reproducibility contract](#10-determinism-and-reproducibility-contract)
11. [Leakage and cross‑validation discipline](#11-leakage-and-cross-validation-discipline)
    * §11.5 added 2026‑06‑03: **Methodology framing — the report language §2.8 mandates**
12. [Submission readiness checklist](#12-submission-readiness-checklist)
13. [Risk register — what could go wrong](#13-risk-register--what-could-go-wrong)
    * R‑10 added 2026‑06‑03: continuous‑contract adjustment artefact on target (esp. `ng1s`)
    * R‑11 added 2026‑06‑03: BBG feature missingness in H2‑2022 hidden test
14. [Glossary and pointers](#14-glossary-and-pointers)

---

## 0. Cardinal rules (read every session)

These are non‑negotiable. If any rule conflicts with the rest of this document, the rule wins.

### R1 — One version of everything

Every code file is the single canonical version of itself. There is **no** `pipeline_v2.py`, no `labels_with_garch.py`, no `features_ext.py`, no `predictions_v0.csv` / `_v1.csv` / `_v2.csv`. If we change methodology, we **edit the file in place** and use git to track the change. Renames happen with `git mv`.

If a teammate or a future agent needs to A/B test two approaches, they do so in a feature branch off `Sreeram_experimental`, evaluate, then either merge in or discard. The main branch holds the single working artefact, always.

**Forbidden file name patterns:**

```
*_v2.py / *_v3.py / *_old.py / *_new.py / *_final.py / *_FINAL.py
*_backup.py / *_copy.py / *_test.py (note: tests live in /tests/, not next to the module)
*_attempt2.py / *_with_*.py (describing a feature in the filename)
predictions_v*.csv / predictions_old.csv
```

**Forbidden code patterns:**

```python
# DO_NOT_USE: legacy_*
# OLD_VERSION: ...
if USE_V2:
    ...
else:
    ...  # the old way kept "for comparison"
```

If you need a feature flag, add it to `config.py` with a default and a docstring. If the flag goes unused for one full stage, **delete** the dead branch.

### R2 — Edit in place, never duplicate

When a fix or improvement lands, it replaces the old behaviour in the same file. The old code is removed in the same commit, not commented out, not moved to a `legacy/` folder. Git history keeps the receipt.

**Exception:** when an experiment is genuinely exploratory and we are not sure which way to go, do it in a branch (`sreeram_experimental/<short‑desc>`), run it, then merge the winner back and delete the branch. Do not carry exploratory branches in the long term.

### R3 — Outputs are deterministic and byte‑identical on re‑emit

Every artefact (predictions CSV, weights CSV, importance CSV, log CSV) is sorted, has pinned column order, ISO date format, fixed float format. A re‑emit on the same data must match byte‑for‑byte. Tested in `tests/experimental/test_byte_identical_emit.py`.

### R4 — Leakage is a hard‑fail bug, not a methodology choice

A feature whose value at time `t` depends on data at time `> t` is broken and must be removed before the rest of the build proceeds. Every feature function ships with a truncation‑invariance property test. Every CV split is purged on `t1` (first‑touch). No model trains on data that overlaps its test window's label span.

The leakage gates we use:

* **Truncation‑invariance** for every E‑class feature: `feature(panel.iloc[:t+1]).iloc[t] == feature(panel).iloc[t]` for several `t`.
* **Frozen‑artefact guard** for fitted features: TF blocks fit on the modelling sample only, applied causally with frozen parameters.
* **Purged CV** with per‑instrument embargo from `instrument_scope.json`.
* **No consumption of pre‑built feature matrices that freeze a global FE‑train cutoff** during the model's CV folds. Features are recomputed per fold from raw OHLCV / signals / Bloomberg.

### R5 — Methodology, not performance, is the grade

The course brief says it on page 9. Every architectural choice in this plan is justified against either (a) a specific finding in the data or (b) a cited reference. We do not optimise hyperparameters by looking at the test slice. We do not pick the model that "happens to win on H1‑2022" unless our validation methodology said it should win.

If we discover we have been selecting on test, we revert to the principled choice and document the revert. This is the lesson of alken's pass 4/5.

### R6 — The prediction window is config‑driven

The grader will rerun on H2‑2022. The prediction window is a `Config` field, not a hardcoded literal. CLI flags `--predict-start` and `--predict-end` exist. Hidden in no `if 2022` checks anywhere.

### R7 — The 10 % volatility cap is the actual binding strategy constraint

User clarification (2026‑06‑02): the only strategy constraint released is **max 10 % annualised volatility**. Our sizing module enforces this. Sreeram's existing `strategy_weights.csv` breaches it (realised 18.1 %); alken passes (7.5 %). We target 7–10 % realised ann vol with `target_vol = 0.08` for headroom.

### R8 — Branch isolation is absolute

We write to:

* `src/stml/experimental/`
* `tests/experimental/`
* `notebooks/sreeram_experimental/`
* `reports/sreeram_experimental/`
* `results/sreeram_experimental/`
* `data/bloomberg/` (raw + cleaned Bloomberg artefacts; treated as data, not features)
* `pyproject.toml` / `uv.lock` (for the deps we need)

We **do not** touch:

* `src/stml/io.py` or `src/stml/na_checks.py` (shared spine, read‑only)
* Any files belonging to the Harry, alken, or main feature branches when we look at them via worktrees

The Harry branch lives at `/tmp/stml-harry/` and the alken branch at `/tmp/stml-alken/`. Both are reference material. We import patterns and code by COPY (with attribution in the docstring) into `src/stml/experimental/`, never by `from harry.* import *` or `from alken_metamodel.* import *`.

### R9 — Every numerical claim must reproduce from an artefact

If the report says "pooled AUC 0.68", there must be a CSV under `results/sreeram_experimental/` that contains that number, produced by code under `src/stml/experimental/`. No screenshots, no hand‑typed tables in markdown without backing files. The `experiment_log.csv` records every emit run.

### R10 — Update this document when methodology changes

If we deviate from the plan, the plan gets updated in the same commit. This document is the source of truth; if it lies, future agents make wrong decisions. The pattern:

```
git add src/stml/experimental/<changed file>
git add reports/sreeram_experimental/plan.md   # always
git commit -m "feat(<stage>): <change>  (plan updated)"
```

### R11 — When in doubt, do the cheap diagnostic before the expensive build

Before adding a Bloomberg feature family, compute its single‑feature AUC and train→test KS on a sample. Before changing the CV splitter, run an A/B on a small dataset. Before stacking N models, plot the OOF predictions and check they have non‑trivial correlation differences. We always do the cheap check first.

---

## 1. Assignment recap and the constraints we operate under

(Cross‑reference: full recap in `/branch_descriptions.md` §1 — kept short here.)

* **Module:** BUSI70575 — Imperial coursework on Systematic Trading Strategies with ML
* **Weight:** 50 % of final grade
* **Marking (100 marks + 10 bonus):**
  * 20 — Feature engineering (creative; technical indicators, latent vars, unsupervised, anything justified)
  * 20 — Labelling (triple‑barrier with justified barrier widths + horizon)
  * 30 — Model development & comparison (≥3 models from linear / tree / NN families with hyperparameter tuning)
  * 10 — Cluster‑level feature importance
  * 20 — OOS evaluation (clean carve‑out, classification metrics, per‑instrument breakdown, baseline comparison)
  * +10 bonus — Strategy construction (position sizing on top of meta probabilities)
* **Data:** `data/ohlcv_data.csv` (daily 1990‑2022) + `data/primary_signals.csv` (Jan 2020 → Jun 2022, signal in {-1, 0, +1}). Hidden test = H2‑2022 (grader reruns code).
* **Universe:** 11 instruments — Equity (es1s, nq1s, fesx1s), Energy (cl1s, ho1s, rb1s, ng1s), Metals (gc1s, si1s, hg1s, pl1s). Must cover ≥1 full asset class; covering more is optional.
* **Strategy constraint (only one):** **max 10 % annualised volatility** (clarified by user 2026‑06‑02).
* **Bloomberg use:** user confirmed (2026‑06‑02) that additional data from the Bloomberg terminal is allowed provided it is point‑in‑time correct and inside the released training period.
* **Deliverables:**
  * Code that reruns end‑to‑end
  * `predictions.csv` for H1‑2022 with `(date, instrument, prediction ∈ [0, 1])`
  * Optional `strategy_weights.csv` with `(date, instrument, weight)`
* **Deadline:** 2026‑06‑04

The grader is a quant researcher at Alken Asset Management. They will:

* throw out anything that overclaims significance
* reward catching your own potential false positives
* reward per‑instrument honesty (no hiding a weak member in a pooled aggregate)
* reward the literature‑grounded methodology choices
* not reward a high Sharpe with no significance test
* not reward a strategy that breaches the 10 % vol cap

---

## 2. Diagnostic findings — what the data actually says

These are empirical findings from the analysis we ran on 2026‑06‑02 (`branch_descriptions.md` and the analysis turn that followed). They are the basis for every architectural choice in §3.

### 2.1 Primary signal characterisation (645 dates, 2020‑01‑03 → 2022‑06‑30)

| inst | n_active | pct_long | pct_short | pct_zero | avg_run | p90_run | notes |
|---|--:|---:|---:|---:|---:|---:|---|
| es1s | 575 | 70.7 % | 18.4 % | 10.9 % | 3.8 | 10 | long‑biased |
| nq1s | 604 | 62.3 % | 31.3 % | 6.4 % | 3.8 | 8 | balanced |
| fesx1s | 637 | 44.3 % | 54.4 % | 1.2 % | 3.9 | 10 | balanced |
| cl1s | 422 | 59.8 % | 5.6 % | 34.6 % | 6.6 | 14 | nearly long‑only |
| ho1s | 63 | 8.2 % | 1.6 % | 90.2 % | 11.3 | 27 | **very thin** |
| rb1s | 628 | 56.9 % | 40.5 % | 2.6 % | 6.6 | 19 | balanced |
| ng1s | 124 | 0.0 % | 19.2 % | 80.8 % | 13.2 | 33 | **short‑only** |
| gc1s | 168 | 21.1 % | 5.0 % | 74.0 % | 5.6 | 12 | thin, long‑biased |
| si1s | 578 | 47.4 % | 42.2 % | 10.4 % | 3.2 | 7 | balanced |
| hg1s | 628 | 47.9 % | 49.5 % | 2.6 % | 4.7 | 11 | balanced |
| pl1s | 557 | 65.3 % | 21.1 % | 13.6 % | 3.9 | 8 | long‑biased |

**Implications:**

* ho1s (63 active days), gc1s (168), ng1s (124) — thin participation. Strict 60‑row coverage flag.
* `p90_run` of 33 for ng1s drives the per‑instrument embargo: in pooled CV, embargo must be ≥33 trading days on ng1s rows.
* ng1s is short‑only — calibration cannot extrapolate to "long ng1s" cases.

### 2.2 Label outcome composition (h=10, symmetric `pt=sl=1.0`, EWMA σ̂)

The triple‑barrier labels resolve at vertical timeout **50–65 % of the time**, with non‑trivial mean returns at vertical (+310 bp cl1s, −214 bp ng1s, +52 bp es1s …). This means the labels are essentially predicting "10‑day drift sign" not "did the bet hit a clean barrier". The σ̂·√10 barriers are too wide for the actual 10‑day move distribution.

| inst | n | PT hit | SL hit | VERT hit | mean ret AT vertical (bp) |
|---|--:|---:|---:|---:|---:|
| cl1s | 411 | 20 % | 14 % | **65 %** | +310 |
| es1s | 564 | 25 % | 21 % | 55 % | +52 |
| fesx1s | 626 | 21 % | 15 % | **64 %** | +13 |
| ng1s | 120 | 33 % | 17 % | 50 % | **−214** |
| hg1s | 617 | 22 % | 21 % | 57 % | −6 |
| si1s | 567 | 19 % | 21 % | 60 % | +49 |

**Implications:**

* **Tighten barriers** to push PT+SL fraction up (target: 60–70 % combined). Either smaller multipliers (`pt = sl = 0.5`) or shorter horizon (`h = 5`).
* **GARCH(1,1) σ̂** instead of EWMA: forward‑aware barrier scale, sharper.
* **CPCV barrier search per asset class:** equity, energy, metals likely want different barriers.
* **Asymmetric barriers** for systematically drifting instruments: cl1s has +310 bp mean vertical → `pt > sl` favours the drift; ng1s has −214 bp → `pt < sl`.

### 2.3 Single‑feature held‑out AUC ceiling per instrument

Direction picked on train+val, AUC measured on test (the proper held‑out single‑feature ceiling):

| inst | top single feature | sf ceiling | alken got | **headroom** |
|---|---|---:|---:|---:|
| cl1s | `f6_macd_12_26` | 0.959 | 0.55 | **+0.41** |
| ng1s | `f7_oi_level` | 0.846 | 0.35 | **+0.50** |
| rb1s | `f5_signal` | 0.801 | 0.50 | **+0.30** |
| es1s | `z_f6_macd_12_26` | 0.756 | 0.60 | +0.16 |
| nq1s | `f12_trend_tval_21` | 0.735 | 0.61 | +0.13 |
| si1s | `z_f7_kyles_lambda_20` | 0.702 | 0.50 | +0.20 |
| hg1s | `f13_mra_energy_d3` | 0.678 | 0.58 | +0.10 |
| pl1s | `f11_be10y_level` | 0.675 | 0.50 | +0.18 |
| fesx1s | `f13_mra_energy_d4` | 0.635 | 0.59 | +0.05 |

Mean headroom vs alken: **+0.22 AUC**.

**Implications:**

* The pooled model leaves substantial AUC on the table per instrument.
* Every instrument has a **different** best feature — pooling forces compromise.
* The single biggest engineering change available is **per‑instrument modelling** (multi‑task NN or per‑class XGBoost with strong instrument conditioning).

### 2.4 Train→test feature drift (KS test by family)

| family | n features | median KS | max KS | translation |
|---|--:|---:|---:|---|
| f7 microstructure | 10 | **0.075** | 0.18 | stable |
| f9 cross‑section | 5 | 0.093 | 0.18 | stable |
| f5 signal | 9 | 0.095 | 0.16 | stable |
| f8 calendar | 4 | 0.106 | 0.28 | stable |
| f1 counter‑trend | 15 | 0.111 | 0.23 | stable |
| f12 path structure | 8 | **0.122** | 0.22 | stable |
| f6 momentum | 7 | 0.159 | 0.26 | mildly drifting |
| f3 GMM regimes | 4 | 0.160 | 0.19 | mildly drifting |
| f17 HMM regimes | 4 | 0.163 | 0.18 | mildly drifting |
| f4 latent | 11 | 0.177 | 0.25 | drifting |
| f2 vol | 12 | 0.179 | 0.34 | drifting |
| f13 wavelet | 5 | 0.223 | 0.50 | drifting |
| f15 conditional risk | 4 | 0.247 | 0.39 | drifting |
| **f11 macro** | **45** | **0.358** | **1.000** | **catastrophic** |

**Implications:**

* Macro LEVELS are catastrophically drifted (rates regime change 2021→2022). Use macro CHANGES or ranks, not levels.
* F7/F9/F5/F8/F1/F12 are stable — overweight these in the feature stack.
* Macro family is too large (45 cols) given its drift profile. Prune aggressively.

### 2.5 Stable AND informative features (the gold)

| feature | KS | test AUC | source |
|---|---:|---:|---|
| `f15_path_tortuosity_20` | 0.082 | 0.598 | Harry |
| `f12_variance_ratio_5_21` | 0.043 | 0.593 | Sreeram |
| `f12_efficiency_ratio_21` | 0.076 | 0.589 | Sreeram |
| `f5_long_bias_20` | 0.102 | 0.578 | shared |
| `f7_oi_level` | 0.080 | 0.553 | shared |
| `f7_kyles_lambda_20` | 0.101 | 0.552 | Harry |
| `f9_ewma_implied_corr_z` | 0.102 | 0.551 | Harry |

Only 9 features are BOTH stable AND have non‑trivial signal. Force the model to lean on these via the drift penalty (see §3.5).

### 2.6 Grinold's Fundamental Law sets the hard ceiling

`IR = IC · √BR`. Primary signal IC ≈ 0.07 (Harry's audit), breadth = 11 → IR ≈ 0.23. Implies AUC ceiling ≈ 0.5 + IC/π ≈ 0.52 for the **pooled** problem.

**Per‑instrument**, the local IC is higher (cl1s ~0.12, es1s ~0.10, etc.), and the regime conditioning can push effective IC higher in regime‑stable periods. So per‑instrument AUC of 0.65–0.75 is achievable; per‑instrument AUC > 0.85 is not.

### 2.7 Sign‑flip features

61 % of features have a train→test sign flip (the direction that worked in train is opposite of what worked in test). The single biggest example is `f5_signal` itself: TR AUC 0.636 (direction +), TEST AUC 0.518 (direction −).

**Implication:** drop `f5_signal` as a direct feature. Keep behavioural signal‑derived features (`f5_long_bias_20`, `f5_signal_entropy_20`, `f5_trailing_run_length`). These describe the signal's *structure*, not its instantaneous direction, and survive the regime break.

### 2.8 `ohlcv_data.csv` is an **adjusted continuous‑futures series**, not raw front‑month prices — and the adjustment artefact is asset‑specific

This finding was diagnosed on 2026‑06‑02 night after the Bloomberg pull arrived and the BBG‑raw front‑month prices could be cross‑checked against the project's OHLCV closes. **The substantive conclusion: `ohlcv_data.csv` is real‑market‑derived data transformed into a continuous contract via ratio / proportional back‑adjustment, not pure additive Panama back‑adjustment, and not synthetic data.** It is internally consistent for the supervised problem (the hidden test uses the same convention), but the adjustment artefact is materially different across asset classes and that drives several modelling decisions.

#### 2.8.1 What `ohlcv_data.csv` actually is

* At the start of each instrument's history, the OHLCV close matches the raw Bloomberg front‑month close exactly (cl1s 22.89, gc1s 402.10, si1s 5.253, ng1s 1.62 …). So the data is **market‑derived**, not synthetic.
* Through the history, the OHLCV closes diverge from raw front‑month closes by an asset‑specific factor — by 2020‑01‑02 the multipliers are roughly cl1s 0.41, gc1s 0.40, ng1s 0.0013, etc. This pattern is consistent with **ratio / proportional continuous‑contract back‑adjustment** (Panama back‑adjustment would preserve dollar differences with a slope close to 1.0; the observed slopes of `diff(OHLCV) ~ diff(raw)` match the level ratios, not 1.0):

| Instrument | Median OHLCV / raw level ratio | Slope of OHLCV‑diff vs raw‑diff |
|---|---:|---:|
| gc1s gold | 0.4040 | 0.4076 |
| si1s silver | 0.3287 | 0.3306 |
| hg1s copper | 0.0118 | 0.0118 |
| rb1s gasoline | 0.0260 | 0.0259 |
| ng1s natural gas | 0.0006 | 0.00045 |

* The OHLCV thus represents an **adjusted continuous trading series**. The implicit return process at time `t` is `course_return = raw_market_return + adjustment_artefact`. For most assets `adjustment_artefact` is tiny on a daily horizon; for natural gas it is large.

#### 2.8.2 Return agreement OHLCV‑vs‑raw, modelling window 2020‑01‑02 → 2022‑06‑30

The decisive empirical test: do the **daily log returns** in `ohlcv_data.csv` track the daily log returns of the raw BBG front contract over the modelling window?

| Asset | Log‑return R² vs raw | Mean abs return diff (bps) | Days > 100 bps diff | Verdict |
|---|---:|---:|---:|---|
| si1s silver | 0.9996 | 1.7 | 0 | **strongly coherent** |
| pl1s platinum | 0.9980 | 3.6 | 0 | **strongly coherent** |
| hg1s copper | 0.9961 | 3.7 | 1 | **strongly coherent** |
| gc1s gold | 0.9952 | 2.1 | 1 | **strongly coherent** |
| cl1s WTI | 0.9748 | 10.2 | 10 | mostly coherent (Apr‑2020 caveat) |
| fesx1s Euro Stoxx | 0.9658 | 4.7 | 14 | mostly coherent |
| nq1s Nasdaq | 0.9518 | 5.3 | 11 | mostly coherent |
| rb1s RBOB | 0.9471 | 23.7 | 36 | **use caution** |
| es1s S&P 500 | 0.9394 | 4.6 | 6 | mostly coherent |
| ho1s heating oil | 0.8722 | 21.5 | 25 | **use caution** |
| **ng1s natural gas** | **0.7212** | **39.3** | **48** | **problematic** |

#### 2.8.3 Label agreement OHLCV‑vs‑raw, same window

Re‑labelling on raw BBG front‑month closes (same labeller, same pt=sl=0.5, h=10) and comparing per‑event labels:

| Instrument | Label agreement | Notes |
|---|---:|---|
| si1s | 99.6 % | clean |
| ho1s | 98.4 % | clean despite noisy returns |
| cl1s | 98.0 % | mostly clean |
| hg1s | 97.8 % | clean |
| pl1s | 97.6 % | clean |
| gc1s | 95.7 % | clean |
| rb1s | 94.3 % | usable |
| **ng1s** | **89.2 %** | **~11 % of labels flip vs raw‑market labels** |

The natural gas label flip rate is the load‑bearing number. It quantifies how much of `ng1s`'s barrier outcomes are driven by adjustment artefacts rather than raw‑market price action — and therefore how much of any "feature → label" signal is genuinely economic vs construction‑specific.

#### 2.8.4 Asset‑class verdict — does external macro / BBG meaning carry across the transformation?

The question the user asked: *"If external features are real but the target is transformed, are the learned relationships meaningful?"* The empirical answer is **asset‑specific**:

| Asset group | Interpretation | What to claim |
|---|---|---|
| **Metals** (gc1s, si1s, hg1s, pl1s) | **Safe** — course returns ≈ raw returns; real macro/BBG features remain economically meaningful | "macro / rates / IV features predict the course `gc1s` move because the course series tracks raw gold" |
| **Equity** (es1s, nq1s, fesx1s) | **Mostly safe** — some Mar‑2020 vol‑event artefacts but R² ≈ 0.94–0.97 | "macro / IV features remain interpretable; flag Mar‑2020 worst days in the limitations section" |
| **Crude** (cl1s) | **Mostly safe with one caveat** — direction match 98 %; the Apr‑2020 negative‑oil day (raw −$37.63 vs course stayed positive) is transformed away and breaks raw‑equivalence | "we predict barrier outcomes on the provided continuous series; do not claim raw front‑month profitability" |
| **HO / RB** (ho1s, rb1s) | **Use caution** — R² 0.87–0.95 but 25–36 days with > 100 bps adjustment noise | "external features may help but the target carries material adjustment noise; report per‑instrument carefully" |
| **Natural gas** (ng1s) | **Problematic** — R² 0.72, 48 days with > 100 bps difference, 11 % label flips | "for `ng1s` external features predict the raw‑market component but not the adjustment component; results interpreted as construction‑specific, not deployable" |

Concrete `ng1s` artefact examples: 2022‑01‑28 raw NG log return −30.05 % vs course +7.98 %; 2022‑01‑27 raw +38.17 % vs course +5.94 %; 2020‑09‑29 raw +19.80 % vs course −8.74 %. The same external feature cannot simultaneously predict both directions on the same day.

#### 2.8.5 Implications for the build

These flow directly into §3 and into the methodology document we ship:

* **Per‑asset‑class modelling becomes more strongly justified.** Already in §3.1 for AUC‑headroom reasons (per‑instrument single‑feature ceilings); §2.8 adds **economic‑interpretability heterogeneity** as a second axis — the price you pay for pooling across asset classes is not only loss of single‑feature signal but also loss of feature‑target‑coherence (because the energy classes have more adjustment artefact than metals). Per‑class models avoid forcing a single decision function to absorb both clean and noisy targets.
* **`ng1s` gets flagged as high‑artefact in the deliverable.** We continue to predict on all 11 instruments (the brief asks for ≥ 1 full class; covering all is rewarded), but the methodology document includes an explicit limitations bullet about `ng1s`: low feature‑target coherence, 11 % label‑flip‑vs‑raw, results interpreted as construction‑specific rather than deployable. We also document the option of running a per‑instrument ablation that drops external BBG / macro features for `ng1s`; this is investigated in S3 if the with‑BBG model under‑performs the without‑BBG model on `ng1s`'s validation slice.
* **HO and RB get a parallel caution.** Lighter than `ng1s` (no explicit limitations bullet) but the methodology mentions them as "noisier‑but‑usable" with R² 0.87–0.95.
* **The headline narrative shifts subtly.** The deliverable predicts *barrier outcomes on the course continuous series* — that is internally consistent with the grader's hidden test (which uses the same convention). It does **not** claim raw front‑month deployable trading profitability. This shapes the report language (see §11 below).
* **No change to the labels themselves.** S1 events.parquet stays as‑is. The course OHLCV is what the grader rates on, so the labels we produce match the assessment target.

#### 2.8.6 Hidden‑test BBG feature missingness — a separate but linked concern (R‑11)

A parallel concern surfaced in the same analysis: our cleaned Bloomberg parquets only cover up to **2022‑06‑30** because that is when we pulled. The grader's hidden test runs on **H2‑2022 (Jul → Dec 2022)**. At rerun time, F18 / F19 / F22 columns will be **all‑NaN** for the prediction window unless we extend the pulls.

This is a deployment mismatch that is independent of the continuous‑contract issue but interacts with it. Mitigations (in decreasing preference order):

1. **Pull BBG data for H2‑2022 too.** The brief allows additional data "inside the released training period (1990 → 2022‑06‑30)" — but the *grader's rerun* will need the BBG series to extend through H2‑2022 for the model to score the hidden window. The user has BBG access now and can pull the H2‑2022 BBG window before submission. This is the cleanest fix.
2. **Validate the model under simulated BBG missingness.** Hold out a fraction of the H1‑2022 validation slice with F18 / F19 / F22 forced to NaN, retrain, and compare AUC. If the model degrades materially, the architecture is BBG‑fragile.
3. **Ship the model in a NaN‑tolerant configuration** (XGBoost native NaN handling, multi‑task NN gets BBG indicators dropped at inference). This works mechanically but feature‑family disappearance is a real distribution shift — Option 1 is preferred.
4. **Train the final model on a "no‑BBG" feature set as a robustness baseline** (the F1–F17 + F21 stack alone). Plan §5.5 already documented this fallback; what we add here is an explicit *with‑vs‑without* ablation in S3 acceptance so the decision is made on numerical evidence, not on belief.

The S3 acceptance gate now requires this ablation. See §3.10 + §13 R‑11 below.

### 2.9 Summary of what §2.8 changes vs the original plan

Three pieces of the plan get updated as a consequence of §2.8 — §3.1 (model architecture justification), §3.10 (what we deliberately do NOT claim), and §13 risk register (R‑10, R‑11). The cardinal rules (§0) do NOT change — the existing R5 ("methodology, not performance, is the grade") already covers the claim discipline that §2.8 sharpens. The labels, the GARCH σ̂, the CPCV machinery, the feature stack, and the deliverable format all stay byte‑identical to what S1 and S2 produced.

---

## 3. Architectural decisions — what we build and why

Each decision is tagged with its origin: **[finding]** (justified by §2), **[best‑of‑breed]** (taking from an existing branch), or **[new]** (novel to this branch).

### 3.1 Two model families, one architecture: per‑class XGBoost + multi‑task neural net

**[finding §2.3 + §2.8]** Two findings now justify per‑asset‑class modelling. **(a)** Pooled modelling leaves +0.22 AUC per instrument on the table (§2.3). **(b)** The economic‑interpretability of features against the course continuous‑contract target is **heterogeneous across asset classes** (§2.8.4 verdict table) — metals are strongly coherent (R² > 0.995), equity mostly coherent, energy noisier with `ng1s` problematic (R² 0.72, 11 % label flip vs raw). A single pooled model would absorb both clean and noisy targets and the noisy ones (`ng1s` especially) would pull the decision boundary against the cleaner ones. Per‑class models avoid this contamination. We address both findings two ways:

**Family A — Per‑asset‑class XGBoost with instrument‑conditioned interactions**

* Three models: one for Equity (es1s/nq1s/fesx1s), one for Energy (cl1s/ho1s/rb1s/ng1s), one for Metals (gc1s/si1s/hg1s/pl1s).
* Instrument one‑hot dummies AND **explicit interaction terms** between every top‑20 feature and the instrument one‑hot. This lets the boosted forest learn `feature × instrument` splits without per‑instrument training.
* Tuned hyperparameters by inner CPCV.
* Acts as the methodological baseline — interpretable, reproducible, deterministic.

**Family B — Multi‑task neural net with instrument heads**

* One shared encoder: 2 hidden layers (64, 32) feeding into a shared representation `h` of dim 16.
* 11 output heads (one per instrument), each a `Linear(16, 1) + sigmoid`.
* Joint training: loss = sum over instruments of `uniqueness_weighted_BCE(y_i, ŷ_i)`. Only the head matching the row's instrument contributes.
* Instrument embedding `e_inst ∈ R^8` learned jointly; concatenated to the row before the encoder.
* Full‑batch Adam, byte‑deterministic.

**Family C — Elastic‑net Logistic baseline**

* Per asset class.
* Sample‑weighted, standardised.
* The linear baseline required by the rubric.

Three families covers the rubric requirement of linear + tree + NN, and the multi‑task NN is the headline architectural novelty.

### 3.2 Triple‑barrier labels with t+1 entry, GARCH σ̂, and CPCV‑optimised barriers per class

**[best‑of‑breed: Harry's t+1 entry + Harry's GARCH σ̂ + Harry's CPCV barrier search]**

* **Entry at `t + 1`** (Harry's fix). Signal observed at close of `t` is acted on at close of `t + 1`. Empirically supported: `corr(s_t, r_{t+1}) > 0` for all 11 instruments. The same‑bar return between `t` and `t + 1` is no longer inside the held window.
* **GARCH(1,1) σ̂** from Harry's `new_work/triple_barrier.py` instead of EWMA. Sharper, forward‑aware. Refit every 21 days; expanding window capped at 2000 bars; `min_obs = 500`.
* **CPCV barrier search per asset class** (`pt_mult`, `sl_mult`, `h`) on IN‑SAMPLE only (modelling period ≤ 2021‑12‑31). Grid:
  * `h ∈ {3, 5, 7, 10, 15}`
  * `pt_mult ∈ {0.5, 0.75, 1.0, 1.25, 1.5}`
  * `sl_mult ∈ {0.5, 0.75, 1.0, 1.25, 1.5}`
  * Objective: maximise `mean(AUC across 15 CPCV paths) − 0.5 × |class_balance − 0.5|`.
  * Per asset class. Three winning configs.
  * **Decision rule:** the winning config must clear two gates — (a) PT+SL fraction > 50 % and (b) class balance ∈ [0.45, 0.65]. If neither config in the grid clears both gates, fall back to a hand‑set `pt = sl = 0.5, h = 5`.
* Sample uniqueness weights from AFML Ch.4, computed on each instrument's native trading‑day index.

### 3.3 Drift‑aware feature engineering

**[finding §2.4 + §2.5]** The 175‑feature alken matrix has too many drift‑prone features. We rebuild the feature stack with drift control baked in.

**Feature stack (target: ~80 cleaned features, vs 175 alken):**

| Block | Cols (est.) | Source | Drift handling |
|---|---:|---|---|
| F1 counter‑trend | 12 | reuse Sreeram/Harry | already stable |
| F2 vol | 6 | reuse | drop levels, keep ratios |
| F5 signal trajectory | 6 | reuse Harry | drop `f5_signal`; keep behavioural |
| F6 momentum | 6 | reuse | already stable |
| F7 microstructure | 8 | reuse Harry's `microstructure_fixed` | zero‑volume mask applied |
| F8 calendar | 4 | reuse | stable by construction |
| F9 cross‑section | 6 | reuse Harry | already stable |
| F10 price action | 4 | reuse | stable |
| F11 macro CHANGES + RANKS | 12 | rebuild | LEVELS → 63‑day rolling ranks; CHANGES kept; CESIUSD added |
| F12 path structure | 8 | reuse Sreeram | the gold (highest stable AUC) |
| F15 conditional risk | 4 | reuse Harry | bootstrap MC |
| F16 concept drift | 1 | reuse Harry | used as a feature AND as a training weight |
| F17 HMM regimes (causal forward filter) | 4 | reuse Sreeram | not the smoothed predict_proba |
| **F18 futures term structure** | **8** | **new (Bloomberg)** | front − second, roll yield, contango/back indicator |
| **F19 options‑implied vol** | **9** | **new (Bloomberg)** | ATM IV, 25Δ skew, risk reversal at 1m and 3m |
| **F20 CFTC COT positioning** | **6** | **new (Bloomberg)** | net spec position z‑score; commercials net position; managed money breakdown |
| **F21 cross‑asset relative value** | **4** | **new (Bloomberg + OHLCV)** | gold/silver, copper/gold, crack spreads, gold‑real‑rate spread |
| **F22 event flags** | **3** | **new (Bloomberg econ calendar)** | "FOMC in window", "EIA release in window", "OPEC in window" |
| EWMA HMM regime | 5 | reuse alken's `regime.py` | causal/fit‑free, no CV seam |

Total: ~120 columns (target; final count after dedup will be lower).

**Drift filter:**

* Compute KS(train, val) per feature before final training.
* Drop features with `KS > 0.25` AND `val_AUC < 0.54`.
* Keep `KS > 0.25` features only if `val_AUC > 0.58` (they're informative enough to justify the drift risk).
* Logged in `results/sreeram_experimental/feature_drift_audit.csv`.

**Macro reformulation:**

* For every level series in F11 (VIX, MOVE, DXY, 10Y_UST, BE10Y, …), the **canonical feature is the rolling 63‑day rank** in `[0, 1]`, not the raw level.
* CHANGES (`chg5`, `chg20`) are kept as‑is.
* The level itself is NOT added as a feature.

**F16 as both feature and training weight:**

* As a feature: `f16_regime_alignment` gives the model an explicit "today looks like / unlike train era" signal.
* As a training weight (inverse): `sample_weight_drift = 1 / (1 + α · f16_score)` with `α = 0.5`. This **down‑weights training rows that look very different from recent data**, focusing the model on the distribution that matters for OOS.

### 3.4 CPCV‑as‑selection + per‑instrument embargo + nested CPCV (actually run)

**[best‑of‑breed: alken's CV machinery]**

* **`CombinatorialPurgedCV(n_groups=6, n_test_groups=2)` → 15 paths.** Used for model selection by 15‑path mean OOS AUC.
* **Per‑instrument embargo** from `results/instrument_scope.json` (already present in alken's results, but we regenerate it). Embargo is the instrument's `embargo_p90` (trading days), advanced on each instrument's own date axis.
* **Nested CPCV (actually run on real data).** Outer CPCV (6, 2) splits the modelling sample; inner CPCV (5, 1) picks the model per outer fold; outer fold scores the winner. Cost: 15 × 5 × 3 (estimators) ≈ 225 fits per asset class. Budget: ~30 minutes per class on CPU. **This time we run it**, because alken left it as deferred and the user wants the strongest methodology.

### 3.5 Per‑class Platt calibration

**[best‑of‑breed: alken's calibration discipline]**

* Fit Platt on the modelling‑OOF predictions of the **selected** model (per asset class), strictly before `predict_start`.
* Platt is monotone → AUC unchanged (unit‑tested invariant).
* Deliverable ships calibrated; raw retained for the §3 Brier story.
* Per‑class only — no per‑instrument calibration (too few samples per instrument; documented decision).

### 3.6 Cluster‑level importance with the four bug fixes

**[best‑of‑breed: alken's `cluster_importance.py`]**

* Mantegna distance `√(1 − |Spearman ρ|)`.
* PCA + optimal‑K K‑means.
* Cluster MDI + **purged cluster MDA** (with `PurgedKFold`) + cluster SHAP via `TreeExplainer`.
* `max_features = 'sqrt'` (not the deprecated `'auto'`).
* Output: per‑class cluster importance table + per‑instrument champion importance (top‑3 clusters per champion).

### 3.7 Strategy: fractional Kelly × vol target at 8 % ann vol

**[finding §1 + alken's sizing module]**

* `κ = 0.25` fractional Kelly.
* Confidence floor `p̂ ≥ 0.55` (hard) — smooth taper is implemented but not the default (rejected by alken's CER gate; we re‑run the gate on our better predictions and decide).
* **Vol target: `target_vol = 0.08`** — headroom under the 10 % cap. Realised vol will be 6‑9 % depending on the OOS window.
* Barrier‑exact backtest: exit on actual `t1` first touch, overlapping labels netted, Grinold‑Kahn cost model (2 bps half‑spread + 10 bps linear impact).
* Sortino computed with full‑T denominator (Sortino & Price 1994 — alken's pass‑5 fix).

### 3.8 Significance and deflation: report both, lead with significance

**[best‑of‑breed: alken's pass‑4/5 inference order]**

* PRIMARY inference: studentised stationary block‑bootstrap CI on per‑period Sharpe.
* Cross‑check: Lo/Opdyke analytic band.
* PSR(0), MinTRL.
* Ljung‑Box(10) gate before any √252 annualisation.
* Deflation (DSR ladder over `N_eff → 4·N_raw`, CSCV‑PBO with `C(16,8) = 12,870`, MinBTL) reported as corroboration.
* Pesaran‑Timmermann as the primary directional skill test.
* Treynor‑Mazuy + Henriksson‑Merton (labelled as base‑rate‑sensitive proxy) for corroboration.

### 3.9 Deterministic emit, byte‑identical re‑emit

**[best‑of‑breed: alken's emit pattern]**

* Rows sorted by `(date, instrument)`.
* Pinned column order: `["date", "instrument", "prediction"]` / `["date", "instrument", "weight"]`.
* ISO dates `%Y-%m-%d`, float format `%.10f`, `lineterminator="\n"`.
* Verified by `tests/experimental/test_byte_identical_emit.py`.
* Outputs: `metamodel_predictions.csv` (calibrated), `metamodel_predictions_raw.csv`, `strategy_weights.csv`, `coverage_caveat.csv`, `experiment_log.csv`.

### 3.10 What we deliberately do NOT build

* **Stacked ensemble of 8 models** (Sreeram v4): the data is too small to fit a meta‑learner without overfit; cf. alken's pass 2 rejection.
* **Per‑instrument calibration**: too few samples per instrument; alken found it hurts.
* **TDA / persistent homology** (Harry deferred): C++ build issues, low expected lift.
* **KerasVSN**: TF op‑determinism is best‑effort; rerun risk too high.
* **Smooth‑taper sizing as default**: rejected by alken's CER gate; we re‑evaluate but expect to revert.
* **Per‑instrument Baker‑McHale κᵢ as default**: rejected by alken's paired bootstrap; we re‑evaluate but expect to revert.
* **Manual selection of "drop equity from training"** (Sreeram v3): selection on test; methodology punishes it.

### 3.11 What we deliberately do NOT *claim* — the §2.8 discipline

In addition to architectural choices, §2.8 surfaces a **framing discipline**: there are statements the deliverable must not make even if the numbers tempt us. These belong on the negative list because they are easy mistakes:

* **Do not claim raw front‑month trading profitability.** The target is the course continuous‑contract series, not raw BBG front prices. Any backtest Sharpe is barrier‑outcome Sharpe on the course series.
* **Do not claim universal economic interpretability.** Macro / BBG features are interpretable for metals and equity; the report must explicitly caveat `ng1s` and (lighter) `ho1s` / `rb1s` as having materially more adjustment artefact in the target.
* **Do not claim per‑instrument `ng1s` performance is deployable.** If the model produces a strong `ng1s` AUC, the methodology document interprets it as construction‑specific (the model has learned the adjustment artefact's predictability, not the raw‑market signal). A neutral or near‑0.5 `ng1s` AUC is the honest reading and should not be presented as a failure.
* **Do not present "with BBG features" results without the parallel "without BBG features" ablation per asset class** (S3 acceptance gate addition — §13 R‑11). The submission ships a model that is robust to BBG feature missingness at the grader's hidden‑test runtime.

---

## 4. What we take from each existing branch (and what we do NOT)

Cross‑reference: full per‑branch detail in `/branch_descriptions.md`. This section is a focused inventory of code/concepts to lift and code to leave alone.

### 4.1 From `main` (the shared spine)

**TAKE — keep using as‑is (do not modify):**

* `src/stml/io.py` — `load_data` / `load_clean_data` / `load_returns_panel`
* `src/stml/na_checks.py` — calendars, missing‑data classifier, native returns, rolling vol, corr_max_info, rolling_pair_corr
* `data/ohlcv_data.csv` — read‑only
* `data/primary_signals.csv` — read‑only
* `reports/missing-data-report.md` — the authoritative NA policy (reference only)

**DO NOT TOUCH:** any of the above files. They are the shared base.

### 4.2 From `Sreeram` (forensic v0 → v5)

**TAKE — lift the patterns (copy with attribution into `src/stml/experimental/`):**

* `cv.PurgedKFold` and `walk_forward_splits` — the splitter implementation. We extend with per‑instrument embargo.
* `regimes.causal_filtered_probs` — the hand‑rolled forward HMM filter. **Critical**: hmmlearn's `predict_proba` is smoothed and would leak. We reuse Sreeram's filtered version.
* `features.py` core: F1 counter‑trend, F2 vol, F6 momentum, F8 calendar — the closed‑form computations. Copy the functions.
* `importance.py` — `cluster_features`, `clustered_mdi`, `clustered_mda` (with shared row permutation). Solid implementation.
* `evaluation.py` — `classification_report`, `per_instrument_breakdown`, `calibration_table`. Drop the regime‑conditional bits (alken's cluster importance covers them).
* The **diagnostic narrative discipline**: `docs/build/10-forensic-improvements.md` is a model template for documenting "we tried 16 fixes, here's what worked and what didn't". We will write the equivalent for this branch.

**LIFT WITH MODIFICATION:**

* `models.ElasticNetLogReg` — keep the structure but switch to single‑threaded determinism (`n_jobs=1`), and use the alken `MetaClassifier` interface for uniformity.
* `models.XGBoostMeta` — same: keep the structure, use the PS5 cell‑43 config (alken's), single‑threaded.
* `strategy.py` — keep the framework but reconfigure to `target_vol = 0.08`, no `target_portfolio_vol` (the brief constraint is ann vol, not portfolio vol).

**DO NOT TAKE:**

* `best_strategy.py` (the "drop equity from training" model) — selection‑on‑test by Sreeram's own v5 doc.
* `v4.py` (8‑model stacked ensemble) — over‑engineered for the data size; alken showed simple averaging is robust.
* `v5.py` as‑is — the architecture is sound but we already lift the principles separately.
* The "v0/v1/v2/v3/v4/v5" naming — violates R1.
* `predictions_v0.csv … _v4.csv` — same.

### 4.3 From `Harry` (independent audit + reconciled pipeline)

**TAKE — lift the patterns:**

* `signal_audit.py` — the block‑bootstrap signal characterisation. Run it on the released window for context (informational, not selection criterion).
* `labels.py` — **the t+1 entry triple barrier**. This is the load‑bearing fix. Copy the per‑instrument concurrency uniqueness implementation. Verify against the off‑by‑one test.
* `persist_events.py` pattern — SHA256 + config hash + per‑instrument balance metadata.
* `new_work/triple_barrier.py` — **the GARCH(1,1) σ̂**. We use this as the barrier scale.
* `new_work/cpcv_search.py` — **the CPCV barrier search machinery**. We feed it the real features (not the 7 placeholder features) and run per asset class.
* `new_work/hmm_vol.py` and `new_work/hmm_macro.py` — the frozen pre‑sample HMM regime features. **Alken's online EWMA HMM is the better choice** for this build (no CV seam artefact), but we keep Harry's Core #1 vol HMM as a supplementary feature because its log‑Garman‑Klass observation gives a different view.
* `harry/features/signal_trajectory.py` — F5 signal‑derived (run length, days since flip, entropy, flip rate, cum PnL).
* `harry/features/conditional_risk.py` — F15 bootstrap first‑passage MC. The path tortuosity feature is the highest‑value individual feature in our stable‑and‑strong list.
* `harry/features/microstructure_fixed.py` — F7 with the correct zero‑volume mask. **Critical**: Sreeram's G4 microstructure doesn't mask, propagates Inf.
* `harry/features/cross_asset.py` — F9 lead‑lag centroid distance, asset‑class dispersion z, EWMA implied‑corr z.
* `harry/features/macro_features.py` — F11 with publication lags. We re‑use the publication‑lag dictionary and the EIA release‑count windows.
* `harry/features/wavelet.py` — keep as optional. Drift is high (KS median 0.22) but a few features have non‑trivial AUC. Include behind a config flag, default ON for the equity class only.
* `harry/features/concept_drift.py` — F16. Used as feature AND inverse training weight.

**LIFT WITH MODIFICATION:**

* `new_work/champion_importance.py` — the structure (per‑champion focused importance) is good. We re‑use the within‑cluster top‑K breakdown + Kendall τ rank agreement, but apply it to OUR champions (per asset class).
* `new_work/reconciliation.py` — the per‑fold mean ± std AUC + STD‑based tie detection + calibration tiebreak is the right methodology. We adopt the same flow.

**DO NOT TAKE:**

* The two coexisting label conventions (`harry/labels.py` symmetric/EWMA + `new_work/triple_barrier.py` asymmetric/GARCH). We use **GARCH σ̂ + per‑class CPCV‑optimised barriers** — one convention.
* `harry/features/wavelet.py` for energy/metals: too drifted on H1‑22 test. (Equity OK.)
* The "4 of 11 instruments have signal" abstention pattern — we predict on all 11; coverage flags handle thin instruments.

### 4.4 From `model/alken-metamodel` (methodology gold standard)

**TAKE — lift the patterns wholesale:**

* The **methodology discipline** — pass plans, action tracker, RED‑first TDD, byte‑identical re‑emit, deterministic emit.
* The **two‑level mental model** but **not the directory layout** — alken's nested uv subproject is overkill for this branch; we use a single `src/stml/experimental/` package.
* `cross_validation.py` — `CombinatorialPurgedCV` (15 paths) + `nested_cpcv` + per‑instrument embargo via `embargo_days` map. Copy.
* `evaluation.py` — sample‑weighted purged OOS harness. Copy.
* `calibration.py` — Platt + isotonic + reliability + ECE. Copy.
* `cluster_importance.py` — Mantegna + the four bug fixes. Copy.
* `significance.py` — t‑stat + studentised stationary bootstrap + Lo CI + MinTRL + paired CER‑diff bootstrap. **Critical**: the studentised bootstrap with `optimal_block_length` is the PRIMARY inference. Copy.
* `deflation.py` — DSR ladder + CSCV‑PBO + MinBTL + ONC N_eff. Copy.
* `signal_analysis.py` — Pesaran‑Timmermann + Treynor‑Mazuy + Henriksson‑Merton (with the base‑rate‑sensitive caveat). Copy.
* `sizing.py` — fractional Kelly + vol target + smooth taper + Baker‑McHale. Copy.
* `backtest.py` — barrier‑exact + cost‑aware + the Sortino‑Price full‑T form + the S6.10 reconciliation. Copy.
* `cost_model.py` — Grinold‑Kahn half‑spread + impact. Copy.
* `regime.py` online EWMA HMM — copy (causal/fit‑free, no CV seam).
* `seeding.py` — copy.
* `_env.py` — copy (the macOS libomp segfault fix).
* `emit.py` — copy the deterministic CSV writer pattern.
* `pipeline.py` — copy the structure of `run_asset_class`, modify for our multi‑task NN + per‑class XGB.

**LIFT WITH MODIFICATION:**

* `features.py` (the per‑instrument feature adapter) — modify to use our drift‑filtered feature catalogue.
* `models.py` — extend the `MetaClassifier` interface to support the multi‑task NN (which takes both features and instrument id).
* `neural.py` — extend with the multi‑task architecture.
* `dim_reduction.py` `ClusterRepSelector` — copy; this is the promoted reducer for VSN input.

**DO NOT TAKE:**

* The pooled model architecture as the default — pooling within a class is fine, pooling across classes is not.
* The Keras VSN — TF op‑determinism risk.
* The autoencoder reducer for the deliverable — keep for EX.2 comparison only.
* The "175 features" matrix philosophy — we want ~120 drift‑filtered features, not maximal coverage.
* Macro LEVELS as features — convert to ranks (decided in §3.3).

### 4.5 From `signal-deep-dive` (referenced but not directly used)

The branch is on `remotes/origin/signal-deep-dive`. We do not directly check it out for this build. However the **feature catalog convention** (F‑prefixed families, provenance per feature) originated there and is propagated through alken. We adopt the convention.

We acknowledge the conflict their characterisation surfaced (counter‑trend at lag 1) but settle it the same way Harry did: **symmetric labels per default**, asymmetric per CPCV search per class.

---

## 5. The Bloomberg data ingestion plan

The user has full Bloomberg terminal access. This section specifies exactly what to pull, how to PIT‑align it, and how to wire it into the feature stack.

### 5.1 Constraints

* Only data inside the released training period (1990‑01‑02 to 2022‑06‑30 inclusive) is admissible. The OHLCV history starts 1990 so we have a long warm‑up.
* Every series gets a conservative **publication lag**. Daily market series: +1 day. Weekly inventory data: +5 days. Monthly macro releases: +30 days. CFTC: +3 days (Friday for Tuesday‑to‑Tuesday window).
* Bloomberg vintages are typically **revised final values**, not real‑time vintages. We note this as a Limitations bullet in the final report (same as alken's X.9 limitation).

### 5.2 What to pull — definitive list

**Block A — Futures term structure (F18 family, ~8 cols)**

For each commodity futures, pull both the front (1st) and second (2nd) generic contract continuous series, daily, full available history:

| BBG Ticker | Series | Used for |
|---|---|---|
| `CL1 Comdty` LAST_PRICE | WTI front | matches existing cl1s OHLCV |
| `CL2 Comdty` LAST_PRICE | WTI 2nd month | new |
| `HO1 Comdty`, `HO2 Comdty` LAST_PRICE | Heating oil 1st/2nd | new |
| `RB1 Comdty`, `RB2 Comdty` LAST_PRICE | RBOB Gasoline 1st/2nd | new |
| `NG1 Comdty`, `NG2 Comdty` LAST_PRICE | Natural gas 1st/2nd | new |
| `GC1 Comdty`, `GC2 Comdty` LAST_PRICE | Gold 1st/2nd | new |
| `SI1 Comdty`, `SI2 Comdty` LAST_PRICE | Silver 1st/2nd | new |
| `HG1 Comdty`, `HG2 Comdty` LAST_PRICE | Copper 1st/2nd | new |
| `PL1 Comdty`, `PL2 Comdty` LAST_PRICE | Platinum 1st/2nd | new |

Derived features per instrument:

* `f18_term_spread = (F2 - F1) / F1` — contango (positive) vs backwardation (negative)
* `f18_roll_yield = - f18_term_spread × (252 / days_to_roll)` — annualised carry
* `f18_term_spread_z63` — 63‑day rolling z‑score of `f18_term_spread`
* `f18_contango_flag = (f18_term_spread > 0).astype(int)` — regime indicator
* `f18_term_chg5 = f18_term_spread.diff(5)` — momentum of the term structure

**For equity (ES, NQ, FESX), use VIX futures term structure as the analogue:**

| BBG Ticker | Series |
|---|---|
| `VX1 Index` LAST_PRICE | VIX front month |
| `VX2 Index` LAST_PRICE | VIX 2nd month |

Derived (applies to all three equity instruments as a shared cross‑asset feature):

* `f18_vix_term = VX2 - VX1` — already in alken via F11 vix_term_slope (VIX3M − VIX). We keep BOTH because VX futures term structure is a tighter, more actionable measure.

Publication lag: **+1 day** (close‑to‑close, traded next day).

**Block B — Options‑implied vol (F19 family, ~9 cols per instrument)**

For each future, pull at‑the‑money and 25‑delta implied vol at 1‑month and 3‑month tenors. BBG fields:

| Field name (BBG) | Meaning |
|---|---|
| `1M_IMPVOL_100MNY_DF` | 1‑month ATM IV |
| `3M_IMPVOL_100MNY_DF` | 3‑month ATM IV |
| `1M_IMPVOL_75DELTA_DF` | 1‑month 25Δ put IV |
| `1M_IMPVOL_125DELTA_DF` | 1‑month 25Δ call IV |

Tickers: same as the futures (`CL1 Comdty`, `GC1 Comdty`, etc.).

For **equity**, the cleaner pull is the index option vol:

* `SPX Index` (or `ES1 Index`) with the same fields
* `NDX Index` (or `NQ1 Index`)
* `SX5E Index` (or `FESX1 Index` if available)

Derived per instrument:

* `f19_atm_iv_1m` — ATM 1‑month IV (level)
* `f19_atm_iv_3m` — ATM 3‑month IV
* `f19_iv_term_slope = iv_3m - iv_1m` — IV term structure
* `f19_iv_rv_spread = iv_1m - f2_vol_20` — VRP proxy
* `f19_skew_1m = put_iv - call_iv` — 25Δ skew at 1m
* `f19_risk_reversal_1m = call_iv - put_iv` — equivalent, opposite sign
* `f19_skew_z63` — 63‑day rolling z‑score of skew
* `f19_iv_pctile_252` — 252‑day rolling percentile of ATM IV (drift‑safe)

Fallback for instruments without listed options (or thin data): use the closest available proxy. For example, if `PL1 Comdty` options are thin, use the gold options as a proxy (their IV correlate ~0.7).

Publication lag: **+1 day**.

**Block C — CFTC Commitment of Traders positioning (F20 family, ~6 cols per instrument)**

CFTC publishes the COT report weekly (Friday for Tuesday's data, so the data is 3 days old). BBG carries it as a series of fields tied to each future. Generic tickers:

| BBG Field | Meaning |
|---|---|
| `CFTC_NONCOMM_LONG_FUT_ONLY` | Non‑commercial (specs) long |
| `CFTC_NONCOMM_SHORT_FUT_ONLY` | Non‑commercial short |
| `CFTC_COMM_LONG_FUT_ONLY` | Commercial long |
| `CFTC_COMM_SHORT_FUT_ONLY` | Commercial short |
| `CFTC_OPEN_INT` | Total open interest |

Pulled per future: CL1, HO1, RB1, NG1 (energy), GC1, SI1, HG1, PL1 (metals), ES1, NQ1 (equity), FESX1 (Euro Stoxx, may not be CFTC — use ICE COT or skip).

Derived per instrument:

* `f20_net_spec = (noncomm_long - noncomm_short) / open_int` — net spec position as fraction of open interest
* `f20_net_comm = (comm_long - comm_short) / open_int` — net commercial position
* `f20_spec_extremity = (f20_net_spec - rolling_median_156w(f20_net_spec)) / rolling_std_156w(f20_net_spec)` — z‑score over 3 years
* `f20_spec_pctile_156w` — 156‑week (~3‑year) rolling percentile
* `f20_open_int_chg5w` — 5‑week change in open interest
* `f20_position_skew = f20_net_spec - f20_net_comm` — divergence indicator

Publication lag: **+3 calendar days** (Tuesday data published Friday COB).

Forward‑fill onto trade calendar between weekly publications.

**Block D — Cross‑asset relative value (F21 family, ~4 cols)**

Compute from the existing OHLCV (no new BBG pull needed; we already have all the futures):

* `f21_gold_silver_ratio = close_gc / close_si`
* `f21_gold_silver_ratio_z63` — 63‑day rolling z‑score
* `f21_copper_gold_ratio = close_hg / close_gc`
* `f21_crack_321 = (2 × close_rb + close_ho) / 3 - close_cl` — the 3‑2‑1 crack spread (used for cl1s, ho1s, rb1s)

These are computed from the existing data; no new BBG ingestion needed.

**Block E — Event flags (F22 family, ~3 binary cols)**

From the BBG econ calendar API or `ECRL` function:

* FOMC meeting dates (typically 8 per year)
* EIA Petroleum Status Report release dates (weekly Wednesday)
* OPEC meeting dates (typically 2 main per year plus several smaller)
* USDA WASDE release dates (monthly)
* Major central bank dates (ECB, BOE, BOJ)

Derived (per row, per instrument, varies by instrument relevance):

* `f22_fomc_in_window = (any FOMC date in [t+1, t+h]).astype(int)` — for all instruments
* `f22_eia_in_window = (any EIA date in [t+1, t+h]).astype(int)` — for energy
* `f22_opec_in_window` — for cl1s/ho1s/rb1s

Encoded as binary; can be expanded to "days until next event" for more granularity.

### 5.3 Ingestion pipeline

**One‑time pull** (done by `src/stml/experimental/bloomberg_ingest.py`):

* Wraps `pdblp` or `xbbg` (Python BBG client) — assumes the BBG terminal is running.
* Pulls all the series above for the period 1990‑01‑02 to 2022‑06‑30 inclusive.
* Saves raw pulls to `data/bloomberg/raw/<series_name>.parquet`.
* Persists a manifest `data/bloomberg/manifest.json` with: tickers pulled, fields, date range, pull timestamp, BBG terminal user.
* No transformations at this stage — just bytes from BBG.

**Cleaning** (done by the same script in a second pass):

* For each series: forward‑fill missing values up to 5 business days (handle short outages).
* Apply publication lag: shift the observation index forward by `lag_days`.
* Forward‑fill onto the trade calendar (each instrument's `close.dropna().index`).
* Save to `data/bloomberg/cleaned/<series_name>.parquet`.
* Each cleaned file is keyed by `date` only (the instrument is in the filename).

**Loader** (`src/stml/experimental/bloomberg_loader.py`):

* `load_bloomberg(date_range, instruments)` returns a tidy DataFrame keyed by `(date, instrument)` with all available series.
* Handles missing instruments gracefully (some cross‑asset series have no instrument key).

**Caching:**

* The cleaned parquet files ARE the cache. They are committed to git (small enough — daily series 1990‑2022 is ~8000 rows × few cols each).
* The raw files are NOT committed (gitignored in `data/bloomberg/raw/`).

**Reproducibility:**

* The manifest JSON records the BBG terminal user and timestamp.
* If the BBG terminal is unavailable to a future agent, they can re‑run from the committed cleaned parquet files — the modelling pipeline reads from cleaned, not raw.

### 5.4 Where it plugs in

* `src/stml/experimental/features/term_structure.py` consumes `data/bloomberg/cleaned/term_structure_*.parquet` and produces F18 columns.
* Similarly for `options_iv.py` (F19), `positioning.py` (F20), `events.py` (F22).
* `relative_value.py` (F21) computes from existing OHLCV — no Bloomberg dependency.

All Bloomberg‑backed features carry an attribute `provenance = "bloomberg"` in the feature catalog so they can be ablated as a block.

### 5.5 If Bloomberg pull is not available immediately

If the user cannot pull BBG data before, say, 2026‑06‑03 morning, the build continues with the existing F1‑F17 stack only. The Bloomberg features are added in a single commit when the data lands, and we rerun the pipeline. The plan stays the same; the timeline shifts by half a day.

**This is the explicit fallback:** the build is robust to Bloomberg being delayed.

---

## 6. Repository structure (the canonical layout)

The new branch adds ONE package. Everything else is left untouched.

```
stml/
├── data/                                       (read-only existing inputs + new BBG)
│   ├── ohlcv_data.csv                          existing
│   ├── primary_signals.csv                     existing
│   ├── meta/                                   existing (na_checks outputs)
│   └── bloomberg/                              NEW
│       ├── raw/                                gitignored
│       ├── cleaned/                            committed (parquet per series)
│       └── manifest.json
│
├── src/stml/                                   shared spine (do not modify)
│   ├── __init__.py
│   ├── io.py                                   unchanged
│   ├── na_checks.py                            unchanged
│   └── experimental/                           NEW — THE ONLY package we write
│       ├── __init__.py
│       ├── _env.py                             single-thread native kernels
│       ├── seeding.py                          set_seeds()
│       ├── config.py                           ONE PipelineConfig dataclass
│       │
│       ├── data_loader.py                      OHLCV + signals + bloomberg consolidated
│       ├── bloomberg_ingest.py                 the BBG pull script (one-time, idempotent)
│       │
│       ├── volatility.py                       GK + GARCH(1,1) σ̂
│       ├── labels.py                           triple-barrier with t+1 entry, GARCH σ̂
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── catalog.py                      FeatureSpec registry + drift filter
│       │   ├── core.py                         F1/F2/F5/F6/F7/F8/F10
│       │   ├── path.py                         F12 + F15
│       │   ├── regimes.py                      F3 GMM + F17 HMM + EWMA online HMM + drift F16
│       │   ├── macro.py                        F11 — ranks not levels, PIT-lagged
│       │   ├── cross_section.py                F9
│       │   ├── term_structure.py               F18 (BBG)
│       │   ├── options_iv.py                   F19 (BBG)
│       │   ├── positioning.py                  F20 (BBG)
│       │   ├── relative_value.py               F21 (computed from existing OHLCV)
│       │   └── events.py                       F22 (BBG econ calendar)
│       │
│       ├── cv.py                               PurgedKFold + CPCV + nested CPCV + per-inst embargo
│       ├── models.py                           ElasticNet + XGBoost (the linear + tree baselines)
│       ├── multitask.py                        the multi-task NN with instrument heads
│       ├── dim_reduction.py                    ClusterRepSelector
│       │
│       ├── evaluation.py                       sample-weighted purged OOS harness
│       ├── importance.py                       Mantegna + clustered MDI/MDA/SHAP (4 bug fixes)
│       ├── calibration.py                      per-class Platt + isotonic + ECE
│       │
│       ├── significance.py                     studentised bootstrap + PT + MinTRL
│       ├── deflation.py                        DSR ladder + CSCV-PBO + MinBTL
│       │
│       ├── sizing.py                           κ=0.25, vol_target=0.08
│       ├── cost_model.py                       Grinold-Kahn
│       ├── backtest.py                         barrier-exact + cost-aware
│       │
│       ├── signal_analysis.py                  Pesaran-Timmermann + TM + H-M
│       ├── experiment_log.py                   deterministic per-run CSV log
│       ├── pipeline.py                         orchestrator: per-asset-class run + emit
│       └── emit.py                             deterministic CSV writer + CLI entry point
│
├── tests/                                      tests on the shared spine (do not modify)
│   └── experimental/                           NEW — mirrors the package, RED-first
│       ├── conftest.py
│       ├── test_byte_identical_emit.py
│       ├── test_volatility.py
│       ├── test_labels.py
│       ├── test_features_truncation_invariance.py
│       ├── test_macro_pit_alignment.py
│       ├── test_bloomberg_ingest.py
│       ├── test_term_structure.py
│       ├── test_options_iv.py
│       ├── test_positioning.py
│       ├── test_cv.py
│       ├── test_cv_per_instrument_embargo.py
│       ├── test_models.py
│       ├── test_multitask.py
│       ├── test_evaluation.py
│       ├── test_importance.py
│       ├── test_calibration.py
│       ├── test_significance.py
│       ├── test_deflation.py
│       ├── test_sizing.py
│       ├── test_backtest.py
│       ├── test_signal_analysis.py
│       └── test_pipeline.py
│
├── notebooks/
│   └── sreeram_experimental/                   NEW — personal scratch
│       ├── 01_data_audit.ipynb                 verifies BBG ingestion + label sanity
│       ├── 02_feature_drift_map.ipynb          KS test results + drift filter rationale
│       └── 03_per_instrument_ceiling.ipynb     single-feature ceiling reproduction
│
├── results/
│   └── sreeram_experimental/                   NEW — outputs
│       ├── feature_drift_audit.csv             per-feature KS, train AUC, val AUC, decision
│       ├── instrument_scope.json               per-instrument embargo_p90, n_eff
│       ├── label_outcome_audit.csv             PT/SL/vert composition per instrument
│       ├── cpcv_barrier_search/                
│       │   ├── equity_results.csv              winning (pt, sl, h) per class
│       │   ├── energy_results.csv
│       │   └── metals_results.csv
│       ├── importance/                         per-instrument cluster importance
│       │   ├── {inst}_findings.txt
│       │   ├── {inst}_clustered_mda.csv
│       │   ├── {inst}_global_shap.csv
│       │   └── {inst}_dendrogram.png
│       ├── master_results.csv                  per (class, model, instrument) AUC/Brier
│       ├── selection_table.csv                 champion per asset class
│       ├── reconciliation_report.md            the per-fold mean ± std verification
│       ├── coverage_caveat.csv                 thin-instrument flag
│       └── experiment_log.csv                  deterministic per-run log
│
├── reports/
│   └── sreeram_experimental/                   NEW
│       ├── plan.md                             THIS DOCUMENT (the golden record)
│       ├── methodology.md                      written as we go (the academic write-up)
│       ├── action_tracker.md                   PM-N chronological log (alken pattern)
│       └── final_report.md                     submission-ready Harvard-referenced report
│
├── outputs/                                    NEW — the DELIVERABLES (gitignored)
│   ├── metamodel_predictions.csv               calibrated
│   ├── metamodel_predictions_raw.csv           uncalibrated
│   └── strategy_weights.csv
│
├── pyproject.toml                              MAY modify (add deps: pdblp/xbbg, arch, shap)
├── uv.lock                                     update via uv add
├── .gitignore                                  MAY modify (add data/bloomberg/raw/)
├── branch_descriptions.md                      existing (the per-branch audit)
└── README.md                                   existing (unchanged)
```

### Why this layout

* **One package, not nested subproject.** Alken's nested `metamodel-apb/` is overkill for our scope. A flat `src/stml/experimental/` package is easier to import, test, and maintain. Single `pyproject.toml`.
* **`features/` is a subpackage** because there are 12+ feature modules; flat layout would be unreadable.
* **Tests mirror src/.** Easy for an agent to find the test for any module.
* **Reports separated from results.** Reports are written prose (markdown). Results are machine‑produced (CSV, JSON, PNG).
* **`outputs/` is the submission directory.** Gitignored. Re‑emit reproduces it.

---

## 7. Naming and "one version only" conventions

### 7.1 File names

* Module: `triple_barrier.py`, not `triple_barrier_garch.py` or `triple_barrier_v2.py`. The contents reflect the current best methodology; the methodology choice is described in the docstring and in this plan.
* Test: `test_<module>.py`. One test file per module, kept in `tests/experimental/`.
* Output CSV: `metamodel_predictions.csv`, `strategy_weights.csv`. No version suffix. Re‑emit overwrites.
* Notebook: `NN_short_description.ipynb` where NN is a 2‑digit ordering. `01_data_audit.ipynb`, `02_feature_drift_map.ipynb`. Notebooks are exploratory; if they evolve they get renamed (`git mv`).

### 7.2 Function and class names

* Functions: snake_case, verb_first. `compute_garch_sigma`, not `sigma_calculator`.
* Classes: PascalCase, noun. `PurgedKFold`, `MultiTaskMetaModel`.
* Private: leading underscore. Private helpers go at the bottom of the module, separated by `# --- private helpers ---`.

### 7.3 Config keys

* All hyperparameters live in `config.PipelineConfig` (a frozen dataclass).
* No magic numbers scattered through code; if a value is referenced more than once, it's a config field with a docstring.
* When we decide to change a hyperparameter (e.g. `kappa = 0.25 → 0.20`), we change the default in `config.py` ONCE. We do not add `kappa_v2` or `kappa_alt`.

### 7.4 Forbidden patterns (enforced by review, not by linter)

```python
# Bad: keeping the old way "for comparison"
if config.use_legacy_calibration:
    ...
else:
    ...

# Bad: parallel implementations
class CalibratorV1: ...
class CalibratorV2: ...

# Bad: time-stamped artefacts
predictions_20260603.csv

# Bad: notebook proliferation
01_attempt.ipynb
01_attempt_v2.ipynb
01_attempt_final.ipynb
```

If a comparison is genuinely needed for the methodology section (e.g. "v1 uses EWMA σ̂, v2 uses GARCH"), it goes in a single experiment script with both branches inline, writes a comparison CSV, and the conclusion is documented in `methodology.md`. The code path that wins is then promoted to default and the loser is deleted in the same commit.

### 7.5 The git workflow

Every commit message follows the conventional commits + stage tag pattern (alken's convention):

```
feat(s2): add F18 term structure features
fix(s1): correct GARCH σ̂ refit cadence (21 days, was 1 day)
docs(plan): update Stage 3 acceptance gate after running the barrier search
test(s4): RED-first test for multi-task NN head allocation
chore(deps): add arch package for GARCH
```

Stage tags: `s0` setup, `s1` data+labels, `s2` features, `s3` baseline model, `s4` multi-task NN, `s5` importance, `s6` calibration+sizing, `s7` significance, `s8` emit+report.

Plan changes always co‑commit with the corresponding code change (R10).

---

## 8. Stage‑by‑stage build plan with acceptance gates

Each stage has a concrete deliverable, a test gate (numerical or property), and a single canonical artefact. We do not advance to stage N+1 until stage N has cleared its gate.

### Stage 0 — Setup (target: half a day)

**Deliverables:**

* This `plan.md` committed.
* `src/stml/experimental/` package skeleton: `__init__.py`, `_env.py`, `seeding.py`, `config.py` (empty dataclass).
* `tests/experimental/conftest.py`: pytest config that sets seeds before each test.
* `pyproject.toml` updated to add: `arch` (GARCH), `shap`, `xbbg` or `pdblp` (BBG client), `lightgbm`, `hmmlearn` (already present via Sreeram).
* `notebooks/sreeram_experimental/` directory with a single `00_setup_verification.ipynb` that imports the package and runs `seeding.set_seeds()`.
* `.gitignore` updated for `data/bloomberg/raw/`.
* `outputs/` added to `.gitignore`.

**Acceptance gate:**

* `uv sync` succeeds with the new deps.
* `pytest tests/experimental/ -q` runs (zero tests pass — that's fine; zero failures matters).
* `python -c "from stml.experimental import _env, seeding, config"` succeeds.

**Single canonical artefact:** the `Sreeram_experimental` branch with the above files committed.

### Stage 1 — Data + labels (target: 1 day)

**Deliverables:**

* `volatility.py` with `garman_klass(ohlc, window)`, `parkinson(ohlc, window)`, `rogers_satchell(ohlc, window)`, `garch_sigma(close, h, refit=21, min_obs=500, max_window=2000)`.
* `labels.py` with:
  * `triple_barrier_labels(close, signal, sigma, pt_mult, sl_mult, max_holding) → events DataFrame`
  * Entry at `t + 1` (the Harry fix).
  * Returns `(t_signal, t_start, t_end, side, ret, label, uniqueness_weight, sigma_at_t)`.
  * Per‑instrument concurrency on the trading‑day index (no calendar‑day weeks).
* `data_loader.py` that loads OHLCV + signals (existing) and prepares the per‑instrument frames.
* Initial labels run: write `results/sreeram_experimental/label_outcome_audit.csv` with PT/SL/vert composition per instrument.
* `tests/experimental/test_volatility.py`, `test_labels.py` — RED‑first.

**Acceptance gate:**

* **Property test:** `test_labels.py::test_t_plus_one_entry` reproduces the Harry 5‑row example with both old (entry at t) and new (entry at t+1) conventions, asserts the labels differ. (This is the Harry test, copied.)
* **Property test:** `test_labels.py::test_uniqueness_weights` — disjoint events → weight 1, fully overlapping → weight 0.5.
* **Sanity test:** total event count matches Harry's `events.csv` (4886) to within ±50 (small differences from σ̂ differences).
* **Numerical gate:** vertical‑barrier fraction is < 65 % per instrument with the default `pt = sl = 0.5` barriers. If not, fall back to `pt = sl = 1.0` and document the trade‑off.

**Single canonical artefact:** `data/sreeram_experimental_events.parquet` (committed).

### Stage 2 — Feature stack (target: 2 days)

**Deliverables:**

* `features/catalog.py` — the `FeatureSpec` registry. Every feature self‑registers with its name, source data, leakage class, warmup window.
* `features/core.py` — F1/F2/F5/F6/F7/F8/F10. Copy from Sreeram/Harry, drift‑prune, drop `f5_signal`.
* `features/path.py` — F12 + F15. The stable + informative families.
* `features/regimes.py` — F3 + F17 + EWMA online HMM + F16 drift score.
* `features/macro.py` — F11 reformulated: LEVELS → 63‑day rolling ranks, CHANGES kept as‑is, PIT‑lagged via `pit_align`.
* `features/cross_section.py` — F9.
* `features/relative_value.py` — F21 (gold/silver, copper/gold, crack 3‑2‑1). Computed from existing OHLCV.
* **`bloomberg_ingest.py`** — the BBG pull script. Idempotent (re‑run skips already‑pulled). Documents the manifest.
* **`bloomberg_loader.py`** — reads cleaned parquet, joins to trade calendar.
* `features/term_structure.py` — F18.
* `features/options_iv.py` — F19.
* `features/positioning.py` — F20.
* `features/events.py` — F22.
* Feature drift audit: `results/sreeram_experimental/feature_drift_audit.csv` with per‑feature KS(train, val), train AUC, val AUC, KEEP/DROP decision.

**Acceptance gates:**

* **Property test:** every feature in the catalog passes `test_features_truncation_invariance.py` — value at `t` is identical on `data[:t+1]` and `data[:T]` for three sample `t`.
* **Property test:** `test_macro_pit_alignment.py` — release deferral test. A macro series observed on day D with lag 5 must not appear before D+5 in the feature.
* **Numerical gate:** ≥ 80 % of features have train→test KS < 0.20.
* **Numerical gate:** at least 15 features (excluding the family overlap) have val AUC > 0.55.
* **Numerical gate:** after drift filter, the matrix has 80–120 columns (target ~100).
* **Provenance:** `results/sreeram_experimental/feature_drift_audit.csv` exists and explains every KEEP/DROP decision.

**Bloomberg deliverable:** `data/bloomberg/manifest.json` lists every ticker/field pulled with date range and timestamp.

**Single canonical artefact:** `data/sreeram_experimental_features.parquet` (committed).

### Stage 3 — Per‑asset‑class XGBoost baseline (target: 1 day)

**Deliverables:**

* `models.py` with `ElasticNetLogReg` (logistic baseline, the linear family slot), `XGBoostMeta` (the tree slot), unified `MetaClassifier` interface.
* `cv.py` — copy alken's `PurgedKFold`, `CombinatorialPurgedCV(6,2)`, `nested_cpcv`, with per‑instrument embargo via `instrument_scope.json`.
* `evaluation.py` — sample‑weighted `cross_val_evaluate`, per‑instrument breakdown.
* `pipeline.py` — `run_asset_class(asset_class, config)` that builds the panel, runs the horse‑race, refits the winner, predicts the window.
* Run on the modelling sample (≤ 2021‑12‑31), produce per‑asset‑class baseline numbers.

**Acceptance gate (numerical):**

* Per‑class CPCV mean AUC ≥ alken's numbers (Equity 0.579, Energy 0.525, Metals 0.530). We expect to clear by 0.02–0.05 each because of the better features + drift filter, even before the multi‑task NN.
* **Per‑instrument AUC:** at least 6 of 11 instruments AUC > 0.55 on the modelling sample (alken got 5).
* **BBG missingness ablation** (§13 R‑11, NEW): produce `results/sreeram_experimental/bbg_missingness_ablation.csv` with three columns per asset class: `with_bbg_auc`, `without_bbg_auc`, `simulated_missingness_auc`. The "simulated missingness" run forces F18 / F19 / F22 to NaN on the validation slice (`> embargo_end`) and rescores. **Gate:** if `simulated_missingness_auc` drops > 0.03 AUC vs `with_bbg_auc` for any asset class, the architecture is BBG‑fragile and the S8 deliverable ships the `without_bbg` variant.

If we miss the gate, debug before moving on. Likely culprits: too aggressive drift filter, wrong σ̂ source, label noise.

**Single canonical artefacts:** `results/sreeram_experimental/baseline_xgb_per_class.csv` + `results/sreeram_experimental/bbg_missingness_ablation.csv`.

### Stage 4 — Multi‑task neural net (target: 2 days)

**Deliverables:**

* `multitask.py` — the multi‑task NN.
  * Input: row features (drop instrument one‑hot since it's encoded via the embedding).
  * Architecture: instrument embedding `Embedding(11, 8)` → concat with row features → `Linear(d_feat+8, 64)` → ReLU → Dropout(0.1) → `Linear(64, 32)` → ReLU → Dropout(0.1) → shared representation `h ∈ R^16` → 11 instrument‑specific heads `Linear(16, 1) + sigmoid`.
  * Joint training: `loss = sum over rows of uniqueness_weight × BCE(y, ŷ_head[inst])`. Only the head matching the row's instrument contributes.
  * Full‑batch Adam, full epoch shuffle disabled for determinism.
  * Early stop on inner CV val log‑loss.
* Trained per asset class (so 3 multi‑task NNs, one per class — within each class, the 3‑4 instruments are jointly trained).
* `tests/experimental/test_multitask.py`: byte‑deterministic forward pass + correct head allocation per row.

**Acceptance gate (numerical):**

* Per‑class CPCV mean AUC: ≥ baseline XGB + 0.03 (so Equity ≥ 0.62, Energy ≥ 0.56, Metals ≥ 0.56 on the modelling sample).
* **Per‑instrument lift:** ≥ 7 of 11 instruments show AUC improvement vs Stage 3 baseline.
* **Per‑instrument AUC headroom:** ≥ 0.05 better than the alken per‑instrument numbers on at least 5 instruments.

If we miss, the NN is likely too big — drop hidden width from 64 to 48 and retry.

**Single canonical artefact:** `results/sreeram_experimental/multitask_per_class.csv`.

### Stage 5 — Per‑class cluster importance with bug fixes (target: half a day)

**Deliverables:**

* `importance.py` — Mantegna + clustered MDI + purged MDA + cluster SHAP, with the four bug fixes.
* `dim_reduction.py` — `ClusterRepSelector` for reduction (used in Stage 4 NN if needed).
* Per‑class cluster importance run.
* `results/sreeram_experimental/importance/<class>_clustered_mda.csv`.

**Acceptance gate:**

* All four bug fixes present (test asserts: max_features='sqrt', PurgedKFold for MDA, SHAP computed, Mantegna distance).
* At least one cluster per class has `MDA > 0.02` (a genuinely informative cluster, like alken's Equity 0.025). If not, the model has near‑zero edge, which agrees with the Grinold ceiling but indicates we should look at the labels again.

### Stage 6 — Calibration, sizing, backtest (target: 1 day)

**Deliverables:**

* `calibration.py` — Platt per asset class, fit on modelling‑OOF.
* `sizing.py` — fractional Kelly with `κ = 0.25`, vol target `0.08`, hard floor `p̂ ≥ 0.55`.
* `cost_model.py` — Grinold‑Kahn 2 bps half‑spread + 10 bps impact.
* `backtest.py` — barrier‑exact + cost‑aware, full‑T Sortino, S6.10 reconciliation.
* `emit.py` — deterministic CSV writer, CLI entry point.
* First end‑to‑end deliverables: `outputs/metamodel_predictions.csv` (calibrated) and `outputs/strategy_weights.csv`.

**Acceptance gate:**

* **Calibration:** Platt monotonicity preserved (AUC invariant before/after Platt). Unit‑tested.
* **Strategy:** **realised annualised vol ∈ [0.06, 0.10]** on the H1‑2022 OOS. If realised vol > 0.10, the strategy breaches the constraint — adjust `target_vol` lower.
* **Byte‑identical emit:** two independent runs produce identical CSVs.

### Stage 7 — Significance + deflation (target: half a day)

**Deliverables:**

* `significance.py` — t‑stat, studentised stationary bootstrap CI, Lo/Opdyke analytic, MinTRL, paired CER‑diff bootstrap.
* `deflation.py` — DSR ladder, CSCV‑PBO, MinBTL, ONC N_eff.
* `signal_analysis.py` — Pesaran‑Timmermann, Treynor‑Mazuy, H‑M proxy.
* Run all of the above on the H1‑2022 OOS deliverable.
* `results/sreeram_experimental/significance_summary.md` summarising t‑stat, CI, MinTRL, DSR ladder, PT stat, PBO.

**Acceptance gate:**

* All numbers reproducible from `outputs/strategy_weights.csv`.
* **Significance verdict:** EITHER (a) bootstrap CI excludes zero AND DSR > 0.95 at N_eff (the strong case) OR (b) bootstrap CI contains zero AND we report the honest negative (the alken case). We do not artificially inflate the headline.

### Stage 8 — Documentation and submission (target: 1 day)

**Deliverables:**

* `reports/sreeram_experimental/methodology.md` — full methodology write‑up. Mirrors alken's structure but reports OUR numbers.
* `reports/sreeram_experimental/final_report.md` — submission‑ready academic report with Harvard references. Mirrors alken's `T3_03_Alken_Metamodel_Report.md`. Honest framing: "we improved AUC from X to Y because of A, B, C; the strategy {is / is not} statistically distinguishable from zero".
* `reports/sreeram_experimental/action_tracker.md` — chronological log of PM‑N entries (one per session).
* Final byte‑identical re‑emit verification.

**Acceptance gate (final submission readiness):**

* `outputs/metamodel_predictions.csv` exists and conforms to the brief format.
* `outputs/strategy_weights.csv` exists and realised vol ≤ 10 %.
* `pytest tests/experimental/ -q` passes 100 %.
* `python -m stml.experimental.emit --predict-start 2022-01-01 --predict-end 2022-06-30` runs end‑to‑end in < 60 minutes.
* `python -m stml.experimental.emit --predict-start 2022-07-01 --predict-end 2022-12-31` runs end‑to‑end (the grader's rerun simulation; we cannot verify the numbers but the pipeline must succeed without errors).
* `methodology.md` and `final_report.md` complete with no `[TBD]` markers.

---

## 9. Testing strategy (RED‑first TDD)

Following alken's discipline. Every new behaviour is tested before it's implemented.

### 9.1 RED‑first workflow

For each feature:

1. Write the test first. It fails (RED).
2. Implement the minimum code to make it pass (GREEN).
3. Refactor for clarity. Test still passes.
4. Commit `feat(sN): <description>` + `test(sN): <description>` together.

### 9.2 Test categories

* **Truncation invariance** — every E‑class feature.
* **Determinism / byte identity** — every artefact‑producing function.
* **Known‑value tests** — e.g. uniqueness weight on disjoint events = 1.
* **Leakage guards** — `test_no_module_consumes_frozen_parquet` (analogous to alken's), `test_pit_alignment_blocks_unreleased_data`.
* **Property tests** — invariants like "Platt is monotone → AUC unchanged".
* **Integration tests** — `test_pipeline.py` runs end‑to‑end on a synthetic 20‑instrument 2‑year panel.

### 9.3 Coverage target

* Every module under `src/stml/experimental/` has a corresponding `test_<module>.py`.
* No module ships without at least one test.
* Test count target: ≥ 150 tests at submission.

### 9.4 What NOT to test

* Don't test sklearn / xgboost / torch internals.
* Don't test private helpers — test the public API.
* Don't write "test that this works" — write "test that this returns X on input Y".

---

## 10. Determinism and reproducibility contract

* `seeding.set_seeds(seed=42)` synchronises `random`, `numpy`, `torch`, `tensorflow` (if loaded), `PYTHONHASHSEED`. Called at every entry point.
* `_env.py` sets single‑thread native kernels (BLAS, OpenMP) and fixes the macOS libomp segfault.
* All RNG‑using functions accept a `seed` parameter with default 42.
* Per‑row seeded operations (e.g. bootstrap MC for `f15_expected_hit_time`) use the formula `seed × 1_000_003 + t_position` so the value at row `t` is identical on truncated and full input.
* CSV emit: sorted rows, pinned column order, ISO date, `%.10f` float, `lineterminator="\n"`.
* **Byte‑identical re‑emit verified** at each stage gate.

---

## 11. Leakage and cross‑validation discipline

### 11.1 Feature‑level

* Every E‑class feature: truncation‑invariance test.
* Every TF (fitted) feature: fits on the contiguous prefix `dates < fit_end`, applied with frozen params on the full series.
* Macro features: PIT‑lagged via `pit_align`. Release‑deferral test asserts a macro series with lag 5 doesn't appear before D+5.
* The frozen alken `feature_matrix.parquet` is **never consumed** by the modelling pipeline. We recompute features per fold from raw OHLCV + signals + Bloomberg.

### 11.2 Label‑level

* Labels are computed once on the full history (they're stateless given the σ̂ series).
* The σ̂ used in labelling (GARCH) is causal — per‑bar `σ̂_t` uses only data `≤ t`.
* The `t1` (first‑touch) is recorded per label for purging.

### 11.3 CV‑level

* **`PurgedKFold`** drops train events whose `[t, t1]` overlaps the test block.
* **Per‑instrument embargo** advances each instrument's `embargo_p90` (from `instrument_scope.json`) on that instrument's own date axis.
* **`CombinatorialPurgedCV(6, 2)`** → 15 OOS paths.
* **Nested CPCV** (outer (6,2), inner (5,1)) for selection‑bias‑aware evaluation.
* Calibration: Platt fit on **modelling‑OOF** (purged predictions on dates `≤ modelling_end`), strictly before `predict_start`.
* No model hyperparameters tuned by looking at the prediction window.

### 11.4 The hidden test (H2‑2022)

* The grader reruns the code on H2‑2022.
* Our `--predict-start` and `--predict-end` are CLI flags. The grader changes the value.
* The pipeline does not have ANY hardcoded H1‑22 references in the modelling code (only in the default config and in result reports).
* Test: `test_pipeline.py::test_runs_on_alternate_window` — invoke the pipeline with `predict_start=2022-07-01, predict_end=2022-12-31` and assert it completes without error (numbers will be NaN because we don't have the H2 data, but the structure must hold).
* **Bloomberg parquets cannot legally cover H2‑2022** — the released period (plan §5.1) ends 2022‑06‑30 and pulling beyond it would violate the brief. The model therefore ships under the simulated‑missingness winner decided in S3 (with‑BBG if robust, otherwise the no‑BBG baseline). See §13 R‑11.

### 11.5 Methodology framing — the report language §2.8 mandates

The final report (`reports/sreeram_experimental/final_report.md`) MUST use language consistent with §2.8 + §3.11. The two canonical sentences:

> *"The released OHLCV data are continuous futures series rather than raw front‑month Bloomberg prices. We therefore define all labels on the provided continuous‑contract series, matching the evaluation target. To assess whether external market features remain economically meaningful, we compared course‑series returns with raw Bloomberg front‑contract returns. Metals show near‑identical return dynamics (R² > 0.995), equity index futures remain mostly coherent (R² 0.94–0.97), and the daily‑return process for these classes is close to raw market; real macro and IV features therefore remain interpretable. Energy contracts, especially natural gas (R² 0.72; 11 % label flip relative to a raw‑futures relabel), show materially larger adjustment artefacts; for these instruments, external features may predict the raw‑market component but not the adjustment component of the course target. We treat energy results with caution and evaluate feature families by asset class."*

> *"Our results should be interpreted as predictions of barrier outcomes on the coursework continuous‑contract target, not as direct evidence of deployable raw front‑month futures trading profitability."*

The report does NOT use:
* "raw futures profitability"
* "tradeable WTI / Brent / NG signal"
* "macro features predict gold/oil/copper" without the per‑asset caveat
* `ng1s` results framed as deployable

---

## 12. Submission readiness checklist

Run through this list before declaring done. Every item must be ✅.

### Code

* [ ] `pytest tests/experimental/ -q` → 100 % pass
* [ ] `python -m stml.experimental.emit` runs end‑to‑end in < 60 min
* [ ] `python -m stml.experimental.emit --predict-start 2022-07-01 --predict-end 2022-12-31` runs end‑to‑end without error
* [ ] No filename matches the forbidden patterns from §7.4
* [ ] Two independent `emit` runs produce byte‑identical `outputs/*.csv`

### Deliverables

* [ ] `outputs/metamodel_predictions.csv` exists, 1000–1100 rows, format `(date, instrument, prediction)`, all in [0, 1]
* [ ] `outputs/strategy_weights.csv` exists, same row count, format `(date, instrument, weight)`
* [ ] `outputs/coverage_caveat.csv` flags thin instruments (ho1s, gc1s, ng1s)
* [ ] Realised ann vol on the strategy ≤ 10 %

### Methodology evidence

* [ ] `reports/sreeram_experimental/methodology.md` complete with no `[TBD]`
* [ ] `reports/sreeram_experimental/final_report.md` complete with Harvard refs
* [ ] All numerical claims in the report reproduce from a CSV under `results/sreeram_experimental/`
* [ ] Per‑instrument AUC table in the report sources from `results/sreeram_experimental/master_results.csv`
* [ ] Significance section reports t‑stat, bootstrap CI, MinTRL, DSR ladder, PT stat
* [ ] At least four of the five lenses (AUC, MDA, significance, deflation, PT) agree on the verdict
* [ ] **§2.8 OHLCV‑adjusted‑continuous framing language present** (the two §11.5 canonical sentences appear in §1 and §6 of `final_report.md`)
* [ ] **`results/sreeram_experimental/bbg_missingness_ablation.csv` exists** (§13 R‑11) with `with_bbg_auc`, `without_bbg_auc`, `simulated_missingness_auc` per asset class
* [ ] **`coverage_caveat.csv` flags `ng1s` for low feature‑target coherence** alongside the thin‑instrument flags (ho1s, gc1s, ng1s)
* [ ] **No use of "raw futures profitability" / "tradeable WTI signal" language** in `final_report.md` — language audit per §11.5

### Hidden‑test BBG coverage (R‑11 — pre‑submission decision)

* [ ] Shipped model is the simulated‑missingness winner decided in S3 (with‑BBG with NaN‑tolerant inference if it survives the < 0.03 AUC delta test, otherwise the no‑BBG baseline) and the decision is documented in `methodology.md`. Extending the BBG pull to H2‑2022 is NOT allowed (released period stops 2022‑06‑30 per §5.1).

### Required by the rubric

* [ ] **Feature engineering** (20): catalogued in `methodology.md`, every feature has a citation or justification
* [ ] **Labelling** (20): triple‑barrier with CPCV‑searched barriers; the search results in `results/sreeram_experimental/cpcv_barrier_search/`
* [ ] **Model development** (30): three families (linear, tree, NN) compared via CPCV, per asset class
* [ ] **Cluster importance** (10): four bug fixes documented, per‑class cluster MDA + SHAP in `results/sreeram_experimental/importance/`
* [ ] **OOS evaluation** (20): per‑instrument breakdown, calibration metrics, blind‑primary baseline, threshold sweep
* [ ] **Strategy (+10)**: barrier‑exact + cost‑aware backtest with full‑T Sortino, significance + deflation gate, realised vol ≤ 10 %

### Plan and tracker hygiene

* [ ] `plan.md` updated to reflect the final state (no `[TBD]` in the plan)
* [ ] `action_tracker.md` has at least one PM‑N entry per build session
* [ ] `git log --oneline` shows the build narrative cleanly (stage‑tagged commits)

---

## 13. Risk register — what could go wrong

### R‑1 — Bloomberg pull is unavailable or thin

**Risk:** terminal access issues, or the BBG fields we want are not available for our instruments / period.

**Mitigation:**
* Fall back to F1‑F17 only. The pipeline must still run.
* For each F18‑F22 family, the code has a `try / except ImportError / FileNotFoundError` guard that drops the family if data is missing.
* Document the fallback in `methodology.md` as a limitation.

### R‑2 — Tighter barriers crash the label balance

**Risk:** `pt = sl = 0.5` produces near‑even PT/SL distributions but pos rate drops to 0.45–0.50 (closer to random), making the meta‑label uninformative.

**Mitigation:**
* CPCV barrier search has both gates: (a) PT+SL fraction > 50 % AND (b) class balance ∈ [0.45, 0.65].
* If both fail, fall back to `pt = sl = 1.0` and document the trade‑off (wider barriers but vertical‑noise problem).

### R‑3 — Multi‑task NN overfits per‑instrument heads

**Risk:** with only 60–600 events per instrument in training, the per‑instrument heads memorise.

**Mitigation:**
* Hard cap hidden layer widths (≤ 64).
* Aggressive dropout (0.1–0.2).
* Early stopping on inner CV val loss.
* If a per‑instrument head's val AUC is < 0.50, fall back to per‑asset‑class XGBoost for that instrument's predictions (selection done by inner CV, not by looking at test).

### R‑4 — Nested CPCV is too slow

**Risk:** 225 fits per class × 3 classes × multi‑task NN at 30 s per fit = 5–6 hours. Eats into the timeline.

**Mitigation:**
* Run nested CPCV ONLY on the tree models (faster). Run plain CPCV on the multi‑task NN.
* Document the asymmetry in `methodology.md`.

### R‑5 — Determinism breaks on a different machine

**Risk:** the user's machine and the grader's machine produce different bytes from the same code.

**Mitigation:**
* `seeding.set_seeds()` covers `random / numpy / torch / tensorflow / PYTHONHASHSEED`.
* `_env.py` forces single‑thread BLAS / OpenMP.
* CSV emit pins format / sort / lineterminator.
* `test_byte_identical_emit.py` runs on every commit (verified locally before push).
* Final submission tested on the .venv that ships with `uv sync` — if it works locally with a fresh `uv sync`, it works for the grader.

### R‑6 — Strategy realised vol exceeds 10 %

**Risk:** Even with `target_vol = 0.08`, realised vol can exceed 10 % if the strategy is more aggressive than the vol model predicted.

**Mitigation:**
* `sizing.py` enforces a leverage cap (`MAX_LEVERAGE = 5.0`).
* Final check: run backtest, measure realised vol. If > 10 %, drop `target_vol` to 0.07 and re‑emit.
* This is iterative until the constraint is satisfied — but we never look at AUC during the vol calibration loop. AUC is locked at Stage 6's end.

### R‑7 — Selection on test

**Risk:** we pick a feature / model / hyperparam because it improves H1‑22 OOS.

**Mitigation:**
* Every selection happens via CPCV on the modelling sample (≤ 2021‑12‑31).
* The H1‑22 test is touched ONCE at the end for the final number.
* `methodology.md` documents every selection criterion.
* Action tracker records selection events with date and rationale.
* If we catch ourselves selecting on test, we revert and document (alken's pass 4/5 discipline).

### R‑8 — The Grinold ceiling means we can't actually hit 0.65–0.75

**Risk:** the AUC ceiling on a near‑zero‑IC primary is around 0.60 even per‑instrument.

**Mitigation:**
* If at Stage 4 acceptance gate we are stuck at 0.55 per class, we accept the result and shift the narrative to "honest negative" like alken.
* The methodology marks are 100; the AUC chase is to support the methodology, not the other way around.
* **A well‑documented honest negative beats a poorly‑documented strong number** under the brief's rubric.

### R‑9 — Bloomberg data has different conventions than our existing OHLCV

**Risk:** BBG `CL1 Comdty` may use a different roll convention than the `cl1s` we have in `ohlcv_data.csv`. Term structure derived from misaligned series is garbage.

**Status (2026‑06‑02 night):** **CONFIRMED in practice — see §2.8.** OHLCV is back‑adjusted continuous (ratio / proportional), BBG raw is unadjusted. We do NOT mix them. F18 term structure uses BBG‑raw front and BBG‑raw 2nd (both from the same raw convention); labels and F1‑F17 + F21 use OHLCV. R‑9 is therefore *handled architecturally*, not by a correlation test.

**Mitigation in code:** `bloomberg_validation_report.md` documents the OHLCV‑vs‑BBG scale divergence (cl1s 24.80 vs $61.18 in 2020); `features/bloomberg.py` uses `_front_second_close()` that pulls both legs from `data/bloomberg/cleaned/futures_term.parquet` only, never mixing in OHLCV.

### R‑10 — Continuous‑contract adjustment artefact on the target (especially `ng1s`)

**Risk:** The course OHLCV is an adjusted continuous series; the implied return process is `course_return = raw_market_return + adjustment_artefact` (§2.8.1). For `ng1s` the artefact is large (R² 0.72 vs raw, 11 % label flip rate). A model trained on course `ng1s` labels learns a mixture of the raw‑market signal and the adjustment artefact; external macro / IV / COT features predict the raw‑market component but not the artefact. The `ng1s` per‑instrument AUC may therefore be **construction‑specific** rather than economically deployable.

**Materiality:** Asset‑specific. Metals strongly coherent (R² > 0.995); equity / crude mostly coherent; HO / RB caution; `ng1s` problematic. The risk is not that the project fails — the grader's hidden test uses the same continuous‑contract convention, so the supervised problem is internally consistent — but that the **methodology document overclaims raw‑market deployability**.

**Mitigation (plan §3.11 framing discipline + S8 methodology language):**

1. Methodology document presents `ng1s` results as construction‑specific, not deployable. The "we predict barrier outcomes on the provided continuous‑contract series" language goes in §1 of the report and is repeated in §6 (limitations).
2. The per‑instrument breakdown table in `final_report.md` includes a `feature_target_R2_vs_raw` column or footnote citing §2.8.2 — so the reader sees which per‑instrument AUCs sit on coherent vs noisy targets.
3. Per‑class XGBoost / multi‑task NN avoid pooling across asset classes (§3.1), so noisy energy targets don't contaminate the metals decision boundary.
4. We DO NOT drop `ng1s` from the deliverable; the brief allows ≥1 full class and we ship all 11. The deliverable's `coverage_caveat.csv` flags `ng1s` alongside thin instruments (ho1s, gc1s) so the reader is told.

### R‑11 — Bloomberg feature missingness in the H2‑2022 hidden test

**Risk:** Cleaned BBG parquets (`data/bloomberg/cleaned/*.parquet`) cover up to **2022‑06‑30**. The grader's hidden test is **H2‑2022 (Jul → Dec 2022)**. At rerun time, F18 / F19 / F22 columns will be all‑NaN for the prediction window — a real distribution shift, not just a NaN‑handling mechanic. XGBoost's native NaN handling masks the issue but does not solve it.

**Constraint (corrected 2026‑06‑03 PM):** Per plan §5.1 the released training period is **1990‑01‑02 → 2022‑06‑30 inclusive**. Pulling BBG data for H2‑2022 **violates the brief** and is therefore NOT a mitigation available to us. The grader extends the OHLCV + signals through H2‑2022 themselves at rerun time; they do not extend our BBG parquets. The architecture must be robust to BBG missingness at H2‑2022 by construction.

**Mitigation (corrected — preference order):**

1. **Validate under simulated missingness during S3.** S3 acceptance gate (mandatory): train the model on the modelling sample, force F18 / F19 / F22 to NaN on a held‑out validation slice that mimics H2‑2022, and report the AUC delta. If the model degrades > 0.03 AUC under simulated missingness on ANY asset class, the architecture is BBG‑fragile and we ship option (2) instead.
2. **Train and ship a "no‑BBG" robustness baseline** (F1–F17 + F21 only). Already documented in §5.5 as a fallback; now a **mandatory parallel model** alongside the with‑BBG variant. The submission picks the variant that wins on the simulated‑missingness validation, not the variant that wins on H1‑2022 OOS (which would be selection on test).
3. **Use NaN‑tolerant inference even on the BBG model**: XGBoost handles NaN natively; the multi‑task NN gets BBG‑feature indicators dropped before the head allocation. This is the implementation backstop if both (1) shows robustness AND (2)–(3) say BBG features are too useful to drop entirely.

**Why pulling H2‑2022 BBG is NOT on this list:** the released period is the hard line. We do not violate it even for a clean engineering fix. The methodology is the grade.

**S3 / S8 acceptance gate addition (mandatory):**

* `results/sreeram_experimental/bbg_missingness_ablation.csv` exists and reports `with_bbg_auc`, `without_bbg_auc`, `simulated_missingness_auc` per asset class.
* `methodology.md` documents which variant was shipped and why.

---

## 14. Glossary and pointers

### Cross‑references

* Full per‑branch audit: `/branch_descriptions.md`
* Sreeram source code: `src/stml/` (existing) and `docs/build/*.md`
* Harry source code: `/tmp/stml-harry/src/stml/harry/` and `/tmp/stml-harry/src/stml/new_work/`
* Alken source code: `/tmp/stml-alken/metamodel-apb/src/alken_metamodel/`
* Alken methodology: `/tmp/stml-alken/metamodel-apb/docs/methodology.md`
* Alken academic report: `/tmp/stml-alken/metamodel-apb/reports/T3_03_Alken_Metamodel_Report.md`

### Glossary

* **AUC** — area under ROC curve; ranking metric, 0.5 = random, 1.0 = perfect.
* **AFML** — Advances in Financial Machine Learning (López de Prado 2018).
* **CPCV** — Combinatorial Purged Cross‑Validation. `n_groups=6, n_test_groups=2` → C(6,2) = 15 paths.
* **CSCV‑PBO** — Combinatorial Symmetric CV Probability of Backtest Overfitting. `n_blocks=16, C(16,8)=12,870` (not 12,780).
* **DSR** — Deflated Sharpe Ratio (Bailey & López de Prado 2014). Adjusts for selection bias.
* **F‑family** — feature family with prefix (e.g. F11 = macro). Convention from signal‑deep‑dive, adopted by alken and us.
* **GARCH(1,1)** — Generalised AutoRegressive Conditional Heteroskedasticity model. Forecast σ̂.
* **GK** — Garman‑Klass volatility estimator. Range‑based, captures gaps.
* **IC** — Information Coefficient. Spearman rank correlation between signal and forward return.
* **IR = IC · √BR** — Grinold's Fundamental Law of active management.
* **MinBTL** — Minimum Backtest Length (Bailey et al. 2014).
* **MinTRL** — Minimum Track Record Length (Bailey & López de Prado 2012).
* **OHLCV** — Open, High, Low, Close, Volume.
* **PIT** — Point‑In‑Time. A series is PIT‑aligned if it shows only data released by time `t`.
* **PSR** — Probabilistic Sharpe Ratio.
* **PT** — Pesaran‑Timmermann directional accuracy test (1992).
* **σ̂** — sigma‑hat, the estimated daily volatility.
* **t1** — first‑touch time of a triple‑barrier label (AFML Ch.3).
* **TM** — Treynor‑Mazuy convexity timing test (1966).
* **TF** — Time‑Fitted feature class (vs E for engineered/stateless).

### Key citations

We will use a subset of the alken bibliography:

* López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. [Triple‑barrier Ch.3, uniqueness Ch.4, purged CV Ch.7, CPCV Ch.12, cluster importance Ch.6 in *Machine Learning for Asset Managers* 2020]
* Bailey, D.H. & López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*.
* Bailey, Borwein, López de Prado & Zhu (2017). The probability of backtest overfitting. *Journal of Computational Finance*.
* Garman, M.B. & Klass, M.J. (1980). On the estimation of security price volatilities from historical data. *Journal of Business*.
* Grinold, R.C. (1989). The fundamental law of active management. *Journal of Portfolio Management*.
* Kelly, J.L. (1956). A new interpretation of information rate. *Bell System Technical Journal*.
* Lim, B., et al. (2021). Variable Selection Networks for Tabular Data (TFT). [VSN reference]
* Mantegna, R.N. (1999). Hierarchical structure in financial markets. *European Physical Journal B*.
* Pesaran, M.H. & Timmermann, A. (1992). A simple nonparametric test of predictive performance. *Journal of Business & Economic Statistics*.
* Politis, D.N. & Romano, J.P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*.
* Sortino, F.A. & Price, L.N. (1994). Performance measurement in a downside risk framework. *Journal of Investing*.
* Treynor, J.L. & Mazuy, K.K. (1966). Can mutual funds outguess the market? *Harvard Business Review*.

---

## Appendix A — Quick start for a fresh agent

If you are a fresh agent (or me after context compacts) picking this up:

1. **Read this file end‑to‑end.** Especially §0 (cardinal rules) and §3 (architectural decisions).
2. **Check the branch:** `git branch --show-current` should be `Sreeram_experimental`. If not: `git checkout Sreeram_experimental`.
3. **Check the stage:** look at `reports/sreeram_experimental/action_tracker.md` for the most recent PM‑N entry. That tells you what stage we're on.
4. **Check the tests:** `pytest tests/experimental/ -q`. Failures point to where the build is.
5. **Pick up where the last entry left off.** Do NOT redo work that's already in.
6. **Update the action tracker** with what you're doing in the current session.
7. **Update this plan** if you discover a methodology change is needed.
8. **One version only.** No new file names that conflict with the layout in §6.

## Appendix B — Quick start commands

```bash
# Setup (one-time on a fresh clone)
cd /Volumes/Sreeram/Systematic/stml
git checkout Sreeram_experimental
uv sync                           # install deps from lock file
uv run pytest tests/experimental/ # verify tests pass

# Run a stage
uv run python -m stml.experimental.bloomberg_ingest        # one-time BBG pull
uv run python -m stml.experimental.emit                    # produce deliverables

# Re-emit (verify determinism)
uv run python -m stml.experimental.emit
diff outputs/metamodel_predictions.csv outputs/metamodel_predictions.csv.prev
# expected: no output (byte identical)

# Run on the grader's hypothetical window
uv run python -m stml.experimental.emit \
    --predict-start 2022-07-01 --predict-end 2022-12-31
# expected: runs end-to-end without error
```

## Appendix C — Stage gate one‑liner table

| Stage | Deliverable | Key gate |
|---|---|---|
| S0 | Branch + skeleton | `pytest` runs |
| S1 | Labels + GARCH σ̂ | vertical‑barrier fraction < 65 % per inst |
| S2 | ~100 drift‑filtered features incl. F18‑F22 | ≥ 80 % features KS < 0.20 |
| S3 | Per‑class XGB baseline | beats alken numbers by ≥ 0.02 per class |
| S4 | Multi‑task NN | per‑class AUC ≥ baseline + 0.03 |
| S5 | Cluster importance | ≥1 cluster per class MDA > 0.02 |
| S6 | Calibration + sizing + backtest | realised vol ∈ [0.06, 0.10] |
| S7 | Significance + deflation | all numbers reproduce from artefact |
| S8 | Report + final check | submission readiness checklist 100 % |

---

**End of plan. Edit in place when methodology changes. Commit with `docs(plan): <change>`. No `plan_v2.md`.**
