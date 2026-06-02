# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 1. What this project is

**Imperial College London / Alken Asset Management — "Systematic Trading Strategies with Machine Learning Algorithms."** Group coursework (team of 5), **50% of the final grade**, deadline **2026-06-04**. Full spec: [`refs/project-instructions.md`](refs/project-instructions.md).

The deliverable is a **metamodel**: given a provided primary trading signal `s ∈ {-1, 0, +1}` for 11 futures, predict for each non-zero signal the **probability in `[0,1]` that following it is profitable** under a **triple-barrier** exit. The required output is a CSV `date,instrument,prediction`.

> **Grading is METHODOLOGY, not performance.** A high mark is achievable even if the metamodel never beats the primary signal. Optimize for rigour, leakage-discipline, and clear analysis — never for a flashy backtest number.

**Marking scheme** (drives where effort goes):

| Section | Marks | Where the work lives |
|---|---|---|
| Feature engineering | 20 | `src/stml/metamodel/` — **DONE** |
| Labeling (triple-barrier) | 20 | `refs/triple_barrier_guide.md`, `src/stml/harry/labels.py` |
| Model development & comparison (≥3 families: linear / tree / NN) | 30 | the meta-model layer (see §3, §4, §5) |
| Feature importance (cluster-level: MDI/MDA/SHAP) | 10 | redundancy map + importance module |
| Model evaluation (precision/recall/F1/AUC, per-instrument, vs blind baseline) | 20 | evaluate module |
| Optional strategy / position-sizing (competition) | +10 bonus | not started |

**Universe (11 instruments, 3 classes):** Equity `es1s nq1s fesx1s` · Energy `cl1s ho1s rb1s ng1s` · Metals `gc1s si1s hg1s pl1s`. Covering ≥1 full asset class is required; more is optional.

**Dates that matter:**
- Released signals: Jan 2020 → **2022-06-30**. The final 6 months (**Jul–Dec 2022**) are a **hidden test set** — not in our data; our code is *re-run* on it for the final mark.
- Deliverable CSV covers **Jan–Jun 2022** (the clean OOS holdout we carve from the training period).
- **FE-train boundary = `2021-07-01`**: all *fitted* features (see §6) are frozen on data `≤ 2021-07-01`. The matrix `partition` column splits train (`2020-01-03`→`2021-07-01`) / val (→`2021-12-30`) / test (→`2022-06-30`).

---

## 2. Current stage & next priorities

**Branch:** `signal-deep-dive` (cut for the deep analysis; the shared model-free base is `feature-base` / merged to `main`).

### What is DONE and on this branch
- **Feature engineering (the 20-mark section).** `src/stml/metamodel/` builds `results/feature_matrix.parquet` — **4,984 non-zero-signal trade-days × 179 cols (175 feature columns)**, families **F1–F17 (no F14)**, leakage-proven, deterministic. Leakage totals **E=108 / TF=65 / LI=2**. **257 tests** across 24 files, ruff-clean. Per-column docs in [`reports/feature-catalog.md`](reports/feature-catalog.md).
- **Triple-barrier labels (reference implementation).** `src/stml/harry/labels.py` produces meta-labels (4,886 events, label-1 share ≈ 0.556) with the off-by-one entry fix, sample-uniqueness weights, and 19 tests. Write-up: [`reports/harry/02-labels.md`](reports/harry/02-labels.md).
- **Signal characterization.** The primary signal is **short-horizon mean-reversion / counter-trend** (10/11 instruments; `ng1s` is short-only). Cross-asset correlation is low (~0.09). See §7.

### What is NOT YET on this branch (the priority)
The **meta-model layer (30+10+20 marks: model / importance / evaluation)** is the next phase. A complete **reference first-pass implementation exists but was rebased off this branch** — it lives only on `backup/signal-deep-dive-prerebase` (commit `94dca2f`) under `src/stml/model/` + `tests/test_model_*.py`, plus a from-scratch training guide at [`refs/meta-model-guide.pdf`](refs/meta-model-guide.pdf). Treat both as the **baseline architecture to build on** (the guide's full methodology is distilled in §5; the reference code in §4). The current working tree has the *guides* in place but no `src/stml/model/` package yet.

**Typical next tasks:** (re)build `src/stml/model/` on this branch from the reference; run barrier search → train XGB/RF/MLP/VSN under purged CV → cluster importance → OOS evaluation → emit the deliverable CSV.

> No recorded model *results* exist yet (the first run was rebased out before any evaluation was committed). When you build the model, you are establishing the first measured baseline — record AUC/per-instrument numbers in `reports/` so future sessions inherit them.

---

## 3. Task → where to look (read these FIRST)

| If you're asked to… | Read first |
|---|---|
| **Build / extend the meta-model** | **§5** (the course-canonical end-to-end pipeline, distilled from [`refs/meta-model-guide.pdf`](refs/meta-model-guide.pdf)) + **§4** (the reference code on `backup/signal-deep-dive-prerebase:src/stml/model/` — start with its `__init__.py` for pipeline order) |
| **Understand / add features** | [`src/README.md`](src/README.md) (module map), `src/stml/metamodel/`, [`reports/feature-catalog.md`](reports/feature-catalog.md) (every column) |
| **Do triple-barrier labeling** | [`refs/triple_barrier_guide.md`](refs/triple_barrier_guide.md) (López de Prado, full code), `src/stml/harry/labels.py` + [`reports/harry/02-labels.md`](reports/harry/02-labels.md); the matrix-wired variant `model/labels.py` (on backup) consumes `side=f5_signal`, `sigma=f2_vol_20` |
| **Cross-validation & hyperparameter tuning** | **§5 Part 2** (CPCV + two-stage AUC→Sharpe selection); `model/cv.py` (`PurgedWalkForward`) + `model/optuna_objective.py` (on backup) + `results/instrument_scope.json` (`embargo_p90`); §6 leakage rules |
| **Cluster-level feature importance** | §5 Part 2 (noise-feature sanity check), `model/importance.py` (on backup: MDI / SHAP / permutation / VSN-gates) + `results/feature_redundancy.json` (clusters) |
| **Evaluate / build deliverable CSV** | **§5 Part 3** (beat the *blind* primary: dual confusion matrix + NAV, EV threshold `p*`), `model/evaluate.py` (on backup), [`refs/project-instructions.md`](refs/project-instructions.md) §5 + Deliverables |
| **Calibrate probabilities & size positions** (competition track) | **§5 Part 4** (Platt scaling on out-of-fold pairs; fixed + estimated sizing incl. SOPS) |
| **Neural end-to-end portfolio** (competition track) | **§5 Parts 5–6** (Sharpe-loss + volatility targeting; meta-signals as `d+2` channels); §10 Sessions 7–8 |
| **Load data / handle NAs** | [`README.md`](README.md) §4, [`reports/missing-data-report.md`](reports/missing-data-report.md), `stml.io` / `stml.na_checks` |
| **Understand the signal's nature** | §7 below, [`reports/harry/01-signal-direction.md`](reports/harry/01-signal-direction.md) |
| **Read the artifact schemas** | [`results/README.md`](results/README.md) (matrix, redundancy, scope, provenance) |
| **Find a worked example of a course technique** | §10 — the course programming-session notebook library (task→notebook routing + per-session index) |

---

## 4. The meta-model — reference baseline architecture

This is the architecture of the rebased-off first run (`backup/signal-deep-dive-prerebase:src/stml/model/`). Use it as the baseline to root new work; vary deliberately.

**Pipeline order** (from `model/__init__.py`):
`labels` → `dataset` → `cv` → `barrier_search` → `optuna_objective` (+ `trees` / `mlp` / `vsn`) → `importance` → `evaluate`.

- **`dataset.py`** — assembles model-ready `(X, y)` from `results/feature_matrix.parquet` (read-only; never refits a TF feature). Attaches each row's **`bar_pos`** (trading-bar index on the instrument's own calendar — the axis CV and the labeler count `h` along). Label interface: **`side = f5_signal`**, **`sigma = f2_vol_20`**. Drops redundant features (one representative per `feature_redundancy.json` cluster), one-hots nominal `f4_cluster_id` + instrument. Iterates **scopes**: `pooled` / `per_class` (EQ/EN/ME) / `per_instrument`. `Preprocessor` (median-impute + standardise) is fit on **purged-train rows only**.
- **`labels.py`** — `triple_barrier_labels(...)`: first-touch (not endpoint) over `h` **trading bars** on each instrument's own calendar; events without a full forward window are **dropped** (no peeking); `price_end` truncates the price series so tuning never touches test prices.
- **`cv.py`** — `PurgedWalkForward`: expanding (anchored) walk-forward, **purge width `h`** + **per-instrument embargo `embargo_p90`** (from `instrument_scope.json`), applied on each instrument's own bar axis; fold cuts on the global date axis (handles cross-sectional F9/F11 leakage). The official test partition is never seen here.
- **`barrier_search.py`** — tunes `(pt, sl, h)` on dev only. Screens a **class-balance floor**, scores by **mean purged-CV AUC of a fixed shallow XGB baseline** (measures *label* quality, not joint barrier+model overfit), and picks by a **robust-plateau** rule (a candidate + its grid neighbours), not the single peak. Returns #grid-points tried (for Sharpe/AUC deflation).
- **Models (uniform `fit` / `predict_proba` / `feature_names_` surface):**
  - `trees.py` — **XGBoost** (`scale_pos_weight=n_neg/n_pos` per fold; NaNs kept via native handling) and **RandomForest** (`class_weight='balanced_subsample'`; median-imputed). Both accept uniqueness `sample_weight`.
  - `mlp.py` — PyTorch BatchNorm+Dropout **MLP** (default hidden `(64,32)`), `BCEWithLogitsLoss` with positive-class weight, seeded/deterministic. A tabular NN family — **one of several co-equal neural options, not a privileged default** (see the sequence-NN note below).
  - `vsn.py` — **Variable Selection Network** (GRN + GLU + softmax selection layer) — gives native, model-internal feature importance via gate weights (a neural complement to SHAP/permutation).
  - **Sequence-NN families to build out — co-equal to MLP/VSN, not optional add-ons:** LSTM / Transformer (Session 7) and the **Temporal Fusion Transformer** (Session 8). The coursework spec lists *"Variable Selection Network **or** Sequential Neural Networks"* as its NN examples, so a strong submission compares a **tabular NN (MLP/VSN) and a sequence NN (LSTM/Transformer/TFT) on equal footing**. These are not in the rebased-off reference code yet — add them. Build each one a **per-instrument causal lookback window** from each row's feature history (no peeking), wrap them in the same `fit`/`predict_proba`/`feature_names_` surface, keep the purged-CV + uniqueness-weight discipline (§6), and register them in `MODEL_REGISTRY` alongside the rest.
- **`optuna_objective.py`** — `cross_val_auc` is the single scoring primitive (mean ± std purged-CV **AUC**, never accuracy; degenerate single-class folds skipped, not scored 0.5). `MODEL_REGISTRY = {xgb, rf, mlp, vsn}` in the reference — **extend it with the sequence families (`lstm`/`transformer`/`tft`) as equal entries**; seeded TPE sampler.
- **`importance.py`** — trees: MDI / SHAP / permutation; NNs: permutation / gradient-saliency / VSN gates. Permutation computed on held-out blocks (generalisation, not train fit).
- **`evaluate.py`** — headline **ROC-AUC** + average-precision + Brier + F1/precision/recall@threshold; **`per_instrument_breakdown`** (thin names like `ho1s`/`ng1s` must be visible, not hidden in the pooled mean); **`release_test`** is the single test-opening tripwire (refuses without `final_confirmation=True`).

---

## 5. The course-canonical end-to-end pipeline (`refs/meta-model-guide.pdf`)

This is the **lecturer's (Madmoun) authoritative methodology** — the standard the graded deliverable is expected to follow. It is **richer than the §4 reference code** (which is one valid implementation of Parts 2–3); **where they differ, this guide is the methodology source.** The guide has six parts: Parts 2–3 are the graded core (model / importance / evaluation); Parts 4–6 are calibration+sizing and the neural-portfolio **competition (+10) track**.

**Part 1 — Framing.** Two-model architecture: the **primary** owns direction `s ∈ {−1,0,+1}`; the **meta-model** is a *binary* classifier `p̂ = Pr(y=1 | x)` answering "is this trade worth taking?" — far easier to learn than direction, and it outputs exactly the `[0,1]` probability the deliverable asks for. **Carve out the holdout (Jul–Dec 2022) before anything else**; CPCV, HP selection, calibration, and threshold-setting all happen on the training period only. Anything that touches the held-out period during development is leakage.

**Part 2 — Hyperparameter optimization with CPCV.**
- **CPCV (Combinatorial Purged Cross-Validation)** is the recommended validator. Standard k-fold leaks (triple-barrier labels span multiple days), and a single walk-forward path is too noisy to compare configs. CPCV = **purge** (drop train samples whose labels reach into the test window) + **embargo** (drop train samples just after each test block) + **combinatorial** (partition into N blocks, test on every k-subset). Worked example: **10 blocks, 2 test/model → C(10,2) = 45 models**, each block tested by 9 → **9 backtest paths per observation**.
  - ⚠️ **Divergence from §4 code:** the reference `model/cv.py` uses a *single-path* expanding `PurgedWalkForward` and explicitly calls CPCV "out of scope." **The guide recommends CPCV** — prefer it for the graded HP comparison (many paths → a robust, non-lucky config choice). Keep the purge = `h` + per-instrument `embargo_p90` discipline (§6) either way.
- **Two-stage HP selection** (recommended, not required): **Stage 1** — rank all configs by **mean AUC across the 45 CPCV models**, keep **top K = 3**. **Stage 2** — among the survivors, plot each one's **Sharpe distribution across the 45 paths** and pick the **highest median + tightest spread**. (AUC narrows statistically; Sharpe breaks ties economically.)
- **Feature importance** (the 10-mark section) is a **required sanity check**: economically-motivated features (regime / volatility state) should rank near the top, and **deliberately injected noise features should sink to the bottom**. Do it at the **cluster level** (§10 Session 2; pair with `feature_redundancy.json`).

**Part 3 — The meta-model.**
- **Label** with triple-barrier (see the labeling section): `y=1` if the **profit barrier is hit first** (long→upper or short→lower), `y=0` if the loss barrier first; time-barrier exit is labeled by the sign of its return. Barriers scale with σ so labels are comparable across regimes.
- **Threshold = filter by "adding zeros."** Take the trade only when expected value is positive: `p·rG + (1−p)·rL > 0` ⇒ **`p* = −rL/(rG−rL)`** (= `L/(G+L)` writing `G=rG>0`, `L=−rL>0`). Estimate `rG` (mean TP return) and `rL` (mean FP return) by **bootstrapping training-fold trade returns** (the threshold inherits the bootstrap uncertainty). Below `p*` set the position to 0; otherwise take it at full size. This *gates* direction; it does **not** size (sizing is Part 4 and needs calibration).
- **Evaluation — ask the right question:** not "is my meta-model accurate?" but **"does it beat following the primary *blindly*?"** Compare two confusion matrices on **out-of-fold** predictions: *primary alone* (always "trade" → recall = 1, precision = base rate) vs *primary + meta filter* (`p̂ ≥ p*`). Track **false positives avoided** (gain) vs **true positives missed** (cost); the net should be a **precision gain with a meaningfully higher hit rate**. Then go classification → P&L: compare **NAV / Sharpe / max drawdown** of the two strategies (flat "sit-out" stretches in the meta NAV are *correct* regime-distrust decisions). The result that matters is that the **precision gain holds out-of-sample**.

**Part 4 — Calibration & position sizing** (prerequisite for sizing; the +10 competition track).
- **Calibration:** to *size* on `p̂` it must mean what it says (`Pr(y=1 | p̂) ≈ p̂`). Defaults aren't calibrated: **RandomForest is under-confident** (pulls toward 0.5), **XGBoost/LightGBM over-confident** (push toward {0,1}), **logistic regression usually fine**. Fix with **Platt scaling** — a 1-D logistic `f(p̂)=σ(a·p̂+b)`, two params, fit **on out-of-fold CPCV pairs `(p̂ᵢ,yᵢ)`** (fitting on in-sample predictions learns "already calibrated" and is wrong live). **Isotonic regression** is the more flexible but data-hungry alternative; Platt is the safer small-sample default.
- **Sizing** `b = g(p̂)`, each returning 0 when `p̂ ≤ 0.5`:
  - *Fixed (no training):* **ModelConfidence** `b=p̂·1{p̂>0.5}`; **AllOrNothing** `b=1{p̂>0.5}` (the binary gate); **NCDF** standardize `z=(p̂−½)/√(p̂(1−p̂))` then `b=Φ(z)`.
  - *Estimated (fit on training `{p̂tr, rtr}`):* **LinearScaling** (stretch realized `p̂` range to [0,1]); **ECDF** (size by the training percentile of `p̂`); **SOPS** (Sharpe-Optimal Position Sizing — fit logistic `f_{a,c}(p)=(1+e^{−(ap−c)})⁻¹`, pick `(a,c)` maximizing `mean/std` of `f_{a,c}(p̂tr)·rtr`, i.e. training Sharpe directly).

**Part 5 — Neural portfolio construction (end-to-end Sharpe).** The Oxford benchmark (Saly-Kaufmann, Wood, Calliess, Zohren). A **feature-agnostic** recipe mapping per-asset feature windows → positions, trained by **maximizing the Sharpe ratio directly** (no supervised target):
1. Build **causal** features `x_{t,k}` and stack a **lookback window** `X_{t,k} ∈ R^{L×d}`.
2. Estimate **ex-ante EWMA volatility** `σ_{t,k}` (information ≤ t only).
3. **Sequence model** `g_ϕ` (**LSTM / PatchTST / Mamba2 / TFT**) → temporal state `h`; **projection head** `ŷ = tanh(wᵀh + b) ∈ [−1,1]` (+ a **ticker embedding** so one shared model specializes per instrument; only `g_ϕ` varies across the benchmark, keeping the comparison fair).
4. **Volatility targeting:** `w_{t,k} = ŷ_{t,k} · σ_tgt / σ_{t,k}` (σ_tgt ≈ 10%; scales low-vol assets up, high-vol down so each contributes comparable risk).
5. **Portfolio return:** `R^port_{t+1} = (1/K) Σ_k w_{t,k} · r_{t+1,k}`.
6. **Loss = negative differentiable annualized Sharpe:** `L = −(Ê[R]/√(Var[R]+ε))·√252`; Adam + gradient clipping, early-stop on validation Sharpe (patience 20). One forward pass produces a return for every day in the window; the Sharpe is computed once on the pooled path so gradients flow from the whole path into `fθ`. **TFT** (recurrent encoder + interpretable multi-head attention + variable-selection networks) is a top performer and is course-covered (§10 Session 8).

**Part 6 — Bridge: meta-signals as portfolio features.** Feed Part 3's outputs into Part 5's net. The per-step input becomes `x_{t,k} = [technical/statistical features (d channels), s_{t,k} (primary side), p̂_{t,k} (meta prob)] ∈ R^{d+2}` — kept as **two separate channels** (side carries direction; `p̂` is direction-agnostic), so the network learns *how trust modulates direction* rather than a fixed `s·p̂` product. **Use the out-of-fold CPCV `p̂`** (in-sample probabilities inject lookahead). Reuse the EWMA vol (span ≈ 60), volatility targeting, and Sharpe loss from Part 5; everything downstream is unchanged, just reading `d+2` features.

> **How this maps to our build:** Parts 2–3 are the graded core — start from the §4 code, but **upgrade its CV toward CPCV** and add the **EV-threshold (`p*`)** and the **blind-primary confusion/NAV comparison**. Parts 4–6 (calibration → sizing → neural end-to-end portfolio) are the **+10 competition track** and the strongest home for the **co-equal sequence-NN families** (§10 Sessions 7–8). Honor the same leakage discipline throughout (§6): holdout first, out-of-fold probabilities everywhere, ex-ante volatility only.

---

## 6. Leakage discipline (non-negotiable)

Every feature uses only information `≤ t`. Two classes (proven in tests):
- **E (engineered)** — no fit; causal by **truncation-invariance** (value at `t` identical on `data[:t+1]` and `data[:T]`).
- **TF (fitted)** — F3/F4/F11/F16/F17. Fit on the **FE-train partition only (≤ `2021-07-01`)** and applied causally with frozen parameters. `feature_matrix_provenance.json` records each fitted artifact's train window.
- **LI (label-interface)** — `f2_vol_20`, `f5_trailing_run_length`: engineered+causal columns earmarked for the downstream label.

When modeling, preserve this discipline:
1. **Purge + embargo, per-instrument.** Triple-barrier labels overlap in time; plain k-fold (even `TimeSeriesSplit`) leaks across the `[t, t+h]` window and inflates AUC. Use `h` purge + `embargo_p90` embargo on each instrument's bar axis.
2. **Fit preprocessing on train rows only** (never on val/test).
3. **Pass sample-uniqueness weights** as `sample_weight` everywhere (overlapping labels are not iid; effective N is ~13% of raw — AFML Ch. 4).
4. **Open the test partition exactly once**, through the `release_test` tripwire. The hidden Jul–Dec 2022 set is never in our data.
5. **Optimize AUC / an economically meaningful metric, never accuracy** (a 95%-one-class labeler scores high accuracy and is useless).

---

## 7. Project-specific lessons & gotchas

**Signal nature (informs feature/label/model design):**
- The primary signal is **short-horizon mean-reversion / counter-trend** — confirmed on 10/11 instruments. **Counter-trend F1 features (`f1_*`) are expected to dominate** the meta-model; sanity-check importance against this.
- `corr(s_t, r_{t+1}) > 0` for all 11, `corr(s_t, r_t) < 0` for 10/11 → the signal predicts the **next** bar. **Enter at `t+1`, not `t`** (the load-bearing label fix; entering at `t` is lookahead). See `reports/harry/02-labels.md`.
- Cross-asset correlation is low (~0.09) — instruments are near-independent.
- `cl1s` is the strongest single instrument (label-1 ≈ 0.70). `ng1s` is **short-only** (every non-zero signal is a short). `ho1s` has only ~63 events (statistically weak). **`low_power = {cl1s, ho1s, ng1s}`** — report these per-instrument, don't bury them.

**Traps that have already bitten this project:**
- **`f2_vol_20` is ANNUALISED.** As a triple-barrier sigma, **de-annualise by `√252` first**, or labels degenerate to all-timeouts. (Repeated in `reports/README.md` and the main README.)
- **Naive instrument-pooling inflates metrics** via cross-instrument base-rate matching (a momentum archetype scored a fake +0.14 kappa pooled while per-member it was ≈ chance, with `cl1s`/`ho1s` negative). **Aggregate within-instrument (per-member-averaged).**
- **First-touch ordering matters.** Don't compare path max/min independently — a position that ends +2% but dipped through the stop first is a `0`. Scan bars in order, take the earliest of {PT, SL, vertical}.
- **Momentum / breakout / trend archetypes fail** (wrong-signed) on this signal; only mean-reversion-flavored constructions replicate. Don't bake a trend bias into labels — keep the triple-barrier target neutral (symmetric `pt=sl=1` default) so importance analysis can *honestly* tell you whether counter-trend features matter.

**Reference / scratch locations:**
- Meta-model reference code: `git show backup/signal-deep-dive-prerebase:src/stml/model/<file>.py`.
- `ignore/` and `.omc/` are git-ignored scratch areas (not tracked).

---

## 8. Commands

Environment is **`uv`** (Python 3.12). `uv run <cmd>` always uses `.venv/` — never manually activate.

```bash
uv sync                          # install pinned deps (first time)
uv sync --group features-extra   # REQUIRED to build the full matrix (pywavelets F13 + hmmlearn F17)
uv run nbstripout --install      # one-time per clone: strip notebook outputs at commit

# Build / regenerate
uv run python -m stml.metamodel.build_features                    # full 11-instrument matrix + all artifacts
uv run python -m stml.metamodel.build_features --instruments si1s # fast smoke subset
uv run python -m stml.na_checks                                   # regenerate data/meta/*.csv missing-data diagnostics

# Tests & lint
uv run pytest                                   # full suite (257 tests)
uv run pytest tests/test_features_leakage.py    # one file
uv run pytest tests/harry/test_labels.py -k off_by_one   # one test by name
uv run ruff check src/ tests/                   # lint

# Notebooks
uv run jupyter lab
```

**Load data the sanctioned way** (auto-finds `data/` from any depth):
```python
from stml.io import load_clean_data            # NA-handled OHLCV (default for modeling)
from stml.metamodel import FeaturePipeline
ohlcv, signals = load_clean_data()
matrix = FeaturePipeline().fit(ohlcv, signals).transform(ohlcv, signals)
```
Never re-read the raw CSVs by hand; never `ffill`/`fillna(0)` structural NaNs (a closed venue is information — see [`reports/missing-data-report.md`](reports/missing-data-report.md)).

---

## 9. Repository layout & collaboration

- **`src/stml/`** — installable package (editable). `io.py` (loaders), `na_checks.py` (cleaning/diagnostics), `metamodel/` (feature layer F1–F17), `harry/` (Harry's features + `labels.py`).
- **`data/`** — `ohlcv_data.csv`, `primary_signals.csv`, `additional_data.xlsx` (F11 macro input), `features/` (one CSV per family), `meta/` (generated). Raw inputs are read-only.
- **`results/`** — feature matrix + redundancy/scope/provenance JSON. **`reports/`** — `feature-catalog.md`, `missing-data-report.md`, `harry/`. **`refs/`** — read-only course materials (the two guides live here).
- **Branching:** never commit to `main`; work on a personal/feature branch; PR at checkpoints. Notebooks live in `notebooks/<initials>/` (personal) — outputs are auto-stripped by nbstripout. Full workflow in [`README.md`](README.md) §6.

> When `pyproject.toml` / `uv.lock` change on a branch you pull, re-run `uv sync`. Never hand-edit `uv.lock` — regenerate with `uv lock`.

---

## 10. Course programming-session notebooks (reference library)

The 8 worked solution notebooks in [`refs/programming-session-sol/`](refs/programming-session-sol/) are the **canonical course implementations** of the techniques the rubric expects. They are read-only reference. The marking criteria explicitly reward "techniques covered in the course," so **cite the relevant session** in write-ups when you apply one.

> **Framework caveat.** These notebooks use **TensorFlow/Keras** for neural nets (Sessions 4, 6, 7, 8) and **sklearn / xgboost / statsmodels / hmmlearn** for the rest. **This project's code is PyTorch + sklearn + xgboost** (e.g. the VSN in `model/vsn.py` is a Keras→torch reimplementation of Session 6). Treat the NN notebooks as *architecture/algorithm* references and **port them to torch**; the sklearn/xgboost/statsmodels/hmmlearn code transfers directly.

### Task → notebook routing

| Project task | Go-to session(s) |
|---|---|
| **Full metamodel = label signal → filter with ML → importance → ROC/AUC** (closest analog to our whole deliverable) | **5** (primary), 4 |
| **Cluster-level feature importance** (10-mark section: MDI / MDA-PFI at cluster level, feature clustering) | **2** (primary), 5 |
| **Triple-barrier labeling** | [`refs/triple_barrier_guide.md`](refs/triple_barrier_guide.md) is primary; **1 & 5** for the trend-scanning *alternative* labeler (t-value trend labels) |
| **Tree models (XGBoost / RandomForest) + hyperparameter tuning** | **4**, 5 |
| **NN family — MLP / feed-forward** (tabular; BatchNorm, Dropout, early stopping) | **4**, 5 |
| **NN family — Variable Selection Network (VSN)** (interpretable gated NN) | **6** |
| **NN family — LSTM / Transformer** (sequence NN — *co-equal, not optional*) | **7** |
| **NN family — Temporal Fusion Transformer (TFT)** (sequence NN — *co-equal, not optional*) | **8** |
| **Latent / regime features (HMM)** — F3 / F17 | **3** (HMM), 2 (GMM/PCA/KMeans) |
| **Unsupervised structure (PCA / K-means / GMM) & feature relationships** — F4 | **2** |
| **Time-series CV, feature scaling, lag/rolling/cyclical features, regression metrics** | **4** |

### Per-session index

### Session 1 — Trend Scanning for Labeling Financial Time Series
**Summary:** Identify and label trends by fitting linear regressions over multiple look-forward horizons and labeling each point by the **sign of the most statistically significant slope (t-value)**. An alternative labeling scheme to triple-barrier.
**Key concepts:** linear-regression trend labeling, t-value significance, observation-horizon sensitivity, supervised label generation.
**Techniques & algorithms:** OLS trend fitting, t-value of slope, trend scanning across window sizes, sign-based labels.
**Key functions / APIs:** `statsmodels.api.OLS`, `np.sign`, custom `tValLinR()`, `trend_labels()`, `plot_trend_lines()`.
**Keywords:** trend scanning, t-statistic, lookahead horizon, price-trend classification, time-series labeling.
**Use when…:** considering or justifying a labeling scheme alternative to triple-barrier, or generating trend-strength labels over variable horizons.

### Session 2 — Unsupervised Learning, Clustering & Cluster-Level Feature Importance
**Summary:** Group related features (linear & non-linear) via a correlation distance matrix, reduce with PCA, cluster with K-means/GMM, pick cluster count by quality scores, then **aggregate MDI and permutation importance to the cluster level**. The direct template for the 10-mark cluster-importance section.
**Key concepts:** feature clustering, distance metrics, dimensionality reduction, cluster-quality selection, soft vs hard clustering, cluster-level importance decomposition.
**Techniques & algorithms:** Spearman-correlation distance, PCA, K-means, GMM, silhouette / Calinski-Harabasz / Davies-Bouldin, MDI, permutation feature importance (PFI), RandomForest.
**Key functions / APIs:** `scipy.stats.spearmanr`, `sklearn.decomposition.PCA`, `sklearn.cluster.KMeans`, `sklearn.mixture.GaussianMixture`, `silhouette_score`/`calinski_harabasz_score`/`davies_bouldin_score`, `sklearn.metrics.log_loss`, `KFold`; custom `OptimalClusterer`, `calculate_cluster_importance_mdi/_pfi`.
**Keywords:** cluster importance, MDI, PFI/MDA, Spearman distance, PCA, K-means, GMM, correlated feature groups.
**Use when…:** doing the cluster-level feature-importance analysis (pair with `results/feature_redundancy.json`), or selecting cluster counts for any unsupervised grouping.

### Session 3 — Discrete Hidden Markov Models
**Summary:** Foundations of HMMs via a narrative example — the three classic problems: **evaluation** (likelihood of observations), **learning** (EM/Baum-Welch parameter estimation), and **prediction** (future states/observations). Conceptual basis for the regime features.
**Key concepts:** hidden states, transition/emission matrices, filtering, log-likelihood, the label-switching problem, Rabiner framework.
**Techniques & algorithms:** Expectation-Maximization, Baum-Welch, forward filtering, state/observation prediction.
**Key functions / APIs:** `hmmlearn.CategoricalHMM` (`startprob_`, `transmat_`, `emissionprob_`, `predict_proba`, `score`), `np.random.choice`, `np.dot`.
**Keywords:** HMM, discrete HMM, regime switching, latent states, Baum-Welch, filtering.
**Use when…:** building or debugging HMM regime features (F3 Markov / F17 Gaussian-HMM), or explaining latent-state estimation in write-ups.

### Session 4 — Supervised Learning for Time-Series Forecasting (trees + NN pipeline)
**Summary:** End-to-end supervised pipeline: time-series feature engineering (lags, rolling stats, cyclical encoding), scaling, tree models with tuning, and feed-forward NNs with regularization — plus multi-metric evaluation and feature importance. The general template for the model-development section.
**Key concepts:** sequential train/test splitting, feature scaling, hyperparameter optimization, ensembles, NN regularization, model comparison.
**Techniques & algorithms:** XGBoost, RandomForest, feed-forward NN, BatchNorm, Dropout, `ReduceLROnPlateau`, early stopping, `TimeSeriesSplit`, `RandomizedSearchCV`.
**Key functions / APIs:** `XGBRegressor`, `RandomForestRegressor`, `StandardScaler`, `TimeSeriesSplit`, `RandomizedSearchCV`, Keras `Sequential`/`Dense`/`Dropout`/`BatchNormalization`/`Adam`/`EarlyStopping`, `mean_absolute_error`/`mean_squared_error`/`r2_score`. *(Regression demo; for our binary meta-label swap to classifiers + AUC.)*
**Keywords:** XGBoost, RandomForest, neural network, lag features, rolling stats, cyclical encoding, hyperparameter tuning, TimeSeriesSplit.
**Use when…:** standing up the model-comparison harness (tree + NN), engineering lag/rolling/cyclical features, or choosing a CV/tuning scaffold (note: our project replaces plain `TimeSeriesSplit` with purged+embargoed CV — see §6).

### Session 5 — Metamodel Signal-Filtering (the closest analog to this coursework)
**Summary:** A **two-stage trading system**: label price moves with Trend Scanning, then train RF / XGBoost / NN **metamodels to filter a primary signal** on market-microstructure features, comparing models by ROC/AUC and analyzing feature importance. This is the single most on-point reference for our deliverable.
**Key concepts:** meta-labeling/signal filtering, multi-horizon trend labels, market microstructure, model-ensemble comparison, importance analysis.
**Techniques & algorithms:** Trend Scanning (t-value maximization), OLS, RandomForest (bagging), XGBoost (boosting, L1/L2), MLP (ReLU + early stopping), MDI, permutation importance.
**Key functions / APIs:** custom `tValLinR`/`trend_labels`, `RandomForestClassifier`, `xgboost.XGBClassifier`, `sklearn.neural_network.MLPClassifier`, `sklearn.inspection.permutation_importance`, `roc_curve`/`roc_auc_score`/`precision`/`recall`/`f1_score`, `train_test_split`.
**Keywords:** metamodel, signal filtering, ROC/AUC, feature importance, RF/XGB/NN comparison, microstructure features.
**Use when…:** structuring the overall metamodel workflow, comparing the three model families, or wiring up classification metrics/ROC analysis (upgrade its `train_test_split` to our purged CV and triple-barrier labels).

### Session 6 — Variable Selection Networks (VSN)
**Summary:** Builds a **VSN** from scratch — an interpretable NN that learns per-sample softmax feature weights — via custom layers (input transformation, **Gated Linear Units**, **Gated Residual Networks**) over mixed numeric/categorical inputs. This is the architecture our `model/vsn.py` reimplements in torch.
**Key concepts:** learnable feature selection/weighting, mixed-feature embeddings, gating, residual connections, NN interpretability.
**Techniques & algorithms:** GLU, GRN, softmax variable selection, embeddings for categoricals, dense projection for numerics, `tf.GradientTape` training.
**Key functions / APIs:** custom Keras layers `InputTransformation`/`GatedLinearUnits`/`GatedResidualNetwork`/`VariableSelectionNetwork`/`FinalModel`, `tf.data.Dataset`, `Embedding`/`Dense`/`LayerNormalization`/`Dropout`, `AUC`/`Precision`/`Recall`.
**Keywords:** VSN, GRN, GLU, gated network, feature weighting, neural interpretability, mixed features.
**Use when…:** implementing/understanding the VSN meta-model or extracting model-internal (gate-weight) feature importance — **port the Keras layers to torch** (`model/vsn.py` already does).

### Session 7 — Temporal Processing with RNNs & Transformers
**Summary:** Sequence-to-label classification for Long/Short/Neutral positions using **LSTM** and **Transformer** architectures (self/multi-head attention, positional encoding, look-ahead masking) in Keras. A **co-equal NN family** to the tabular MLP/VSN — the "Sequential Neural Networks" the coursework lists alongside VSN.
**Key concepts:** sequence classification, temporal dependencies, self-attention, multi-head attention, positional encoding, masking, regularization.
**Techniques & algorithms:** LSTM, scaled dot-product attention, multi-head attention, sinusoidal positional encoding, early stopping + LR scheduling.
**Key functions / APIs:** `tf.keras.layers.LSTM`/`Dense`/`BatchNormalization`/`Dropout`, `EarlyStopping`/`ReduceLROnPlateau`, custom `scaled_dot_product_attention`/`MultiHeadAttention`/`positional_encoding`/`create_look_ahead_mask`, `sklearn.metrics.classification_report`.
**Keywords:** LSTM, transformer, self-attention, sequence model, positional encoding, time-series classification.
**Use when…:** building an LSTM/Transformer meta-model — a **first-class NN family to compare head-to-head with MLP/VSN**. Feed it a per-instrument **causal lookback window** over each row's feature history (no peeking), keep the purged-CV + uniqueness-weight discipline (§6), and port the Keras code to torch. The panel is short/thin per instrument, so windowing choices (length, padding thin names) are a design task to address — not a reason to deprioritise.

### Session 8 — Temporal Fusion Transformer (TFT) for Volatility Forecasting
**Summary:** Multi-horizon forecasting with the **Temporal Fusion Transformer** via Nixtla `neuralforecast`, including quantile loss, robust scaling, rolling-window cross-validation, SMAPE/MAE/MASE evaluation, and attention/feature-importance interpretation. A **co-equal sequence-NN family** — TFT natively handles static + temporal covariates and yields built-in attention-based feature importance.
**Key concepts:** quantile-regression forecasting, realized volatility, multi-horizon prediction, attention interpretability, rolling-window CV.
**Techniques & algorithms:** TFT, quantile loss, robust scaling, forward-fill imputation, one-hot static covariates, log-transform.
**Key functions / APIs:** `neuralforecast.models.TFT`, `NeuralForecast.cross_validation`, `utilsforecast.evaluation.evaluate`, `smape`/`mae`/`mse`/`mase`, model `attention_weights()`/`feature_importances()`.
**Keywords:** TFT, NeuralForecast (Nixtla), quantile regression, multi-step forecasting, realized volatility, attention importance.
**Use when…:** building a TFT meta-model as a **first-class sequence-NN family** (adapt its quantile head to a single binary-probability output), using its native attention/feature-importance for the importance section, or for the optional competition/volatility track. Note: `neuralforecast` / `utilsforecast` aren't in `pyproject.toml` yet — `uv add` them if you take this route.
