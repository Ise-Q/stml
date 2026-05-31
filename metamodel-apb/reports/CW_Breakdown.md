# T3.03 Coursework — Detailed Breakdown v2

**Project:** Alken Asset Management metamodel challenge — group of 5, marked /100, +10 bonus.
**This version adds:** (i) a method evaluation across all eight programming sessions with a "best method" verdict per problem area; (ii) the eight methodological commitments from the literature review, mapped onto the brief; (iii) full 11-instrument coverage guidance; (iv) the bonus/strategy track; (v) validation options with pros and cons.

Sources now in hand: the brief (`T3_03_CW_Brief.docx`, identical content to the `.md`); the literature review `nlr-cw-v1.md` (60 references, 8 commitments); all PS1–PS8 solutions + PS2_O/PS4_O optional notebooks; the nine project files; and the `sts-ml` skill (references + runnable `scripts/`).

---

## Part A — Method evaluation across the programming sessions

The brief says explicitly: *"For implementation guidance, refer to all programming sessions"* (`T3_03_CW_Brief.docx`, Getting Started). So the right way to choose methods is to evaluate what each PS actually implements and pick the best-fit per task. I read the **solutions** notebooks (the `_O` files are optional deep-dives). Citations below are `notebook : cell` (cells are 0-indexed as they load).

### A.0 What each notebook actually contains (inventory)

| PS | Topic | Labels | Models | CV / leakage handling | Importance | Reusable for CW? |
| ---- | ------- | -------- | -------- | ---------------------- | ------------ | ------------------ |
| PS1 | Trend scanning | trend-scan (`tValLinR`,`trend_labels`) | — | — | — | As a **feature** only (Sec 1) |
| PS2 | Clustering & importance | — | RF | none (in-sample) | **per-feature + cluster MDI/PFI** | **Yes — Sec 4 core** (`PS2_Solutions:41–66`) |
| PS3 | Discrete HMM | — | `CategoricalHMM` | — | — | As a **regime feature** (Sec 1) |
| PS4 | Supervised TS forecast | regression target | XGB, RF, Keras NN | **`TimeSeriesSplit`+`RandomizedSearchCV`, scaler on train, shifted feats** | MDI + permutation | **Yes — leakage gold-standard** (`PS4_Solutions:2,13,15,24`) |
| PS5 | Crypto **metamodel** | trend-scan | RF, XGB, MLP (dict) | plain `GridSearchCV` (no time-awareness) | MDI(trees)+perm(NN), train&test | **Yes — pipeline template** (`PS5_Solutions:39–73`) |
| PS6 | Variable Selection Network | synthetic | trees, NN, **VSN** (GLU/GRN) | none | VSN selection weights | **Yes — NN family option** (Sec 3) |
| PS7 | RNN / Transformer | synthetic seq | LSTM, Transformer from scratch | — | attention weights | Optional (sequential NN) |
| PS8 | Temporal Fusion Transformer | realized vol | TFT (`neuralforecast`) | built-in | TFT var-selection + attention | Optional (heavy; vol-forecasting) |

### A.1 Labelling — which method is "best" for THIS coursework

- **PS1/PS5 implement trend scanning.** It is elegant (pick the horizon that maximises |t-stat| of the slope, label by its sign — `PS5_Solutions:21–22`) and last year's coursework used it.
- **But the brief mandates the triple-barrier method** (`T3_03_CW_Brief.docx`, §2) and marks the *justification of barrier widths and time-limit* (20 marks). **No notebook implements triple-barrier.** The lit review independently lands on triple-barrier and explains *why* it beats trend scanning here: the trend-scanning alternative "has not been independently validated on multi-asset futures universes" and adds a horizon-selection hyperparameter that "would inflate the multiple-testing burden" (`nlr-cw-v1.md` §1).
- **Verdict:** **Triple-barrier with volatility-adaptive symmetric barriers** (±k·σ̂ₜ, vertical horizon T_max). σ̂ₜ from an EWMA or an OHLC estimator; the lit review's specific recommendation is **Garman–Klass** volatility for energy/rates (overnight-gap aware) with Parkinson as a robustness check (`nlr-cw-v1.md` §A1, §3). This must be **written** — the L1 slides give the definition (`T3_03_L1_Slides.md` L155–175), and trend scanning from PS1/PS5 is *repurposed as a feature*, not the label.
- **Two refinements the lit review justifies and the notebooks lack:** (a) **sample-uniqueness weighting** — triple-barrier labels overlap in time, so they're not iid; downweight concurrent labels (`nlr-cw-v1.md` §1, citing López de Prado 2018 Ch.4); (b) **class weighting** — the positive-class share for trend-following primaries runs ~30–40%, enough to warrant `class_weight`/threshold tuning, not resampling (`nlr-cw-v1.md` §1).

### A.2 Models — which family wins, and which notebook spec to copy

- **PS5 is the template:** three families in a dict — RandomForest (`criterion='entropy'`, depth-limited), XGBoost (L1/L2 + subsampling), `MLPClassifier` (early stopping) — then evaluate in a loop (`PS5_Solutions:39,43,47,49`). These exact anti-overfit settings are the ones to reuse.
- **The brief's required families** are linear / tree / neural (`T3_03_CW_Brief.docx`, §3). PS5 covers tree + neural; you must **add a regularised logistic regression** for the linear slot.
- **Empirical "best model" expectation:** the horse-race literature is consistent — **gradient-boosted trees** (XGBoost/LightGBM) match or beat both linear and deep nets on tabular financial data, at modest margins (`nlr-cw-v1.md` §2, citing Gu-Kelly-Xiu 2020, Krauss-Do-Huck 2017). The lit review's chosen trio is **elastic-net logistic / LightGBM / a VSN-only neural net** deliberately kept small for sample-efficiency (`nlr-cw-v1.md` §2). LightGBM isn't in any notebook but is a drop-in for the XGBoost slot.
- **Neural option:** the cheap, course-supported route is the PS5 `MLPClassifier`; the higher-mark route is the **VSN** from PS6 (`scripts/vsn.py`), which the brief names explicitly and which yields feature-selection weights for free. The Transformer (PS7) and TFT (PS8) are over-powered for a binary act/skip metamodel and the lit review warns against importing CV-scale architectures into a small-data problem (`nlr-cw-v1.md` §2, Israel-Kelly-Moskowitz 2020).
- **Verdict:** linear = elastic-net logistic; tree = **XGBoost or LightGBM (likely the winner)**; neural = MLP first, VSN if time. Tune all three with the **PS4** machinery, not the PS5 one (next point).

### A.3 Tuning & leakage — PS4 beats PS5

- **PS5's tuning leaks:** it uses plain `GridSearchCV` with no time-aware folds (confirmed: no `TimeSeriesSplit` in PS5). On autocorrelated, overlapping-label data this overstates performance.
- **PS4 is the gold standard:** seeds fixed across `random`/`numpy`/`tf`/`PYTHONHASHSEED` (`PS4_Solutions:2`), `featurize` shifts every lag/rolling feature (`PS4_Solutions:13`), scaler fit on train only (`PS4_Solutions:15`), and tuning via **`TimeSeriesSplit` + `RandomizedSearchCV`** (`PS4_Solutions:24`).
- **One bug to fix:** PS4's RF grid contains `max_features=['auto', ...]` (`PS4_Solutions:24`), removed in scikit-learn ≥1.3 — use `'sqrt'`/`'log2'` (`SKILL.md` L170–174).
- **Verdict:** adopt PS4's preprocessing + `TimeSeriesSplit`/`RandomizedSearchCV`, but go further to **purged CV + embargo (and ideally CPCV)** for the metamodel — see Part D, the lit review's strongest single methodological theme (`nlr-cw-v1.md` §6).

### A.4 Cluster-level feature importance — PS2 is the only source

- **PS2 implements the whole pipeline:** Spearman distance `1−|ρ|` (`PS2_Solutions:41`) → PCA → silhouette-K → K-means → GMM soft membership (`OptimalClusterer`, `PS2_Solutions:47–55`) → **cluster MDI** (sum per-feature tree importances within a cluster, `PS2_Solutions:60`) and **cluster PFI** (permute a whole cluster with one shared index, `PS2_Solutions:62`). This is mirrored verbatim in `scripts/cluster_feature_importance.py`.
- **PS5's importance is per-feature only** (MDI for trees, permutation for NN — `PS5_Solutions:55,64`); it does **not** cluster. The brief's Section 4 requires cluster level, so **PS2 is binding**, not PS5.
- **One leakage caveat to fix:** PS2's cluster-PFI uses `KFold(shuffle=True)` (confirmed) — fine for the synthetic iid demo, wrong for time series. Swap in a purged/time-aware split for the coursework (Part D).
- **The brief allows MDI / MDA / SHAP** (`T3_03_CW_Brief.docx`, §4). **No notebook uses SHAP** (the "shap" string in the notebooks is only `.shape`). The lit review flags that **cluster-level SHAP with correlation-based clustering has no peer-reviewed antecedent in finance** — so doing it is a genuine, write-up-worthy contribution, not a replication (`nlr-cw-v1.md` §5). TreeExplainer makes it cheap on trees (`nlr-cw-v1.md` §5, Lundberg et al. 2020).
- **Verdict:** PS2 pipeline as the spine; cluster MDI + cluster MDA (permutation) as the robust baseline; **cluster SHAP as the differentiator**. Distance metric √(1−|ρ|) is the lit-review/Mantegna-1999 standard (`nlr-cw-v1.md` §5).

### A.5 Regime / latent features — PS3 (HMM) and PS2 (GMM)

- **GMM** soft-membership probabilities (`PS2_Solutions:55`) and **discrete HMM** via `hmmlearn.CategoricalHMM` (`PS3_Solutions`) both give regime features. HMM is temporally aware (Markov chain on hidden state) where GMM treats rows as iid — so HMM is the better *regime* feature for sequential data (`T3_03_L3_Slides.md` L40–48).
- The lit review backs a **two-state Gaussian HMM with time-varying (EWMA-updated) parameters** (`nlr-cw-v1.md` §4, Nystrup-Madsen-Lindström 2017), and notes using HMM regime-probabilities as downstream features is itself near-novel (`nlr-cw-v1.md` §4). Caveat to state honestly: regime models lag at turning points (`nlr-cw-v1.md` §4).

---

## Part B — Section-by-section breakdown (brief order)

Each section: brief requirement → best method (from Part A) → action items with citations. The cross-cutting checklist (seeds, leakage, DataFrame hygiene) from v1 still applies and isn't repeated.

### Section 1 — Feature Engineering (20 marks)

**Brief:** technical indicators; latent-variable (GMM/HMM); other unsupervised; anything justifiable; document each (`T3_03_CW_Brief.docx`, §1).
**Best methods:** PS4 `featurize` discipline + PS1 backward trend feature + PS2 GMM + PS3 HMM, plus the futures-specific features the lit review motivates.

- [ ] **1.1** Technical block per instrument: multi-horizon returns, realised vol, momentum, MA crossovers, RSI, volume/open-interest changes — all shifted (`PS4_Solutions:13`; `SKILL.md` L124–138).
- [ ] **1.2** **Futures-specific features the lit review justifies** (these lift the "creativity" mark): log **calendar/basis spread** (front-vs-second), **open-interest growth**, and for energy **scheduled-release/overnight-gap** vol; for metals, gold↔real-rates/USD and copper↔China-demand proxies; for equity indices, **VIX level + term-structure slope + variance-risk-premium** (`nlr-cw-v1.md` §3). Note: some require data beyond OHLCV — include where derivable, flag where not.
- [ ] **1.3** Backward **trend-strength** feature via `trend_labels(look_forward=False)` (`scripts/trend_scanning.py`; `T3_03_L1_Slides.md` L201–204).
- [ ] **1.4** **GMM** regime-probability features (`PS2_Solutions:55`).
- [ ] **1.5** **HMM** regime-probability features, two-state Gaussian, periodic re-estimation (`PS3_Solutions`; `nlr-cw-v1.md` §4).
- [ ] **1.6** One-line "what it captures" comment per feature (`T3_03_CW_Brief.docx`, §1).

### Section 2 — Labelling: Triple-Barrier (20 marks)

**Brief:** triple-barrier as taught; justify widths + time-limit (`T3_03_CW_Brief.docx`, §2).
**Best method:** volatility-adaptive symmetric triple-barrier — **must be written** (Part A.1).

- [ ] **2.1** Label only dates where the **provided primary signal ≠ 0**: simulate the trade forward, label T/¬T by first barrier touched (`T3_03_L1_Slides.md` L295–303).
- [ ] **2.2** Barrier widths = ±k·σ̂ₜ with σ̂ₜ from **Garman–Klass** (energy/rates) / EWMA, justified in prose (`nlr-cw-v1.md` §A1, §3; `T3_03_L1_Slides.md` L169–171).
- [ ] **2.3** Set + justify vertical barrier T_max (`T3_03_L1_Slides.md` L161).
- [ ] **2.4** Record `t1` (first-touch time) for every label — needed for purging (Section 5 / Part D).
- [ ] **2.5** Compute **sample-uniqueness weights** from label concurrency (`nlr-cw-v1.md` §1) and note per-instrument class balance for metric choice.

### Section 3 — Model Development & Comparison (30 marks)

**Brief:** ≥3 tuned models, one per family; clear comparison with reasoning (`T3_03_CW_Brief.docx`, §3).
**Best methods:** elastic-net logistic + XGBoost/LightGBM + MLP→VSN; tuned with PS4 machinery (Part A.2–A.3).

- [ ] **3.1** Linear: `LogisticRegression` (elastic-net) in a `Pipeline` with train-fit scaler.
- [ ] **3.2** Tree: XGBoost with PS5 anti-overfit settings (`PS5_Solutions:43`); optionally LightGBM (lit-review pick, `nlr-cw-v1.md` §2).
- [ ] **3.3** Neural: MLP (`PS5_Solutions:47`) first; VSN (`scripts/vsn.py`, PS6) for the higher mark.
- [ ] **3.4** Tune each with **`TimeSeriesSplit`+`RandomizedSearchCV`** (`PS4_Solutions:24`; fix `max_features`), `scoring='neg_log_loss'` or AUC; apply **class weights**.
- [ ] **3.5** Pass **sample-uniqueness weights** into `.fit(... sample_weight=)` where supported (`nlr-cw-v1.md` §1).
- [ ] **3.6** Comparison table + ROC overlay; pick winner by **test AUC** *and* a calibration metric (Brier/ECE) — calibration matters because the deliverable is a probability and (if bonus) Kelly needs reliable p̂ (`nlr-cw-v1.md` §2). Tree models are usually well-calibrated; NNs may need Platt/isotonic (`nlr-cw-v1.md` §2).

### Section 4 — Cluster-Level Feature Importance (10 marks)

**Brief:** cluster correlated features; MDI/MDA/SHAP at cluster level; discuss the groups (`T3_03_CW_Brief.docx`, §4).
**Best method:** PS2 pipeline + cluster SHAP differentiator (Part A.4).

- [ ] **4.1** Spearman distance `1−|ρ|` (`PS2_Solutions:41`).
- [ ] **4.2** `OptimalClusterer` PCA→silhouette-K→K-means→GMM (`PS2_Solutions:47–55`).
- [ ] **4.3** Cluster **MDI** (`PS2_Solutions:60`).
- [ ] **4.4** Cluster **MDA/PFI** — same-permutation shuffle — but **replace shuffled KFold with a purged/time-aware split** (`PS2_Solutions:62`; Part D).
- [ ] **4.5** Cluster **SHAP** via TreeExplainer, aggregated per cluster — flag as a contribution (`nlr-cw-v1.md` §5).
- [ ] **4.6** Discuss which feature *groups* drive the metamodel; confirm noise clusters ≈ 0 (`T3_03_CW_Brief.docx`, §4; `PS2_Solutions` cluster interpretation).

### Section 5 — Model Evaluation (20 marks)

**Brief:** precision/recall/F1/AUC; confusion matrix + threshold analysis; per-instrument breakdown; baseline = follow primary blindly (`T3_03_CW_Brief.docx`, §5).
**Best method:** PS5 `evaluate_model` (train+test metric set) on a purged OOS split (Part A.3, Part D).

- [ ] **5.1** PS5 `evaluate_model` for every model (`PS5_Solutions:72`).
- [ ] **5.2** Confusion matrix + threshold sweep (precision↔recall trade-off; `T3_03_L1_Slides.md` L321–329).
- [ ] **5.3** **Per-instrument** breakdown — say where it helps and where it doesn't (`T3_03_CW_Brief.docx`, §5).
- [ ] **5.4** Baseline comparison vs blind-primary (`T3_03_CW_Brief.docx`, §5); frame as precision-for-recall (`nlr-cw-v1.md` §1).
- [ ] **5.5** Apply purge+embargo between train and the internal OOS block using `t1` (Part D).
- [ ] **5.6** If you tuned across many configs, report a **deflated Sharpe / note the multiple-testing hurdle** even on classification metrics (`nlr-cw-v1.md` §6, Harvey-Liu-Zhu t>3).

### Section 6 — Bonus: Strategy Construction (+10)

**Brief:** size positions from metamodel probabilities; backtest CAGR / vol / Sharpe / Sortino / max drawdown / avg holding period / turnover; constraints released 20 May (`T3_03_CW_Brief.docx`, Optional track).
**Best method:** fractional-Kelly sizing + vol targeting (lit review §7; no notebook implements this).

- [ ] **6.1** Combine signal as `primary × take` → Long/Short/Neutral (`references/L1_labeling_and_evaluation.md` L158–165), then **size by probability**.
- [ ] **6.2** **Fractional Kelly** f* = (p̂·b − (1−p̂)·d)/(b·d) with b/d the upper/lower barrier multipliers; baseline **κ = 0.25**, confidence floor **p̂ ≥ 0.55** (no position below) (`nlr-cw-v1.md` §7).
- [ ] **6.3** **Volatility-targeting** overlay (Garman–Klass realised vol input), per Carver's 25% annualised-vol convention (`nlr-cw-v1.md` §7).
- [ ] **6.4** Backtest metrics (`T3_03_CW_Brief.docx`, Optional). ⚠ **Not in any T3.03 notebook** — pull from `python-quant-finance` or `empirical-finance` (Q3).
- [ ] **6.5** Apply the **20 May constraints** doc (position limits, gross/net, rebalancing, target vol) — not in the files (Q2).
- [ ] **6.6** Emit `date,instrument,weight` CSV (`T3_03_CW_Brief.docx`, Deliverables).

---

## Part C — Covering all 11 instruments

The brief requires ≥1 full asset class; all 11 is optional and is the competition-grade scope (`T3_03_CW_Brief.docx`, The Universe). Universe: **Equity index** ES1S/NQ1S/FESX1S; **Energy** CL1S/HO1S/RB1S/NG1S; **Metals** GC1S/SI1S/HG1S/PL1S.

**Recommended architecture — one metamodel per asset class, instrument identity as a feature.** The lit review argues this directly: per-instrument models starve for data; one pooled model mixes heterogeneous data-generating processes; **per-asset-class is the standard middle ground**, and term-structure/carry features "differ qualitatively across classes" (`nlr-cw-v1.md` §1, citing Hurst-Ooi-Pedersen 2017, Moskowitz-Ooi-Pedersen 2012). So: **3 metamodels** (Equity, Energy, Metals), each trained on its instruments pooled with an instrument-id feature, returns standardised by ex-ante vol before pooling.

**Practical sequencing for 11 instruments:**

- [ ] **C.1** Build the pipeline **end-to-end on one class first (Metals or Energy, 4 instruments)** — cleaner than equities, which start late (ES1S 1997, FESX1S 1998, NQ1S 1999; `T3_03_CW_Brief.docx`, Dataset).
- [ ] **C.2** Parameterise everything by instrument/class so extension is a loop, not a rewrite. Keep features in labelled DataFrames (`SKILL.md` L150–157).
- [ ] **C.3** Asset-class-specific features: Energy → Garman–Klass vol + storage/scheduled-release; Metals → basis + (gold: real-rates/USD; copper: China demand); Equity → VIX/VRP (`nlr-cw-v1.md` §3, §A1).
- [ ] **C.4** Per-instrument **diagnostics**: label balance, OOS metrics, calibration — the metamodel will help unevenly across the 11 and the brief wants that stated (`T3_03_CW_Brief.docx`, §5).
- [ ] **C.5** One deterministic CSV emitter over all covered instruments, exact format (`T3_03_CW_Brief.docx`, Deliverables); the grader re-runs on the hidden Jul–Dec 2022 half (`T3_03_CW_Brief.docx`, Dataset).

---

## Part D — Validation options (pros & cons)

This is the rigour the brief marks ("rigour of your labelling and validation protocol", Evaluation) and the lit review's single biggest theme: standard k-fold leaks on overlapping-label, autocorrelated financial data (`nlr-cw-v1.md` §6). The data ends **30 Jun 2022**; **Jul–Dec 2022 is the hidden test** — so all options below operate *within* the released data, and you must carve a clean internal OOS block.

| Option | What it is | Pros | Cons | Verdict for CW |
| --- | --- | --- | --- | --- |
| **Plain k-fold (shuffled)** | random folds | simple; max data per fold | **leaks**: future→past via overlapping labels + autocorrelation; inflates Sharpe/AUC (`nlr-cw-v1.md` §6, Schnaubelt 2022). This is what **PS2's cluster-PFI** uses (`PS2_Solutions:62`) | ❌ do not use for time series |
| **`TimeSeriesSplit` (forward-chained)** | expanding train, future val | no shuffle leakage; in sklearn; **used in PS4** (`PS4_Solutions:24`) | ignores label *overlap* at the train/val boundary; single path | ✅ minimum acceptable; good for tuning |
| **Purged k-fold + embargo** | drop train labels overlapping val `t1`; embargo ⌈0.01·T⌉ after val | removes the overlapping-label leak triple-barrier *creates*; the course's stated target (`T3_03_L1_Slides.md` L367–383) | must track `t1`; more code; still one path | ✅✅ **recommended baseline** (`nlr-cw-v1.md` §6) |
| **CPCV (combinatorial purged)** | N=6 groups, k=2 out → 15 paths | a **distribution** of OOS metrics, not one number; supports PBO/robustness; lit-review pick (`nlr-cw-v1.md` §6) | most code/compute; overkill if time-poor | ✅✅✅ best-in-class; do if competing |
| **Nested CPCV** | inner CPCV tunes, outer CPCV evaluates | removes hyperparameter-tuning leakage (`nlr-cw-v1.md` §6, Schnaubelt 2022) | heaviest | ⭐ only if going for the internship |
| **Single hold-out (e.g. Jan–Jun 2022 block)** | one contiguous OOS slice | mirrors the real hidden test; trivially clean if purged at the seam | small sample; one estimate; high variance | ✅ use **as the final reporting split**, on top of CV for selection |
| **Block / stationary bootstrap** | resample contiguous blocks | preserves serial dependence; good for **uncertainty bands** | for inference, not train/val separation (`nlr-cw-v1.md` §6, Künsch 1989 / Politis-Romano 1994) | ➕ optional, for confidence intervals |

**Recommended protocol:** **purged k-fold + embargo** for model selection (CPCV/nested-CPCV if competing), then a **single contiguous Jan–Jun 2022 hold-out** for final reporting — which also rehearses the grader's hidden-half re-run. Track `t1` from Section 2 so purging is possible. Embargo = 1% of sample is the published default; don't retune it (`nlr-cw-v1.md` §6).

---

## Part E — The eight methodological commitments (lit review → brief)

The lit review codifies eight choices with citations; here's where each lands and whether code exists.

| # | Commitment | Brief section | Code status |
| --- | --- | --- | --- |
| 1 | Meta-labelling (secondary act/skip model) | whole CW | template in PS5; **purged CV must be added** |
| 2 | Volatility-adaptive triple barriers | §2 | **write it** (no notebook) |
| 3 | Purged CV + embargo (+CPCV) | §5 / Part D | **write it** (PS4 only has `TimeSeriesSplit`) |
| 4 | Garman–Klass volatility | §1, §2, §6 | **write it** (no notebook) |
| 5 | 3-family horse-race (logistic / LightGBM / VSN) | §3 | PS5 (RF/XGB/MLP) + PS6 (VSN); add logistic/LightGBM |
| 6 | Cluster-level importance (+ cluster SHAP) | §4 | PS2 (MDI/PFI); **SHAP is the novel add** |
| 7 | Fractional-Kelly sizing + vol targeting | §6 bonus | **write it** (no notebook) |
| 8 | Regime features (2-state Gaussian HMM, TV params) | §1 | PS3 (`CategoricalHMM`); adapt to Gaussian + EWMA update |

**Net:** the notebooks give you the *pipeline shape* (PS5), the *leakage discipline* (PS4), the *cluster importance* (PS2), the *regime/NN building blocks* (PS3/PS6). The five things you must build from the slides + lit review are: **triple-barrier, purged/CPCV, Garman–Klass, cluster-SHAP, and Kelly sizing**. None of these is large; each has a clear published spec.

---

## Open questions (carried from v1, refined)

1. **Triple-barrier labeller — confirm I should write it.** No PS implements it; the brief mandates it; the L1 slides define it (`T3_03_L1_Slides.md` L155–175). Shall I write `triple_barrier.py` (volatility-adaptive, symmetric, with `t1` + uniqueness weights) matching course conventions?
2. **20 May constraints doc** for the bonus track — not in the uploads. Are you competing, and can you share it (`T3_03_CW_Brief.docx`, Optional)?
3. **Backtest metrics source** (CAGR/Sharpe/Sortino/drawdown/turnover) — absent from T3.03. OK to pull from `python-quant-finance`/`empirical-finance`?
4. **Scope & split:** (a) all 11, or one class first? (Recommend Metals/Energy first, then extend — Part C.) (b) Validation: purged k-fold for selection + Jan–Jun 2022 hold-out for reporting — agreed, or do you want CPCV/nested for the internship track?
5. **What do I build next?** (a) `triple_barrier.py` + purged-CV module; (b) end-to-end notebook skeleton wiring PS2/PS4/PS5 + the new modules across one asset class; (c) a written methodology section (academic style, Harvard, drawing on `nlr-cw-v1.md`'s references); (d) cluster-SHAP module. Tell me the order.
