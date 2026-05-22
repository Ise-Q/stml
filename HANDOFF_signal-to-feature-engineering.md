# HANDOFF: Signal Deep-Dive → Feature Engineering / Metamodel

**Audience:** a fresh agent session picking up the **graded coursework**.
**Status of prior phase:** complete, verified, committed (`signal-deep-dive` @ `4336699`).
**Date:** 2026-05-22. **Deadline:** 2026-06-04. **Branch you'll likely work on:** cut a new branch from `signal-deep-dive` (or `main` after it merges).

## 0. How to use this document
Read §1 (the real task) and §5 (what to do next) first. §2–§4 explain what the prior phase did and what it *means for your features/labels/eval*. The signal deep-dive was **not** the graded deliverable — it was a pre-feature-engineering investigation of the primary signal so the metamodel is built on understanding, not guesses. **Reuse the framework in `src/stml/replication/` and `src/stml/` — do not reimplement returns, splits, metrics, or NA handling.**

## 1. The main task (from `refs/project-instructions.md`)
Build a **metamodel** on top of the provided primary signal `s_t ∈ {-1,0,+1}` (11 futures, 3 classes). For each signal, predict the **probability in [0,1] that following it is profitable** under a **triple-barrier** exit. **Graded on METHODOLOGY, not performance** (a high mark is possible even if the metamodel doesn't beat the signal).

Marking scheme (100 + 10 bonus):
| Section | Marks | What it needs |
|---|--:|---|
| Feature Engineering | 20 | Rich feature set (technical indicators, GMM/HMM latent models, unsupervised methods); document what each captures |
| Triple-Barrier Labeling | 20 | Apply triple-barrier; **justify barrier widths + time-limit** |
| Model Development & Comparison | 30 | ≥3 models across linear / tree / NN families, with tuning; clear comparison |
| Feature Importance (Cluster-Level) | 10 | Cluster correlated features; MDI/MDA/SHAP at cluster level |
| Model Evaluation | 20 | precision/recall/F1/AUC, confusion, threshold analysis, **per-instrument breakdown**, **vs a follow-signal-blindly baseline** |
| Competition (optional) | +10 | Position-sizing strategy on the metamodel probs (constraints released 2026-05-20; check `refs/`) |

**Universe:** Equity {es1s,nq1s,fesx1s}; Energy {cl1s,ho1s,rb1s,ng1s}; Metals {gc1s,si1s,hg1s,pl1s}. Must cover ≥1 full class.
**Critical dates:** train **2021-01-01 → 2022-06-30**; **hidden test = Jul–Dec 2022** (rerun on submission). Signals released Jan 2020 → 2022-06-30.
**Deliverable CSV:** `date,instrument,prediction` (prob in [0,1]).
**Data:** `data/ohlcv_data.csv` (long, 1990→; equities start late: ES 1997/FESX 1998/NQ 1999), `data/primary_signals.csv` (645 rows, wide, no NaN).

## 2. What the prior phase did, and why
**Workflow:** deep-interview → consensus plan (`.omc/plans/signal-reverse-engineering-plan.md`) → autonomous "ralph" execution in two segments, each built by parallel agents against a locked interface contract, then independently reviewed (architect for the foundation, critic for the search) and corrected.

- **Segment 1 — C1 characterization** (*why:* know the signal's nature + lock the evaluation conventions before modeling). Foundation modules + diagnostics answering 6 questions (alpha type, lead-lag/holding convention, regime, cross-asset, drift, model-family fingerprint). Paused at a **human checkpoint** (you approved: chance-baseline bar, n_eff FLOOR=10, next-day convention).
- **Segment 2 — replication search** (*why:* if a simple, structurally-explicit strategy reproduces the signal, that *is* its alpha; what reproduces it tells you which features matter). 6 archetype families × 9 cells, tiered TPE/grid search on **train+val only**, judged by 4 robustness gates. Held-out internal test touched **exactly once**.

**Key methodological choices (and why they matter to you):**
- **Effective sample size = independent regime-runs, not calendar days.** The signal is piecewise-constant; 645 days collapse to ~2–35 independent decisions per instrument. *You must apply the same discipline to triple-barrier labels — overlapping/sticky labels are not independent samples.*
- **No leakage, ever:** chronological splits, per-instrument embargo ≥ run-length p90, train-only threshold calibration, and a `get_test(final_confirmation=True)` tripwire touched once. *Carry this into metamodel training/eval.*
- **Imbalance-/baseline-robust metrics:** an all-flat guess must score ≈0, not ≈90%. Ordinal skill is **chance-corrected** (kappa-style). *Your metamodel eval needs the same — accuracy is meaningless on this imbalance.*
- **Robustness over peak score:** gates require a parameter *plateau* (G3) + multi-metric agreement (G4) + drift-aware generalization (G2), not a single max.
- **Cross-instrument pooling artifact (important, hard-won):** naively concatenating instruments and computing one metric inflates agreement via per-instrument base-rate matching (momentum scored a fake +0.14 while *anti*-replicating 2 of 3 pooled members). Fixed to **within-instrument aggregation**. *If you pool instruments to train the metamodel, evaluate within-instrument or you will fool yourself.*

## 3. Results
### C1 — what the signal IS (`reports/signal-characterization.md`, `results/jj/thresholds.json`)
- **Alpha type: short-horizon MEAN-REVERSION / counter-trend.** 10/11 instruments have negative correlation with trailing returns; ng1s is single-direction (never long).
- **Holding convention: next-day CONFIRMED** — `corr(s_t, r_{t+1}) > 0` for all 11 (0.036–0.129). PnL = `s_t · r_{t+1}`.
- **Regime ("avoids high vol") is instrument-specific**, not universal: es/cl/si pull back in high vol; ng1s does the opposite (1%→39%); fesx/rb/hg are always-on.
- **Cross-asset mean |corr| ≈ 0.09** → signals are nearly independent across instruments.
- **Base-rate DRIFT is real** across splits (e.g. ng1s participation 0.07→0.31→0.43; hg1s long-bias swings +0.37 train→test).
- **Model-family fingerprint: INCONCLUSIVE** — no shallow tree/linear/forest surrogate beats the majority baseline → the generator is *not* a simple TA rule.
- **Effective-n (post-embargo val):** es35 nq20 fesx25 cl9 ho9 rb13 ng2 gc11 si19 hg29 pl26. At FLOOR=10, energy {cl1s,ho1s,ng1s} are pooled.
- *(Note: the C1 split is 60/20/20 INSIDE released data — train→2021-07-01, val→2021-12-30, test→2022-06-30 — distinct from the coursework's hidden Jul–Dec 2022.)*

### Replication — what reproduces it (`reports/replication-summary.md`, `results/jj/top_candidates.json`)
**Honest 3 of 6 families replicate ≥1 standalone instrument** (target was 5; the plan explicitly allows an honest shortfall, and methodology is what's graded):

| Family | Replicates | val→test composite |
|---|---|---|
| **mean_reversion** | es1s, fesx1s, si1s, hg1s | si1s 0.36 → 0.17 (genuine positive transfer) |
| **vol_regime_gated** (base=mean_reversion) | es1s, si1s | 0.30 → 0.06 (weak transfer) |
| **xsect_rank** (score=reversal) | pl1s | 0.31 → −0.03 (**flips negative OOS — caution**) |
| ts_momentum / breakout_donchian / hybrid | none | momentum is wrong-signed for a counter-trend signal |

## 4. Interpretation (what the results mean for the metamodel)
1. **The signal is a counter-trend strategy that "fades" recent moves and earns a small next-day edge.** Features that capture *deviation from recent levels* are the highest-value predictors of when following it works.
2. **No simple rule fully reproduces it.** Only mean-reversion-flavored archetypes pass, and even the best transfers only partially out-of-sample. → The metamodel is genuinely useful: it should learn *when* the counter-trend bet is reliable vs noise, which simple rules can't.
3. **It's instrument-specific and time-varying.** Don't assume one global model; build per-instrument or per-class features, and validate drift-aware.
4. **Cross-sectional structure is nearly absent (corr 0.09).** Cross-asset/ranking features are low-value (report this as a justified negative); per-instrument dynamics dominate.
5. **Thin effective-n is the central statistical constraint.** Be conservative about per-instrument claims; pool *within asset class* for power but evaluate within-instrument.

## 5. What to do next (prioritized; maps to the marking scheme)
**Build under `src/stml/` (e.g. a new `src/stml/metamodel/` package) and a notebook in `notebooks/jay/`. Reuse `stml.io`, `stml.na_checks`, and `stml.replication`.**

1. **Triple-barrier labeling (20 marks) — do this first; it defines the target.**
   - Label each primary-signal *trade* (a non-zero `s_t`): following `s_t`, does it hit the profit barrier before the stop or the time-limit? Label = 1 if the realized triple-barrier outcome is profitable, else 0.
   - **Justify barriers from C1:** time-limit (vertical barrier) ≈ the signal's holding behaviour (run-length p90 ranges 8–33 days; reuse `splits.run_length_p90`); barrier widths from instrument volatility (reuse `na_checks.rolling_vol` for σ-scaled barriers). Note the next-day execution convention (C1-confirmed).
   - Track **label overlap / uniqueness** (López de Prado) — overlapping labels break the i.i.d. assumption; this connects to the effective-n discipline.
2. **Feature engineering (20 marks) — C1-informed.**
   - **Counter-trend / mean-reversion (highest value):** z-score of `close − SMA_L`, distance-from-MA, short-horizon return reversal, RSI/Bollinger-style. (Mirror `archetypes.mean_reversion`'s score.)
   - **Volatility-regime:** rolling vol, and **GMM/Markov regime** features — reuse `characterize.regime` (statsmodels Markov-switching + sklearn GaussianMixture) directly for the "latent variable models" the rubric rewards.
   - **Signal-derived:** the signal value, its run-length / persistence, recency since last flip.
   - **Momentum/trend (for contrast + nonlinearity):** include them so the model can learn the counter-trend interaction; C1 says they're wrong-signed alone.
   - **Per-instrument** focus; document why cross-sectional features are low-value (corr 0.09).
3. **Model development (30 marks):** ≥3 families — Logistic (L1/L2), tree (RandomForest/XGBoost/LightGBM), NN (sequential/VSN). Tune with **purged/embargoed CV** (the fingerprint being inconclusive means *don't* pre-judge linear; compare honestly). If pooling within a class for power, beware the cross-instrument artifact (§2).
4. **Cluster-level feature importance (10 marks):** cluster correlated features (reuse `na_checks.corr_max_info` for the correlation matrix), then MDA/MDI/SHAP at cluster level; discuss which groups drive predictions (expect mean-reversion + vol-regime clusters to dominate).
5. **Evaluation (20 marks):** **reuse `stml.replication`** — `splits` (chronological + embargo + n_eff), a metrics panel (precision/recall/F1/AUC + the chance-corrected/imbalance-robust scores), **per-instrument breakdown**, and the **follow-signal-blindly baseline**. Carve the OOS period cleanly from train; keep the hidden Jul–Dec 2022 untouched until the final rerun (mirror the `get_test` tripwire).
6. **Deliverable:** an end-to-end notebook that produces `date,instrument,prediction` CSV for the required window; ensure it reruns on the hidden test.
7. **(Optional, +10) Competition:** position-sizing from metamodel probs; read the constraints released 2026-05-20 (`refs/`); report CAGR/vol/Sharpe/Sortino/maxDD/holding/turnover.

## 6. Pitfalls / learnings to carry forward
- **Effective-n, not row count** — report it; don't over-claim on thin instruments (energy especially).
- **No leakage** — chronological + purge/embargo; train-only fit; touch hidden test once.
- **Within-instrument evaluation** when pooling — the concatenation artifact is subtle and convincing.
- **Drift-aware** — compare against each split's own base rate; report base rates beside metrics.
- **Imbalance-robust metrics** — accuracy lies here; use chance-corrected/balanced metrics.
- **Honest reporting** — the rubric rewards rigorous methodology and critical analysis over performance; a well-explained negative is full marks.

## 7. Pointers (reproduce / reuse)
- **Reusable code:** `stml.io` (`load_clean_data`, `load_returns_panel`), `stml.na_checks` (`native_returns`, `rolling_vol`, `corr_max_info` — honor the NA policy: native returns before pivot, never ffill structural NaNs), `stml.replication.{align,splits,baselines,metrics,nav,characterize}`.
- **Reports to read:** `reports/signal-characterization.md`, `reports/replication-summary.md`, per-archetype `reports/*.md`, `reports/missing-data-report.md`.
- **Frozen artifacts:** `results/jj/{thresholds.json,top_candidates.json,ledger.json}`.
- **Reproduce the deep-dive:** `uv run python -m stml.replication.run_characterize` then `uv run python -m stml.replication.run_replicate --budget 64 --seed 0`. Tests: `uv run pytest tests/ -q` (229) ; lint `uv run ruff check src/ tests/`.
- **Plan + spec of record:** `.omc/plans/signal-reverse-engineering-plan.md`, `.omc/specs/deep-interview-signal-reverse-engineering.md`.
- **Env note:** uses `uv`; `optuna` + `pytest` were added. `results/jj/ledger.json` is ~3.8 MB (regenerable) — fine to ignore/slim.
