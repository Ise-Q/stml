# 00 — Project Context & Harry's Role (Critical-Analysis Seed)

> Audience: the team. This is the seed for the team-synthesis memo (see
> `99-team-synthesis-memo.md`), not a graded artifact in itself. It is a
> read-only snapshot of where the project stood when Harry's branch began work
> (2026-05-25), so future readers can trace what was inherited vs what was
> added.

## 1. The coursework, in one paragraph

Build a **metamodel** that, conditional on a primary signal `s_t ∈ {-1, 0, +1}`
for 11 futures across 3 asset classes, predicts the probability that following
that signal is profitable under a **triple-barrier** exit. Marked out of 100
(Features 20 / Labeling 20 / Models 30 / Cluster Importance 10 / Evaluation 20)
with an optional +10 for a strategy backtest. **Methodology, not performance,
is what is graded.** Deadline 2026-06-04.

Released signal window: 2020-01-03 → 2022-06-30. Hidden test: 2022-07-01 →
2022-12-31 (the grader reruns our code on it).

## 2. What each branch had built by 2026-05-25

### `main` (the shared spine)
- `stml.io` (loaders) and `stml.na_checks` (hand-coded NYMEX / COMEX /
  CME-equity / Eurex calendars 1990–2022, holiday-scope classifier, native
  per-instrument returns, pairwise-complete PSD-repaired correlations).
- Project README, `pyproject.toml` (editable install), `nbstripout` enforced
  via `.gitattributes`, and the missing-data report (`reports/missing-data-report.md`).
- **No labels, no features, no model.** It is the launchpad, not the
  deliverable.

### `Sreeram` (full pipeline, v0 → v5)
- Triple-barrier labelling (`labeling.py`), purged-K-fold CV (`cv.py`), 75
  features G1–G8 (`features.py`), filtered HMM + GMM regime posteriors
  (`regimes.py`), ElasticNet / XGBoost / MLP models with isotonic
  calibration (`models.py`), clustered MDI + MDA importance (`importance.py`),
  full evaluation suite incl. regime-conditional AUC (`evaluation.py`), five
  iterated pipelines (v0/v1, v3 commodity-only, v4 stacked, v5 principled).
- Documented per-stage in `docs/build/01-…12-…md`.
- Strongest piece: the v4 → v5 selection-on-test critique (v4's OOS AUC of
  0.562 was driven by architecture choices made *while looking at H1-2022*;
  v5 rebuilt under strict TRAIN/VAL/TEST gives an honest 95 % CI of
  [0.43, 0.50]).
- **Working assumption: the primary signal is trend-following.** Features
  weight momentum/trend heavily (G2 is 19 % of MDI in-sample); the strategy
  is "follow primary when meta-prob is high."

### `signal-deep-dive` (reverse-engineering + feature layer)
- `stml.replication` (~6 modules + 20 test files): align / splits /
  baselines / metrics / NAV / characterize / archetypes / gates / search /
  ledger. Searches 6 archetype families against the released signal with
  TPE above the n_eff floor (10) and exhaustive grid below.
- `stml.metamodel` (~8 modules, ~3 000 LoC): 75-feature matrix F1–F10
  with provenance, a redundancy map, per-instrument fitting scope, and
  causality proofs (truncation-invariance for engineered features,
  fit-provenance for fitted ones).
- Empirical finding: **`corr(s_t, r_{t+1}) > 0` for all 11**, and `s_t`
  loads **negatively** on trailing returns in 10/11 instruments. The
  primary signal is **mean-reversion / counter-trend**, not trend-following.
  Cross-asset mean |corr| ≈ 0.09, so cross-sectional features are
  expected-negative.
- **No metamodel built yet.** The `HANDOFF_signal-to-feature-engineering.md`
  document explicitly hands off to the next agent.

### `Harry`, `research/jay`
- Empty. Both point at `main` head as of 2026-05-25.

## 3. The contradiction this project has to resolve

> Sreeram's pipeline is built for trend-following. signal-deep-dive's
> characterization says the signal is counter-trend. Both pieces of work are
> technically rigorous in isolation; together they are inconsistent.

The contradiction matters because the two branches engineered features for
*opposite* alpha priors:

- Sreeram's **G2 trend cluster** (momentum, MA distance, trend t-statistics)
  is large and prominent in-sample.
- signal-deep-dive's **F1 counter-trend** family (`-zscore_L(close - SMA_L)`,
  hi-lo position, RSI, Bollinger %b) is the headline feature family.

If the signal-deep-dive characterization is right, Sreeram's MDA finding that
the trend cluster has **negative OOS importance** (permuting trend features
*helps*) is the same fact arriving as a diagnostic.

## 4. What Harry's branch will add

The deliverables on this branch are intentionally scoped to be the
**synthesis + creative layer** the rubric rewards.

| Step | Artifact | Why it matters |
|---|---|---|
| 1 | `signal_audit.py` → `results/harry/signal_direction.csv` + `reports/harry/01-signal-direction.md` | Resolves the trend-vs-reversion contradiction with one tight measurement, independent of either teammate's framing. |
| 2 | `labels.py` | Canonical triple-barrier labels with **next-day entry** (the convention `signal-deep-dive` empirically confirmed and `Sreeram`'s `labeling.py` does not state explicitly), **asymmetric barriers** (so we can later sweep a counter-trend-bias label), and **trading-day** uniqueness weights. |
| 3 | `src/stml/harry/features/` — eight families | Features no other branch ships: signal-trajectory, conditional first-passage risk, information-theoretic (MI + transfer entropy), microstructure with the zero-volume mask correctly applied, cross-asset lead-lag / dispersion, MODWT wavelet energy bands, concept-drift alignment score, optional persistent homology. |
| 4 | `pipeline.py` | ElasticNet + LightGBM (with monotone constraints) + a re-implemented **VSN** for `torch >= 2.12`. Cluster importance via MDI + MDA + tree-SHAP side-by-side. |
| 5 | `strategy.py` | Kelly-fractional sizing + vol target; rubric bonus track. |
| 6 | `reports/harry/02-…06-…md` + `99-team-synthesis-memo.md` | The "critical analysis" the rubric explicitly rewards: a 1-page memo addressing the trend-vs-reversion contradiction, citing the Step-1 audit, and explaining why Harry's architecture picked the working assumption it picked. |

## 5. Constraints Harry's branch operates under

1. **Absolute branch isolation.** Writes confined to `src/stml/harry/`,
   `notebooks/harry/`, `tests/harry/`, `reports/harry/`, `results/harry/`,
   `pyproject.toml`, `uv.lock`. Everything else is read-only — including
   anything Sreeram or signal-deep-dive added.
2. **Reuse via copy, not import.** If code from another branch is needed,
   it is *copied* into `src/stml/harry/` with a header citing the source
   commit, then modified in place. The two work-branches stay
   self-consistent in their own worlds.
3. **Determinism.** Default `random_state=42` everywhere; every CV split is
   seeded; every model fit is seeded.
4. **Truncation-invariance for every feature.** A feature value at time `t`
   computed on `data[:t+1]` must equal the value at time `t` computed on
   `data[:T]` for any `T >= t+1`. There is a single causality test
   (`tests/harry/test_causality.py`) every feature must pass.
5. **Next-day execution convention.** `PnL_t = s_t · r_{t+1}`. Labels are
   evaluated over `[t+1, t+h+1]`, not `[t, t+h]`.

## 6. Open questions deferred until after Step 1

These are intentionally not pre-decided. The Step-1 audit produces the
evidence; the team-synthesis memo (Step 6) gives the answer.

- Whether to label with symmetric barriers (`pt = sl = 1.0`) or with a
  counter-trend tilt (e.g. `pt = 1.5, sl = 1.0` if the signal is reverting).
- Whether to drop equity from training (Sreeram's v3 finding) or pool all
  instruments and let regularisation do the work.
- Whether to ensemble Sreeram's `predictions_v5.csv` with Harry's
  `predictions.csv` for the final group submission, or to submit one only.

---

*This document is the seed. The decisions and their evidence go in
`99-team-synthesis-memo.md` once Steps 1–5 have produced numbers.*
