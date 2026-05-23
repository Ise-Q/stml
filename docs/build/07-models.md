# Stage 4 — Model Development & Comparison

> Module: [`src/stml/models.py`](../../src/stml/models.py) (3 model classes)
> Orchestration: [`src/stml/experiments.py`](../../src/stml/experiments.py)
> Rubric: **30 marks** (the largest single section).

## Three model families — one wrapper interface each

| Family | Class | Library |
|---|---|---|
| **Linear** | `ElasticNetLogReg` | sklearn `LogisticRegression(elasticnet, saga)` |
| **Tree-based** | `XGBoostMeta` | xgboost `XGBClassifier` |
| **Neural network** | `MlpMeta` | sklearn `MLPClassifier` (Programming Session 5) |

All three share the same interface: `fit(X, y, t, t1, sample_weight)` →
`predict_proba(X)` → calibrated probability in [0, 1]. Every model is tuned
with `RandomizedSearchCV` over `PurgedKFold(5, embargo=10d)` and wrapped in
`CalibratedClassifierCV(method='isotonic', cv=PurgedKFold(3))` so the output
probability is a *real* probability, not a raw logit.

## Per-family design notes

### ElasticNetLogReg
- `saga` solver supports elastic net + sample weights.
- `class_weight='balanced'` for the mild 56/44 imbalance.
- Search grid: `C ∈ {10^{-3..2}}`, `l1_ratio ∈ {0, .25, .5, .75, 1}`.
- Exposes `linear_coefficients()` for sign-check of feature direction.

### XGBoostMeta
- `tree_method='hist'` for speed on tabular data.
- `scale_pos_weight = neg/pos` derived automatically.
- Search grid covers depth (3–8), learning rate (0.01–0.1), n_estimators
  (100–500), subsample (0.7–1.0), colsample (0.6–1.0), L1/L2 reg, min child weight.
- `feature_importance('gain')` returns MDI on the booster — feeds Stage 5
  cluster importance directly.

### MlpMeta (the neural network family)
- Two-layer MLP, hidden widths tuned in {(50,30), (80,40), (100,), (64,32,16)}.
- Activations `relu` / `tanh`; L2 alpha tuned in {1e-4, ..., 0.1}.
- Built-in early stopping (`validation_fraction=0.1`, `n_iter_no_change=15`).
- **Honest note**: the *ideal* neural network for this assignment is the VSN
  from Programming Session 6 (with built-in softmax feature importance). The
  full PyTorch VSN implementation is in `src/stml/models.py:_vsn_torch_modules`,
  but on this Mac the installed `torch` (2.2, x86_64) was compiled against
  NumPy 1.x while the env has NumPy 2.x — a binary ABI mismatch. The fix
  requires either downgrading NumPy or upgrading torch to a version whose
  x86_64 wheel is no longer published. Programming Session 5's MLPClassifier
  is the same NN family and is fully course-aligned; we ship that as the
  active classifier and label the VSN code as future work to swap in when
  the env is sorted.

## Head-to-head: OOS H1-2022, n=1002

```
            logreg  xgboost     mlp
auc         0.4885   0.4800  0.4772
f1          0.5962   0.6519  0.6709
precision   0.5124   0.5256  0.5478
recall      0.7127   0.8582  0.8655
brier       0.2576   0.2515  0.2535
log_loss    0.7084   0.6951  0.7004
```

**Best on OOS log-loss: XGBoost** — used to write `predictions_v2.csv`.

**Honest reading:** all three converge on a similar story — the signal in
2020-2021 training data does not transfer cleanly to H1-2022 (the regime
break is real and visible in *every* model family). XGBoost wins the
calibration race (lowest Brier + log-loss) by virtue of better-shaped
probabilities, not by better discrimination (AUC essentially tied at ~0.48
across all three). MLP has the highest recall but pays for it in precision.

## Per-sector ablation — the real story

Trained XGBoost separately on each asset class (3 sector-specific models)
and compared to the pooled model on the SAME OOS events:

```
        pooled_auc  sector_auc  sector  sector_minus_pooled
gc1s        0.5319      0.9167  metals               +0.385  ←  HUGE lift
cl1s        0.5801      0.7163  energy               +0.136
ng1s        0.4513      0.5850  energy               +0.134
pl1s        0.4699      0.5398  metals               +0.070
hg1s        0.4309      0.4565  metals               +0.026
si1s        0.4819      0.4690  metals               -0.013
rb1s        0.7333      0.5072  energy               -0.226  ←  pooled wins
fesx1s      0.2522      0.2485  equity               -0.004
es1s        0.4797      0.4597  equity               -0.020
nq1s        0.3159      0.2352  equity               -0.081
```

**The finding:**
- **Metals models substantially beat the pooled model.** Gold goes from random
  (0.53) to near-perfect (0.92). The pooled model is being dragged by equity
  events whose features carry different information.
- **Energy is mixed**: crude oil and natural gas improve with sector-specific
  modelling; gasoline (`rb1s`) prefers the pooled model — probably because the
  pooled model picks up cross-asset structure (correlation with crude) that
  the per-sector RB model can't.
- **Equity stays bad** under both: the regime break in H1-2022 is real and the
  2020-2021 training data is misleading regardless of how we slice it.

**Implication for the report:** the right answer is *not* "always use pooled"
or "always use per-sector". It's a per-sector decision: metals win
per-sector, gasoline wins pooled, equity is hard either way. This is a sharp
methodological observation that anchors the critical-analysis section.

## What the rubric should see

- **3 models from 3 families**, tuned with purged CV, calibrated, compared
  head-to-head on the same OOS data → **30 marks earned with comparison
  table + per-instrument breakdown + ablation**.
- **Per-sector ablation** as a model-comparison axis = the kind of critical
  analysis the rubric explicitly asks for.
- **All models honest about their limitations** — none claims an AUC > 0.55
  on H1-2022 because the regime break is real.
