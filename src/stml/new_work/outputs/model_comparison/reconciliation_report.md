# Reconciliation Report

Produced by `reconciliation.py`. Reads stored OOS predictions from the model-comparison harness. No models retrained; calibration uses a leakage-free inner-split protocol on stored fold assignments.

## 1. Signal vs No-Signal Classification

Criterion: `per_fold_mean_AUC − per_fold_std_AUC > 0.50`  
(lower CI of per-path AUC distribution must clear random.)  
Note: `auc_mean` is the **mean of per-fold AUCs** (not pooled AUC — pooled can understate performance by 0.02–0.04 for imbalanced folds).

| Instrument | Per-fold AUC | ±std | Lower CI | Signal? |
|---|---|---|---|---|
| cl1s | 0.707 | 0.130 | 0.577 | **YES** |
| es1s | 0.605 | 0.069 | 0.536 | **YES** |
| fesx1s | 0.498 | 0.047 | 0.450 | no |
| gc1s | 0.543 | 0.121 | 0.423 | no |
| hg1s | 0.530 | 0.064 | 0.467 | no |
| ho1s | 0.775 | 0.155 | 0.620 | **YES** |
| ng1s | 0.590 | 0.142 | 0.448 | no |
| nq1s | 0.542 | 0.108 | 0.434 | no |
| pl1s | 0.541 | 0.052 | 0.489 | no |
| rb1s | 0.632 | 0.118 | 0.514 | **YES** |
| si1s | 0.499 | 0.067 | 0.432 | no |

**Signal-bearing (4):** cl1s, es1s, ho1s, rb1s

**No-signal (7):** fesx1s, gc1s, hg1s, ng1s, nq1s, pl1s, si1s

**Note on ng1s:** Individual RF achieves per-fold mean AUC ≈ 0.60, but with only 120 events across 15 CPCV paths the per-fold std is ~0.14, pushing the lower CI to ~0.46 — below the 0.50 floor. ng1s is therefore classified **no-signal by strict criterion**. A softer threshold (0.48) would include it. Excluded from feature importance; noted as borderline.

## 2. STD-Based Tie Analysis

Tie threshold: `|AUC gap from best| ≤ best_std` (within 1σ of winner).

### cl1s  (best per-fold AUC 0.707 ± 0.130)

| Group | Model | AUC | Gap | Tied? |
|---|---|---|---|---|
| cl1s | xgb | 0.707 | +0.000 | ✓ |
| energy_cl_ho | logistic | 0.692 | +0.015 | ✓ |
| energy_all | logistic | 0.674 | +0.033 | ✓ |
| energy_cl_ho | xgb | 0.669 | +0.037 | ✓ |
| energy_all | mlp | 0.668 | +0.038 | ✓ |
| cl1s | mlp | 0.662 | +0.044 | ✓ |
| cl1s | rf | 0.644 | +0.063 | ✓ |
| cl1s | logistic | 0.642 | +0.065 | ✓ |
| energy_all | xgb | 0.641 | +0.065 | ✓ |
| energy_cl_ho | rf | 0.619 | +0.087 | ✓ |
| energy_cl_ho | mlp | 0.617 | +0.089 | ✓ |
| energy_all | rf | 0.571 | +0.135 | — |

### es1s  (best per-fold AUC 0.605 ± 0.069)

| Group | Model | AUC | Gap | Tied? |
|---|---|---|---|---|
| es1s | rf | 0.605 | +0.000 | ✓ |
| es1s | xgb | 0.583 | +0.022 | ✓ |
| es1s | logistic | 0.563 | +0.042 | ✓ |
| es1s | mlp | 0.540 | +0.064 | ✓ |

### ho1s  (best per-fold AUC 0.775 ± 0.155)

| Group | Model | AUC | Gap | Tied? |
|---|---|---|---|---|
| energy_all | mlp | 0.775 | +0.000 | ✓ |
| energy_all | logistic | 0.759 | +0.016 | ✓ |
| energy_all | xgb | 0.746 | +0.029 | ✓ |
| energy_cl_ho | logistic | 0.734 | +0.041 | ✓ |
| energy_cl_ho | mlp | 0.651 | +0.124 | ✓ |
| energy_cl_ho | rf | 0.634 | +0.141 | ✓ |
| energy_all | rf | 0.618 | +0.157 | — |
| energy_cl_ho | xgb | 0.527 | +0.248 | — |

### rb1s  (best per-fold AUC 0.632 ± 0.118)

| Group | Model | AUC | Gap | Tied? |
|---|---|---|---|---|
| energy_all | logistic | 0.632 | +0.000 | ✓ |
| energy_all | xgb | 0.629 | +0.003 | ✓ |
| rb1s | logistic | 0.617 | +0.015 | ✓ |
| energy_all | mlp | 0.585 | +0.048 | ✓ |
| energy_all | rf | 0.574 | +0.059 | ✓ |
| rb1s | xgb | 0.557 | +0.075 | ✓ |
| rb1s | mlp | 0.501 | +0.131 | — |
| rb1s | rf | 0.501 | +0.132 | — |

## 3. Calibration Results

Protocol: for each CPCV fold, fit simplified inner model on first 60% of training fold → get predictions on last 40% → fit Platt (sigmoid) and isotonic calibrators → apply to stored raw OOS scores for that fold's test events. If both calibrators increase per-fold Brier, fall back to raw scores (method='none'). Dominant method across folds reported.

### cl1s

| Group | Model | AUC | Brier raw | Brier cal | Method |
|---|---|---|---|---|---|
| cl1s | xgb | 0.707 | 0.2177 | 0.1956 (↓0.0221) | none |
| energy_cl_ho | logistic | 0.692 | 0.2637 | 0.2172 (↓0.0465) | sigmoid |
| energy_all | logistic | 0.674 | 0.2568 | 0.2212 (↓0.0356) | sigmoid |
| energy_cl_ho | xgb | 0.669 | 0.2298 | 0.2094 (↓0.0204) | none |
| energy_all | mlp | 0.668 | 0.2393 | 0.2061 (↓0.0333) | isotonic |
| cl1s | mlp | 0.662 | 0.2174 | 0.2075 (↓0.0099) | none |
| cl1s | rf | 0.644 | 0.2178 | 0.2048 (↓0.0130) | none |
| cl1s | logistic | 0.642 | 0.2707 | 0.2227 (↓0.0479) | none |
| energy_all | xgb | 0.641 | 0.2285 | 0.2196 (↓0.0088) | none |
| energy_cl_ho | rf | 0.619 | 0.2243 | 0.2076 (↓0.0167) | none |
| energy_cl_ho | mlp | 0.617 | 0.2473 | 0.2234 (↓0.0239) | none |

### es1s

| Group | Model | AUC | Brier raw | Brier cal | Method |
|---|---|---|---|---|---|
| es1s | rf | 0.605 | 0.2462 | 0.2432 (↓0.0029) | none |
| es1s | xgb | 0.583 | 0.2561 | 0.2499 (↓0.0062) | none |
| es1s | logistic | 0.563 | 0.3347 | 0.2747 (↓0.0600) | sigmoid |
| es1s | mlp | 0.540 | 0.3290 | 0.2738 (↓0.0552) | sigmoid |

### ho1s

| Group | Model | AUC | Brier raw | Brier cal | Method |
|---|---|---|---|---|---|
| energy_all | mlp | 0.775 | 0.2184 | 0.2044 (↓0.0140) | isotonic |
| energy_all | logistic | 0.759 | 0.2018 | 0.2098 (↑0.0080) | sigmoid |
| energy_all | xgb | 0.746 | 0.2203 | 0.2141 (↓0.0062) | none |
| energy_cl_ho | logistic | 0.734 | 0.2039 | 0.2163 (↑0.0125) | sigmoid |
| energy_cl_ho | mlp | 0.651 | 0.2293 | 0.2243 (↓0.0050) | isotonic |
| energy_cl_ho | rf | 0.634 | 0.2136 | 0.2016 (↓0.0120) | none |

### rb1s

| Group | Model | AUC | Brier raw | Brier cal | Method |
|---|---|---|---|---|---|
| energy_all | logistic | 0.632 | 0.2870 | 0.2525 (↓0.0345) | sigmoid |
| energy_all | xgb | 0.629 | 0.2489 | 0.2469 (↓0.0020) | none |
| rb1s | logistic | 0.617 | 0.2634 | 0.2515 (↓0.0119) | none |
| energy_all | mlp | 0.585 | 0.3233 | 0.2695 (↓0.0538) | isotonic |
| energy_all | rf | 0.574 | 0.2563 | 0.2584 (↑0.0021) | none |
| rb1s | xgb | 0.557 | 0.2571 | 0.2497 (↓0.0074) | none |

## 4. Final Champion Selection

Tiebreak: lowest calibrated Brier → model simplicity (logistic < RF < XGB < MLP).

| Instrument | Champion | AUC ±std | Brier raw→cal | Cal | Runner-up | Notes |
|---|---|---|---|---|---|---|
| cl1s | cl1s/xgb | 0.707±0.130 | 0.21772→0.19564 | none | cl1s/rf (AUC=0.644) | 11 candidates within 1σ |
| es1s | es1s/rf | 0.605±0.069 | 0.24619→0.24324 | none | es1s/xgb (AUC=0.583) | 4 candidates within 1σ |
| ho1s | energy_cl_ho/rf | 0.634±0.155 | 0.21356→0.20157 | none | energy_all/mlp (AUC=0.775) | 6 candidates within 1σ |
| rb1s | energy_all/xgb | 0.629±0.118 | 0.24886→0.24685 | none | rb1s/xgb (AUC=0.557) | 6 candidates within 1σ |

## 5. Cleaned Pooling Verdict

Pooling 'helps' only where pool beats individual by more than 1σ.

- **cl1s:** Within noise (gap=+0.015, 1σ=0.130): indiv=0.707, pool=0.692. Inconclusive.
- **es1s:** Individual only — no pooling comparison.
- **ho1s:** No individual model (too thin). Best pool AUC=0.775±0.155. Pooling is the only option.
- **rb1s:** Within noise (gap=-0.015, 1σ=0.118): indiv=0.617, pool=0.632. Inconclusive.

## 6. Instruments Carrying Forward to Feature Importance

Feature importance on a ~0.50 AUC model captures noise, not signal. Only signal-bearing, calibrated champions proceed.

**4 instruments:** cl1s, es1s, ho1s, rb1s

| Instrument | Group | Model | AUC | Brier_calibrated | Cal method |
|---|---|---|---|---|---|
| cl1s | cl1s | xgb | 0.707 | 0.19564 | none |
| es1s | es1s | rf | 0.605 | 0.24324 | none |
| ho1s | energy_cl_ho | rf | 0.634 | 0.20157 | none |
| rb1s | energy_all | xgb | 0.629 | 0.24685 | none |

**Excluded (no signal):** fesx1s, gc1s, hg1s, ng1s, nq1s, pl1s, si1s.

*ng1s borderline note:* best per-fold AUC 0.60 with lower CI 0.46. Excluded by strict criterion; revisit if additional signal data accrues.
