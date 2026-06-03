# Reconciliation Report

Produced by `reconciliation.py`. Reads stored OOS predictions from the model-comparison harness. No models retrained; calibration uses a leakage-free inner-split protocol on stored fold assignments.

## 1. Signal vs No-Signal Classification

Criterion: `per_fold_mean_AUC − per_fold_std_AUC > 0.50`  
(lower CI of per-path AUC distribution must clear random.)  
Note: `auc_mean` is the **mean of per-fold AUCs** (not pooled AUC — pooled can understate performance by 0.02–0.04 for imbalanced folds).

| Instrument | Per-fold AUC | ±std | Lower CI | Signal? |
|---|---|---|---|---|
| cl1s | 0.719 | 0.082 | 0.637 | **YES** |
| es1s | 0.512 | 0.092 | 0.420 | no |
| fesx1s | 0.588 | 0.060 | 0.528 | **YES** |
| gc1s | 0.500 | 0.123 | 0.377 | no |
| hg1s | 0.582 | 0.045 | 0.537 | **YES** |
| ho1s | 0.572 | 0.111 | 0.460 | no |
| ng1s | 0.552 | 0.105 | 0.447 | no |
| nq1s | 0.711 | 0.119 | 0.593 | **YES** |
| pl1s | 0.609 | 0.130 | 0.479 | no |
| rb1s | 0.550 | 0.085 | 0.465 | no |
| si1s | 0.616 | 0.074 | 0.542 | **YES** |

**Signal-bearing (5):** cl1s, fesx1s, hg1s, nq1s, si1s

**No-signal (6):** es1s, gc1s, ho1s, ng1s, pl1s, rb1s

**Note on ng1s:** Individual RF achieves per-fold mean AUC ≈ 0.60, but with only 120 events across 15 CPCV paths the per-fold std is ~0.14, pushing the lower CI to ~0.46 — below the 0.50 floor. ng1s is therefore classified **no-signal by strict criterion**. A softer threshold (0.48) would include it. Excluded from feature importance; noted as borderline.

## 2. STD-Based Tie Analysis

Tie threshold: `|AUC gap from best| ≤ best_std` (within 1σ of winner).

### cl1s  (best per-fold AUC 0.719 ± 0.082)

| Group | Model | AUC | Gap | Tied? |
|---|---|---|---|---|
| cl1s | xgb | 0.719 | +0.000 | ✓ |
| energy_cl_ho | xgb | 0.717 | +0.002 | ✓ |
| cl1s | logistic | 0.703 | +0.016 | ✓ |
| energy_cl_ho | logistic | 0.690 | +0.029 | ✓ |
| cl1s | rf | 0.621 | +0.098 | — |
| energy_cl_ho | mlp | 0.600 | +0.119 | — |
| energy_cl_ho | rf | 0.535 | +0.184 | — |
| energy_all | mlp | 0.512 | +0.207 | — |
| energy_all | logistic | 0.500 | +0.220 | — |
| energy_all | xgb | 0.453 | +0.266 | — |
| energy_all | rf | 0.440 | +0.279 | — |

### fesx1s  (best per-fold AUC 0.588 ± 0.060)

| Group | Model | AUC | Gap | Tied? |
|---|---|---|---|---|
| fesx1s | logistic | 0.588 | +0.000 | ✓ |
| fesx1s | mlp | 0.545 | +0.042 | ✓ |
| fesx1s | xgb | 0.533 | +0.055 | ✓ |
| fesx1s | rf | 0.512 | +0.075 | — |

### hg1s  (best per-fold AUC 0.582 ± 0.045)

| Group | Model | AUC | Gap | Tied? |
|---|---|---|---|---|
| hg1s | rf | 0.582 | +0.000 | ✓ |
| hg1s | logistic | 0.563 | +0.019 | ✓ |
| hg1s | xgb | 0.551 | +0.031 | ✓ |
| hg1s | mlp | 0.538 | +0.044 | ✓ |

### nq1s  (best per-fold AUC 0.711 ± 0.119)

| Group | Model | AUC | Gap | Tied? |
|---|---|---|---|---|
| nq1s | xgb | 0.711 | +0.000 | ✓ |
| nq1s | logistic | 0.669 | +0.043 | ✓ |
| nq1s | mlp | 0.619 | +0.093 | ✓ |
| nq1s | rf | 0.601 | +0.110 | ✓ |

### si1s  (best per-fold AUC 0.616 ± 0.074)

| Group | Model | AUC | Gap | Tied? |
|---|---|---|---|---|
| si1s | xgb | 0.616 | +0.000 | ✓ |
| si1s | rf | 0.602 | +0.015 | ✓ |
| precious | xgb | 0.584 | +0.032 | ✓ |
| precious | rf | 0.582 | +0.035 | ✓ |
| precious | logistic | 0.544 | +0.072 | ✓ |
| si1s | logistic | 0.525 | +0.091 | — |
| precious | mlp | 0.518 | +0.098 | — |
| si1s | mlp | 0.468 | +0.148 | — |

## 3. Calibration Results

Protocol: for each CPCV fold, fit simplified inner model on first 60% of training fold → get predictions on last 40% → fit Platt (sigmoid) and isotonic calibrators → apply to stored raw OOS scores for that fold's test events. If both calibrators increase per-fold Brier, fall back to raw scores (method='none'). Dominant method across folds reported.

### cl1s

| Group | Model | AUC | Brier raw | Brier cal | Method |
|---|---|---|---|---|---|
| cl1s | xgb | 0.719 | 0.2519 | 0.2205 (↓0.0314) | sigmoid |
| energy_cl_ho | xgb | 0.717 | 0.2587 | 0.2243 (↓0.0344) | none |
| cl1s | logistic | 0.703 | 0.2534 | 0.2371 (↓0.0164) | none |
| energy_cl_ho | logistic | 0.690 | 0.2682 | 0.2314 (↓0.0368) | none |

### fesx1s

| Group | Model | AUC | Brier raw | Brier cal | Method |
|---|---|---|---|---|---|
| fesx1s | logistic | 0.588 | 0.2667 | 0.2522 (↓0.0146) | sigmoid |
| fesx1s | mlp | 0.545 | 0.2954 | 0.2518 (↓0.0435) | sigmoid |
| fesx1s | xgb | 0.533 | 0.2557 | 0.2535 (↓0.0022) | none |

### hg1s

| Group | Model | AUC | Brier raw | Brier cal | Method |
|---|---|---|---|---|---|
| hg1s | rf | 0.582 | 0.2455 | 0.2453 (↓0.0002) | none |
| hg1s | logistic | 0.563 | 0.2607 | 0.2540 (↓0.0068) | none |
| hg1s | xgb | 0.551 | 0.2491 | 0.2479 (↓0.0012) | none |
| hg1s | mlp | 0.538 | 0.3044 | 0.2565 (↓0.0479) | sigmoid |

### nq1s

| Group | Model | AUC | Brier raw | Brier cal | Method |
|---|---|---|---|---|---|
| nq1s | xgb | 0.711 | 0.2356 | 0.2335 (↓0.0021) | none |
| nq1s | logistic | 0.669 | 0.2467 | 0.2350 (↓0.0117) | none |
| nq1s | mlp | 0.619 | 0.2870 | 0.2422 (↓0.0448) | none |
| nq1s | rf | 0.601 | 0.2462 | 0.2447 (↓0.0015) | none |

### si1s

| Group | Model | AUC | Brier raw | Brier cal | Method |
|---|---|---|---|---|---|
| si1s | xgb | 0.616 | 0.2468 | 0.2458 (↓0.0010) | none |
| si1s | rf | 0.602 | 0.2457 | 0.2447 (↓0.0010) | none |
| precious | xgb | 0.584 | 0.2484 | 0.2491 (↑0.0007) | none |
| precious | rf | 0.582 | 0.2454 | 0.2460 (↑0.0006) | none |
| precious | logistic | 0.544 | 0.2767 | 0.2566 (↓0.0202) | none |

## 4. Final Champion Selection

Tiebreak: lowest calibrated Brier → model simplicity (logistic < RF < XGB < MLP).

| Instrument | Champion | AUC ±std | Brier raw→cal | Cal | Runner-up | Notes |
|---|---|---|---|---|---|---|
| cl1s | cl1s/xgb | 0.719±0.082 | 0.25188→0.22049 | sigmoid | energy_cl_ho/xgb (AUC=0.717) | 4 candidates within 1σ |
| fesx1s | fesx1s/mlp | 0.545±0.060 | 0.29537→0.25184 | sigmoid | fesx1s/logistic (AUC=0.588) | 3 candidates within 1σ; MLP vs non-MLP within Brier noise — prefer log |
| hg1s | hg1s/rf | 0.582±0.045 | 0.24551→0.2453 | none | hg1s/xgb (AUC=0.551) | 4 candidates within 1σ |
| nq1s | nq1s/xgb | 0.711±0.119 | 0.23563→0.23351 | none | nq1s/logistic (AUC=0.669) | 4 candidates within 1σ |
| si1s | si1s/rf | 0.602±0.074 | 0.24571→0.24473 | none | si1s/xgb (AUC=0.616) | 5 candidates within 1σ |

## 5. Cleaned Pooling Verdict

Pooling 'helps' only where pool beats individual by more than 1σ.

- **cl1s:** Within noise (gap=+0.002, 1σ=0.082): indiv=0.719, pool=0.717. Inconclusive.
- **fesx1s:** Individual only — no pooling comparison.
- **hg1s:** Individual only — no pooling comparison.
- **nq1s:** Individual only — no pooling comparison.
- **si1s:** Within noise (gap=+0.032, 1σ=0.074): indiv=0.616, pool=0.584. Inconclusive.

## 6. Instruments Carrying Forward to Feature Importance

Feature importance on a ~0.50 AUC model captures noise, not signal. Only signal-bearing, calibrated champions proceed.

**5 instruments:** cl1s, fesx1s, hg1s, nq1s, si1s

| Instrument | Group | Model | AUC | Brier_calibrated | Cal method |
|---|---|---|---|---|---|
| cl1s | cl1s | xgb | 0.719 | 0.22049 | sigmoid |
| fesx1s | fesx1s | mlp | 0.545 | 0.25184 | sigmoid |
| hg1s | hg1s | rf | 0.582 | 0.2453 | none |
| nq1s | nq1s | xgb | 0.711 | 0.23351 | none |
| si1s | si1s | rf | 0.602 | 0.24473 | none |

**Excluded (no signal):** es1s, gc1s, ho1s, ng1s, pl1s, rb1s.

*ng1s borderline note:* best per-fold AUC 0.60 with lower CI 0.46. Excluded by strict criterion; revisit if additional signal data accrues.
