"""v5 stress test: multiple boundaries, multiple configs.

Tests the robustness of the principled pipeline across:
  - Different train/val/test boundary placements (simulates different rerun
    scenarios)
  - Configuration variants (with/without calibration, more/less shrinkage)
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from stml.v5 import run_v5, V5Config

# Multiple boundaries — including ones that simulate the actual grader's rerun.
configs_to_try = [
    # boundary, predict_end, val_months, label
    (pd.Timestamp("2021-10-01"), pd.Timestamp("2022-01-01"), 6, "boundary=2021-10 (early test)"),
    (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-07-01"), 6, "boundary=2022-01 (our submission)"),
    (pd.Timestamp("2022-04-01"), pd.Timestamp("2022-07-01"), 6, "boundary=2022-04 (val SPANS regime break)"),
]

print(f"{'config':<50s} {'VAL AUC':>8s} {'TEST AUC':>9s} {'95% CI':>20s} {'TEST Brier':>11s}")
print("-" * 100)

results_by_boundary = {}

for boundary, predict_end, val_months, label in configs_to_try:
    try:
        cfg = V5Config(boundary=boundary, predict_end=predict_end, val_months=val_months)
        r = run_v5(cfg, verbose=False, do_stability=False)
        val_auc = "n/a"  # We don't return this directly
        test_auc = r["report_test_final"]["auc"]
        test_br = r["report_test_final"]["brier"]
        m, lo, hi = r["bootstrap_auc"]
        ci = f"[{lo:.3f}, {hi:.3f}]"
        print(f"{label:<50s} {val_auc:>8s} {test_auc:>9.3f} {ci:>20s} {test_br:>11.3f}")
        results_by_boundary[label] = r
    except Exception as e:
        print(f"{label:<50s} FAILED: {e}")

# Sanity check: pure baseline (constant prediction)
print()
print("Sanity baselines (predict_v3-style commodity-only, predict label_1_share, etc.):")
print(f"  Constant label_1_share = {0.549:.3f}  → AUC = 0.500 by construction")

# Cross-boundary stability: compare predictions across boundaries on overlapping events
print()
print("=== Cross-boundary stability: how stable are predictions when boundary shifts? ===")
if len(results_by_boundary) >= 2:
    # Take v5 at 2022-01-01 vs v5 at 2022-04-01; compare predictions on the
    # 2022-04 to 2022-07 events (which are TEST for both — but trained differently).
    keys = list(results_by_boundary.keys())
    if "boundary=2022-01 (our submission)" in results_by_boundary and \
       "boundary=2022-04 (val SPANS regime break)" in results_by_boundary:
        r1 = results_by_boundary["boundary=2022-01 (our submission)"]
        r2 = results_by_boundary["boundary=2022-04 (val SPANS regime break)"]
        # r2 test was 2022-04-01 to 2022-07-01; r1 test includes that span
        # Both predicted on that quarter. Are the predictions correlated?
        p1_df = r1["predictions"]
        p2_df = r2["predictions"]
        merged = p1_df.merge(p2_df, on=["date", "instrument"], suffixes=("_b22Q1", "_b22Q2"))
        if len(merged) > 0:
            # Pearson correlation between predictions
            corr = merged[["prediction_b22Q1", "prediction_b22Q2"]].corr().iloc[0, 1]
            print(f"  Pearson correlation between predictions at boundary=2022-01 vs 2022-04: {corr:.3f}")
            print(f"  (close to 1.0 = stable predictions; close to 0 = chaotic)")
