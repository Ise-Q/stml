"""Tests for the OOS evaluation harness (Stage 2, RED-first).

Adapts PS5's ``evaluate_model`` into a sample-weighted, threshold-aware, calibration-aware
scorer wired to the purged CV. Known-value assertions pin the metric maths; the integration
test runs a real estimator through PurgedKFold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alken_metamodel.cross_validation import PurgedKFold
from alken_metamodel.evaluation import (
    always_act_baseline,
    cross_val_evaluate,
    evaluate_predictions,
    threshold_sweep,
)
from alken_metamodel.models import make_xgb


def test_perfect_predictions_score_perfectly():
    y = np.array([1, 1, 0, 0])
    p = np.array([1.0, 1.0, 0.0, 0.0])
    m = evaluate_predictions(y, p)
    for k in ("accuracy", "precision", "recall", "f1", "auc", "avg_precision"):
        assert m[k] == 1.0, k
    assert m["brier"] == 0.0
    assert m["log_loss"] < 1e-6


def test_known_confusion_values():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1])  # pred @0.5 -> [1,0,1,0]
    m = evaluate_predictions(y, p, threshold=0.5)
    assert m["accuracy"] == 0.5
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5
    assert m["auc"] == 0.75  # 3 of 4 pos>neg pairs
    assert abs(m["brier"] - 0.185) < 1e-12  # mean((p-y)^2)
    assert m["n"] == 4


def test_sample_weight_changes_metrics():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1])
    # zero-weight the false-negative (idx 1): recall over the remaining positive is perfect
    w = np.array([1.0, 0.0, 1.0, 1.0])
    m = evaluate_predictions(y, p, sample_weight=w, threshold=0.5)
    assert m["recall"] == 1.0
    assert abs(m["accuracy"] - 2.0 / 3.0) < 1e-12


def test_lower_threshold_raises_recall():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1])
    hi = evaluate_predictions(y, p, threshold=0.5)
    lo = evaluate_predictions(y, p, threshold=0.35)  # now 0.4 counts as act
    assert lo["recall"] > hi["recall"]
    assert lo["recall"] == 1.0


def test_degenerate_single_class_is_nan_not_crash():
    y = np.array([1, 1, 1])
    p = np.array([0.9, 0.8, 0.7])
    m = evaluate_predictions(y, p)
    assert np.isnan(m["auc"])
    assert np.isnan(m["avg_precision"])
    assert m["accuracy"] == 1.0  # all predicted act, all are act


def test_always_act_baseline_is_the_blind_primary():
    y = np.array([1, 1, 0, 0])
    m = always_act_baseline(y)
    assert m["recall"] == 1.0          # acts on everything
    assert m["precision"] == 0.5       # = base rate of positives
    assert m["auc"] == 0.5             # constant score = no discrimination


def test_threshold_sweep_is_monotone_in_recall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(0.5 + 0.3 * (2 * y - 1) + rng.normal(0, 0.2, 200), 0, 1)
    sweep = threshold_sweep(y, p, thresholds=[0.2, 0.4, 0.6, 0.8])
    assert list(sweep["threshold"]) == [0.2, 0.4, 0.6, 0.8]
    # recall is non-increasing as the threshold rises
    assert (sweep["recall"].diff().dropna() <= 1e-9).all()


def test_cross_val_evaluate_runs_purged_and_learns():
    rng = np.random.default_rng(1)
    n = 400
    h = n // 2
    x = np.vstack([rng.normal(-1.5, 0.7, (h, 3)), rng.normal(1.5, 0.7, (n - h, 3))])
    y = np.array([0] * h + [1] * (n - h))
    perm = rng.permutation(n)
    x, y = x[perm], y[perm]
    idx = pd.bdate_range("2020-01-01", periods=n)
    x = pd.DataFrame(x, index=idx)  # feature matrix is index-aligned to t1 for the splitter
    t1 = pd.Series(idx[np.minimum(np.arange(n) + 3, n - 1)], index=idx)
    cv = PurgedKFold(n_splits=3, t1=t1)
    res = cross_val_evaluate(lambda: make_xgb(seed=42), x, y, cv)
    assert len(res) == 3
    assert list(res["fold"]) == [0, 1, 2]
    assert (res["auc"] > 0.9).all()  # learns the separable OOS signal in every fold
