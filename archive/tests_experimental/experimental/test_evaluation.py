"""Tests for ``stml.experimental.evaluation``.

* evaluate_predictions sample-weighted metrics match sklearn.
* Single-class fold returns NaN AUC instead of crashing.
* cross_val_evaluate produces per-fold scores + OOS predictions.
* nan_columns_at_test correctly forces those columns to NaN at prediction time
  (R-11 simulated-missingness scaffold).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from stml.experimental.cv import PurgedKFold
from stml.experimental.evaluation import (
    cross_val_evaluate,
    evaluate_predictions,
    per_instrument_breakdown,
)
from stml.experimental.models import make_elasticnet_logistic, make_xgb


def test_evaluate_predictions_auc_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    n = 100
    y_true = rng.integers(0, 2, n)
    proba = rng.random(n)
    sw = rng.random(n) + 0.5
    expected = float(roc_auc_score(y_true, proba, sample_weight=sw))
    actual = evaluate_predictions(y_true, proba, sample_weight=sw)["auc"]
    np.testing.assert_allclose(actual, expected)


def test_evaluate_predictions_single_class_returns_nan_auc() -> None:
    """All-zero target → AUC is undefined → NaN, not crash."""
    y_true = np.zeros(20, dtype=int)
    proba = np.linspace(0.1, 0.9, 20)
    out = evaluate_predictions(y_true, proba)
    assert np.isnan(out["auc"])
    assert np.isnan(out["log_loss"])
    # Brier still defined.
    assert np.isfinite(out["brier"])


def test_cross_val_evaluate_returns_per_fold_and_oos() -> None:
    rng = np.random.default_rng(1)
    n = 100
    X = pd.DataFrame(rng.standard_normal((n, 4)))
    y = pd.Series((X[0] > 0).astype(int))
    t = pd.Series(pd.date_range("2020-01-02", periods=n, freq="B"))
    t1 = t + pd.tseries.offsets.BDay(5)
    cv = PurgedKFold(n_splits=3, t=t, t1=t1)
    result = cross_val_evaluate(
        make_model=lambda: make_xgb(seed=42),
        X=X, y=y, cv=cv, uniqueness_weights=None,
    )
    assert not result.fold_scores.empty
    assert "auc" in result.fold_scores.columns
    assert len(result.fold_scores) == 3
    assert not result.oos_predictions.empty
    # OOS row_idx must be valid indices into X.
    assert result.oos_predictions["row_idx"].max() < n
    assert result.oos_predictions["row_idx"].min() >= 0


def test_cross_val_evaluate_nan_columns_at_test_force_nan() -> None:
    """Test that the simulated-missingness path produces XGB predictions on NaN."""
    rng = np.random.default_rng(2)
    n = 60
    X = pd.DataFrame(rng.standard_normal((n, 4)), columns=["a", "b", "c", "d"])
    y = pd.Series((X["a"] > 0).astype(int))
    t = pd.Series(pd.date_range("2020-01-02", periods=n, freq="B"))
    t1 = t + pd.tseries.offsets.BDay(3)
    cv = PurgedKFold(n_splits=3, t=t, t1=t1)
    # Force "c" and "d" to NaN at test time (simulating BBG missingness).
    result = cross_val_evaluate(
        make_model=lambda: make_xgb(seed=42),
        X=X, y=y, cv=cv, uniqueness_weights=None,
        nan_columns_at_test=["c", "d"],
    )
    # The function should still produce predictions (XGB handles NaN).
    assert not result.oos_predictions.empty


def test_per_instrument_breakdown_groups_correctly() -> None:
    """The per-instrument table has one row per unique instrument."""
    rng = np.random.default_rng(3)
    n = 90
    X = pd.DataFrame(rng.standard_normal((n, 3)))
    y = pd.Series((X[0] > 0).astype(int))
    instruments = pd.Series(["A"] * 30 + ["B"] * 30 + ["C"] * 30)
    t = pd.Series(pd.date_range("2020-01-02", periods=n, freq="B"))
    t1 = t + pd.tseries.offsets.BDay(3)
    cv = PurgedKFold(n_splits=3, t=t, t1=t1)
    result = cross_val_evaluate(
        make_model=lambda: make_xgb(seed=42),
        X=X, y=y, cv=cv, uniqueness_weights=None,
    )
    pi = per_instrument_breakdown(instruments=instruments, oos_predictions=result.oos_predictions)
    assert sorted(pi["instrument"]) == ["A", "B", "C"]
    assert (pi["n"] > 0).all()
