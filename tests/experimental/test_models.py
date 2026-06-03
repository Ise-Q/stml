"""Tests for ``stml.experimental.models``.

Verifies:
* balanced_sample_weight: per-class total = n / n_classes invariant.
* All three estimators fit + predict with the uniform interface.
* XGBoost handles NaN inputs natively (critical for R-11 simulated missingness).
* feature_importance() works on the tree estimators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.experimental.models import (
    balanced_sample_weight,
    default_roster,
    make_elasticnet_logistic,
    make_random_forest,
    make_xgb,
)


# ---------------------------------------------------------------------------
# balanced_sample_weight.
# ---------------------------------------------------------------------------


def test_balanced_sample_weight_per_class_mass_equal() -> None:
    """Each class's total weight should equal n / n_classes."""
    y = np.array([0, 0, 0, 0, 1, 1])  # 4 zeros, 2 ones
    w = balanced_sample_weight(y, base=None)
    assert pytest.approx(3.0) == w[y == 0].sum()  # 6 / 2 = 3 per class
    assert pytest.approx(3.0) == w[y == 1].sum()


def test_balanced_sample_weight_preserves_base_proportions() -> None:
    """Within a class, base proportions are preserved."""
    y = np.array([1, 1, 1, 1])
    base = np.array([1.0, 2.0, 3.0, 4.0])
    w = balanced_sample_weight(y, base=base)
    # All same class → weights = base * target/total; ratios preserved.
    ratios_w = w / w[0]
    ratios_base = base / base[0]
    np.testing.assert_allclose(ratios_w, ratios_base)


def test_balanced_sample_weight_degenerate_single_class() -> None:
    """When y has only one class, return base unchanged."""
    y = np.zeros(5, dtype=int)
    base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w = balanced_sample_weight(y, base=base)
    np.testing.assert_allclose(w, base)


# ---------------------------------------------------------------------------
# Estimators.
# ---------------------------------------------------------------------------


@pytest.fixture
def synth_xy():
    """100-row easy classification problem."""
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)])
    # Linear decision boundary on the first feature.
    y = pd.Series((X["f0"] + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


def test_logistic_fits_and_predicts(synth_xy) -> None:
    X, y = synth_xy
    m = make_elasticnet_logistic(seed=42)
    m.fit(X, y)
    proba = m.predict_act_proba(X)
    assert proba.shape == (len(X),)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_xgb_fits_and_predicts(synth_xy) -> None:
    X, y = synth_xy
    m = make_xgb(seed=42)
    m.fit(X, y)
    proba = m.predict_act_proba(X)
    assert proba.shape == (len(X),)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_random_forest_fits_and_predicts(synth_xy) -> None:
    X, y = synth_xy
    m = make_random_forest(seed=42)
    m.fit(X, y)
    proba = m.predict_act_proba(X)
    assert proba.shape == (len(X),)


def test_xgb_handles_nan_inputs(synth_xy) -> None:
    """XGBoost must handle NaN features natively — critical for R-11 ablation."""
    X, y = synth_xy
    X_with_nan = X.copy()
    # Force the BBG-like columns to NaN.
    X_with_nan.iloc[:, -2:] = np.nan
    m = make_xgb(seed=42)
    m.fit(X_with_nan, y)
    # And predict on NaN-laden inputs.
    proba = m.predict_act_proba(X_with_nan)
    assert proba.shape == (len(X),)
    assert np.isfinite(proba).all()


def test_logistic_imputes_nan_via_scale_pipeline(synth_xy) -> None:
    """Elastic-net path median-imputes NaNs before standardising."""
    X, y = synth_xy
    X_with_nan = X.copy()
    X_with_nan.iloc[0, 0] = np.nan
    m = make_elasticnet_logistic(seed=42)
    m.fit(X_with_nan, y)
    proba = m.predict_act_proba(X_with_nan)
    assert np.isfinite(proba).all()


def test_default_roster_returns_four_estimators() -> None:
    """Roster includes linear (logistic) + 2 boosted-tree (xgb, lgbm) + bagged (rf)."""
    roster = default_roster(seed=42)
    assert set(roster.keys()) == {
        "elasticnet_logistic", "xgboost", "lightgbm", "random_forest",
    }


def test_xgb_feature_importance(synth_xy) -> None:
    X, y = synth_xy
    m = make_xgb(seed=42)
    m.fit(X, y)
    imp = m.feature_importance()
    assert imp is not None
    assert len(imp) == X.shape[1]


def test_xgb_deterministic_refit(synth_xy) -> None:
    """Two XGBoost fits with the same seed produce identical predictions."""
    X, y = synth_xy
    m1 = make_xgb(seed=42)
    m2 = make_xgb(seed=42)
    m1.fit(X, y)
    m2.fit(X, y)
    np.testing.assert_allclose(m1.predict_act_proba(X), m2.predict_act_proba(X))
