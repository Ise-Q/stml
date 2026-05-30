"""Tests for the act/skip model roster (Stage 2, RED-first).

The tree/linear horse-race core: elastic-net logistic + XGBoost + LightGBM, behind a uniform
``MetaClassifier`` interface (``fit(X, y, sample_weight=)`` / ``predict_act_proba(X)``). Both
label overlap (uniqueness weights) and class imbalance are carried by a SINGLE sample_weight
channel, so the comparison stays apples-to-apples.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from alken_metamodel.models import (
    balanced_sample_weight,
    make_elasticnet_logistic,
    make_lightgbm,
    make_xgb,
    tree_linear_roster,
)

ROSTER_NAMES = {"elasticnet_logistic", "xgboost", "lightgbm"}


def _separable(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    h = n // 2
    x0 = rng.normal(-1.5, 0.7, (h, 3))
    x1 = rng.normal(+1.5, 0.7, (n - h, 3))
    x = np.vstack([x0, x1])
    y = np.array([0] * h + [1] * (n - h))
    perm = rng.permutation(n)
    return x[perm], y[perm]


def test_roster_is_the_three_named_models():
    roster = tree_linear_roster(seed=42)
    assert set(roster) == ROSTER_NAMES
    for clf in roster.values():
        assert hasattr(clf, "fit") and hasattr(clf, "predict_act_proba")


def test_each_model_learns_separable_signal():
    x, y = _separable(seed=1)
    for name, clf in tree_linear_roster(seed=42).items():
        clf.fit(x, y)
        p = clf.predict_act_proba(x)
        assert p.shape == (len(y),)
        assert ((p >= 0.0) & (p <= 1.0)).all(), f"{name} produced out-of-range proba"
        assert roc_auc_score(y, p) > 0.95, f"{name} failed to learn the separable signal"


def test_fit_is_deterministic():
    x, y = _separable(seed=2)
    for clf_a, clf_b in zip(
        tree_linear_roster(seed=42).values(),
        tree_linear_roster(seed=42).values(),
        strict=True,
    ):
        clf_a.fit(x, y)
        clf_b.fit(x, y)
        np.testing.assert_allclose(clf_a.predict_act_proba(x), clf_b.predict_act_proba(x))


def test_sample_weight_is_consumed():
    """Zero-weighting a contradictory cluster makes every model ignore it."""
    rng = np.random.default_rng(3)
    x_pos = rng.uniform(1.0, 2.0, (150, 1))            # trustworthy: x>0 -> 1
    x_neg = rng.uniform(-2.0, -1.0, (150, 1))          # trustworthy: x<0 -> 0
    x_bad = rng.uniform(1.0, 2.0, (90, 1))             # contradiction: x>0 -> 0
    x = np.vstack([x_pos, x_neg, x_bad])
    y = np.concatenate([np.ones(150), np.zeros(150), np.zeros(90)])
    w_ignore = np.concatenate([np.ones(150), np.ones(150), np.zeros(90)])
    probe = np.array([[1.5]])
    for name in ROSTER_NAMES:
        factory = {
            "elasticnet_logistic": make_elasticnet_logistic,
            "xgboost": make_xgb,
            "lightgbm": make_lightgbm,
        }[name]
        clf_ignore = factory(seed=42)
        clf_naive = factory(seed=42)
        clf_ignore.fit(x, y, sample_weight=w_ignore)
        clf_naive.fit(x, y, sample_weight=np.ones(len(y)))
        p_ignore = clf_ignore.predict_act_proba(probe)[0]
        p_naive = clf_naive.predict_act_proba(probe)[0]
        assert p_ignore > p_naive + 0.1, f"{name} ignored sample_weight"
        assert p_ignore > 0.5, f"{name} should call x=1.5 'act' once the contradiction is muted"


def test_logistic_handles_nan_features_via_imputation():
    """The scaled (logistic) path imputes structural NaNs the pooled matrix will contain."""
    x, y = _separable(seed=10)
    x = x.copy()
    x[::7, 0] = np.nan  # inject structural NaNs into a feature column
    clf = make_elasticnet_logistic(seed=42)
    clf.fit(x, y)
    p = clf.predict_act_proba(x)
    assert np.isfinite(p).all()
    assert ((p >= 0.0) & (p <= 1.0)).all()
    assert roc_auc_score(y, p) > 0.9  # still learns despite the missing values


def test_balanced_sample_weight_equalises_class_mass():
    y = np.array([0, 0, 0, 0, 1, 1])
    w = balanced_sample_weight(y)
    assert w[y == 0].sum() == np.float64(w[y == 1].sum())
    # a base (uniqueness) weight multiplies through but classes stay balanced
    base = np.array([1.0, 2.0, 1.0, 1.0, 3.0, 1.0])
    wb = balanced_sample_weight(y, base=base)
    np.testing.assert_allclose(wb[y == 0].sum(), wb[y == 1].sum())
    # within-class proportionality of the base weights is preserved
    np.testing.assert_allclose(wb[4] / wb[5], base[4] / base[5])
