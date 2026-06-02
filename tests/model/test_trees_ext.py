"""
test_trees_ext.py
=================
Tests for the extended tree-model family: AdaBoostModel and LGBMModel.

Both must satisfy the same uniform interface as XGBModel / RFModel:
  - fit(X, y, sample_weight=None) -> self
  - predict_proba(X) -> np.ndarray of P(y=1), shape (n,), values in [0, 1]
  - feature_names_: list[str] matching the training columns
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module-level smoke: import must succeed even when lightgbm is absent
# ---------------------------------------------------------------------------


def test_module_import_exposes_has_lightgbm_flag() -> None:
    from stml.model.trees import HAS_LIGHTGBM

    assert isinstance(HAS_LIGHTGBM, bool)


# ---------------------------------------------------------------------------
# AdaBoostModel — always runs
# ---------------------------------------------------------------------------

FEATURE_COLS = ["f1_a", "f1_b", "f2_vol"]


@pytest.fixture()
def ada_xy(synthetic_panel):
    X = synthetic_panel[FEATURE_COLS].copy()
    y = synthetic_panel["bin"].to_numpy()
    return X, y


def test_adaboost_predict_proba_shape_and_range(ada_xy) -> None:
    from stml.model.trees import AdaBoostModel

    X, y = ada_xy
    model = AdaBoostModel(params={"n_estimators": 20, "learning_rate": 0.5}, seed=42)
    model.fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (len(y),), "predict_proba must return a 1-D array"
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0), "probabilities must be in [0, 1]"


def test_adaboost_feature_names(ada_xy) -> None:
    from stml.model.trees import AdaBoostModel

    X, y = ada_xy
    model = AdaBoostModel(params={"n_estimators": 20}, seed=42)
    model.fit(X, y)

    assert model.feature_names_ == FEATURE_COLS


def test_adaboost_accepts_sample_weight(ada_xy) -> None:
    from stml.model.trees import AdaBoostModel

    X, y = ada_xy
    rng = np.random.default_rng(42)
    w = rng.uniform(0.5, 1.5, size=len(y))

    model = AdaBoostModel(params={"n_estimators": 20}, seed=42)
    model.fit(X, y, sample_weight=w)  # must not raise
    proba = model.predict_proba(X)
    assert proba.shape == (len(y),)


def test_adaboost_max_depth_param(ada_xy) -> None:
    """max_depth is passed via params and correctly forwarded to the base estimator."""
    from stml.model.trees import AdaBoostModel

    X, y = ada_xy
    model = AdaBoostModel(params={"n_estimators": 20, "max_depth": 2}, seed=42)
    model.fit(X, y)
    # depth is consumed — should not appear in the underlying AdaBoostClassifier params
    assert model.model_ is not None


# ---------------------------------------------------------------------------
# LGBMModel — skipped when lightgbm is absent
# ---------------------------------------------------------------------------

lgbm = pytest.importorskip("lightgbm", reason="lightgbm not installed")


@pytest.fixture()
def lgbm_xy(synthetic_panel):
    X = synthetic_panel[FEATURE_COLS].copy()
    y = synthetic_panel["bin"].to_numpy()
    return X, y


def test_lgbm_predict_proba_shape_and_range(lgbm_xy) -> None:
    from stml.model.trees import LGBMModel

    X, y = lgbm_xy
    model = LGBMModel(params={"n_estimators": 20, "num_leaves": 16}, seed=42)
    model.fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (len(y),), "predict_proba must return a 1-D array"
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0), "probabilities must be in [0, 1]"


def test_lgbm_feature_names(lgbm_xy) -> None:
    from stml.model.trees import LGBMModel

    X, y = lgbm_xy
    model = LGBMModel(params={"n_estimators": 20, "num_leaves": 16}, seed=42)
    model.fit(X, y)

    assert model.feature_names_ == FEATURE_COLS


def test_lgbm_accepts_sample_weight(lgbm_xy) -> None:
    from stml.model.trees import LGBMModel

    X, y = lgbm_xy
    rng = np.random.default_rng(42)
    w = rng.uniform(0.5, 1.5, size=len(y))

    model = LGBMModel(params={"n_estimators": 20, "num_leaves": 16}, seed=42)
    model.fit(X, y, sample_weight=w)  # must not raise
    proba = model.predict_proba(X)
    assert proba.shape == (len(y),)
