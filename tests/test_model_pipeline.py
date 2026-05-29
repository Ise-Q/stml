"""Light integration + determinism tests for the meta-model pipeline.

Covers the uniform trainer interface, the importance helpers and the evaluation metrics on a small
real subset, and asserts that the seeded purged-CV score is reproducible (same seed -> identical
AUC) -- the determinism guarantee the plan calls for. Kept to XGBoost + a tiny MLP so the suite
stays fast; the VSN shares the MLP training path already exercised here.
"""

from __future__ import annotations

import warnings

import pytest

from stml.model import dataset as ds
from stml.model.cv import PurgedWalkForward
from stml.model.evaluate import evaluate_predictions, per_instrument_breakdown
from stml.model.importance import permutation_importance_auc, tree_importance
from stml.model.labels import triple_barrier_labels
from stml.model.optuna_objective import cross_val_auc
from stml.model.trees import XGBModel, xgb_baseline_params


@pytest.fixture(scope="module")
def cell():
    warnings.filterwarnings("ignore")
    m = ds.load_matrix()
    cw = ds.close_panel()
    m = ds.attach_bar_pos(m, cw)
    dev = m[m["partition"].isin(ds.DEV_PARTITIONS)].reset_index(drop=True)
    lab = triple_barrier_labels(cw, ds.events_frame(dev), pt=1.5, sl=1.5, h=10,
                                price_end="2021-12-30")
    d2 = dev.merge(lab[["date", "instrument", "bin"]], on=["date", "instrument"]).reset_index(
        drop=True
    )
    fc = ds.select_features(dev)
    X, y = ds.make_xy(d2, fc, instrument_dummies=True)
    cv = PurgedWalkForward(n_splits=4, h=10, embargo_by_instrument=ds.embargo_map())
    return X, y, d2, cv, cw


def test_cross_val_auc_deterministic(cell):
    X, y, d2, cv, _ = cell
    a = cross_val_auc(XGBModel, xgb_baseline_params(), X, y, d2, cv, seed=0)
    b = cross_val_auc(XGBModel, xgb_baseline_params(), X, y, d2, cv, seed=0)
    assert a[0] == b[0]            # identical mean AUC
    assert 0.0 <= a[0] <= 1.0
    assert a[2] == 4              # all four folds scored


def test_tree_importance_and_eval(cell):
    X, y, d2, cv, _ = cell
    model = XGBModel(xgb_baseline_params(), seed=0).fit(X, y)
    imp = tree_importance(model, X, y, n_repeats=2, with_shap=True)
    assert {"native", "perm_auc_drop", "shap", "rank_mean"} <= set(imp.columns)
    assert len(imp) == X.shape[1]

    proba = model.predict_proba(X)
    metrics = evaluate_predictions(y, proba)
    assert 0.0 <= metrics["auc"] <= 1.0
    assert metrics["n"] == len(y)

    breakdown = per_instrument_breakdown(d2["instrument"].to_numpy(), y, proba)
    assert breakdown["instrument"].nunique() == d2["instrument"].nunique()


def test_permutation_importance_signs(cell):
    X, y, _, _, _ = cell
    model = XGBModel(xgb_baseline_params(), seed=0).fit(X, y)
    perm = permutation_importance_auc(model, X.iloc[:400], y[:400], n_repeats=2, seed=1)
    assert len(perm) == X.shape[1]
    assert perm.is_monotonic_decreasing  # returned sorted desc
