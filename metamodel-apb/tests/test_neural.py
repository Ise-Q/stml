"""Neural act/skip variants (Stage 2 enrichment, RED-first).

torch-MLP first: must learn a separable signal, produce valid probabilities, consume
sample_weight, tolerate NaN (impute), and be deterministic (the grader re-runs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from alken_metamodel.dim_reduction import ClusterRepSelector
from alken_metamodel.neural import (
    KerasVSN,
    ReducedEstimator,
    TorchMLP,
    TorchVSN,
    default_roster,
    full_roster,
    neural_roster,
)


def _separable(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    h = n // 2
    x = np.vstack([rng.normal(-1.5, 0.7, (h, 4)), rng.normal(1.5, 0.7, (n - h, 4))])
    y = np.array([0] * h + [1] * (n - h))
    perm = rng.permutation(n)
    return x[perm], y[perm]


def test_torch_mlp_learns_separable_signal():
    x, y = _separable(seed=1)
    clf = TorchMLP(seed=42, epochs=120)
    clf.fit(x, y)
    p = clf.predict_act_proba(x)
    assert p.shape == (len(y),)
    assert ((p >= 0.0) & (p <= 1.0)).all()
    assert roc_auc_score(y, p) > 0.9


def test_torch_mlp_is_deterministic():
    x, y = _separable(seed=2)
    a = TorchMLP(seed=42, epochs=80)
    a.fit(x, y)
    b = TorchMLP(seed=42, epochs=80)
    b.fit(x, y)
    np.testing.assert_allclose(a.predict_act_proba(x), b.predict_act_proba(x), rtol=1e-5, atol=1e-6)


def test_torch_mlp_handles_nan_and_sample_weight():
    x, y = _separable(seed=3)
    x = x.copy()
    x[::9, 0] = np.nan  # structural NaN -> imputed
    w = np.where(y == 1, 3.0, 1.0)  # up-weight positives
    clf = TorchMLP(seed=42, epochs=80)
    clf.fit(x, y, sample_weight=w)
    p = clf.predict_act_proba(x)
    assert np.isfinite(p).all()
    assert ((p >= 0.0) & (p <= 1.0)).all()


def test_torch_mlp_has_roster_interface():
    clf = TorchMLP(seed=42)
    assert clf.name == "torch_mlp"
    assert hasattr(clf, "fit") and hasattr(clf, "predict_act_proba")


# --- torch VSN --------------------------------------------------------------

def test_torch_vsn_learns_and_is_deterministic():
    x, y = _separable(seed=4)
    a = TorchVSN(seed=42, epochs=120)
    a.fit(x, y)
    pa = a.predict_act_proba(x)
    assert pa.shape == (len(y),)
    assert ((pa >= 0.0) & (pa <= 1.0)).all()
    assert roc_auc_score(y, pa) > 0.85
    b = TorchVSN(seed=42, epochs=120)
    b.fit(x, y)
    np.testing.assert_allclose(pa, b.predict_act_proba(x), rtol=1e-4, atol=1e-5)


def test_torch_vsn_handles_sample_weight():
    x, y = _separable(seed=5)
    w = np.where(y == 1, 2.0, 1.0)
    clf = TorchVSN(seed=42, epochs=60)
    clf.fit(x, y, sample_weight=w)
    p = clf.predict_act_proba(x)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


# --- Keras VSN (best-effort determinism; documented TF caveat) --------------

def test_keras_vsn_learns_separable_signal():
    x, y = _separable(seed=6)
    clf = KerasVSN(seed=42, epochs=50)
    clf.fit(x, y)
    p = clf.predict_act_proba(x)
    assert p.shape == (len(y),)
    assert ((p >= 0.0) & (p <= 1.0)).all()
    assert roc_auc_score(y, p) > 0.7


# --- rosters ----------------------------------------------------------------

def test_rosters_have_expected_members():
    assert set(neural_roster()) == {"torch_mlp", "torch_vsn", "keras_vsn"}
    assert set(full_roster()) == {
        "elasticnet_logistic",
        "xgboost",
        "lightgbm",
        "torch_mlp",
        "torch_vsn",
        "keras_vsn",
    }


# --- S3.7: torch default roster + cluster-rep reduction ---------------------

def _corr_frame(n: int = 160, seed: int = 0):
    """A wide DataFrame with two tight correlation blocks + noise, and a learnable label."""
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=n), rng.normal(size=n)
    cols = {f"a{k}": 0.9 * a + 0.1 * rng.normal(size=n) for k in range(5)}
    cols |= {f"b{k}": 0.9 * b + 0.1 * rng.normal(size=n) for k in range(5)}
    cols["n0"], cols["n1"] = rng.normal(size=n), rng.normal(size=n)
    X = pd.DataFrame(cols, index=pd.bdate_range("2020-01-01", periods=n))
    y = (0.9 * a + 0.4 * rng.normal(size=n) > 0).astype(float)
    return X, y


def test_default_roster_is_torch_only_no_keras():
    roster = default_roster(seed=42)
    assert set(roster) == {"elasticnet_logistic", "xgboost", "lightgbm", "torch_mlp", "torch_vsn"}
    assert "keras_vsn" not in roster  # TF non-determinism kept off the selectable path


def test_reduced_estimator_reduces_and_predicts():
    X, y = _corr_frame()
    est = ReducedEstimator(
        TorchVSN(seed=42, epochs=12), ClusterRepSelector(seed=42, max_clusters=6)
    )
    est.fit(X, y)
    assert 0 < len(est.reducer.selected_) < X.shape[1]  # the VSN saw a reduced feature set
    p = est.predict_act_proba(X)
    assert p.shape == (len(X),)
    assert ((p >= 0.0) & (p <= 1.0)).all()
    assert est.name == "torch_vsn"  # roster key stays stable through the wrapper


def test_reduced_estimator_is_deterministic():
    X, y = _corr_frame(n=120, seed=2)

    def _mk():
        return ReducedEstimator(
            TorchVSN(seed=42, epochs=10), ClusterRepSelector(seed=42, max_clusters=5)
        )

    a = _mk().fit(X, y).predict_act_proba(X)
    b = _mk().fit(X, y).predict_act_proba(X)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
