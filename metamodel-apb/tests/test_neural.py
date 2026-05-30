"""Neural act/skip variants (Stage 2 enrichment, RED-first).

torch-MLP first: must learn a separable signal, produce valid probabilities, consume
sample_weight, tolerate NaN (impute), and be deterministic (the grader re-runs).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from alken_metamodel.neural import TorchMLP


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
