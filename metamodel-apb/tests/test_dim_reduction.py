"""EX.2 dimensionality reduction (RED-first).

Three reducers feed the NN horse-race (S3.7): cluster-representative selection (the promoted
one — one feature per Mantegna §4 cluster), PCA, and a small torch autoencoder. The invariants
that matter: each reducer is fit on TRAIN only (no test-time fitting), the reduced dimension is
strictly smaller than the full feature set, cluster-rep picks exactly one feature per cluster,
and the autoencoder is deterministic under a fixed seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alken_metamodel.dim_reduction import (
    AutoencoderReducer,
    ClusterRepSelector,
    PCAReducer,
)


def _synthetic_features(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Two tight correlation clusters (A: a0/a1/a2, B: b0/b1) + two independent noise columns."""
    rng = np.random.default_rng(seed)
    base_a = rng.normal(size=n)
    base_b = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "a0": base_a,
            "a1": 0.96 * base_a + 0.04 * rng.normal(size=n),
            "a2": 0.94 * base_a + 0.06 * rng.normal(size=n),
            "b0": base_b,
            "b1": 0.95 * base_b + 0.05 * rng.normal(size=n),
            "n0": rng.normal(size=n),
            "n1": rng.normal(size=n),
        },
        index=pd.bdate_range("2020-01-01", periods=n),
    )
    return df


# --- ClusterRepSelector -----------------------------------------------------

def test_cluster_rep_picks_exactly_one_per_cluster():
    X = _synthetic_features()
    sel = ClusterRepSelector(seed=42, max_clusters=6).fit(X)
    assert len(sel.clusters_) >= 2  # the two correlation blocks are separated
    # exactly one representative per (non-empty) cluster
    for _cid, feats in sel.clusters_.items():
        chosen = [f for f in sel.selected_ if f in feats]
        assert len(chosen) == 1
    assert len(sel.selected_) == len(sel.clusters_)
    assert set(sel.selected_).issubset(set(X.columns))


def test_cluster_rep_reduces_dimension():
    X = _synthetic_features()
    sel = ClusterRepSelector(seed=42, max_clusters=6).fit(X)
    assert 0 < len(sel.selected_) < X.shape[1]
    out = sel.transform(X)
    assert list(out.columns) == sel.selected_  # transform preserves fit-time selection order
    assert out.shape == (len(X), len(sel.selected_))


def test_cluster_rep_fit_is_train_only():
    X = _synthetic_features(n=400)
    train, test = X.iloc[:300], X.iloc[300:]
    sel = ClusterRepSelector(seed=42, max_clusters=6).fit(train)
    chosen = list(sel.selected_)
    # transforming the held-out slice re-uses the fit-time columns — no re-selection on test
    assert list(sel.transform(test).columns) == chosen


def test_cluster_rep_handles_nan_and_constant_columns():
    X = _synthetic_features()
    X = X.copy()
    X.iloc[:5, 0] = np.nan          # leading NaNs (as real causal features have)
    X["const"] = 1.0                # zero-variance column must not crash clustering
    sel = ClusterRepSelector(seed=42, max_clusters=6).fit(X)
    assert "const" not in sel.selected_  # a constant feature is never a representative
    assert len(sel.selected_) >= 2


# --- PCAReducer -------------------------------------------------------------

def test_pca_reduces_dimension_and_names_components():
    X = _synthetic_features()
    r = PCAReducer(var_threshold=0.9, seed=42).fit(X)
    out = r.transform(X)
    assert out.shape[1] < X.shape[1]
    assert out.shape[0] == len(X)
    assert list(out.columns) == [f"pc_{i}" for i in range(out.shape[1])]


def test_pca_fit_is_train_only():
    X = _synthetic_features(n=400)
    train, test = X.iloc[:300], X.iloc[300:]
    r1 = PCAReducer(var_threshold=0.9, seed=42).fit(train)
    r2 = PCAReducer(var_threshold=0.9, seed=42).fit(train)
    # basis fit on train only -> transforming test is reproducible and row-wise independent
    pd.testing.assert_frame_equal(r1.transform(test), r2.transform(test))
    pd.testing.assert_frame_equal(r1.transform(X).iloc[300:], r1.transform(test))


# --- AutoencoderReducer -----------------------------------------------------

def test_autoencoder_is_deterministic_and_reduces():
    X = _synthetic_features(n=200)
    a = AutoencoderReducer(latent_dim=3, epochs=40, seed=42).fit(X)
    b = AutoencoderReducer(latent_dim=3, epochs=40, seed=42).fit(X)
    pd.testing.assert_frame_equal(a.transform(X), b.transform(X))
    out = a.transform(X)
    assert out.shape == (len(X), 3)
    assert out.shape[1] < X.shape[1]


def test_autoencoder_latent_dim_must_be_smaller():
    X = _synthetic_features(n=120)
    with pytest.raises(ValueError):
        AutoencoderReducer(latent_dim=X.shape[1] + 1, epochs=5, seed=42).fit(X)
