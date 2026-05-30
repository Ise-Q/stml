"""Cluster-level feature importance + the four Stage-3 bug fixes (RED-first).

- #1 max_features 'auto' -> 'sqrt' (the cluster MDI/SHAP forest; PS4 grid bug).
- #2 KFold(shuffle=True) -> PurgedKFold for cluster MDA (overlapping-label leakage).
- #3 cluster SHAP via TreeExplainer (PS2/sts-ml had MDI + PFI only) — the contribution.
- #4 Spearman distance 1-|rho| -> Mantegna sqrt(1-|rho|).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from alken_metamodel._vendor.cluster_feature_importance import compute_spearman_distance_matrix
from alken_metamodel.cluster_importance import (
    CLUSTER_RF_MAX_FEATURES,
    cluster_feature_importance,
    cluster_forest,
    cluster_shap,
    make_clusters,
)


def _signal_noise(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    X = pd.DataFrame(
        {
            "sig_0": base,
            "sig_1": 0.85 * base + 0.15 * rng.normal(0, 1, n),  # correlated signal cluster
            "noise_0": rng.normal(0, 1, n),
            "noise_1": rng.normal(0, 1, n),
        },
        index=idx,
    )
    p = 1.0 / (1.0 + np.exp(-2.0 * base))  # y driven by the signal cluster only
    y = pd.Series((rng.uniform(0, 1, n) < p).astype(int), index=idx)
    t1 = pd.Series(idx[np.minimum(np.arange(n) + 3, n - 1)], index=idx)
    return X, y, t1


# --- #4 Mantegna distance ---------------------------------------------------

def test_mantegna_distance_formula():
    X, _, _ = _signal_noise()
    D = compute_spearman_distance_matrix(X)
    corr = spearmanr(X).correlation
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    expected = np.sqrt(np.clip(1 - np.abs(corr), 0, None))
    np.testing.assert_allclose(D.to_numpy(), expected, atol=1e-12)
    assert D.loc["sig_0", "sig_0"] == 0.0
    rho = abs(corr[0, 1])
    assert D.loc["sig_0", "sig_1"] == pytest.approx(np.sqrt(1 - rho))
    # the fix: sqrt(1-|rho|) > the old 1-|rho| for any correlated pair (proves Mantegna)
    assert D.loc["sig_0", "sig_1"] > (1 - rho)


# --- #1 the forest uses 'sqrt', never 'auto' --------------------------------

def test_cluster_forest_uses_sqrt_not_auto():
    assert CLUSTER_RF_MAX_FEATURES == "sqrt"
    rf = cluster_forest(seed=42)
    assert rf.max_features == "sqrt"


# --- clustering groups the correlated signal features -----------------------

def test_clustering_groups_correlated_features():
    X, _, _ = _signal_noise()
    clusters, _ = make_clusters(X, seed=42, max_clusters=3)
    sig_cluster = next(c for c in clusters.values() if "sig_0" in c)
    assert "sig_1" in sig_cluster  # the correlated pair lands together
    assert any("sig" in f for f in sig_cluster)


# --- #3 cluster SHAP: per-cluster == sum of member |SHAP|; noise ~ 0 --------

def test_cluster_shap_sums_members_and_ranks_signal_over_noise():
    X, y, _ = _signal_noise()
    clusters, _ = make_clusters(X, seed=42, max_clusters=3)
    model = cluster_forest(seed=42).fit(X, y)
    shap_imp = cluster_shap(model, X, clusters)
    # normalised cluster importances sum to ~1 and are non-negative
    assert shap_imp.sum() == pytest.approx(1.0, abs=1e-9)
    assert (shap_imp >= 0).all()
    sig_cid = next(cid for cid, feats in clusters.items() if "sig_0" in feats)
    noise_cid = next(
        cid for cid, feats in clusters.items() if all("noise" in f for f in feats)
    )
    assert shap_imp[f"Cluster_{sig_cid}"] > shap_imp[f"Cluster_{noise_cid}"]


# --- #2 purged cluster MDA + the combined table -----------------------------

def test_cluster_feature_importance_table_ranks_signal_cluster_top():
    X, y, t1 = _signal_noise()
    table, clusters = cluster_feature_importance(X, y, t1, seed=42, n_splits=3, max_clusters=3)
    assert set(table.columns) == {"mdi", "mda", "shap"}
    sig_cid = next(cid for cid, feats in clusters.items() if "sig_0" in feats)
    noise_cid = next(
        cid for cid, feats in clusters.items() if all("noise" in f for f in feats)
    )
    # the signal cluster outranks the pure-noise cluster on every method
    for method in ("mdi", "mda", "shap"):
        assert table.loc[f"Cluster_{sig_cid}", method] > table.loc[f"Cluster_{noise_cid}", method]
