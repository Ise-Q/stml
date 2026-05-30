"""Cluster-level feature importance for §4 (MDI + purged MDA + cluster SHAP).

Substitution effects make per-feature importance unreliable when features are correlated
(LdP 2020 Ch.6): a feature's importance is diluted across its correlated siblings. We cluster
features by Mantegna distance (``_vendor`` bug fix #4), then score importance per CLUSTER three
ways:

- **MDI** — sum of member tree importances (reused ``calculate_cluster_importance_mdi``).
- **MDA** — permute a whole cluster with one shared permutation across a **purged** CV
  (``_vendor`` bug fix #2: a ``PurgedKFold`` keyed on ``t1`` replaces the leaky shuffled KFold).
- **SHAP** — cluster SHAP via ``shap.TreeExplainer`` (bug fix #3, the contribution): mean |SHAP|
  per feature summed within each cluster. PS2/sts-ml shipped MDI + PFI only.

The forest used for MDI/SHAP fixes bug #1 — ``max_features='sqrt'`` (never the deprecated
``'auto'`` of the PS4 grid).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ._vendor.cluster_feature_importance import (
    OptimalClusterer,
    calculate_cluster_importance_mdi,
    calculate_cluster_importance_pfi,
    compute_spearman_distance_matrix,
)
from .cross_validation import PurgedKFold

CLUSTER_RF_MAX_FEATURES = "sqrt"  # BUG FIX #1: never the deprecated 'auto' (PS4 grid)


def cluster_forest(*, seed: int = 42, n_estimators: int = 200) -> RandomForestClassifier:
    """RandomForest for cluster MDI/SHAP — ``max_features='sqrt'`` (bug fix #1)."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=CLUSTER_RF_MAX_FEATURES,
        random_state=seed,
        n_jobs=1,
    )


def make_clusters(
    X: pd.DataFrame, *, seed: int = 42, max_clusters: int = 10
) -> tuple[dict, OptimalClusterer]:
    """Mantegna distance -> PCA -> optimal-K K-means clusters of the feature set."""
    dist = compute_spearman_distance_matrix(X)
    clusterer = OptimalClusterer(max_clusters=max_clusters, random_state=seed)
    clusterer.apply_pca(dist)
    clusterer.find_optimal_clusters()
    clusters = clusterer.apply_kmeans()
    return clusters, clusterer


def _positive_class_shap(shap_values) -> np.ndarray:
    """Reduce TreeExplainer output (list / 2-D / 3-D across shap versions) to (n, n_features)."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[-1])  # positive class
    arr = np.asarray(shap_values)
    if arr.ndim == 3:  # (n_samples, n_features, n_classes)
        return arr[:, :, -1]
    return arr


def cluster_shap(model, X: pd.DataFrame, clusters: dict) -> pd.Series:
    """BUG FIX #3 — cluster SHAP: per-cluster importance = sum of member mean |SHAP|, normalised.

    A pure-noise cluster contributes ~0; the sum-of-members decomposition is the additive
    analogue of cluster MDI/MDA (the §4 write-up contribution; nlr-cw §5).
    """
    import shap

    explainer = shap.TreeExplainer(model)
    values = _positive_class_shap(explainer.shap_values(X))
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=X.columns)
    out = {}
    for cid, feats in clusters.items():
        valid = [f for f in feats if f in X.columns]
        out[f"Cluster_{cid}"] = float(mean_abs[valid].sum()) if valid else 0.0
    importance = pd.Series(out)
    total = importance.sum()
    return importance / total if total > 0 else importance


def cluster_feature_importance(
    X: pd.DataFrame,
    y,
    t1: pd.Series,
    *,
    seed: int = 42,
    n_splits: int = 5,
    pct_embargo: float = 0.01,
    max_clusters: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """Cluster the features and score each cluster by MDI, purged MDA and cluster SHAP.

    ``X`` must be index-aligned to ``t1`` (event dates) so the purged MDA splits leak-free.
    Returns a per-cluster table (columns ``mdi``, ``mda``, ``shap``) and the cluster map.
    """
    y = pd.Series(np.asarray(y), index=X.index) if not isinstance(y, pd.Series) else y
    clusters, _ = make_clusters(X, seed=seed, max_clusters=max_clusters)

    mdi_model = cluster_forest(seed=seed).fit(X, y)
    mdi = calculate_cluster_importance_mdi(mdi_model, list(X.columns), clusters)

    cv = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)
    mda = calculate_cluster_importance_pfi(cluster_forest(seed=seed), X, y, clusters, cv)

    shap_imp = cluster_shap(mdi_model, X, clusters)

    table = pd.DataFrame(
        {"mdi": mdi["mean"], "mda": mda["mean"], "shap": shap_imp}
    ).astype(float)
    return table, clusters
