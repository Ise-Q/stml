"""
importance.py
=============
Feature-importance analysis at the **cluster level** (the assignment's 10-mark
section, drawn from Programming Session 2 and AFML Ch. 8 + Ch. 6).

The motivation: per-feature importance is *biased* when features are
correlated. Random Forest's MDI splits the credit between two near-identical
features; permutation importance hides the contribution of a correlated
feature because permuting one still leaves a correlated proxy in place. The
fix is to compute importance at the **cluster** level: cluster correlated
features together, then attribute importance to the whole cluster.

Public API:
  - :func:`cluster_features`   -- hierarchical clustering on 1-|Spearman| distance
  - :func:`pick_optimal_k`     -- silhouette-based K selection
  - :func:`clustered_mdi`      -- sum of tree MDI within each cluster
  - :func:`clustered_mda`      -- permute all features in a cluster, measure drop

The two importance lenses agree on the cluster ranking most of the time —
disagreement is itself informative (e.g. an in-sample / out-of-sample split).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
)


# --------------------------------------------------------------------------- #
# 1. Feature clustering                                                       #
# --------------------------------------------------------------------------- #
def feature_distance_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Pairwise distance between features = ``1 - |Spearman correlation|``.

    Spearman is rank-based ⇒ robust to monotone transforms (so e.g. ``vol_5d``
    and ``z_vol_5d`` will sit at distance ≈ 0). Returns a symmetric
    ``(n_features, n_features)`` matrix indexed by feature name.
    """
    corr_arr, _ = spearmanr(X.values, axis=0)
    # Guard against NaN from constant columns.
    corr_arr = np.where(np.isnan(corr_arr), 0.0, corr_arr)
    if corr_arr.ndim == 0:  # single feature edge case
        corr_arr = np.array([[1.0]])
    np.fill_diagonal(corr_arr, 1.0)
    dist = 1.0 - np.abs(corr_arr)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    return pd.DataFrame(dist, index=X.columns, columns=X.columns)


def cluster_features(
    X: pd.DataFrame,
    n_clusters: Optional[int] = None,
    max_k: int = 12,
    linkage_method: str = "average",
    random_state: int = 42,  # noqa: ARG001  -- not used by hierarchical, kept for API
) -> tuple[dict[str, int], pd.DataFrame]:
    """Hierarchical clustering of features by ``1 - |Spearman|`` distance.

    Returns
    -------
    feature_to_cluster : dict[str, int]
        Mapping from feature name to cluster id (0-indexed).
    info : pd.DataFrame
        Per-cluster summary: ``size``, ``members``.
    """
    dist = feature_distance_matrix(X)
    # Convert to condensed form for scipy.cluster.hierarchy.
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method=linkage_method)

    if n_clusters is None:
        n_clusters = pick_optimal_k(X, Z, max_k=max_k)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1  # 0-indexed
    feature_to_cluster = {f: int(c) for f, c in zip(X.columns, labels)}

    # Build info table.
    members_by_c: dict[int, list[str]] = {}
    for f, c in feature_to_cluster.items():
        members_by_c.setdefault(c, []).append(f)
    info = pd.DataFrame([
        {"cluster": c, "size": len(m), "members": ", ".join(sorted(m))}
        for c, m in sorted(members_by_c.items())
    ])
    return feature_to_cluster, info


def pick_optimal_k(
    X: pd.DataFrame,
    linkage_matrix: np.ndarray,
    max_k: int = 12,
    min_k: int = 3,
) -> int:
    """Pick the number of clusters by maximising silhouette over the
    1-|Spearman| feature distance matrix."""
    from sklearn.metrics import silhouette_score

    dist = feature_distance_matrix(X).values
    p = dist.shape[0]
    max_k = min(max_k, p - 1)
    if max_k < min_k:
        return max(2, min_k)

    best_k, best_score = min_k, -np.inf
    for k in range(min_k, max_k + 1):
        labels = fcluster(linkage_matrix, t=k, criterion="maxclust") - 1
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(dist, labels, metric="precomputed")
        except ValueError:
            continue
        if score > best_score:
            best_k, best_score = k, float(score)
    return best_k


# --------------------------------------------------------------------------- #
# 2. Clustered MDI                                                            #
# --------------------------------------------------------------------------- #
def clustered_mdi(
    feature_importance: pd.Series,
    feature_to_cluster: dict[str, int],
) -> pd.DataFrame:
    """Sum per-feature MDI within each cluster.

    ``feature_importance`` is a Series like that returned by
    :meth:`XGBoostMeta.feature_importance`. The output is a DataFrame indexed
    by cluster id with columns ``mdi`` (sum) and ``mdi_share`` (normalised).
    """
    rows = []
    for f, imp in feature_importance.items():
        c = feature_to_cluster.get(f)
        if c is None:
            continue
        rows.append({"cluster": c, "feature": f, "mdi": imp})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["mdi", "mdi_share"]).rename_axis("cluster")
    g = df.groupby("cluster")["mdi"].sum().rename("mdi")
    out = g.to_frame()
    total = out["mdi"].sum()
    out["mdi_share"] = out["mdi"] / total if total > 0 else 0.0
    return out.sort_values("mdi_share", ascending=False)


# --------------------------------------------------------------------------- #
# 3. Clustered MDA (cluster-permutation importance)                           #
# --------------------------------------------------------------------------- #
def clustered_mda(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    feature_to_cluster: dict[str, int],
    scoring: str = "neg_log_loss",
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permute all features in a cluster together, measure metric drop.

    For each cluster:
      1. Compute the model's baseline score on (X, y).
      2. For ``n_repeats`` random permutations of the rows of ``X[cluster]``,
         compute the perturbed score.
      3. Importance = baseline_score - mean(perturbed_score).
         (Positive ⇒ permuting hurt the model ⇒ cluster mattered.)

    ``scoring``:
      - ``"neg_log_loss"`` : log-loss is negated so higher = better; importance
        = drop in (negated) log-loss = increase in log-loss when permuted.
      - ``"roc_auc"``      : AUC; importance = AUC drop.

    Parameters
    ----------
    model : object with ``predict_proba(X) -> ndarray[:, 1] or float``.
    X, y : evaluation data (typically OOS).
    feature_to_cluster : output of :func:`cluster_features`.
    n_repeats : permutations per cluster.
    random_state : reproducibility.

    Returns
    -------
    pd.DataFrame indexed by cluster with columns:
        ``mda``   = mean importance
        ``std``   = std across repeats
        ``rank``  = 1-indexed rank by mda (higher mda = lower rank)
    """
    rng = np.random.default_rng(random_state)
    if isinstance(model.predict_proba, type(list.append)):
        raise TypeError("model.predict_proba must be a method, not a function")

    def _score(p: np.ndarray) -> float:
        if scoring == "neg_log_loss":
            return -log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))
        if scoring == "roc_auc":
            return roc_auc_score(y, p)
        raise ValueError(f"Unknown scoring {scoring!r}")

    # Baseline.
    p_base = _get_proba(model, X)
    base = _score(p_base)

    # Group features by cluster.
    members_by_c: dict[int, list[str]] = {}
    for f, c in feature_to_cluster.items():
        if f in X.columns:
            members_by_c.setdefault(c, []).append(f)

    rows = []
    for c, members in members_by_c.items():
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            perm = rng.permutation(len(X_perm))
            # Permute the WHOLE block of cluster members with the SAME permutation —
            # this preserves intra-cluster correlation structure and so isolates
            # the cluster's marginal contribution.
            X_perm[members] = X[members].iloc[perm].values
            p_perm = _get_proba(model, X_perm)
            drops.append(base - _score(p_perm))
        drops_arr = np.asarray(drops)
        rows.append({
            "cluster": c,
            "mda": float(drops_arr.mean()),
            "std": float(drops_arr.std(ddof=1)) if len(drops_arr) > 1 else 0.0,
        })
    out = pd.DataFrame(rows).set_index("cluster").sort_values("mda", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def _get_proba(model, X) -> np.ndarray:
    """Adapter so we accept both sklearn-style (returns N x 2) and the stml-style
    (returns N x 1 of class-1 probability)."""
    p = model.predict_proba(X)
    if p.ndim == 2 and p.shape[1] == 2:
        return p[:, 1]
    return p


# --------------------------------------------------------------------------- #
# 4. Cluster-by-group summary                                                  #
# --------------------------------------------------------------------------- #
def cluster_economic_overlap(
    feature_to_cluster: dict[str, int],
    feature_groups: dict[str, str],
) -> pd.DataFrame:
    """Cross-tabulate the data-driven clusters against the declared economic
    groups (G1_vol, G2_trend, G3_meanrev, G4_microstructure, G5_signal, G6_regime,
    G7_calendar). Useful for the report: "do the data-driven clusters recover
    the economic groups?"
    """
    rows = []
    for f, c in feature_to_cluster.items():
        rows.append({"feature": f, "cluster": c,
                     "group": feature_groups.get(f, "OTHER")})
    df = pd.DataFrame(rows)
    return df.pivot_table(
        index="cluster", columns="group", values="feature",
        aggfunc="count", fill_value=0,
    )
