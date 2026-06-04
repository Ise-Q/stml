"""Dimensionality reduction — methodology spec

ClusterRepSelector — pick one feature per Mantegna cluster (the **medoid**,
the feature with smallest mean Spearman distance to its cluster-mates).
Deterministic, unsupervised, interpretable.

Used in two places:
  1. Feature-set reduction for the S4 multi-task NN if it's overfit-prone with
     the full 80-feature matrix (methodology spec calls out reducing-then-NN as the
     alternative shipped path).
  2. As a clean baseline for understanding which features ALONE carry signal
     after redundancy is removed.

Fit on TRAIN-ONLY rows (cluster computed on train Spearman) — re-applied to
val/test by simply selecting the same medoid columns. No fitted statistics
beyond the column list → fold-safe.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=RuntimeWarning)


class ClusterRepSelector:
    """Pick one medoid feature per Mantegna cluster.

    Parameters
    ----------
    max_clusters : int
        Maximum number of clusters (= number of features kept). The actual K
        is chosen by silhouette over [3, max_clusters] on the Mantegna
        distance matrix.
    """

    def __init__(self, *, max_clusters: int = 12, seed: int = 42) -> None:
        self.max_clusters = max_clusters
        self.seed = seed
        self.selected_features_: list[str] = []
        self.cluster_membership_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "ClusterRepSelector":
        from stml.experimental.importance import (
            cluster_features_mantegna,
            mantegna_distance_matrix,
        )

        # Filter to numeric, drop zero-variance.
        Xnum = X.select_dtypes(include=[np.number]).copy()
        var = Xnum.var(skipna=True)
        Xnum = Xnum.loc[:, var > 1e-10]
        if Xnum.shape[1] <= self.max_clusters:
            # Fewer features than clusters — return them all.
            self.selected_features_ = list(Xnum.columns)
            self.cluster_membership_ = pd.Series(
                np.arange(len(Xnum.columns)), index=Xnum.columns, name="cluster_id",
            )
            return self

        membership, dist_matrix, _ = cluster_features_mantegna(
            Xnum, k_range=(3, self.max_clusters)
        )
        self.cluster_membership_ = membership

        # For each cluster, pick the MEDOID — the feature with smallest
        # mean Spearman distance to its cluster-mates.
        medoids: list[str] = []
        for cid in membership.unique():
            members = membership.index[membership == cid].tolist()
            if len(members) == 1:
                medoids.append(members[0])
                continue
            sub_d = dist_matrix.loc[members, members].copy()
            # Mean distance to other cluster members (exclude self).
            arr = sub_d.values.copy()
            np.fill_diagonal(arr, np.nan)
            mean_d = pd.Series(np.nanmean(arr, axis=1), index=members)
            medoid = mean_d.idxmin()
            medoids.append(medoid)

        # Preserve original column order.
        self.selected_features_ = [c for c in X.columns if c in medoids]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_features_:
            raise RuntimeError("must call fit before transform")
        # Defensive: subset to present columns.
        present = [c for c in self.selected_features_ if c in X.columns]
        return X.loc[:, present].copy()

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
