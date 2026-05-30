"""Dimensionality reduction for the NN horse-race (EX.2).

The Variable Selection Network builds one Gated Residual Network *per feature*, so the ~60+
pooled features are intractable at CV scale. EX.2 reduces the feature set three ways and the
write-up compares them; **cluster-representative selection is the promoted reducer** (S3.7):

- ``ClusterRepSelector`` — one feature per Mantegna §4 cluster (the cluster MEDOID, the feature
  with minimum total Spearman-distance to its cluster-mates). Deterministic, unsupervised,
  interpretable, and it re-uses the exact §4 clustering — so the reduced set *is* the cluster
  structure, not an opaque projection.
- ``PCAReducer`` — variance-threshold PCA (standardised). Dense, not interpretable.
- ``AutoencoderReducer`` — a small deterministic torch autoencoder; non-linear, over-fit-prone
  at this sample size, kept for the comparison only.

Every reducer is **fit on train only** (the basis/selection is frozen at fit time and re-applied
to OOS rows), median-imputes leading NaNs, and reduces strictly below the input dimension.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ._vendor.cluster_feature_importance import compute_spearman_distance_matrix
from .cluster_importance import make_clusters
from .seeding import set_seeds

VARIANCE_FLOOR = 1e-12


def _median_impute(X: pd.DataFrame) -> pd.DataFrame:
    """Train-only median imputation that keeps every column (all-NaN -> 0)."""
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    arr = imp.fit_transform(X)
    return pd.DataFrame(arr, index=X.index, columns=X.columns)


def _medoid(dist: pd.DataFrame, feats: list[str]) -> str:
    """Cluster representative = feature with min total distance to its mates (name tie-break)."""
    if len(feats) == 1:
        return feats[0]
    sub = dist.loc[feats, feats]
    totals = sub.sum(axis=1)
    return min(feats, key=lambda f: (float(totals[f]), f))


class ClusterRepSelector:
    """Select one representative feature per Mantegna cluster (promoted EX.2 reducer)."""

    def __init__(self, *, seed: int = 42, max_clusters: int = 10):
        self.seed = seed
        self.max_clusters = max_clusters
        self.selected_: list[str] = []
        self.clusters_: dict = {}

    def fit(self, X: pd.DataFrame, y=None) -> ClusterRepSelector:
        imputed = _median_impute(X)
        # drop zero-variance columns: spearmanr returns NaN for a constant series, which would
        # poison the distance matrix; such features carry no information and cannot represent.
        keep = [c for c in imputed.columns if imputed[c].var() > VARIANCE_FLOOR]
        clustering_input = imputed[keep]
        max_k = min(self.max_clusters, max(2, clustering_input.shape[1] - 1))
        clusters, _ = make_clusters(clustering_input, seed=self.seed, max_clusters=max_k)
        dist = compute_spearman_distance_matrix(clustering_input)

        reps: set[str] = set()
        non_empty: dict = {}
        for cid, feats in clusters.items():
            valid = [f for f in feats if f in dist.index]
            if not valid:
                continue
            non_empty[cid] = valid
            reps.add(_medoid(dist, valid))
        self.clusters_ = non_empty
        self.selected_ = [c for c in X.columns if c in reps]  # original column order
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_]

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class PCAReducer:
    """Variance-threshold PCA on standardised, median-imputed features (fit train-only)."""

    def __init__(self, *, var_threshold: float = 0.95, n_components=None, seed: int = 42):
        self.var_threshold = var_threshold
        self.n_components = n_components
        self.seed = seed

    def fit(self, X: pd.DataFrame, y=None) -> PCAReducer:
        self._imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
        self._scaler = StandardScaler().fit(self._imp.transform(X))
        n = self.n_components if self.n_components is not None else self.var_threshold
        self._pca = PCA(n_components=n, svd_solver="full", random_state=self.seed)
        self._pca.fit(self._scaler.transform(self._imp.transform(X)))
        self.n_components_ = int(self._pca.n_components_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        comps = self._pca.transform(self._scaler.transform(self._imp.transform(X)))
        cols = [f"pc_{i}" for i in range(comps.shape[1])]
        return pd.DataFrame(comps, index=X.index, columns=cols)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class AutoencoderReducer:
    """Small deterministic torch autoencoder; comparison-only (over-fit-prone at this N)."""

    def __init__(self, *, latent_dim: int = 8, epochs: int = 200, lr: float = 1e-3, seed: int = 42):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.seed = seed

    def fit(self, X: pd.DataFrame, y=None) -> AutoencoderReducer:
        import torch
        from torch import nn

        n_features = X.shape[1]
        if self.latent_dim >= n_features:
            raise ValueError(
                f"latent_dim ({self.latent_dim}) must be < n_features ({n_features})"
            )
        set_seeds(self.seed)
        torch.manual_seed(self.seed)

        self._imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
        self._scaler = StandardScaler().fit(self._imp.transform(X))
        z = self._scaler.transform(self._imp.transform(X)).astype(np.float32)
        tensor = torch.from_numpy(z)

        hidden = max(self.latent_dim * 2, n_features // 2)
        self._encoder = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, self.latent_dim)
        )
        decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_features)
        )
        params = list(self._encoder.parameters()) + list(decoder.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        loss_fn = nn.MSELoss()
        self._encoder.train()
        decoder.train()
        for _ in range(self.epochs):  # full-batch -> deterministic
            opt.zero_grad()
            loss_fn(decoder(self._encoder(tensor)), tensor).backward()
            opt.step()
        self._encoder.eval()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        import torch

        z = self._scaler.transform(self._imp.transform(X)).astype(np.float32)
        with torch.no_grad():
            latent = self._encoder(torch.from_numpy(z)).numpy()
        cols = [f"ae_{i}" for i in range(latent.shape[1])]
        return pd.DataFrame(latent, index=X.index, columns=cols)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
