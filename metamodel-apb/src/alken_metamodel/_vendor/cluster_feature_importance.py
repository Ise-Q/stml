"""Cluster-level feature importance (substitution effects).

Source: T3.03_PS2_Solutions.ipynb (Madmoun, L2 "The Correlation Problem in
Feature Importance"). Plotting side-effects from the notebook's OptimalClusterer
are removed so the module is importable; the algorithm is unchanged.

Pipeline: Spearman distance (1 - |rho|) -> PCA -> K-means (K by silhouette) ->
GMM (soft membership) -> cluster-level MDI and PFI. The crux is that cluster PFI
permutes all features in a cluster with the SAME permutation index, removing the
information that leaks through correlated siblings.

Stack: numpy, pandas, scipy, scikit-learn.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, log_loss)


def compute_spearman_distance_matrix(X):
    """Mantegna metric distance d = sqrt(1 - |Spearman rho|), symmetrised, unit diagonal.

    BUG FIX #4 (Stage 3): the source used the non-metric ``1 - |rho|``. The Mantegna (1999)
    ultrametric ``sqrt(1 - |rho|)`` is a proper distance, which is what the downstream
    hierarchical/feature clustering assumes (LdP 2020 Ch.4; nlr-cw §5).
    """
    corr = spearmanr(X).correlation
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    dist = np.sqrt(np.clip(1 - np.abs(corr), 0, None))  # BUG FIX #4: Mantegna (was 1 - |corr|)
    names = X.columns if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(dist, index=names, columns=names)


class OptimalClusterer:
    """PCA -> optimal-K K-means -> GMM soft clustering of a distance matrix."""

    def __init__(self, variance_threshold=0.95, max_clusters=10, random_state=42):
        self.variance_threshold = variance_threshold
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.features = self.pca_data = self.optimal_k = self.kmeans = None

    def apply_pca(self, dist_matrix):
        self.features = dist_matrix.index
        pca = PCA(random_state=self.random_state)
        points = pca.fit_transform(dist_matrix)
        cum = np.cumsum(pca.explained_variance_ratio_)
        n = int(np.argmax(cum >= self.variance_threshold) + 1)
        self.pca_data = points[:, :n]
        return self.pca_data

    def find_optimal_clusters(self, method='silhouette'):
        scorers = {'silhouette': silhouette_score,
                   'calinski': calinski_harabasz_score,
                   'davies': davies_bouldin_score}
        scores = []
        for k in range(2, self.max_clusters + 1):
            km = KMeans(n_clusters=k, random_state=self.random_state).fit(self.pca_data)
            s = scorers[method](self.pca_data, km.labels_)
            scores.append(-s if method == 'davies' else s)   # davies: lower is better
        self.optimal_k = int(np.argmax(scores) + 2)
        return self.optimal_k

    def apply_kmeans(self):
        self.kmeans = KMeans(n_clusters=self.optimal_k,
                             random_state=self.random_state).fit(self.pca_data)
        clusters = {i: [] for i in range(self.optimal_k)}
        for feat, lab in zip(self.features, self.kmeans.labels_):
            clusters[lab].append(feat)
        return clusters

    def apply_gmm(self, reg_covar=1e-3):
        gmm = GaussianMixture(n_components=self.optimal_k, random_state=self.random_state,
                              means_init=self.kmeans.cluster_centers_, reg_covar=reg_covar)
        gmm.fit(self.pca_data)
        return pd.DataFrame(gmm.predict_proba(self.pca_data), index=self.features,
                            columns=[f'Cluster_{i}' for i in range(self.optimal_k)])


def calculate_cluster_importance_mdi(model, feature_names, clusters):
    """Cluster-level MDI: sum per-feature tree importances within each cluster."""
    if hasattr(model, 'estimators_'):
        importances = {i: t.feature_importances_ for i, t in enumerate(model.estimators_)}
    else:
        importances = {0: model.feature_importances_}
    imp_df = pd.DataFrame.from_dict(importances, orient='index')
    imp_df.columns = feature_names
    imp_df = imp_df.replace(0, np.nan)
    out = pd.DataFrame(columns=['mean', 'std'])
    for cid, feats in clusters.items():
        valid = [f for f in feats if f in feature_names]
        if valid:
            ci = imp_df[valid].sum(axis=1)
            out.loc[f'Cluster_{cid}', 'mean'] = ci.mean()
            out.loc[f'Cluster_{cid}', 'std'] = ci.std() * ci.shape[0] ** -0.5 if len(ci) > 1 else 0
    total = out['mean'].sum()
    if total > 0:
        out['mean'] = out['mean'] / total
    return out


def calculate_cluster_importance_pfi(model, X, y, clusters, cv_splitter):
    """Cluster-level MDA: permute a whole cluster with ONE shared permutation, scored across
    a leakage-safe CV.

    BUG FIX #2 (Stage 3): the source used ``KFold(n_splits=cv, shuffle=True)``, invalid for
    overlapping triple-barrier labels (a shuffled split lands a label's near-duplicate in both
    train and test). The purged splitter is now INJECTED — pass a ``PurgedKFold``/CPCV keyed on
    the label ``t1`` spans so train/test never share a label horizon (LdP 2018 Ch.7; nlr-cw §5).
    """
    baseline = pd.Series(dtype='float64')
    perm = pd.DataFrame(columns=clusters.keys())
    for i, (tr, te) in enumerate(cv_splitter.split(X)):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        Xte, yte = X.iloc[te], y.iloc[te]
        model.fit(Xtr, ytr)
        baseline.loc[i] = -log_loss(yte, model.predict_proba(Xte), labels=model.classes_)
        for cid in clusters:
            Xp = Xte.copy()
            feats = [f for f in clusters[cid] if f in X.columns]
            if feats:
                idx = np.random.permutation(len(Xte))      # ONE permutation per cluster
                for f in feats:
                    Xp[f] = Xp[f].values[idx]
                perm.loc[i, cid] = -log_loss(yte, model.predict_proba(Xp), labels=model.classes_)
    importance = (-1 * perm).add(baseline, axis=0) / (-1 * perm)
    out = pd.DataFrame({'mean': importance.mean(),
                        'std': importance.std() * importance.shape[0] ** -0.5})
    out.index = [f'Cluster_{i}' for i in out.index]
    return out


if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    base = np.random.normal(0, 1, n)
    X = pd.DataFrame({
        'Base_0': base,
        'Corr_0_1': 0.8 * base + np.sqrt(1 - 0.64) * np.random.normal(0, 1, n),
        'Noise_0': np.random.normal(0, 1, n),
        'Noise_1': np.random.normal(0, 1, n)})
    D = compute_spearman_distance_matrix(X)
    c = OptimalClusterer(max_clusters=3)
    c.apply_pca(D); c.find_optimal_clusters(); clusters = c.apply_kmeans()
    print("optimal_k:", c.optimal_k)
    print(clusters)
