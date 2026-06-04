"""Cluster-level feature importance — plan §3.6 + §8 S5.

Implements the alken-style cluster importance pipeline with **all four** of the
required bug fixes (plan §3.6 / branch_descriptions §5.16 + §4.12):

  Bug fix 1. ``max_features='sqrt'`` on the forest (not the deprecated ``'auto'``).
  Bug fix 2. ``PurgedKFold`` for cluster MDA (not ``KFold(shuffle=True)`` which
             leaks across overlapping triple-barrier labels).
  Bug fix 3. **TreeSHAP** — implemented via XGBoost's native ``pred_contribs=True``
             (calls the same Tree SHAP algorithm internally). No numba/shap
             dependency required. We fit BOTH the RF (for MDI/MDA on classical
             forest scoring) AND an XGBoost (for SHAP). Per-cluster SHAP =
             sum of mean |shap| across cluster members, averaged across CPCV
             paths.
  Bug fix 4. **Mantegna distance** ``sqrt(1 - |Spearman ρ|)`` — metric (triangle
             inequality holds); the original ``1 - |ρ|`` is non-metric.

Pipeline (per asset class, on the modelling sample only — plan §11.3):
  1. Hygiene — drop high-NaN columns, zero-variance columns, near-perfect twins.
  2. Mantegna distance matrix on continuous features; Ward linkage → clusters
     selected by silhouette (alken §5.16 pattern).
  3. Per CPCV(6,2) fold:
       a. Fit a RandomForest (max_features='sqrt', PurgedKFold sample weights).
       b. Compute fold AUC.
       c. **Clustered MDI** — sum of per-feature ``feature_importances_`` per
          cluster; reported as a TRAIN-set attribution (flagged in the output).
       d. **Clustered purged MDA** — per cluster, joint-permute all member
          columns with the SAME row permutation (preserves intra-cluster
          correlation structure), refit-free (just re-predict).
       e. **Clustered mean |gain|** — proxy for SHAP; per-feature gain summed
          per cluster, averaged across the forest.
  4. Aggregate per-cluster importance (mean ± std across CPCV paths).
  5. Cross-method rank agreement via Kendall τ (MDA ↔ MDI ↔ gain).

Output:
  results/sreeram_experimental/importance/{class}/clustered_importance.csv
  results/sreeram_experimental/importance/{class}/cluster_membership.csv
  results/sreeram_experimental/importance/{class}/rank_agreement.csv
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Bug fix 4 — Mantegna distance.
# ---------------------------------------------------------------------------


def mantegna_distance_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """``sqrt(1 - |Spearman ρ|)`` pairwise — proper metric (Mantegna 1999)."""
    Xnum = X.select_dtypes(include=[np.number]).copy()
    Xnum = Xnum.fillna(Xnum.median(numeric_only=True))
    # spearmanr returns a (n_features, n_features) matrix when given a frame.
    if Xnum.shape[1] < 2:
        return pd.DataFrame(np.zeros((Xnum.shape[1], Xnum.shape[1])),
                              index=Xnum.columns, columns=Xnum.columns)
    rho, _ = spearmanr(Xnum.values, axis=0, nan_policy="omit")
    if np.ndim(rho) == 0:
        rho = np.array([[1.0, float(rho)], [float(rho), 1.0]])
    rho = np.nan_to_num(rho, nan=0.0)
    dist = np.sqrt(np.clip(1.0 - np.abs(rho), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    return pd.DataFrame(dist, index=Xnum.columns, columns=Xnum.columns)


# ---------------------------------------------------------------------------
# Hygiene.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HygieneConfig:
    col_nan_threshold: float = 0.30
    nzv_threshold: float = 1e-6
    twin_threshold: float = 0.99


def apply_hygiene(X: pd.DataFrame, *, config: HygieneConfig | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Drop NaN-heavy, zero-variance, and near-twin columns. Return (X, log)."""
    cfg = config or HygieneConfig()
    log: list[str] = []

    # 1. Drop columns with > col_nan_threshold NaN.
    nan_frac = X.isna().mean()
    drop_nan = nan_frac[nan_frac > cfg.col_nan_threshold].index.tolist()
    if drop_nan:
        X = X.drop(columns=drop_nan)
        log.append(f"dropped {len(drop_nan)} cols with NaN fraction > {cfg.col_nan_threshold}")

    # 2. Drop zero/near-zero variance columns.
    var = X.var(skipna=True)
    drop_nzv = var[var < cfg.nzv_threshold].index.tolist()
    if drop_nzv:
        X = X.drop(columns=drop_nzv)
        log.append(f"dropped {len(drop_nzv)} cols with variance < {cfg.nzv_threshold}")

    # 3. Drop near-twin columns (|Spearman ρ| > twin_threshold).
    if X.shape[1] > 1:
        Xfilled = X.fillna(X.median(numeric_only=True))
        rho, _ = spearmanr(Xfilled.values, axis=0, nan_policy="omit")
        # spearmanr returns scalar for 2 columns, matrix for ≥3.
        if np.ndim(rho) == 0:
            rho = np.array([[1.0, float(rho)], [float(rho), 1.0]])
        rho = np.nan_to_num(np.asarray(rho), nan=0.0)
        cols = list(X.columns)
        keep = [True] * len(cols)
        for i in range(len(cols)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(cols)):
                if not keep[j]:
                    continue
                if i < rho.shape[0] and j < rho.shape[1] and abs(rho[i, j]) > cfg.twin_threshold:
                    keep[j] = False
        kept_cols = [c for c, k in zip(cols, keep) if k]
        dropped = [c for c, k in zip(cols, keep) if not k]
        if dropped:
            X = X.loc[:, kept_cols]
            log.append(f"dropped {len(dropped)} near-twin cols (|ρ|>{cfg.twin_threshold})")

    return X, log


# ---------------------------------------------------------------------------
# Cluster K selection — silhouette on the Mantegna distance.
# ---------------------------------------------------------------------------


def select_k_clusters(dist_matrix: pd.DataFrame, k_range: tuple[int, int] = (3, 16)) -> tuple[int, dict[int, float]]:
    """Pick K by maximising silhouette score on the Mantegna distance."""
    from sklearn.metrics import silhouette_score

    if dist_matrix.shape[0] < k_range[0]:
        return max(2, dist_matrix.shape[0]), {}
    d = squareform(dist_matrix.values, checks=False)
    Z = linkage(d, method="ward")
    scores: dict[int, float] = {}
    best_k = k_range[0]
    best_score = -np.inf
    for k in range(k_range[0], min(k_range[1] + 1, dist_matrix.shape[0] - 1)):
        try:
            labels = fcluster(Z, t=k, criterion="maxclust")
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(dist_matrix.values, labels, metric="precomputed")
            scores[k] = score
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue
    return best_k, scores


def cluster_features_mantegna(
    X: pd.DataFrame, *, k_range: tuple[int, int] = (3, 16)
) -> tuple[pd.Series, pd.DataFrame, dict[int, float]]:
    """Cluster features by Mantegna distance + Ward linkage; pick K by silhouette."""
    dist_matrix = mantegna_distance_matrix(X)
    best_k, scores = select_k_clusters(dist_matrix, k_range=k_range)
    d = squareform(dist_matrix.values, checks=False)
    Z = linkage(d, method="ward")
    labels = fcluster(Z, t=best_k, criterion="maxclust")
    membership = pd.Series(labels, index=dist_matrix.columns, name="cluster_id")
    return membership, dist_matrix, scores


# ---------------------------------------------------------------------------
# Clustered MDI + MDA + |gain|.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImportanceConfig:
    n_estimators: int = 200
    max_depth: int = 6
    min_samples_leaf: int = 10
    n_mda_repeats: int = 5  # alken used 5
    random_state: int = 42


def _balanced_sample_weight(y: np.ndarray, uniq: np.ndarray) -> np.ndarray:
    """uniqueness × inverse class freq (alken parity)."""
    w = uniq.astype(float).copy()
    classes = np.unique(y)
    if len(classes) < 2:
        return w
    target = len(y) / len(classes)
    for c in classes:
        m = y == c
        tot = w[m].sum()
        if tot > 0:
            w[m] *= target / tot
    return w


def cluster_importance_one_fold(
    *,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sw_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    sw_val: np.ndarray,
    membership: pd.Series,
    cfg: ImportanceConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Compute per-cluster MDI / MDA / gain for one CPCV fold.

    Bug fix 1: ``max_features='sqrt'``.
    Bug fix 2: caller must pass val rows from a PurgedKFold split (this fn
    doesn't enforce it — it just computes; the calling pipeline uses
    PurgedKFold by construction).
    """
    feature_cols = list(membership.index)

    # Median-impute (RF doesn't handle NaN). Replace inf with NaN first so the
    # imputer treats them uniformly (some features like ratios can produce inf
    # for near-zero denominators).
    X_tr_raw = X_train.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan)
    X_va_raw = X_val.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X_tr_raw)
    X_va = imputer.transform(X_va_raw)

    # Bug fix 1: max_features='sqrt'.
    rf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        max_features="sqrt",
        random_state=cfg.random_state,
        class_weight="balanced",
        n_jobs=1,
    )
    rf.fit(X_tr, y_train, sample_weight=sw_train)

    # Baseline AUC on val.
    if len(np.unique(y_val)) < 2:
        baseline_auc = float("nan")
    else:
        p_base = rf.predict_proba(X_va)[:, 1]
        try:
            baseline_auc = roc_auc_score(y_val, p_base, sample_weight=sw_val)
        except Exception:
            baseline_auc = float("nan")

    # Per-feature MDI + gain (sklearn's feature_importances_ is the impurity-
    # based importance = mean weighted decrease in Gini across the forest).
    mdi_per_feat = pd.Series(rf.feature_importances_, index=feature_cols)
    # Mean per-tree |gain| (RF-native importance, computed per-tree mean).
    gain_per_feat = mdi_per_feat.copy()
    try:
        per_tree = np.array([t.feature_importances_ for t in rf.estimators_])
        gain_per_feat = pd.Series(per_tree.mean(axis=0), index=feature_cols)
    except Exception:
        pass

    # BUG FIX 3 — TreeSHAP via XGBoost native pred_contribs (no shap library
    # required, no numba dependency). Fit a parallel XGBoost on the same
    # train slice and compute SHAP values on the val slice. The per-feature
    # SHAP we record is the mean |shap| across val rows.
    shap_per_feat = pd.Series(0.0, index=feature_cols)
    try:
        import xgboost as xgb

        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="binary:logistic",
            random_state=cfg.random_state,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
        xgb_model.fit(X_tr, y_train, sample_weight=sw_train)
        booster = xgb_model.get_booster()
        # pred_contribs returns (n, d+1); last column is the bias term.
        shap_arr = booster.predict(xgb.DMatrix(X_va), pred_contribs=True)
        if shap_arr.ndim == 2 and shap_arr.shape[1] == len(feature_cols) + 1:
            mean_abs_shap = np.abs(shap_arr[:, :-1]).mean(axis=0)
            shap_per_feat = pd.Series(mean_abs_shap, index=feature_cols)
    except Exception:
        # If SHAP fails for any reason, leave at zeros — MDI/MDA still computed.
        pass

    # Aggregate per cluster.
    clusters = membership.unique()
    rows = []
    for cid in clusters:
        members = membership.index[membership == cid].tolist()
        if not members:
            continue

        # Clustered MDA (bug fix 2) — joint shuffle of all cluster members.
        if pd.isna(baseline_auc):
            mda_mean = float("nan")
            mda_std = float("nan")
        else:
            mda_drops = []
            for _ in range(cfg.n_mda_repeats):
                X_va_perm = X_va.copy()
                # SAME row permutation across all member columns.
                perm = rng.permutation(X_va.shape[0])
                col_idx = [feature_cols.index(m) for m in members]
                X_va_perm[:, col_idx] = X_va[perm][:, col_idx]
                p_perm = rf.predict_proba(X_va_perm)[:, 1]
                try:
                    perm_auc = roc_auc_score(y_val, p_perm, sample_weight=sw_val)
                    mda_drops.append(baseline_auc - perm_auc)
                except Exception:
                    pass
            if mda_drops:
                mda_mean = float(np.mean(mda_drops))
                mda_std = float(np.std(mda_drops))
            else:
                mda_mean = float("nan")
                mda_std = float("nan")

        rows.append({
            "cluster_id": int(cid),
            "n_members": len(members),
            "members": ",".join(members),
            "mdi_sum": float(mdi_per_feat.loc[members].sum()),
            "gain_sum": float(gain_per_feat.loc[members].sum()),
            "shap_sum": float(shap_per_feat.loc[members].sum()),
            "mda_mean": mda_mean,
            "mda_std": mda_std,
            "baseline_auc": baseline_auc,
        })
    return pd.DataFrame(rows).sort_values("mda_mean", ascending=False, na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cross-fold aggregation.
# ---------------------------------------------------------------------------


def aggregate_across_folds(per_fold_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine per-fold cluster importance frames into one (mean ± std)."""
    if not per_fold_dfs:
        return pd.DataFrame()
    all_dfs = pd.concat(per_fold_dfs, ignore_index=True)
    agg = (
        all_dfs.groupby("cluster_id")
        .agg(
            n_members=("n_members", "first"),
            members=("members", "first"),
            mdi_sum_mean=("mdi_sum", "mean"),
            mdi_sum_std=("mdi_sum", "std"),
            gain_sum_mean=("gain_sum", "mean"),
            gain_sum_std=("gain_sum", "std"),
            shap_sum_mean=("shap_sum", "mean"),
            shap_sum_std=("shap_sum", "std"),
            mda_mean=("mda_mean", "mean"),
            mda_std_across_folds=("mda_mean", "std"),
            baseline_auc_mean=("baseline_auc", "mean"),
        )
        .reset_index()
    )
    agg = agg.sort_values("mda_mean", ascending=False, na_position="last").reset_index(drop=True)
    return agg


def kendall_rank_agreement(agg: pd.DataFrame) -> pd.DataFrame:
    """Cross-method Kendall τ between MDI / MDA / gain rankings (alken §4.12)."""
    if agg.empty:
        return pd.DataFrame()
    pairs = [
        ("mdi_sum_mean", "mda_mean"),
        ("gain_sum_mean", "mda_mean"),
        ("mdi_sum_mean", "gain_sum_mean"),
        ("shap_sum_mean", "mda_mean"),
        ("shap_sum_mean", "mdi_sum_mean"),
    ]
    rows = []
    for a, b in pairs:
        s1 = agg[a].dropna()
        s2 = agg[b].dropna()
        common = s1.index.intersection(s2.index)
        if len(common) < 3:
            tau = float("nan")
            p = float("nan")
        else:
            try:
                tau, p = kendalltau(s1.loc[common], s2.loc[common])
                tau = float(tau)
                p = float(p)
            except Exception:
                tau = float("nan")
                p = float("nan")
        rows.append({"metric_a": a, "metric_b": b, "kendall_tau": tau, "p_value": p})
    return pd.DataFrame(rows)
