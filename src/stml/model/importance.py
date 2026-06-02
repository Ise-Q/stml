"""
importance.py
=============
Feature-importance analysis for both model families, on a common footing.

Trees expose three complementary views:
* **native gain / MDI** -- the model's own split-based importance (fast, but biased toward
  high-cardinality features),
* **SHAP** (TreeExplainer) -- additive, theoretically grounded per-feature attributions,
* **permutation** -- model-agnostic drop in validation AUC when a feature is shuffled.

Neural nets get the model-agnostic **permutation** view, a **gradient saliency** view (mean
absolute gradient of the logit w.r.t. each standardised input), and -- for the VSN -- its own
**selection-gate weights**. Permutation is computed on a held-out (validation/test) block so it
reflects generalisation, not train fit.

All functions return tidy, sorted Series/DataFrames keyed by the model's ``feature_names_`` so the
notebook can rank them against the catalog's expectation that the counter-trend F1 family
(``f1_mr_*``) should dominate a counter-trend signal's meta-model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def permutation_importance_auc(
    model, X: pd.DataFrame, y: np.ndarray, *, n_repeats: int = 5, seed: int = 0
) -> pd.Series:
    """Mean drop in ROC-AUC when each feature column is permuted (model already fitted)."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X))
    drops: dict[str, float] = {}
    for col in X.columns:
        vals = X[col].to_numpy()
        acc = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(vals)
            acc.append(base - roc_auc_score(y, model.predict_proba(Xp)))
        drops[col] = float(np.mean(acc))
    return pd.Series(drops, name="perm_auc_drop").sort_values(ascending=False)


def native_tree_importance(model) -> pd.Series:
    """Native gain (XGBoost) / MDI (RandomForest) importance from a fitted tree model."""
    imp = model.model_.feature_importances_
    return pd.Series(imp, index=model.feature_names_, name="native").sort_values(ascending=False)


def shap_tree_importance(model, X: pd.DataFrame, *, max_samples: int = 500,
                         seed: int = 0) -> pd.Series:
    """Mean |SHAP value| per feature for a fitted tree model (TreeExplainer)."""
    import shap

    Xs = X.sample(min(len(X), max_samples), random_state=seed) if len(X) > max_samples else X
    # RandomForest needs the imputed/scaled matrix; XGBoost handles raw NaNs.
    if getattr(model, "prep_", None) is not None:
        data = pd.DataFrame(model.prep_.transform(Xs), columns=model.feature_names_)
    else:
        data = Xs
    explainer = shap.TreeExplainer(model.model_)
    sv = explainer.shap_values(data, check_additivity=False)
    arr = np.asarray(sv)
    if arr.ndim == 3:           # (n, features, classes) -> positive class
        arr = arr[:, :, -1]
    elif isinstance(sv, list):  # [class0, class1]
        arr = np.asarray(sv[-1])
    mean_abs = np.abs(arr).mean(axis=0)
    return pd.Series(mean_abs, index=model.feature_names_,
                     name="shap").sort_values(ascending=False)


def gradient_importance(model, X: pd.DataFrame) -> pd.Series:
    """Mean |d logit / d input| per feature for a fitted torch model (saliency)."""
    import torch

    net = model.model_
    net.eval()
    Xt = torch.tensor(model.prep_.transform(X), dtype=torch.float32, requires_grad=True)
    out = net(Xt)
    out.sum().backward()
    grad = Xt.grad.abs().mean(dim=0).numpy()
    return pd.Series(grad, index=model.feature_names_,
                     name="grad_saliency").sort_values(ascending=False)


def tree_importance(model, X: pd.DataFrame, y: np.ndarray, *, n_repeats: int = 5,
                    seed: int = 0, with_shap: bool = True) -> pd.DataFrame:
    """Combine native, SHAP and permutation importances for a tree model into one frame."""
    cols = {
        "native": native_tree_importance(model),
        "perm_auc_drop": permutation_importance_auc(model, X, y, n_repeats=n_repeats, seed=seed),
    }
    if with_shap:
        cols["shap"] = shap_tree_importance(model, X, seed=seed)
    df = pd.DataFrame(cols)
    df["rank_mean"] = df.rank(ascending=False).mean(axis=1)
    return df.sort_values("rank_mean")


def nn_importance(model, X: pd.DataFrame, y: np.ndarray, *, n_repeats: int = 5,
                  seed: int = 0) -> pd.DataFrame:
    """Combine permutation, gradient saliency and (if VSN) gate weights for a torch model."""
    cols = {
        "perm_auc_drop": permutation_importance_auc(model, X, y, n_repeats=n_repeats, seed=seed),
        "grad_saliency": gradient_importance(model, X),
    }
    if hasattr(model, "gate_weights"):
        cols["vsn_gate"] = model.gate_weights(X)
    df = pd.DataFrame(cols)
    df["rank_mean"] = df.rank(ascending=False).mean(axis=1)
    return df.sort_values("rank_mean")


# ---------------------------------------------------------------------------
# Cluster-level importance (the 10-mark section).
#
# Correlated features split their split-credit / SHAP / permutation signal, so a per-feature
# ranking under-states a redundancy group whose members each look weak. The redundancy map
# (``results/feature_redundancy.json``) already grouped features with ``|corr| > threshold``; here
# we aggregate importance back to that cluster level so the ranking reflects *economic* groups
# (regime / volatility / counter-trend), and the noise-injection / SFI / iterative-drop checks the
# guide's Part 2 asks for fall out of the same machinery.
#
# A subtlety the design pins down: importance runs on the **model's** columns
# (``model.feature_names_`` -- the redundancy-pruned base features PLUS one-hot ``f4_clust_*`` /
# ``inst_*`` dummies), not the full 175-feature redundancy universe. Dummies belong to no
# redundancy cluster -> an explicit ``"dummies"`` bucket. Clusters whose representative was pruned
# out have no in-matrix member -> reported as ``missing_cluster_ids`` (never silently zeroed).
# ---------------------------------------------------------------------------

DUMMY_PREFIXES = ("f4_clust", "inst_")


def load_feature_clusters(redundancy_path: str | Path | None = None) -> dict[int, list[str]]:
    """Invert ``feature_redundancy.json["clusters"]`` (``{feat: cid}``) to ``{cid: [feats...]}``.

    Member lists are sorted for determinism. Default path resolved via ``_find_repo_root`` like
    :mod:`stml.model.dataset` does.
    """
    if redundancy_path is None:
        from stml.io import _find_repo_root

        redundancy_path = (
            _find_repo_root(Path.cwd().resolve()) / "results" / "feature_redundancy.json"
        )
    feat_to_cid: dict[str, int] = json.loads(Path(redundancy_path).read_text())["clusters"]
    clusters: dict[int, list[str]] = {}
    for feat, cid in feat_to_cid.items():
        clusters.setdefault(int(cid), []).append(feat)
    return {cid: sorted(members) for cid, members in sorted(clusters.items())}


def map_model_features_to_clusters(
    feature_names: list[str], clusters: dict[int, list[str]]
) -> tuple[dict[str, int | str], list[int]]:
    """Map each model column to its redundancy cluster id, else the ``"dummies"`` bucket.

    A name maps to a cluster id when it is a member of some cluster; otherwise it falls into
    ``"dummies"`` -- one-hot ``f4_clust_*`` / ``inst_*`` columns belong to no redundancy cluster,
    and any other non-clustered column is caught there too (the catch-all). Every ``feature_name``
    maps to exactly one cluster id OR ``"dummies"``.

    Returns ``(assignment, missing_cluster_ids)`` where ``missing_cluster_ids`` is the sorted list
    of cluster ids with ZERO members present in ``feature_names`` (representative pruned -- surfaced,
    not silently dropped).
    """
    member_to_cid = {m: cid for cid, members in clusters.items() for m in members}
    assignment: dict[str, int | str] = {}
    for name in feature_names:
        cid = member_to_cid.get(name)
        assignment[name] = cid if cid is not None else "dummies"
    present = {a for a in assignment.values() if a != "dummies"}
    missing = sorted(cid for cid in clusters if cid not in present)
    return assignment, missing


def _cluster_members_in_matrix(
    feature_names: list[str], clusters: dict[int, list[str]]
) -> tuple[dict[int, list[str]], list[str]]:
    """Split model columns into ``{cid: [in-matrix members]}`` plus the dummy/non-clustered list."""
    assignment, _ = map_model_features_to_clusters(feature_names, clusters)
    in_matrix: dict[int, list[str]] = {}
    dummies: list[str] = []
    for name in feature_names:
        cid = assignment[name]
        if cid == "dummies":
            dummies.append(name)
        else:
            in_matrix.setdefault(int(cid), []).append(name)
    return in_matrix, dummies


def clustered_mdi(model, clusters: dict[int, list[str]]) -> pd.Series:
    """Cluster-summed native (MDI / gain) importance, plus a ``"dummies"`` row.

    Each cluster row = sum of :func:`native_tree_importance` over its in-matrix members; the
    ``"dummies"`` row sums the dummy / non-clustered columns. Decomposition identity: cluster rows
    + dummies row == total native importance. Sorted descending.
    """
    native = native_tree_importance(model)
    in_matrix, dummies = _cluster_members_in_matrix(list(native.index), clusters)
    out: dict[int | str, float] = {
        cid: float(native[members].sum()) for cid, members in in_matrix.items()
    }
    out["dummies"] = float(native[dummies].sum()) if dummies else 0.0
    return pd.Series(out, name="cluster_mdi").sort_values(ascending=False)


def clustered_mda(
    model, X: pd.DataFrame, y: np.ndarray, clusters: dict[int, list[str]],
    *, n_repeats: int = 5, seed: int = 42,
) -> pd.Series:
    """Cluster-level permutation importance: ΔAUC when a whole cluster is permuted **jointly**.

    All in-matrix columns of a cluster share one row permutation per repeat, so within-cluster
    correlation is preserved while the cluster->y link is broken (permuting members independently
    would understate a correlated group). Model-agnostic (only ``predict_proba``). A ``"dummies"``
    row covers the dummy / non-clustered columns the same way.
    """
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X))
    in_matrix, dummies = _cluster_members_in_matrix(list(X.columns), clusters)
    groups: dict[int | str, list[str]] = dict(in_matrix)
    if dummies:
        groups["dummies"] = dummies
    n = len(X)
    drops: dict[int | str, float] = {}
    for key, members in groups.items():
        acc = []
        for _ in range(n_repeats):
            perm = rng.permutation(n)
            Xp = X.copy()
            for col in members:
                Xp[col] = X[col].to_numpy()[perm]
            acc.append(base - roc_auc_score(y, model.predict_proba(Xp)))
        drops[key] = float(np.mean(acc))
    return pd.Series(drops, name="cluster_mda").sort_values(ascending=False)


def clustered_shap(
    model, X: pd.DataFrame, clusters: dict[int, list[str]],
    *, max_samples: int = 500, seed: int = 42,
) -> pd.Series:
    """Per-feature mean ``|SHAP|`` (:func:`shap_tree_importance`) summed to cluster level + dummies.

    SHAP is additive, so the cluster total is the sum of its members' attributions.
    """
    per_feat = shap_tree_importance(model, X, max_samples=max_samples, seed=seed)
    in_matrix, dummies = _cluster_members_in_matrix(list(per_feat.index), clusters)
    out: dict[int | str, float] = {
        cid: float(per_feat[members].sum()) for cid, members in in_matrix.items()
    }
    out["dummies"] = float(per_feat[dummies].sum()) if dummies else 0.0
    return pd.Series(out, name="cluster_shap").sort_values(ascending=False)


def single_feature_importance(
    model_cls, params: dict, X: pd.DataFrame, y: np.ndarray, df: pd.DataFrame, cv,
    *, seed: int = 42,
) -> pd.Series:
    """SFI (López de Prado): per-feature mean purged-CV AUC when trained on that ONE column.

    For each feature, fit ``model_cls`` on the single column under ``cv.split(df)`` and score
    validation ROC-AUC, averaged over folds where both classes are present in train AND val (a
    degenerate single-class fold is skipped, not scored 0.5). Unlike permutation/MDI (which share
    a fitted model), SFI is leakage-free per feature -- each feature stands on its own out-of-fold.
    Sorted descending.
    """
    X = X.reset_index(drop=True)
    df = df.reset_index(drop=True)
    y = np.asarray(y)
    scores: dict[str, float] = {}
    for col in X.columns:
        fold_aucs = []
        for tr, va in cv.split(df):
            if np.unique(y[tr]).size < 2 or np.unique(y[va]).size < 2:
                continue
            model = model_cls(params, seed).fit(X[[col]].iloc[tr], y[tr])
            fold_aucs.append(roc_auc_score(y[va], model.predict_proba(X[[col]].iloc[va])))
        if fold_aucs:
            scores[col] = float(np.mean(fold_aucs))
    return pd.Series(scores, name="sfi_auc").sort_values(ascending=False)


def noise_injection_check(
    model_cls, params: dict, X: pd.DataFrame, y: np.ndarray, df: pd.DataFrame, cv,
    *, n_noise: int = 5, seed: int = 42,
) -> pd.DataFrame:
    """Inject ``n_noise`` standard-normal columns and rank them by permutation importance.

    The guide's required sanity check: deliberately-injected noise should sink **below** the real
    informative features. Fits one model on the full dev ``X`` (augmented with ``noise_0..``) and
    computes :func:`permutation_importance_auc`. Returns a frame with the importance and an
    ``is_noise`` flag, sorted by importance descending. ``cv`` is accepted for a uniform call
    signature (the check is a single full-dev fit, mirroring the model used downstream).
    """
    rng = np.random.default_rng(seed)
    Xn = X.reset_index(drop=True).copy()
    noise_cols = [f"noise_{i}" for i in range(n_noise)]
    for col in noise_cols:
        Xn[col] = rng.standard_normal(len(Xn))
    model = model_cls(params, seed).fit(Xn, np.asarray(y))
    imp = permutation_importance_auc(model, Xn, np.asarray(y), seed=seed)
    out = imp.to_frame()
    out["is_noise"] = out.index.isin(noise_cols)
    return out.sort_values("perm_auc_drop", ascending=False)


def iterative_cluster_drop(
    model_cls, params: dict, X: pd.DataFrame, y: np.ndarray, df: pd.DataFrame, cv,
    clusters: dict[int, list[str]], *, importance_fn, seed: int = 42,
    min_clusters: int = 3, max_iters: int = 8,
) -> pd.DataFrame:
    """Backward cluster elimination ("Feature Track A"): drop the weakest cluster, refit, re-score.

    Each round: fit on the surviving columns, rank clusters via ``importance_fn(model, ...)`` (a
    callable returning a per-cluster Series, e.g. :func:`clustered_mdi`), drop the lowest cluster's
    in-matrix columns, and record ``(dropped_cluster, n_features_left, cv_auc, delta_auc)`` where
    ``cv_auc`` is the mean purged-CV AUC over folds with both classes. Bounded by ``min_clusters``
    and ``max_iters`` so it terminates fast. ``delta_auc`` is the change vs the previous round.
    """
    X = X.reset_index(drop=True)
    df = df.reset_index(drop=True)
    y = np.asarray(y)
    cols = list(X.columns)
    rows: list[dict] = []
    prev_auc = None

    def _cv_auc(active: list[str]) -> float:
        fold_aucs = []
        for tr, va in cv.split(df):
            if np.unique(y[tr]).size < 2 or np.unique(y[va]).size < 2:
                continue
            m = model_cls(params, seed).fit(X[active].iloc[tr], y[tr])
            fold_aucs.append(roc_auc_score(y[va], m.predict_proba(X[active].iloc[va])))
        return float(np.mean(fold_aucs)) if fold_aucs else float("nan")

    for _ in range(max_iters):
        in_matrix, _ = _cluster_members_in_matrix(cols, clusters)
        if len(in_matrix) <= min_clusters:
            break
        model = model_cls(params, seed).fit(X[cols], y)
        ranked = importance_fn(model, clusters)
        # lowest-ranked real cluster (skip the "dummies" bucket; never drop dummies)
        cluster_only = ranked[[k for k in ranked.index if k != "dummies"]]
        if cluster_only.empty:
            break
        worst = cluster_only.idxmin()
        drop_cols = in_matrix.get(int(worst), [])
        cols = [c for c in cols if c not in drop_cols]
        auc = _cv_auc(cols)
        delta = (auc - prev_auc) if prev_auc is not None else 0.0
        rows.append({
            "dropped_cluster": int(worst),
            "n_features_left": len(cols),
            "cv_auc": auc,
            "delta_auc": delta,
        })
        prev_auc = auc
    return pd.DataFrame(rows, columns=["dropped_cluster", "n_features_left", "cv_auc", "delta_auc"])
