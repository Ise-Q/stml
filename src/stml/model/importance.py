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
