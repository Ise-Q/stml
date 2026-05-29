"""
optuna_objective.py
====================
Glue between the trainers, the purged walk-forward CV, and Optuna.

``cross_val_auc`` is the single scoring primitive used everywhere: it runs a model across the
purged folds and returns the mean (and std) validation ROC-AUC. The barrier search calls it with
a fixed baseline; the model studies call it through an Optuna objective that samples each model's
hyperparameter space. Folds whose train or validation block lacks both classes are skipped (not
scored as 0.5), so a degenerate thin-instrument cell can't poison the mean.

AUC -- never accuracy -- is the objective: the guide is explicit that an accuracy-optimised
labeler/model is worthless under imbalance. Studies use a seeded TPE sampler for reproducibility.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from stml.model.cv import PurgedWalkForward
from stml.model.mlp import MLPModel, mlp_param_space
from stml.model.trees import RFModel, XGBModel, rf_param_space, xgb_param_space
from stml.model.vsn import VSNModel, vsn_param_space

# model key -> (wrapper class, optuna param-space fn)
MODEL_REGISTRY: dict[str, tuple[type, Callable]] = {
    "xgb": (XGBModel, xgb_param_space),
    "rf": (RFModel, rf_param_space),
    "mlp": (MLPModel, mlp_param_space),
    "vsn": (VSNModel, vsn_param_space),
}


def cross_val_auc(
    model_cls: type,
    params: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    dev_df: pd.DataFrame,
    cv: PurgedWalkForward,
    *,
    seed: int = 0,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, float, int]:
    """Mean / std purged-CV validation AUC for one model + param set.

    ``X`` must be row-aligned with ``dev_df`` (same order, reset index). Returns
    ``(mean_auc, std_auc, n_scored_folds)``; mean is NaN if no fold was scorable.
    """
    aucs: list[float] = []
    for tr, va in cv.split(dev_df):
        ytr, yva = y[tr], y[va]
        if np.unique(ytr).size < 2 or np.unique(yva).size < 2:
            continue
        sw = sample_weight[tr] if sample_weight is not None else None
        model = model_cls(params, seed).fit(X.iloc[tr], ytr, sample_weight=sw)
        proba = model.predict_proba(X.iloc[va])
        aucs.append(roc_auc_score(yva, proba))
    if not aucs:
        return float("nan"), float("nan"), 0
    return float(np.mean(aucs)), float(np.std(aucs)), len(aucs)


def make_objective(
    model_key: str,
    X: pd.DataFrame,
    y: np.ndarray,
    dev_df: pd.DataFrame,
    cv: PurgedWalkForward,
    *,
    seed: int = 0,
    sample_weight: np.ndarray | None = None,
) -> Callable[[optuna.Trial], float]:
    """Build an Optuna objective maximising mean purged-CV AUC for ``model_key``."""
    model_cls, space = MODEL_REGISTRY[model_key]

    def objective(trial: optuna.Trial) -> float:
        params = space(trial)
        mean, std, n = cross_val_auc(
            model_cls, params, X, y, dev_df, cv, seed=seed, sample_weight=sample_weight
        )
        trial.set_user_attr("auc_std", std)
        trial.set_user_attr("n_folds", n)
        return mean if np.isfinite(mean) else 0.0

    return objective


def run_study(
    objective: Callable[[optuna.Trial], float],
    n_trials: int,
    *,
    seed: int = 0,
    direction: str = "maximize",
) -> optuna.Study:
    """Run a seeded TPE study quietly and return it."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study
