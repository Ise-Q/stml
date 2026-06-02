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
from math import ceil

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from stml.model.cv import CombinatorialPurgedCV, PurgedWalkForward
from stml.model.linear import LogRegModel, logreg_param_space
from stml.model.mlp import MLPModel, mlp_param_space
from stml.model.trees import (
    HAS_LIGHTGBM,
    AdaBoostModel,
    LGBMModel,
    RFModel,
    XGBModel,
    adaboost_param_space,
    lgbm_param_space,
    rf_param_space,
    xgb_param_space,
)
from stml.model.vsn import VSNModel, vsn_param_space

# model key -> (wrapper class, optuna param-space fn). Families: linear / tree / neural.
# AdaBoost is sklearn-native (always present); LightGBM is registered only when importable so the
# tree family gracefully degrades to RF + XGB + AdaBoost when lightgbm is absent.
MODEL_REGISTRY: dict[str, tuple[type, Callable]] = {
    "logreg": (LogRegModel, logreg_param_space),
    "xgb": (XGBModel, xgb_param_space),
    "rf": (RFModel, rf_param_space),
    "ada": (AdaBoostModel, adaboost_param_space),
    "mlp": (MLPModel, mlp_param_space),
    "vsn": (VSNModel, vsn_param_space),
}
if HAS_LIGHTGBM:
    MODEL_REGISTRY["lgbm"] = (LGBMModel, lgbm_param_space)


def cross_val_auc(
    model_cls: type,
    params: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    dev_df: pd.DataFrame,
    cv: PurgedWalkForward | CombinatorialPurgedCV,
    *,
    seed: int = 0,
    sample_weight: np.ndarray | None = None,
    weight_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[float, float, int]:
    """Mean / std purged-CV validation AUC for one model + param set.

    ``X`` must be row-aligned with ``dev_df`` (same order, reset index). ``cv`` is duck-typed on
    ``.split(df) -> (train_pos, val_pos)``, so either :class:`PurgedWalkForward` or
    :class:`CombinatorialPurgedCV` works. Returns ``(mean_auc, std_auc, n_scored_folds)``; mean is
    NaN if no fold was scorable.

    Sample weights: pass ``weight_fn`` to **recompute** the López de Prado uniqueness weights on
    each fold's *purged-train* rows (``weight_fn(train_pos) -> weights``). This is the leakage-safe
    path -- slicing a globally-computed weight vector (``sample_weight[tr]``) carries the
    concurrency of events that purging removed from the fold, a subtle Channel-2 leak. The
    ``sample_weight`` argument is retained for the weightless barrier-search baseline and is used
    only when ``weight_fn`` is None.
    """
    aucs: list[float] = []
    for tr, va in cv.split(dev_df):
        ytr, yva = y[tr], y[va]
        if np.unique(ytr).size < 2 or np.unique(yva).size < 2:
            continue
        if weight_fn is not None:
            sw = weight_fn(tr)
        elif sample_weight is not None:
            sw = sample_weight[tr]
        else:
            sw = None
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
    cv: PurgedWalkForward | CombinatorialPurgedCV,
    *,
    seed: int = 0,
    sample_weight: np.ndarray | None = None,
    weight_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Callable[[optuna.Trial], float]:
    """Build an Optuna objective maximising mean purged-CV AUC for ``model_key``.

    Pass ``weight_fn`` to recompute uniqueness weights per purged-train fold (leakage-safe; see
    :func:`cross_val_auc`).
    """
    model_cls, space = MODEL_REGISTRY[model_key]

    def objective(trial: optuna.Trial) -> float:
        params = space(trial)
        mean, std, n = cross_val_auc(
            model_cls, params, X, y, dev_df, cv,
            seed=seed, sample_weight=sample_weight, weight_fn=weight_fn,
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


# --------------------------------------------------------------------------------------------- #
# CPCV-aware scoring + two-stage selection (guide Part 2).
#
# Combinatorial purged CV gives every config a *distribution* of scores across backtest paths
# (``C(N-1,k-1)`` of them), not a single walk-forward number. Stage 1 ranks configs by their mean
# path-AUC (statistical strength); Stage 2 breaks the survivors apart by Sharpe-spread -- here the
# dispersion ``std(path_AUC)``, lower = more robust -- so a config that is merely *lucky on one
# path* loses to one that is *consistently good across paths*. Uniqueness weights are recomputed
# per purged-train fold via ``weight_fn`` (never sliced), exactly as in :func:`cross_val_auc`.
# --------------------------------------------------------------------------------------------- #


def cpcv_oof_auc(
    model_cls: type,
    params: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    df: pd.DataFrame,
    cpcv: CombinatorialPurgedCV,
    *,
    seed: int = 42,
    weight_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[float, float, int]:
    """Mean / std ROC-AUC across the CPCV **backtest paths** for one model + param set.

    Each block combination is fitted on its purged-embargoed train rows and predicts its test
    block(s); the per-block predictions are then reassembled (via
    :meth:`CombinatorialPurgedCV.path_assignments`) into ``n_paths`` full-development OOF paths, and
    one AUC is scored per path. Returns ``(mean_path_auc, std_path_auc, n_paths_scored)`` -- the
    mean is the Stage-1 ranking score, the std the Stage-2 robustness (Sharpe-spread) score.

    A block combination whose purged train lacks both classes cannot be fitted; any path that
    depends on it is dropped, lowering ``n_paths_scored`` (mean/std/NaN if none survive). ``X`` must
    be row-aligned with ``df``.
    """
    df = df.reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    n = len(df)
    dates = pd.to_datetime(df["date"])
    blocks = cpcv._date_blocks(dates)
    block_of = np.full(n, -1, dtype=int)
    for b, blk in enumerate(blocks):
        block_of[dates.isin(blk).to_numpy()] = b

    # Fit each block combination once; cache its test-row predictions (NaN elsewhere).
    split_pred: dict[int, np.ndarray | None] = {}
    for si, (tr, te) in enumerate(cpcv.split(df)):
        if np.unique(y[tr]).size < 2:
            split_pred[si] = None
            continue
        sw = weight_fn(tr) if weight_fn is not None else None
        model = model_cls(params, seed).fit(X.iloc[tr], y[tr], sample_weight=sw)
        pr = np.full(n, np.nan)
        pr[te] = model.predict_proba(X.iloc[te])
        split_pred[si] = pr

    # Reassemble each path (one block-combination per block) and score it.
    assignments = cpcv.path_assignments(df)
    path_aucs: list[float] = []
    for path_id in range(cpcv.n_paths()):
        proba = np.full(n, np.nan)
        complete = True
        for rec in (r for r in assignments if r["path"] == path_id):
            pr = split_pred.get(rec["split"])
            if pr is None:
                complete = False
                break
            rows_b = block_of == rec["block"]
            proba[rows_b] = pr[rows_b]
        if not complete:
            continue
        mask = ~np.isnan(proba)
        if np.unique(y[mask]).size < 2:
            continue
        path_aucs.append(float(roc_auc_score(y[mask], proba[mask])))

    if not path_aucs:
        return float("nan"), float("nan"), 0
    return float(np.mean(path_aucs)), float(np.std(path_aucs)), len(path_aucs)


def make_objective_cpcv(
    model_key: str,
    X: pd.DataFrame,
    y: np.ndarray,
    df: pd.DataFrame,
    cpcv: CombinatorialPurgedCV,
    *,
    seed: int = 42,
    weight_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Callable[[optuna.Trial], float]:
    """Optuna objective maximising **mean CPCV path-AUC**; stores ``path_auc_std`` per trial.

    Use with :func:`two_stage_select` to pick the robust (low-spread) survivor of the high-mean
    configs.
    """
    model_cls, space = MODEL_REGISTRY[model_key]

    def objective(trial: optuna.Trial) -> float:
        params = space(trial)
        mean, std, n = cpcv_oof_auc(
            model_cls, params, X, y, df, cpcv, seed=seed, weight_fn=weight_fn
        )
        trial.set_user_attr("path_auc_std", std)
        trial.set_user_attr("n_paths_scored", n)
        return mean if np.isfinite(mean) else 0.0

    return objective


def two_stage_select(study: optuna.Study, *, top_frac: float = 0.25) -> optuna.trial.FrozenTrial:
    """Two-stage HP selection (guide Part 2): mean path-AUC, then lowest Sharpe-spread.

    **Stage 1** keeps the top ``top_frac`` of completed trials by mean path-AUC (``trial.value``).
    **Stage 2** returns the survivor with the **lowest** ``path_auc_std`` user-attr (lower spread =
    more robust across paths), ties broken by higher mean AUC. Raises if no trial is scorable.
    """
    trials = [t for t in study.trials if t.value is not None and np.isfinite(t.value)]
    if not trials:
        raise ValueError("no completed trials with a finite value to select from")
    trials.sort(key=lambda t: t.value, reverse=True)
    survivors = trials[: max(1, ceil(top_frac * len(trials)))]

    def spread(t: optuna.trial.FrozenTrial) -> float:
        s = t.user_attrs.get("path_auc_std", float("inf"))
        return float(s) if s is not None and np.isfinite(s) else float("inf")

    survivors.sort(key=lambda t: (spread(t), -t.value))
    return survivors[0]
