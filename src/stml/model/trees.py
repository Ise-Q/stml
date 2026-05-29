"""
trees.py
========
Tree-based meta-model trainers: gradient-boosted trees (XGBoost) and a Random Forest.

Both are wrapped in a tiny uniform interface -- ``fit(X, y, sample_weight=None)`` /
``predict_proba(X) -> P(class=1)`` / ``feature_names_`` -- so the Optuna objective and the
importance/evaluation code can treat any model identically (the torch nets in ``mlp.py`` /
``vsn.py`` expose the same surface).

Imbalance handling is built in: XGBoost gets ``scale_pos_weight = n_neg/n_pos`` computed on each
fold's own training labels; RandomForest uses ``class_weight='balanced_subsample'``. Both accept
the López de Prado sample-uniqueness weights (overlapping labels are not iid). XGBoost consumes
the design matrix with NaNs intact (native missing-value handling); RandomForest can't, so it
median-imputes via :class:`~stml.model.dataset.Preprocessor` fit on the training rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from stml.model.dataset import Preprocessor


def _scale_pos_weight(y: np.ndarray) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return (n_neg / n_pos) if n_pos > 0 else 1.0


class XGBModel:
    """XGBoost binary classifier; consumes raw (NaN-bearing) features."""

    def __init__(self, params: dict, seed: int = 0) -> None:
        self.params = dict(params)
        self.seed = seed
        self.model_: XGBClassifier | None = None
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None):
        self.feature_names_ = list(X.columns)
        self.model_ = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=self.seed,
            n_jobs=-1,
            scale_pos_weight=_scale_pos_weight(y),
            **self.params,
        )
        self.model_.fit(X.to_numpy(dtype=float), y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X.to_numpy(dtype=float))[:, 1]


class RFModel:
    """RandomForest binary classifier; median-imputes + standardises first (no native NaN)."""

    def __init__(self, params: dict, seed: int = 0) -> None:
        self.params = dict(params)
        self.seed = seed
        self.model_: RandomForestClassifier | None = None
        self.prep_: Preprocessor | None = None
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None):
        self.feature_names_ = list(X.columns)
        self.prep_ = Preprocessor().fit(X)
        Xt = self.prep_.transform(X)
        self.model_ = RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=self.seed,
            n_jobs=-1,
            **self.params,
        )
        self.model_.fit(Xt, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self.prep_.transform(X))[:, 1]


def xgb_param_space(trial) -> dict:
    """Optuna search space for XGBoost (regularised, shallow -- small noisy panel)."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 2.0, log=True),
    }


def rf_param_space(trial) -> dict:
    """Optuna search space for RandomForest."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
    }


def xgb_baseline_params() -> dict:
    """Fixed shallow XGB used by the barrier search (cheap, no per-config Optuna)."""
    return {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    }
