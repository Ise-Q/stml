"""
linear.py
=========
The **Linear** meta-model family: regularised logistic regression. This is the third model
family the coursework requires (linear / tree / neural), and a natural calibrated-by-default
baseline -- the guide notes logistic regression is usually already well-calibrated, unlike the
tree and boosting families.

Wrapped in the same uniform interface as the trees and nets -- ``fit(X, y, sample_weight=None)``
/ ``predict_proba(X) -> P(class=1)`` / ``feature_names_`` -- so the Optuna objective, importance
and evaluation code treat it identically. Logistic regression cannot consume NaNs and is scale
sensitive, so (like :class:`~stml.model.trees.RFModel`) it median-imputes + standardises via a
:class:`~stml.model.dataset.Preprocessor` fit on the training rows only -- never on validation.

Imbalance is handled with ``class_weight='balanced'`` (mirroring the trees' built-in handling),
and the López de Prado sample-uniqueness weights are passed straight through to ``fit``. The
``saga`` solver supports the full L1 / L2 / elastic-net penalty range the search explores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from stml.model.dataset import Preprocessor


class LogRegModel:
    """Regularised logistic-regression binary classifier; standardises inputs first."""

    def __init__(self, params: dict, seed: int = 0) -> None:
        self.params = dict(params)
        self.seed = seed
        self.model_: LogisticRegression | None = None
        self.prep_: Preprocessor | None = None
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None):
        self.feature_names_ = list(X.columns)
        self.prep_ = Preprocessor().fit(X)
        Xt = self.prep_.transform(X)
        self.model_ = LogisticRegression(
            solver="saga",
            class_weight="balanced",
            max_iter=5000,
            random_state=self.seed,
            **self.params,
        )
        self.model_.fit(Xt, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self.prep_.transform(X))[:, 1]

    @property
    def coef_(self) -> np.ndarray:
        """Fitted linear weights (one per feature), for inspection / linear importance."""
        return self.model_.coef_.ravel()


def logreg_param_space(trial) -> dict:
    """Optuna search space for regularised logistic regression.

    Explores penalty strength ``C`` (inverse regularisation, log-scale) across L1 / L2 /
    elastic-net. ``l1_ratio`` is only sampled for the elastic-net penalty (conditional space).
    """
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    params: dict = {
        "penalty": penalty,
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
    }
    if penalty == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
    return params
