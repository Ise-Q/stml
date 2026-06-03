"""Model roster — plan §3.1 / §8 Stage 3.

Three estimators behind one uniform :class:`MetaClassifier` interface:

* :func:`make_elasticnet_logistic` — sklearn elastic-net LogisticRegression,
  standardised inputs, median-imputed NaNs.
* :func:`make_xgb` — XGBoost with the PS5 cell-43 / alken-parity config
  (binary:logistic, max_depth=4, lr=0.05, subsample=0.8, colsample=0.8,
  reg_alpha=0.1, reg_lambda=1.0, n_jobs=1). Single-threaded for byte-stable
  re-fit. **Handles NaN natively** — critical for the R-11 BBG-missingness
  ablation (XGB consumes NaNs without imputation).
* :func:`make_random_forest` — sklearn RandomForestClassifier (max_features='sqrt'
  — Harry's PS4 bug fix; auto would error on sklearn ≥1.3).

All estimators implement:

    fit(X, y, sample_weight=...) -> self
    predict_proba(X)              -> ndarray (n, 2)
    predict_act_proba(X)          -> ndarray (n,)  — P(class == 1)

Sample weighting (plan §3.1):

    balanced_sample_weight(y, base=uniqueness) ↦
        per-sample weight = uniqueness × inverse_class_frequency

Lifted with attribution from ``metamodel-apb/src/alken_metamodel/models.py``
(alken parity) and ``src/stml/models.py`` (Sreeram, the original wrappers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def balanced_sample_weight(
    y: np.ndarray | pd.Series, base: np.ndarray | pd.Series | None = None
) -> np.ndarray:
    """Per-sample weight = ``base`` (e.g. uniqueness) × inverse class frequency.

    Each class's total weight is rescaled to ``n / n_classes`` so the classes
    carry equal mass while within-class proportions of ``base`` are preserved.
    """
    y = np.asarray(y)
    w = np.ones(len(y), dtype=float) if base is None else np.asarray(base, dtype=float).copy()
    classes = np.unique(y)
    if len(classes) < 2:
        return w
    target = len(y) / len(classes)
    for c in classes:
        mask = y == c
        total = w[mask].sum()
        if total > 0:
            w[mask] *= target / total
    return w


@dataclass
class MetaClassifier:
    """Uniform fit / predict interface.

    Attributes
    ----------
    name : human-readable name.
    base : the sklearn-compatible underlying estimator.
    scale : if True, the inputs get median-imputed and standardised before fit
        (logistic path). XGBoost / RF consume NaNs natively, so this is False
        for them.
    """

    name: str
    base: BaseEstimator
    scale: bool = False
    _scaler: StandardScaler | None = field(default=None, init=False, repr=False)
    _imputer: SimpleImputer | None = field(default=None, init=False, repr=False)
    _feature_names: list[str] | None = field(default=None, init=False, repr=False)

    def _to_array(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self._feature_names is None:
                self._feature_names = list(X.columns)
            arr = X.values.astype(float)
        else:
            arr = np.asarray(X, dtype=float)
        # Replace +/- inf with NaN — XGBoost rejects inf but handles NaN.
        # Affects features like ratios with near-zero denominators.
        arr = np.where(np.isfinite(arr), arr, np.nan)
        return arr

    def fit(self, X, y, sample_weight=None) -> "MetaClassifier":
        x = self._to_array(X)
        if self.scale:
            self._imputer = SimpleImputer(strategy="median").fit(x)
            self._scaler = StandardScaler().fit(self._imputer.transform(x))
            x = self._scaler.transform(self._imputer.transform(x))
        # sklearn estimators that don't support sample_weight gracefully will
        # raise; both XGB and LogReg support it.
        self.base.fit(x, np.asarray(y), sample_weight=sample_weight)
        return self

    def predict_proba(self, X) -> np.ndarray:
        x = self._to_array(X)
        if self._imputer is not None:
            x = self._imputer.transform(x)
        if self._scaler is not None:
            x = self._scaler.transform(x)
        return self.base.predict_proba(x)

    def predict_act_proba(self, X) -> np.ndarray:
        """P(class == 1) — the 'act' probability in meta-labelling."""
        proba = self.predict_proba(X)
        classes = list(self.base.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        return np.zeros(proba.shape[0])  # degenerate single-class fold

    @property
    def feature_names(self) -> list[str] | None:
        return self._feature_names

    def feature_importance(self) -> pd.Series | None:
        """Return feature importance as a Series if the base supports it."""
        if hasattr(self.base, "feature_importances_"):
            imp = self.base.feature_importances_
            if self._feature_names:
                return pd.Series(imp, index=self._feature_names).sort_values(ascending=False)
            return pd.Series(imp).sort_values(ascending=False)
        if hasattr(self.base, "coef_"):
            coef = self.base.coef_.ravel()
            if self._feature_names:
                return pd.Series(coef, index=self._feature_names)
            return pd.Series(coef)
        return None


# ---------------------------------------------------------------------------
# Constructors.
# ---------------------------------------------------------------------------


def make_elasticnet_logistic(
    *, seed: int = 42, l1_ratio: float = 0.5, C: float = 1.0, max_iter: int = 5000
) -> MetaClassifier:
    """Elastic-net penalised LogReg, saga solver, standardised inputs."""
    # sklearn ≥1.8: penalty='elasticnet' deprecated — l1_ratio drives the mix
    # (0=ridge, 1=lasso). n_jobs has no effect since 1.8 — omitted.
    base = LogisticRegression(
        solver="saga",
        l1_ratio=l1_ratio,
        C=C,
        max_iter=max_iter,
        random_state=seed,
    )
    return MetaClassifier("elasticnet_logistic", base, scale=True)


def make_xgb(*, seed: int = 42, **overrides: Any) -> MetaClassifier:
    """XGBoost — PS5 cell-43 config (plan §3.1 / alken parity).

    NaN-tolerant by default — critical for the R-11 BBG-missingness ablation.
    """
    from xgboost import XGBClassifier  # lazy import

    config = dict(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
        tree_method="hist",  # deterministic, fast, NaN-aware
    )
    config.update(overrides)
    return MetaClassifier("xgboost", XGBClassifier(**config), scale=False)


def make_random_forest(
    *, seed: int = 42, max_depth: int = 6, n_estimators: int = 200,
    min_samples_leaf: int = 10, **overrides: Any
) -> MetaClassifier:
    """RandomForest with the four PS4 bug fixes (max_features='sqrt' included)."""
    config = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",  # plan §3.6 — alken's first §4 bug fix
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )
    config.update(overrides)
    # RF doesn't handle NaN natively — wrap with imputation via scale=True path
    # (which median-imputes then standardises; RF doesn't need the standardisation
    # but it does no harm and keeps the interface uniform).
    return MetaClassifier("random_forest", RandomForestClassifier(**config), scale=True)


def default_roster(seed: int = 42) -> dict[str, MetaClassifier]:
    """The plan §8 S3 roster — three families (linear / tree / tree-bagging)."""
    return {
        "elasticnet_logistic": make_elasticnet_logistic(seed=seed),
        "xgboost": make_xgb(seed=seed),
        "random_forest": make_random_forest(seed=seed),
    }
