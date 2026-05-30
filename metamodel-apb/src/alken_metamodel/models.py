"""Act/skip model roster for the meta-labelling horse-race (Stage 2).

This module hosts the **tree/linear core** of the §3 horse-race — elastic-net logistic
regression, XGBoost, and LightGBM — behind one uniform interface (``MetaClassifier``) so the
comparison is apples-to-apples and the validation harness (``evaluation.py``) can treat every
estimator identically. The three neural variants (torch-MLP, torch-VSN, Keras-VSN) are added
in a follow-up before the fan-out gate.

Design choices (justified against the dossier of PS4/PS5/PS6):
- **One weighting channel.** PS4/5/6 ship no class- or sample-weighting at all. Meta-labels
  are both *overlapping* (need López de Prado uniqueness weights, Ch.4) and *imbalanced*
  (~30–40% positive, nlr-cw §1). Both are folded into a single ``sample_weight`` passed
  identically to every estimator's ``fit`` (``balanced_sample_weight`` composes uniqueness ×
  inverse-class-frequency), instead of estimator-specific ``scale_pos_weight``/``class_weight``.
- **XGBoost** uses the PS5 cell-43 configuration (binary:logistic, the fixed classifier — note
  PS5 already uses ``max_features='sqrt'``; the deprecated ``'auto'`` bug lives only in PS4's RF
  grid and is fixed in Stage 3). **LightGBM** mirrors that regularised configuration.
- **Determinism.** ``set_seeds`` at every fit; single-threaded tree building and LightGBM
  ``deterministic=True`` so a re-fit is byte-stable (the grader re-runs on the hidden half).
- **NaN policy is per-model.** Trees consume NaN natively (better than imputing). The scaled
  (logistic) path imputes with a train-fitted median, then standardises — so the pooled
  real-data matrix's structural NaNs don't break the linear model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def balanced_sample_weight(y, base=None) -> np.ndarray:
    """Compose a per-sample weight = ``base`` (e.g. uniqueness) × inverse class frequency.

    Each class's total weight is rescaled to ``n / n_classes`` so the classes carry equal
    mass, while the within-class proportions of ``base`` are preserved. ``base=None`` -> ones.
    """
    y = np.asarray(y)
    w = np.ones(len(y), dtype=float) if base is None else np.asarray(base, dtype=float).copy()
    classes = np.unique(y)
    target = len(y) / len(classes)
    for c in classes:
        mask = y == c
        total = w[mask].sum()
        if total > 0:
            w[mask] *= target / total
    return w


@dataclass
class MetaClassifier:
    """Uniform act/skip wrapper: ``fit(X, y, sample_weight=)`` / ``predict_act_proba(X)``.

    ``predict_act_proba`` returns P(class == 1) ("act"), mapped through ``classes_`` so it is
    robust to a degenerate single-class fold.
    """

    name: str
    base: BaseEstimator
    scale: bool = False
    _scaler: StandardScaler | None = field(default=None, init=False, repr=False)
    _imputer: SimpleImputer | None = field(default=None, init=False, repr=False)

    def fit(self, X, y, sample_weight=None) -> MetaClassifier:
        # Determinism comes from each estimator's constructor ``random_state`` plus the
        # single-threaded native kernels (conftest / entry-point env), NOT a per-fit global
        # reseed — seeding belongs once at the pipeline entry point (CLAUDE.md convention).
        x = np.asarray(X, dtype=float)
        if self.scale:  # logistic: train-fitted median impute -> standardise
            self._imputer = SimpleImputer(strategy="median").fit(x)
            self._scaler = StandardScaler().fit(self._imputer.transform(x))
            x = self._scaler.transform(self._imputer.transform(x))
        self.base.fit(x, np.asarray(y), sample_weight=sample_weight)
        return self

    def predict_proba(self, X) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        if self._imputer is not None:
            x = self._imputer.transform(x)
        if self._scaler is not None:
            x = self._scaler.transform(x)
        return self.base.predict_proba(x)

    def predict_act_proba(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        classes = list(self.base.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        return np.zeros(proba.shape[0])  # no positive class present in this fold


def make_elasticnet_logistic(
    *, seed: int = 42, l1_ratio: float = 0.5, C: float = 1.0, max_iter: int = 5000
) -> MetaClassifier:
    """Elastic-net penalised logistic regression (saga solver), standardised inputs."""
    # sklearn >=1.8 drives the elastic-net mix via l1_ratio (0=ridge..1=lasso) with the saga
    # solver; the explicit penalty= kwarg is deprecated.
    base = LogisticRegression(
        solver="saga",
        l1_ratio=l1_ratio,
        C=C,
        max_iter=max_iter,
        random_state=seed,
    )
    return MetaClassifier("elasticnet_logistic", base, scale=True)


def make_xgb(*, seed: int = 42) -> MetaClassifier:
    """XGBoost classifier — PS5 cell-43 configuration, single-threaded for determinism."""
    base = XGBClassifier(
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
    )
    return MetaClassifier("xgboost", base, scale=False)


def make_lightgbm(*, seed: int = 42) -> MetaClassifier:
    """LightGBM classifier mirroring the regularised XGBoost configuration (deterministic)."""
    base = LGBMClassifier(
        n_estimators=200,
        num_leaves=15,
        max_depth=-1,
        learning_rate=0.05,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=5,
        random_state=seed,
        n_jobs=1,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    return MetaClassifier("lightgbm", base, scale=False)


def tree_linear_roster(*, seed: int = 42) -> dict[str, MetaClassifier]:
    """The three tree/linear act/skip estimators, keyed by name."""
    return {
        "elasticnet_logistic": make_elasticnet_logistic(seed=seed),
        "xgboost": make_xgb(seed=seed),
        "lightgbm": make_lightgbm(seed=seed),
    }
