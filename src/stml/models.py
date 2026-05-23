"""
models.py
=========
Model wrappers for the meta-model. Stage 2b provides the baseline:

  - :class:`ElasticNetLogReg`  -- L1+L2 penalised logistic regression with
    purged-CV hyperparameter tuning + isotonic probability calibration.

The model interface is sklearn-flavoured (``fit`` / ``predict_proba``) so it
plugs cleanly into the master pipeline and into evaluation utilities.

Stages 4 (XGBoost, VSN) will add additional model classes in this module or
sibling modules — all sharing the same calling convention.

Conventions
-----------
- ``X``: pd.DataFrame indexed by event id; columns are features.
- ``y``: pd.Series of binary labels {0, 1} indexed identically to X.
- ``sample_weight``: pd.Series of non-negative weights, same index. We pass
  through to sklearn's ``fit(..., sample_weight=...)``.
- ``predict_proba``: returns the probability of class 1 (label = profitable).
- ``threshold``: optional decision threshold (default 0.5) for hard predictions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stml.cv import PurgedKFold


class ElasticNetLogReg:
    """L1+L2-penalised logistic regression with purged-CV tuning and isotonic
    probability calibration.

    Pipeline (post-fit):
        StandardScaler -> LogisticRegression(elasticnet) -> Isotonic calibration

    The internal scaler is fit on training data only; calibration uses a
    held-out fold to remap raw scores to true probabilities (so reliability is
    a first-class output rather than an after-thought).
    """

    def __init__(
        self,
        n_splits_inner: int = 5,
        n_iter: int = 20,
        embargo_td: Optional[pd.Timedelta] = None,
        random_state: int = 42,
        l1_ratio_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
        C_log_grid: tuple[float, ...] = (-3, -2, -1, 0, 1, 2),
        verbose: int = 0,
    ):
        self.n_splits_inner = n_splits_inner
        self.n_iter = n_iter
        self.embargo_td = embargo_td
        self.random_state = random_state
        self.l1_ratio_grid = list(l1_ratio_grid)
        self.C_log_grid = list(C_log_grid)
        self.verbose = verbose

        # Set at fit time:
        self.best_params_: Optional[dict] = None
        self.calibrator_: Optional[CalibratedClassifierCV] = None
        self.feature_names_: Optional[list[str]] = None

    # ------------------------------------------------------------------ #
    def _make_pipeline(self) -> Pipeline:
        # `liblinear` doesn't support elasticnet; use `saga`.
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                max_iter=5000,
                random_state=self.random_state,
                class_weight="balanced",
                l1_ratio=0.5,
                C=1.0,
            )),
        ])

    def _param_distributions(self) -> dict:
        return {
            "clf__C": [10**g for g in self.C_log_grid],
            "clf__l1_ratio": self.l1_ratio_grid,
        }

    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        t: pd.Series,
        t1: pd.Series,
        sample_weight: Optional[pd.Series] = None,
    ) -> "ElasticNetLogReg":
        """Tune hyperparameters with purged K-fold CV, then refit + calibrate.

        Parameters
        ----------
        X : pd.DataFrame indexed by event id.
        y : pd.Series of {0,1} labels.
        t, t1 : event start / end dates, indexed identically to X.
        sample_weight : optional uniqueness weights (passed through to fit).
        """
        self.feature_names_ = list(X.columns)

        cv = PurgedKFold(
            n_splits=self.n_splits_inner, t=t, t1=t1, embargo_td=self.embargo_td
        )

        pipe = self._make_pipeline()
        # Use the SAGA solver, which supports elastic-net and sample_weight.
        search = RandomizedSearchCV(
            pipe,
            param_distributions=self._param_distributions(),
            n_iter=self.n_iter,
            cv=cv,
            scoring="neg_log_loss",
            n_jobs=-1,
            random_state=self.random_state,
            verbose=self.verbose,
            refit=True,
        )

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["clf__sample_weight"] = sample_weight.values

        # Pass X as DataFrame so PurgedKFold can read the event-id index;
        # the sklearn pipeline (StandardScaler -> LogisticRegression) handles
        # the DataFrame -> array conversion internally.
        search.fit(X, y.values, **fit_kwargs)
        self.best_params_ = search.best_params_
        self._search = search

        # Calibrate the *tuned* pipeline on a held-out fold (cv=PurgedKFold).
        # CalibratedClassifierCV with cv='prefit' would skip the held-out
        # split; instead pass our PurgedKFold so calibration is also purged.
        cal_cv = PurgedKFold(
            n_splits=min(3, self.n_splits_inner),
            t=t, t1=t1, embargo_td=self.embargo_td,
        )
        self.calibrator_ = CalibratedClassifierCV(
            estimator=search.best_estimator_,
            method="isotonic",
            cv=cal_cv,
        )
        # CalibratedClassifierCV needs the splitter to be applied to (X, y)
        # but our PurgedKFold reads its index from X. Calling with arrays
        # would lose the index alignment — pass DataFrames.
        cal_fit_kwargs = {}
        if sample_weight is not None:
            cal_fit_kwargs["sample_weight"] = sample_weight.values
        self.calibrator_.fit(X, y, **cal_fit_kwargs)
        return self

    # ------------------------------------------------------------------ #
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated P(label = 1) for each row."""
        if self.calibrator_ is None:
            raise RuntimeError("Model is not fitted. Call .fit() first.")
        return self.calibrator_.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # ------------------------------------------------------------------ #
    def linear_coefficients(self) -> pd.Series:
        """Coefficients of the *underlying* (uncalibrated) elastic-net.

        Useful for sanity-checking feature signs. Note these are on the
        standardised feature scale (scaler is fit inside the pipeline).
        """
        if self._search is None:
            raise RuntimeError("Model is not fitted.")
        pipe = self._search.best_estimator_
        coefs = pipe.named_steps["clf"].coef_[0]
        return pd.Series(coefs, index=self.feature_names_, name="coef").sort_values()
