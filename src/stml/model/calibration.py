"""
calibration.py
==============
Per-family probability calibration on dev out-of-fold predictions only.

Tree/boosting models are systematically miscalibrated: RandomForest is under-confident
(predictions cluster near 0.5) while XGBoost/LightGBM are over-confident (predictions
push toward {0,1}); logistic regression is typically well-calibrated by construction.
To size positions on ``p̂`` it must satisfy ``Pr(y=1 | p̂) ≈ p̂`` -- i.e. a predicted
probability of 0.7 should correspond to ~70 % of trades being profitable.

**Fitting discipline:** calibrators MUST be fit on out-of-fold CPCV pairs ``(p̂_i, y_i)``.
Fitting on in-sample predictions learns "already calibrated" and gives a false reliability
curve that collapses the moment the model runs on unseen data.

Two methods are provided:

* **Platt scaling** -- a 1-D logistic fit on the log-odds of the raw probability.
  Only two parameters; robust on small datasets; the default choice.
* **Isotonic regression** -- a non-parametric monotone fit; more flexible but requires
  substantially more data to avoid overfitting the calibration itself.

Usage::

    calibrator = calibrate_oof(p_oof, y, method="platt")
    p_calibrated = calibrator.transform(p_test)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """1-D logistic calibrator fit on the log-odds of the raw probability.

    Platt scaling learns two parameters ``(a, b)`` such that
    ``p_cal = σ(a · logit(p_raw) + b)``, aligning the reliability curve without
    distorting the rank ordering of predictions.

    Attributes
    ----------
    lr_ : LogisticRegression
        Fitted sklearn estimator (available after :meth:`fit`).
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self.lr_: LogisticRegression | None = None

    def fit(self, p_oof: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        """Fit on out-of-fold ``(p_oof, y)`` pairs only."""
        p_oof = np.asarray(p_oof, dtype=float)
        y = np.asarray(y, dtype=int)
        z = _logit(p_oof).reshape(-1, 1)
        self.lr_ = LogisticRegression(
            C=1e9,  # near-zero regularisation; Platt is already low-capacity
            solver="lbfgs",
            random_state=self._seed,
            max_iter=1000,
        )
        self.lr_.fit(z, y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        """Map raw probabilities to calibrated probabilities in [0, 1]."""
        if self.lr_ is None:
            raise RuntimeError("PlattCalibrator must be fitted before transform.")
        p = np.asarray(p, dtype=float)
        z = _logit(p).reshape(-1, 1)
        return self.lr_.predict_proba(z)[:, 1]

    def predict(self, p: np.ndarray) -> np.ndarray:
        """Alias of :meth:`transform`."""
        return self.transform(p)


class IsotonicCalibrator:
    """Non-parametric monotone calibrator via isotonic regression.

    More flexible than Platt scaling but requires substantially more data; prefer
    Platt on small datasets (< ~1 000 OOF pairs). The ``out_of_bounds="clip"``
    flag ensures predictions outside the training range are clamped rather than
    extrapolated.

    Attributes
    ----------
    iso_ : IsotonicRegression
        Fitted sklearn estimator (available after :meth:`fit`).
    """

    def __init__(self) -> None:
        self.iso_: IsotonicRegression | None = None

    def fit(self, p_oof: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        """Fit on out-of-fold ``(p_oof, y)`` pairs only."""
        p_oof = np.asarray(p_oof, dtype=float)
        y = np.asarray(y, dtype=int)
        self.iso_ = IsotonicRegression(out_of_bounds="clip")
        self.iso_.fit(p_oof, y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        """Map raw probabilities to calibrated probabilities in [0, 1]."""
        if self.iso_ is None:
            raise RuntimeError("IsotonicCalibrator must be fitted before transform.")
        p = np.asarray(p, dtype=float)
        return self.iso_.transform(p)

    def predict(self, p: np.ndarray) -> np.ndarray:
        """Alias of :meth:`transform`."""
        return self.transform(p)


def calibrate_oof(
    p_oof: np.ndarray,
    y: np.ndarray,
    *,
    method: str = "platt",
    seed: int = 42,
) -> PlattCalibrator | IsotonicCalibrator:
    """Fit and return a calibrator on out-of-fold ``(p_oof, y)`` pairs.

    Parameters
    ----------
    p_oof:
        Raw predicted probabilities from OOF folds; shape ``(n,)``.
    y:
        Binary labels; shape ``(n,)``.
    method:
        ``"platt"`` (default) or ``"isotonic"``.
    seed:
        Random seed for Platt's LogisticRegression.

    Returns
    -------
    PlattCalibrator | IsotonicCalibrator
        A fitted calibrator with :meth:`transform` and :meth:`predict`.

    Raises
    ------
    ValueError
        If ``method`` is not ``"platt"`` or ``"isotonic"``.
    """
    if method == "platt":
        return PlattCalibrator(seed=seed).fit(p_oof, y)
    elif method == "isotonic":
        return IsotonicCalibrator().fit(p_oof, y)
    else:
        raise ValueError(f"Unknown calibration method {method!r}; choose 'platt' or 'isotonic'.")


def reliability_table(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute a reliability (calibration) table for equal-width bins on [0, 1].

    Parameters
    ----------
    y:
        Binary labels; shape ``(n,)``.
    p:
        Predicted probabilities; shape ``(n,)``.
    n_bins:
        Number of equal-width bins on [0, 1].

    Returns
    -------
    pd.DataFrame
        Columns: ``bin`` (int, 0-indexed), ``p_mean``, ``y_freq``, ``count``.
        Only non-empty bins are returned; ``count`` column sums to ``len(y)``.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize gives 1-based; clip to [0, n_bins-1] for 0-based bin index
    bin_idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            rows.append({
                "bin": b,
                "p_mean": float(p[mask].mean()),
                "y_freq": float(y[mask].mean()),
                "count": int(mask.sum()),
            })
    return pd.DataFrame(rows, columns=["bin", "p_mean", "y_freq", "count"])


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Brier score (lower is better, 0 = perfect).

    Thin wrapper around :func:`sklearn.metrics.brier_score_loss`.
    """
    return float(brier_score_loss(np.asarray(y, dtype=int), np.asarray(p, dtype=float)))


def calibration_report(
    y: np.ndarray,
    p_raw: np.ndarray,
    p_cal: np.ndarray,
) -> dict:
    """Summarise calibration quality before and after recalibration.

    Parameters
    ----------
    y:
        Binary labels.
    p_raw:
        Raw (uncalibrated) predicted probabilities.
    p_cal:
        Calibrated predicted probabilities.

    Returns
    -------
    dict with keys:

    ``brier_raw``
        Brier score of the raw probabilities.
    ``brier_cal``
        Brier score of the calibrated probabilities.
    ``ece_raw``
        Expected Calibration Error of raw probabilities (weighted mean absolute
        reliability-table deviation).
    ``ece_cal``
        Expected Calibration Error of calibrated probabilities.
    ``monotonic``
        ``True`` if ``p_cal`` is a non-decreasing function of ``p_raw`` (up to
        floating-point noise of 1e-9); ``False`` otherwise.
    """
    y = np.asarray(y, dtype=int)
    p_raw = np.asarray(p_raw, dtype=float)
    p_cal = np.asarray(p_cal, dtype=float)
    n = len(y)

    def _ece(p: np.ndarray) -> float:
        tbl = reliability_table(y, p)
        return float((tbl["count"] / n * (tbl["p_mean"] - tbl["y_freq"]).abs()).sum())

    # Monotonicity: sort by raw prob and check calibrated is non-decreasing
    order = np.argsort(p_raw)
    p_cal_sorted = p_cal[order]
    monotonic = bool(np.all(np.diff(p_cal_sorted) >= -1e-9))

    return {
        "brier_raw": brier(y, p_raw),
        "brier_cal": brier(y, p_cal),
        "ece_raw": _ece(p_raw),
        "ece_cal": _ece(p_cal),
        "monotonic": monotonic,
    }
