"""Probability calibration — methodology spec / §8 S6.

Per-class Platt scaling fit on **purged modelling-OOF predictions** (strictly
inside the ``train + val`` partition so no test-period information leaks).

Platt is monotone → AUC is unchanged before/after calibration (unit-tested
invariant). Only Brier / log-loss / ECE / the sizing stake move.

Lifted with attribution from

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class PlattCalibrator:
    """1-D sigmoid calibration via LogReg on the raw probabilities."""

    _logreg: LogisticRegression | None = None

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        scores = np.asarray(scores, dtype=float).reshape(-1, 1)
        y_true = np.asarray(y_true, dtype=int)
        if len(np.unique(y_true)) < 2:
            # Degenerate — return identity calibrator.
            self._logreg = None
            return self
        self._logreg = LogisticRegression(solver="lbfgs", C=1e6, max_iter=2000)
        self._logreg.fit(scores, y_true)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float).reshape(-1, 1)
        if self._logreg is None:
            return scores.ravel()
        return self._logreg.predict_proba(scores)[:, 1]


@dataclass
class IsotonicCalibrator:
    """Free-form monotone calibration (clipped to [0, 1] out of range)."""

    _iso: IsotonicRegression | None = None

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        scores = np.asarray(scores, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        if len(np.unique(y_true)) < 2:
            self._iso = None
            return self
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(scores, y_true)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self._iso is None:
            return np.asarray(scores, dtype=float)
        return self._iso.transform(np.asarray(scores, dtype=float))


def reliability_curve(
    y_true: np.ndarray, proba: np.ndarray, *, n_bins: int = 10
) -> pd.DataFrame:
    """Per-bin mean predicted prob vs realised positive frequency."""
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(proba, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin": b,
                "n": int(mask.sum()),
                "mean_pred": float(proba[mask].mean()),
                "actual_pos": float(y_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: np.ndarray, proba: np.ndarray, *, n_bins: int = 10
) -> float:
    """Weighted L1 gap between predicted and actual positive frequency."""
    rc = reliability_curve(y_true, proba, n_bins=n_bins)
    if rc.empty:
        return float("nan")
    n_total = rc["n"].sum()
    return float((rc["n"] * (rc["mean_pred"] - rc["actual_pos"]).abs()).sum() / n_total)
