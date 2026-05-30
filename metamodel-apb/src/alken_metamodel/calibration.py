"""Probability calibration for EX.4 (reliability curve, ECE, Platt, isotonic).

The metamodel's p̂ feeds fractional-Kelly sizing directly, so a miscalibrated probability is a
mis-sized bet — a candidate explanation for the AUC≠P&L finding (a model can rank act/skip well
yet be systematically over/under-confident). This module measures calibration (binned reliability
curve + expected calibration error) and supplies the two standard post-hoc fixes: Platt scaling
(a logistic fit, nlr-cw §2 / Gramegna–Giudici 2021) and isotonic regression (monotone, free-form).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def reliability_curve(y_true, proba, *, n_bins: int = 10) -> pd.DataFrame:
    """Binned reliability table: mean predicted prob vs realised positive frequency per bin."""
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(proba, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            rows.append(
                {
                    "bin": b,
                    "mean_pred": float(proba[mask].mean()),
                    "frac_pos": float(y_true[mask].mean()),
                    "count": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, proba, *, n_bins: int = 10) -> float:
    """ECE = Σ_bins (count/N) · |mean_pred − frac_pos| (weighted gap from the diagonal)."""
    curve = reliability_curve(y_true, proba, n_bins=n_bins)
    if curve.empty:
        return float("nan")
    total = curve["count"].sum()
    gap = (curve["mean_pred"] - curve["frac_pos"]).abs()
    return float((curve["count"] / total * gap).sum())


class PlattCalibrator:
    """Platt scaling: a 1-D logistic fit mapping raw scores -> calibrated probabilities."""

    def fit(self, proba, y) -> PlattCalibrator:
        x = np.asarray(proba, dtype=float).reshape(-1, 1)
        self._lr = LogisticRegression().fit(x, np.asarray(y, dtype=float))
        return self

    def transform(self, proba) -> np.ndarray:
        x = np.asarray(proba, dtype=float).reshape(-1, 1)
        return self._lr.predict_proba(x)[:, 1]


class IsotonicCalibrator:
    """Isotonic regression: a free-form monotone score -> probability map (clipped out of range)."""

    def fit(self, proba, y) -> IsotonicCalibrator:
        self._iso = IsotonicRegression(out_of_bounds="clip").fit(
            np.asarray(proba, dtype=float), np.asarray(y, dtype=float)
        )
        return self

    def transform(self, proba) -> np.ndarray:
        return self._iso.predict(np.asarray(proba, dtype=float))
