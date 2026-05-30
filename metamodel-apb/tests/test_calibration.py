"""EX.4 probability calibration (RED-first).

Kelly sizing consumes p̂ directly, so miscalibration -> mis-sizing (part of the AUC≠P&L story).
We assess calibration (reliability curve + ECE) and post-process with Platt / isotonic. The
invariants: a well-calibrated input has a near-diagonal reliability curve / low ECE, an
overconfident input has higher ECE, isotonic mapping is monotone, and Platt fit on miscalibrated
scores lowers the ECE.
"""

from __future__ import annotations

import numpy as np

from alken_metamodel.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    expected_calibration_error,
    reliability_curve,
)


def _calibrated(n=8000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(size=n) < p).astype(float)  # P(y=1) == p  -> perfectly calibrated
    return p, y


def _overconfident(n=8000, seed=1):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(size=n) < p * 0.5).astype(float)  # true prob is half the stated p
    return p, y


def test_reliability_curve_is_diagonal_when_calibrated():
    p, y = _calibrated()
    curve = reliability_curve(y, p, n_bins=10)
    assert {"mean_pred", "frac_pos", "count"}.issubset(curve.columns)
    diag_gap = (curve["mean_pred"] - curve["frac_pos"]).abs().max()
    assert diag_gap < 0.06  # close to the y=x diagonal


def test_ece_lower_for_calibrated_than_overconfident():
    ece_cal = expected_calibration_error(*reversed(_calibrated()))
    ece_bad = expected_calibration_error(*reversed(_overconfident()))
    assert ece_cal < 0.05
    assert ece_bad > ece_cal


def test_isotonic_mapping_is_monotone():
    p, y = _overconfident(seed=3)
    cal = IsotonicCalibrator().fit(p, y)
    grid = np.linspace(0.0, 1.0, 50)
    mapped = cal.transform(grid)
    assert np.all(np.diff(mapped) >= -1e-9)  # non-decreasing


def test_platt_reduces_ece_on_overconfident_scores():
    p, y = _overconfident(seed=4)
    before = expected_calibration_error(y, p)
    cal = PlattCalibrator().fit(p, y)
    after = expected_calibration_error(y, cal.transform(p))
    assert after < before  # Platt scaling pulls the overconfident scores toward truth
