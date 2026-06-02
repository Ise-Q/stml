"""
Tests for src/stml/model/calibration.py.

Uses a deterministic miscalibrated probability set built with rng=42 so results
are reproducible without loading any real data.
"""

from __future__ import annotations

import numpy as np
import pytest

from stml.model.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    brier,
    calibrate_oof,
    calibration_report,
    reliability_table,
)


# ---------------------------------------------------------------------------
# Shared synthetic fixture
# ---------------------------------------------------------------------------


def _make_miscalibrated(n: int = 500, seed: int = 42):
    """Return (y, p_over, p_under) -- two deliberately miscalibrated prob vectors.

    p_over:  over-confident (XGBoost-style) -- raw logits pushed toward {0,1}.
    p_under: under-confident (RF-style) -- squeezed toward 0.5.
    """
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(n)
    true_prob = 1 / (1 + np.exp(-latent))
    y = (rng.uniform(size=n) < true_prob).astype(int)

    # Over-confident: exaggerate the logit
    logit_over = latent * 2.5
    p_over = 1 / (1 + np.exp(-logit_over))

    # Under-confident: squash logit toward 0
    logit_under = latent * 0.3
    p_under = 1 / (1 + np.exp(-logit_under))

    return y, p_over, p_under


Y, P_OVER, P_UNDER = _make_miscalibrated()


# ---------------------------------------------------------------------------
# IsotonicCalibrator
# ---------------------------------------------------------------------------


class TestIsotonicCalibrator:
    def test_transform_monotone(self):
        """Calibrated outputs are non-decreasing over sorted raw inputs."""
        cal = IsotonicCalibrator().fit(P_UNDER, Y)
        sorted_p = np.sort(P_UNDER)
        p_cal = cal.transform(sorted_p)
        assert np.all(np.diff(p_cal) >= -1e-9), "IsotonicCalibrator transform is not monotone"

    def test_brier_improves_in_sample(self):
        """In-sample Brier strictly improves after isotonic calibration (well-known property)."""
        cal = IsotonicCalibrator().fit(P_UNDER, Y)
        p_cal = cal.transform(P_UNDER)
        assert brier(Y, p_cal) <= brier(Y, P_UNDER), "Brier did not improve after isotonic calibration"

    def test_outputs_in_unit_interval(self):
        cal = IsotonicCalibrator().fit(P_OVER, Y)
        p_cal = cal.transform(P_OVER)
        assert np.all(p_cal >= 0.0) and np.all(p_cal <= 1.0)

    def test_predict_alias(self):
        cal = IsotonicCalibrator().fit(P_OVER, Y)
        np.testing.assert_array_equal(cal.predict(P_OVER), cal.transform(P_OVER))

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            IsotonicCalibrator().transform(P_OVER)


# ---------------------------------------------------------------------------
# PlattCalibrator
# ---------------------------------------------------------------------------


class TestPlattCalibrator:
    def test_monotone_function_of_raw(self):
        """Platt is a monotone transformation -- sorted raw => sorted calibrated."""
        cal = PlattCalibrator(seed=42).fit(P_OVER, Y)
        sorted_p = np.sort(P_OVER)
        p_cal = cal.transform(sorted_p)
        assert np.all(np.diff(p_cal) >= -1e-9), "PlattCalibrator output is not monotone in raw prob"

    def test_outputs_in_unit_interval(self):
        cal = PlattCalibrator(seed=42).fit(P_UNDER, Y)
        p_cal = cal.transform(P_UNDER)
        assert np.all(p_cal >= 0.0) and np.all(p_cal <= 1.0)

    def test_predict_alias(self):
        cal = PlattCalibrator(seed=42).fit(P_OVER, Y)
        np.testing.assert_array_equal(cal.predict(P_OVER), cal.transform(P_OVER))

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            PlattCalibrator().transform(P_OVER)

    def test_fit_returns_self(self):
        cal = PlattCalibrator(seed=42)
        result = cal.fit(P_OVER, Y)
        assert result is cal


# ---------------------------------------------------------------------------
# reliability_table
# ---------------------------------------------------------------------------


class TestReliabilityTable:
    def test_count_sums_to_n(self):
        tbl = reliability_table(Y, P_OVER, n_bins=10)
        assert tbl["count"].sum() == len(Y)

    def test_p_mean_in_unit_interval(self):
        tbl = reliability_table(Y, P_OVER, n_bins=10)
        assert tbl["p_mean"].between(0.0, 1.0).all()

    def test_y_freq_in_unit_interval(self):
        tbl = reliability_table(Y, P_OVER, n_bins=10)
        assert tbl["y_freq"].between(0.0, 1.0).all()

    def test_columns_present(self):
        tbl = reliability_table(Y, P_OVER)
        assert set(tbl.columns) == {"bin", "p_mean", "y_freq", "count"}

    def test_non_empty_bins_only(self):
        """Uniform probs should fill all bins; no extra empty rows."""
        rng = np.random.default_rng(0)
        p_uniform = rng.uniform(size=1000)
        y_uniform = (rng.uniform(size=1000) < 0.5).astype(int)
        tbl = reliability_table(y_uniform, p_uniform, n_bins=10)
        assert len(tbl) == 10  # all bins populated for large uniform sample

    def test_custom_n_bins(self):
        tbl = reliability_table(Y, P_OVER, n_bins=5)
        assert tbl["count"].sum() == len(Y)
        assert len(tbl) <= 5


# ---------------------------------------------------------------------------
# calibrate_oof
# ---------------------------------------------------------------------------


class TestCalibrateOof:
    def test_platt_returns_fitted_calibrator(self):
        cal = calibrate_oof(P_OVER, Y, method="platt", seed=42)
        assert isinstance(cal, PlattCalibrator)
        p_cal = cal.transform(P_OVER)
        assert p_cal.shape == P_OVER.shape

    def test_isotonic_returns_fitted_calibrator(self):
        cal = calibrate_oof(P_UNDER, Y, method="isotonic")
        assert isinstance(cal, IsotonicCalibrator)
        p_cal = cal.transform(P_UNDER)
        assert p_cal.shape == P_UNDER.shape

    def test_bad_method_raises(self):
        with pytest.raises(ValueError, match="Unknown calibration method"):
            calibrate_oof(P_OVER, Y, method="magic")

    def test_default_method_is_platt(self):
        cal = calibrate_oof(P_OVER, Y)
        assert isinstance(cal, PlattCalibrator)


# ---------------------------------------------------------------------------
# calibration_report
# ---------------------------------------------------------------------------


class TestCalibrationReport:
    def _make_report_isotonic(self):
        cal = calibrate_oof(P_UNDER, Y, method="isotonic")
        p_cal = cal.transform(P_UNDER)
        return calibration_report(Y, P_UNDER, p_cal)

    def test_five_keys_present(self):
        report = self._make_report_isotonic()
        assert set(report.keys()) == {"brier_raw", "brier_cal", "ece_raw", "ece_cal", "monotonic"}

    def test_brier_cal_le_brier_raw_isotonic(self):
        """In-sample isotonic calibration must lower the Brier score."""
        report = self._make_report_isotonic()
        assert report["brier_cal"] <= report["brier_raw"], (
            f"brier_cal={report['brier_cal']:.4f} > brier_raw={report['brier_raw']:.4f}"
        )

    def test_monotonic_is_bool(self):
        report = self._make_report_isotonic()
        assert isinstance(report["monotonic"], bool)

    def test_monotonic_true_for_platt(self):
        """Platt is a strictly monotone logistic -- should always be monotonic."""
        cal = calibrate_oof(P_OVER, Y, method="platt", seed=42)
        p_cal = cal.transform(P_OVER)
        report = calibration_report(Y, P_OVER, p_cal)
        assert report["monotonic"] is True

    def test_ece_values_are_non_negative(self):
        report = self._make_report_isotonic()
        assert report["ece_raw"] >= 0.0
        assert report["ece_cal"] >= 0.0
