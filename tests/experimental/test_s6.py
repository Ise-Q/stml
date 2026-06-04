"""S6 tests — calibration / sizing / cost_model / backtest / emit.

Plan §8 S6 RED-first invariants.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Calibration — Platt is monotone → AUC invariant.
# ---------------------------------------------------------------------------


def test_platt_preserves_auc():
    """Platt calibration is monotone → AUC unchanged."""
    from stml.experimental.calibration import PlattCalibrator

    rng = np.random.default_rng(0)
    n = 500
    y = rng.integers(0, 2, n)
    raw = np.clip(rng.normal(0.5, 0.2, n) + 0.3 * (y - 0.5), 0, 1)
    pc = PlattCalibrator().fit(raw, y)
    cal = pc.transform(raw)
    auc_raw = roc_auc_score(y, raw)
    auc_cal = roc_auc_score(y, cal)
    np.testing.assert_allclose(auc_raw, auc_cal, atol=1e-9)


def test_platt_clips_extreme_probabilities():
    """Calibrated probabilities are in [0, 1]."""
    from stml.experimental.calibration import PlattCalibrator
    rng = np.random.default_rng(1)
    n = 200
    y = rng.integers(0, 2, n)
    raw = rng.uniform(0, 1, n)
    pc = PlattCalibrator().fit(raw, y)
    cal = pc.transform(raw)
    assert (cal >= 0).all() and (cal <= 1).all()


def test_isotonic_monotone():
    """Isotonic transform of sorted input is non-decreasing."""
    from stml.experimental.calibration import IsotonicCalibrator
    rng = np.random.default_rng(2)
    n = 200
    raw = np.sort(rng.uniform(0, 1, n))
    y = (raw + 0.2 * rng.standard_normal(n) > 0.5).astype(int)
    ic = IsotonicCalibrator().fit(raw, y)
    out = ic.transform(np.sort(rng.uniform(0, 1, 50)))
    assert (np.diff(out) >= -1e-9).all()


def test_ece_formula_on_perfect_calibration():
    """ECE of perfectly-calibrated predictions = 0."""
    from stml.experimental.calibration import expected_calibration_error
    n = 1000
    rng = np.random.default_rng(3)
    proba = rng.uniform(0, 1, n)
    y_true = (rng.uniform(0, 1, n) < proba).astype(int)  # perfectly calibrated
    ece = expected_calibration_error(y_true, proba, n_bins=10)
    assert ece < 0.05


# ---------------------------------------------------------------------------
# Sizing.
# ---------------------------------------------------------------------------


def test_kelly_fraction_symmetric():
    """Symmetric Kelly: f* = 2p - 1."""
    from stml.experimental.sizing import kelly_fraction
    assert kelly_fraction(0.6) == pytest.approx(0.2, abs=1e-9)
    assert kelly_fraction(0.5) == pytest.approx(0.0, abs=1e-9)
    assert kelly_fraction(0.4) == pytest.approx(-0.2, abs=1e-9)


def test_fractional_kelly_hard_floor():
    """Probability below floor → zero weight."""
    from stml.experimental.sizing import fractional_kelly
    assert fractional_kelly(0.50) == 0.0  # below 0.55 floor
    assert fractional_kelly(0.54) == 0.0
    # Above floor — kappa * (2p-1) * 1.
    val = fractional_kelly(0.70, kappa=0.25)
    assert val == pytest.approx(0.25 * 0.4, abs=1e-9)


def test_vol_target_leverage_clips():
    """Realised vol > target → leverage <= 1; realised < target → leverage > 1
    but clipped to max_leverage."""
    from stml.experimental.sizing import vol_target_leverage
    # Realised 0.2, target 0.08 → 0.4
    assert vol_target_leverage(0.20, target_vol=0.08) == pytest.approx(0.4, abs=1e-9)
    # Realised tiny → clipped to max_leverage=5.
    assert vol_target_leverage(0.001, target_vol=0.08, max_leverage=5.0) == 5.0
    # Zero vol → 0.
    assert vol_target_leverage(0.0) == 0.0


def test_position_weight_sign_follows_side():
    """side = +1 → positive weight; side = -1 → negative weight."""
    from stml.experimental.sizing import position_weight
    long_w = position_weight(+1, 0.7, realised_vol=0.10)
    short_w = position_weight(-1, 0.7, realised_vol=0.10)
    assert long_w > 0
    assert short_w < 0
    assert long_w == -short_w


# ---------------------------------------------------------------------------
# Cost model — Grinold-Kahn linearity.
# ---------------------------------------------------------------------------


def test_transaction_costs_linear_in_delta():
    """Cost = (half_spread + impact) × |Δw| under impact_exponent=1."""
    from stml.experimental.cost_model import transaction_costs
    idx = pd.date_range("2020-01-01", periods=4)
    weights = pd.DataFrame(
        {"A": [0.0, 0.5, 0.5, 0.0], "B": [0.0, 0.0, 0.5, 0.5]},
        index=idx,
    )
    costs = transaction_costs(weights, half_spread_bps=2.0, impact_bps=10.0)
    # Day 0: turnover = |0.0 - flat| = 0; cost = 0.
    # Day 1: |Δw_A| = 0.5; cost = (2+10)/10000 × 0.5 = 0.0006
    # Day 2: |Δw_B| = 0.5; cost = 0.0006
    # Day 3: |Δw_A| = 0.5; cost = 0.0006
    assert costs.iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert costs.iloc[1] == pytest.approx(0.0006, abs=1e-9)


# ---------------------------------------------------------------------------
# Backtest — Sortino-Price full-T known-value.
# ---------------------------------------------------------------------------


def test_sortino_full_t_known_value():
    """Hand-computed Sortino-Price on a tiny series."""
    from stml.experimental.backtest import sortino_full_t
    # Returns: 0.05, -0.03, 0.02, -0.04
    r = pd.Series([0.05, -0.03, 0.02, -0.04])
    # Excess (target 0) = same as r.
    # Downside = [0, -0.03, 0, -0.04].
    # Semivariance = (0 + 0.0009 + 0 + 0.0016) / 4 = 0.000625
    # Per-period Sortino = mean(r) / sqrt(semivar)
    # mean = (0.05 - 0.03 + 0.02 - 0.04) / 4 = 0
    # So per-period Sortino = 0; annualised = 0.
    val = sortino_full_t(r, ann=252.0)
    np.testing.assert_allclose(val, 0.0, atol=1e-9)


def test_sortino_full_t_differs_from_std_negatives_only_misimpl():
    """The Sortino-Price full-T form differs from the std(neg-returns) mis-impl.

    The common mis-implementation uses ``std(r[r<0], ddof=1)`` which subtracts
    the mean OF THE NEGATIVES (itself negative), producing a smaller spread
    estimate than the semivariance around 0 → inflates Sortino. Our full-T
    form correctly uses semivariance around 0 with denominator T.
    """
    from stml.experimental.backtest import sortino_full_t
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 252))
    val_full_t = sortino_full_t(r, ann=252.0)
    # Mis-impl: std of negative returns around their own mean.
    neg = r[r < 0]
    std_neg = neg.std(ddof=1)
    mis = (r.mean() / std_neg) * np.sqrt(252) if std_neg > 0 else float("nan")
    # Values must be different — the test is that we are NOT computing
    # the mis-impl; magnitude direction depends on distribution.
    assert not np.isclose(val_full_t, mis, atol=1e-6)


def test_performance_metrics_basic_shape():
    from stml.experimental.backtest import performance_metrics
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 252))
    m = performance_metrics(r)
    for k in ("n", "total_return", "ann_return", "ann_vol", "sharpe", "sortino", "max_dd"):
        assert k in m
    assert m["n"] == 252


# ---------------------------------------------------------------------------
# Emit — byte-identical re-emit.
# ---------------------------------------------------------------------------


def test_emit_predictions_byte_identical(tmp_path):
    """Two emits of the same predictions frame → byte-identical CSV."""
    from stml.experimental.emit import emit_predictions
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=10),
        "instrument": ["cl1s"] * 10,
        "prediction": rng.uniform(0, 1, 10),
    })
    p1 = tmp_path / "p1.csv"
    p2 = tmp_path / "p2.csv"
    emit_predictions(df, p1)
    emit_predictions(df, p2)
    assert p1.read_bytes() == p2.read_bytes()


def test_emit_predictions_sorted_and_formatted(tmp_path):
    """Output is sorted by (date, instrument); ISO dates; %.10f floats."""
    from stml.experimental.emit import emit_predictions
    df = pd.DataFrame({
        "date": pd.to_datetime(["2022-01-03", "2022-01-02"]),
        "instrument": ["es1s", "cl1s"],
        "prediction": [0.7, 0.5],
    })
    p = tmp_path / "out.csv"
    emit_predictions(df, p)
    text = p.read_text()
    assert text.startswith("date,instrument,prediction\n")
    # First data row should be 2022-01-02,cl1s (earlier date).
    assert "2022-01-02,cl1s,0.5000000000" in text
