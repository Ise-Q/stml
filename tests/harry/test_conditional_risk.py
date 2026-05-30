"""Unit tests for ``stml.harry.features.conditional_risk``.

Truncation-invariance, shape, and no-NaN-after-warmup are covered by
``tests/harry/test_causality.py``. The tests here are hand-computed
correctness checks and per-feature property assertions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.harry.features.conditional_risk import (
    _simulate_first_passage,
    expected_hit_time,
    path_tortuosity_20d,
    prob_timeout,
    realized_semi_vol_ratio,
)


# --------------------------------------------------------------------------- #
# Hand-computed: path_tortuosity_20d                                           #
# --------------------------------------------------------------------------- #
def test_path_tortuosity_monotone_path_is_one():
    """A monotone-positive return path has |net move| == sum |moves| → 1."""
    r = pd.Series([0.01] * 6)
    out = path_tortuosity_20d(r, window=5)
    # rows 4 and 5 use the full 5-bar window of [0.01]*5: sum=0.05, abs_sum=0.05.
    assert out.iloc[4] == pytest.approx(1.0)
    assert out.iloc[5] == pytest.approx(1.0)


def test_path_tortuosity_hand_computed_zigzag():
    """A 4-bar window with alternating +/− 0.01 has net 0, abs_sum 0.04 →
    very large tortuosity (4e10 with the documented epsilon)."""
    r = pd.Series([0.01, -0.01, 0.01, -0.01])
    out = path_tortuosity_20d(r, window=4)
    # net = 0, abs_sum = 0.04, denom = 0 + 1e-12 = 1e-12 → 4e10.
    assert out.iloc[3] == pytest.approx(0.04 / 1e-12, rel=1e-6)


def test_path_tortuosity_partial_zigzag():
    """A path with net move 0.01 and abs_sum 0.07 → ratio 7."""
    r = pd.Series([0.04, -0.03, 0.02, -0.02])
    # sum = 0.01, abs_sum = 0.11 → 11. Let me recompute:
    # values: 0.04, -0.03, 0.02, -0.02
    # sum = 0.04 - 0.03 + 0.02 - 0.02 = 0.01
    # abs_sum = 0.04 + 0.03 + 0.02 + 0.02 = 0.11
    # ratio = 0.11 / (0.01 + 1e-12) ≈ 11.0
    out = path_tortuosity_20d(r, window=4)
    assert out.iloc[3] == pytest.approx(11.0, rel=1e-6)


# --------------------------------------------------------------------------- #
# Hand-computed: realized_semi_vol_ratio                                       #
# --------------------------------------------------------------------------- #
def test_realized_semi_vol_ratio_hand_computed():
    """r = [-0.01, +0.02, -0.03, +0.04] with window=4:
    pos² mean = (0.0004 + 0.0016) / 4 = 0.0005 → rms = √0.0005
    neg² mean = (0.0001 + 0.0009) / 4 = 0.00025 → rms = √0.00025
    ratio = √(0.0005 / 0.00025) = √2.
    """
    r = pd.Series([-0.01, 0.02, -0.03, 0.04])
    out = realized_semi_vol_ratio(r, window=4)
    assert out.iloc[3] == pytest.approx(np.sqrt(2.0), rel=1e-6)


def test_realized_semi_vol_ratio_zero_negative_returns():
    """When the trailing window has no negative returns, the ratio should
    blow up but remain finite (epsilon-protected). All positives means
    rms_neg = 0 → ratio = rms_pos / eps."""
    r = pd.Series([0.01, 0.02, 0.03, 0.04])
    out = realized_semi_vol_ratio(r, window=4)
    assert np.isfinite(out.iloc[3])
    assert out.iloc[3] > 1e6  # very large but finite


# --------------------------------------------------------------------------- #
# Property checks                                                              #
# --------------------------------------------------------------------------- #
def test_path_tortuosity_always_non_negative():
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(0, 0.01, 300))
    out = path_tortuosity_20d(r).dropna()
    assert (out >= 0).all()


def test_realized_semi_vol_ratio_non_negative():
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(0, 0.01, 300))
    out = realized_semi_vol_ratio(r).dropna()
    assert (out >= 0).all()


def test_prob_timeout_in_zero_one():
    rng = np.random.default_rng(42)
    n = 400
    r = pd.Series(rng.normal(0, 0.01, n))
    vol = r.rolling(60, min_periods=60).std()
    out = prob_timeout(
        r, vol, pt_mult=1.0, sl_mult=1.0, h=10,
        window=80, n_sims=40,
    ).dropna()
    assert (out >= 0).all()
    assert (out <= 1.0).all()


def test_expected_hit_time_in_one_to_h_plus_one():
    """First-passage time is an integer-valued day index in [1, h] when
    a touch occurs, with h+1 reserved for "no path touched" sentinel.
    The median falls in the same range."""
    rng = np.random.default_rng(42)
    n = 400
    r = pd.Series(rng.normal(0, 0.01, n))
    vol = r.rolling(60, min_periods=60).std()
    h = 10
    out = expected_hit_time(
        r, vol, pt_mult=1.0, sl_mult=1.0, h=h,
        window=80, n_sims=40,
    ).dropna()
    assert (out >= 1.0).all()
    assert (out <= h + 1).all()


def test_first_passage_simulation_returns_both_outputs():
    """``_simulate_first_passage`` returns two aligned Series."""
    rng = np.random.default_rng(42)
    n = 300
    r = pd.Series(rng.normal(0, 0.01, n))
    vol = r.rolling(60, min_periods=60).std()
    hit, pto = _simulate_first_passage(
        r, vol, pt_mult=1.0, sl_mult=1.0, h=10,
        window=80, n_sims=40, seed=42,
    )
    assert len(hit) == n
    assert len(pto) == n
    # Defined past the window; same NaN pattern across both.
    assert hit.dropna().index.equals(pto.dropna().index)


def test_first_passage_simulation_determinism():
    """Same seed and inputs produce identical outputs."""
    rng = np.random.default_rng(42)
    n = 300
    r = pd.Series(rng.normal(0, 0.01, n))
    vol = r.rolling(60, min_periods=60).std()
    a_hit, a_pto = _simulate_first_passage(
        r, vol, pt_mult=1.0, sl_mult=1.0, h=10,
        window=80, n_sims=40, seed=42,
    )
    b_hit, b_pto = _simulate_first_passage(
        r, vol, pt_mult=1.0, sl_mult=1.0, h=10,
        window=80, n_sims=40, seed=42,
    )
    pd.testing.assert_series_equal(a_hit, b_hit)
    pd.testing.assert_series_equal(a_pto, b_pto)


# --------------------------------------------------------------------------- #
# Input validation                                                             #
# --------------------------------------------------------------------------- #
def test_first_passage_simulation_rejects_bad_inputs():
    r = pd.Series([0.0] * 100)
    vol = pd.Series([0.01] * 100)
    with pytest.raises(ValueError):
        _simulate_first_passage(r, vol, pt_mult=1.0, sl_mult=1.0, h=0)
    with pytest.raises(ValueError):
        _simulate_first_passage(r, vol, pt_mult=1.0, sl_mult=1.0, h=10, window=1)
    with pytest.raises(ValueError):
        _simulate_first_passage(r, vol, pt_mult=1.0, sl_mult=1.0, h=10, n_sims=0)
