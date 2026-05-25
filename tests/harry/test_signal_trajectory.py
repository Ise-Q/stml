"""Unit tests for ``stml.harry.features.signal_trajectory``.

Truncation-invariance, shape, and no-NaN-past-warmup are covered by
``tests/harry/test_causality.py`` via the auto-discovered
``CAUSALITY_REGISTRATIONS`` constant. The tests in this file cover
hand-computed correctness and the per-feature property checks the
Step-3 spec required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.harry.features.signal_trajectory import (
    signal_cum_pnl_20d,
    signal_entropy_20d,
    signal_flip_rate_60d,
    signal_run_length,
    time_since_last_flip,
)


# --------------------------------------------------------------------------- #
# Hand-computed correctness                                                     #
# --------------------------------------------------------------------------- #
def test_signal_run_length_hand_computed():
    s = pd.Series([1, 1, -1, -1, 0, -1, -1])
    expected = pd.Series([1, 2, 1, 2, 1, 1, 2], dtype="int64")
    out = signal_run_length(s)
    pd.testing.assert_series_equal(
        out.reset_index(drop=True),
        expected,
        check_names=False,
    )


def test_time_since_last_flip_hand_computed():
    s = pd.Series([1, 1, -1, -1, 0, -1, -1])
    expected = pd.Series([0, 1, 0, 1, 0, 0, 1], dtype="int64")
    out = time_since_last_flip(s)
    pd.testing.assert_series_equal(
        out.reset_index(drop=True),
        expected,
        check_names=False,
    )


def test_signal_entropy_hand_computed_uniform_window():
    # 3 distinct values, equal frequency → entropy = log(3).
    s = pd.Series([1, 1, 0, 0, -1, -1])
    out = signal_entropy_20d(s, window=6)
    assert np.isnan(out.iloc[:5]).all()
    assert out.iloc[5] == pytest.approx(np.log(3))


def test_signal_entropy_hand_computed_concentrated_window():
    # All identical → entropy = 0.
    s = pd.Series([1, 1, 1, 1])
    out = signal_entropy_20d(s, window=4)
    assert out.iloc[3] == pytest.approx(0.0)


def test_signal_entropy_hand_computed_two_state_window():
    # 1, 1, 0, -1 → p(1)=0.5, p(0)=0.25, p(-1)=0.25.
    # H = -[0.5·log(0.5) + 0.25·log(0.25)·2]
    s = pd.Series([1, 1, 0, -1])
    out = signal_entropy_20d(s, window=4)
    expected = -(0.5 * np.log(0.5) + 2 * 0.25 * np.log(0.25))
    assert out.iloc[3] == pytest.approx(expected)


def test_signal_flip_rate_hand_computed():
    # window=2; flips at u=1..4 are [(1!=1) F, (0!=1) T, (-1!=0) T, (1!=-1) T].
    # Rolling-2 mean: NaN, NaN (row-0 flip is NaN), 0.5 (mean(NaN_skipped? no:
    # rolling with min_periods=2 needs both non-NaN -> row 1 is NaN, row 2
    # uses flips[1..2] = [F, T] -> 0.5), row 3: flips[2..3] = [T, T] -> 1.0,
    # row 4: flips[3..4] = [T, T] -> 1.0.
    s = pd.Series([1, 1, 0, -1, 1])
    out = signal_flip_rate_60d(s, window=2)
    assert np.isnan(out.iloc[0])
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(0.5)
    assert out.iloc[3] == pytest.approx(1.0)
    assert out.iloc[4] == pytest.approx(1.0)


def test_signal_cum_pnl_hand_computed():
    s = pd.Series([1.0, -1.0, 1.0, 0.0])
    r = pd.Series([0.01, -0.02, 0.005, 0.03])
    out = signal_cum_pnl_20d(s, r, window=3)
    # products: [0.01, 0.02, 0.005, 0.0]
    # rolling(3).sum from row 2: 0.01 + 0.02 + 0.005 = 0.035 at row 2,
    # 0.02 + 0.005 + 0.0 = 0.025 at row 3.
    assert np.isnan(out.iloc[0])
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(0.035)
    assert out.iloc[3] == pytest.approx(0.025)


# --------------------------------------------------------------------------- #
# Property checks                                                              #
# --------------------------------------------------------------------------- #
def test_signal_run_length_is_positive_integer():
    rng = np.random.default_rng(42)
    s = pd.Series(rng.choice([-1, 0, 1], size=500))
    out = signal_run_length(s)
    assert out.dtype == np.int64
    assert (out >= 1).all()


def test_time_since_last_flip_is_non_negative_integer():
    rng = np.random.default_rng(42)
    s = pd.Series(rng.choice([-1, 0, 1], size=500))
    out = time_since_last_flip(s)
    assert out.dtype == np.int64
    assert (out >= 0).all()


def test_signal_entropy_in_zero_log3():
    rng = np.random.default_rng(42)
    s = pd.Series(rng.choice([-1, 0, 1], size=500))
    h = signal_entropy_20d(s)
    h = h.dropna()
    assert (h >= -1e-12).all()
    assert (h <= np.log(3) + 1e-12).all()


def test_signal_flip_rate_in_zero_one():
    rng = np.random.default_rng(42)
    s = pd.Series(rng.choice([-1, 0, 1], size=500))
    fr = signal_flip_rate_60d(s)
    fr = fr.dropna()
    assert (fr >= -1e-12).all()
    assert (fr <= 1.0 + 1e-12).all()


def test_signal_cum_pnl_can_be_negative_and_positive():
    """It's an unbounded PnL accumulator — both signs should occur on
    a random panel."""
    rng = np.random.default_rng(42)
    s = pd.Series(rng.choice([-1, 0, 1], size=500))
    r = pd.Series(rng.normal(0, 0.01, 500))
    out = signal_cum_pnl_20d(s, r)
    out = out.dropna()
    assert (out > 0).any()
    assert (out < 0).any()


# --------------------------------------------------------------------------- #
# Cross-consistency with time_since_last_flip                                  #
# --------------------------------------------------------------------------- #
def test_run_length_minus_one_equals_time_since_last_flip():
    rng = np.random.default_rng(42)
    s = pd.Series(rng.choice([-1, 0, 1], size=300))
    a = signal_run_length(s) - 1
    b = time_since_last_flip(s)
    pd.testing.assert_series_equal(
        a.astype("int64").reset_index(drop=True),
        b.reset_index(drop=True),
        check_names=False,
    )
