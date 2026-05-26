"""Unit tests for ``stml.harry.features.microstructure_fixed``.

Universal causality / shape / no-NaN-past-warmup checks live in
``tests/harry/test_causality.py``. The tests here cover hand-computed
correctness, the zero-volume mask (the headline fix vs. other branches),
and property assertions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.harry.features.microstructure_fixed import (
    _mask_zero_volume,
    amihud_illiquidity,
    kyles_lambda,
    overnight_gap,
    rolls_effective_spread,
)


# --------------------------------------------------------------------------- #
# Zero-volume mask                                                             #
# --------------------------------------------------------------------------- #
def test_mask_zero_volume_drops_zero_and_nan():
    v = pd.Series([1.0, 0.0, 5.0, np.nan, 3.0, 0.0])
    out = _mask_zero_volume(v)
    assert out.iloc[0] == 1.0
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == 5.0
    assert np.isnan(out.iloc[3])
    assert out.iloc[4] == 3.0
    assert np.isnan(out.iloc[5])


def test_amihud_zero_volume_yields_nan_not_inf():
    """The headline fix: divide-by-zero on zero-volume rows must NOT
    produce Inf. The output for a window containing a zero-volume row is
    NaN under strict min_periods, but the surrounding finite values are
    untouched."""
    r = pd.Series([0.01, -0.02, 0.005, 0.01, -0.015, 0.02, 0.01])
    v = pd.Series([100.0, 200.0, 0.0, 100.0, 150.0, 200.0, 100.0])
    out = amihud_illiquidity(r, v, window=3)
    assert np.isfinite(out.dropna()).all()
    # The middle window (rows 0..2) contains the zero-volume row → NaN.
    assert np.isnan(out.iloc[2])
    # Once past the zero-volume row's window, rows are finite.
    assert np.isfinite(out.iloc[5])


# --------------------------------------------------------------------------- #
# amihud_illiquidity — hand-computed                                            #
# --------------------------------------------------------------------------- #
def test_amihud_hand_computed():
    """r = [0.02, -0.04, 0.06], v = [100, 200, 300], window=3:
    illiq = [0.02/100, 0.04/200, 0.06/300] = [2e-4, 2e-4, 2e-4]
    mean over 3 = 2e-4 at row 2.
    """
    r = pd.Series([0.02, -0.04, 0.06])
    v = pd.Series([100.0, 200.0, 300.0])
    out = amihud_illiquidity(r, v, window=3)
    assert np.isnan(out.iloc[0])
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2e-4)


# --------------------------------------------------------------------------- #
# rolls_effective_spread — hand-computed                                       #
# --------------------------------------------------------------------------- #
def test_rolls_spread_zero_when_returns_positively_correlated():
    """A monotone price series has Cov(Δp_t, Δp_{t-1}) > 0 → -Cov < 0 →
    clipped to 0 → spread = 0."""
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    out = rolls_effective_spread(close, window=4)
    assert out.dropna().iloc[-1] == pytest.approx(0.0)


def test_rolls_spread_positive_on_bouncing_series():
    """A perfectly alternating series Δp = [+1, -1, +1, -1, ...] has
    Cov(Δp_t, Δp_{t-1}) = -1 → -Cov = +1 → spread = 2."""
    close = pd.Series([100.0, 101.0, 100.0, 101.0, 100.0, 101.0])
    out = rolls_effective_spread(close, window=4)
    val = out.dropna().iloc[-1]
    assert val > 0


# --------------------------------------------------------------------------- #
# kyles_lambda — property checks                                               #
# --------------------------------------------------------------------------- #
def test_kyles_lambda_non_negative():
    rng = np.random.default_rng(42)
    n = 200
    r = pd.Series(rng.normal(0, 0.01, n))
    v = pd.Series(rng.integers(1_000, 100_000, n).astype(float))
    out = kyles_lambda(r, v).dropna()
    assert (out >= 0).all()


def test_kyles_lambda_higher_for_lower_volume():
    """Same return series, divided by lower volume → larger lambda."""
    r = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01])
    v_low = pd.Series([100.0] * 5)
    v_high = pd.Series([10_000.0] * 5)
    a = kyles_lambda(r, v_low, window=3).dropna().iloc[-1]
    b = kyles_lambda(r, v_high, window=3).dropna().iloc[-1]
    assert a > b


# --------------------------------------------------------------------------- #
# overnight_gap — hand-computed                                                #
# --------------------------------------------------------------------------- #
def test_overnight_gap_hand_computed():
    """log(101 / 100) = ~0.00995."""
    open_ = pd.Series([100.0, 101.0, 102.0])
    close_prev = pd.Series([np.nan, 100.0, 101.0])
    out = overnight_gap(open_, close_prev)
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(np.log(1.01))
    assert out.iloc[2] == pytest.approx(np.log(102.0 / 101.0))


def test_overnight_gap_signs():
    """log(open/close_prev) > 0 iff open > close_prev."""
    open_ = pd.Series([105.0, 95.0])
    close_prev = pd.Series([100.0, 100.0])
    out = overnight_gap(open_, close_prev)
    assert out.iloc[0] > 0
    assert out.iloc[1] < 0
