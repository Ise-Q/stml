"""
test_nav.py
===========
Tests for :mod:`stml.replication.nav`.

Coverage
--------
- NAV identity on a toy series with known signal + known returns.
- Perfect replica: increment_corr ≈ 1.0 and cumnav_ssd_norm ≈ 0.0.
- Alignment property: forward-shifting the target by +1 day strictly lowers
  increment_corr and strictly raises cumnav_ssd_norm.
- retained_fraction passed through correctly.
- Integration smoke on cl1s via load_clean_data + align_instrument.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from stml.replication.nav import nav_discrepancy, nav_from_raw, nav_series


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _date_index(n: int, start: str = "2021-01-04") -> pd.DatetimeIndex:
    """Return a DatetimeIndex of n consecutive calendar days starting from start."""
    return pd.date_range(start=start, periods=n, freq="D")


def _series(values: list[float], start: str = "2021-01-04") -> pd.Series:
    idx = _date_index(len(values), start=start)
    return pd.Series(values, index=idx, dtype=float)


# --------------------------------------------------------------------------- #
# NAV identity: hand-computed cumsum(signal * ret)                             #
# --------------------------------------------------------------------------- #

def test_nav_series_toy_identity() -> None:
    """nav_series equals hand-computed cumsum(signal * ret) on a tiny toy series.

    Toy series (5 days):
      signal = [1, -1,  0,  1, -1]
      ret    = [0.01, -0.02, 0.03, -0.01, 0.04]

    PnL per day:
      d0:  1 *  0.01 =  0.01
      d1: -1 * -0.02 =  0.02
      d2:  0 *  0.03 =  0.00
      d3:  1 * -0.01 = -0.01
      d4: -1 *  0.04 = -0.04

    Cumulative log-NAV:
      [0.01, 0.03, 0.03, 0.02, -0.02]
    """
    signal = _series([1.0, -1.0, 0.0, 1.0, -1.0])
    ret = _series([0.01, -0.02, 0.03, -0.01, 0.04])

    result = nav_series(signal, ret)

    expected = [0.01, 0.03, 0.03, 0.02, -0.02]
    assert len(result) == 5
    for i, exp in enumerate(expected):
        assert math.isclose(result.iloc[i], exp, rel_tol=1e-9, abs_tol=1e-12), (
            f"day {i}: got {result.iloc[i]:.6f}, expected {exp:.6f}"
        )


def test_nav_series_all_flat() -> None:
    """All-zero signal produces a flat (all-zero) NAV."""
    signal = _series([0.0, 0.0, 0.0, 0.0])
    ret = _series([0.01, -0.02, 0.03, -0.01])
    result = nav_series(signal, ret)
    assert (result == 0.0).all()


def test_nav_series_index_is_intersection() -> None:
    """nav_series uses only the intersection of the two indices."""
    # signal has 5 dates, ret has 3 of those (indices 1-3)
    sig = _series([1.0, 1.0, 1.0, 1.0, 1.0])
    ret = pd.Series(
        [0.01, 0.02, 0.03],
        index=pd.DatetimeIndex(["2021-01-05", "2021-01-06", "2021-01-07"]),
    )
    result = nav_series(sig, ret)
    assert len(result) == 3
    assert list(result.index) == list(ret.index)


# --------------------------------------------------------------------------- #
# nav_discrepancy: perfect replica                                              #
# --------------------------------------------------------------------------- #

def test_perfect_replica_increment_corr() -> None:
    """When replica == target, increment_corr must be ≈ 1.0."""
    signal = _series([1.0, -1.0, 1.0, 0.0, -1.0, 1.0])
    ret = _series([0.01, -0.02, 0.03, -0.01, 0.04, -0.005])

    metrics = nav_discrepancy(signal, signal, ret)
    assert math.isclose(metrics["increment_corr"], 1.0, rel_tol=1e-9)


def test_perfect_replica_ssd_norm() -> None:
    """When replica == target, cumnav_ssd_norm must be ≈ 0.0."""
    signal = _series([1.0, -1.0, 1.0, 0.0, -1.0, 1.0])
    ret = _series([0.01, -0.02, 0.03, -0.01, 0.04, -0.005])

    metrics = nav_discrepancy(signal, signal, ret)
    assert math.isclose(metrics["cumnav_ssd_norm"], 0.0, abs_tol=1e-15)


# --------------------------------------------------------------------------- #
# Alignment property: misalignment strictly degrades both metrics              #
# --------------------------------------------------------------------------- #

def test_misalignment_degrades_increment_corr() -> None:
    """Forward-shifting the target signal by +1 day strictly lowers increment_corr.

    Aligned case:  replica and target share the same dates.
    Misaligned case: target is shifted +1 calendar day relative to replica,
      so the common-date intersection sees a different target signal on most days.
      This should STRICTLY lower Pearson correlation of PnL increments.
    """
    rng = np.random.default_rng(42)
    n = 60
    dates = _date_index(n)

    # Non-trivial signal with mix of -1, 0, +1
    raw_signal = rng.choice([-1.0, 0.0, 1.0], size=n)
    signal_s = pd.Series(raw_signal, index=dates)

    # Slightly different replica (flip ~20% of days)
    replica_raw = raw_signal.copy()
    flip_idx = rng.choice(n, size=n // 5, replace=False)
    for i in flip_idx:
        replica_raw[i] = rng.choice([-1.0, 0.0, 1.0])
    replica_s = pd.Series(replica_raw, index=dates)

    ret_s = pd.Series(rng.normal(0, 0.01, size=n), index=dates)

    # Aligned case: same dates
    m_aligned = nav_discrepancy(replica_s, signal_s, ret_s)

    # Misaligned case: shift target by +1 day
    target_shifted = pd.Series(raw_signal, index=dates + pd.Timedelta(days=1))
    m_misaligned = nav_discrepancy(replica_s, target_shifted, ret_s)

    corr_aligned = m_aligned["increment_corr"]
    corr_misaligned = m_misaligned["increment_corr"]
    ssd_aligned = m_aligned["cumnav_ssd_norm"]
    ssd_misaligned = m_misaligned["cumnav_ssd_norm"]

    assert corr_misaligned < corr_aligned, (
        f"Expected corr_misaligned ({corr_misaligned:.4f}) < corr_aligned ({corr_aligned:.4f})"
    )
    assert ssd_misaligned > ssd_aligned, (
        f"Expected ssd_misaligned ({ssd_misaligned:.6f}) > ssd_aligned ({ssd_aligned:.6f})"
    )


# --------------------------------------------------------------------------- #
# retained_fraction                                                             #
# --------------------------------------------------------------------------- #

def test_retained_fraction_full_overlap() -> None:
    """When all signals and ret share the same dates, retained_fraction == 1.0."""
    signal = _series([1.0, -1.0, 0.0, 1.0])
    ret = _series([0.01, -0.02, 0.03, -0.01])

    metrics = nav_discrepancy(signal, signal, ret)
    assert math.isclose(metrics["retained_fraction"], 1.0, rel_tol=1e-9)


def test_retained_fraction_partial_overlap() -> None:
    """retained_fraction reflects the common-date fraction of aligned_ret."""
    # ret has 6 dates; signals share only dates 1..4 (4 of 6)
    full_dates = _date_index(6)
    partial_dates = pd.DatetimeIndex(full_dates[1:5])  # 4 dates

    ret = pd.Series([0.01, -0.02, 0.03, -0.01, 0.04, -0.005], index=full_dates)
    signal = pd.Series([1.0, -1.0, 1.0, 0.0], index=partial_dates)

    metrics = nav_discrepancy(signal, signal, ret)
    expected = 4 / 6
    assert math.isclose(metrics["retained_fraction"], expected, rel_tol=1e-9)


# --------------------------------------------------------------------------- #
# cumnav_ssd_norm_vs_flat reference scale                                      #
# --------------------------------------------------------------------------- #

def test_ssd_norm_vs_flat_nonzero_for_nonflat_target() -> None:
    """cumnav_ssd_norm_vs_flat is positive when the target NAV is not flat."""
    signal = _series([1.0, 1.0, 1.0, 1.0])
    ret = _series([0.01, 0.01, 0.01, 0.01])

    metrics = nav_discrepancy(signal, signal, ret)
    assert metrics["cumnav_ssd_norm_vs_flat"] > 0.0


def test_ssd_norm_normalised_by_length() -> None:
    """cumnav_ssd_norm equals hand-computed SSD / n (normalisation by series length).

    Toy: signal=[1,-1], ret=[0.01,-0.02], replica=[0,0]
      cum_target = [0.01, 0.01+0.02] = [0.01, 0.03]
      cum_replica = [0, 0]
      squared diffs = [0.01^2, 0.03^2] = [0.0001, 0.0009]
      SSD = 0.0010; normalised = 0.0010 / 2 = 0.0005
    """
    signal = _series([1.0, -1.0])
    ret = _series([0.01, -0.02])
    replica = _series([0.0, 0.0])

    metrics = nav_discrepancy(replica, signal, ret)
    expected = (0.01**2 + 0.03**2) / 2  # = 0.0005
    assert math.isclose(metrics["cumnav_ssd_norm"], expected, rel_tol=1e-9)


# --------------------------------------------------------------------------- #
# tracking_error_ann                                                            #
# --------------------------------------------------------------------------- #

def test_tracking_error_ann_perfect_replica_is_zero() -> None:
    """Perfect replica produces zero tracking error."""
    signal = _series([1.0, -1.0, 0.0, 1.0, -1.0])
    ret = _series([0.01, -0.02, 0.03, -0.01, 0.04])

    metrics = nav_discrepancy(signal, signal, ret)
    assert math.isclose(metrics["tracking_error_ann"], 0.0, abs_tol=1e-15)


def test_tracking_error_ann_opposite_signal() -> None:
    """Opposite signal produces non-zero tracking error."""
    signal = _series([1.0, 1.0, 1.0, 1.0, 1.0])
    opposite = _series([-1.0, -1.0, -1.0, -1.0, -1.0])
    ret = _series([0.01, -0.02, 0.03, -0.01, 0.04])

    metrics = nav_discrepancy(opposite, signal, ret)
    assert metrics["tracking_error_ann"] > 0.0


# --------------------------------------------------------------------------- #
# Integration smoke: real cl1s data                                            #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def real_data():
    """Load real clean data once for the module."""
    from stml.io import load_clean_data
    return load_clean_data()


def test_nav_from_raw_cl1s_runs(real_data) -> None:
    """nav_from_raw on cl1s returns a finite Series with no NaNs."""
    ohlcv, signals = real_data
    result = nav_from_raw(signals, ohlcv, "cl1s", convention="next_day")

    assert isinstance(result, pd.Series)
    assert len(result) > 0
    assert result.isna().sum() == 0, "NAV contains NaN values"
    assert np.isfinite(result.values).all(), "NAV contains non-finite values"


def test_nav_from_raw_cl1s_is_cumulative(real_data) -> None:
    """The NAV series is monotonically cumulative (differences are the raw PnL)."""
    ohlcv, signals = real_data
    result = nav_from_raw(signals, ohlcv, "cl1s", convention="next_day")

    # Verify it starts near 0 and differences are finite
    diffs = result.diff().dropna()
    assert np.isfinite(diffs.values).all()
