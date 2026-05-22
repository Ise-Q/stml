"""
test_align.py
=============
Tests for :mod:`stml.replication.align`.

Coverage
--------
- Structural invariants on real data (cl1s, next_day convention).
- No-ffill invariant: a synthetic gapped series leaves the gap excluded.
- same_day convention: log(close_t / close_{t-1}) is attached correctly.
- n_dropped / retained_fraction arithmetic identity.
- frame dates are a subset of signal dates and all ret values are non-null.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from stml.replication.align import AlignResult, align_instrument, align_panel


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _make_signals(dates: list, instrument: str, value: int = 1) -> pd.DataFrame:
    """Minimal wide signals frame for one instrument."""
    return pd.DataFrame({"date": pd.to_datetime(dates), instrument: value})


def _make_ohlcv(dates: list, closes: list, instrument: str) -> pd.DataFrame:
    """Minimal long OHLCV frame with only the columns native_returns needs."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "instrument": instrument,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1.0,
            "open_interest": 0.0,
        }
    )


# --------------------------------------------------------------------------- #
# Real-data tests (cl1s, next_day)                                             #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def real_data():
    """Load real clean data once for the module."""
    from stml.io import load_clean_data
    return load_clean_data()


@pytest.fixture(scope="module")
def cl1s_result(real_data):
    ohlcv, signals = real_data
    return align_instrument(signals, ohlcv, "cl1s", convention="next_day")


def test_cl1s_n_signal_days(cl1s_result: AlignResult) -> None:
    """Signal series has 645 rows per CONTRACT.md."""
    assert cl1s_result.n_signal_days == 645


def test_cl1s_n_dropped_plausible(cl1s_result: AlignResult) -> None:
    """cl1s drops ≈18 days (calendar misses) under next_day."""
    assert 0 < cl1s_result.n_dropped < 40


def test_frame_dates_subset_of_signal_dates(cl1s_result: AlignResult, real_data) -> None:
    """Every date in frame must appear in the original signal series."""
    _, signals = real_data
    signal_dates = set(signals["date"])
    frame_dates = set(cl1s_result.frame["date"])
    assert frame_dates.issubset(signal_dates)


def test_frame_ret_no_nulls(cl1s_result: AlignResult) -> None:
    """All ret values in the aligned frame must be non-null."""
    assert cl1s_result.frame["ret"].notna().all()


def test_arithmetic_identity(cl1s_result: AlignResult) -> None:
    """n_dropped == n_signal_days - len(frame)."""
    r = cl1s_result
    assert r.n_dropped == r.n_signal_days - len(r.frame)


def test_retained_fraction_correct(cl1s_result: AlignResult) -> None:
    """retained_fraction == len(frame) / n_signal_days."""
    r = cl1s_result
    expected = len(r.frame) / r.n_signal_days
    assert math.isclose(r.retained_fraction, expected, rel_tol=1e-9)


def test_retained_fraction_in_range(cl1s_result: AlignResult) -> None:
    """retained_fraction must be in (0, 1]."""
    assert 0 < cl1s_result.retained_fraction <= 1.0


def test_frame_columns(cl1s_result: AlignResult) -> None:
    """frame must have exactly [date, signal, ret] columns."""
    assert list(cl1s_result.frame.columns) == ["date", "signal", "ret"]


# --------------------------------------------------------------------------- #
# align_panel smoke test                                                       #
# --------------------------------------------------------------------------- #

def test_align_panel_returns_all_instruments(real_data) -> None:
    ohlcv, signals = real_data
    panel = align_panel(signals, ohlcv, convention="next_day")
    instruments = [c for c in signals.columns if c != "date"]
    assert set(panel.keys()) == set(instruments)
    for inst, r in panel.items():
        assert isinstance(r, AlignResult)
        assert r.n_signal_days == 645
        assert 0 < r.retained_fraction <= 1.0


def test_align_panel_instruments_subset(real_data) -> None:
    ohlcv, signals = real_data
    panel = align_panel(signals, ohlcv, convention="next_day", instruments=["cl1s", "es1s"])
    assert set(panel.keys()) == {"cl1s", "es1s"}


# --------------------------------------------------------------------------- #
# NO-FFILL invariant: synthetic gapped series                                  #
# --------------------------------------------------------------------------- #

def test_no_ffill_gap_excluded() -> None:
    """A signal on a date with no OHLCV row must be EXCLUDED, not forward-filled.

    Setup
    -----
    OHLCV has closes on [d0, d1, d3] -- d2 is deliberately absent (interior gap).
    Signals have a signal on [d0, d1, d2, d3].  Under same_day convention:
      - d1 has a defined return (log(close_d1/close_d0)).
      - d2 has no OHLCV row, so no return can be computed.
      - d3 has a defined return (log(close_d3/close_d1) -- spanning the gap correctly).
    The aligned frame must contain d1 and d3 but NOT d2.
    """
    inst = "xx1s"
    dates_ohlcv = ["2021-01-04", "2021-01-05", "2021-01-07"]  # 2021-01-06 missing
    closes = [100.0, 102.0, 101.0]

    dates_signal = ["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"]

    ohlcv = _make_ohlcv(dates_ohlcv, closes, inst)
    signals = _make_signals(dates_signal, inst, value=1)

    result = align_instrument(signals, ohlcv, inst, convention="same_day")
    frame_dates = set(result.frame["date"].dt.strftime("%Y-%m-%d"))

    # d2 (2021-01-06) must be absent -- it has no OHLCV row, hence no return
    assert "2021-01-06" not in frame_dates

    # d1 and d3 have defined returns and signals -- they must be present
    assert "2021-01-05" in frame_dates
    assert "2021-01-07" in frame_dates

    # All ret values must be non-null
    assert result.frame["ret"].notna().all()


def test_no_ffill_next_day_gap_excluded() -> None:
    """Under next_day, signal on d_gap is excluded when d_{gap+1} has no close."""
    inst = "yy1s"
    # OHLCV: d0, d1, d3 (d2 missing -- so r_{d2} is also missing for d1 under next_day)
    dates_ohlcv = ["2021-01-04", "2021-01-05", "2021-01-07"]
    closes = [100.0, 102.0, 101.0]

    # Signal on all four dates including the gap date
    dates_signal = ["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"]

    ohlcv = _make_ohlcv(dates_ohlcv, closes, inst)
    signals = _make_signals(dates_signal, inst, value=1)

    result = align_instrument(signals, ohlcv, inst, convention="next_day")

    # 2021-01-06 is not in OHLCV so there is no next-day return for 2021-01-05
    # (the return on 2021-01-06 doesn't exist), meaning 2021-01-05 should be dropped.
    # 2021-01-06 itself is not in OHLCV returns at all.
    # 2021-01-07 is the last OHLCV date so its next-day return is NaN (shift -1).
    frame_dates = set(result.frame["date"].dt.strftime("%Y-%m-%d"))

    # The gap date itself has no OHLCV row, so it appears in neither returns series
    assert "2021-01-06" not in frame_dates

    # All ret values must be non-null
    assert result.frame["ret"].notna().all()


# --------------------------------------------------------------------------- #
# same_day convention: known log-return values                                 #
# --------------------------------------------------------------------------- #

def test_same_day_known_returns() -> None:
    """same_day: ret on date t == log(close_t / close_{t-1})."""
    inst = "zz1s"
    dates = ["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"]
    closes = [100.0, 110.0, 99.0, 105.0]

    ohlcv = _make_ohlcv(dates, closes, inst)
    signals = _make_signals(dates, inst, value=1)

    result = align_instrument(signals, ohlcv, inst, convention="same_day")
    frame = result.frame.set_index("date")

    # d0 has no prior close, so no return -- only d1..d3 in frame
    assert pd.Timestamp("2021-01-04") not in frame.index

    # Check known values
    expected = {
        "2021-01-05": math.log(110.0 / 100.0),
        "2021-01-06": math.log(99.0 / 110.0),
        "2021-01-07": math.log(105.0 / 99.0),
    }
    for date_str, exp_ret in expected.items():
        ts = pd.Timestamp(date_str)
        assert ts in frame.index, f"{date_str} missing from frame"
        assert math.isclose(frame.loc[ts, "ret"], exp_ret, rel_tol=1e-9)


def test_next_day_known_returns() -> None:
    """next_day: ret on signal day t == log(close_{t+1} / close_t).

    native_returns computes r_t = log(close_t/close_{t-1}), dropping d0.
    The native return series is therefore indexed on [d1, d2, d3].
    After shift(-1): d1 carries r_{d2}, d2 carries r_{d3}, d3 becomes NaN.
    Inner-join with signals [d0, d1, d2, d3] and drop NaN rows yields [d1, d2].
    """
    inst = "aa1s"
    dates = ["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"]
    closes = [100.0, 110.0, 99.0, 105.0]

    ohlcv = _make_ohlcv(dates, closes, inst)
    signals = _make_signals(dates, inst, value=1)

    result = align_instrument(signals, ohlcv, inst, convention="next_day")
    frame = result.frame.set_index("date")

    # d0 (2021-01-04): not in native return index, so absent from frame
    assert pd.Timestamp("2021-01-04") not in frame.index
    # d3 (2021-01-07): last native return date, shift(-1) gives NaN, so dropped
    assert pd.Timestamp("2021-01-07") not in frame.index

    # d1 -> next-day return = log(close_d2 / close_d1) = log(99/110)
    assert math.isclose(frame.loc[pd.Timestamp("2021-01-05"), "ret"], math.log(99.0 / 110.0), rel_tol=1e-9)
    # d2 -> next-day return = log(close_d3 / close_d2) = log(105/99)
    assert math.isclose(frame.loc[pd.Timestamp("2021-01-06"), "ret"], math.log(105.0 / 99.0), rel_tol=1e-9)


# --------------------------------------------------------------------------- #
# Invalid convention                                                           #
# --------------------------------------------------------------------------- #

def test_invalid_convention_raises() -> None:
    inst = "bb1s"
    ohlcv = _make_ohlcv(["2021-01-04", "2021-01-05"], [100.0, 101.0], inst)
    signals = _make_signals(["2021-01-04", "2021-01-05"], inst)
    with pytest.raises(ValueError, match="convention"):
        align_instrument(signals, ohlcv, inst, convention="bad")


# --------------------------------------------------------------------------- #
# n_dropped / retained_fraction arithmetic on synthetic data                   #
# --------------------------------------------------------------------------- #

def test_arithmetic_identity_synthetic() -> None:
    """Arithmetic identity holds on synthetic data."""
    inst = "cc1s"
    ohlcv = _make_ohlcv(
        ["2021-01-04", "2021-01-05", "2021-01-06"],
        [100.0, 101.0, 102.0],
        inst,
    )
    # Signal on 4 dates; one (2021-01-08) has no OHLCV row -> dropped
    signals = _make_signals(
        ["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-08"], inst
    )
    result = align_instrument(signals, ohlcv, inst, convention="same_day")
    assert result.n_dropped == result.n_signal_days - len(result.frame)
    assert math.isclose(result.retained_fraction, len(result.frame) / result.n_signal_days, rel_tol=1e-9)
