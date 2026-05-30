"""Known-value + invariant tests for OHLC volatility estimators (Stage 1, RED-first).

Expected values are computed independently from the closed forms (not from the code):
constants ln2 = 0.69314718..., 4·ln2 = 2.77258872..., 2·ln2−1 = 0.38629436...

Bar A (O=C=100, H=100·e^0.02, L=100·e^-0.02): ln(H/L)=0.04, ln(C/O)=0.
  Parkinson var = 0.04²/(4·ln2) = 5.7707807e-4 → vol = 0.024022449
  Garman–Klass var = 0.5·0.04² = 8.0e-4           → vol = 0.028284271
  Rogers–Satchell var = 0.02·0.02 + 0.02·0.02 = 8.0e-4 → vol = 0.028284271

Bar B (O=100, C=100·e^0.01, H=100·e^0.02, L=100·e^-0.01): ln(H/L)=0.03, ln(C/O)=0.01.
  Garman–Klass var = 0.5·0.03² − (2·ln2−1)·0.01² = 4.113706e-4 → vol = 0.020282273
  Parkinson var = 0.03²/(4·ln2) = 3.246051e-4              → vol = 0.018016801
  Rogers–Satchell var = 0.01·0.02 + 0.02·0.01 = 4.0e-4    → vol = 0.020000000
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alken_metamodel.volatility import garman_klass, parkinson, rogers_satchell

REL = 1e-9


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],  # row0 flat, row1=A, row2=B
            "high": [100.0, 100.0 * np.exp(0.02), 100.0 * np.exp(0.02)],
            "low": [100.0, 100.0 * np.exp(-0.02), 100.0 * np.exp(-0.01)],
            "close": [100.0, 100.0, 100.0 * np.exp(0.01)],
        },
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
    )


_LN2 = float(np.log(2.0))
_FOUR_LN2 = 4.0 * _LN2
_GK_C = 2.0 * _LN2 - 1.0

# Analytic per-bar volatilities (exact for the constructed bars, derived from the
# published closed forms with the *known* log-ratios for bars A and B — independent
# of how the module turns prices into ratios):
#   bar A: ln(H/L)=0.04, ln(C/O)=0; RS legs ±0.02
#   bar B: ln(H/L)=0.03, ln(C/O)=0.01; RS legs 0.01·0.02 + 0.02·0.01 = 4e-4
_EXPECTED = {
    "parkinson": (np.sqrt(0.04**2 / _FOUR_LN2), np.sqrt(0.03**2 / _FOUR_LN2)),
    "garman_klass": (np.sqrt(0.5 * 0.04**2), np.sqrt(0.5 * 0.03**2 - _GK_C * 0.01**2)),
    "rogers_satchell": (np.sqrt(8.0e-4), np.sqrt(4.0e-4)),
}


@pytest.mark.parametrize("estimator", [parkinson, garman_klass, rogers_satchell])
def test_per_bar_matches_closed_form(estimator):
    expected_a, expected_b = _EXPECTED[estimator.__name__]
    vol = estimator(_bars(), window=None)
    assert vol.iloc[1] == pytest.approx(expected_a, rel=REL)
    assert vol.iloc[2] == pytest.approx(expected_b, rel=REL)
    # hard decimal anchors so the test isn't purely self-referential
    if estimator is garman_klass:
        assert vol.iloc[1] == pytest.approx(0.0282842712474619, rel=1e-13)
    if estimator is rogers_satchell:
        assert vol.iloc[2] == pytest.approx(0.02, rel=1e-13)


@pytest.mark.parametrize("estimator", [parkinson, garman_klass, rogers_satchell])
def test_flat_bar_is_zero(estimator):
    vol = estimator(_bars(), window=None)
    assert vol.iloc[0] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("estimator", [parkinson, garman_klass, rogers_satchell])
def test_non_negative(estimator):
    # wider random-ish but valid OHLC bars
    rng = np.random.default_rng(0)
    n = 200
    o = 100 * np.exp(rng.normal(0, 0.01, n))
    c = o * np.exp(rng.normal(0, 0.01, n))
    hi = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.01, n)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c})
    assert (estimator(df, window=None).to_numpy() >= 0).all()


@pytest.mark.parametrize("estimator", [parkinson, garman_klass, rogers_satchell])
def test_monotonic_in_range(estimator):
    # same C=O, strictly wider high-low range -> strictly larger vol
    narrow = pd.DataFrame(
        {
            "open": [100.0],
            "high": [100 * np.exp(0.01)],
            "low": [100 * np.exp(-0.01)],
            "close": [100.0],
        }
    )
    wide = pd.DataFrame(
        {
            "open": [100.0],
            "high": [100 * np.exp(0.03)],
            "low": [100 * np.exp(-0.03)],
            "close": [100.0],
        }
    )
    assert estimator(wide, window=None).iloc[0] > estimator(narrow, window=None).iloc[0]


def test_annualize_scales_by_sqrt_trading_days():
    daily = garman_klass(_bars(), window=None, annualize=False)
    ann = garman_klass(_bars(), window=None, annualize=True, trading_days=252)
    assert ann.iloc[1] == pytest.approx(daily.iloc[1] * np.sqrt(252), rel=REL)


def test_rolling_window_smooths_variance():
    # rolling(window=2) vol of bars A,B = sqrt(mean of the two per-bar variances)
    df = _bars()
    gk_win = garman_klass(df, window=2)
    var_A, var_B = 8.0e-4, 4.113706e-4
    assert gk_win.iloc[2] == pytest.approx(np.sqrt((var_A + var_B) / 2.0), rel=1e-6)
    assert np.isnan(gk_win.iloc[0])  # min_periods=window
