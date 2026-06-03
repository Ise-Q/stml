"""Causality + correctness tests for the barrier-σ̂ estimators (EX.5, RED-first).

The load-bearing invariant is the same RIGHT-EDGE TRUNCATION-INVARIANCE the feature stack
enforces (``test_features.py``): a causal σ̂_t must be identical whether computed on
``close[:t+1]`` or on the full series. A trailing rolling/EWMA std of returns satisfies this;
a forward-looking or full-sample estimator does not. This is the property that lets the swept
barrier widths claim "causal vol only" (spec protocol step 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alken_metamodel.barrier_vol import barrier_sigma, ewma_vol, rolling_std_vol


def _synthetic_ohlcv(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Valid long-format OHLCV for one instrument (mirrors test_features helper)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    log_close = np.cumsum(rng.normal(0.0, 0.012, n)) + np.log(100.0)
    close = np.exp(log_close)
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    intraday = np.abs(rng.normal(0, 0.008, n))
    high = np.maximum(open_, close) * np.exp(intraday)
    low = np.minimum(open_, close) * np.exp(-intraday)
    return pd.DataFrame(
        {
            "date": dates,
            "instrument": "tst1s",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(1_000, 50_000, n).astype(float),
        }
    )


def _close(ohlcv: pd.DataFrame) -> pd.Series:
    s = ohlcv.set_index("date")["close"].sort_index()
    s.index = pd.DatetimeIndex(s.index)
    return s.astype(float)


# --- the crux: right-edge truncation invariance (causality) -----------------


@pytest.mark.parametrize(
    "estimator,param", [("ewma", 20), ("ewma", 50), ("rolling", 20), ("gk", 20)]
)
def test_barrier_sigma_truncation_invariance(estimator, param):
    """σ̂_t identical on data[:t+1] vs full series — no forward leakage."""
    ohlcv = _synthetic_ohlcv(n=300, seed=7)
    full = barrier_sigma(ohlcv, estimator, param)
    t_iloc = 250  # well past warm-up
    t = full.index[t_iloc]

    trunc = barrier_sigma(ohlcv.iloc[: t_iloc + 1].copy(), estimator, param)
    assert t in trunc.index and t in full.index
    assert trunc.loc[t] == pytest.approx(full.loc[t], rel=1e-9, nan_ok=True)


def test_ewma_vol_truncation_invariance():
    close = _close(_synthetic_ohlcv(n=200, seed=3))
    full = ewma_vol(close, span=30)
    t = full.index[180]
    trunc = ewma_vol(close.iloc[:181], span=30)
    assert trunc.loc[t] == pytest.approx(full.loc[t], rel=1e-9)


def test_vols_are_nonnegative_and_finite_after_warmup():
    close = _close(_synthetic_ohlcv(n=200, seed=1))
    for v in (ewma_vol(close, span=20), rolling_std_vol(close, window=20)):
        tail = v.dropna()
        assert len(tail) > 100
        assert (tail >= 0).all()
        assert np.isfinite(tail.to_numpy()).all()


def test_rolling_vol_matches_pandas_std_of_log_returns():
    """rolling_std_vol == rolling std of log-returns (daily, unannualised)."""
    close = _close(_synthetic_ohlcv(n=120, seed=2))
    r = np.log(close / close.shift(1))
    expected = r.rolling(20, min_periods=20).std()
    got = rolling_std_vol(close, window=20)
    pd.testing.assert_series_equal(got.dropna(), expected.dropna(), check_names=False)


def test_gk_is_genuinely_distinct_from_rolling_close_to_close():
    """Guard: the GK arm must be a real OHLC-range estimator, NOT close-to-close realized vol.

    (The shipped ``daily_barrier_sigma`` = ``f2_vol_20``/√252 IS realized-vol-20, so a GK arm that
    silently equalled rolling-20 would make the vol sweep meaningless — the exact bug this catches.)
    """
    ohlcv = _synthetic_ohlcv(n=200, seed=4)
    gk = barrier_sigma(ohlcv, "gk", 20).dropna()
    rolling = barrier_sigma(ohlcv, "rolling", 20).dropna()
    common = gk.index.intersection(rolling.index)
    assert len(common) > 50
    # materially different series, not a relabel of the same numbers
    assert not np.allclose(gk.loc[common].to_numpy(), rolling.loc[common].to_numpy())


def test_barrier_sigma_rejects_unknown_estimator():
    with pytest.raises(ValueError, match="estimator"):
        barrier_sigma(_synthetic_ohlcv(n=30), "garch", 20)


def test_lower_span_reacts_faster_than_higher_span():
    """A vol shock moves the short-span EWMA more than the long-span one (sanity)."""
    close = _close(_synthetic_ohlcv(n=200, seed=5))
    # inject a shock at the tail
    close.iloc[-1] = close.iloc[-2] * 1.10
    fast = ewma_vol(close, span=10).iloc[-1]
    slow = ewma_vol(close, span=100).iloc[-1]
    assert fast > slow
