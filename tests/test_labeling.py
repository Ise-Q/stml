"""Unit tests for src/stml/labeling.py.

Covers:
    - get_daily_vol: causality (no peeking ahead) + known-input check
    - apply_triple_barrier_one: hit PT / hit SL / vertical / both barriers /
      side handling / NaN sigma / event at end of data
    - get_meta_labels: shape + label = sign(ret) invariant + 0/1 only
    - get_uniqueness_weights: weights in (0, 1] before normalization,
      mean=1 after; effective sample size < N when events overlap
    - get_fixed_horizon_labels: matches naive computation on a small input
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.labeling import (
    apply_triple_barrier_one,
    extract_signal_events,
    get_daily_vol,
    get_fixed_horizon_labels,
    get_meta_labels,
    get_uniqueness_weights,
    label_summary,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture
def trading_dates() -> pd.DatetimeIndex:
    """50 contiguous business days starting 2020-01-02."""
    return pd.bdate_range("2020-01-02", periods=50)


@pytest.fixture
def flat_close(trading_dates) -> pd.Series:
    """Flat close prices = 100 everywhere -> zero returns -> zero vol."""
    return pd.Series(100.0, index=trading_dates, name="close")


@pytest.fixture
def trend_close(trading_dates) -> pd.Series:
    """Smooth upward trend, 1% per day."""
    return pd.Series(100.0 * np.exp(0.01 * np.arange(len(trading_dates))),
                     index=trading_dates, name="close")


# --------------------------------------------------------------------------- #
# 1. get_daily_vol                                                             #
# --------------------------------------------------------------------------- #
class TestGetDailyVol:

    def test_causality_no_peeking(self):
        """Vol at date t must not change if we append future data."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-02", periods=200)
        ret = rng.normal(0, 0.01, size=len(dates))
        close = pd.Series(100 * np.exp(np.cumsum(ret)), index=dates)

        vol_full = get_daily_vol(close, span=50, min_periods=20)
        vol_trunc = get_daily_vol(close.iloc[:100], span=50, min_periods=20)

        # Values up to index 99 should agree exactly.
        common = vol_full.iloc[:100].dropna().index.intersection(vol_trunc.dropna().index)
        assert len(common) > 0
        np.testing.assert_array_almost_equal(
            vol_full.loc[common].values, vol_trunc.loc[common].values, decimal=12
        )

    def test_flat_prices_zero_vol(self, flat_close):
        vol = get_daily_vol(flat_close, span=10, min_periods=5)
        # After min_periods, vol of zero returns is zero.
        assert vol.dropna().abs().max() < 1e-12

    def test_min_periods_nan_prefix(self, trend_close):
        vol = get_daily_vol(trend_close, span=10, min_periods=20)
        # First 19 values must be NaN (need at least min_periods returns).
        assert vol.iloc[:19].isna().all()
        assert vol.iloc[20:].notna().all()

    def test_requires_sorted_index(self, flat_close):
        shuffled = flat_close.sample(frac=1, random_state=0)
        with pytest.raises(ValueError):
            get_daily_vol(shuffled)


# --------------------------------------------------------------------------- #
# 2. apply_triple_barrier_one                                                  #
# --------------------------------------------------------------------------- #
class TestApplyTripleBarrierOne:

    def _make_close(self, prices: list[float], start_date: str = "2020-01-02") -> pd.Series:
        idx = pd.bdate_range(start_date, periods=len(prices))
        return pd.Series(prices, index=idx, name="close")

    def test_long_hits_pt(self):
        """A long bet on a steadily rising path: PT must touch."""
        close = self._make_close([100, 100.5, 101, 102, 103, 104, 105, 106])
        t_event = close.index[0]
        # vol very small ⇒ barrier ≈ 0.001 ⇒ even tiny moves trigger.
        t1, hit, ret = apply_triple_barrier_one(
            close, t_event, side=+1, sigma_at_t=0.001, h=5, pt_mult=1.0, sl_mult=1.0
        )
        assert hit == "pt"
        assert ret > 0
        assert t1 > t_event

    def test_long_hits_sl(self):
        """A long bet on a steadily falling path: SL must touch."""
        close = self._make_close([100, 99.5, 99, 98, 97, 96, 95])
        t_event = close.index[0]
        t1, hit, ret = apply_triple_barrier_one(
            close, t_event, side=+1, sigma_at_t=0.001, h=5, pt_mult=1.0, sl_mult=1.0
        )
        assert hit == "sl"
        assert ret < 0

    def test_short_hits_pt_on_falling(self):
        """A SHORT bet on falling path: PT touches (signed-return positive in short direction)."""
        close = self._make_close([100, 99, 98, 97, 96])
        t_event = close.index[0]
        t1, hit, ret = apply_triple_barrier_one(
            close, t_event, side=-1, sigma_at_t=0.001, h=4, pt_mult=1.0, sl_mult=1.0
        )
        assert hit == "pt"
        assert ret > 0  # short profitable

    def test_vertical_when_flat(self):
        """Flat path within barriers: vertical hit, ret = 0."""
        close = self._make_close([100, 100, 100, 100, 100, 100])
        t_event = close.index[0]
        t1, hit, ret = apply_triple_barrier_one(
            close, t_event, side=+1, sigma_at_t=1.0, h=5, pt_mult=1.0, sl_mult=1.0
        )
        assert hit == "vertical"
        assert abs(ret) < 1e-10
        assert t1 == close.index[5]

    def test_earliest_touch_wins(self):
        """If PT and SL would both eventually touch, the EARLIEST one wins."""
        # Path: down hard (SL on day 1), then up hard (PT on day 3)
        close = self._make_close([100, 90, 95, 110, 110])
        t_event = close.index[0]
        # sigma small enough that 90 hits SL and 110 hits PT, with SL hit first.
        t1, hit, ret = apply_triple_barrier_one(
            close, t_event, side=+1, sigma_at_t=0.01, h=4, pt_mult=1.0, sl_mult=1.0
        )
        assert hit == "sl"
        assert t1 == close.index[1]

    def test_event_at_end_of_data(self):
        close = self._make_close([100, 101])
        t_event = close.index[-1]
        # No future data after t_event.
        t1, hit, ret = apply_triple_barrier_one(
            close, t_event, side=+1, sigma_at_t=0.01, h=5, pt_mult=1.0, sl_mult=1.0
        )
        assert np.isnan(ret)
        assert hit == "vertical"

    def test_nan_sigma_falls_back_to_vertical(self):
        close = self._make_close([100, 101, 102, 103, 104])
        t_event = close.index[0]
        t1, hit, ret = apply_triple_barrier_one(
            close, t_event, side=+1, sigma_at_t=np.nan, h=4, pt_mult=1.0, sl_mult=1.0
        )
        assert hit == "vertical"
        assert ret > 0  # rising path, long bet ⇒ positive

    def test_invalid_side_raises(self, flat_close):
        with pytest.raises(ValueError):
            apply_triple_barrier_one(flat_close, flat_close.index[0], side=2,
                                     sigma_at_t=0.01, h=5)


# --------------------------------------------------------------------------- #
# 3. get_meta_labels — end to end on synthetic panel                          #
# --------------------------------------------------------------------------- #
class TestGetMetaLabels:

    def _build_synthetic_panel(self, n_days: int = 60, seed: int = 0):
        """Build a tiny synthetic OHLCV + signals panel for one instrument."""
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2020-01-02", periods=n_days)
        # Two-instrument panel; instrument "up1s" trends up, "dn1s" trends down.
        rows = []
        for inst, drift in [("up1s", 0.01), ("dn1s", -0.01)]:
            ret = rng.normal(drift, 0.005, size=n_days)
            close = 100.0 * np.exp(np.cumsum(ret))
            for d, c in zip(dates, close):
                rows.append({"date": d, "instrument": inst,
                             "open": c, "high": c * 1.001, "low": c * 0.999,
                             "close": c, "volume": 1000, "open_interest": 1000})
        ohlcv = pd.DataFrame(rows)
        # Signal: long every day on both instruments.
        signals = pd.DataFrame(
            {"date": dates, "up1s": [1] * n_days, "dn1s": [1] * n_days}
        )
        return ohlcv, signals

    def test_shape_and_columns(self):
        ohlcv, signals = self._build_synthetic_panel()
        labels = get_meta_labels(ohlcv, signals, h=5, vol_span=20, vol_min_periods=10)
        expected = {"t", "instrument", "side", "sigma_at_t", "t1",
                    "barrier_hit", "ret", "label", "h"}
        assert expected.issubset(labels.columns)
        assert labels["label"].isin([0, 1]).all()
        assert (labels["side"].isin([-1, 1])).all()

    def test_label_equals_sign_of_ret(self):
        ohlcv, signals = self._build_synthetic_panel()
        labels = get_meta_labels(ohlcv, signals, h=5, vol_span=20, vol_min_periods=10)
        # By construction label = 1 iff ret > 0.
        assert ((labels["ret"] > 0) == (labels["label"] == 1)).all()

    def test_trending_up_high_long_label_share(self):
        """On the up-trending instrument, long bets should mostly succeed."""
        ohlcv, signals = self._build_synthetic_panel(n_days=80, seed=1)
        labels = get_meta_labels(ohlcv, signals, h=5, vol_span=20, vol_min_periods=10)
        up_share = labels.loc[labels["instrument"] == "up1s", "label"].mean()
        dn_share = labels.loc[labels["instrument"] == "dn1s", "label"].mean()
        # Up-trending should win much more often than down-trending (both long).
        assert up_share > 0.55
        assert dn_share < 0.45
        assert up_share - dn_share > 0.10

    def test_t1_after_t(self):
        ohlcv, signals = self._build_synthetic_panel()
        labels = get_meta_labels(ohlcv, signals, h=5, vol_span=20, vol_min_periods=10)
        assert (labels["t1"] >= labels["t"]).all()

    def test_extract_signal_events_drops_zeros(self):
        # 3 dates, 2 instruments, with one zero signal.
        dates = pd.bdate_range("2020-01-02", periods=3)
        signals = pd.DataFrame({"date": dates, "a": [1, 0, -1], "b": [0, 1, 1]})
        ev = extract_signal_events(signals)
        assert len(ev) == 4  # 6 cells - 2 zeros
        assert ev["side"].isin([-1, 1]).all()


# --------------------------------------------------------------------------- #
# 4. get_uniqueness_weights                                                    #
# --------------------------------------------------------------------------- #
class TestUniquenessWeights:

    def test_disjoint_events_get_equal_weight(self):
        """Three non-overlapping events on the same instrument ⇒ weights all = 1."""
        events = pd.DataFrame({
            "t": pd.to_datetime(["2020-01-02", "2020-02-03", "2020-03-04"]),
            "t1": pd.to_datetime(["2020-01-09", "2020-02-10", "2020-03-11"]),
            "instrument": ["cl1s"] * 3,
        })
        w = get_uniqueness_weights(events)
        np.testing.assert_array_almost_equal(w.values, [1.0, 1.0, 1.0])

    def test_fully_overlapping_events_equal_weight_lower_than_one_in_raw(self):
        """Two events on the same instrument with identical spans: both share
        the path 50/50 ⇒ raw uniqueness = 0.5 ⇒ after mean-normalisation = 1."""
        events = pd.DataFrame({
            "t": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            "t1": pd.to_datetime(["2020-01-09", "2020-01-09"]),
            "instrument": ["cl1s", "cl1s"],
        })
        w = get_uniqueness_weights(events, normalize=True)
        np.testing.assert_array_almost_equal(w.values, [1.0, 1.0])
        w_raw = get_uniqueness_weights(events, normalize=False)
        # raw uniqueness = 1/2 since c=2 across the whole span
        np.testing.assert_array_almost_equal(w_raw.values, [0.5, 0.5])

    def test_concurrency_is_per_instrument(self):
        """Events on different instruments should not affect each other's
        weights (concurrency is per-instrument)."""
        events = pd.DataFrame({
            "t": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            "t1": pd.to_datetime(["2020-01-09", "2020-01-09"]),
            "instrument": ["cl1s", "gc1s"],
        })
        w = get_uniqueness_weights(events, normalize=False)
        # Each instrument has 1 event ⇒ uniqueness = 1.
        np.testing.assert_array_almost_equal(w.values, [1.0, 1.0])

    def test_mean_after_normalisation_is_one(self):
        # Heterogeneous spans
        events = pd.DataFrame({
            "t": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
            "t1": pd.to_datetime(["2020-01-10", "2020-01-09", "2020-01-13"]),
            "instrument": ["cl1s"] * 3,
        })
        w = get_uniqueness_weights(events, normalize=True)
        assert abs(w.mean() - 1.0) < 1e-12


# --------------------------------------------------------------------------- #
# 5. get_fixed_horizon_labels                                                  #
# --------------------------------------------------------------------------- #
class TestFixedHorizonLabels:

    def test_matches_naive_computation(self):
        """For one instrument, label_i = 1 iff side * log(close_{t+h} / close_t) > 0."""
        dates = pd.bdate_range("2020-01-02", periods=20)
        close = 100 * np.exp(np.linspace(0, 0.1, 20))  # up-trend
        ohlcv = pd.DataFrame({
            "date": dates, "instrument": "up1s",
            "open": close, "high": close, "low": close, "close": close,
            "volume": 0, "open_interest": 0,
        })
        signals = pd.DataFrame({"date": dates, "up1s": [1] * 20})
        out = get_fixed_horizon_labels(ohlcv, signals, h=5, threshold=0.0)
        # Up-trend, long bet, h=5 ⇒ all labels should be 1.
        assert (out["label"] == 1).all()
        # ret_i should equal log(close_{t+5} / close_t)
        for _, row in out.iterrows():
            idx = list(dates).index(row["t"])
            expected = float(np.log(close[idx + 5] / close[idx]))
            assert abs(row["ret"] - expected) < 1e-10
