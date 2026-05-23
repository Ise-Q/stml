"""Unit tests for src/stml/regimes.py — the HMM/GMM regime features.

The CRITICAL invariant being tested is CAUSALITY: filtered HMM posteriors at
time t must depend only on observations up to time t, never on future data.
This is the single subtlest leakage trap in the project (hmmlearn's
``predict_proba`` uses smoothed posteriors that DO use future data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.regimes import (
    causal_filtered_probs,
    compute_regime_features,
    fit_instrument_hmm,
    gmm_features_for_instrument,
    hmm_features_for_instrument,
    _instrument_obs,
)


# --------------------------------------------------------------------------- #
# Synthetic price panel — bicephalous vol regime                              #
# --------------------------------------------------------------------------- #
@pytest.fixture
def synth_ohlcv() -> pd.DataFrame:
    """Two-instrument synthetic OHLCV. Instrument 'a1s' has two clear vol
    regimes (low for the first 500 days, high for the next 500), which the
    HMM should pick up."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-02", periods=1500)
    rows = []
    for inst in ("a1s", "b1s"):
        # 1500-day series with a vol regime change at day 500 and day 1100.
        sigma = np.where(
            (np.arange(1500) < 500) | (np.arange(1500) >= 1100),
            0.005,  # low vol
            0.025,  # high vol
        )
        ret = rng.normal(0, sigma)
        close = 100.0 * np.exp(np.cumsum(ret))
        for d, c in zip(dates, close):
            rows.append({
                "date": d, "instrument": inst,
                "open": c * 0.999, "high": c * 1.002, "low": c * 0.998,
                "close": c, "volume": 1000 + rng.integers(0, 500),
                "open_interest": 5000,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 1. causal_filtered_probs — the headline causality invariant                  #
# --------------------------------------------------------------------------- #
class TestCausalFilteredProbs:

    def _fit_and_obs(self, synth_ohlcv):
        obs = _instrument_obs(synth_ohlcv, "a1s")
        hmm = fit_instrument_hmm(obs.values[:800], n_states=2, random_state=0)
        return hmm, obs

    def test_filtered_no_peeking(self, synth_ohlcv):
        """The core invariant: filtered[t] computed on X[:t+k] must equal
        filtered[t] computed on X[:t+1] for any k > 0."""
        hmm, obs = self._fit_and_obs(synth_ohlcv)
        X = obs.values
        partial = causal_filtered_probs(hmm, X[:1000])
        full = causal_filtered_probs(hmm, X[:1400])[:1000]
        np.testing.assert_array_almost_equal(partial, full, decimal=12)

    def test_rows_sum_to_one(self, synth_ohlcv):
        hmm, obs = self._fit_and_obs(synth_ohlcv)
        f = causal_filtered_probs(hmm, obs.values[:500])
        sums = f.sum(axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones_like(sums), decimal=10)

    def test_filtered_differs_from_smoothed(self, synth_ohlcv):
        """We must verify we are NOT accidentally using smoothed posteriors.
        If filtered == smoothed everywhere, our causality protection is moot."""
        hmm, obs = self._fit_and_obs(synth_ohlcv)
        X = obs.values[:1000]
        filt = causal_filtered_probs(hmm, X)
        smooth = hmm.predict_proba(X)
        # On at least some rows filtered and smoothed differ noticeably.
        assert np.max(np.abs(filt - smooth)) > 1e-3


# --------------------------------------------------------------------------- #
# 2. HMM features per instrument                                              #
# --------------------------------------------------------------------------- #
class TestHmmFeatures:

    def test_shape_and_columns(self, synth_ohlcv):
        boundary = pd.Timestamp("2018-01-01")
        out = hmm_features_for_instrument(synth_ohlcv, "a1s", boundary, n_states=3)
        for c in ["hmm_state_lo", "hmm_state_mid", "hmm_state_hi", "hmm_state_argmax"]:
            assert c in out.columns
        # Row probabilities sum to 1.
        sums = out[["hmm_state_lo", "hmm_state_mid", "hmm_state_hi"]].sum(axis=1)
        np.testing.assert_array_almost_equal(sums.values, np.ones_like(sums.values),
                                              decimal=10)

    def test_states_are_vol_ordered(self, synth_ohlcv):
        """After reordering, hmm_state_hi should activate during the
        high-vol regime far more than during the low-vol regimes."""
        boundary = pd.Timestamp("2018-01-01")
        out = hmm_features_for_instrument(synth_ohlcv, "a1s", boundary, n_states=2)
        # 2-state version: use the second column ("mid") as the higher state.
        # n_states=2 ⇒ columns are state_0..state_K-1, no lo/mid/hi naming.
        # Fall back to argmax check.
        dates = out.index
        high_mask = (dates >= pd.Timestamp("2016-12-30")) & (dates < pd.Timestamp("2019-04-10"))
        low_mask = ~high_mask
        # During the high-vol regime, the "high" state should dominate.
        # With n_states=2, the higher state is index 1 (by vol ordering).
        # Use argmax to check most-likely state distribution.
        in_high = out.loc[high_mask, "hmm_state_argmax"].value_counts(normalize=True)
        in_low = out.loc[low_mask, "hmm_state_argmax"].value_counts(normalize=True)
        # In the high-vol regime, state 1 (higher) should be more frequent than in low-vol.
        share_high_state_in_high = in_high.get(1, 0.0)
        share_high_state_in_low = in_low.get(1, 0.0)
        assert share_high_state_in_high > share_high_state_in_low + 0.10

    def test_training_only_uses_pre_boundary_data(self, synth_ohlcv):
        """Same instrument, different boundaries should yield different
        fitted HMMs (because they see different training sets)."""
        out_early = hmm_features_for_instrument(synth_ohlcv, "a1s",
                                                 pd.Timestamp("2017-01-01"), n_states=3)
        out_late = hmm_features_for_instrument(synth_ohlcv, "a1s",
                                                pd.Timestamp("2019-01-01"), n_states=3)
        # The two should diverge meaningfully somewhere in the OVERLAP region.
        common = out_early.index.intersection(out_late.index)
        diff = (out_early.loc[common, ["hmm_state_lo", "hmm_state_mid", "hmm_state_hi"]].values
                - out_late.loc[common, ["hmm_state_lo", "hmm_state_mid", "hmm_state_hi"]].values)
        assert np.max(np.abs(diff)) > 0.01

    def test_too_little_data_returns_empty_or_indexed_only(self, synth_ohlcv):
        # Boundary so early there are <200 training observations
        out = hmm_features_for_instrument(synth_ohlcv, "a1s",
                                           pd.Timestamp("2015-03-01"), n_states=3)
        # Should return empty or index-only DataFrame.
        assert out.empty or set(out.columns).isdisjoint(
            {"hmm_state_lo", "hmm_state_mid", "hmm_state_hi"}
        )


# --------------------------------------------------------------------------- #
# 3. GMM features                                                              #
# --------------------------------------------------------------------------- #
class TestGmmFeatures:

    def test_shape_and_columns(self, synth_ohlcv):
        out = gmm_features_for_instrument(synth_ohlcv, "a1s",
                                           pd.Timestamp("2018-01-01"))
        cols = ["gmm_cluster_lo", "gmm_cluster_mid", "gmm_cluster_hi",
                "gmm_cluster_argmax"]
        for c in cols:
            assert c in out.columns

    def test_row_probs_sum_to_one(self, synth_ohlcv):
        out = gmm_features_for_instrument(synth_ohlcv, "a1s",
                                           pd.Timestamp("2018-01-01"))
        sums = out[["gmm_cluster_lo", "gmm_cluster_mid", "gmm_cluster_hi"]].sum(axis=1)
        np.testing.assert_array_almost_equal(sums.values, np.ones_like(sums.values),
                                              decimal=8)


# --------------------------------------------------------------------------- #
# 4. compute_regime_features — full panel                                     #
# --------------------------------------------------------------------------- #
class TestComputeRegimeFeatures:

    def test_returns_one_row_per_event(self, synth_ohlcv):
        # Build a tiny events frame.
        events = pd.DataFrame({
            "t": pd.to_datetime(["2018-01-03", "2018-02-03", "2019-01-03"] * 2),
            "instrument": ["a1s"] * 3 + ["b1s"] * 3,
            "side": [1, -1, 1, 1, -1, 1],
        })
        out = compute_regime_features(synth_ohlcv, events,
                                       boundary=pd.Timestamp("2018-01-01"))
        assert len(out) == len(events)
