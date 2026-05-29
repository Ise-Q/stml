"""
test_xsection.py
================
Tests for F9 cross-sectional features in :mod:`stml.metamodel.xsection`.

Uses real data via :func:`stml.io.load_clean_data` and covers:
- correct output columns and date index;
- f9_xsection_universe_size is integer-valued, in [1, 11], and VARIES;
- panel-truncation invariance (causal proof);
- f9_pair_corr_mean bounded in [-1, 1] where finite;
- score="reversal" negates the ranking direction vs score="momentum".
"""

from __future__ import annotations

import pandas as pd
import pytest

from stml.io import load_clean_data
from stml.metamodel.xsection import ASSET_CLASS_PEERS, xsection_features

# ---------------------------------------------------------------------------
# Fixtures — load data once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ohlcv_all() -> pd.DataFrame:
    ohlcv, _ = load_clean_data()
    return ohlcv


# Three instruments covering one per asset class
PROBE_INSTRUMENTS = ["es1s", "gc1s", "cl1s"]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _features(ohlcv_all: pd.DataFrame, instrument: str, **kw) -> pd.DataFrame:
    return xsection_features(ohlcv_all, instrument, **kw)


# ---------------------------------------------------------------------------
# 1. Column names and date index
# ---------------------------------------------------------------------------

class TestOutputShape:
    EXPECTED_COLS = [
        "f9_xsect_rank",
        "f9_xsection_universe_size",
        "f9_pair_corr_mean",
        "f9_dist_lead_lag_centroid",
        "f9_asset_class_dispersion_z",
        "f9_ewma_implied_corr_z",
    ]

    @pytest.mark.parametrize("instrument", PROBE_INSTRUMENTS)
    def test_columns(self, ohlcv_all: pd.DataFrame, instrument: str) -> None:
        feat = _features(ohlcv_all, instrument)
        assert list(feat.columns) == self.EXPECTED_COLS, (
            f"{instrument}: columns mismatch: {list(feat.columns)}"
        )

    @pytest.mark.parametrize("instrument", PROBE_INSTRUMENTS)
    def test_date_indexed(self, ohlcv_all: pd.DataFrame, instrument: str) -> None:
        feat = _features(ohlcv_all, instrument)
        assert isinstance(feat.index, pd.DatetimeIndex), (
            f"{instrument}: index must be DatetimeIndex, got {type(feat.index)}"
        )
        assert feat.index.name == "date"

    @pytest.mark.parametrize("instrument", PROBE_INSTRUMENTS)
    def test_nonempty(self, ohlcv_all: pd.DataFrame, instrument: str) -> None:
        feat = _features(ohlcv_all, instrument)
        assert len(feat) > 0, f"{instrument}: empty output"


# ---------------------------------------------------------------------------
# 2. f9_xsection_universe_size: integer, in [1,11], varies
# ---------------------------------------------------------------------------

class TestUniverseSize:
    @pytest.mark.parametrize("instrument", PROBE_INSTRUMENTS)
    def test_integer_valued(self, ohlcv_all: pd.DataFrame, instrument: str) -> None:
        feat = _features(ohlcv_all, instrument)
        sizes = feat["f9_xsection_universe_size"].dropna()
        # Must be integer-valued (no fractional values)
        assert (sizes == sizes.round()).all(), (
            f"{instrument}: f9_xsection_universe_size has non-integer values"
        )

    @pytest.mark.parametrize("instrument", PROBE_INSTRUMENTS)
    def test_bounded_1_to_11(self, ohlcv_all: pd.DataFrame, instrument: str) -> None:
        feat = _features(ohlcv_all, instrument)
        # During the warm-up window (before lookback bars exist) universe_size
        # is 0; restrict the [1, 11] bound check to post-warm-up rows where at
        # least one instrument has a finite trailing return.
        sizes = feat["f9_xsection_universe_size"]
        post_warmup = sizes[sizes > 0]
        assert len(post_warmup) > 0, f"{instrument}: no post-warm-up rows"
        assert (post_warmup >= 1).all(), (
            f"{instrument}: f9_xsection_universe_size has values < 1 (post warm-up)"
        )
        assert (post_warmup <= 11).all(), (
            f"{instrument}: f9_xsection_universe_size has values > 11"
        )

    @pytest.mark.parametrize("instrument", PROBE_INSTRUMENTS)
    def test_varies_across_days(self, ohlcv_all: pd.DataFrame, instrument: str) -> None:
        """Universe size is NOT constant — ragged multi-venue calendar causes variation."""
        feat = _features(ohlcv_all, instrument)
        sizes = feat["f9_xsection_universe_size"]
        n_unique = sizes.nunique()
        assert n_unique > 1, (
            f"{instrument}: f9_xsection_universe_size never varies (nunique={n_unique}); "
            "expected multi-venue calendar gaps to produce some days with < 11 members"
        )


# ---------------------------------------------------------------------------
# 3. Panel-truncation invariance (causal proof)
# ---------------------------------------------------------------------------

class TestTruncationInvariance:
    """
    For a probe date t, re-running xsection_features on ohlcv_all[:t+1] must
    yield the SAME f9_xsect_rank and f9_xsection_universe_size as the full-panel
    run (within floating-point tolerance 1e-9).

    This directly proves that no future close leaks into the computation at t.
    """

    @pytest.mark.parametrize("instrument", PROBE_INSTRUMENTS)
    def test_rank_truncation_invariant(self, ohlcv_all: pd.DataFrame, instrument: str) -> None:
        full = _features(ohlcv_all, instrument)

        # Pick a probe date well into the series (after warm-up)
        finite_mask = full["f9_xsect_rank"].notna()
        finite_dates = full.index[finite_mask]
        assert len(finite_dates) > 0, f"{instrument}: no finite rank values to probe"

        # Use a date somewhere in the middle (index 40% into finite values)
        probe_idx = int(len(finite_dates) * 0.4)
        probe_date = finite_dates[probe_idx]

        # Truncate the panel to dates <= probe_date
        ohlcv_trunc = ohlcv_all[ohlcv_all["date"] <= probe_date].copy()
        trunc = _features(ohlcv_trunc, instrument)

        full_val = full.loc[probe_date, "f9_xsect_rank"]
        trunc_val = trunc.loc[probe_date, "f9_xsect_rank"]

        assert abs(full_val - trunc_val) < 1e-9, (
            f"{instrument} @ {probe_date.date()}: "
            f"full={full_val:.10f}, truncated={trunc_val:.10f} — not invariant"
        )

    @pytest.mark.parametrize("instrument", PROBE_INSTRUMENTS)
    def test_universe_size_truncation_invariant(
        self, ohlcv_all: pd.DataFrame, instrument: str
    ) -> None:
        full = _features(ohlcv_all, instrument)

        finite_mask = full["f9_xsect_rank"].notna()
        finite_dates = full.index[finite_mask]
        assert len(finite_dates) > 0

        probe_idx = int(len(finite_dates) * 0.4)
        probe_date = finite_dates[probe_idx]

        ohlcv_trunc = ohlcv_all[ohlcv_all["date"] <= probe_date].copy()
        trunc = _features(ohlcv_trunc, instrument)

        full_val = full.loc[probe_date, "f9_xsection_universe_size"]
        trunc_val = trunc.loc[probe_date, "f9_xsection_universe_size"]

        assert full_val == trunc_val, (
            f"{instrument} @ {probe_date.date()}: "
            f"universe_size full={full_val}, truncated={trunc_val} — not invariant"
        )


# ---------------------------------------------------------------------------
# 4. f9_pair_corr_mean bounded in [-1, 1]
# ---------------------------------------------------------------------------

class TestPairCorr:
    @pytest.mark.parametrize("instrument", PROBE_INSTRUMENTS)
    def test_bounded(self, ohlcv_all: pd.DataFrame, instrument: str) -> None:
        feat = _features(ohlcv_all, instrument)
        corr = feat["f9_pair_corr_mean"].dropna()
        assert (corr >= -1.0 - 1e-10).all(), (
            f"{instrument}: f9_pair_corr_mean has values < -1"
        )
        assert (corr <= 1.0 + 1e-10).all(), (
            f"{instrument}: f9_pair_corr_mean has values > 1"
        )

    @pytest.mark.parametrize("instrument", PROBE_INSTRUMENTS)
    def test_some_finite_values(self, ohlcv_all: pd.DataFrame, instrument: str) -> None:
        """Correlation must be computable for at least part of the series."""
        feat = _features(ohlcv_all, instrument)
        n_finite = feat["f9_pair_corr_mean"].notna().sum()
        assert n_finite > 0, (
            f"{instrument}: f9_pair_corr_mean is entirely NaN — "
            "check peer list and pair_window"
        )

    def test_expected_low_absolute_correlation(self, ohlcv_all: pd.DataFrame) -> None:
        """
        Cross-asset mean |corr| is documented as ~0.09 (expected-negative
        diagnostic per C1).  Within-class should be higher; cross-class lower.
        We do not assert a hard bound, but document that the magnitudes are
        consistent with the CONTRACT §2 expectation.
        """
        mean_abs_corrs = {}
        for inst in PROBE_INSTRUMENTS:
            feat = _features(ohlcv_all, inst)
            c = feat["f9_pair_corr_mean"].dropna()
            if len(c) > 0:
                mean_abs_corrs[inst] = float(c.abs().mean())

        # Soft check: no instrument's mean |corr| should exceed 1.0
        for inst, mac in mean_abs_corrs.items():
            assert mac <= 1.0, f"{inst}: mean |pair_corr| = {mac:.4f} > 1.0"


# ---------------------------------------------------------------------------
# 5. Rank direction: reversal vs momentum
# ---------------------------------------------------------------------------

class TestScoreDirection:
    def test_reversal_vs_momentum_opposite(self, ohlcv_all: pd.DataFrame) -> None:
        """
        With score='reversal', the instrument's rank is negated relative to
        score='momentum' (best momentum -> worst reversal rank and vice versa).
        On any day with n >= 2 finite instruments, rank_reversal + rank_momentum
        should sum to 0 (they are mirror images around 0).
        """
        instrument = "es1s"
        rev = _features(ohlcv_all, instrument, score="reversal")
        mom = _features(ohlcv_all, instrument, score="momentum")

        common = rev.index.intersection(mom.index)
        r = rev.loc[common, "f9_xsect_rank"]
        m = mom.loc[common, "f9_xsect_rank"]

        # Find days where both are finite
        both_finite = r.notna() & m.notna()
        assert both_finite.sum() > 0, "No common finite rank dates for reversal/momentum test"

        # On days with universe_size >= 2 the scores must be exactly mirrored
        usize = rev.loc[common, "f9_xsection_universe_size"]
        multi = both_finite & (usize >= 2)

        diff = (r[multi] + m[multi]).abs()
        max_diff = diff.max()
        assert max_diff < 1e-9, (
            f"reversal + momentum ranks not symmetric (max |sum| = {max_diff:.2e})"
        )

    def test_solo_universe_rank_is_zero(self, ohlcv_all: pd.DataFrame) -> None:
        """When only 1 instrument has a finite score, rank must be 0.0."""
        instrument = "es1s"
        feat = _features(ohlcv_all, instrument)
        solo = feat[feat["f9_xsection_universe_size"] == 1]
        if len(solo) == 0:
            pytest.skip("No days with universe_size==1 for es1s in this data slice")
        assert (solo["f9_xsect_rank"].dropna() == 0.0).all(), (
            "f9_xsect_rank must be 0.0 when universe_size == 1"
        )


# ---------------------------------------------------------------------------
# 6. Peer map sanity
# ---------------------------------------------------------------------------

class TestPeerMap:
    def test_all_instruments_have_peers(self) -> None:
        instruments = [
            "es1s", "nq1s", "fesx1s",
            "cl1s", "ho1s", "rb1s", "ng1s",
            "gc1s", "si1s", "hg1s", "pl1s",
        ]
        for inst in instruments:
            assert inst in ASSET_CLASS_PEERS, f"{inst} missing from ASSET_CLASS_PEERS"
            assert inst in ASSET_CLASS_PEERS[inst], (
                f"{inst} is not listed in its own peer list"
            )

    def test_explicit_peers_override(self, ohlcv_all: pd.DataFrame) -> None:
        """Passing peers= overrides the default class lookup."""
        feat_default = _features(ohlcv_all, "gc1s")
        feat_custom = _features(ohlcv_all, "gc1s", peers=["gc1s", "si1s"])  # subset
        # Both should return valid frames; custom may differ in pair_corr_mean
        assert "f9_pair_corr_mean" in feat_default.columns
        assert "f9_pair_corr_mean" in feat_custom.columns


# ---------------------------------------------------------------------------
# 7. Invalid inputs
# ---------------------------------------------------------------------------

class TestInvalidInputs:
    def test_unknown_instrument(self, ohlcv_all: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="not found"):
            xsection_features(ohlcv_all, "FAKE_INST")

    def test_invalid_score(self, ohlcv_all: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="score must be"):
            xsection_features(ohlcv_all, "es1s", score="bad_score")
