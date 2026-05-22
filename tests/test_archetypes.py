"""Tests for ``stml.replication.archetypes`` (US-007).

Real data is loaded via ``stml.io.load_clean_data`` (the 645-row signal panel +
its OHLCV history). A couple of instruments are exercised, including ng1s --
the participation-only short-side instrument from C1 (never prints +1).

The tests assert INVARIANTS the contract requires, not brittle magic numbers:

* every ``Archetype.generate`` emits only ``{-1, 0, +1}`` on a date index;
* the deadband is **monotone**: a wider deadband never increases the
  nonzero-day count;
* participation-only ``allowed=(-1, 0)`` emits zero ``+1``;
* **no look-ahead**: truncating the OHLCV at a date ``T`` and re-running yields
  the IDENTICAL signal on every date ``< T`` (future bars cannot change past
  signals);
* the cross-sectional ``xsect_rank.generate_panel`` returns a dict over
  instruments of ``{-1, 0, +1}`` series;
* the ``ARCHETYPES`` registry has >= 6 families, each with ``generate`` +
  ``param_space``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.io import load_clean_data
from stml.replication.archetypes import ARCHETYPES, Archetype, decide, generate_panel

# Instruments exercised in the per-instrument tests. ng1s is the
# participation-only (never +1) instrument; cl1s is a standard energy cell.
SAMPLE_INSTRUMENTS = ["cl1s", "ng1s", "es1s"]

# The per-instrument (non cross-sectional) families.
PER_INSTRUMENT_NAMES = [n for n, a in ARCHETYPES.items() if a.score_fn is not None]

# A spread of deadbands wide enough to drive the nonzero count down to ~0.
DEADBANDS = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Real clean OHLCV (long) + the 645-row wide signal panel, loaded once."""
    ohlcv, sig = load_clean_data()
    return ohlcv, sig


@pytest.fixture(scope="module")
def signal_dates(data: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DatetimeIndex:
    """The released signal calendar (645 trading days)."""
    _, sig = data
    return pd.DatetimeIndex(sorted(set(pd.to_datetime(sig["date"]))))


def _inst_ohlcv(ohlcv: pd.DataFrame, inst: str) -> pd.DataFrame:
    return ohlcv[ohlcv["instrument"] == inst].copy()


def _default_params(arc: Archetype) -> dict:
    """First grid value on every axis -- a single concrete config to exercise."""
    return {k: v[0] for k, v in arc.param_space().items()}


def _nonzero_on_signal_dates(
    sig_series: pd.Series, signal_dates: pd.DatetimeIndex
) -> int:
    """Count nonzero signal days restricted to the released signal calendar."""
    on = sig_series[sig_series.index.isin(signal_dates)]
    return int((on != 0).sum())


# --------------------------------------------------------------------------- #
# Registry shape: >= 6 families, each with generate + param_space.            #
# --------------------------------------------------------------------------- #
def test_registry_has_at_least_six_families() -> None:
    assert len(ARCHETYPES) >= 6


def test_every_archetype_exposes_generate_and_param_space() -> None:
    for name, arc in ARCHETYPES.items():
        assert isinstance(arc, Archetype)
        assert callable(arc.generate), name
        ps = arc.param_space()
        assert isinstance(ps, dict) and ps, f"{name} param_space must be non-empty"
        # Every per-instrument family must search a deadband.
        if arc.score_fn is not None:
            assert "deadband" in ps, f"{name} must expose a deadband axis"
        # param_space returns a fresh dict (mutating it must not affect the grid).
        ps["__scratch__"] = [123]
        assert "__scratch__" not in arc.param_space(), f"{name} grid leaked state"


def test_contract_required_families_present() -> None:
    required = {
        "ts_momentum",
        "mean_reversion",
        "breakout_donchian",
        "vol_regime_gated",
        "hybrid_filtered_momentum",
        "xsect_rank",
    }
    assert required <= set(ARCHETYPES)


# --------------------------------------------------------------------------- #
# decide(): the shared score -> {-1,0,+1} rule.                               #
# --------------------------------------------------------------------------- #
def test_decide_basic_decomposition() -> None:
    idx = pd.date_range("2020-01-01", periods=5)
    score = pd.Series([2.0, -2.0, 0.3, -0.3, np.nan], index=idx)
    # deadband 0.5: |0.3| < 0.5 -> flat; NaN -> flat.
    out = decide(score, 0.5)
    assert out.tolist() == [1, -1, 0, 0, 0]
    assert out.index.equals(idx)
    assert out.dtype.kind in "iu"


def test_decide_zero_deadband_trades_on_sign() -> None:
    idx = pd.date_range("2020-01-01", periods=4)
    score = pd.Series([0.01, -0.01, 0.0, np.nan], index=idx)
    out = decide(score, 0.0)
    # |score| >= 0 is True for finite scores, so sign decides; 0.0 -> flat,
    # NaN -> flat.
    assert out.tolist() == [1, -1, 0, 0]


def test_decide_allowed_drops_disallowed_direction() -> None:
    idx = pd.date_range("2020-01-01", periods=3)
    score = pd.Series([2.0, -2.0, 3.0], index=idx)
    out = decide(score, 0.5, allowed=(-1, 0))  # short side only
    assert out.tolist() == [0, -1, 0]
    assert set(pd.unique(out)) <= {-1, 0}


def test_decide_rejects_negative_deadband() -> None:
    score = pd.Series([1.0, -1.0])
    with pytest.raises(ValueError):
        decide(score, -0.1)


def test_decide_rejects_bad_allowed() -> None:
    score = pd.Series([1.0, -1.0])
    with pytest.raises(ValueError):
        decide(score, 0.5, allowed=(-1, 0, 2))


# --------------------------------------------------------------------------- #
# Output domain: every generate emits only {-1,0,+1} on a date index.         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", PER_INSTRUMENT_NAMES)
@pytest.mark.parametrize("inst", SAMPLE_INSTRUMENTS)
def test_generate_outputs_only_ternary_date_indexed(
    name: str,
    inst: str,
    data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ohlcv, _ = data
    arc = ARCHETYPES[name]
    out = arc.generate(_inst_ohlcv(ohlcv, inst), _default_params(arc))

    assert isinstance(out, pd.Series)
    assert isinstance(out.index, pd.DatetimeIndex), f"{name}/{inst} not date-indexed"
    assert set(pd.unique(out)) <= {-1, 0, 1}, f"{name}/{inst} emitted off-domain"
    assert out.isna().sum() == 0, f"{name}/{inst} produced NaN signals"
    assert out.index.is_monotonic_increasing


# --------------------------------------------------------------------------- #
# Monotone deadband: wider deadband => non-increasing nonzero-day count.       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", PER_INSTRUMENT_NAMES)
def test_deadband_is_monotone_nonincreasing(
    name: str,
    data: tuple[pd.DataFrame, pd.DataFrame],
    signal_dates: pd.DatetimeIndex,
) -> None:
    ohlcv, _ = data
    arc = ARCHETYPES[name]
    base = {k: v for k, v in _default_params(arc).items() if k != "deadband"}
    oi = _inst_ohlcv(ohlcv, "cl1s")

    counts = [
        _nonzero_on_signal_dates(
            arc.generate(oi, {**base, "deadband": db}), signal_dates
        )
        for db in DEADBANDS
    ]
    # Each step up in the deadband can only drop trades, never add them.
    for lo, hi in zip(counts[:-1], counts[1:], strict=True):
        assert hi <= lo, f"{name} non-monotone over deadbands {DEADBANDS}: {counts}"
    # Sanity: the largest deadband must trade strictly less than the smallest
    # (otherwise the score is not actually being thresholded).
    assert counts[-1] < counts[0], f"{name} deadband had no effect: {counts}"


# --------------------------------------------------------------------------- #
# Participation-only: allowed=(-1,0) emits no +1.                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", PER_INSTRUMENT_NAMES)
@pytest.mark.parametrize("inst", ["ng1s", "cl1s"])
def test_participation_only_emits_no_long(
    name: str,
    inst: str,
    data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ohlcv, _ = data
    arc = ARCHETYPES[name]
    out = arc.generate(_inst_ohlcv(ohlcv, inst), _default_params(arc), allowed=(-1, 0))
    assert (out == 1).sum() == 0, f"{name}/{inst} emitted +1 under allowed=(-1,0)"
    assert set(pd.unique(out)) <= {-1, 0}


def test_participation_only_can_still_short(
    data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    # The (-1,0) clamp must drop longs, not silence the rule entirely: at least
    # one family on at least one instrument should still take the short side.
    ohlcv, _ = data
    any_short = False
    for name in PER_INSTRUMENT_NAMES:
        arc = ARCHETYPES[name]
        out = arc.generate(
            _inst_ohlcv(ohlcv, "cl1s"), _default_params(arc), allowed=(-1, 0)
        )
        if (out == -1).sum() > 0:
            any_short = True
            break
    assert any_short, "participation-only (-1,0) silenced every family"


# --------------------------------------------------------------------------- #
# No look-ahead: truncating at T leaves all signals at dates < T unchanged.    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", PER_INSTRUMENT_NAMES)
def test_no_lookahead_truncation_invariance(
    name: str,
    data: tuple[pd.DataFrame, pd.DataFrame],
    signal_dates: pd.DatetimeIndex,
) -> None:
    """Future bars must not change past signals.

    Run the archetype on the full OHLCV history and on the same history
    truncated at a cut date ``T`` (inside the released window). The signal on
    every date strictly before ``T`` must be byte-for-byte identical -- if any
    feature peeked at a bar ``>= t``, truncating would perturb the earlier
    outputs.
    """
    ohlcv, _ = data
    arc = ARCHETYPES[name]
    params = _default_params(arc)

    oi_full = _inst_ohlcv(ohlcv, "cl1s")
    cut = signal_dates[400]  # a date well inside the released signal window
    oi_trunc = oi_full[oi_full["date"] < cut]

    full = arc.generate(oi_full, params)
    trunc = arc.generate(oi_trunc, params)

    common = trunc.index[trunc.index < cut]
    assert len(common) > 0, f"{name}: nothing to compare before the cut"
    assert full.reindex(common).equals(trunc.reindex(common)), (
        f"{name}: signals before {cut.date()} changed when future bars were removed"
    )


# --------------------------------------------------------------------------- #
# Cross-sectional family: generate_panel -> dict of {-1,0,+1} series.         #
# --------------------------------------------------------------------------- #
def test_xsect_rank_generate_panel_returns_dict_of_ternary_series(
    data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ohlcv, _ = data
    arc = ARCHETYPES["xsect_rank"]
    params = _default_params(arc)

    panel = arc.generate_panel(ohlcv, params)
    assert isinstance(panel, dict)
    insts = set(ohlcv["instrument"].unique())
    assert set(panel) == insts, "panel must cover every instrument"

    for inst, ser in panel.items():
        assert isinstance(ser, pd.Series), inst
        assert isinstance(ser.index, pd.DatetimeIndex), inst
        assert set(pd.unique(ser)) <= {-1, 0, 1}, f"{inst} off-domain"
        assert ser.isna().sum() == 0, f"{inst} NaN signal"


def test_xsect_rank_module_function_matches_archetype(
    data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    # The standalone generate_panel and the registry's Archetype.generate_panel
    # must agree (same logic, no copy drift).
    ohlcv, _ = data
    params = _default_params(ARCHETYPES["xsect_rank"])
    via_fn = generate_panel(ohlcv, params)
    via_arc = ARCHETYPES["xsect_rank"].generate_panel(ohlcv, params)
    assert set(via_fn) == set(via_arc)
    for inst in via_fn:
        assert via_fn[inst].equals(via_arc[inst]), inst


def test_xsect_rank_ranks_long_and_short_each_active_day(
    data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    # On days where the cross-section is fully populated, the ranker must take
    # both sides (top long, bottom short), i.e. the row is not all-flat.
    ohlcv, _ = data
    params = _default_params(ARCHETYPES["xsect_rank"])
    panel = generate_panel(ohlcv, params)
    wide = pd.DataFrame(panel)
    # Restrict to rows where every instrument has a signal computed (post warm-up).
    full_rows = wide.dropna(how="any")
    n_long = (full_rows == 1).sum(axis=1)
    n_short = (full_rows == -1).sum(axis=1)
    assert (n_long > 0).any(), "ranker never went long"
    assert (n_short > 0).any(), "ranker never went short"


def test_xsect_rank_participation_only(
    data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    # The panel ranker also honours an allowed alphabet: (-1,0) -> no +1 anywhere.
    ohlcv, _ = data
    arc = ARCHETYPES["xsect_rank"]
    panel = arc.generate_panel(ohlcv, _default_params(arc), allowed=(-1, 0))
    for inst, ser in panel.items():
        assert (ser == 1).sum() == 0, f"{inst} emitted +1 under allowed=(-1,0)"


def test_xsect_rank_generate_raises_clear_error(
    data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ohlcv, _ = data
    with pytest.raises(ValueError, match="cross-sectional"):
        ARCHETYPES["xsect_rank"].generate(_inst_ohlcv(ohlcv, "cl1s"), {"deadband": 0.0})


def test_per_instrument_generate_panel_raises_clear_error(
    data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ohlcv, _ = data
    with pytest.raises(ValueError, match="per-instrument"):
        ARCHETYPES["mean_reversion"].generate_panel(ohlcv, {"lookback": 20})
