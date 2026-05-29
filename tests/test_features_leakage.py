"""Leakage tests for ``stml.metamodel.features`` (AC-6a).

The engineered (E-class) feature families must be **look-ahead-free by
construction**, which the contract operationalises as **truncation-invariance**:
truncating the inputs at a cut date ``T`` and recomputing reproduces the
IDENTICAL value on every date ``< T`` (no future bar can perturb a past value).

These tests assert INVARIANTS, not magic numbers, on REAL data
(:func:`stml.io.load_clean_data`) across >= 3 instruments including ng1s (the
participation-only, sparse-signal energy instrument from C1):

* **truncation-invariance for EVERY family** — exact-equal for integer/exact
  scans, else within 1e-9 — with the high-risk F5 ``f5_trailing_run_length`` and
  ``f5_days_since_flip`` columns (MUST-FIX-2) explicitly enumerated;
* **F7 Amihud zero-volume guard** — an all-zero-volume series yields all-NaN
  Amihud (never a divide-by-zero ``inf``);
* **no forward-fill of structural NaN** — a deliberately gapped input leaves NaN
  exactly where it was, never a filled value;
* **every family returns only finite-or-NaN floats on a date index** and does
  not crash on ng1s.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.io import load_clean_data
from stml.metamodel import features as F

# Instruments exercised. ng1s is the sparse / participation-only energy cell;
# cl1s a standard energy; es1s a high-liquidity equity index. (Contract: >=3
# instruments incl. ng1s.)
SAMPLE_INSTRUMENTS = ["cl1s", "ng1s", "es1s"]

# Price families (computed from OHLCV alone), keyed by the public function.
PRICE_FAMILIES = {
    "f1": F.f1_counter_trend,
    "f2": F.f2_vol_dispersion,
    "f6": F.f6_momentum_contrast,
    "f7": F.f7_microstructure,
    "f10": F.f10_price_action,
}

# The high-risk trailing F5 columns the contract demands be enumerated.
F5_HIGH_RISK_COLS = ["f5_trailing_run_length", "f5_days_since_flip"]
F5_EXACT_COLS = [
    "f5_signal",
    "f5_abs_signal",
    "f5_trailing_run_length",
    "f5_days_since_flip",
    "f5_days_since_nonzero",
]


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
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


def _inst_signal(sig: pd.DataFrame, inst: str) -> pd.Series:
    return pd.Series(sig.set_index("date")[inst]).sort_index()


def _frames_match(a: pd.DataFrame, b: pd.DataFrame, tol: float = 1e-9) -> None:
    """Assert two aligned frames match: NaN pattern identical, finite within tol."""
    assert list(a.columns) == list(b.columns)
    assert (a.isna() == b.isna()).all().all(), "NaN pattern differs after truncation"
    diff = (a - b).abs().to_numpy(dtype=float)
    finite = diff[np.isfinite(diff)]
    if finite.size:
        assert finite.max() <= tol, f"max abs diff {finite.max():.3e} exceeds {tol:.0e}"


# --------------------------------------------------------------------------- #
# Output domain: every family is finite-or-NaN floats on a date index, no NG1s #
# crash.                                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("inst", SAMPLE_INSTRUMENTS)
def test_every_family_returns_finite_or_nan_floats(
    inst: str, data: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    ohlcv, sig = data
    oi = _inst_ohlcv(ohlcv, inst)
    s = _inst_signal(sig, inst)

    blocks = {
        "f1": F.f1_counter_trend(oi),
        "f2": F.f2_vol_dispersion(oi),
        "f5": F.f5_signal_derived(s),
        "f6": F.f6_momentum_contrast(oi),
        "f7": F.f7_microstructure(oi),
        "f10": F.f10_price_action(oi),
        "f8": F.f8_calendar(oi["date"].drop_duplicates().sort_values()),
    }
    for name, block in blocks.items():
        assert isinstance(block, pd.DataFrame), f"{name}/{inst} not a DataFrame"
        assert isinstance(block.index, pd.DatetimeIndex), f"{name}/{inst} not date-indexed"
        assert block.index.is_monotonic_increasing, f"{name}/{inst} unsorted index"
        assert block.dtypes.map(pd.api.types.is_float_dtype).all(), (
            f"{name}/{inst} has a non-float column"
        )
        arr = block.to_numpy(dtype=float)
        assert not np.isinf(arr).any(), f"{name}/{inst} produced an inf"


def test_assemble_engineered_runs_on_ng1s(
    data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ohlcv, sig = data
    asm = F.assemble_engineered(_inst_ohlcv(ohlcv, "ng1s"), _inst_signal(sig, "ng1s"))
    assert isinstance(asm.index, pd.DatetimeIndex)
    assert asm.index.is_monotonic_increasing
    assert asm.dtypes.map(pd.api.types.is_float_dtype).all()
    assert not asm.columns.duplicated().any(), "duplicate feature columns"
    assert not np.isinf(asm.to_numpy(dtype=float)).any()
    # The contract's required named columns must be present.
    for col in (
        "f1_mr_score_20",
        "f2_vol_20",
        "f5_trailing_run_length",
        "f5_days_since_flip",
        "f5_signal",
        "f5_abs_signal",
    ):
        assert col in asm.columns, f"missing required column {col}"


# NOTE: the former ``test_f1_mr_score_20_mirrors_archetype`` cross-checked the
# F1 feature against the signal-replication ``archetypes._score_mean_reversion``
# implementation. That package is intentionally absent from the feature-base
# branch, so the cross-check was removed. ``f1_mr_score_20`` remains covered by
# the truncation-invariance and catalog-coverage tests below.


# --------------------------------------------------------------------------- #
# Truncation-invariance: PRICE families (info <= t).                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("inst", SAMPLE_INSTRUMENTS)
@pytest.mark.parametrize("fam", list(PRICE_FAMILIES))
def test_price_family_truncation_invariant(
    inst: str,
    fam: str,
    data: tuple[pd.DataFrame, pd.DataFrame],
    signal_dates: pd.DatetimeIndex,
) -> None:
    """Future bars must not change past values for any price family.

    Compute the family on the full OHLCV history and on the same history
    truncated at a cut date ``T`` (well inside the released window). Every value
    on dates strictly before ``T`` must match within tolerance, with an
    identical NaN pattern.
    """
    ohlcv, _ = data
    fn = PRICE_FAMILIES[fam]
    oi_full = _inst_ohlcv(ohlcv, inst)
    cut = signal_dates[400]
    oi_trunc = oi_full[oi_full["date"] <= cut]

    full = fn(oi_full)
    trunc = fn(oi_trunc)
    common = trunc.index[trunc.index < cut]
    assert len(common) > 0, f"{fam}/{inst}: nothing to compare before the cut"
    _frames_match(full.reindex(common), trunc.reindex(common))


# --------------------------------------------------------------------------- #
# Truncation-invariance: F5 signal-derived, INCL. the enumerated high-risk     #
# run-length / days-since-flip columns (MUST-FIX-2).                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("inst", SAMPLE_INSTRUMENTS)
def test_f5_truncation_invariant_all_columns(
    inst: str,
    data: tuple[pd.DataFrame, pd.DataFrame],
    signal_dates: pd.DatetimeIndex,
) -> None:
    ohlcv, sig = data
    s_full = _inst_signal(sig, inst)
    mr_full = F.f1_counter_trend(_inst_ohlcv(ohlcv, inst))["f1_mr_score_20"]

    cut = signal_dates[400]
    s_trunc = s_full[s_full.index <= cut]

    full = F.f5_signal_derived(s_full, mr_score=mr_full.reindex(s_full.index))
    trunc = F.f5_signal_derived(s_trunc, mr_score=mr_full.reindex(s_trunc.index))
    common = trunc.index[trunc.index < cut]
    assert len(common) > 0, f"f5/{inst}: nothing to compare before the cut"
    _frames_match(full.reindex(common), trunc.reindex(common))


@pytest.mark.parametrize("inst", SAMPLE_INSTRUMENTS)
@pytest.mark.parametrize("col", F5_EXACT_COLS)
def test_f5_high_risk_columns_exact_truncation_invariant(
    inst: str,
    col: str,
    data: tuple[pd.DataFrame, pd.DataFrame],
    signal_dates: pd.DatetimeIndex,
) -> None:
    """The integer-valued F5 scans (incl. the MUST-FIX-2 columns) must be
    BYTE-FOR-BYTE identical before the cut, not merely within tolerance — they
    are cumulative-from-left counts, so any look-ahead would perturb them."""
    ohlcv, sig = data
    s_full = _inst_signal(sig, inst)
    cut = signal_dates[400]
    s_trunc = s_full[s_full.index <= cut]

    full = F.f5_signal_derived(s_full)[col]
    trunc = F.f5_signal_derived(s_trunc)[col]
    common = trunc.index[trunc.index < cut]
    assert full.reindex(common).equals(trunc.reindex(common)), (
        f"f5/{inst}: {col} changed when future signal rows were removed"
    )


def test_f5_high_risk_columns_are_enumerated() -> None:
    """Guard the contract requirement that the two MUST-FIX-2 columns exist and
    are exercised by the exact-equality test above."""
    cols = set(F.f5_signal_derived(
        pd.Series([0, 1, -1, -1, 0], index=pd.bdate_range("2020-01-01", periods=5))
    ).columns)
    for c in F5_HIGH_RISK_COLS:
        assert c in cols, f"missing high-risk F5 column {c}"
        assert c in set(F5_EXACT_COLS)


def test_f5_trailing_run_length_is_cumulative_not_full_period() -> None:
    """``f5_trailing_run_length`` is the CURRENT run ending at ``t`` (expanding,
    cumulative-from-left) — NOT the full-period run-length statistic
    ``splits.run_length_p90`` (which would leak the future)."""
    idx = pd.bdate_range("2020-01-01", periods=6)
    s = pd.Series([1, 1, 0, 0, 0, 1], index=idx)
    f5 = F.f5_signal_derived(s)
    # Expanding run lengths: 1,2 then reset, 1,2,3 then reset to 1.
    assert f5["f5_trailing_run_length"].tolist() == [1.0, 2.0, 1.0, 2.0, 3.0, 1.0]
    assert f5["f5_days_since_flip"].tolist() == [0.0, 1.0, 0.0, 1.0, 2.0, 0.0]
    # And it is NOT constant at the full-period p90 (the leaky alternative).
    from stml.metamodel.splits import run_length_p90

    p90 = run_length_p90(s)
    assert not (f5["f5_trailing_run_length"] == p90).all()


# --------------------------------------------------------------------------- #
# F8 calendar: pure function of the index, trivially truncation-invariant.     #
# --------------------------------------------------------------------------- #
def test_f8_calendar_truncation_invariant(
    data: tuple[pd.DataFrame, pd.DataFrame], signal_dates: pd.DatetimeIndex
) -> None:
    ohlcv, _ = data
    idx_full = pd.DatetimeIndex(
        _inst_ohlcv(ohlcv, "cl1s")["date"].drop_duplicates().sort_values()
    )
    cut = signal_dates[400]
    idx_trunc = idx_full[idx_full < cut]
    full = F.f8_calendar(idx_full)
    trunc = F.f8_calendar(idx_trunc)
    _frames_match(full.reindex(idx_trunc), trunc)


# --------------------------------------------------------------------------- #
# F7 Amihud zero-volume guard: |ret| / 0 must NEVER blow up.                   #
# --------------------------------------------------------------------------- #
def test_f7_amihud_zero_volume_is_nan_not_inf() -> None:
    n = 30
    idx = pd.bdate_range("2020-01-01", periods=n)
    df = pd.DataFrame(
        {
            "date": idx,
            "instrument": "z",
            "open": np.linspace(10.0, 11.0, n),
            "high": np.linspace(10.1, 11.1, n),
            "low": np.linspace(9.9, 10.9, n),
            "close": np.linspace(10.0, 11.0, n),
            "volume": np.zeros(n),  # every row zero-volume
            "open_interest": np.arange(n, dtype=float),
        }
    )
    f7 = F.f7_microstructure(df)
    # With ALL volume zero, the guard makes every Amihud row NaN (never inf).
    assert f7["f7_amihud_20"].isna().all(), "zero-volume Amihud must be NaN"
    assert not np.isinf(f7["f7_amihud_20"].to_numpy(dtype=float)).any()


def test_f7_amihud_mixed_zero_volume_never_inf() -> None:
    n = 30
    idx = pd.bdate_range("2020-01-01", periods=n)
    df = pd.DataFrame(
        {
            "date": idx,
            "instrument": "z",
            "open": np.linspace(10.0, 11.0, n),
            "high": np.linspace(10.1, 11.1, n),
            "low": np.linspace(9.9, 10.9, n),
            "close": np.linspace(10.0, 11.0, n),
            "volume": np.full(n, 100.0),
            "open_interest": np.arange(n, dtype=float),
        }
    )
    df.loc[15, "volume"] = 0.0  # a single zero-volume day among positives
    f7 = F.f7_microstructure(df)
    assert not np.isinf(f7["f7_amihud_20"].to_numpy(dtype=float)).any()


# --------------------------------------------------------------------------- #
# F10 price-action returns: closed-form math, first-row NaN, non-positive guard #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("inst", SAMPLE_INSTRUMENTS)
def test_f10_matches_log_definitions(
    inst: str, data: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """The f10 columns equal their closed-form log definitions on real data."""
    ohlcv, _ = data
    oi = _inst_ohlcv(ohlcv, inst)
    f10 = F.f10_price_action(oi)
    df = F._ohlcv_indexed(oi)  # same sorted/deduped frame the feature uses

    hl = np.log(df["high"] / df["low"])
    oto = np.log(df["open"] / df["open"].shift(1))

    # Daily values match the closed forms (finite rows within tolerance).
    for col, ref in (("f10_hl_range", hl), ("f10_oto_ret", oto)):
        a = f10[col]
        b = ref.reindex(a.index)
        diff = (a - b).abs().to_numpy(dtype=float)
        finite = diff[np.isfinite(diff)]
        assert finite.size and finite.max() <= 1e-12, f"{inst}: {col} != log def"

    # The *_mean_20 columns are exactly a trailing right-aligned rolling(20) mean.
    assert np.allclose(
        f10["f10_hl_range_mean_20"].to_numpy(dtype=float),
        f10["f10_hl_range"].rolling(20, min_periods=20).mean().to_numpy(dtype=float),
        equal_nan=True,
    )
    assert np.allclose(
        f10["f10_oto_ret_mean_20"].to_numpy(dtype=float),
        f10["f10_oto_ret"].rolling(20, min_periods=20).mean().to_numpy(dtype=float),
        equal_nan=True,
    )


def _synthetic_ohlcv(n: int = 25) -> pd.DataFrame:
    """A clean strictly-positive OHLCV frame for the F10 edge-case tests."""
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {
            "date": idx,
            "instrument": "z",
            "open": np.linspace(10.0, 12.0, n),
            "high": np.linspace(10.2, 12.2, n),
            "low": np.linspace(9.8, 11.8, n),
            "close": np.linspace(10.0, 12.0, n),
            "volume": np.full(n, 100.0),
            "open_interest": np.arange(n, dtype=float),
        }
    )


def test_f10_first_open_to_open_is_nan() -> None:
    """The first row has no prior open, so the open-to-open return is NaN."""
    f10 = F.f10_price_action(_synthetic_ohlcv())
    assert np.isnan(f10["f10_oto_ret"].iloc[0]), "first open-to-open must be NaN"
    assert np.isfinite(f10["f10_oto_ret"].iloc[1:].to_numpy(dtype=float)).all()


def test_f10_nonpositive_price_is_nan_not_inf() -> None:
    """A zero high/low/open yields NaN, never an inf (mirrors the Parkinson guard)."""
    df = _synthetic_ohlcv()
    df.loc[10, "low"] = 0.0  # zero low -> high/low range NaN on that row
    df.loc[12, "open"] = 0.0  # zero open -> oto NaN on rows 12 and 13
    f10 = F.f10_price_action(df)
    assert not np.isinf(f10.to_numpy(dtype=float)).any(), "f10 produced an inf"
    assert np.isnan(f10["f10_hl_range"].iloc[10]), "zero-low range must be NaN"
    assert np.isnan(f10["f10_oto_ret"].iloc[12]), "zero-open oto must be NaN"
    assert np.isnan(f10["f10_oto_ret"].iloc[13]), "oto off a zero prior open is NaN"


# --------------------------------------------------------------------------- #
# No forward-fill of structural NaN: a gapped input keeps NaN, never filled.   #
# --------------------------------------------------------------------------- #
def test_f7_open_interest_gap_not_forward_filled() -> None:
    n = 30
    idx = pd.bdate_range("2020-01-01", periods=n)
    df = pd.DataFrame(
        {
            "date": idx,
            "instrument": "z",
            "open": np.linspace(10.0, 11.0, n),
            "high": np.linspace(10.1, 11.1, n),
            "low": np.linspace(9.9, 10.9, n),
            "close": np.linspace(10.0, 11.0, n),
            "volume": np.full(n, 100.0),
            "open_interest": np.arange(n, dtype=float),
        }
    )
    df.loc[10:14, "open_interest"] = np.nan  # a deliberate structural gap
    f7 = F.f7_microstructure(df)
    # The gapped block stays NaN (no ffill), and the next finite row is its own
    # raw value — not the value carried across the gap.
    assert f7["f7_oi_level"].iloc[10:15].isna().all(), "OI gap was forward-filled"
    assert f7["f7_oi_level"].iloc[15] == df.loc[15, "open_interest"]


def test_f5_days_since_nonzero_nan_before_first_participation() -> None:
    """Before the first non-zero signal there is no prior participation to count
    from, so the trailing days-since-nonzero is NaN (not a fabricated 0)."""
    idx = pd.bdate_range("2020-01-01", periods=6)
    s = pd.Series([0, 0, 0, 1, 0, 0], index=idx)
    f5 = F.f5_signal_derived(s)
    dsn = f5["f5_days_since_nonzero"]
    assert dsn.iloc[:3].isna().all(), "days-since-nonzero must be NaN pre-participation"
    assert dsn.iloc[3:].tolist() == [0.0, 1.0, 2.0]
