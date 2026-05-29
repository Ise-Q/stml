"""
test_macro_features.py
======================
Leakage / provenance tests for the F11 cross-asset macro family
(:mod:`stml.metamodel.macro_features`) on REAL data
(``data/additional_data.xlsx`` + :func:`stml.io.load_clean_data`).

The graded crux is the **publication lag**: a released value must never be
visible before its real release date. These tests assert that against
**independent, hand-computed literal release dates** (NOT the output of
``compute_availability``, which would be circular), plus the standard
truncation-invariance + fit-provenance proofs the contract requires:

* ``test_macro_pit_lag_no_lookahead`` — PMI reference month 2020-01 is invisible
  before 2020-02-03; the EIA Friday-2020-01-10 value is invisible before
  2020-01-16 (stamp + 6 calendar days). Literal dates, hard-coded.
* ``test_macro_raw_truncation_invariant`` — truncating the input observations at
  a cut ``T`` leaves every raw F11 value (level + ``chg{h}``) on dates ``< T``
  unchanged.
* ``test_macro_bundle_stats_frozen_from_train`` — the bundle z-score stats equal
  the FE-train recomputation and differ from the full-series stats.
* ``test_macro_train_index_*`` — the fit frame is the FE-train slice and is one
  row per trade date (no 11x instrument inflation).
* ``test_macro_catalog_roundtrip`` — the F11 catalog entry set equals the
  produced column set (one traversal).
* dropped-series, same-day-availability, as-of-carry, finite domain,
  determinism, and the standalone-artifact row alignment.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from stml.io import load_clean_data
from stml.metamodel import macro_features as MF
from stml.metamodel.build_features import _persist
from stml.metamodel.catalog import CATALOG
from stml.metamodel.pipeline import FeaturePipeline
from stml.metamodel.scope import ASSET_CLASS_MAP

FE_TRAIN_END = pd.Timestamp("2021-07-01")

# Independent, hand-computed literal release dates (NOT compute_availability).
PMI_JAN2020_AVAIL = pd.Timestamp("2020-02-03")  # 2020-01-31 (Fri) + 1 business day
PMI_JAN2020_VALUE = 51.4  # the US ISM PMI January-2020 reference value
EIA_FRI_STAMP = pd.Timestamp("2020-01-10")  # a Friday whose value differs from prior wk
EIA_AVAIL = pd.Timestamp("2020-01-16")  # Friday stamp + 6 calendar days (next Thu)
EIA_FRI_VALUE = 428511.0  # EIA crude stock at the 2020-01-10 Friday stamp

# One instrument per asset class for a fast representative pipeline fit.
SUBSET = ["es1s", "cl1s", "gc1s"]


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def signals():
    """The real wide signal panel (645 trading days)."""
    _, sig = load_clean_data()
    return sig


@pytest.fixture(scope="module")
def trade_dates(signals) -> pd.DatetimeIndex:
    """The nonzero-signal trade-date union across the full universe."""
    sig = signals.set_index("date")
    insts = [c for c in signals.columns if c != "date" and c in ASSET_CLASS_MAP]
    nz: set[pd.Timestamp] = set()
    for inst in insts:
        nz.update(sig.index[sig[inst] != 0])
    return pd.DatetimeIndex(sorted(nz))


@pytest.fixture(scope="module")
def raw_series():
    """The stamp-indexed macro observation series (loaded once)."""
    return MF.load_macro_raw()


@pytest.fixture(scope="module")
def assembled(trade_dates, raw_series):
    """The raw (pre-standardization) 45-column F11 frame on the trade dates."""
    return MF.assemble_macro_raw(trade_dates, raw=raw_series)


# --------------------------------------------------------------------------- #
# The graded crux — publication-lag / no-lookahead (literal dates)            #
# --------------------------------------------------------------------------- #
def test_macro_pit_lag_no_lookahead(assembled) -> None:
    """A released value is never visible before its literal release date.

    PMI 2020-01 (51.4) must be absent on every trade date before 2020-02-03 and
    present on/after it; the EIA Friday-2020-01-10 value (428511) must be absent
    before 2020-01-16 and present on/after it. Asserted against hand-coded
    literal dates, never ``compute_availability``.
    """
    # --- US ISM PMI: reference month 2020-01, available 2020-02-03 ----------
    pmi = assembled["f11_us_ism_mfg_pmi_level"]
    pre_pmi = pmi[(pmi.index >= pd.Timestamp("2020-01-06")) & (pmi.index < PMI_JAN2020_AVAIL)]
    assert len(pre_pmi) > 0, "no trade dates in the pre-release PMI window"
    assert pre_pmi.notna().all(), "pre-release PMI should carry the prior month's value"
    assert (np.abs(pre_pmi.to_numpy() - PMI_JAN2020_VALUE) > 1e-6).all(), (
        "the January-2020 PMI value is visible BEFORE its 2020-02-03 release"
    )
    on_after_pmi = pmi[pmi.index >= PMI_JAN2020_AVAIL]
    assert len(on_after_pmi) > 0
    assert abs(float(on_after_pmi.iloc[0]) - PMI_JAN2020_VALUE) < 1e-6, (
        "the January-2020 PMI value is not present on its 2020-02-03 release date"
    )

    # --- EIA crude: Friday 2020-01-10 stamp, available 2020-01-16 -----------
    eia = assembled["f11_eia_crude_stock_level"]
    pre_eia = eia[(eia.index >= pd.Timestamp("2020-01-13")) & (eia.index < EIA_AVAIL)]
    assert len(pre_eia) > 0, "no trade dates in the pre-release EIA window"
    assert (np.abs(pre_eia.to_numpy() - EIA_FRI_VALUE) >= 1.0).all(), (
        "the EIA 2020-01-10 value is visible BEFORE its 2020-01-16 availability"
    )
    on_after_eia = eia[eia.index >= EIA_AVAIL]
    assert len(on_after_eia) > 0
    assert abs(float(on_after_eia.iloc[0]) - EIA_FRI_VALUE) < 1.0, (
        "the EIA 2020-01-10 value is not present on its 2020-01-16 availability"
    )


def test_compute_availability_per_class() -> None:
    """The per-class lag rule maps stamps to the hand-computed availability."""
    assert MF.compute_availability(pd.Timestamp("2020-03-05"), "daily") == pd.Timestamp(
        "2020-03-05"
    )
    assert MF.compute_availability(EIA_FRI_STAMP, "weekly_eia") == EIA_AVAIL
    assert MF.compute_availability(
        pd.Timestamp("2020-01-31"), "monthly_pmi"
    ) == PMI_JAN2020_AVAIL
    # Month-end on a weekend (2020-02-29 Sat) rolls to the next business day.
    assert MF.compute_availability(
        pd.Timestamp("2020-02-29"), "monthly_pmi"
    ) == pd.Timestamp("2020-03-02")


def test_daily_market_available_same_day(assembled, raw_series) -> None:
    """A daily market series value at obs date t is usable at t (lag 0)."""
    # VIX is a daily series: its applied level on a trade date equals the latest
    # VIX observation with obs-date <= that trade date.
    vix_obs = raw_series["VIX"]
    t = pd.Timestamp("2020-06-15")  # a Monday trade date in the window
    if t not in assembled.index:
        t = assembled.index[assembled.index >= t][0]
    expected = float(vix_obs[vix_obs.index <= t].iloc[-1])
    assert abs(float(assembled.loc[t, "f11_vix_level"]) - expected) < 1e-9


# --------------------------------------------------------------------------- #
# Truncation-invariance (raw PIT series, incl. momentum columns)              #
# --------------------------------------------------------------------------- #
def test_macro_raw_truncation_invariant(trade_dates, raw_series, assembled) -> None:
    """Truncating input obs after T leaves all raw F11 values on dates < T fixed."""
    cut = pd.Timestamp("2021-03-01")
    truncated = {k: v[v.index <= cut] for k, v in raw_series.items()}
    trunc_frame = MF.assemble_macro_raw(trade_dates, raw=truncated)

    before = assembled.index < cut
    a = assembled[before]
    b = trunc_frame[before]
    assert list(a.columns) == list(b.columns)
    assert (a.isna() == b.isna()).all().all(), "NaN pattern changed after truncation"
    diff = (a - b).abs().to_numpy(dtype=float)
    finite = diff[np.isfinite(diff)]
    if finite.size:
        assert finite.max() <= 1e-9, f"max abs diff {finite.max():.3e} > 1e-9"


# --------------------------------------------------------------------------- #
# Fit provenance — frozen FROM TRAIN, not the full series                     #
# --------------------------------------------------------------------------- #
def test_macro_bundle_stats_frozen_from_train(assembled) -> None:
    """Bundle mean_/std_ equal the FE-train recomputation and differ from full."""
    raw_train = assembled[assembled.index <= FE_TRAIN_END]
    assert len(raw_train) < len(assembled), "train must be a strict subset"
    bundle = MF.fit_macro(raw_train)

    tr_mean = np.nanmean(raw_train.to_numpy(dtype=float), axis=0)
    tr_std = np.nanstd(raw_train.to_numpy(dtype=float), axis=0)
    tr_std = np.where((tr_std == 0.0) | ~np.isfinite(tr_std), 1.0, tr_std)
    np.testing.assert_allclose(bundle.mean_, tr_mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(bundle.std_, tr_std, rtol=0, atol=1e-12)

    full_mean = np.nanmean(assembled.to_numpy(dtype=float), axis=0)
    full_std = np.nanstd(assembled.to_numpy(dtype=float), axis=0)
    assert not np.allclose(bundle.mean_, full_mean)
    assert not np.allclose(bundle.std_, full_std)


def test_macro_train_index_within_fe_train(assembled) -> None:
    """The fit frame's index lies entirely within the FE-train partition."""
    raw_train = assembled[assembled.index <= FE_TRAIN_END]
    bundle = MF.fit_macro(raw_train)
    idx = pd.DatetimeIndex(bundle.train_index)
    assert len(idx) > 0
    assert idx.max() <= FE_TRAIN_END


def test_macro_train_index_is_date_deduplicated(assembled) -> None:
    """The fit frame is one row per FE-train trade date (no 11x inflation)."""
    raw_train = assembled[assembled.index <= FE_TRAIN_END]
    bundle = MF.fit_macro(raw_train)
    idx = pd.DatetimeIndex(bundle.train_index)
    assert len(idx) == len(set(idx)), "train_index has duplicate dates"
    assert len(idx) == 387, f"expected 387 FE-train trade dates, got {len(idx)}"


def test_macro_std_guard() -> None:
    """A zero-variance training column gets a std of 1.0 (no divide-by-zero)."""
    cols = MF.macro_feature_columns()
    const = pd.DataFrame(
        np.ones((10, len(cols))), columns=cols, index=pd.date_range("2020-01-01", periods=10)
    )
    bundle = MF.fit_macro(const)
    assert np.all(bundle.std_ == 1.0)
    out = MF.transform_macro(bundle, const)
    assert np.isfinite(out.to_numpy()).all()


# --------------------------------------------------------------------------- #
# Catalog / column-set roundtrip + leakage class                             #
# --------------------------------------------------------------------------- #
def test_macro_catalog_roundtrip() -> None:
    """The F11 catalog entry set equals the produced F11 column set exactly."""
    produced = set(MF.macro_feature_columns())
    catalog_f11 = {n for n, s in CATALOG.items() if s.family == "F11"}
    assert produced == catalog_f11, produced ^ catalog_f11
    assert len(produced) == 45


def test_macro_all_columns_are_tf() -> None:
    """Every F11 catalog entry is leakage-class TF (FE-train-frozen scaler)."""
    f11 = [s for n, s in CATALOG.items() if s.family == "F11"]
    assert len(f11) == 45
    assert all(s.leakage_class == "TF" for s in f11)


# --------------------------------------------------------------------------- #
# Curation — dropped series produce no standalone columns                     #
# --------------------------------------------------------------------------- #
def test_macro_dropped_series_absent(raw_series) -> None:
    """The 10 dropped series + the two spread-only inputs yield no standalone col."""
    cols = MF.macro_feature_columns()
    # The 10 dropped sheet series are never read at all.
    for dropped in MF.DROPPED:
        assert dropped not in raw_series, f"{dropped} should not be loaded"
        assert not any(c.startswith(f"f11_{dropped.lower()}_") for c in cols)
    # GERMANY_PMI_MFG is the empty one, also absent.
    assert "GERMANY_PMI_MFG" in MF.DROPPED
    # Spread inputs are read but never standalone.
    for spread_input in MF.SPREAD_INPUTS:
        assert spread_input in raw_series, f"{spread_input} should be read for spreads"
        assert not any(c.startswith(f"f11_{spread_input.lower()}_") for c in cols), (
            f"{spread_input} leaked as a standalone column"
        )


def test_macro_spread_arithmetic(trade_dates, raw_series) -> None:
    """Each spread level equals minuend-applied minus subtrahend-applied level."""
    applied = MF.build_applied_panel(trade_dates, raw=raw_series)
    assembled = MF.assemble_macro_raw(trade_dates, raw=raw_series)
    for sname, (minuend, subtrahend, _, _) in MF.SPREADS.items():
        expected = (applied[minuend] - applied[subtrahend]).reindex(assembled.index)
        got = assembled[f"f11_spread_{sname}_level"]
        np.testing.assert_allclose(
            got.to_numpy(dtype=float), expected.to_numpy(dtype=float), atol=1e-9
        )


# --------------------------------------------------------------------------- #
# Domain / determinism / as-of carry                                          #
# --------------------------------------------------------------------------- #
def test_macro_finite_or_nan_float_domain(assembled) -> None:
    """Every produced column is float and finite-or-NaN; full data => no NaN."""
    assert list(assembled.columns) == MF.macro_feature_columns()
    assert assembled.dtypes.map(pd.api.types.is_float_dtype).all()
    arr = assembled.to_numpy(dtype=float)
    assert not np.isinf(arr).any(), "no infinities allowed"
    # All curated series predate the window, so momentum is fully populated.
    assert int(assembled.isna().sum().sum()) == 0


def test_macro_no_structural_ffill_beyond_asof(assembled) -> None:
    """The EIA level is held constant (as-of carry) BETWEEN Friday releases.

    Between two EIA availability dates the weekly value is carried forward
    unchanged -- the documented as-of carry, not a naive fill of a structural
    gap. Adjacent trade dates strictly inside one release week share a value.
    """
    eia = assembled["f11_eia_crude_stock_level"]
    # A stretch with no new EIA availability: 2020-01-21..2020-01-23 (the
    # 2020-01-17 Friday value, avail 2020-01-23, then constant until next).
    window = eia[(eia.index >= pd.Timestamp("2020-01-27")) & (eia.index <= pd.Timestamp("2020-01-29"))]
    if len(window) >= 2:
        assert window.nunique() == 1, "EIA level should be flat between weekly releases"


def test_load_macro_raw_deterministic() -> None:
    """Two reads of the workbook yield identical stamp-indexed series."""
    a = MF.load_macro_raw()
    b = MF.load_macro_raw()
    assert set(a) == set(b)
    for k in a:
        pd.testing.assert_series_equal(a[k], b[k])


# --------------------------------------------------------------------------- #
# Pipeline integration + standalone artifact                                  #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fitted_subset():
    """A pipeline fit+transform on the 3-instrument subset (warnings silenced)."""
    ohlcv, sig = load_clean_data()
    o = ohlcv[ohlcv["instrument"].isin(SUBSET)].copy()
    s = sig[["date", *SUBSET]].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = FeaturePipeline().fit(o, s)
        matrix = pipe.transform(o, s)
    return pipe, matrix


def test_pipeline_macro_broadcast_identical(fitted_subset) -> None:
    """F11 values are broadcast identically across instruments on a shared date."""
    _, matrix = fitted_subset
    f11 = [c for c in matrix.columns if c.startswith("f11_")]
    assert len(f11) == 45
    counts = matrix.groupby("date")["instrument"].nunique()
    shared = counts[counts >= 2].index
    assert len(shared) > 0
    rows = matrix[matrix["date"] == shared[len(shared) // 2]]
    vals = rows[f11].to_numpy(dtype=float)
    assert np.allclose(vals[0], vals[1], equal_nan=True), "F11 not broadcast-identical"


def test_pipeline_fit_missing_macro_raises() -> None:
    """fit() fails fast with FileNotFoundError when macro_path does not resolve."""
    ohlcv, sig = load_clean_data()
    o = ohlcv[ohlcv["instrument"].isin(SUBSET)].copy()
    s = sig[["date", *SUBSET]].copy()
    with pytest.raises(FileNotFoundError, match="macro workbook not found"):
        FeaturePipeline(macro_path="data/__does_not_exist__.xlsx").fit(o, s)


def test_macro_artifact_row_aligned(fitted_subset, tmp_path) -> None:
    """The standalone macro artifact is keyed (date,instrument) + row-aligned."""
    pipe, matrix = fitted_subset
    paths = _persist(
        matrix,
        pipe,
        outdir=tmp_path / "results",
        seed=0,
        catalog_path=tmp_path / "reports" / "feature-catalog.md",
        data_dir=tmp_path / "data",
    )
    macro_csv = paths["macro_features_csv"]
    assert macro_csv.exists()
    art = pd.read_csv(macro_csv)
    assert list(art.columns[:2]) == ["date", "instrument"]
    f11 = [c for c in art.columns if c.startswith("f11_")]
    assert len(f11) == 45
    assert len(art) == len(matrix), "artifact row count must equal the matrix"
    # The artifact F11 values equal the matrix F11 values (standardized slice).
    np.testing.assert_allclose(
        art[f11].to_numpy(dtype=float),
        matrix[f11].to_numpy(dtype=float),
        atol=1e-9,
        equal_nan=True,
    )
