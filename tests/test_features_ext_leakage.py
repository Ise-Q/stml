"""Leakage / causality tests for the extended E-class families
(:mod:`stml.metamodel.features_ext`) folded in from the Harry / Sreeram branches.

Every extended engineered family must be **truncation-invariant**: truncating
the inputs at a cut date ``T`` and recomputing reproduces the IDENTICAL value on
every date ``< T``. For the F15 conditional-risk bootstrap this is the real
test — its per-row RNG is seeded from the positional index, so truncating the
FUTURE (the tail) must leave past rows byte-stable.

Tests run on REAL data (:func:`stml.io.load_clean_data`) across >= 3 instruments
including ng1s, on a recent slice with small windows so the expensive bootstrap
/ wavelet loops stay fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.io import load_clean_data
from stml.metamodel import features_ext as FX

SAMPLE_INSTRUMENTS = ["cl1s", "ng1s", "es1s"]

# Recent slice length + small windows keep the bootstrap / wavelet loops fast.
_SLICE = 600
_CUT_FROM_END = 120


def _ext_families(oi: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build every extended price-family block with small (fast) windows."""
    return {
        "f2rs": FX.f2_rogers_satchell(oi),
        "f7e": FX.f7_microstructure_ext(oi),
        "f12": FX.f12_path_structure(oi, hurst_window=100),
        "f13": FX.f13_wavelet_energy(oi, window=64),
        "f15": FX.f15_conditional_risk(oi, window=120, n_sims=40),
    }


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.DataFrame]:
    ohlcv, sig = load_clean_data()
    return ohlcv, sig


def _inst_ohlcv_recent(ohlcv: pd.DataFrame, inst: str) -> pd.DataFrame:
    oi = ohlcv[ohlcv["instrument"] == inst].sort_values("date")
    return oi.tail(_SLICE).copy()


def _frames_match(a: pd.DataFrame, b: pd.DataFrame, tol: float = 1e-9) -> None:
    assert list(a.columns) == list(b.columns)
    assert (a.isna() == b.isna()).all().all(), "NaN pattern differs after truncation"
    diff = (a - b).abs().to_numpy(dtype=float)
    finite = diff[np.isfinite(diff)]
    if finite.size:
        assert finite.max() <= tol, f"max abs diff {finite.max():.3e} exceeds {tol:.0e}"


@pytest.mark.parametrize("inst", SAMPLE_INSTRUMENTS)
@pytest.mark.parametrize("fam", ["f2rs", "f7e", "f12", "f13", "f15"])
def test_ext_price_family_truncation_invariant(
    inst: str, fam: str, data: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    ohlcv, _ = data
    oi_full = _inst_ohlcv_recent(ohlcv, inst)
    dates = pd.DatetimeIndex(sorted(oi_full["date"].unique()))
    cut = dates[-_CUT_FROM_END]
    oi_trunc = oi_full[oi_full["date"] <= cut]

    full = _ext_families(oi_full)[fam]
    trunc = _ext_families(oi_trunc)[fam]
    common = trunc.index[trunc.index < cut]
    assert len(common) > 0, f"{fam}/{inst}: nothing to compare before the cut"
    _frames_match(full.reindex(common), trunc.reindex(common))


@pytest.mark.parametrize("inst", SAMPLE_INSTRUMENTS)
def test_f5_ext_truncation_invariant(
    inst: str, data: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """F5 signal-trajectory adds (entropy / flip-rate) on the released signal."""
    _, sig = data
    s_full = pd.Series(sig.set_index("date")[inst]).sort_index()
    cut = s_full.index[400]
    s_trunc = s_full[s_full.index <= cut]
    full = FX.f5_signal_trajectory(s_full)
    trunc = FX.f5_signal_trajectory(s_trunc)
    common = trunc.index[trunc.index < cut]
    assert len(common) > 0
    _frames_match(full.reindex(common), trunc.reindex(common))


@pytest.mark.parametrize("inst", SAMPLE_INSTRUMENTS)
def test_ext_families_return_finite_or_nan_floats(
    inst: str, data: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    ohlcv, _ = data
    oi = _inst_ohlcv_recent(ohlcv, inst)
    for name, block in _ext_families(oi).items():
        assert isinstance(block.index, pd.DatetimeIndex), f"{name}/{inst} not date-indexed"
        assert block.index.is_monotonic_increasing, f"{name}/{inst} unsorted"
        assert block.dtypes.map(pd.api.types.is_float_dtype).all(), f"{name}/{inst} non-float"
        arr = block.to_numpy(dtype=float)
        assert not np.isinf(arr).any(), f"{name}/{inst} produced an inf"


def test_z_twin_columns_all_have_a_base() -> None:
    """Every z-twin base is a real produced column (guards against drift between
    Z_TWIN_COLUMNS and the catalog)."""
    from stml.metamodel.catalog import CATALOG

    for base in FX.Z_TWIN_COLUMNS:
        assert base in CATALOG, f"z-twin base {base!r} missing from CATALOG"
        assert f"z_{base}" in CATALOG, f"z-twin z_{base} missing from CATALOG"


def test_expanding_zscore_is_causal() -> None:
    """The expanding z-score at t uses only data[:t+1] (truncation-invariant)."""
    idx = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(0)
    s = pd.Series(rng.standard_normal(200).cumsum(), index=idx)
    full = FX.expanding_zscore(s, min_periods=20)
    cut = idx[150]
    trunc = FX.expanding_zscore(s[s.index <= cut], min_periods=20)
    common = trunc.index[trunc.index < cut]
    a, b = full.reindex(common), trunc.reindex(common)
    assert (a.isna() == b.isna()).all()
    fin = (a - b).abs().dropna()
    assert fin.empty or fin.max() <= 1e-12
