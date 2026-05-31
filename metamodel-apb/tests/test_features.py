"""Fold-safety + schema tests for the per-instrument feature adapter (Stage 2, RED-first).

The load-bearing invariant is RIGHT-EDGE TRUNCATION-INVARIANCE: for a causal feature,
its value at date t must be identical whether it was computed on ``data[:t+1]`` or on the
full series ``data[:T]`` (López de Prado's E-class definition; stml CLAUDE.md). This is the
single property that proves the whole assembled stack (stml F1/F2/F5/F6/F7/F8/F10 +
F12/F13/F15 + z-twins + the backward trend feature) contains no forward-looking leak.

The contract the adapter enforces (verified against the stml source): every assemble_* is
stateless, so we compute on each instrument's FULL fixed-start history and RIGHT-SLICE the
output — anchoring both series at index 0 keeps f15's positional-seed bootstrap and the
expanding z-twins invariant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alken_metamodel.features import (
    SQRT_252,
    TREND_TVAL_CAP,
    assemble_instrument_features,
    attach_instrument,
    backward_trend_feature,
    daily_barrier_sigma,
    filter_signal_days,
    right_slice,
)


def _synthetic_ohlcv(n: int = 300, seed: int = 0, drift: float = 0.0) -> pd.DataFrame:
    """Valid long-format OHLCV for ONE instrument (cols: date, instrument, OHLCV, OI)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    log_close = np.cumsum(rng.normal(drift, 0.01, n)) + np.log(100.0)
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
            "open_interest": rng.integers(5_000, 80_000, n).astype(float),
        }
    )


def _signal(ohlcv: pd.DataFrame, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    vals = rng.choice([-1, 0, 1], size=len(ohlcv), p=[0.3, 0.4, 0.3])
    return pd.Series(vals, index=pd.DatetimeIndex(ohlcv["date"]), dtype=int)


# --- the crux: right-edge truncation invariance -----------------------------

def test_right_edge_truncation_invariance():
    """feature[t] is identical on data[:t+1] and on data[:T] for EVERY column."""
    ohlcv = _synthetic_ohlcv(n=300, seed=7)
    sig = _signal(ohlcv, seed=2)

    full = assemble_instrument_features(ohlcv, sig)
    t_iloc = 290  # well past the 252-bar f13/f15 warm-up
    t = full.index[t_iloc]

    truncated_ohlcv = ohlcv.iloc[: t_iloc + 1].copy()
    truncated_sig = sig.iloc[: t_iloc + 1]
    trunc = assemble_instrument_features(truncated_ohlcv, truncated_sig)

    assert t in trunc.index and t in full.index
    # compare the row at t across the common columns; NaN must align with NaN
    cols = [c for c in full.columns if c in trunc.columns]
    assert len(cols) > 60  # the full stack actually assembled
    a = full.loc[t, cols].to_numpy(dtype=float)
    b = trunc.loc[t, cols].to_numpy(dtype=float)
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_no_column_is_entirely_future_dependent():
    """Sanity: at least the ext families (f13/f15) produced finite values at t (so the
    invariance test above is actually exercising them, not comparing NaN==NaN)."""
    ohlcv = _synthetic_ohlcv(n=300, seed=7)
    sig = _signal(ohlcv, seed=2)
    full = assemble_instrument_features(ohlcv, sig)
    row = full.iloc[290]
    for col in ["f13_mra_energy_d1", "f15_prob_timeout", "f2_garman_klass_20"]:
        assert col in full.columns
        assert np.isfinite(row[col]), f"{col} should be finite at a post-warmup row"


# --- schema -----------------------------------------------------------------

def test_expected_feature_families_present():
    ohlcv = _synthetic_ohlcv(n=300, seed=3)
    sig = _signal(ohlcv)
    feats = assemble_instrument_features(ohlcv, sig)
    expected = [
        "f1_rsi_14", "f2_vol_20", "f2_garman_klass_20", "f5_signal",
        "f6_macd_12_26", "f7_oi_change", "f8_dow_sin", "f10_hl_range",
        "f2_rogers_satchell_20", "f12_hurst_100", "f13_mra_energy_d1",
        "f15_prob_timeout", "z_f2_vol_20", "trend_tval_back", "trend_sign_back",
    ]
    missing = [c for c in expected if c not in feats.columns]
    assert not missing, f"missing expected columns: {missing}"
    # per-instrument DatetimeIndex, all numeric
    assert isinstance(feats.index, pd.DatetimeIndex)
    assert all(pd.api.types.is_numeric_dtype(feats[c]) for c in feats.columns)


# --- F16 concept-drift wiring (S1.8-b) --------------------------------------

def _feat_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A finite synthetic feature frame (no warm-up NaN) to exercise F16 directly."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    cols = [f"x{i}" for i in range(6)]
    return pd.DataFrame(rng.normal(size=(n, 6)), index=idx, columns=cols)


def test_f16_regime_alignment_is_truncation_invariant():
    """F16's score at t is identical on data[:t+1] and data[:T] (right-edge causal)."""
    from stml.metamodel.drift_features import regime_alignment_score

    feats = _feat_frame(400, seed=1)
    train_end = feats.index[120]
    full = regime_alignment_score(feats, train_end=train_end, window=63, refit_every=21, seed=42)
    t_iloc = 360
    t = feats.index[t_iloc]
    trunc = regime_alignment_score(
        feats.iloc[: t_iloc + 1], train_end=train_end, window=63, refit_every=21, seed=42
    )
    assert np.isfinite(full.loc[t]) and np.isfinite(trunc.loc[t])  # actually scored, not NaN==NaN
    np.testing.assert_allclose(float(full.loc[t]), float(trunc.loc[t]), rtol=1e-12, atol=1e-12)
    scored = full.dropna()
    assert ((scored >= 0.0) & (scored <= 1.0)).all()  # it is a probability


def test_assemble_wires_f16_only_when_drift_enabled():
    """F16 is added (exactly one column) iff a drift_train_end is supplied; off by default."""
    ohlcv = _synthetic_ohlcv(n=320, seed=7)
    sig = _signal(ohlcv, seed=2)
    base = assemble_instrument_features(ohlcv, sig)
    drifted = assemble_instrument_features(
        ohlcv, sig, drift_train_end=pd.Timestamp(ohlcv["date"].iloc[200])
    )
    assert "f16_regime_alignment_score" not in base.columns
    assert "f16_regime_alignment_score" in drifted.columns
    assert set(base.columns).issubset(set(drifted.columns))
    assert len(drifted.columns) == len(base.columns) + 1  # re-locked count: +1


def test_no_metamodel_module_reads_frozen_parquet():
    """Leakage guard: no metamodel module consumes the frozen feature_matrix.parquet."""
    import pathlib

    import alken_metamodel

    src = pathlib.Path(alken_metamodel.__file__).parent
    offenders = [
        p.relative_to(src).as_posix()
        for p in src.rglob("*.py")
        if "_vendor" not in p.parts and "read_parquet" in p.read_text()
    ]
    assert not offenders, f"modules read parquet (frozen-matrix leak risk): {offenders}"


# --- backward trend feature -------------------------------------------------

def test_backward_trend_sign_and_cap():
    n = 200
    dates = pd.bdate_range("2019-01-01", periods=n)
    close = pd.Series(np.exp(np.linspace(np.log(100), np.log(160), n)), index=dates)  # up-trend
    out = backward_trend_feature(close, span=(5, 25))
    assert set(["trend_tval_back", "trend_sign_back", "trend_window_back"]).issubset(out.columns)
    late = out["trend_sign_back"].dropna().iloc[-1]
    assert late == 1.0  # a clean up-trend is a +1 backward trend
    # the cosmetic global-variance cap is overridden by a fixed constant
    assert out["trend_tval_back"].abs().max() <= TREND_TVAL_CAP + 1e-9
    # warm-up rows (< max window of backward history) are NaN, not fabricated
    assert out["trend_sign_back"].iloc[:25].isna().all()


def test_segment_tval_matches_vendored_tvallinr():
    """The fast closed-form slope t-value equals the vendored statsmodels tValLinR."""
    from alken_metamodel._vendor.trend_scanning import tValLinR
    from alken_metamodel.features import _segment_tval

    rng = np.random.default_rng(11)
    for _ in range(25):
        y = rng.normal(0.0, 1.0, int(rng.integers(5, 40)))
        t_ref, _ = tValLinR(y)
        np.testing.assert_allclose(_segment_tval(np.asarray(y, float)), t_ref, rtol=1e-9, atol=1e-9)


def test_backward_trend_is_truncation_invariant():
    """The trend tVal at t must not depend on data after t (the global-cap leak fixed)."""
    n = 180
    dates = pd.bdate_range("2019-01-01", periods=n)
    rng = np.random.default_rng(5)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=dates)
    full = backward_trend_feature(close, span=(5, 25))
    t_iloc = 150
    t = dates[t_iloc]
    trunc = backward_trend_feature(close.iloc[: t_iloc + 1], span=(5, 25))
    for col in ["trend_tval_back", "trend_sign_back", "trend_window_back"]:
        np.testing.assert_allclose(
            float(full.loc[t, col]), float(trunc.loc[t, col]), rtol=1e-9, atol=1e-12
        )


# --- fold-safety + LI helpers ----------------------------------------------

def test_right_slice_keeps_left_anchor():
    ohlcv = _synthetic_ohlcv(n=120, seed=9)
    sig = _signal(ohlcv)
    feats = assemble_instrument_features(ohlcv, sig)
    end = feats.index[80]
    sliced = right_slice(feats, end)
    assert sliced.index.max() == end
    assert sliced.index.min() == feats.index.min()  # anchored at inception
    # values on the retained dates are untouched (same as full)
    pd.testing.assert_frame_equal(sliced, feats.loc[:end])


def test_daily_barrier_sigma_is_deannualised():
    ohlcv = _synthetic_ohlcv(n=300, seed=4)
    sig = _signal(ohlcv)
    feats = assemble_instrument_features(ohlcv, sig)
    sigma = daily_barrier_sigma(feats)
    ref = feats["f2_vol_20"] / SQRT_252
    pd.testing.assert_series_equal(sigma, ref, check_names=False)
    assert abs(SQRT_252 - np.sqrt(252.0)) < 1e-12


def test_filter_signal_days_keeps_only_nonzero():
    ohlcv = _synthetic_ohlcv(n=120, seed=6)
    sig = _signal(ohlcv, seed=3)
    feats = assemble_instrument_features(ohlcv, sig)
    kept = filter_signal_days(feats, sig)
    expected = sig.index[sig != 0].intersection(feats.index)
    assert list(kept.index) == list(expected)
    assert len(kept) < len(feats)


def test_attach_instrument_adds_key_without_dropping_columns():
    ohlcv = _synthetic_ohlcv(n=80, seed=8)
    sig = _signal(ohlcv)
    feats = assemble_instrument_features(ohlcv, sig)
    keyed = attach_instrument(feats, "tst1s")
    assert (keyed["instrument"] == "tst1s").all()
    assert set(feats.columns).issubset(set(keyed.columns))
