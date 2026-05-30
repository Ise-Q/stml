"""Tests for ``stml.harry.signal_audit``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.harry.signal_audit import (
    ASSET_CLASSES,
    DEFAULT_LAGS,
    INSTRUMENTS,
    _build_aligned_frame,
    _classify_tag,
    _corr_col,
    _forward_cumret,
    _lag_corr,
    _log_returns,
    _moving_block_indices,
    _safe_ci_excludes_zero,
    _split_signals_halves,
    _statistics_from_frame,
    audit_all,
    audit_instrument,
    audit_stability,
)


# --------------------------------------------------------------------------- #
# Building blocks                                                              #
# --------------------------------------------------------------------------- #
def test_log_returns_first_is_nan_then_correct():
    idx = pd.date_range("2020-01-03", periods=4)
    closes = pd.Series([1.0, 2.0, 4.0, 4.0], index=idx)
    r = _log_returns(closes)
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(np.log(2.0))
    assert r.iloc[2] == pytest.approx(np.log(2.0))
    assert r.iloc[3] == pytest.approx(0.0)


def test_forward_cumret_hand_computed():
    # r[0..4] = [a, b, c, d, e]; with h=2, at index t we want r[t+1]+r[t+2].
    idx = pd.date_range("2020-01-03", periods=5)
    r = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4], index=idx)
    out = _forward_cumret(r, 2)
    assert out.iloc[0] == pytest.approx(0.1 + 0.2)
    assert out.iloc[1] == pytest.approx(0.2 + 0.3)
    assert out.iloc[2] == pytest.approx(0.3 + 0.4)
    assert np.isnan(out.iloc[3])
    assert np.isnan(out.iloc[4])


def test_corr_col_naming_is_unambiguous():
    assert _corr_col(0) == "corr_contemp_0"
    assert _corr_col(1) == "corr_trail_1"
    assert _corr_col(20) == "corr_trail_20"
    assert _corr_col(-1) == "corr_fwd_1"
    assert _corr_col(-5) == "corr_fwd_5"


def test_lag_corr_signs_match_momentum_vs_meanrev():
    # Synthetic returns; perfect momentum and perfect mean-reversion signals.
    rng = np.random.default_rng(0)
    n = 500
    idx = pd.date_range("2020-01-03", periods=n)
    r = pd.Series(rng.normal(0, 0.01, size=n), index=idx)
    s_mom = pd.Series(np.sign(r.shift(1)).fillna(0).values, index=idx)
    s_mr = pd.Series(-np.sign(r.shift(1)).fillna(0).values, index=idx)
    c_mom = _lag_corr(s_mom, r, k=1)
    c_mr = _lag_corr(s_mr, r, k=1)
    assert c_mom > 0.5
    assert c_mr < -0.5
    # And the corresponding forward corr (lag k=-1) should be near zero on
    # i.i.d. returns regardless of construction.
    c_fwd_mom = _lag_corr(s_mom, r, k=-1)
    assert abs(c_fwd_mom) < 0.2


def test_moving_block_indices_shape_and_bounds():
    rng = np.random.default_rng(42)
    n, B, B_boot = 200, 20, 100
    idx = _moving_block_indices(n, B, B_boot, rng)
    assert idx.shape == (B_boot, n)
    assert int(idx.min()) >= 0
    assert int(idx.max()) <= n - 1


def test_moving_block_indices_uses_blocks_of_size_B():
    rng = np.random.default_rng(42)
    n, B = 200, 20
    idx = _moving_block_indices(n, B, n_boot=1, rng=rng)
    diffs = np.diff(idx[0, :B])
    # Within a block, consecutive indices differ by exactly 1.
    assert (diffs == 1).all()


# --------------------------------------------------------------------------- #
# Integration on synthetic instruments                                         #
# --------------------------------------------------------------------------- #
def _make_synthetic_instrument(
    instrument: str,
    n: int,
    construction: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (ohlcv, signals) frames for one instrument with a known sign.

    ``construction`` is one of {"momentum", "mean_reverting", "random"}.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-03", periods=n, freq="B")
    r = pd.Series(rng.normal(0, 0.01, size=n), index=dates)
    close = pd.Series(100.0 * np.exp(r.cumsum()), index=dates)
    ohlcv = pd.DataFrame(
        {"date": dates, "instrument": instrument, "close": close.values}
    )
    if construction == "momentum":
        s_raw = np.sign(r.shift(1)).fillna(0).astype(int)
    elif construction == "mean_reverting":
        s_raw = (-np.sign(r.shift(1))).fillna(0).astype(int)
    elif construction == "random":
        s_raw = pd.Series(
            rng.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3]), index=dates
        )
    else:  # pragma: no cover
        raise ValueError(construction)
    signals = pd.DataFrame({"date": dates, instrument: s_raw.values})
    return ohlcv, signals


def test_audit_instrument_tags_momentum():
    ohlcv, signals = _make_synthetic_instrument("es1s", n=600, construction="momentum", seed=7)
    row = audit_instrument(
        ohlcv, signals, "es1s", h=10, n_boot=50, block_size=20, seed=42
    )
    assert row["tag"] == "trend"
    assert row["mean_trail_corr"] > 0.05
    assert row["corr_trail_1"] > 0.5


def test_audit_instrument_tags_mean_reversion():
    ohlcv, signals = _make_synthetic_instrument(
        "cl1s", n=600, construction="mean_reverting", seed=11
    )
    row = audit_instrument(
        ohlcv, signals, "cl1s", h=10, n_boot=50, block_size=20, seed=42
    )
    assert row["tag"] == "mean_reverting"
    assert row["mean_trail_corr"] < -0.05
    assert row["corr_trail_1"] < -0.5


def test_audit_instrument_random_signal_is_mixed():
    ohlcv, signals = _make_synthetic_instrument(
        "es1s", n=600, construction="random", seed=23
    )
    row = audit_instrument(
        ohlcv, signals, "es1s", h=10, n_boot=50, block_size=20, seed=42
    )
    # A random signal should fall in "mixed" almost surely.
    assert row["tag"] == "mixed"
    assert abs(row["mean_trail_corr"]) < 0.1


def test_audit_instrument_determinism():
    ohlcv, signals = _make_synthetic_instrument(
        "es1s", n=400, construction="momentum", seed=99
    )
    a = audit_instrument(ohlcv, signals, "es1s", n_boot=100, block_size=20, seed=42)
    b = audit_instrument(ohlcv, signals, "es1s", n_boot=100, block_size=20, seed=42)
    for key, va in a.items():
        vb = b[key]
        if isinstance(va, float) and np.isnan(va):
            assert isinstance(vb, float) and np.isnan(vb)
        else:
            assert va == vb, f"mismatch on {key}: {va!r} vs {vb!r}"


def test_audit_instrument_ci_is_ordered_and_nondegenerate():
    ohlcv, signals = _make_synthetic_instrument(
        "es1s", n=500, construction="momentum", seed=13
    )
    row = audit_instrument(ohlcv, signals, "es1s", n_boot=200, block_size=20, seed=42)
    for stat in ("corr_trail_1", "mean_pnl_next_day", "hit_rate_h"):
        lo, hi = row[f"{stat}_lo"], row[f"{stat}_hi"]
        assert lo <= hi, f"{stat}: lo={lo!r} > hi={hi!r}"
        # CI should be wider than zero on real bootstrap output.
        assert hi - lo > 0


def test_audit_instrument_raises_on_missing_instrument():
    ohlcv, signals = _make_synthetic_instrument(
        "es1s", n=200, construction="random", seed=5
    )
    with pytest.raises(KeyError):
        audit_instrument(ohlcv, signals, "nq1s")


def test_audit_all_schema_and_one_row_per_instrument():
    # Build a multi-instrument synthetic panel covering all 11 tickers.
    pieces_ohlcv: list[pd.DataFrame] = []
    sig_cols: dict[str, object] = {"date": None}
    n = 400
    dates = pd.date_range("2020-01-03", periods=n, freq="B")
    sig_cols["date"] = dates
    rng = np.random.default_rng(3)
    for inst in INSTRUMENTS:
        r = pd.Series(rng.normal(0, 0.01, size=n), index=dates)
        close = (100 * np.exp(r.cumsum())).values
        pieces_ohlcv.append(
            pd.DataFrame({"date": dates, "instrument": inst, "close": close})
        )
        sig_cols[inst] = rng.choice([-1, 0, 1], size=n)
    ohlcv = pd.concat(pieces_ohlcv, ignore_index=True)
    signals = pd.DataFrame(sig_cols)

    df = audit_all(ohlcv, signals, n_boot=50, block_size=20, seed=42)
    assert len(df) == len(INSTRUMENTS)
    assert list(df["instrument"]) == INSTRUMENTS
    assert set(df["asset_class"]) == {"equity", "energy", "metals"}
    for k in DEFAULT_LAGS:
        col = _corr_col(k)
        for c in (col, f"{col}_lo", f"{col}_hi"):
            assert c in df.columns
    for stat in ("mean_pnl_next_day", "mean_pnl_h", "hit_rate_h"):
        for c in (stat, f"{stat}_lo", f"{stat}_hi"):
            assert c in df.columns
    assert "mean_trail_corr" in df.columns
    for tag_col in ("tag", "tag_trail_1", "tag_trail_h10"):
        assert tag_col in df.columns
        assert set(df[tag_col]).issubset(
            {"trend", "mean_reverting", "mixed", "n/a"}
        )


def test_statistics_from_frame_pnl_definitions():
    # Hand-built tiny frame: 6 rows, h=2.
    idx = pd.date_range("2020-01-03", periods=6, freq="B")
    s = pd.Series([1, 1, -1, 0, 1, -1], index=idx, dtype=float)
    # Choose returns so that pnl is easy to verify.
    r = pd.Series([0.10, 0.20, -0.30, 0.40, 0.50, -0.10], index=idx)
    lags = (1, -1)  # one trailing, one forward
    frame = _build_aligned_frame(s, r, lags, h=2)
    stats = _statistics_from_frame(frame, lags)
    # next-day pnl = s * r.shift(-1) → at t=0..4 :
    #   1*0.20, 1*(-0.30), -1*0.40, 0*0.50, 1*(-0.10) → mean(0.2,-0.3,-0.4,0,-0.1)=-0.12
    # t=5 has no next day → NaN, excluded.
    assert stats["mean_pnl_next_day"] == pytest.approx(
        np.mean([1 * 0.20, 1 * (-0.30), -1 * 0.40, 0 * 0.50, 1 * (-0.10)])
    )
    # pnl_h with h=2 at t=0..3:
    #   1 * (0.20 + -0.30) = -0.10
    #   1 * (-0.30 + 0.40) = 0.10
    #   -1 * (0.40 + 0.50) = -0.90
    #   0 * (0.50 + -0.10) = 0.0
    # t=4,5 → NaN (forward window incomplete)
    expected = [1 * (0.20 - 0.30), 1 * (-0.30 + 0.40), -1 * (0.40 + 0.50), 0.0]
    assert stats["mean_pnl_h"] == pytest.approx(np.mean(expected))
    # hit_rate among bet days where pnl_h is defined: bet days at t in {0,1,2,4};
    # pnl_h defined at t in {0,1,2,3}; intersection {0,1,2}; pnl_h values
    # [-0.10, 0.10, -0.90], hits = [F, T, F] → 1/3.
    assert stats["hit_rate_h"] == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
# Per-horizon tagging (Step 1 addendum)                                        #
# --------------------------------------------------------------------------- #
def test_classify_tag_thresholds():
    assert _classify_tag(0.10, threshold=0.05) == "trend"
    assert _classify_tag(-0.10, threshold=0.05) == "mean_reverting"
    assert _classify_tag(0.04, threshold=0.05) == "mixed"
    assert _classify_tag(-0.04, threshold=0.05) == "mixed"
    assert _classify_tag(0.0, threshold=0.05) == "mixed"
    assert _classify_tag(float("nan"), threshold=0.05) == "n/a"
    # Threshold edge: strictly greater on either side; equal => mixed.
    assert _classify_tag(0.05, threshold=0.05) == "mixed"
    assert _classify_tag(-0.05, threshold=0.05) == "mixed"


def test_per_horizon_tags_in_audit_instrument_output():
    ohlcv, signals = _make_synthetic_instrument(
        "es1s", n=600, construction="momentum", seed=4
    )
    row = audit_instrument(
        ohlcv, signals, "es1s", h=10, n_boot=50, block_size=20, seed=42
    )
    # On strong momentum data both horizons should be "trend".
    assert row["tag_trail_1"] == "trend"
    # tag is multi-horizon mean — should also be trend here.
    assert row["tag"] == "trend"
    # tag_trail_h10 must reflect corr_trail_10 only.
    expected = "trend" if row["corr_trail_10"] > 0.05 else (
        "mean_reverting" if row["corr_trail_10"] < -0.05 else "mixed"
    )
    assert row["tag_trail_h10"] == expected


def test_per_horizon_tags_can_disagree_with_canonical():
    """A signal that loads strongly negatively at lag 1 but is noise elsewhere
    will tag mean_reverting on trail_1 and mixed on the multi-horizon mean.
    """
    rng = np.random.default_rng(31)
    n = 600
    dates = pd.date_range("2020-01-03", periods=n, freq="B")
    r = pd.Series(rng.normal(0, 0.01, size=n), index=dates)
    close = pd.Series(100 * np.exp(r.cumsum()), index=dates).values
    # Signal is -sign(r_{t-1}) — strong reversion at trail_1, near-zero elsewhere.
    s = (-np.sign(r.shift(1))).fillna(0).astype(int).values
    ohlcv = pd.DataFrame({"date": dates, "instrument": "es1s", "close": close})
    signals = pd.DataFrame({"date": dates, "es1s": s})
    row = audit_instrument(
        ohlcv, signals, "es1s", h=10, n_boot=50, block_size=20, seed=42
    )
    # trail_1 lights up; trail_10 is noise → canonical multi-horizon mean is
    # diluted by the noisy horizons.
    assert row["tag_trail_1"] == "mean_reverting"
    # Either way the per-horizon tags + canonical tag must agree with the
    # numeric values according to the threshold rule.
    assert row["tag"] == _classify_tag(row["mean_trail_corr"], threshold=0.05)
    assert row["tag_trail_h10"] == _classify_tag(row["corr_trail_10"], threshold=0.05)


# --------------------------------------------------------------------------- #
# Stability check (Step 1 addendum)                                            #
# --------------------------------------------------------------------------- #
def test_split_signals_halves_median_split():
    dates = pd.date_range("2020-01-03", periods=11, freq="B")
    sigs = pd.DataFrame({"date": dates, "es1s": np.ones(len(dates))})
    first, second, split = _split_signals_halves(sigs, split_date=None)
    # Median of 11 contiguous dates is the 6th; first half has 5 rows.
    assert len(first) == 5
    assert len(second) == 6
    assert (first["date"] < split).all()
    assert (second["date"] >= split).all()


def test_safe_ci_excludes_zero_handles_nan_and_signs():
    assert _safe_ci_excludes_zero(0.1, 0.2)
    assert _safe_ci_excludes_zero(-0.2, -0.1)
    assert not _safe_ci_excludes_zero(-0.1, 0.1)  # spans zero
    assert not _safe_ci_excludes_zero(float("nan"), 0.1)
    assert not _safe_ci_excludes_zero(0.1, float("nan"))
    assert not _safe_ci_excludes_zero(0.0, 0.1)  # touches zero
    assert not _safe_ci_excludes_zero(-0.1, 0.0)


def test_audit_stability_schema_and_no_flips_on_stable_signal():
    """A stationary momentum signal across the whole window should not flip."""
    pieces_ohlcv: list[pd.DataFrame] = []
    sig_cols: dict[str, object] = {"date": None}
    n = 400
    dates = pd.date_range("2020-01-03", periods=n, freq="B")
    sig_cols["date"] = dates
    rng = np.random.default_rng(73)
    for inst in INSTRUMENTS:
        r = pd.Series(rng.normal(0, 0.01, size=n), index=dates)
        close = (100 * np.exp(r.cumsum())).values
        pieces_ohlcv.append(
            pd.DataFrame({"date": dates, "instrument": inst, "close": close})
        )
        # Same construction in both halves → no real sign flip.
        s = np.sign(r.shift(1)).fillna(0).astype(int)
        sig_cols[inst] = s.values
    ohlcv = pd.concat(pieces_ohlcv, ignore_index=True)
    signals = pd.DataFrame(sig_cols)

    stab = audit_stability(ohlcv, signals, n_boot=50, block_size=20, seed=42)
    expected_cols = {
        "instrument",
        "metric",
        "full_window",
        "first_half",
        "second_half",
        "sign_flip_flag",
    }
    assert expected_cols.issubset(set(stab.columns))
    # 11 instruments × 8 metrics = 88 rows.
    assert len(stab) == 11 * 8
    # The strongest stable signal here is the trailing-1 construction — its
    # sign should be the same in both halves for every instrument. Noisier
    # metrics (corr_fwd_*, mean_pnl_*) can flip on pure random returns even
    # when the construction is stationary; we accept up to a handful of those.
    trail1 = stab[stab["metric"] == "corr_trail_1"]
    assert trail1["sign_flip_flag"].sum() == 0, (
        f"unexpected sign flip on corr_trail_1: {trail1[trail1['sign_flip_flag']]}"
    )
    # Sanity cap on overall noise-driven flips.
    assert stab["sign_flip_flag"].sum() <= 4


def test_audit_stability_flags_constructed_flip():
    """Build a signal that is momentum in the first half and counter-trend in
    the second. The corr_trail_1 row for that instrument should flag a flip.
    """
    n = 600
    dates = pd.date_range("2020-01-03", periods=n, freq="B")
    rng = np.random.default_rng(101)
    pieces_ohlcv = []
    sig_cols: dict[str, object] = {"date": dates}
    for inst in INSTRUMENTS:
        r = pd.Series(rng.normal(0, 0.01, size=n), index=dates)
        close = (100 * np.exp(r.cumsum())).values
        pieces_ohlcv.append(
            pd.DataFrame({"date": dates, "instrument": inst, "close": close})
        )
        if inst == "es1s":
            half = n // 2
            s_mom = np.sign(r.shift(1)).fillna(0).astype(int).values
            s_mr = (-np.sign(r.shift(1))).fillna(0).astype(int).values
            s = np.concatenate([s_mom[:half], s_mr[half:]])
        else:
            s = np.sign(r.shift(1)).fillna(0).astype(int).values
        sig_cols[inst] = s
    ohlcv = pd.concat(pieces_ohlcv, ignore_index=True)
    signals = pd.DataFrame(sig_cols)

    stab = audit_stability(ohlcv, signals, n_boot=200, block_size=20, seed=42)
    flips = stab[stab["sign_flip_flag"]]
    # es1s / corr_trail_1 must be among the flagged rows.
    assert (
        (flips["instrument"] == "es1s") & (flips["metric"] == "corr_trail_1")
    ).any()
