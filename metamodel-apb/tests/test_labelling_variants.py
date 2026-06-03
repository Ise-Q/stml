"""Hand-built-path tests for the EX.5 labelling variants + baselines (RED-first).

Variants under test (all share the ``side, t1, ret, bin, weight`` schema of the shipped
``triple_barrier_labels`` so the sweep can swap labellers freely):
- ``triple_barrier_labels_ext`` — superset adding ``barrier_type`` (pt/sl/vertical) and a
  per-event vertical horizon (V2). Must equal the shipped labeller on the default config.
- ``vol_scaled_horizon`` — V2 per-event horizon: longer when σ is low, clipped + causal.
- ``fixed_horizon_labels`` — B1: ``bin = 1{ side·r_{t,t+h} > τ }`` (L1 §2.1).
- ``trend_scan_labels`` — B2: act iff the (forward) trend sign agrees with the primary side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alken_metamodel.labelling_variants import (
    fixed_horizon_labels,
    trend_scan_labels,
    triple_barrier_labels_ext,
    vol_scaled_horizon,
)
from alken_metamodel.triple_barrier import triple_barrier_labels

DAYS = pd.to_datetime([f"2020-01-{d:02d}" for d in range(1, 8)])  # d0..d6


def _label_one_ext(prices, side, *, k=1.0, trgt=0.02, max_holding=5):
    close = pd.Series(prices, index=DAYS[: len(prices)], dtype=float)
    signal = pd.Series(0, index=close.index, dtype=int)
    signal.iloc[0] = side
    target = pd.Series(trgt, index=close.index, dtype=float)
    return triple_barrier_labels_ext(close, signal, target, pt_sl=(k, k), max_holding=max_holding)


# --- regression: ext == shipped core on the default config ------------------


def test_ext_matches_core_on_default_config():
    rng = np.random.default_rng(11)
    n = 60
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    signal = pd.Series(rng.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3]), index=idx)
    target = pd.Series(0.015, index=idx)

    core = triple_barrier_labels(close, signal, target, pt_sl=(1.0, 1.0), max_holding=10)
    ext = triple_barrier_labels_ext(close, signal, target, pt_sl=(1.0, 1.0), max_holding=10)

    assert list(core.index) == list(ext.index)
    for col in ("side", "t1", "ret", "bin", "weight"):
        pd.testing.assert_series_equal(ext[col], core[col], check_names=False)
    assert "barrier_type" in ext.columns  # the one added column


# --- barrier_type provenance on constructed paths ---------------------------


def test_barrier_type_pt_first():
    out = _label_one_ext([100, 101, 102.5, 101, 100, 100], side=1)  # +2.5% crosses +2% PT at d2
    assert out.loc[DAYS[0], "barrier_type"] == "pt"
    assert out.loc[DAYS[0], "bin"] == 1.0


def test_barrier_type_sl_first():
    out = _label_one_ext([100, 99.5, 97.5, 100, 101, 100], side=1)  # -2.5% crosses -2% SL at d2
    assert out.loc[DAYS[0], "barrier_type"] == "sl"
    assert out.loc[DAYS[0], "bin"] == 0.0


def test_barrier_type_timeout_is_vertical():
    out = _label_one_ext([100, 100.5, 101, 100.8, 101.2, 101], side=1, max_holding=5)
    assert out.loc[DAYS[0], "barrier_type"] == "vertical"
    assert out.loc[DAYS[0], "t1"] == DAYS[5]


def test_short_side_pt_when_price_falls():
    out = _label_one_ext([100, 99, 97.5, 99, 100, 100], side=-1)  # side-adjusted +2.5% -> PT
    assert out.loc[DAYS[0], "barrier_type"] == "pt"
    assert out.loc[DAYS[0], "bin"] == 1.0


# --- V2 vol-scaled horizon --------------------------------------------------


def test_vol_scaled_horizon_longer_when_sigma_low():
    idx = pd.bdate_range("2020-01-01", periods=40)
    sigma = pd.Series(0.02, index=idx)  # baseline so σ̄ ≈ 0.02
    sigma.iloc[30] = 0.01  # low-σ event -> longer horizon
    sigma.iloc[35] = 0.04  # high-σ event -> shorter horizon
    t_events = idx[[30, 35]]
    h = vol_scaled_horizon(sigma, t_events, h0=10, h_min=2, h_max=40)
    assert h.loc[idx[30]] > h.loc[idx[35]]
    assert (h >= 2).all() and (h <= 40).all()
    assert h.dtype.kind in "iu"  # integer horizons


def test_vol_scaled_horizon_is_clipped():
    idx = pd.bdate_range("2020-01-01", periods=40)
    sigma = pd.Series(0.02, index=idx)
    sigma.iloc[30] = 1e-6  # would explode h -> must clip to h_max
    h = vol_scaled_horizon(sigma, idx[[30]], h0=10, h_min=2, h_max=20)
    assert h.loc[idx[30]] == 20


def test_vol_scaled_horizon_causal_uses_only_past_sigma():
    """σ̄ is a trailing mean: changing a FUTURE σ must not alter an event's horizon."""
    idx = pd.bdate_range("2020-01-01", periods=40)
    base = pd.Series(0.02, index=idx)
    base.iloc[25] = 0.015
    h0_ = vol_scaled_horizon(base, idx[[25]], h0=10, h_min=2, h_max=40)
    bumped = base.copy()
    bumped.iloc[30] = 0.10  # a later shock
    h1 = vol_scaled_horizon(bumped, idx[[25]], h0=10, h_min=2, h_max=40)
    assert h0_.loc[idx[25]] == h1.loc[idx[25]]


# --- B1 fixed-time-horizon labeller -----------------------------------------


def test_fixed_horizon_sign_and_schema():
    close = pd.Series([100, 101, 103, 102, 104, 105], index=DAYS[:6], dtype=float)
    signal = pd.Series([1, 0, 0, 0, 0, 0], index=DAYS[:6])
    out = fixed_horizon_labels(close, signal, h=2, tau=0.0)
    assert set(["side", "t1", "ret", "bin", "weight"]).issubset(out.columns)
    assert out.loc[DAYS[0], "t1"] == DAYS[2]  # t1 = t + h
    assert out.loc[DAYS[0], "ret"] == pytest.approx(0.03)  # 103/100 - 1, long
    assert out.loc[DAYS[0], "bin"] == 1.0


def test_fixed_horizon_threshold_gate():
    close = pd.Series([100, 100.5, 100.6], index=DAYS[:3], dtype=float)
    signal = pd.Series([1, 0, 0], index=DAYS[:3])
    # +0.6% move; τ=1% -> below threshold -> skip
    out = fixed_horizon_labels(close, signal, h=2, tau=0.01)
    assert out.loc[DAYS[0], "bin"] == 0.0


def test_fixed_horizon_drops_last_h_bars():
    close = pd.Series([100, 101, 102, 103, 104], index=DAYS[:5], dtype=float)
    signal = pd.Series([1, 0, 0, 1, 1], index=DAYS[:5])  # events d0,d3,d4
    out = fixed_horizon_labels(close, signal, h=2, tau=0.0)
    assert DAYS[0] in out.index  # has a full 2-bar horizon
    assert DAYS[3] not in out.index  # d3+2 = d5 is out of range -> dropped
    assert DAYS[4] not in out.index


def test_fixed_horizon_short_side_sign():
    close = pd.Series([100, 99, 97], index=DAYS[:3], dtype=float)
    signal = pd.Series([-1, 0, 0], index=DAYS[:3])
    out = fixed_horizon_labels(close, signal, h=2, tau=0.0)
    assert out.loc[DAYS[0], "side"] == -1.0
    assert out.loc[DAYS[0], "ret"] == pytest.approx(0.03)  # short gains as price falls
    assert out.loc[DAYS[0], "bin"] == 1.0


# --- B2 trend-scanning baseline ---------------------------------------------


def test_trend_scan_acts_when_trend_agrees_with_side():
    idx = pd.bdate_range("2020-01-01", periods=40)
    close = pd.Series(100 * np.exp(np.cumsum(np.full(40, 0.01))), index=idx)  # steady uptrend
    long_sig = pd.Series(0, index=idx)
    long_sig.iloc[5] = 1  # long into an uptrend -> agreement -> act
    out = trend_scan_labels(close, long_sig, span=(5, 20))
    assert out.loc[idx[5], "bin"] == 1.0

    short_sig = pd.Series(0, index=idx)
    short_sig.iloc[5] = -1  # short into an uptrend -> disagreement -> skip
    out2 = trend_scan_labels(close, short_sig, span=(5, 20))
    assert out2.loc[idx[5], "bin"] == 0.0


def test_trend_scan_schema():
    idx = pd.bdate_range("2020-01-01", periods=40)
    close = pd.Series(100 * np.exp(np.cumsum(np.full(40, 0.01))), index=idx)
    sig = pd.Series(0, index=idx)
    sig.iloc[5] = 1
    out = trend_scan_labels(close, sig, span=(5, 20))
    assert set(["side", "t1", "ret", "bin", "weight"]).issubset(out.columns)
