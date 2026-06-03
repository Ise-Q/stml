"""Unit tests for the EX.5 barrier-sweep harness helpers (RED-first).

The end-to-end run is exercised on real data in the verification step; here we pin the pure,
leakage-critical helpers: the config set (must include the GK arm, EWMA arms, baselines, and
the always-act economic floor), the common-event-set intersection, the barrier-INDEPENDENT
sizing (so only labels drive the ranking), and the anti-snooping confinement guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alken_metamodel.barrier_sweep import (
    assert_modelling_only,
    build_configs,
    build_meta,
    common_event_index,
    lag_placebo_sharpe,
    run_class,
)
from alken_metamodel.pipeline import PipelineConfig

# --- config set --------------------------------------------------------------


def test_build_configs_covers_arms_baselines_and_floor():
    cfgs = {c.name: c for c in build_configs()}
    families = {c.family for c in cfgs.values()}
    assert {"tb", "b1_fixed", "b2_trend", "always_act"} <= families
    # shipped (realized-vol-20) status-quo arm + a GENUINE GK arm + EWMA + rolling all present
    vol_arms = {(c.vol_estimator, c.vol_param) for c in cfgs.values() if c.family == "tb"}
    assert any(est == "shipped" for est, _ in vol_arms)  # the live pipeline default
    assert ("gk", 20) in vol_arms  # a genuine Garman-Klass arm, distinct from the shipped rv20
    assert any(est == "ewma" for est, _ in vol_arms)
    assert any(est == "rolling" for est, _ in vol_arms)
    # exactly one always-act economic floor
    floors = [c for c in cfgs.values() if c.family == "always_act"]
    assert len(floors) == 1


def test_build_configs_have_unique_names():
    names = [c.name for c in build_configs()]
    assert len(names) == len(set(names))


def test_build_configs_has_tight_stop_and_larger_horizon():
    """PDF (jay) finding: robust geometries cluster at tight stops (sl=0.25) and larger h."""
    cfgs = [c for c in build_configs() if c.family == "tb"]
    ptsls = {c.pt_sl for c in cfgs}
    assert any(sl == 0.25 for _, sl in ptsls)  # a tight-stop width arm exists
    horizons = {c.horizon for c in cfgs if c.vol_scaled is None}
    assert 15 in horizons  # the PDF's mid-horizon picks (h∈{15,20})


# --- lag-0/1/2 placebo-in-time diagnostic (oracle, never the ranker) --------


def test_lag_placebo_lag1_dominates_when_edge_is_forward():
    """A label whose edge sits on the first tradeable bar (lag-1) must beat lag-0 (placebo)."""
    rng = np.random.default_rng(0)
    n = 200
    u = rng.normal(0, 0.01, n)
    ev = np.arange(50, 150)
    u[ev] = -0.005 + rng.normal(0, 0.002, len(ev))       # lag-0 (signal-bar return): weak/negative
    u[ev + 1] = 0.010 + rng.normal(0, 0.002, len(ev))    # lag-1 (first tradeable bar): real edge
    close = pd.Series(100 * np.cumprod(1 + u), index=pd.bdate_range("2020-01-01", periods=n))
    dates = close.index[ev]
    mi = pd.MultiIndex.from_arrays([dates, ["x"] * len(ev)], names=["date", "instrument"])
    label_common = pd.DataFrame({"side": 1.0, "bin": 1.0}, index=mi)

    res = lag_placebo_sharpe(label_common, {"x": close})
    assert set(res) == {"adj_sharpe_lag0", "adj_sharpe_lag1", "adj_sharpe_lag2"}
    assert res["adj_sharpe_lag1"] > res["adj_sharpe_lag0"]  # the placebo-in-time signature
    assert res["adj_sharpe_lag1"] > 0


def test_lag_placebo_only_counts_acted_labels():
    """y=0 (skip) events contribute nothing; an all-skip label set yields NaN Sharpe."""
    close = pd.Series(
        100 * np.cumprod(1 + np.full(60, 0.001)), index=pd.bdate_range("2020-01-01", periods=60)
    )
    dates = close.index[20:40]
    mi = pd.MultiIndex.from_arrays([dates, ["x"] * 20], names=["date", "instrument"])
    label_common = pd.DataFrame({"side": 1.0, "bin": 0.0}, index=mi)  # every event skipped
    res = lag_placebo_sharpe(label_common, {"x": close})
    assert all(np.isnan(v) for v in res.values())


# --- common event set --------------------------------------------------------


def test_common_event_index_is_the_intersection():
    d = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"])
    a = pd.DataFrame(
        {"bin": [1, 0, 1]},
        index=pd.MultiIndex.from_tuples(
            [(d[0], "x"), (d[1], "x"), (d[2], "x")], names=["date", "instrument"]
        ),
    )
    b = pd.DataFrame(
        {"bin": [0, 1, 1]},
        index=pd.MultiIndex.from_tuples(
            [(d[1], "x"), (d[2], "x"), (d[3], "x")], names=["date", "instrument"]
        ),
    )
    common = common_event_index([a, b])
    assert list(common) == [(d[1], "x"), (d[2], "x")]


def test_common_event_index_sorted_by_date_then_instrument():
    d = pd.to_datetime(["2020-01-02", "2020-01-01"])
    a = pd.DataFrame(
        {"bin": [1, 1, 1]},
        index=pd.MultiIndex.from_tuples(
            [(d[0], "b"), (d[1], "a"), (d[1], "b")], names=["date", "instrument"]
        ),
    )
    common = common_event_index([a, a])
    assert list(common) == [(d[1], "a"), (d[1], "b"), (d[0], "b")]  # 01-01 a,b then 01-02 b


# --- barrier-independent sizing ---------------------------------------------


def test_build_meta_gates_on_threshold_and_signs_by_side():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02"])
    meta = build_meta(
        dates=idx,
        instruments=pd.Series(["x", "x"]),
        side=np.array([1.0, -1.0]),
        proba=np.array([0.6, 0.4]),  # second is below tau -> skip
        ann_vol=np.array([0.25, 0.25]),  # vol-target leverage = 0.25/0.25 = 1.0
        t1=pd.Series(idx),
        tau=0.5,
    )
    assert list(meta.columns) == ["date", "instrument", "weight", "t1"]
    assert meta["weight"].tolist() == [1.0, 0.0]


def test_build_meta_act_all_ignores_proba():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02"])
    meta = build_meta(
        dates=idx,
        instruments=pd.Series(["x", "x"]),
        side=np.array([1.0, -1.0]),
        proba=np.array([0.1, 0.1]),
        ann_vol=np.array([0.25, 0.25]),
        t1=pd.Series(idx),
        tau=0.5,
        act_all=True,
    )
    assert meta["weight"].tolist() == [1.0, -1.0]  # every signal acted


def test_build_meta_vol_targets_leverage():
    idx = pd.to_datetime(["2020-01-01"])
    meta = build_meta(
        dates=idx,
        instruments=pd.Series(["x"]),
        side=np.array([1.0]),
        proba=np.array([0.9]),
        ann_vol=np.array([0.5]),  # lev = 0.25/0.5 = 0.5
        t1=pd.Series(idx),
        tau=0.5,
        target_vol=0.25,
    )
    assert meta["weight"].iloc[0] == pytest.approx(0.5)


# --- anti-snooping confinement ----------------------------------------------


def test_assert_modelling_only_passes_within_window():
    idx = pd.to_datetime(["2021-06-01", "2021-12-31"])
    assert_modelling_only(idx, pd.Timestamp("2021-12-31"))  # no raise


def test_assert_modelling_only_rejects_oos_dates():
    idx = pd.to_datetime(["2021-12-31", "2022-01-03"])  # touches the locked OOS window
    with pytest.raises(AssertionError, match="modelling"):
        assert_modelling_only(idx, pd.Timestamp("2021-12-31"))


# --- integration: full run_class wiring on synthetic data -------------------


def _synth_ohlcv(instrument: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    close = np.exp(np.cumsum(rng.normal(0.0, 0.012, n)) + np.log(100.0))
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    intr = np.abs(rng.normal(0, 0.008, n))
    return pd.DataFrame(
        {
            "date": dates, "instrument": instrument,
            "open": open_,
            "high": np.maximum(open_, close) * np.exp(intr),
            "low": np.minimum(open_, close) * np.exp(-intr),
            "close": close,
            "volume": rng.integers(1_000, 50_000, n).astype(float),
            "open_interest": rng.integers(5_000, 80_000, n).astype(float),
        }
    )


def test_run_class_smoke_on_synthetic_equity():
    """Full harness wiring (labels → common set → design → purged-OOS → size → backtest) on
    synthetic data: catches PurgedKFold alignment, sizing, and confinement bugs cheaply."""
    from alken_metamodel.pipeline import class_members

    insts = class_members("equity")
    n = 800
    ohlcv = pd.concat(
        [_synth_ohlcv(inst, n, seed=i) for i, inst in enumerate(insts)], ignore_index=True
    )
    dates = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(99)
    cols = {"date": dates}
    in_window = (dates >= "2020-01-01") & (dates <= "2021-12-31")
    for inst in insts:
        s = np.zeros(n, dtype=int)
        s[in_window] = rng.choice([-1, 0, 1], size=int(in_window.sum()), p=[0.3, 0.4, 0.3])
        cols[inst] = s
    signals = pd.DataFrame(cols)

    keep = {"anchor_shipped_1_1_h10", "volA_gk20", "b1_fixed_h10", "always_act_floor"}
    subset = [c for c in build_configs() if c.name in keep]
    embargo = {inst: 3 for inst in insts}
    cfg = PipelineConfig()

    table = run_class("equity", ohlcv, signals, cfg, embargo, configs=subset, model_name="xgboost")

    assert len(table) == len(subset)
    assert "net_sharpe_1x" in table.columns
    assert table.attrs["common_n"] > 0
    assert (table["family"] == "always_act").sum() == 1  # the economic floor is present
    assert table["n"].min() > 0  # every config scored a non-empty common set
    # confinement: run_class asserts no event/t1/return date exceeds modelling_end (else it raises)
