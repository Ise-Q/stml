"""Per-class barrier specs wired through the pipeline at the emit boundary (Strategy B).

EX.5 ranked labelling schemes by downstream net Sharpe and recommended a per-asset-class
``(vol estimator, width, horizon)`` triple. These tests pin the WIRING that lets ``emit`` opt
into those per-class barriers while every other consumer (and the synthetic integration tests)
stays byte-identical:

- ``BarrierSpec`` validation + the ``DEFAULT_BARRIERS`` the analysis recommended,
- ``resolve_barrier``'s ``None``→shipped-global / set→per-class behaviour,
- ``build_instrument_panel`` backward-compat when ``barrier`` is ``None`` (the shipped path is
  byte-for-byte the pre-change ``triple_barrier_labels`` call),
- a non-shipped spec actually changes the labels and NEVER leaks the string ``barrier_type``
  column into the (all-numeric) feature matrix,
- ``strategy_weights``' per-class ``pt_sl`` override feeds the Kelly geometry.

CAVEAT (baked into the design): the EX.5 winners are XGBoost-selected and OFAT (each stage
varied one factor; the composite was never measured jointly). Only equity ``(2,1)`` survives the
xgb→lgbm swap, and NONE are validated under the shipped torch ``roster="default"``. These tests
verify the wiring is correct — NOT that the barriers improve the deliverable.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest
from stml.metamodel.scope import ASSET_CLASS_MAP
from test_pipeline import (  # sibling test module (pytest prepend-import; no tests/__init__.py)
    ENERGY,
    _synthetic_ohlcv,
    _synthetic_signals,
)

from alken_metamodel.barrier_vol import barrier_sigma
from alken_metamodel.emit import strategy_weights
from alken_metamodel.features import daily_barrier_sigma
from alken_metamodel.labelling_variants import vol_scaled_horizon
from alken_metamodel.pipeline import (
    DEFAULT_BARRIERS,
    BarrierSpec,
    PipelineConfig,
    build_instrument_panel,
    class_members,
    feature_columns,
    per_instrument_pt_sl,
    resolve_barrier,
    run_asset_class,
)
from alken_metamodel.sizing import position_weight
from alken_metamodel.triple_barrier import triple_barrier_labels

_LABEL_COLS = ["side", "t1", "ret", "bin", "weight"]


def _barrier_config(barriers=None, **kw) -> PipelineConfig:
    base = dict(
        modelling_end=pd.Timestamp("2021-12-31"),
        predict_start=pd.Timestamp("2022-01-01"),
        predict_end=pd.Timestamp("2022-03-31"),
        n_splits=3,
        use_regime=False,
    )
    base.update(kw)
    return PipelineConfig(barriers=barriers, **base)


# --- BarrierSpec validation -------------------------------------------------


def test_barrier_spec_defaults_are_shipped_fixed_horizon():
    spec = BarrierSpec()
    assert spec.vol_estimator == "shipped"
    assert spec.vol_param == 20
    assert spec.pt_sl == (1.0, 1.0)
    assert spec.max_holding == 10
    assert spec.vol_scaled is None


def test_barrier_spec_rejects_both_horizon_modes():
    # exactly one of {max_holding, vol_scaled} must be set — both is contradictory
    with pytest.raises(ValueError, match="exactly one"):
        BarrierSpec(max_holding=10, vol_scaled=(10, 2, 40))


def test_barrier_spec_rejects_neither_horizon_mode():
    with pytest.raises(ValueError, match="exactly one"):
        BarrierSpec(max_holding=None, vol_scaled=None)


def test_barrier_spec_rejects_unknown_estimator():
    with pytest.raises(ValueError, match="estimator"):
        BarrierSpec(vol_estimator="garch")


def test_barrier_spec_is_frozen():
    spec = BarrierSpec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.pt_sl = (2.0, 1.0)


# --- DEFAULT_BARRIERS reproduce the EX.5 per-class recommendations -----------


def test_default_barriers_match_ex5_recommendations():
    # equity: volA_rolling50 + widthB_2_1 + anchor h10 (robust to the xgb->lgbm swap)
    assert DEFAULT_BARRIERS["equity"] == BarrierSpec("rolling", 50, (2.0, 1.0), 10, None)
    # energy: volA_ewma20 + widthB_0p5_0p25 + horizC_volscaled (survives swap; 0.25 > 0.09 floor)
    assert DEFAULT_BARRIERS["energy"] == BarrierSpec("ewma", 20, (0.5, 0.25), None, (10, 2, 40))
    # METALS is deliberately omitted: its EX.5 width flips to a negative net Sharpe under lightgbm
    # (below the always-act floor) and the class shows no learnable edge — so it is NOT shipped and
    # falls back to the shipped global barrier via resolve_barrier.
    assert set(DEFAULT_BARRIERS) == {"equity", "energy"}


def test_metals_falls_back_to_shipped_under_default_barriers():
    cfg = PipelineConfig(barriers=DEFAULT_BARRIERS, pt_sl=(1.0, 1.0))
    # equity/energy resolve to their EX.5 picks; metals (omitted) reverts to the shipped global
    assert resolve_barrier(cfg, "equity") == DEFAULT_BARRIERS["equity"]
    assert resolve_barrier(cfg, "metals") == BarrierSpec("shipped", 20, (1.0, 1.0), 10, None)


# --- resolve_barrier --------------------------------------------------------


def test_resolve_barrier_none_returns_shipped_global():
    cfg = PipelineConfig(pt_sl=(1.0, 1.0), max_holding=10)  # barriers default None
    spec = resolve_barrier(cfg, "equity")
    assert spec == BarrierSpec("shipped", 20, (1.0, 1.0), 10, None)


def test_resolve_barrier_shipped_global_tracks_config_pt_sl_and_holding():
    cfg = PipelineConfig(pt_sl=(2.0, 0.5), max_holding=7)
    spec = resolve_barrier(cfg, "metals")
    assert spec.vol_estimator == "shipped"
    assert spec.pt_sl == (2.0, 0.5)
    assert spec.max_holding == 7


def test_resolve_barrier_picks_per_class():
    cfg = PipelineConfig(barriers=DEFAULT_BARRIERS)
    assert resolve_barrier(cfg, "equity") == DEFAULT_BARRIERS["equity"]
    assert resolve_barrier(cfg, "energy") == DEFAULT_BARRIERS["energy"]


def test_resolve_barrier_missing_class_falls_back_to_shipped():
    cfg = PipelineConfig(barriers={"equity": DEFAULT_BARRIERS["equity"]}, pt_sl=(1.0, 1.0))
    # energy is absent from this partial map -> shipped global, not a KeyError
    assert resolve_barrier(cfg, "energy") == BarrierSpec("shipped", 20, (1.0, 1.0), 10, None)


# --- build_instrument_panel backward-compat (shipped path unchanged) --------


def test_build_instrument_panel_barrier_none_equals_explicit_shipped():
    cfg = _barrier_config()
    ohlcv = _synthetic_ohlcv(["cl1s"])
    signals = _synthetic_signals(["cl1s"])
    default = build_instrument_panel(ohlcv, signals, "cl1s", cfg)
    explicit = build_instrument_panel(
        ohlcv, signals, "cl1s", cfg, barrier=BarrierSpec("shipped", 20, cfg.pt_sl, cfg.max_holding)
    )
    pd.testing.assert_frame_equal(default, explicit)


def test_build_instrument_panel_shipped_path_matches_core_labeller():
    # Regression guard: the barrier=None branch must still call the shipped daily_barrier_sigma +
    # triple_barrier_labels exactly — not silently reroute through the _ext labeller.
    cfg = _barrier_config()
    ohlcv = _synthetic_ohlcv(["cl1s"])
    signals = _synthetic_signals(["cl1s"])
    panel = build_instrument_panel(ohlcv, signals, "cl1s", cfg)

    inst = ohlcv[ohlcv["instrument"] == "cl1s"]
    signal = signals.set_index("date")["cl1s"].sort_index()
    signal.index = pd.DatetimeIndex(signal.index)
    from alken_metamodel.features import assemble_instrument_features

    feats = assemble_instrument_features(inst, signal)
    close = inst.set_index("date")["close"].sort_index().astype(float)
    close.index = pd.DatetimeIndex(close.index)
    labels = triple_barrier_labels(
        close, signal, daily_barrier_sigma(feats), pt_sl=cfg.pt_sl, max_holding=cfg.max_holding
    )
    common = panel.index.intersection(labels.index)
    assert len(common) > 50
    pd.testing.assert_series_equal(
        panel.loc[common, "bin"], labels.loc[common, "bin"], check_names=False
    )
    pd.testing.assert_series_equal(
        panel.loc[common, "t1"], labels.loc[common, "t1"], check_names=False
    )


# --- a non-shipped spec changes labels and never leaks barrier_type ---------


def test_non_shipped_barrier_changes_labels_without_barrier_type_leak():
    cfg = _barrier_config()
    ohlcv = _synthetic_ohlcv(["cl1s"])
    signals = _synthetic_signals(["cl1s"])
    shipped = build_instrument_panel(ohlcv, signals, "cl1s", cfg)
    # metals-style spec: genuine Garman-Klass vol, tight asymmetric stop, h=15
    gk = build_instrument_panel(
        ohlcv, signals, "cl1s", cfg, barrier=BarrierSpec("gk", 20, (0.5, 0.25), 15, None)
    )
    # the provenance column must be dropped — a string column would poison feature_columns()
    assert "barrier_type" not in gk.columns
    assert "barrier_type" not in feature_columns(gk)
    assert set(gk["bin"].dropna().unique()).issubset({0.0, 1.0})
    # the labels genuinely differ from the shipped path on the shared event dates
    common = shipped.index.intersection(gk.index)
    assert len(common) > 0
    assert not shipped.loc[common, "bin"].equals(gk.loc[common, "bin"])


def test_vol_scaled_barrier_path_is_clean_and_non_degenerate():
    # The vol-scaled (V2) path is the trickiest barrier wiring. Prove build_instrument_panel routes
    # it cleanly (valid numeric labels, no barrier_type leak) AND that the per-event horizon fed in
    # is genuinely non-constant — so the path is live, not collapsed to a single h. (The horizon's
    # longer-when-low-sigma / clipped / causal properties are pinned in test_labelling_variants.)
    cfg = _barrier_config()
    ohlcv = _synthetic_ohlcv(["cl1s"])
    signals = _synthetic_signals(["cl1s"])
    scaled = build_instrument_panel(
        ohlcv, signals, "cl1s", cfg, barrier=BarrierSpec("ewma", 20, (0.5, 0.25), None, (10, 2, 40))
    )
    assert "barrier_type" not in scaled.columns
    assert "barrier_type" not in feature_columns(scaled)
    assert set(scaled["bin"].dropna().unique()).issubset({0.0, 1.0})
    assert len(scaled) > 50

    # the same ewma σ the panel uses produces a genuinely per-event (non-constant) horizon in [2,40]
    inst = ohlcv[ohlcv["instrument"] == "cl1s"]
    signal = signals.set_index("date")["cl1s"].sort_index()
    signal.index = pd.DatetimeIndex(signal.index)
    sigma = barrier_sigma(inst, "ewma", 20)
    t_events = signal.index[signal.to_numpy() != 0]
    horizon = vol_scaled_horizon(sigma, t_events, 10, h_min=2, h_max=40)
    assert horizon.nunique() > 1
    assert horizon.min() >= 2 and horizon.max() <= 40


# --- strategy_weights per-class pt_sl override ------------------------------


def test_strategy_weights_pt_sl_override_changes_kelly_geometry():
    preds = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-04"]),
            "instrument": ["ho1s"],
            "prediction": [0.61],
            "side": [1],
            "ann_vol": [0.20],
        }
    )
    cfg = PipelineConfig(pt_sl=(1.0, 1.0))
    w_default = strategy_weights(preds, cfg)  # uses config.pt_sl = (1,1)
    w_override = strategy_weights(preds, cfg, pt_sl=(2.0, 1.0))  # per-class (2,1)
    assert w_default["weight"].iloc[0] != w_override["weight"].iloc[0]
    expected = position_weight(side=1, p=0.61, b=2.0, d=1.0, realised_vol=0.20, target_vol=0.25)
    assert w_override["weight"].iloc[0] == pytest.approx(expected)


# --- per_instrument_pt_sl: pooled-sizing bet-geometry map -------------------


def test_per_instrument_pt_sl_maps_each_class_to_its_barrier():
    # Pooled sizing (e.g. S6 resize over all 11 instruments) needs each instrument's class barrier
    # pt_sl so the Kelly geometry matches that class's labels.
    cfg = PipelineConfig(barriers=DEFAULT_BARRIERS, pt_sl=(1.0, 1.0))
    m = per_instrument_pt_sl(cfg)
    assert set(m) == set(ASSET_CLASS_MAP)  # all 11 instruments present
    for inst in class_members("equity"):
        assert m[inst] == (2.0, 1.0)  # EX.5 equity width
    for inst in class_members("energy"):
        assert m[inst] == (0.5, 0.25)  # EX.5 energy width
    for inst in class_members("metals"):
        assert m[inst] == (1.0, 1.0)  # omitted from DEFAULT_BARRIERS -> shipped global


def test_per_instrument_pt_sl_none_barriers_all_shipped():
    cfg = PipelineConfig(pt_sl=(1.0, 1.0))  # barriers default None
    m = per_instrument_pt_sl(cfg)
    assert set(m) == set(ASSET_CLASS_MAP)
    assert all(v == (1.0, 1.0) for v in m.values())


def test_per_instrument_pt_sl_shipped_tracks_config_pt_sl():
    # with no per-class overrides, the map echoes config.pt_sl for every instrument
    cfg = PipelineConfig(pt_sl=(2.0, 0.5))
    m = per_instrument_pt_sl(cfg)
    assert all(v == (2.0, 0.5) for v in m.values())


# --- integration: run_asset_class threads + carries the resolved barrier ----


def test_run_asset_class_carries_resolved_barrier_and_predicts():
    # The energy spec is the hardest path (ext labeller + vol-scaled horizon + tight stop). A clean
    # run proves the wiring produces an all-numeric feature matrix — i.e. no barrier_type leak.
    cfg = _barrier_config(barriers=DEFAULT_BARRIERS)
    ohlcv = _synthetic_ohlcv(ENERGY)
    signals = _synthetic_signals(ENERGY)
    res = run_asset_class(ohlcv, signals, "energy", cfg)
    assert res.barrier == DEFAULT_BARRIERS["energy"]
    preds = res.predictions
    assert ((preds["prediction"] >= 0) & (preds["prediction"] <= 1)).all()
    assert res.n_modelling > 0


def test_run_asset_class_barrier_none_carries_shipped_spec():
    cfg = _barrier_config()  # barriers None
    ohlcv = _synthetic_ohlcv(ENERGY)
    signals = _synthetic_signals(ENERGY)
    res = run_asset_class(ohlcv, signals, "energy", cfg)
    assert res.barrier == BarrierSpec("shipped", 20, cfg.pt_sl, cfg.max_holding, None)
