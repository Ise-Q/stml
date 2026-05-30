"""End-to-end pipeline + emit tests (Stage 5).

Fast emit unit tests (determinism, schema, sizing) plus a synthetic Energy integration test
that exercises the whole chain (features -> labels -> pool -> purged-CV horse-race -> predict)
with ``use_regime=False`` for speed. The real-data run is a one-off CLI invocation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alken_metamodel.emit import (
    PREDICTION_COLUMNS,
    emit_predictions,
    select_window,
    strategy_weights,
)
from alken_metamodel.pipeline import (
    PipelineConfig,
    _roster_factory,
    build_instrument_panel,
    class_members,
    run_asset_class,
)

ENERGY = ["cl1s", "ho1s", "rb1s", "ng1s"]


def _synthetic_ohlcv(instruments, start="2020-01-01", periods=650, seed=0) -> pd.DataFrame:
    frames = []
    for k, inst in enumerate(instruments):
        rng = np.random.default_rng(seed + k)
        dates = pd.bdate_range(start, periods=periods)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, periods)))
        open_ = close * np.exp(rng.normal(0.0, 0.003, periods))
        intra = np.abs(rng.normal(0.0, 0.008, periods))
        high = np.maximum(open_, close) * np.exp(intra)
        low = np.minimum(open_, close) * np.exp(-intra)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "instrument": inst,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": rng.integers(1_000, 50_000, periods).astype(float),
                    "open_interest": rng.integers(5_000, 80_000, periods).astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _synthetic_signals(instruments, start="2020-01-01", periods=650, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {"date": pd.bdate_range(start, periods=periods)}
    for inst in instruments:
        data[inst] = rng.choice([-1, 0, 1], size=periods, p=[0.3, 0.4, 0.3])
    return pd.DataFrame(data)


def _fast_config() -> PipelineConfig:
    return PipelineConfig(
        modelling_end=pd.Timestamp("2021-12-31"),
        predict_start=pd.Timestamp("2022-01-01"),
        predict_end=pd.Timestamp("2022-03-31"),
        n_splits=3,
        use_regime=False,
    )


# --- emit unit tests (fast) -------------------------------------------------

def _toy_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-04", "2022-01-04", "2022-01-03"]),
            "instrument": ["ho1s", "cl1s", "cl1s"],
            "prediction": [0.61, 0.72, 0.55],
            "side": [1, -1, 1],
            "ann_vol": [0.20, 0.20, np.nan],
        }
    )


def test_emit_predictions_schema_and_sorted(tmp_path):
    out = emit_predictions(_toy_predictions(), tmp_path / "preds.csv")
    assert list(out.columns) == PREDICTION_COLUMNS
    back = pd.read_csv(tmp_path / "preds.csv")
    assert list(back.columns) == PREDICTION_COLUMNS
    # sorted by (date, instrument)
    assert list(back["date"]) == ["2022-01-03", "2022-01-04", "2022-01-04"]
    assert list(back["instrument"]) == ["cl1s", "cl1s", "ho1s"]


def test_emit_is_byte_identical_on_reemit(tmp_path):
    emit_predictions(_toy_predictions(), tmp_path / "a.csv")
    emit_predictions(_toy_predictions(), tmp_path / "b.csv")
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()


def test_select_window_filters_dates():
    df = pd.DataFrame({"date": pd.to_datetime(["2021-12-31", "2022-01-03", "2022-07-01"])})
    win = select_window(df, "2022-01-01", "2022-06-30")
    assert list(pd.to_datetime(win["date"]).dt.strftime("%Y-%m-%d")) == ["2022-01-03"]


def test_strategy_weights_known_values_and_sign():
    cfg = PipelineConfig(pt_sl=(1.0, 1.0))
    w = strategy_weights(_toy_predictions(), cfg)
    # p=0.7? no: p=0.61/0.72; fractional_kelly(0.72,1,1,kappa=.25,floor=.55)=.25*(0.72-0.28)=0.11
    # vol leverage 0.25/0.20=1.25 -> |w|=0.1375 for the confident rows
    long_w = w.loc[w["instrument"] == "ho1s", "weight"].iloc[0]   # side +1, p=0.61
    short_w = w.loc[(w["instrument"] == "cl1s") & (w["weight"] != 0), "weight"].iloc[0]
    assert long_w > 0
    assert short_w < 0
    # the NaN-vol row (cl1s 2022-01-03) is flat
    flat = w.loc[w["instrument"] == "cl1s", "weight"]
    assert (flat == 0.0).any()


# --- synthetic Energy integration (slower) ---------------------------------

def test_class_members_energy():
    assert class_members("energy") == ENERGY


def test_roster_factory_resolves_tree_linear_and_full():
    tree = _roster_factory(PipelineConfig(roster="tree_linear"))(seed=42)
    assert set(tree) == {"elasticnet_logistic", "xgboost", "lightgbm"}
    full = _roster_factory(PipelineConfig(roster="full"))(seed=42)
    assert set(full) == {
        "elasticnet_logistic",
        "xgboost",
        "lightgbm",
        "torch_mlp",
        "torch_vsn",
        "keras_vsn",
    }


def test_build_instrument_panel_has_features_and_labels():
    cfg = _fast_config()
    ohlcv = _synthetic_ohlcv(["cl1s"])
    signals = _synthetic_signals(["cl1s"])
    panel = build_instrument_panel(ohlcv, signals, "cl1s", cfg)
    assert {"bin", "t1", "weight", "side", "instrument", "date"}.issubset(panel.columns)
    assert "f2_vol_20" in panel.columns
    assert set(panel["bin"].dropna().unique()).issubset({0.0, 1.0})
    assert (panel["instrument"] == "cl1s").all()


def test_run_asset_class_predicts_config_window():
    cfg = _fast_config()
    ohlcv = _synthetic_ohlcv(ENERGY)
    signals = _synthetic_signals(ENERGY)
    res = run_asset_class(ohlcv, signals, "energy", cfg)
    preds = res.predictions
    assert list(preds.columns) == ["date", "instrument", "prediction", "side", "ann_vol"]
    d = pd.to_datetime(preds["date"])
    assert (d >= cfg.predict_start).all() and (d <= cfg.predict_end).all()
    assert ((preds["prediction"] >= 0) & (preds["prediction"] <= 1)).all()
    assert res.best_model in {"elasticnet_logistic", "xgboost", "lightgbm"}
    assert res.n_modelling > 0
    assert set(preds["instrument"]).issubset(set(ENERGY))
    # per-instrument OOS diagnostics: one row per instrument (§5 before-aggregate reporting)
    diag = res.diagnostics
    assert {"instrument", "n", "pos_rate", "auc", "precision"}.issubset(diag.columns)
    assert set(diag["instrument"]) == set(ENERGY)


def test_run_asset_class_is_deterministic():
    cfg = _fast_config()
    ohlcv = _synthetic_ohlcv(ENERGY)
    signals = _synthetic_signals(ENERGY)
    a = run_asset_class(ohlcv, signals, "energy", cfg).predictions
    b = run_asset_class(ohlcv, signals, "energy", cfg).predictions
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
