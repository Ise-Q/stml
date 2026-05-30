"""End-to-end pipeline + emit tests (Stage 5).

Fast emit unit tests (determinism, schema, sizing) plus a synthetic Energy integration test
that exercises the whole chain (features -> labels -> pool -> purged-CV horse-race -> predict)
with ``use_regime=False`` for speed. The real-data run is a one-off CLI invocation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alken_metamodel.cross_validation import CombinatorialPurgedCV, PurgedKFold
from alken_metamodel.emit import (
    PREDICTION_COLUMNS,
    coverage_caveat,
    emit_predictions,
    select_window,
    strategy_weights,
)
from alken_metamodel.pipeline import (
    PipelineConfig,
    _make_cv,
    _roster_factory,
    build_instrument_panel,
    class_members,
    nested_cpcv_select_and_evaluate,
    run_asset_class,
    select_model,
)

ENERGY = ["cl1s", "ho1s", "rb1s", "ng1s"]


def _toy_panel(n: int = 120, seed: int = 0, pos_rate: float | None = None):
    """Small (X, y, t1, sample_weight) modelling panel with overlapping label spans."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    X = pd.DataFrame(
        rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)], index=idx
    )
    if pos_rate is None:
        lin = X["f0"].to_numpy() * 0.8 + rng.normal(size=n) * 0.5
        y = (lin > np.median(lin)).astype(float)
    else:
        y = (rng.random(n) < pos_rate).astype(float)
    end = np.minimum(np.arange(n) + 3, n - 1)
    t1 = pd.Series(idx[end], index=idx)
    return X, y, t1, np.ones(n)


# --- S3.8: CPCV-as-selection + nested headline -----------------------------

def test_make_cv_honours_scheme():
    _, t1, _, _ = _toy_panel(40)
    purged = _make_cv(t1, PipelineConfig(cv_scheme="purged", n_splits=5))
    assert isinstance(purged, PurgedKFold)
    assert purged.get_n_splits() == 5
    cpcv = _make_cv(t1, PipelineConfig(cv_scheme="cpcv"))
    assert isinstance(cpcv, CombinatorialPurgedCV)
    assert cpcv.get_n_splits() == 15  # C(6,2)


def test_select_model_cpcv_survives_single_class_folds():
    X, y, t1, sw = _toy_panel(120, pos_rate=0.05)  # rare positives -> single-class CPCV folds
    best, scores = select_model(X, y, t1, sw, PipelineConfig(cv_scheme="cpcv", use_regime=False))
    assert best in scores
    assert set(scores) == {"elasticnet_logistic", "xgboost", "lightgbm"}


def test_nested_cpcv_select_and_evaluate_shape_and_determinism():
    X, y, t1, sw = _toy_panel(120, seed=3)
    cfg = PipelineConfig(roster="tree_linear", use_regime=False, seed=42)
    a = nested_cpcv_select_and_evaluate(X, y, t1, sw, cfg)
    assert {"outer_fold", "selected", "auc", "n_test"}.issubset(a.columns)
    assert len(a) == 15  # C(6,2) outer folds
    assert a["selected"].isin({"elasticnet_logistic", "xgboost", "lightgbm"}).all()
    b = nested_cpcv_select_and_evaluate(X, y, t1, sw, cfg)
    pd.testing.assert_frame_equal(a, b)  # deterministic


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


def test_coverage_caveat_flags_thin_instruments():
    dates = ["2022-01-03", "2022-01-04", "2022-01-03"] + ["2022-01-03"] * 40
    preds = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "instrument": ["ho1s", "ho1s", "gc1s"] + ["cl1s"] * 40,
            "prediction": [0.5] * 43,
        }
    )
    tab = coverage_caveat(preds, min_rows=30)
    counts = dict(zip(tab["instrument"], tab["n_oos_rows"], strict=True))
    thin = dict(zip(tab["instrument"], tab["thin"], strict=True))
    assert counts == {"cl1s": 40, "gc1s": 1, "ho1s": 2}
    assert thin["ho1s"] and thin["gc1s"]  # near-empty -> flagged unreliable
    assert not thin["cl1s"]
    assert list(tab["instrument"]) == ["cl1s", "gc1s", "ho1s"]  # sorted


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


def test_roster_factory_resolves_default():
    r = _roster_factory(PipelineConfig(roster="default"))(seed=42)
    assert set(r) == {"elasticnet_logistic", "xgboost", "lightgbm", "torch_mlp", "torch_vsn"}


def test_build_instrument_panel_with_macro_adds_macro_columns():
    cfg = PipelineConfig(
        modelling_end=pd.Timestamp("2021-12-31"),
        predict_end=pd.Timestamp("2022-03-31"),
        n_splits=3,
        use_regime=False,
        use_macro=True,
    )
    ohlcv = _synthetic_ohlcv(["cl1s"])
    signals = _synthetic_signals(["cl1s"])
    panel = build_instrument_panel(ohlcv, signals, "cl1s", cfg)
    macro_cols = [c for c in panel.columns if c.startswith("macro_")]
    assert len(macro_cols) >= 5  # the PIT-lagged block is joined into the feature panel
    assert "macro_vix_term_slope" in macro_cols


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
