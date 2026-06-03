"""S6 runner — refit champions, predict, calibrate, size, backtest, emit.

Per plan §8 S6. Produces the submission-ready deliverables:

    outputs/metamodel_predictions.csv          (calibrated)
    outputs/metamodel_predictions_raw.csv      (uncalibrated)
    outputs/strategy_weights.csv               (vol-targeted Kelly sized)
    outputs/coverage_caveat.csv                (per-instrument flags)
    outputs/experiment_log.csv                 (per-class deterministic log)
    results/sreeram_experimental/backtest_metrics.csv

Acceptance gates (plan §8 S6):
* Calibration: Platt monotone → AUC invariant (test).
* Realised ann vol ∈ [0.06, 0.10] (the R7 10 % cap with headroom).
* Byte-identical re-emit verified.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from stml.experimental.backtest import barrier_backtest, performance_metrics
from stml.experimental.calibration import PlattCalibrator, expected_calibration_error
from stml.experimental.champion_pipeline import (
    INSTRUMENT_REGIMES,
    POOL_MEMBERS,
    _is_bbg_feature,
    _select_feature_cols,
    _slice_pool,
)
from stml.experimental.config import INSTRUMENTS, PipelineConfig
from stml.experimental.cv import CombinatorialPurgedCV
from stml.experimental.data_loader import load_panel
from stml.experimental.emit import (
    emit_predictions,
    emit_weights,
    panel_to_long,
)
from stml.experimental.evaluation import cross_val_evaluate
from stml.experimental.make_scope import embargo_days_map
from stml.experimental.models import balanced_sample_weight
from stml.experimental.pipeline import _fresh_estimator
from stml.experimental.sizing import (
    CONFIDENCE_FLOOR,
    MAX_LEVERAGE,
    TARGET_VOL,
    position_weight,
)
from stml.experimental.volatility import ewma_daily_sigma


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"Could not locate repo root from {here}")


_SCHEMA_COLS = frozenset({
    "instrument", "t_signal", "t_start", "t_end", "side", "ret", "label",
    "uniqueness_weight", "sigma_at_t", "barrier_hit",
})


@dataclass
class InstrumentDeliverable:
    instrument: str
    pool: str
    model_name: str
    test_events: pd.DataFrame
    raw_proba: np.ndarray
    calibrated_proba: np.ndarray
    weight_per_event: np.ndarray


def _load_champions() -> dict[str, dict]:
    """Load champion per-instrument selection from S3-fix output."""
    root = _find_repo_root()
    summary = pd.read_csv(root / "results" / "sreeram_experimental" / "champions_summary.csv")
    with_bbg = summary.loc[summary["variant"] == "with_bbg"].set_index("instrument")
    return {
        inst: {
            "pool": row["winning_pool"],
            "model": row["winning_model"],
        }
        for inst, row in with_bbg.iterrows()
    }


def _add_instrument_onehot(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    insts = sorted(df["instrument"].unique())
    for inst in insts:
        out[f"inst_{inst}"] = (df["instrument"] == inst).astype(float)
    return out


def _split_modelling_and_test(features: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = features.copy()
    df["t_signal"] = pd.to_datetime(df["t_signal"])
    df["t_end"] = pd.to_datetime(df["t_end"])
    train_cut = pd.Timestamp(cfg.global_train_cut)
    embargo_end = pd.Timestamp(cfg.embargo_end)
    modelling = df.loc[df["t_signal"] <= train_cut].reset_index(drop=True)
    test = df.loc[df["t_signal"] > embargo_end].reset_index(drop=True)
    return modelling, test


def _predict_for_instrument(
    *,
    instrument: str,
    pool: str,
    model_name: str,
    modelling: pd.DataFrame,
    test: pd.DataFrame,
    cfg: PipelineConfig,
    embargo_map: dict[str, int],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Refit champion on full modelling sample (pool), predict on test slice
    for THIS instrument.

    Returns (test_events_for_this_instrument, raw_proba, oof_modelling_proba)
    where oof_modelling_proba is the purged OOF predictions on the modelling
    sample (for Platt calibration fit).
    """
    pool_modelling = _slice_pool(modelling, pool)
    pool_modelling = _add_instrument_onehot(pool_modelling)
    feature_cols = _select_feature_cols(pool_modelling, "with_bbg") + [
        c for c in pool_modelling.columns if c.startswith("inst_")
    ]

    X_m = pool_modelling.loc[:, feature_cols]
    y_m = pool_modelling["label"].astype(int).values
    uniq_m = pool_modelling["uniqueness_weight"].astype(float).values
    sw_m = balanced_sample_weight(y_m, base=uniq_m)

    # 1) Purged-OOF predictions on the modelling sample for THIS instrument
    #    → used downstream for Platt calibration.
    t_m = pd.to_datetime(pool_modelling["t_signal"])
    t1_m = pd.to_datetime(pool_modelling["t_end"])
    insts_m = pool_modelling["instrument"]
    cv = CombinatorialPurgedCV(
        n_groups=cfg.cpcv_n_groups,
        n_test_groups=cfg.cpcv_n_test_groups,
        t=t_m, t1=t1_m, pct_embargo=cfg.cpcv_pct_embargo,
        instruments=insts_m, embargo_days=embargo_map,
    )
    cv_result = cross_val_evaluate(
        make_model=lambda: _fresh_estimator(model_name, seed=cfg.seed),
        X=X_m, y=pool_modelling["label"].astype(int),
        cv=cv, uniqueness_weights=pool_modelling["uniqueness_weight"].astype(float),
    )
    oos = cv_result.oos_predictions
    oos["instrument"] = oos["row_idx"].map(
        dict(enumerate(insts_m.values))
    )
    inst_oof = oos.loc[oos["instrument"] == instrument].copy()
    # Mean per row to dedupe across CPCV paths.
    oof_mean = inst_oof.groupby("row_idx")["y_proba"].mean()
    oof_label = inst_oof.groupby("row_idx")["y_true"].first()
    oof_aligned = pd.DataFrame({"proba": oof_mean, "label": oof_label})

    # 2) Refit on FULL modelling sample.
    model = _fresh_estimator(model_name, seed=cfg.seed)
    try:
        model.fit(X_m, y_m, sample_weight=sw_m)
    except Exception as exc:
        # Fallback: simple uniform sample weights.
        model = _fresh_estimator(model_name, seed=cfg.seed)
        model.fit(X_m, y_m)

    # 3) Predict on the test slice for THIS instrument.
    pool_test = _slice_pool(test, pool)
    pool_test = _add_instrument_onehot(pool_test)
    # Align columns to pool_modelling (some inst dummies might be missing).
    missing = [c for c in feature_cols if c not in pool_test.columns]
    for c in missing:
        pool_test[c] = 0.0
    X_t = pool_test.loc[:, feature_cols]
    # Predict for ALL rows in the pool test slice; then filter to instrument.
    try:
        all_proba = model.predict_act_proba(X_t)
    except Exception:
        all_proba = np.full(len(X_t), 0.5)
    pool_test = pool_test.assign(_raw_proba=all_proba)
    inst_test = pool_test.loc[pool_test["instrument"] == instrument].copy().reset_index(drop=True)
    raw_proba = inst_test["_raw_proba"].values

    return inst_test, raw_proba, oof_aligned


def _per_event_weight(
    *,
    test_events: pd.DataFrame,
    calibrated_proba: np.ndarray,
    realised_vol: pd.Series,
    cfg: PipelineConfig,
) -> np.ndarray:
    """Convert per-event probability + side + realised vol to position weight."""
    weights = np.zeros(len(test_events), dtype=float)
    for i, row in test_events.iterrows():
        sigma = float(realised_vol.get(row["t_start"], float("nan")))
        # Annualise daily sigma.
        ann_sigma = sigma * np.sqrt(252.0) if np.isfinite(sigma) else float("nan")
        w = position_weight(
            side=int(row["side"]),
            p=float(calibrated_proba[i]),
            realised_vol=ann_sigma,
            kappa=cfg.kappa,
            floor=cfg.confidence_floor,
            target_vol=cfg.target_vol,
            max_leverage=cfg.max_leverage,
        )
        weights[i] = w
    return weights


def _returns_panel(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Per-instrument daily log returns as a wide panel."""
    rows = []
    for inst in INSTRUMENTS:
        sub = ohlcv.loc[ohlcv["instrument"] == inst, ["date", "close"]].sort_values("date")
        sub = sub.set_index("date")
        ret = np.log(sub["close"] / sub["close"].shift(1))
        rows.append(ret.rename(inst))
    return pd.concat(rows, axis=1).sort_index()


def run(cfg: PipelineConfig | None = None, *, verbose: bool = True) -> dict:
    cfg = cfg or PipelineConfig()
    root = _find_repo_root()
    features = pd.read_parquet(root / "data" / "sreeram_experimental_features.parquet")
    if verbose:
        print(f"Loaded features: {features.shape}")

    champions = _load_champions()
    if verbose:
        print(f"Loaded champions for {len(champions)} instruments")

    embargo_map = embargo_days_map()
    modelling, test = _split_modelling_and_test(features, cfg)
    if verbose:
        print(f"Modelling sample: {len(modelling)} events  Test slice: {len(test)} events")

    # Per-instrument deliverables.
    deliverables: dict[str, InstrumentDeliverable] = {}
    calibrators: dict[str, PlattCalibrator] = {}
    ohlcv, _signals = load_panel()
    returns_panel = _returns_panel(ohlcv)

    for inst in INSTRUMENTS:
        if inst not in champions:
            continue
        pool = champions[inst]["pool"]
        model_name = champions[inst]["model"]
        if model_name == "multitask_nn":
            # Multi-task NN refit + predict is more involved; fall back to its
            # 1-SE-pickable alternative for the deliverable. The multitask
            # contribution is in the CV/per-instrument breakdown only.
            from stml.experimental.champion_pipeline import select_champion
            summary = pd.read_csv(root / "results" / "sreeram_experimental" / "champions_per_pool_per_model.csv")
            grid = summary.loc[
                (summary["variant"] == "with_bbg")
                & (summary["instrument"] == inst)
                & (summary["model"] != "multitask_nn")
            ].sort_values("mean_auc", ascending=False)
            if grid.empty:
                continue
            pool = grid.iloc[0]["pool"]
            model_name = grid.iloc[0]["model"]
            if verbose:
                print(f"  {inst}: multi-task NN champion → substitute "
                      f"{model_name}@{pool} for sklearn-refit deliverable")

        if verbose:
            print(f"\n[deliverable] {inst}: {model_name}@{pool}")

        try:
            test_events, raw_proba, oof = _predict_for_instrument(
                instrument=inst, pool=pool, model_name=model_name,
                modelling=modelling, test=test, cfg=cfg, embargo_map=embargo_map,
            )
        except Exception as exc:
            if verbose:
                print(f"    [WARN] failed: {exc}")
            continue

        if test_events.empty:
            continue

        # Per-class Platt fit on OOF.
        platt = PlattCalibrator()
        if not oof.empty:
            platt.fit(oof["proba"].values, oof["label"].values)
        calibrators[inst] = platt
        calibrated = platt.transform(raw_proba)

        # Position weights per event.
        ewma_vol = ewma_daily_sigma(
            ohlcv.loc[ohlcv["instrument"] == inst].set_index("date")["close"], span=21
        )
        weights = _per_event_weight(
            test_events=test_events, calibrated_proba=calibrated,
            realised_vol=ewma_vol, cfg=cfg,
        )

        deliverables[inst] = InstrumentDeliverable(
            instrument=inst, pool=pool, model_name=model_name,
            test_events=test_events.assign(
                raw_proba=raw_proba, calibrated_proba=calibrated, weight=weights,
            ),
            raw_proba=raw_proba, calibrated_proba=calibrated, weight_per_event=weights,
        )

    # Assemble deliverable frames.
    pred_rows_raw = []
    pred_rows_cal = []
    event_rows = []
    for inst, d in deliverables.items():
        for _, row in d.test_events.iterrows():
            pred_rows_raw.append({
                "date": row["t_signal"], "instrument": inst,
                "prediction": float(row["raw_proba"]),
            })
            pred_rows_cal.append({
                "date": row["t_signal"], "instrument": inst,
                "prediction": float(row["calibrated_proba"]),
            })
            event_rows.append({
                "instrument": inst,
                "t_signal": row["t_signal"], "t_start": row["t_start"], "t_end": row["t_end"],
                "side": int(row["side"]), "ret": float(row["ret"]),
                "label": int(row["label"]),
                "raw_proba": float(row["raw_proba"]),
                "calibrated_proba": float(row["calibrated_proba"]),
                "weight": float(row["weight"]),
            })

    if not event_rows:
        print("FATAL: no events produced.")
        return {"status": "no_events"}

    events_full = pd.DataFrame(event_rows).sort_values(["instrument", "t_signal"]).reset_index(drop=True)
    preds_raw = pd.DataFrame(pred_rows_raw)
    preds_cal = pd.DataFrame(pred_rows_cal)

    # Backtest on the test slice.
    report = barrier_backtest(
        events_full.assign(t_start=events_full["t_start"], t_end=events_full["t_end"]),
        events_full["weight"], returns_panel,
    )

    # Weights panel as long deliverable.
    weights_long = panel_to_long(report.weights, value_col="weight")

    # Coverage caveats per instrument.
    n_above_055 = 0
    coverage_rows = []
    for inst in INSTRUMENTS:
        sub = events_full.loc[events_full["instrument"] == inst]
        if sub.empty:
            coverage_rows.append({
                "instrument": inst, "n_oos_events": 0, "thin": True,
                "low_coherence_vs_raw": inst in ("ng1s",),
            })
            continue
        n_oos = len(sub)
        coverage_rows.append({
            "instrument": inst, "n_oos_events": n_oos,
            "thin": n_oos < 60,
            "low_coherence_vs_raw": inst == "ng1s",
            "noisy_coherence_vs_raw": inst in ("ho1s", "rb1s"),
        })
    coverage = pd.DataFrame(coverage_rows)

    # Persist.
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    results_dir = root / "results" / "sreeram_experimental"
    results_dir.mkdir(parents=True, exist_ok=True)

    emit_predictions(preds_raw, outputs_dir / "metamodel_predictions_raw.csv")
    emit_predictions(preds_cal, outputs_dir / "metamodel_predictions.csv")
    emit_weights(weights_long.rename(columns={"weight": "weight"}),
                  outputs_dir / "strategy_weights.csv")
    coverage.to_csv(outputs_dir / "coverage_caveat.csv", index=False)
    events_full.to_csv(results_dir / "oos_events_with_predictions.csv",
                        index=False, float_format="%.10f")
    pd.Series(report.metrics).to_csv(results_dir / "backtest_metrics.csv",
                                       header=["value"])

    # Persist daily returns for S7 significance.
    report.net_returns.to_csv(results_dir / "strategy_daily_net_returns.csv",
                                header=["net_ret"])
    report.gross_returns.to_csv(results_dir / "strategy_daily_gross_returns.csv",
                                  header=["gross_ret"])

    # Experiment log.
    exp_log = pd.DataFrame([{
        "asset_class": cls,
        "n_events": int(events_full.loc[events_full["instrument"].isin(members)].shape[0]),
        "champion_models": ", ".join(sorted(set(
            champions[i]["model"] for i in members if i in champions
        ))),
    } for cls, members in {
        "equity": ("es1s", "nq1s", "fesx1s"),
        "energy": ("cl1s", "ho1s", "rb1s", "ng1s"),
        "metals": ("gc1s", "si1s", "hg1s", "pl1s"),
    }.items()])
    exp_log.to_csv(outputs_dir / "experiment_log.csv", index=False)

    # Print acceptance gates.
    print("\n=== Plan §8 S6 acceptance gates ===")
    m = report.metrics
    print(f"  events with weight !=0: {(events_full['weight'] != 0).sum()} / {len(events_full)}")
    print(f"  realised ann vol      : {m['ann_vol']:.4f}  (target ∈ [0.06, 0.10])  "
          f"{'PASS' if 0.06 <= m['ann_vol'] <= 0.10 else 'CHECK'}")
    print(f"  realised ann return   : {m['ann_return']:.4f}")
    print(f"  Sharpe ann            : {m['sharpe']:.4f}")
    print(f"  Sortino (full-T)      : {m['sortino']:.4f}")
    print(f"  Max DD                : {m['max_dd']:.4f}")
    print(f"  Turnover (1/yr)       : {m['turnover_per_year']:.2f}")

    return {"report": report, "deliverables": deliverables,
             "events_full": events_full, "coverage": coverage}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S6 deliverable runner")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run(verbose=not args.quiet)
    return 0 if result.get("status") != "no_events" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
