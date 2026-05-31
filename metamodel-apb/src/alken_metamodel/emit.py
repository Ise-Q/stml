"""Deterministic CSV emitters, §6 strategy sizing, and the CLI entry point (Stage 5).

Two deliverables (brief, Deliverables):
- ``metamodel_predictions.csv`` — ``date,instrument,prediction`` (P(act) in [0,1]).
- ``strategy_weights.csv`` — ``date,instrument,weight`` (the +10 bonus track).

Determinism: rows are sorted by ``(date, instrument)``, the column order is pinned, dates are
ISO strings and floats use a fixed format, so a re-emit is byte-identical (the grader re-runs
on the hidden Jul–Dec 2022 half). The prediction window is config-driven, never hardcoded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .experiment_log import log_run
from .pipeline import PipelineConfig, run_asset_class  # importing the package pins env first
from .seeding import set_seeds
from .sizing import TARGET_VOL, position_weight

PREDICTION_COLUMNS = ["date", "instrument", "prediction"]
WEIGHT_COLUMNS = ["date", "instrument", "weight"]
FLOAT_FORMAT = "%.10f"
DEFAULT_ASSET_CLASSES = ("equity", "energy", "metals")
# Below this an instrument's OOS deliverable rests on too few rows to read its metrics (S5.9
# widened the pass-2 threshold from 30 to 60 so gc1s=30 and ng1s=56 also flag, not just ho1s=2).
COVERAGE_MIN_ROWS = 60


def _emit(df: pd.DataFrame, path, columns: list[str], *, float_format: str = FLOAT_FORMAT):
    """Write a tidy CSV deterministically: pinned columns, ISO dates, sorted rows, fixed floats."""
    out = df.loc[:, columns].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out = out.sort_values(["date", "instrument"]).reset_index(drop=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format=float_format, lineterminator="\n")
    return out


def emit_predictions(predictions: pd.DataFrame, path) -> pd.DataFrame:
    return _emit(predictions, path, PREDICTION_COLUMNS)


def emit_weights(weights: pd.DataFrame, path) -> pd.DataFrame:
    return _emit(weights, path, WEIGHT_COLUMNS)


def select_window(df: pd.DataFrame, start, end, *, date_col: str = "date") -> pd.DataFrame:
    d = pd.to_datetime(df[date_col])
    return df[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))].copy()


def strategy_weights(predictions: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """§6 sizing: fractional-Kelly stake × vol-target leverage, signed by the primary side.

    ``predictions`` carries ``side`` (primary signal) and ``ann_vol`` (annualised realised vol).
    A non-finite vol yields a flat (zero) weight rather than a NaN position.
    """
    pt, sl = config.pt_sl
    weights = []
    for row in predictions.itertuples(index=False):
        if not pd.notna(row.ann_vol) or not pd.notna(row.side):
            weights.append(0.0)
            continue
        weights.append(
            position_weight(
                side=row.side,
                p=row.prediction,
                b=pt,
                d=sl,
                realised_vol=row.ann_vol,
                target_vol=TARGET_VOL,
            )
        )
    out = predictions[["date", "instrument"]].copy()
    out["weight"] = weights
    return out


def coverage_caveat(
    predictions: pd.DataFrame,
    *,
    min_rows: int = COVERAGE_MIN_ROWS,
    instrument_ic: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Per-instrument OOS row counts, flagging thin-coverage instruments (S5.7 / widened S5.9).

    The deliverable emits all 11 instruments (no abstention), but a few rest on very few OOS rows
    (ho1s=2, gc1s=30, ng1s=56 in H1 2022). An instrument is flagged ``thin`` when it has fewer than
    ``min_rows`` OOS rows **or** an undefined information coefficient (``instrument_ic`` NaN) — the
    deliverable for those names cannot be read with confidence, and the write-up says so.
    """
    table = predictions.groupby("instrument").size().rename("n_oos_rows").reset_index()
    if instrument_ic is not None:
        table["ic"] = table["instrument"].map(instrument_ic)
        table["ic_undefined"] = table["ic"].isna()
    else:
        table["ic_undefined"] = False
    table["thin"] = (table["n_oos_rows"] < min_rows) | table["ic_undefined"]
    return table.sort_values("instrument").reset_index(drop=True)


def build_deliverables(
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    config: PipelineConfig,
    asset_classes=DEFAULT_ASSET_CLASSES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Run each asset-class metamodel; assemble raw + calibrated prediction frames, the
    calibrated-sized weight frame, and diagnostics (pass-3: Kelly is sized on the calibrated p̂)."""
    raw, calibrated, weights, diagnostics = [], [], [], {}
    for ac in asset_classes:
        result = run_asset_class(ohlcv, signals, ac, config)
        preds = result.predictions
        raw.append(preds[["date", "instrument", "prediction"]])
        calibrated.append(
            preds[["date", "instrument", "prediction_calibrated"]].rename(
                columns={"prediction_calibrated": "prediction"}
            )
        )
        # size Kelly on the calibrated probability (feed it in as the ``prediction`` column)
        sized = strategy_weights(preds.assign(prediction=preds["prediction_calibrated"]), config)
        weights.append(sized)
        diagnostics[ac] = {
            "best_model": result.best_model,
            "cv_scores": result.cv_scores,
            "per_instrument": result.diagnostics,
            "oos_brier": result.oos_brier,
            "oos_precision": result.oos_precision,
        }
    return (
        pd.concat(raw, ignore_index=True),
        pd.concat(calibrated, ignore_index=True),
        pd.concat(weights, ignore_index=True),
        diagnostics,
    )


def instrument_oos_ic(predictions: pd.DataFrame, returns_panel: pd.DataFrame) -> dict[str, float]:
    """Per-instrument OOS information coefficient: Spearman(p̂, next-day return) on the deliverable.

    Undefined (NaN) when an instrument has too few rows or a constant prediction — the S5.9
    'undefined IC' coverage flag. Reads only released returns, never the frozen parquet.
    """
    from .signal_analysis import information_coefficient

    ic: dict[str, float] = {}
    for inst, group in predictions.groupby("instrument"):
        if inst not in returns_panel.columns:
            ic[inst] = float("nan")
            continue
        forward = returns_panel[inst].shift(-1)
        series = group.set_index("date")["prediction"]
        aligned = pd.concat([series, forward.reindex(series.index)], axis=1).dropna()
        ic[inst] = (
            information_coefficient(aligned.iloc[:, 0], aligned.iloc[:, 1])
            if len(aligned) >= 3
            else float("nan")
        )
    return ic


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Emit the Alken metamodel deliverable CSVs.")
    parser.add_argument("--asset-classes", nargs="+", default=list(DEFAULT_ASSET_CLASSES))
    parser.add_argument("--predict-start", default="2022-01-01")
    parser.add_argument("--predict-end", default="2022-06-30")
    parser.add_argument("--outdir", default="outputs")
    # Shipped default path (pass 2): torch NN family + CPCV selection + PIT macro block.
    parser.add_argument("--roster", default="default")
    parser.add_argument("--cv-scheme", default="cpcv")
    parser.add_argument("--no-macro", action="store_true")
    # Pass-4 methodology, on by default: per-instrument embargo (S2.6) + F16 drift (S1.8-b).
    parser.add_argument(
        "--no-per-instrument-embargo", dest="per_instrument_embargo", action="store_false"
    )
    parser.add_argument("--no-drift", dest="use_drift", action="store_false")
    args = parser.parse_args(argv)

    set_seeds()
    from stml.io import load_clean_data, load_returns_panel

    ohlcv, signals = load_clean_data()
    config = PipelineConfig(
        predict_start=pd.Timestamp(args.predict_start),
        predict_end=pd.Timestamp(args.predict_end),
        roster=args.roster,
        cv_scheme=args.cv_scheme,
        use_macro=not args.no_macro,
        per_instrument_embargo=args.per_instrument_embargo,
        use_drift=args.use_drift,
    )
    raw_preds, cal_preds, weights, diagnostics = build_deliverables(
        ohlcv, signals, config, asset_classes=args.asset_classes
    )
    outdir = Path(args.outdir)
    # The calibrated file is the brief deliverable (``metamodel_predictions.csv``); the raw file
    # retains the uncalibrated p̂ for the §3 Brier/ECE-improvement story; the explicit
    # ``_calibrated`` name is a byte-identical alias so the pass-3 contract name also resolves.
    emit_predictions(cal_preds, outdir / "metamodel_predictions.csv")
    emit_predictions(cal_preds, outdir / "metamodel_predictions_calibrated.csv")
    emit_predictions(raw_preds, outdir / "metamodel_predictions_raw.csv")
    emit_weights(weights, outdir / "strategy_weights.csv")
    for ac, diag in diagnostics.items():
        cv = {k: round(v, 4) for k, v in diag["cv_scores"].items()}
        print(f"\n[{ac}] selected={diag['best_model']}  cv_auc={cv}")
        print(diag["per_instrument"].to_string(index=False))  # per-instrument BEFORE the aggregate
    log_path = outdir / "experiment_log.csv"
    if log_path.exists():
        log_path.unlink()  # fresh per emit run -> deterministic, rows in asset-class order
    for ac, diag in diagnostics.items():
        log_run(
            {
                "run_id": f"{config.predict_start.date()}_{ac}",
                "asset_class": ac,
                "roster": config.roster,
                "cv_scheme": config.cv_scheme,
                "reducer": "cluster_rep" if config.roster == "default" else "",
                "use_macro": config.use_macro,
                "best_model": diag["best_model"],
                "oos_auc": round(diag["cv_scores"].get(diag["best_model"], float("nan")), 6),
                "oos_brier": round(diag["oos_brier"], 6),  # XT.2: now measured, not blank
                "oos_precision": round(diag["oos_precision"], 6),
                "notes": "shipped default path; calibrated deliverable",
            },
            log_path,
        )
    # S5.9 coverage caveat on the deliverable (calibrated) preds + a real per-instrument OOS IC.
    returns_panel = load_returns_panel(kind="simple")
    ic = instrument_oos_ic(cal_preds, returns_panel)
    caveat = coverage_caveat(cal_preds, instrument_ic=ic)
    outdir.mkdir(parents=True, exist_ok=True)
    caveat.to_csv(outdir / "coverage_caveat.csv", index=False, lineterminator="\n")
    print("\nOOS COVERAGE (thin: <60 rows or undefined IC):")
    print(caveat.to_string(index=False))
    print(f"\nAGGREGATE: {len(cal_preds)} predictions / {len(weights)} weights -> {outdir}/")


if __name__ == "__main__":
    main()
