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

from .pipeline import PipelineConfig, run_asset_class  # importing the package pins env first
from .seeding import set_seeds
from .sizing import TARGET_VOL, position_weight

PREDICTION_COLUMNS = ["date", "instrument", "prediction"]
WEIGHT_COLUMNS = ["date", "instrument", "weight"]
FLOAT_FORMAT = "%.10f"
DEFAULT_ASSET_CLASSES = ("equity", "energy", "metals")
COVERAGE_MIN_ROWS = 30  # below this, an instrument's OOS deliverable rests on too few rows (S5.7)


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
    predictions: pd.DataFrame, *, min_rows: int = COVERAGE_MIN_ROWS
) -> pd.DataFrame:
    """Per-instrument OOS row counts, flagging near-empty instruments as thin coverage (S5.7).

    The deliverable emits all 11 instruments (no abstention), but a few rest on very few OOS rows
    (e.g. ho1s/gc1s in H1 2022); their numbers are flagged ``thin`` so the write-up can state the
    coverage honestly rather than imply uniform support.
    """
    table = predictions.groupby("instrument").size().rename("n_oos_rows").reset_index()
    table["thin"] = table["n_oos_rows"] < min_rows
    return table.sort_values("instrument").reset_index(drop=True)


def build_deliverables(
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    config: PipelineConfig,
    asset_classes=DEFAULT_ASSET_CLASSES,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run each asset-class metamodel and assemble the two deliverable frames + diagnostics."""
    preds, weights, diagnostics = [], [], {}
    for ac in asset_classes:
        result = run_asset_class(ohlcv, signals, ac, config)
        preds.append(result.predictions)
        weights.append(strategy_weights(result.predictions, config))
        diagnostics[ac] = {
            "best_model": result.best_model,
            "cv_scores": result.cv_scores,
            "per_instrument": result.diagnostics,
        }
    return pd.concat(preds, ignore_index=True), pd.concat(weights, ignore_index=True), diagnostics


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
    args = parser.parse_args(argv)

    set_seeds()
    from stml.io import load_clean_data

    ohlcv, signals = load_clean_data()
    config = PipelineConfig(
        predict_start=pd.Timestamp(args.predict_start),
        predict_end=pd.Timestamp(args.predict_end),
        roster=args.roster,
        cv_scheme=args.cv_scheme,
        use_macro=not args.no_macro,
    )
    preds, weights, diagnostics = build_deliverables(
        ohlcv, signals, config, asset_classes=args.asset_classes
    )
    outdir = Path(args.outdir)
    emit_predictions(preds, outdir / "metamodel_predictions.csv")
    emit_weights(weights, outdir / "strategy_weights.csv")
    for ac, diag in diagnostics.items():
        cv = {k: round(v, 4) for k, v in diag["cv_scores"].items()}
        print(f"\n[{ac}] selected={diag['best_model']}  cv_auc={cv}")
        print(diag["per_instrument"].to_string(index=False))  # per-instrument BEFORE the aggregate
    caveat = coverage_caveat(preds)
    outdir.mkdir(parents=True, exist_ok=True)
    caveat.to_csv(outdir / "coverage_caveat.csv", index=False, lineterminator="\n")
    print("\nOOS COVERAGE (thin instruments rest on few rows):")
    print(caveat.to_string(index=False))
    print(f"\nAGGREGATE: emitted {len(preds)} predictions / {len(weights)} weights to {outdir}/")


if __name__ == "__main__":
    main()
