"""Build the H1 2022 deliverable CSVs from the pipeline's cached events.

Reads:
    data/primary_signals.csv                                — the full panel
    results/submission/oos_events_with_predictions.csv — events with model output

Writes:
    outputs/metamodel_predictions.csv  — schema (date, instrument, prediction)
    outputs/strategy_weights.csv       — schema (date, instrument, weight)

Brief contract (https://hm-ai.github.io/BUSI70575/coursework/):
    * H1 2022 = 2022-01-03 → 2022-06-30 (the first half of the released window).
    * One row per (date, instrument) in the window — full grid, not just events.
    * prediction ∈ [0, 1].
    * Rerunnable on H2 2022 by passing --start 2022-07-01 --end 2022-12-31.

Conventions for prediction fill:
    * signal == 0          → prediction = 0.0  (no trade taken, no probability)
    * signal != 0, event resolved → prediction = calibrated_proba (from the pipeline)
    * signal != 0, no event resolution (e.g. near window end where t1 spills out
                              of the released window) → prediction = 0.5
                              (neutral: meta-model abstains)

Conventions for weight fill:
    * weight = 0.0 wherever prediction <= 0.5 OR no model row
    * else weight = strategy weight from oos_events_with_predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_SIGNALS = ROOT / "data" / "primary_signals.csv"
EVENTS_WITH_PRED = ROOT / "results" / "submission" / "oos_events_with_predictions.csv"

OUT_DIR = ROOT / "outputs"
PREDICTIONS_OUT = OUT_DIR / "metamodel_predictions.csv"
WEIGHTS_OUT = OUT_DIR / "strategy_weights.csv"

FLOAT_FORMAT = "%.10f"
DATE_FORMAT = "%Y-%m-%d"
LINETERMINATOR = "\n"


def build_full_grid(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    primary_signals_path: Path = PRIMARY_SIGNALS,
    events_with_pred_path: Path = EVENTS_WITH_PRED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (predictions_df, weights_df) for the (start, end] window inclusive.

    Returns two DataFrames with columns [date, instrument, prediction] and
    [date, instrument, weight] respectively, sorted by (date, instrument),
    ISO-formatted date strings, full date × instrument grid.
    """
    # Load the full primary signal panel and the resolved-event predictions.
    sig = pd.read_csv(primary_signals_path, parse_dates=["date"])
    events = pd.read_csv(events_with_pred_path, parse_dates=["t_signal"])

    # Restrict the panel to the window. Use t_signal (the signal date) — that
    # is what (date, instrument) refers to in the deliverable.
    sig = sig[(sig["date"] >= start) & (sig["date"] <= end)].copy()
    events = events[(events["t_signal"] >= start) & (events["t_signal"] <= end)].copy()
    events = events.rename(columns={"t_signal": "date"})

    # Melt the panel to long.
    long = sig.melt(id_vars=["date"], var_name="instrument", value_name="signal")

    # Left-join model predictions.
    pred_lookup = events[["date", "instrument", "calibrated_proba"]]
    weight_lookup = events[["date", "instrument", "weight"]]

    merged = (
        long.merge(pred_lookup, on=["date", "instrument"], how="left")
            .merge(weight_lookup, on=["date", "instrument"], how="left")
    )

    # Fill predictions per conventions above.
    merged["prediction"] = np.where(
        merged["signal"] == 0,
        0.0,
        merged["calibrated_proba"].fillna(0.5),
    )
    merged["weight"] = np.where(
        merged["signal"] == 0,
        0.0,
        merged["weight"].fillna(0.0),
    )

    # Sort and shape.
    merged = merged.sort_values(["date", "instrument"]).reset_index(drop=True)

    predictions = merged[["date", "instrument", "prediction"]].copy()
    weights = merged[["date", "instrument", "weight"]].copy()

    return predictions, weights


def write_csv_deterministic(df: pd.DataFrame, path: Path) -> None:
    """Write the deliverable with the same conventions as `experimental.emit`:
    ISO dates, %.10f floats, LF line terminator, no index, sorted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime(DATE_FORMAT)
    out.to_csv(
        path,
        index=False,
        float_format=FLOAT_FORMAT,
        lineterminator=LINETERMINATOR,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2022-01-03", help="Window start (ISO).")
    p.add_argument("--end", default="2022-06-30", help="Window end (ISO).")
    p.add_argument(
        "--predictions-out",
        type=Path,
        default=PREDICTIONS_OUT,
        help="Path for the predictions CSV.",
    )
    p.add_argument(
        "--weights-out",
        type=Path,
        default=WEIGHTS_OUT,
        help="Path for the weights CSV.",
    )
    args = p.parse_args(argv)

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    predictions, weights = build_full_grid(start, end)

    write_csv_deterministic(predictions, args.predictions_out)
    write_csv_deterministic(weights, args.weights_out)

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)

    n = len(predictions)
    n_inst = predictions["instrument"].nunique()
    n_dates = predictions["date"].nunique()
    sig_nonzero = int((predictions["prediction"] != 0.0).sum())
    print(f"window: {start.date()} -> {end.date()}")
    print(f"  predictions: {n} rows = {n_dates} dates x {n_inst} instruments")
    print(f"    non-zero predictions: {sig_nonzero}")
    print(f"    zero predictions (signal=0): {n - sig_nonzero}")
    if n:
        print(f"    range: [{predictions['prediction'].min():.4f}, "
              f"{predictions['prediction'].max():.4f}]")
    print(f"  weights:     {len(weights)} rows")
    print(f"    non-zero weights: {int((weights['weight'] != 0).sum())}")
    if len(weights):
        print(f"    range: [{weights['weight'].min():.4f}, "
              f"{weights['weight'].max():.4f}]")
    print(f"  predictions -> {_rel(args.predictions_out)}")
    print(f"  weights     -> {_rel(args.weights_out)}")
    if n == 0:
        print("  (window is outside the released primary_signals window: "
              "supply an extended primary_signals.csv to populate H2 2022.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
