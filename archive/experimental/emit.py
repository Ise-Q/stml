"""Deterministic CSV writer — plan §3.9 / §8 S6.

Contract (plan §3.9):
* Rows sorted by ``(date, instrument)``.
* Pinned column order.
* ISO date ``%Y-%m-%d``.
* Float format ``%.10f``.
* ``lineterminator="\\n"``.

Byte-identical re-emit verified by test.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FLOAT_FORMAT = "%.10f"
DATE_FORMAT = "%Y-%m-%d"
LINETERMINATOR = "\n"


def emit_predictions(
    predictions: pd.DataFrame, out_path: Path, *, column: str = "prediction"
) -> None:
    """Write the ``(date, instrument, prediction)`` deliverable CSV."""
    df = predictions.loc[:, ["date", "instrument", column]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime(DATE_FORMAT)
    df = df.sort_values(["date", "instrument"]).reset_index(drop=True)
    df.columns = ["date", "instrument", "prediction"]
    df.to_csv(out_path, index=False, float_format=FLOAT_FORMAT, lineterminator=LINETERMINATOR)


def emit_weights(weights_long: pd.DataFrame, out_path: Path) -> None:
    """Write ``(date, instrument, weight)`` strategy CSV from a long frame."""
    df = weights_long.loc[:, ["date", "instrument", "weight"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime(DATE_FORMAT)
    df = df.sort_values(["date", "instrument"]).reset_index(drop=True)
    df.to_csv(out_path, index=False, float_format=FLOAT_FORMAT, lineterminator=LINETERMINATOR)


def panel_to_long(panel: pd.DataFrame, value_col: str = "weight") -> pd.DataFrame:
    """(date × instrument) wide → tidy long (date, instrument, value)."""
    out = panel.copy()
    out.index.name = "date"
    out = out.reset_index().melt(id_vars="date", var_name="instrument", value_name=value_col)
    return out
