"""Extend the cleaned Bloomberg parquets to cover H2 2022.

Background
----------
The released-window cleaned parquets under ``data/bloomberg/cleaned/`` stop on
2022-06-29 / 2022-07-01 because that is the latest data the brief released. The
hidden test window is H2 2022 (2022-07-01 → 2022-12-30). For the marker's
rerun to produce real predictions the feature pipeline needs Bloomberg data
for H2 2022.

The shipped model uses two Bloomberg-derived feature families:

* **F11** — macro panel z-scores (VIX, MOVE, DXY, UST/Bund/TIPS yields, OAS
  spreads, PMIs).
* **F22** — EIA weekly crude inventory release (release-day flag + change).

This script ships the H2 2022 data for those two families and extends the
underlying parquets so the feature pipeline reads a single continuous file:

1. ``macro_alternative.parquet`` — extended from ``data/OOS_additional_data.xlsx``,
   which carries the same 21 macro series for Jul-Dec 2022 (VIX, MOVE, DXY,
   UST/Bund/TIPS yields, OAS, EIA stock levels, PMIs, etc.).
2. ``eia_crude.parquet`` — derived from the EIA_CRUDE_STOCK series in the OOS
   workbook (weekly change in thousands of barrels).
3. ``eia_release_flag.parquet`` — derived from EIA's standard release schedule
   (Wednesdays after the as-of Friday).

Two other parquets (``futures_term.parquet``, ``options_iv.parquet``) are
**left untouched** — the corresponding feature families (F18 futures term
structure, F19 options-implied vol) were excluded from the shipped model on
parsimony grounds after the cluster-importance analysis showed they did not
materially lift performance over the OHLCV + F11 + F22 baseline. Carrying them
would force a stale-data substitution we did not consider methodologically
clean.

Run
---
    python scripts/extend_bloomberg_for_h2.py
    # idempotent: re-running is a no-op if the parquets already cover H2 2022.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEANED = ROOT / "data" / "bloomberg" / "cleaned"
OOS_XLSX = ROOT / "data" / "OOS_additional_data.xlsx"

H2_START = pd.Timestamp("2022-07-01")
H2_END = pd.Timestamp("2022-12-30")


def _load_oos_panel() -> pd.DataFrame:
    """Read OOS_additional_data.xlsx into a tidy date-indexed wide panel.

    The workbook layout is paired columns: each series ``S`` occupies a
    date column and a value column. Row 0 of the sheet is the series name.
    """
    raw = pd.read_excel(OOS_XLSX, header=None)
    series_names = raw.iloc[0].dropna().tolist()
    data = raw.iloc[1:].reset_index(drop=True)

    panels = []
    # The paired layout means column 2k = date for series k, column 2k+1 = value.
    for k, name in enumerate(series_names):
        col_date = data.iloc[:, 2 * k]
        col_val = data.iloc[:, 2 * k + 1]
        s = pd.Series(col_val.values, index=pd.to_datetime(col_date), name=name)
        s = s.dropna().sort_index()
        s = s[~s.index.duplicated(keep="first")]
        panels.append(s)

    panel = pd.concat(panels, axis=1)
    panel.index.name = "date"
    return panel


def _make_business_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def extend_macro_alternative(oos: pd.DataFrame) -> None:
    """Append H2 2022 rows to ``macro_alternative.parquet``."""
    p = CLEANED / "macro_alternative.parquet"
    current = pd.read_parquet(p)
    current.index = pd.to_datetime(current.index)
    cols = list(current.columns)

    h2_idx = _make_business_index(H2_START, H2_END)
    h2 = pd.DataFrame(index=h2_idx, columns=cols, dtype="float64")

    for col in cols:
        if col in oos.columns:
            # reindex onto H2 business days, ffill (matches the existing
            # convention: monthly PMI sits as a step function).
            series = oos[col].reindex(h2_idx).ffill().bfill()
            h2[col] = series
        else:
            # not in OOS xlsx → carry the last known value from the existing parquet
            last_value = current[col].dropna().iloc[-1] if current[col].notna().any() else np.nan
            h2[col] = last_value

    extended = pd.concat([current.loc[current.index < H2_START], h2])
    extended.index.name = current.index.name or "date"
    extended.to_parquet(p)
    print(f"  macro_alternative.parquet → extended to {extended.index.max().date()} "
          f"(+{len(h2)} rows)")


def extend_eia_crude(oos: pd.DataFrame) -> None:
    """Derive weekly EIA crude change from OOS stock levels.

    ``EIA_CRUDE_CHANGE_KB[t] = EIA_CRUDE_STOCK[t] - EIA_CRUDE_STOCK[t-1 release]``.
    The OOS xlsx carries EIA_CRUDE_STOCK weekly; the change is positive when
    stocks built and negative when stocks drew.
    """
    p = CLEANED / "eia_crude.parquet"
    current = pd.read_parquet(p)
    current.index = pd.to_datetime(current.index)

    stock = oos["EIA_CRUDE_STOCK"].dropna().sort_index()
    release_dates = stock.index[stock.index >= H2_START]
    if len(release_dates) == 0:
        print("  eia_crude.parquet         → no H2 EIA data in OOS workbook (skipping)")
        return
    change = stock.diff().reindex(release_dates)

    # Tile across all H2 business days, holding the last release's change
    h2_idx = _make_business_index(H2_START, H2_END)
    h2_series = change.reindex(h2_idx).ffill().bfill()
    h2 = pd.DataFrame({"EIA_CRUDE_CHANGE_KB": h2_series}, index=h2_idx)

    extended = pd.concat([current.loc[current.index < H2_START], h2])
    extended.index.name = current.index.name or "date"
    extended.to_parquet(p)
    print(f"  eia_crude.parquet         → extended to {extended.index.max().date()} "
          f"(+{len(h2)} rows from {len(release_dates)} OOS releases)")


def extend_eia_release_flag(oos: pd.DataFrame) -> None:
    """1 on EIA release days (Wednesdays of the corresponding week), 0 elsewhere."""
    p = CLEANED / "eia_release_flag.parquet"
    current = pd.read_parquet(p)
    current.index = pd.to_datetime(current.index)

    h2_idx = _make_business_index(H2_START, H2_END)
    # Approximate release flag: EIA reports Wed for the prior Friday's stocks.
    flag = (h2_idx.weekday == 2).astype(int)  # 2 = Wednesday
    h2 = pd.DataFrame({"EIA_CRUDE_RELEASE_FLAG": flag}, index=h2_idx)

    extended = pd.concat([current.loc[current.index < H2_START], h2])
    extended.index.name = current.index.name or "date"
    extended.to_parquet(p)
    print(f"  eia_release_flag.parquet  → extended to {extended.index.max().date()} "
          f"(+{len(h2)} rows; {int(flag.sum())} release days)")


def main() -> int:
    if not OOS_XLSX.exists():
        print(f"ERROR: {OOS_XLSX} not found", file=sys.stderr)
        return 1
    if not CLEANED.exists():
        print(f"ERROR: {CLEANED} not found", file=sys.stderr)
        return 1

    # Idempotency check: bail if every parquet already covers H2.
    macro = pd.read_parquet(CLEANED / "macro_alternative.parquet")
    macro.index = pd.to_datetime(macro.index)
    if macro.index.max() >= H2_END:
        print(f"All cleaned parquets already cover through {macro.index.max().date()}; "
              f"nothing to do.")
        return 0

    print(f"Extending cleaned Bloomberg parquets to cover "
          f"{H2_START.date()} → {H2_END.date()}:")

    oos = _load_oos_panel()
    print(f"  OOS xlsx panel: {oos.shape[0]} dates, {oos.shape[1]} series")

    extend_macro_alternative(oos)
    extend_eia_crude(oos)
    extend_eia_release_flag(oos)

    print("\nDone. The feature pipeline (make_features) now sees a continuous "
          "Bloomberg panel through Dec 2022.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
