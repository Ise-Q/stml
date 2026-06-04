"""Bloomberg / alternate-data ingestion + PIT alignment.

Plan §5.3 / §8 Stage 2 deliverable.

Reads raw BBG-style pulls + the macro CSV from ``data/bloomberg/raw/`` and
``data/alternate_data_cleaned.csv`` (when present)
and produces **PIT-aligned cleaned parquets** under
``data/bloomberg/cleaned/`` that the feature modules consume.

The cleaner applies:
    * Publication lag (Block A daily / Block B daily: +1 day;
      Block E weekly EIA: +5 calendar days from the Friday data date).
    * Forward-fill onto a daily business calendar (so feature modules can
      reindex by trade date without worrying about weekly cadence).
    * Schema normalisation — every cleaned parquet has a ``DatetimeIndex``
      named ``date`` and one column per series; column names are the
      pipeline's canonical short names (e.g. ``CL1_LAST``, ``SPX_IV1M_ATM``).

Run via:

    uv run python -m stml.experimental.bloomberg_ingest

Idempotent — re-running over the same raw files re-emits byte-identical
parquets. The raw files themselves are gitignored (``data/bloomberg/raw/``);
the cleaned parquets are committed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------

# Publication lags (calendar days), methodology spec
LAG_DAILY = 1
LAG_EIA = 5  # Friday data-as-of -> Wednesday release

# Block A — tickers we care about (the suffix _LAST is added by the cleaner).
BLOCK_A_FRONT_TICKERS = [
    "CL1_Comdty", "HO1_Comdty", "XB1_Comdty", "NG1_Comdty",
    "GC1_Comdty", "SI1_Comdty", "HG1_Comdty", "PL1_Comdty",
]
BLOCK_A_SECOND_TICKERS = [
    "CL2_Comdty", "HO2_Comdty", "XB2_Comdty", "NG2_Comdty",
    "GC2_Comdty", "SI2_Comdty", "HG2_Comdty", "PL2_Comdty",
    "UX1_Index", "UX2_Index",
]

# Block B — sheet name template.
BLOCK_B_UNDERLYINGS = [
    "SPX", "NDX", "SX5E",
    "CL1", "HO1", "XB1", "NG1",
    "GC1", "SI1", "HG1", "XPT",
]
BLOCK_B_FIELDS = ["IV1M_ATM", "IV3M_ATM", "IV1M_90MNY", "IV1M_110MNY"]


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"Could not locate repo root from {here}")


# ---------------------------------------------------------------------------
# Block A — futures term structure.
# ---------------------------------------------------------------------------


def _read_block_a_front(raw_dir: Path) -> pd.DataFrame:
    """Read the front-month CSV and keep only last_price columns (+1d lag).

    Returns a date-indexed frame with columns ``CL1_LAST``, ``HO1_LAST`` etc.
    OI and volume columns are dropped here (not used by F18). All values are
    raw BBG (not adjusted) — see the validation report; OHLCV is on a
    different, back-adjusted convention.
    """
    csv = raw_dir / "block_a_front_playbook_consolidated.csv"
    if not csv.exists():
        raise FileNotFoundError(csv)
    df = pd.read_csv(csv, parse_dates=["date"]).set_index("date").sort_index()
    out = pd.DataFrame(index=df.index)
    for ticker in BLOCK_A_FRONT_TICKERS:
        col = f"{ticker}__last_price"
        if col in df.columns:
            short = ticker.replace("_Comdty", "").replace("_Index", "") + "_LAST"
            out[short] = df[col]
    return _apply_calendar_lag(out, LAG_DAILY)


def _read_block_a_second(raw_dir: Path) -> pd.DataFrame:
    """Read the 2nd-month CSV (same shape, includes UX1/UX2 VIX futures) (+1d lag)."""
    csv = raw_dir / "block_a_playbook_consolidated.csv"
    if not csv.exists():
        raise FileNotFoundError(csv)
    df = pd.read_csv(csv, parse_dates=["date"]).set_index("date").sort_index()
    out = pd.DataFrame(index=df.index)
    for ticker in BLOCK_A_SECOND_TICKERS:
        col = f"{ticker}__last_price"
        if col in df.columns:
            short = ticker.replace("_Comdty", "").replace("_Index", "") + "_LAST"
            out[short] = df[col]
    return _apply_calendar_lag(out, LAG_DAILY)


# ---------------------------------------------------------------------------
# Block B — options-implied volatility (Excel, multi-sheet, header quirks).
# ---------------------------------------------------------------------------


def _read_block_b(raw_dir: Path) -> pd.DataFrame:
    """Parse the Block B Excel workbook into a single PIT-aligned frame.

    Per-sheet schema (no header row):
        col 0: date (datetime)
        col 1: IV value (float)
        cols 2-5: 'Unnamed: 2', 'Ticker formula', duplicate date, duplicate value
            — these are header-row leakage from the BBG paste; ignored.

    Empty sheets (#N/A — happens for XPT) become empty columns; the loader
    substitutes GC1 IV for pl1s downstream (methodology spec).
    """
    candidates = list(raw_dir.glob("block_b_iv_extraction*.xlsx"))
    if not candidates:
        raise FileNotFoundError(f"No block_b_iv_extraction*.xlsx in {raw_dir}")
    xlsx_path = candidates[0]

    out_frames: dict[str, pd.Series] = {}
    for underlying in BLOCK_B_UNDERLYINGS:
        for field in BLOCK_B_FIELDS:
            sheet = f"{underlying}_{field}"
            try:
                # We don't trust the header — read with header=None and pick cols 0/1.
                raw = pd.read_excel(
                    xlsx_path, sheet_name=sheet, header=None, usecols=[0, 1]
                )
            except Exception:
                # Sheet missing or corrupt — skip silently; methodology already
                # documents which underlyings have data.
                continue
            if raw.empty or raw.shape[1] < 2:
                continue
            raw.columns = ["date", "value"]
            # The very first row IS sometimes a data row (the BBG paste's
            # quirk). Coerce to datetime + float and drop rows where either
            # fails.
            raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
            raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
            raw = raw.dropna(subset=["date", "value"]).set_index("date").sort_index()
            if raw.empty:
                continue
            col_name = f"{underlying}_{field}"
            out_frames[col_name] = raw["value"]

    if not out_frames:
        return pd.DataFrame()

    out = pd.DataFrame(out_frames)
    return _apply_calendar_lag(out, LAG_DAILY)


# ---------------------------------------------------------------------------
# Block E — EIA weekly crude changes (Friday data, +5d lag -> Wed release).
# ---------------------------------------------------------------------------


def _read_block_e_eia(raw_dir: Path) -> pd.DataFrame:
    """Read the EIA weekly-crude-change CSV with +5d lag (Friday -> Wed)."""
    csv = raw_dir / "block_e_eia_playbook_consolidated.csv"
    if not csv.exists():
        raise FileNotFoundError(csv)
    df = pd.read_csv(csv, parse_dates=["date"]).set_index("date").sort_index()
    # Rename the verbose BBG ticker column.
    rename = {}
    for col in df.columns:
        if "DOEASCRD" in col:
            rename[col] = "EIA_CRUDE_CHANGE_KB"
    df = df.rename(columns=rename)
    return _apply_calendar_lag(df, LAG_EIA)


def _eia_release_calendar(eia_change: pd.DataFrame) -> pd.DataFrame:
    """Build a daily binary flag = 1 on the EIA release Wednesday, 0 elsewhere.

    The release-date is computed as the as-of Friday + 5 calendar days (= the
    following Wednesday). The flag is keyed by calendar date and aligned to a
    daily business-day index via _to_daily.
    """
    if eia_change.empty:
        return pd.DataFrame()
    release_dates = eia_change.index  # already lagged by +5d in _read_block_e_eia
    # Build a daily flag, defaulting to 0 elsewhere. End boundary is the
    # LAST release date the EIA file carries (data-driven; not hard-coded
    # so the marker's H2-2022 re-run still produces a valid flag).
    full_bd = pd.bdate_range(release_dates.min(), release_dates.max())
    flag = pd.Series(0, index=full_bd, dtype=int)
    # If the release day fell on a holiday, snap to the next business day so
    # the flag still fires on a tradeable calendar date.
    snapped = pd.to_datetime(release_dates).map(
        lambda d: full_bd[full_bd.searchsorted(d, side="left")]
        if full_bd.searchsorted(d, side="left") < len(full_bd)
        else None
    )
    snapped = [d for d in snapped if d is not None]
    flag.loc[snapped] = 1
    return pd.DataFrame({"EIA_CRUDE_RELEASE_FLAG": flag}, index=full_bd)


# ---------------------------------------------------------------------------
# the macro CSV (already in repo via the alternate-data branch, read into the bloomberg
# pipeline so all macro lives in one place after S2).
# ---------------------------------------------------------------------------


def _read_alternative_macro(raw_dir: Path) -> pd.DataFrame:
    """Read the alternate_data_cleaned.csv via repo-root.

    The file is COMMITTED on the alternate-data branch but NOT  (we
    keep the experimental branch lean — see the orphan-rewrite history). We
    pull it via ``git show`` and persist a local copy to ``data/bloomberg/raw/``
    so the ingest is self-contained on the experimental branch from then on.
    """
    cache = raw_dir / "alternative_alternate_data_cleaned.csv"
    if not cache.exists():
        import subprocess
        repo_root = _find_repo_root()
        text = subprocess.check_output(
            ["git", "show", "origin/main:data/alternate_data_cleaned.csv"],
            cwd=repo_root,
        ).decode()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text)
    df = pd.read_csv(cache, parse_dates=["Date"]).rename(columns={"Date": "date"})
    df = df.set_index("date").sort_index()
    # Drop nothing — the +1d lag applies to daily series; weekly EIA macro
    # series in the CSV are already forward-filled to daily.
    return _apply_calendar_lag(df, LAG_DAILY)


# ---------------------------------------------------------------------------
# PIT alignment + daily reindex helpers.
# ---------------------------------------------------------------------------


def _apply_calendar_lag(df: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    """Shift the index FORWARD by ``lag_days`` calendar days (PIT lag).

    A value observed at ``t`` is *actionable* at ``t + lag_days``. We move the
    timestamp forward so the value at the resulting index is "what's safe to
    use at this date".
    """
    if df.empty:
        return df
    shifted = df.copy()
    shifted.index = shifted.index + pd.Timedelta(days=lag_days)
    return shifted


def _to_daily(df: pd.DataFrame, fill_method: str = "ffill") -> pd.DataFrame:
    """Reindex to daily business-day calendar with forward-fill.

    Weekly inputs (EIA) get carried forward day-by-day until the next release.
    Daily inputs (Block A, Block B) get a no-op on most days; weekends/holidays
    that aren't in the source get forward-filled from the prior trading day.
    """
    if df.empty:
        return df
    full = pd.bdate_range(df.index.min(), df.index.max())
    out = df.reindex(full)
    if fill_method == "ffill":
        out = out.ffill()
    return out


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestResult:
    """Summary of what the ingest produced — printed and persisted as JSON."""

    cleaned_dir: Path
    raw_dir: Path
    parquets: dict[str, tuple[int, int]]  # name -> (rows, cols)


def ingest(raw_dir: Path | None = None, cleaned_dir: Path | None = None) -> IngestResult:
    """Run the full ingest pipeline; returns a summary dataclass."""
    root = _find_repo_root()
    raw_dir = raw_dir or (root / "data" / "bloomberg" / "raw")
    cleaned_dir = cleaned_dir or (root / "data" / "bloomberg" / "cleaned")
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    parquets: dict[str, tuple[int, int]] = {}

    # 1. Block A — front + 2nd month combined into one parquet
    a_front = _read_block_a_front(raw_dir)
    a_second = _read_block_a_second(raw_dir)
    a_combined = pd.concat([a_front, a_second], axis=1, sort=True)
    a_combined = _to_daily(a_combined, fill_method="ffill")
    a_path = cleaned_dir / "futures_term.parquet"
    a_combined.to_parquet(a_path)
    parquets["futures_term"] = a_combined.shape

    # 2. Block B — options IV
    b_combined = _read_block_b(raw_dir)
    if not b_combined.empty:
        b_combined = _to_daily(b_combined, fill_method="ffill")
        b_path = cleaned_dir / "options_iv.parquet"
        b_combined.to_parquet(b_path)
        parquets["options_iv"] = b_combined.shape

    # 3. Block E — EIA crude change (weekly) + release flag (daily)
    e_change = _read_block_e_eia(raw_dir)
    e_change_daily = _to_daily(e_change, fill_method="ffill")
    e_path = cleaned_dir / "eia_crude.parquet"
    e_change_daily.to_parquet(e_path)
    parquets["eia_crude"] = e_change_daily.shape

    e_flag = _eia_release_calendar(e_change)
    if not e_flag.empty:
        e_flag_path = cleaned_dir / "eia_release_flag.parquet"
        e_flag.to_parquet(e_flag_path)
        parquets["eia_release_flag"] = e_flag.shape

    # 4. the macro — the 21-series CSV.
    h_macro = _read_alternative_macro(raw_dir)
    h_macro = _to_daily(h_macro, fill_method="ffill")
    h_path = cleaned_dir / "macro_alternative.parquet"
    h_macro.to_parquet(h_path)
    parquets["macro_alternative"] = h_macro.shape

    return IngestResult(cleaned_dir=cleaned_dir, raw_dir=raw_dir, parquets=parquets)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ingest BBG + the macro into cleaned parquets")
    args = ap.parse_args(argv)
    result = ingest()
    print(f"Cleaned dir: {result.cleaned_dir}")
    print(f"Raw dir: {result.raw_dir}")
    print(f"\nParquets emitted:")
    for name, (rows, cols) in result.parquets.items():
        print(f"  {name:25s}  {rows:6d} rows × {cols:3d} cols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
