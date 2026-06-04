"""Regenerate ``data/triple_barrier_labels.csv`` to include H2 2022 events.

H2 2022 is the brief's hidden test window. The shipped labels CSV covers only
the released window (2020-01-03 → 2022-06-29) because that is the universe the
per-instrument barrier-geometry grid search ran on. When the held-out window
arrives, the pipeline needs labels for the new events too — and they must use
*the same per-instrument geometry* the model was selected against, otherwise
the deployment slice is not the slice that was tuned.

What this script does
---------------------
1. Loads the shipped labels CSV to read the locked per-instrument
   ``(pt, sl, h)`` tuple chosen during development.
2. Loads ``data/ohlcv_data.csv`` and ``data/primary_signals.csv``, which the
   marker has (presumably) replaced with H2-extended versions.
3. For every non-zero-signal day strictly after the released window
   (``> 2022-06-30``) it walks the instrument's forward price path and resolves
   the triple-barrier exit using the instrument's locked geometry. The first
   barrier touched fixes ``t_end``, ``ret``, ``label``, and ``touch``.
4. Concatenates the original released-window rows (unchanged) with the new
   H2 rows (``partition = "test"``) and overwrites
   ``data/triple_barrier_labels.csv``.

Convention (mirrors the released labels exactly)
-------------------------------------------------
* **Entry**: at the close of the signal bar (``t_start == t_signal``).
* **σ**: 20-bar trailing standard deviation of close-to-close arithmetic
  returns (``f2_vol_20`` style; same as the shipped CSV's ``sigma`` column).
* **Path**: ``close[t+1 : t+h]`` (h bars strictly after entry).
* **First-touch ordering**: scan bars in order, take the earliest of
  {profit-take, stop-loss, vertical}.
* **Drop events without a full ``h``-bar forward window** (no peeking past
  the end of the price series).

Run
---
    python scripts/relabel_for_h2.py

The script is idempotent: if the inputs already cover H2 and the labels CSV
already includes a row for every H2 non-zero-signal event, it is a no-op and
exits 0 with a brief printout. Re-running after a fresh OHLCV replacement is
safe.

This script is only needed for the H2 rerun. For the H1-only deliverable the
shipped labels CSV is authoritative and this script does not need to run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

LABELS_CSV = DATA / "triple_barrier_labels.csv"
OHLCV_CSV = DATA / "ohlcv_data.csv"
SIGNALS_CSV = DATA / "primary_signals.csv"

H2_BOUNDARY = pd.Timestamp("2022-06-30")
SIGMA_WINDOW = 20  # rolling-std bars for σ (= f2_vol_20)


def _load_geometry(labels: pd.DataFrame) -> pd.DataFrame:
    """Per-instrument (pt, sl, h) from the shipped labels."""
    geo = labels.groupby("instrument")[["pt", "sl", "h"]].first().reset_index()
    geo["h"] = geo["h"].astype(int)
    return geo


def _causal_sigma(close: pd.Series, window: int = SIGMA_WINDOW) -> pd.Series:
    """20-bar rolling std of LOG returns (causal).

    Matches the shipped CSV's `sigma` column, which is `f2_vol_20 / sqrt(252)`
    where `f2_vol_20 = log_returns.rolling(20).std() * sqrt(252)`. The two
    sqrt(252) factors cancel, leaving `log_returns.rolling(20).std()`. Verified
    against the shipped sigma column on a sample of released-window events.
    """
    r = np.log(close / close.shift(1))
    return r.rolling(window=window, min_periods=window).std()


def _resolve_event(
    *,
    close_arr: np.ndarray,
    p_entry: int,
    side: float,
    sigma: float,
    pt: float,
    sl: float,
    h: int,
) -> tuple[int, float, int, str] | None:
    """Triple-barrier resolution for one event.

    Returns ``(end_idx, ret, label, touch)`` or ``None`` if the forward window
    is incomplete or sigma is invalid.
    """
    end = p_entry + h
    if end >= len(close_arr):
        return None
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    entry = close_arr[p_entry]
    path_fwd = close_arr[p_entry + 1 : end + 1]
    rel_rets = (path_fwd / entry - 1.0) * side

    up = pt * sigma if pt > 0 else np.inf
    dn = -sl * sigma if sl > 0 else -np.inf

    pt_hits = np.flatnonzero(rel_rets >= up)
    sl_hits = np.flatnonzero(rel_rets <= dn)
    first_pt = int(pt_hits[0]) if pt_hits.size else None
    first_sl = int(sl_hits[0]) if sl_hits.size else None

    if first_pt is None and first_sl is None:
        # Vertical
        touch = "vert"
        end_offset = h - 1  # last bar of the path
    elif first_pt is None:
        touch = "sl"
        end_offset = first_sl
    elif first_sl is None:
        touch = "pt"
        end_offset = first_pt
    else:
        # Both touched in window — take the earlier one
        if first_pt <= first_sl:
            touch = "pt"; end_offset = first_pt
        else:
            touch = "sl"; end_offset = first_sl

    end_idx = p_entry + 1 + end_offset
    ret = float(rel_rets[end_offset])
    label = 1 if ret > 0 else 0
    return end_idx, ret, label, touch


def relabel_h2(*, dry_run: bool = False) -> int:
    if not LABELS_CSV.exists():
        print(f"ERROR: {LABELS_CSV} not found", file=sys.stderr)
        return 1
    if not OHLCV_CSV.exists() or not SIGNALS_CSV.exists():
        print(f"ERROR: missing {OHLCV_CSV} or {SIGNALS_CSV}", file=sys.stderr)
        return 1

    shipped = pd.read_csv(LABELS_CSV, parse_dates=["date", "t1"])
    geo = _load_geometry(shipped)
    print(f"Shipped labels: {len(shipped)} rows, dates {shipped['date'].min().date()} → "
          f"{shipped['date'].max().date()}")
    print("Per-instrument geometry (pt / sl / h):")
    print(geo.to_string(index=False))

    ohlcv = pd.read_csv(OHLCV_CSV, parse_dates=["date"])
    signals = pd.read_csv(SIGNALS_CSV, parse_dates=["date"])

    ohlcv_window_end = ohlcv["date"].max()
    signals_window_end = signals["date"].max()
    print(f"\nOHLCV covers through {ohlcv_window_end.date()}")
    print(f"Signals cover through {signals_window_end.date()}")
    if ohlcv_window_end <= H2_BOUNDARY:
        print("\nNo H2 2022 data in OHLCV — the input CSV still ends in the released "
              "window. The marker has not yet swapped in the H2-extended OHLCV; nothing "
              "to do.")
        return 0
    if signals_window_end <= H2_BOUNDARY:
        print("\nERROR: OHLCV covers H2 but signals do not. The brief requires the "
              "marker to supply both H2-extended OHLCV and H2-extended primary signals.",
              file=sys.stderr)
        return 1

    # Build per-instrument close series + causal σ.
    close_panel = ohlcv.pivot(index="date", columns="instrument", values="close").sort_index()
    signals_long = signals.melt(id_vars="date", var_name="instrument", value_name="signal")
    h2_signals = signals_long[
        (signals_long["date"] > H2_BOUNDARY) & (signals_long["signal"].astype(int) != 0)
    ].copy()
    h2_signals["signal"] = h2_signals["signal"].astype(int)
    print(f"\nH2 non-zero signal events to label: {len(h2_signals)}")

    new_rows: list[dict] = []
    skipped: dict[str, int] = {"incomplete_forward_window": 0, "bad_sigma": 0,
                                "date_not_on_calendar": 0, "no_geometry": 0}
    for inst, ev_g in h2_signals.groupby("instrument"):
        if inst not in close_panel.columns:
            skipped["date_not_on_calendar"] += len(ev_g)
            continue
        geom = geo.loc[geo["instrument"] == inst]
        if geom.empty:
            skipped["no_geometry"] += len(ev_g)
            continue
        pt, sl, h = float(geom.iloc[0]["pt"]), float(geom.iloc[0]["sl"]), int(geom.iloc[0]["h"])
        s = close_panel[inst].dropna().sort_index()
        sigma_series = _causal_sigma(s)
        idx = s.index
        close_arr = s.to_numpy(dtype=float)

        for _, row in ev_g.sort_values("date").iterrows():
            event_date = row["date"]
            pos = idx.get_indexer([event_date])[0]
            if pos < 0:
                skipped["date_not_on_calendar"] += 1
                continue
            sigma = float(sigma_series.iloc[pos]) if pos < len(sigma_series) else np.nan
            res = _resolve_event(
                close_arr=close_arr, p_entry=pos, side=float(row["signal"]),
                sigma=sigma, pt=pt, sl=sl, h=h,
            )
            if res is None:
                if not np.isfinite(sigma) or sigma <= 0:
                    skipped["bad_sigma"] += 1
                else:
                    skipped["incomplete_forward_window"] += 1
                continue
            end_idx, ret, label, touch = res
            new_rows.append({
                "instrument": inst,
                "date": event_date.strftime("%Y-%m-%d"),
                "t1": idx[end_idx].strftime("%Y-%m-%d"),
                "partition": "test",
                "side": float(row["signal"]),
                "sigma": sigma,
                "pt": pt, "sl": sl, "h": h,
                "ret": ret, "label": int(label), "touch": touch,
            })

    print(f"\nResolved H2 events: {len(new_rows)}")
    for k, v in skipped.items():
        if v:
            print(f"  skipped — {k}: {v}")

    if not new_rows:
        print("\nNo H2 rows could be resolved. The marker should verify that the "
              "H2-extended OHLCV and signals are aligned and that the OHLCV has at "
              "least h bars past every event date.")
        return 1

    h2_df = pd.DataFrame(new_rows)
    h2_df["date"] = pd.to_datetime(h2_df["date"])
    h2_df["t1"] = pd.to_datetime(h2_df["t1"])

    # Drop any duplicates that may already be in the shipped CSV (idempotency)
    existing_keys = set(zip(shipped["instrument"], shipped["date"]))
    h2_df["key"] = list(zip(h2_df["instrument"], h2_df["date"]))
    h2_new = h2_df[~h2_df["key"].isin(existing_keys)].drop(columns="key")
    print(f"H2 rows new vs already in CSV: {len(h2_new)} new / "
          f"{len(h2_df) - len(h2_new)} duplicates")
    if h2_new.empty:
        print("CSV already contains every H2 event — nothing to write.")
        return 0

    # Order columns to match shipped CSV exactly.
    h2_new = h2_new[list(shipped.columns)]
    combined = pd.concat([shipped, h2_new], ignore_index=True)
    combined = combined.sort_values(["instrument", "date"]).reset_index(drop=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")
    combined["t1"] = pd.to_datetime(combined["t1"]).dt.strftime("%Y-%m-%d")

    if dry_run:
        print(f"\n[dry-run] would write {len(combined)} rows to {LABELS_CSV} "
              f"({len(shipped)} shipped + {len(h2_new)} H2)")
        return 0

    combined.to_csv(LABELS_CSV, index=False)
    print(f"\nWrote {LABELS_CSV} — {len(combined)} rows "
          f"({len(shipped)} released + {len(h2_new)} H2 test)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve H2 events but do not overwrite the labels CSV")
    args = parser.parse_args()
    return relabel_h2(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
