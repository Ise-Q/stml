"""S1 runner — load the per-instrument triple-barrier labels into the
canonical events parquet.

Replaces the previous GARCH(1,1) + global ``pt=sl=0.5, h=10`` label
generator. The shipped labelling methodology: per-instrument
``(pt, sl, h)`` geometry selected by adjusted-Sharpe over a 343-geometry
grid (``pt, sl ∈ {0.25, 0.5, 0.75, 1, 1.5, 2, 2.5}`` × ``h ∈ {1, 2, 3, 5,
10, 15, 20}``) with a held-out 2022-H1 validation slice and a
placebo-in-time check at return lags 0 / 1 / 2.

Inputs:

    data/triple_barrier_labels.csv
        Columns: instrument, date, t1, partition, side, sigma, pt, sl, h,
        ret, label, touch.

Outputs:

    data/events.parquet
        Canonical events schema (instrument, t_signal, t_start, t_end, side,
        ret, label, uniqueness_weight, sigma_at_t, barrier_hit) + the team's
        per-instrument geometry columns (pt, sl, h) + `partition` column
        that controls the train/val/test split downstream.

    results/submission/label_outcome_audit.csv
        Per-instrument composition: n_events, n_long/n_short, pos_rate,
        PT/SL/vert counts and fractions, mean_uniqueness, plus the
        adopted (pt, sl, h) geometry.

    results/submission/per_instrument_geometry_summary.csv
        Per-instrument (pt, sl, h) + adjusted-Sharpe context — the "what
        geometry was picked for each instrument and why".

Run:

    uv run python -m stml.experimental.make_labels

Notes:

  * ``t_start`` is the next trading day after ``date`` on the instrument's
    own calendar (entry-at-t+1 convention preserved).
  * ``t_end`` is the CSV's ``t1`` column (first barrier touch or vertical).
  * ``uniqueness_weight`` is computed per instrument as the mean of
    ``1 / concurrency[bar]`` over the held window ``[t_start, t_end]`` —
    AFML Ch.4, recomputed on the spans.
  * The CSV's ``touch`` column uses ``vert``; we rename to ``vertical`` to
    match the downstream consumers (backtest, evaluation).
  * The ``partition`` column drives the train/val/test split in Phase B;
    no global cut is used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stml.experimental.config import INSTRUMENTS, PipelineConfig
from stml.experimental.data_loader import load_panel, per_instrument_frames


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"Could not locate repo root from {here}")


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


_REQUIRED_COLS = (
    "instrument", "date", "t1", "partition",
    "side", "sigma", "pt", "sl", "h", "ret", "label", "touch",
)

_VALID_PARTITIONS = ("train", "val", "test")
_VALID_TOUCHES = ("pt", "sl", "vert")


def _validate_raw(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"labels CSV missing columns: {missing}")
    parts = set(df["partition"].dropna().unique())
    bad_parts = parts - set(_VALID_PARTITIONS)
    if bad_parts:
        raise ValueError(f"unknown partition values: {bad_parts}")
    touches = set(df["touch"].dropna().unique())
    bad_touches = touches - set(_VALID_TOUCHES)
    if bad_touches:
        raise ValueError(f"unknown touch values: {bad_touches}")
    if not df["instrument"].isin(INSTRUMENTS).all():
        bad = sorted(set(df["instrument"]) - set(INSTRUMENTS))
        raise ValueError(f"Unknown instruments in CSV: {bad}")


def _per_instrument_uniqueness(
    inst_events: pd.DataFrame, trading_days: pd.DatetimeIndex,
) -> np.ndarray:
    """AFML Ch.4 uniqueness weights on [t_start, t_end] per instrument.

    Vectorised diff/cumsum trick. Days outside ``trading_days`` are dropped
    from the span (defensive — shouldn't occur if t_start / t_end come from
    the instrument's own calendar).
    """
    n = len(inst_events)
    if n == 0:
        return np.zeros(0, dtype=float)
    pos_map = {ts: i for i, ts in enumerate(trading_days)}
    pos_starts = np.empty(n, dtype=int)
    pos_ends = np.empty(n, dtype=int)
    for i, (ts, te) in enumerate(zip(inst_events["t_start"], inst_events["t_end"])):
        if ts not in pos_map or te not in pos_map:
            # Fallback: clip to nearest available day.
            ts_idx = trading_days.searchsorted(ts, side="left")
            te_idx = trading_days.searchsorted(te, side="right") - 1
            ts_idx = int(np.clip(ts_idx, 0, len(trading_days) - 1))
            te_idx = int(np.clip(te_idx, 0, len(trading_days) - 1))
            pos_starts[i] = ts_idx
            pos_ends[i] = max(te_idx, ts_idx)
        else:
            pos_starts[i] = pos_map[ts]
            pos_ends[i] = pos_map[te]
            if pos_ends[i] < pos_starts[i]:
                pos_ends[i] = pos_starts[i]
    # HALF-OPEN convention [pos_start, pos_end) — matches the t1 semantics
    # and ``backtest.build_position_panel``'s ``< t_end`` clipping. A h=1
    # event has span_len = 1 (just the entry bar t); consecutive h=1 events
    # are disjoint.
    n_bars = len(trading_days)
    delta = np.zeros(n_bars + 1, dtype=int)
    for i in range(n):
        # Defensive: if pos_end <= pos_start, treat as a 1-bar event on the
        # entry bar (so uniqueness is well defined).
        end_excl = max(pos_ends[i], pos_starts[i] + 1)
        delta[pos_starts[i]] += 1
        delta[end_excl] -= 1
    concurrency = np.cumsum(delta)[:n_bars]
    concurrency = np.maximum(concurrency, 1)
    inv_conc = 1.0 / concurrency.astype(float)
    inv_cumsum = np.concatenate([[0.0], np.cumsum(inv_conc)])
    weights = np.empty(n, dtype=float)
    for i in range(n):
        end_excl = max(pos_ends[i], pos_starts[i] + 1)
        span_len = end_excl - pos_starts[i]
        s = inv_cumsum[end_excl] - inv_cumsum[pos_starts[i]]
        weights[i] = s / max(span_len, 1)
    return weights


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def build_events(
    cfg: PipelineConfig | None = None,
    *,
    csv_path: Path | str | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the CSV, map to the canonical events schema, compute uniqueness.

    Returns
    -------
    events : pd.DataFrame
        Canonical events frame plus ``pt, sl, h, partition``.
    audit : pd.DataFrame
        Per-instrument composition.
    geometry : pd.DataFrame
        Per-instrument adopted geometry.
    """
    cfg = cfg or PipelineConfig()
    root = _find_repo_root()
    csv_path = Path(csv_path) if csv_path else (root / "data" / "triple_barrier_labels.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"the labels CSV not found at {csv_path}")

    if verbose:
        print(f"S1 make_labels — loading the per-instrument labels from {csv_path.name}")

    raw = pd.read_csv(csv_path)
    _validate_raw(raw)
    raw["date"] = pd.to_datetime(raw["date"])
    raw["t1"] = pd.to_datetime(raw["t1"])

    # Per-instrument calendars from OHLCV (entry-at-t+1 requires the
    # instrument's own trading-day index, not a global calendar).
    ohlcv, _signals = load_panel()
    panel = per_instrument_frames(ohlcv, _signals)

    parts = []
    audit_rows = []
    geo_rows = []

    for inst in INSTRUMENTS:
        sub = raw.loc[raw["instrument"] == inst].copy()
        if sub.empty:
            continue
        if inst not in panel:
            if verbose:
                print(f"  [WARN] {inst}: no OHLCV calendar; skipping")
            continue

        frame = panel[inst]
        trading_days = pd.DatetimeIndex(frame.index)

        # the convention (PDF): signal observed at close of `date`, position
        # entered at close of `date`, exited at close of `t1`. The "first
        # tradeable bar" is lag-1 = u_{t+1} = log(close_{t+1}/close_t), which
        # requires entry at close(t). So t_start = t_signal (not next-trading-
        # day) — this matches the realised `ret` column in the CSV.
        sub = sub.sort_values("date").reset_index(drop=True)
        sub["t_signal"] = sub["date"]
        sub["t_start"] = sub["date"]
        sub["t_end"] = sub["t1"]
        # Defensive: drop any rows whose date is past the instrument's history.
        in_calendar = sub["t_signal"].isin(trading_days)
        if not in_calendar.all():
            dropped = int((~in_calendar).sum())
            if verbose:
                print(f"  [WARN] {inst}: dropping {dropped} events whose date is outside the OHLCV calendar")
            sub = sub.loc[in_calendar].reset_index(drop=True)

        # Schema mapping.
        events_inst = pd.DataFrame({
            "instrument": inst,
            "t_signal": sub["t_signal"],
            "t_start": sub["t_start"],
            "t_end": sub["t_end"],
            "side": sub["side"].astype(int),
            "ret": sub["ret"].astype(float),
            "label": sub["label"].astype(int),
            "sigma_at_t": sub["sigma"].astype(float),
            "barrier_hit": sub["touch"].map({"pt": "pt", "sl": "sl", "vert": "vertical"}),
            "pt": sub["pt"].astype(float),
            "sl": sub["sl"].astype(float),
            "h": sub["h"].astype(int),
            "partition": sub["partition"].astype(str),
        })

        # Uniqueness weights on this instrument's calendar.
        events_inst["uniqueness_weight"] = _per_instrument_uniqueness(
            events_inst, trading_days,
        )

        parts.append(events_inst)

        # Audit row.
        n = len(events_inst)
        pt_n = int((events_inst["barrier_hit"] == "pt").sum())
        sl_n = int((events_inst["barrier_hit"] == "sl").sum())
        vert_n = int((events_inst["barrier_hit"] == "vertical").sum())
        n_long = int((events_inst["side"] == 1).sum())
        n_short = int((events_inst["side"] == -1).sum())
        n_label_1 = int(events_inst["label"].sum())
        audit_rows.append({
            "instrument": inst,
            "sigma_source": "f2_vol_20",  # the methodology.
            "n_events": n,
            "n_long": n_long,
            "n_short": n_short,
            "n_label_1": n_label_1,
            "pos_rate": n_label_1 / n if n else float("nan"),
            "n_pt": pt_n, "n_sl": sl_n, "n_vertical": vert_n,
            "frac_pt": pt_n / n if n else float("nan"),
            "frac_sl": sl_n / n if n else float("nan"),
            "frac_vertical": vert_n / n if n else float("nan"),
            "mean_uniqueness": float(events_inst["uniqueness_weight"].mean()) if n else float("nan"),
            "pt_mult": float(events_inst["pt"].iloc[0]),
            "sl_mult": float(events_inst["sl"].iloc[0]),
            "h": int(events_inst["h"].iloc[0]),
            "n_train": int((events_inst["partition"] == "train").sum()),
            "n_val": int((events_inst["partition"] == "val").sum()),
            "n_test": int((events_inst["partition"] == "test").sum()),
        })

        geo_rows.append({
            "instrument": inst,
            "pt": float(events_inst["pt"].iloc[0]),
            "sl": float(events_inst["sl"].iloc[0]),
            "h": int(events_inst["h"].iloc[0]),
            "n_events": n,
            "n_train": int((events_inst["partition"] == "train").sum()),
            "n_val": int((events_inst["partition"] == "val").sum()),
            "n_test": int((events_inst["partition"] == "test").sum()),
            "pos_rate": n_label_1 / n if n else float("nan"),
            "frac_pt": pt_n / n if n else float("nan"),
            "frac_sl": sl_n / n if n else float("nan"),
            "frac_vert": vert_n / n if n else float("nan"),
        })

    if not parts:
        raise RuntimeError("labels CSV produced 0 events for every instrument")

    events_all = pd.concat(parts, ignore_index=True)
    events_all = events_all.sort_values(["instrument", "t_signal"]).reset_index(drop=True)
    audit = pd.DataFrame(audit_rows)
    geometry = pd.DataFrame(geo_rows)
    return events_all, audit, geometry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="S1 runner — load the per-instrument triple-barrier "
                    "labels into the canonical events parquet.",
    )
    ap.add_argument("--csv", type=str, default=None,
                     help="Path to the labels CSV (default: data/triple_barrier_labels.csv).")
    ap.add_argument("--no-persist", action="store_true",
                     help="Don't write parquet/CSV.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    events, audit, geometry = build_events(
        csv_path=args.csv, verbose=not args.quiet,
    )

    print("\n=== Per-instrument audit ===")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(audit.to_string(index=False))

    print("\n=== Per-instrument geometry ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(geometry.to_string(index=False))

    print("\n=== Totals ===")
    print(f"Total events: {len(events)}")
    print(f"Pos rate: {events['label'].mean():.3f}")
    print(f"PT fraction: {(events['barrier_hit'] == 'pt').mean():.3f}")
    print(f"SL fraction: {(events['barrier_hit'] == 'sl').mean():.3f}")
    print(f"Vertical fraction: {(events['barrier_hit'] == 'vertical').mean():.3f}")
    print(f"Partition: train={int((events['partition']=='train').sum())}  "
          f"val={int((events['partition']=='val').sum())}  "
          f"test={int((events['partition']=='test').sum())}")

    # Acceptance gates — adapted to the spec.
    print("\n=== S1 acceptance gates ===")
    n_obs = len(events)
    gate1 = 4800 <= n_obs <= 5000
    print(f"[{'PASS' if gate1 else 'CHECK'}] event count = {n_obs} (target ~4917 from CSV)")
    coverage = sorted(events["instrument"].unique())
    gate2 = set(coverage) == set(INSTRUMENTS)
    print(f"[{'PASS' if gate2 else 'CHECK'}] all 11 instruments present = {gate2}")
    bad_partition = (~events["partition"].isin(("train", "val", "test"))).sum()
    gate3 = bad_partition == 0
    print(f"[{'PASS' if gate3 else 'CHECK'}] no rogue partitions = {gate3}")
    bad_uniq = ((events["uniqueness_weight"] <= 0) | (events["uniqueness_weight"] > 1)).sum()
    gate4 = bad_uniq == 0
    print(f"[{'PASS' if gate4 else 'CHECK'}] uniqueness in (0, 1] for all events = {gate4}")

    if not args.no_persist:
        root = _find_repo_root()
        events_path = root / "data" / "events.parquet"
        audit_path = root / "results" / "submission" / "label_outcome_audit.csv"
        geo_path = root / "results" / "submission" / "per_instrument_geometry_summary.csv"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        events.to_parquet(events_path, index=False)
        audit.to_csv(audit_path, index=False, float_format="%.6f")
        geometry.to_csv(geo_path, index=False, float_format="%.6f")
        print(f"\nWrote {events_path.relative_to(root)} ({len(events)} rows)")
        print(f"Wrote {audit_path.relative_to(root)} ({len(audit)} rows)")
        print(f"Wrote {geo_path.relative_to(root)} ({len(geometry)} rows)")

    all_gates = gate1 and gate2 and gate3 and gate4
    return 0 if all_gates else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
