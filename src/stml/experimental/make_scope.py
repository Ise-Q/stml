"""Per-instrument scope generator — emits `results/submission/instrument_scope.json`.

Plan §3.4 + §5.2. Each instrument's CV embargo is the p90 of
that instrument's own label-window length (``t_end - t_signal`` in trading
days on the instrument's calendar). The wide variation across instruments
(ng1s 33d, ho1s 26d, equity 8-10d) is what the per-instrument embargo
machinery in ``cv.py`` consumes.

Schema (mirrors the reference ``results/instrument_scope.json``):

    {
        "<inst>": {
            "instrument":        "<inst>",
            "asset_class":       "EQ" | "EN" | "ME",
            "n_events":          int,
            "embargo_p90":       int,  # trading days
            "n_eff_gate":        int,  # effective sample size at 90% uniqueness
            "low_power":         bool,
            "p90_label_span":    int,  # alias for embargo_p90, kept for clarity
            "median_label_span": int,
            "median_uniqueness": float,
        },
        ...
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stml.experimental.config import INSTRUMENT_TO_CLASS, INSTRUMENTS
from stml.experimental.data_loader import load_panel, per_instrument_frames

ASSET_CLASS_SHORT = {"equity": "EQ", "energy": "EN", "metals": "ME"}

# Power threshold from reference: any instrument with n_eff < this gets low_power=True.
LOW_POWER_THRESHOLD = 15


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"Could not locate repo root from {here}")


def _trading_day_span(t_signal: pd.Timestamp, t_end: pd.Timestamp, inst_index: pd.DatetimeIndex) -> int:
    """Number of bars on the INSTRUMENT'S OWN trading calendar between
    ``t_signal`` and ``t_end`` (used as a diagnostic, NOT the embargo value)."""
    pos_signal = int(inst_index.searchsorted(t_signal, side="left"))
    pos_end = int(inst_index.searchsorted(t_end, side="left"))
    return max(pos_end - pos_signal, 0)


def _business_day_span(t_signal: pd.Timestamp, t_end: pd.Timestamp) -> int:
    """Number of POOLED business days (Mon-Fri) between t_signal and t_end.

    This is the embargo_p90 unit the convention uses: for thin instruments like ng1s
    that only trade ~1 in 3 business days, the on-instrument h=10 window
    stretches over many more business days on the pooled calendar — that's
    where the per-instrument differentiation comes from.
    """
    return int(np.busday_count(t_signal.date(), t_end.date()))


def build_scope(events: pd.DataFrame) -> dict[str, dict]:
    """Compute the per-instrument scope dict from the S1 events frame.

    For each instrument:
      * embargo_p90 = 90th percentile of label-window length, **measured in
        bars on the instrument's own trading calendar**. This is what the reference
        CV machinery wants (33d for ng1s, 26d for ho1s) — the wide variation
        comes from instruments whose trade calendar has gaps where the h=10
        horizon stretches over more calendar days.
      * n_eff_gate = sum of uniqueness weights (AFML Ch.4 effective sample size).
      * low_power = n_eff < LOW_POWER_THRESHOLD.

    Parameters
    ----------
    events : the S1 events.parquet — needs ``instrument``, ``t_signal``,
        ``t_end``, ``uniqueness_weight`` columns.
    """
    # Build per-instrument trading-day indices from OHLCV.
    ohlcv, signals = load_panel()
    panel = per_instrument_frames(ohlcv, signals)
    inst_indices = {inst: pd.DatetimeIndex(frame.index) for inst, frame in panel.items()}

    out: dict[str, dict] = {}
    for inst in INSTRUMENTS:
        sub = events.loc[events["instrument"] == inst]
        if sub.empty:
            out[inst] = {
                "instrument": inst,
                "asset_class": ASSET_CLASS_SHORT.get(INSTRUMENT_TO_CLASS[inst], "??"),
                "n_events": 0,
                "embargo_p90": 0,
                "n_eff_gate": 0,
                "low_power": True,
                "p90_label_span": 0,
                "median_label_span": 0,
                "median_uniqueness": float("nan"),
            }
            continue

        # POOLED business-day spans (Mon-Fri) — the reference convention. For thin
        # instruments like ng1s the on-instrument 11-bar window can stretch
        # over 30+ business days on the pooled calendar, which is what the
        # per-instrument embargo machinery needs to know about.
        spans = np.array([
            _business_day_span(ts, te) for ts, te in zip(sub["t_signal"], sub["t_end"])
        ])
        span_p90 = int(np.percentile(spans, 90))
        span_med = int(np.percentile(spans, 50))

        # Worst-case held window from the per-instrument geometry: the label
        # could be held for up to h trading days even if most events close
        # early at PT or SL. The embargo must cover this so events starting
        # right after a test fold cannot have their t_end overlap the test
        # block via the forward leak window.
        h_max = int(sub["h"].max()) if "h" in sub.columns else 0
        # Take the MAX of (data-driven p90, the worst-case h, AFML default 10).
        embargo = int(max(span_p90, h_max, 10))

        n_events = int(len(sub))
        uniq = sub["uniqueness_weight"].astype(float)
        median_uniq = float(uniq.median())
        n_eff = int(uniq.sum())
        low_power = bool(n_eff < LOW_POWER_THRESHOLD)

        out[inst] = {
            "instrument": inst,
            "asset_class": ASSET_CLASS_SHORT.get(INSTRUMENT_TO_CLASS[inst], "??"),
            "n_events": n_events,
            "embargo_p90": embargo,
            "n_eff_gate": n_eff,
            "low_power": low_power,
            "p90_label_span": span_p90,
            "median_label_span": span_med,
            "h_max": h_max,
            "median_uniqueness": median_uniq,
        }
    return out


def load_scope(path: Path | None = None) -> dict[str, dict]:
    """Load the scope JSON from disk."""
    if path is None:
        root = _find_repo_root()
        path = root / "results" / "submission" / "instrument_scope.json"
    with open(path) as f:
        return json.load(f)


def embargo_days_map(path: Path | None = None) -> dict[str, int]:
    """Helper for cv.py — returns ``{inst: embargo_p90_days}``."""
    scope = load_scope(path)
    return {inst: int(meta["embargo_p90"]) for inst, meta in scope.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate results/submission/instrument_scope.json"
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    root = _find_repo_root()
    events_path = root / "data" / "events.parquet"
    if not events_path.exists():
        print(f"FATAL: {events_path} missing — run S1 first.")
        return 1
    events = pd.read_parquet(events_path)

    scope = build_scope(events)
    out_path = root / "results" / "submission" / "instrument_scope.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(scope, f, indent=2, sort_keys=True)
    if not args.quiet:
        print(f"Wrote {out_path.relative_to(root)}")
        print()
        print(f"{'inst':6s} {'class':4s} {'n':>4s} {'embargo_p90':>12s} {'n_eff':>6s} {'low_power':>10s}")
        for inst, meta in scope.items():
            print(f"{inst:6s} {meta['asset_class']:4s} {meta['n_events']:>4d} "
                  f"{meta['embargo_p90']:>12d} {meta['n_eff_gate']:>6d} "
                  f"{str(meta['low_power']):>10s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
