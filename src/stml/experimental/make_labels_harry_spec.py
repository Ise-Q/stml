"""S1 variant runner — Harry's label spec (pt=1.5, sl=1.0, cumulative h-day GARCH).

Per branch_descriptions §4.9, Harry's ``new_work/triple_barrier.py`` uses
asymmetric barriers (``pt=1.5, sl=1.0``) with **cumulative h-day GARCH variance**
as the scale (not one-step-ahead daily σ̂ × √h). This is a different label
spec from our plan §3.2 default (pt=sl=0.5, daily GARCH).

Built as an ablation per the user's directive to integrate the best
configurations from other branches AFTER testing. Output goes to a separate
parquet so the plan-spec labels remain the canonical version.

Why this matters: Harry's per-instrument AUCs (nq1s 0.689, hg1s 0.604) are
above ours (0.537, 0.533) on those instruments. The hypothesis is that wider
asymmetric barriers + cumulative GARCH produce labels with more directional
content for momentum-driven instruments.

Output: ``data/sreeram_experimental_events_harry_spec.parquet``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stml.experimental.config import INSTRUMENTS, PipelineConfig
from stml.experimental.data_loader import load_panel, per_instrument_frames
from stml.experimental.labels import LabelConfig, triple_barrier_labels
from stml.experimental.volatility import (
    ewma_daily_sigma,
    garch_h_cumulative_sigma,
    garch_sigma,
)


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"Could not locate repo root from {here}")


def _compute_sigma_harry(close: pd.Series, *, h: int, signal_dates: pd.DatetimeIndex,
                          refit: int = 21, min_obs: int = 500, max_window: int = 2000) -> pd.Series:
    """Harry's cumulative h-day GARCH σ̂ (vs our default one-step-ahead daily)."""
    if signal_dates.empty:
        return ewma_daily_sigma(close)
    first_signal = pd.Timestamp(signal_dates.min())
    warm_start = first_signal - pd.DateOffset(years=10)
    close_trimmed = close.loc[close.index >= warm_start]
    if len(close_trimmed) < min_obs + 50:
        return ewma_daily_sigma(close)
    try:
        sigma_h = garch_h_cumulative_sigma(
            close_trimmed, h=h, refit=refit, min_obs=min_obs, max_window=max_window
        )
        sigma_h = sigma_h.reindex(close.index)
        if sigma_h.loc[signal_dates].isna().any():
            ewma = ewma_daily_sigma(close)
            # Convert daily EWMA to h-day equivalent for fallback.
            ewma_h = ewma * np.sqrt(h)
            sigma_h = sigma_h.fillna(ewma_h)
        return sigma_h
    except Exception:
        ewma = ewma_daily_sigma(close)
        return ewma * np.sqrt(h)


def build_events_harry_spec(*, pt_mult: float = 1.5, sl_mult: float = 1.0, h: int = 10,
                             verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build labelled events with Harry's spec: pt=1.5/sl=1.0 + cumulative h-day GARCH.

    Note on barrier scale: Harry's labels.py uses ``barrier_width = pt_mult × σ̂``
    where ``σ̂`` is the h-day cumulative GARCH. Our triple_barrier_labels expects
    daily σ̂ and multiplies by sqrt(h). To match Harry's convention we pass
    ``σ̂_daily_equivalent = σ̂_h / sqrt(h)`` so the barrier becomes
    ``pt_mult × (σ̂_h / sqrt(h)) × sqrt(h) = pt_mult × σ̂_h`` — same as Harry.
    """
    cfg = PipelineConfig()
    label_cfg = LabelConfig(pt_mult=pt_mult, sl_mult=sl_mult, max_holding=h)
    ohlcv, signals = load_panel()
    panel = per_instrument_frames(ohlcv, signals)

    parts = []
    audit_rows = []
    for inst in INSTRUMENTS:
        if inst not in panel:
            continue
        frame = panel[inst]
        close = frame["close"].dropna()
        signal = frame["signal"].reindex(close.index).fillna(0).astype(int)
        signal_dates = close.index[signal != 0]
        if verbose:
            print(f"  {inst}: {len(close)} bars, {len(signal_dates)} signal dates")
        sigma_h = _compute_sigma_harry(close, h=h, signal_dates=pd.DatetimeIndex(signal_dates))
        # Convert to daily equivalent for our labeller's barrier-width formula.
        sigma_daily_equiv = sigma_h / np.sqrt(h)
        events = triple_barrier_labels(
            close=close, signal=signal, sigma=sigma_daily_equiv,
            instrument=inst, config=label_cfg,
        )
        n = len(events)
        if n:
            parts.append(events)
            pt = int((events["barrier_hit"] == "pt").sum())
            sl = int((events["barrier_hit"] == "sl").sum())
            vert = int((events["barrier_hit"] == "vertical").sum())
            n_long = int((events["side"] == +1).sum())
            n_short = int((events["side"] == -1).sum())
            n_label_1 = int(events["label"].sum())
        else:
            pt = sl = vert = n_long = n_short = n_label_1 = 0
        audit_rows.append({
            "instrument": inst, "n_events": n, "n_long": n_long, "n_short": n_short,
            "n_label_1": n_label_1, "pos_rate": (n_label_1 / n) if n else float("nan"),
            "n_pt": pt, "n_sl": sl, "n_vertical": vert,
            "frac_pt": (pt / n) if n else float("nan"),
            "frac_sl": (sl / n) if n else float("nan"),
            "frac_vertical": (vert / n) if n else float("nan"),
        })

    events_all = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    events_all = events_all.sort_values(["instrument", "t_signal"]).reset_index(drop=True)
    return events_all, pd.DataFrame(audit_rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S1-variant: build Harry-spec labels")
    ap.add_argument("--pt", type=float, default=1.5)
    ap.add_argument("--sl", type=float, default=1.0)
    ap.add_argument("--h", type=int, default=10)
    args = ap.parse_args(argv)

    print(f"Building labels with Harry's spec: pt={args.pt}, sl={args.sl}, h={args.h}, GARCH-h cumulative")
    t0 = time.time()
    events, audit = build_events_harry_spec(pt_mult=args.pt, sl_mult=args.sl, h=args.h)
    print(f"Built in {time.time() - t0:.1f}s")
    print("\nPer-instrument audit:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(audit.to_string(index=False))
    print(f"\nTotal events: {len(events)}")
    print(f"Pooled vertical fraction: {(events['barrier_hit'] == 'vertical').mean():.3f}")
    print(f"Pooled PT fraction: {(events['barrier_hit'] == 'pt').mean():.3f}")
    print(f"Pooled SL fraction: {(events['barrier_hit'] == 'sl').mean():.3f}")
    print(f"Pooled pos rate: {events['label'].mean():.3f}")

    root = _find_repo_root()
    events_path = root / "data" / "sreeram_experimental_events_harry_spec.parquet"
    audit_path = root / "results" / "sreeram_experimental" / "label_outcome_audit_harry_spec.csv"
    events.to_parquet(events_path, index=False)
    audit.to_csv(audit_path, index=False, float_format="%.6f")
    print(f"\nWrote {events_path.relative_to(root)} ({len(events)} rows)")
    print(f"Wrote {audit_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
