"""S1 runner — produce ``data/sreeram_experimental_events.parquet`` +
``results/sreeram_experimental/label_outcome_audit.csv`` for plan §8 acceptance.

Run via:

    uv run python -m stml.experimental.make_labels

Sequence:

1. Load cleaned OHLCV + wide signals via :func:`stml.experimental.data_loader.load_panel`.
2. Per instrument:
   a) Compute daily one-step-ahead GARCH(1,1) σ̂ over the close series (truncated
      to ~10y before the first signal date for tractable wall-clock).
   b) Run :func:`stml.experimental.labels.triple_barrier_labels` at the plan §3.2
      default ``pt=sl=0.5, h=10``.
3. Concatenate events into one frame.
4. Compute the label-outcome audit (PT / SL / VERT composition per instrument).
5. Persist ``data/sreeram_experimental_events.parquet`` and
   ``results/sreeram_experimental/label_outcome_audit.csv``.

Plan §8 S1 acceptance gates (verified at the end):
   * Total event count matches Harry's ``events.csv`` (4886) to within ±50.
   * Vertical-barrier fraction < 65 % per instrument at ``pt=sl=0.5``.

If the GARCH fit fails for a particular instrument, the runner falls back to
the EWMA-σ̂ surrogate (``ewma_daily_sigma`` with span 100) for that instrument
and notes the substitution in the audit CSV.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from stml.experimental.config import INSTRUMENTS, PipelineConfig
from stml.experimental.data_loader import load_panel, per_instrument_frames
from stml.experimental.labels import LabelConfig, triple_barrier_labels
from stml.experimental.volatility import ewma_daily_sigma, garch_sigma


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"Could not locate repo root from {here}")


def _compute_sigma(
    close: pd.Series,
    *,
    signal_dates: pd.DatetimeIndex,
    cfg: PipelineConfig,
    verbose: bool = True,
) -> tuple[pd.Series, str]:
    """Compute the per-instrument daily σ̂, falling back to EWMA on GARCH failure.

    Truncates the close series to start ~10 years before the first signal date
    so the GARCH fits remain tractable (the modelling window is 2.5y so 10y of
    warm-up dwarfs it).

    Returns
    -------
    (sigma, method) where method is 'garch' on success, 'ewma' on fallback.
    """
    if signal_dates.empty:
        return ewma_daily_sigma(close), "ewma"

    first_signal = pd.Timestamp(signal_dates.min())
    warm_start = first_signal - pd.DateOffset(years=10)
    close_trimmed = close.loc[close.index >= warm_start]
    # Need at least min_obs bars to attempt the GARCH fit.
    if len(close_trimmed) < cfg.garch_min_obs + 50:
        # Too thin for GARCH — surrogate.
        return ewma_daily_sigma(close), "ewma"

    if cfg.sigma_source == "garch":
        try:
            t0 = time.time()
            sigma = garch_sigma(
                close_trimmed,
                refit=cfg.garch_refit_every,
                min_obs=cfg.garch_min_obs,
                max_window=cfg.garch_max_window,
            )
            if verbose:
                print(f"    GARCH fit time: {time.time() - t0:.1f}s "
                      f"(over {len(close_trimmed)} bars)")
            # Reindex to full close.index — pre-warm-start σ̂ is NaN.
            sigma = sigma.reindex(close.index)
            # Defensive: any NaN inside the modelling window is filled with EWMA.
            if sigma.loc[signal_dates].isna().any():
                ewma = ewma_daily_sigma(close)
                sigma = sigma.fillna(ewma)
            return sigma, "garch"
        except Exception as exc:
            if verbose:
                print(f"    GARCH failed ({exc.__class__.__name__}: {exc}); EWMA fallback.")
            return ewma_daily_sigma(close), "ewma"

    return ewma_daily_sigma(close), "ewma"


def build_events(
    cfg: PipelineConfig | None = None,
    *,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the canonical events frame + per-instrument audit table.

    Returns
    -------
    events : pd.DataFrame
        One row per labelled event, with the schema from
        :func:`stml.experimental.labels.triple_barrier_labels`.
    audit : pd.DataFrame
        Per-instrument composition: n_events, n_long, n_short, n_label_1,
        pt / sl / vertical counts and fractions, sigma_source used.
    """
    cfg = cfg or PipelineConfig()
    label_cfg = LabelConfig(
        pt_mult=cfg.pt_mult, sl_mult=cfg.sl_mult, max_holding=cfg.max_holding
    )

    if verbose:
        print(f"S1 make_labels — pt={cfg.pt_mult}, sl={cfg.sl_mult}, h={cfg.max_holding}, "
              f"sigma_source={cfg.sigma_source}")

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
            print(f"  {inst}: {len(close)} bars, {len(signal_dates)} non-zero signal dates")

        sigma, method = _compute_sigma(
            close=close, signal_dates=pd.DatetimeIndex(signal_dates), cfg=cfg, verbose=verbose
        )

        events = triple_barrier_labels(
            close=close, signal=signal, sigma=sigma, instrument=inst, config=label_cfg
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

        audit_rows.append(
            {
                "instrument": inst,
                "sigma_source": method,
                "n_events": n,
                "n_long": n_long,
                "n_short": n_short,
                "n_label_1": n_label_1,
                "pos_rate": (n_label_1 / n) if n else float("nan"),
                "n_pt": pt,
                "n_sl": sl,
                "n_vertical": vert,
                "frac_pt": (pt / n) if n else float("nan"),
                "frac_sl": (sl / n) if n else float("nan"),
                "frac_vertical": (vert / n) if n else float("nan"),
                "mean_uniqueness": (
                    float(events["uniqueness_weight"].mean()) if n else float("nan")
                ),
            }
        )

    events_all = (
        pd.concat(parts, ignore_index=True)
        if parts
        else triple_barrier_labels(
            close=pd.Series(dtype=float),
            signal=pd.Series(dtype=int),
            sigma=pd.Series(dtype=float),
            instrument="x",
        )
    )
    events_all = events_all.sort_values(["instrument", "t_signal"]).reset_index(drop=True)
    audit = pd.DataFrame(audit_rows)
    return events_all, audit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="S1 runner — build triple-barrier events with t+1 entry "
                    "(plan §8 Stage 1 deliverable)."
    )
    ap.add_argument("--pt-mult", type=float, default=PipelineConfig().pt_mult)
    ap.add_argument("--sl-mult", type=float, default=PipelineConfig().sl_mult)
    ap.add_argument("--h", type=int, default=PipelineConfig().max_holding)
    ap.add_argument(
        "--sigma-source", choices=["garch", "ewma"], default=PipelineConfig().sigma_source
    )
    ap.add_argument("--no-persist", action="store_true", help="Don't write parquet/CSV.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg = PipelineConfig(
        pt_mult=args.pt_mult,
        sl_mult=args.sl_mult,
        max_holding=args.h,
        sigma_source=args.sigma_source,
    )
    events, audit = build_events(cfg, verbose=not args.quiet)

    print("\n=== Per-instrument audit ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(audit.to_string(index=False))
    print(f"\n=== Totals ===")
    print(f"Total events: {len(events)}")
    print(f"Pos rate: {events['label'].mean():.3f}")
    print(f"PT fraction: {(events['barrier_hit'] == 'pt').mean():.3f}")
    print(f"SL fraction: {(events['barrier_hit'] == 'sl').mean():.3f}")
    print(f"Vertical fraction: {(events['barrier_hit'] == 'vertical').mean():.3f}")

    # Acceptance gates.
    print("\n=== Plan §8 S1 acceptance gates ===")
    target_n = 4886
    n_obs = len(events)
    delta_n = n_obs - target_n
    gate1 = abs(delta_n) <= 50
    print(f"[{'PASS' if gate1 else 'CHECK'}] event count = {n_obs} (target {target_n} ±50, "
          f"delta = {delta_n:+d})")
    vert_frac = (events["barrier_hit"] == "vertical").mean()
    max_inst_vert = audit["frac_vertical"].max()
    gate2 = float(max_inst_vert) < 0.65
    print(f"[{'PASS' if gate2 else 'CHECK'}] per-instrument vertical fraction max = "
          f"{max_inst_vert:.3f} (< 0.65 target)")
    print(f"        pooled vertical fraction = {vert_frac:.3f}")

    if not args.no_persist:
        root = _find_repo_root()
        events_path = root / "data" / "sreeram_experimental_events.parquet"
        audit_path = root / "results" / "sreeram_experimental" / "label_outcome_audit.csv"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        events.to_parquet(events_path, index=False)
        audit.to_csv(audit_path, index=False, float_format="%.6f")
        print(f"\nWrote {events_path.relative_to(root)} ({len(events)} rows)")
        print(f"Wrote {audit_path.relative_to(root)} ({len(audit)} rows)")

    return 0 if (gate1 and gate2) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
