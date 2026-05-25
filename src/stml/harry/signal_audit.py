"""signal_audit.py — direction of the primary signal, measured.

Step 1 of Harry's contribution. Resolves the team's trend-vs-reversion
contradiction with one tight, independent measurement per instrument.

For each of the 11 instruments we compute:

* ``corr(s_t, r_{t-k})`` for ``k ∈ {-5, -3, -1, 0, 1, 3, 5, 10, 20}``.

  - ``k > 0`` measures the relationship with TRAILING returns. A positive
    value means the signal LOADS POSITIVELY on past returns (momentum /
    trend-following construction); a negative value means the signal
    loads negatively on past returns (counter-trend / mean-reversion
    construction).
  - ``k = 0`` is the contemporaneous correlation (typically dominated by
    the construction lag, not the PnL convention).
  - ``k < 0`` is the FORWARD return ``r_{t+|k|}``. A positive value at
    ``k = -1`` is the empirical test of the next-day PnL convention:
    ``PnL_t = s_t · r_{t+1}``.

* ``mean(s_t · r_{t+1})``           — next-day PnL convention.
* ``mean(s_t · cumret_{t+1..t+h})`` — h-day forward PnL (h defaults to 10).
* ``hit_rate_h`` = ``P(s_t · cumret_{t+1..t+h} > 0 | s_t ≠ 0)`` — fraction
  of participating signals that pay off over the next h days.

Bootstrap 95 % CIs are produced by a **moving-block bootstrap** on the
aligned (s, pre-computed-lag-returns) panel. Block size defaults to 20
trading days ≈ the signal run-length p90 in the released data, configurable.
Block-bootstrap preserves the autocorrelation in both the piecewise-constant
signal and the returns; an i.i.d. bootstrap would massively understate CI
widths because the signal has runs of length 5–30+ days.

Per-instrument sign tag (the headline column):

    sign_label =
        "trend"          if mean_trail_corr > +trend_threshold
        "mean_reverting" if mean_trail_corr < -trend_threshold
        "mixed"          otherwise

    mean_trail_corr = mean( corr_trail_1, corr_trail_5, corr_trail_10,
                            corr_trail_20 )

Convention reminder:
    r_t                = log(close_t / close_{t-1})
    cumret_{t+1..t+h}  = r_{t+1} + … + r_{t+h}
                       = log(close_{t+h} / close_t)

CLI::

    python -m stml.harry.signal_audit --out results/harry/signal_direction.csv

Programmatic::

    from stml.harry.signal_audit import audit_all
    df = audit_all(ohlcv, signals, h=10, n_boot=1000, seed=42)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ASSET_CLASSES: dict[str, str] = {
    "es1s": "equity",   "nq1s": "equity",   "fesx1s": "equity",
    "cl1s": "energy",   "ho1s": "energy",   "rb1s": "energy",   "ng1s": "energy",
    "gc1s": "metals",   "si1s": "metals",   "hg1s": "metals",   "pl1s": "metals",
}

INSTRUMENTS: list[str] = list(ASSET_CLASSES.keys())

DEFAULT_LAGS: tuple[int, ...] = (-5, -3, -1, 0, 1, 3, 5, 10, 20)
DEFAULT_H: int = 10
DEFAULT_N_BOOT: int = 1000
DEFAULT_BLOCK_SIZE: int = 20
DEFAULT_TREND_THRESHOLD: float = 0.05


# --------------------------------------------------------------------------- #
# Core scalar statistics                                                       #
# --------------------------------------------------------------------------- #
def _log_returns(close: pd.Series) -> pd.Series:
    """``r_t = log(close_t / close_{t-1})``; first value is NaN."""
    close = close.sort_index()
    return np.log(close).diff().rename("r")


def _forward_cumret(r: pd.Series, h: int) -> pd.Series:
    """``cumret_{t+1..t+h}`` indexed at ``t``. NaN for the last ``h`` rows."""
    csum = r.cumsum()
    return (csum.shift(-h) - csum).rename(f"cumret_h{h}")


def _corr_col(k: int) -> str:
    """Stable column name for ``corr(s_t, r_{t-k})``.

    k > 0 → trailing return (``corr_trail_{k}``);
    k = 0 → contemporaneous (``corr_contemp_0``);
    k < 0 → forward return  (``corr_fwd_{|k|}``).
    """
    if k > 0:
        return f"corr_trail_{k}"
    if k < 0:
        return f"corr_fwd_{-k}"
    return "corr_contemp_0"


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    """Pearson correlation that returns NaN for degenerate inputs."""
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 3:
        return float("nan")
    a, b = df.iloc[:, 0], df.iloc[:, 1]
    if a.std(ddof=0) == 0 or b.std(ddof=0) == 0:
        return float("nan")
    return float(a.corr(b))


def _lag_corr(s: pd.Series, r: pd.Series, k: int) -> float:
    """``corr(s_t, r_{t-k})`` over aligned non-NaN dates."""
    return _safe_corr(s, r.shift(k))


# --------------------------------------------------------------------------- #
# Frame builder + statistics                                                   #
# --------------------------------------------------------------------------- #
_FLOAT_STATS: tuple[str, ...] = ("mean_pnl_next_day", "mean_pnl_h", "hit_rate_h")


def _build_aligned_frame(
    s: pd.Series, r: pd.Series, lags: tuple[int, ...], h: int
) -> pd.DataFrame:
    """Pre-compute everything per-position so bootstrap is cheap.

    Columns produced:
      s                — the signal at ``t``.
      r_lag_{k}        — ``r_{t-k}`` for each ``k`` in ``lags``.
      pnl_next_day     — ``s_t · r_{t+1}``.
      pnl_h            — ``s_t · cumret_{t+1..t+h}``.
      bet              — 1 if ``s_t != 0`` else 0.
      hit_h            — 1 if ``bet`` and ``pnl_h > 0`` else 0.
    """
    out = pd.DataFrame(index=s.index)
    out["s"] = s.astype("float64")
    for k in lags:
        out[f"r_lag_{k}"] = r.shift(k)
    out["pnl_next_day"] = s * r.shift(-1)
    out["pnl_h"] = s * _forward_cumret(r, h)
    out["bet"] = (s.fillna(0) != 0).astype("int64")
    out["hit_h"] = ((out["pnl_h"] > 0) & out["bet"].astype(bool)).astype("int64")
    return out


def _statistics_from_frame(
    frame: pd.DataFrame, lags: tuple[int, ...]
) -> dict[str, float]:
    """Compute the scalar statistics from a (possibly resampled) frame."""
    out: dict[str, float] = {}
    for k in lags:
        out[_corr_col(k)] = _safe_corr(frame["s"], frame[f"r_lag_{k}"])

    p_nd = frame["pnl_next_day"].dropna()
    out["mean_pnl_next_day"] = float(p_nd.mean()) if len(p_nd) else float("nan")

    p_h = frame["pnl_h"].dropna()
    out["mean_pnl_h"] = float(p_h.mean()) if len(p_h) else float("nan")

    bet_mask = frame["bet"].astype(bool) & frame["pnl_h"].notna()
    out["hit_rate_h"] = (
        float(frame.loc[bet_mask, "hit_h"].mean()) if bet_mask.sum() else float("nan")
    )
    return out


# --------------------------------------------------------------------------- #
# Moving-block bootstrap                                                       #
# --------------------------------------------------------------------------- #
def _moving_block_indices(
    n: int, block_size: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate ``(n_boot, n)`` resample-index matrix via moving-block bootstrap.

    Each bootstrap row is the concatenation of ``ceil(n / block_size)`` blocks
    of length ``block_size`` chosen uniformly at random from the legal start
    positions ``[0, n - block_size]`` (with replacement), then truncated to
    length ``n``. Standard moving-block bootstrap (Künsch 1989).
    """
    if n <= 0:
        return np.empty((n_boot, 0), dtype=np.int64)
    block_size = max(1, min(block_size, n))
    n_blocks = (n + block_size - 1) // block_size
    starts = rng.integers(0, n - block_size + 1, size=(n_boot, n_blocks))
    offsets = np.arange(block_size, dtype=np.int64)
    idx = starts[:, :, None] + offsets[None, None, :]
    idx = idx.reshape(n_boot, n_blocks * block_size)
    return idx[:, :n]


def _bootstrap_ci(
    frame: pd.DataFrame,
    lags: tuple[int, ...],
    block_size: int,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    """Return ``{stat: (lo, hi)}`` for every float statistic."""
    n = len(frame)
    if n < 2:
        return {}
    idx_matrix = _moving_block_indices(n, block_size, n_boot, rng)
    cols = [_corr_col(k) for k in lags] + list(_FLOAT_STATS)
    records: list[dict[str, float]] = []
    for ix in idx_matrix:
        records.append(_statistics_from_frame(frame.iloc[ix], lags))
    boot_df = pd.DataFrame(records, columns=cols)
    return {
        c: (
            float(boot_df[c].quantile(0.025)),
            float(boot_df[c].quantile(0.975)),
        )
        for c in cols
    }


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def audit_instrument(
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    instrument: str,
    *,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    h: int = DEFAULT_H,
    n_boot: int = DEFAULT_N_BOOT,
    block_size: int = DEFAULT_BLOCK_SIZE,
    seed: int = 42,
    trend_threshold: float = DEFAULT_TREND_THRESHOLD,
) -> dict:
    """Per-instrument audit row. See module docstring for the convention."""
    if instrument not in signals.columns:
        raise KeyError(f"Instrument {instrument!r} not in signals columns")

    inst_rows = ohlcv.loc[ohlcv["instrument"] == instrument, ["date", "close"]]
    if inst_rows.empty:
        raise KeyError(f"Instrument {instrument!r} not in ohlcv")
    inst_rows = inst_rows.copy()
    inst_rows["date"] = pd.to_datetime(inst_rows["date"])
    close = inst_rows.sort_values("date").set_index("date")["close"]
    r = _log_returns(close)

    sigs = signals.copy()
    sigs["date"] = pd.to_datetime(sigs["date"])
    s = sigs.set_index("date")[instrument].sort_index().astype("float64")

    # Restrict frame to dates with both a signal and a return defined.
    s_aligned, r_aligned = s.align(r, join="inner")
    frame = _build_aligned_frame(s_aligned, r_aligned, lags, h)
    # Drop rows where the signal is NaN (outside the released window) — those
    # carry no information about the convention either way.
    frame = frame.dropna(subset=["s"])

    point = _statistics_from_frame(frame, lags)

    rng = np.random.default_rng(seed)
    ci = _bootstrap_ci(frame, lags, block_size, n_boot, rng)

    # Build the row in a stable column order.
    row: dict[str, float | str | int] = {
        "instrument": instrument,
        "asset_class": ASSET_CLASSES.get(instrument, "unknown"),
        "n_signal_dates": int(frame["s"].notna().sum()),
        "n_bets": int(frame["bet"].sum()),
    }
    for k in lags:
        col = _corr_col(k)
        row[col] = point[col]
        lo, hi = ci.get(col, (float("nan"), float("nan")))
        row[f"{col}_lo"] = lo
        row[f"{col}_hi"] = hi
    for stat in _FLOAT_STATS:
        row[stat] = point[stat]
        lo, hi = ci.get(stat, (float("nan"), float("nan")))
        row[f"{stat}_lo"] = lo
        row[f"{stat}_hi"] = hi

    # Sign label.
    trail_keys = [_corr_col(k) for k in (1, 5, 10, 20) if k in lags]
    trail_vals = [point[c] for c in trail_keys if not np.isnan(point[c])]
    mean_trail = float(np.mean(trail_vals)) if trail_vals else float("nan")
    if not np.isfinite(mean_trail):
        sign = "n/a"
    elif mean_trail > trend_threshold:
        sign = "trend"
    elif mean_trail < -trend_threshold:
        sign = "mean_reverting"
    else:
        sign = "mixed"
    row["mean_trail_corr"] = mean_trail
    row["sign_label"] = sign
    return row


def audit_all(
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    h: int = DEFAULT_H,
    n_boot: int = DEFAULT_N_BOOT,
    block_size: int = DEFAULT_BLOCK_SIZE,
    seed: int = 42,
    trend_threshold: float = DEFAULT_TREND_THRESHOLD,
) -> pd.DataFrame:
    """Audit every instrument in ``INSTRUMENTS`` present in ``signals``.

    Bootstrap seeds are decorrelated across instruments by adding the
    instrument index so a teammate inspecting one row can reproduce just
    that row without rerunning the others.
    """
    rows: list[dict] = []
    for i, inst in enumerate(INSTRUMENTS):
        if inst not in signals.columns:
            continue
        rows.append(
            audit_instrument(
                ohlcv,
                signals,
                inst,
                lags=lags,
                h=h,
                n_boot=n_boot,
                block_size=block_size,
                seed=seed + i,
                trend_threshold=trend_threshold,
            )
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Signal-direction audit (Step 1 of Harry's contribution).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/harry/signal_direction.csv"),
        help="Destination CSV (default: results/harry/signal_direction.csv)",
    )
    parser.add_argument("--h", type=int, default=DEFAULT_H)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--trend-threshold",
        type=float,
        default=DEFAULT_TREND_THRESHOLD,
        help="|mean trailing corr| threshold for the trend / mean-reverting tag.",
    )
    args = parser.parse_args(argv)

    # Imported here so unit tests don't pay the cost.
    from stml.io import load_clean_data

    ohlcv, signals = load_clean_data()
    df = audit_all(
        ohlcv,
        signals,
        h=args.h,
        n_boot=args.n_boot,
        block_size=args.block_size,
        seed=args.seed,
        trend_threshold=args.trend_threshold,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(df)} rows)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
