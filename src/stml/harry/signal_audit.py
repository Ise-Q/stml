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

Per-instrument sign tags (three columns in the output CSV):

    tag             — canonical, uses ``mean_trail_corr``.
    tag_trail_1     — same threshold but only on ``corr_trail_1``.
    tag_trail_h10   — same threshold but only on ``corr_trail_10``.

    classification rule (applied identically to all three):
        "trend"          if value > +trend_threshold
        "mean_reverting" if value < -trend_threshold
        "mixed"          otherwise

    mean_trail_corr = mean( corr_trail_1, corr_trail_5, corr_trail_10,
                            corr_trail_20 )

The per-horizon tags exist to reconcile with the signal-deep-dive branch's
"10/11 mean-reversion" headline (which is driven by trail_1) and to show
that the trail_10 picture is materially weaker — relevant because we use
h=10 barriers, so the trail_10 signal is the structurally-aligned one.

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
# Tagging                                                                      #
# --------------------------------------------------------------------------- #
def _classify_tag(value: float, threshold: float) -> str:
    """Threshold-based sign tag used by the canonical and per-horizon labels.

    Returns one of ``"trend"``, ``"mean_reverting"``, ``"mixed"``, or
    ``"n/a"`` (the last only when ``value`` is non-finite). ``threshold``
    is applied symmetrically: ``value > +threshold`` ⇒ trend,
    ``value < -threshold`` ⇒ mean_reverting, else mixed.
    """
    if not np.isfinite(value):
        return "n/a"
    if value > threshold:
        return "trend"
    if value < -threshold:
        return "mean_reverting"
    return "mixed"


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

    # Sign tags — canonical multi-horizon plus per-horizon variants.
    trail_keys = [_corr_col(k) for k in (1, 5, 10, 20) if k in lags]
    trail_vals = [point[c] for c in trail_keys if not np.isnan(point[c])]
    mean_trail = float(np.mean(trail_vals)) if trail_vals else float("nan")
    row["mean_trail_corr"] = mean_trail
    row["tag"] = _classify_tag(mean_trail, trend_threshold)
    # Per-horizon: same threshold, only the single-lag correlation.
    row["tag_trail_1"] = _classify_tag(point.get(_corr_col(1), float("nan")), trend_threshold)
    row["tag_trail_h10"] = _classify_tag(point.get(_corr_col(10), float("nan")), trend_threshold)
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
# Stability check (descriptive only — not a train/test split)                  #
# --------------------------------------------------------------------------- #
_STABILITY_METRICS: tuple[str, ...] = (
    "corr_trail_1",
    "corr_trail_5",
    "corr_trail_10",
    "corr_trail_20",
    "corr_fwd_1",
    "mean_pnl_next_day",
    "mean_pnl_h",
    "hit_rate_h",
)


def _split_signals_halves(
    signals: pd.DataFrame, split_date: pd.Timestamp | None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Cut ``signals`` into two halves at ``split_date`` (default: median).

    Returns ``(first_half, second_half, resolved_split_date)``. The split is
    over signal *dates*, not over a random shuffle: ``first_half`` is dates
    strictly less than ``split_date`` and ``second_half`` is dates greater
    than or equal to it.
    """
    sigs = signals.copy()
    sigs["date"] = pd.to_datetime(sigs["date"])
    sigs = sigs.sort_values("date").reset_index(drop=True)
    if split_date is None:
        split_date = pd.Timestamp(sigs["date"].median()).normalize()
    else:
        split_date = pd.Timestamp(split_date).normalize()
    first = sigs.loc[sigs["date"] < split_date].copy()
    second = sigs.loc[sigs["date"] >= split_date].copy()
    return first, second, split_date


def _safe_ci_excludes_zero(lo: float, hi: float) -> bool:
    """``True`` if ``(lo, hi)`` is a finite interval that does not span zero."""
    if not np.isfinite(lo) or not np.isfinite(hi):
        return False
    return lo > 0 or hi < 0


def audit_stability(
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    split_date: pd.Timestamp | None = None,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    h: int = DEFAULT_H,
    n_boot: int = DEFAULT_N_BOOT,
    block_size: int = DEFAULT_BLOCK_SIZE,
    seed: int = 42,
    trend_threshold: float = DEFAULT_TREND_THRESHOLD,
    metrics: tuple[str, ...] = _STABILITY_METRICS,
) -> pd.DataFrame:
    """Re-run the audit on the full window and on each half.

    This is a **descriptive stability exhibit**, not a train/test split: no
    model decisions are conditioned on the result and the held-out half of
    2022 (the grader's hidden block) is not touched.

    Returns a long DataFrame with one row per ``(instrument, metric)`` and
    columns:

      instrument        — ticker.
      metric            — one of ``_STABILITY_METRICS``.
      full_window       — statistic on the full released signal window.
      first_half        — statistic on dates ``< split_date``.
      second_half       — statistic on dates ``>= split_date``.
      sign_flip_flag    — ``True`` iff the two halves have opposite signs
                          AND both bootstrap 95 % CIs exclude zero (a
                          stricter "real" sign-flip rather than a noise flip).

    ``split_date`` defaults to the median signal date (~2021-09 for the
    released window).
    """
    first, second, _ = _split_signals_halves(signals, split_date)

    full = audit_all(
        ohlcv, signals,
        lags=lags, h=h, n_boot=n_boot, block_size=block_size,
        seed=seed, trend_threshold=trend_threshold,
    ).set_index("instrument")
    first_df = audit_all(
        ohlcv, first,
        lags=lags, h=h, n_boot=n_boot, block_size=block_size,
        seed=seed, trend_threshold=trend_threshold,
    ).set_index("instrument")
    second_df = audit_all(
        ohlcv, second,
        lags=lags, h=h, n_boot=n_boot, block_size=block_size,
        seed=seed + 7919, trend_threshold=trend_threshold,
    ).set_index("instrument")

    rows: list[dict] = []
    for inst in full.index:
        for metric in metrics:
            f = float(full.at[inst, metric]) if metric in full.columns else float("nan")
            h1 = (
                float(first_df.at[inst, metric])
                if (inst in first_df.index and metric in first_df.columns)
                else float("nan")
            )
            h2 = (
                float(second_df.at[inst, metric])
                if (inst in second_df.index and metric in second_df.columns)
                else float("nan")
            )
            h1_lo = (
                float(first_df.at[inst, f"{metric}_lo"])
                if (inst in first_df.index and f"{metric}_lo" in first_df.columns)
                else float("nan")
            )
            h1_hi = (
                float(first_df.at[inst, f"{metric}_hi"])
                if (inst in first_df.index and f"{metric}_hi" in first_df.columns)
                else float("nan")
            )
            h2_lo = (
                float(second_df.at[inst, f"{metric}_lo"])
                if (inst in second_df.index and f"{metric}_lo" in second_df.columns)
                else float("nan")
            )
            h2_hi = (
                float(second_df.at[inst, f"{metric}_hi"])
                if (inst in second_df.index and f"{metric}_hi" in second_df.columns)
                else float("nan")
            )
            both_sig = _safe_ci_excludes_zero(h1_lo, h1_hi) and _safe_ci_excludes_zero(
                h2_lo, h2_hi
            )
            opposite_sign = np.isfinite(h1) and np.isfinite(h2) and ((h1 > 0) != (h2 > 0))
            sign_flip = bool(opposite_sign and both_sig)
            rows.append(
                {
                    "instrument": inst,
                    "metric": metric,
                    "full_window": f,
                    "first_half": h1,
                    "second_half": h2,
                    "sign_flip_flag": sign_flip,
                }
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
    parser.add_argument(
        "--stability-out",
        type=Path,
        default=None,
        help=(
            "If set, also write a stability CSV (full window + first/second "
            "halves with sign-flip flags). Pass a path such as "
            "results/harry/signal_direction_stability.csv to enable."
        ),
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

    if args.stability_out is not None:
        stab = audit_stability(
            ohlcv,
            signals,
            h=args.h,
            n_boot=args.n_boot,
            block_size=args.block_size,
            seed=args.seed,
            trend_threshold=args.trend_threshold,
        )
        args.stability_out.parent.mkdir(parents=True, exist_ok=True)
        stab.to_csv(args.stability_out, index=False)
        n_flips = int(stab["sign_flip_flag"].sum())
        print(f"wrote {args.stability_out} ({len(stab)} rows, {n_flips} sign flips)")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
