"""Shared volatility-targeting layer.

Implements the position-weight formula from Madmoun Optional Session 3, slide 40:

    w_{t,k} = ŷ_{t,k} · σ_tgt / σ̂_{t,k}

clipped to [-max_leverage, +max_leverage].

Here ŷ_{t,k} ∈ [-1, +1] is the signed conviction from any sizing method:
  - Benchmark: ŷ = side ∈ {-1, 0, +1}
  - SOPS/fixed:  ŷ = side · g(p̂)     where g ∈ [0, 1]
  - Neural:      ŷ ∈ [-1, +1] from the tanh head

σ̂_{t,k} is the ex-ante annualised vol from `vol.compute_vol` or passed directly.

The module also provides `build_vol_panel`, which builds a date × instrument
vol panel from the long-format OHLCV DataFrame, using the config defaults.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stml.new_work import config
from stml.new_work.vol import compute_vol


# ---------------------------------------------------------------------------
# Per-scalar weight
# ---------------------------------------------------------------------------


def vol_targeted_weight(
    y_hat: float,
    ann_sigma: float,
    *,
    sigma_tgt: float = config.SIGMA_TGT,
    max_leverage: float = config.MAX_LEVERAGE,
) -> float:
    """Apply vol-targeting to a single signed conviction.

    Returns 0 when ann_sigma is not strictly positive-finite or y_hat == 0.
    """
    if y_hat == 0.0:
        return 0.0
    if not (np.isfinite(ann_sigma) and ann_sigma > 0.0):
        return 0.0
    raw = y_hat * sigma_tgt / ann_sigma
    return float(np.clip(raw, -max_leverage, max_leverage))


# ---------------------------------------------------------------------------
# Vectorised weight series
# ---------------------------------------------------------------------------


def apply_vol_targeting(
    y_hat: pd.Series,
    ann_sigma: pd.Series,
    *,
    sigma_tgt: float = config.SIGMA_TGT,
    max_leverage: float = config.MAX_LEVERAGE,
) -> pd.Series:
    """Vectorised version of vol_targeted_weight for a single instrument.

    Parameters
    ----------
    y_hat
        Signed conviction series, index = date.
    ann_sigma
        Annualised daily σ̂ series, same index as y_hat.

    Returns
    -------
    pd.Series
        Position weights clipped to [-max_leverage, max_leverage], same index.
        NaN where ann_sigma is NaN (warmup period before vol estimate is available).
    """
    y = y_hat.reindex(ann_sigma.index)
    sigma = ann_sigma.copy()
    raw = y * sigma_tgt / sigma
    raw = raw.where(sigma > 0)  # NaN where sigma <= 0 or NaN
    return raw.clip(lower=-max_leverage, upper=max_leverage).rename("weight")


# ---------------------------------------------------------------------------
# Vol panel builder
# ---------------------------------------------------------------------------


def build_vol_panel(
    ohlcv: pd.DataFrame,
    *,
    instruments: list[str] | None = None,
    method: str | None = None,
    window: int | None = None,
    span: int | None = None,
    trading_days: float = config.TRADING_DAYS,
    boundary: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build a (date × instrument) annualised vol panel from long-format OHLCV.

    Filters to dates strictly before `boundary` (default: config.BOUNDARY) so
    the panel is safe to pass to any training routine without leakage risk.
    Dates after the boundary are retained but must be treated as OOS.

    Parameters
    ----------
    ohlcv
        Long-format DataFrame: columns [date, instrument, open, high, low, close, ...].
        Must have a DatetimeIndex OR a `date` column.
    instruments
        Subset of instruments to include. Defaults to config.INSTRUMENTS.
    method
        Vol estimator: yang_zhang | ewma_close | garch | gjr.
        Defaults to config.VOL_METHOD. GJR is auto-switched for equity
        instruments when method == "gjr".
    window
        Rolling window override (bars). Defaults to config.LOOKBACK_L.
    span
        EWMA span override (bars). Defaults to config.EWMA_SPAN.
    trading_days
        Annualisation factor.
    boundary
        Upper (exclusive) date limit for the returned vol panel. Pass None to
        include all dates (required for OOS application). The function itself
        applies no boundary cut — the caller controls scope.

    Returns
    -------
    pd.DataFrame
        Wide vol panel: index = date (DatetimeIndex), columns = instruments.
        NaN in warmup rows.
    """
    if instruments is None:
        instruments = config.INSTRUMENTS
    method = method or config.VOL_METHOD
    window = window if window is not None else config.LOOKBACK_L
    span = span if span is not None else config.EWMA_SPAN

    # Ensure date is a column (handle both index and column forms).
    if "date" not in ohlcv.columns:
        ohlcv = ohlcv.reset_index().rename(columns={"index": "date"})

    panels: dict[str, pd.Series] = {}
    for inst in instruments:
        inst_df = ohlcv[ohlcv["instrument"] == inst].copy()
        if inst_df.empty:
            continue
        inst_df = inst_df.set_index("date").sort_index()
        inst_df.index = pd.to_datetime(inst_df.index)

        inst_method = method
        if method == "gjr" and inst not in config.EQUITY_INSTRUMENTS:
            # Fall back to GARCH for non-equity instruments.
            inst_method = "garch"

        sigma = compute_vol(
            inst_df,
            method=inst_method,
            window=window,
            span=span,
            trading_days=trading_days,
        )
        panels[inst] = sigma

    vol_panel = pd.DataFrame(panels)
    vol_panel.index = pd.to_datetime(vol_panel.index)
    vol_panel = vol_panel.sort_index()
    return vol_panel


# ---------------------------------------------------------------------------
# Convenience: weight panel from y_hat panel + vol panel
# ---------------------------------------------------------------------------


def weights_from_conviction(
    y_hat_panel: pd.DataFrame,
    vol_panel: pd.DataFrame,
    *,
    sigma_tgt: float = config.SIGMA_TGT,
    max_leverage: float = config.MAX_LEVERAGE,
) -> pd.DataFrame:
    """Apply vol-targeting across an instrument panel.

    Parameters
    ----------
    y_hat_panel
        Wide conviction panel: index = date, columns = instruments, values ∈ [-1, +1].
    vol_panel
        Wide annualised vol panel from :func:`build_vol_panel`.

    Returns
    -------
    pd.DataFrame
        Position weight panel: same shape as y_hat_panel, clipped to
        [-max_leverage, +max_leverage]. Rows with missing vol are NaN.
    """
    # Align both panels to the same date index.
    common_idx = y_hat_panel.index.intersection(vol_panel.index)
    y = y_hat_panel.loc[common_idx]
    v = vol_panel.loc[common_idx]

    raw = y * sigma_tgt / v
    # Zero where conviction is zero (preserves NaN from missing vol).
    raw = raw.where(y != 0.0, other=0.0)
    return raw.clip(lower=-max_leverage, upper=max_leverage)
