"""OHLC range-based volatility estimators: Garman-Klass, Parkinson, Rogers-Satchell.

These provide the σ̂ₜ that sets the triple-barrier widths (Section 2, ±k·σ̂ₜ) and
feeds the vol-targeting overlay (Section 6). Garman-Klass is preferred for energy
and rates futures because it incorporates an open-close term that captures the
overnight gaps produced by scheduled releases (EIA reports, auctions); Parkinson
(high-low only) systematically under-estimates vol on gap days, and Rogers-Satchell
is drift-independent. See ``reports/apb/nlr-cw-v1.md`` §A1 (Garman & Klass 1980;
Parkinson 1980; Korkusuz, Kambouroudis & McMillan 2023).

These wrap the same closed forms used in stml's feature layer
(``stml.metamodel.features.f2_vol_dispersion``) but expose them as standalone,
unit-tested σ̂ sources so Section 2 is self-contained.

Per-bar variances (O,H,L,C; r ≡ natural-log ratios):
    Parkinson:        σ² = (ln(H/L))² / (4·ln2)
    Garman-Klass:     σ² = ½·(ln(H/L))² − (2·ln2 − 1)·(ln(C/O))²
    Rogers-Satchell:  σ² = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)

Each estimator returns a *daily* volatility (standard deviation) Series aligned to
the input index. ``window=None`` gives the per-bar estimate; an integer ``window``
returns sqrt of the rolling mean of the per-bar variances (``min_periods=window``).
``annualize`` scales by sqrt(``trading_days``). All three are non-negative for valid
OHLC bars (H ≥ max(O,C), L ≤ min(O,C)).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0
_FOUR_LN2 = 4.0 * np.log(2.0)
_GK_CO_COEF = 2.0 * np.log(2.0) - 1.0


def _finalize(
    var: pd.Series, window: int | None, annualize: bool, trading_days: float
) -> pd.Series:
    # Clip away tiny negative values from floating-point error; valid bars are >= 0.
    var = var.clip(lower=0.0)
    if window is not None:
        var = var.rolling(window, min_periods=window).mean()
    vol = np.sqrt(var)
    if annualize:
        vol = vol * np.sqrt(trading_days)
    return vol


def parkinson(
    ohlc: pd.DataFrame,
    window: int | None = None,
    annualize: bool = False,
    trading_days: float = TRADING_DAYS,
) -> pd.Series:
    """Parkinson (1980) high-low range volatility."""
    hl = np.log(ohlc["high"] / ohlc["low"])
    var = hl.pow(2) / _FOUR_LN2
    return _finalize(var, window, annualize, trading_days)


def garman_klass(
    ohlc: pd.DataFrame,
    window: int | None = None,
    annualize: bool = False,
    trading_days: float = TRADING_DAYS,
) -> pd.Series:
    """Garman-Klass (1980) OHLC volatility (overnight-gap aware via the open-close term)."""
    hl = np.log(ohlc["high"] / ohlc["low"])
    co = np.log(ohlc["close"] / ohlc["open"])
    var = 0.5 * hl.pow(2) - _GK_CO_COEF * co.pow(2)
    return _finalize(var, window, annualize, trading_days)


def rogers_satchell(
    ohlc: pd.DataFrame,
    window: int | None = None,
    annualize: bool = False,
    trading_days: float = TRADING_DAYS,
) -> pd.Series:
    """Rogers-Satchell (1991) drift-independent OHLC volatility."""
    hc = np.log(ohlc["high"] / ohlc["close"])
    ho = np.log(ohlc["high"] / ohlc["open"])
    lc = np.log(ohlc["low"] / ohlc["close"])
    lo = np.log(ohlc["low"] / ohlc["open"])
    var = hc * ho + lc * lo
    return _finalize(var, window, annualize, trading_days)
