"""microstructure_fixed.py — microstructure features with the
zero-volume mask correctly applied.

ECONOMIC INTUITION
==================
Liquidity and microstructure noise condition every trade's expected
slippage. The signal-deep-dive branch's F7 family computes Amihud
illiquidity and other microstructure quantities but Sreeram's `G4`
family did not consistently mask the 765 zero-volume rows in our panel
(documented in ``reports/missing-data-report.md`` — those rows carry a
valid settle price but the volume field was not recorded). Dividing by
zero volume produces Inf; the rolling mean propagates the Inf and
contaminates the feature. This module is the corrected implementation:

* ``amihud_illiquidity``    — mean ``|r| / volume`` over a trailing
  window, with zero-volume rows masked to NaN. Range ``[0, ∞)``.
* ``rolls_effective_spread`` — Roll (1984) implied bid–ask spread
  ``2 · √(max(-Cov(Δp_t, Δp_{t-1}), 0))``. Range ``[0, ∞)``.
* ``kyles_lambda``          — practical Hasbrouck (2009) form
  ``mean(|r| / √volume)``, rolling. Higher = more price impact per
  share. Range ``[0, ∞)``.
* ``overnight_gap``         — ``log(open_t / close_{t-1})``, the
  overnight return.

CAUSALITY CONTRACT
==================
All trailing windows close at row ``t``; outputs use only data at
indices ``<= t``. ``overnight_gap`` requires the caller to pass the
already-lagged previous close (``close.shift(1)``); the explicit-lag
design makes the causality contract obvious at the call site.

WARMUP WINDOWS
==============
* ``amihud_illiquidity``     : ``window - 1`` (default 19).
* ``rolls_effective_spread`` : ``window`` (default 20; needs Δp at the
  start of the window).
* ``kyles_lambda``           : ``window - 1`` (default 19).
* ``overnight_gap``          : 1 (lagged close at row 0 is NaN).

ZERO-VOLUME MASK
================
The mask predicate vendored from ``stml.na_checks`` is:
``volume.fillna(0).eq(0)``. Vendored rather than imported so this
module's logic stays self-contained on the Harry branch; the source-of-
truth implementation in ``na_checks.detect_anomalous_rows`` is unchanged.

CITATIONS
=========
* Amihud, Y. (2002) "Illiquidity and Stock Returns: Cross-Section and
  Time-Series Effects", Journal of Financial Markets 5: 31–56.
* Roll, R. (1984) "A Simple Implicit Measure of the Effective Bid–Ask
  Spread in an Efficient Market", Journal of Finance 39: 1127–1139.
* Kyle, A. S. (1985) "Continuous Auctions and Insider Trading",
  Econometrica 53: 1315–1335; Hasbrouck (2009) for the daily-bar
  approximation used here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "amihud_illiquidity",
    "rolls_effective_spread",
    "kyles_lambda",
    "overnight_gap",
]


def _mask_zero_volume(volume: pd.Series) -> pd.Series:
    """Vendored mask from ``stml.na_checks``: zero-volume rows -> NaN.

    Treats NaN as zero (the na_checks pattern is
    ``volume.fillna(0).eq(0)``).
    """
    v = volume.astype("float64")
    return v.where(v > 0, np.nan)


def amihud_illiquidity(
    r: pd.Series, volume: pd.Series, window: int = 20
) -> pd.Series:
    """Rolling Amihud (2002) illiquidity: ``mean(|r| / volume)``.

    Zero-volume rows are masked to NaN before the division so the rolling
    mean is finite. ``rolling(min_periods=window)`` is strict on the
    synthetic harness panel; for real data with the 765 documented
    zero-volume rows the caller can either accept the NaN gaps or use a
    looser ``min_periods`` in their own wrapper.
    """
    v = _mask_zero_volume(volume)
    illiq = r.abs().astype("float64") / v
    return (
        illiq.rolling(window, min_periods=window)
        .mean()
        .rename(f"amihud_illiquidity_{window}d")
    )


def rolls_effective_spread(close: pd.Series, window: int = 20) -> pd.Series:
    """Roll (1984) implied bid–ask spread.

    ``s_t = 2 · √(max(-Cov(Δp_t, Δp_{t-1}), 0))``

    where ``Δp = close.diff()`` and the covariance is taken on the
    trailing ``window`` bars. Negative covariance is the Roll bounce
    signature; positive covariance (no measurable bounce) returns
    spread 0.
    """
    dp = close.astype("float64").diff()
    dp_lag = dp.shift(1)
    cov = (
        (dp * dp_lag).rolling(window, min_periods=window).mean()
        - dp.rolling(window, min_periods=window).mean()
        * dp_lag.rolling(window, min_periods=window).mean()
    )
    spread = 2.0 * np.sqrt(np.clip(-cov, a_min=0.0, a_max=None))
    return spread.rename(f"rolls_spread_{window}d")


def kyles_lambda(
    r: pd.Series, volume: pd.Series, window: int = 20
) -> pd.Series:
    """Daily-bar Kyle's lambda (Hasbrouck 2009 form): mean ``|r| / √volume``.

    Higher = more price impact per share traded. Zero-volume rows are
    NaN-masked before the division.
    """
    v = _mask_zero_volume(volume)
    impact = r.abs().astype("float64") / np.sqrt(v)
    return (
        impact.rolling(window, min_periods=window)
        .mean()
        .rename(f"kyles_lambda_{window}d")
    )


def overnight_gap(open_: pd.Series, close_prev: pd.Series) -> pd.Series:
    """Overnight log-return: ``log(open_t / close_prev_t)``.

    ``close_prev`` is the lagged close; the caller must pass
    ``close.shift(1)`` (the explicit-lag design makes the causality
    contract obvious). Output is NaN at row 0 (no prior close).
    """
    return np.log(open_.astype("float64") / close_prev.astype("float64")).rename(
        "overnight_gap"
    )


# --------------------------------------------------------------------------- #
# Causality harness registry                                                  #
# --------------------------------------------------------------------------- #
CAUSALITY_REGISTRATIONS: list[dict] = [
    {
        "name": "amihud_illiquidity",
        "module": __name__,
        "func": "amihud_illiquidity",
        "adapter": "returns_volume",
        "kwargs": {"window": 20},
        "warmup": 19,
        "data_kind": "single_instrument",
    },
    {
        "name": "rolls_effective_spread",
        "module": __name__,
        "func": "rolls_effective_spread",
        "adapter": "close_only",
        "kwargs": {"window": 20},
        # Δp = close.diff() is NaN at row 0; the lagged Δp is NaN at rows
        # 0–1; the rolling-window covariance needs both, so the first
        # defined output is at row 21 (one past the strict ``window`` of
        # 20 because the lagged Δp contributes one extra NaN row).
        "warmup": 21,
        "data_kind": "single_instrument",
    },
    {
        "name": "kyles_lambda",
        "module": __name__,
        "func": "kyles_lambda",
        "adapter": "returns_volume",
        "kwargs": {"window": 20},
        "warmup": 19,
        "data_kind": "single_instrument",
    },
    {
        "name": "overnight_gap",
        "module": __name__,
        "func": "overnight_gap",
        "adapter": "open_close_lagged",
        "kwargs": {},
        "warmup": 1,
        "data_kind": "single_instrument",
    },
]
