"""Position sizing: fractional Kelly + volatility targeting (Section 6 bonus).

The meta-model supplies a calibrated probability p̂ that the primary signal's
trade hits its upper barrier first; the *side* is fixed by the primary signal, so
we only size. We use the asymmetric-payoff Kelly fraction (Kelly 1956)

    f* = (p̂·b − (1 − p̂)·d) / (b·d)

with b, d the upper / lower barrier multipliers, scaled by a fractional κ = 0.25
(MacLean-Ziemba-Blazenko 1992: ~half the full-Kelly growth, far smaller drawdowns,
robust to estimation error in p̂), a confidence floor p̂ ≥ 0.55 (no bet below it),
and a vol-targeting overlay to a 25% annualised target (Carver 2015). See
``reports/apb/nlr-cw-v1.md`` §7.

Functions are elementwise: scalars return floats; arrays return ``np.ndarray``.
"""

from __future__ import annotations

import numpy as np

KAPPA = 0.25
CONFIDENCE_FLOOR = 0.55
TARGET_VOL = 0.25
MAX_LEVERAGE = 5.0


def _scalarize(a: np.ndarray):
    return float(a) if a.ndim == 0 else a


def kelly_fraction(p, b, d):
    """Asymmetric-payoff Kelly fraction f* = (p·b − (1−p)·d)/(b·d)."""
    p = np.asarray(p, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    return _scalarize((p * b - (1.0 - p) * d) / (b * d))


def fractional_kelly(
    p,
    b,
    d,
    kappa: float = KAPPA,
    floor: float = CONFIDENCE_FLOOR,
    cap: float = 1.0,
):
    """κ·f*, zeroed below the confidence floor and clipped to [0, cap].

    Clipping to [0, cap] keeps sizing non-negative (the side is the primary's) and
    bounds leverage when a tight stop-loss ``d`` inflates the raw Kelly fraction.
    """
    f = kappa * np.asarray(kelly_fraction(p, b, d), dtype=float)
    f = np.where(np.asarray(p, dtype=float) < floor, 0.0, f)
    return _scalarize(np.clip(f, 0.0, cap))


def vol_target_leverage(
    realised_vol,
    target_vol: float = TARGET_VOL,
    max_leverage: float = MAX_LEVERAGE,
):
    """Leverage = target_vol / realised_vol, clipped to [0, max_leverage].

    Zero (or near-zero) realised vol maps to ``max_leverage`` rather than dividing
    by zero.
    """
    rv = np.asarray(realised_vol, dtype=float)
    safe = np.where(rv > 0, rv, 1.0)  # avoid divide-by-zero warning
    lev = np.where(rv > 0, target_vol / safe, max_leverage)
    return _scalarize(np.clip(lev, 0.0, max_leverage))


def position_weight(
    side,
    p,
    b,
    d,
    realised_vol,
    target_vol: float = TARGET_VOL,
    kappa: float = KAPPA,
    floor: float = CONFIDENCE_FLOOR,
    cap: float = 1.0,
    max_leverage: float = MAX_LEVERAGE,
):
    """Signed position weight = side · fractional_kelly(p,b,d) · vol_target_leverage.

    The deliverable ``strategy_weights.csv`` value: positive = long, negative = short,
    zero when p̂ is below the confidence floor.
    """
    size = np.asarray(fractional_kelly(p, b, d, kappa=kappa, floor=floor, cap=cap), dtype=float)
    lev = np.asarray(
        vol_target_leverage(realised_vol, target_vol=target_vol, max_leverage=max_leverage),
        dtype=float,
    )
    return _scalarize(np.asarray(side, dtype=float) * size * lev)
