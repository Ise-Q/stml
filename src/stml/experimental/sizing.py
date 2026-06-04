"""Position sizing — plan §3.7 / §8 S6.

Fractional Kelly κ=0.25 + hard confidence floor 0.55 + vol target 0.08
(under the R7 10 % cap with 2 % headroom). Lifted from
``metamodel-apb/src/alken_metamodel/sizing.py`` (alken parity).
"""

from __future__ import annotations

import numpy as np

KAPPA = 0.25
CONFIDENCE_FLOOR = 0.55
TARGET_VOL = 0.08
MAX_LEVERAGE = 5.0


def kelly_fraction(p: float, b: float = 1.0, d: float = 1.0) -> float:
    """Asymmetric-payoff Kelly: f* = (p·b − (1−p)·d) / (b·d).

    For symmetric barriers (b=d=1): f* = 2p − 1.
    """
    if b <= 0 or d <= 0:
        return 0.0
    f = (p * b - (1.0 - p) * d) / (b * d)
    return float(f)


def fractional_kelly(
    p: float,
    *,
    b: float = 1.0,
    d: float = 1.0,
    kappa: float = KAPPA,
    floor: float = CONFIDENCE_FLOOR,
    cap: float = 1.0,
) -> float:
    """κ·f* clipped to [0, cap]; zero when p < floor."""
    if p < floor:
        return 0.0
    f_star = kelly_fraction(p, b=b, d=d)
    raw = kappa * max(f_star, 0.0)
    return float(min(raw, cap))


def vol_target_leverage(
    realised_vol: float,
    *,
    target_vol: float = TARGET_VOL,
    max_leverage: float = MAX_LEVERAGE,
) -> float:
    """target / realised, clipped to [0, max_leverage]. Zero on NaN/zero vol."""
    if not np.isfinite(realised_vol) or realised_vol <= 0:
        return 0.0
    lev = target_vol / realised_vol
    return float(np.clip(lev, 0.0, max_leverage))


def position_weight(
    side: int,
    p: float,
    realised_vol: float,
    *,
    b: float = 1.0,
    d: float = 1.0,
    kappa: float = KAPPA,
    floor: float = CONFIDENCE_FLOOR,
    target_vol: float = TARGET_VOL,
    max_leverage: float = MAX_LEVERAGE,
    confidence_cap: float = 1.0,
) -> float:
    """Signed position weight = side × fractional_kelly × vol_target_leverage."""
    if side == 0:
        return 0.0
    k = fractional_kelly(p, b=b, d=d, kappa=kappa, floor=floor, cap=confidence_cap)
    lev = vol_target_leverage(realised_vol, target_vol=target_vol, max_leverage=max_leverage)
    return float(np.sign(side) * k * lev)
