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
TAPER_WIDTH = 0.05  # S6.15: half-width of the smooth ramp around the confidence floor
TARGET_VOL = 0.25
MAX_LEVERAGE = 5.0


def _scalarize(a: np.ndarray):
    return float(a) if a.ndim == 0 else a


def confidence_taper(p, floor: float = CONFIDENCE_FLOOR, width: float = TAPER_WIDTH):
    """Smooth [0, 1] ramp replacing the hard ``p ≥ floor`` cutoff (S6.15, LR-7).

    A C¹-continuous smoothstep: 0 for ``p ≤ floor − width``, 1 for ``p ≥ floor + width``,
    ``0.5`` exactly at the floor. Multiplying conviction by this taper removes the sizing
    discontinuity at 0.55 (the hard floor jumps from 0 to κ·f*), which is the over-/under-betting
    artefact LR-7 flags; the gentle ramp is the calibrated growth-optimal shape.
    """
    p = np.asarray(p, dtype=float)
    t = np.clip((p - (floor - width)) / (2.0 * width), 0.0, 1.0)
    return _scalarize(t * t * (3.0 - 2.0 * t))


def kappa_baker_mchale(edge, resid_var):
    """Per-instrument Kelly shrinkage κᵢ = eᵢ² / (eᵢ² + σᵢ²) (Baker–McHale).

    ``edge`` is the instrument's signal edge (e.g. mean calibrated ``p̂ − ½``); ``resid_var`` its
    post-calibration residual variance σᵢ². κᵢ → 1 when the edge dominates the noise and → 0 when
    residual variance dominates, so it shrinks size hardest exactly where p̂ is least reliable.
    Monotonically decreasing in ``resid_var``; bounded in [0, 1].
    """
    e2 = np.asarray(edge, dtype=float) ** 2
    s2 = np.asarray(resid_var, dtype=float)
    denom = e2 + s2
    return _scalarize(np.where(denom > 0.0, e2 / denom, 0.0))


def cer_improves(candidate: float, baseline: float, *, min_gain: float = 0.0) -> bool:
    """The S6.15 CER gate: adopt a sizing change iff it strictly raises the OOS certainty-
    equivalent by more than ``min_gain`` (else revert to the flat-κ baseline — the honest
    default when more elaborate sizing buys no out-of-sample utility)."""
    if not (np.isfinite(candidate) and np.isfinite(baseline)):
        return False
    return bool(candidate - baseline > min_gain)


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
    taper_width: float | None = None,
):
    """κ·f*, gated by the confidence floor and clipped to [0, cap].

    Clipping to [0, cap] keeps sizing non-negative (the side is the primary's) and
    bounds leverage when a tight stop-loss ``d`` inflates the raw Kelly fraction. The floor is
    applied two ways: ``taper_width is None`` (default) hard-zeros below ``floor`` — the shipped
    behaviour; a positive ``taper_width`` (S6.15) instead multiplies by ``confidence_taper`` for a
    continuous ramp. ``kappa`` may be a per-instrument array (Baker–McHale κᵢ).
    """
    f = np.asarray(kappa, dtype=float) * np.asarray(kelly_fraction(p, b, d), dtype=float)
    p_arr = np.asarray(p, dtype=float)
    if taper_width is None:
        f = np.where(p_arr < floor, 0.0, f)
    else:
        f = f * np.asarray(confidence_taper(p_arr, floor=floor, width=taper_width), dtype=float)
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
    taper_width: float | None = None,
):
    """Signed position weight = side · fractional_kelly(p,b,d) · vol_target_leverage.

    The deliverable ``strategy_weights.csv`` value: positive = long, negative = short,
    zero when p̂ is below the confidence floor. ``kappa`` may be a per-instrument Baker–McHale
    κᵢ array and ``taper_width`` switches the hard floor for the S6.15 smooth ramp — both adopted
    only when the OOS certainty-equivalent improves (see ``cer_improves``).
    """
    size = np.asarray(
        fractional_kelly(p, b, d, kappa=kappa, floor=floor, cap=cap, taper_width=taper_width),
        dtype=float,
    )
    lev = np.asarray(
        vol_target_leverage(realised_vol, target_vol=target_vol, max_leverage=max_leverage),
        dtype=float,
    )
    return _scalarize(np.asarray(side, dtype=float) * size * lev)
