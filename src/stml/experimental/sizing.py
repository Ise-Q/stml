"""Position sizing — Optional Session 3 (Madmoun) recipe.

This module implements the six sizing functions prescribed in the lecturer's
Calibration & Position Sizing section (slides 32-34) plus the volatility-
targeting weight assembly from the Portfolio Construction section
(slides 39-40).

Every sizing function maps a calibrated meta-probability :math:`\\hat p \\in [0,1]`
to a unit-less conviction :math:`b \\in [0,1]`, returning 0 when
:math:`\\hat p \\le 0.5` (the side is wrong on average there).

Six methods, two families:

Fixed (no training):
    1. ``model_confidence``  --  :math:`b = \\hat p \\cdot \\mathbf 1\\{\\hat p > 0.5\\}`
    2. ``all_or_nothing``    --  :math:`b = \\mathbf 1\\{\\hat p > 0.5\\}`
    3. ``ncdf``              --  :math:`b = \\Phi\\!\\bigl((\\hat p - 0.5)/\\sqrt{\\hat p(1-\\hat p)}\\bigr)`

Estimated (fit on training pairs :math:`(p_{tr}, r_{tr})`):
    4. ``linear_scaling``    --  stretch training range to [0,1]
    5. ``ecdf``              --  size by the training percentile of :math:`\\hat p`
    6. ``sops``              --  Sharpe-Optimal Position Sizing; fits a logistic
                                 sigmoid :math:`f_{a,c}(p) = (1+e^{-(ap-c)})^{-1}`
                                 such that the Sharpe of
                                 :math:`m_{a,c} = f_{a,c}(p_{tr}) \\odot r_{tr}`
                                 is maximised.

The final position weight assembled by :func:`position_weight` is::

    w_{t,k} = side · g(p̂_{t,k}) · σ_tgt / σ_{t,k}_annualised

clipped to ``[-max_leverage, +max_leverage]`` per the released competition
constraints (released 20 May; assumed 10% target vol). ``g`` is the chosen
sizing function. ``σ_{t,k}`` is causal EWMA daily std of returns
(:mod:`stml.experimental.volatility.ewma_daily_sigma`), annualised by √252.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


SizingMethodName = Literal[
    "model_confidence", "all_or_nothing", "ncdf",
    "linear_scaling", "ecdf", "sops",
]

# Lecturer-prescribed defaults. The competition target volatility is 10%
# annualised (slide 40); max_leverage is a defensive clamp against pathological
# σ̂_t estimates and is not in the lecturer's formula.
TARGET_VOL: float = 0.10
MAX_LEVERAGE: float = 10.0
TRADING_DAYS: float = 252.0


# ---------------------------------------------------------------------------
# Fixed sizing methods (no training data needed).
# ---------------------------------------------------------------------------


def model_confidence(p: np.ndarray | float) -> np.ndarray | float:
    """Slide 33, method 1 -- size equals the calibrated probability.

    :math:`b = \\hat p \\cdot \\mathbf 1\\{\\hat p > 0.5\\}`.
    """
    p_arr = np.asarray(p, dtype=float)
    return np.where(p_arr > 0.5, p_arr, 0.0)


def all_or_nothing(p: np.ndarray | float) -> np.ndarray | float:
    """Slide 33, method 2 -- binary gate, full size or zero.

    :math:`b = \\mathbf 1\\{\\hat p > 0.5\\}`.
    """
    p_arr = np.asarray(p, dtype=float)
    return (p_arr > 0.5).astype(float)


def ncdf(p: np.ndarray | float) -> np.ndarray | float:
    """Slide 33, method 3 -- standardise then push through the normal CDF.

    :math:`b = \\Phi\\!\\bigl((\\hat p - 0.5) / \\sqrt{\\hat p (1 - \\hat p)}\\bigr)`
    for :math:`\\hat p > 0.5`, else 0.
    """
    p_arr = np.asarray(p, dtype=float)
    out = np.zeros_like(p_arr, dtype=float)
    mask = p_arr > 0.5
    if mask.any():
        pm = p_arr[mask]
        var = np.clip(pm * (1.0 - pm), 1e-12, None)
        z = (pm - 0.5) / np.sqrt(var)
        out[mask] = norm.cdf(z)
    return out if isinstance(p, np.ndarray) else float(out)


# ---------------------------------------------------------------------------
# Estimated sizing methods — fit on training (p_tr, r_tr) pairs.
# ---------------------------------------------------------------------------


@dataclass
class LinearScalingFit:
    """Fitted parameters for :func:`linear_scaling`."""

    p_min: float
    p_max: float

    def transform(self, p: np.ndarray | float) -> np.ndarray:
        """Apply the fit; returns 0 below 0.5."""
        p_arr = np.asarray(p, dtype=float)
        denom = max(self.p_max - self.p_min, 1e-12)
        scaled = (p_arr - self.p_min) / denom
        scaled = np.clip(scaled, 0.0, 1.0)
        return np.where(p_arr > 0.5, scaled, 0.0)


def fit_linear_scaling(p_tr: np.ndarray) -> LinearScalingFit:
    """Slide 34, method 4 -- stretch the training range to [0,1].

    :math:`g(\\hat p) = (\\hat p - \\min p_{tr}) / (\\max p_{tr} - \\min p_{tr})`,
    zero below 0.5.
    """
    p_tr = np.asarray(p_tr, dtype=float)
    if p_tr.size == 0:
        return LinearScalingFit(p_min=0.5, p_max=1.0)
    return LinearScalingFit(p_min=float(np.min(p_tr)), p_max=float(np.max(p_tr)))


@dataclass
class ECDFFit:
    """Fitted ECDF for :func:`ecdf`."""

    sorted_p_tr: np.ndarray = field(repr=False)

    def transform(self, p: np.ndarray | float) -> np.ndarray:
        """Empirical CDF of ``p`` on the training distribution; 0 below 0.5."""
        p_arr = np.asarray(p, dtype=float)
        if self.sorted_p_tr.size == 0:
            return np.zeros_like(p_arr)
        ranks = np.searchsorted(self.sorted_p_tr, p_arr, side="right")
        cdf = ranks / float(self.sorted_p_tr.size)
        return np.where(p_arr > 0.5, cdf, 0.0)


def fit_ecdf(p_tr: np.ndarray) -> ECDFFit:
    """Slide 34, method 5 -- size by the training percentile of :math:`\\hat p`."""
    p_tr = np.asarray(p_tr, dtype=float)
    return ECDFFit(sorted_p_tr=np.sort(p_tr))


@dataclass
class SOPSFit:
    """Fitted parameters for :func:`sops` (Sharpe-Optimal Position Sizing)."""

    a: float
    c: float
    train_sharpe: float

    def _f(self, p: np.ndarray) -> np.ndarray:
        # Logistic sigmoid f_{a,c}(p) = 1 / (1 + exp(-(a·p - c))).
        z = self.a * p - self.c
        # Numerically stable sigmoid.
        return np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-z)),
            np.exp(z) / (1.0 + np.exp(z)),
        )

    def transform(self, p: np.ndarray | float) -> np.ndarray:
        """Apply the fitted sigmoid; zero below 0.5."""
        p_arr = np.asarray(p, dtype=float)
        out = self._f(p_arr)
        return np.where(p_arr > 0.5, out, 0.0)


def fit_sops(
    p_tr: np.ndarray,
    r_tr: np.ndarray,
    *,
    a_grid: np.ndarray | None = None,
    c_grid: np.ndarray | None = None,
    refine: bool = True,
) -> SOPSFit:
    """Slide 34, method 6 -- fit the sizing sigmoid to maximise the Sharpe.

    For each candidate ``(a, c)`` compute the per-trade sized return
    :math:`m_{a,c}(i) = f_{a,c}(p_{tr,i}) \\cdot r_{tr,i}` and pick the
    ``(a, c)`` that maximises ``mean(m) / std(m)`` -- the lecturer's exact
    objective.

    Parameters
    ----------
    p_tr, r_tr
        Equal-length 1-D arrays of training calibrated probabilities and
        realised per-trade returns (signed by primary direction).
    a_grid, c_grid
        Optional coarse search grids. Defaults: ``a ∈ {1, 5, 10, 20, 50, 100}``,
        ``c ∈ {0.0, 0.5·a, 0.6·a, 0.7·a, 0.8·a, 0.9·a}`` -- spans the lecturer's
        sigmoid family centred near 0.5.
    refine
        If True (default), polish the best grid point with L-BFGS-B on the
        Sharpe objective for sub-grid resolution.
    """
    p_tr = np.asarray(p_tr, dtype=float)
    r_tr = np.asarray(r_tr, dtype=float)
    if len(p_tr) != len(r_tr):
        raise ValueError("p_tr and r_tr must be the same length")
    finite = np.isfinite(p_tr) & np.isfinite(r_tr)
    p_tr, r_tr = p_tr[finite], r_tr[finite]
    if len(p_tr) < 5:
        # Fall back to all-or-nothing-equivalent sigmoid.
        return SOPSFit(a=20.0, c=10.0, train_sharpe=0.0)

    if a_grid is None:
        a_grid = np.array([1.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    if c_grid is None:
        # Centre the sigmoid around p ≈ c/a; for a=10, c=5 gives midpoint 0.5.
        c_grid = np.array([0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])  # multiplied by a

    def neg_sharpe(params: np.ndarray) -> float:
        a, c = float(params[0]), float(params[1])
        z = a * p_tr - c
        f = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                              np.exp(z) / (1.0 + np.exp(z)))
        m = f * r_tr
        mu, sd = float(np.mean(m)), float(np.std(m, ddof=1))
        if sd <= 0:
            return 0.0
        return -mu / sd  # minimise negative Sharpe

    # Coarse grid.
    best = (0.0, None)
    for a in a_grid:
        for c_frac in c_grid:
            c = float(c_frac * a)
            s = -neg_sharpe(np.array([a, c]))
            if not np.isfinite(s):
                continue
            if s > best[0] or best[1] is None:
                best = (s, (float(a), c))
    if best[1] is None:
        return SOPSFit(a=20.0, c=10.0, train_sharpe=0.0)
    a_opt, c_opt = best[1]

    # Refine.
    if refine:
        res = minimize(
            neg_sharpe, x0=np.array([a_opt, c_opt]), method="L-BFGS-B",
            bounds=[(0.1, 500.0), (-500.0, 500.0)],
            options={"maxiter": 200, "ftol": 1e-8},
        )
        if res.success and np.isfinite(res.fun):
            a_opt, c_opt = float(res.x[0]), float(res.x[1])
            best = (-float(res.fun), (a_opt, c_opt))

    return SOPSFit(a=a_opt, c=c_opt, train_sharpe=float(best[0]))


# ---------------------------------------------------------------------------
# Sizing-method dispatcher.
# ---------------------------------------------------------------------------


@dataclass
class SizingPolicy:
    """Per-instrument sizing policy. Fit on training trades, then transform."""

    method: SizingMethodName
    fit: object | None = None  # LinearScalingFit | ECDFFit | SOPSFit | None
    threshold: float = 0.5     # gating threshold (≥ 0.5); may come from p*

    def transform(self, p: np.ndarray | float) -> np.ndarray:
        """Apply the policy to calibrated probabilities -- returns b in [0, 1]."""
        p_arr = np.asarray(p, dtype=float)
        if self.method == "model_confidence":
            b = model_confidence(p_arr)
        elif self.method == "all_or_nothing":
            b = all_or_nothing(p_arr)
        elif self.method == "ncdf":
            b = ncdf(p_arr)
        elif self.method == "linear_scaling":
            if not isinstance(self.fit, LinearScalingFit):
                raise RuntimeError("linear_scaling policy is missing its fit")
            b = self.fit.transform(p_arr)
        elif self.method == "ecdf":
            if not isinstance(self.fit, ECDFFit):
                raise RuntimeError("ecdf policy is missing its fit")
            b = self.fit.transform(p_arr)
        elif self.method == "sops":
            if not isinstance(self.fit, SOPSFit):
                raise RuntimeError("sops policy is missing its fit")
            b = self.fit.transform(p_arr)
        else:
            raise ValueError(f"unknown sizing method: {self.method!r}")
        # Apply the optional p* gate. The lecturer's per-method "zero below 0.5"
        # is already enforced inside each function; threshold ≥ 0.5 overrides.
        b = np.asarray(b, dtype=float)
        if self.threshold > 0.5:
            b = np.where(p_arr >= self.threshold, b, 0.0)
        return b


def fit_sizing_policy(
    method: SizingMethodName,
    *,
    p_tr: np.ndarray | None = None,
    r_tr: np.ndarray | None = None,
    threshold: float = 0.5,
) -> SizingPolicy:
    """Build a :class:`SizingPolicy` for ``method``, fitting if needed.

    Fixed methods (model_confidence, all_or_nothing, ncdf) ignore ``p_tr``
    and ``r_tr``. Estimated methods require both (ECDF/LinearScaling need
    only ``p_tr``; SOPS needs both).
    """
    if method in ("model_confidence", "all_or_nothing", "ncdf"):
        return SizingPolicy(method=method, fit=None, threshold=threshold)
    if method == "linear_scaling":
        if p_tr is None:
            raise ValueError("linear_scaling requires p_tr")
        return SizingPolicy(method=method, fit=fit_linear_scaling(p_tr),
                             threshold=threshold)
    if method == "ecdf":
        if p_tr is None:
            raise ValueError("ecdf requires p_tr")
        return SizingPolicy(method=method, fit=fit_ecdf(p_tr),
                             threshold=threshold)
    if method == "sops":
        if p_tr is None or r_tr is None:
            raise ValueError("sops requires p_tr and r_tr")
        return SizingPolicy(method=method, fit=fit_sops(p_tr, r_tr),
                             threshold=threshold)
    raise ValueError(f"unknown sizing method: {method!r}")


# ---------------------------------------------------------------------------
# Volatility-targeted weight assembly (slides 39-40).
# ---------------------------------------------------------------------------


def vol_target_leverage(
    daily_sigma: float,
    *,
    target_vol: float = TARGET_VOL,
    max_leverage: float = MAX_LEVERAGE,
    trading_days: float = TRADING_DAYS,
) -> float:
    """Annualise the daily σ̂ and return ``target_vol / ann_sigma``.

    Clipped to ``[0, max_leverage]``. Returns 0 on NaN / zero σ̂.
    """
    if not np.isfinite(daily_sigma) or daily_sigma <= 0:
        return 0.0
    ann_sigma = daily_sigma * np.sqrt(trading_days)
    lev = target_vol / ann_sigma
    return float(np.clip(lev, 0.0, max_leverage))


def position_weight(
    side: int,
    p: float,
    daily_sigma: float,
    *,
    policy: SizingPolicy,
    target_vol: float = TARGET_VOL,
    max_leverage: float = MAX_LEVERAGE,
    trading_days: float = TRADING_DAYS,
) -> float:
    """Signed weight :math:`w = \\mathrm{side} \\cdot g(\\hat p) \\cdot
    \\sigma_{tgt} / \\sigma^{ann}_{t,k}`.

    The conviction :math:`g(\\hat p)` is from ``policy``; the volatility
    leverage uses an annualised version of ``daily_sigma``. ``side = 0``
    or zero-conviction collapses to zero weight.
    """
    if side == 0:
        return 0.0
    b = float(np.asarray(policy.transform(np.array([p]))).ravel()[0])
    if b <= 0.0:
        return 0.0
    lev = vol_target_leverage(
        daily_sigma, target_vol=target_vol, max_leverage=max_leverage,
        trading_days=trading_days,
    )
    return float(np.sign(side) * b * lev)
