"""Primary-signal characterisation for EX.5 (/aqms-python L3/L4).

The metamodel is a *secondary* act/skip filter, so its ceiling is set by the primary signal it
sits on. This module measures the provided signal in its own right: turnover (flip frequency),
hit-rate (directional accuracy on non-zero days), the information coefficient (rank correlation
with the forward return), and the fundamental-law information ratio ``IR = IC·√breadth`` (Grinold
1989). Together they contextualise how much edge is even available for the metamodel to refine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr


def signal_hit_rate(signal, forward_returns) -> float:
    """Fraction of non-zero-signal days whose sign matches the forward return (NaN if none)."""
    s = np.asarray(signal, dtype=float)
    r = np.asarray(forward_returns, dtype=float)
    active = s != 0
    if not active.any():
        return float("nan")
    return float((np.sign(s[active]) == np.sign(r[active])).mean())


def signal_turnover(signal) -> float:
    """Mean number of unit position changes per period = mean(|Δsignal|)/2."""
    s = pd.Series(np.asarray(signal, dtype=float))
    return float(s.diff().abs().div(2.0).dropna().mean())


def information_coefficient(signal, forward_returns, *, method: str = "spearman") -> float:
    """Rank (Spearman) correlation between the signal and the forward return — the IC."""
    s = np.asarray(signal, dtype=float)
    r = np.asarray(forward_returns, dtype=float)
    if method != "spearman":
        raise ValueError(f"unsupported IC method: {method}")
    return float(spearmanr(s, r).correlation)


def information_ratio(ic: float, breadth: float) -> float:
    """Fundamental law of active management: IR = IC·√breadth (Grinold 1989)."""
    return float(ic * np.sqrt(breadth))


def henriksson_merton(realised_direction, predicted_direction) -> tuple[float, float, float]:
    """Directional-accuracy proxy (a simplified, base-rate-SENSITIVE sign test).

    Returns ``(hit_rate, z_stat, p_value)`` for the naive null that the directional hit rate is
    ½: ``z = (p̂ − ½) / √(¼/P)``, ``p = 1 − Φ(z)``. **Caveat (S5.10):** this is *not* canonical
    Henriksson–Merton — it is neither the parametric regression form nor the conditional
    non-parametric form. Because it does not condition on the directional base rates, it is
    biased *toward* "skill" whenever the up/down sample is imbalanced (any trending window): an
    always-long call in an up market scores p̂ > ½ with no skill at all. Use
    :func:`pesaran_timmermann` (which conditions on the base rates) as the PRIMARY timing test
    and :func:`treynor_mazuy` for convexity; this proxy is reported only as a complement, and a
    proxy biased toward skill that still shows none is the conservative reading.
    """
    real = np.sign(np.asarray(realised_direction, dtype=float))
    pred = np.sign(np.asarray(predicted_direction, dtype=float))
    p = len(real)
    if p == 0:
        return float("nan"), float("nan"), float("nan")
    hit = float((real == pred).mean())
    z = (hit - 0.5) / np.sqrt(0.25 / p)
    return hit, float(z), float(1.0 - norm.cdf(z))


def pesaran_timmermann(realised_direction, predicted_direction) -> tuple[float, float]:
    """Pesaran–Timmermann (1992) non-parametric test of directional predictive accuracy.

    The base-rate-aware market-timing test (the PRIMARY §5 statistic). Let ``P̂`` be the fraction
    of correctly signed calls and ``P̂*`` the fraction expected under independence of the
    predicted and realised directions; then ``S = (P̂ − P̂*) / √(var P̂ − var P̂*) → N(0, 1)``
    under the no-skill null. Unlike the hit-rate proxy it conditions on the directional base
    rates, so a constant (no-information) predictor scores ``S = 0`` regardless of the market's
    drift. Returns ``(stat, one_sided_p_value)``; a degenerate (constant) predictor returns
    ``(0.0, 0.5)`` — exactly no directional information. Cite PT 1992, pp. 461–465.
    """
    y = (np.asarray(realised_direction, dtype=float) > 0).astype(float)
    x = (np.asarray(predicted_direction, dtype=float) > 0).astype(float)
    n = len(y)
    if n == 0:
        return float("nan"), float("nan")
    p_hat = float((x == y).mean())
    py = float(y.mean())
    px = float(x.mean())
    p_star = py * px + (1.0 - py) * (1.0 - px)
    var_p = p_star * (1.0 - p_star) / n
    var_pstar = (
        (2.0 * py - 1.0) ** 2 * px * (1.0 - px) / n
        + (2.0 * px - 1.0) ** 2 * py * (1.0 - py) / n
        + 4.0 * py * px * (1.0 - py) * (1.0 - px) / n**2
    )
    denom = var_p - var_pstar
    if denom <= 0.0:  # constant predictor → no directional information
        return 0.0, 0.5
    stat = (p_hat - p_star) / np.sqrt(denom)
    return float(stat), float(1.0 - norm.cdf(stat))


def treynor_mazuy(market, portfolio) -> tuple[float, float, float]:
    """Treynor–Mazuy (1966) convexity timing test: regress portfolio on market + market².

    Fits ``r_p = α + β·r_m + γ·r_m² + ε`` by OLS; the quadratic coefficient ``γ`` is the timing
    signature — ``γ > 0`` means the strategy holds more (less) exposure when the market move is
    large (small), the convex payoff of a successful timer; ``γ ≈ 0`` means no timing. Here the
    "market" is each acted trade's realised return and the "portfolio" the signed-side PnL, so a
    correctly-sided, conviction-scaled book is convex in the move. Returns ``(gamma, t_stat,
    two_sided_p)`` with the OLS HC0-free analytic SE. Cite Treynor–Mazuy 1966.
    """
    m = np.asarray(market, dtype=float)
    p = np.asarray(portfolio, dtype=float)
    n = len(m)
    if n < 4:
        return float("nan"), float("nan"), float("nan")
    design = np.column_stack([np.ones(n), m, m * m])
    beta, *_ = np.linalg.lstsq(design, p, rcond=None)
    gamma = float(beta[2])
    resid = p - design @ beta
    dof = n - 3
    sigma2 = float(resid @ resid) / dof if dof > 0 else float("nan")
    try:
        se = float(np.sqrt(sigma2 * np.linalg.inv(design.T @ design)[2, 2]))
    except np.linalg.LinAlgError:
        se = float("nan")
    t = gamma / se if se and np.isfinite(se) and se > 0 else float("nan")
    pval = float(2.0 * (1.0 - norm.cdf(abs(t)))) if np.isfinite(t) else float("nan")
    return gamma, float(t), pval
