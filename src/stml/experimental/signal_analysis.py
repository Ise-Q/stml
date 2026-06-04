"""Signal analysis — plan §3.8 / §8 S7.

* Pesaran-Timmermann (PRIMARY directional skill test, base-rate aware)
* Treynor-Mazuy convexity timing
* Henriksson-Merton (base-rate-sensitive proxy)
* Information Coefficient + Grinold's Fundamental Law (IR = IC·√BR)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def pesaran_timmermann(
    realised: np.ndarray, predicted: np.ndarray
) -> tuple[float, float]:
    """Pesaran-Timmermann 1992 — base-rate-aware directional skill.

        S = (P̂ − P̂*) / √(var P̂ − var P̂*) → N(0,1) under no-skill null
    where:
        P̂  = empirical hit rate (sign agreement)
        P̂* = base-rate hit rate under independence

    A constant call in a trending market scores S = 0 — does NOT spuriously
    look like skill. Returns ``(stat, one_sided_p_value)``.
    """
    realised = np.asarray(realised, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(realised) & np.isfinite(predicted)
    realised = realised[mask]
    predicted = predicted[mask]
    n = len(realised)
    if n < 10:
        return (float("nan"), float("nan"))

    real_pos = (realised > 0).astype(float)
    pred_pos = (predicted > 0).astype(float)

    Py = real_pos.mean()
    Px = pred_pos.mean()
    P_hat = (real_pos == pred_pos).mean()
    P_star = Py * Px + (1.0 - Py) * (1.0 - Px)

    var_P_hat = (P_star * (1.0 - P_star)) / n
    # Pesaran-Timmermann 1992 Theorem 4.1: first two terms are O(1/n);
    # the cross term is O(1/n²) — NOT 1/n.
    var_P_star = (
        (2.0 * Py - 1.0) ** 2 * Px * (1.0 - Px) / n
        + (2.0 * Px - 1.0) ** 2 * Py * (1.0 - Py) / n
        + 4.0 * Py * Px * (1.0 - Py) * (1.0 - Px) / (n ** 2)
    )

    denom = var_P_hat - var_P_star
    if denom < 1e-10:
        return (float("nan"), float("nan"))
    S = (P_hat - P_star) / np.sqrt(denom)
    p = 1.0 - stats.norm.cdf(S)  # one-sided (skill = S > 0)
    return (float(S), float(p))


def treynor_mazuy(
    market: np.ndarray, portfolio: np.ndarray
) -> tuple[float, float]:
    """Treynor-Mazuy 1966 convexity timing: r_p = α + β·r_m + γ·r_m² + ε.

    Returns (γ, t-statistic on γ).
    """
    market = np.asarray(market, dtype=float)
    portfolio = np.asarray(portfolio, dtype=float)
    mask = np.isfinite(market) & np.isfinite(portfolio)
    market = market[mask]
    portfolio = portfolio[mask]
    n = len(market)
    if n < 20:
        return (float("nan"), float("nan"))
    X = np.column_stack([np.ones(n), market, market ** 2])
    # OLS coef.
    try:
        coef, residuals, rank, _ = np.linalg.lstsq(X, portfolio, rcond=None)
    except Exception:
        return (float("nan"), float("nan"))
    gamma = float(coef[2])
    # t-stat: gamma / SE.
    yhat = X @ coef
    sse = float(((portfolio - yhat) ** 2).sum())
    df = n - 3
    if df <= 0 or sse <= 0:
        return (gamma, float("nan"))
    sigma2 = sse / df
    try:
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        se_gamma = float(np.sqrt(cov[2, 2]))
    except Exception:
        return (gamma, float("nan"))
    if se_gamma <= 0:
        return (gamma, float("nan"))
    return (gamma, float(gamma / se_gamma))


def henriksson_merton_proxy(
    realised: np.ndarray, predicted: np.ndarray
) -> tuple[float, float, float]:
    """Base-rate-SENSITIVE Henriksson-Merton proxy (alken §5.21 caveat).

    Returns ``(hit_rate, z_statistic, p_value)``.

    Caveat: this is NOT canonical Henriksson-Merton — neither the parametric
    regression form nor the conditional non-parametric form. It is biased
    TOWARD "skill" in any trending window; reported only as a complement to
    Pesaran-Timmermann (which IS base-rate aware).
    """
    realised = np.asarray(realised, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(realised) & np.isfinite(predicted) & (predicted != 0)
    realised = realised[mask]
    predicted = predicted[mask]
    n = len(realised)
    if n < 10:
        return (float("nan"), float("nan"), float("nan"))
    hits = (np.sign(realised) == np.sign(predicted)).astype(float)
    hit_rate = hits.mean()
    # Naive z-stat against the 0.5 null (BIASED — that's the caveat).
    z = (hit_rate - 0.5) / np.sqrt(0.25 / n)
    p = 1.0 - stats.norm.cdf(z)
    return (float(hit_rate), float(z), float(p))


# ---------------------------------------------------------------------------
# Information Coefficient + Grinold's Fundamental Law.
# ---------------------------------------------------------------------------


def information_coefficient(
    signal: np.ndarray, forward_returns: np.ndarray, *, method: str = "spearman"
) -> float:
    """Spearman / Pearson rank correlation between signal and forward return."""
    s = np.asarray(signal, dtype=float)
    r = np.asarray(forward_returns, dtype=float)
    mask = np.isfinite(s) & np.isfinite(r)
    s = s[mask]
    r = r[mask]
    if len(s) < 10:
        return float("nan")
    if method == "spearman":
        rho, _ = stats.spearmanr(s, r)
    else:
        rho, _ = stats.pearsonr(s, r)
    return float(rho)


def information_ratio(ic: float, breadth: int) -> float:
    """Grinold 1989: IR = IC · √BR."""
    if not np.isfinite(ic) or breadth <= 0:
        return float("nan")
    return float(ic * np.sqrt(breadth))
