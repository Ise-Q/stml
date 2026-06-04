"""Sharpe-significance — plan §3.8 / §8 S7 PRIMARY inference.

* t-statistic = SR · √n (Bailey-Lopez de Prado 2012 baseline)
* Studentised stationary block-bootstrap CI (Politis-Romano 1994, Lo 2002 SE)
* Lo / Opdyke analytic band (parametric cross-check, non-Normal corrected)
* Probabilistic Sharpe Ratio (PSR)
* Minimum Track Record Length (MinTRL)
* Ljung-Box(10) IID gate before √252 annualisation

Lifted from ``metamodel-apb/src/alken_metamodel/significance.py`` (alken parity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def sharpe_ratio(returns: np.ndarray, ddof: int = 1) -> float:
    """Per-period Sharpe ratio (mean / std)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=ddof)
    if sd <= 0:
        return float("nan")
    return float(r.mean() / sd)


def t_statistic(returns: np.ndarray) -> float:
    """t = SR · √n — the AFML honest first read."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sr = sharpe_ratio(r)
    return float(sr * np.sqrt(len(r)))


def sharpe_std(sr: float, n: int, skew: float, kurt: float) -> float:
    """Mertens 2002 / Lo 2002 SE under non-Normality.

    Var(SR̂) = (1 − skew·SR + ((kurt−1)/4)·SR²) / (n − 1)
    """
    if n <= 1 or not np.isfinite(sr):
        return float("nan")
    var = (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2) / (n - 1)
    return float(np.sqrt(max(var, 0.0)))


def sharpe_ci_analytic(
    returns: np.ndarray, *, alpha: float = 0.05
) -> tuple[float, float]:
    """Lo/Opdyke band: SR ± z_{1-α/2} · σ(SR̂) — non-Normal corrected."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sr = sharpe_ratio(r)
    if not np.isfinite(sr):
        return (float("nan"), float("nan"))
    s = stats.skew(r, bias=False)
    k = stats.kurtosis(r, bias=False, fisher=False)  # kurt (not excess)
    se = sharpe_std(sr, len(r), s, k)
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return (float(sr - z * se), float(sr + z * se))


# ---------------------------------------------------------------------------
# Studentised stationary block-bootstrap CI (PRIMARY — plan §3.8).
# ---------------------------------------------------------------------------


def politis_white_block_length(returns: np.ndarray) -> float:
    """Politis-White data-driven optimal block length for the stationary boot."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 30:
        return 1.0
    # Estimate ρ at lags up to floor(sqrt(n)) and find first lag where the
    # autocorrelation is statistically insignificant (Politis & White 2004).
    max_lag = int(np.floor(np.sqrt(n)))
    centered = r - r.mean()
    var = (centered ** 2).mean()
    if var <= 0:
        return 1.0
    rho = np.array([
        (centered[:-k] * centered[k:]).mean() / var
        if k > 0 else 1.0
        for k in range(max_lag + 1)
    ])
    # Pick the lag at which |rho| first drops below 2/sqrt(n) (Politis 2004).
    threshold = 2.0 / np.sqrt(n)
    m_hat = 1
    for k in range(1, max_lag + 1):
        if abs(rho[k]) < threshold:
            m_hat = k
            break
    # Block length L = 1.5 × m̂ × (n / log n)^(1/3) (Politis-Romano-Wolf 2010 form).
    L = 1.5 * m_hat * (n / max(np.log(n), 1.0)) ** (1.0 / 3.0)
    return float(max(1.0, min(L, n)))


def stationary_bootstrap_indices(
    n: int, block_length: float, *, seed: int = 42, n_reps: int = 2000
) -> np.ndarray:
    """Generate ``(n_reps, n)`` resample indices via Politis-Romano stationary boot."""
    rng = np.random.default_rng(seed)
    p = 1.0 / max(block_length, 1.0)
    out = np.zeros((n_reps, n), dtype=int)
    for b in range(n_reps):
        idx = np.empty(n, dtype=int)
        i = 0
        while i < n:
            start = int(rng.integers(0, n))
            # Geometric block length.
            L = int(rng.geometric(p))
            for k in range(L):
                if i >= n:
                    break
                idx[i] = (start + k) % n
                i += 1
        out[b] = idx
    return out


def stationary_bootstrap_sharpe_ci(
    returns: np.ndarray,
    *,
    alpha: float = 0.05,
    n_reps: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """**PRIMARY** §3.8 inference: studentised stationary block-bootstrap CI.

    Returns ``(lower, upper, block_length_used)``. Studentised by the
    Lo (2002) analytic SE per replicate so the band is not too narrow.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 30:
        return (float("nan"), float("nan"), float("nan"))
    sr_hat = sharpe_ratio(r)
    block_L = politis_white_block_length(r)
    indices = stationary_bootstrap_indices(n, block_L, seed=seed, n_reps=n_reps)
    boot_sr = np.empty(n_reps, dtype=float)
    boot_t = np.empty(n_reps, dtype=float)
    for b in range(n_reps):
        rs = r[indices[b]]
        sr_b = sharpe_ratio(rs)
        s = stats.skew(rs, bias=False)
        k = stats.kurtosis(rs, bias=False, fisher=False)
        se_b = sharpe_std(sr_b, len(rs), s, k)
        boot_sr[b] = sr_b
        boot_t[b] = (sr_b - sr_hat) / se_b if se_b > 0 else 0.0
    # Studentised pivot: invert quantiles of (sr_b − sr_hat) / se_b.
    se_hat = sharpe_std(
        sr_hat, n, stats.skew(r, bias=False), stats.kurtosis(r, bias=False, fisher=False)
    )
    q_lo, q_hi = np.quantile(boot_t, [alpha / 2.0, 1.0 - alpha / 2.0])
    lower = sr_hat - q_hi * se_hat
    upper = sr_hat - q_lo * se_hat
    return (float(lower), float(upper), float(block_L))


# ---------------------------------------------------------------------------
# PSR + MinTRL.
# ---------------------------------------------------------------------------


def probabilistic_sharpe_ratio(
    returns: np.ndarray, *, sr_benchmark: float = 0.0
) -> float:
    """PSR(SR*) = Φ((SR − SR*) / σ(SR̂))."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sr = sharpe_ratio(r)
    if not np.isfinite(sr):
        return float("nan")
    s = stats.skew(r, bias=False)
    k = stats.kurtosis(r, bias=False, fisher=False)
    se = sharpe_std(sr, len(r), s, k)
    if se <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr_benchmark) / se))


def min_track_record_length(
    returns: np.ndarray, *, sr_benchmark: float = 0.0, prob: float = 0.95
) -> float:
    """MinTRL — how many periods of track record needed to be confident of skill.

    Bailey-Lopez de Prado 2012:
        MinTRL = 1 + (1 − skew·SR + (kurt−1)/4 · SR²) · (z_p / (SR − SR*))²
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sr = sharpe_ratio(r)
    if not np.isfinite(sr) or sr <= sr_benchmark:
        return float("nan")
    s = stats.skew(r, bias=False)
    k = stats.kurtosis(r, bias=False, fisher=False)
    z = stats.norm.ppf(prob)
    var_factor = 1.0 - s * sr + ((k - 1.0) / 4.0) * sr ** 2
    return float(1.0 + max(var_factor, 0.0) * (z / (sr - sr_benchmark)) ** 2)


# ---------------------------------------------------------------------------
# Ljung-Box gate.
# ---------------------------------------------------------------------------


def ljung_box_test(returns: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """Ljung-Box Q(lags) — IID gate before √252 annualisation."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < lags + 1:
        return (float("nan"), float("nan"))
    res = acorr_ljungbox(r, lags=[lags], return_df=True)
    return (float(res["lb_stat"].iloc[0]), float(res["lb_pvalue"].iloc[0]))


# ---------------------------------------------------------------------------
# Convenience summary.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignificanceReport:
    sr: float
    n: int
    t_stat: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    bootstrap_block_length: float
    analytic_ci_low: float
    analytic_ci_high: float
    psr_zero: float
    min_trl_days: float
    ljung_box_stat: float
    ljung_box_p: float


def significance_report(
    returns: np.ndarray,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 42,
) -> SignificanceReport:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sr = sharpe_ratio(r)
    n = len(r)
    t_stat = t_statistic(r)
    bl, bh, bL = stationary_bootstrap_sharpe_ci(r, alpha=alpha, n_reps=n_boot, seed=seed)
    al, ah = sharpe_ci_analytic(r, alpha=alpha)
    psr = probabilistic_sharpe_ratio(r)
    mtrl = min_track_record_length(r)
    lb_stat, lb_p = ljung_box_test(r)
    return SignificanceReport(
        sr=sr, n=n, t_stat=t_stat,
        bootstrap_ci_low=bl, bootstrap_ci_high=bh, bootstrap_block_length=bL,
        analytic_ci_low=al, analytic_ci_high=ah,
        psr_zero=psr, min_trl_days=mtrl,
        ljung_box_stat=lb_stat, ljung_box_p=lb_p,
    )
