"""Significance-first inference for §6 (S6.14, LR-6).

The §6 Sharpe is selected from a horse-race, but the more basic question — is it even
distinguishable from zero on the ~128-day OOS window? — comes *before* any selection-bias
deflation. This module reports significance in assumption-strength order:

1. ``t_statistic`` — the raw t = SR·√n (no annualisation needed; an honest first read).
2. ``stationary_bootstrap_sharpe_ci`` — the PRIMARY inference: a studentised stationary
   block-bootstrap CI (Politis–Romano 1994), block length from Politis–White's
   ``optimal_block_length``, studentised with the Lo (2002) analytic SE to avoid arch's
   slow/fragile nested resampling. Seeded → deterministic.
3. ``sharpe_ci_analytic`` — the Lo/Opdyke closed-form band (a parametric cross-check).
4. ``min_track_record_length`` — MinTRL: the track length PSR needs to clear a benchmark.
5. PSR / DSR / CSCV-PBO live in ``deflation.py`` and are reported *demoted*, as corroboration.

``ljung_box_test`` is a SEPARATE IID gate (run it before any √252 annualisation); it is NOT
the bootstrap block-length selector (that is Politis–White's, above). All Sharpes are
per-period; annualise only after the Ljung–Box check supports it.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis as _scipy_kurtosis
from scipy.stats import norm
from scipy.stats import skew as _scipy_skew
from statsmodels.stats.diagnostic import acorr_ljungbox

from .deflation import sharpe_ratio, sharpe_std


def _moments(r: np.ndarray) -> tuple[float, float]:
    """(skew, raw-kurtosis) — the standardised 3rd and 4th moments the Lo SE consumes."""
    return float(_scipy_skew(r)), float(_scipy_kurtosis(r, fisher=False))


def t_statistic(returns) -> float:
    """Student t of the mean return = SR·√n (per-period; no annualisation). NaN if degenerate."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sr = sharpe_ratio(r)
    if not np.isfinite(sr):
        return float("nan")
    return float(sr * np.sqrt(r.size))


def ljung_box_test(returns, lags: int = 10) -> tuple[float, float]:
    """Ljung–Box Q test for autocorrelation up to ``lags`` → ``(stat, p_value)``.

    The IID gate to run BEFORE √252-annualising a Sharpe: a small p-value means the returns
    are serially correlated, so the naive √252 scaling overstates the annualised Sharpe.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    res = acorr_ljungbox(r, lags=[lags], return_df=True)
    return float(res["lb_stat"].iloc[-1]), float(res["lb_pvalue"].iloc[-1])


def sharpe_ci_analytic(returns, *, alpha: float = 0.05) -> tuple[float, float]:
    """Lo (2002) / Opdyke (2007) analytic Sharpe CI: ``SR ± z_{1-α/2}·σ(SR̂)`` (per-period)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sr = sharpe_ratio(r)
    if not np.isfinite(sr):
        return float("nan"), float("nan")
    skew, kurt = _moments(r)
    se = sharpe_std(sr, r.size, skew=skew, kurt=kurt)
    z = norm.ppf(1.0 - alpha / 2.0)
    return sr - z * se, sr + z * se


def stationary_bootstrap_sharpe_ci(
    returns, *, alpha: float = 0.05, reps: int = 2000, seed: int = 42
) -> tuple[float, float]:
    """Studentised stationary block-bootstrap CI for the per-period Sharpe (the PRIMARY §6 read).

    Block length is Politis–White's ``optimal_block_length`` (data-driven, deterministic);
    each replicate is studentised by the Lo analytic SE (``std_err_func``), so arch does not
    fall back to a nested bootstrap. The RNG is seeded → byte-stable. Returns ``(lo, hi)``.
    """
    from arch.bootstrap import StationaryBootstrap, optimal_block_length

    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 8:
        return float("nan"), float("nan")
    block = float(optimal_block_length(r)["stationary"].iloc[0])

    def _sr(data) -> np.ndarray:
        d = np.asarray(data).ravel()
        return np.array([d.mean() / d.std(ddof=1)])

    def _se(theta, data) -> np.ndarray:
        d = np.asarray(data).ravel()
        skew, kurt = _moments(d)
        return np.array([sharpe_std(float(np.atleast_1d(theta)[0]), d.size, skew=skew, kurt=kurt)])

    bs = StationaryBootstrap(block, r, seed=seed)
    ci = np.asarray(
        bs.conf_int(_sr, reps=reps, method="studentized", size=1.0 - alpha, std_err_func=_se)
    ).ravel()
    return float(ci[0]), float(ci[1])


def min_track_record_length(
    sr: float,
    sr_benchmark: float = 0.0,
    *,
    skew: float = 0.0,
    kurt: float = 3.0,
    prob: float = 0.95,
) -> float:
    """MinTRL (Bailey & López de Prado 2012): track length for PSR(SR*) to reach ``prob``.

    ``MinTRL = 1 + (1 − skew·SR + ((kurt−1)/4)·SR²)·(z_prob/(SR − SR*))²`` (per-period units).
    Returns ``inf`` when the Sharpe does not exceed the benchmark (never enough evidence).
    """
    if sr <= sr_benchmark:
        return float("inf")
    z = norm.ppf(prob)
    var_term = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    return float(1.0 + var_term * (z / (sr - sr_benchmark)) ** 2)
