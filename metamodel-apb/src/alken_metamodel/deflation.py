"""Backtest deflation for the §6 deployment gate (S6.8).

A single reported Sharpe is selected from N configurations tried during model selection, so it
overstates skill (selection bias) and ignores non-normality. This module supplies the three
standard corrections so §6 is reported *deflated* — never as "the strategy works":

- **Probabilistic / Deflated Sharpe Ratio** (Bailey & López de Prado 2014): PSR of the observed
  Sharpe against the selection-bias-adjusted benchmark ``SR0 = E[max of N trials]``.
- **Minimum Backtest Length** (Bailey et al. 2014): the track length below which an N-trial max
  Sharpe is indistinguishable from luck.
- **Probability of Backtest Overfitting** via CSCV (Bailey, Borwein, López de Prado & Zhu 2017).

The grade is methodology: the gate is reported as a *range* over N ∈ [N_eff → N_raw] (a single
backtest cannot pin N), and if even the optimistic end does not clear, that is the honest finding.
All Sharpes are per-period; the observed Sharpe and the benchmark must share a frequency.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy.stats import norm, rankdata

EULER_GAMMA = 0.5772156649015329  # Euler–Mascheroni constant


def sharpe_ratio(returns, *, ddof: int = 1) -> float:
    """Per-period Sharpe (mean / std); NaN on fewer than two finite points or zero variance."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=ddof)
    if sd == 0:
        return float("nan")
    return float(r.mean() / sd)


def sharpe_std(sr: float, n: int, *, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Standard error of the Sharpe estimator under non-normality (Mertens 2002 / Lo 2002).

    ``Var = (1 − skew·SR + ((kurt−1)/4)·SR²) / (n−1)``; ``kurt`` is the raw standardised fourth
    moment (3 for a normal), so zero-skew normal returns recover the familiar ``(1 + ½SR²)/(n−1)``.
    """
    var = (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr) / (n - 1)
    return float(math.sqrt(max(var, 0.0)))


def expected_max_sharpe(n_trials: int, *, trials_std: float = 1.0) -> float:
    """SR0: expected maximum Sharpe of N independent zero-skill trials (extreme-value estimate).

    ``SR0 = trials_std · [(1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e))]`` (Bailey & López de Prado 2014).
    A single trial carries no selection, so ``SR0 = 0``.
    """
    if n_trials <= 1:
        return 0.0
    z = (1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n_trials) + EULER_GAMMA * norm.ppf(
        1.0 - 1.0 / (n_trials * math.e)
    )
    return float(trials_std * z)


def probabilistic_sharpe_ratio(
    sr: float, sr_benchmark: float, n: int, *, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """PSR(SR*) = P(true SR > SR*) = Φ[(SR̂ − SR*) / σ(SR̂)]."""
    se = sharpe_std(sr, n, skew=skew, kurt=kurt)
    if se == 0 or not math.isfinite(se):
        return float("nan")
    return float(norm.cdf((sr - sr_benchmark) / se))


def deflated_sharpe_ratio(returns, *, n_trials: int, trials_sharpe_std: float) -> float:
    """DSR = PSR of the observed Sharpe vs ``SR0 = E[max of N trials]`` (per-period throughout)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sr = sharpe_ratio(r)
    if not math.isfinite(sr):
        return float("nan")
    sr0 = expected_max_sharpe(n_trials, trials_std=trials_sharpe_std)
    return probabilistic_sharpe_ratio(sr, sr0, r.size, skew=_skew(r), kurt=_kurt(r))


def min_backtest_length(n_trials: int, *, target_sharpe: float) -> float:
    """MinBTL = (E[max of N standard trials])² / target_Sharpe² (years if target is annualised).

    The shortest track over which an N-trial maximum Sharpe of a zero-skill strategy stays below
    ``target_sharpe``. Returns ``inf`` for a non-positive target.
    """
    if target_sharpe <= 0:
        return float("inf")
    z = expected_max_sharpe(n_trials, trials_std=1.0)
    return float((z * z) / (target_sharpe * target_sharpe))


def _skew(r) -> float:
    r = np.asarray(r, dtype=float)
    s = r.std()
    if s == 0:
        return 0.0
    return float(np.mean(((r - r.mean()) / s) ** 3))


def _kurt(r) -> float:
    r = np.asarray(r, dtype=float)
    s = r.std()
    if s == 0:
        return 3.0
    return float(np.mean(((r - r.mean()) / s) ** 4))


def _block_sharpe(matrix: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Per-trial (per-column) Sharpe over the selected rows; zero-variance columns score 0."""
    sub = matrix[rows, :]
    mean = sub.mean(axis=0)
    sd = sub.std(axis=0, ddof=1)
    return np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)


def probability_of_backtest_overfitting(matrix, *, n_blocks: int = 16) -> dict:
    """CSCV PBO: fraction of combinatorial IS/OOS splits where the IS-best trial is OOS-sub-median.

    ``matrix`` is (T observations × N trials) of per-period performance. The rows split into
    ``n_blocks`` contiguous blocks; for each C(S, S/2) choice of in-sample blocks the IS-best
    trial's relative OOS rank gives a logit λ, and ``PBO = P(λ < 0)``. PBO≈0 for a dominant trial,
    ≈0.5 for noise, →1 for an overfit selection. Returns ``{pbo, n_splits, logits}``.
    """
    matrix = np.asarray(matrix, dtype=float)
    n_obs, n_trials = matrix.shape
    if n_blocks % 2 != 0:
        raise ValueError("n_blocks must be even for symmetric CSCV splits")
    blocks = np.array_split(np.arange(n_obs), n_blocks)
    logits = []
    for is_blocks in combinations(range(n_blocks), n_blocks // 2):
        is_set = set(is_blocks)
        is_rows = np.concatenate([blocks[b] for b in is_blocks])
        oos_rows = np.concatenate([blocks[b] for b in range(n_blocks) if b not in is_set])
        n_star = int(np.argmax(_block_sharpe(matrix, is_rows)))
        omega = rankdata(_block_sharpe(matrix, oos_rows))[n_star] / (n_trials + 1.0)
        omega = min(max(omega, 1e-9), 1.0 - 1e-9)
        logits.append(math.log(omega / (1.0 - omega)))
    arr = np.asarray(logits, dtype=float)
    return {"pbo": float(np.mean(arr < 0)), "n_splits": int(arr.size), "logits": arr}


def effective_n_trials(perf_matrix, *, seed: int = 42, max_clusters: int = 10) -> int:
    """ONC effective trial count: cluster the trial-correlation matrix (Mantegna distance) and
    count clusters. Correlated trials collapse, so ``N_eff ≤ N_raw`` — the optimistic end of the
    DSR range. Reuses the §4 ``make_clusters`` machinery; falls back to N on a clustering failure.
    """
    import pandas as pd

    from .cluster_importance import make_clusters

    frame = pd.DataFrame(np.asarray(perf_matrix, dtype=float))
    n_trials = frame.shape[1]
    if n_trials < 2:
        return int(n_trials)
    try:
        clusters, _ = make_clusters(
            frame, seed=seed, max_clusters=min(max_clusters, n_trials - 1)
        )
        n_eff = int(len(clusters))
    except Exception:
        return int(n_trials)
    return max(1, min(n_eff, n_trials))
