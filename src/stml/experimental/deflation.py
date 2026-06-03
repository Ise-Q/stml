"""Backtest deflation — plan §3.8 / §8 S7 corroboration.

* DSR (Bailey-Lopez de Prado 2014) ladder over N_eff → 4·N_raw
* CSCV-PBO with C(16, 8) = 12,870 (corrects the long-propagated "12,780" typo)
* MinBTL — Minimum Backtest Length
* ONC N_eff — effective trial count via Mantegna-clustered trial correlations

Lifted from ``metamodel-apb/src/alken_metamodel/deflation.py`` (alken parity).
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from scipy import stats

from stml.experimental.significance import sharpe_ratio, sharpe_std


def expected_max_sharpe(n_trials: int, trials_std: float = 1.0) -> float:
    """E[max of N standard Sharpe trials] — Bailey-Lopez de Prado 2014.

        SR0 = trials_std · [(1 − γ) · Φ⁻¹(1 − 1/N) + γ · Φ⁻¹(1 − 1/(N·e))]
    γ = Euler-Mascheroni constant ≈ 0.5772.
    """
    if n_trials < 2:
        return 0.0
    gamma = 0.5772156649015329
    return float(
        trials_std * (
            (1.0 - gamma) * stats.norm.ppf(1.0 - 1.0 / n_trials)
            + gamma * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        )
    )


def deflated_sharpe_ratio(
    returns: np.ndarray,
    *,
    n_trials: int,
    trials_sharpe_std: float = 1.0,
) -> float:
    """DSR = PSR(SR_0) where SR_0 = E[max of N trials].

    Higher is better; > 0.95 is the deployment threshold (Bailey-Lopez de Prado).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sr = sharpe_ratio(r)
    if not np.isfinite(sr):
        return float("nan")
    sr0 = expected_max_sharpe(n_trials, trials_sharpe_std)
    s = stats.skew(r, bias=False)
    k = stats.kurtosis(r, bias=False, fisher=False)
    se = sharpe_std(sr, len(r), s, k)
    if se <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr0) / se))


def dsr_ladder(
    returns: np.ndarray,
    *,
    n_eff: int,
    n_raw: int,
    trials_std: float = 1.0,
) -> pd.DataFrame:
    """DSR over the trial-count ladder N_eff → N_raw → 2·N_raw → 4·N_raw."""
    rungs = [
        ("N_eff", n_eff),
        ("N_raw", n_raw),
        ("2·N_raw", 2 * n_raw),
        ("4·N_raw", 4 * n_raw),
    ]
    rows = []
    for label, n in rungs:
        dsr = deflated_sharpe_ratio(returns, n_trials=n, trials_sharpe_std=trials_std)
        rows.append({"rung": label, "n_trials": n, "dsr": dsr})
    return pd.DataFrame(rows)


def min_backtest_length(n_trials: int, target_sharpe: float = 1.0) -> float:
    """MinBTL = (E[max of N standard trials])² / target_Sharpe²."""
    if target_sharpe <= 0:
        return float("nan")
    sr0 = expected_max_sharpe(n_trials)
    return float(sr0 ** 2 / target_sharpe ** 2)


# ---------------------------------------------------------------------------
# CSCV-PBO — Probability of Backtest Overfitting.
# ---------------------------------------------------------------------------


def probability_of_backtest_overfitting(
    perf_matrix: np.ndarray, *, n_blocks: int = 16
) -> float:
    """CSCV PBO (Bailey, Borwein, Lopez de Prado, Zhu 2017).

    perf_matrix: shape (T, n_trials) where each column is one trial's per-period
    perf (e.g. returns or per-block Sharpe). The function splits T into
    n_blocks; for each of C(n_blocks, n_blocks/2) IS-block choices, the IS-best
    trial's relative OOS rank produces a logit λ. PBO = P(λ < 0).

    For n_blocks=16, C(16, 8) = 12,870.
    """
    perf = np.asarray(perf_matrix, dtype=float)
    if perf.ndim != 2:
        raise ValueError("perf_matrix must be 2-D")
    T, n_trials = perf.shape
    block_size = T // n_blocks
    if block_size < 2 or n_trials < 2:
        return float("nan")
    half = n_blocks // 2
    n_combos = comb(n_blocks, half)
    blocks = [perf[k * block_size:(k + 1) * block_size, :] for k in range(n_blocks)]
    lambdas = []
    for is_idx in combinations(range(n_blocks), half):
        oos_idx = tuple(i for i in range(n_blocks) if i not in is_idx)
        is_perf = np.concatenate([blocks[i] for i in is_idx]).mean(axis=0)
        oos_perf = np.concatenate([blocks[i] for i in oos_idx]).mean(axis=0)
        best_trial = int(np.argmax(is_perf))
        oos_rank = (oos_perf < oos_perf[best_trial]).sum()  # 0..n_trials-1
        # Relative rank ω = oos_rank / (n_trials - 1).
        omega = oos_rank / max(n_trials - 1, 1)
        # Logit λ = log(ω / (1 − ω)).
        if omega <= 0:
            lam = -np.inf
        elif omega >= 1:
            lam = np.inf
        else:
            lam = np.log(omega / (1.0 - omega))
        lambdas.append(lam)
    return float(np.mean(np.array(lambdas) < 0))


def cscv_pbo_combinations_count(n_blocks: int = 16) -> int:
    """Return C(n_blocks, n_blocks/2). 12,870 for n_blocks=16."""
    return comb(n_blocks, n_blocks // 2)


# ---------------------------------------------------------------------------
# ONC effective trial count.
# ---------------------------------------------------------------------------


def effective_n_trials(
    perf_matrix: np.ndarray, *, max_clusters: int = 20, seed: int = 42
) -> int:
    """ONC N_eff: cluster trial returns by Mantegna distance, count clusters."""
    perf = np.asarray(perf_matrix, dtype=float)
    if perf.ndim != 2:
        raise ValueError("perf_matrix must be 2-D")
    T, n_trials = perf.shape
    if n_trials <= 1:
        return n_trials
    # Standardise columns.
    centered = perf - perf.mean(axis=0, keepdims=True)
    sd = centered.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    centered = centered / sd
    # Pearson corr matrix.
    corr = (centered.T @ centered) / max(T - 1, 1)
    # Mantegna distance.
    dist = np.sqrt(np.clip(1.0 - np.abs(corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    # Hierarchical clustering, silhouette K selection over [2, min(max_clusters, n_trials)].
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import silhouette_score

    try:
        Z = linkage(squareform(dist, checks=False), method="ward")
    except Exception:
        return n_trials
    best_k = 1
    best_score = -np.inf
    for k in range(2, min(max_clusters, n_trials) + 1):
        try:
            labels = fcluster(Z, t=k, criterion="maxclust")
            if len(np.unique(labels)) < 2:
                continue
            sc = silhouette_score(dist, labels, metric="precomputed")
            if sc > best_score:
                best_score = sc
                best_k = k
        except Exception:
            continue
    return int(best_k)
