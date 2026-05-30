"""Deflation gate (S6.8) — DSR / MinBTL / CSCV-PBO, RED-first known-answer tests.

The §6 strategy Sharpe is selected from N configurations, so it overstates skill; before it can be
read as anything but a number it must be deflated. These tests pin the three statistics to their
closed forms (Bailey & López de Prado 2014; Bailey–Borwein–LdP–Zhu 2017) on synthetic tracks of
known (SR, N, skew, kurtosis) — not on plausibility — because a subtly-wrong expected-max or
variance term looks right and is the whole failure mode here.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from alken_metamodel.deflation import (
    deflated_sharpe_ratio,
    effective_n_trials,
    expected_max_sharpe,
    min_backtest_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
    sharpe_std,
)

EULER_GAMMA = 0.5772156649015329


def _track(target_sharpe: float, n: int, seed: int = 0) -> np.ndarray:
    """A return series with sample per-period Sharpe == target_sharpe (standardise then shift)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    x = (x - x.mean()) / x.std(ddof=1)  # sample mean 0, sample std (ddof=1) 1
    return x + target_sharpe  # sample mean target, std 1 -> sample sharpe == target


# ---- sharpe_ratio ----------------------------------------------------------


def test_sharpe_ratio_recovers_planted_value():
    assert abs(sharpe_ratio(_track(0.25, 1000, seed=7)) - 0.25) < 1e-9


def test_sharpe_ratio_nan_on_zero_variance():
    assert math.isnan(sharpe_ratio(np.full(50, 0.01)))


# ---- sharpe_std (Mertens/Lo non-normal SE) ---------------------------------


def test_sharpe_std_normal_reduces_to_lo_formula():
    sr, n = 0.1, 250
    expected = math.sqrt((1.0 + 0.5 * sr * sr) / (n - 1))  # skew 0, kurt 3
    assert abs(sharpe_std(sr, n, skew=0.0, kurt=3.0) - expected) < 1e-12


def test_sharpe_std_increases_with_kurtosis():
    assert sharpe_std(0.2, 250, kurt=12.0) > sharpe_std(0.2, 250, kurt=3.0)


def test_sharpe_std_negative_skew_raises_se_for_positive_sr():
    # variance term is (1 - skew*SR + ...); negative skew with positive SR => larger
    assert sharpe_std(0.2, 250, skew=-1.0) > sharpe_std(0.2, 250, skew=0.0)


# ---- expected_max_sharpe (extreme-value selection benchmark) ---------------


def test_expected_max_sharpe_matches_closed_form():
    n = 10
    expected = (1 - EULER_GAMMA) * norm.ppf(1 - 1.0 / n) + EULER_GAMMA * norm.ppf(
        1 - 1.0 / (n * math.e)
    )
    assert abs(expected_max_sharpe(n, trials_std=1.0) - expected) < 1e-12


def test_expected_max_sharpe_scales_linearly_with_trials_std():
    one = expected_max_sharpe(20, trials_std=1.0)
    two = expected_max_sharpe(20, trials_std=2.0)
    assert abs(two - 2.0 * one) < 1e-12


def test_expected_max_sharpe_monotone_increasing_in_n():
    vals = [expected_max_sharpe(n) for n in (2, 5, 10, 50, 200, 1000)]
    assert all(b > a for a, b in zip(vals, vals[1:], strict=False))


def test_expected_max_sharpe_single_trial_is_zero():
    assert expected_max_sharpe(1) == 0.0  # no selection bias with one trial


# ---- probabilistic_sharpe_ratio --------------------------------------------


def test_psr_half_when_sr_equals_benchmark():
    assert abs(probabilistic_sharpe_ratio(0.15, 0.15, 250) - 0.5) < 1e-12


def test_psr_matches_phi_closed_form():
    sr, bench, n = 0.2, 0.05, 300
    expected = norm.cdf((sr - bench) / sharpe_std(sr, n))
    assert abs(probabilistic_sharpe_ratio(sr, bench, n) - expected) < 1e-12


def test_psr_monotone_in_observed_sr():
    lo = probabilistic_sharpe_ratio(0.1, 0.0, 250)
    hi = probabilistic_sharpe_ratio(0.3, 0.0, 250)
    assert 0.5 < lo < hi < 1.0


# ---- deflated_sharpe_ratio -------------------------------------------------


def test_dsr_decreases_as_n_trials_rises():
    r = _track(0.2, 500)
    small = deflated_sharpe_ratio(r, n_trials=2, trials_sharpe_std=0.5)
    big = deflated_sharpe_ratio(r, n_trials=100, trials_sharpe_std=0.5)
    assert small > big


def test_dsr_half_when_observed_equals_expected_max():
    n_trials, tstd = 10, 0.5
    sr0 = expected_max_sharpe(n_trials, trials_std=tstd)
    r = _track(sr0, 800)  # sample sharpe == SR0 => numerator 0 => PSR 0.5
    assert abs(deflated_sharpe_ratio(r, n_trials=n_trials, trials_sharpe_std=tstd) - 0.5) < 1e-6


# ---- min_backtest_length ---------------------------------------------------


def test_min_backtest_length_known_value():
    n, target = 10, 1.0
    z = expected_max_sharpe(n, trials_std=1.0)
    assert abs(min_backtest_length(n, target_sharpe=target) - z * z) < 1e-12


def test_min_backtest_length_increases_with_n():
    assert min_backtest_length(500, target_sharpe=1.0) > min_backtest_length(5, target_sharpe=1.0)


def test_min_backtest_length_scales_inverse_square_in_target():
    base = min_backtest_length(50, target_sharpe=1.0)
    half = min_backtest_length(50, target_sharpe=0.5)
    assert abs(half - 4.0 * base) < 1e-9


# ---- probability_of_backtest_overfitting (CSCV) ----------------------------


def _noise(t=240, n=8, seed=1):
    return np.random.default_rng(seed).standard_normal((t, n)) * 0.01


def test_pbo_zero_for_dominant_trial():
    M = _noise(seed=0)
    M[:, 0] += 0.05  # trial 0 dominates every block IS and OOS
    res = probability_of_backtest_overfitting(M, n_blocks=8)
    assert res["pbo"] < 0.05


def test_pbo_near_half_for_pure_noise():
    # E[PBO] under noise is exactly 0.5 (the IS-best trial's OOS rank is uniform); the C(8,4) splits
    # share blocks, so a single draw is high-variance — average over seeds to test the expectation.
    pbos = [
        probability_of_backtest_overfitting(_noise(seed=s), n_blocks=8)["pbo"] for s in range(12)
    ]
    assert 0.35 < float(np.mean(pbos)) < 0.65


def test_pbo_high_for_anticorrelated_halves():
    # first half rewards the high-bias trial, second half punishes it by the same amount:
    # the IS-best on either half is the OOS-worst on the other -> PBO == 1
    t, n = 240, 8
    M = _noise(t=t, n=n, seed=2)
    half = t // 2
    bias = np.linspace(0.02, 0.08, n)
    M[:half, :] += bias
    M[half:, :] -= bias
    res = probability_of_backtest_overfitting(M, n_blocks=2)
    assert res["pbo"] > 0.9


def test_pbo_split_count_is_combinatorial():
    res = probability_of_backtest_overfitting(_noise(n=5), n_blocks=6)
    assert res["n_splits"] == math.comb(6, 3)


def test_pbo_rejects_odd_n_blocks():
    try:
        probability_of_backtest_overfitting(_noise(), n_blocks=7)
    except ValueError:
        return
    raise AssertionError("odd n_blocks must raise (CSCV needs symmetric halves)")


# ---- effective_n_trials (ONC) ----------------------------------------------


def test_effective_n_trials_collapses_correlated_trials():
    # 6 trial columns = 2 underlying signals each duplicated 3x with small noise
    rng = np.random.default_rng(3)
    a = rng.standard_normal(300)
    b = rng.standard_normal(300)
    cols = [a, a + 0.01 * rng.standard_normal(300), a + 0.01 * rng.standard_normal(300),
            b, b + 0.01 * rng.standard_normal(300), b + 0.01 * rng.standard_normal(300)]
    M = np.column_stack(cols)
    n_eff = effective_n_trials(M)
    assert 1 <= n_eff <= 4  # well below the 6 raw trials
    assert n_eff < 6
