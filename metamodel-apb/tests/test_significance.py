"""S6.14 significance-first inference (RED-first, known-answer).

At T≈128 the raw annual Sharpe (~1.4–1.55) gives t≈1.0 — not significant before any
deflation. §6 must therefore be led by significance, in assumption-strength order: t-stat →
studentised stationary block-bootstrap CI (primary) → Lo/Opdyke analytic band → PSR/MinTRL →
DSR/PBO (demoted, in deflation.py). These tests pin each new primitive to its closed form.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kurtosis as _kurt
from scipy.stats import norm
from scipy.stats import skew as _skew

from alken_metamodel.deflation import sharpe_ratio, sharpe_std
from alken_metamodel.significance import (
    ljung_box_test,
    min_track_record_length,
    sharpe_ci_analytic,
    stationary_bootstrap_cer_diff_ci,
    stationary_bootstrap_sharpe_ci,
    t_statistic,
)


def test_t_statistic_is_sharpe_times_sqrt_n():
    # r = 1..5: mean 3, var(ddof=1)=2.5 -> SR=1.897367, t=SR*sqrt(5)=4.242641
    r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert t_statistic(r) == pytest.approx(4.242640687, rel=1e-7)
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 200)
    assert t_statistic(x) == pytest.approx(sharpe_ratio(x) * np.sqrt(len(x)), rel=1e-9)


def test_ljung_box_flags_ar1_and_passes_white_noise():
    rng = np.random.default_rng(1)
    wn = rng.normal(0.0, 1.0, 600)
    _, p_wn = ljung_box_test(wn, lags=10)
    assert p_wn > 0.05  # cannot reject IID for white noise

    ar = np.zeros(600)
    e = rng.normal(0.0, 1.0, 600)
    for t in range(1, 600):
        ar[t] = 0.6 * ar[t - 1] + e[t]
    _, p_ar = ljung_box_test(ar, lags=10)
    assert p_ar < 0.01  # strongly rejects IID for AR(1)


def test_sharpe_ci_analytic_matches_lo_band():
    rng = np.random.default_rng(2)
    r = rng.normal(0.001, 0.01, 300)
    lo, hi = sharpe_ci_analytic(r, alpha=0.05)
    sr = sharpe_ratio(r)
    se = sharpe_std(sr, len(r), skew=float(_skew(r)), kurt=float(_kurt(r, fisher=False)))
    z = norm.ppf(0.975)
    assert lo < sr < hi
    assert (hi - lo) == pytest.approx(2.0 * z * se, rel=1e-9)


def test_min_track_record_length_matches_formula():
    z = norm.ppf(0.975)
    # skew 0, kurt 3 -> variance term = 1 + 0.5*SR^2; SR=0.1, SR*=0
    expected = 1.0 + (1.0 + 0.5 * 0.1**2) * (z / 0.1) ** 2
    got = min_track_record_length(0.1, 0.0, skew=0.0, kurt=3.0, prob=0.975)
    assert got == pytest.approx(expected, rel=1e-9)
    # an SR at or below the benchmark can never accumulate enough track record
    assert np.isinf(min_track_record_length(0.0, 0.05))


def test_bootstrap_ci_recovers_analytic_se_on_iid():
    """On IID returns the studentised stationary-bootstrap CI ≈ the Lo analytic band."""
    rng = np.random.default_rng(7)
    r = rng.normal(0.0008, 0.01, 750)
    lo_b, hi_b = stationary_bootstrap_sharpe_ci(r, alpha=0.05, reps=1500, seed=42)
    lo_a, hi_a = sharpe_ci_analytic(r, alpha=0.05)
    assert (hi_b - lo_b) == pytest.approx(hi_a - lo_a, rel=0.25)
    assert lo_b < sharpe_ratio(r) < hi_b


def test_bootstrap_ci_is_deterministic():
    rng = np.random.default_rng(3)
    r = rng.normal(0.001, 0.012, 400)
    a = stationary_bootstrap_sharpe_ci(r, seed=42, reps=800)
    b = stationary_bootstrap_sharpe_ci(r, seed=42, reps=800)
    assert a == b


# --- EX.6: paired studentised CER-difference bootstrap -----------------------


def test_cer_diff_identical_series_contains_zero():
    """Identical paired series → CER difference is exactly 0; the CI is the point {0}."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, size=200)
    lo, hi = stationary_bootstrap_cer_diff_ci(r, r, risk_aversion=5.0, seed=42)
    assert lo <= 0.0 <= hi
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert hi == pytest.approx(0.0, abs=1e-12)


def test_cer_diff_constant_shift_is_deterministic_and_excludes_zero():
    """A constant +c level shift raises the mean by c, leaves variance unchanged → CER diff = c
    exactly; the paired difference has zero sampling variance so the CI collapses to {c} > 0."""
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 0.01, size=200)
    alt = base + 0.005
    lo, hi = stationary_bootstrap_cer_diff_ci(alt, base, risk_aversion=5.0, seed=42)
    assert lo == pytest.approx(0.005, abs=1e-9)
    assert hi == pytest.approx(0.005, abs=1e-9)
    assert lo > 0.0


def test_cer_diff_is_paired_and_deterministic():
    """A genuine (non-degenerate) difference yields a finite CI, deterministic under a fixed seed,
    and bracketing the plug-in CER difference."""
    rng = np.random.default_rng(5)
    base = rng.normal(0.0005, 0.01, size=300)
    alt = 0.5 * base  # a rescaled sibling: lower mean and lower variance, correlated with base
    a = stationary_bootstrap_cer_diff_ci(alt, base, risk_aversion=5.0, seed=42, reps=1000)
    b = stationary_bootstrap_cer_diff_ci(alt, base, risk_aversion=5.0, seed=42, reps=1000)
    assert a == b  # seeded → byte-stable
    lo, hi = a
    point = (alt.mean() - 0.5 * 5.0 * alt.var(ddof=1)) - (
        base.mean() - 0.5 * 5.0 * base.var(ddof=1)
    )
    assert np.isfinite(lo) and np.isfinite(hi) and lo < point < hi
