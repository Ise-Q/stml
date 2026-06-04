"""S7 tests — significance + deflation + signal_analysis.

Plan §8 S7 RED-first invariants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# t-statistic = SR · √n.
# ---------------------------------------------------------------------------


def test_t_stat_formula():
    from stml.experimental.significance import sharpe_ratio, t_statistic
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 252)
    sr = sharpe_ratio(r)
    t = t_statistic(r)
    np.testing.assert_allclose(t, sr * np.sqrt(252), atol=1e-9)


def test_sharpe_std_formula_on_known_inputs():
    """Mertens 2002: Var = (1 − skew·SR + (kurt−1)/4 · SR²) / (n−1)."""
    from stml.experimental.significance import sharpe_std
    val = sharpe_std(0.1, n=100, skew=0.0, kurt=3.0)  # Normal: kurt=3
    expected = np.sqrt((1.0 - 0 + (3 - 1) / 4 * 0.01) / 99)
    np.testing.assert_allclose(val, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# Stationary bootstrap CI — seed-stable + bracket Sharpe.
# ---------------------------------------------------------------------------


def test_stationary_bootstrap_seed_deterministic():
    from stml.experimental.significance import stationary_bootstrap_sharpe_ci
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 250)
    c1 = stationary_bootstrap_sharpe_ci(r, n_reps=200, seed=42)
    c2 = stationary_bootstrap_sharpe_ci(r, n_reps=200, seed=42)
    np.testing.assert_allclose(c1[:2], c2[:2])


def test_stationary_bootstrap_brackets_sharpe():
    from stml.experimental.significance import sharpe_ratio, stationary_bootstrap_sharpe_ci
    rng = np.random.default_rng(1)
    r = rng.normal(0.0005, 0.01, 500)
    sr = sharpe_ratio(r)
    lo, hi, _ = stationary_bootstrap_sharpe_ci(r, n_reps=500, seed=42)
    assert lo <= sr <= hi


def test_ljung_box_returns_p_value():
    from stml.experimental.significance import ljung_box_test
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.01, 300)
    stat, p = ljung_box_test(r, lags=10)
    assert np.isfinite(stat)
    assert 0 <= p <= 1


# ---------------------------------------------------------------------------
# MinTRL + PSR.
# ---------------------------------------------------------------------------


def test_psr_increases_with_sharpe():
    """Higher Sharpe → higher PSR."""
    from stml.experimental.significance import probabilistic_sharpe_ratio
    rng = np.random.default_rng(3)
    r_low = rng.normal(0.0001, 0.01, 252)
    r_high = rng.normal(0.001, 0.01, 252)
    psr_low = probabilistic_sharpe_ratio(r_low)
    psr_high = probabilistic_sharpe_ratio(r_high)
    assert psr_high > psr_low


def test_min_trl_formula():
    """MinTRL is positive when SR > benchmark."""
    from stml.experimental.significance import min_track_record_length
    rng = np.random.default_rng(4)
    r = rng.normal(0.001, 0.01, 252)
    mtrl = min_track_record_length(r, sr_benchmark=0.0)
    assert mtrl > 0


# ---------------------------------------------------------------------------
# Deflation — CSCV-PBO combinations + DSR ladder.
# ---------------------------------------------------------------------------


def test_cscv_pbo_combinations_count_16_blocks_is_12870():
    """The famous "12,780 typo" — actual count is C(16, 8) = 12,870."""
    from stml.experimental.deflation import cscv_pbo_combinations_count
    assert cscv_pbo_combinations_count(16) == 12_870


def test_expected_max_sharpe_increases_with_n():
    from stml.experimental.deflation import expected_max_sharpe
    assert expected_max_sharpe(5) < expected_max_sharpe(50)
    assert expected_max_sharpe(50) < expected_max_sharpe(500)


def test_dsr_ladder_dsr_falls_as_n_grows():
    """DSR decreases as more trials are assumed (more selection bias)."""
    from stml.experimental.deflation import dsr_ladder
    rng = np.random.default_rng(5)
    r = rng.normal(0.001, 0.01, 252)
    ladder = dsr_ladder(r, n_eff=2, n_raw=10)
    dsr_values = ladder["dsr"].values
    # Most rungs should be non-increasing (greater N → smaller DSR).
    assert dsr_values[0] >= dsr_values[-1] - 1e-6


def test_probability_of_backtest_overfitting_in_unit_interval():
    """PBO ∈ [0, 1]."""
    from stml.experimental.deflation import probability_of_backtest_overfitting
    rng = np.random.default_rng(6)
    # 6 trials × 32 periods.
    perf = rng.normal(0, 1, (32, 6))
    pbo = probability_of_backtest_overfitting(perf, n_blocks=4)
    assert 0 <= pbo <= 1


# ---------------------------------------------------------------------------
# Pesaran-Timmermann — constant call → S = 0.
# ---------------------------------------------------------------------------


def test_pesaran_timmermann_constant_call_zero_stat():
    """A constant positive call in a balanced market → S → 0 (no spurious skill)."""
    from stml.experimental.signal_analysis import pesaran_timmermann
    rng = np.random.default_rng(7)
    realised = rng.normal(0, 1, 200)
    predicted = np.full(200, 1.0)  # always predict +1
    # PT is undefined (var becomes 0) when predicted is constant — should return NaN
    # gracefully rather than blowing up.
    S, p = pesaran_timmermann(realised, predicted)
    assert np.isnan(S) or abs(S) < 1.5  # near zero or NaN, NOT a "huge skill" signal


def test_pesaran_timmermann_finite_on_balanced_real_skill():
    """When PT is well-defined, it returns finite stat with a valid p-value."""
    from stml.experimental.signal_analysis import pesaran_timmermann
    rng = np.random.default_rng(8)
    realised = rng.normal(0.1, 1.0, 200)
    predicted = rng.normal(0, 1, 200)
    S, p = pesaran_timmermann(realised, predicted)
    if np.isfinite(S):
        assert 0 <= p <= 1


def test_pesaran_timmermann_real_skill_rejects_null():
    """Genuine directional skill -> PT stat large and positive, p < 0.01.

    Constructs a sample where predicted aligns with realised 65 % of the
    time on n = 500. The bug-fixed denominator (cross term /n^2) must give
    a positive denom and a meaningful S statistic.
    """
    from stml.experimental.signal_analysis import pesaran_timmermann
    rng = np.random.default_rng(11)
    n = 500
    realised = rng.normal(0, 1, n)
    predicted = realised.copy()
    flip = rng.random(n) < 0.35
    predicted[flip] = -predicted[flip]
    S, p = pesaran_timmermann(realised, predicted)
    assert np.isfinite(S) and S > 3.0, f"expected S > 3, got {S}"
    assert p < 0.01, f"expected p < 0.01, got {p}"


def test_pesaran_timmermann_denom_positive_balanced():
    """Bug regression: with balanced Py, Px around 0.5 on a realistic n,
    var(P_hat) - var(P_star) must be POSITIVE (i.e. PT not NaN). The
    pre-fix formula treated the cross term as /n instead of /n^2 and
    flipped the sign of the denominator for our H1-2022 OOS sample.
    """
    from stml.experimental.signal_analysis import pesaran_timmermann
    rng = np.random.default_rng(13)
    n = 1342
    realised = rng.normal(0, 1, n)
    predicted = rng.normal(0, 1, n)
    S, p = pesaran_timmermann(realised, predicted)
    assert np.isfinite(S), "PT denom collapsed — pre-fix bug regression"
    assert 0 <= p <= 1


# ---------------------------------------------------------------------------
# Treynor-Mazuy.
# ---------------------------------------------------------------------------


def test_treynor_mazuy_zero_convexity_for_linear_strategy():
    """Linear β-1 strategy → γ ~ 0."""
    from stml.experimental.signal_analysis import treynor_mazuy
    rng = np.random.default_rng(9)
    market = rng.normal(0.001, 0.01, 200)
    portfolio = market + 0.001 * rng.normal(0, 0.01, 200)
    gamma, t = treynor_mazuy(market, portfolio)
    assert abs(t) < 3.0  # not strongly significant


def test_information_ratio_formula():
    """IR = IC · √BR."""
    from stml.experimental.signal_analysis import information_ratio
    np.testing.assert_allclose(information_ratio(0.07, 11), 0.07 * np.sqrt(11))
