"""EX.5 primary-signal characterisation (RED-first).

Before judging what the metamodel adds, we characterise the *provided* primary signal: how often
it flips (turnover), how often it is directionally right (hit-rate), its information coefficient
(rank corr with the forward return), and the fundamental-law information ratio IR = IC·√breadth
(/aqms-python L4). Known-value tests pin each.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from alken_metamodel.signal_analysis import (
    henriksson_merton,
    information_coefficient,
    information_ratio,
    signal_hit_rate,
    signal_turnover,
)


def test_signal_hit_rate_counts_only_nonzero_signals():
    signal = pd.Series([1, -1, 0, 1])
    fwd_ret = pd.Series([0.01, -0.02, 0.50, -0.01])
    # nonzero at 0,1,3: +/+ hit, -/- hit, +/- miss -> 2 of 3
    assert signal_hit_rate(signal, fwd_ret) == pytest.approx(2 / 3)


def test_signal_turnover_counts_position_flips():
    signal = pd.Series([1, 1, -1, 0])
    # |Δ| = [_, 0, 2, 1]; /2 -> [0, 1, 0.5]; mean = 0.5
    assert signal_turnover(signal) == pytest.approx(0.5)


def test_information_coefficient_perfect_rank():
    signal = pd.Series([-1, 0, 1, 2])
    fwd_ret = pd.Series([-0.1, 0.0, 0.1, 0.2])  # perfectly rank-aligned
    assert information_coefficient(signal, fwd_ret) == pytest.approx(1.0)


def test_information_ratio_is_ic_root_breadth():
    assert information_ratio(0.05, breadth=100) == pytest.approx(0.5)
    assert information_ratio(0.10, breadth=4) == pytest.approx(0.2)


def test_hit_rate_nan_when_no_signal():
    assert np.isnan(signal_hit_rate(pd.Series([0, 0, 0]), pd.Series([0.1, -0.1, 0.2])))


# --- S5.8: Henriksson–Merton market-timing test ----------------------------


def test_henriksson_merton_perfect_timing_is_significant():
    real = np.array([1, -1, 1, -1, 1, -1, 1, -1.0] * 10)
    hit, z, p = henriksson_merton(real, real.copy())  # perfect directional calls
    assert hit == pytest.approx(1.0)
    assert p < 1e-6  # overwhelmingly significant timing


def test_henriksson_merton_no_timing_not_significant():
    rng = np.random.default_rng(0)
    real = rng.choice([-1.0, 1.0], 500)
    pred = rng.choice([-1.0, 1.0], 500)  # independent of the realised direction
    hit, z, p = henriksson_merton(real, pred)
    assert 0.40 < hit < 0.60
    assert p > 0.05  # cannot reject the no-timing null


def test_henriksson_merton_known_z_and_pvalue():
    # 60 of 100 correct -> p_hat=0.6, z=(0.6-0.5)/sqrt(0.25/100)=2.0
    real = np.ones(100)
    pred = np.ones(100)
    pred[:40] = -1.0  # 60 correct, 40 wrong
    hit, z, p = henriksson_merton(real, pred)
    assert hit == pytest.approx(0.6)
    assert z == pytest.approx(2.0)
    assert p == pytest.approx(1.0 - norm.cdf(2.0))
