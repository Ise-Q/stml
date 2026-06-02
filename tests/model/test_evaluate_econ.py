"""
test_evaluate_econ.py
=====================
Tests for the economic-evaluation utilities appended to stml.model.evaluate:
bootstrap_returns, decision_threshold, adding_zeros_eval, nav_sharpe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.model.evaluate import (
    adding_zeros_eval,
    bootstrap_returns,
    decision_threshold,
    nav_sharpe,
)


# ---------------------------------------------------------------------------
# decision_threshold
# ---------------------------------------------------------------------------


def test_decision_threshold_symmetric():
    """Symmetric G=L => p* = 0.5."""
    assert decision_threshold(0.02, -0.02) == pytest.approx(0.5)


def test_decision_threshold_asymmetric():
    """G=0.03, L=0.01 => p* = 0.01/(0.03+0.01) = 0.25."""
    assert decision_threshold(0.03, -0.01) == pytest.approx(0.25)


def test_decision_threshold_edge_guard_zeros():
    """G=0, L=0 => edge case returns 0.5."""
    assert decision_threshold(0.0, 0.0) == pytest.approx(0.5)


def test_decision_threshold_near_zero_sum():
    """Very small G+L still returns 0.5 (guard)."""
    assert decision_threshold(1e-13, -1e-13) == pytest.approx(0.5)


def test_decision_threshold_high_loss():
    """G=0.01, L=0.09 => p* = 0.09/0.10 = 0.90."""
    assert decision_threshold(0.01, -0.09) == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# bootstrap_returns
# ---------------------------------------------------------------------------


def _make_labels(rets: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"ret": rets})


def test_bootstrap_returns_sign_correct():
    """r_G > 0 and r_L < 0 when both groups are non-empty."""
    labels = _make_labels([0.05, 0.03, -0.02, -0.04])
    r_G, r_L = bootstrap_returns(labels, seed=42)
    assert r_G > 0.0
    assert r_L < 0.0


def test_bootstrap_returns_tp_fp_split():
    """Gains group = [0.05, 0.03], losses group = [-0.02, -0.04].
    r_G should be close to mean(gains)=0.04, r_L close to mean(losses)=-0.03."""
    labels = _make_labels([0.05, 0.03, -0.02, -0.04])
    r_G, r_L = bootstrap_returns(labels, seed=42, n_boot=5000)
    assert r_G == pytest.approx(0.04, abs=0.005)
    assert r_L == pytest.approx(-0.03, abs=0.005)


def test_bootstrap_returns_empty_gains():
    """All losses => r_G == 0.0, r_L < 0."""
    labels = _make_labels([-0.01, -0.02, -0.03])
    r_G, r_L = bootstrap_returns(labels, seed=42)
    assert r_G == 0.0
    assert r_L < 0.0


def test_bootstrap_returns_empty_losses():
    """All gains => r_G > 0, r_L == 0.0."""
    labels = _make_labels([0.01, 0.02, 0.03])
    r_G, r_L = bootstrap_returns(labels, seed=42)
    assert r_G > 0.0
    assert r_L == 0.0


def test_bootstrap_returns_zero_boundary():
    """ret == 0.0 is treated as a loss (ret <= 0)."""
    labels = _make_labels([0.05, 0.0, -0.03])
    r_G, r_L = bootstrap_returns(labels, seed=42)
    # gains = [0.05], losses = [0.0, -0.03]
    assert r_G > 0.0
    assert r_L <= 0.0


# ---------------------------------------------------------------------------
# adding_zeros_eval
# ---------------------------------------------------------------------------


def _simple_scenario():
    """7 events: 4 label-1 (profitable), 3 label-0 (loss).
    Probabilities designed so filtering at p*=0.6 keeps the 4 true positives
    and drops 2 of 3 false positives => precision rises from 4/7 to 4/5.
    """
    y = np.array([1, 1, 1, 1, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.65, 0.55, 0.45, 0.35])
    p_star = 0.6
    return y, p, p_star


def test_adding_zeros_primary_alone_recall_one():
    y, p, p_star = _simple_scenario()
    result = adding_zeros_eval(y, p, p_star=p_star)
    assert result["primary_alone"]["recall"] == pytest.approx(1.0)


def test_adding_zeros_primary_alone_precision_base_rate():
    y, p, p_star = _simple_scenario()
    result = adding_zeros_eval(y, p, p_star=p_star)
    base = y.mean()
    assert result["primary_alone"]["precision"] == pytest.approx(base)


def test_adding_zeros_precision_lift_positive():
    y, p, p_star = _simple_scenario()
    result = adding_zeros_eval(y, p, p_star=p_star)
    assert result["precision_lift"] > 0.0


def test_adding_zeros_consistency_primary_alone():
    """tp + fp == n_taken for primary_alone."""
    y, p, p_star = _simple_scenario()
    result = adding_zeros_eval(y, p, p_star=p_star)
    cm = result["primary_alone"]
    assert cm["tp"] + cm["fp"] == cm["n_taken"]


def test_adding_zeros_consistency_meta():
    """tp + fp == n_taken for primary_meta."""
    y, p, p_star = _simple_scenario()
    result = adding_zeros_eval(y, p, p_star=p_star)
    cm = result["primary_meta"]
    assert cm["tp"] + cm["fp"] == cm["n_taken"]


def test_adding_zeros_false_positives_avoided():
    y, p, p_star = _simple_scenario()
    result = adding_zeros_eval(y, p, p_star=p_star)
    assert result["false_positives_avoided"] >= 0


def test_adding_zeros_p_star_in_output():
    y, p, p_star = _simple_scenario()
    result = adding_zeros_eval(y, p, p_star=p_star)
    assert result["p_star"] == pytest.approx(p_star)


def test_adding_zeros_all_filtered_out():
    """p_star > max(p) => meta takes nothing; precision is 0.0."""
    y = np.array([1, 0, 1])
    p = np.array([0.3, 0.2, 0.1])
    result = adding_zeros_eval(y, p, p_star=0.9)
    assert result["primary_meta"]["n_taken"] == 0
    assert result["primary_meta"]["precision"] == 0.0


# ---------------------------------------------------------------------------
# nav_sharpe
# ---------------------------------------------------------------------------


def _make_events(rets: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"ret": rets})


def test_nav_sharpe_all_take_vs_subset():
    """Taking all positive returns gives higher NAV than a strict subset."""
    rets = [0.01, 0.02, 0.015, 0.005, 0.008]
    events = _make_events(rets)
    all_mask = np.ones(len(rets), dtype=bool)
    sub_mask = np.array([True, True, False, False, False])
    ns_all = nav_sharpe(events, all_mask)
    ns_sub = nav_sharpe(events, sub_mask)
    assert ns_all["nav"] >= ns_sub["nav"]


def test_nav_sharpe_n_taken_correct():
    rets = [0.01, -0.02, 0.03]
    events = _make_events(rets)
    mask = np.array([True, False, True])
    ns = nav_sharpe(events, mask)
    assert ns["n_taken"] == 2


def test_nav_sharpe_additive_pnl_formula():
    """NAV = sum of realized returns (0 for not-taken) — additive P&L, not a compounded product."""
    rets = [0.10, -0.05, 0.20]
    events = _make_events(rets)
    mask = np.array([True, True, True])
    ns = nav_sharpe(events, mask)
    assert ns["nav"] == pytest.approx(0.10 - 0.05 + 0.20, rel=1e-6)
    # additive (sum), NOT the compounded product 1.10*0.95*1.20-1 = 0.254
    assert ns["nav"] != pytest.approx((1.10 * 0.95 * 1.20) - 1.0, rel=1e-3)


def test_nav_sharpe_not_taken_contribute_zero_to_nav():
    """Untaken events contribute 0 to NAV (not their raw return)."""
    rets = [0.10, -0.50]
    events = _make_events(rets)
    mask_one = np.array([True, False])
    ns_one = nav_sharpe(events, mask_one)
    # product(1+0.10, 1+0) - 1 = 0.10
    assert ns_one["nav"] == pytest.approx(0.10, rel=1e-6)


def test_nav_sharpe_fewer_than_2_taken_sharpe_zero():
    """Less than 2 taken events => sharpe == 0.0."""
    events = _make_events([0.05])
    mask = np.array([True])
    ns = nav_sharpe(events, mask)
    assert ns["sharpe"] == 0.0


def test_nav_sharpe_zero_std_sharpe_zero():
    """Constant returns => std == 0 => sharpe == 0.0."""
    events = _make_events([0.05, 0.05, 0.05])
    mask = np.ones(3, dtype=bool)
    ns = nav_sharpe(events, mask)
    assert ns["sharpe"] == 0.0
