"""Strategy backtest metrics for §6 (RED-first).

The metamodel + sizing emit a position weight on each non-zero-signal trade day; the backtest
forward-holds that weight for the barrier horizon and marks it to the next-day return. Metrics
(CAGR, vol, Sharpe, Sortino, max drawdown) are pinned by known-value tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alken_metamodel.backtest import (
    average_holding_period,
    barrier_backtest,
    build_barrier_position_panel,
    build_position_panel,
    performance_metrics,
    strategy_returns,
)


def test_max_drawdown_known_path():
    # equity 1.0 -> 1.1 -> 0.55 -> 0.605; peak 1.1; trough at 0.55 -> DD = 0.55/1.1 - 1 = -0.5
    r = pd.Series([0.1, -0.5, 0.1])
    m = performance_metrics(r)
    assert m["max_drawdown"] == pytest.approx(-0.5)
    assert m["n"] == 3


def test_sharpe_and_vol_closed_form():
    r = pd.Series([0.02, -0.01, 0.03, 0.00, 0.01])
    m = performance_metrics(r)
    arr = r.to_numpy()
    assert m["ann_vol"] == pytest.approx(arr.std(ddof=1) * np.sqrt(252))
    assert m["sharpe"] == pytest.approx(arr.mean() / arr.std(ddof=1) * np.sqrt(252))
    assert m["total_return"] == pytest.approx(np.prod(1 + arr) - 1)


def test_flat_strategy_is_zero_return_no_drawdown():
    m = performance_metrics(pd.Series([0.0, 0.0, 0.0]))
    assert m["total_return"] == 0.0
    assert m["max_drawdown"] == 0.0
    assert np.isnan(m["sharpe"])  # zero variance -> undefined Sharpe, not a crash


def test_build_position_panel_holds_then_expires():
    cal = pd.bdate_range("2022-01-03", periods=8)
    returns_panel = pd.DataFrame(0.0, index=cal, columns=["cl1s", "ho1s"])
    weights = pd.DataFrame(
        {"date": [cal[0], cal[4]], "instrument": ["cl1s", "cl1s"], "weight": [0.5, -0.3]}
    )
    pos = build_position_panel(weights, returns_panel, max_holding=2)
    assert pos.loc[cal[0], "cl1s"] == 0.5
    assert pos.loc[cal[2], "cl1s"] == 0.5   # still held (forward-filled 2 days)
    assert pos.loc[cal[3], "cl1s"] == 0.0   # horizon expired -> flat
    assert pos.loc[cal[4], "cl1s"] == -0.3  # new signal overwrites
    assert (pos["ho1s"] == 0.0).all()       # never signalled -> always flat


def test_strategy_returns_marks_to_next_day():
    cal = pd.bdate_range("2022-01-03", periods=3)
    positions = pd.DataFrame({"cl1s": [1.0, 0.0, 0.0]}, index=cal)
    returns_panel = pd.DataFrame({"cl1s": [0.0, 0.02, 0.05]}, index=cal)
    daily = strategy_returns(positions, returns_panel)
    assert daily.iloc[0] == pytest.approx(0.02)  # day-0 position earns the day-1 return


# --- S6.7: barrier-exact backtest -------------------------------------------

def _energy_cal():
    return pd.bdate_range("2022-01-03", periods=8)


def test_barrier_panel_holds_until_touch_and_nets_overlap():
    cal = _energy_cal()
    returns_panel = pd.DataFrame(0.0, index=cal, columns=["cl1s", "ho1s"])
    meta = pd.DataFrame(
        {
            "date": [cal[0], cal[2]],
            "instrument": ["cl1s", "cl1s"],
            "weight": [0.5, 0.3],
            "t1": [cal[4], cal[5]],  # exit on the actual first-touch, not a fixed horizon
        }
    )
    pos = build_barrier_position_panel(meta, returns_panel)
    assert pos.loc[cal[0], "cl1s"] == 0.5
    assert pos.loc[cal[1], "cl1s"] == 0.5
    assert pos.loc[cal[2], "cl1s"] == pytest.approx(0.8)  # overlapping labels NET (0.5 + 0.3)
    assert pos.loc[cal[3], "cl1s"] == pytest.approx(0.8)
    assert pos.loc[cal[4], "cl1s"] == pytest.approx(0.3)  # first label's barrier touched -> exits
    assert pos.loc[cal[5], "cl1s"] == 0.0  # both touched -> flat
    assert (pos["ho1s"] == 0.0).all()


def test_average_holding_period_counts_business_days_of_real_trades():
    cal = _energy_cal()
    meta = pd.DataFrame(
        {
            "date": [cal[0], cal[2]],
            "instrument": ["cl1s", "cl1s"],
            "weight": [0.5, 0.0],  # the zero-weight row is not a trade -> excluded
            "t1": [cal[4], cal[5]],
        }
    )
    assert average_holding_period(meta) == pytest.approx(4.0)  # cal[0]->cal[4] = 4 business days


def test_barrier_backtest_net_is_below_gross_when_costs_charged():
    cal = _energy_cal()
    returns_panel = pd.DataFrame(0.01, index=cal, columns=["cl1s"])  # steady +1%/day
    meta = pd.DataFrame(
        {"date": [cal[0]], "instrument": ["cl1s"], "weight": [0.5], "t1": [cal[6]]}
    )
    _net, report = barrier_backtest(meta, returns_panel, impact_bps=20.0)
    assert report["gross_total_return"] > report["net_total_return"]  # costs bite
    assert report["total_cost"] > 0
    assert report["avg_holding_period"] == pytest.approx(6.0)
    assert report["ann_turnover"] > 0
    assert "sharpe" in report and "sortino" in report
