"""Transaction-cost model for §6 (RED-first; /aqms-python Grinold–Kahn).

Cost per rebalance = half-spread on the traded size + a Grinold–Kahn market-impact term, summed
across instruments and charged in return units. Turnover is the |Δposition| flow. The invariants:
entering from flat counts the full first position as a trade, turnover is the absolute change,
and cost is monotone increasing in trade size.
"""

from __future__ import annotations

import pandas as pd
import pytest

from alken_metamodel.cost_model import position_changes, transaction_costs, turnover


def test_position_changes_enters_from_flat():
    pos = pd.DataFrame({"cl1s": [0.5, 0.5, 0.0]})
    dw = position_changes(pos)
    assert dw["cl1s"].tolist() == [0.5, 0.0, -0.5]  # row 0 is the entry from flat


def test_turnover_known_value():
    pos = pd.DataFrame({"cl1s": [0.5, 0.5, 0.0], "ho1s": [0.0, -0.2, -0.2]})
    t = turnover(pos)
    # cl1s |Δ| = [0.5,0,0.5]; ho1s |Δ| = [0,0.2,0]; daily total turnover = [0.5,0.2,0.5]
    assert t.tolist() == [0.5, 0.2, 0.5]


def test_transaction_costs_known_value():
    pos = pd.DataFrame({"x": [1.0, 1.0]})  # one trade of size 1.0, then hold (no trade)
    c = transaction_costs(pos, half_spread_bps=2.0, impact_bps=10.0, impact_exponent=1.0)
    assert c.iloc[0] == pytest.approx(12e-4)  # (2 + 10) bps on |Δ|=1
    assert c.iloc[1] == 0.0


def test_transaction_costs_monotonic_in_size():
    small = transaction_costs(pd.DataFrame({"x": [0.1, 0.0]})).sum()
    big = transaction_costs(pd.DataFrame({"x": [0.9, 0.0]})).sum()
    assert big > small


def test_quadratic_impact_penalises_large_trades_more():
    # impact_exponent=2 makes a single 1.0 trade dearer than ten 0.1 trades of equal turnover
    one_big = transaction_costs(
        pd.DataFrame({"x": [1.0, 0.0]}), half_spread_bps=0.0, impact_bps=10.0, impact_exponent=2.0
    ).sum()
    many_small = transaction_costs(
        pd.DataFrame({"x": [0.1] * 10}), half_spread_bps=0.0, impact_bps=10.0, impact_exponent=2.0
    ).sum()
    assert one_big > many_small
