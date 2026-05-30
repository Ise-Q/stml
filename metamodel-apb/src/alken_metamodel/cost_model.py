"""Transaction-cost model for the §6 backtest (S6.7; /aqms-python L4, Grinold–Kahn).

Each rebalance pays a **half-spread** on the traded size plus a **market-impact** term. The
classic Grinold–Kahn impact scales with trade size; we use ``impact_bps · |Δw|^impact_exponent``
(linear by default; ``impact_exponent=2`` gives the convex penalty that discourages dumping a
large position in one go). Without per-instrument ADV we charge impact on the position-fraction
traded, a documented simplification. Costs are in return units and summed across instruments.
"""

from __future__ import annotations

import pandas as pd

HALF_SPREAD_BPS = 2.0  # per side, conservative for liquid front-month futures
IMPACT_BPS = 10.0  # Grinold–Kahn impact coefficient (bps per unit turnover)
_BPS = 1e-4


def position_changes(positions: pd.DataFrame) -> pd.DataFrame:
    """Per-day traded size ``Δw`` per instrument; the first row enters from flat."""
    delta = positions.diff()
    delta.iloc[0] = positions.iloc[0]  # entering from flat: the whole opening position is a trade
    return delta


def turnover(positions: pd.DataFrame) -> pd.Series:
    """Daily total turnover = Σ_i |Δw_i| (the absolute position flow across instruments)."""
    return position_changes(positions).abs().sum(axis=1)


def transaction_costs(
    positions: pd.DataFrame,
    *,
    half_spread_bps: float = HALF_SPREAD_BPS,
    impact_bps: float = IMPACT_BPS,
    impact_exponent: float = 1.0,
) -> pd.Series:
    """Per-day transaction cost in return units = Σ_i [spread·|Δw_i| + impact·|Δw_i|^exponent]."""
    dw = position_changes(positions).abs()
    spread = (half_spread_bps * _BPS) * dw
    impact = (impact_bps * _BPS) * dw.pow(impact_exponent)
    return (spread + impact).sum(axis=1)
