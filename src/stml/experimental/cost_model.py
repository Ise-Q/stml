"""Grinold-Kahn cost model — plan §3.7 / §8 S6.

Half-spread + market-impact cost on |Δw|. Lifted from
``metamodel-apb/src/alken_metamodel/cost_model.py`` (alken parity).

Defaults:
    half_spread_bps = 2.0   conservative for liquid front-month futures
    impact_bps      = 10.0  Grinold-Kahn impact coefficient on |Δw|
    impact_exponent = 1.0   linear (set to 2 for convex)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HALF_SPREAD_BPS = 2.0
IMPACT_BPS = 10.0
IMPACT_EXPONENT = 1.0


def transaction_costs(
    weights: pd.DataFrame,
    *,
    half_spread_bps: float = HALF_SPREAD_BPS,
    impact_bps: float = IMPACT_BPS,
    impact_exponent: float = IMPACT_EXPONENT,
) -> pd.Series:
    """Per-day total cost as a fraction of NAV under the lecturer's 1/K
    risk-budget aggregation (slide 41), K = universe size.

    Per-asset cost is ``(half_spread + impact·|Δw|^exp) · |Δw|``; aggregated as
    ``(1/K) · Σ_k`` consistent with :func:`strategy_returns`.
    """
    delta = weights.fillna(0.0).diff().abs()
    # Day-0 turnover (open from flat) = |w_0|.
    delta.iloc[0] = weights.iloc[0].abs()
    hs = half_spread_bps / 10_000.0
    im = impact_bps / 10_000.0
    spread_cost = hs * delta.sum(axis=1)
    impact_cost = im * (delta ** impact_exponent).sum(axis=1)
    raw = spread_cost + impact_cost
    k_universe = weights.shape[1]
    return (raw / k_universe).rename("daily_cost")


def annualised_turnover(weights: pd.DataFrame, ann: float = 252.0) -> float:
    """Annualised one-way notional turnover, per-instrument average (1/K basis)."""
    delta = weights.fillna(0.0).diff().abs()
    delta.iloc[0] = weights.iloc[0].abs()
    k_universe = weights.shape[1]
    daily_turnover = delta.sum(axis=1).mean() / k_universe
    return float(daily_turnover * ann)
