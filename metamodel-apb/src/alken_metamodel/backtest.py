"""Strategy backtest for the §6 bonus track (Carver 2015 vol-targeting context, nlr-cw §7).

The metamodel (act/skip) + sizing (``sizing.py``: fractional Kelly × vol-target, signed by the
primary side) emit a position weight on each non-zero-signal trade day. This module marks that
to market: each weight is forward-held for the barrier horizon (the vertical ``max_holding``),
then the daily strategy return is the position dotted with the next day's instrument returns,
and standard performance metrics are computed.

This is a deliberately simple holding model (hold ``max_holding`` days, latest signal wins) —
documented as such in the write-up; a barrier-exact backtest is left as a refinement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cost_model import transaction_costs, turnover

ANNUALISATION = 252


def build_position_panel(
    weights: pd.DataFrame, returns_panel: pd.DataFrame, *, max_holding: int
) -> pd.DataFrame:
    """Daily (date × instrument) position panel: each signal-day weight held ``max_holding`` days.

    A later signal overwrites the held weight; after the horizon the position goes flat. Aligned
    to the ``returns_panel`` calendar and column order.
    """
    wide = weights.pivot_table(index="date", columns="instrument", values="weight")
    wide.index = pd.DatetimeIndex(wide.index)
    wide = wide.reindex(returns_panel.index)
    held = wide.ffill(limit=max_holding)  # carry each weight forward for the holding horizon
    return held.reindex(columns=returns_panel.columns).fillna(0.0)


def strategy_returns(positions: pd.DataFrame, returns_panel: pd.DataFrame) -> pd.Series:
    """Daily portfolio return: position on day t earns instrument returns on day t+1."""
    forward = returns_panel.shift(-1)
    daily = (positions * forward).sum(axis=1)
    return daily.iloc[:-1]  # the last day has no realised next-day return


def performance_metrics(returns: pd.Series, *, ann: int = ANNUALISATION) -> dict:
    """Total/CAGR/vol/Sharpe/Sortino/max-drawdown for a daily return series.

    Degenerate inputs (zero variance, no losers) yield NaN for the affected ratio rather than a
    division error.
    """
    r = np.asarray(returns.dropna(), dtype=float)
    n = len(r)
    ratio_keys = ("total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown")
    if n == 0:
        return {"n": 0, **{k: float("nan") for k in ratio_keys}}

    equity = np.cumprod(1.0 + r)
    total = float(equity[-1] - 1.0)
    cagr = float(equity[-1] ** (ann / n) - 1.0) if equity[-1] > 0 else float("nan")
    sd = r.std(ddof=1) if n > 1 else 0.0
    ann_vol = float(sd * np.sqrt(ann)) if n > 1 else float("nan")
    sharpe = float(r.mean() / sd * np.sqrt(ann)) if sd > 0 else float("nan")
    downside = r[r < 0]
    dsd = downside.std(ddof=1) if len(downside) > 1 else 0.0
    sortino = float(r.mean() / dsd * np.sqrt(ann)) if dsd > 0 else float("nan")
    peak = np.maximum.accumulate(equity)
    max_dd = float((equity / peak - 1.0).min())

    return {
        "n": n,
        "total_return": total,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
    }


def backtest_strategy(
    weights: pd.DataFrame, returns_panel: pd.DataFrame, *, max_holding: int = 10
) -> tuple[pd.Series, dict]:
    """End-to-end: weights -> daily position panel -> daily returns -> performance metrics."""
    positions = build_position_panel(weights, returns_panel, max_holding=max_holding)
    daily = strategy_returns(positions, returns_panel)
    return daily, performance_metrics(daily)


# --- S6.7: barrier-exact backtest with costs --------------------------------

def build_barrier_position_panel(
    meta: pd.DataFrame, returns_panel: pd.DataFrame
) -> pd.DataFrame:
    """Daily position panel where each label is held until its **actual** first-touch ``t1``.

    ``meta`` carries one row per labelled trade (``date``, ``instrument``, ``weight``, ``t1``).
    A position is active over ``[date, t1)`` — entered at the event date, exited when the triple
    barrier is touched (not at a fixed ``max_holding``). Overlapping labels on the same instrument
    are **netted** (summed), the LdP convention for concurrent bets.
    """
    cal = returns_panel.index
    panel = pd.DataFrame(0.0, index=cal, columns=returns_panel.columns)
    for row in meta.itertuples(index=False):
        if row.instrument not in panel.columns:
            continue
        active = (cal >= row.date) & (cal < row.t1)
        panel.loc[active, row.instrument] += float(row.weight)
    return panel


def average_holding_period(meta: pd.DataFrame) -> float:
    """Mean business-day span ``[date, t1)`` across real trades (non-zero weight); NaN if none."""
    trades = meta[meta["weight"] != 0.0]
    if trades.empty:
        return float("nan")
    held = np.busday_count(
        trades["date"].to_numpy().astype("datetime64[D]"),
        trades["t1"].to_numpy().astype("datetime64[D]"),
    )
    return float(np.mean(held))


def barrier_backtest(
    meta: pd.DataFrame,
    returns_panel: pd.DataFrame,
    *,
    half_spread_bps: float = 2.0,
    impact_bps: float = 10.0,
    impact_exponent: float = 1.0,
) -> tuple[pd.Series, dict]:
    """Barrier-exact, cost-aware backtest -> (net daily returns, report).

    The report folds the standard performance metrics on the **net** series together with the
    brief's missing measures — turnover (annualised) and average holding period — plus the gross
    vs net split so the transaction-cost drag is explicit (§6.7).
    """
    positions = build_barrier_position_panel(meta, returns_panel)
    gross = strategy_returns(positions, returns_panel)
    costs = transaction_costs(
        positions,
        half_spread_bps=half_spread_bps,
        impact_bps=impact_bps,
        impact_exponent=impact_exponent,
    ).reindex(gross.index).fillna(0.0)
    net = gross - costs

    metrics = performance_metrics(net)
    daily_turnover = turnover(positions)
    report = {
        **metrics,
        "gross_total_return": float(np.prod(1.0 + gross.dropna().to_numpy()) - 1.0),
        "net_total_return": metrics["total_return"],
        "total_cost": float(costs.sum()),
        "ann_turnover": float(daily_turnover.mean() * ANNUALISATION),
        "avg_holding_period": average_holding_period(meta),
    }
    return net, report
