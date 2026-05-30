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
