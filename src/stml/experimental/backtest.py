"""Barrier-exact + cost-aware backtest — Madmoun Optional Session 3 recipe.

Per-instrument position held from ``t_start`` to actual ``t_end`` (first-touch);
weights are stamped on each held day. Portfolio aggregation follows the
lecturer's slide 41:

    R^port_{t+1} = (1 / K_active(t)) · Σ_k w_{t,k} · r_{t+1,k}

with ``K_active(t)`` the number of instruments carrying a non-zero weight at
``t`` (i.e. equal risk-capital allocation across the active universe at each
day). Transaction costs use the same Grinold-Kahn (half-spread + linear
impact) model, applied per asset and aggregated under the same 1/K scheme.

Sortino is the **Sortino-Price 1994 full-T form** (denominator is full sample
length with target 0; NOT std-of-negatives).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stml.experimental.cost_model import (
    HALF_SPREAD_BPS,
    IMPACT_BPS,
    IMPACT_EXPONENT,
    annualised_turnover,
    transaction_costs,
)


@dataclass(frozen=True)
class BacktestReport:
    weights: pd.DataFrame
    gross_returns: pd.Series
    net_returns: pd.Series
    daily_costs: pd.Series
    metrics: dict[str, float]


def build_position_panel(
    events: pd.DataFrame,
    weights_per_event: pd.Series,
    *,
    business_calendar: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Build a (date × instrument) position panel from labelled events.

    Each event contributes its sized weight from ``t_start`` (inclusive) to
    ``t_end`` (exclusive). Overlapping events on the same instrument are
    SUMMED — natural behaviour when the model fires on consecutive signal
    days. The caller may downstream-clip to per-instrument caps if desired.
    """
    events = events.copy()
    events["t_start"] = pd.to_datetime(events["t_start"])
    events["t_end"] = pd.to_datetime(events["t_end"])

    if business_calendar is None:
        business_calendar = pd.bdate_range(
            events["t_start"].min(), events["t_end"].max()
        )

    instruments = sorted(events["instrument"].unique())
    weights_panel = pd.DataFrame(
        0.0, index=business_calendar, columns=instruments
    )
    for i, row in events.iterrows():
        w = float(weights_per_event.iloc[i])
        if w == 0.0:
            continue
        # Held on [t_start, t_end) — exclusive of t_end so we don't double-count
        # at next event's t_start.
        date_range = business_calendar[
            (business_calendar >= row["t_start"]) & (business_calendar < row["t_end"])
        ]
        if len(date_range) == 0:
            continue
        weights_panel.loc[date_range, row["instrument"]] = (
            weights_panel.loc[date_range, row["instrument"]] + w
        )
    return weights_panel


def strategy_returns(
    weights: pd.DataFrame, returns_panel: pd.DataFrame
) -> pd.Series:
    """Cross-sectional risk-budgeted portfolio return -- slide 41.

    ``R^port_{t+1} = (1/K_active(t)) · Σ_k w_{t,k} · r_{t+1,k}`` with
    ``K_active(t) = #{k : w_{t,k} != 0}``. If no asset is active on a day,
    that day's return is 0.
    """
    aligned = returns_panel.reindex(weights.index).reindex(
        columns=weights.columns
    )
    # Shift returns BACKWARD by 1 so r_{t+1} aligns to w_t.
    fwd = aligned.shift(-1).fillna(0.0)
    w = weights.fillna(0.0)
    raw = (w * fwd).sum(axis=1)
    k_active = (w != 0.0).sum(axis=1).clip(lower=1)
    out = (raw / k_active).rename("strategy_ret")
    # Zero out days with no active assets (clip set them to /1, so check w).
    no_active = (weights.fillna(0.0).abs().sum(axis=1) == 0.0)
    out = out.where(~no_active, 0.0)
    # Drop last row (no forward return).
    return out.iloc[:-1]


# ---------------------------------------------------------------------------
# Sortino-Price 1994 full-T form (plan §3.7 + pass-5 fix).
# ---------------------------------------------------------------------------


def sortino_full_t(returns: pd.Series, target: float = 0.0, ann: float = 252.0) -> float:
    """Sortino with the **full-T denominator** (NOT a "std over negatives only")."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    excess = r - target
    downside = np.minimum(excess, 0.0)
    # Sortino-Price: denominator = (1/T) Σ min(r_t − target, 0)^2 over ALL T,
    # NOT just the negatives.
    semivar = (downside ** 2).sum() / len(r)
    sortino_per = excess.mean() / np.sqrt(semivar) if semivar > 0 else float("nan")
    return float(sortino_per * np.sqrt(ann))


def performance_metrics(returns: pd.Series, *, ann: float = 252.0) -> dict[str, float]:
    """Standard per-period and annualised metrics; Sortino is Sortino-Price."""
    r = returns.dropna()
    if len(r) < 2:
        return {k: float("nan") for k in (
            "n", "total_return", "ann_return", "ann_vol", "sharpe", "sortino", "max_dd"
        )}
    n = len(r)
    cum = (1.0 + r).cumprod() - 1.0
    total = float(cum.iloc[-1])
    ann_ret = float(r.mean() * ann)
    ann_vol = float(r.std(ddof=1) * np.sqrt(ann))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    sortino = sortino_full_t(r, ann=ann)
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = (eq / peak - 1.0).min()
    return {
        "n": int(n),
        "total_return": total,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": float(dd),
    }


# ---------------------------------------------------------------------------
# End-to-end backtest.
# ---------------------------------------------------------------------------


def barrier_backtest(
    events: pd.DataFrame,
    weights_per_event: pd.Series,
    returns_panel: pd.DataFrame,
    *,
    half_spread_bps: float = HALF_SPREAD_BPS,
    impact_bps: float = IMPACT_BPS,
    impact_exponent: float = IMPACT_EXPONENT,
) -> BacktestReport:
    """Build positions, compute gross/net returns, performance metrics, costs."""
    weights = build_position_panel(events, weights_per_event)
    gross = strategy_returns(weights, returns_panel)
    costs = transaction_costs(
        weights,
        half_spread_bps=half_spread_bps,
        impact_bps=impact_bps,
        impact_exponent=impact_exponent,
    )
    costs_aligned = costs.reindex(gross.index).fillna(0.0)
    net = (gross - costs_aligned).rename("net_ret")
    metrics = performance_metrics(net)
    metrics["turnover_per_year"] = annualised_turnover(weights)
    metrics["avg_holding_days"] = float(
        (events["t_end"] - events["t_start"]).dt.days.mean()
    )
    metrics["total_cost_bps"] = float(costs.sum() * 10_000)
    metrics["gross_ann_return"] = float(gross.mean() * 252.0)
    metrics["net_ann_return"] = float(net.mean() * 252.0)
    return BacktestReport(
        weights=weights,
        gross_returns=gross,
        net_returns=net,
        daily_costs=costs_aligned,
        metrics=metrics,
    )
