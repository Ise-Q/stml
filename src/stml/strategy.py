"""
strategy.py — Stage 6 — Competition / Strategy Track
=====================================================

Builds a position-sizing strategy on top of the meta-model probabilities.
Operating without the official 20-May constraints document, we use *standard
quant-fund defaults*; when the constraints are released, only the constants
in :class:`StrategyConfig` need to change.

Strategy design (clearly justified — each rule has an economic rationale):

  1. RAW POSITION:
       raw_w[t, i] = sign(primary_signal[t, i]) * conviction(meta_prob[t, i])
     where ``conviction`` maps meta-probability through a thresholded ramp:
       conviction(p) = max(0, (p - threshold) / (1 - threshold))
     i.e. abstain below threshold, ramp to full size as p approaches 1.0.

  2. VOLATILITY TARGETING:
       w[t, i] = raw_w[t, i] * (target_vol / forecast_vol[t, i])
     so each position contributes roughly the target_vol of risk.

  3. POSITION CAPS: |w[t, i]| <= max_per_instrument.

  4. PORTFOLIO RISK CAP: scale all weights down by a single factor so that
     the ex-ante portfolio volatility (using historical covariance) does not
     exceed target_portfolio_vol.

  5. GROSS / NET EXPOSURE CAPS: enforce sum(|w|) <= gross_cap and
     |sum(w)| <= net_cap, scaling down if violated.

Realised returns are computed simply as:
       portfolio_ret[t] = sum_i w[t, i] * forward_1d_log_return[t+1, i]
with no transaction-cost model (placeholder for when constraints arrive).

Metrics reported: CAGR, ann. vol, Sharpe, Sortino, MDD, average holding
period (in trading days), turnover (sum |w[t]-w[t-1]| over time).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    threshold: float = 0.55        # below this prob, abstain
    target_vol: float = 0.10       # 10% per-instrument target vol contribution
    target_portfolio_vol: float = 0.10  # 10% portfolio vol
    max_per_instrument: float = 0.30
    gross_cap: float = 2.0          # |w1|+|w2|+... <= 2.0 (200% gross)
    net_cap: float = 1.0            # |w1+w2+...| <= 1.0
    risk_free_rate: float = 0.0     # daily; raise on rerun
    vol_lookback: int = 21
    vol_ann_factor: float = 252.0
    cov_min_periods: int = 252


# --------------------------------------------------------------------------- #
def _conviction(p: np.ndarray, threshold: float) -> np.ndarray:
    """Ramp conviction from 0 (at threshold) to 1 (at p=1.0). 0 below threshold."""
    above = (p - threshold) / max(1.0 - threshold, 1e-6)
    return np.clip(above, 0.0, 1.0)


def _forecast_vol(ret_panel: pd.DataFrame, window: int = 21,
                   ann_factor: float = 252.0) -> pd.DataFrame:
    """Rolling realised vol per instrument (annualised)."""
    return ret_panel.rolling(window).std() * np.sqrt(ann_factor)


def _scale_weights(w: pd.DataFrame, cov: pd.DataFrame,
                   cfg: StrategyConfig) -> pd.DataFrame:
    """Apply portfolio vol cap + gross/net caps. Returns scaled weights."""
    w = w.copy()
    # Per-instrument cap
    w = w.clip(lower=-cfg.max_per_instrument, upper=cfg.max_per_instrument)
    # Portfolio vol scaling
    w_arr = w.values
    if cov.shape[0] == w.shape[1]:
        sigma_p = np.sqrt(np.einsum("ti,ij,tj->t", w_arr, cov.values, w_arr))
        scale = np.where(sigma_p > cfg.target_portfolio_vol,
                          cfg.target_portfolio_vol / sigma_p, 1.0)
        w = w.multiply(scale, axis=0)
    # Gross / net caps
    gross = w.abs().sum(axis=1)
    g_scale = np.where(gross > cfg.gross_cap, cfg.gross_cap / gross, 1.0)
    w = w.multiply(g_scale, axis=0)
    net = w.sum(axis=1)
    n_scale = np.where(net.abs() > cfg.net_cap, cfg.net_cap / net.abs(), 1.0)
    w = w.multiply(n_scale, axis=0)
    return w


def backtest(
    predictions: pd.DataFrame,
    signals: pd.DataFrame,
    ohlcv: pd.DataFrame,
    cfg: StrategyConfig = StrategyConfig(),
) -> dict:
    """Position-sizing backtest of the meta-model strategy.

    Parameters
    ----------
    predictions : DataFrame with columns ['date', 'instrument', 'prediction']
        The deliverable format. Predictions = 0 mean abstain.
    signals : wide signals frame indexed by date.
    ohlcv : long-format OHLCV.

    Returns
    -------
    dict with:
        weights      : DataFrame (date x instrument)
        portfolio_ret: Series (daily portfolio log returns)
        metrics      : dict {CAGR, ann_vol, Sharpe, Sortino, MDD, avg_holding_days, turnover}
        equity_curve : Series of cumulative return
    """
    if "date" in signals.columns:
        sig_panel = signals.set_index("date")
    else:
        sig_panel = signals
    instruments = list(sig_panel.columns)

    # Wide predictions
    pred = predictions.copy()
    pred["date"] = pd.to_datetime(pred["date"])
    pred_wide = pred.pivot(index="date", columns="instrument", values="prediction")
    # Align columns to signals
    pred_wide = pred_wide.reindex(columns=instruments).fillna(0.0)
    sig_wide = sig_panel.loc[pred_wide.index, instruments].fillna(0)

    # 1. raw conviction
    conv = _conviction(pred_wide.values, cfg.threshold)  # ramped
    raw_w = pd.DataFrame(np.sign(sig_wide.values) * conv,
                         index=pred_wide.index, columns=instruments)

    # 2. Vol-target each instrument
    # Build daily log-return panel for the prediction window (and an extended
    # window so we have lookback vol).
    closes = ohlcv.pivot(index="date", columns="instrument", values="close")
    closes = closes.reindex(columns=instruments)
    ret_panel = np.log(closes).diff()
    vol_panel = _forecast_vol(ret_panel, window=cfg.vol_lookback,
                               ann_factor=cfg.vol_ann_factor)
    vol_pred = vol_panel.reindex(pred_wide.index).reindex(columns=instruments)
    vol_pred = vol_pred.replace(0, np.nan).ffill()
    target_w = raw_w.copy()
    for inst in instruments:
        v = vol_pred[inst]
        target_w[inst] = (raw_w[inst] * cfg.target_vol / v).fillna(0.0)

    # 3. Portfolio scaling using historical covariance
    # Use a 1y window from BEFORE prediction window for covariance.
    cov_window_end = pred_wide.index.min()
    cov_window = ret_panel.loc[:cov_window_end].dropna(how="all").tail(252)
    if cov_window.shape[0] >= cfg.cov_min_periods:
        cov = cov_window.cov() * cfg.vol_ann_factor
        cov = cov.reindex(index=instruments, columns=instruments).fillna(0.0)
    else:
        # Fallback: identity-ish
        cov = pd.DataFrame(np.eye(len(instruments)) * 0.04,
                            index=instruments, columns=instruments)

    weights = _scale_weights(target_w, cov, cfg)

    # 4. Compute realised PnL using NEXT-day returns
    fwd_ret = ret_panel.shift(-1).reindex(weights.index).reindex(columns=instruments)
    pnl = (weights * fwd_ret.fillna(0.0)).sum(axis=1)
    equity = pnl.cumsum()

    # 5. Metrics
    daily_mean = pnl.mean()
    daily_std = pnl.std(ddof=1)
    ann_factor = cfg.vol_ann_factor
    ann_ret = daily_mean * ann_factor
    ann_vol = daily_std * np.sqrt(ann_factor)
    sharpe = (ann_ret - cfg.risk_free_rate) / (ann_vol + 1e-12)
    downside = pnl[pnl < 0]
    downside_std = downside.std(ddof=1) if len(downside) else 1e-12
    sortino = (ann_ret - cfg.risk_free_rate) / (downside_std * np.sqrt(ann_factor) + 1e-12)
    cum = equity.values
    peak = np.maximum.accumulate(cum)
    drawdown = cum - peak  # in log units
    mdd = float(np.exp(drawdown.min()) - 1) if len(drawdown) else 0.0

    # Avg holding period: 1 / mean fraction of position changes
    pos_change = (np.sign(weights).diff().abs() > 0).sum().sum()
    total_pos_days = (np.sign(weights) != 0).sum().sum()
    avg_holding = float(total_pos_days / max(pos_change / 2, 1)) if pos_change > 0 else float("nan")
    turnover = float(weights.diff().abs().sum().sum() / len(weights)) if len(weights) > 0 else 0.0

    n_days = len(pnl)
    years = n_days / ann_factor
    cagr = float(np.exp(pnl.sum()) ** (1 / years) - 1) if years > 0 else 0.0

    return {
        "weights": weights,
        "portfolio_ret": pnl,
        "equity_curve": equity,
        "fwd_ret": fwd_ret,
        "metrics": {
            "CAGR": cagr,
            "ann_return": float(ann_ret),
            "ann_vol": float(ann_vol),
            "Sharpe": float(sharpe),
            "Sortino": float(sortino),
            "MDD": float(mdd),
            "avg_holding_days": avg_holding,
            "turnover_per_day": turnover,
            "n_days": int(n_days),
            "n_positions_avg": float((np.sign(weights) != 0).sum(axis=1).mean()),
        },
    }


# --------------------------------------------------------------------------- #
def blind_baseline_strategy(
    predictions: pd.DataFrame,
    signals: pd.DataFrame,
    ohlcv: pd.DataFrame,
    cfg: StrategyConfig = StrategyConfig(),
) -> dict:
    """Baseline strategy: take EVERY ±1 signal at fixed size (no meta filter).

    For apples-to-apples comparison we still apply vol-targeting + portfolio
    caps; only the meta-probability ramp is replaced by a constant 1.0.
    """
    if "date" in signals.columns:
        sig_panel = signals.set_index("date")
    else:
        sig_panel = signals
    instruments = list(sig_panel.columns)
    pred = predictions.copy()
    pred["date"] = pd.to_datetime(pred["date"])
    # Blind = use the sign of the signal with conviction = 1.0
    blind_pred = pred.copy()
    blind_pred["prediction"] = blind_pred.apply(
        lambda r: 0.0 if r["prediction"] == 0.0 else 1.0, axis=1
    )
    return backtest(blind_pred, signals, ohlcv,
                     cfg=StrategyConfig(threshold=0.0, **{
                         k: v for k, v in cfg.__dict__.items() if k != "threshold"
                     }))


def write_strategy_weights(
    weights: pd.DataFrame, output_path: str | Path,
) -> None:
    """Write strategy_weights.csv in the assignment's required format."""
    rows = []
    for d, row in weights.iterrows():
        for inst, w in row.items():
            rows.append({"date": d, "instrument": inst, "weight": float(w)})
    df = pd.DataFrame(rows)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_csv(output_path, index=False, float_format="%.4f")
