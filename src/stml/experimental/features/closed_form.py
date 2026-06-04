"""Closed-form OHLCV + signal features.

Families covered (methodology spec):
    F1  counter-trend     — Bollinger %b, RSI, mean-reversion score
    F2  vol / dispersion  — rolling vol, vol ratio, parkinson, garman-klass, vol-of-vol
    F5  signal trajectory — run length, days since flip, signal entropy, flip rate, long bias
    F6  momentum          — 20/60d TS momentum, MA cross, MACD, ADX-like
    F7  microstructure    — volume z, OI level, Amihud illiq, Kyle's lambda, overnight gap
    F8  calendar          — day-of-week sin/cos, month sin/cos
    F10 price action      — high-low range, open-to-open return
    F12 path structure    — variance ratio, efficiency ratio, trend t-val, Hurst

Lifted with attribution from:
* ``src/stml/features.py`` (Sreeram)  — F1, F2, F5, F6, F8 closed forms
* ``src/stml/alternative/features/microstructure_fixed.py`` (Harry) — F7 (with zero-volume mask)
* ``src/stml/alternative/features/signal_trajectory.py`` (Harry) — F5 signal-derived
* ``src/stml/metamodel/features.py`` — F12 path-structure forms

Every feature is E-class (no fit), right-edge truncation invariant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stml.experimental.features.catalog import (
    FeatureContext,
    FeatureSpec,
    expanding_zscore,
    log_returns,
    register,
    rolling_zscore,
)
from stml.experimental.volatility import (
    garman_klass,
    parkinson,
)

# ---------------------------------------------------------------------------
# F1 — Counter-trend / mean-reversion family.
# ---------------------------------------------------------------------------


def _f1_bb_pctb_20(ctx: FeatureContext) -> pd.Series:
    """Bollinger %b: (close - lower) / (upper - lower); 20d window."""
    close = ctx.frame["close"]
    ma = close.rolling(20).mean()
    sd = close.rolling(20).std()
    upper = ma + 2 * sd
    lower = ma - 2 * sd
    return (close - lower) / (upper - lower).replace(0, np.nan)


def _f1_rsi_14(ctx: FeatureContext) -> pd.Series:
    """Wilder's RSI(14) on close-to-close changes."""
    delta = ctx.frame["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _f1_mr_score_20(ctx: FeatureContext) -> pd.Series:
    """Mean-reversion score: z-scored (close - rolling_mean) / rolling_std."""
    close = ctx.frame["close"]
    ma = close.rolling(20).mean()
    sd = close.rolling(20).std()
    return -(close - ma) / sd.replace(0, np.nan)  # negative sign: high = reversion potential


register(FeatureSpec("f1_bb_pctb_20", "F1", _f1_bb_pctb_20, "E", 20, "engineered"))
register(FeatureSpec("f1_rsi_14", "F1", _f1_rsi_14, "E", 14, "shared"))
register(FeatureSpec("f1_mr_score_20", "F1", _f1_mr_score_20, "E", 20, "engineered"))


# ---------------------------------------------------------------------------
# F2 — Volatility / dispersion family.
# ---------------------------------------------------------------------------


def _f2_vol_20(ctx: FeatureContext) -> pd.Series:
    """Annualised rolling std of log returns over 20 bars."""
    return log_returns(ctx.frame["close"]).rolling(20).std() * np.sqrt(252.0)


def _f2_vol_60(ctx: FeatureContext) -> pd.Series:
    return log_returns(ctx.frame["close"]).rolling(60).std() * np.sqrt(252.0)


def _f2_vol_ratio_20_60(ctx: FeatureContext) -> pd.Series:
    """20d vol divided by 60d vol — vol acceleration indicator."""
    r = log_returns(ctx.frame["close"])
    return (r.rolling(20).std() / r.rolling(60).std()).replace([np.inf, -np.inf], np.nan)


def _f2_parkinson_20(ctx: FeatureContext) -> pd.Series:
    return parkinson(ctx.frame[["open", "high", "low", "close"]], window=20)


def _f2_garman_klass_20(ctx: FeatureContext) -> pd.Series:
    return garman_klass(ctx.frame[["open", "high", "low", "close"]], window=20)


def _f2_vol_of_vol_20(ctx: FeatureContext) -> pd.Series:
    """Std of 5d vol over a 20-bar window — vol-of-vol."""
    vol_5d = log_returns(ctx.frame["close"]).rolling(5).std()
    return vol_5d.rolling(20).std()


register(FeatureSpec("f2_vol_20", "F2", _f2_vol_20, "E", 21, "engineered"))
register(FeatureSpec("f2_vol_60", "F2", _f2_vol_60, "E", 61, "engineered"))
register(FeatureSpec("f2_vol_ratio_20_60", "F2", _f2_vol_ratio_20_60, "E", 61, "engineered"))
register(FeatureSpec("f2_parkinson_20", "F2", _f2_parkinson_20, "E", 20, "baseline"))
register(FeatureSpec("f2_garman_klass_20", "F2", _f2_garman_klass_20, "E", 20, "baseline"))
register(FeatureSpec("f2_vol_of_vol_20", "F2", _f2_vol_of_vol_20, "E", 25, "engineered"))


# ---------------------------------------------------------------------------
# F5 — Signal trajectory family (the alternative/features/signal_trajectory.py).
# ---------------------------------------------------------------------------


def _f5_signal_trailing_run_length(ctx: FeatureContext) -> pd.Series:
    """Number of consecutive bars the signal has been at its current value."""
    s = ctx.frame["signal"].fillna(0).astype(int)
    # Reset run length when the value changes.
    new_run = (s != s.shift()).astype(int)
    return new_run.groupby(new_run.cumsum()).cumcount() + 1


def _f5_days_since_flip(ctx: FeatureContext) -> pd.Series:
    """Trading days since the signal last changed value."""
    s = ctx.frame["signal"].fillna(0).astype(int)
    flips = (s.diff() != 0).cumsum()
    return s.groupby(flips).cumcount()


def _f5_signal_entropy_20(ctx: FeatureContext) -> pd.Series:
    """Shannon entropy of the {-1, 0, +1} signal distribution over 20 bars."""
    s = ctx.frame["signal"].fillna(0).astype(int)

    def _ent(window):
        vals, counts = np.unique(window, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log(p + 1e-12))

    return s.rolling(20, min_periods=20).apply(_ent, raw=True)


def _f5_flip_rate_60(ctx: FeatureContext) -> pd.Series:
    """Fraction of bars in the last 60 where the signal flipped."""
    s = ctx.frame["signal"].fillna(0).astype(int)
    flipped = (s.diff() != 0).astype(float)
    return flipped.rolling(60, min_periods=60).mean()


def _f5_long_bias_20(ctx: FeatureContext) -> pd.Series:
    """Mean signal value over 20 bars — long bias (+1 = always long, -1 = always short)."""
    return ctx.frame["signal"].fillna(0).astype(float).rolling(20, min_periods=20).mean()


def _f5_participation_60(ctx: FeatureContext) -> pd.Series:
    """Fraction of last 60 bars with a non-zero signal."""
    s = ctx.frame["signal"].fillna(0).astype(int)
    return (s != 0).astype(float).rolling(60, min_periods=60).mean()


register(FeatureSpec("f5_trailing_run_length", "F5", _f5_signal_trailing_run_length, "E", 1, "alternative"))
register(FeatureSpec("f5_days_since_flip", "F5", _f5_days_since_flip, "E", 1, "alternative"))
register(FeatureSpec("f5_signal_entropy_20", "F5", _f5_signal_entropy_20, "E", 20, "alternative"))
register(FeatureSpec("f5_flip_rate_60", "F5", _f5_flip_rate_60, "E", 60, "alternative"))
register(FeatureSpec("f5_long_bias_20", "F5", _f5_long_bias_20, "E", 20, "shared"))
register(FeatureSpec("f5_participation_60", "F5", _f5_participation_60, "E", 60, "shared"))


# ---------------------------------------------------------------------------
# F6 — Momentum family.
# ---------------------------------------------------------------------------


def _f6_ts_momentum_20(ctx: FeatureContext) -> pd.Series:
    return np.log(ctx.frame["close"] / ctx.frame["close"].shift(20))


def _f6_ts_momentum_60(ctx: FeatureContext) -> pd.Series:
    return np.log(ctx.frame["close"] / ctx.frame["close"].shift(60))


def _f6_ma_cross_20_60(ctx: FeatureContext) -> pd.Series:
    """(MA20 - MA60) / MA60 — MA-crossover indicator."""
    close = ctx.frame["close"]
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    return (ma20 - ma60) / ma60.replace(0, np.nan)


def _f6_macd_12_26(ctx: FeatureContext) -> pd.Series:
    """Classic MACD = EMA12(close) - EMA26(close)."""
    close = ctx.frame["close"]
    ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
    return ema12 - ema26


def _f6_macd_signal_9(ctx: FeatureContext) -> pd.Series:
    """MACD signal line — EMA9 of MACD."""
    return _f6_macd_12_26(ctx).ewm(span=9, min_periods=9, adjust=False).mean()


def _f6_macd_hist(ctx: FeatureContext) -> pd.Series:
    """MACD histogram = MACD - MACD signal."""
    return _f6_macd_12_26(ctx) - _f6_macd_signal_9(ctx)


register(FeatureSpec("f6_ts_momentum_20", "F6", _f6_ts_momentum_20, "E", 20, "engineered"))
register(FeatureSpec("f6_ts_momentum_60", "F6", _f6_ts_momentum_60, "E", 60, "engineered"))
register(FeatureSpec("f6_ma_cross_20_60", "F6", _f6_ma_cross_20_60, "E", 60, "engineered"))
register(FeatureSpec("f6_macd_12_26", "F6", _f6_macd_12_26, "E", 26, "engineered"))
register(FeatureSpec("f6_macd_signal_9", "F6", _f6_macd_signal_9, "E", 35, "engineered"))
register(FeatureSpec("f6_macd_hist", "F6", _f6_macd_hist, "E", 35, "engineered"))


# ---------------------------------------------------------------------------
# F7 — Microstructure family (the microstructure_fixed.py with zero-vol mask).
# ---------------------------------------------------------------------------


def _f7_volume_z_20(ctx: FeatureContext) -> pd.Series:
    """20d rolling z-score of volume — captures volume surprises."""
    v = ctx.frame["volume"].astype(float)
    return rolling_zscore(v, 20)


def _f7_volume_trend_20(ctx: FeatureContext) -> pd.Series:
    """(volume - 20d mean) / 20d mean."""
    v = ctx.frame["volume"].astype(float)
    ma = v.rolling(20).mean()
    return (v - ma) / ma.replace(0, np.nan)


def _f7_oi_level(ctx: FeatureContext) -> pd.Series:
    """Open interest level (raw; very informative on `ng1s` per methodology spec)."""
    return ctx.frame.get("open_interest", pd.Series(dtype=float, index=ctx.frame.index))


def _f7_oi_change_20(ctx: FeatureContext) -> pd.Series:
    """20d log-change in open interest."""
    oi = ctx.frame.get("open_interest")
    if oi is None or oi.isna().all():
        return pd.Series(np.nan, index=ctx.frame.index)
    oi = oi.astype(float)
    return np.log(oi / oi.shift(20))


def _f7_amihud_illiquidity_20(ctx: FeatureContext) -> pd.Series:
    """Amihud (2002) illiquidity proxy = |r| / dollar_volume; rolling mean over 20 bars.

    Zero-volume bars are masked (the fix vs the G4 which propagates Inf).
    """
    r = log_returns(ctx.frame["close"]).abs()
    dv = ctx.frame["close"] * ctx.frame["volume"].astype(float)
    # Mask zero-volume bars.
    mask = ctx.frame["volume"].fillna(0) > 0
    amihud_per_bar = (r / dv.replace(0, np.nan)).where(mask)
    return amihud_per_bar.rolling(20, min_periods=10).mean()


def _f7_kyles_lambda_20(ctx: FeatureContext) -> pd.Series:
    """Kyle's lambda (1985) proxy: mean(|r| / sqrt(volume)) over 20 bars.

    Zero-volume bars masked.
    """
    r = log_returns(ctx.frame["close"]).abs()
    v = ctx.frame["volume"].astype(float)
    mask = v.fillna(0) > 0
    lam_per_bar = (r / np.sqrt(v.replace(0, np.nan))).where(mask)
    return lam_per_bar.rolling(20, min_periods=10).mean()


def _f7_overnight_gap(ctx: FeatureContext) -> pd.Series:
    """log(open / prior close) — overnight gap."""
    return np.log(ctx.frame["open"] / ctx.frame["close"].shift(1))


register(FeatureSpec("f7_volume_z_20", "F7", _f7_volume_z_20, "E", 20, "alternative"))
register(FeatureSpec("f7_volume_trend_20", "F7", _f7_volume_trend_20, "E", 20, "alternative"))
register(FeatureSpec("f7_oi_level", "F7", _f7_oi_level, "E", 1, "shared"))
register(FeatureSpec("f7_oi_change_20", "F7", _f7_oi_change_20, "E", 20, "alternative"))
register(FeatureSpec("f7_amihud_20", "F7", _f7_amihud_illiquidity_20, "E", 20, "alternative"))
register(FeatureSpec("f7_kyles_lambda_20", "F7", _f7_kyles_lambda_20, "E", 20, "alternative"))
register(FeatureSpec("f7_overnight_gap", "F7", _f7_overnight_gap, "E", 1, "alternative"))


# ---------------------------------------------------------------------------
# F8 — Calendar family.
# ---------------------------------------------------------------------------


def _f8_dow_sin(ctx: FeatureContext) -> pd.Series:
    dow = ctx.frame.index.dayofweek.to_numpy(dtype=float)
    return pd.Series(np.sin(2 * np.pi * dow / 5.0), index=ctx.frame.index)


def _f8_dow_cos(ctx: FeatureContext) -> pd.Series:
    dow = ctx.frame.index.dayofweek.to_numpy(dtype=float)
    return pd.Series(np.cos(2 * np.pi * dow / 5.0), index=ctx.frame.index)


def _f8_month_sin(ctx: FeatureContext) -> pd.Series:
    month = ctx.frame.index.month.to_numpy(dtype=float)
    return pd.Series(np.sin(2 * np.pi * month / 12.0), index=ctx.frame.index)


def _f8_month_cos(ctx: FeatureContext) -> pd.Series:
    month = ctx.frame.index.month.to_numpy(dtype=float)
    return pd.Series(np.cos(2 * np.pi * month / 12.0), index=ctx.frame.index)


register(FeatureSpec("f8_dow_sin", "F8", _f8_dow_sin, "E", 0, "engineered"))
register(FeatureSpec("f8_dow_cos", "F8", _f8_dow_cos, "E", 0, "engineered"))
register(FeatureSpec("f8_month_sin", "F8", _f8_month_sin, "E", 0, "engineered"))
register(FeatureSpec("f8_month_cos", "F8", _f8_month_cos, "E", 0, "engineered"))


# ---------------------------------------------------------------------------
# F10 — Price action family.
# ---------------------------------------------------------------------------


def _f10_hl_range(ctx: FeatureContext) -> pd.Series:
    """(High - Low) / close — bar-level intraday range."""
    return (ctx.frame["high"] - ctx.frame["low"]) / ctx.frame["close"]


def _f10_hl_range_mean_20(ctx: FeatureContext) -> pd.Series:
    return _f10_hl_range(ctx).rolling(20).mean()


def _f10_open_close_ret(ctx: FeatureContext) -> pd.Series:
    """log(close / open) — intraday momentum."""
    return np.log(ctx.frame["close"] / ctx.frame["open"])


def _f10_oc_ret_mean_20(ctx: FeatureContext) -> pd.Series:
    return _f10_open_close_ret(ctx).rolling(20).mean()


register(FeatureSpec("f10_hl_range", "F10", _f10_hl_range, "E", 1, "engineered"))
register(FeatureSpec("f10_hl_range_mean_20", "F10", _f10_hl_range_mean_20, "E", 20, "engineered"))
register(FeatureSpec("f10_oc_ret", "F10", _f10_open_close_ret, "E", 1, "engineered"))
register(FeatureSpec("f10_oc_ret_mean_20", "F10", _f10_oc_ret_mean_20, "E", 20, "engineered"))


# ---------------------------------------------------------------------------
# F12 — Path structure (the reference / Sreeram-promoted family).
# ---------------------------------------------------------------------------


def _f12_efficiency_ratio_21(ctx: FeatureContext) -> pd.Series:
    """Kaufman efficiency ratio: |close_t - close_{t-21}| / sum(|diff|)."""
    close = ctx.frame["close"]
    direction = (close - close.shift(21)).abs()
    volatility = close.diff().abs().rolling(21).sum()
    return direction / volatility.replace(0, np.nan)


def _f12_variance_ratio_5_21(ctx: FeatureContext) -> pd.Series:
    """VR = var(r_5) / (5 * var(r_1)) — random walk has VR=1, mean-revert <1, trend >1."""
    r = log_returns(ctx.frame["close"])
    r5 = r.rolling(5).sum()
    return r5.rolling(21).var() / (5.0 * r.rolling(21).var().replace(0, np.nan))


def _f12_trend_tval_21(ctx: FeatureContext) -> pd.Series:
    """OLS t-statistic of close ~ a + b·t over a 21-bar window — trend strength."""
    close = ctx.frame["close"]
    n = 21
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    sx2 = ((x - x_mean) ** 2).sum()

    def _tval(window):
        if np.any(np.isnan(window)):
            return np.nan
        y = window
        y_mean = y.mean()
        slope_num = ((x - x_mean) * (y - y_mean)).sum()
        b = slope_num / sx2
        a = y_mean - b * x_mean
        resid = y - (a + b * x)
        sse = (resid ** 2).sum()
        s2 = sse / (n - 2)
        se = np.sqrt(s2 / sx2) if s2 > 0 else np.nan
        return b / se if se and se > 0 else np.nan

    return close.rolling(n, min_periods=n).apply(_tval, raw=True)


def _f12_autocorr_21(ctx: FeatureContext) -> pd.Series:
    """Lag-1 autocorrelation of log returns over 21 bars."""
    r = log_returns(ctx.frame["close"])
    return r.rolling(21).apply(lambda w: pd.Series(w).autocorr(lag=1), raw=False)


def _f12_hurst_100(ctx: FeatureContext) -> pd.Series:
    """Rolling rescaled-range Hurst exponent over 100 bars.

    Simplified estimator: H = log(R/S) / log(n) on the residuals from the mean.
    """
    r = log_returns(ctx.frame["close"])
    n = 100

    def _hurst(window):
        if np.any(np.isnan(window)):
            return np.nan
        m = window.mean()
        y = (window - m).cumsum()
        r_range = y.max() - y.min()
        s = window.std()
        if s == 0 or r_range == 0:
            return np.nan
        return np.log(r_range / s) / np.log(n)

    return r.rolling(n, min_periods=n).apply(_hurst, raw=True)


register(FeatureSpec("f12_efficiency_ratio_21", "F12", _f12_efficiency_ratio_21, "E", 22, "engineered"))
register(FeatureSpec("f12_variance_ratio_5_21", "F12", _f12_variance_ratio_5_21, "E", 25, "engineered"))
register(FeatureSpec("f12_trend_tval_21", "F12", _f12_trend_tval_21, "E", 21, "engineered"))
register(FeatureSpec("f12_autocorr_21", "F12", _f12_autocorr_21, "E", 21, "engineered"))
register(FeatureSpec("f12_hurst_100", "F12", _f12_hurst_100, "E", 100, "engineered"))
