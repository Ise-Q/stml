"""Causal (ex-ante) volatility estimators for the strategy-construction package.

Four estimators, all returning an annualised daily σ̂ series:

    yang_zhang   — Yang-Zhang (2000) OHLC range estimator; handles overnight gaps.
                   Default choice: lower variance than Parkinson, accounts for
                   drift via the Rogers-Satchell intraday term.
    ewma_close   — EWMA of squared log close-to-close returns; the AFML Ch.3
                   surrogate. Fast, requires only close prices.
    garch        — GARCH(1,1) one-step-ahead σ̂. Fit refits every `refit` bars
                   on an expanding-capped window.
    gjr          — GJR-GARCH(1,1,1) with asymmetric leverage term I(r<0). Used
                   for equity indices where down-moves widen vol more than up-moves.

Causality contract: σ̂_t uses only data up to and including bar t.
All series are returned with the same DatetimeIndex as the input; warm-up
periods where estimation is impossible are NaN.

References
----------
Yang, D. & Zhang, Q. (2000). Drift-independent volatility estimation based on
  high, low, open, and close prices. Journal of Business 73 (3): 477-491.
Bollerslev, T. (1986). GARCH. Journal of Econometrics 31 (3): 307-327.
Glosten, L.R., Jagannathan, R. & Runkle, D.E. (1993). On the relation between
  the expected value and the volatility of the nominal excess return on stocks.
  Journal of Finance 48 (5): 1779-1801.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_log_ratio(num: pd.Series, denom: pd.Series) -> pd.Series:
    """log(num/denom) with NaN for non-finite or non-positive inputs."""
    a, b = num.astype(float), denom.astype(float)
    out = pd.Series(np.nan, index=a.index, dtype=float)
    mask = (a > 0) & (b > 0)
    out.loc[mask] = np.log(a[mask].values) - np.log(b[mask].values)
    return out


def _annualise(daily_var: pd.Series, trading_days: float = 252.0) -> pd.Series:
    """Annualise a per-bar variance series → σ̂_ann (standard deviation)."""
    clipped = daily_var.clip(lower=0.0)
    return (clipped * trading_days).pow(0.5)


# ---------------------------------------------------------------------------
# 1. Yang-Zhang (2000)
# ---------------------------------------------------------------------------


def yang_zhang(
    ohlc: pd.DataFrame,
    window: int = 20,
    *,
    trading_days: float = 252.0,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling Yang-Zhang annualised σ̂.

    YZ combines three components:
      σ²_overnight = Var[log(O_t / C_{t-1})]   — overnight/gap variance
      σ²_open      = Var[log(C_t / O_t)]        — open-to-close (RS-complementary)
      σ²_RS        = Rogers-Satchell intraday

    Combined as:
      σ²_YZ = σ²_overnight + k · σ²_open + (1 − k) · σ²_RS
    with the bias-minimising k = 0.34 / (1.34 + (N+1)/(N-1)).

    Parameters
    ----------
    ohlc
        DataFrame with columns open, high, low, close and a DatetimeIndex.
    window
        Rolling window in bars (N in the YZ formula).
    trading_days
        Annualisation factor.
    min_periods
        Minimum non-NaN bars; defaults to `window`.
    """
    if min_periods is None:
        min_periods = window
    n = window

    o = ohlc["open"].astype(float)
    h = ohlc["high"].astype(float)
    lo = ohlc["low"].astype(float)
    c = ohlc["close"].astype(float)

    # Overnight: log(O_t / C_{t-1}) — requires a one-bar lag of close.
    ln_oc_prev = _safe_log_ratio(o, c.shift(1))  # log(O / C_prev)
    # Open-to-close: log(C_t / O_t).
    ln_co = _safe_log_ratio(c, o)
    # Rogers-Satchell: drift-independent intraday variance.
    ln_hc = _safe_log_ratio(h, c)
    ln_ho = _safe_log_ratio(h, o)
    ln_lc = _safe_log_ratio(lo, c)
    ln_lo = _safe_log_ratio(lo, o)
    rs_var = ln_hc * ln_ho + ln_lc * ln_lo

    k = 0.34 / (1.34 + (n + 1) / max(n - 1, 1))

    def _roll_var(s: pd.Series) -> pd.Series:
        """Biased-corrected rolling variance (ddof=1)."""
        roll_mean = s.rolling(window=n, min_periods=min_periods).mean()
        roll_sq_mean = s.pow(2).rolling(window=n, min_periods=min_periods).mean()
        count = s.rolling(window=n, min_periods=min_periods).count()
        # ddof=1 correction: multiply biased var by n/(n-1)
        biased_var = roll_sq_mean - roll_mean.pow(2)
        return biased_var * (count / (count - 1).clip(lower=1.0))

    var_overnight = _roll_var(ln_oc_prev)
    var_open = _roll_var(ln_co)
    # RS uses rolling mean of per-bar variance (not variance of RS).
    var_rs = rs_var.rolling(window=n, min_periods=min_periods).mean()

    yz_var = var_overnight + k * var_open + (1.0 - k) * var_rs
    return _annualise(yz_var, trading_days).rename("sigma_yang_zhang")


# ---------------------------------------------------------------------------
# 2. EWMA close-to-close
# ---------------------------------------------------------------------------


def ewma_close(
    close: pd.Series,
    span: int = 60,
    *,
    min_periods: int = 10,
    trading_days: float = 252.0,
    vol_floor: float | None = None,
) -> pd.Series:
    """Causal EWMA annualised σ̂ — exact lecture recurrence (Madmoun OS3, slide 39).

        λ = 2 / (span + 1)
        μ_t = λ·r_t + (1 − λ)·μ_{t-1}
        σ²_t = λ·(r_t − μ_t)² + (1 − λ)·σ²_{t-1}

    Initialised: μ_0 = r_0, σ²_0 = sample-var of the first min(21, n) returns.
    Output: annualised σ̂ = σ_daily × √trading_days, with optional vol floor.
    """
    from stml.new_work import config as _cfg  # lazy to avoid circular import

    if vol_floor is None:
        vol_floor = _cfg.VOL_FLOOR

    close = close.astype(float)
    ret = (close / close.shift(1) - 1.0).dropna()
    n = len(ret)
    if n == 0:
        return pd.Series(np.nan, index=close.index, name="sigma_ewma_close")

    lam = 2.0 / (float(span) + 1.0)
    mu  = np.empty(n, dtype=float)
    var = np.empty(n, dtype=float)

    mu[0] = float(ret.iloc[0])
    seed  = min(21, n)
    var[0] = float(np.var(ret.iloc[:seed].values, ddof=1)) if seed > 1 else 1e-12
    if var[0] <= 0:
        var[0] = 1e-12

    for t in range(1, n):
        rt     = float(ret.iloc[t])
        mu[t]  = lam * rt + (1.0 - lam) * mu[t - 1]
        var[t] = lam * (rt - mu[t]) ** 2 + (1.0 - lam) * var[t - 1]

    daily_sigma = np.sqrt(np.maximum(var, 0.0))
    ann_sigma   = daily_sigma * np.sqrt(trading_days)

    # Apply floor and NaN warm-up period.
    ann_sigma = np.maximum(ann_sigma, vol_floor)
    out = pd.Series(ann_sigma, index=ret.index, name="sigma_ewma_close")
    out.iloc[:min_periods] = np.nan

    return out.reindex(close.index)


# ---------------------------------------------------------------------------
# 3. GARCH(1,1) — shared with experimental/volatility.py but standalone here.
# ---------------------------------------------------------------------------


def garch(
    close: pd.Series,
    *,
    refit: int = 21,
    min_obs: int = 252,
    max_window: int = 2000,
    scale: float = 100.0,
    trading_days: float = 252.0,
) -> pd.Series:
    """Causal GARCH(1,1) one-step-ahead annualised σ̂.

    Fits an expanding-window GARCH(1,1) every `refit` bars; between refits σ̂
    evolves via the GARCH recursion. Returns annualised σ̂ (per-bar daily σ × √252).
    Pre-min_obs entries are NaN.
    """
    from arch import arch_model  # lazy import

    close = close.astype(float)
    if (close.dropna() <= 0).any():
        raise ValueError("close must be strictly positive")

    log_ret = np.log(close / close.shift(1)).fillna(0.0)
    n = len(log_ret)
    daily_sigma = pd.Series(np.nan, index=close.index, dtype=float)

    last_fit_pos: int = -1
    omega = alpha = beta = None

    for pos in range(min_obs, n):
        # Refit?
        if last_fit_pos < 0 or (pos - last_fit_pos) >= refit:
            window_start = max(0, pos - max_window + 1)
            train = log_ret.iloc[window_start: pos + 1].values * scale
            try:
                res = arch_model(train, mean="Zero", vol="GARCH", p=1, q=1,
                                 dist="normal", rescale=False).fit(
                    disp="off", show_warning=False, options={"maxiter": 200})
                omega = float(res.params["omega"])
                alpha = float(res.params["alpha[1]"])
                beta = float(res.params["beta[1]"])
                last_fit_pos = pos
            except Exception:
                if omega is None:
                    continue  # still in warmup

        # One-step-ahead σ̂_{t+1}.
        try:
            r_now = float(log_ret.iloc[pos]) * scale
            persist = alpha + beta
            long_run = omega / max(1.0 - persist, 1e-8)
            # Propagate variance to the current bar via the recursion.
            cond_var = _garch_terminal_variance(
                log_ret.iloc[:pos + 1].values * scale,
                omega=omega, alpha=alpha, beta=beta, long_run=long_run)
            next_var = omega + alpha * r_now ** 2 + beta * cond_var
            daily_sigma.iloc[pos] = float(np.sqrt(max(next_var, 0.0))) / scale
        except Exception:
            pass

    return _annualise(daily_sigma.pow(2), trading_days).rename("sigma_garch")


def _garch_terminal_variance(
    log_ret: np.ndarray, *, omega: float, alpha: float, beta: float, long_run: float
) -> float:
    """Run GARCH recursion on log_ret and return the last conditional variance."""
    var = long_run
    for r in log_ret:
        var = omega + alpha * r ** 2 + beta * var
        if not np.isfinite(var):
            var = long_run
    return var


# ---------------------------------------------------------------------------
# 4. GJR-GARCH(1,1,1) — asymmetric leverage term for equity indices.
# ---------------------------------------------------------------------------


def gjr(
    close: pd.Series,
    *,
    refit: int = 21,
    min_obs: int = 252,
    max_window: int = 2000,
    scale: float = 100.0,
    trading_days: float = 252.0,
) -> pd.Series:
    """Causal GJR-GARCH(1,1,1) one-step-ahead annualised σ̂.

    The GJR model adds an asymmetric term:
        σ²_t = ω + (α + γ I(r_{t-1} < 0)) r²_{t-1} + β σ²_{t-1}

    This captures the leverage effect where negative returns produce larger
    vol increases than positive ones (Glosten et al. 1993).
    """
    from arch import arch_model  # lazy import

    close = close.astype(float)
    if (close.dropna() <= 0).any():
        raise ValueError("close must be strictly positive")

    log_ret = np.log(close / close.shift(1)).fillna(0.0)
    n = len(log_ret)
    daily_sigma = pd.Series(np.nan, index=close.index, dtype=float)

    last_fit_pos: int = -1
    params: dict | None = None

    for pos in range(min_obs, n):
        if last_fit_pos < 0 or (pos - last_fit_pos) >= refit:
            window_start = max(0, pos - max_window + 1)
            train = log_ret.iloc[window_start: pos + 1].values * scale
            try:
                res = arch_model(train, mean="Zero", vol="GARCH", p=1, o=1, q=1,
                                 dist="normal", rescale=False).fit(
                    disp="off", show_warning=False, options={"maxiter": 200})
                params = {
                    "omega": float(res.params["omega"]),
                    "alpha": float(res.params["alpha[1]"]),
                    "gamma": float(res.params["gamma[1]"]),
                    "beta": float(res.params["beta[1]"]),
                }
                last_fit_pos = pos
            except Exception:
                if params is None:
                    continue

        if params is None:
            continue

        try:
            r_now = float(log_ret.iloc[pos]) * scale
            r_prev = float(log_ret.iloc[pos - 1]) * scale if pos > 0 else 0.0
            omega, alpha, gamma, beta = (
                params["omega"], params["alpha"], params["gamma"], params["beta"])
            long_run = omega / max(1.0 - alpha - 0.5 * gamma - beta, 1e-8)
            cond_var = _gjr_terminal_variance(
                log_ret.iloc[:pos + 1].values * scale,
                omega=omega, alpha=alpha, gamma=gamma, beta=beta, long_run=long_run)
            indicator = 1.0 if r_prev < 0 else 0.0
            next_var = omega + (alpha + gamma * indicator) * r_now ** 2 + beta * cond_var
            daily_sigma.iloc[pos] = float(np.sqrt(max(next_var, 0.0))) / scale
        except Exception:
            pass

    return _annualise(daily_sigma.pow(2), trading_days).rename("sigma_gjr")


def _gjr_terminal_variance(
    log_ret: np.ndarray,
    *,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    long_run: float,
) -> float:
    """Run GJR recursion and return the last conditional variance."""
    var = long_run
    for i, r in enumerate(log_ret):
        r_prev = log_ret[i - 1] if i > 0 else 0.0
        ind = 1.0 if r_prev < 0 else 0.0
        var = omega + (alpha + gamma * ind) * r_prev ** 2 + beta * var
        if not np.isfinite(var):
            var = long_run
    return var


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def compute_vol(
    ohlc: pd.DataFrame,
    method: str = "yang_zhang",
    *,
    window: int = 20,
    span: int = 60,
    trading_days: float = 252.0,
) -> pd.Series:
    """Dispatch to the chosen estimator.

    Parameters
    ----------
    ohlc
        OHLC DataFrame (columns: open, high, low, close) with DatetimeIndex.
    method
        One of: yang_zhang, ewma_close, garch, gjr.
    window
        Rolling window for yang_zhang (bars).
    span
        EWMA span for ewma_close (bars).
    trading_days
        Annualisation factor.
    """
    if method == "yang_zhang":
        return yang_zhang(ohlc, window=window, trading_days=trading_days)
    if method == "ewma_close":
        return ewma_close(ohlc["close"], span=span, trading_days=trading_days)
    if method == "garch":
        return garch(ohlc["close"], trading_days=trading_days)
    if method == "gjr":
        return gjr(ohlc["close"], trading_days=trading_days)
    raise ValueError(
        f"Unknown vol method {method!r}. Choose from: yang_zhang, ewma_close, garch, gjr"
    )
