"""Volatility estimators for the labels and feature pipeline.

Plan §8 Stage 1 deliverable; the σ̂ source for the triple-barrier scale.

Three OHLC range-based estimators and one forward-aware GARCH(1,1) model.

Closed forms (returned as **bar-level variance**, not annualised; the rolling
helpers below produce per-bar realised vol after averaging over a window):

    Parkinson:        σ² = (ln(H/L))² / (4·ln 2)            (Parkinson 1980)
    Garman-Klass:     σ² = 0.5·(ln(H/L))² − (2·ln 2 − 1)·(ln(C/O))²   (GK 1980)
    Rogers-Satchell:  σ² = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)  (R-S 1991)

Garman-Klass is the preferred OHLC σ̂ for the label barrier scale because the
open-close term captures overnight gaps (EIA reports, FOMC, auctions) — the
moments when the barrier scale most needs sharpening. Parkinson under-estimates
on gap days; Rogers-Satchell is drift-independent and corroborates GK.

GARCH(1,1):
    h-day cumulative σ̂ via ``arch.arch_model`` (zero mean, Normal innovations,
    constant variance distribution).
    Causal: every per-bar σ̂_t uses only data ``t' ≤ t``.
    Refits every ``refit`` trading days; σ̂ forward-filled in between.
    Expanding window capped at ``max_window`` bars to keep the fit time bounded
    on long histories.

Citations:
    Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity.
        Journal of Econometrics 31 (3): 307-327.
    Garman, M.B. & Klass, M.J. (1980). On the estimation of security price
        volatilities from historical data. Journal of Business 53 (1): 67-78.
    Parkinson, M. (1980). The extreme value method for estimating the variance of
        the rate of return. Journal of Business 53 (1): 61-65.
    Rogers, L.C.G. & Satchell, S.E. (1991). Estimating variance from high, low and
        closing prices. Annals of Applied Probability 1 (4): 504-512.

Lifted from / informed by:
    metamodel-apb/src/alken_metamodel/volatility.py (closed forms, alken parity).
    src/stml/new_work/triple_barrier.py (Harry's GARCH(1,1) refit cadence pattern).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Per-bar variance closed forms (annualisation NOT applied; the caller decides).
# ---------------------------------------------------------------------------

_LOG2 = float(np.log(2.0))


def _safe_log_ratio(num: pd.Series, denom: pd.Series) -> pd.Series:
    """``ln(num / denom)`` with NaN for any non-finite input — never raises."""
    a = num.astype(float)
    b = denom.astype(float)
    # Both inputs must be strictly positive for log() to be real-valued.
    out = pd.Series(np.nan, index=a.index, dtype=float)
    mask = (a > 0) & (b > 0)
    out.loc[mask] = np.log(a[mask].values) - np.log(b[mask].values)
    return out


def parkinson_variance(ohlc: pd.DataFrame) -> pd.Series:
    """Bar-level Parkinson σ², ``(ln(H/L))² / (4·ln 2)``."""
    ln_hl = _safe_log_ratio(ohlc["high"], ohlc["low"])
    return ln_hl.pow(2) / (4.0 * _LOG2)


def garman_klass_variance(ohlc: pd.DataFrame) -> pd.Series:
    """Bar-level Garman-Klass σ², ``0.5·(ln H/L)² − (2·ln 2 − 1)·(ln C/O)²``."""
    ln_hl = _safe_log_ratio(ohlc["high"], ohlc["low"])
    ln_co = _safe_log_ratio(ohlc["close"], ohlc["open"])
    return 0.5 * ln_hl.pow(2) - (2.0 * _LOG2 - 1.0) * ln_co.pow(2)


def rogers_satchell_variance(ohlc: pd.DataFrame) -> pd.Series:
    """Bar-level Rogers-Satchell σ², ``ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)``."""
    ln_hc = _safe_log_ratio(ohlc["high"], ohlc["close"])
    ln_ho = _safe_log_ratio(ohlc["high"], ohlc["open"])
    ln_lc = _safe_log_ratio(ohlc["low"], ohlc["close"])
    ln_lo = _safe_log_ratio(ohlc["low"], ohlc["open"])
    return ln_hc * ln_ho + ln_lc * ln_lo


# ---------------------------------------------------------------------------
# Rolling realised vol estimators (annualised, daily-frequency input).
# ---------------------------------------------------------------------------


def _rolling_sigma(per_bar_var: pd.Series, window: int, ann: float = 252.0) -> pd.Series:
    """Rolling mean of per-bar σ² → annualised σ. ``min_periods=window`` (warmup NaN)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    rolled_var = per_bar_var.rolling(window=window, min_periods=window).mean()
    # Clamp tiny negative finite values (arithmetic noise) to 0 but preserve NaN
    # so the caller can see the warmup boundary.
    rolled_var = rolled_var.mask(np.isfinite(rolled_var) & (rolled_var < 0.0), other=0.0)
    return (rolled_var * ann).pow(0.5)


def garman_klass(ohlc: pd.DataFrame, window: int = 20, ann: float = 252.0) -> pd.Series:
    """Rolling annualised Garman-Klass σ̂ over ``window`` bars."""
    return _rolling_sigma(garman_klass_variance(ohlc), window, ann)


def parkinson(ohlc: pd.DataFrame, window: int = 20, ann: float = 252.0) -> pd.Series:
    """Rolling annualised Parkinson σ̂ over ``window`` bars."""
    return _rolling_sigma(parkinson_variance(ohlc), window, ann)


def rogers_satchell(ohlc: pd.DataFrame, window: int = 20, ann: float = 252.0) -> pd.Series:
    """Rolling annualised Rogers-Satchell σ̂ over ``window`` bars."""
    return _rolling_sigma(rogers_satchell_variance(ohlc), window, ann)


# ---------------------------------------------------------------------------
# EWMA close-to-close (fallback when GARCH is unavailable).
# ---------------------------------------------------------------------------


def ewma_daily_sigma(close: pd.Series, span: int = 100, min_periods: int = 20) -> pd.Series:
    """Causal EWMA std of log returns — the AFML Ch.3 daily σ̂ surrogate.

    Returned UN-annualised (daily-frequency) so the caller multiplies by
    ``sqrt(h)`` when sizing the triple-barrier widths.
    """
    log_ret = np.log(close / close.shift(1))
    return log_ret.ewm(span=span, min_periods=min_periods, adjust=False).std()


# ---------------------------------------------------------------------------
# GARCH(1,1) — the plan §3.2 default barrier scale.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GarchParams:
    """Per-refit fitted GARCH(1,1) parameters (omega, alpha, beta) + history range."""

    omega: float
    alpha: float
    beta: float
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp


def garch_sigma(
    close: pd.Series,
    refit: int = 21,
    min_obs: int = 500,
    max_window: int = 2000,
    *,
    scale: float = 100.0,
) -> pd.Series:
    """Causal expanding-window GARCH(1,1) **one-step-ahead daily σ̂** per bar.

    Returns a Series of the same length as ``close`` whose value at bar ``t`` is
    the GARCH(1,1)-forecast one-step-ahead daily standard deviation of log
    returns — i.e. σ̂_{t+1}. The triple-barrier labels then size barriers as
    ``pt_mult · σ̂_t · √h`` (plan §3.2 convention).

    Causality contract: σ̂_t uses only ``close.iloc[:t+1]`` (no lookahead).
    The function refits every ``refit`` bars; between refits, σ̂ evolves via the
    GARCH recursion fed with realised log returns up to and including ``t``.

    Parameters
    ----------
    close
        Strictly positive close prices, ``DatetimeIndex``.
    refit
        Refit cadence in bars (21 = monthly business-day; plan §3.2).
    min_obs
        Bars required before the first fit attempt; pre-min_obs σ̂ are NaN.
    max_window
        Cap on the expanding training window to keep MLE tractable.
    scale
        Multiplier applied to the log returns before fitting; ``arch`` is
        numerically more stable when returns are ~unit variance. The output
        is rescaled so the returned σ̂ matches the unscaled log-return units.

    Notes
    -----
    Lifted from Harry's ``src/stml/new_work/triple_barrier.py::sigma_garch`` with
    attribution. We adopt the **daily** convention (one-step-ahead) rather than
    Harry's cumulative h-day form because the plan §3.2 barrier formula
    multiplies by ``√h`` itself. ``garch_h_cumulative_sigma`` below is offered
    as the alternative when an experiment wants the term-structure-aware form.
    """
    return _garch_run(
        close,
        refit=refit,
        min_obs=min_obs,
        max_window=max_window,
        scale=scale,
        emit="daily",
        h=1,
    )


def garch_h_cumulative_sigma(
    close: pd.Series,
    h: int = 10,
    refit: int = 21,
    min_obs: int = 500,
    max_window: int = 2000,
    *,
    scale: float = 100.0,
) -> pd.Series:
    """GARCH(1,1) **cumulative h-day σ̂** per bar — Harry's term-structure variant.

    Identical fitting protocol to :func:`garch_sigma`; only the per-bar emission
    differs: this function returns ``√(Σ_{k=1}^h σ²_{t+k})`` so a barrier sized
    as ``pt_mult · σ̂`` (no extra √h) absorbs the GARCH variance term structure.
    Offered for experimentation; not the default barrier scale.
    """
    if h < 1:
        raise ValueError("h must be >= 1")
    return _garch_run(
        close,
        refit=refit,
        min_obs=min_obs,
        max_window=max_window,
        scale=scale,
        emit="cumulative_h",
        h=h,
    )


def _garch_run(
    close: pd.Series,
    *,
    refit: int,
    min_obs: int,
    max_window: int,
    scale: float,
    emit: str,
    h: int,
) -> pd.Series:
    """Internal GARCH iterator — shared by daily and cumulative-h emitters."""
    if min_obs < 50:
        raise ValueError("min_obs must be >= 50 for a stable GARCH fit")
    if refit < 1:
        raise ValueError("refit must be >= 1")
    if max_window < min_obs:
        raise ValueError("max_window must be >= min_obs")
    if emit not in {"daily", "cumulative_h"}:
        raise ValueError("emit must be 'daily' or 'cumulative_h'")

    from arch import arch_model  # lazy import — ~200ms

    close = close.astype(float)
    if (close <= 0).any():
        raise ValueError("close must be strictly positive")

    log_ret = np.log(close / close.shift(1))
    log_ret = log_ret.fillna(0.0).astype(float)
    n = len(log_ret)
    out = pd.Series(np.nan, index=close.index, dtype=float)

    last_fit_end_pos: int = -1
    params: GarchParams | None = None

    for pos in range(min_obs, n):
        if last_fit_end_pos < 0 or (pos - last_fit_end_pos) >= refit:
            window_start = max(0, pos - max_window + 1)
            train = log_ret.iloc[window_start : pos + 1].values * scale
            try:
                model = arch_model(
                    train,
                    mean="Zero",
                    vol="GARCH",
                    p=1,
                    q=1,
                    dist="normal",
                    rescale=False,
                )
                res = model.fit(disp="off", show_warning=False, options={"maxiter": 200})
                omega = float(res.params["omega"])
                alpha = float(res.params["alpha[1]"])
                beta = float(res.params["beta[1]"])
            except Exception:
                if params is None:
                    continue
                omega, alpha, beta = params.omega, params.alpha, params.beta
            params = GarchParams(
                omega=omega,
                alpha=alpha,
                beta=beta,
                fit_start=close.index[window_start],
                fit_end=close.index[pos],
            )
            last_fit_end_pos = pos

        if params is None:
            continue

        try:
            cond_var_now = _garch_conditional_variance(
                log_ret.iloc[: pos + 1].values * scale,
                omega=params.omega,
                alpha=params.alpha,
                beta=params.beta,
            )[-1]
            if emit == "daily":
                # One-step-ahead variance: σ²_{t+1} = ω + α r_t² + β σ²_t.
                next_var = (
                    params.omega
                    + params.alpha * (log_ret.iloc[pos] * scale) ** 2
                    + params.beta * cond_var_now
                )
                sigma = float(np.sqrt(max(next_var, 0.0))) / scale
            else:
                cum_var = _garch_h_step_cumulative_variance(
                    cond_var_now,
                    omega=params.omega,
                    alpha=params.alpha,
                    beta=params.beta,
                    h=h,
                )
                sigma = float(np.sqrt(max(cum_var, 0.0))) / scale
            out.iloc[pos] = sigma
        except Exception:
            pass

    return out


def _garch_conditional_variance(
    log_ret: np.ndarray, *, omega: float, alpha: float, beta: float
) -> np.ndarray:
    """One-step-ahead conditional variances under fitted GARCH(1,1) params.

    σ²_{t+1} = ω + α·r_t² + β·σ²_t, warm-started by the long-run variance.
    """
    n = len(log_ret)
    var = np.empty(n, dtype=float)
    # Long-run variance under stationarity ω / (1 − α − β).
    persist = alpha + beta
    long_run = omega / max(1.0 - persist, 1e-6) if persist < 1.0 else float(np.var(log_ret))
    var[0] = float(long_run)
    for t in range(1, n):
        var[t] = omega + alpha * (log_ret[t - 1] ** 2) + beta * var[t - 1]
        if not np.isfinite(var[t]):
            var[t] = float(long_run)
    return var


def _garch_h_step_cumulative_variance(
    sigma_sq_t: float, *, omega: float, alpha: float, beta: float, h: int
) -> float:
    """Cumulative h-step-ahead forecast variance under GARCH(1,1).

    Standard closed form (Engle & Patton 2001 §3 / arch docs):

        σ²_{t+1} starts from the GARCH recursion.
        For k ≥ 1: σ²_{t+k} = ω·(1 + (α+β) + ... + (α+β)^{k-1}) + (α+β)^{k-1}·σ²_{t+1}
        Cumulative variance = Σ_{k=1}^{h} σ²_{t+k}.
    """
    persist = alpha + beta
    cum = 0.0
    for k in range(1, h + 1):
        if persist < 1.0:
            geom = (1.0 - persist ** k) / (1.0 - persist)
            sigma_sq_tk = omega * geom + (persist ** (k - 1)) * sigma_sq_t
        else:
            # Non-stationary: variance grows linearly; degenerate case but
            # arises in extreme regimes. Use an undamped recursion.
            sigma_sq_tk = omega * k + sigma_sq_t
        cum += max(sigma_sq_tk, 0.0)
    return cum
