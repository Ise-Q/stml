"""Causal daily-volatility estimators for the triple-barrier width sweep (EX.5).

The shipped pipeline scales barriers off **Garman-Klass** ``f2_vol_20`` (``daily_barrier_sigma``),
but the literature review (``reports/research/nlr-cw-v1.md`` §1) committed to an **EWMA** σ̂_t. The
EX.5 barrier sweep can only resolve that documented inconsistency if GK, EWMA and rolling-std are
compared on equal footing — this module supplies the EWMA/rolling arms and a by-name resolver.

All estimators are **causal**: σ̂_t depends only on returns up to and including bar ``t``, so it is
known at the event-bar close where the trade is entered (López de Prado, AFML 2018, Ch. 3 daily-vol
target). A trailing rolling/EWMA std is **right-edge truncation-invariant** — its value at ``t`` is
identical whether computed on ``close[:t+1]`` or the full series — the same E-class contract the
feature stack enforces (``features.py``). An optional ``shift`` excludes the latest bar for an
extra-cautious σ that ignores bar ``t``'s own return.
<!-- TODO: verify the intended EWMA span(s) against T3.03 L1 source; the lit review says
"exponentially weighted" but does not pin a span. -->
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .volatility import garman_klass


def _close_series(close: pd.Series) -> pd.Series:
    out = close.sort_index()
    out.index = pd.DatetimeIndex(out.index)
    return out.astype(float)


def _log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def ewma_vol(close: pd.Series, span: int, *, shift: int = 0) -> pd.Series:
    """Causal EWMA daily volatility of log-returns (``span`` in trading days).

    ``σ̂_t = ewm(span).std()`` of the log-returns up to bar ``t`` (recursive, ``adjust=False`` —
    RiskMetrics style). Exponential trailing weighting ⇒ right-edge truncation-invariant.
    ``shift`` (default 0) lags σ̂ by that many bars to exclude the latest return(s).
    """
    r = _log_returns(_close_series(close))
    vol = r.ewm(span=span, adjust=False).std()
    return vol.shift(shift) if shift else vol


def rolling_std_vol(close: pd.Series, window: int, *, shift: int = 0) -> pd.Series:
    """Causal rolling-std daily volatility of log-returns (``window`` trading days)."""
    r = _log_returns(_close_series(close))
    vol = r.rolling(window, min_periods=window).std()
    return vol.shift(shift) if shift else vol


def barrier_sigma(ohlc: pd.DataFrame, estimator: str, param: int) -> pd.Series:
    """Daily barrier σ̂_t by estimator name, indexed on the OHLC date calendar.

    ``estimator``:
    - ``"gk"``      — Garman-Klass daily vol over ``param`` bars (the shipped barrier family;
      ``annualize=False`` returns the daily σ directly).
    - ``"ewma"``    — EWMA daily log-return vol, ``span=param``.
    - ``"rolling"`` — rolling-std daily log-return vol, ``window=param``.

    ``ohlc`` is long-format (``date`` column + OHLC). NOTE: the shipped ``daily_barrier_sigma``
    (``f2_vol_20``/√252) is a **realized-vol-20** (close-to-close rolling std), i.e. it equals the
    ``rolling`` arm at window 20 — it is NOT Garman-Klass. The ``gk`` arm here is therefore a
    *genuinely different* OHLC-range estimator the sweep adds on top of the shipped realized vol.
    """
    if estimator == "ewma":
        return ewma_vol(_close_from_long(ohlc), span=param)
    if estimator == "rolling":
        return rolling_std_vol(_close_from_long(ohlc), window=param)
    if estimator == "gk":
        ohlc_idx = ohlc.set_index("date").sort_index()
        ohlc_idx.index = pd.DatetimeIndex(ohlc_idx.index)
        return garman_klass(ohlc_idx, window=param, annualize=False)
    raise ValueError(f"unknown estimator {estimator!r}; expected one of gk|ewma|rolling")


def _close_from_long(ohlc: pd.DataFrame) -> pd.Series:
    return _close_series(ohlc.set_index("date")["close"])
