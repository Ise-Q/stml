"""Triple-barrier meta-labelling (AFML Ch. 3–4), side fixed by primary signal.

Concept
-------
Each non-zero primary-signal bar is an *event*.  Three barriers are placed
around the entry price:

  PT (profit-take)  : side-adjusted return >= pt_mult * trgt
  SL (stop-loss)    : side-adjusted return <= -sl_mult * trgt
  Vertical          : h trading days have elapsed without PT/SL

where ``trgt = ewma_daily_vol * sqrt(h)`` is the forward-horizon volatility
estimate.  The first barrier touched determines t1 (exit date) and the
realised side-adjusted return ``ret = side * (P_t1 / P_entry - 1)``.

The *meta-label* ``bin`` is 1 when ``ret > min_ret`` (i.e. the primary signal
was worth following), 0 otherwise.  Side is always fixed by the primary
signal — the meta-model is not asked to predict direction, only whether the
direction predicted by the primary model is profitable to trade.

Fixed documented config (Step 3):
    h=10 days, pt_mult=1.5, sl_mult=1.0, sigma_source="ewma" (span=50).

TODO (Step 4): Optimise (pt_mult, sl_mult, h) via CPCV / PBO.
    Do NOT implement barrier search here — this module is the labeller only.

Sigma source is pluggable via ``sigma_source``:
    "ewma"         : EWMA std of daily log-returns, span=``vol_span`` (default).
    callable       : any ``f(close: pd.Series) -> pd.Series`` returning a
                     causal daily-vol series on the same index.
    # Future hooks (not implemented here):
    # "garch"      : requires the ``arch`` package.
    # "implied"    : requires an external implied-vol feed.

Public API
----------
    from stml.new_work.triple_barrier import label_signals

    labels = label_signals(ohlcv, signals)
    # overrides:
    labels = label_signals(ohlcv, signals, h=5, pt_mult=2.0, sl_mult=0.5)

Output columns
--------------
    date           — signal date (close of this bar = entry price P_entry)
    instrument     — ticker
    side           — primary-signal direction: +1 (long), -1 (short)
    t1             — resolution date (first barrier touch or vertical bar)
    ret            — side * (P_t1 / P_entry - 1), arithmetic side-adj return
    bin            — 1 if ret > min_ret, else 0 (the meta-label)
    trgt           — barrier scale: ewma_vol * sqrt(h) at signal date
    h              — holding horizon used
    pt_mult        — profit-take multiplier used
    sl_mult        — stop-loss multiplier used
    avg_uniqueness — AFML Ch.4 mean(1/c_t) over [date, t1] trading days
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Universe / defaults
# ──────────────────────────────────────────────────────────────────────────────

INSTRUMENTS: tuple[str, ...] = (
    "es1s", "nq1s", "fesx1s",
    "cl1s", "ho1s", "rb1s", "ng1s",
    "gc1s", "si1s", "hg1s", "pl1s",
)

H: int = 10
PT_MULT: float = 1.5
SL_MULT: float = 1.0
VOL_SPAN: int = 50
MIN_RET: float = 0.0

_OUTPUT_COLS: tuple[str, ...] = (
    "date", "instrument", "side", "t1", "ret", "bin",
    "trgt", "h", "pt_mult", "sl_mult", "avg_uniqueness",
)

# ──────────────────────────────────────────────────────────────────────────────
# Sigma (volatility) sources — pluggable
# ──────────────────────────────────────────────────────────────────────────────

SigmaFn = Callable[[pd.Series], pd.Series]


def _ewma_vol(close: pd.Series, span: int) -> pd.Series:
    """Causal EWMA std of daily log-returns on the instrument's own dense index."""
    r = np.log(close).diff()
    return r.ewm(span=span, adjust=False).std()


def _get_sigma_fn(source: str | SigmaFn, span: int) -> SigmaFn:
    """Resolve ``sigma_source`` to a callable ``f(close) -> daily_vol``."""
    if callable(source):
        return source
    if source == "ewma":
        return lambda c: _ewma_vol(c, span)
    # Future hooks — do not implement unless the dependency is installed:
    # if source == "garch":
    #     try:
    #         from arch import arch_model  # noqa: F401
    #         return lambda c: _garch_vol(c, span)
    #     except ImportError:
    #         raise ImportError("Install 'arch' for sigma_source='garch'") from None
    # if source == "implied":
    #     raise NotImplementedError("Implied-vol source requires an external feed.")
    raise ValueError(
        f"Unknown sigma_source {source!r}. "
        "Supported: 'ewma' or any callable f(close) -> pd.Series."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Per-event barrier resolution
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_event(
    prices: np.ndarray,
    side: int,
    pt_thresh: float,
    sl_thresh: float,
) -> tuple[int, float]:
    """Return (offset, ret) for one event.

    Parameters
    ----------
    prices     : 1-D array; prices[0] = entry close, prices[1..h] = hold window.
    side       : +1 (long) or -1 (short).
    pt_thresh  : profit-take level in side-adjusted simple-return space.
    sl_thresh  : stop-loss level (positive number; loss direction).

    Returns
    -------
    offset : index within ``prices`` of the resolution bar (>= 1).
    ret    : side * (prices[offset] / prices[0] - 1).
    """
    entry = prices[0]
    # side-adjusted arithmetic return at each bar (0 at entry by construction)
    r = side * (prices / entry - 1.0)
    for i in range(1, len(r)):
        if r[i] >= pt_thresh or r[i] <= -sl_thresh:
            return i, float(r[i])
    return len(r) - 1, float(r[-1])


# ──────────────────────────────────────────────────────────────────────────────
# Per-instrument labelling
# ──────────────────────────────────────────────────────────────────────────────


def _label_instrument(
    close: pd.Series,
    signal: pd.Series,
    h: int,
    pt_mult: float,
    sl_mult: float,
    sigma_fn: SigmaFn,
    min_ret: float,
) -> pd.DataFrame:
    close = close.sort_index().dropna()
    signal = signal.reindex(close.index)

    daily_vol = sigma_fn(close)
    # trgt = horizon-scaled sigma; used as barrier width multiplier base
    trgt_series = daily_vol * float(np.sqrt(h))

    dates = close.index
    close_arr = close.to_numpy(dtype=np.float64)
    trgt_arr = trgt_series.to_numpy(dtype=np.float64)
    sig_arr = signal.reindex(dates).to_numpy(dtype=np.float64)
    n = len(close_arr)

    records: list[dict] = []
    for i in range(n):
        s = sig_arr[i]
        if not np.isfinite(s) or s == 0:
            continue
        trgt = trgt_arr[i]
        if not np.isfinite(trgt) or trgt <= 0:
            continue
        if i + h >= n:
            continue  # insufficient forward data for full h-bar window
        side = int(np.sign(s))
        window = close_arr[i : i + h + 1]  # prices[0]=entry, prices[1..h]=hold
        offset, ret = _resolve_event(window, side, pt_mult * trgt, sl_mult * trgt)
        records.append(
            {
                "date": dates[i],
                "t1": dates[i + offset],
                "side": side,
                "ret": ret,
                "bin": int(ret > min_ret),
                "trgt": trgt,
            }
        )
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# AFML Ch.4 average uniqueness
# ──────────────────────────────────────────────────────────────────────────────


def _avg_uniqueness(events: pd.DataFrame, bar_index: pd.DatetimeIndex) -> np.ndarray:
    """Compute AFML Ch.4 average label uniqueness for each event.

    Concurrency c_u = number of events whose [date, t1] window covers bar u.
    Uniqueness of event i = mean(1/c_u) over its window.
    Computed on the instrument's native trading-day index (no calendar gaps).

    Parameters
    ----------
    events    : DataFrame with 'date' and 't1' columns (entry and exit dates).
    bar_index : cleaned trading-day DatetimeIndex for the instrument.
    """
    if events.empty:
        return np.array([], dtype=np.float64)

    bar_index = pd.DatetimeIndex(bar_index)
    pos_start = bar_index.get_indexer(events["date"])
    pos_end = bar_index.get_indexer(events["t1"])
    n_bars = len(bar_index)

    # Concurrency via diff/cumsum — O(n_events + n_bars)
    delta = np.zeros(n_bars + 1, dtype=np.float64)
    for ps, pe in zip(pos_start, pos_end):
        if ps >= 0 and pe >= 0:
            delta[ps] += 1.0
            if pe + 1 <= n_bars:
                delta[pe + 1] -= 1.0
    concurrency = np.cumsum(delta)[:n_bars]

    uniqueness = np.empty(len(events), dtype=np.float64)
    for i, (ps, pe) in enumerate(zip(pos_start, pos_end)):
        if ps < 0 or pe < 0 or pe < ps:
            uniqueness[i] = np.nan
        else:
            c = concurrency[ps : pe + 1]
            # clamp to 1 to avoid division by zero if a bar had concurrency < 1
            uniqueness[i] = float(np.mean(1.0 / np.maximum(c, 1.0)))
    return uniqueness


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def label_signals(
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    h: int = H,
    pt_mult: float = PT_MULT,
    sl_mult: float = SL_MULT,
    min_ret: float = MIN_RET,
    sigma_source: str | SigmaFn = "ewma",
    vol_span: int = VOL_SPAN,
    instruments: list[str] | None = None,
) -> pd.DataFrame:
    """Triple-barrier meta-labelling for the panel.

    Parameters
    ----------
    ohlcv        : long-format OHLCV from :func:`stml.io.load_clean_data`;
                   must have columns ``date``, ``instrument``, ``close``.
    signals      : wide-format primary signals from :func:`stml.io.load_clean_data`;
                   must have column ``date`` plus one column per instrument with
                   values in ``{-1, 0, +1}``.
    h            : vertical-barrier horizon (trading days). Default 10.
    pt_mult      : profit-take multiplier on ``trgt``. Default 1.5.
    sl_mult      : stop-loss multiplier on ``trgt``. Default 1.0.
    min_ret      : minimum side-adjusted return for ``bin = 1``. Default 0.
    sigma_source : ``"ewma"`` (default) or any callable ``f(close) -> pd.Series``.
    vol_span     : EWMA span for daily-vol estimation (ignored if callable). 50.
    instruments  : subset of tickers; defaults to all 11 in :data:`INSTRUMENTS`.

    Returns
    -------
    DataFrame with columns:
        date, instrument, side, t1, ret, bin, trgt, h, pt_mult, sl_mult,
        avg_uniqueness.
    """
    sigma_fn = _get_sigma_fn(sigma_source, vol_span)
    if instruments is None:
        instruments = list(INSTRUMENTS)

    sigs = signals.copy()
    sigs["date"] = pd.to_datetime(sigs["date"])
    sigs = sigs.set_index("date").sort_index()

    ohlcv_local = ohlcv.copy()
    ohlcv_local["date"] = pd.to_datetime(ohlcv_local["date"])

    parts: list[pd.DataFrame] = []
    for inst in instruments:
        if inst not in sigs.columns:
            continue
        sub = ohlcv_local.loc[ohlcv_local["instrument"] == inst, ["date", "close"]]
        if sub.empty:
            continue
        close = sub.sort_values("date").set_index("date")["close"]
        signal = sigs[inst].astype("float64")

        evts = _label_instrument(close, signal, h, pt_mult, sl_mult, sigma_fn, min_ret)
        if evts.empty:
            continue

        bar_index = close.dropna().sort_index().index
        evts["avg_uniqueness"] = _avg_uniqueness(evts, bar_index)
        evts["instrument"] = inst
        evts["h"] = h
        evts["pt_mult"] = pt_mult
        evts["sl_mult"] = sl_mult
        parts.append(evts)

    if not parts:
        return pd.DataFrame(columns=list(_OUTPUT_COLS))

    out = pd.concat(parts, ignore_index=True)
    return out[list(_OUTPUT_COLS)]


# ──────────────────────────────────────────────────────────────────────────────
# GARCH(1,1) sigma source
# ──────────────────────────────────────────────────────────────────────────────

def sigma_garch(
    close: pd.Series,
    h: int,
    *,
    refit: int = 21,
    min_obs: int = 500,
    max_window: int = 2000,
) -> pd.Series:
    """Causal expanding-window GARCH(1,1) h-horizon sigma.

    At each bar t, refits GARCH(1,1) (zero mean, Normal) on log-returns from
    ``max(0, t - max_window)`` through ``t-1`` (strictly causal), forecasts the
    h-step ahead variances, and sums across the h steps to obtain the total
    variance of the h-day log-return.  Refits happen every ``refit`` bars;
    sigma is forward-filled between refits.

    Returns are scaled ×100 internally for GARCH numerical stability, then
    unscaled before returning.  Falls back to ``EWMA(span=50) * sqrt(h)`` with
    a warning if ``arch`` is not installed.

    Parameters
    ----------
    close      : sorted, NaN-free price series.
    h          : holding horizon (trading days).
    refit      : refit every this many bars. Default 21 (≈monthly).
    min_obs    : minimum returns before first GARCH fit. Default 500.
    max_window : cap on expanding window (GARCH memory decays exponentially;
                 older data adds negligible information). Default 2000 ≈ 8 yr.

    Returns
    -------
    pd.Series of h-day log-return sigma aligned to ``close.index``.
    Bars before the first successful GARCH fit are filled with EWMA warm-up.
    """
    try:
        from arch import arch_model as _arch_model
        _have_arch = True
    except ImportError:
        import warnings
        warnings.warn(
            "arch not installed — falling back to EWMA(span=50)*sqrt(h). "
            "Install with: uv add arch",
            stacklevel=2,
        )
        _have_arch = False

    if not _have_arch:
        return _ewma_vol(close, 50) * float(np.sqrt(h))

    close = close.sort_index().dropna()
    log_ret = np.log(close).diff()
    n = len(close)

    sigma_arr = np.full(n, np.nan, dtype=np.float64)
    last_sigma: float = float("nan")
    last_refit_pos: int = -refit  # trigger first refit as soon as min_obs reached

    import warnings as _warnings

    for i in range(n):
        if i >= min_obs and (i - last_refit_pos) >= refit:
            start = max(0, i - max_window)
            rets = log_ret.iloc[start:i].dropna().to_numpy(dtype=np.float64)
            if len(rets) >= max(min_obs // 4, 50):
                try:
                    am = _arch_model(
                        rets * 100.0, vol="Garch", p=1, q=1,
                        mean="Zero", dist="Normal",
                    )
                    with _warnings.catch_warnings():
                        _warnings.simplefilter("ignore")
                        res = am.fit(disp="off")
                    fcast = res.forecast(horizon=h, reindex=False)
                    h_var_scaled = float(fcast.variance.iloc[-1].sum())
                    last_sigma = np.sqrt(max(h_var_scaled, 0.0)) / 100.0
                    last_refit_pos = i
                except Exception:
                    pass  # keep last sigma; retry at next refit interval
        sigma_arr[i] = last_sigma

    sigma_h = pd.Series(sigma_arr, index=close.index, dtype=np.float64)
    # Fill pre-warmup NaNs with EWMA (warm-up placeholder, not used in IS)
    ewma_warmup = _ewma_vol(close, 50) * float(np.sqrt(h))
    sigma_h = sigma_h.where(sigma_h.notna(), ewma_warmup)
    return sigma_h


# ──────────────────────────────────────────────────────────────────────────────
# Per-instrument labelling with pre-computed trgt
# ──────────────────────────────────────────────────────────────────────────────

def _label_instrument_with_trgt(
    close: pd.Series,
    signal: pd.Series,
    trgt_series: pd.Series,
    h: int,
    pt_mult: float,
    sl_mult: float,
    min_ret: float,
) -> pd.DataFrame:
    """Like ``_label_instrument`` but accepts a pre-computed h-scaled trgt series.

    ``trgt_series`` must be in the same return units as ``_resolve_event``'s
    output (i.e. log-return std × sqrt(h) for EWMA, or GARCH h-step cumulative
    sigma).  No additional sqrt(h) scaling is applied here.
    """
    close = close.sort_index().dropna()
    signal = signal.reindex(close.index)
    trgt_series = trgt_series.reindex(close.index)

    dates = close.index
    close_arr = close.to_numpy(dtype=np.float64)
    trgt_arr = trgt_series.to_numpy(dtype=np.float64)
    sig_arr = signal.to_numpy(dtype=np.float64)
    n = len(close_arr)

    records: list[dict] = []
    for i in range(n):
        s = sig_arr[i]
        if not np.isfinite(s) or s == 0:
            continue
        trgt = trgt_arr[i]
        if not np.isfinite(trgt) or trgt <= 0:
            continue
        if i + h >= n:
            continue
        side = int(np.sign(s))
        window = close_arr[i : i + h + 1]
        offset, ret = _resolve_event(window, side, pt_mult * trgt, sl_mult * trgt)
        records.append(
            {
                "date": dates[i],
                "t1": dates[i + offset],
                "side": side,
                "ret": ret,
                "bin": int(ret > min_ret),
                "trgt": trgt,
            }
        )
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Fixed-config labeller (GARCH vol)
# ──────────────────────────────────────────────────────────────────────────────

FIXED_H: int = 10
FIXED_PT_MULT: float = 1.5
FIXED_SL_MULT: float = 1.0

_FIXED_OUTPUT_COLS: tuple[str, ...] = (
    "date", "instrument", "side", "t1", "ret", "bin",
    "trgt", "h", "pt_mult", "sl_mult", "sigma_method", "avg_uniqueness",
)


def label_signals_fixed(
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    h: int = FIXED_H,
    pt_mult: float = FIXED_PT_MULT,
    sl_mult: float = FIXED_SL_MULT,
    min_ret: float = MIN_RET,
    garch_refit: int = 21,
    garch_min_obs: int = 500,
    garch_max_window: int = 2000,
    instruments: list[str] | None = None,
) -> pd.DataFrame:
    """Triple-barrier meta-labelling with GARCH(1,1) vol scaling.

    Fixed documented config: h=10, pt_mult=1.5, sl_mult=1.0, sigma=GARCH(1,1).

    The GARCH sigma is computed causally (expanding window capped at
    ``garch_max_window`` bars, refitted every ``garch_refit`` trading days).
    Falls back to EWMA if ``arch`` is not installed.

    Returns
    -------
    DataFrame with columns:
        date, instrument, side, t1, ret, bin, trgt, h, pt_mult, sl_mult,
        sigma_method, avg_uniqueness.
    """
    if instruments is None:
        instruments = list(INSTRUMENTS)

    sigs = signals.copy()
    sigs["date"] = pd.to_datetime(sigs["date"])
    sigs = sigs.set_index("date").sort_index()

    ohlcv_local = ohlcv.copy()
    ohlcv_local["date"] = pd.to_datetime(ohlcv_local["date"])

    try:
        import arch  # noqa: F401
        method = "garch"
    except ImportError:
        method = "ewma_fallback"

    parts: list[pd.DataFrame] = []
    for inst in instruments:
        if inst not in sigs.columns:
            continue
        sub = ohlcv_local.loc[ohlcv_local["instrument"] == inst, ["date", "close"]]
        if sub.empty:
            continue
        close = sub.sort_values("date").set_index("date")["close"].dropna()
        signal = sigs[inst].astype("float64")

        trgt_series = sigma_garch(
            close, h,
            refit=garch_refit,
            min_obs=garch_min_obs,
            max_window=garch_max_window,
        )

        evts = _label_instrument_with_trgt(
            close, signal, trgt_series, h, pt_mult, sl_mult, min_ret
        )
        if evts.empty:
            continue

        bar_index = close.sort_index().index
        evts["avg_uniqueness"] = _avg_uniqueness(evts, bar_index)
        evts["instrument"] = inst
        evts["h"] = h
        evts["pt_mult"] = pt_mult
        evts["sl_mult"] = sl_mult
        evts["sigma_method"] = method
        parts.append(evts)

    if not parts:
        return pd.DataFrame(columns=list(_FIXED_OUTPUT_COLS))

    out = pd.concat(parts, ignore_index=True)
    return out[list(_FIXED_OUTPUT_COLS)]


__all__ = [
    "INSTRUMENTS",
    "H",
    "PT_MULT",
    "SL_MULT",
    "VOL_SPAN",
    "MIN_RET",
    "FIXED_H",
    "FIXED_PT_MULT",
    "FIXED_SL_MULT",
    "sigma_garch",
    "label_signals",
    "label_signals_fixed",
    "_label_instrument_with_trgt",
    "_avg_uniqueness",
    "_resolve_event",
]
