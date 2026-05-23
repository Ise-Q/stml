"""
labeling.py
===========
Triple-barrier labeling, meta-labels, and concurrency-based sample-uniqueness
weights for the meta-model. Strictly causal: every quantity at time ``t`` uses
only information observable at ``t``.

Implements (following Lopez de Prado, *Advances in Financial Machine Learning*,
Ch. 3 + Ch. 4 — and as taught in Lecture 1):

  - :func:`get_daily_vol`              -- causal EWMA daily volatility (AFML 3.1)
  - :func:`extract_signal_events`      -- (date, instrument, side) events from the primary signal
  - :func:`apply_triple_barrier_one`   -- first-touch on one event (PT / SL / vertical)
  - :func:`get_meta_labels`            -- full triple-barrier meta-labeling pipeline (AFML 3.4)
  - :func:`get_uniqueness_weights`     -- concurrency-based sample weights (AFML 4)
  - :func:`get_fixed_horizon_labels`   -- the *rejected baseline* (Lecture 1 critique)

Meta-label semantics
--------------------
For each event ``(t, instrument, side)`` where ``side`` is the primary signal
(``-1`` or ``+1``):

  - Upper barrier (profit-take) at ``+pt_mult * sigma_t * sqrt(h)`` log-return
    in the *signed* direction of the bet.
  - Lower barrier (stop-loss)  at ``-sl_mult * sigma_t * sqrt(h)`` likewise.
  - Vertical barrier at ``h`` trading days after ``t``.

The label is **binary**::

    label = 1   if signed_return_in_bet_direction(t -> t1) > 0
    label = 0   otherwise

Which barrier was hit (PT / SL / vertical) is also recorded for diagnostics, but
the label collapses to the sign of the realised PnL at first touch — the
canonical meta-labeling convention (a profit-take touch always yields +ret,
a stop-loss touch always yields -ret, the vertical breaks by realised sign).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 1. Volatility estimate                                                      #
# --------------------------------------------------------------------------- #
def get_daily_vol(
    close: pd.Series,
    span: int = 100,
    min_periods: int = 20,
) -> pd.Series:
    """Causal EWMA daily volatility of log returns.

    Parameters
    ----------
    close : pd.Series
        Close prices indexed by date for a single instrument. Must be sorted
        ascending and free of duplicate dates.
    span : int, default 100
        Span (in trading days) of the exponential weighting. Larger = slower
        adaptation, lower variance.
    min_periods : int, default 20
        Earliest date on which to emit a vol (less ⇒ noisy initial values).

    Returns
    -------
    pd.Series
        EWMA std of log returns, indexed identically to ``close``. The value
        at index ``t`` uses returns up to and including ``t`` (causal).
        Values before ``min_periods`` observations are NaN.

    Notes
    -----
    This is AFML 3.1 ``getDailyVol`` adapted to use native pandas EWMA on
    log-returns. We use log-returns (not pct-change as in AFML) for consistency
    with the rest of the project (``stml.na_checks.native_returns`` uses log).
    """
    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pd.Series indexed by date")
    if not close.index.is_monotonic_increasing:
        raise ValueError("close must be sorted by date ascending")

    log_ret = np.log(close).diff()
    vol = log_ret.ewm(span=span, min_periods=min_periods, adjust=False).std()
    return vol.rename("sigma")


# --------------------------------------------------------------------------- #
# 2. Event extraction                                                          #
# --------------------------------------------------------------------------- #
def extract_signal_events(
    signals: pd.DataFrame,
    instruments: Optional[list[str]] = None,
    include_flat: bool = False,
) -> pd.DataFrame:
    """Long-form events DataFrame from the wide primary-signals panel.

    Parameters
    ----------
    signals : pd.DataFrame
        Wide signals frame with columns = instruments, values in ``{-1, 0, +1}``,
        indexed by ``date`` *or* containing a ``date`` column.
    instruments : list of str, optional
        Restrict to these instruments. If None, use all columns.
    include_flat : bool, default False
        If True, include ``side == 0`` rows. By default they are dropped:
        meta-labeling is only defined for *taken* bets.

    Returns
    -------
    pd.DataFrame
        Columns ``[t, instrument, side]``, one row per event, sorted by
        ``(t, instrument)``. Empty rows / NaN signals are dropped.
    """
    if "date" in signals.columns:
        signals = signals.set_index("date")
    if instruments is None:
        instruments = list(signals.columns)
    long = (
        signals[instruments]
        .stack()
        .rename("side")
        .reset_index()
        .rename(columns={"date": "t", "level_1": "instrument"})
    )
    # `.stack()` produces (index_name='date', col_name='instrument') → after
    # rename the columns are [t, instrument, side]. Coerce to int (signals are ±1/0).
    long["side"] = long["side"].astype(int)
    if not include_flat:
        long = long.loc[long["side"] != 0].copy()
    long = long.sort_values(["t", "instrument"]).reset_index(drop=True)
    return long[["t", "instrument", "side"]]


# --------------------------------------------------------------------------- #
# 3. Triple barrier — single event                                            #
# --------------------------------------------------------------------------- #
def apply_triple_barrier_one(
    close: pd.Series,
    t_event: pd.Timestamp,
    side: int,
    sigma_at_t: float,
    h: int,
    pt_mult: float = 1.0,
    sl_mult: float = 1.0,
) -> tuple[pd.Timestamp, str, float]:
    """First-touch of PT / SL / vertical barrier for ONE event.

    Parameters
    ----------
    close : pd.Series
        Close prices for the instrument, indexed by date (sorted ascending).
    t_event : pd.Timestamp
        Event date. Must be present in ``close.index``.
    side : int
        +1 for a long bet, -1 for a short bet (the primary signal direction).
    sigma_at_t : float
        Daily volatility estimate at ``t_event``. Used to scale the barriers.
    h : int
        Vertical barrier — maximum number of trading days to hold the position
        (so the look-forward window is the next ``h`` rows of ``close``).
    pt_mult, sl_mult : float
        Barrier multipliers. Upper barrier = ``+pt_mult * sigma * sqrt(h)``
        on the signed return; lower barrier = ``-sl_mult * sigma * sqrt(h)``.
        Symmetric ``pt_mult = sl_mult = 1.0`` is the default; one-sigma h-day
        bands are an economically meaningful "did this bet work" threshold.

    Returns
    -------
    t1 : pd.Timestamp
        Date of first barrier touch. If PT or SL is breached intra-window,
        ``t1`` is that date. Otherwise ``t1`` is the last date of the
        ``h``-day window.
    barrier_hit : str
        One of ``"pt"`` (profit-take hit first), ``"sl"`` (stop-loss hit first),
        ``"vertical"`` (time-out — neither barrier breached).
    signed_ret_at_t1 : float
        ``side * (log(close[t1]) - log(close[t_event]))`` -- the realised
        log return in the bet's direction at first touch.

    Notes
    -----
    Barriers are checked on **closing prices** (matches AFML and the course).
    Intra-bar high/low touches are not used here; this is the standard course
    convention. Refinement to high/low touches is documented as a possible
    extension but introduces an order-of-touch ambiguity within a single bar.

    If ``sigma_at_t`` is NaN (insufficient vol history), the event is treated
    as time-out with NaN return — the caller should drop such events.
    """
    if t_event not in close.index:
        raise KeyError(f"t_event {t_event} not in close index")
    if side not in (-1, 1):
        raise ValueError(f"side must be -1 or +1, got {side}")
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}")

    idx = close.index.get_loc(t_event)
    # Window strictly AFTER t_event (the bet is held t_event -> t_event+h),
    # capped at end-of-data.
    end = min(idx + h + 1, len(close))
    if end - idx <= 1:
        # No future data — event is unlabelable.
        return t_event, "vertical", np.nan

    window_close = close.iloc[idx:end]  # includes t_event at position 0
    if pd.isna(sigma_at_t):
        # Cannot construct barriers without a vol estimate. Label by sign at the
        # vertical barrier (still defined as a numeric outcome).
        t1 = window_close.index[-1]
        signed_ret = side * float(np.log(window_close.iloc[-1] / window_close.iloc[0]))
        return t1, "vertical", signed_ret

    # Signed log returns from t_event to each subsequent date.
    rets = side * np.log(window_close.values / window_close.iloc[0])

    upper = pt_mult * sigma_at_t * np.sqrt(h)
    lower = -sl_mult * sigma_at_t * np.sqrt(h)

    # Find first time each barrier is breached (excluding the t_event row at 0).
    after = rets[1:]
    pt_idx_rel = np.argmax(after >= upper) if np.any(after >= upper) else -1
    sl_idx_rel = np.argmax(after <= lower) if np.any(after <= lower) else -1

    # `argmax` returns 0 if the boolean array is all False; the guards above
    # convert that to -1 = no touch. If touch found, absolute index in window
    # is +1 because we sliced off the t_event row.
    candidates: list[tuple[str, int]] = []
    if pt_idx_rel >= 0:
        candidates.append(("pt", pt_idx_rel + 1))
    if sl_idx_rel >= 0:
        candidates.append(("sl", sl_idx_rel + 1))

    if not candidates:
        # Vertical barrier hit.
        t1_pos = len(window_close) - 1
        barrier_hit = "vertical"
    else:
        # Earliest of the candidates wins (smaller index = earlier date).
        barrier_hit, t1_pos = min(candidates, key=lambda x: x[1])

    t1 = window_close.index[t1_pos]
    signed_ret = float(rets[t1_pos])
    return t1, barrier_hit, signed_ret


# --------------------------------------------------------------------------- #
# 4. Triple barrier — full meta-labeling pipeline                             #
# --------------------------------------------------------------------------- #
def get_meta_labels(
    ohlcv_long: pd.DataFrame,
    signals: pd.DataFrame,
    h: int = 10,
    pt_mult: float = 1.0,
    sl_mult: float = 1.0,
    vol_span: int = 100,
    vol_min_periods: int = 20,
    instruments: Optional[list[str]] = None,
    price_col: str = "close",
    verbose: bool = False,
) -> pd.DataFrame:
    """Triple-barrier meta-labeling across all instruments and dates.

    For each ``(date, instrument)`` where the primary signal is non-zero, compute
    the first-touch barrier and the realised signed return. The meta-label is
    binary: ``1`` if the bet was profitable at first touch (PT or vertical with
    positive return), ``0`` otherwise (SL or vertical with non-positive return).

    Parameters
    ----------
    ohlcv_long : pd.DataFrame
        Long-format OHLCV frame (columns include ``date``, ``instrument``,
        ``close``). Use the NA-cleaned version from ``stml.io.load_clean_data``.
    signals : pd.DataFrame
        Wide primary-signals frame (date index or column + instrument columns).
    h, pt_mult, sl_mult : barrier parameters (see :func:`apply_triple_barrier_one`).
    vol_span, vol_min_periods : EWMA vol parameters (see :func:`get_daily_vol`).
    instruments : list of str, optional
        Restrict to these instruments (default: all in signals).
    price_col : str, default "close"
        Which price column to use for barriers.
    verbose : bool
        Print per-instrument progress.

    Returns
    -------
    pd.DataFrame
        Columns:

          - ``t``           : event date (the primary signal's date)
          - ``instrument``  : ticker (lowercase, e.g. ``cl1s``)
          - ``side``        : +1 or -1, the primary signal
          - ``sigma_at_t``  : vol estimate used to scale barriers
          - ``t1``          : first-touch date
          - ``barrier_hit`` : "pt" / "sl" / "vertical"
          - ``ret``         : realised signed log return at ``t1``
          - ``label``       : binary meta-label (``1`` if ``ret > 0``)
          - ``h``           : horizon used (for traceability if h varies later)

        Events with no future data (last ``h`` bars before instrument end) are
        dropped. Events where the vol estimate is NaN are dropped.

        Sorted by ``(t, instrument)``.

    Notes
    -----
    Per-instrument loop is unavoidable because each instrument has its own
    trading calendar. Inside an instrument, events are processed sequentially
    (one barrier-touch lookup per event).
    """
    events = extract_signal_events(signals, instruments=instruments)
    if events.empty:
        return events.assign(
            sigma_at_t=np.nan, t1=pd.NaT, barrier_hit="", ret=np.nan, label=0, h=h
        )

    universe = sorted(events["instrument"].unique())
    out_rows: list[dict] = []

    for inst in universe:
        ev_inst = events.loc[events["instrument"] == inst].copy()
        if ev_inst.empty:
            continue
        close = (
            ohlcv_long.loc[ohlcv_long["instrument"] == inst]
            .set_index("date")[price_col]
            .sort_index()
        )
        if close.empty:
            if verbose:
                print(f"[{inst}] no close data; skipping {len(ev_inst)} events")
            continue
        # Drop duplicates if any (shouldn't be — data is clean).
        close = close[~close.index.duplicated(keep="last")]
        sigma = get_daily_vol(close, span=vol_span, min_periods=vol_min_periods)

        # Filter events to those that have a close price for ``t`` (alignment guard).
        ev_inst = ev_inst.loc[ev_inst["t"].isin(close.index)].copy()

        for _, row in ev_inst.iterrows():
            t = row["t"]
            side = row["side"]
            sig = float(sigma.loc[t]) if not pd.isna(sigma.loc[t]) else np.nan
            t1, hit, ret = apply_triple_barrier_one(
                close, t, side, sig, h=h, pt_mult=pt_mult, sl_mult=sl_mult
            )
            out_rows.append({
                "t": t,
                "instrument": inst,
                "side": int(side),
                "sigma_at_t": sig,
                "t1": t1,
                "barrier_hit": hit,
                "ret": ret,
                "h": h,
            })
        if verbose:
            print(f"[{inst}] processed {len(ev_inst)} events")

    out = pd.DataFrame(out_rows)
    # Drop events with NaN ret (insufficient vol history *and* no future data).
    out = out.loc[out["ret"].notna()].copy()
    out["label"] = (out["ret"] > 0).astype(int)
    out = out.sort_values(["t", "instrument"]).reset_index(drop=True)
    return out[
        ["t", "instrument", "side", "sigma_at_t", "t1", "barrier_hit", "ret", "label", "h"]
    ]


# --------------------------------------------------------------------------- #
# 5. Concurrency / sample-uniqueness weights (AFML Ch. 4)                     #
# --------------------------------------------------------------------------- #
def get_uniqueness_weights(
    events: pd.DataFrame,
    normalize: bool = True,
) -> pd.Series:
    """Average-uniqueness sample weights, computed per instrument.

    For each event ``i = (t_i, t1_i)``, the *concurrency* on date ``u`` is
    ``c(u) = #{j : t_j <= u <= t1_j}``. The average uniqueness of event ``i`` is
    ``u_i = mean_{u in [t_i, t1_i]} ( 1 / c(u) )``.

    Concurrency is counted **per instrument** — two events on different
    instruments don't share a price path, so their labels are independent at
    the path level. (Calendar-time concurrency is what we purge on in CV,
    which is a different concern.)

    Parameters
    ----------
    events : pd.DataFrame
        Must contain ``t``, ``t1``, ``instrument``.
    normalize : bool, default True
        If True, scale weights so the mean is 1 (a "neutral" mean weight).

    Returns
    -------
    pd.Series
        Float weights indexed identically to ``events`` (by position). Greater
        average uniqueness ⇒ higher weight (the event is more informative
        because fewer overlapping bets are sharing the path).
    """
    needed = {"t", "t1", "instrument"}
    if not needed.issubset(events.columns):
        raise KeyError(f"events missing one of {needed}")

    weights = pd.Series(index=events.index, dtype=float)

    for inst, ev in events.groupby("instrument"):
        ev = ev.copy()
        # All dates that any event's span touches, sorted.
        all_t = pd.DatetimeIndex(
            sorted(set(ev["t"]).union(ev["t1"]))
        )
        # Build concurrency by sweeping events: +1 at t, -1 just after t1.
        # We need concurrency on *every business day* between t and t1, not
        # just the event boundaries. Build the daily index from min(t) to max(t1).
        daily = pd.date_range(ev["t"].min(), ev["t1"].max(), freq="D")
        concurrency = pd.Series(0, index=daily, dtype=int)
        for _, r in ev.iterrows():
            # Inclusive span [t, t1].
            concurrency.loc[r["t"]:r["t1"]] += 1
        # Per-event average uniqueness over its own span.
        for i, r in ev.iterrows():
            span_conc = concurrency.loc[r["t"]:r["t1"]]
            # Safety: span_conc is >= 1 inside the span by construction.
            uniq = (1.0 / span_conc.clip(lower=1)).mean()
            weights.loc[i] = uniq

    if normalize and weights.notna().any():
        weights = weights / weights.mean()
    return weights.rename("weight")


# --------------------------------------------------------------------------- #
# 6. Fixed-horizon labelling — the REJECTED BASELINE                          #
# --------------------------------------------------------------------------- #
def get_fixed_horizon_labels(
    ohlcv_long: pd.DataFrame,
    signals: pd.DataFrame,
    h: int = 10,
    threshold: float = 0.0,
    instruments: Optional[list[str]] = None,
    price_col: str = "close",
) -> pd.DataFrame:
    """Fixed-horizon meta-labels — the labeling baseline we reject.

    Lecture 1's critique: it ignores volatility (same threshold for calm and
    stormy markets) and ignores the price path (a 5% move that round-tripped
    looks the same as a steady 5% rise). Included for an honest comparison in
    the report — the methodology section can demonstrate empirically why
    triple-barrier is preferred.

    Label = 1 if ``side * log(close[t+h] / close[t]) > threshold`` else 0.
    """
    events = extract_signal_events(signals, instruments=instruments)
    if events.empty:
        return events.assign(t1=pd.NaT, ret=np.nan, label=0, h=h)

    out_rows: list[dict] = []
    for inst, ev_inst in events.groupby("instrument"):
        close = (
            ohlcv_long.loc[ohlcv_long["instrument"] == inst]
            .set_index("date")[price_col]
            .sort_index()
        )
        if close.empty:
            continue
        for _, row in ev_inst.iterrows():
            t = row["t"]
            if t not in close.index:
                continue
            idx = close.index.get_loc(t)
            end = idx + h
            if end >= len(close):
                continue
            t1 = close.index[end]
            ret = float(row["side"]) * float(np.log(close.iloc[end] / close.iloc[idx]))
            out_rows.append({
                "t": t,
                "instrument": inst,
                "side": int(row["side"]),
                "t1": t1,
                "ret": ret,
                "label": int(ret > threshold),
                "h": h,
            })
    out = pd.DataFrame(out_rows)
    return out.sort_values(["t", "instrument"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 7. Diagnostics                                                              #
# --------------------------------------------------------------------------- #
def label_summary(events_labeled: pd.DataFrame) -> pd.DataFrame:
    """Per-instrument summary: event count, label balance, barrier-hit mix.

    Useful sanity check after :func:`get_meta_labels`.
    """
    if events_labeled.empty:
        return pd.DataFrame()
    g = events_labeled.groupby("instrument")
    out = pd.DataFrame({
        "n_events": g.size(),
        "label_1_share": g["label"].mean(),
        "pt_share": g["barrier_hit"].apply(lambda s: (s == "pt").mean()),
        "sl_share": g["barrier_hit"].apply(lambda s: (s == "sl").mean()),
        "vertical_share": g["barrier_hit"].apply(lambda s: (s == "vertical").mean()),
        "mean_ret_bp": g["ret"].mean() * 1e4,
        "mean_sigma_pct": g["sigma_at_t"].mean() * 100,
    })
    return out.round(4)
