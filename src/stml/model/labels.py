"""
labels.py
=========
Triple-barrier **meta-labeling** of the primary signal.

This is the canonical Lopez de Prado triple-barrier method (``refs/triple_barrier_guide.md``)
specialised for *meta-labeling* on the released futures panel: the primary signal supplies
the **side** (long ``+1`` / short ``-1``), and the label answers a single binary question --
*was taking this bet profitable?* (``1``) *or not* (``0``).

For each non-zero-signal event at ``(date, instrument)`` we walk the instrument's forward
price path over ``h`` trading bars and find the **first** barrier touched:

* upper / profit-taking barrier at ``+pt * sigma``  -> side-adjusted return > 0 -> label ``1``
* lower / stop-loss barrier at ``-sl * sigma``       -> side-adjusted return < 0 -> label ``0``
* vertical / time-out barrier after ``h`` bars       -> label ``sign(ret)`` (or ``0`` if
  ``vertical_zero=True``)

``sigma`` is the per-event volatility target -- in this project the designated label-interface
feature ``f2_vol_20`` (a return-space daily vol), so the barriers are volatility-scaled exactly
as the guide prescribes. ``pt`` / ``sl`` may differ (asymmetric barriers); setting either to
``0`` disables that horizontal barrier.

Two deliberate choices keep this leakage-honest and faithful to the path:

1. **First-touch, not endpoint.** We scan the path in trading-bar order and take the earliest
   of {PT, SL, vertical}. A position that ends up at +2% but dipped through the stop first is
   correctly labelled ``0`` (it would have been stopped out). This is the bug the WIP notebook
   had -- it compared the path max/min independently and ignored ordering.
2. **Trading-bar windows on each instrument's own calendar.** ``h`` counts *bars*, not calendar
   days, so ragged exchange calendars never fabricate or skip a barrier. Events without a full
   ``h``-bar forward window inside the available price series are **dropped** (no peeking at
   prices that do not exist yet). Pass ``price_end`` to truncate the price series -- during
   barrier/model tuning we set ``price_end`` to the validation-block end so label construction
   never touches a single test-period price.

Zero-signal days are label ``0`` by definition and are absent from the feature matrix, so they
never reach this function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_LABEL_COLS = ["date", "instrument", "t1", "ret", "bin", "touch"]


def triple_barrier_labels(
    close_wide: pd.DataFrame,
    events: pd.DataFrame,
    *,
    pt: float,
    sl: float,
    h: int,
    vertical_zero: bool = False,
    price_end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Meta-label every event by its first-touched triple barrier.

    Parameters
    ----------
    close_wide : wide ``date x instrument`` close-price panel (e.g. pivoted clean OHLCV). Each
        column is read on its own non-NaN trading calendar.
    events : DataFrame with columns ``date``, ``instrument``, ``side`` (+1/-1) and ``sigma``
        (the volatility target, e.g. ``f2_vol_20``). One row per event to label.
    pt, sl : profit-taking / stop-loss barrier multiples of ``sigma`` (asymmetric allowed;
        ``0`` disables a barrier).
    h : vertical-barrier horizon in **trading bars**.
    vertical_zero : label time-outs ``0`` regardless of return sign (guide Exercise 3.3).
    price_end : optional inclusive upper bound on the price series used for labeling. Set this
        to the validation-block end during tuning so no test-period price is ever consulted.

    Returns
    -------
    DataFrame with columns ``[date, instrument, t1, ret, bin, touch]`` -- one row per event that
        had a full ``h``-bar forward window and a finite positive ``sigma``. Events without a
        full window (or with bad sigma / unknown date) are silently dropped; the caller can
        compare lengths to see how many were lost.
    """
    if h < 1:
        raise ValueError(f"h must be >= 1 bar, got {h}")
    if price_end is not None:
        price_end = pd.Timestamp(price_end)

    rows: list[tuple] = []
    for inst, ev_g in events.groupby("instrument", sort=False):
        if inst not in close_wide.columns:
            continue
        s = close_wide[inst].dropna()
        if price_end is not None:
            s = s[s.index <= price_end]
        if s.empty:
            continue
        idx = s.index
        vals = s.to_numpy(dtype=float)
        pos = idx.get_indexer(pd.DatetimeIndex(ev_g["date"]))

        sides = ev_g["side"].to_numpy(dtype=float)
        sigmas = ev_g["sigma"].to_numpy(dtype=float)
        dates = ev_g["date"].to_numpy()

        for k in range(len(ev_g)):
            p = int(pos[k])
            if p < 0:  # event date not on this instrument's price calendar
                continue
            end = p + h
            if end >= len(vals):  # not enough forward bars -> drop (no peeking)
                continue
            sigma = sigmas[k]
            if not np.isfinite(sigma) or sigma <= 0:
                continue
            entry = vals[p]
            path_fwd = vals[p + 1 : end + 1]  # h bars strictly after entry
            side = sides[k]
            rel_rets = (path_fwd / entry - 1.0) * side

            up = pt * sigma if pt > 0 else np.inf
            dn = sl * sigma if sl > 0 else np.inf
            pt_hits = np.flatnonzero(rel_rets >= up)
            sl_hits = np.flatnonzero(rel_rets <= -dn)
            t_pt = int(pt_hits[0]) if pt_hits.size else np.iinfo(np.int64).max
            t_sl = int(sl_hits[0]) if sl_hits.size else np.iinfo(np.int64).max
            t_vert = rel_rets.size - 1

            first = min(t_pt, t_sl, t_vert)
            touch = "pt" if first == t_pt else ("sl" if first == t_sl else "vert")
            ret = float(rel_rets[first])
            label = 0 if (touch == "vert" and vertical_zero) else (1 if ret > 0 else 0)

            rows.append((dates[k], inst, idx[p + 1 + first], ret, int(label), touch))

    out = pd.DataFrame(rows, columns=_LABEL_COLS)
    if not out.empty:
        out = out.sort_values(["instrument", "date"]).reset_index(drop=True)
    return out


def sample_uniqueness(labels: pd.DataFrame, close_wide: pd.DataFrame) -> pd.Series:
    """López de Prado average-uniqueness weight per label (Ch. 4), per instrument.

    Overlapping triple-barrier windows make labels non-iid: an event spanning a busy stretch
    shares its outcome with many concurrent events and should count for less. For each instrument
    we count label concurrency ``c_t`` (how many label spans ``[t0, t1]`` cover bar ``t``) and set
    a label's weight to the mean of ``1/c_t`` over its own span. Passed as ``sample_weight`` to the
    tree trainers so concurrent labels don't dominate.

    Returns a Series of weights aligned to ``labels.index``.
    """
    w = pd.Series(1.0, index=labels.index)
    for inst, g in labels.groupby("instrument", sort=False):
        if inst not in close_wide.columns:
            continue
        cal = close_wide[inst].dropna().index
        pos0 = cal.get_indexer(pd.DatetimeIndex(g["date"]))
        pos1 = cal.get_indexer(pd.DatetimeIndex(g["t1"]))
        conc = np.zeros(len(cal))
        for a, b in zip(pos0, pos1):
            if a < 0 or b < 0:
                continue
            conc[a : b + 1] += 1.0
        for k, (a, b) in enumerate(zip(pos0, pos1)):
            if a < 0 or b < 0:
                continue
            seg = conc[a : b + 1]
            seg = seg[seg > 0]
            if seg.size:
                w.loc[g.index[k]] = float(np.mean(1.0 / seg))
    return w


def class_balance(labels: pd.DataFrame) -> dict[str, float]:
    """Summarise the label distribution: counts, positive rate, and minority fraction.

    The minority fraction is the headline number for the barrier-search class-balance floor --
    a labeler that emits 95% of one class is useless no matter its accuracy (guide section 7.1).
    """
    if labels.empty:
        return {"n": 0, "n_pos": 0, "n_neg": 0, "pos_rate": float("nan"),
                "minority_frac": 0.0}
    y = labels["bin"].to_numpy()
    n = int(y.size)
    n_pos = int((y == 1).sum())
    n_neg = n - n_pos
    pos_rate = n_pos / n
    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_rate": pos_rate,
        "minority_frac": float(min(pos_rate, 1.0 - pos_rate)),
    }
