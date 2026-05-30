"""Triple-barrier meta-labelling (López de Prado 2018, Ch.3-4).

Adapted from ``refs/triple_barrier_guide.md`` §4/§6 to the metamodel's setting:
the *events* are the days where the provided primary signal is non-zero, the
*side* is the sign of that signal, and the labels are **meta-labels** in {0, 1}
("act" / "skip"). Each event gets symmetric (configurable) volatility-adaptive
barriers ±k·σ̂ₜ and a vertical barrier ``max_holding`` bars out; the label is the
sign of the side-adjusted P&L at the first barrier touched, and we record ``t1``
(first-touch time) for purging and the average-uniqueness weight for training.

Why these conventions (Section 2 justification):
- **Barriers scale with σ̂ₜ** (not a fixed %): fixed thresholds ignore the
  heteroskedasticity of returns and make the label distribution pro-cyclical
  (guide §1; LdP 2018 Ch.3).
- **Timeout labelled by the sign of expiry P&L** (>0 -> 1 else 0): the guide's
  default for meta-labelling (``get_bins`` §4.7); whether a hard 0 works better is
  empirical and left to barrier tuning (§7).
- **Average-uniqueness weights** down-weight labels whose horizons overlap many
  others, because triple-barrier labels are not iid (LdP 2018 Ch.4); used as
  ``sample_weight`` in model training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_vertical_barrier(close: pd.Series, t_events: pd.Index, max_holding: int) -> pd.Series:
    """Timestamp ``max_holding`` bars after each event (the time-out barrier).

    Indexed by event time; value is the expiry timestamp (clipped to the last bar
    for events too close to the end of the series).
    """
    pos = close.index.get_indexer(t_events)
    vb = np.minimum(pos + max_holding, len(close.index) - 1)
    return pd.Series(close.index[vb], index=t_events)


def apply_pt_sl_on_t1(
    close: pd.Series, events: pd.DataFrame, pt_sl: tuple[float, float]
) -> pd.DataFrame:
    """First time the profit-take / stop-loss barrier is touched, per event.

    ``events`` has columns ``t1`` (vertical barrier), ``trgt`` (σ̂ width), ``side``.
    Returns a frame with columns ``t1`` (vertical, copied), ``sl``, ``pt`` (touch
    times, NaT if never touched). A barrier factor of 0 disables that barrier.
    """
    out = events[["t1"]].copy(deep=True)
    pt = pt_sl[0] * events["trgt"] if pt_sl[0] > 0 else pd.Series(np.nan, index=events.index)
    sl = -pt_sl[1] * events["trgt"] if pt_sl[1] > 0 else pd.Series(np.nan, index=events.index)

    for loc, t1 in events["t1"].fillna(close.index[-1]).items():
        path = close.loc[loc:t1]
        path_ret = (path / close.loc[loc] - 1.0) * events.at[loc, "side"]  # side-adjusted
        out.loc[loc, "sl"] = path_ret[path_ret < sl[loc]].index.min()
        out.loc[loc, "pt"] = path_ret[path_ret > pt[loc]].index.min()
    return out


def get_events(
    close: pd.Series,
    t_events: pd.Index,
    pt_sl: tuple[float, float],
    trgt: pd.Series,
    side: pd.Series,
    t1: pd.Series,
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """Assemble events and resolve the first barrier touch (meta-label path).

    ``t1`` after this call holds the *first-touch* time (earliest of pt/sl/vertical).
    """
    trgt = trgt.loc[trgt.index.intersection(t_events)]
    trgt = trgt[trgt > min_ret]
    side_ = side.loc[trgt.index]

    events = pd.concat({"t1": t1, "trgt": trgt, "side": side_}, axis=1).dropna(subset=["trgt"])
    touches = apply_pt_sl_on_t1(close, events, pt_sl)
    events["t1"] = touches.dropna(how="all").min(axis=1)  # earliest of {pt, sl, vertical}
    return events


def get_bins(events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Realised P&L and meta-label in {0, 1} (1 = side-adjusted return at t1 > 0)."""
    events_ = events.dropna(subset=["t1"])
    px_idx = events_.index.union(pd.DatetimeIndex(events_["t1"].values)).drop_duplicates()
    px = close.reindex(px_idx, method="bfill")

    out = pd.DataFrame(index=events_.index)
    out["ret"] = px.loc[events_["t1"].to_numpy()].to_numpy() / px.loc[events_.index].to_numpy() - 1.0
    out["ret"] = out["ret"] * events_["side"]
    out["bin"] = (out["ret"] > 0).astype(float)
    return out


def get_num_co_events(close_index: pd.Index, t1: pd.Series) -> pd.Series:
    """Number of concurrent label spans covering each bar (LdP 2018 §4.1)."""
    t1 = t1.fillna(close_index[-1])
    count = pd.Series(0.0, index=close_index)
    for t_in, t_out in t1.items():
        count.loc[t_in:t_out] += 1.0
    return count


def average_uniqueness(t1: pd.Series, num_co_events: pd.Series) -> pd.Series:
    """Average uniqueness of each label = mean over its span of 1/concurrency (LdP §4.2).

    1.0 when the label's span overlaps no other; < 1.0 under overlap.
    """
    t1 = t1.fillna(num_co_events.index[-1])
    out = pd.Series(index=t1.index, dtype=float)
    for t_in, t_out in t1.items():
        out.loc[t_in] = (1.0 / num_co_events.loc[t_in:t_out]).mean()
    return out


def triple_barrier_labels(
    close: pd.Series,
    signal: pd.Series,
    target: pd.Series,
    *,
    pt_sl: tuple[float, float] = (1.0, 1.0),
    max_holding: int = 10,
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """Meta-labels for the days where ``signal != 0``.

    Parameters
    ----------
    close : daily price Series indexed by date.
    signal : primary signal Series in {-1, 0, +1}, aligned to ``close``.
    target : σ̂ₜ Series (barrier unit width, e.g. Garman-Klass vol), aligned to ``close``.
    pt_sl : (profit-take, stop-loss) multiples of ``target`` (symmetric by default).
    max_holding : vertical-barrier horizon in bars.
    min_ret : skip events whose σ̂ target is <= this.

    Returns
    -------
    DataFrame indexed by event date with columns ``side``, ``t1`` (first-touch),
    ``ret`` (side-adjusted P&L), ``bin`` (meta-label {0,1}), ``weight`` (avg uniqueness).
    """
    t_events = signal.index[signal != 0]
    side = np.sign(signal.loc[t_events]).astype(float)
    vertical = add_vertical_barrier(close, t_events, max_holding)

    events = get_events(close, t_events, pt_sl, target, side, vertical, min_ret)
    bins = get_bins(events, close)

    num_co = get_num_co_events(close.index, events["t1"])
    weight = average_uniqueness(events["t1"], num_co)

    out = pd.DataFrame(index=events.index)
    out["side"] = events["side"]
    out["t1"] = events["t1"]
    out["ret"] = bins["ret"]
    out["bin"] = bins["bin"]
    out["weight"] = weight
    return out
