"""Labelling variants + baselines for the EX.5 barrier sweep.

All functions emit the shipped ``triple_barrier_labels`` schema (``side, t1, ret, bin, weight``)
plus a ``barrier_type`` provenance column, so the sweep harness can swap labellers freely and
report the label distribution (% pt / sl / timeout). They **reuse** the shipped triple-barrier
primitives (``apply_pt_sl_on_t1``, ``get_bins``, ``get_num_co_events``, ``average_uniqueness``)
rather than re-deriving them — the core ``triple_barrier.py`` is left untouched.

Contents:
- ``triple_barrier_labels_ext`` — superset of the shipped labeller: adds ``barrier_type`` and
  accepts a **per-event** vertical horizon (enables V2). Equal to the core on the default config.
- ``vol_scaled_horizon`` — V2 per-event horizon ``h_t = round(h0·σ̄/σ_t)``, longer when σ is low,
  with a **causal** trailing σ̄ (expanding mean, shifted), clipped to ``[h_min, h_max]``.
- ``fixed_horizon_labels`` — baseline B1: ``bin = 1{ side·r_{t,t+h} > τ }`` (López de Prado AFML
  Ch. 3 fixed-time-horizon; T3.03 L1 §2.1). The label may look forward (it is a *label*).
- ``trend_scan_labels`` — baseline B2: forward trend scanning mapped to act/skip via agreement
  with the primary side. Forward-looking is correct *for a label* (leakage only bites features).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._vendor.trend_scanning import trend_labels
from .triple_barrier import (
    apply_pt_sl_on_t1,
    average_uniqueness,
    get_bins,
    get_num_co_events,
)


def _vertical_barrier(close: pd.Series, t_events: pd.Index, max_holding) -> pd.Series:
    """Vertical-barrier timestamps; ``max_holding`` may be an int OR a per-event Series (V2)."""
    idx = close.index
    pos = idx.get_indexer(t_events)
    if isinstance(max_holding, pd.Series):
        h = max_holding.reindex(t_events).to_numpy(dtype=float)
        h = np.where(np.isfinite(h), h, 0.0)
    else:
        h = np.full(len(pos), float(max_holding))
    vb = np.minimum(pos + h.astype(int), len(idx) - 1)
    return pd.Series(idx[vb], index=t_events)


def _barrier_type(touches: pd.DataFrame, first: pd.Series) -> pd.Series:
    """Which barrier was touched first: 'pt' | 'sl' | 'vertical' (pt wins exact ties)."""
    bt = pd.Series("vertical", index=touches.index, dtype=object)
    bt[touches["sl"] == first] = "sl"
    bt[touches["pt"] == first] = "pt"
    return bt


def triple_barrier_labels_ext(
    close: pd.Series,
    signal: pd.Series,
    target: pd.Series,
    *,
    pt_sl: tuple[float, float] = (1.0, 1.0),
    max_holding=10,
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """Triple-barrier meta-labels with ``barrier_type`` and a per-event-capable horizon.

    Mirrors ``triple_barrier.triple_barrier_labels`` exactly on the shared columns (regression
    tested) while exposing which barrier resolved each label and accepting a per-event
    ``max_holding`` Series for the vol-scaled vertical (V2).
    """
    close = close.sort_index()
    t_events = signal.index[signal.to_numpy() != 0]
    side = np.sign(signal.loc[t_events]).astype(float)
    vertical = _vertical_barrier(close, t_events, max_holding)

    trgt = target.loc[target.index.intersection(t_events)]
    trgt = trgt[trgt > min_ret]
    side_ = side.loc[trgt.index]
    events = pd.concat({"t1": vertical, "trgt": trgt, "side": side_}, axis=1).dropna(
        subset=["trgt"]
    )

    touches = apply_pt_sl_on_t1(close, events, pt_sl)  # cols t1(vertical), sl, pt
    first = touches.min(axis=1)  # earliest of {pt, sl, vertical}
    barrier_type = _barrier_type(touches, first)
    events["t1"] = first

    bins = get_bins(events, close)
    num_co = get_num_co_events(close.index, events["t1"])
    weight = average_uniqueness(events["t1"], num_co)

    out = pd.DataFrame(index=events.index)
    out["side"] = events["side"]
    out["t1"] = events["t1"]
    out["ret"] = bins["ret"]
    out["bin"] = bins["bin"]
    out["weight"] = weight
    out["barrier_type"] = barrier_type
    return out


def vol_scaled_horizon(
    sigma: pd.Series,
    t_events: pd.Index,
    h0: int,
    *,
    h_min: int,
    h_max: int,
) -> pd.Series:
    """V2 per-event vertical horizon: longer when σ is low, ``h_t = round(h0·σ̄/σ_t)``.

    σ̄ is a **causal** trailing mean (expanding mean of σ, shifted one bar so the current event's
    own σ is excluded from the normaliser). Warm-up events with no trailing mean fall back to
    ``h0``. Clipped to ``[h_min, h_max]``. Returned as integer horizons indexed by ``t_events``.
    """
    sigma = sigma.sort_index()
    sigma_bar = sigma.expanding(min_periods=2).mean().shift(1)  # trailing, excludes current bar
    ratio = sigma_bar / sigma
    h = (h0 * ratio).round().reindex(t_events).fillna(h0)
    return h.clip(lower=h_min, upper=h_max).astype(int)


def fixed_horizon_labels(
    close: pd.Series, signal: pd.Series, h: int, tau: float = 0.0
) -> pd.DataFrame:
    """Baseline B1 — fixed-time-horizon: ``bin = 1{ side·r_{t,t+h} > τ }``.

    Each non-zero-signal event is labelled by the sign (vs threshold ``τ``) of its side-adjusted
    ``h``-bar-ahead return; ``t1 = t+h``. The last ``h`` bars (no full horizon) are dropped. Same
    output schema as the triple-barrier labeller; ``barrier_type`` is always ``'vertical'``.
    """
    close = close.sort_index()
    idx = close.index
    t_events = signal.index[signal.to_numpy() != 0]
    pos = idx.get_indexer(t_events)
    valid = (pos >= 0) & (pos + h < len(idx))
    t_events, pos = t_events[valid], pos[valid]
    side = np.sign(signal.loc[t_events].to_numpy()).astype(float)

    prices = close.to_numpy()
    ret = (prices[pos + h] / prices[pos] - 1.0) * side
    t1 = pd.Series(idx[pos + h], index=t_events)
    num_co = get_num_co_events(idx, t1)
    weight = average_uniqueness(t1, num_co).reindex(t_events)

    out = pd.DataFrame(index=t_events)
    out["side"] = side
    out["t1"] = t1
    out["ret"] = ret
    out["bin"] = (ret > tau).astype(float)
    out["weight"] = weight.to_numpy()
    out["barrier_type"] = "vertical"
    return out


def trend_scan_labels(
    close: pd.Series, signal: pd.Series, span: tuple[int, int] = (5, 20)
) -> pd.DataFrame:
    """Baseline B2 — trend scanning as a meta-label: act iff the trend agrees with the side.

    Runs the vendored forward trend-scanning (``look_forward=True`` — forward-looking is correct
    *for a label*), then ``bin = 1{ sign(tVal) == side }``. ``t1`` is the most-significant trend
    window end; ``ret`` is the side-adjusted realised return to ``t1``. Same output schema.
    """
    close = close.sort_index()
    ts = trend_labels(close, span, look_forward=True)  # cols: t1, tVal, bin(sign), windowSize
    t_events = signal.index[signal.to_numpy() != 0]
    common = t_events.intersection(ts.index)

    side = np.sign(signal.loc[common].to_numpy()).astype(float)
    trend_sign = np.sign(ts.loc[common, "bin"].to_numpy().astype(float))
    t1 = pd.DatetimeIndex(pd.to_datetime(ts.loc[common, "t1"]))

    entry = close.reindex(common).to_numpy()
    exit_px = close.reindex(t1).to_numpy()
    ret = (exit_px / entry - 1.0) * side
    t1s = pd.Series(t1, index=common)
    num_co = get_num_co_events(close.index, t1s)
    weight = average_uniqueness(t1s, num_co).reindex(common)

    out = pd.DataFrame(index=common)
    out["side"] = side
    out["t1"] = t1
    out["ret"] = ret
    out["bin"] = (trend_sign == side).astype(float)
    out["weight"] = weight.to_numpy()
    out["barrier_type"] = "trend"
    return out
