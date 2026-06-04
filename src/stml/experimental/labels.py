"""Triple-barrier labels with **t+1 entry** — methodology spec / §8 Stage 1 deliverable.

This module is the single labelling implementation .
Three load-bearing decisions vs the AFML Ch.3 default and vs the prior
labeller:

* **Entry at the close of t+1, not t.** A signal observed at the close of bar
  ``t`` is acted on at the close of ``t+1``. The held window is
  ``[t+1, t+1+h]``; the bar between ``t`` and ``t+1`` is no longer inside the
  event. This is the load-bearing fix (prior audit) —
  empirically ``corr(s_t, r_{t+1}) > 0`` for all 11 instruments.

* **Tighter symmetric barriers by default** (``pt = sl = 0.5``). The branch
  audit (prior audit and the methodology spec) shows the EWMA-σ̂ ×
  ``pt=sl=1.0`` barrier resolves at the vertical line 50–65 % of the time on
  the released data — labels degenerate to "10-day drift sign" rather than a
  triple-barrier first-touch. The S2 CPCV barrier search may override pt/sl/h
  per asset class; this is the *initial* config.

* **Per-instrument concurrency on each instrument's native trading-day index.**
  Concurrency for AFML Ch.4 uniqueness weights is computed on the instrument's
  own dense index (no calendar reindexing) — so weekend / cross-venue holiday
  rows that don't trade for the instrument are not counted as overlap.

Output schema (one row per non-zero signal day that resolves to a barrier):

    instrument          str
    t_signal            pd.Timestamp        — signal observation date (close of t)
    t_start             pd.Timestamp        — entry date = t+1 trading day
    t_end               pd.Timestamp        — resolution date (first PT/SL touch or vertical)
    side                int                 — +1 (long bet) / -1 (short bet)
    ret                 float               — side · log(close[t_end] / close[t_start])
    label               int                 — 1 if ret > 0 else 0
    uniqueness_weight   float               — AFML Ch.4, in (0, 1]
    sigma_at_t          float               — daily σ̂ at the signal date (barrier scale)
    barrier_hit         str                 — 'pt' / 'sl' / 'vertical'

Citations
---------
López de Prado, M. (2018). *Advances in Financial Machine Learning*, Ch.3-4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration container.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelConfig:
    """Per-call configuration for :func:`triple_barrier_labels`.

    The defaults mirror :class:`stml.experimental.config.PipelineConfig` so the
    label module is callable both with a full ``PipelineConfig`` (via
    ``from_pipeline_config``) and as a standalone helper.
    """

    pt_mult: float = 0.5
    sl_mult: float = 0.5
    max_holding: int = 10  # h, in trading days
    min_uniqueness: float = 0.0  # drop events with weight < this (default keeps all)


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def triple_barrier_labels(
    close: pd.Series,
    signal: pd.Series,
    sigma: pd.Series,
    *,
    instrument: str,
    config: LabelConfig | None = None,
) -> pd.DataFrame:
    """Compute triple-barrier labels for one instrument with t+1 entry.

    Parameters
    ----------
    close
        Strictly-positive close-price Series, ``DatetimeIndex`` = the
        instrument's dense trading-day index (no calendar reindexing).
    signal
        Primary signal Series in ``{-1, 0, +1}`` aligned to ``close.index``.
        Zero rows are silently dropped; non-zero rows become candidate events.
    sigma
        **Daily** σ̂ Series aligned to ``close.index``. The barrier widths at
        event ``t`` are ``pt_mult · σ̂_t · √h`` (PT) and ``-sl_mult · σ̂_t · √h``
        (SL), in *signed-bet-direction log-return* units.
    instrument
        Instrument ticker (e.g. ``'cl1s'``) — copied verbatim into the output
        ``instrument`` column.
    config
        Per-call config; defaults to symmetric ``pt=sl=0.5``, ``h=10``.

    Returns
    -------
    pd.DataFrame
        One row per event with the schema documented in the module docstring.
        Empty DataFrame if no signal rows resolve to a labelled event.

    Notes
    -----
    * Events whose entry bar ``t+1`` is past the last available close emit
      no row (truncation at the right edge).
    * Events whose σ̂ is NaN at the signal date emit no row.
    * Same-instrument concurrency is computed on the bar position index after
      labels are resolved — so the uniqueness weights are consistent regardless
      of the calendar gaps within the held window.
    """
    cfg = config or LabelConfig()

    # Defensive copy + alignment. Drop signal rows where sigma is NaN; those
    # events can't be sized.
    close = close.astype(float).copy()
    signal = signal.reindex(close.index).astype("Int64")
    sigma_aligned = sigma.reindex(close.index).astype(float)

    event_signals = signal.dropna().astype(int)
    event_signals = event_signals[event_signals != 0]
    if event_signals.empty:
        return _empty_label_frame()

    rows = []
    bar_positions: dict[pd.Timestamp, int] = {
        ts: i for i, ts in enumerate(close.index)
    }

    for t_signal, side in event_signals.items():
        if t_signal not in bar_positions:
            continue
        pos_t = bar_positions[t_signal]
        pos_entry = pos_t + 1
        pos_vert = pos_entry + cfg.max_holding
        # Drop events whose full [t+1, t+1+h] window is not available on this
        # instrument's calendar — matches the labels.py convention (§4.3).
        # Truncating to fewer than h bars would change the barrier semantics
        # (barrier width is sized by sqrt(h)).
        if pos_vert >= len(close):
            continue

        sigma_t = float(sigma_aligned.iloc[pos_t])
        if not np.isfinite(sigma_t) or sigma_t <= 0.0:
            continue

        t_start = close.index[pos_entry]
        entry_close = float(close.iloc[pos_entry])
        if not np.isfinite(entry_close) or entry_close <= 0.0:
            continue

        h = cfg.max_holding
        pt_width = cfg.pt_mult * sigma_t * np.sqrt(h)
        sl_width = cfg.sl_mult * sigma_t * np.sqrt(h)

        t_end_pos, barrier_hit, ret = _first_barrier_touch(
            close=close,
            pos_entry=pos_entry,
            pos_vert=pos_vert,
            side=int(side),
            entry_close=entry_close,
            pt_width=pt_width,
            sl_width=sl_width,
        )
        t_end = close.index[t_end_pos]
        rows.append(
            {
                "instrument": instrument,
                "t_signal": t_signal,
                "t_start": t_start,
                "t_end": t_end,
                "_pos_start": pos_entry,
                "_pos_end": t_end_pos,
                "side": int(side),
                "ret": float(ret),
                "label": 1 if ret > 0.0 else 0,
                "sigma_at_t": sigma_t,
                "barrier_hit": barrier_hit,
            }
        )

    if not rows:
        return _empty_label_frame()

    events = pd.DataFrame(rows)
    events["uniqueness_weight"] = _uniqueness_weights(
        events["_pos_start"].values,
        events["_pos_end"].values,
        n_bars=len(close),
    )
    events = events.drop(columns=["_pos_start", "_pos_end"])

    if cfg.min_uniqueness > 0.0:
        events = events.loc[events["uniqueness_weight"] >= cfg.min_uniqueness].reset_index(drop=True)

    column_order = [
        "instrument", "t_signal", "t_start", "t_end",
        "side", "ret", "label", "uniqueness_weight", "sigma_at_t", "barrier_hit",
    ]
    return events.loc[:, column_order].reset_index(drop=True)


def label_panel(
    panel: dict[str, pd.DataFrame],
    sigmas: dict[str, pd.Series],
    *,
    config: LabelConfig | None = None,
) -> pd.DataFrame:
    """Convenience wrapper — label every instrument in ``panel`` and concat.

    Parameters
    ----------
    panel
        ``{instrument: DataFrame[open, high, low, close, volume, ..., signal]}``
        — output of :func:`stml.experimental.data_loader.per_instrument_frames`.
    sigmas
        ``{instrument: pd.Series}`` of daily σ̂, one per instrument.
    config
        Optional ``LabelConfig``; default symmetric pt=sl=0.5 h=10.

    Returns
    -------
    pd.DataFrame
        Concatenated event frame, sorted by ``(instrument, t_signal)``.
    """
    parts = []
    for inst, frame in panel.items():
        if "close" not in frame.columns or "signal" not in frame.columns:
            continue
        sigma = sigmas.get(inst)
        if sigma is None:
            continue
        labelled = triple_barrier_labels(
            close=frame["close"],
            signal=frame["signal"],
            sigma=sigma,
            instrument=inst,
            config=config,
        )
        if not labelled.empty:
            parts.append(labelled)
    if not parts:
        return _empty_label_frame()
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["instrument", "t_signal"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Internals.
# ---------------------------------------------------------------------------


_LABEL_COLUMNS = [
    "instrument", "t_signal", "t_start", "t_end",
    "side", "ret", "label", "uniqueness_weight", "sigma_at_t", "barrier_hit",
]


def _empty_label_frame() -> pd.DataFrame:
    """Empty event frame with the canonical schema (so consumers can rely on it)."""
    return pd.DataFrame({col: pd.Series(dtype=_LABEL_DTYPE[col]) for col in _LABEL_COLUMNS})


_LABEL_DTYPE: dict[str, object] = {
    "instrument": "string",
    "t_signal": "datetime64[ns]",
    "t_start": "datetime64[ns]",
    "t_end": "datetime64[ns]",
    "side": "int64",
    "ret": "float64",
    "label": "int64",
    "uniqueness_weight": "float64",
    "sigma_at_t": "float64",
    "barrier_hit": "string",
}


def _first_barrier_touch(
    *,
    close: pd.Series,
    pos_entry: int,
    pos_vert: int,
    side: int,
    entry_close: float,
    pt_width: float,
    sl_width: float,
) -> tuple[int, str, float]:
    """Scan close[pos_entry+1 ... pos_vert] for first PT or SL touch.

    Per the methodology spec / algorithm:
        for u in (t+2, …, t+1+h):
            signed_dist = side · log(close[u] / entry_close)
            if signed_dist ≥ +pt_width: PT touch
            if signed_dist ≤ -sl_width: SL touch
        t1 = u_first_touch or t+1+h
        ret = side · log(close[t1] / entry_close)
    """
    sign = float(side)
    for pos in range(pos_entry + 1, pos_vert + 1):
        c = float(close.iloc[pos])
        if not np.isfinite(c) or c <= 0.0:
            continue
        signed_dist = sign * (np.log(c) - np.log(entry_close))
        if signed_dist >= pt_width:
            return pos, "pt", float(signed_dist)
        if signed_dist <= -sl_width:
            return pos, "sl", float(signed_dist)

    # No touch — vertical timeout. Compute signed return at the vertical bar.
    c = float(close.iloc[pos_vert])
    ret = sign * (np.log(c) - np.log(entry_close)) if c > 0.0 else 0.0
    return pos_vert, "vertical", float(ret)


def _uniqueness_weights(
    pos_starts: np.ndarray, pos_ends: np.ndarray, *, n_bars: int
) -> np.ndarray:
    """AFML Ch.4 average-uniqueness weights — vectorised O(N + B) per instrument.

    For event ``i`` spanning ``[pos_starts[i], pos_ends[i]]`` (inclusive), the
    weight is ``mean_{bar in span}(1 / concurrency[bar])`` where
    ``concurrency[bar]`` is the number of events containing ``bar``. Computed
    via the standard diff/cumsum trick.
    """
    pos_starts = np.asarray(pos_starts, dtype=int)
    pos_ends = np.asarray(pos_ends, dtype=int)
    n = len(pos_starts)
    if n == 0:
        return np.zeros(0, dtype=float)

    # diff/cumsum to build concurrency.
    delta = np.zeros(n_bars + 1, dtype=int)
    for i in range(n):
        delta[pos_starts[i]] += 1
        delta[pos_ends[i] + 1] -= 1
    concurrency = np.cumsum(delta)[:n_bars]
    concurrency = np.maximum(concurrency, 1)  # defensive; should always be >= 1 within any span

    inv_conc = 1.0 / concurrency.astype(float)
    # Prefix-sum to compute mean over a span in O(1) per event.
    inv_conc_cumsum = np.concatenate([[0.0], np.cumsum(inv_conc)])
    weights = np.empty(n, dtype=float)
    for i in range(n):
        span_len = pos_ends[i] - pos_starts[i] + 1
        s = inv_conc_cumsum[pos_ends[i] + 1] - inv_conc_cumsum[pos_starts[i]]
        weights[i] = s / max(span_len, 1)
    return weights
