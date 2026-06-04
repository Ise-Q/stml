"""labels.py — canonical triple-barrier labels with next-day execution.

Step 2 of Harry's contribution. Produces meta-labels for every non-zero
primary-signal event in the panel, with three load-bearing decisions vs the
implementations on other branches:

1. **Entry at t+1, not t.** The signal observed at the close of day ``t`` is
   acted on with execution beginning at the close of day ``t+1`` — the
   convention the Step 1 signal-direction audit empirically confirmed
   (``corr(s_t, r_{t+1}) > 0`` for all 11 instruments). Sreeram's
   ``labeling.py`` resolves the event over ``[t, t+h]``, double-counting the
   close return between ``t`` and ``t+1`` inside the held window. Harry's
   resolves over ``[t+1, t+1+h]``. See the worked 5-row example in
   ``reports/harry/02-labels.md`` for the case where these two conventions
   produce different labels.

2. **Asymmetric barrier parameters, symmetric default.** ``pt_mult`` and
   ``sl_mult`` are independently configurable, but the default is
   ``1.0 / 1.0``. The Step 1 audit produced 0 trend / 4 mean-reverting / 7
   mixed sign labels under the canonical multi-horizon classifier — too
   equivocal to bake a counter-trend bias into the label itself.
   Asymmetric variants live in the Step 4 sensitivity sweep, not in the
   label default.

3. **Trading-day concurrency for uniqueness weights.** AFML Ch. 4 sample
   uniqueness is computed on each instrument's native trading-day index,
   not on calendar days. A 3-day weekend between two events does not
   inflate or shrink the computed concurrency — see
   ``_per_instrument_uniqueness`` and the corresponding test.

EWMA vol-scaling::

    sigma_t        = EWMA daily-log-return std up to and INCLUDING bar t
                     (span = ``vol_span``, default 100).
    barrier width  = mult * sigma_t * sqrt(h)
    PT  (long)     : close >= entry * exp(+pt_mult * sigma_t * sqrt(h))
    SL  (long)     : close <= entry * exp(-sl_mult * sigma_t * sqrt(h))
    PT  (short)    : close <= entry * exp(-pt_mult * sigma_t * sqrt(h))
    SL  (short)    : close >= entry * exp(+sl_mult * sigma_t * sqrt(h))

Touch logic is close-to-close. First barrier hit wins; if neither is hit by
``t+1+h`` the vertical barrier resolves the event with ``label =
sign(realized signed return)``. The entry bar itself is excluded from the
touch scan (the signed log-distance from entry to entry is zero).

Public API::

    from stml.harry.labels import get_meta_labels
    events = get_meta_labels(
        ohlcv, signals,
        h=10, pt_mult=1.0, sl_mult=1.0, vol_span=100,
    )

returns a DataFrame with columns:

  instrument          — ticker.
  t_signal            — date the primary signal was observed (close of t).
  t_start             — entry date = ``t_signal + 1`` trading day.
  t_end               — resolution date (first barrier touch or vertical).
  side                — primary signal value in {-1, +1}.
  ret                 — ``side * (log(close[t_end]) - log(close[t_start]))``.
  label               — ``1`` if ``ret > 0`` else ``0``.
  uniqueness_weight   — AFML Ch.4 uniqueness in [0, 1] for this event.
  sigma               — ``sigma_t`` used to set the barrier widths.

Events with the released signal ``= 0`` are skipped (no bet). Events
without enough forward data to fill the ``[t+1, t+1+h]`` window are also
skipped. Both filters are silent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Universe + defaults                                                          #
# --------------------------------------------------------------------------- #
INSTRUMENTS: tuple[str, ...] = (
    "es1s", "nq1s", "fesx1s",
    "cl1s", "ho1s", "rb1s", "ng1s",
    "gc1s", "si1s", "hg1s", "pl1s",
)

DEFAULT_H: int = 10
DEFAULT_PT_MULT: float = 1.0
DEFAULT_SL_MULT: float = 1.0
DEFAULT_VOL_SPAN: int = 100


@dataclass(frozen=True)
class TripleBarrierConfig:
    """Parameters of the triple-barrier label.

    Attributes
    ----------
    h         : holding horizon in trading days (vertical barrier).
    pt_mult   : profit-take multiplier on ``sigma_t * sqrt(h)``.
    sl_mult   : stop-loss multiplier on ``sigma_t * sqrt(h)``.
    vol_span  : EWMA span (days) for the daily-log-return std used as
                ``sigma_t``.
    """

    h: int = DEFAULT_H
    pt_mult: float = DEFAULT_PT_MULT
    sl_mult: float = DEFAULT_SL_MULT
    vol_span: int = DEFAULT_VOL_SPAN

    def __post_init__(self) -> None:
        if self.h < 1:
            raise ValueError(f"h must be >= 1, got {self.h}")
        if self.pt_mult < 0:
            raise ValueError(f"pt_mult must be >= 0, got {self.pt_mult}")
        if self.sl_mult < 0:
            raise ValueError(f"sl_mult must be >= 0, got {self.sl_mult}")
        if self.vol_span < 2:
            raise ValueError(f"vol_span must be >= 2, got {self.vol_span}")

    @property
    def sqrt_h(self) -> float:
        return float(np.sqrt(self.h))


# --------------------------------------------------------------------------- #
# Volatility                                                                   #
# --------------------------------------------------------------------------- #
def ewma_daily_vol(close: pd.Series, span: int = DEFAULT_VOL_SPAN) -> pd.Series:
    """Causal EWMA std of daily log returns.

    Index is the close series's date index after sorting and dropping
    NaN values. The first value is NaN (``diff`` requires two points);
    subsequent values are the EWMA std with ``adjust=False`` — strictly
    causal (one-pass, no peeking at the future).
    """
    close = close.sort_index().dropna()
    r = np.log(close).diff()
    return r.ewm(span=span, adjust=False).std()


# --------------------------------------------------------------------------- #
# Per-event resolution                                                         #
# --------------------------------------------------------------------------- #
def _resolve_event(
    close_window: np.ndarray,
    side: int,
    pt_threshold: float,
    sl_threshold: float,
) -> tuple[int, float]:
    """Resolve one event.

    Parameters
    ----------
    close_window  : 1-D array of close prices; ``close_window[0]`` is the
                    entry bar (already shifted to t+1 by the caller).
    side          : +1 (long) or -1 (short).
    pt_threshold  : non-negative log-return distance to the profit barrier
                    in the direction of the bet.
    sl_threshold  : non-negative log-return distance to the stop barrier
                    against the bet.

    Returns
    -------
    end_offset    : positional offset within ``close_window`` of the
                    resolution bar. ``0`` is impossible (entry bar
                    excluded); the maximum is ``len(window) - 1`` (the
                    vertical barrier).
    signed_ret    : ``side * (log(close[end]) - log(close[entry]))`` —
                    the signed log-return realised over [entry, end].
    """
    if len(close_window) < 2:
        raise ValueError(
            f"close_window must have length >= 2, got {len(close_window)}"
        )
    if side not in (-1, 1):
        raise ValueError(f"side must be -1 or +1, got {side}")
    entry = float(close_window[0])
    log_dist = np.log(close_window) - np.log(entry)
    signed_dist = log_dist * side  # +ve toward profit, -ve toward loss
    profit_mask = signed_dist >= pt_threshold
    loss_mask = signed_dist <= -sl_threshold
    touch_mask = (profit_mask | loss_mask).copy()
    # The entry bar has zero signed distance and is never a "touch", even if
    # someone passes pt_threshold == 0 (degenerate). Force it off.
    touch_mask[0] = False
    if touch_mask.any():
        end_offset = int(np.argmax(touch_mask))
    else:
        end_offset = len(close_window) - 1
    return end_offset, float(signed_dist[end_offset])


# --------------------------------------------------------------------------- #
# Per-instrument labelling                                                     #
# --------------------------------------------------------------------------- #
def _label_one_instrument(
    close: pd.Series,
    signal: pd.Series,
    cfg: TripleBarrierConfig,
    sigma: pd.Series | None = None,
) -> pd.DataFrame:
    """Produce the events frame for one instrument.

    ``sigma`` may be provided to bypass EWMA (used by tests that need a
    known barrier width). If ``None`` it is computed from ``close`` via
    :func:`ewma_daily_vol`.
    """
    close = close.sort_index().dropna()
    signal = signal.sort_index().reindex(close.index)
    if sigma is None:
        sigma = ewma_daily_vol(close, span=cfg.vol_span)
    else:
        sigma = sigma.reindex(close.index)

    dates = close.index
    close_vals = close.to_numpy(dtype=np.float64)
    sigma_vals = sigma.to_numpy(dtype=np.float64)
    signal_vals = signal.to_numpy(dtype=np.float64)
    n = len(close)
    sqrt_h = cfg.sqrt_h

    records: list[dict] = []
    for pos in range(n):
        s = signal_vals[pos]
        if not np.isfinite(s) or s == 0:
            continue
        entry_pos = pos + 1
        end_pos_max = entry_pos + cfg.h
        if end_pos_max >= n:
            continue  # not enough forward data for a full [t+1, t+1+h] window
        sigma_t = sigma_vals[pos]
        if not np.isfinite(sigma_t) or sigma_t <= 0:
            continue
        window = close_vals[entry_pos:end_pos_max + 1]
        pt_thresh = cfg.pt_mult * sigma_t * sqrt_h
        sl_thresh = cfg.sl_mult * sigma_t * sqrt_h
        side = int(np.sign(s))
        offset, signed_ret = _resolve_event(window, side, pt_thresh, sl_thresh)
        records.append(
            {
                "t_signal": dates[pos],
                "t_start": dates[entry_pos],
                "t_end": dates[entry_pos + offset],
                "side": side,
                "ret": float(signed_ret),
                "label": int(signed_ret > 0),
                "sigma": float(sigma_t),
            }
        )
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Sample uniqueness (AFML Ch.4) on the instrument's native trading-day index   #
# --------------------------------------------------------------------------- #
def _per_instrument_uniqueness(
    events_inst: pd.DataFrame, bar_index: pd.DatetimeIndex
) -> np.ndarray:
    """AFML Ch. 4 uniqueness weights, indexed on the trading-day bar set.

    For each bar ``u``::

        concurrency[u] = #{ events i : pos_start_i <= u <= pos_end_i }

    For each event ``i``::

        uniqueness_i   = mean over u in [pos_start_i, pos_end_i] of
                         (1 / concurrency[u])

    ``bar_index`` MUST be the cleaned trading-day index of the instrument
    (e.g. ``close.dropna().index``). Any NaN-filled weekend rows in the
    underlying panel must be dropped before calling this function — that
    is how trading-day concurrency is guaranteed regardless of how the
    caller represents the calendar upstream.
    """
    if len(events_inst) == 0:
        return np.array([])
    bar_index = pd.Index(bar_index)
    pos_start = bar_index.get_indexer(events_inst["t_start"])
    pos_end = bar_index.get_indexer(events_inst["t_end"])
    if (pos_start < 0).any() or (pos_end < 0).any():
        raise ValueError(
            "event t_start / t_end dates not all present in bar_index — "
            "uniqueness can only be computed on the cleaned trading-day index"
        )
    n_bars = len(bar_index)
    # Concurrency via the diff/cumsum trick — O(n_events + n_bars) rather
    # than O(n_events * avg_span) and correct for overlapping events.
    delta = np.zeros(n_bars + 1, dtype=np.float64)
    np.add.at(delta, pos_start, 1.0)
    np.add.at(delta, pos_end + 1, -1.0)
    concurrency = np.cumsum(delta)[:n_bars]

    uniqueness = np.empty(len(events_inst), dtype=np.float64)
    for i, (ps, pe) in enumerate(zip(pos_start, pos_end, strict=True)):
        c = concurrency[ps : pe + 1]
        uniqueness[i] = float((1.0 / c).mean())
    return uniqueness


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
_EVENT_COLS: tuple[str, ...] = (
    "instrument",
    "t_signal",
    "t_start",
    "t_end",
    "side",
    "ret",
    "label",
    "uniqueness_weight",
    "sigma",
)


def get_meta_labels(
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    h: int = DEFAULT_H,
    pt_mult: float = DEFAULT_PT_MULT,
    sl_mult: float = DEFAULT_SL_MULT,
    vol_span: int = DEFAULT_VOL_SPAN,
    instruments: list[str] | None = None,
    sigma: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Build the events + labels + uniqueness DataFrame for the panel.

    Parameters
    ----------
    ohlcv       : long-format DataFrame with columns ``date``,
                  ``instrument``, ``close``.
    signals     : wide-format DataFrame with columns ``date`` plus one
                  per instrument; values in ``{-1, 0, +1}``.
    h, pt_mult, sl_mult, vol_span : triple-barrier parameters; see
                  :class:`TripleBarrierConfig`.
    instruments : optional subset; defaults to :data:`INSTRUMENTS`. Unknown
                  tickers are silently skipped.
    sigma       : optional ``{instrument: Series}`` mapping to inject a
                  known sigma per instrument (bypasses EWMA). Used by
                  tests; not needed in production.

    Returns
    -------
    DataFrame with columns
    ``[instrument, t_signal, t_start, t_end, side, ret, label,
       uniqueness_weight, sigma]``.
    """
    cfg = TripleBarrierConfig(
        h=h, pt_mult=pt_mult, sl_mult=sl_mult, vol_span=vol_span
    )
    if instruments is None:
        instruments = list(INSTRUMENTS)

    sigs = signals.copy()
    sigs["date"] = pd.to_datetime(sigs["date"])
    sigs = sigs.set_index("date").sort_index()

    ohlcv_local = ohlcv.copy()
    ohlcv_local["date"] = pd.to_datetime(ohlcv_local["date"])

    all_events: list[pd.DataFrame] = []
    for inst in instruments:
        if inst not in sigs.columns:
            continue
        sub = ohlcv_local.loc[
            ohlcv_local["instrument"] == inst, ["date", "close"]
        ]
        if sub.empty:
            continue
        close = sub.sort_values("date").set_index("date")["close"]
        signal = sigs[inst].astype("float64")
        inst_sigma = None if sigma is None else sigma.get(inst)
        events = _label_one_instrument(close, signal, cfg, sigma=inst_sigma)
        if events.empty:
            continue
        bar_index = close.dropna().sort_index().index
        events["uniqueness_weight"] = _per_instrument_uniqueness(events, bar_index)
        events["instrument"] = inst
        all_events.append(events)

    if not all_events:
        return pd.DataFrame(columns=list(_EVENT_COLS))
    out = pd.concat(all_events, ignore_index=True)
    return out[list(_EVENT_COLS)]


__all__ = [
    "INSTRUMENTS",
    "DEFAULT_H",
    "DEFAULT_PT_MULT",
    "DEFAULT_SL_MULT",
    "DEFAULT_VOL_SPAN",
    "TripleBarrierConfig",
    "ewma_daily_vol",
    "get_meta_labels",
]
