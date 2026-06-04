"""Data loading for the experimental pipeline.

Plan §8 Stage 1 deliverable.

Wraps ``stml.io.load_clean_data`` (the shared spine, R8 read-only) and exposes
per-instrument frames in the shape the rest of the pipeline expects:

* ``load_panel()`` → ``(ohlcv_clean, signals)`` — clean tidy-long OHLCV +
  wide-format signals, both from the released CSVs.
* ``per_instrument_frames(ohlcv, signals)`` → ``dict[instrument, DataFrame]``
  with columns ``[open, high, low, close, volume, open_interest, signal]``
  indexed by trading date for each of the 11 instruments. Zero-volume weekday
  rows are KEPT (valid settle bars) per the shared NA policy.
* ``trading_day_index(close)`` → the instrument's own dense trading-day index,
  used for per-instrument concurrency in the labels and per-instrument embargo
  in CV.
* ``trading_day_offset(close, t, offset)`` → t advanced by ``offset`` trading
  days on the instrument's own calendar.

The loader does NOT compute any features; that's `features/`. It also does not
filter on the predict window — that's the responsibility of `splits.py` and
``config.PipelineConfig.global_train_cut``.
"""

from __future__ import annotations

import pandas as pd

from stml import io as stml_io
from stml.experimental.config import INSTRUMENTS

# Required OHLCV columns. ``open_interest`` is optional — both teams
# observed that some instruments have OI populated and some have all-NaN; the
# microstructure features tolerate that.
OHLCV_COLUMNS: tuple[str, ...] = (
    "open", "high", "low", "close", "volume", "open_interest",
)


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the cleaned OHLCV panel + wide signals frame.

    Returns
    -------
    ohlcv_clean : long format — columns ``[date, instrument, *OHLCV_COLUMNS]``,
        sorted by ``(instrument, date)``, with the shared NA policy applied
        (weekend / non-finite / bad-bounds rows dropped, zero-volume weekday
        settles kept).
    signals : wide format — columns ``[date, *INSTRUMENTS]`` with values in
        ``{-1, 0, +1}``.
    """
    ohlcv, signals = stml_io.load_clean_data()
    return ohlcv, signals


def per_instrument_frames(
    ohlcv: pd.DataFrame, signals: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Pivot the long OHLCV + wide signals into one ``DataFrame`` per instrument.

    Each output frame is indexed by ``pd.DatetimeIndex`` (the instrument's own
    dense trading-day axis — no calendar reindexing) with columns
    ``[open, high, low, close, volume, open_interest, signal]``. The ``signal``
    column is the wide-frame's instrument column reindexed to the OHLCV dates
    (so any signal date the instrument was closed becomes NaN).

    Parameters
    ----------
    ohlcv : long OHLCV (the output of :func:`load_panel`).
    signals : wide signals (same).

    Returns
    -------
    dict mapping ``instrument → frame``, restricted to the 11 instruments in
    ``config.INSTRUMENTS`` (any extras silently ignored).
    """
    # Pre-sort and pre-set the index once on the full frame so per-instrument
    # slicing is O(1) per instrument rather than O(N) per slice.
    panel = (
        ohlcv.loc[:, ["date", "instrument", *OHLCV_COLUMNS]]
        .sort_values(["instrument", "date"])
        .reset_index(drop=True)
    )

    out: dict[str, pd.DataFrame] = {}
    sig_indexed = signals.set_index("date")
    for inst in INSTRUMENTS:
        if inst not in sig_indexed.columns:
            continue
        rows = panel.loc[panel["instrument"] == inst].set_index("date")
        if rows.empty:
            continue
        frame = rows.loc[:, list(OHLCV_COLUMNS)].copy()
        # Align the wide signal column to this instrument's trading dates.
        frame["signal"] = sig_indexed[inst].reindex(frame.index)
        # Cast volumes / OI to float so NaNs propagate cleanly downstream.
        frame["volume"] = frame["volume"].astype(float)
        if "open_interest" in frame.columns:
            frame["open_interest"] = frame["open_interest"].astype(float)
        out[inst] = frame
    return out


def trading_day_index(close: pd.Series) -> pd.DatetimeIndex:
    """Return ``close``'s own dense trading-day index (no calendar grid)."""
    return pd.DatetimeIndex(close.dropna().index)


def trading_day_offset(
    close: pd.Series, t: pd.Timestamp, offset: int
) -> pd.Timestamp | None:
    """Advance ``t`` by ``offset`` trading days on ``close``'s own calendar.

    Returns ``None`` if the resulting position falls outside the instrument's
    history (e.g. ``t + max_holding`` is past the last bar).

    Parameters
    ----------
    close : the instrument's close series (sorted, ``DatetimeIndex``).
    t : the anchor date. Must be in ``close.index`` (else ``KeyError``).
    offset : positive = forward, negative = backward, 0 = ``t`` itself.
    """
    idx = trading_day_index(close)
    pos = idx.get_loc(t)
    if isinstance(pos, slice):
        # Duplicate index — take the first occurrence. Should not happen on
        # the cleaned panel but defend against it.
        pos = pos.start
    target = pos + int(offset)
    if target < 0 or target >= len(idx):
        return None
    return idx[target]
