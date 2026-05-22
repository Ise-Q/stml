"""
align.py
========
Align per-instrument trading signals with forward or same-day returns.

The primary use-case is attaching the return that a signal *causes* execution
on, so that ``PnL_t = s_t * r_{t+1}`` (next-day convention) or
``PnL_t = s_t * r_t`` (same-day convention) can be computed without look-ahead.

Returns are computed via :func:`stml.na_checks.native_returns` on each
instrument's own dense close series (holiday-spanning moves are correct; no
fabricated zeros).  Structural NaNs -- dates where one venue traded but the
instrument did not -- are NEVER forward-filled or filled with zero.  The
inner-join between signal dates and dates-with-a-defined-return is the only
source of dropped rows.

Public API
----------
- :class:`AlignResult` -- named result container.
- :func:`align_instrument` -- align one instrument.
- :func:`align_panel` -- align all (or a subset of) instruments.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stml.na_checks import native_returns

__all__ = ["AlignResult", "align_instrument", "align_panel"]

# Canonical instrument list (matches io.INSTRUMENTS order)
_INSTRUMENTS: list[str] = [
    "es1s",
    "nq1s",
    "fesx1s",
    "cl1s",
    "ho1s",
    "rb1s",
    "ng1s",
    "gc1s",
    "si1s",
    "hg1s",
    "pl1s",
]


@dataclass(frozen=True)
class AlignResult:
    """Result of aligning one instrument's signals with its returns.

    Attributes
    ----------
    frame : pd.DataFrame
        Columns ``[date, signal, ret]``.  Contains only signal days that have a
        well-defined return under the chosen convention.  All ``ret`` values are
        non-null by construction.
    n_signal_days : int
        Total signal days before any dropping (len of the raw signal column).
    n_dropped : int
        Signal days dropped because no return was available under the convention
        (``n_signal_days - len(frame)``).
    retained_fraction : float
        ``len(frame) / n_signal_days``; always in ``(0, 1]`` for real data.
    """

    frame: pd.DataFrame
    n_signal_days: int
    n_dropped: int
    retained_fraction: float


def align_instrument(
    signals_wide: pd.DataFrame,
    ohlcv_long: pd.DataFrame,
    instrument: str,
    convention: str = "next_day",
) -> AlignResult:
    """Align one instrument's signal series with its returns.

    Parameters
    ----------
    signals_wide : pd.DataFrame
        Wide signals frame -- columns are ``date`` plus one column per
        instrument containing values in ``{-1, 0, 1}``.  No NaNs expected.
    ohlcv_long : pd.DataFrame
        Long OHLCV frame (artifact rows already removed by
        :func:`stml.io.load_clean_data`).  Must include columns
        ``[date, instrument, close]``.
    instrument : str
        The instrument to align (must be a column in ``signals_wide`` and
        present in ``ohlcv_long``).
    convention : str
        ``'next_day'`` (default): attach ``r_{t+1}`` to signal day ``t``
        (shift the per-instrument return series by -1 so day ``t`` carries
        the log-return realised from ``close_t`` to ``close_{t+1}``).
        ``'same_day'``: attach ``r_t`` (= ``log(close_t / close_{t-1})``).

    Returns
    -------
    AlignResult
        See :class:`AlignResult`.

    Raises
    ------
    ValueError
        If ``convention`` is not one of ``'next_day'`` or ``'same_day'``.
    """
    if convention not in {"next_day", "same_day"}:
        raise ValueError(f"convention must be 'next_day' or 'same_day', got {convention!r}")

    # --- 1. Extract signal series for this instrument -------------------------
    sig = signals_wide[["date", instrument]].rename(columns={instrument: "signal"}).copy()
    sig = sig.sort_values("date").reset_index(drop=True)
    n_signal_days = len(sig)

    # --- 2. Compute native returns for this instrument ------------------------
    inst_ohlcv = ohlcv_long[ohlcv_long["instrument"] == instrument].copy()
    inst_rets_long = native_returns(inst_ohlcv, kind="log")  # drops first row (NaN ret)

    # Build a date-indexed return series (per-instrument dense, no gaps filled)
    ret_series = inst_rets_long.set_index("date")["ret"].sort_index()

    # --- 3. Apply convention shift --------------------------------------------
    # 'next_day': shift(-1) moves r_{t+1} onto day t; after shifting, day t
    #             holds the return earned by holding from close_t to close_{t+1}.
    # 'same_day': no shift; day t holds r_t = log(close_t / close_{t-1}).
    if convention == "next_day":
        ret_series = ret_series.shift(-1)
        # shift(-1) introduces NaN at the last date; that date is excluded below

    # --- 4. Inner-join signal dates with dates that have a defined return -----
    # Merge on date; only keep rows where ret is non-null (no ffill/fillna).
    ret_df = ret_series.reset_index()  # columns: date, ret
    merged = sig.merge(ret_df, on="date", how="inner")
    merged = merged.dropna(subset=["ret"])  # drops convention-induced NaN at boundary
    merged = merged.sort_values("date").reset_index(drop=True)

    # Ensure output column order
    frame = merged[["date", "signal", "ret"]].copy()

    # --- 5. Compute summary statistics ----------------------------------------
    n_dropped = n_signal_days - len(frame)
    retained_fraction = len(frame) / n_signal_days

    return AlignResult(
        frame=frame,
        n_signal_days=n_signal_days,
        n_dropped=n_dropped,
        retained_fraction=retained_fraction,
    )


def align_panel(
    signals_wide: pd.DataFrame,
    ohlcv_long: pd.DataFrame,
    convention: str = "next_day",
    instruments: list[str] | None = None,
) -> dict[str, AlignResult]:
    """Align all (or a subset of) instruments.

    Parameters
    ----------
    signals_wide : pd.DataFrame
        Wide signals frame -- see :func:`align_instrument`.
    ohlcv_long : pd.DataFrame
        Long OHLCV frame -- see :func:`align_instrument`.
    convention : str
        Holding convention; forwarded to :func:`align_instrument`.
    instruments : list[str] or None
        Instruments to align.  Defaults to all columns in ``signals_wide``
        that are not ``'date'``.

    Returns
    -------
    dict[str, AlignResult]
        Mapping ``instrument -> AlignResult``.
    """
    if instruments is None:
        instruments = [c for c in signals_wide.columns if c != "date"]
    return {
        inst: align_instrument(signals_wide, ohlcv_long, inst, convention=convention)
        for inst in instruments
    }
