"""Tests for ``stml.experimental.data_loader`` — plan §8 Stage 1 deliverable.

Verifies the per-instrument framing of OHLCV + signal data against the released
CSVs (so this test depends on ``data/ohlcv_data.csv`` + ``data/primary_signals.csv``
being available — they are committed in the shared spine).
"""

from __future__ import annotations

import pandas as pd

from stml.experimental.config import INSTRUMENTS
from stml.experimental.data_loader import (
    OHLCV_COLUMNS,
    load_panel,
    per_instrument_frames,
    trading_day_index,
    trading_day_offset,
)


def test_load_panel_returns_long_ohlcv_and_wide_signals(repo_root: object) -> None:  # noqa: ARG001
    """Smoke-test ``load_panel`` on the real released data."""
    ohlcv, signals = load_panel()
    assert isinstance(ohlcv, pd.DataFrame)
    assert isinstance(signals, pd.DataFrame)
    # Long OHLCV has the required columns + the index keys.
    required = {"date", "instrument", *OHLCV_COLUMNS}
    assert required.issubset(ohlcv.columns)
    # Wide signals: 'date' + 11 instrument columns.
    assert "date" in signals.columns
    for inst in INSTRUMENTS:
        assert inst in signals.columns, f"missing signal column {inst}"


def test_per_instrument_frames_covers_universe(repo_root: object) -> None:  # noqa: ARG001
    """Every instrument in ``config.INSTRUMENTS`` has a frame with the right columns."""
    ohlcv, signals = load_panel()
    panel = per_instrument_frames(ohlcv, signals)
    assert set(panel.keys()) == set(INSTRUMENTS), (
        f"missing/extra instruments: {set(INSTRUMENTS) - panel.keys()} / "
        f"{panel.keys() - set(INSTRUMENTS)}"
    )
    for inst, frame in panel.items():
        # Required OHLCV columns + signal column.
        for col in [*OHLCV_COLUMNS, "signal"]:
            assert col in frame.columns, f"{inst} missing column {col}"
        # DatetimeIndex, sorted, no duplicates.
        assert isinstance(frame.index, pd.DatetimeIndex)
        assert frame.index.is_monotonic_increasing
        assert frame.index.is_unique


def test_trading_day_offset_advances_on_instrument_calendar() -> None:
    """``trading_day_offset(close, t, +5)`` returns the 5th trading day AFTER ``t``."""
    import numpy as np
    n = 20
    idx = pd.bdate_range("2020-01-02", periods=n)
    close = pd.Series(100.0 + 0.1 * np.arange(n), index=idx)
    t = idx[7]
    assert trading_day_offset(close, t, 0) == t
    assert trading_day_offset(close, t, +1) == idx[8]
    assert trading_day_offset(close, t, +5) == idx[12]
    # Going past the end returns None.
    assert trading_day_offset(close, idx[-1], +1) is None
    assert trading_day_offset(close, idx[0], -1) is None


def test_trading_day_index_drops_nan_bars() -> None:
    """Bars with NaN close are not part of the instrument's dense trading-day index."""
    idx = pd.bdate_range("2020-01-02", periods=10)
    close = pd.Series([100.0, float("nan"), 102.0, 103.0, 104.0,
                       105.0, 106.0, 107.0, 108.0, 109.0], index=idx)
    tdi = trading_day_index(close)
    assert len(tdi) == 9  # one NaN dropped
    assert idx[1] not in tdi
