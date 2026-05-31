"""Per-instrument embargo (S2.6, RED-first).

The pooled CV panel interleaves instruments at duplicate timestamps, so the old
``⌈pct·len(X)⌉``-*position* embargo covers only ~embargo/K days per instrument.
``instrument_scope.json`` specifies the embargo in *trading days* (``embargo_p90``,
e.g. ng1s=33, ho1s=26, cl1s=14), varying within a fold. These tests assert that
each instrument's forward-purged window equals its own ``embargo_p90`` measured on
its own trading-day axis, while the instrument-agnostic overlap purge still holds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alken_metamodel.cross_validation import (
    PurgedKFold,
    _instrument_date_axes,
    _purge_train,
)


def _intersects(a0, a1, b0, b1) -> bool:
    return a0 <= b1 and b0 <= a1


def _panel(dates_per_inst: dict[str, int], span: int = 0):
    """Pooled multi-instrument panel sorted by (date, instrument).

    Each instrument trades every business day in its range; label end = start +
    ``span`` business days (span=0 isolates the embargo from the overlap purge).
    Returns (X, t1, instruments) sharing one DatetimeIndex.
    """
    frames = []
    for tk, n in dates_per_inst.items():
        idx = pd.bdate_range("2020-01-01", periods=n)
        end_pos = np.minimum(np.arange(n) + span, n - 1)
        frames.append(pd.DataFrame({"date": idx, "instrument": tk, "t1": idx[end_pos]}))
    df = pd.concat(frames, ignore_index=True).sort_values(["date", "instrument"])
    df = df.reset_index(drop=True)
    index = pd.DatetimeIndex(df["date"])
    x = pd.DataFrame({"f": np.arange(len(df), dtype=float)}, index=index)
    t1 = pd.Series(pd.DatetimeIndex(df["t1"]).to_numpy(), index=index)
    instruments = pd.Series(df["instrument"].to_numpy(), index=index)
    return x, t1, instruments


def test_per_instrument_embargo_purges_each_instruments_own_window():
    """A's 2-day and B's 5-day embargo windows are purged on each one's own axis."""
    x, t1, inst = _panel({"A": 10, "B": 10}, span=0)
    dates = pd.bdate_range("2020-01-01", periods=10)
    d3 = dates[3]
    test_idx = np.where(t1.index == d3)[0]  # the A@d3 and B@d3 events
    axes = _instrument_date_axes(inst, x.index)

    keep = set(
        _purge_train(
            np.arange(len(x)),
            test_idx,
            t1,
            x.index,
            0,
            instruments=inst,
            embargo_days={"A": 2, "B": 5},
            date_axes=axes,
        )
    )

    def pos(tk: str, k: int) -> int:
        return int(np.where((t1.index == dates[k]) & (inst.to_numpy() == tk))[0][0])

    # A: embargo window (d3, d5] purged; d6 kept
    assert pos("A", 4) not in keep
    assert pos("A", 5) not in keep
    assert pos("A", 6) in keep
    # B: embargo window (d3, d8] purged; d9 kept
    for k in range(4, 9):
        assert pos("B", k) not in keep
    assert pos("B", 9) in keep
    # pre-test events kept (span-0 → no overlap)
    assert pos("A", 0) in keep
    assert pos("B", 2) in keep


def test_per_instrument_embargo_preserves_zero_overlap_invariant():
    """With per-instrument embargo active, no train span intersects a test block."""
    x, t1, inst = _panel({"A": 24, "B": 24}, span=3)
    cv = PurgedKFold(
        n_splits=3,
        t1=t1,
        pct_embargo=0.0,
        instruments=inst,
        embargo_days={"A": 2, "B": 5},
    )
    n = 0
    for train, test in cv.split(x):
        n += 1
        assert set(train).isdisjoint(set(test))
        ts = np.sort(test)
        for block in np.split(ts, np.where(np.diff(ts) != 1)[0] + 1):
            b0 = t1.index[block[0]]
            b1 = t1.iloc[block].max()
            for tr in train:
                assert not _intersects(t1.index[tr], t1.iloc[tr], b0, b1)
    assert n == 3


def test_larger_embargo_purges_at_least_as_much():
    """A strictly larger per-instrument embargo never keeps more training rows."""
    x, t1, inst = _panel({"A": 20, "B": 20}, span=2)
    axes = _instrument_date_axes(inst, x.index)
    test_idx = np.arange(18, 24)  # one contiguous interior block
    small = set(
        _purge_train(np.arange(len(x)), test_idx, t1, x.index, 0,
                     instruments=inst, embargo_days={"A": 1, "B": 1}, date_axes=axes)
    )
    large = set(
        _purge_train(np.arange(len(x)), test_idx, t1, x.index, 0,
                     instruments=inst, embargo_days={"A": 8, "B": 8}, date_axes=axes)
    )
    assert large.issubset(small)
