"""Leakage-safe cross-validation: purged k-fold, CPCV, nested CPCV.

Triple-barrier labels overlap in time, so a training label whose span [t0, t1]
reaches into a validation window leaks the future into the past. These splitters
**purge** every training label whose span intersects the test window and apply a
forward **embargo** of ⌈pct·T⌉ bars to block serial-correlation leakage
(López de Prado 2018, Ch.7 & 12; ``reports/apb/nlr-cw-v1.md`` §6).

- ``PurgedKFold`` — K contiguous test folds, purged + embargoed; minimum acceptable
  discipline and the tuning splitter.
- ``CombinatorialPurgedCV`` — N groups, k held out per split → C(N,k) splits
  (N=6, k=2 → 15), giving a *distribution* of OOS metrics for PBO/robustness.
- ``nested_cpcv`` — inner CPCV (tune) nested in outer CPCV (evaluate); removes
  hyperparameter-tuning leakage (Schnaubelt 2022).

``t1`` is a Series indexed by event-start with values = label-end (first-touch),
and must share the index of ``X``. Splitters yield integer-position arrays so they
plug into scikit-learn (``GridSearchCV(cv=PurgedKFold(...))``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import combinations

import numpy as np
import pandas as pd


def embargo_size(n: int, pct: float = 0.01) -> int:
    """Forward embargo length = ⌈pct·n⌉ bars (the published 1% default; not retuned)."""
    return int(math.ceil(pct * n))


def _instrument_date_axes(instruments, x_index: pd.Index) -> dict:
    """Map each instrument ticker → its own sorted, unique trading-date axis.

    The pooled panel interleaves instruments at duplicate timestamps; the embargo
    is in *trading days*, so it must be advanced on each instrument's OWN calendar.
    """
    inst = np.asarray(instruments)
    dates = np.asarray(x_index.values)
    return {tk: np.unique(dates[inst == tk]) for tk in pd.unique(inst)}


def _advance_trading_days(axis: np.ndarray, ref, days: int):
    """Date ``days`` trading days after ``ref`` on this instrument's own ``axis``.

    ``ref`` is located by the last axis date ≤ ``ref`` (robust if ``ref`` is a
    first-touch ``t1`` that is not itself in the axis); the result is clamped to
    the final available date. ``days == 0`` returns the located date (no window).
    """
    pos = int(np.searchsorted(axis, np.datetime64(ref), side="right")) - 1
    pos = max(pos, 0)
    return axis[min(pos + days, len(axis) - 1)]


def _contiguous_blocks(sorted_idx: np.ndarray) -> list[np.ndarray]:
    """Split a sorted integer array into runs of consecutive values."""
    if len(sorted_idx) == 0:
        return []
    cuts = np.where(np.diff(sorted_idx) != 1)[0] + 1
    return np.split(sorted_idx, cuts)


def _purge_train(
    indices: np.ndarray,
    test_idx: np.ndarray,
    t1: pd.Series,
    x_index: pd.Index,
    embargo: int,
    *,
    instruments=None,
    embargo_days: Mapping[str, int] | None = None,
    date_axes: dict | None = None,
) -> np.ndarray:
    """Training indices with all test samples and test-overlapping labels removed.

    For each contiguous block of the test set, purge training labels whose span
    [start, end] intersects the test window. The forward **embargo** is applied two
    ways:

    - ``embargo_days is None`` (legacy/uniform): one ⌈pct·T⌉-*position* buffer past
      the block end, instrument-agnostic — the published 1% default path, unchanged.
    - ``embargo_days`` given (S2.6): the instrument-agnostic overlap purge, plus a
      **per-instrument** forward embargo referenced to each instrument's OWN max
      test-label-end and advanced ``embargo_days[i]`` trading days on that
      instrument's own date axis. Cross-instrument leakage stays covered by the
      overlap purge; same-instrument serial-correlation leakage by the embargo.
    """
    starts = t1.index
    ends = t1.to_numpy()
    start_vals = starts.values
    keep = ~np.isin(indices, test_idx)
    inst_arr = None if instruments is None else np.asarray(instruments)

    for block in _contiguous_blocks(np.sort(test_idx)):
        b_t0 = starts[block[0]]
        b_t1 = ends[block].max()
        if embargo_days is None:
            end_pos = int(x_index.searchsorted(b_t1))
            emb_pos = min(end_pos + embargo, len(x_index) - 1)
            emb_t = x_index[emb_pos]
            overlaps = (starts <= emb_t) & (ends >= b_t0)  # interval intersection
            keep &= ~np.asarray(overlaps)
            continue
        # per-instrument embargo path
        overlaps = (starts <= b_t1) & (ends >= b_t0)
        keep &= ~np.asarray(overlaps)
        block_inst = inst_arr[block]
        block_ends = ends[block]
        for tk in np.unique(block_inst):
            e_i = block_ends[block_inst == tk].max()
            emb_end = _advance_trading_days(date_axes[tk], e_i, int(embargo_days.get(tk, 0)))
            emb_mask = (inst_arr == tk) & (start_vals > np.datetime64(e_i)) & (
                start_vals <= emb_end
            )
            keep &= ~emb_mask
    return indices[keep]


def _check_aligned(x: pd.DataFrame | pd.Series, t1: pd.Series) -> None:
    if len(x) != len(t1) or not x.index.equals(t1.index):
        raise ValueError("X and t1 must share the same index (event order).")


def _require_instruments(instruments, embargo_days) -> None:
    if embargo_days is not None and instruments is None:
        raise ValueError("per-instrument embargo_days requires an aligned `instruments` Series")


def _axes_for(instruments, embargo_days, x_index: pd.Index) -> dict | None:
    """Per-instrument date axes when the per-instrument embargo is active, else ``None``."""
    return None if embargo_days is None else _instrument_date_axes(instruments, x_index)


class PurgedKFold:
    """K-fold over contiguous time blocks with purging + embargo.

    Pass ``embargo_days`` (a ticker→trading-days map) with an aligned ``instruments``
    Series to use the per-instrument embargo (S2.6); omit both for the uniform path.
    """

    def __init__(
        self,
        n_splits: int,
        t1: pd.Series,
        pct_embargo: float = 0.01,
        *,
        instruments=None,
        embargo_days: Mapping[str, int] | None = None,
    ) -> None:
        _require_instruments(instruments, embargo_days)
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo
        self.instruments = instruments
        self.embargo_days = embargo_days

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: N803 (sklearn API)
        return self.n_splits

    def split(self, X, y=None, groups=None):  # noqa: N803 (sklearn API)
        _check_aligned(X, self.t1)
        indices = np.arange(len(X))
        embargo = embargo_size(len(X), self.pct_embargo)
        axes = _axes_for(self.instruments, self.embargo_days, X.index)
        for test in np.array_split(indices, self.n_splits):
            train = _purge_train(
                indices, test, self.t1, X.index, embargo,
                instruments=self.instruments, embargo_days=self.embargo_days, date_axes=axes,
            )
            yield train, test


class CombinatorialPurgedCV:
    """Combinatorial purged CV: C(n_groups, n_test_groups) purged splits."""

    def __init__(
        self,
        n_groups: int,
        n_test_groups: int,
        t1: pd.Series,
        pct_embargo: float = 0.01,
        *,
        instruments=None,
        embargo_days: Mapping[str, int] | None = None,
    ) -> None:
        _require_instruments(instruments, embargo_days)
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.t1 = t1
        self.pct_embargo = pct_embargo
        self.instruments = instruments
        self.embargo_days = embargo_days

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: N803
        return math.comb(self.n_groups, self.n_test_groups)

    def split(self, X, y=None, groups=None):  # noqa: N803
        _check_aligned(X, self.t1)
        indices = np.arange(len(X))
        groups_idx = np.array_split(indices, self.n_groups)
        embargo = embargo_size(len(X), self.pct_embargo)
        axes = _axes_for(self.instruments, self.embargo_days, X.index)
        for combo in combinations(range(self.n_groups), self.n_test_groups):
            test = np.sort(np.concatenate([groups_idx[g] for g in combo]))
            train = _purge_train(
                indices, test, self.t1, X.index, embargo,
                instruments=self.instruments, embargo_days=self.embargo_days, date_axes=axes,
            )
            yield train, test


def nested_cpcv(
    X,  # noqa: N803
    t1: pd.Series,
    *,
    outer_groups: int = 6,
    outer_test_groups: int = 2,
    inner_groups: int = 5,
    inner_test_groups: int = 1,
    pct_embargo: float = 0.01,
    instruments=None,
    embargo_days: Mapping[str, int] | None = None,
):
    """Yield ``(outer_train, outer_test, inner_cv)`` for nested CPCV.

    Tune hyperparameters with ``inner_cv`` over ``X.iloc[outer_train]`` (its splits
    index into that subset), then evaluate the chosen model on ``outer_test``. The
    inner folds are built only from the outer-train rows, so they never touch the
    outer test fold — no tuning leakage. The per-instrument embargo (S2.6) is
    threaded through and the ``instruments`` Series is subset to the inner rows.
    """
    _require_instruments(instruments, embargo_days)
    outer = CombinatorialPurgedCV(
        outer_groups, outer_test_groups, t1, pct_embargo,
        instruments=instruments, embargo_days=embargo_days,
    )
    for outer_train, outer_test in outer.split(X):
        t1_tr = t1.iloc[outer_train]
        inst_tr = None if instruments is None else instruments.iloc[outer_train]
        inner = CombinatorialPurgedCV(
            inner_groups, inner_test_groups, t1_tr, pct_embargo,
            instruments=inst_tr, embargo_days=embargo_days,
        )
        yield outer_train, outer_test, inner
