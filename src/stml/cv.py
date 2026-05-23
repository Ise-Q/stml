"""
cv.py
=====
Purged K-fold cross-validation, embargo, and expanding-window walk-forward
splitter for the meta-model. Implements AFML Ch. 7 (also covered in Lecture 1).

The standard sklearn ``KFold`` / ``TimeSeriesSplit`` are *unsafe* for triple-
barrier labels because:

  1. Labels are autocorrelated (the EDA showed lag-1 autocorrelation of 0.6-0.9
     for the primary signal) → random shuffling leaks the same regime into
     train and test.
  2. Labels have **overlapping information spans**: event ``i`` at time ``t_i``
     uses prices through its first-touch time ``t1_i``. If ``t1_i`` lies inside
     a test window for some test event ``j``, then event ``i``'s label was
     determined using information from inside ``j``'s test period — keeping
     ``i`` in train leaks.

The fix (AFML Ch. 7):

  - **Purging**: drop from training any event whose ``[t, t1]`` span overlaps
    the test block.
  - **Embargo**: in addition, drop training events whose ``t`` falls within a
    short buffer *after* the test block ends — to prevent serial-correlation
    leakage even after t1 has resolved.

Public API:

  - :class:`PurgedKFold`            -- sklearn-compatible CV splitter
  - :func:`walk_forward_splits`     -- generator of expanding-train / fixed-test splits
  - :class:`WalkForwardSplitter`    -- the same as a splitter class
  - :func:`assert_no_leakage`       -- defensive assertion used in tests
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 1. Purged K-fold                                                            #
# --------------------------------------------------------------------------- #
class PurgedKFold:
    """K-fold CV with purging on label-span overlaps and optional embargo.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds. Test blocks are *contiguous in event-time* (= the
        events sorted by ``t`` are divided into ``n_splits`` equal chunks).
    t : pd.Series
        Event start dates, indexed by event id. Required.
    t1 : pd.Series
        Event end dates (first-touch from triple-barrier), indexed identically
        to ``t``. Required.
    embargo_td : pd.Timedelta, optional
        Embargo as a timedelta after each test block. Default: ``None``
        (no embargo). Recommended: ``pd.Timedelta(days=h)`` where ``h`` is the
        labeling horizon, so the next training event begins after the longest
        possible test-label span has resolved.
    embargo_pct : float, default 0.0
        Embargo as a fraction of the total unique-date range. Used only if
        ``embargo_td`` is None. AFML's convention.

    Notes
    -----
    The splitter is sklearn-compatible: ``get_n_splits`` and ``split`` are
    provided. It can be passed to ``RandomizedSearchCV`` and friends.

    The ``X`` passed to :meth:`split` must have an index that matches
    ``self.t.index`` (event ids). Yields ``(train_pos, test_pos)`` where
    ``pos`` are integer positional indices into ``X`` after :meth:`_align_X`.
    """

    def __init__(
        self,
        n_splits: int = 5,
        t: pd.Series = None,
        t1: pd.Series = None,
        embargo_td: Optional[pd.Timedelta] = None,
        embargo_pct: float = 0.0,
    ):
        if t is None or t1 is None:
            raise ValueError("Both `t` and `t1` are required")
        if not t.index.equals(t1.index):
            raise ValueError("t and t1 must share the same index (event ids)")
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")

        self.n_splits = int(n_splits)
        self.t = pd.to_datetime(t).rename("t")
        self.t1 = pd.to_datetime(t1).rename("t1")
        self.embargo_td = embargo_td
        self.embargo_pct = float(embargo_pct)

    # sklearn interface
    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: ARG002
        return self.n_splits

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,  # noqa: ARG002
        groups: Optional[pd.Series] = None,  # noqa: ARG002
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` as positional integer arrays into X.

        ``X`` must be indexed by event id, with the same ids as ``self.t.index``.
        """
        if not X.index.isin(self.t.index).all():
            raise ValueError("X.index must be a subset of t.index (event ids)")

        # Restrict t/t1 to the events present in X, in X's order, then sort by t.
        t = self.t.reindex(X.index)
        t1 = self.t1.reindex(X.index)
        order = np.argsort(t.values, kind="stable")
        t_sorted = t.iloc[order]
        t1_sorted = t1.iloc[order]
        pos_sorted = np.arange(len(X))[order]  # original positions in X.index

        # Divide events into n_splits contiguous chunks (in event-time order).
        n = len(X)
        if n < self.n_splits:
            raise ValueError(f"n_samples ({n}) < n_splits ({self.n_splits})")
        # Roughly equal-sized chunks.
        chunk_bounds = np.linspace(0, n, self.n_splits + 1, dtype=int)

        embargo_td = self._resolve_embargo(t_sorted)

        for k in range(self.n_splits):
            lo, hi = chunk_bounds[k], chunk_bounds[k + 1]
            test_rows = pos_sorted[lo:hi]
            test_t = t_sorted.iloc[lo:hi]
            test_t1 = t1_sorted.iloc[lo:hi]
            if test_t.empty:
                continue
            test_start = test_t.min()
            test_end = test_t1.max()  # latest *end* of any test event
            embargo_end = test_end + embargo_td

            # Training events: span [t_i, t1_i] must NOT overlap [test_start, test_end];
            # AND t_i must be > embargo_end (if event starts after test block).
            # Equivalent: keep event i if  t1_i < test_start  OR  t_i > embargo_end.
            train_mask = (t1.values < test_start) | (t.values > embargo_end)
            train_rows = np.where(train_mask)[0]
            yield train_rows, test_rows

    # ------------------------------------------------------------------ #
    def _resolve_embargo(self, t_sorted: pd.Series) -> pd.Timedelta:
        """Resolve the effective embargo timedelta."""
        if self.embargo_td is not None:
            return pd.Timedelta(self.embargo_td)
        if self.embargo_pct > 0:
            span = t_sorted.iloc[-1] - t_sorted.iloc[0]
            return pd.Timedelta(seconds=int(span.total_seconds() * self.embargo_pct))
        return pd.Timedelta(0)


# --------------------------------------------------------------------------- #
# 2. Walk-forward                                                             #
# --------------------------------------------------------------------------- #
def walk_forward_splits(
    t: pd.Series,
    boundaries: list[pd.Timestamp],
    embargo_td: Optional[pd.Timedelta] = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, pd.Timestamp, pd.Timestamp]]:
    """Expanding-train / fixed-test walk-forward splits.

    Parameters
    ----------
    t : pd.Series
        Event start dates, indexed by event id.
    boundaries : list of pd.Timestamp
        Sorted ascending. Each consecutive pair (b_i, b_{i+1}) defines a test
        window ``[b_i, b_{i+1})``; training is ``t < b_i`` (with optional
        embargo before b_i).
    embargo_td : pd.Timedelta, optional
        Embargo *before* each test window — drop training events whose ``t``
        falls within ``embargo_td`` of ``b_i``. Default: ``None``.

    Yields
    ------
    train_pos, test_pos, test_start, test_end
        Positional integer indices into ``t.index`` for the train and test
        events, plus the test-window boundaries for traceability.
    """
    t = pd.to_datetime(t)
    boundaries = sorted(pd.to_datetime(boundaries))
    if len(boundaries) < 2:
        raise ValueError("boundaries must have at least 2 timestamps")

    embargo = pd.Timedelta(0) if embargo_td is None else pd.Timedelta(embargo_td)

    for b_lo, b_hi in zip(boundaries[:-1], boundaries[1:]):
        train_cutoff = b_lo - embargo
        train_mask = t.values < train_cutoff
        test_mask = (t.values >= b_lo) & (t.values < b_hi)
        train_pos = np.where(train_mask)[0]
        test_pos = np.where(test_mask)[0]
        yield train_pos, test_pos, b_lo, b_hi


class WalkForwardSplitter:
    """Expanding-train / fixed-test walk-forward as a class.

    Parameters
    ----------
    boundaries : list of pd.Timestamp
        Sorted ascending. Defines the test windows ``[b_i, b_{i+1})``.
    embargo_td : pd.Timedelta, optional
        Embargo BEFORE each test window.
    """

    def __init__(
        self,
        boundaries: list[pd.Timestamp],
        embargo_td: Optional[pd.Timedelta] = None,
    ):
        self.boundaries = sorted(pd.to_datetime(boundaries))
        self.embargo_td = embargo_td

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: ARG002
        return len(self.boundaries) - 1

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,  # noqa: ARG002
        groups: Optional[pd.Series] = None,  # noqa: ARG002
        t: Optional[pd.Series] = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if t is None:
            raise ValueError("WalkForwardSplitter.split needs explicit `t` series (event start dates).")
        if not t.index.equals(X.index):
            raise ValueError("t must share the same index as X")
        for train_pos, test_pos, _b_lo, _b_hi in walk_forward_splits(
            t, self.boundaries, embargo_td=self.embargo_td
        ):
            yield train_pos, test_pos


# --------------------------------------------------------------------------- #
# 3. Train/predict boundary helper                                            #
# --------------------------------------------------------------------------- #
def split_by_boundary(
    t: pd.Series,
    boundary: pd.Timestamp,
    embargo_td: Optional[pd.Timedelta] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Single train/predict split at a boundary date.

    The master pipeline trains on everything before ``boundary`` (less an
    optional embargo) and predicts events with ``t >= boundary``.

    Parameters
    ----------
    t : pd.Series
        Event start dates indexed by event id.
    boundary : pd.Timestamp
        For our submission ``boundary = 2022-01-01``. On rerun the grader sets
        it to ``2022-07-01``.
    embargo_td : pd.Timedelta, optional
        Embargo before the boundary (excludes training events too close to
        the predict window).

    Returns
    -------
    train_pos, predict_pos : np.ndarray
        Positional integer indices into ``t.index``.
    """
    t = pd.to_datetime(t)
    boundary = pd.to_datetime(boundary)
    embargo = pd.Timedelta(0) if embargo_td is None else pd.Timedelta(embargo_td)
    train_pos = np.where(t.values < (boundary - embargo))[0]
    predict_pos = np.where(t.values >= boundary)[0]
    return train_pos, predict_pos


# --------------------------------------------------------------------------- #
# 4. Defensive leakage check                                                  #
# --------------------------------------------------------------------------- #
def assert_no_leakage(
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    t: pd.Series,
    t1: pd.Series,
    embargo_td: Optional[pd.Timedelta] = None,
) -> None:
    """Assert that no training event's [t, t1] span overlaps any test event's
    [t, t1] span. Also assert the embargo gap is respected.

    Raises ``AssertionError`` with a descriptive message on the first violation.
    """
    if len(train_pos) == 0 or len(test_pos) == 0:
        return  # vacuously true; let the caller decide if this is a problem

    t_arr = pd.to_datetime(t.values)
    t1_arr = pd.to_datetime(t1.values)
    test_start = t_arr[test_pos].min()
    test_end = t1_arr[test_pos].max()
    embargo = pd.Timedelta(0) if embargo_td is None else pd.Timedelta(embargo_td)
    embargo_end = test_end + embargo

    # Overlap rule: an event [a, b] overlaps [c, d] iff a <= d AND b >= c.
    train_t = t_arr[train_pos]
    train_t1 = t1_arr[train_pos]
    overlap_mask = (train_t <= test_end) & (train_t1 >= test_start)
    if overlap_mask.any():
        bad = np.where(overlap_mask)[0][:3]
        raise AssertionError(
            f"Leakage: {overlap_mask.sum()} training events overlap the test span "
            f"[{test_start.date()} .. {test_end.date()}]. "
            f"First offenders (train_pos): {train_pos[bad].tolist()}, "
            f"their (t, t1): {list(zip(train_t[bad], train_t1[bad]))}"
        )

    # Embargo rule: post-test training events must start strictly after embargo_end.
    post_mask = train_t > test_end
    if embargo > pd.Timedelta(0) and post_mask.any():
        too_soon = train_t[post_mask] <= embargo_end
        if too_soon.any():
            n = int(too_soon.sum())
            raise AssertionError(
                f"Embargo violation: {n} post-test training events start within "
                f"embargo window (test_end={test_end.date()}, embargo={embargo})."
            )
