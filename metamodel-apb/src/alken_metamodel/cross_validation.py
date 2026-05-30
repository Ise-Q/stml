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
from itertools import combinations

import numpy as np
import pandas as pd


def embargo_size(n: int, pct: float = 0.01) -> int:
    """Forward embargo length = ⌈pct·n⌉ bars (the published 1% default; not retuned)."""
    return int(math.ceil(pct * n))


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
) -> np.ndarray:
    """Training indices with all test samples and test-overlapping labels removed.

    For each contiguous block of the test set, purge training labels whose span
    [start, end] intersects [block_start, block_end + embargo bars].
    """
    starts = t1.index
    ends = t1.to_numpy()
    keep = ~np.isin(indices, test_idx)

    for block in _contiguous_blocks(np.sort(test_idx)):
        b_t0 = starts[block[0]]
        b_t1 = ends[block].max()
        end_pos = int(x_index.searchsorted(b_t1))
        emb_pos = min(end_pos + embargo, len(x_index) - 1)
        emb_t = x_index[emb_pos]
        overlaps = (starts <= emb_t) & (ends >= b_t0)  # interval intersection
        keep &= ~np.asarray(overlaps)
    return indices[keep]


def _check_aligned(x: pd.DataFrame | pd.Series, t1: pd.Series) -> None:
    if len(x) != len(t1) or not x.index.equals(t1.index):
        raise ValueError("X and t1 must share the same index (event order).")


class PurgedKFold:
    """K-fold over contiguous time blocks with purging + embargo."""

    def __init__(self, n_splits: int, t1: pd.Series, pct_embargo: float = 0.01) -> None:
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: N803 (sklearn API)
        return self.n_splits

    def split(self, X, y=None, groups=None):  # noqa: N803 (sklearn API)
        _check_aligned(X, self.t1)
        indices = np.arange(len(X))
        embargo = embargo_size(len(X), self.pct_embargo)
        for test in np.array_split(indices, self.n_splits):
            train = _purge_train(indices, test, self.t1, X.index, embargo)
            yield train, test


class CombinatorialPurgedCV:
    """Combinatorial purged CV: C(n_groups, n_test_groups) purged splits."""

    def __init__(
        self,
        n_groups: int,
        n_test_groups: int,
        t1: pd.Series,
        pct_embargo: float = 0.01,
    ) -> None:
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: N803
        return math.comb(self.n_groups, self.n_test_groups)

    def split(self, X, y=None, groups=None):  # noqa: N803
        _check_aligned(X, self.t1)
        indices = np.arange(len(X))
        groups_idx = np.array_split(indices, self.n_groups)
        embargo = embargo_size(len(X), self.pct_embargo)
        for combo in combinations(range(self.n_groups), self.n_test_groups):
            test = np.sort(np.concatenate([groups_idx[g] for g in combo]))
            train = _purge_train(indices, test, self.t1, X.index, embargo)
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
):
    """Yield ``(outer_train, outer_test, inner_cv)`` for nested CPCV.

    Tune hyperparameters with ``inner_cv`` over ``X.iloc[outer_train]`` (its splits
    index into that subset), then evaluate the chosen model on ``outer_test``. The
    inner folds are built only from the outer-train rows, so they never touch the
    outer test fold — no tuning leakage.
    """
    outer = CombinatorialPurgedCV(outer_groups, outer_test_groups, t1, pct_embargo)
    for outer_train, outer_test in outer.split(X):
        t1_tr = t1.iloc[outer_train]
        inner = CombinatorialPurgedCV(inner_groups, inner_test_groups, t1_tr, pct_embargo)
        yield outer_train, outer_test, inner
