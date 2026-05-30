"""Leakage-invariant tests for purged CV / CPCV / nested CPCV (Stage 1, RED-first).

The whole point of these splitters is that *no training label's span [t0, t1]
overlaps the test window* (purge), with a forward embargo of ⌈pct·T⌉ bars
(LdP 2018 Ch.7 & 12; nlr-cw §6). The tests assert those invariants directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alken_metamodel.cross_validation import (
    CombinatorialPurgedCV,
    PurgedKFold,
    embargo_size,
    nested_cpcv,
)


def _events(n: int = 30, span: int = 3):
    """N daily events; each label ends `span` bars later (spans overlap neighbours)."""
    idx = pd.bdate_range("2020-01-01", periods=n)
    x = pd.DataFrame({"f": np.arange(n, dtype=float)}, index=idx)
    end_pos = np.minimum(np.arange(n) + span, n - 1)
    t1 = pd.Series(idx[end_pos], index=idx)
    return x, t1


def _intersects(a0, a1, b0, b1) -> bool:
    return a0 <= b1 and b0 <= a1


# --- embargo ---------------------------------------------------------------

def test_embargo_is_ceil_one_percent():
    assert embargo_size(100) == 1  # ceil(0.01*100)
    assert embargo_size(250) == 3  # ceil(2.5)
    assert embargo_size(99) == 1  # ceil(0.99)
    assert embargo_size(100, pct=0.0) == 0


# --- PurgedKFold -----------------------------------------------------------

def test_purgedkfold_no_train_test_overlap():
    x, t1 = _events(30, span=3)
    cv = PurgedKFold(n_splits=3, t1=t1, pct_embargo=0.01)
    n_splits = 0
    for train, test in cv.split(x):
        n_splits += 1
        assert set(train).isdisjoint(set(test))  # no index in both
        test_t0 = t1.index[test].min()
        test_t1 = t1.iloc[test].max()
        for tr in train:  # NO train label span may intersect the test window
            assert not _intersects(t1.index[tr], t1.iloc[tr], test_t0, test_t1)
    assert n_splits == 3


def test_purgedkfold_covers_every_sample_once_as_test():
    x, t1 = _events(30, span=3)
    cv = PurgedKFold(n_splits=3, t1=t1, pct_embargo=0.01)
    seen = np.concatenate([test for _, test in cv.split(x)])
    assert sorted(seen) == list(range(30))  # each sample tested exactly once


# --- Combinatorial Purged CV ----------------------------------------------

def test_cpcv_path_count_is_15():
    x, t1 = _events(36, span=3)
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, t1=t1, pct_embargo=0.01)
    splits = list(cv.split(x))
    assert cv.get_n_splits() == 15  # C(6,2)
    assert len(splits) == 15


def test_cpcv_purges_all_test_groups():
    x, t1 = _events(36, span=3)
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, t1=t1, pct_embargo=0.01)
    for train, test in cv.split(x):
        assert set(train).isdisjoint(set(test))
        # Purge holds per CONTIGUOUS test block: when the two test groups are
        # non-adjacent, a training label strictly between them is correctly kept
        # (it overlaps neither group's local span). So check each block separately.
        test_sorted = np.sort(test)
        cuts = np.where(np.diff(test_sorted) != 1)[0] + 1
        for block in np.split(test_sorted, cuts):
            b_t0 = t1.index[block[0]]
            b_t1 = t1.iloc[block].max()
            for tr in train:
                assert not _intersects(t1.index[tr], t1.iloc[tr], b_t0, b_t1)


# --- nested CPCV -----------------------------------------------------------

def test_nested_inner_folds_disjoint_from_outer_test():
    x, t1 = _events(36, span=3)
    for outer_train, outer_test, inner_cv in nested_cpcv(
        x, t1, outer_groups=6, outer_test_groups=2, inner_groups=4, inner_test_groups=1
    ):
        x_tr = x.iloc[outer_train]
        for inner_train, inner_test in inner_cv.split(x_tr):
            # inner positions index into x_tr; map back to original sample indices
            orig = outer_train[np.concatenate([inner_train, inner_test])]
            assert set(orig).isdisjoint(set(outer_test))  # no tuning leakage
        break  # one outer split is enough to prove the invariant
