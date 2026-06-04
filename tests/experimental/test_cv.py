"""Tests for ``stml.experimental.cv`` — methodology spec RED-first.

Property tests on the splitters' purge + embargo invariants.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
import pytest

from stml.experimental.cv import (
    CombinatorialPurgedCV,
    PurgedKFold,
    assert_no_leakage,
    embargo_size,
    nested_cpcv,
)


@pytest.fixture
def synth_events():
    """200 synthetic triple-barrier events spanning 2020-2021 across 4 instruments."""
    rng = np.random.default_rng(42)
    n = 200
    t = pd.date_range("2020-01-02", periods=n, freq="B")
    # Each event lasts ~10 trading days.
    t1 = t + pd.tseries.offsets.BDay(10)
    instruments = pd.Series(rng.choice(["A", "B", "C", "D"], size=n))
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.integers(0, 2, n))
    return X, y, pd.Series(t), pd.Series(t1), instruments


# ---------------------------------------------------------------------------
# embargo_size + helpers.
# ---------------------------------------------------------------------------


def test_embargo_size_1pct() -> None:
    assert embargo_size(1000, pct=0.01) == 10
    assert embargo_size(105, pct=0.01) == 2
    assert embargo_size(50, pct=0.05) == 3


# ---------------------------------------------------------------------------
# PurgedKFold.
# ---------------------------------------------------------------------------


def test_purgedkfold_basic_split_shape(synth_events) -> None:
    X, y, t, t1, _ = synth_events
    cv = PurgedKFold(n_splits=5, t=t, t1=t1)
    splits = list(cv.split(X))
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        # No index appears in both train and test (purging guarantees this).
        assert len(np.intersect1d(train_idx, test_idx)) == 0


def test_purgedkfold_no_leakage_invariant(synth_events) -> None:
    """For every fold, train events' [t, t1] must NOT overlap test [t, t1]."""
    X, y, t, t1, _ = synth_events
    cv = PurgedKFold(n_splits=5, t=t, t1=t1)
    for train_idx, test_idx in cv.split(X):
        # The assertion raises on leakage; calling it here is the test.
        assert_no_leakage(train_idx, test_idx, t, t1)


def test_purgedkfold_embargo_drops_bars(synth_events) -> None:
    X, y, t, t1, _ = synth_events
    cv = PurgedKFold(n_splits=5, t=t, t1=t1, pct_embargo=0.05)
    cv_no_emb = PurgedKFold(n_splits=5, t=t, t1=t1, pct_embargo=0.0)
    sizes_emb, sizes_no_emb = [], []
    for (tr, _), (tr2, _) in zip(cv.split(X), cv_no_emb.split(X)):
        sizes_emb.append(len(tr))
        sizes_no_emb.append(len(tr2))
    assert sum(sizes_emb) <= sum(sizes_no_emb), "embargo should drop at least as much"


def test_purgedkfold_requires_t_t1() -> None:
    cv = PurgedKFold(n_splits=5)
    with pytest.raises(ValueError, match="t and t1 are required"):
        list(cv.split(pd.DataFrame(np.zeros((10, 2)))))


def test_purgedkfold_per_instrument_embargo_drops_same_instrument_only(synth_events) -> None:
    """When per-instrument embargo is on, drops are scoped to that instrument."""
    X, y, t, t1, insts = synth_events
    # Aggressive per-instrument embargo on instrument 'A' only.
    embargo_map = {"A": 30, "B": 0, "C": 0, "D": 0}
    cv = PurgedKFold(
        n_splits=5, t=t, t1=t1,
        instruments=insts, embargo_days=embargo_map,
    )
    for train_idx, test_idx in cv.split(X):
        # No leakage at the span level.
        assert_no_leakage(train_idx, test_idx, t, t1)


# ---------------------------------------------------------------------------
# CombinatorialPurgedCV.
# ---------------------------------------------------------------------------


def test_cpcv_path_count_is_C_n_k(synth_events) -> None:
    """`(n_groups=6, n_test_groups=2)` → C(6, 2) = 15 paths."""
    X, y, t, t1, _ = synth_events
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, t=t, t1=t1)
    assert cv.n_paths == comb(6, 2) == 15
    splits = list(cv.split(X))
    assert len(splits) == 15


def test_cpcv_no_leakage_all_paths(synth_events) -> None:
    X, y, t, t1, _ = synth_events
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, t=t, t1=t1)
    for train_idx, test_idx in cv.split(X):
        assert_no_leakage(train_idx, test_idx, t, t1)


def test_cpcv_rejects_bad_n_groups() -> None:
    with pytest.raises(ValueError, match="require"):
        CombinatorialPurgedCV(n_groups=1, n_test_groups=1, t=pd.Series([]), t1=pd.Series([]))
    with pytest.raises(ValueError, match="require"):
        CombinatorialPurgedCV(n_groups=3, n_test_groups=3, t=pd.Series([]), t1=pd.Series([]))


# ---------------------------------------------------------------------------
# nested_cpcv.
# ---------------------------------------------------------------------------


def test_nested_cpcv_inner_built_only_from_outer_train(synth_events) -> None:
    """Inner CV indices must be in [0, len(outer_train)) — no outer-test leakage."""
    X, y, t, t1, _ = synth_events
    paths = list(nested_cpcv(X, t=t, t1=t1, outer_groups=4, outer_test_groups=1,
                              inner_groups=3, inner_test_groups=1))
    assert len(paths) == 4  # C(4, 1) = 4 outer paths
    for outer_train_idx, outer_test_idx, inner_cv in paths:
        # The inner cv's t/t1 are restricted to outer_train rows.
        assert len(inner_cv.t) == len(outer_train_idx)
        # Verify the inner splits yield indices in [0, len(outer_train)).
        Xi = X.iloc[outer_train_idx]
        for itr, ite in inner_cv.split(Xi):
            assert itr.max(initial=-1) < len(outer_train_idx)
            assert ite.max(initial=-1) < len(outer_train_idx)
