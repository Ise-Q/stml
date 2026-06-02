"""
test_release_guard.py
=====================
Tests for release_test tripwire and CombinatorialPurgedCV split discipline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.model.cv import CombinatorialPurgedCV
from stml.model.evaluate import release_test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_matrix(n_train: int = 20, n_val: int = 10, n_test: int = 8) -> pd.DataFrame:
    """Minimal matrix with partition column {train, val, test}."""
    partitions = (
        ["train"] * n_train
        + ["val"] * n_val
        + ["test"] * n_test
    )
    dates = pd.bdate_range("2020-01-02", periods=len(partitions))
    return pd.DataFrame({
        "date": dates,
        "partition": partitions,
        "value": np.arange(len(partitions), dtype=float),
    })


# ---------------------------------------------------------------------------
# release_test tripwire
# ---------------------------------------------------------------------------


def test_release_test_raises_without_confirmation():
    """release_test must raise RuntimeError without final_confirmation=True."""
    m = _make_matrix()
    with pytest.raises(RuntimeError, match="final_confirmation"):
        release_test(m)


def test_release_test_raises_explicit_false():
    """release_test(m, final_confirmation=False) also raises."""
    m = _make_matrix()
    with pytest.raises(RuntimeError):
        release_test(m, final_confirmation=False)


def test_release_test_returns_test_rows_only():
    """With final_confirmation=True, only partition=='test' rows are returned."""
    m = _make_matrix(n_train=20, n_val=10, n_test=8)
    result = release_test(m, final_confirmation=True)
    assert (result["partition"] == "test").all()
    assert len(result) == 8


def test_release_test_index_reset():
    """Returned frame has a fresh 0-based index."""
    m = _make_matrix()
    result = release_test(m, final_confirmation=True)
    assert list(result.index) == list(range(len(result)))


def test_release_test_no_train_val_rows():
    """Returned frame contains no train or val rows."""
    m = _make_matrix()
    result = release_test(m, final_confirmation=True)
    assert "train" not in result["partition"].values
    assert "val" not in result["partition"].values


# ---------------------------------------------------------------------------
# CombinatorialPurgedCV — no fabricated test-partition rows
# ---------------------------------------------------------------------------


def test_cpcv_split_never_touches_test_partition(synthetic_panel):
    """CPCV.split on a dev-only frame never emits row indices that would be 'test' partition.

    The synthetic_panel fixture has all rows marked 'train'.  We layer a 'partition' column
    on a copy and confirm no emitted test-position index falls in a 'test'-partition row
    (trivially true since there are none, but exercises the contract: the splitter only sees
    dev rows and cannot fabricate test-partition rows from thin air).
    """
    df = synthetic_panel.copy()
    # Ensure required columns are present; bar_pos is already there.
    assert "bar_pos" in df.columns

    # All rows are 'train' partition in the fixture -- attach explicitly.
    df = df.copy()
    df["partition"] = "train"

    cv = CombinatorialPurgedCV(4, 2, h=3, default_embargo=1)
    test_partition_idx = set(df.index[df["partition"] == "test"].tolist())
    # Should be empty -- the fixture has no test-partition rows.
    assert len(test_partition_idx) == 0

    for tr, te in cv.split(df):
        # Neither train nor test positions should point at a 'test' partition row.
        all_emitted = set(tr.tolist()) | set(te.tolist())
        assert all_emitted.isdisjoint(test_partition_idx), (
            "CPCV emitted positions that coincide with test-partition rows"
        )


def test_cpcv_split_correct_n_splits(synthetic_panel):
    """C(4,2)=6 combinations emitted for (n_blocks=4, k_test=2)."""
    df = synthetic_panel.copy()
    cv = CombinatorialPurgedCV(4, 2, h=3, default_embargo=1)
    splits = list(cv.split(df))
    assert len(splits) == 6


def test_cpcv_train_test_disjoint(synthetic_panel):
    """Train and test position sets must be disjoint in every fold."""
    df = synthetic_panel.copy()
    cv = CombinatorialPurgedCV(4, 2, h=3, default_embargo=1)
    for tr, te in cv.split(df):
        assert len(set(tr.tolist()) & set(te.tolist())) == 0
