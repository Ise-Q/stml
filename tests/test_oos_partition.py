"""Tests for the OOS-stable date-pinned split and partition labelling.

These guard the property the final-submission feature-regeneration path depends
on: extending the released signal axis with a hidden Jul-Dec 2022 block must NOT
move the FE-train boundary or re-bucket already-released rows, and the new rows
must be tagged ``"oos"`` (never ``train``/``val``/``test``).

Ground truth is the real released panel (645 trading days,
2020-01-03..2022-06-30). The historical fractional cut is train[0:387],
val[387:516], test[516:645]; the date-pinned split must reproduce it exactly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stml.io import load_clean_data
from stml.metamodel.pipeline import FeaturePipeline
from stml.metamodel.splits import chronological_split, fixed_date_split

FE_TRAIN_END = pd.Timestamp("2021-07-01")
VAL_END = pd.Timestamp("2021-12-30")
TEST_END = pd.Timestamp("2022-06-30")


@pytest.fixture(scope="module")
def released_dates() -> pd.DatetimeIndex:
    _, signals = load_clean_data()
    return pd.DatetimeIndex(signals["date"].sort_values().to_numpy())


def test_fixed_matches_chronological_on_released(released_dates):
    """On the released window the date-pinned split == the (0.6,0.2,0.2) cut."""
    old = chronological_split(released_dates)
    new = fixed_date_split(released_dates)
    assert new.train_dates.equals(old.train_dates)
    assert new.val_dates.equals(old.val_dates)
    assert new.test_dates.equals(old.test_dates)
    assert (len(new.train_dates), len(new.val_dates), len(new.test_dates)) == (
        387,
        129,
        129,
    )


def test_fixed_split_invariant_under_extension(released_dates):
    """Appending a hidden H2-2022 block leaves train/val/test blocks unchanged."""
    base = fixed_date_split(released_dates)
    # Synthetic Jul-Dec 2022 extension (business days strictly after test_end).
    oos = pd.bdate_range("2022-07-01", "2022-12-30")
    extended = released_dates.append(pd.DatetimeIndex(oos))
    ext = fixed_date_split(extended)

    # The three released blocks are byte-identical -- no fractional drift.
    assert ext.train_dates.equals(base.train_dates)
    assert ext.val_dates.equals(base.val_dates)
    assert ext.test_dates.equals(base.test_dates)
    # The OOS dates fall in NONE of the three blocks.
    assigned = set(ext.train_dates) | set(ext.val_dates) | set(ext.test_dates)
    assert not (set(oos) & assigned)


def test_partition_for_tags_oos():
    """_partition_for buckets each region by date, with >test_end -> 'oos'."""
    pipe = FeaturePipeline()
    idx = pd.DatetimeIndex(
        [
            "2020-01-03",  # train
            FE_TRAIN_END,  # train (inclusive upper edge)
            "2021-07-02",  # val
            VAL_END,  # val (inclusive upper edge)
            "2021-12-31",  # test
            TEST_END,  # test (inclusive upper edge)
            "2022-07-01",  # oos
            "2022-12-30",  # oos
        ]
    )
    labels = pipe._partition_for(idx)
    assert list(labels) == [
        "train",
        "train",
        "val",
        "val",
        "test",
        "test",
        "oos",
        "oos",
    ]


def test_fixed_split_rejects_bad_input():
    with pytest.raises(ValueError):
        fixed_date_split(pd.DatetimeIndex([]))
    with pytest.raises(ValueError):
        fixed_date_split(
            pd.bdate_range("2020-01-01", "2020-02-01"),
            train_end="2021-07-01",
            val_end="2021-01-01",  # not strictly increasing
            test_end="2022-06-30",
        )
