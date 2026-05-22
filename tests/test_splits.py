"""Tests for ``stml.replication.splits`` (US-002).

Ground truth is recomputed against the real released panel
(``stml.io.load_clean_data``): 645 trading days, 2020-01-03..2022-06-30, no NaN.
The chronological 60/20/20 cut is train[0:387], val[387:516], test[516:645].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.io import load_clean_data
from stml.replication.splits import (
    Split,
    chronological_split,
    embargoed_val,
    get_test,
    n_eff,
    run_length_p90,
)

# Panel cut points: int(645*0.6)=387, int(645*0.8)=516.
N_PANEL = 645
TRAIN_END = 387
VAL_END = 516

INSTRUMENTS = [
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


@pytest.fixture(scope="module")
def signals() -> pd.DataFrame:
    """The real 645-row wide signal panel (date + 11 instrument columns)."""
    _, sig = load_clean_data()
    assert len(sig) == N_PANEL
    assert not sig.isna().any().any()
    return sig


@pytest.fixture(scope="module")
def split(signals: pd.DataFrame) -> Split:
    return chronological_split(signals["date"])


# --------------------------------------------------------------------------- #
# chronological_split: contiguous, non-overlapping, ordered, tiling.          #
# --------------------------------------------------------------------------- #
def test_split_exact_boundaries(split: Split) -> None:
    assert np.array_equal(split.train_idx, np.arange(0, TRAIN_END))
    assert np.array_equal(split.val_idx, np.arange(TRAIN_END, VAL_END))
    assert np.array_equal(split.test_idx, np.arange(VAL_END, N_PANEL))
    assert (len(split.train_idx), len(split.val_idx), len(split.test_idx)) == (
        387,
        129,
        129,
    )


def test_split_dates_match_indices(split: Split, signals: pd.DataFrame) -> None:
    dates = pd.DatetimeIndex(signals["date"])
    assert split.train_dates.equals(dates[split.train_idx])
    assert split.val_dates.equals(dates[split.val_idx])
    assert split.test_dates.equals(dates[split.test_idx])


def test_split_no_overlap_and_tiles_all(split: Split) -> None:
    tr, va, te = set(split.train_idx), set(split.val_idx), set(split.test_idx)
    # pairwise disjoint
    assert tr & va == set()
    assert va & te == set()
    assert tr & te == set()
    # union is exactly the full range
    union = np.sort(np.concatenate([split.train_idx, split.val_idx, split.test_idx]))
    assert np.array_equal(union, np.arange(N_PANEL))


def test_split_strictly_increasing(split: Split) -> None:
    full = np.concatenate([split.train_idx, split.val_idx, split.test_idx])
    assert np.all(np.diff(full) == 1)  # contiguous and strictly increasing
    for block in (split.train_idx, split.val_idx, split.test_idx):
        assert np.all(np.diff(block) > 0)


def test_split_custom_fracs_floor_and_tile() -> None:
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    sp = chronological_split(dates, fracs=(0.7, 0.15, 0.15))
    assert len(sp.train_idx) == 70  # int(100*0.7)
    assert len(sp.val_idx) == 15  # int(100*0.85) - 70
    assert len(sp.test_idx) == 15
    union = np.concatenate([sp.train_idx, sp.val_idx, sp.test_idx])
    assert np.array_equal(np.sort(union), np.arange(100))


@pytest.mark.parametrize(
    "bad", [(0.6, 0.2), (0.5, 0.2, 0.2), (0.6, 0.0, 0.4), (-0.1, 0.5, 0.6)]
)
def test_split_rejects_bad_fracs(bad: tuple) -> None:
    with pytest.raises(ValueError):
        chronological_split(pd.date_range("2020-01-01", periods=10), fracs=bad)


# --------------------------------------------------------------------------- #
# run_length_p90 / n_eff primitives.                                          #
# --------------------------------------------------------------------------- #
def test_run_length_p90_known_series() -> None:
    # runs of length [2, 3, 1] -> p90 (linear interp) = 2.8 -> int 2
    s = pd.Series([1, 1, 0, 0, 0, 1])
    assert run_length_p90(s) == int(np.percentile([2, 3, 1], 90))


def test_n_eff_counts_runs() -> None:
    assert n_eff(pd.Series([1, 1, 0, 0, 0, 1])) == 3  # three runs
    assert n_eff(pd.Series([0, 0, 0, 0])) == 1  # one flat run
    assert n_eff(pd.Series([1, -1, 1, -1])) == 4  # flips every step


def test_empty_primitives() -> None:
    empty = pd.Series([], dtype=int)
    assert run_length_p90(empty) == 0
    assert n_eff(empty) == 0


# --------------------------------------------------------------------------- #
# Embargo: default >= TRAIN p90, and >= p90 removed at EACH val edge.         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("inst", INSTRUMENTS)
def test_default_embargo_is_train_p90_and_guards_both_edges(
    inst: str, signals: pd.DataFrame, split: Split
) -> None:
    sig = signals[inst]
    full_p90 = run_length_p90(sig)

    pos = embargoed_val(sig, split)  # default embargo == full-period p90
    val_start = int(split.val_idx[0])
    val_end = int(split.val_idx[-1]) + 1

    if pos.size == 0:
        # Defensive only: would require 2*p90 >= len(val). Does NOT occur for
        # any of the 11 instruments under full-period p90 (max is ng1s p90~33,
        # and 2*33 < 129), so this branch is unreached on the real panel.
        assert 2 * full_p90 >= len(split.val_idx)
        return

    lo_removed = int(pos[0]) - val_start  # rows dropped at the train/val edge
    hi_removed = val_end - 1 - int(pos[-1])  # rows dropped at the val/test edge
    # The default embargo is exactly the full-period run p90 (>= by construction)...
    assert lo_removed >= full_p90
    assert hi_removed >= full_p90
    # ...so every retained val position is >= p90 away from each boundary,
    # i.e. no constant-signal run can straddle the train/val or val/test cut.
    assert int(pos[0]) - val_start >= full_p90
    assert val_end - 1 - int(pos[-1]) >= full_p90
    # positions are a contiguous interior block of the val window
    assert np.array_equal(pos, np.arange(int(pos[0]), int(pos[-1]) + 1))
    assert val_start <= int(pos[0]) and int(pos[-1]) < val_end


def test_embargo_explicit_override(signals: pd.DataFrame, split: Split) -> None:
    sig = signals["es1s"]
    pos = embargoed_val(sig, split, embargo=5)
    val_start, val_end = int(split.val_idx[0]), int(split.val_idx[-1]) + 1
    assert int(pos[0]) == val_start + 5
    assert int(pos[-1]) == val_end - 1 - 5


def test_embargo_rejects_negative(signals: pd.DataFrame, split: Split) -> None:
    with pytest.raises(ValueError):
        embargoed_val(signals["es1s"], split, embargo=-1)


def test_large_embargo_empties_window(signals: pd.DataFrame, split: Split) -> None:
    # An embargo wider than half the val window must yield an empty array,
    # never a malformed/negative range.
    pos = embargoed_val(signals["es1s"], split, embargo=999)
    assert pos.size == 0


# --------------------------------------------------------------------------- #
# Effective sample size on the POST-embargo window (implementation note #1).  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("inst", INSTRUMENTS)
def test_post_embargo_n_eff_le_raw(
    inst: str, signals: pd.DataFrame, split: Split
) -> None:
    sig = signals[inst]
    raw_val = sig.iloc[split.val_idx]
    post_val = sig.iloc[embargoed_val(sig, split)]
    # Embargoing removes whole runs at each edge, so it can only lower n_eff.
    assert n_eff(post_val) <= n_eff(raw_val)


def test_ng1s_post_embargo_n_eff_collapses(signals: pd.DataFrame, split: Split) -> None:
    # ng1s never goes +1 and holds very long flat runs; its raw val n_eff is
    # small and the post-embargo gateable n_eff drops further (to <= 3, and in
    # fact to 0 here because the train-p90 embargo consumes the whole window).
    sig = signals["ng1s"]
    raw = n_eff(sig.iloc[split.val_idx])
    post = n_eff(sig.iloc[embargoed_val(sig, split)])
    assert post <= raw
    assert post <= 3


def test_gateable_n_eff_recipe(signals: pd.DataFrame, split: Split) -> None:
    # The documented recipe must run for every instrument without crashing,
    # including the degenerate (ng1s never +1) and ~80%-flat (gc1s/ho1s) cases.
    for inst in INSTRUMENTS:
        sig = signals[inst]
        gateable = n_eff(sig.iloc[embargoed_val(sig, split)])
        assert gateable >= 0


# --------------------------------------------------------------------------- #
# get_test tripwire.                                                          #
# --------------------------------------------------------------------------- #
def test_get_test_blocks_without_confirmation(split: Split) -> None:
    with pytest.raises(RuntimeError):
        get_test(split)


def test_get_test_returns_with_confirmation(split: Split) -> None:
    out = get_test(split, final_confirmation=True)
    assert np.array_equal(out, split.test_idx)
    assert np.array_equal(out, np.arange(VAL_END, N_PANEL))
