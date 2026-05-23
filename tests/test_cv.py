"""Unit tests for src/stml/cv.py.

Covers:
    - PurgedKFold: shape, completeness of test sets, no-leakage invariant,
      embargo enforcement, exact equal-chunk sizes for small N.
    - walk_forward_splits: monotonic expanding train, disjoint test windows,
      pre-test embargo enforcement.
    - split_by_boundary: partition is exact, embargo cuts train cleanly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.cv import (
    PurgedKFold,
    WalkForwardSplitter,
    assert_no_leakage,
    split_by_boundary,
    walk_forward_splits,
)


# --------------------------------------------------------------------------- #
# Synthetic event panel                                                        #
# --------------------------------------------------------------------------- #
@pytest.fixture
def synth_events() -> pd.DataFrame:
    """20 contiguous events, one per business day, with h=5 day labels."""
    dates = pd.bdate_range("2020-01-02", periods=20)
    t1 = dates + pd.tseries.offsets.BDay(5)
    df = pd.DataFrame({
        "t": dates,
        "t1": t1,
        "instrument": ["cl1s"] * 20,
        "feat": np.arange(20, dtype=float),
    })
    df.index = range(len(df))
    return df


# --------------------------------------------------------------------------- #
# 1. PurgedKFold                                                              #
# --------------------------------------------------------------------------- #
class TestPurgedKFold:

    def test_n_splits_and_shape(self, synth_events):
        cv = PurgedKFold(n_splits=4, t=synth_events["t"], t1=synth_events["t1"])
        assert cv.get_n_splits() == 4
        splits = list(cv.split(synth_events[["feat"]]))
        assert len(splits) == 4

    def test_test_sets_are_disjoint_and_cover_everything(self, synth_events):
        cv = PurgedKFold(n_splits=5, t=synth_events["t"], t1=synth_events["t1"])
        all_test = []
        for _, te in cv.split(synth_events[["feat"]]):
            all_test.extend(te.tolist())
        # All 20 events appear in exactly one test fold.
        assert sorted(all_test) == list(range(20))

    def test_no_leakage_invariant(self, synth_events):
        cv = PurgedKFold(
            n_splits=4,
            t=synth_events["t"],
            t1=synth_events["t1"],
            embargo_td=pd.Timedelta(days=2),
        )
        for tr, te in cv.split(synth_events[["feat"]]):
            # No training event's span overlaps any test event's span.
            assert_no_leakage(
                tr, te, synth_events["t"], synth_events["t1"],
                embargo_td=pd.Timedelta(days=2),
            )

    def test_embargo_excludes_immediate_next_events(self, synth_events):
        """With a 10-day embargo, the events immediately after a test block
        cannot be in train."""
        cv = PurgedKFold(
            n_splits=4,
            t=synth_events["t"],
            t1=synth_events["t1"],
            embargo_td=pd.Timedelta(days=10),
        )
        # Pick the second fold.
        splits = list(cv.split(synth_events[["feat"]]))
        tr, te = splits[1]
        test_end_t1 = synth_events.loc[te, "t1"].max()
        embargo_cutoff = test_end_t1 + pd.Timedelta(days=10)
        # No training event with t in (test_end, embargo_cutoff] allowed.
        train_t = synth_events.loc[tr, "t"]
        # train events that start AFTER the test window must start strictly
        # after embargo_cutoff.
        post = train_t[train_t > synth_events.loc[te, "t"].max()]
        if len(post):
            assert post.min() > embargo_cutoff

    def test_mismatched_index_raises(self, synth_events):
        # t.index != X.index → error
        events_with_funky_index = synth_events.copy()
        events_with_funky_index.index = events_with_funky_index.index + 100
        cv = PurgedKFold(n_splits=3, t=synth_events["t"], t1=synth_events["t1"])
        with pytest.raises(ValueError):
            list(cv.split(events_with_funky_index[["feat"]]))

    def test_n_splits_must_be_ge_two(self, synth_events):
        with pytest.raises(ValueError):
            PurgedKFold(n_splits=1, t=synth_events["t"], t1=synth_events["t1"])

    def test_n_samples_must_exceed_n_splits(self, synth_events):
        cv = PurgedKFold(n_splits=5, t=synth_events["t"], t1=synth_events["t1"])
        with pytest.raises(ValueError):
            list(cv.split(synth_events.iloc[:3][["feat"]]))


# --------------------------------------------------------------------------- #
# 2. walk_forward_splits                                                       #
# --------------------------------------------------------------------------- #
class TestWalkForward:

    def test_train_expands_monotonically(self, synth_events):
        boundaries = pd.date_range("2020-01-05", "2020-01-30", freq="W-MON").tolist()
        sizes = []
        for tr, te, _b_lo, _b_hi in walk_forward_splits(synth_events["t"], boundaries):
            sizes.append(len(tr))
        assert all(b >= a for a, b in zip(sizes, sizes[1:]))

    def test_test_windows_are_consecutive_and_disjoint(self, synth_events):
        boundaries = pd.date_range("2020-01-05", "2020-02-01", freq="W-MON").tolist()
        all_test = []
        windows = []
        for tr, te, b_lo, b_hi in walk_forward_splits(synth_events["t"], boundaries):
            all_test.extend(te.tolist())
            windows.append((b_lo, b_hi))
        # No double-counting.
        assert len(all_test) == len(set(all_test))
        # Windows tile [b_0, b_last) without gaps.
        for (_, hi), (lo, _) in zip(windows[:-1], windows[1:]):
            assert hi == lo

    def test_pre_test_embargo_cuts_train(self, synth_events):
        boundaries = [pd.Timestamp("2020-01-10"), pd.Timestamp("2020-01-20")]
        embargo = pd.Timedelta(days=3)
        for tr, _te, b_lo, _b_hi in walk_forward_splits(
            synth_events["t"], boundaries, embargo_td=embargo
        ):
            cutoff = b_lo - embargo
            train_t = synth_events.loc[tr, "t"]
            if len(train_t):
                assert train_t.max() < cutoff

    def test_walk_forward_splitter_class(self, synth_events):
        boundaries = pd.date_range("2020-01-05", "2020-02-01", freq="W-MON").tolist()
        wfs = WalkForwardSplitter(boundaries)
        # Class version requires explicit t kwarg.
        splits = list(wfs.split(synth_events[["feat"]], t=synth_events["t"]))
        assert len(splits) == wfs.get_n_splits()


# --------------------------------------------------------------------------- #
# 3. split_by_boundary                                                         #
# --------------------------------------------------------------------------- #
class TestSplitByBoundary:

    def test_partition_is_exact(self, synth_events):
        boundary = pd.Timestamp("2020-01-15")
        tr, pr = split_by_boundary(synth_events["t"], boundary, embargo_td=None)
        # All events ≤ 2020-01-14 in train, all ≥ 2020-01-15 in predict.
        assert (synth_events.loc[tr, "t"] < boundary).all()
        assert (synth_events.loc[pr, "t"] >= boundary).all()
        # And every event is in one or the other.
        assert len(tr) + len(pr) == len(synth_events)

    def test_embargo_pulls_back_train(self, synth_events):
        boundary = pd.Timestamp("2020-01-15")
        embargo = pd.Timedelta(days=5)
        tr_no, _ = split_by_boundary(synth_events["t"], boundary)
        tr_emb, _ = split_by_boundary(synth_events["t"], boundary, embargo_td=embargo)
        assert len(tr_emb) <= len(tr_no)
        if len(tr_emb):
            assert synth_events.loc[tr_emb, "t"].max() < boundary - embargo


# --------------------------------------------------------------------------- #
# 4. assert_no_leakage helper                                                  #
# --------------------------------------------------------------------------- #
class TestAssertNoLeakage:

    def test_flags_overlap(self, synth_events):
        # train = event 5, test = event 6. Event 5 has t1 spans 5 BDays → overlaps event 6.
        tr = np.array([5])
        te = np.array([6])
        with pytest.raises(AssertionError):
            assert_no_leakage(tr, te, synth_events["t"], synth_events["t1"])

    def test_passes_clearly_disjoint(self, synth_events):
        # Event 0 ends well before event 15 starts → no overlap.
        tr = np.array([0])
        te = np.array([15])
        assert_no_leakage(tr, te, synth_events["t"], synth_events["t1"])
