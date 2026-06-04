"""Leakage + path-accounting tests for :class:`stml.model.cv.CombinatorialPurgedCV`.

The load-bearing assertion (``test_no_train_test_label_overlap``) proves that for every CPCV
split, no train event's label window ``[bar_pos, bar_pos+h]`` overlaps any test event's window on
the same instrument -- the exact leakage triple-barrier labels would otherwise inject.
"""

from __future__ import annotations

import numpy as np
import pytest

from stml.model.cv import CombinatorialPurgedCV, PurgedWalkForward
from stml.model.linear import LogRegModel

H = 3
EMB = 2


def _cpcv(n_blocks=6, k_test=2, h=H, emb=EMB):
    return CombinatorialPurgedCV(n_blocks, k_test, h=h, default_embargo=emb)


@pytest.mark.parametrize(
    "n_blocks,k_test,exp_splits,exp_paths",
    [(6, 2, 15, 5), (8, 2, 28, 7), (10, 2, 45, 9), (6, 3, 20, 10)],
)
def test_path_and_split_counts(n_blocks, k_test, exp_splits, exp_paths):
    cv = _cpcv(n_blocks, k_test)
    assert cv.n_splits() == exp_splits
    assert cv.n_paths() == exp_paths
    # n_paths == C(N,k) * k / N
    assert cv.n_paths() == exp_splits * k_test // n_blocks


def test_split_count_matches_combinations(synthetic_panel):
    cv = _cpcv()
    splits = list(cv.split(synthetic_panel))
    assert len(splits) == cv.n_splits() == 15


def test_train_test_disjoint_and_within_bounds(synthetic_panel):
    n = len(synthetic_panel)
    for tr, te in _cpcv().split(synthetic_panel):
        assert set(tr).isdisjoint(set(te))
        assert tr.max() < n and te.max() < n
        assert te.size > 0 and tr.size > 0


def test_no_train_test_label_overlap(synthetic_panel):
    """For every split + instrument: no kept train bar lies within the purge+embargo envelope of
    any test bar (``|p - t| > h``), and the embargo separation (``> h + emb``) holds."""
    df = synthetic_panel.reset_index(drop=True)
    inst = df["instrument"].to_numpy()
    bar = df["bar_pos"].to_numpy()
    for tr, te in _cpcv().split(df):
        for g in np.unique(inst[te]):
            test_bars = np.sort(bar[te][inst[te] == g])
            train_bars = bar[tr][inst[tr] == g]
            if train_bars.size == 0 or test_bars.size == 0:
                continue
            ins = np.searchsorted(test_bars, train_bars)
            left = test_bars[np.clip(ins - 1, 0, test_bars.size - 1)]
            right = test_bars[np.clip(ins, 0, test_bars.size - 1)]
            dist = np.minimum(np.abs(train_bars - left), np.abs(train_bars - right))
            # hard leakage invariant: label windows [p,p+h] and [t,t+h] must not overlap
            assert dist.min() > H, f"window overlap in split, instrument {g}"
            # embargo: at least (h + emb) bars of separation
            assert dist.min() > H + EMB, f"embargo violated, instrument {g}"


def test_oof_predict_aligned_and_no_nan_where_scored(synthetic_panel):
    df = synthetic_panel
    X = df[["f1_a", "f1_b", "f2_vol"]].astype(float).reset_index(drop=True)
    y = df["bin"].to_numpy()
    cv = _cpcv()
    oof, counts, n_scored = cv.oof_predict(LogRegModel, {"C": 1.0}, X, y, df, seed=42)
    assert oof.shape == (len(df),) and counts.shape == (len(df),)
    assert n_scored == cv.n_splits()  # clean panel: every fold has both classes in train
    scored = counts > 0
    assert scored.any()
    assert np.isfinite(oof[scored]).all()  # no NaN where a prediction was made
    assert ((oof[scored] >= 0) & (oof[scored] <= 1)).all()
    # each observation is predicted on as many paths as it has appearances
    assert counts[scored].max() <= cv.n_paths()


def test_class_skipped_fold_decrements_scored(synthetic_panel):
    """Class 0 present ONLY in the last date block -> every combination that tests that block has
    an all-ones (single-class) train and is skipped: 5 of 15 folds drop."""
    df = synthetic_panel.reset_index(drop=True)
    X = df[["f1_a", "f1_b", "f2_vol"]].astype(float)
    y = np.ones(len(df), dtype=int)
    last_block = np.sort(df["date"].unique())[-12:]  # block 5 of 6 (72/6 = 12)
    zero_mask = df["date"].isin(last_block).to_numpy()
    y[zero_mask] = (df.loc[zero_mask, "bar_pos"].to_numpy() % 2)  # mixed 0/1 inside last block
    cv = _cpcv()
    _, _, n_scored = cv.oof_predict(LogRegModel, {"C": 1.0}, X, y, df, seed=42)
    # combos containing the last block (its complement train is all-ones) = C(5,1) = 5 -> skipped
    assert n_scored == cv.n_splits() - 5 == 10


def test_weight_fn_called_with_train_positions(synthetic_panel):
    """``oof_predict`` recomputes weights per fold via ``weight_fn(train_pos)`` (not a slice)."""
    df = synthetic_panel
    X = df[["f1_a", "f1_b", "f2_vol"]].astype(float).reset_index(drop=True)
    y = df["bin"].to_numpy()
    seen_lengths = []

    def weight_fn(tr_pos: np.ndarray) -> np.ndarray:
        seen_lengths.append(tr_pos.size)
        assert tr_pos.ndim == 1 and tr_pos.dtype.kind in "iu"
        return np.ones(tr_pos.size)

    cv = _cpcv()
    cv.oof_predict(LogRegModel, {"C": 1.0}, X, y, df, seed=42, weight_fn=weight_fn)
    assert len(seen_lengths) == cv.n_splits()
    assert all(n > 0 for n in seen_lengths)


def test_missing_bar_pos_raises(synthetic_panel):
    bad = synthetic_panel.drop(columns="bar_pos")
    with pytest.raises(KeyError):
        list(_cpcv().split(bad))


def test_purged_walk_forward_untouched(synthetic_panel):
    """The existing expanding splitter still works (additive change, no regression)."""
    wf = PurgedWalkForward(n_splits=3, h=H, default_embargo=EMB)
    folds = list(wf.split(synthetic_panel))
    assert len(folds) == 3
    for tr, va in folds:
        assert set(tr).isdisjoint(set(va))


# --- per-instrument h (h_by_instrument) -------------------------------------------------------

def _kept_train_by_inst(df, splitter):
    """Per fold: ``({instrument: set(kept-train bar_pos)}, holdout_pos)`` for either splitter."""
    inst = df["instrument"].to_numpy()
    bar = df["bar_pos"].to_numpy()
    out = []
    for tr, hold in splitter.split(df):
        per = {g: set(bar[tr][inst[tr] == g]) for g in np.unique(inst)}
        out.append((per, hold))
    return out


def test_per_instrument_h_none_matches_scalar(synthetic_panel):
    """``h_by_instrument=None`` reproduces the single-``h`` splits byte-for-byte (backward compat)."""
    base = list(_cpcv().split(synthetic_panel))
    mapped = list(
        CombinatorialPurgedCV(6, 2, h=H, h_by_instrument=None, default_embargo=EMB).split(synthetic_panel)
    )
    assert len(base) == len(mapped)
    for (tr0, te0), (tr1, te1) in zip(base, mapped):
        assert np.array_equal(tr0, tr1) and np.array_equal(te0, te1)
    # an empty map is the same as None
    empty = list(
        CombinatorialPurgedCV(6, 2, h=H, h_by_instrument={}, default_embargo=EMB).split(synthetic_panel)
    )
    for (tr0, _), (tr1, _) in zip(base, empty):
        assert np.array_equal(tr0, tr1)


def test_per_instrument_h_widens_purge_cpcv(synthetic_panel):
    """A larger per-instrument ``h`` purges *that* instrument more (kept-train shrinks) and respects
    the wider ``h_g + emb`` envelope; other instruments and all test sets are unchanged."""
    df = synthetic_panel.reset_index(drop=True)
    inst = df["instrument"].to_numpy()
    bar = df["bar_pos"].to_numpy()
    big = H + 5
    scalar = _kept_train_by_inst(df, _cpcv())
    wide = _kept_train_by_inst(df, CombinatorialPurgedCV(6, 2, h=H, h_by_instrument={"AAA": big},
                                                         default_embargo=EMB))
    strictly_more = False
    for (s_per, s_te), (w_per, w_te) in zip(scalar, wide):
        assert np.array_equal(s_te, w_te)            # purge touches only the train side
        assert w_per["AAA"] <= s_per["AAA"]          # AAA purged at least as hard
        if w_per["AAA"] < s_per["AAA"]:
            strictly_more = True
        for g in ("BBB", "CCC"):                      # untouched instruments identical
            assert w_per[g] == s_per[g]
        test_bars = np.sort(bar[w_te][inst[w_te] == "AAA"])
        train_bars = np.array(sorted(w_per["AAA"]))
        if test_bars.size and train_bars.size:
            ins = np.searchsorted(test_bars, train_bars)
            left = test_bars[np.clip(ins - 1, 0, test_bars.size - 1)]
            right = test_bars[np.clip(ins, 0, test_bars.size - 1)]
            dist = np.minimum(np.abs(train_bars - left), np.abs(train_bars - right))
            assert dist.min() > big + EMB             # wider envelope honored
    assert strictly_more, "wider h must purge AAA strictly more in at least one split"


def test_per_instrument_h_narrower_superset_cpcv(synthetic_panel):
    """A smaller per-instrument ``h`` purges *that* instrument less -- its kept-train is a superset
    of the scalar-``h`` run (others unchanged)."""
    df = synthetic_panel.reset_index(drop=True)
    scalar = _kept_train_by_inst(df, _cpcv())
    narrow = _kept_train_by_inst(df, CombinatorialPurgedCV(6, 2, h=H, h_by_instrument={"AAA": 1},
                                                           default_embargo=EMB))
    strictly_more = False
    for (s_per, _), (n_per, _) in zip(scalar, narrow):
        assert s_per["AAA"] <= n_per["AAA"]          # narrower h keeps a superset
        if s_per["AAA"] < n_per["AAA"]:
            strictly_more = True
        for g in ("BBB", "CCC"):
            assert n_per[g] == s_per[g]
    assert strictly_more, "narrower h must keep AAA strictly more in at least one split"


def test_per_instrument_h_purged_walk_forward(synthetic_panel):
    """``PurgedWalkForward`` honors ``h_by_instrument`` the same way (one-sided, earlier tail)."""
    df = synthetic_panel.reset_index(drop=True)
    scalar = _kept_train_by_inst(df, PurgedWalkForward(n_splits=3, h=H, default_embargo=EMB))
    wide = _kept_train_by_inst(
        df, PurgedWalkForward(n_splits=3, h=H, h_by_instrument={"AAA": H + 5}, default_embargo=EMB))
    narrow = _kept_train_by_inst(
        df, PurgedWalkForward(n_splits=3, h=H, h_by_instrument={"AAA": 1}, default_embargo=EMB))
    wider_seen = narrower_seen = False
    for (s_per, s_va), (w_per, w_va), (n_per, _) in zip(scalar, wide, narrow):
        assert np.array_equal(s_va, w_va)            # val blocks unchanged by purge width
        assert w_per["AAA"] <= s_per["AAA"] <= n_per["AAA"]
        wider_seen |= w_per["AAA"] < s_per["AAA"]
        narrower_seen |= s_per["AAA"] < n_per["AAA"]
        for g in ("BBB", "CCC"):
            assert w_per[g] == s_per[g] == n_per[g]
    assert wider_seen and narrower_seen
