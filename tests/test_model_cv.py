"""Tests for the purged walk-forward CV and the test-set tripwire.

The central invariant: across every fold, no training event's label window ``[t, t+h]`` may reach
within ``embargo_p90`` bars of its instrument's validation block -- otherwise overlapping
triple-barrier labels leak future into the score. We also check the walk-forward is expanding and
time-ordered, that only development rows are ever returned, and that the test partition stays
behind the :func:`release_test` confirmation gate. Run on the real dev panel.
"""

from __future__ import annotations

import pytest

from stml.model import dataset as ds
from stml.model.cv import PurgedWalkForward
from stml.model.evaluate import release_test


@pytest.fixture(scope="module")
def dev():
    m = ds.load_matrix()
    cw = ds.close_panel()
    m = ds.attach_bar_pos(m, cw)
    return m[m["partition"].isin(ds.DEV_PARTITIONS)].reset_index(drop=True)


def test_no_label_window_leak_into_val(dev):
    h = 10
    emb = ds.embargo_map()
    cv = PurgedWalkForward(n_splits=4, h=h, embargo_by_instrument=emb)
    for tr, va in cv.split(dev):
        tr_df, va_df = dev.iloc[tr], dev.iloc[va]
        # global ordering: every train row predates the validation block
        assert tr_df["date"].max() < va_df["date"].min()
        # per-instrument purge+embargo invariant
        for inst, vg in va_df.groupby("instrument"):
            pos_v = vg["bar_pos"].min()
            tg = tr_df[tr_df["instrument"] == inst]
            if not tg.empty:
                gap = pos_v - tg["bar_pos"].max()
                assert gap > h + emb.get(inst, 10)


def test_expanding_window(dev):
    cv = PurgedWalkForward(n_splits=4, h=5, embargo_by_instrument=ds.embargo_map())
    val_starts, train_sizes = [], []
    for tr, va in cv.split(dev):
        val_starts.append(dev.iloc[va]["date"].min())
        train_sizes.append(tr.size)
    assert val_starts == sorted(val_starts)        # validation marches forward
    assert train_sizes == sorted(train_sizes)      # train block only grows


def test_only_dev_rows_returned(dev):
    cv = PurgedWalkForward(n_splits=4, h=5, embargo_by_instrument=ds.embargo_map())
    n = len(dev)
    for tr, va in cv.split(dev):
        assert tr.max() < n and va.max() < n
        assert set(tr).isdisjoint(set(va))         # no row is both train and val


def test_release_test_tripwire():
    m = ds.load_matrix()
    with pytest.raises(RuntimeError):
        release_test(m)
    out = release_test(m, final_confirmation=True)
    assert (out["partition"] == "test").all()
    assert len(out) > 0


def test_fold_n_eff_reported(dev):
    cv = PurgedWalkForward(n_splits=4, h=5, embargo_by_instrument=ds.embargo_map())
    info = cv.fold_n_eff(dev.assign(side=dev[ds.SIDE_COL]))
    assert len(info) == 4
    assert all(d["val_n_eff"] > 0 for d in info)
