"""Pipeline wiring of the per-instrument embargo (S2.6, RED-first).

``_make_cv`` must build a per-instrument-embargoed splitter when the config asks
for it (loading ``embargo_p90`` from the released ``instrument_scope.json``), and
the legacy uniform splitter otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alken_metamodel.pipeline import PipelineConfig, _make_cv, load_embargo_days


def test_load_embargo_days_matches_instrument_scope():
    e = load_embargo_days()
    # ground-truth embargo_p90 (trading days) from results/instrument_scope.json
    assert e["ng1s"] == 33
    assert e["ho1s"] == 26
    assert e["rb1s"] == 19
    assert e["cl1s"] == 14
    assert set(e) >= {
        "es1s", "nq1s", "fesx1s", "cl1s", "ho1s", "rb1s",
        "ng1s", "gc1s", "si1s", "hg1s", "pl1s",
    }


def _small_panel(n_per: int = 12):
    frames = []
    for tk in ("cl1s", "ng1s"):
        idx = pd.bdate_range("2021-01-01", periods=n_per)
        frames.append(pd.DataFrame({"date": idx, "instrument": tk, "t1": idx}))
    df = pd.concat(frames, ignore_index=True).sort_values(["date", "instrument"])
    df = df.reset_index(drop=True)
    index = pd.DatetimeIndex(df["date"])
    t1 = pd.Series(pd.DatetimeIndex(df["t1"]).to_numpy(), index=index)
    inst = pd.Series(df["instrument"].to_numpy(), index=index)
    return t1, inst


def test_make_cv_enables_per_instrument_embargo_when_configured():
    t1, inst = _small_panel()
    cfg_on = PipelineConfig(per_instrument_embargo=True, cv_scheme="cpcv")
    cv_on = _make_cv(t1, cfg_on, instruments=inst)
    assert cv_on.embargo_days is not None
    assert cv_on.instruments is not None
    assert cv_on.embargo_days["ng1s"] == 33

    cfg_off = PipelineConfig(per_instrument_embargo=False, cv_scheme="cpcv")
    cv_off = _make_cv(t1, cfg_off, instruments=inst)
    assert cv_off.embargo_days is None


def test_make_cv_per_instrument_embargo_splits_are_disjoint():
    t1, inst = _small_panel(n_per=18)
    x = pd.DataFrame({"f": np.arange(len(t1), dtype=float)}, index=t1.index)
    cfg = PipelineConfig(per_instrument_embargo=True, cv_scheme="purged", n_splits=3)
    cv = _make_cv(t1, cfg, instruments=inst)
    for train, test in cv.split(x):
        assert set(train).isdisjoint(set(test))
