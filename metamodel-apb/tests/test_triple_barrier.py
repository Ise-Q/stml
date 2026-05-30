"""Hand-built-path tests for the triple-barrier meta-labeller (Stage 1, RED-first).

Conventions (adapted from refs/triple_barrier_guide.md §4/§6, López de Prado 2018 Ch.3-4):
- events are the days where the primary signal != 0; side = sign(signal).
- barriers are ±k·σ̂ₜ (symmetric here); vertical barrier = max_holding bars.
- meta-label bin = 1 iff side-adjusted P&L at first touch > 0, else 0
  (PT-first -> 1, SL-first -> 0, timeout -> sign of expiry P&L). t1 = first-touch time.
- sample-uniqueness weight = average uniqueness over the label's span (LdP Ch.4):
  1.0 for a label whose span overlaps no other, < 1.0 when spans overlap.
"""

from __future__ import annotations

import pandas as pd
import pytest

from alken_metamodel.triple_barrier import (
    average_uniqueness,
    get_num_co_events,
    triple_barrier_labels,
)

DAYS = pd.to_datetime([f"2020-01-{d:02d}" for d in range(1, 8)])  # d0..d6


def _label_one(prices, side, *, k=1.0, trgt=0.02, max_holding=5):
    close = pd.Series(prices, index=DAYS[: len(prices)], dtype=float)
    signal = pd.Series(0, index=close.index, dtype=int)
    signal.iloc[0] = side
    target = pd.Series(trgt, index=close.index, dtype=float)
    return triple_barrier_labels(close, signal, target, pt_sl=(k, k), max_holding=max_holding)


def test_only_nonzero_signal_days_are_labelled():
    out = _label_one([100, 101, 102.5, 101, 100, 100], side=1)
    assert list(out.index) == [DAYS[0]]  # only the single nonzero-signal day


def test_upper_first_long_is_act():
    # +2.5% at d2 crosses the +2% PT before any SL
    out = _label_one([100, 101, 102.5, 101, 100, 100], side=1)
    assert out.loc[DAYS[0], "bin"] == 1.0
    assert out.loc[DAYS[0], "t1"] == DAYS[2]  # first-touch at d2
    assert out.loc[DAYS[0], "side"] == 1.0


def test_lower_first_long_is_skip():
    # -2.5% at d2 crosses the -2% SL first
    out = _label_one([100, 99.5, 97.5, 100, 101, 100], side=1)
    assert out.loc[DAYS[0], "bin"] == 0.0
    assert out.loc[DAYS[0], "t1"] == DAYS[2]


def test_vertical_positive_is_act():
    # never touches ±2%; ends +1% at the vertical barrier d5 -> act
    out = _label_one([100, 100.5, 101, 100.8, 101.2, 101], side=1, max_holding=5)
    assert out.loc[DAYS[0], "t1"] == DAYS[5]
    assert out.loc[DAYS[0], "bin"] == 1.0


def test_vertical_nonpositive_is_skip():
    # never touches ±2%; ends -1% at d5 -> skip (timeout labelled by sign of P&L)
    out = _label_one([100, 100.5, 99, 100, 99.5, 99], side=1, max_holding=5)
    assert out.loc[DAYS[0], "t1"] == DAYS[5]
    assert out.loc[DAYS[0], "bin"] == 0.0


def test_short_side_profit_when_price_falls():
    # side=-1: a 2.5% price drop at d2 is a +2.5% side-adjusted gain -> PT, act
    out = _label_one([100, 99, 97.5, 99, 100, 100], side=-1)
    assert out.loc[DAYS[0], "side"] == -1.0
    assert out.loc[DAYS[0], "t1"] == DAYS[2]
    assert out.loc[DAYS[0], "bin"] == 1.0


def test_disjoint_labels_have_unit_uniqueness():
    # spans [d0,d1] and [d3,d4] never overlap -> uniqueness 1.0 each
    t1 = pd.Series([DAYS[1], DAYS[4]], index=[DAYS[0], DAYS[3]])
    num_co = get_num_co_events(DAYS, t1)
    w = average_uniqueness(t1, num_co)
    assert w.loc[DAYS[0]] == pytest.approx(1.0)
    assert w.loc[DAYS[3]] == pytest.approx(1.0)


def test_overlapping_labels_have_sub_unit_uniqueness():
    # A=[d0,d3] fully contains B=[d1,d2]; co-event counts d0:1,d1:2,d2:2,d3:1
    t1 = pd.Series([DAYS[3], DAYS[2]], index=[DAYS[0], DAYS[1]])
    num_co = get_num_co_events(DAYS, t1)
    assert list(num_co.loc[DAYS[0] : DAYS[3]]) == [1.0, 2.0, 2.0, 1.0]
    w = average_uniqueness(t1, num_co)
    assert w.loc[DAYS[0]] == pytest.approx(0.75)  # mean(1, 1/2, 1/2, 1)
    assert w.loc[DAYS[1]] == pytest.approx(0.50)  # mean(1/2, 1/2)
    assert w.loc[DAYS[0]] < 1.0 and w.loc[DAYS[1]] < 1.0


def test_uniqueness_weight_in_pipeline_output():
    out = _label_one([100, 101, 102.5, 101, 100, 100], side=1)
    assert "weight" in out.columns
    assert 0.0 < out.loc[DAYS[0], "weight"] <= 1.0
