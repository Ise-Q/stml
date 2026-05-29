"""Tests for the triple-barrier meta-labeler (:mod:`stml.model.labels`).

These assert the two properties the WIP notebook got wrong -- **first-touch ordering** (a path
that hits the stop before the target is a loss even if it ends up positive) and **no peeking past
the available window** -- plus the meta-label sign convention and sane uniqueness weights. They use
small hand-built price paths so the expected label is unambiguous.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stml.model.labels import class_balance, sample_uniqueness, triple_barrier_labels


def _panel(prices: list[float], inst: str = "X") -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="D")
    return pd.DataFrame({inst: prices}, index=idx)


def _event(date, inst="X", side=1.0, sigma=0.01):
    return pd.DataFrame({"date": [pd.Timestamp(date)], "instrument": [inst],
                         "side": [side], "sigma": [sigma]})


def test_profit_target_first_is_label_1():
    # entry 100, next bar +1% hits the +1sigma target first -> label 1
    close = _panel([100, 101.5, 100, 100, 100])
    ev = _event("2020-01-01", side=1.0, sigma=0.01)
    out = triple_barrier_labels(close, ev, pt=1.0, sl=1.0, h=3)
    assert len(out) == 1
    assert out.iloc[0]["touch"] == "pt"
    assert out.iloc[0]["bin"] == 1


def test_stop_loss_first_beats_later_target():
    # entry 100, bar1 = 98 (hits -1sigma stop FIRST), bar2 = 105 (would hit target).
    # First-touch ordering must label this a loss (0), not a win.
    close = _panel([100, 98.0, 105.0, 105.0, 105.0])
    ev = _event("2020-01-01", side=1.0, sigma=0.01)
    out = triple_barrier_labels(close, ev, pt=1.0, sl=1.0, h=3)
    assert out.iloc[0]["touch"] == "sl"
    assert out.iloc[0]["bin"] == 0


def test_short_side_inverts_direction():
    # short (side=-1): a price DROP is profitable. entry 100 -> 98 is +2% side-adjusted -> PT.
    close = _panel([100, 98.0, 100, 100, 100])
    ev = _event("2020-01-01", side=-1.0, sigma=0.01)
    out = triple_barrier_labels(close, ev, pt=1.0, sl=1.0, h=3)
    assert out.iloc[0]["touch"] == "pt"
    assert out.iloc[0]["bin"] == 1


def test_vertical_timeout_uses_return_sign():
    # stays inside both barriers for h bars -> vertical; ends slightly up -> label 1.
    close = _panel([100, 100.2, 100.3, 100.4, 100.5])
    ev = _event("2020-01-01", side=1.0, sigma=0.05, )  # 5% barriers, never touched
    out = triple_barrier_labels(close, ev, pt=1.0, sl=1.0, h=3)
    assert out.iloc[0]["touch"] == "vert"
    assert out.iloc[0]["bin"] == 1
    # vertical_zero forces a time-out to 0 regardless of sign
    out0 = triple_barrier_labels(close, ev, pt=1.0, sl=1.0, h=3, vertical_zero=True)
    assert out0.iloc[0]["bin"] == 0


def test_event_without_full_window_is_dropped():
    # only 2 forward bars available but h=3 -> no full window -> event dropped (no peeking).
    close = _panel([100, 101, 102])
    ev = _event("2020-01-01", side=1.0, sigma=0.01)
    out = triple_barrier_labels(close, ev, pt=2.0, sl=2.0, h=3)
    assert out.empty


def test_price_end_truncates_and_drops():
    close = _panel([100, 100, 100, 100, 100, 100, 100, 100])
    ev = _event("2020-01-03", side=1.0, sigma=0.05)
    # truncating before a full window exists past the event -> dropped
    out = triple_barrier_labels(close, ev, pt=1.0, sl=1.0, h=4, price_end="2020-01-05")
    assert out.empty


def test_bad_sigma_dropped():
    close = _panel([100, 101, 102, 103, 104])
    for bad in (0.0, np.nan, -0.01):
        out = triple_barrier_labels(close, _event("2020-01-01", sigma=bad), pt=1.0, sl=1.0, h=3)
        assert out.empty


def test_class_balance_and_uniqueness():
    close = _panel([100, 101.5, 99, 100, 101.5, 99, 100, 101.5])
    ev = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "instrument": ["X", "X", "X"], "side": [1.0, 1.0, 1.0], "sigma": [0.01, 0.01, 0.01],
    })
    out = triple_barrier_labels(close, ev, pt=1.0, sl=1.0, h=3)
    bal = class_balance(out)
    assert 0.0 <= bal["minority_frac"] <= 0.5
    w = sample_uniqueness(out, close)
    assert ((w > 0) & (w <= 1.0 + 1e-9)).all()
