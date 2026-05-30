"""Tests for ``stml.harry.labels``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.harry.labels import (
    DEFAULT_H,
    INSTRUMENTS,
    TripleBarrierConfig,
    _label_one_instrument,
    _per_instrument_uniqueness,
    _resolve_event,
    ewma_daily_vol,
    get_meta_labels,
)


# --------------------------------------------------------------------------- #
# Config validation                                                            #
# --------------------------------------------------------------------------- #
def test_triple_barrier_config_validates_inputs():
    TripleBarrierConfig(h=1, pt_mult=0.0, sl_mult=0.5, vol_span=2)  # boundary OK
    with pytest.raises(ValueError):
        TripleBarrierConfig(h=0)
    with pytest.raises(ValueError):
        TripleBarrierConfig(pt_mult=-0.1)
    with pytest.raises(ValueError):
        TripleBarrierConfig(sl_mult=-0.1)
    with pytest.raises(ValueError):
        TripleBarrierConfig(vol_span=1)


def test_triple_barrier_config_sqrt_h():
    cfg = TripleBarrierConfig(h=10)
    assert cfg.sqrt_h == pytest.approx(np.sqrt(10))


# --------------------------------------------------------------------------- #
# Causality of the EWMA vol estimator                                          #
# --------------------------------------------------------------------------- #
def test_ewma_daily_vol_truncation_invariance():
    """``sigma_t`` computed on ``close[:t+1]`` equals ``sigma_t`` on the full
    series for every ``t`` — strict no-peeking guarantee."""
    rng = np.random.default_rng(13)
    n = 200
    dates = pd.date_range("2020-01-06", periods=n, freq="B")
    close = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=dates
    )
    full = ewma_daily_vol(close, span=50)
    for t_pos in (50, 100, 150, 199):
        trunc = ewma_daily_vol(close.iloc[: t_pos + 1], span=50)
        assert full.iloc[t_pos] == pytest.approx(trunc.iloc[t_pos], abs=1e-12)


# --------------------------------------------------------------------------- #
# Off-by-one: hand-computed 5-row example                                       #
# --------------------------------------------------------------------------- #
def test_off_by_one_fix_changes_label_5_row_ohlc():
    """5-row close series where entry-at-t and entry-at-t+1 produce different
    labels. Worked example also lives in ``reports/harry/02-labels.md``.

    closes = [100, 90, 110, 115, 120], signal +1 at t=0, h=3, thresholds 0.05.

    OLD convention (entry at t, window=closes[0:4]):
        log-dist from entry=100: [0, log(0.9)=-0.105, log(1.1)=+0.095, log(1.15)=+0.140]
        SL at index 1 first (log(0.9) ≤ -0.05) → end=1, ret=-0.105, label=0.

    NEW convention (entry at t+1, window=closes[1:5]):
        log-dist from entry=90: [0, log(110/90)=+0.201, log(115/90)=+0.245, log(120/90)=+0.288]
        PT at index 1 first (log(110/90) ≥ +0.05) → end=1, ret=+0.201, label=1.
    """
    closes = np.array([100.0, 90.0, 110.0, 115.0, 120.0])
    pt = sl = 0.05

    end_old, ret_old = _resolve_event(
        closes[:4], side=+1, pt_threshold=pt, sl_threshold=sl
    )
    end_new, ret_new = _resolve_event(
        closes[1:5], side=+1, pt_threshold=pt, sl_threshold=sl
    )

    # OLD convention: stops out at offset 1, label 0.
    assert end_old == 1
    assert ret_old == pytest.approx(np.log(0.9), abs=1e-9)
    assert int(ret_old > 0) == 0

    # NEW convention: hits PT at offset 1, label 1.
    assert end_new == 1
    assert ret_new == pytest.approx(np.log(110 / 90), abs=1e-9)
    assert int(ret_new > 0) == 1

    # And the two labels disagree on this constructed example.
    assert (ret_old > 0) != (ret_new > 0)


def test_get_meta_labels_uses_t_plus_one_entry_integration():
    """End-to-end with an injected sigma: the event resolves on
    ``[t+1, t+1+h]`` (not ``[t, t+h]``) and the realised return is the
    hand-computed ``log(110/90)``."""
    # We pad with 60 days of trivial pre-history so signal at index 60 has a
    # well-defined position and the scenario starts at index 60.
    n_pad = 60
    rng = np.random.default_rng(7)
    pad_returns = rng.normal(0, 0.005, size=n_pad)
    pad_close = 100.0 * np.exp(np.cumsum(pad_returns))
    anchor = float(pad_close[-1])
    # Scenario closes appended AFTER the pad: relative to anchor,
    # multipliers [0.9, 1.1, 1.15, 1.2] (i.e. "90, 110, 115, 120" rescaled).
    scenario = anchor * np.array([0.9, 1.1, 1.15, 1.2])
    closes = np.concatenate([pad_close, scenario])
    n = len(closes)  # n_pad + 4
    dates = pd.date_range("2020-01-06", periods=n, freq="B")
    ohlcv = pd.DataFrame(
        {"date": dates, "instrument": "es1s", "close": closes}
    )
    # Signal +1 at the last pad day; entry must therefore be at index n_pad
    # (the "90" bar relative to anchor).
    s = np.zeros(n)
    s[n_pad - 1] = 1
    signals = pd.DataFrame({"date": dates, "es1s": s})

    # Inject sigma so the threshold is exactly 0.05.
    #   pt_thresh = pt_mult * sigma * sqrt(h) = 1.0 * sigma * sqrt(3) = 0.05
    #   => sigma = 0.05 / sqrt(3)
    sigma_value = 0.05 / float(np.sqrt(3))
    sigma_inject = pd.Series(sigma_value, index=dates)

    events = get_meta_labels(
        ohlcv,
        signals,
        h=3,
        pt_mult=1.0,
        sl_mult=1.0,
        vol_span=50,
        instruments=["es1s"],
        sigma={"es1s": sigma_inject},
    )

    assert len(events) == 1
    ev = events.iloc[0]
    assert ev["t_signal"] == dates[n_pad - 1]
    # t_start = signal date + 1 trading day — the "90" bar.
    assert ev["t_start"] == dates[n_pad]
    # Resolution at offset 1 in the window (the "110" bar).
    assert ev["t_end"] == dates[n_pad + 1]
    assert ev["ret"] == pytest.approx(np.log(110 / 90), abs=1e-9)
    assert ev["label"] == 1
    assert ev["side"] == 1
    assert ev["sigma"] == pytest.approx(sigma_value, abs=1e-12)


# --------------------------------------------------------------------------- #
# Causality: full-pipeline truncation invariance                               #
# --------------------------------------------------------------------------- #
def test_truncation_invariance_for_resolved_events():
    """An event resolved on the full history must produce identical
    ``t_end``, ``ret``, ``label``, and ``sigma`` when re-resolved on a
    series truncated just past ``t_end``."""
    rng = np.random.default_rng(11)
    n = 200
    dates = pd.date_range("2020-01-06", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    ohlcv = pd.DataFrame({"date": dates, "instrument": "es1s", "close": close})
    s = np.zeros(n)
    s[40] = 1
    s[80] = -1
    s[120] = 1
    signals = pd.DataFrame({"date": dates, "es1s": s})

    full = get_meta_labels(ohlcv, signals, h=10, instruments=["es1s"])
    assert len(full) >= 1, "test setup did not produce any events"

    bar_pos = {d: i for i, d in enumerate(dates)}
    for _, ev in full.iterrows():
        # Truncate just past the FULL forward window the labeller needs
        # (signal pos + 1 entry + h forward bars + 1-bar buffer). An event
        # that resolved early at t_end <= t+1+h still requires the full
        # window to be considered by the labeller; truncating only to t_end
        # would (correctly) drop the event from the labeller's output.
        cut = bar_pos[ev["t_signal"]] + 1 + DEFAULT_H + 2
        trunc_ohlcv = ohlcv.iloc[:cut].reset_index(drop=True)
        trunc_signals = signals.iloc[:cut].reset_index(drop=True)
        trunc = get_meta_labels(
            trunc_ohlcv, trunc_signals, h=10, instruments=["es1s"]
        )
        match = trunc[trunc["t_signal"] == ev["t_signal"]]
        assert len(match) == 1, f"event {ev['t_signal']} missing in truncation"
        m = match.iloc[0]
        assert m["t_end"] == ev["t_end"]
        assert m["label"] == ev["label"]
        assert m["ret"] == pytest.approx(ev["ret"], abs=1e-12)
        assert m["sigma"] == pytest.approx(ev["sigma"], abs=1e-12)


# --------------------------------------------------------------------------- #
# Asymmetric barriers                                                          #
# --------------------------------------------------------------------------- #
def test_asymmetric_barriers_change_label_distribution():
    """Three hand-constructed events on the same instrument with injected
    sigma, designed so the asymmetric-barrier direction is mechanically
    inevitable.

    With sigma = 0.01 and h = 10, the barrier widths are::

        tight (pt=0.5, sl=1.0) :  PT at ±1.58 %,  SL at ±3.16 %.
        sym   (pt=1.0, sl=1.0) :  PT at ±3.16 %,  SL at ±3.16 %.
        wide  (pt=2.0, sl=1.0) :  PT at ±6.32 %,  SL at ±3.16 %.

    Event A — round-trip up-then-down (+5 % on day 1, then declining to
              −5 % by day 10):
        sym  → PT touches at day 1 (+4.88 % ≥ +3.16 %)  → label 1.
        wide → PT never touches (peak +4.88 % < +6.32 %); SL touches at
               day 10 (−5.13 % ≤ −3.16 %)              → label 0.
        tight→ PT touches at day 1 (+4.88 % ≥ +1.58 %) → label 1.

    Event B — monotone +1 % per day (+10 % cumulative):
        sym  → PT at day 4.   wide → PT at day 7.   tight → PT at day 2.
        All three label 1.

    Event C — +2 % on day 1, then declining to −5 % by day 10:
        sym  → PT not touched at day 1 (+1.98 % < +3.16 %); SL touches at
               day 7 (−4.08 %)                          → label 0.
        wide → PT not touched; SL touches at day 7      → label 0.
        tight→ PT touches at day 1 (+1.98 % ≥ +1.58 %)  → label 1.

    Expected label rates: tight 1.0 > sym 0.667 > wide 0.333. The original
    Step-2 spec said pt=2.0, sl=1.0 should produce MORE +1 labels than
    symmetric — the direction is reversed, as this test shows.
    """
    h = 10
    sigma_val = 0.01
    n_pad = 60
    pad_close = np.full(n_pad, 100.0)

    a_path = np.array([100.0, 105, 104, 103, 102, 101, 100, 99, 98, 97, 95.0])
    b_path = np.array([100.0 * 1.01**k for k in range(h + 1)])
    c_path = np.array([100.0, 102, 101, 100, 99, 98, 97, 96, 95, 95, 95.0])

    buffer = np.full(20, 100.0)
    closes = np.concatenate(
        [pad_close, a_path, buffer, b_path, buffer, c_path]
    )

    # Position layout (n_pad=60, h+1=11, buffer=20):
    #   [0 .. 59]   pad_close          (60)
    #   [60 .. 70]  a_path             (11)
    #   [71 .. 90]  buffer             (20)
    #   [91 .. 101] b_path             (11)
    #   [102 .. 121] buffer            (20)
    #   [122 .. 132] c_path            (11)
    # Signals are placed at the position whose NEXT bar is the path[0]
    # (entry). a_path[0] is at index 60, so signal A is at index 59. b_path
    # at 91, signal at 90. c_path at 122, signal at 121.
    n = len(closes)
    dates = pd.date_range("2020-01-06", periods=n, freq="B")
    ohlcv = pd.DataFrame(
        {"date": dates, "instrument": "es1s", "close": closes}
    )
    s = np.zeros(n)
    s[59] = 1
    s[90] = 1
    s[121] = 1
    signals = pd.DataFrame({"date": dates, "es1s": s})

    sigma_inject = pd.Series(sigma_val, index=dates)
    common = dict(
        h=h, sl_mult=1.0, vol_span=50,
        instruments=["es1s"], sigma={"es1s": sigma_inject},
    )
    sym = get_meta_labels(ohlcv, signals, pt_mult=1.0, **common)
    tight = get_meta_labels(ohlcv, signals, pt_mult=0.5, **common)
    wide = get_meta_labels(ohlcv, signals, pt_mult=2.0, **common)

    # Three events each. Same number across configurations — barrier width
    # never changes which signals get labelled, only how they resolve.
    assert len(sym) == 3
    assert len(tight) == 3
    assert len(wide) == 3

    sym_lbl = sym.set_index("t_signal").sort_index()["label"].tolist()
    tight_lbl = tight.set_index("t_signal").sort_index()["label"].tolist()
    wide_lbl = wide.set_index("t_signal").sort_index()["label"].tolist()
    assert sym_lbl == [1, 1, 0], f"sym labels {sym_lbl}"
    assert tight_lbl == [1, 1, 1], f"tight labels {tight_lbl}"
    assert wide_lbl == [0, 1, 0], f"wide labels {wide_lbl}"

    sym_rate = sum(sym_lbl) / len(sym_lbl)
    tight_rate = sum(tight_lbl) / len(tight_lbl)
    wide_rate = sum(wide_lbl) / len(wide_lbl)
    # Headline: tighter PT → more +1; wider PT → fewer +1.
    assert tight_rate > sym_rate, f"tight {tight_rate} should exceed sym {sym_rate}"
    assert wide_rate < sym_rate, f"wide {wide_rate} should be less than sym {sym_rate}"


def test_wider_pt_can_only_lower_label_rate_per_event():
    """Per-event monotonicity property: for any path and ``sl_mult`` fixed,
    increasing ``pt_mult`` from the symmetric value can only flip a label
    from 1 to 0 (or leave it unchanged). It can never flip 0 → 1.

    Verified across a synthetic panel of ~80 events.
    """
    n = 1500
    rng = np.random.default_rng(17)
    dates = pd.date_range("2020-01-06", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))  # larger vol
    ohlcv = pd.DataFrame({"date": dates, "instrument": "es1s", "close": close})
    s = np.zeros(n)
    s[200::15] = 1
    signals = pd.DataFrame({"date": dates, "es1s": s})

    sym = get_meta_labels(
        ohlcv, signals, h=10, pt_mult=1.0, sl_mult=1.0, instruments=["es1s"]
    )
    wide = get_meta_labels(
        ohlcv, signals, h=10, pt_mult=3.0, sl_mult=1.0, instruments=["es1s"]
    )

    sym_lbl = sym.set_index("t_signal")["label"]
    wide_lbl = wide.set_index("t_signal")["label"]
    common = sym_lbl.index.intersection(wide_lbl.index)
    diffs = (sym_lbl[common] - wide_lbl[common])
    # No event should go from 0 in sym to 1 in wide.
    assert (diffs >= 0).all(), (
        f"unexpected promotion: {diffs[diffs < 0]}"
    )


def test_asymmetric_barriers_short_side_consistency():
    """The same asymmetric-barrier behaviour holds for short bets: a tighter
    PT (smaller ``pt_mult``) produces more +1 labels on the short side too,
    because the PT-in-the-direction-of-the-bet is what's being scaled."""
    n = 500
    rng = np.random.default_rng(19)
    dates = pd.date_range("2020-01-06", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    ohlcv = pd.DataFrame({"date": dates, "instrument": "es1s", "close": close})
    s = np.zeros(n)
    s[100::10] = -1  # all-short events
    signals = pd.DataFrame({"date": dates, "es1s": s})

    sym = get_meta_labels(
        ohlcv, signals, h=10, pt_mult=1.0, sl_mult=1.0, instruments=["es1s"]
    )
    tight_pt = get_meta_labels(
        ohlcv, signals, h=10, pt_mult=0.5, sl_mult=1.0, instruments=["es1s"]
    )
    assert tight_pt["label"].mean() > sym["label"].mean()
    # All events should be short-sided.
    assert (sym["side"] == -1).all()
    assert (tight_pt["side"] == -1).all()


# --------------------------------------------------------------------------- #
# Uniqueness weights                                                           #
# --------------------------------------------------------------------------- #
def test_uniqueness_in_zero_one_and_sum_bounded():
    n = 300
    rng = np.random.default_rng(23)
    dates = pd.date_range("2020-01-06", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    ohlcv = pd.DataFrame({"date": dates, "instrument": "es1s", "close": close})
    s = np.zeros(n)
    s[::5] = 1  # events every 5 days → h=10 windows overlap heavily
    signals = pd.DataFrame({"date": dates, "es1s": s})

    events = get_meta_labels(ohlcv, signals, h=10, instruments=["es1s"])
    assert len(events) > 0
    w = events["uniqueness_weight"]
    # Per-event bound: [0, 1].
    assert (w >= 0).all()
    assert (w <= 1.0 + 1e-9).all()
    # Sum bounded by number of events (only achievable when fully disjoint).
    assert w.sum() <= len(events) + 1e-9
    # With h=10 and signals every 5 days, events overlap → some must be <1.
    assert (w < 0.99).any()


def test_uniqueness_disjoint_events_all_one():
    """Events that don't overlap on the trading-day index get uniqueness 1.0
    each."""
    n = 100
    rng = np.random.default_rng(29)
    dates = pd.date_range("2020-01-06", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    ohlcv = pd.DataFrame({"date": dates, "instrument": "es1s", "close": close})
    s = np.zeros(n)
    s[10] = 1
    s[40] = 1
    s[70] = -1  # gaps > h=5 → all disjoint
    signals = pd.DataFrame({"date": dates, "es1s": s})
    events = get_meta_labels(ohlcv, signals, h=5, instruments=["es1s"])
    assert len(events) == 3
    assert (events["uniqueness_weight"] == 1.0).all()


def test_uniqueness_two_fully_overlapping_events_get_half():
    """Two events that span identical bars must each get uniqueness 0.5
    (concurrency=2 across the entire span)."""
    # We can call _per_instrument_uniqueness directly with a hand-built
    # events frame so we don't depend on the full pipeline.
    bar_index = pd.date_range("2020-01-06", periods=10, freq="B")
    events = pd.DataFrame(
        {
            "t_start": [bar_index[2], bar_index[2]],
            "t_end": [bar_index[6], bar_index[6]],
        }
    )
    u = _per_instrument_uniqueness(events, bar_index)
    assert u.shape == (2,)
    assert u[0] == pytest.approx(0.5)
    assert u[1] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Trading-day concurrency invariance                                           #
# --------------------------------------------------------------------------- #
def test_trading_day_concurrency_invariant_to_weekend_padding():
    """Inserting NaN weekend rows must not change uniqueness, t_end, label,
    or ret. Concurrency is computed on the cleaned trading-day index, so
    weekend dates the caller might happen to include in the input panel are
    immaterial."""
    n_bday = 60
    rng = np.random.default_rng(31)
    bd_dates = pd.date_range("2020-01-06", periods=n_bday, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_bday)))
    ohlcv_clean = pd.DataFrame(
        {"date": bd_dates, "instrument": "es1s", "close": close}
    )
    s = np.zeros(n_bday)
    s[5::7] = 1  # ~8 events
    signals_clean = pd.DataFrame({"date": bd_dates, "es1s": s})

    events_clean = get_meta_labels(
        ohlcv_clean, signals_clean, h=5, instruments=["es1s"]
    )
    assert len(events_clean) > 0

    # Build a calendar-day-indexed version: insert NaN close + NaN signal
    # for every weekend date between the first and last business day.
    full_cal = pd.date_range(bd_dates[0], bd_dates[-1], freq="D")
    ohlcv_padded = (
        ohlcv_clean.set_index("date")
        .reindex(full_cal)
        .rename_axis("date")
        .reset_index()
    )
    # Re-fill the instrument column (reindex left it NaN on weekends).
    ohlcv_padded["instrument"] = "es1s"
    signals_padded = (
        signals_clean.set_index("date")
        .reindex(full_cal)
        .rename_axis("date")
        .reset_index()
    )

    events_padded = get_meta_labels(
        ohlcv_padded, signals_padded, h=5, instruments=["es1s"]
    )

    assert len(events_clean) == len(events_padded)
    # Sort by t_signal to make the comparison position-independent.
    a = events_clean.sort_values("t_signal").reset_index(drop=True)
    b = events_padded.sort_values("t_signal").reset_index(drop=True)
    pd.testing.assert_series_equal(
        a["uniqueness_weight"], b["uniqueness_weight"],
        check_names=False, atol=1e-12,
    )
    pd.testing.assert_series_equal(a["t_end"], b["t_end"], check_names=False)
    pd.testing.assert_series_equal(a["label"], b["label"], check_names=False)
    pd.testing.assert_series_equal(
        a["ret"], b["ret"], check_names=False, atol=1e-12,
    )


# --------------------------------------------------------------------------- #
# Resolution semantics                                                         #
# --------------------------------------------------------------------------- #
def test_resolve_event_vertical_when_no_touch_label_by_sign():
    """No PT/SL touch → vertical resolution at the last bar, label by sign
    of the realised return."""
    # Drift upward but never hit the +0.5 threshold over 6 bars.
    closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    end, ret = _resolve_event(closes, side=+1, pt_threshold=0.5, sl_threshold=0.5)
    assert end == len(closes) - 1
    assert ret > 0  # positive drift → label 1
    # Down-drift case:
    closes_down = np.array([100.0, 99.0, 98.0, 97.0, 96.0, 95.0])
    end_d, ret_d = _resolve_event(
        closes_down, side=+1, pt_threshold=0.5, sl_threshold=0.5
    )
    assert end_d == len(closes_down) - 1
    assert ret_d < 0


def test_resolve_event_first_touch_wins_among_pt_and_sl():
    """When the path would eventually hit both, the EARLIER touch decides."""
    # Drops 8 % then jumps to +20 %. With pt=sl=0.05, SL fires at offset 1.
    closes = np.array([100.0, 92.0, 120.0, 120.0])
    end, ret = _resolve_event(closes, side=+1, pt_threshold=0.05, sl_threshold=0.05)
    assert end == 1
    assert ret < 0  # SL → label 0
    # Reverse: jumps up before dropping.
    closes2 = np.array([100.0, 108.0, 90.0, 90.0])
    end2, ret2 = _resolve_event(closes2, side=+1, pt_threshold=0.05, sl_threshold=0.05)
    assert end2 == 1
    assert ret2 > 0  # PT → label 1


def test_resolve_event_short_side_pt_means_price_down():
    """For a short bet, PT means price MOVED DOWN by ``pt_threshold`` (in
    log units), SL means it moved up."""
    closes = np.array([100.0, 90.0, 90.0, 90.0])
    end, ret = _resolve_event(closes, side=-1, pt_threshold=0.05, sl_threshold=0.05)
    assert end == 1
    assert ret > 0  # short profited → label 1


def test_resolve_event_invalid_inputs():
    with pytest.raises(ValueError):
        _resolve_event(np.array([100.0]), side=+1, pt_threshold=0.05, sl_threshold=0.05)
    with pytest.raises(ValueError):
        _resolve_event(
            np.array([100.0, 101.0]),
            side=0,
            pt_threshold=0.05,
            sl_threshold=0.05,
        )


# --------------------------------------------------------------------------- #
# Output schema                                                                #
# --------------------------------------------------------------------------- #
def test_get_meta_labels_output_schema_and_dtypes():
    n = 200
    rng = np.random.default_rng(37)
    dates = pd.date_range("2020-01-06", periods=n, freq="B")
    pieces_ohlcv = []
    sig_cols: dict[str, object] = {"date": dates}
    for inst in INSTRUMENTS:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        pieces_ohlcv.append(
            pd.DataFrame({"date": dates, "instrument": inst, "close": close})
        )
        s = rng.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3])
        sig_cols[inst] = s
    ohlcv = pd.concat(pieces_ohlcv, ignore_index=True)
    signals = pd.DataFrame(sig_cols)

    events = get_meta_labels(ohlcv, signals, h=DEFAULT_H)
    expected = {
        "instrument",
        "t_signal",
        "t_start",
        "t_end",
        "side",
        "ret",
        "label",
        "uniqueness_weight",
        "sigma",
    }
    assert set(events.columns) == expected
    assert set(events["instrument"].unique()).issubset(set(INSTRUMENTS))
    assert set(events["side"].unique()).issubset({-1, 1})
    assert set(events["label"].unique()).issubset({0, 1})
    # t_start should always be strictly after t_signal (next trading day).
    assert (events["t_start"] > events["t_signal"]).all()
    # t_end should never be before t_start.
    assert (events["t_end"] >= events["t_start"]).all()


def test_get_meta_labels_empty_when_no_signals():
    """With an all-zero signal panel, the function returns an empty frame
    with the correct columns."""
    n = 50
    dates = pd.date_range("2020-01-06", periods=n, freq="B")
    rng = np.random.default_rng(41)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    ohlcv = pd.DataFrame({"date": dates, "instrument": "es1s", "close": close})
    signals = pd.DataFrame({"date": dates, "es1s": np.zeros(n)})
    events = get_meta_labels(ohlcv, signals, h=5, instruments=["es1s"])
    assert events.empty
    assert "uniqueness_weight" in events.columns
