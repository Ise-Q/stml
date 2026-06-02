"""
test_csv_grid.py
================
Tests for predictions_grid -- the deliverable CSV builder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.model.evaluate import predictions_grid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INSTRUMENTS = ["es1s", "si1s"]
DATES_ALL = pd.date_range("2022-01-03", periods=5, freq="B")   # 5 business days


def _make_primary_signals(dates=None, instruments=None):
    """Tiny wide primary_signals frame: 'date' + one col per instrument."""
    if dates is None:
        dates = DATES_ALL
    if instruments is None:
        instruments = INSTRUMENTS
    rng = np.random.default_rng(42)
    data = {"date": [d.strftime("%Y-%m-%d") for d in dates]}
    for inst in instruments:
        # Mix of -1, 0, +1
        data[inst] = rng.choice([-1, 0, 1], size=len(dates)).tolist()
    return pd.DataFrame(data)


def _make_prob_lookup(primary_signals: pd.DataFrame) -> pd.DataFrame:
    """Long prob_lookup covering only the first 3 dates for es1s."""
    sig_long = primary_signals.melt(id_vars=["date"], var_name="instrument", value_name="signal")
    # Only cover non-zero signals for the first 3 dates for es1s
    mask = (
        (sig_long["instrument"] == "es1s")
        & (sig_long["date"].astype(str).isin(
            [d.strftime("%Y-%m-%d") for d in DATES_ALL[:3]]
        ))
        & (sig_long["signal"] != 0)
    )
    covered = sig_long[mask][["date", "instrument"]].copy()
    covered["prob"] = 0.75
    return covered.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_grid_row_count_equals_dates_times_instruments():
    """3-date window x 2 instruments = 6 rows exactly."""
    sig = _make_primary_signals()
    lookup = _make_prob_lookup(sig)
    window_start = DATES_ALL[0]
    window_end = DATES_ALL[2]   # inclusive, 3 dates
    result = predictions_grid(sig, lookup, date_start=window_start, date_end=window_end)
    assert len(result) == 3 * len(INSTRUMENTS)


def test_grid_columns_exactly():
    """Output columns must be exactly ['date','instrument','prediction']."""
    sig = _make_primary_signals()
    lookup = _make_prob_lookup(sig)
    result = predictions_grid(sig, lookup, date_start=DATES_ALL[0], date_end=DATES_ALL[4])
    assert list(result.columns) == ["date", "instrument", "prediction"]


def test_grid_zero_signal_cells_have_zero_prediction():
    """Any cell where the primary signal is 0 must have prediction == 0.0."""
    sig = _make_primary_signals()
    # Force all es1s signals to 0 for visibility
    sig["es1s"] = 0
    lookup = pd.DataFrame({"date": [], "instrument": [], "prob": []})
    result = predictions_grid(sig, lookup, date_start=DATES_ALL[0], date_end=DATES_ALL[4])
    es1s_rows = result[result["instrument"] == "es1s"]
    assert (es1s_rows["prediction"] == 0.0).all()


def test_grid_predictions_in_unit_interval():
    """All predictions must be in [0, 1]."""
    sig = _make_primary_signals()
    lookup = _make_prob_lookup(sig)
    result = predictions_grid(sig, lookup, date_start=DATES_ALL[0], date_end=DATES_ALL[4])
    assert result["prediction"].between(0.0, 1.0).all()


def test_grid_different_window_changes_only_row_count():
    """Narrowing the date window changes row count but not schema/columns."""
    sig = _make_primary_signals()
    lookup = _make_prob_lookup(sig)
    r_full = predictions_grid(sig, lookup, date_start=DATES_ALL[0], date_end=DATES_ALL[4])
    r_narrow = predictions_grid(sig, lookup, date_start=DATES_ALL[2], date_end=DATES_ALL[4])
    assert list(r_full.columns) == list(r_narrow.columns) == ["date", "instrument", "prediction"]
    assert len(r_full) == 5 * len(INSTRUMENTS)
    assert len(r_narrow) == 3 * len(INSTRUMENTS)


def test_grid_missing_prob_defaults_to_zero():
    """Non-zero signal with no matching prob_lookup entry => prediction == 0.0 (not NaN)."""
    sig = pd.DataFrame({
        "date": ["2022-01-03"],
        "es1s": [1],      # non-zero signal, no prob coverage
        "si1s": [0],
    })
    lookup = pd.DataFrame({"date": [], "instrument": [], "prob": []})
    result = predictions_grid(sig, lookup, date_start="2022-01-03", date_end="2022-01-03")
    es_row = result[result["instrument"] == "es1s"]
    assert float(es_row["prediction"].iloc[0]) == 0.0


def test_grid_sorted_by_date_instrument():
    """Output is sorted by (date, instrument)."""
    sig = _make_primary_signals()
    lookup = _make_prob_lookup(sig)
    result = predictions_grid(sig, lookup, date_start=DATES_ALL[0], date_end=DATES_ALL[4])
    expected_dates = result["date"].tolist()
    expected_inst = result["instrument"].tolist()
    pairs = list(zip(expected_dates, expected_inst))
    assert pairs == sorted(pairs)


def test_grid_prob_used_for_nonzero_signals():
    """A known prob value appears in the output for a matched non-zero signal."""
    sig = pd.DataFrame({
        "date": ["2022-01-03"],
        "es1s": [1],
        "si1s": [0],
    })
    lookup = pd.DataFrame({
        "date": ["2022-01-03"],
        "instrument": ["es1s"],
        "prob": [0.82],
    })
    result = predictions_grid(sig, lookup, date_start="2022-01-03", date_end="2022-01-03")
    es_row = result[result["instrument"] == "es1s"]
    assert float(es_row["prediction"].iloc[0]) == pytest.approx(0.82)
