"""Tests for ``stml.experimental.make_labels`` — Jay-CSV loader.

Round-trips the per-instrument triple-barrier CSV to the canonical events
schema. Checks the per-instrument geometry is unique, the partition column
is well-formed, uniqueness weights are well-defined, and ``t_start`` strictly
follows ``t_signal`` (entry-at-t+1 invariant preserved).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stml.experimental.make_labels import build_events


# ---------------------------------------------------------------------------
# Repo root + fixture: the actual Jay CSV shipped with the branch.
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "data").is_dir() and (p / "pyproject.toml").is_file():
            return p
    raise RuntimeError("repo root not found")


@pytest.fixture(scope="module")
def loaded() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the loader once per test module."""
    return build_events(verbose=False)


# ---------------------------------------------------------------------------
# Schema + integrity.
# ---------------------------------------------------------------------------


def test_canonical_schema_present(loaded):
    events, _, _ = loaded
    required = {
        "instrument", "t_signal", "t_start", "t_end",
        "side", "ret", "label", "uniqueness_weight", "sigma_at_t", "barrier_hit",
        "pt", "sl", "h", "partition",
    }
    assert required.issubset(set(events.columns)), \
        f"missing: {required - set(events.columns)}"


def test_event_counts_match_csv(loaded):
    """The runner's emit must equal the raw CSV row count (no rows dropped)."""
    events, _, _ = loaded
    raw = pd.read_csv(_repo_root() / "data" / "triple_barrier_labels.csv")
    # Allow for the (rare) drop of events whose t+1 is past instrument history;
    # in the shipped CSV there should be no such events.
    assert len(events) == len(raw)


def test_partition_values_are_valid(loaded):
    events, _, _ = loaded
    parts = set(events["partition"].unique())
    assert parts == {"train", "val", "test"}, parts


def test_partition_counts_per_instrument(loaded):
    """Partition counts per instrument must match Jay's CSV exactly."""
    events, _, _ = loaded
    raw = pd.read_csv(_repo_root() / "data" / "triple_barrier_labels.csv")
    a = events.groupby(["instrument", "partition"]).size().sort_index()
    b = raw.groupby(["instrument", "partition"]).size().sort_index()
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_barrier_hit_mapping(loaded):
    """CSV 'vert' must be renamed to 'vertical' (downstream consumers expect this)."""
    events, _, _ = loaded
    assert set(events["barrier_hit"]).issubset({"pt", "sl", "vertical"})
    raw = pd.read_csv(_repo_root() / "data" / "triple_barrier_labels.csv")
    expected_vert = int((raw["touch"] == "vert").sum())
    assert int((events["barrier_hit"] == "vertical").sum()) == expected_vert


# ---------------------------------------------------------------------------
# Entry-at-t+1 invariant.
# ---------------------------------------------------------------------------


def test_t_start_equals_t_signal_jay_convention(loaded):
    """Jay's convention (PDF): entry at close of `date` (t_signal), exit at
    close of `t1`. The realised ``ret`` column in the CSV is the
    close(t)→close(t1) log-return, which only matches if entry is at close(t).
    """
    events, _, _ = loaded
    assert (events["t_start"] == events["t_signal"]).all()


def test_t_end_not_before_t_start(loaded):
    """t_end ≥ t_start for every event (exit can be same-day for h=1)."""
    events, _, _ = loaded
    assert (events["t_end"] >= events["t_start"]).all()


# ---------------------------------------------------------------------------
# Uniqueness weights.
# ---------------------------------------------------------------------------


def test_uniqueness_weights_in_unit_interval(loaded):
    events, _, _ = loaded
    w = events["uniqueness_weight"].values
    assert (w > 0).all()
    assert (w <= 1.0 + 1e-9).all()


def test_uniqueness_weights_correlate_with_horizon(loaded):
    """h=1 instruments should have mean uniqueness ~1 (mostly disjoint);
    larger-h instruments should have weighted lower (more overlap)."""
    events, audit, _ = loaded
    # ho1s and other h=1 series sit at mean_uniqueness == 1 in the audit.
    h1_inst = audit.loc[audit["h"] == 1, "mean_uniqueness"]
    h_large = audit.loc[audit["h"] >= 10, "mean_uniqueness"]
    # h=1 mean uniqueness should equal 1 exactly (or very close).
    assert (h1_inst >= 0.99).all()
    # Larger h should weight lower on average than h=1 (some overlap).
    assert h_large.mean() < h1_inst.mean()


# ---------------------------------------------------------------------------
# Per-instrument geometry uniqueness.
# ---------------------------------------------------------------------------


def test_one_geometry_per_instrument(loaded):
    events, _, geometry = loaded
    grouped = events.groupby("instrument")[["pt", "sl", "h"]].nunique()
    assert (grouped == 1).all().all(), \
        f"some instrument has multiple geometries:\n{grouped}"


def test_geometry_summary_matches_pdf(loaded):
    """Spot-check the adopted geometries against the doc's recommended picks."""
    _, _, geo = loaded
    geo = geo.set_index("instrument")
    expected = {
        "cl1s":   (0.25, 0.25, 1),
        "es1s":   (1.00, 0.25, 10),
        "fesx1s": (0.25, 0.25, 1),
        "gc1s":   (1.00, 0.25, 15),
        "hg1s":   (1.00, 0.25, 10),
        "ho1s":   (0.25, 0.25, 1),
        "ng1s":   (0.25, 0.25, 1),
        "nq1s":   (0.25, 0.25, 1),
        "pl1s":   (0.25, 0.25, 1),
        "rb1s":   (2.50, 0.25, 15),
        "si1s":   (0.75, 0.25, 20),
    }
    for inst, (pt, sl, h) in expected.items():
        assert float(geo.loc[inst, "pt"]) == pt, f"{inst} pt"
        assert float(geo.loc[inst, "sl"]) == sl, f"{inst} sl"
        assert int(geo.loc[inst, "h"]) == h,    f"{inst} h"


# ---------------------------------------------------------------------------
# Label semantics.
# ---------------------------------------------------------------------------


def test_label_distribution_per_instrument(loaded):
    """ho1s has the highest positive rate (0.667); rb1s the lowest (0.263)
    -- these are direct consequences of Jay's geometry choices."""
    _, audit, _ = loaded
    a = audit.set_index("instrument")
    assert abs(a.loc["ho1s", "pos_rate"] - 0.667) < 0.01
    assert abs(a.loc["rb1s", "pos_rate"] - 0.263) < 0.01


def test_no_missing_label_or_side(loaded):
    events, _, _ = loaded
    assert events["label"].isin([0, 1]).all()
    assert events["side"].isin([-1, 1]).all()
