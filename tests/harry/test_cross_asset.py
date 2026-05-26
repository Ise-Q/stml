"""Unit tests for ``stml.harry.features.cross_asset``.

Universal causality / shape / no-NaN-past-warmup checks live in
``tests/harry/test_causality.py``. The tests here are hand-computed
correctness checks and per-feature property assertions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.harry.features.cross_asset import (
    ASSET_CLASSES,
    asset_class_dispersion_z,
    distance_to_lead_lag_centroid,
    ewma_implied_corr_z,
)


# --------------------------------------------------------------------------- #
# Synthetic helpers                                                            #
# --------------------------------------------------------------------------- #
def _panel(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {
            inst: rng.normal(0, 0.01, n)
            for inst in ASSET_CLASSES
        },
        index=dates,
    )


# --------------------------------------------------------------------------- #
# distance_to_lead_lag_centroid                                                #
# --------------------------------------------------------------------------- #
def test_distance_lead_lag_zero_when_instrument_equals_centroid():
    """If the instrument's returns equal the lag-shifted peer mean
    *exactly*, the L2 distance is 0."""
    dates = pd.date_range("2018-01-02", periods=20, freq="B")
    # Build a panel where the centroid (mean of peer returns) is a known
    # series; then set the instrument's returns equal to centroid.shift(-1)
    # so that ``inst[t] == centroid_lag1[t]``.
    peer_a = pd.Series(np.arange(20, dtype=float) / 100.0, index=dates)
    peer_b = pd.Series(np.arange(20, dtype=float) / 100.0, index=dates)
    centroid = (peer_a + peer_b) / 2
    inst = centroid.shift(-1)  # so inst[t] == centroid[t+1] → inst[t] equals
                                # the lag=1 shifted centroid (centroid[t-1])? No —
                                # we want inst[t] == centroid_lag1[t] == centroid[t-1].
    inst = centroid.shift(1)
    panel = pd.DataFrame({"target": inst, "peer_a": peer_a, "peer_b": peer_b})
    # With our function, centroid_t = mean(peer)[t-1]; here both peers are
    # identical to centroid, so centroid_lag1[t] = peer_a[t-1] = inst[t].
    # Distance must be 0.
    out = distance_to_lead_lag_centroid(panel, "target", lag=1, window=5)
    tail = out.dropna()
    assert (tail.abs() < 1e-12).all()


def test_distance_lead_lag_non_negative_on_random_panel():
    panel = _panel(n=300, seed=42)
    out = distance_to_lead_lag_centroid(
        panel, "es1s", lag=1, window=60,
    ).dropna()
    assert (out >= 0).all()


def test_distance_lead_lag_raises_on_missing_instrument():
    panel = _panel(n=50, seed=42)
    with pytest.raises(KeyError):
        distance_to_lead_lag_centroid(panel, "missing_ticker")


def test_distance_lead_lag_rejects_bad_inputs():
    panel = _panel(n=50, seed=42)
    with pytest.raises(ValueError):
        distance_to_lead_lag_centroid(panel, "es1s", lag=-1)
    with pytest.raises(ValueError):
        distance_to_lead_lag_centroid(panel, "es1s", window=1)


# --------------------------------------------------------------------------- #
# asset_class_dispersion_z                                                     #
# --------------------------------------------------------------------------- #
def test_asset_class_dispersion_hand_computed_three_member_class():
    """3-member class with returns ``[1, 2, 3]`` at every row → constant
    cross-section std → rolling z-score is NaN (std of constant is 0)."""
    n = 20
    dates = pd.date_range("2018-01-02", periods=n, freq="B")
    panel = pd.DataFrame(
        {"a1s": [0.01] * n, "a2s": [0.02] * n, "a3s": [0.03] * n,
         "b1s": [0.0] * n},
        index=dates,
    )
    classes = {"a1s": "A", "a2s": "A", "a3s": "A", "b1s": "B"}
    out = asset_class_dispersion_z(panel, "a1s", classes=classes, window=5)
    # All within-class dispersions are equal → rolling std = 0 → z = NaN.
    assert out.dropna().empty


def test_asset_class_dispersion_z_in_reasonable_range():
    panel = _panel(n=300, seed=42)
    out = asset_class_dispersion_z(
        panel, "es1s", window=60,
    ).dropna()
    # A rolling z-score should be near-normal — comfortably within +/- 5.
    assert (out.abs() < 5).all()


def test_asset_class_dispersion_nan_when_class_has_one_member():
    """If the class has only one member, dispersion is undefined → NaN
    everywhere."""
    panel = _panel(n=50, seed=42)
    classes = {inst: ("equity" if inst == "es1s" else "other")
               for inst in panel.columns}
    out = asset_class_dispersion_z(panel, "es1s", classes=classes, window=20)
    assert out.isna().all()


def test_asset_class_dispersion_raises_on_missing_instrument_or_class():
    panel = _panel(n=50, seed=42)
    with pytest.raises(KeyError):
        asset_class_dispersion_z(panel, "missing")
    with pytest.raises(KeyError):
        asset_class_dispersion_z(panel, "es1s", classes={"nq1s": "equity"})


# --------------------------------------------------------------------------- #
# ewma_implied_corr_z                                                          #
# --------------------------------------------------------------------------- #
def test_ewma_implied_corr_z_in_reasonable_range():
    panel = _panel(n=400, seed=42)
    out = ewma_implied_corr_z(
        panel, "es1s", halflife=20, window=100,
    ).dropna()
    # Z-score on a roughly stationary series → bounded.
    assert (out.abs() < 6).all()


def test_ewma_implied_corr_z_rejects_bad_inputs():
    panel = _panel(n=50, seed=42)
    with pytest.raises(ValueError):
        ewma_implied_corr_z(panel, "es1s", halflife=0)
    with pytest.raises(ValueError):
        ewma_implied_corr_z(panel, "es1s", window=1)


def test_ewma_implied_corr_z_raises_on_missing_instrument():
    panel = _panel(n=50, seed=42)
    with pytest.raises(KeyError):
        ewma_implied_corr_z(panel, "missing")


# --------------------------------------------------------------------------- #
# ASSET_CLASSES export                                                         #
# --------------------------------------------------------------------------- #
def test_asset_classes_covers_full_universe():
    assert set(ASSET_CLASSES.keys()) == {
        "es1s", "nq1s", "fesx1s",
        "cl1s", "ho1s", "rb1s", "ng1s",
        "gc1s", "si1s", "hg1s", "pl1s",
    }
    assert set(ASSET_CLASSES.values()) == {"equity", "energy", "metals"}
