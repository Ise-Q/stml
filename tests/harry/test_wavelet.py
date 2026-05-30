"""Unit tests for ``stml.harry.features.wavelet``.

Universal causality / shape / no-NaN-past-warmup checks live in
``tests/harry/test_causality.py``. The tests here check the
mathematical properties of the MRA energy decomposition on hand-built
signals where the energy is known to live in a specific band.

Skipped cleanly when PyWavelets is missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pywt")

from stml.harry.features.wavelet import mra_energy_bands  # noqa: E402


# --------------------------------------------------------------------------- #
# Schema                                                                       #
# --------------------------------------------------------------------------- #
def test_mra_energy_bands_returns_levels_columns():
    rng = np.random.default_rng(42)
    n = 400
    r = pd.Series(rng.normal(0, 0.01, n))
    out = mra_energy_bands(r, levels=5, window=128)
    assert out.shape == (n, 5)
    assert list(out.columns) == [f"mra_energy_D{k}" for k in range(1, 6)]


def test_mra_energy_rows_in_zero_one():
    """Each level's fraction must be in [0, 1] and the row sum in [0, 1]
    (the missing mass lives in the approximation coefficient that we
    intentionally exclude)."""
    rng = np.random.default_rng(42)
    n = 400
    r = pd.Series(rng.normal(0, 0.01, n))
    out = mra_energy_bands(r, levels=5, window=128).dropna()
    assert (out >= 0).all().all()
    assert (out <= 1.0).all().all()
    row_sums = out.sum(axis=1)
    assert (row_sums >= 0).all()
    assert (row_sums <= 1.0 + 1e-9).all()


# --------------------------------------------------------------------------- #
# Signal-localised tests                                                       #
# --------------------------------------------------------------------------- #
def test_mra_energy_concentrated_at_low_frequency():
    """A slow ramp (zero high-frequency content) puts energy mostly in
    the COARSEST bands (highest level numbers we keep)."""
    n = 300
    # Long-period sine wave: period = window; one cycle inside the
    # trailing window. Period 128 is the lowest detail band.
    window = 128
    t = np.arange(n)
    r = pd.Series(np.sin(2 * np.pi * t / window))
    out = mra_energy_bands(r, levels=5, window=window).dropna()
    last = out.iloc[-1]
    # The highest-level (coarsest) detail bands should dominate.
    assert (last["mra_energy_D5"] + last["mra_energy_D4"]) > (
        last["mra_energy_D1"] + last["mra_energy_D2"]
    )


def test_mra_energy_concentrated_at_high_frequency():
    """A bar-by-bar alternating series ``[+1, -1, +1, ...]`` puts essentially
    all energy at the FINEST detail band (D1)."""
    n = 300
    window = 128
    r = pd.Series(np.array([1.0, -1.0] * (n // 2)))
    out = mra_energy_bands(r, levels=5, window=window).dropna()
    last = out.iloc[-1]
    # D1 captures the alternation; other bands are negligible.
    assert last["mra_energy_D1"] > 0.95
    assert last[[f"mra_energy_D{k}" for k in (2, 3, 4, 5)]].sum() < 0.05


# --------------------------------------------------------------------------- #
# Input validation                                                             #
# --------------------------------------------------------------------------- #
def test_mra_energy_rejects_levels_zero():
    with pytest.raises(ValueError):
        mra_energy_bands(pd.Series([0.0] * 64), levels=0, window=32)


def test_mra_energy_rejects_window_too_small_for_levels():
    """``window`` must be >= 2^levels."""
    with pytest.raises(ValueError):
        mra_energy_bands(pd.Series([0.0] * 64), levels=5, window=10)
