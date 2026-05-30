"""Unit tests for ``stml.harry.features.information_theoretic``.

Truncation-invariance, shape, and no-NaN-after-warmup are covered by
``tests/harry/test_causality.py``. The tests here are hand-computed
correctness checks and property assertions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.harry.features.information_theoretic import (
    _mutual_information,
    _shannon_entropy,
    rolling_mutual_information_252d,
    transfer_entropy_vol_to_signal_acc,
)


# --------------------------------------------------------------------------- #
# Hand-computed: entropy + MI primitives                                       #
# --------------------------------------------------------------------------- #
def test_shannon_entropy_uniform_two_bins():
    """Entropy of fair coin = log 2."""
    p = np.array([0.5, 0.5])
    assert _shannon_entropy(p) == pytest.approx(np.log(2))


def test_shannon_entropy_concentrated():
    """Entropy of point mass = 0."""
    p = np.array([1.0, 0.0, 0.0])
    assert _shannon_entropy(p) == pytest.approx(0.0)


def test_mutual_information_independent_variables():
    """X and Y independent → MI ≈ 0."""
    rng = np.random.default_rng(42)
    n = 5_000
    x = rng.integers(0, 3, size=n)
    y = rng.integers(0, 3, size=n)
    mi = _mutual_information(x.astype(np.int64), y.astype(np.int64))
    assert mi < 0.02  # numerical noise on 5k samples


def test_mutual_information_perfectly_correlated():
    """X == Y → MI = H(X) = log(n_categories)."""
    rng = np.random.default_rng(42)
    n = 5_000
    x = rng.integers(0, 3, size=n).astype(np.int64)
    y = x.copy()
    mi = _mutual_information(x, y)
    # H(X) for a roughly uniform 3-cat dist ≈ log 3.
    assert mi == pytest.approx(np.log(3), abs=0.02)


# --------------------------------------------------------------------------- #
# rolling_mutual_information_252d — sanity                                     #
# --------------------------------------------------------------------------- #
def test_rolling_mi_independent_inputs_near_zero():
    """Independent inputs → rolling MI hovers near zero."""
    rng = np.random.default_rng(42)
    n = 400
    x = pd.Series(rng.choice([-1, 0, 1], size=n))
    y = pd.Series(rng.normal(0, 0.01, n))
    mi = rolling_mutual_information_252d(x, y, window=200, n_bins=5)
    tail = mi.dropna()
    assert (tail >= -1e-12).all()  # non-negative
    assert tail.mean() < 0.10  # well below log(5) ≈ 1.61


def test_rolling_mi_constant_window_is_zero():
    """If one input is constant within the trailing window, MI must be 0."""
    n = 200
    x = pd.Series([1.0] * n)  # constant
    y = pd.Series(np.linspace(-1.0, 1.0, n))
    mi = rolling_mutual_information_252d(x, y, window=100, n_bins=5)
    tail = mi.dropna()
    assert (tail == 0.0).all()


def test_rolling_mi_strongly_dependent_inputs():
    """Strongly dependent inputs (y = x + tiny noise) → high MI."""
    rng = np.random.default_rng(42)
    n = 400
    x = pd.Series(rng.normal(0, 1, n))
    y = pd.Series(x + rng.normal(0, 0.05, n))  # noisy copy
    mi = rolling_mutual_information_252d(x, y, window=200, n_bins=5)
    tail = mi.dropna()
    assert (tail > 0.5).all()  # well above the independence baseline


def test_rolling_mi_rejects_bad_inputs():
    x = pd.Series([1.0] * 100)
    y = pd.Series([1.0] * 100)
    with pytest.raises(ValueError):
        rolling_mutual_information_252d(x, y, window=1)
    with pytest.raises(ValueError):
        rolling_mutual_information_252d(x, y, n_bins=1)


# --------------------------------------------------------------------------- #
# transfer_entropy_vol_to_signal_acc — sanity                                  #
# --------------------------------------------------------------------------- #
def test_transfer_entropy_independent_vol_near_zero():
    """Vol independent of accuracy → TE ≈ 0."""
    rng = np.random.default_rng(42)
    n = 500
    s = pd.Series(rng.choice([-1, 1], size=n))
    r = pd.Series(rng.normal(0, 0.01, n))
    vol = pd.Series(rng.lognormal(mean=np.log(0.01), sigma=0.2, size=n))
    te = transfer_entropy_vol_to_signal_acc(
        vol, s, r, window=200, n_bins=3,
    )
    tail = te.dropna()
    assert (tail >= -1e-12).all()
    assert tail.mean() < 0.10  # well below log(2) ≈ 0.69


def test_transfer_entropy_non_negative():
    """TE is non-negative by construction (after the FP-clip in the
    implementation)."""
    rng = np.random.default_rng(42)
    n = 400
    s = pd.Series(rng.choice([-1, 1], size=n))
    r = pd.Series(rng.normal(0, 0.01, n))
    vol = pd.Series(rng.lognormal(mean=np.log(0.01), sigma=0.2, size=n))
    te = transfer_entropy_vol_to_signal_acc(vol, s, r, window=150, n_bins=3)
    tail = te.dropna()
    assert (tail >= 0).all()


def test_transfer_entropy_rejects_bad_inputs():
    n = 50
    s = pd.Series(np.zeros(n))
    r = pd.Series(np.zeros(n))
    vol = pd.Series(np.zeros(n))
    with pytest.raises(ValueError):
        transfer_entropy_vol_to_signal_acc(vol, s, r, window=3)
    with pytest.raises(ValueError):
        transfer_entropy_vol_to_signal_acc(vol, s, r, n_bins=1)
