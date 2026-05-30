"""Unit tests for ``stml.harry.features.concept_drift``.

Universal causality / shape / no-NaN-past-warmup checks live in
``tests/harry/test_causality.py``. The tests here verify hand-built
scenarios where the regime alignment score must move in the right
direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.harry.features.concept_drift import (
    _resolve_train_end_pos,
    regime_alignment_score,
)


# --------------------------------------------------------------------------- #
# Train-end resolution                                                         #
# --------------------------------------------------------------------------- #
def test_resolve_train_end_pos_accepts_int():
    df = pd.DataFrame({"x": [0, 1, 2, 3]})
    assert _resolve_train_end_pos(df, 2) == 2


def test_resolve_train_end_pos_accepts_timestamp():
    dates = pd.date_range("2018-01-02", periods=4, freq="B")
    df = pd.DataFrame({"x": [0, 1, 2, 3]}, index=dates)
    assert _resolve_train_end_pos(df, dates[2]) == 2


def test_resolve_train_end_pos_rejects_negative_int():
    df = pd.DataFrame({"x": [0]})
    with pytest.raises(ValueError):
        _resolve_train_end_pos(df, -1)


# --------------------------------------------------------------------------- #
# Output bounds + warmup                                                       #
# --------------------------------------------------------------------------- #
def test_regime_alignment_score_in_zero_one():
    rng = np.random.default_rng(42)
    n = 300
    dates = pd.date_range("2018-01-02", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
        },
        index=dates,
    )
    out = regime_alignment_score(df, train_end=100, window=30, refit_every=15)
    tail = out.dropna()
    assert (tail >= 0).all()
    assert (tail <= 1.0).all()


def test_regime_alignment_score_nan_before_first_refit():
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.date_range("2018-01-02", periods=n, freq="B")
    df = pd.DataFrame(
        {"f1": rng.normal(0, 1, n)}, index=dates,
    )
    out = regime_alignment_score(df, train_end=100, window=30, refit_every=15)
    # First refit at 100 + 30 = 130, so rows 0..129 must be NaN.
    assert out.iloc[:130].isna().all()
    # Row 130 onwards: defined (as long as the feature is finite).
    assert out.iloc[130:].notna().all()


# --------------------------------------------------------------------------- #
# Behaviour: clear drift                                                       #
# --------------------------------------------------------------------------- #
def test_regime_alignment_score_high_for_distinct_recent():
    """If the post-train_end rows have markedly different features from
    the train pool, the alignment score for those rows should be high
    (close to 1) — the discriminator easily picks them out."""
    n = 300
    dates = pd.date_range("2018-01-02", periods=n, freq="B")
    train_end = 100
    rng = np.random.default_rng(0)
    f1 = np.concatenate([
        rng.normal(-2.0, 0.5, train_end),    # train regime
        rng.normal(+2.0, 0.5, n - train_end),  # recent regime
    ])
    df = pd.DataFrame({"f1": f1}, index=dates)
    out = regime_alignment_score(
        df, train_end=train_end, window=30, refit_every=15, seed=42,
    )
    # Past the first refit, predictions should be uniformly close to 1
    # on the recent-regime rows.
    tail = out.dropna()
    assert tail.mean() > 0.9


def test_regime_alignment_score_low_for_same_distribution():
    """If train and recent share the same distribution, the alignment
    score should be near 0.5 (no discriminative signal)."""
    n = 300
    dates = pd.date_range("2018-01-02", periods=n, freq="B")
    rng = np.random.default_rng(0)
    f1 = rng.normal(0, 1, n)
    df = pd.DataFrame({"f1": f1}, index=dates)
    out = regime_alignment_score(
        df, train_end=100, window=30, refit_every=15, seed=42,
    )
    tail = out.dropna()
    # Within reasonable bounds of 0.5 on a noisy classifier.
    assert 0.2 < tail.mean() < 0.8


def test_regime_alignment_score_determinism():
    """Same inputs → identical outputs."""
    n = 200
    dates = pd.date_range("2018-01-02", periods=n, freq="B")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"f1": rng.normal(0, 1, n)}, index=dates)
    a = regime_alignment_score(df, train_end=80, window=30, refit_every=15, seed=42)
    b = regime_alignment_score(df, train_end=80, window=30, refit_every=15, seed=42)
    pd.testing.assert_series_equal(a, b)


# --------------------------------------------------------------------------- #
# Input validation                                                             #
# --------------------------------------------------------------------------- #
def test_regime_alignment_score_rejects_bad_inputs():
    df = pd.DataFrame({"f1": [0.0] * 200})
    with pytest.raises(ValueError):
        regime_alignment_score(df, train_end=50, window=1)
    with pytest.raises(ValueError):
        regime_alignment_score(df, train_end=50, refit_every=0)


def test_regime_alignment_score_returns_all_nan_when_no_train_pool():
    df = pd.DataFrame({"f1": [0.0] * 200})
    # train_end too small → train pool < MIN_RECENT.
    out = regime_alignment_score(df, train_end=5, window=30)
    assert out.isna().all()
