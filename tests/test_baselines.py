"""Unit tests for :mod:`stml.replication.baselines`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.replication.baselines import (
    always_flat,
    majority_class,
    persistence,
    stratified_random,
)

_PREDICTORS = [always_flat, majority_class, persistence]


@pytest.fixture
def y() -> np.ndarray:
    """A mixed {-1, 0, +1} series with a clear majority (+1)."""
    return np.array([1, 1, 0, -1, 1, 1, 0, 1, -1, 1])


@pytest.mark.parametrize("predictor", _PREDICTORS)
def test_shape_matches_input_length(predictor, y) -> None:
    out = predictor(y)
    assert isinstance(out, np.ndarray)
    assert out.shape == y.shape


def test_stratified_random_shape(y) -> None:
    out = stratified_random(y, seed=0)
    assert isinstance(out, np.ndarray)
    assert out.shape == y.shape


def test_always_flat_is_all_zero(y) -> None:
    out = always_flat(y)
    assert np.all(out == 0)


def test_majority_class_returns_the_mode(y) -> None:
    # +1 occurs 6 times, more than any other label.
    out = majority_class(y)
    assert np.all(out == 1)
    assert out.shape == y.shape


def test_majority_class_flat_dominant() -> None:
    y = np.array([0, 0, 0, 1, -1])
    assert np.all(majority_class(y) == 0)


def test_persistence_is_one_step_shift_with_leading_zero(y) -> None:
    out = persistence(y)
    assert out[0] == 0
    np.testing.assert_array_equal(out[1:], y[:-1])


def test_persistence_single_element() -> None:
    assert np.array_equal(persistence(np.array([1])), np.array([0]))


def test_stratified_random_reproducible(y) -> None:
    a = stratified_random(y, seed=0)
    b = stratified_random(y, seed=0)
    np.testing.assert_array_equal(a, b)


def test_stratified_random_seed_changes_output(y) -> None:
    a = stratified_random(y, seed=0)
    b = stratified_random(y, seed=1)
    # Different seeds should (with overwhelming probability) differ on n=10.
    assert not np.array_equal(a, b)


def test_stratified_random_only_emits_observed_labels() -> None:
    y = np.array([-1, -1, 0, 0, 0])  # no +1 present
    out = stratified_random(y, seed=0)
    assert set(np.unique(out)).issubset({-1, 0})


@pytest.mark.parametrize("predictor", [*_PREDICTORS, stratified_random])
def test_accepts_pandas_series(predictor, y) -> None:
    s = pd.Series(y)
    out = predictor(s)
    assert isinstance(out, np.ndarray)
    assert out.shape == (len(s),)


@pytest.mark.parametrize("predictor", _PREDICTORS)
def test_output_dtype_is_integer(predictor, y) -> None:
    assert np.issubdtype(predictor(y).dtype, np.integer)
