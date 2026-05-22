"""
baselines.py
============
Naive predictors for the primary trading signal ``s_t in {-1, 0, +1}``.

Every predictor takes a 1-D array-like of integer labels and returns a
same-length ``np.ndarray`` of predicted labels. These exist so the
discrepancy metrics in :mod:`stml.replication.metrics` can be benchmarked
against trivially-uninformed references: a good replica must beat *all* of
these, and the ordinal skill-score uses ``always_flat`` / ``stratified_random``
as its zero points (see ``.omc/scratch/CONTRACT.md``, US-003a/b).

All functions are pure and side-effect-free; the only randomness lives in
:func:`stratified_random`, which is fully determined by its ``seed``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_int_array(y: np.ndarray | pd.Series | list[int]) -> np.ndarray:
    """Coerce a label container to a 1-D integer ``np.ndarray``."""
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D label array, got shape {arr.shape}")
    return arr.astype(int)


def always_flat(y: np.ndarray | pd.Series | list[int]) -> np.ndarray:
    """Predict ``0`` (no position) everywhere.

    The most naive reference: it never trades. On an imbalance-robust metric
    this must score at chance even when the true series is ~80% flat.
    """
    arr = _as_int_array(y)
    return np.zeros(arr.shape[0], dtype=int)


def majority_class(y: np.ndarray | pd.Series | list[int]) -> np.ndarray:
    """Predict the single most frequent label of ``y`` everywhere.

    Ties are broken toward the smallest label (``np.unique`` is sorted), which
    keeps the output deterministic.
    """
    arr = _as_int_array(y)
    values, counts = np.unique(arr, return_counts=True)
    mode = int(values[np.argmax(counts)])
    return np.full(arr.shape[0], mode, dtype=int)


def stratified_random(
    y: np.ndarray | pd.Series | list[int], seed: int = 0
) -> np.ndarray:
    """Draw labels i.i.d. from ``y``'s empirical class distribution.

    Reproducible: the same ``y`` and ``seed`` always yield the same output.
    Preserves the marginal label frequencies in expectation but carries no
    information about the ordering, so it sits at chance on every metric.
    """
    arr = _as_int_array(y)
    values, counts = np.unique(arr, return_counts=True)
    probs = counts / counts.sum()
    rng = np.random.default_rng(seed)
    return rng.choice(values, size=arr.shape[0], p=probs).astype(int)


def persistence(y: np.ndarray | pd.Series | list[int]) -> np.ndarray:
    """Predict the previous label: ``s_t = s_{t-1}`` with ``s_0 = 0``.

    A one-step lag of ``y``. The leading element has no predecessor and is
    filled with ``0`` (flat).
    """
    arr = _as_int_array(y)
    out = np.zeros(arr.shape[0], dtype=int)
    if arr.shape[0] > 1:
        out[1:] = arr[:-1]
    return out
