"""Causality + contract tests for F16 concept-drift / regime-alignment
(:mod:`stml.metamodel.drift_features`).

The rolling discriminator refits only on rows ``<= t_r`` and scores the next
``refit_every`` rows, so the score at ``t`` depends only on data ``<= t``. The
per-refit subsample is seeded from the positional refit index, so truncating the
FUTURE leaves already-scored rows byte-stable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.metamodel.drift_features import DRIFT_COLUMN, regime_alignment_score


def _synth_features(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Synthetic multi-feature panel with a regime shift past the midpoint."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    base = rng.standard_normal((n, 4))
    # Shift the mean of the second half so a discriminator has signal.
    base[n // 2 :] += 1.5
    return pd.DataFrame(base, index=idx, columns=[f"x{i}" for i in range(4)])


def test_output_in_unit_interval_or_nan() -> None:
    feats = _synth_features()
    s = regime_alignment_score(feats, train_end=150, window=30, refit_every=15, seed=42)
    assert s.name == DRIFT_COLUMN
    finite = s.dropna()
    assert len(finite) > 0, "expected some scored rows past the first refit"
    assert (finite >= 0.0).all() and (finite <= 1.0).all()
    # Warm-up rows before the first refit (train_end + window) are NaN.
    assert s.iloc[: 150 + 30].isna().all()


def test_truncation_invariant() -> None:
    """Scores at rows < T must not change when future rows are removed."""
    feats = _synth_features()
    cut = feats.index[330]
    full = regime_alignment_score(feats, train_end=150, window=30, refit_every=15, seed=42)
    trunc = regime_alignment_score(
        feats[feats.index <= cut], train_end=150, window=30, refit_every=15, seed=42
    )
    common = trunc.index[trunc.index < cut]
    a, b = full.reindex(common), trunc.reindex(common)
    assert (a.isna() == b.isna()).all(), "NaN pattern changed after truncation"
    diff = (a - b).abs().dropna()
    assert diff.empty or diff.max() <= 1e-12, "drift score leaked the future"


def test_detects_a_regime_shift() -> None:
    """The mean-shifted second half should score higher (more 'recent') than the
    training era on average — a sanity check the discriminator learns something."""
    feats = _synth_features()
    s = regime_alignment_score(feats, train_end=200, window=30, refit_every=15, seed=42)
    late = s.dropna()
    assert late.mean() > 0.5


def test_all_nan_column_does_not_wipe_output() -> None:
    """A fully-NaN feature column must not force every row to drop out."""
    feats = _synth_features()
    feats["dead"] = np.nan
    s = regime_alignment_score(feats, train_end=150, window=30, refit_every=15, seed=42)
    assert s.dropna().size > 0
