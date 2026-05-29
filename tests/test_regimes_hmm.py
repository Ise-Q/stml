"""Causality + contract tests for F17 HMM regime posteriors
(:mod:`stml.metamodel.regime_features_hmm`).

The HMM is fit on FE-train ``(ret, vol)`` only and the **filtered** (forward-
only) posteriors are causal: the posterior at ``t`` uses only observations
``<= t``. The decisive test is that, for a FIXED frozen bundle, transforming the
full series vs a future-truncated series yields IDENTICAL posteriors on the
shared dates (a smoothed posterior would fail this).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.metamodel.regime_features_hmm import (
    HMM_COLUMNS,
    fit_hmm,
    transform_hmm,
)


def _synth_ret_vol(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Synthetic two-regime (ret, vol) series with a clear low/high-vol switch."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    # Alternating ~60-day vol regimes so the HMM has structure to find.
    block = 60
    hi = (np.arange(n) // block) % 2 == 1
    scale = np.where(hi, 0.03, 0.008)
    ret = rng.standard_normal(n) * scale
    vol = pd.Series(ret, index=idx).rolling(20).std() * np.sqrt(252)
    return pd.DataFrame({"ret": ret, "vol": vol.to_numpy()}, index=idx).dropna()


@pytest.fixture(scope="module")
def bundle_and_data():
    rv = _synth_ret_vol()
    boundary = rv.index[300]
    train = rv[rv.index <= boundary]
    bundle = fit_hmm(train, seed=0, instrument="synth", min_train=200)
    return bundle, rv, boundary


def test_fit_succeeds_and_records_train_index(bundle_and_data) -> None:
    bundle, _, boundary = bundle_and_data
    assert bundle.ok, "HMM fit should succeed on the synthetic two-regime series"
    assert bundle.n_states == 3
    assert len(bundle.train_index) > 0
    assert bundle.train_index.max() <= boundary


def test_transform_columns_and_simplex(bundle_and_data) -> None:
    bundle, rv, _ = bundle_and_data
    out = transform_hmm(bundle, rv)
    assert list(out.columns) == list(HMM_COLUMNS)
    probs = out[["f17_hmm_state_lo", "f17_hmm_state_mid", "f17_hmm_state_hi"]].dropna()
    # Each row of posteriors sums to 1 (a proper filtered posterior).
    row_sums = probs.sum(axis=1).to_numpy()
    assert np.allclose(row_sums, 1.0, atol=1e-9)
    argmax = out["f17_hmm_state_argmax"].dropna().unique()
    assert set(argmax).issubset({0.0, 1.0, 2.0})


def test_filtered_posteriors_are_causal(bundle_and_data) -> None:
    """A FIXED bundle's filtered posterior at t must not change when future rows
    are removed (forward-only filter; a smoothed posterior would fail)."""
    bundle, rv, _ = bundle_and_data
    cut = rv.index[420]
    full = transform_hmm(bundle, rv)
    trunc = transform_hmm(bundle, rv[rv.index <= cut])
    common = trunc.index[trunc.index < cut]
    assert len(common) > 50
    a = full.reindex(common)
    b = trunc.reindex(common)
    assert (a.isna() == b.isna()).all().all()
    diff = (a - b).abs().to_numpy(dtype=float)
    fin = diff[np.isfinite(diff)]
    assert fin.size and fin.max() <= 1e-9, "HMM filtered posterior leaked the future"


def test_failed_fit_yields_structural_nan() -> None:
    """Too few training rows -> ok=False bundle -> all-structural-NaN transform."""
    rv = _synth_ret_vol(n=120)
    bundle = fit_hmm(rv, seed=0, instrument="tiny", min_train=200)
    assert not bundle.ok
    out = transform_hmm(bundle, rv)
    assert list(out.columns) == list(HMM_COLUMNS)
    assert out.isna().all().all()
