"""
test_build_determinism.py
=========================
Determinism + persistence round-trip tests for the metamodel feature build
(US-FE-009, AC-11) on REAL data via :func:`stml.io.load_clean_data`.

These prove the build is reproducible and lossless without paying for the full
11-instrument fit: a FAST two-instrument subset (``es1s`` EQ + ``gc1s`` ME --
one equity, one metal, exercising both a per-instrument regime fit and a
class-pooled latent fit) is built twice from fresh :class:`FeaturePipeline`
objects with the same seed, then checked for frame-equality:

* The deterministic columns (everything except the float-valued fitted-transform
  outputs) must be **exactly** equal -- same column order, same dtypes, same
  values. This includes ``f4_cluster_id`` (the KMeans label is bit-exact).
* The float-valued fitted-transform columns -- the autoencoder-derived
  ``f4_ae_code*`` / ``f4_ae_recon_err`` and ``f4_cluster_dist`` (a Euclidean
  distance to the KMeans centroid, a float reduction over the scaled vector) --
  must reproduce within ``1e-10`` (CONTRACT_FE §3 / the AE determinism tolerance
  pinned in :func:`stml.metamodel.latent`). On CPU these reproduce to ~1e-15;
  the ``1e-10`` band absorbs the last-ULP non-associativity of the float sums.

A second test writes the matrix to ``tmp_path`` as parquet and re-reads it,
asserting the persisted artifact round-trips frame-equal (including the
``partition`` and ``fe_train_end_date`` provenance columns) so the on-disk
deliverable matches what the pipeline produced.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from stml.io import load_clean_data
from stml.metamodel.catalog import META_COLS
from stml.metamodel.pipeline import FeaturePipeline

# Fast subset: one equity + one metal (distinct asset classes so both a
# per-instrument regime fit and a class-pooled latent fit are exercised).
SUBSET = ["es1s", "gc1s"]

# Float-valued fitted-transform columns reproduce to a tight tolerance (not
# bit-exact): the AE codes / reconstruction error and the KMeans centroid
# distance are float reductions whose summation order is not bit-stable across
# independent fits. Every OTHER column (including the KMeans cluster id) must be
# exactly equal.
TOL_COLS = [
    "f4_ae_code1",
    "f4_ae_code2",
    "f4_ae_code3",
    "f4_ae_code4",
    "f4_ae_recon_err",
    "f4_cluster_dist",
]


@pytest.fixture(scope="module")
def data():
    """Real clean OHLCV + signals restricted to the fast two-instrument subset."""
    ohlcv, signals = load_clean_data()
    ohlcv_sub = ohlcv[ohlcv["instrument"].isin(SUBSET)].copy()
    signals_sub = signals[["date", *SUBSET]].copy()
    return ohlcv_sub, signals_sub


def _build(ohlcv: pd.DataFrame, signals: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Fresh fit + transform of the subset (EM-fit warnings silenced)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = FeaturePipeline(seed=seed).fit(ohlcv, signals)
        return pipe.transform(ohlcv, signals)


# --------------------------------------------------------------------------- #
# AC-11: rebuild -> frame-equality (deterministic cols exact, AE within 1e-10) #
# --------------------------------------------------------------------------- #
def test_rebuild_is_frame_equal(data) -> None:
    """Two fresh builds on the same data + seed produce an identical matrix:
    same column order and dtypes; deterministic columns exactly equal; the
    float-valued fitted-transform columns equal within ``1e-10``."""
    ohlcv_sub, signals_sub = data
    m1 = _build(ohlcv_sub, signals_sub, seed=0)
    m2 = _build(ohlcv_sub, signals_sub, seed=0)

    # Identical schema: same columns in the same order, same dtypes.
    assert list(m1.columns) == list(m2.columns)
    assert list(m1.dtypes) == list(m2.dtypes)
    assert len(m1) == len(m2)

    # Float-valued fitted-transform columns reproduce within 1e-10.
    tol_present = [c for c in TOL_COLS if c in m1.columns]
    assert tol_present, "expected fitted-transform float columns in the matrix"
    np.testing.assert_allclose(
        m1[tol_present].to_numpy(dtype=float),
        m2[tol_present].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-10,
    )

    # Every other column is EXACTLY equal (meta + all bit-exact feature columns,
    # including the KMeans cluster id and every engineered / regime feature).
    rest = [c for c in m1.columns if c not in tol_present]
    pd.testing.assert_frame_equal(
        m1[rest], m2[rest], check_exact=True, check_dtype=True
    )


def test_rebuild_meta_and_partition_identical(data) -> None:
    """The provenance / identity columns reproduce exactly across rebuilds."""
    ohlcv_sub, signals_sub = data
    m1 = _build(ohlcv_sub, signals_sub, seed=0)
    m2 = _build(ohlcv_sub, signals_sub, seed=0)

    for col in META_COLS:
        assert col in m1.columns
        pd.testing.assert_series_equal(m1[col], m2[col], check_exact=True)


# --------------------------------------------------------------------------- #
# Persistence round-trip: parquet write -> read is frame-equal                 #
# --------------------------------------------------------------------------- #
def test_parquet_round_trip(data, tmp_path) -> None:
    """Writing the matrix to parquet and re-reading it yields a frame-equal
    matrix (including the ``partition`` + ``fe_train_end_date`` columns)."""
    ohlcv_sub, signals_sub = data
    matrix = _build(ohlcv_sub, signals_sub, seed=0)

    dest = tmp_path / "feature_matrix.parquet"
    matrix.to_parquet(dest, index=False)
    reloaded = pd.read_parquet(dest)

    # Meta columns survive the round-trip in canonical leading order.
    assert list(reloaded.columns[:4]) == [
        "date",
        "instrument",
        "partition",
        "fe_train_end_date",
    ]

    # Whole-frame equality (parquet preserves dtypes for this matrix). reset the
    # index defensively so a stored range index does not trip the comparison.
    pd.testing.assert_frame_equal(
        matrix.reset_index(drop=True),
        reloaded.reset_index(drop=True),
        check_exact=True,
        check_dtype=True,
    )
