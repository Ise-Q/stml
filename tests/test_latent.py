"""Tests for ``stml.metamodel.latent`` (F4: PCA + KMeans + dense AE).

Covers the feature-engineering acceptance criteria for the pooled latent stack:

* **AC-4** -- ``fit_latent`` then ``transform_latent`` emits the contracted
  columns (``f4_pc1..4``, ``f4_cluster_id``, ``f4_cluster_dist``,
  ``f4_ae_code1..4``, ``f4_ae_recon_err``) and ``recon_mse`` has finite
  ``pca_k4`` / ``ae_k4`` keys.
* **AC-11** -- determinism: refit on the same seed + data, then transform, and
  the AE-derived columns match within ``1e-10`` (PCA columns are frame-equal).
* **AC-12** -- per-instrument-series: ``transform_latent`` on one instrument
  alone equals (within ``1e-8``) transforming a two-instrument concatenation and
  slicing that instrument's rows -- proving the transform is row-wise, not
  panel-coupled.

The fixtures are SEEDED SYNTHETIC numeric feature frames with a low-rank
structure plus noise (no dependence on ``features.py``, which is built in
parallel): a pooled FE-train block of ~600 rows x ~30 cols and two
per-instrument blocks drawn from the same generative model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.metamodel.latent import LatentBundle, fit_latent, transform_latent

K = 4
N_FEATURES = 30
N_POOLED = 600
N_INST_A = 120
N_INST_B = 90
LATENT_RANK = 5

PCA_COLS = [f"f4_pc{j + 1}" for j in range(K)]
AE_CODE_COLS = [f"f4_ae_code{j + 1}" for j in range(K)]
EXPECTED_COLS = (
    PCA_COLS + ["f4_cluster_id", "f4_cluster_dist"] + AE_CODE_COLS + ["f4_ae_recon_err"]
)
FEATURE_COLS = [f"feat_{i:02d}" for i in range(N_FEATURES)]


def _low_rank_frame(
    n_rows: int, seed: int, start: str = "2020-01-03"
) -> pd.DataFrame:
    """A seeded low-rank-plus-noise numeric feature frame (date-indexed).

    ``rank``-dimensional latent factors are projected up to ``N_FEATURES`` and
    perturbed with small Gaussian noise, giving a matrix whose intrinsic
    dimensionality (~5) is well below ``N_FEATURES`` -- so PCA(4) and the AE
    have real structure to compress.
    """
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((n_rows, LATENT_RANK))
    loadings = rng.standard_normal((LATENT_RANK, N_FEATURES))
    noise = 0.1 * rng.standard_normal((n_rows, N_FEATURES))
    mat = factors @ loadings + noise
    index = pd.bdate_range(start=start, periods=n_rows, name="date")
    return pd.DataFrame(mat, columns=FEATURE_COLS, index=index)


@pytest.fixture(scope="module")
def pooled_block() -> pd.DataFrame:
    """Pooled FE-train block (~600 x 30) tagged with an asset class."""
    block = _low_rank_frame(N_POOLED, seed=11)
    block.attrs["asset_class"] = "EQ"
    return block


@pytest.fixture(scope="module")
def inst_a() -> pd.DataFrame:
    """One instrument's feature rows (distinct dates from inst_b)."""
    return _low_rank_frame(N_INST_A, seed=101, start="2021-08-02")


@pytest.fixture(scope="module")
def inst_b() -> pd.DataFrame:
    """A second instrument's feature rows (date-disjoint from inst_a).

    Started well after inst_a's 120-bday span ends so the AC-12 concat-then-
    slice via ``.loc[inst_a.index]`` selects exactly inst_a's rows (no
    duplicate-date label matches across the two synthetic instruments).
    """
    return _low_rank_frame(N_INST_B, seed=202, start="2023-01-02")


@pytest.fixture(scope="module")
def bundle(pooled_block: pd.DataFrame) -> LatentBundle:
    return fit_latent(pooled_block, k=K, seed=0)


# --------------------------------------------------------------------------- #
# AC-4: output columns + recon_mse dict.                                      #
# --------------------------------------------------------------------------- #
def test_transform_emits_contract_columns(
    bundle: LatentBundle, inst_a: pd.DataFrame
) -> None:
    out = transform_latent(bundle, inst_a)
    assert list(out.columns) == EXPECTED_COLS
    assert len(out) == len(inst_a)
    assert out.index.equals(inst_a.index)
    # Every emitted value is finite (no NaN/inf leaked through the transform).
    assert np.isfinite(out.to_numpy(dtype=float)).all()


def test_cluster_id_in_range(bundle: LatentBundle, inst_a: pd.DataFrame) -> None:
    out = transform_latent(bundle, inst_a)
    ids = out["f4_cluster_id"].to_numpy()
    assert set(np.unique(ids)).issubset(set(range(K)))
    assert (out["f4_cluster_dist"].to_numpy() >= 0).all()


def test_recon_mse_has_both_keys_finite(bundle: LatentBundle) -> None:
    assert set(bundle.recon_mse) == {"pca_k4", "ae_k4"}
    assert np.isfinite(bundle.recon_mse["pca_k4"])
    assert np.isfinite(bundle.recon_mse["ae_k4"])
    assert bundle.recon_mse["pca_k4"] > 0
    assert bundle.recon_mse["ae_k4"] > 0


def test_bundle_records_class_and_k(bundle: LatentBundle) -> None:
    assert bundle.asset_class == "EQ"
    assert bundle.k == K


# --------------------------------------------------------------------------- #
# AC-11: determinism (refit -> AE cols within 1e-10, PCA frame-equal).        #
# --------------------------------------------------------------------------- #
def test_determinism_refit_matches(
    pooled_block: pd.DataFrame, inst_a: pd.DataFrame
) -> None:
    b1 = fit_latent(pooled_block, k=K, seed=0)
    b2 = fit_latent(pooled_block, k=K, seed=0)
    out1 = transform_latent(b1, inst_a)
    out2 = transform_latent(b2, inst_a)

    # AE-derived columns must reproduce to a very tight tolerance.
    ae_cols = AE_CODE_COLS + ["f4_ae_recon_err"]
    np.testing.assert_allclose(
        out1[ae_cols].to_numpy(), out2[ae_cols].to_numpy(), atol=1e-10, rtol=0.0
    )
    # recon_mse reproduces too.
    assert abs(b1.recon_mse["ae_k4"] - b2.recon_mse["ae_k4"]) < 1e-10

    # PCA columns are frame-equal (deterministic by construction).
    pd.testing.assert_frame_equal(out1[PCA_COLS], out2[PCA_COLS])
    # Cluster assignments are identical too.
    pd.testing.assert_series_equal(out1["f4_cluster_id"], out2["f4_cluster_id"])


# --------------------------------------------------------------------------- #
# AC-12: per-instrument-series (concat-then-slice == transform-alone).        #
# --------------------------------------------------------------------------- #
def test_transform_is_row_wise_not_panel_coupled(
    bundle: LatentBundle, inst_a: pd.DataFrame, inst_b: pd.DataFrame
) -> None:
    alone = transform_latent(bundle, inst_a)

    panel = pd.concat([inst_a, inst_b])
    panel_out = transform_latent(bundle, panel)
    sliced = panel_out.loc[inst_a.index]

    # Row-wise transform: instA's rows are unaffected by instB being present.
    np.testing.assert_allclose(
        alone.to_numpy(dtype=float),
        sliced.to_numpy(dtype=float),
        atol=1e-8,
        rtol=0.0,
    )


# --------------------------------------------------------------------------- #
# Bundle invariants: train_index, frozen feature_cols, NaN -> frozen median.  #
# --------------------------------------------------------------------------- #
def test_train_index_subset_of_given_index(
    bundle: LatentBundle, pooled_block: pd.DataFrame
) -> None:
    assert bundle.train_index.isin(pooled_block.index).all()
    assert len(bundle.train_index) == len(pooled_block)


def test_feature_cols_frozen_order(bundle: LatentBundle) -> None:
    assert bundle.feature_cols == FEATURE_COLS
    assert len(bundle.impute_median) == len(FEATURE_COLS)


def test_nan_cell_uses_frozen_median(
    bundle: LatentBundle, inst_a: pd.DataFrame
) -> None:
    # Punch a NaN into one cell; the transform must impute the frozen train
    # median for that column rather than crash or propagate NaN.
    holed = inst_a.copy()
    holed.iloc[0, 0] = np.nan
    out = transform_latent(bundle, holed)
    assert np.isfinite(out.to_numpy(dtype=float)).all()

    # The imputed row equals the row obtained by manually substituting the
    # frozen median for that NaN cell (proves the median, not 0/ffill, is used).
    manual = inst_a.copy()
    manual.iloc[0, 0] = bundle.impute_median[0]
    expected = transform_latent(bundle, manual)
    np.testing.assert_allclose(
        out.to_numpy(dtype=float), expected.to_numpy(dtype=float), atol=1e-10
    )


def test_transform_reindexes_to_frozen_cols(
    bundle: LatentBundle, inst_a: pd.DataFrame
) -> None:
    # Shuffle + drop a column on input; reindex-to-feature_cols must still
    # produce the full contract output (missing col -> all-NaN -> frozen median).
    shuffled = inst_a[FEATURE_COLS[::-1]].drop(columns=[FEATURE_COLS[5]])
    out = transform_latent(bundle, shuffled)
    assert list(out.columns) == EXPECTED_COLS
    assert np.isfinite(out.to_numpy(dtype=float)).all()
