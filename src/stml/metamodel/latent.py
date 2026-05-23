"""
latent.py
=========
Unsupervised latent-structure features (F4) for the triple-barrier metamodel.

This is a TF-class (fitted) feature group: the transforms are *fit on the
FE-train partition only* and then applied causally, frozen, forward. Unlike the
per-instrument F3 regime models, F4 is **pooled within an asset class** -- the
scaler, PCA, KMeans and autoencoder all see the FE-train engineered-feature
vectors of every instrument in the class stacked together, so the latent axes
describe the class as a whole. The fitted bundle is then applied **one
instrument-series at a time** (never on a concatenated multi-instrument panel),
which keeps the transform row-wise and avoids the C1 cross-instrument artifact.

Four complementary latent views, all on the FROZEN, scaled feature matrix:

* **PCA** -- linear principal axes (``f4_pc1..f4_pc{k}``); the standard
  variance-maximizing reduction, and the reconstruction baseline the
  autoencoder is benchmarked against.
* **KMeans** -- a hard clustering of the scaled vectors (``f4_cluster_id`` plus
  ``f4_cluster_dist``, the Euclidean distance to the assigned centroid -- a
  cheap "how typical is today" score).
* **Dense autoencoder** -- a shallow ``n_in -> 8 -> k -> 8 -> n_in`` MSE
  autoencoder (``f4_ae_code1..f4_ae_code{k}`` from the bottleneck, plus
  ``f4_ae_recon_err``, the per-row reconstruction MSE -- a nonlinear novelty
  score). Trained FULL-BATCH (no shuffling) with a fixed seed so it is bit-for-
  bit reproducible on CPU.

Leakage / determinism contract (see ``.omc/scratch/CONTRACT_FE.md`` §0, §3)
---------------------------------------------------------------------------
* Every fitted object (``scaler``, ``pca``, ``kmeans``, ``ae_state``) and the
  per-column impute medians (``impute_median``) are frozen from FE-train and
  recorded in the :class:`LatentBundle`; nothing is ever recomputed on the full
  series at transform time.
* NaN cells in the feature matrix are imputed with the FROZEN per-column train
  MEDIAN before scaling. This is feature-matrix imputation for an unsupervised
  learner (the bundle's frozen statistic), **not** a time-series ffill -- the
  persisted feature matrix elsewhere keeps its structural NaNs.
* The autoencoder fit pins ``torch.use_deterministic_algorithms(True)``,
  ``torch.set_num_threads(1)``, ``torch.manual_seed(seed)`` and
  ``np.random.seed(seed)`` so a refit on the same data + seed reproduces the AE
  weights (and hence ``f4_ae_*``) within ``1e-10``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn

__all__ = [
    "LatentBundle",
    "DenseAutoencoder",
    "fit_latent",
    "transform_latent",
]


class DenseAutoencoder(nn.Module):
    """Shallow dense autoencoder ``n_in -> 8 -> k -> 8 -> n_in`` (MSE).

    The encoder compresses an ``n_in``-dimensional scaled feature vector through
    a single ``ReLU`` hidden layer of width 8 down to a ``k``-dimensional code
    (the bottleneck); the symmetric decoder expands it back to ``n_in``. The
    code feeds ``f4_ae_code1..f4_ae_code{k}`` and the squared reconstruction
    error feeds ``f4_ae_recon_err``.

    Parameters
    ----------
    n_in : int
        Number of input features (``len(bundle.feature_cols)``).
    k : int
        Bottleneck (code) width; equals the PCA component count so the AE and
        PCA latent dimensionalities match.
    """

    def __init__(self, n_in: int, k: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, 8),
            nn.ReLU(),
            nn.Linear(8, k),
        )
        self.decoder = nn.Sequential(
            nn.Linear(k, 8),
            nn.ReLU(),
            nn.Linear(8, n_in),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the ``k``-dimensional bottleneck code for ``x``."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the reconstruction ``decode(encode(x))``."""
        return self.decoder(self.encoder(x))


@dataclass
class LatentBundle:
    """Frozen FE-train artifacts for the F4 pooled latent transform.

    Every field is fit on the pooled FE-train block of one asset class and
    applied forward, unchanged, at transform time. ``feature_cols`` pins the
    column order so transform inputs are reindexed to exactly the fitted layout.

    Attributes
    ----------
    scaler : StandardScaler
        Frozen standardizer fit on the (median-imputed) pooled train matrix.
    impute_median : np.ndarray
        Per-column FE-train medians (length ``len(feature_cols)``) used to fill
        NaN cells before scaling -- a frozen statistic, not a time-series fill.
    pca : PCA
        Fitted ``PCA(n_components=k)`` on the scaled train matrix.
    kmeans : KMeans
        Fitted ``KMeans(n_clusters=k)`` on the scaled train matrix.
    ae_state : dict
        ``state_dict`` of the trained :class:`DenseAutoencoder` (CPU tensors).
    recon_mse : dict
        ``{"pca_k4": ..., "ae_k4": ...}`` -- train reconstruction MSE of the
        PCA(k) and AE reconstructions on the scaled train matrix (both finite).
    train_index : pd.Index
        Row index of the pooled FE-train block the bundle was fit on.
    feature_cols : list[str]
        Frozen feature-column order; transform inputs are reindexed to this.
    asset_class : str
        Asset-class tag (e.g. ``"EQ"``) carried for provenance.
    k : int
        Latent dimensionality (PCA components, KMeans clusters, AE code width).
    """

    scaler: StandardScaler
    impute_median: np.ndarray
    pca: PCA
    kmeans: KMeans
    ae_state: dict
    recon_mse: dict
    train_index: pd.Index
    feature_cols: list[str]
    asset_class: str
    k: int


def _set_determinism(seed: int) -> None:
    """Pin every RNG / threading knob the AE fit depends on (CPU).

    Sets ``torch.use_deterministic_algorithms(True)``, single-threaded BLAS,
    and seeds both the torch and numpy generators so a refit on identical data
    reproduces the autoencoder weights bit-for-bit.
    """
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)


def _impute_scale(
    block: pd.DataFrame,
    feature_cols: list[str],
    impute_median: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    """Reindex to ``feature_cols``, impute frozen medians, then scale.

    Mirrors the fit-time preprocessing exactly: the block is reindexed to the
    frozen column order (missing columns become all-NaN), NaN cells are filled
    column-wise with ``impute_median``, and the result is transformed by the
    frozen ``scaler``. Returns a ``float64`` array of shape ``(n_rows, n_in)``.
    """
    mat = block.reindex(columns=feature_cols).to_numpy(dtype=float)
    # Column-wise frozen-median fill (broadcast medians across rows).
    fill = np.broadcast_to(impute_median, mat.shape)
    mat = np.where(np.isnan(mat), fill, mat)
    return scaler.transform(mat)


def _train_autoencoder(
    scaled: np.ndarray,
    k: int,
    seed: int,
    max_epochs: int = 500,
    patience: int = 20,
    lr: float = 1e-3,
) -> tuple[DenseAutoencoder, float]:
    """Full-batch train the dense AE on the scaled matrix; early-stop on plateau.

    Determinism is pinned by :func:`_set_determinism` *before* the module is
    constructed (so the weight initialization is seeded too). Training is
    full-batch -- the entire scaled matrix is one Adam step per epoch, with no
    ``DataLoader`` shuffling -- and stops when the train MSE fails to improve by
    more than ``1e-9`` for ``patience`` consecutive epochs (or at
    ``max_epochs``). Returns the trained module (in ``eval`` mode) and the final
    train reconstruction MSE.
    """
    _set_determinism(seed)
    n_in = scaled.shape[1]
    model = DenseAutoencoder(n_in, k)
    x = torch.from_numpy(scaled.astype(np.float32))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    best_loss = float("inf")
    epochs_no_improve = 0
    for _ in range(max_epochs):
        optimizer.zero_grad()
        recon = model(x)
        loss = loss_fn(recon, x)
        loss.backward()
        optimizer.step()

        cur = float(loss.detach())
        if cur < best_loss - 1e-9:
            best_loss = cur
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.eval()
    with torch.no_grad():
        final_mse = float(loss_fn(model(x), x).detach())
    return model, final_mse


def fit_latent(
    pooled_train_block: pd.DataFrame, k: int = 4, seed: int = 0
) -> LatentBundle:
    """Fit the pooled F4 latent stack on one asset class's FE-train block.

    The pooled block is the FE-train nonzero-signal engineered-feature vectors
    of every instrument in an asset class, stacked. Fitting proceeds: freeze the
    per-column train medians, impute NaN cells with them, fit a
    :class:`~sklearn.preprocessing.StandardScaler`, then on the scaled matrix
    fit ``PCA(n_components=k)``, ``KMeans(n_clusters=k, n_init=10)`` and the
    full-batch dense autoencoder. ``recon_mse`` records the train reconstruction
    MSE of PCA(k) and the AE.

    Parameters
    ----------
    pooled_train_block : pd.DataFrame
        Rows = pooled FE-train days; columns = numeric engineered features. The
        column order is frozen into ``feature_cols``; the row index is recorded
        as ``train_index``.
    k : int, default 4
        Latent dimensionality (PCA components / KMeans clusters / AE code).
    seed : int, default 0
        Seed for KMeans and the deterministic AE fit.

    Returns
    -------
    LatentBundle
        Frozen bundle carrying the scaler, impute medians, PCA, KMeans, AE
        ``state_dict``, ``recon_mse``, ``train_index``, ``feature_cols``,
        ``asset_class`` and ``k``.

    Notes
    -----
    The PERSISTED feature matrix keeps structural NaNs; the median imputation
    here is internal to the unsupervised fit only (CONTRACT_FE §0.4).
    """
    feature_cols = list(pooled_train_block.columns)
    train_index = pooled_train_block.index
    asset_class = str(pooled_train_block.attrs.get("asset_class", ""))

    raw = pooled_train_block.to_numpy(dtype=float)
    # Frozen per-column train medians (ignore NaN); guard an all-NaN column.
    impute_median = np.nanmedian(raw, axis=0)
    impute_median = np.where(np.isnan(impute_median), 0.0, impute_median)

    filled = np.where(np.isnan(raw), np.broadcast_to(impute_median, raw.shape), raw)

    scaler = StandardScaler().fit(filled)
    scaled = scaler.transform(filled)

    pca = PCA(n_components=k).fit(scaled)
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(scaled)

    # PCA(k) reconstruction MSE on the scaled train matrix.
    pca_recon = pca.inverse_transform(pca.transform(scaled))
    pca_mse = float(np.mean((scaled - pca_recon) ** 2))

    model, ae_mse = _train_autoencoder(scaled, k=k, seed=seed)
    ae_state = {key: val.detach().clone() for key, val in model.state_dict().items()}

    recon_mse = {"pca_k4": pca_mse, "ae_k4": ae_mse}

    return LatentBundle(
        scaler=scaler,
        impute_median=impute_median,
        pca=pca,
        kmeans=kmeans,
        ae_state=ae_state,
        recon_mse=recon_mse,
        train_index=train_index,
        feature_cols=feature_cols,
        asset_class=asset_class,
        k=int(k),
    )


def transform_latent(bundle: LatentBundle, block_inst: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen F4 bundle to ONE instrument's feature rows (row-wise).

    The block is reindexed to ``bundle.feature_cols``, NaN-imputed with the
    frozen ``impute_median`` and scaled by the frozen ``scaler``; then the four
    latent views are emitted on the same date index. Every step is row-wise and
    frozen, so transforming one instrument alone is identical (within numerical
    tolerance) to transforming a concatenated panel and slicing that
    instrument's rows -- which is why this MUST be called per-instrument-series,
    never on a stacked multi-instrument panel.

    Parameters
    ----------
    bundle : LatentBundle
        The frozen FE-train bundle from :func:`fit_latent`.
    block_inst : pd.DataFrame
        ONE instrument's engineered-feature rows, date-indexed.

    Returns
    -------
    pd.DataFrame
        Date-indexed, with columns ``f4_pc1..f4_pc{k}``, ``f4_cluster_id``,
        ``f4_cluster_dist``, ``f4_ae_code1..f4_ae_code{k}`` and
        ``f4_ae_recon_err``.
    """
    k = bundle.k
    scaled = _impute_scale(
        block_inst, bundle.feature_cols, bundle.impute_median, bundle.scaler
    )

    out: dict[str, np.ndarray] = {}

    # --- PCA principal-component scores -------------------------------------
    pcs = bundle.pca.transform(scaled)
    for j in range(k):
        out[f"f4_pc{j + 1}"] = pcs[:, j]

    # --- KMeans cluster id + distance to assigned centroid ------------------
    cluster_id = bundle.kmeans.predict(scaled)
    centroids = bundle.kmeans.cluster_centers_[cluster_id]
    cluster_dist = np.sqrt(np.sum((scaled - centroids) ** 2, axis=1))
    out["f4_cluster_id"] = cluster_id.astype(float)
    out["f4_cluster_dist"] = cluster_dist

    # --- AE code (bottleneck) + per-row reconstruction MSE ------------------
    model = DenseAutoencoder(len(bundle.feature_cols), k)
    model.load_state_dict(bundle.ae_state)
    model.eval()
    x = torch.from_numpy(scaled.astype(np.float32))
    with torch.no_grad():
        code = model.encode(x).numpy()
        recon = model(x).numpy()
    for j in range(k):
        out[f"f4_ae_code{j + 1}"] = code[:, j]
    out["f4_ae_recon_err"] = np.mean((scaled - recon) ** 2, axis=1)

    return pd.DataFrame(out, index=block_inst.index)
