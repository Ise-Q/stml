"""
build_features.py
=================
Persistence CLI for the triple-barrier metamodel feature-engineering layer
(US-FE-009). This is the one-shot build entrypoint that turns the released
universe into the graded deliverable artifacts:

* loads the clean OHLCV + signal panel (:func:`stml.io.load_clean_data`);
* fits and transforms the full feature stack
  (:class:`stml.metamodel.pipeline.FeaturePipeline`) into one tidy-long matrix
  restricted to the nonzero-signal trade-days, tagged with the chronological
  ``partition`` and the ``fe_train_end_date`` provenance column;
* asserts the produced columns are in exact 1:1 correspondence with the feature
  catalog (:func:`stml.metamodel.catalog.assert_coverage`);
* persists the canonical artifacts as **new siblings** of ``results/jj/`` (the
  replication ledger is never clobbered).

Persisted artifacts (CONTRACT_FE Sections 3 / 4)
------------------------------------------------
``results/feature_matrix.parquet`` / ``.csv``
    The canonical tidy-long feature matrix.
``data/features/<family>.csv``
    One CSV per feature family (e.g. ``f1_counter_trend.csv`` ...
    ``f17_hmm_regimes.csv``), each keyed by ``(date, instrument)`` and carrying
    that family's raw columns plus their ``z_`` standardization twins.
``data/macro_features_engineered.parquet`` / ``.csv``
    The standalone F11 cross-asset macro dataset: the standardized, matrix-
    aligned macro columns keyed by ``(date, instrument)``, row-aligned to the
    matrix's nonzero-signal rows.
``results/feature_redundancy.json`` / ``.csv``
    Pairwise feature correlation (reusing
    :func:`stml.na_checks.corr_max_info`) plus a
    :mod:`scipy.cluster.hierarchy` flat-cluster assignment on the
    ``1 - |corr|`` distance and the highest-``|corr|`` partner per feature.
``results/instrument_scope.json``
    The D5 per-instrument scope registry
    (:func:`stml.metamodel.scope.persist_scope`).
``results/feature_matrix_provenance.json``
    FE-train boundary, split boundary dates, per-instrument regime
    ``train_index`` summary, per-class latent ``train_index`` count, and the row
    / feature-column / seed bookkeeping.
``reports/feature-catalog.md``
    The rendered feature catalog with the AE-vs-PCA(k=4) reconstruction-MSE
    table (:func:`stml.metamodel.catalog.render_catalog`).

Leakage note (CONTRACT_FE Section 0)
------------------------------------
This module only orchestrates persistence; the causal fit-on-train / frozen
transform contract is enforced inside :class:`FeaturePipeline` and the feature
modules. The persisted matrix keeps structural NaNs exactly as produced -- it is
never forward-filled or ``fillna(0)``-ed (the redundancy correlation tolerates
the structural NaNs via the pairwise-complete :func:`corr_max_info`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from stml.io import load_clean_data
from stml.metamodel.catalog import META_COLS, assert_coverage, render_catalog
from stml.metamodel.pipeline import FeaturePipeline
from stml.metamodel.scope import ASSET_CLASS_MAP, persist_scope

__all__ = [
    "ALL_INSTRUMENTS",
    "build_feature_matrix",
    "compute_redundancy",
    "main",
]

#: The full released universe (the D5 asset-class map keys), sorted.
ALL_INSTRUMENTS: list[str] = sorted(ASSET_CLASS_MAP)

#: Minimum pairwise overlap for the redundancy correlation. 252 trading days
#: (one year) is the project-wide floor used by :func:`corr_max_info`; with
#: ~4984 stacked observations every retained feature pair clears it comfortably.
_CORR_MIN_PERIODS: int = 252

#: Flat-cluster distance threshold on ``1 - |corr|``. Features whose absolute
#: correlation exceeds ``1 - threshold`` (here ``|corr| > 0.7``) merge into one
#: redundancy cluster.
_REDUNDANCY_DISTANCE_THRESHOLD: float = 0.30

#: Per-family CSV filename slugs (under ``data/features/``). One CSV per
#: feature family, each keyed by ``(date, instrument)`` and carrying that
#: family's raw columns plus their ``z_`` standardization twins.
_FAMILY_SLUGS: dict[str, str] = {
    "F1": "f1_counter_trend",
    "F2": "f2_volatility",
    "F3": "f3_regime_posteriors",
    "F4": "f4_latent",
    "F5": "f5_signal_derived",
    "F6": "f6_momentum",
    "F7": "f7_microstructure",
    "F8": "f8_calendar",
    "F9": "f9_cross_sectional",
    "F10": "f10_price_action",
    "F11": "f11_macro_context",
    "F12": "f12_path_structure",
    "F13": "f13_wavelet",
    "F15": "f15_conditional_risk",
    "F16": "f16_concept_drift",
    "F17": "f17_hmm_regimes",
}


def _write_family_csvs(matrix: pd.DataFrame, data_dir: Path) -> dict[str, Path]:
    """Write one CSV per feature family under ``data_dir/features/``.

    Each family CSV is keyed by ``(date, instrument)`` and carries that family's
    feature columns (raw + their ``z_`` twins) in matrix-column order. The union
    of the family files' feature columns equals the master matrix's feature
    columns (no column lost or duplicated).

    Parameters
    ----------
    matrix : pd.DataFrame
        The tidy-long feature matrix.
    data_dir : pathlib.Path
        The ``data/`` directory; the CSVs land in ``data_dir/features/``.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping ``"family_<slug>" -> written path``.
    """
    from stml.metamodel.catalog import CATALOG

    feat_dir = data_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in matrix.columns if c not in META_COLS]
    by_family: dict[str, list[str]] = {}
    for col in feature_cols:
        by_family.setdefault(CATALOG[col].family, []).append(col)

    written: dict[str, Path] = {}
    for family, cols in by_family.items():
        slug = _FAMILY_SLUGS.get(family, family.lower())
        path = feat_dir / f"{slug}.csv"
        matrix[["date", "instrument", *cols]].to_csv(path, index=False)
        written[f"family_{slug}"] = path
    return written


# --------------------------------------------------------------------------- #
# Build the feature matrix                                                    #
# --------------------------------------------------------------------------- #
def build_feature_matrix(
    instruments: list[str] | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, FeaturePipeline]:
    """Load, fit and transform the metamodel feature matrix.

    Loads the clean OHLCV + signal panel, optionally restricts it to the
    requested instruments, fits a :class:`FeaturePipeline` on the FE-train
    partition and transforms it into the tidy-long feature matrix, then asserts
    exact catalog coverage.

    Parameters
    ----------
    instruments : list of str, optional
        Instrument tickers to build. ``None`` (the default) builds the full
        :data:`ALL_INSTRUMENTS` universe. Unknown tickers are ignored by the
        pipeline's universe filter.
    seed : int, default 0
        Determinism seed threaded into the regime GMM and the latent
        KMeans / autoencoder fits.

    Returns
    -------
    matrix : pd.DataFrame
        The tidy-long feature matrix
        (``["date", "instrument", "partition", "fe_train_end_date", <features...>]``).
    pipe : FeaturePipeline
        The fitted pipeline (its ``_regime`` / ``_latent`` / ``scope`` bundles
        back the provenance and catalog artifacts).
    """
    ohlcv, signals = load_clean_data()

    if instruments is not None:
        keep = [i for i in instruments if i in ASSET_CLASS_MAP]
        ohlcv = ohlcv[ohlcv["instrument"].isin(keep)].copy()
        signal_cols = ["date", *[i for i in keep if i in signals.columns]]
        signals = signals[signal_cols].copy()

    pipe = FeaturePipeline(seed=seed).fit(ohlcv, signals)
    matrix = pipe.transform(ohlcv, signals)
    assert_coverage(matrix.columns)
    return matrix, pipe


# --------------------------------------------------------------------------- #
# Redundancy map                                                              #
# --------------------------------------------------------------------------- #
def compute_redundancy(
    matrix: pd.DataFrame,
    min_periods: int = _CORR_MIN_PERIODS,
    distance_threshold: float = _REDUNDANCY_DISTANCE_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, dict[str, float | str]]]:
    """Build the feature-redundancy map from the tidy-long matrix.

    The numeric feature columns (the non-meta columns) are treated as a wide
    observation frame -- each ``(date, instrument)`` row is one observation --
    and their pairwise-complete correlation is computed via
    :func:`stml.na_checks.corr_max_info` (AC-9 reuse). Features are then
    hierarchically clustered on the ``1 - |corr|`` distance
    (:func:`scipy.cluster.hierarchy.linkage` with average linkage,
    :func:`fcluster` at ``distance_threshold``), and the single highest-absolute
    correlation partner of every feature is recorded.

    Parameters
    ----------
    matrix : pd.DataFrame
        The tidy-long feature matrix from :func:`build_feature_matrix`.
    min_periods : int, default 252
        Minimum pairwise overlap forwarded to :func:`corr_max_info`.
    distance_threshold : float, default 0.30
        ``fcluster`` distance cut on ``1 - |corr|`` (``|corr| > 0.70`` merges).

    Returns
    -------
    corr : pd.DataFrame
        The (PSD-repaired) feature x feature correlation matrix.
    clusters : dict[str, int]
        Mapping ``feature -> flat cluster id`` (1-based, from ``fcluster``).
    partners : dict[str, dict]
        Mapping ``feature -> {"partner": str, "abs_corr": float}`` -- the
        off-diagonal highest-``|corr|`` partner of each feature.
    """
    # Reuse na_checks.corr_max_info on the feature-only frame: each (date,
    # instrument) row is an observation, columns are the 71 features, so the
    # pairwise-complete .corr() inside corr_max_info yields feature x feature
    # correlations (NaN-filled to 0 off-diagonal, unit diagonal, PSD-repaired).
    feature_cols = [c for c in matrix.columns if c not in META_COLS]
    feature_frame = matrix[feature_cols].apply(pd.to_numeric, errors="coerce")

    from stml.na_checks import corr_max_info

    corr = corr_max_info(feature_frame, min_periods=min_periods)

    abs_corr = corr.abs().to_numpy(dtype=float)
    # Guard: corr_max_info already fills NaN -> 0 and sets the unit diagonal, but
    # be defensive so the distance matrix is always finite and zero-diagonal.
    abs_corr = np.where(np.isnan(abs_corr), 0.0, abs_corr)
    np.fill_diagonal(abs_corr, 1.0)
    abs_corr = np.clip(abs_corr, 0.0, 1.0)

    distance = 1.0 - abs_corr
    distance = (distance + distance.T) / 2.0  # enforce exact symmetry
    np.fill_diagonal(distance, 0.0)

    # Hierarchical clustering on the condensed distance (NaN corr -> distance 1,
    # i.e. maximally far, so unrelated features never merge spuriously).
    condensed = squareform(distance, checks=False)
    linkage_matrix = linkage(condensed, method="average")
    flat = fcluster(linkage_matrix, t=distance_threshold, criterion="distance")
    clusters = {col: int(cid) for col, cid in zip(feature_cols, flat)}

    # Highest-|corr| off-diagonal partner per feature.
    partners: dict[str, dict[str, float | str]] = {}
    masked = abs_corr.copy()
    np.fill_diagonal(masked, -1.0)
    for i, col in enumerate(feature_cols):
        j = int(np.argmax(masked[i]))
        partners[col] = {
            "partner": feature_cols[j],
            "abs_corr": float(abs_corr[i, j]),
        }

    return corr, clusters, partners


def _write_redundancy(
    corr: pd.DataFrame,
    clusters: dict[str, int],
    partners: dict[str, dict[str, float | str]],
    outdir: Path,
) -> tuple[Path, Path]:
    """Persist the redundancy map to ``feature_redundancy.{json,csv}``.

    The JSON carries the full correlation matrix plus, per feature, its cluster
    id and highest-``|corr|`` partner; the CSV is the tidy per-feature view.

    Parameters
    ----------
    corr : pd.DataFrame
        Feature x feature correlation matrix.
    clusters : dict[str, int]
        Feature -> flat cluster id.
    partners : dict[str, dict]
        Feature -> ``{"partner", "abs_corr"}``.
    outdir : pathlib.Path
        Destination directory (created if absent).

    Returns
    -------
    json_path, csv_path : pathlib.Path
        The two written paths.
    """
    feature_cols = list(corr.columns)
    json_path = outdir / "feature_redundancy.json"
    csv_path = outdir / "feature_redundancy.csv"

    payload = {
        "n_features": len(feature_cols),
        "features": feature_cols,
        "distance_threshold": _REDUNDANCY_DISTANCE_THRESHOLD,
        "min_periods": _CORR_MIN_PERIODS,
        "correlation": {
            row: {col: float(corr.at[row, col]) for col in feature_cols}
            for row in feature_cols
        },
        "clusters": clusters,
        "max_abs_corr_partner": partners,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    tidy = pd.DataFrame(
        {
            "feature": feature_cols,
            "cluster_id": [clusters[c] for c in feature_cols],
            "max_corr_partner": [partners[c]["partner"] for c in feature_cols],
            "max_abs_corr": [partners[c]["abs_corr"] for c in feature_cols],
        }
    )
    tidy.to_csv(csv_path, index=False)
    return json_path, csv_path


# --------------------------------------------------------------------------- #
# Provenance                                                                  #
# --------------------------------------------------------------------------- #
def _build_provenance(
    matrix: pd.DataFrame,
    pipe: FeaturePipeline,
    seed: int,
) -> dict:
    """Assemble the provenance record for ``feature_matrix_provenance.json``.

    Parameters
    ----------
    matrix : pd.DataFrame
        The produced feature matrix.
    pipe : FeaturePipeline
        The fitted pipeline (source of the split dates and fitted bundles).
    seed : int
        The determinism seed the matrix was built with.

    Returns
    -------
    dict
        A JSON-serialisable provenance mapping (see module docstring).
    """

    def _bounds(index: pd.DatetimeIndex) -> dict:
        idx = pd.DatetimeIndex(index)
        if len(idx) == 0:
            return {"count": 0, "min": None, "max": None}
        return {
            "count": int(len(idx)),
            "min": idx.min().date().isoformat(),
            "max": idx.max().date().isoformat(),
        }

    feature_cols = [c for c in matrix.columns if c not in META_COLS]

    regime_train = {
        inst: _bounds(bundle.train_index)
        for inst, bundle in pipe._regime.items()
    }
    latent_train = {
        cls: {
            "count": int(len(pd.Index(bundle.train_index))),
            "recon_mse": dict(bundle.recon_mse),
        }
        for cls, bundle in pipe._latent.items()
    }

    macro = None
    if pipe._macro is not None:
        macro = {
            "train_index": _bounds(pipe._macro.train_index),
            "n_macro_features": int(len(pipe._macro.feature_cols)),
            "kept_series": sorted(pipe._macro.lag_config.get("series_classes", {})),
            "lag_config": pipe._macro.lag_config,
        }

    return {
        "fe_train_end_date": pipe.fe_train_end,
        "seed": int(seed),
        "n_rows": int(len(matrix)),
        "n_feature_cols": int(len(feature_cols)),
        "instruments": sorted(matrix["instrument"].unique().tolist()),
        "split_boundaries": {
            "train": _bounds(pipe._train_dates),
            "val": _bounds(pipe._val_dates),
            "test": _bounds(pipe._test_dates),
        },
        "partition_row_counts": {
            str(k): int(v)
            for k, v in matrix["partition"].value_counts().items()
        },
        "regime_train_index": regime_train,
        "latent_train_index": latent_train,
        "macro": macro,
    }


# --------------------------------------------------------------------------- #
# Persist everything                                                          #
# --------------------------------------------------------------------------- #
def _persist(
    matrix: pd.DataFrame,
    pipe: FeaturePipeline,
    outdir: Path,
    seed: int,
    catalog_path: Path,
    data_dir: Path = Path("data"),
) -> dict[str, Path]:
    """Write every deliverable artifact and return the path map.

    Parameters
    ----------
    matrix : pd.DataFrame
        The produced feature matrix.
    pipe : FeaturePipeline
        The fitted pipeline.
    outdir : pathlib.Path
        Destination directory for the ``results/`` artifacts.
    seed : int
        The build seed (recorded in the provenance JSON).
    catalog_path : pathlib.Path
        Destination markdown path for the rendered feature catalog.
    data_dir : pathlib.Path, default ``Path("data")``
        Destination directory for the standalone F11 macro dataset
        (``macro_features_engineered.{parquet,csv}``).

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping of artifact key -> written path.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    parquet_path = outdir / "feature_matrix.parquet"
    csv_path = outdir / "feature_matrix.csv"
    matrix.to_parquet(parquet_path, index=False)
    matrix.to_csv(csv_path, index=False)

    # Per-family CSVs (data/features/<slug>.csv), keyed by (date, instrument).
    data_dir.mkdir(parents=True, exist_ok=True)
    family_paths = _write_family_csvs(matrix, data_dir)

    # Standalone F11 macro dataset: the STANDARDIZED, matrix-aligned macro
    # columns keyed by (date, instrument), row-aligned to the matrix's
    # nonzero-signal rows (the ML-ready macro slice the spec asked for).
    data_dir.mkdir(parents=True, exist_ok=True)
    macro_cols = [c for c in matrix.columns if c.startswith("f11_")]
    macro_frame = matrix[["date", "instrument", *macro_cols]]
    macro_parquet = data_dir / "macro_features_engineered.parquet"
    macro_csv = data_dir / "macro_features_engineered.csv"
    macro_frame.to_parquet(macro_parquet, index=False)
    macro_frame.to_csv(macro_csv, index=False)

    corr, clusters, partners = compute_redundancy(matrix)
    redundancy_json, redundancy_csv = _write_redundancy(
        corr, clusters, partners, outdir
    )

    scope_path = outdir / "instrument_scope.json"
    persist_scope(pipe.scope, scope_path)

    recon_mse = {cls: pipe._latent[cls].recon_mse for cls in pipe._latent}
    render_catalog(matrix.columns, recon_mse=recon_mse, path=str(catalog_path))

    provenance = _build_provenance(matrix, pipe, seed)
    provenance_path = outdir / "feature_matrix_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2))

    return {
        "feature_matrix_parquet": parquet_path,
        "feature_matrix_csv": csv_path,
        **family_paths,
        "macro_features_parquet": macro_parquet,
        "macro_features_csv": macro_csv,
        "feature_redundancy_json": redundancy_json,
        "feature_redundancy_csv": redundancy_csv,
        "instrument_scope_json": scope_path,
        "feature_matrix_provenance_json": provenance_path,
        "feature_catalog_md": catalog_path,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the build CLI arguments.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector (defaults to ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Parsed ``instruments`` / ``outdir`` / ``seed``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m stml.metamodel.build_features",
        description=(
            "Build, verify and persist the triple-barrier metamodel feature "
            "matrix and its companion artifacts."
        ),
    )
    parser.add_argument(
        "--instruments",
        nargs="*",
        default=None,
        metavar="TICKER",
        help=(
            "Instrument tickers to build (default: all 11). Unknown tickers "
            "are ignored."
        ),
    )
    parser.add_argument(
        "--outdir",
        default="results",
        help="Directory for the results/ artifacts (default: 'results').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Determinism seed for the fitted feature groups (default: 0).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Build, verify and persist the feature matrix + companion artifacts.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector (defaults to ``sys.argv[1:]``). See :func:`_parse_args`.
    """
    args = _parse_args(argv)
    outdir = Path(args.outdir)
    catalog_path = Path("reports") / "feature-catalog.md"

    matrix, pipe = build_feature_matrix(
        instruments=args.instruments, seed=args.seed
    )
    paths = _persist(matrix, pipe, outdir, args.seed, catalog_path)

    feature_cols = [c for c in matrix.columns if c not in META_COLS]
    partition_counts = matrix["partition"].value_counts().to_dict()

    print("=" * 70)
    print("stml.metamodel.build_features — feature matrix built")
    print("=" * 70)
    print(f"instruments        : {sorted(matrix['instrument'].unique().tolist())}")
    print(f"rows               : {len(matrix):,}")
    print(f"feature columns    : {len(feature_cols)}")
    print(f"partitions (rows)  : {partition_counts}")
    print(f"fe_train_end_date  : {pipe.fe_train_end}")
    print(f"seed               : {args.seed}")
    print("-" * 70)
    print("artifacts written:")
    for key, path in paths.items():
        size = path.stat().st_size if path.exists() else 0
        print(f"  {key:32s} {str(path):42s} {size:>12,} B")
    print("=" * 70)


if __name__ == "__main__":
    main()
