"""
dataset.py
==========
Assemble model-ready ``(X, y)`` from the frozen feature matrix and the triple-barrier labels,
plus the small registries the modeling layer reads from the FE handoff (instrument scope,
feature redundancy).

Responsibilities:

* Load ``results/feature_matrix.parquet`` and the clean close panel, and attach each row's
  **trading-bar position** (``bar_pos``) on its instrument's own calendar -- the axis the purged
  CV splitter and the labeler both count ``h`` along.
* Pull the **label interface**: ``side = f5_signal`` (the primary signal) and
  ``sigma = f2_vol_20`` (the volatility target), the two columns the catalog earmarks for the
  deferred triple-barrier label.
* Build the numeric design matrix: drop redundant features (one representative per
  ``feature_redundancy.json`` cluster), one-hot the nominal ``f4_cluster_id``, and -- for the
  pooled / per-class scopes -- one-hot the instrument so a single model can specialise.
* Iterate **scopes**: ``pooled`` (one cell), ``per_class`` (EQ/EN/ME), ``per_instrument`` (11
  cells; the caller skips the non-viable thin ones).
* A leakage-safe :class:`Preprocessor` (median impute + standardise) fit on a fold's purged
  train rows only -- never on validation.

Everything here consumes the ``metamodel`` FE outputs read-only; it never refits a fitted (TF)
feature, preserving the frozen 2021-07-01 provenance contract.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from stml.io import _find_repo_root, load_clean_data

# Feature columns are exactly the family columns ``f<digit>...`` (f1_*..f11_*). This deliberately
# excludes meta columns that also start with "f" -- notably ``fe_train_end_date`` (a string).
_FEATURE_RE = re.compile(r"^f\d")

SIDE_COL = "f5_signal"
SIGMA_COL = "f2_vol_20"
CLUSTER_COL = "f4_cluster_id"
# f2_vol_20 is an ANNUALISED volatility (median ~0.16 vs realised daily log-return std ~0.012);
# the triple barrier needs a per-bar return-space target, so de-annualise by sqrt(252).
TRADING_DAYS_PER_YEAR = 252.0
META_COLS = ("date", "instrument", "partition", "bar_pos")
DEV_PARTITIONS = ("train", "val")


def _results_dir() -> Path:
    return _find_repo_root(Path.cwd().resolve()) / "results"


def load_matrix(path: str | Path | None = None) -> pd.DataFrame:
    """Load the feature matrix with ``date`` parsed and canonical sort order."""
    if path is None:
        path = _results_dir() / "feature_matrix.parquet"
    m = pd.read_parquet(path)
    m["date"] = pd.to_datetime(m["date"])
    return m.sort_values(["instrument", "date"]).reset_index(drop=True)


def load_scope(path: str | Path | None = None) -> dict[str, dict]:
    """Load ``instrument_scope.json`` -> ``{instrument: {asset_class, embargo_p90, ...}}``."""
    if path is None:
        path = _results_dir() / "instrument_scope.json"
    return json.load(open(path))


def asset_class_map(scope: dict[str, dict] | None = None) -> dict[str, str]:
    """Map ``instrument -> asset class`` (EQ/EN/ME) from the scope registry."""
    scope = scope or load_scope()
    return {k: v["asset_class"] for k, v in scope.items()}


def embargo_map(scope: dict[str, dict] | None = None) -> dict[str, int]:
    """Map ``instrument -> embargo_p90`` (trading-bar embargo width) from the scope registry."""
    scope = scope or load_scope()
    return {k: int(v["embargo_p90"]) for k, v in scope.items()}


def close_panel(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Wide ``date x instrument`` clean close-price panel (each column on its own calendar)."""
    ohlcv, _ = load_clean_data(data_dir)
    wide = ohlcv.pivot(index="date", columns="instrument", values="close")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def attach_bar_pos(matrix: pd.DataFrame, close_wide: pd.DataFrame) -> pd.DataFrame:
    """Add ``bar_pos`` = each row's integer position on its instrument's clean trading calendar.

    This is the axis ``h`` (label horizon) and the embargo are counted along -- never calendar
    days, so ragged exchange calendars cannot fabricate a barrier or a fold gap.
    """
    out = matrix.copy()
    out["bar_pos"] = -1
    for inst, idx_pos in out.groupby("instrument").groups.items():
        if inst not in close_wide.columns:
            continue
        cal = close_wide[inst].dropna().index
        sub = out.loc[idx_pos]
        out.loc[idx_pos, "bar_pos"] = cal.get_indexer(pd.DatetimeIndex(sub["date"]))
    return out


def events_frame(matrix: pd.DataFrame, *, de_annualize: bool = True) -> pd.DataFrame:
    """Extract the labeler input ``[date, instrument, side, sigma]`` from the matrix.

    ``sigma`` is ``f2_vol_20`` converted to a per-bar return-space target (divided by
    ``sqrt(252)``) when ``de_annualize`` is True, so a barrier of ``pt * sigma`` is comparable to
    the realised single-bar return. Set ``de_annualize=False`` only if you pass a sigma column
    that is already per-bar.
    """
    sigma = matrix[SIGMA_COL].to_numpy(dtype=float)
    if de_annualize:
        sigma = sigma / np.sqrt(TRADING_DAYS_PER_YEAR)
    return pd.DataFrame({
        "date": matrix["date"].to_numpy(),
        "instrument": matrix["instrument"].to_numpy(),
        "side": matrix[SIDE_COL].to_numpy(),
        "sigma": sigma,
    })


def select_features(
    matrix: pd.DataFrame,
    *,
    drop_redundant: bool = True,
    redundancy_path: str | Path | None = None,
) -> list[str]:
    """Numeric base feature columns to model on (``f*`` minus the nominal ``f4_cluster_id``).

    When ``drop_redundant`` is True, keep a single representative per ``feature_redundancy.json``
    cluster (the lexicographically-first member) and drop the rest -- the redundancy map already
    grouped features with ``|corr| > threshold`` (e.g. the parkinson/garman-klass vol twins).
    """
    feats = [c for c in matrix.columns if _FEATURE_RE.match(c) and c != CLUSTER_COL]
    if not drop_redundant:
        return sorted(feats)

    if redundancy_path is None:
        redundancy_path = _results_dir() / "feature_redundancy.json"
    clusters: dict[str, int] = json.load(open(redundancy_path))["clusters"]

    keep, seen = [], set()
    for feat in sorted(feats):
        cid = clusters.get(feat)
        if cid is None:  # not in the map -> keep (singleton)
            keep.append(feat)
            continue
        if cid in seen:  # a representative of this cluster already kept
            continue
        seen.add(cid)
        keep.append(feat)
    return keep


def make_xy(
    cell_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    instrument_dummies: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Numeric design matrix and binary target for one scope cell.

    ``cell_df`` must already carry a ``bin`` label column (merged from the labeler). One-hots the
    nominal ``f4_cluster_id`` and -- when ``instrument_dummies`` -- the instrument, so a pooled or
    per-class model can specialise per instrument. Columns are deterministic for a given cell.
    """
    X = cell_df[feature_cols].copy()
    if CLUSTER_COL in cell_df.columns:
        dummies = pd.get_dummies(
            cell_df[CLUSTER_COL].astype("Int64").astype(str), prefix="f4_clust"
        )
        X = pd.concat([X, dummies.set_index(X.index)], axis=1)
    if instrument_dummies:
        inst_d = pd.get_dummies(cell_df["instrument"], prefix="inst")
        X = pd.concat([X, inst_d.set_index(X.index)], axis=1)
    X = X.astype(float)
    y = cell_df["bin"].to_numpy(dtype=int)
    return X.reset_index(drop=True), y


def align_columns(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Reindex ``X`` to a fixed column layout (missing dummies -> 0). Used for the test frame."""
    return X.reindex(columns=columns, fill_value=0.0).astype(float)


def scope_iter(
    dev_df: pd.DataFrame, scope: str, *, scope_reg: dict[str, dict] | None = None
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield ``(cell_id, cell_df)`` for a scope: ``pooled`` / ``per_class`` / ``per_instrument``."""
    if scope == "pooled":
        yield "pooled", dev_df
    elif scope == "per_class":
        ac = asset_class_map(scope_reg)
        dev = dev_df.assign(_ac=dev_df["instrument"].map(ac))
        for cls, sub in dev.groupby("_ac", sort=True):
            yield cls, sub.drop(columns="_ac")
    elif scope == "per_instrument":
        for inst, sub in dev_df.groupby("instrument", sort=True):
            yield inst, sub
    else:
        raise ValueError(f"unknown scope {scope!r}")


class Preprocessor:
    """Median-impute then standardise, fit on a fold's purged train rows only.

    Trees (XGBoost) need neither, but RandomForest and the torch nets require finite, scaled
    inputs. Fitting on the purged-train subset (and transforming validation with frozen stats) is
    what keeps each fold honest -- never fit on validation.
    """

    def __init__(self) -> None:
        self.medians_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame | np.ndarray) -> Preprocessor:
        A = np.asarray(X, dtype=float)
        self.medians_ = np.nanmedian(A, axis=0)
        self.medians_ = np.where(np.isfinite(self.medians_), self.medians_, 0.0)
        filled = np.where(np.isnan(A), self.medians_, A)
        self.mean_ = filled.mean(axis=0)
        std = filled.std(axis=0)
        self.std_ = np.where(std > 1e-12, std, 1.0)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        A = np.asarray(X, dtype=float)
        filled = np.where(np.isnan(A), self.medians_, A)
        return (filled - self.mean_) / self.std_

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
