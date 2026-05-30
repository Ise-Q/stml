"""Shared helpers for the EX.* / S4.7 real-data probe scripts.

These scripts are reproducible diagnostics, not part of the deliverable pipeline. They build the
real pooled modelling panel per asset class and write a findings markdown under ``results/``.
Determinism via ``set_seeds``; everything operates on the modelling sample only (<= modelling_end)
unless a script explicitly needs the OOS window — the EX.1/EX.3 probes are DIAGNOSTIC-ONLY and
never feed barrier hyperparameters back into the locked config.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

from stml.io import load_clean_data  # noqa: E402

from alken_metamodel.pipeline import (  # noqa: E402
    PipelineConfig,
    build_class_panel,
    class_members,
    feature_columns,
)

RESULTS = Path(__file__).resolve().parent / "results"
CLASSES = ("energy", "equity", "metals")


def results_dir() -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    return RESULTS


def modelling_panel(asset_class: str, cfg: PipelineConfig):
    """Return (pooled, feature_cols, modelling_mask) for an asset class on the real data."""
    ohlcv, signals = load_clean_data()
    pooled = build_class_panel(ohlcv, signals, class_members(asset_class), cfg)
    cols = feature_columns(pooled)
    dates = pd.DatetimeIndex(pooled["date"])
    mask = np.asarray(dates <= cfg.modelling_end)
    return pooled, cols, mask


def imputed_modelling_X(pooled, cols, mask) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Median-imputed, zero-variance-dropped modelling design matrix + (y, t1)."""
    x_raw = pooled[cols][mask]
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    X = pd.DataFrame(imp.fit_transform(x_raw), index=x_raw.index, columns=cols)
    keep = [c for c in X.columns if X[c].var() > 1e-12]
    X = X[keep]
    y = pooled["bin"].to_numpy()[mask]
    t1 = pooled["t1"][mask]
    return X, y, t1
