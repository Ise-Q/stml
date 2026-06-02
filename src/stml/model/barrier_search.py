"""
barrier_search.py
=================
Tune the triple-barrier geometry ``(pt, sl, h)`` on the development (train+val) panel only.

Per the guide (section 7) we do **not** optimise classification accuracy -- a labeler that emits
95% of one class scores high accuracy and is useless. Instead each candidate barrier set is:

1. used to relabel the dev events (``sigma`` = de-annualised ``f2_vol_20``),
2. screened against a **class-balance floor** (minority fraction >= ``min_minority``), and
3. scored by the **mean purged-CV validation AUC of a fixed shallow XGBoost baseline** on the
   pooled panel -- a cheap, constant model so we measure label quality, not joint barrier+model
   overfitting.

The winner is chosen by a **robust-plateau** rule, not the single peak: each candidate is rescored
as the mean AUC of itself and its immediate grid neighbours (adjacent ``pt``/``sl`` at the same
``h``), so an isolated lucky spike loses to a configuration surrounded by good ones. ``h`` enters
both the labels and the CV purge width, so the CV splitter is rebuilt per candidate via
``cv_factory(h)``.

The number of grid points tried is returned for the deflation note (more configs -> more chance
the best is luck). The test partition is never labelled or scored here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stml.model.cv import PurgedWalkForward
from stml.model.dataset import embargo_map, events_frame, make_xy, select_features
from stml.model.labels import class_balance, triple_barrier_labels
from stml.model.optuna_objective import cross_val_auc
from stml.model.trees import XGBModel, xgb_baseline_params


@dataclass
class BarrierResult:
    """Outcome of a barrier search: the full scored grid and the robust pick."""

    table: pd.DataFrame      # one row per (pt, sl, h): minority, n, auc, auc_std, plateau, skipped
    best: dict               # the chosen {pt, sl, h, auc, plateau, ...}
    n_configs: int           # grid points evaluated (for deflation)


def _plateau_scores(table: pd.DataFrame) -> pd.Series:
    """Neighbour-averaged AUC over the (pt, sl) grid at fixed h (robust-plateau rule)."""
    scored = table[table["auc"].notna()].copy()
    pts = np.sort(scored["pt"].unique())
    sls = np.sort(scored["sl"].unique())
    pt_rank = {v: i for i, v in enumerate(pts)}
    sl_rank = {v: i for i, v in enumerate(sls)}
    out = pd.Series(index=table.index, dtype=float)
    for idx, row in table.iterrows():
        if not np.isfinite(row["auc"]):
            out[idx] = np.nan
            continue
        pr, sr = pt_rank[row["pt"]], sl_rank[row["sl"]]
        nb = scored[
            (scored["h"] == row["h"])
            & (scored["pt"].map(pt_rank).sub(pr).abs() <= 1)
            & (scored["sl"].map(sl_rank).sub(sr).abs() <= 1)
        ]
        out[idx] = float(nb["auc"].mean())
    return out


def search_barriers(
    dev_matrix: pd.DataFrame,
    close_wide: pd.DataFrame,
    *,
    grid: list[tuple[float, float, int]],
    n_splits: int = 4,
    price_end: str | pd.Timestamp | None = None,
    min_minority: float = 0.30,
    seed: int = 0,
    baseline_params: dict | None = None,
    feature_cols: list[str] | None = None,
) -> BarrierResult:
    """Search ``grid`` of ``(pt, sl, h)`` and return the scored table + robust pick.

    ``dev_matrix`` must be the train+val rows of the feature matrix WITH a ``bar_pos`` column.
    ``price_end`` should be the validation-block end date so no test-period price is consulted.
    """
    baseline_params = baseline_params or xgb_baseline_params()
    feature_cols = feature_cols or select_features(dev_matrix)
    emb = embargo_map()
    ev = events_frame(dev_matrix)

    rows = []
    for pt, sl, h in grid:
        labels = triple_barrier_labels(close_wide, ev, pt=pt, sl=sl, h=h, price_end=price_end)
        bal = class_balance(labels)
        rec = {"pt": pt, "sl": sl, "h": h, "n": bal["n"],
               "minority": bal["minority_frac"], "pos_rate": bal["pos_rate"],
               "auc": np.nan, "auc_std": np.nan, "n_folds": 0, "skipped": True}
        if bal["minority_frac"] >= min_minority and bal["n"] > 0:
            dev = dev_matrix.merge(labels[["date", "instrument", "bin"]],
                                   on=["date", "instrument"], how="inner").reset_index(drop=True)
            X, y = make_xy(dev, feature_cols, instrument_dummies=True)
            cv = PurgedWalkForward(n_splits=n_splits, h=h, embargo_by_instrument=emb)
            mean, std, n = cross_val_auc(XGBModel, baseline_params, X, y, dev, cv, seed=seed)
            rec.update(auc=mean, auc_std=std, n_folds=n, skipped=False)
        rows.append(rec)

    table = pd.DataFrame(rows)
    table["plateau"] = _plateau_scores(table)
    viable = table[table["auc"].notna()]
    if viable.empty:
        raise ValueError("No barrier config passed the class-balance floor with a scorable AUC.")
    best_row = viable.loc[viable["plateau"].idxmax()]
    best = best_row.to_dict()
    return BarrierResult(table=table, best=best, n_configs=len(grid))
