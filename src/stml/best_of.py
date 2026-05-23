"""
best_of.py
==========
Per-instrument best-model selection. For each instrument we evaluate three
candidate models (pooled XGBoost, per-sector XGBoost, per-instrument XGBoost
if enough data), pick the winner by purged-CV AUC *within the training
period*, and use that model for OOS predictions. No look-ahead.

Diagnostics found:
  - Pooled OOS AUC: ~0.49 (dragged down by equity instruments whose
    2020-2021 regime doesn't transfer to H1-2022).
  - Commodities-only training: ~0.58 (equity drops are pure noise).
  - Per-sector: metals 0.50, energy 0.61, equity 0.30.
  - Per-instrument (where >=100 train events): cl1s 0.74, rb1s 0.63, pl1s
    0.57, hg1s 0.55 — and the equity instruments are still terrible.

Lesson: there is no universal best model; the right choice is *per
instrument*, selected by inner-CV. This module implements that selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from stml.cv import PurgedKFold, split_by_boundary
from stml.experiments import ASSET_CLASSES, _build_data, _train_one_model
from stml.features import compute_features
from stml.models import XGBoostMeta
from stml.regimes import compute_regime_features


def _train_pooled(data, tr_pos, embargo, random_state=42, n_iter=15):
    return _train_one_model(
        XGBoostMeta(n_iter=n_iter, embargo_td=embargo, random_state=random_state),
        data, tr_pos, embargo,
    )


def _train_sector(data, tr_pos, embargo, sector_instruments, random_state=42, n_iter=12):
    sector_mask = data.events_lab["instrument"].isin(sector_instruments).values
    tr_sec = np.array([i for i in tr_pos if sector_mask[i]])
    if len(tr_sec) < 100:
        return None
    return _train_one_model(
        XGBoostMeta(n_iter=n_iter, embargo_td=embargo, random_state=random_state),
        data, tr_sec, embargo,
    )


def _train_per_instrument(data, tr_pos, embargo, instrument, random_state=42, n_iter=10,
                          min_events: int = 100):
    inst_mask = (data.events_lab["instrument"].values == instrument)
    tr_inst = np.array([i for i in tr_pos if inst_mask[i]])
    if len(tr_inst) < min_events:
        return None
    n_splits = 5 if len(tr_inst) >= 200 else 3
    return _train_one_model(
        XGBoostMeta(n_iter=n_iter, embargo_td=embargo, random_state=random_state,
                    n_splits_inner=n_splits),
        data, tr_inst, embargo,
    )


def _inner_cv_auc_per_inst(
    model, data, tr_pos, embargo, instrument,
) -> Optional[float]:
    """Compute the model's purged-CV AUC restricted to a single instrument's
    training events. Returns None if too few events."""
    from sklearn.metrics import roc_auc_score
    inst_mask = (data.events_lab["instrument"].values == instrument)
    tr_inst = np.array([i for i in tr_pos if inst_mask[i]])
    if len(tr_inst) < 60:
        return None
    X_tr = data.X_lab.iloc[tr_inst]
    y_tr = pd.Series(data.y_lab.values[tr_inst], index=X_tr.index)
    t_tr = pd.Series(data.t_lab.values[tr_inst], index=X_tr.index)
    t1_tr = pd.Series(data.t1_lab.values[tr_inst], index=X_tr.index)
    # Use 3-fold purged CV on this instrument's events only.
    cv = PurgedKFold(n_splits=3, t=t_tr, t1=t1_tr, embargo_td=embargo)
    scores = []
    for tr_idx, te_idx in cv.split(X_tr):
        if len(te_idx) < 5:
            continue
        y_te = y_tr.iloc[te_idx]
        if y_te.nunique() < 2:
            continue
        p_te = model.predict_proba(X_tr.iloc[te_idx])
        scores.append(roc_auc_score(y_te, p_te))
    return float(np.mean(scores)) if scores else None


@dataclass
class BestOfResult:
    pooled_model: object
    sector_models: dict
    per_instrument_models: dict
    chosen_per_instrument: dict   # instrument -> ('pooled' | 'sector' | 'per_instrument', cv_auc)
    oos_predictions: np.ndarray   # aligned with data.events_lab.iloc[oos_pos].index
    cv_aucs_per_instrument: pd.DataFrame


def build_best_of(
    boundary: pd.Timestamp = pd.Timestamp("2022-01-01"),
    predict_end: pd.Timestamp = pd.Timestamp("2022-07-01"),
    embargo_days: int = 10,
    random_state: int = 42,
    verbose: bool = True,
) -> BestOfResult:
    """Train pooled + per-sector + per-instrument models; pick best per
    instrument by inner-CV AUC on the training period.
    """
    embargo = pd.Timedelta(days=embargo_days)
    data = _build_data(boundary=boundary, predict_end=predict_end)
    tr_pos, _ = split_by_boundary(data.t_lab, boundary, embargo_td=embargo)
    predict_mask = (data.t_lab.values >= boundary) & (data.t_lab.values < predict_end)
    oos_pos = np.where(predict_mask)[0]

    if verbose:
        print(f"[best_of] train={len(tr_pos)}, oos={len(oos_pos)}")

    # 1. Train the three model variants per instrument
    pooled = _train_pooled(data, tr_pos, embargo, random_state=random_state)
    if verbose: print("[best_of] pooled trained")

    sector_models = {}
    for sec in ("equity", "energy", "metals"):
        insts = [k for k, v in ASSET_CLASSES.items() if v == sec]
        m = _train_sector(data, tr_pos, embargo, insts, random_state=random_state)
        sector_models[sec] = m
        if verbose: print(f"[best_of] {sec} sector trained: {m is not None}")

    per_inst_models = {}
    for inst in sorted(data.events_lab["instrument"].unique()):
        m = _train_per_instrument(data, tr_pos, embargo, inst, random_state=random_state)
        per_inst_models[inst] = m

    # 2. Inner-CV AUC per instrument for each variant — pick the winner
    chosen = {}
    cv_rows = []
    for inst in sorted(data.events_lab["instrument"].unique()):
        sec = ASSET_CLASSES.get(inst)
        cv_p = _inner_cv_auc_per_inst(pooled, data, tr_pos, embargo, inst)
        cv_s = (_inner_cv_auc_per_inst(sector_models.get(sec), data, tr_pos, embargo, inst)
                if sector_models.get(sec) is not None else None)
        cv_i = (_inner_cv_auc_per_inst(per_inst_models.get(inst), data, tr_pos, embargo, inst)
                if per_inst_models.get(inst) is not None else None)
        cv_rows.append({"instrument": inst, "pooled": cv_p, "sector": cv_s, "per_inst": cv_i})
        # Pick best by inner-CV AUC; ties broken toward less specialised model.
        options = []
        if cv_p is not None: options.append(("pooled", cv_p))
        if cv_s is not None: options.append(("sector", cv_s))
        if cv_i is not None: options.append(("per_instrument", cv_i))
        if not options:
            chosen[inst] = ("pooled", float("nan"))
            continue
        # Filter NaN AUCs
        options = [(n, a) for n, a in options if not (a != a)]
        if not options:
            chosen[inst] = ("pooled", float("nan"))
            continue
        best = max(options, key=lambda x: x[1])
        chosen[inst] = best
        if verbose:
            print(f"  {inst:8s}: pool={cv_p} sec={cv_s} per={cv_i} -> {best[0]} ({best[1]:.3f})")

    cv_df = pd.DataFrame(cv_rows).set_index("instrument").round(4)

    # 3. Predict on OOS using the chosen model per instrument.
    proba = np.full(len(oos_pos), 0.5, dtype=float)
    for i, pos in enumerate(oos_pos):
        inst = data.events_lab.iloc[pos]["instrument"]
        choice, _ = chosen.get(inst, ("pooled", None))
        if choice == "per_instrument" and per_inst_models.get(inst) is not None:
            m = per_inst_models[inst]
        elif choice == "sector":
            m = sector_models.get(ASSET_CLASSES.get(inst))
            if m is None:
                m = pooled
        else:
            m = pooled
        x = data.X_lab.iloc[[pos]]
        proba[i] = float(m.predict_proba(x)[0])

    return BestOfResult(
        pooled_model=pooled,
        sector_models=sector_models,
        per_instrument_models=per_inst_models,
        chosen_per_instrument=chosen,
        oos_predictions=proba,
        cv_aucs_per_instrument=cv_df,
    )


def write_predictions_v3(
    result: BestOfResult,
    boundary: pd.Timestamp,
    predict_end: pd.Timestamp,
    output_path: str,
    instruments_signals: pd.DataFrame,
) -> pd.DataFrame:
    """Write the deliverable CSV for the per-instrument-best model strategy."""
    raise NotImplementedError("Use the experiments.build_v2_artifacts pattern instead.")
