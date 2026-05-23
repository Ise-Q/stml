"""
evaluation.py
=============
Classification metrics, baseline comparisons, threshold analysis, and the
per-instrument breakdown required by the assignment's evaluation rubric
(20 marks).

Public API:
  - :func:`classification_report`   -- precision/recall/F1/AUC/log-loss/Brier
  - :func:`per_instrument_breakdown` -- same metrics, broken out per instrument
  - :func:`confusion_matrix_df`     -- a labelled 2x2 confusion matrix
  - :func:`threshold_sweep`         -- precision/recall vs decision threshold
  - :func:`baseline_compare`        -- meta-model vs "follow primary blindly"
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_report(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float = 0.5,
    sample_weight: Optional[pd.Series] = None,
) -> dict[str, float]:
    """Compact dict of headline classification metrics."""
    y_pred = (y_score >= threshold).astype(int)
    sw = None if sample_weight is None else sample_weight.values
    # Guard against single-class y for AUC.
    auc = roc_auc_score(y_true, y_score, sample_weight=sw) if y_true.nunique() > 1 else np.nan
    return {
        "n": int(len(y_true)),
        "label_1_share": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred, sample_weight=sw)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0, sample_weight=sw)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0, sample_weight=sw)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0, sample_weight=sw)),
        "auc": float(auc),
        "avg_precision": float(average_precision_score(y_true, y_score, sample_weight=sw))
            if y_true.nunique() > 1 else np.nan,
        "log_loss": float(log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7), sample_weight=sw)),
        "brier": float(brier_score_loss(y_true, y_score, sample_weight=sw)),
    }


def per_instrument_breakdown(
    events: pd.DataFrame,
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Per-instrument classification metrics.

    ``events`` must have an ``instrument`` column aligned (by index) with
    ``y_true`` and ``y_score``.
    """
    rows = []
    for inst, idx in events.groupby("instrument").groups.items():
        sub_y = y_true.loc[idx]
        sub_s = pd.Series(y_score, index=y_true.index).loc[idx].values
        if sub_y.nunique() < 2:
            # Single-class instrument: AUC undefined; report partial metrics.
            rows.append({
                "instrument": inst, "n": len(sub_y), "label_1_share": float(sub_y.mean()),
                "auc": np.nan, "f1": np.nan, "precision": np.nan,
                "recall": np.nan, "brier": float(brier_score_loss(sub_y, sub_s)),
            })
            continue
        rows.append({"instrument": inst, **classification_report(sub_y, sub_s, threshold=threshold)})
    return pd.DataFrame(rows).set_index("instrument").round(4)


def confusion_matrix_df(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=pd.Index(["true_0", "true_1"], name="actual"),
        columns=pd.Index(["pred_0", "pred_1"], name="predicted"),
    )


def threshold_sweep(
    y_true: pd.Series,
    y_score: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Precision, recall, F1, # selected at each threshold."""
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 51)
    rows = []
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        rows.append({
            "threshold": float(thr),
            "n_selected": int(y_pred.sum()),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
        })
    return pd.DataFrame(rows)


def baseline_compare(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Meta-model (filtered) vs "follow primary blindly" baseline.

    Both are evaluated as binary classifiers of the same y_true. The blind
    baseline always predicts 1 (take every primary bet).
    """
    meta = classification_report(y_true, y_score, threshold=threshold)
    # Blind baseline = predict 1 for everyone. Score = 1.0 for all.
    blind_score = np.ones_like(y_score)
    blind = classification_report(y_true, blind_score, threshold=0.5)
    return pd.DataFrame({"meta": meta, "blind_primary": blind}).round(4)
