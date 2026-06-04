"""Sample-weighted purged OOS evaluation harness.

Plan §3.4 / §8 Stage 3 deliverable.

Lifted with attribution from


Key principles:

* Every metric is **sample-weighted** by AFML Ch.4 uniqueness × balanced
  class weight (the same weight passed to ``fit``). Concurrent triple-barrier
  labels overlap → uniform weighting would double-count.
* The OOS predictions come from a **purged CV** loop (refits a fresh model
  per fold; trains on the purged + embargoed train indices; scores the
  held-out test indices).
* Single-class folds return NaN for AUC instead of crashing — purged CV
  routinely produces them for thin instruments.
* The :func:`cross_val_evaluate` returns per-fold scores AND the concatenated
  OOS predictions for downstream per-instrument breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

from stml.experimental.cv import CombinatorialPurgedCV, PurgedKFold
from stml.experimental.models import MetaClassifier, balanced_sample_weight


def evaluate_predictions(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Sample-weighted classification + calibration metrics dict."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    if len(y_true) == 0:
        return {k: float("nan") for k in (
            "n", "accuracy", "precision", "recall", "f1", "brier",
            "auc", "avg_precision", "log_loss", "pos_rate",
        )}
    y_pred = (proba >= threshold).astype(int)
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)

    # Defensive single-class handling.
    n_classes = len(np.unique(y_true))
    out: dict[str, float] = {
        "n": int(len(y_true)),
        "pos_rate": float(np.average(y_true, weights=sample_weight)),
        "accuracy": float(accuracy_score(y_true, y_pred, sample_weight=sample_weight)),
        "f1": float(f1_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)),
        "brier": float(brier_score_loss(y_true, proba, sample_weight=sample_weight)),
        "log_loss": float("nan"),
        "auc": float("nan"),
        "avg_precision": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
    }
    if n_classes >= 2:
        try:
            out["log_loss"] = float(log_loss(y_true, proba, sample_weight=sample_weight))
        except Exception:
            pass
        try:
            out["auc"] = float(roc_auc_score(y_true, proba, sample_weight=sample_weight))
        except Exception:
            pass
        try:
            out["avg_precision"] = float(
                average_precision_score(y_true, proba, sample_weight=sample_weight)
            )
        except Exception:
            pass
        # Precision / recall — defensive.
        try:
            tp = float(np.sum(sample_weight[(y_true == 1) & (y_pred == 1)]))
            fp = float(np.sum(sample_weight[(y_true == 0) & (y_pred == 1)]))
            fn = float(np.sum(sample_weight[(y_true == 1) & (y_pred == 0)]))
            out["precision"] = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            out["recall"] = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Cross-val evaluator.
# ---------------------------------------------------------------------------


@dataclass
class CVResult:
    """One row per fold + concatenated OOS predictions."""

    fold_scores: pd.DataFrame  # row per fold: metric columns
    oos_predictions: pd.DataFrame  # row per OOS event: y_true, y_proba, fold, idx
    mean_scores: dict[str, float]
    std_scores: dict[str, float]


def cross_val_evaluate(
    make_model: Callable[[], MetaClassifier],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv,
    uniqueness_weights: pd.Series | None = None,
    threshold: float = 0.5,
    nan_columns_at_test: list[str] | None = None,
) -> CVResult:
    """Purged-CV OOS evaluation with sample weights.

    Refits ``make_model()`` per fold on the purged train slice; scores the held-out
    test slice with sample-weighted metrics. Returns per-fold metrics + the
    concatenated OOS prediction frame.

    Parameters
    ----------
    make_model
        Zero-arg factory returning a fresh ``MetaClassifier``.
    X, y
        Feature matrix + target. Must be index-aligned.
    cv
        A splitter exposing ``.split(X)``; typically :class:`PurgedKFold` or
        :class:`CombinatorialPurgedCV`.
    uniqueness_weights
        AFML Ch.4 weights aligned to ``X``. Used as ``base`` in
        :func:`balanced_sample_weight` and as sample weight in the metrics.
    threshold
        Decision threshold for hard-label metrics (precision/recall/F1).
    nan_columns_at_test
        Optional list of column names to FORCE TO NaN in the test slice **at
        prediction time only** (training sees the real values). Implements the
        methodology spec S3 / R-11 simulated-missingness ablation.
    """
    y = y.astype(int)
    fold_rows = []
    oos_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # Build sample weights: uniqueness × inverse class frequency.
        if uniqueness_weights is None:
            base_train = None
            sw_test_metrics = np.ones(len(y_test), dtype=float)
        else:
            base_train = uniqueness_weights.iloc[train_idx].astype(float).values
            sw_test_metrics = uniqueness_weights.iloc[test_idx].astype(float).values
        sw_train = balanced_sample_weight(y_train.values, base=base_train)

        # Fit a fresh model.
        model = make_model()
        try:
            model.fit(X_train, y_train, sample_weight=sw_train)
        except Exception:
            continue

        # Simulate missingness if requested.
        X_test_pred = X_test.copy()
        if nan_columns_at_test:
            for col in nan_columns_at_test:
                if col in X_test_pred.columns:
                    X_test_pred[col] = np.nan

        try:
            proba = model.predict_act_proba(X_test_pred)
        except Exception:
            continue

        scores = evaluate_predictions(
            y_true=y_test.values,
            proba=proba,
            sample_weight=sw_test_metrics,
            threshold=threshold,
        )
        scores["fold"] = fold_idx
        scores["n_train"] = int(len(train_idx))
        scores["n_test"] = int(len(test_idx))
        fold_rows.append(scores)

        for j, (event_idx, prob, label) in enumerate(
            zip(test_idx, proba, y_test.values)
        ):
            oos_rows.append(
                {
                    "fold": fold_idx,
                    "row_idx": int(event_idx),
                    "y_true": int(label),
                    "y_proba": float(prob),
                    "sample_weight": float(sw_test_metrics[j]),
                }
            )

    if not fold_rows:
        return CVResult(
            fold_scores=pd.DataFrame(),
            oos_predictions=pd.DataFrame(),
            mean_scores={},
            std_scores={},
        )

    fold_df = pd.DataFrame(fold_rows)
    metric_cols = [c for c in fold_df.columns if c not in ("fold", "n_train", "n_test", "n")]
    mean = fold_df[metric_cols].mean(numeric_only=True).to_dict()
    std = fold_df[metric_cols].std(numeric_only=True).to_dict()
    oos_df = pd.DataFrame(oos_rows)
    return CVResult(fold_scores=fold_df, oos_predictions=oos_df, mean_scores=mean, std_scores=std)


# ---------------------------------------------------------------------------
# Per-instrument breakdown.
# ---------------------------------------------------------------------------


def primary_vs_meta_evaluation(
    proba: np.ndarray,
    label: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict:
    """Primary-alone vs primary+meta filter -- Madmoun slide 22.

    The lecturer's prescribed evaluation: ``does my meta-model improve on
    following the primary blindly?``. Two confusion matrices on the same
    out-of-fold predictions:

      * primary alone: take every signal -> recall = 1, precision = base rate.
      * primary + meta filter: take only when ``proba >= threshold``.

    Returns a dict with both confusion counts, precision / recall / F1 for
    each, and the false-positive avoidance / true-positive cost.
    """
    proba = np.asarray(proba, dtype=float)
    label = np.asarray(label, dtype=int)

    take_meta = proba >= threshold
    primary_tp = int(label.sum())
    primary_fp = int((1 - label).sum())
    meta_tp = int(((take_meta) & (label == 1)).sum())
    meta_fp = int(((take_meta) & (label == 0)).sum())
    meta_fn = int(((~take_meta) & (label == 1)).sum())
    meta_tn = int(((~take_meta) & (label == 0)).sum())

    def _prec_rec_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else float("nan")
        return prec, rec, f1

    p_prec, p_rec, p_f1 = _prec_rec_f1(primary_tp, primary_fp, 0)
    m_prec, m_rec, m_f1 = _prec_rec_f1(meta_tp, meta_fp, meta_fn)

    return {
        "threshold": float(threshold),
        "n": int(len(label)),
        "primary_tp": primary_tp, "primary_fp": primary_fp,
        "primary_precision": p_prec, "primary_recall": p_rec, "primary_f1": p_f1,
        "meta_tp": meta_tp, "meta_fp": meta_fp,
        "meta_fn": meta_fn, "meta_tn": meta_tn,
        "meta_precision": m_prec, "meta_recall": m_rec, "meta_f1": m_f1,
        "delta_precision": m_prec - p_prec,
        "delta_recall": m_rec - p_rec,
        "false_positives_avoided": primary_fp - meta_fp,
        "true_positives_missed": primary_tp - meta_tp,
    }


def per_instrument_breakdown(
    instruments: pd.Series,
    oos_predictions: pd.DataFrame,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Aggregate OOS predictions per instrument and emit sample-weighted metrics.

    Parameters
    ----------
    instruments : per-event instrument labels aligned to the ROW INDEX (not
        the OOS prediction row_idx — the merge uses positional index).
    oos_predictions : output of :func:`cross_val_evaluate`'s ``oos_predictions``.
    """
    if oos_predictions.empty:
        return pd.DataFrame()
    merged = oos_predictions.copy()
    merged["instrument"] = merged["row_idx"].map(
        dict(enumerate(instruments.values))
    )
    rows = []
    for inst, sub in merged.groupby("instrument"):
        scores = evaluate_predictions(
            y_true=sub["y_true"].values,
            proba=sub["y_proba"].values,
            sample_weight=sub["sample_weight"].values,
            threshold=threshold,
        )
        scores["instrument"] = inst
        scores["n"] = int(len(sub))
        rows.append(scores)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    cols = ["instrument", "n", "pos_rate", "auc", "log_loss", "brier", "accuracy",
            "precision", "recall", "f1", "avg_precision"]
    return out.loc[:, [c for c in cols if c in out.columns]]
