"""Out-of-sample evaluation harness for the meta-labelling horse-race (Stage 2, §5).

Adapts PS5's ``evaluate_model`` (cell 72) — a fixed-0.5-threshold accuracy/precision/recall/
F1/AUC dict — into the metamodel's needs:

- **Sample-weighted** metrics. Triple-barrier labels overlap, so every score is weighted by
  the López de Prado uniqueness weight (passed identically to the model's ``fit`` and to the
  metric functions) — otherwise concurrent labels double-count in the OOS estimate.
- **Calibration** metrics (Brier, log-loss, average-precision) alongside ranking AUC: the
  downstream fractional-Kelly sizing consumes the probability itself, so calibration matters
  (nlr-cw §2). A tunable **decision threshold** drives the hard-label metrics (PS5 was 0.5-only).
- **Degenerate-fold safety.** Under purging a fold can be single-class; ranking metrics then
  return NaN (documented) rather than crashing.
- **Purged CV wiring.** ``cross_val_evaluate`` refits the estimator fresh per fold on the
  purged+embargoed train slice (``cross_validation.PurgedKFold`` / CPCV) and scores the held-out
  OOS slice — the harness that produces the §5 OOS distribution and the blind-primary comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(
    y_true,
    proba,
    *,
    sample_weight=None,
    threshold: float = 0.5,
) -> dict:
    """Sample-weighted classification metrics for one set of OOS probabilities.

    Hard-label metrics use ``proba >= threshold``; ranking/calibration metrics use the raw
    probability. Returns ``n``, ``threshold``, ``accuracy``, ``precision``, ``recall``, ``f1``,
    ``brier`` and (NaN if the fold is single-class) ``auc``, ``avg_precision``, ``log_loss``.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    pred = (p >= threshold).astype(int)

    out = {
        "n": int(len(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred, sample_weight=w)),
        "precision": float(precision_score(y, pred, sample_weight=w, zero_division=0)),
        "recall": float(recall_score(y, pred, sample_weight=w, zero_division=0)),
        "f1": float(f1_score(y, pred, sample_weight=w, zero_division=0)),
        "brier": float(brier_score_loss(y, p, sample_weight=w)),
    }
    if len(np.unique(y)) == 2:  # ranking/calibration need both classes present
        out["auc"] = float(roc_auc_score(y, p, sample_weight=w))
        out["avg_precision"] = float(average_precision_score(y, p, sample_weight=w))
        out["log_loss"] = float(log_loss(y, p, sample_weight=w, labels=[0, 1]))
    else:
        out["auc"] = float("nan")
        out["avg_precision"] = float("nan")
        out["log_loss"] = float("nan")
    return out


def evaluate_oos(
    model,
    x_test,
    y_test,
    *,
    sample_weight=None,
    threshold: float = 0.5,
    model_name: str = "",
) -> dict:
    """Score an already-fitted model on a held-out OOS slice (model fit on the purged train)."""
    proba = model.predict_act_proba(x_test)
    out = evaluate_predictions(y_test, proba, sample_weight=sample_weight, threshold=threshold)
    out["model"] = model_name
    return out


def always_act_baseline(y_true, *, sample_weight=None, threshold: float = 0.5) -> dict:
    """The blind-primary baseline: act on every signal (constant P(act)=1).

    The metamodel must beat this on precision/F1 to justify its complexity (§5). AUC is NaN
    (a constant score cannot rank).
    """
    y = np.asarray(y_true)
    return evaluate_predictions(
        y, np.ones(len(y)), sample_weight=sample_weight, threshold=threshold
    )


def threshold_sweep(y_true, proba, *, sample_weight=None, thresholds=None) -> pd.DataFrame:
    """Precision/recall/F1/accuracy across decision thresholds (§5 threshold sweep)."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.1, 0.91, 0.05), 2)
    rows = [
        evaluate_predictions(y_true, proba, sample_weight=sample_weight, threshold=float(t))
        for t in thresholds
    ]
    return pd.DataFrame(rows)


def cross_val_evaluate(
    make_model,
    X,
    y,
    cv,
    *,
    sample_weight=None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Refit ``make_model()`` fresh per purged fold and score the held-out OOS slice.

    ``cv`` is a ``PurgedKFold`` / ``CombinatorialPurgedCV`` keyed on the label ``t1`` spans, so
    each train slice is purged + embargoed against its test window. ``sample_weight`` (e.g.
    uniqueness) is sliced per fold and passed to both ``fit`` and the OOS metrics. Returns one
    row per fold with a ``fold`` index.
    """
    y = np.asarray(y)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

    rows = []
    for fold, (train, test) in enumerate(cv.split(X)):  # X is index-aligned to cv's t1
        model = make_model()
        x_tr = X.iloc[train] if hasattr(X, "iloc") else np.asarray(X)[train]
        x_te = X.iloc[test] if hasattr(X, "iloc") else np.asarray(X)[test]
        w_tr = None if w is None else w[train]
        model.fit(x_tr, y[train], sample_weight=w_tr)
        proba = model.predict_act_proba(x_te)
        w_te = None if w is None else w[test]
        metrics = evaluate_predictions(
            y[test], proba, sample_weight=w_te, threshold=threshold
        )
        metrics["fold"] = fold
        rows.append(metrics)
    return pd.DataFrame(rows)
