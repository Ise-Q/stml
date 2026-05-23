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


# --------------------------------------------------------------------------- #
# Deep evaluation utilities (Stage 5b)                                        #
# --------------------------------------------------------------------------- #
def calibration_table(
    y_true: pd.Series,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Reliability diagram in tabular form: per-decile of predicted probability,
    report the mean predicted prob and the actual fraction of label=1.

    A well-calibrated model has `mean_pred ≈ actual_pos_rate` in every bin.
    """
    df = pd.DataFrame({"y": y_true.values, "p": y_score})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop", labels=False)
    out = (
        df.groupby("bin")
        .agg(n=("y", "size"), mean_pred=("p", "mean"), actual_pos=("y", "mean"))
        .round(4)
    )
    out["calibration_gap"] = (out["mean_pred"] - out["actual_pos"]).round(4)
    return out


def optimal_threshold(
    y_true: pd.Series,
    y_score: np.ndarray,
    metric: str = "f1",
    grid: Optional[np.ndarray] = None,
) -> tuple[float, dict[str, float]]:
    """Pick the threshold maximising ``metric`` ('f1' or 'precision' or 'youden').

    Returns ``(best_threshold, metrics_at_best)``.
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)
    sweep = threshold_sweep(y_true, y_score, thresholds=grid)
    if metric == "youden":
        # TPR - FPR
        from sklearn.metrics import confusion_matrix as _cm
        for thr in grid:
            y_pred = (y_score >= thr).astype(int)
            tn, fp, fn, tp = _cm(y_true, y_pred, labels=[0, 1]).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            sweep.loc[sweep.threshold == thr, "youden"] = tpr - fpr
    if metric not in sweep.columns:
        raise ValueError(f"Unknown metric {metric}; have {list(sweep.columns)}")
    best_idx = sweep[metric].idxmax()
    row = sweep.loc[best_idx]
    return float(row["threshold"]), row.to_dict()


def regime_conditional_performance(
    y_true: pd.Series,
    y_score: np.ndarray,
    regime_state: pd.Series,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Performance broken down by a categorical regime variable.

    ``regime_state`` should be a Series aligned with y_true/y_score (e.g. the
    HMM ``hmm_state_argmax`` column at each event date).

    Reports AUC, F1, log-loss, n per regime.
    """
    out_rows = []
    for state, idx in regime_state.groupby(regime_state).groups.items():
        sub_y = y_true.loc[idx]
        sub_p = pd.Series(y_score, index=y_true.index).loc[idx].values
        if sub_y.nunique() < 2:
            out_rows.append({"regime": state, "n": int(len(sub_y)),
                             "label_1_share": float(sub_y.mean()),
                             "auc": np.nan, "f1": np.nan, "log_loss": np.nan})
            continue
        rep = classification_report(sub_y, sub_p, threshold=threshold)
        out_rows.append({"regime": state, **{k: rep[k] for k in
                         ["n", "label_1_share", "auc", "f1", "log_loss"]}})
    return pd.DataFrame(out_rows).round(4).set_index("regime")


def model_comparison_table(
    results: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Convenience: take a dict like {model_name: classification_report dict}
    and return a side-by-side comparison DataFrame.
    """
    return pd.DataFrame(results).round(4)


def filtered_strategy_metrics(
    y_true: pd.Series,
    y_score: np.ndarray,
    side: pd.Series,
    ret: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Strategy-style metrics: if we ONLY take bets with score >= threshold,
    in the primary signal's direction, what's the realised mean ret, hit-rate,
    and Sharpe?

    Compares to "blind" (take every bet).
    """
    df = pd.DataFrame({
        "y": y_true.values, "p": y_score, "side": side.values, "ret": ret.values
    })
    df["taken_meta"] = (df["p"] >= threshold).astype(int)
    df["taken_blind"] = 1
    out = {}
    for tag, col in [("blind", "taken_blind"), ("meta", "taken_meta")]:
        sub = df.loc[df[col] == 1]
        if sub.empty:
            out[f"{tag}_n"] = 0
            out[f"{tag}_hit_rate"] = np.nan
            out[f"{tag}_mean_ret_bp"] = np.nan
            out[f"{tag}_sharpe"] = np.nan
            continue
        # signed_ret was computed in labeling as side * log(p_t1/p_t)
        signed = sub["ret"].values
        out[f"{tag}_n"] = int(len(sub))
        out[f"{tag}_hit_rate"] = float((signed > 0).mean())
        out[f"{tag}_mean_ret_bp"] = float(signed.mean() * 1e4)
        # Annualised "Sharpe-like": mean/std times sqrt(252/h)
        out[f"{tag}_sharpe"] = float(
            (signed.mean() / signed.std()) * np.sqrt(252 / 10)
        ) if signed.std() > 0 else np.nan
    return out
