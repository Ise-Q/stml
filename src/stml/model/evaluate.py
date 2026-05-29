"""
evaluate.py
===========
Metrics, plots and the single test-set gate for the meta-model.

The headline metric is **ROC-AUC** (threshold-free, imbalance-robust); we also report
average-precision, Brier (calibration), and F1/precision/recall at a chosen threshold. Per the
project's leakage discipline the test partition is opened exactly once, through
:func:`release_test` -- a tripwire mirroring :func:`stml.replication.splits.get_test` that refuses
to hand back test rows without an explicit ``final_confirmation=True``.

Because the pooled AUC is dominated by the data-rich instruments, :func:`per_instrument_breakdown`
reports AUC per instrument so thin / degenerate names (ho1s, ng1s) are visible rather than hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(y: np.ndarray, proba: np.ndarray, *, threshold: float = 0.5) -> dict:
    """Standard binary metrics for one set of probabilities."""
    y = np.asarray(y).astype(int)
    pred = (proba >= threshold).astype(int)
    n_classes = np.unique(y).size
    return {
        "n": int(y.size),
        "pos_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, proba)) if n_classes == 2 else float("nan"),
        "ap": float(average_precision_score(y, proba)) if n_classes == 2 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, proba)),
    }


def per_instrument_breakdown(
    instruments: np.ndarray, y: np.ndarray, proba: np.ndarray, *, min_n: int = 10
) -> pd.DataFrame:
    """AUC / positive-rate / count per instrument; AUC is NaN where a class is missing."""
    df = pd.DataFrame({"instrument": instruments, "y": np.asarray(y).astype(int), "p": proba})
    rows = []
    for inst, g in df.groupby("instrument", sort=True):
        two = g["y"].nunique() == 2
        rows.append({
            "instrument": inst,
            "n": int(len(g)),
            "pos_rate": float(g["y"].mean()),
            "auc": float(roc_auc_score(g["y"], g["p"])) if (two and len(g) >= min_n) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("instrument").reset_index(drop=True)


def plot_roc_pr(curves: dict[str, tuple[np.ndarray, np.ndarray]], ax=None):
    """Overlay ROC curves for ``{label: (y, proba)}`` on one axis."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    for label, (y, proba) in curves.items():
        fpr, tpr, _ = roc_curve(y, proba)
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc_score(y, proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC")
    ax.legend(loc="lower right", fontsize=8)
    return ax


def plot_calibration(y: np.ndarray, proba: np.ndarray, *, n_bins: int = 10, ax=None):
    """Reliability curve: mean predicted prob vs observed frequency per bin."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(proba, bins) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            xs.append(proba[m].mean())
            ys.append(np.asarray(y)[m].mean())
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(xs, ys, "o-", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration")
    ax.legend(fontsize=8)
    return ax


def release_test(matrix: pd.DataFrame, *, final_confirmation: bool = False) -> pd.DataFrame:
    """Return the test-partition rows -- only behind an explicit confirmation (tripwire).

    Mirrors :func:`stml.replication.splits.get_test`: the test block is the one-time, untouchable
    evaluation set, so accessing it is forced to be a deliberate, auditable act. Call this exactly
    once, in the final-report cell.
    """
    if not final_confirmation:
        raise RuntimeError(
            "Refusing to expose the test partition. It is the final evaluation set and must not "
            "be touched during barrier search or model tuning. Pass final_confirmation=True only "
            "for the one-time final report."
        )
    return matrix[matrix["partition"] == "test"].reset_index(drop=True)
