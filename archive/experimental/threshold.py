"""Threshold gate :math:`p^* = L / (G + L)` -- Optional Session 3, slide 21.

The lecturer's simple decision rule: take the trade only when its expected
value is positive,

    p · r_G + (1 - p) · r_L > 0,    rearranging gives    p > p* = -r_L / (r_G - r_L).

When the average true-positive trade earns r_G > 0 and the average false-positive
trade loses r_L < 0, write G = r_G and L = -r_L (both positive) and the formula
becomes the standard form

    p* = L / (G + L).

The threshold inherits the bootstrap uncertainty of r_G and r_L since both
are sample averages. This module estimates the threshold AND a bootstrap
confidence interval, both from the training OOF (calibrated probability,
realised trade return) pairs.

We define a "trade" as an OOF row where the meta-model would have taken the
trade under a baseline gate (p̂ > 0.5):

    true-positive  (TP):  y = 1 and p̂ > 0.5  -- model said go, was right.
    false-positive (FP):  y = 0 and p̂ > 0.5  -- model said go, was wrong.

This is the lecturer's exact definition (slide 21).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdEstimate:
    """A point estimate of :math:`p^*` plus its bootstrap CI."""

    p_star: float
    ci_low: float
    ci_high: float
    n_tp: int
    n_fp: int
    rG_mean: float
    rL_mean: float
    bootstrap_p_stars: np.ndarray  # shape (B,) -- the bootstrap distribution


def _safe_p_star(rG: float, rL_signed: float, *, default: float = 0.5) -> float:
    """Compute p* = -rL / (rG - rL) with guard rails.

    ``rL_signed`` is the AVERAGE signed return on false positives (typically
    negative). The denominator ``rG - rL_signed`` must be positive for the
    formula to make sense; otherwise we clamp to a defensive default 0.5.
    """
    if not (np.isfinite(rG) and np.isfinite(rL_signed)):
        return default
    denom = rG - rL_signed
    if denom <= 0.0:
        return default
    p_star = -rL_signed / denom
    # Clamp to [0, 1]; a p* outside this is not actionable as a gate.
    return float(np.clip(p_star, 0.0, 1.0))


def estimate_threshold(
    proba: np.ndarray,
    label: np.ndarray,
    ret: np.ndarray,
    *,
    base_gate: float = 0.5,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> ThresholdEstimate:
    """Bootstrap-estimate :math:`p^* = L / (G + L)` from OOF training trades.

    Parameters
    ----------
    proba
        OOF calibrated probabilities, shape (N,).
    label
        Binary triple-barrier labels, shape (N,).
    ret
        Realised per-event signed returns, shape (N,).
    base_gate
        Defines "the trades we would have taken" -- only events with
        ``proba > base_gate`` enter the TP/FP pools. Default 0.5 per the lecture.
    n_bootstrap
        Number of resamples for the CI (default 2000).
    alpha
        Two-sided CI level; default 0.05 -> 95% CI.
    seed
        RNG seed for the resampling.
    """
    proba = np.asarray(proba, dtype=float)
    label = np.asarray(label, dtype=int)
    ret = np.asarray(ret, dtype=float)
    if not (len(proba) == len(label) == len(ret)):
        raise ValueError("proba, label, ret must be the same length")

    # Restrict to events the model would have taken under the base gate.
    take = proba > base_gate
    if not take.any():
        # Degenerate: model never wants to act. Threshold is the gate itself.
        return ThresholdEstimate(
            p_star=base_gate, ci_low=base_gate, ci_high=base_gate,
            n_tp=0, n_fp=0, rG_mean=float("nan"), rL_mean=float("nan"),
            bootstrap_p_stars=np.empty(0, dtype=float),
        )

    sub_label = label[take]
    sub_ret = ret[take]
    tp_mask = sub_label == 1
    fp_mask = sub_label == 0
    n_tp = int(tp_mask.sum())
    n_fp = int(fp_mask.sum())

    rG_mean = float(sub_ret[tp_mask].mean()) if n_tp else float("nan")
    rL_mean = float(sub_ret[fp_mask].mean()) if n_fp else float("nan")
    p_star = _safe_p_star(rG_mean, rL_mean)

    # Bootstrap CI: resample (event, ret) within each pool independently.
    rng = np.random.default_rng(seed)
    boot = np.full(n_bootstrap, np.nan, dtype=float)
    tp_returns = sub_ret[tp_mask]
    fp_returns = sub_ret[fp_mask]
    if n_tp >= 2 and n_fp >= 2:
        for b in range(n_bootstrap):
            tp_b = rng.choice(tp_returns, size=n_tp, replace=True)
            fp_b = rng.choice(fp_returns, size=n_fp, replace=True)
            boot[b] = _safe_p_star(float(tp_b.mean()), float(fp_b.mean()))

    boot_clean = boot[np.isfinite(boot)]
    if boot_clean.size:
        lo = float(np.quantile(boot_clean, alpha / 2.0))
        hi = float(np.quantile(boot_clean, 1.0 - alpha / 2.0))
    else:
        lo, hi = p_star, p_star

    return ThresholdEstimate(
        p_star=p_star, ci_low=lo, ci_high=hi,
        n_tp=n_tp, n_fp=n_fp, rG_mean=rG_mean, rL_mean=rL_mean,
        bootstrap_p_stars=boot_clean,
    )


def confusion_matrix_primary_vs_meta(
    proba: np.ndarray,
    label: np.ndarray,
    *,
    threshold: float,
) -> dict:
    """Two confusion matrices on out-of-fold predictions -- slide 22.

    "Primary alone" always predicts take (recall = 1, precision = base rate).
    "Primary + meta filter" takes only when ``proba >= threshold``.

    Returns a dict with both confusion counts plus precision/recall/F1 deltas.
    """
    proba = np.asarray(proba, dtype=float)
    label = np.asarray(label, dtype=int)

    take_meta = proba >= threshold
    # Primary alone -- always take.
    primary_tp = int(label.sum())
    primary_fp = int((1 - label).sum())
    # Meta-filtered.
    meta_tp = int(((take_meta) & (label == 1)).sum())
    meta_fp = int(((take_meta) & (label == 0)).sum())
    meta_fn = int(((~take_meta) & (label == 1)).sum())
    meta_tn = int(((~take_meta) & (label == 0)).sum())

    def _prec_rec(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else float("nan")
        return prec, rec, f1

    p_prec, p_rec, p_f1 = _prec_rec(primary_tp, primary_fp, 0)
    m_prec, m_rec, m_f1 = _prec_rec(meta_tp, meta_fp, meta_fn)

    return {
        "primary": {
            "tp": primary_tp, "fp": primary_fp, "fn": 0, "tn": 0,
            "precision": p_prec, "recall": p_rec, "f1": p_f1,
        },
        "meta": {
            "tp": meta_tp, "fp": meta_fp, "fn": meta_fn, "tn": meta_tn,
            "precision": m_prec, "recall": m_rec, "f1": m_f1,
        },
        "delta_precision": m_prec - p_prec,
        "delta_recall": m_rec - p_rec,
        "false_positives_avoided": primary_fp - meta_fp,
        "true_positives_missed": primary_tp - meta_tp,
    }
