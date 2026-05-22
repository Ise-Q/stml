"""
metrics.py
==========
Imbalance- and baseline-robust discrepancy metrics for the primary signal
``s_t in {-1, 0, +1}``.

The headline entry point is :func:`panel`, which scores a predicted signal
``y_pred`` against the released signal ``y_true`` with a battery of metrics
chosen so a *degenerate* guess (always-flat, majority-class, label-shuffled)
scores at chance (~0), not at the deceptively-high raw accuracy that class
imbalance would otherwise produce. A perfect replica scores 1.

Design points (see ``.omc/scratch/CONTRACT.md``, US-003b):

* Every scikit-learn call passes ``zero_division=0`` where the parameter
  exists, so absent predicted classes never raise or emit warnings.
* Labels that are structurally **absent from ``y_true``** (e.g. ng1s never
  prints ``+1``) are dropped from the macro aggregates and reported as
  ``None`` -- never ``0`` -- in ``per_class``: a class the signal can't take
  carries no recall information and must not dilute the macro mean.
* Single-class ``y_true`` (e.g. an all-flat split) leaves Cohen's kappa
  mathematically undefined (0/0); we coerce it to ``0.0`` so the scalar
  metrics never contain NaN.
* The ordinal skill-score weights a sign-flip (``-1 <-> +1``) as twice as
  costly as a move to/from flat. It is **chance-corrected** -- normalised by
  the expected cost of a prediction whose label marginal is independent of
  ``y_true`` -- so that, exactly as for kappa/MCC, *any* marginal-only
  predictor (always-flat, majority-class, stratified-random, label-shuffle)
  scores ~0 while a perfect replica scores 1. A pure ``1 - D_model/D_baseline``
  ratio cannot do this: on an 80%-flat series the flat baseline's cost
  collapses toward 0, so the ratio explodes and a constant-mode guess beats it
  — the very imbalance failure this module exists to prevent.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from stml.replication.baselines import always_flat, stratified_random

# The fixed signal alphabet. per_class is reported over this domain (plus any
# stray labels actually seen) so a class structurally absent from y_true is
# explicitly None rather than silently missing from the dict.
_SIGNAL_LABELS = (-1, 0, 1)

# Ordinal misclassification cost: a full sign flip is twice as bad as a move
# to/from flat. cost[a + 1, b + 1] indexes labels {-1, 0, +1} -> {0, 1, 2}.
_ORDINAL_COST = np.array(
    [
        [0.0, 1.0, 2.0],  # true = -1
        [1.0, 0.0, 1.0],  # true =  0
        [2.0, 1.0, 0.0],  # true = +1
    ]
)


def _as_int_array(y: np.ndarray | pd.Series | list[int]) -> np.ndarray:
    """Coerce a label container to a 1-D integer ``np.ndarray``."""
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D label array, got shape {arr.shape}")
    return arr.astype(int)


@contextmanager
def _quiet_degenerate_warnings():
    """Silence the two benign scikit-learn UserWarnings that the contract's
    degenerate inputs deliberately trigger: a single observed class (kappa /
    confusion ambiguity) and ``y_pred`` carrying a class absent from ``y_true``.
    Both are expected here -- supporting those cases without NaN/crash is the
    point -- so callers should not see spurious noise."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="A single label was found", category=UserWarning)
        warnings.filterwarnings(
            "ignore", message="y_pred contains classes not in y_true", category=UserWarning
        )
        yield


def _ordinal_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean ordinal cost ``D(a, b) = mean_i cost(a_i, b_i)`` over labels +-1/0."""
    return float(_ORDINAL_COST[a + 1, b + 1].mean())


def _label_marginal(y: np.ndarray) -> np.ndarray:
    """Empirical probability of each label in ``_SIGNAL_LABELS`` within ``y``."""
    n = y.shape[0]
    return np.array([(y == lab).sum() / n for lab in _SIGNAL_LABELS])


def _expected_independent_cost(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Expected ordinal cost if ``y_pred``'s labels were drawn independently of
    ``y_true`` -- i.e. the cross-product of the two empirical marginals,
    ``sum_{j,k} p_pred[j] p_true[k] cost(j, k)``.

    This is the chance reference: a prediction that knows only its own label
    frequencies and nothing about *when* each label occurs. Because a constant
    predictor's marginal is a point mass, its actual cost equals this expected
    cost exactly, so any constant guess (flat or majority) scores 0 -- the same
    chance-correction that makes Cohen's kappa 0 for a constant prediction.
    """
    p_pred = _label_marginal(y_pred)
    p_true = _label_marginal(y_true)
    return float(p_pred @ _ORDINAL_COST @ p_true)


def _safe_kappa(
    y_true: np.ndarray, y_pred: np.ndarray, weights: str | None = None
) -> float:
    """Cohen's kappa, coerced to ``0.0`` when it is undefined (single class).

    With only one observed class kappa is ``0 / 0``; sklearn returns NaN. There
    is no agreement-beyond-chance to measure in that case, so chance-level
    ``0.0`` is the correct, NaN-free convention.
    """
    if np.unique(np.concatenate([y_true, y_pred])).size < 2:
        return 0.0
    k = cohen_kappa_score(y_true, y_pred, weights=weights)
    return 0.0 if np.isnan(k) else float(k)


def _chance_corrected_skill(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Intrinsic chance-corrected ordinal skill of ``y_pred`` against ``y_true``.

    ``SS = 1 - D(y_pred, y_true) / D_chance(y_pred)``, where ``D_chance`` is the
    cost expected if ``y_pred``'s *own* label marginal were realised
    independently of ``y_true`` (:func:`_expected_independent_cost`). Computing
    the reference from ``y_pred``'s marginal is what makes the score
    imbalance-robust: every marginal-only predictor -- flat, majority,
    stratified-random, shuffle -- has ``D_model == D_chance`` in expectation and
    so scores ~0, while a perfect replica scores 1. ``D_chance == 0`` only for a
    degenerate single-mass predictor that also matches ``y_true`` perfectly; we
    return ``0.0`` there.
    """
    d_chance = _expected_independent_cost(y_pred, y_true)
    if d_chance == 0.0:
        return 0.0
    d_model = _ordinal_distance(y_pred, y_true)
    return 1.0 - d_model / d_chance


def _ordinal_skill(
    y_true: np.ndarray, y_pred: np.ndarray, baseline: np.ndarray
) -> float:
    """Skill of ``y_pred`` *relative to* a named ``baseline`` predictor.

    ``SS = skill(y_pred) - skill(baseline)`` on the chance-corrected scale of
    :func:`_chance_corrected_skill`. Because every marginal-only baseline has
    ``skill ~ 0``:

    * ``vs_flat`` (baseline = always-flat, ``skill == 0`` exactly) reduces to
      the intrinsic skill of ``y_pred``: a perfect replica scores 1, a flat or
      majority guess scores 0 -- the contract's ordinal identities hold exactly
      and the baseline-robustness bound ``|SS| < 0.15`` holds on every series.
    * ``vs_random`` (baseline = stratified-random) reports skill in excess of a
      marginal-matching random guess, so it too sits at ~0 for uninformed
      predictors.
    """
    return _chance_corrected_skill(y_pred, y_true) - _chance_corrected_skill(
        baseline, y_true
    )


def panel(
    y_true: np.ndarray | pd.Series | list[int],
    y_pred: np.ndarray | pd.Series | list[int],
) -> dict:
    """Score a predicted signal against the true signal.

    Parameters
    ----------
    y_true, y_pred
        Equal-length 1-D label containers over ``{-1, 0, +1}``.

    Returns
    -------
    dict with keys
        ``kappa``            : Cohen's kappa (0.0 if single-class).
        ``kappa_quadratic``  : quadratically-weighted kappa (ordinal-aware).
        ``mcc``              : Matthews correlation coefficient.
        ``balanced_acc``     : balanced accuracy (mean per-class recall).
        ``macro_f1``         : macro-F1 over labels PRESENT in ``y_true``.
        ``per_class``        : ``{label: {"p", "r", "f1"}}``; a label absent
                               from ``y_true`` maps to ``None``.
        ``confusion``        : confusion matrix over ``sorted(y_true | y_pred)``.
        ``ordinal_skill``    : ``{"vs_flat", "vs_random"}`` skill-scores.
    """
    yt = _as_int_array(y_true)
    yp = _as_int_array(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"length mismatch: y_true {yt.shape} vs y_pred {yp.shape}")

    present_true = sorted(np.unique(yt).tolist())
    present_set = set(present_true)
    # Confusion matrix spans exactly the labels that occur (sorted union).
    union_labels = sorted(np.unique(np.concatenate([yt, yp])).tolist())
    # per_class is reported over the fixed signal alphabet plus any stray label
    # actually seen, so a class structurally absent from y_true is explicitly
    # None (not 0, not silently dropped). Macro aggregates use ONLY the labels
    # y_true can take, so an absent class never dilutes the mean.
    report_labels = sorted(set(_SIGNAL_LABELS) | set(union_labels))

    with _quiet_degenerate_warnings():
        macro_f1 = float(
            f1_score(yt, yp, labels=present_true, average="macro", zero_division=0)
        )
        p, r, f1, _ = precision_recall_fscore_support(
            yt, yp, labels=report_labels, zero_division=0
        )
        per_class: dict[int, dict[str, float] | None] = {}
        for i, label in enumerate(report_labels):
            if label in present_set:
                per_class[label] = {
                    "p": float(p[i]),
                    "r": float(r[i]),
                    "f1": float(f1[i]),
                }
            else:
                per_class[label] = None

        return {
            "kappa": _safe_kappa(yt, yp),
            "kappa_quadratic": _safe_kappa(yt, yp, weights="quadratic"),
            "mcc": float(matthews_corrcoef(yt, yp)),
            "balanced_acc": float(balanced_accuracy_score(yt, yp)),
            "macro_f1": macro_f1,
            "per_class": per_class,
            "confusion": confusion_matrix(yt, yp, labels=union_labels),
            "ordinal_skill": {
                "vs_flat": _ordinal_skill(yt, yp, always_flat(yt)),
                "vs_random": _ordinal_skill(yt, yp, stratified_random(yt, seed=0)),
            },
        }
