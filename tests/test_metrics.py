"""Unit tests for :mod:`stml.replication.metrics`.

The headline guarantee is *baseline-robustness*: on every one of the 11
released signal series, an uninformed predictor (always-flat, majority-class,
stratified-random, or a shuffled copy of the truth) must score at chance --
``|kappa|``, ``|mcc|`` and ``|ordinal_skill['vs_flat']|`` all ``< 0.15`` --
while a perfect copy scores ~1. Class imbalance (some series are ~80% flat)
must NOT inflate these scores.
"""

from __future__ import annotations

import numpy as np
import pytest

from stml.io import load_clean_data
from stml.replication.baselines import (
    always_flat,
    majority_class,
    stratified_random,
)
from stml.replication.metrics import panel

# Tolerance for "at chance" on the scalar discrepancy metrics.
_CHANCE_TOL = 0.15
# Tolerance for the exact ordinal-skill identities (0.0 / 1.0).
_EXACT_TOL = 1e-9

_SCALAR_KEYS = ["kappa", "kappa_quadratic", "mcc", "balanced_acc", "macro_f1"]


@pytest.fixture(scope="module")
def signals():
    """The released wide signal frame (645 x 11, values in {-1, 0, +1})."""
    _, sig = load_clean_data()
    return sig


@pytest.fixture(scope="module")
def instruments(signals) -> list[str]:
    return [c for c in signals.columns if c != "date"]


def _shuffled(y: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = y.copy()
    rng.shuffle(out)
    return out


def test_eleven_instruments(instruments) -> None:
    assert len(instruments) == 11


# --------------------------------------------------------------------------- #
# Baseline-robustness across all 11 instruments.                              #
# --------------------------------------------------------------------------- #
def test_uninformed_predictors_score_at_chance(signals, instruments) -> None:
    """always_flat / majority_class / stratified_random / shuffled ~ chance."""
    for inst in instruments:
        y = signals[inst].to_numpy()
        candidates = {
            "always_flat": always_flat(y),
            "majority_class": majority_class(y),
            "stratified_random": stratified_random(y, seed=0),
            "shuffled": _shuffled(y, seed=0),
        }
        for name, y_pred in candidates.items():
            res = panel(y, y_pred)
            assert abs(res["kappa"]) < _CHANCE_TOL, (inst, name, res["kappa"])
            assert abs(res["mcc"]) < _CHANCE_TOL, (inst, name, res["mcc"])
            assert abs(res["ordinal_skill"]["vs_flat"]) < _CHANCE_TOL, (
                inst,
                name,
                res["ordinal_skill"]["vs_flat"],
            )


def test_perfect_copy_scores_one(signals, instruments) -> None:
    """A perfect replica is the sanity upper bound: kappa ~ 1, SS ~ 1."""
    for inst in instruments:
        y = signals[inst].to_numpy()
        res = panel(y, y.copy())
        assert res["kappa"] == pytest.approx(1.0, abs=1e-9)
        assert res["ordinal_skill"]["vs_flat"] == pytest.approx(1.0, abs=1e-9)


def test_no_nan_in_scalar_metrics_across_baselines(signals, instruments) -> None:
    """No baseline on any instrument may produce a NaN scalar metric."""
    for inst in instruments:
        y = signals[inst].to_numpy()
        for y_pred in (always_flat(y), majority_class(y), stratified_random(y, 0)):
            res = panel(y, y_pred)
            for key in _SCALAR_KEYS:
                assert not np.isnan(res[key]), (inst, key)
            assert not np.isnan(res["ordinal_skill"]["vs_flat"])
            assert not np.isnan(res["ordinal_skill"]["vs_random"])


# --------------------------------------------------------------------------- #
# Degenerate cases must be defined and NaN-free.                              #
# --------------------------------------------------------------------------- #
def test_degenerate_ng1s_never_plus_one(signals) -> None:
    """ng1s never prints +1: +1 absent from y_true -> None, no NaN, no crash."""
    y = signals["ng1s"].to_numpy()
    assert 1 not in set(np.unique(y))
    res = panel(y, stratified_random(y, seed=0))
    for key in _SCALAR_KEYS:
        assert not np.isnan(res[key])
    # +1 is absent from y_true -> reported as None and dropped from macro.
    assert res["per_class"].get(1, None) is None
    assert np.isfinite(res["macro_f1"])


def test_degenerate_single_class_all_zero() -> None:
    """y_true all zeros (single class) -> kappa coerced to 0.0, no NaN."""
    y = np.zeros(50, dtype=int)
    res = panel(y, always_flat(y))
    for key in _SCALAR_KEYS:
        assert not np.isnan(res[key]), key
    assert res["kappa"] == 0.0
    assert res["kappa_quadratic"] == 0.0


def test_degenerate_missing_one_class_macro_is_finite() -> None:
    """A y_true missing one class: that class is None in per_class; macro_f1
    is a finite float over the present classes only."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])  # no -1 present
    y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1])
    res = panel(y_true, y_pred)
    assert res["per_class"][-1] is None
    # Present classes carry a real per-class dict.
    assert isinstance(res["per_class"][0], dict)
    assert isinstance(res["per_class"][1], dict)
    assert isinstance(res["macro_f1"], float)
    assert np.isfinite(res["macro_f1"])


def test_per_class_absent_label_is_none_not_zero() -> None:
    """The absent-class entry is None, explicitly NOT a zeroed metric dict."""
    y_true = np.array([0, 0, 1, 1])  # no -1
    y_pred = np.array([0, -1, 1, 1])  # predictor emits -1
    res = panel(y_true, y_pred)
    assert res["per_class"][-1] is None


# --------------------------------------------------------------------------- #
# Ordinal skill-score identities.                                             #
# --------------------------------------------------------------------------- #
def test_ordinal_skill_flat_identity(signals, instruments) -> None:
    """y_pred == always_flat(y_true) -> ordinal_skill['vs_flat'] == 0.0."""
    for inst in instruments:
        y = signals[inst].to_numpy()
        res = panel(y, always_flat(y))
        assert res["ordinal_skill"]["vs_flat"] == pytest.approx(0.0, abs=_EXACT_TOL)


def test_ordinal_skill_perfect_identity(signals, instruments) -> None:
    """y_pred == y_true -> ordinal_skill['vs_flat'] == 1.0."""
    for inst in instruments:
        y = signals[inst].to_numpy()
        res = panel(y, y.copy())
        assert res["ordinal_skill"]["vs_flat"] == pytest.approx(1.0, abs=_EXACT_TOL)


# --------------------------------------------------------------------------- #
# Structural checks on the returned panel.                                    #
# --------------------------------------------------------------------------- #
def test_confusion_is_over_sorted_union_labels() -> None:
    y_true = np.array([-1, 0, 1, 1])
    y_pred = np.array([0, 0, 1, -1])
    res = panel(y_true, y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    cm = res["confusion"]
    assert cm.shape == (len(labels), len(labels))
    # Total count is conserved.
    assert int(cm.sum()) == len(y_true)


def test_panel_keys_present() -> None:
    y = np.array([1, 0, -1, 1, 0])
    res = panel(y, y.copy())
    expected = {
        "kappa",
        "kappa_quadratic",
        "mcc",
        "balanced_acc",
        "macro_f1",
        "per_class",
        "confusion",
        "ordinal_skill",
    }
    assert expected <= set(res)
    assert set(res["ordinal_skill"]) == {"vs_flat", "vs_random"}


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        panel(np.array([0, 1, -1]), np.array([0, 1]))
