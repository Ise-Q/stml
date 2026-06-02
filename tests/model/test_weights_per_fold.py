"""Per-fold uniqueness-weight recompute (plan item 2) — the Channel-2 weight-leak fix.

Slicing a globally-computed uniqueness vector to a purged fold's train rows carries the
concurrency of events that purging removed; recomputing ``sample_uniqueness`` on the fold's own
purged-train labels is the leakage-safe behaviour. These tests prove the two differ, that the
scoring primitives invoke ``weight_fn`` with the train positions, and that the barrier search is
NOT re-run under CPCV (barriers stay frozen, searched with PurgedWalkForward).
"""

from __future__ import annotations

import inspect

import numpy as np

from stml.model import barrier_search
from stml.model.cv import CombinatorialPurgedCV
from stml.model.labels import sample_uniqueness
from stml.model.linear import LogRegModel
from stml.model.optuna_objective import cpcv_oof_auc, cross_val_auc


def _cpcv():
    return CombinatorialPurgedCV(6, 2, h=3, default_embargo=2)


def test_recomputed_uniqueness_differs_from_global_slice(
    synthetic_panel, synthetic_labels, synthetic_close
):
    """fold weights == sample_uniqueness(labels.iloc[train], close) and != global[train] slice."""
    df = synthetic_panel.reset_index(drop=True)
    labels = synthetic_labels.reset_index(drop=True)
    global_w = sample_uniqueness(labels, synthetic_close).to_numpy()

    tr, _ = next(iter(_cpcv().split(df)))
    recomputed = sample_uniqueness(
        labels.iloc[tr].reset_index(drop=True), synthetic_close
    ).to_numpy()
    sliced = global_w[tr]

    assert recomputed.shape == sliced.shape == (tr.size,)
    # purging removes overlapping neighbours -> lower concurrency -> higher (different) uniqueness
    assert not np.allclose(recomputed, sliced)
    assert (recomputed >= sliced - 1e-9).all()  # recompute never *understates* uniqueness


def test_cross_val_auc_invokes_weight_fn_with_train_positions(synthetic_panel):
    df = synthetic_panel.reset_index(drop=True)
    X = df[["f1_a", "f1_b", "f2_vol"]].astype(float)
    y = df["bin"].to_numpy()
    seen: list[np.ndarray] = []

    def weight_fn(tr: np.ndarray) -> np.ndarray:
        seen.append(np.asarray(tr))
        return np.ones(tr.size)

    cross_val_auc(LogRegModel, {"C": 1.0}, X, y, df, _cpcv(), seed=42, weight_fn=weight_fn)
    assert len(seen) >= 1
    assert all(t.ndim == 1 and t.dtype.kind in "iu" for t in seen)


def test_cpcv_oof_auc_invokes_weight_fn(synthetic_panel):
    df = synthetic_panel.reset_index(drop=True)
    X = df[["f1_a", "f1_b", "f2_vol"]].astype(float)
    y = df["bin"].to_numpy()
    calls = {"n": 0}

    def weight_fn(tr: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return np.ones(tr.size)

    mean, std, n_paths = cpcv_oof_auc(
        LogRegModel, {"C": 1.0}, X, y, df, _cpcv(), seed=42, weight_fn=weight_fn
    )
    assert calls["n"] >= 1  # one recompute per fitted block-combination
    assert 0.0 <= mean <= 1.0 and std >= 0.0
    assert n_paths == _cpcv().n_paths() == 5


def test_barriers_not_researched_under_cpcv():
    """Barrier search must use the single-path PurgedWalkForward, never CPCV -> barriers frozen."""
    src = inspect.getsource(barrier_search)
    assert "PurgedWalkForward" in src
    assert "CombinatorialPurgedCV" not in src
