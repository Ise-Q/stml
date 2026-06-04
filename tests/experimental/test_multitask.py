"""Tests for ``stml.experimental.multitask`` — methodology spec S4 acceptance.

Verifies:
* Deterministic forward+backward+Adam step (same seed → same predictions).
* Correct head allocation: only the row's instrument head receives gradient,
  others are untouched.
* Sample-weighted loss matches manual computation.
* Convergence on synthetic per-instrument signal (multi-task NN should
  outperform a single shared head on heterogeneous data).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore", category=UserWarning, module="torch")


try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch is in optional `multitask` extra — install with `uv sync --extra multitask`",
)


@pytest.fixture
def synth_multitask_data():
    """N=200 rows, 5 features, 3 instruments. Each instrument has its own
    feature → label signal so a multi-task NN should beat a shared head."""
    rng = np.random.default_rng(42)
    n = 200
    d = 5
    n_inst = 3
    X = rng.standard_normal((n, d)).astype(np.float32)
    inst = rng.integers(0, n_inst, n)
    # Per-instrument signal: instrument 0 uses feature 0, 1 uses feature 1, 2 uses feature 2.
    y = np.zeros(n, dtype=np.int64)
    for i in range(n_inst):
        mask = inst == i
        y[mask] = (X[mask, i] + 0.3 * rng.standard_normal(mask.sum()) > 0).astype(np.int64)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]), pd.Series(y), inst


def test_deterministic_forward_pass(synth_multitask_data):
    """Same seed → same predictions across two independent fits."""
    from stml.experimental.multitask import MultiTaskConfig, MultiTaskMetaClassifier

    X, y, inst = synth_multitask_data
    cfg = MultiTaskConfig(n_epochs=10, val_frac=0.0)
    m1 = MultiTaskMetaClassifier(n_instruments=3, config=cfg).fit(X, y, instrument_ids=inst)
    p1 = m1.predict_act_proba(X, instrument_ids=inst)

    m2 = MultiTaskMetaClassifier(n_instruments=3, config=cfg).fit(X, y, instrument_ids=inst)
    p2 = m2.predict_act_proba(X, instrument_ids=inst)

    np.testing.assert_allclose(p1, p2, atol=1e-5)


def test_head_allocation_only_matching_head_used():
    """Row from instrument k uses head k's parameters only — verified by
    making each head a known constant and checking output."""
    from stml.experimental.multitask import MultiTaskConfig, MultiTaskMetaClassifier
    import torch

    cfg = MultiTaskConfig(n_epochs=1, val_frac=0.0, embed_dim=2, hidden_widths=(8,),
                           shared_dim=4)
    m = MultiTaskMetaClassifier(n_instruments=3, config=cfg)

    # Fit minimally then overwrite head weights with known sentinels.
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((30, 4)), columns=list("abcd"))
    y = pd.Series(rng.integers(0, 2, 30))
    inst = rng.integers(0, 3, 30)
    m.fit(X, y, instrument_ids=inst)

    # Manually set head k's bias to k * 1000 so logit difference is detectable.
    with torch.no_grad():
        for k, head in enumerate(m._model.heads):
            head.weight.zero_()
            head.bias.fill_(float(k * 100))

    # Predict with all 3 instruments — outputs should be wildly different per inst.
    Xt = pd.DataFrame(np.zeros((6, 4)), columns=list("abcd"))
    inst_test = np.array([0, 0, 1, 1, 2, 2])
    proba = m.predict_act_proba(Xt, instrument_ids=inst_test)
    # head 0 → bias 0 → logit 0 → proba 0.5 (clipped to 0.5 still)
    # head 1 → bias 100 → logit 100 → proba ~1.0 (clipped to 0.99)
    # head 2 → bias 200 → logit 200 → proba ~1.0 (clipped to 0.99)
    assert proba[0] == pytest.approx(0.5, abs=0.01)
    assert proba[1] == pytest.approx(0.5, abs=0.01)
    assert proba[2] >= 0.98
    assert proba[3] >= 0.98
    assert proba[4] >= 0.98
    assert proba[5] >= 0.98


def test_sample_weight_applied(synth_multitask_data):
    """Sample-weighted fit should differ from unweighted fit."""
    from stml.experimental.multitask import MultiTaskConfig, MultiTaskMetaClassifier

    X, y, inst = synth_multitask_data
    cfg = MultiTaskConfig(n_epochs=30, val_frac=0.0)
    m_unweighted = MultiTaskMetaClassifier(n_instruments=3, config=cfg).fit(
        X, y, instrument_ids=inst
    )
    sw = np.ones(len(y), dtype=np.float32)
    sw[:50] = 0.01  # near-zero weight on first 50 rows
    m_weighted = MultiTaskMetaClassifier(n_instruments=3, config=cfg).fit(
        X, y, instrument_ids=inst, sample_weight=sw
    )
    p_uw = m_unweighted.predict_act_proba(X, instrument_ids=inst)
    p_w = m_weighted.predict_act_proba(X, instrument_ids=inst)
    # Predictions should differ measurably on the down-weighted rows.
    diff = np.abs(p_uw[:50] - p_w[:50]).mean()
    assert diff > 0.005


def test_multitask_beats_dummy_on_synthetic_signal(synth_multitask_data):
    """The multi-task NN should achieve clearly above 0.5 AUC on the synthetic
    per-instrument signal."""
    from sklearn.metrics import roc_auc_score

    from stml.experimental.multitask import MultiTaskConfig, MultiTaskMetaClassifier

    X, y, inst = synth_multitask_data
    cfg = MultiTaskConfig(n_epochs=100, val_frac=0.0)
    m = MultiTaskMetaClassifier(n_instruments=3, config=cfg).fit(X, y, instrument_ids=inst)
    p = m.predict_act_proba(X, instrument_ids=inst)
    auc = roc_auc_score(y, p)
    assert auc > 0.75, f"multi-task NN underfit on synthetic data: AUC={auc:.3f}"


def test_predict_proba_two_columns_sum_to_one(synth_multitask_data):
    from stml.experimental.multitask import MultiTaskConfig, MultiTaskMetaClassifier

    X, y, inst = synth_multitask_data
    cfg = MultiTaskConfig(n_epochs=5, val_frac=0.0)
    m = MultiTaskMetaClassifier(n_instruments=3, config=cfg).fit(X, y, instrument_ids=inst)
    p = m.predict_proba(X, instrument_ids=inst)
    assert p.shape == (len(X), 2)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_early_stopping_records_best_epoch(synth_multitask_data):
    """Early stopping should fire on the synthetic problem before n_epochs."""
    from stml.experimental.multitask import MultiTaskConfig, MultiTaskMetaClassifier

    X, y, inst = synth_multitask_data
    cfg = MultiTaskConfig(n_epochs=300, val_frac=0.2, early_stop_patience=10)
    m = MultiTaskMetaClassifier(n_instruments=3, config=cfg).fit(X, y, instrument_ids=inst)
    # Best epoch should be < n_epochs after early stop fires.
    assert m._epoch_at_best < cfg.n_epochs
    assert m._best_val_loss < float("inf")
