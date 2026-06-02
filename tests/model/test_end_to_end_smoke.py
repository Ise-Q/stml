"""End-to-end P1 composition smoke test on a small REAL matrix slice.

Proves the graded pipeline wires together: triple-barrier labels -> CPCV out-of-fold probabilities
-> probability calibration -> EV decision threshold p* -> adding-zeros evaluation. Kept small
(2 instruments, shallow CPCV) so it runs in a few seconds; correctness of each piece is covered by
its own unit test, this only checks they compose.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from stml.model.calibration import calibrate_oof
from stml.model.cv import CombinatorialPurgedCV
from stml.model.dataset import embargo_map, events_frame, make_xy, select_features
from stml.model.evaluate import adding_zeros_eval, bootstrap_returns, decision_threshold
from stml.model.labels import triple_barrier_labels
from stml.model.linear import LogRegModel


def test_cpcv_to_adding_zeros_pipeline(real_slice):
    matrix, close_wide = real_slice
    h = 5

    # 1) triple-barrier meta-labels (sigma de-annualised inside events_frame)
    events = events_frame(matrix)
    labels = triple_barrier_labels(close_wide, events, pt=1.0, sl=1.0, h=h)
    assert not labels.empty

    dev = matrix.merge(
        labels[["date", "instrument", "bin"]], on=["date", "instrument"], how="inner"
    ).reset_index(drop=True)
    assert "bar_pos" in dev.columns and (dev["bar_pos"] >= 0).all()

    # 2) design matrix + shallow CPCV out-of-fold probabilities
    feature_cols = select_features(dev)
    X, y = make_xy(dev, feature_cols, instrument_dummies=True)
    assert np.unique(y).size == 2

    cpcv = CombinatorialPurgedCV(
        4, 2, h=h, embargo_by_instrument=embargo_map(), default_embargo=10
    )
    oof, counts, n_scored = cpcv.oof_predict(LogRegModel, {"C": 1.0}, X, y, dev, seed=42)
    assert n_scored > 0
    scored = counts > 0
    assert scored.any()
    auc = roc_auc_score(y[scored], oof[scored])
    assert np.isfinite(auc) and 0.0 <= auc <= 1.0

    # 3) calibrate OOF probabilities (Platt, fit on OOF pairs only)
    calibrator = calibrate_oof(oof[scored], y[scored], method="platt", seed=42)
    p_cal = calibrator.transform(oof[scored])
    assert ((p_cal >= 0) & (p_cal <= 1)).all()

    # 4) EV threshold p* from bootstrapped TP/FP returns
    r_g, r_l = bootstrap_returns(labels, n_boot=200, seed=42)
    p_star = decision_threshold(r_g, r_l)
    assert 0.0 <= p_star <= 1.0

    # 5) adding-zeros evaluation: primary-alone (recall=1) vs primary+meta filter
    result = adding_zeros_eval(y[scored], p_cal, p_star=p_star)
    assert result["primary_alone"]["recall"] == 1.0
    assert {"tp", "fp", "fn", "tn"} <= set(result["primary_meta"])
    assert result["primary_meta"]["tp"] + result["primary_meta"]["fp"] == \
        result["primary_meta"]["n_taken"]
