"""Methodology guard tests — protect against subtle leakage / aliasing bugs.

1. OOF row-index alignment: cross_val_evaluate's row_idx must point to the
   ORIGINAL pool_df row. If sklearn ever reorders, downstream consumers
   (Platt, threshold, SOPS) would silently mis-align labels and probabilities.

2. NN dataset p_hat forward-fill is causal: between two event days, the
   forward-filled p̂ on day t may only use events with date <= t.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# OOF row_idx alignment.
# ---------------------------------------------------------------------------


def test_cross_val_evaluate_row_idx_aligns_to_input():
    """For every OOF row, y_true must match the input frame's label at row_idx."""
    from stml.experimental.cv import PurgedKFold
    from stml.experimental.evaluation import cross_val_evaluate

    # Synthetic well-behaved problem: 200 rows, 5 features, mostly separable.
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 5)),
                     columns=[f"f{i}" for i in range(5)])
    y = pd.Series((X["f0"] > 0).astype(int), name="label")
    # Non-overlapping windows (1-day horizon) so PurgedKFold has no purges.
    t = pd.Series(pd.date_range("2020-01-01", periods=n, freq="B"))
    t1 = t.shift(-1).fillna(t.iloc[-1] + pd.Timedelta(days=1))

    cv = PurgedKFold(n_splits=5, t=t, t1=t1, pct_embargo=0.0)
    from stml.experimental.models import make_elasticnet_logistic

    result = cross_val_evaluate(
        make_model=lambda: make_elasticnet_logistic(seed=42),
        X=X, y=y, cv=cv,
    )
    oof = result.oos_predictions
    assert not oof.empty
    # For every OOF row, y_true must equal the input y at row_idx.
    merged = oof.merge(
        y.reset_index().rename(columns={"index": "row_idx", "label": "y_input"}),
        on="row_idx",
    )
    np.testing.assert_array_equal(merged["y_true"].values, merged["y_input"].values)


# ---------------------------------------------------------------------------
# NN p̂ forward-fill causality.
# ---------------------------------------------------------------------------


def test_nn_dataset_p_hat_forward_fill_is_causal(tmp_path, monkeypatch):
    """Inject events at dates t1 and t2 (t1 < t2) with distinct p̂; verify
    the panel between t1 and t2 carries p̂(t1) (not p̂(t2)) at every date."""
    import torch  # noqa: F401 -- ensure torch importable

    from stml.experimental.nn_dataset import build_portfolio_panel

    # Build a small fake p̂ OOF with two events on cl1s.
    p_hat = pd.DataFrame({
        "date": pd.to_datetime(["2021-03-10", "2021-04-20"]),
        "instrument": ["cl1s", "cl1s"],
        "calibrated_proba": [0.10, 0.90],  # very different so the test is sharp.
    })
    # Force the panel to span only this window.
    panel = build_portfolio_panel(
        p_hat_oof=p_hat, lookback=2,
        start="2021-03-10", end="2021-04-20",
        include_inst_id=False,
    )
    # Find the channel index for p_hat_cal.
    fn = panel.feature_names
    assert "p_hat_cal" in fn
    ch_idx = fn.index("p_hat_cal")
    # Find the cl1s instrument index.
    k_idx = panel.instruments.index("cl1s")
    # Locate trading days strictly between 2021-03-10 and 2021-04-20.
    dates = pd.DatetimeIndex(panel.dates)
    interior = (dates > pd.Timestamp("2021-03-10")) & (dates < pd.Timestamp("2021-04-20"))
    assert interior.any(), "expected interior trading days between the two events"
    # For each interior day, the LAST timestep of the lookback window is the
    # current day's feature row. p_hat_cal at that step must equal 0.10
    # (forward-filled from the first event) -- never the future 0.90.
    interior_idx = np.where(interior)[0]
    for t_idx in interior_idx:
        last_step = panel.X[t_idx, k_idx, -1, ch_idx]
        assert float(last_step) == pytest.approx(0.10, abs=1e-6), (
            f"on {dates[t_idx]} p_hat={last_step!r} (expected 0.10 from "
            f"forward-fill; got future leakage if 0.90)"
        )


# ---------------------------------------------------------------------------
# Drift filter must use TRAIN only.
# ---------------------------------------------------------------------------


def test_drift_filter_uses_train_only_no_val_leak():
    """The drift filter must compare an early-train vs late-train slice;
    val/test rows must contribute zero to the KS calculation."""
    from stml.experimental.make_features import drift_filter

    rng = np.random.default_rng(0)
    n_train = 120; n_val = 40; n_test = 40
    parts = (["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
    dates = pd.date_range("2020-01-01", periods=n_train + n_val + n_test, freq="B")
    # Train: standard normal. Val: shifted by +5. Test: shifted by -5.
    # If drift filter looks at val/test, KS will be huge. If it looks at
    # train only (an early/late chronological split with same distribution),
    # KS will be tiny.
    train_vals = rng.standard_normal(n_train)
    val_vals = rng.standard_normal(n_val) + 5.0
    test_vals = rng.standard_normal(n_test) - 5.0
    feature_matrix = pd.DataFrame({
        "instrument": ["cl1s"] * (n_train + n_val + n_test),
        "t_signal": dates,
        "t_start": dates,
        "t_end": dates,
        "side": 1,
        "ret": 0.0,
        "label": rng.integers(0, 2, n_train + n_val + n_test),
        "uniqueness_weight": 1.0,
        "sigma_at_t": 0.01,
        "barrier_hit": "pt",
        "pt": 0.25, "sl": 0.25, "h": 1,
        "partition": parts,
        "f_test_feature": np.concatenate([train_vals, val_vals, test_vals]),
    })
    _kept, audit = drift_filter(feature_matrix)
    assert len(audit) == 1
    ks = float(audit["ks"].iloc[0])
    # If val/test were included, the +5/-5 shifts would push KS near 1.
    # Train-only -> early vs late of the same distribution -> KS should be
    # small (well under 0.3).
    assert ks < 0.3, (
        f"drift filter KS={ks:.3f} is too high; val/test rows are leaking "
        f"into the comparison (expected train-only -> small KS)."
    )
