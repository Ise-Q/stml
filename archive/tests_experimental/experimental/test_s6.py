"""S6 tests — calibration / sizing / cost_model / backtest / emit.

Plan §8 S6 RED-first invariants.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Calibration — Platt is monotone → AUC invariant.
# ---------------------------------------------------------------------------


def test_platt_preserves_auc():
    """Platt calibration is monotone → AUC unchanged."""
    from stml.experimental.calibration import PlattCalibrator

    rng = np.random.default_rng(0)
    n = 500
    y = rng.integers(0, 2, n)
    raw = np.clip(rng.normal(0.5, 0.2, n) + 0.3 * (y - 0.5), 0, 1)
    pc = PlattCalibrator().fit(raw, y)
    cal = pc.transform(raw)
    auc_raw = roc_auc_score(y, raw)
    auc_cal = roc_auc_score(y, cal)
    np.testing.assert_allclose(auc_raw, auc_cal, atol=1e-9)


def test_platt_clips_extreme_probabilities():
    """Calibrated probabilities are in [0, 1]."""
    from stml.experimental.calibration import PlattCalibrator
    rng = np.random.default_rng(1)
    n = 200
    y = rng.integers(0, 2, n)
    raw = rng.uniform(0, 1, n)
    pc = PlattCalibrator().fit(raw, y)
    cal = pc.transform(raw)
    assert (cal >= 0).all() and (cal <= 1).all()


def test_isotonic_monotone():
    """Isotonic transform of sorted input is non-decreasing."""
    from stml.experimental.calibration import IsotonicCalibrator
    rng = np.random.default_rng(2)
    n = 200
    raw = np.sort(rng.uniform(0, 1, n))
    y = (raw + 0.2 * rng.standard_normal(n) > 0.5).astype(int)
    ic = IsotonicCalibrator().fit(raw, y)
    out = ic.transform(np.sort(rng.uniform(0, 1, 50)))
    assert (np.diff(out) >= -1e-9).all()


def test_ece_formula_on_perfect_calibration():
    """ECE of perfectly-calibrated predictions = 0."""
    from stml.experimental.calibration import expected_calibration_error
    n = 1000
    rng = np.random.default_rng(3)
    proba = rng.uniform(0, 1, n)
    y_true = (rng.uniform(0, 1, n) < proba).astype(int)  # perfectly calibrated
    ece = expected_calibration_error(y_true, proba, n_bins=10)
    assert ece < 0.05


# ---------------------------------------------------------------------------
# Sizing — Madmoun Optional Session 3, slides 33-34.
# ---------------------------------------------------------------------------


# 3 fixed methods


def test_model_confidence_below_half_is_zero():
    """ModelConfidence(p) = p · 1{p > 0.5}."""
    from stml.experimental.sizing import model_confidence
    assert model_confidence(0.30) == 0.0
    assert model_confidence(0.50) == 0.0
    assert model_confidence(0.70) == pytest.approx(0.70, abs=1e-12)


def test_all_or_nothing_binary():
    """AllOrNothing(p) = 1{p > 0.5}."""
    from stml.experimental.sizing import all_or_nothing
    assert all_or_nothing(0.49) == 0.0
    assert all_or_nothing(0.50) == 0.0
    assert all_or_nothing(0.51) == 1.0
    assert all_or_nothing(0.99) == 1.0


def test_ncdf_below_half_is_zero():
    """NCDF returns 0 below 0.5 and Φ(z) above."""
    from stml.experimental.sizing import ncdf
    assert ncdf(0.40) == 0.0
    assert ncdf(0.50) == 0.0
    # At p slightly above 0.5: z is small and positive → output > 0.5.
    val = float(np.atleast_1d(ncdf(0.55)).ravel()[0])
    assert val > 0.5
    # At p = 1.0: z = ∞, Φ(z) → 1.
    val = float(np.atleast_1d(ncdf(0.999)).ravel()[0])
    assert val > 0.95


def test_ncdf_monotone_above_half():
    """NCDF is monotone non-decreasing above 0.5."""
    from stml.experimental.sizing import ncdf
    p = np.linspace(0.51, 0.99, 25)
    vals = np.asarray(ncdf(p)).ravel()
    assert (np.diff(vals) >= -1e-9).all()


# 3 estimated methods


def test_linear_scaling_stretches_training_range():
    """LinearScaling stretches [min p_tr, max p_tr] to [0, 1]."""
    from stml.experimental.sizing import fit_linear_scaling
    p_tr = np.array([0.55, 0.65, 0.75])
    fit = fit_linear_scaling(p_tr)
    # Min maps to 0, max maps to 1.
    assert float(fit.transform(np.array([0.55]))[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(fit.transform(np.array([0.75]))[0]) == pytest.approx(1.0, abs=1e-9)
    # Below 0.5 always zero.
    assert float(fit.transform(np.array([0.40]))[0]) == 0.0


def test_ecdf_percentile_above_half():
    """ECDF gives 0/N for the min and N/N for the max of the training set."""
    from stml.experimental.sizing import fit_ecdf
    p_tr = np.array([0.51, 0.60, 0.70, 0.80, 0.90])
    fit = fit_ecdf(p_tr)
    # Query > max → CDF = 1.
    assert float(fit.transform(np.array([0.95]))[0]) == pytest.approx(1.0, abs=1e-9)
    # Below 0.5 always zero (the lecturer's "return 0 when p ≤ 0.5").
    assert float(fit.transform(np.array([0.40]))[0]) == 0.0


def test_sops_fits_a_sigmoid_that_separates_profitable_trades():
    """SOPS should give larger sizes to higher-probability trades when those
    trades are profitable on training data."""
    from stml.experimental.sizing import fit_sops
    rng = np.random.default_rng(0)
    n = 400
    p_tr = rng.uniform(0.45, 0.95, n)
    # Mean return increases linearly with p — the kind of structure SOPS
    # is designed to exploit.
    r_tr = (p_tr - 0.5) * 0.05 + 0.005 * rng.standard_normal(n)
    fit = fit_sops(p_tr, r_tr, refine=False)
    # Larger sizes for larger probabilities.
    sizes = np.asarray(fit.transform(np.array([0.55, 0.80, 0.95]))).ravel()
    assert (np.diff(sizes) > 0).all()
    # Below 0.5 → zero.
    assert float(np.asarray(fit.transform(np.array([0.40]))).ravel()[0]) == 0.0


def test_sizing_policy_dispatch_for_all_six_methods():
    """fit_sizing_policy + transform smoke-checks all six methods."""
    from stml.experimental.sizing import fit_sizing_policy
    rng = np.random.default_rng(1)
    p_tr = rng.uniform(0.45, 0.90, 100)
    r_tr = rng.standard_normal(100) * 0.01
    for m in ("model_confidence", "all_or_nothing", "ncdf",
              "linear_scaling", "ecdf", "sops"):
        pol = fit_sizing_policy(m, p_tr=p_tr, r_tr=r_tr)
        b = pol.transform(np.array([0.30, 0.55, 0.85]))
        b = np.asarray(b).ravel()
        # Below 0.5 always zero across all six methods.
        assert b[0] == 0.0
        # Range is [0, 1].
        assert (b >= 0.0).all() and (b <= 1.0 + 1e-9).all()


def test_vol_target_leverage_lecturer_formula():
    """Daily σ̂ annualised by √252; leverage = target / ann_σ, clipped."""
    from stml.experimental.sizing import vol_target_leverage
    # Daily σ̂ = 0.0126 → annualised ≈ 0.20; target 0.10 → leverage ≈ 0.5.
    val = vol_target_leverage(0.0126, target_vol=0.10, max_leverage=10.0)
    assert val == pytest.approx(0.10 / (0.0126 * np.sqrt(252.0)), rel=1e-6)
    # Pathological tiny σ̂ → clipped to max_leverage.
    assert vol_target_leverage(1e-6, target_vol=0.10, max_leverage=10.0) == 10.0
    # NaN / zero / negative → 0.
    assert vol_target_leverage(0.0) == 0.0
    assert vol_target_leverage(float("nan")) == 0.0


def test_position_weight_sign_follows_side_lecturer():
    """side = +1 → positive weight; side = -1 → negative; equal magnitudes."""
    from stml.experimental.sizing import fit_sizing_policy, position_weight
    pol = fit_sizing_policy("model_confidence")
    long_w = position_weight(+1, 0.70, daily_sigma=0.012, policy=pol)
    short_w = position_weight(-1, 0.70, daily_sigma=0.012, policy=pol)
    assert long_w > 0
    assert short_w < 0
    assert long_w == pytest.approx(-short_w, abs=1e-9)


# ---------------------------------------------------------------------------
# Threshold p* = L / (G + L).
# ---------------------------------------------------------------------------


def test_threshold_p_star_basic():
    """Hand-crafted G/L: rG=+0.02, rL=-0.01 → p* = 0.01 / 0.03 = 1/3."""
    from stml.experimental.threshold import estimate_threshold
    # 50 TP with ret +0.02; 50 FP with ret -0.01; all with proba > 0.5.
    n = 100
    proba = np.linspace(0.51, 0.99, n)
    label = np.array([1] * 50 + [0] * 50)
    ret = np.where(label == 1, 0.02, -0.01)
    est = estimate_threshold(proba, label, ret, n_bootstrap=200, seed=0)
    assert est.p_star == pytest.approx(1.0 / 3.0, abs=1e-9)
    assert est.n_tp == 50 and est.n_fp == 50
    # CI brackets the point estimate.
    assert est.ci_low <= est.p_star <= est.ci_high


def test_threshold_clamps_to_unit_interval():
    """If rG ≤ rL the formula degenerates → defensive p* = 0.5."""
    from stml.experimental.threshold import estimate_threshold
    n = 60
    proba = np.linspace(0.55, 0.95, n)
    label = np.array([1] * 30 + [0] * 30)
    ret = np.where(label == 1, -0.01, 0.02)  # inverted: TP loses, FP wins
    est = estimate_threshold(proba, label, ret, n_bootstrap=50, seed=0)
    assert est.p_star == 0.5  # default clamp


def test_primary_vs_meta_evaluation():
    """Primary-alone vs primary+meta filter -- expected counts."""
    from stml.experimental.evaluation import primary_vs_meta_evaluation
    proba = np.array([0.4, 0.6, 0.7, 0.3, 0.8, 0.55, 0.45, 0.9])
    label = np.array([0, 1, 1, 0, 1, 0, 0, 1])  # base rate = 4/8 = 0.5
    out = primary_vs_meta_evaluation(proba, label, threshold=0.55)
    assert out["primary_tp"] == 4
    assert out["primary_fp"] == 4
    # Meta-filter takes proba >= 0.55: rows 1,2,4,5,7 → labels 1,1,1,0,1.
    assert out["meta_tp"] == 4
    assert out["meta_fp"] == 1
    assert out["meta_fn"] == 0
    assert out["false_positives_avoided"] == 3
    assert out["delta_precision"] > 0


# ---------------------------------------------------------------------------
# Cost model — Grinold-Kahn linearity.
# ---------------------------------------------------------------------------


def test_transaction_costs_linear_in_delta():
    """Cost = (half_spread + impact) × |Δw| under impact_exponent=1."""
    from stml.experimental.cost_model import transaction_costs
    idx = pd.date_range("2020-01-01", periods=4)
    weights = pd.DataFrame(
        {"A": [0.0, 0.5, 0.5, 0.0], "B": [0.0, 0.0, 0.5, 0.5]},
        index=idx,
    )
    costs = transaction_costs(weights, half_spread_bps=2.0, impact_bps=10.0)
    # Day 0: turnover = |0.0 - flat| = 0; cost = 0.
    # Day 1: |Δw_A| = 0.5; cost = (2+10)/10000 × 0.5 = 0.0006
    # Day 2: |Δw_B| = 0.5; cost = 0.0006
    # Day 3: |Δw_A| = 0.5; cost = 0.0006
    assert costs.iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert costs.iloc[1] == pytest.approx(0.0006, abs=1e-9)


# ---------------------------------------------------------------------------
# Backtest — Sortino-Price full-T known-value.
# ---------------------------------------------------------------------------


def test_sortino_full_t_known_value():
    """Hand-computed Sortino-Price on a tiny series."""
    from stml.experimental.backtest import sortino_full_t
    # Returns: 0.05, -0.03, 0.02, -0.04
    r = pd.Series([0.05, -0.03, 0.02, -0.04])
    # Excess (target 0) = same as r.
    # Downside = [0, -0.03, 0, -0.04].
    # Semivariance = (0 + 0.0009 + 0 + 0.0016) / 4 = 0.000625
    # Per-period Sortino = mean(r) / sqrt(semivar)
    # mean = (0.05 - 0.03 + 0.02 - 0.04) / 4 = 0
    # So per-period Sortino = 0; annualised = 0.
    val = sortino_full_t(r, ann=252.0)
    np.testing.assert_allclose(val, 0.0, atol=1e-9)


def test_sortino_full_t_differs_from_std_negatives_only_misimpl():
    """The Sortino-Price full-T form differs from the std(neg-returns) mis-impl.

    The common mis-implementation uses ``std(r[r<0], ddof=1)`` which subtracts
    the mean OF THE NEGATIVES (itself negative), producing a smaller spread
    estimate than the semivariance around 0 → inflates Sortino. Our full-T
    form correctly uses semivariance around 0 with denominator T.
    """
    from stml.experimental.backtest import sortino_full_t
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 252))
    val_full_t = sortino_full_t(r, ann=252.0)
    # Mis-impl: std of negative returns around their own mean.
    neg = r[r < 0]
    std_neg = neg.std(ddof=1)
    mis = (r.mean() / std_neg) * np.sqrt(252) if std_neg > 0 else float("nan")
    # Values must be different — the test is that we are NOT computing
    # the mis-impl; magnitude direction depends on distribution.
    assert not np.isclose(val_full_t, mis, atol=1e-6)


def test_performance_metrics_basic_shape():
    from stml.experimental.backtest import performance_metrics
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 252))
    m = performance_metrics(r)
    for k in ("n", "total_return", "ann_return", "ann_vol", "sharpe", "sortino", "max_dd"):
        assert k in m
    assert m["n"] == 252


# ---------------------------------------------------------------------------
# Emit — byte-identical re-emit.
# ---------------------------------------------------------------------------


def test_emit_predictions_byte_identical(tmp_path):
    """Two emits of the same predictions frame → byte-identical CSV."""
    from stml.experimental.emit import emit_predictions
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=10),
        "instrument": ["cl1s"] * 10,
        "prediction": rng.uniform(0, 1, 10),
    })
    p1 = tmp_path / "p1.csv"
    p2 = tmp_path / "p2.csv"
    emit_predictions(df, p1)
    emit_predictions(df, p2)
    assert p1.read_bytes() == p2.read_bytes()


def test_emit_predictions_sorted_and_formatted(tmp_path):
    """Output is sorted by (date, instrument); ISO dates; %.10f floats."""
    from stml.experimental.emit import emit_predictions
    df = pd.DataFrame({
        "date": pd.to_datetime(["2022-01-03", "2022-01-02"]),
        "instrument": ["es1s", "cl1s"],
        "prediction": [0.7, 0.5],
    })
    p = tmp_path / "out.csv"
    emit_predictions(df, p)
    text = p.read_text()
    assert text.startswith("date,instrument,prediction\n")
    # First data row should be 2022-01-02,cl1s (earlier date).
    assert "2022-01-02,cl1s,0.5000000000" in text
