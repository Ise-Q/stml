"""
test_regime_features.py
=======================
Leakage / causality tests for :mod:`stml.metamodel.regime_features` (F3, the
filtered GMM + Markov-switching per-day regime posteriors).

These tests prove the contract's TF-class causality requirements (CONTRACT_FE
Sections 0 and 3, AC-3 / AC-12) on REAL data via :func:`stml.io.load_clean_data`:

* **Filtered truncation-invariance** -- the Markov FILTERED marginal
  probabilities at time ``t`` computed on ``ret_all[:t+1]`` equal those computed
  on the full ``ret_all`` (within ``1e-6``). The model is built with a FROZEN
  params vector so the only variable is the data length; equality therefore
  isolates the one-sided/causal property of ``.filter`` (a SMOOTHED pass would
  differ because it conditions on future returns).
* **Shape** -- ``filtered_marginal_probabilities`` is ``(nobs, 2)``.
* **Fit provenance** -- ``train_index`` is a subset of FE-train (all dates
  ``<= 2021-07-01``), and the FROZEN GMM ``(ret, vol)`` standardization stats
  equal the TRAIN mean/std, NOT the full-series stats.
* **Degenerate robustness** -- the signal-degenerate instrument ``ng1s`` does
  not raise; and on a forced fit-failure the bundle's ``ok`` is ``False`` and
  every column is a structural NaN that is never forward-filled.
* **No characterize import** -- the module re-implements a fresh causal path and
  must not import any smoothed, signal-era-fit ``characterize`` estimator.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from stml.metamodel.regime_features import (
    FE_TRAIN_END,
    fit_regime,
    transform_regime,
)
from stml.na_checks import native_returns, rolling_vol

NORMAL = "si1s"
DEGENERATE = "ng1s"  # signal is never +1 (degenerate signal; dense price series)
VOL_WINDOW = 20


# --------------------------------------------------------------------------- #
# Fixtures / helpers                                                          #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def data():
    """Real clean OHLCV + signals loaded once for the module."""
    from stml.io import load_clean_data

    return load_clean_data()


def _ret_vol(ohlcv: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """Date-indexed ``(ret, vol)`` on the instrument's FULL dense series.

    Returns and rolling vol both reuse :mod:`stml.na_checks`, so a return
    spanning a holiday is the correct multi-day move and the 20-day vol at the
    start of the signal era uses real history, not a truncated window.
    """
    inst = ohlcv[ohlcv["instrument"] == instrument]
    rets = native_returns(inst, kind="log")
    ret = rets.set_index("date")["ret"].sort_index()
    vol = rolling_vol(rets, instrument, window=VOL_WINDOW)
    return pd.DataFrame({"ret": ret, "vol": vol}).dropna().sort_index()


def _train_slice(ret_vol: pd.DataFrame) -> pd.DataFrame:
    """FE-train rows (dates ``<= 2021-07-01``)."""
    return ret_vol[ret_vol.index <= FE_TRAIN_END]


# --------------------------------------------------------------------------- #
# Filtered truncation-invariance -- the core causality proof                  #
# --------------------------------------------------------------------------- #
def test_markov_filtered_truncation_invariant(data) -> None:
    """The FILTERED marginal probabilities are one-sided/causal: building the
    model on ``ret[:t+1]`` and on the full ``ret`` (with the SAME frozen params)
    gives identical filtered probabilities at every retained ``t`` (within
    ``1e-6``). A smoothed pass would NOT satisfy this."""
    ohlcv, _ = data
    rv = _ret_vol(ohlcv, NORMAL)
    train = _train_slice(rv)

    # Fit params on FE-train only, then FREEZE them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_train = MarkovRegression(
            train["ret"].to_numpy(dtype=float),
            k_regimes=2,
            trend="c",
            switching_variance=True,
        )
        params = np.asarray(m_train.fit(maxiter=100, disp=False).params, dtype=float)

    ret_all = rv["ret"].to_numpy(dtype=float)

    # Filtered probs on the FULL series.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = MarkovRegression(ret_all, k_regimes=2, trend="c", switching_variance=True)
        fp_full = np.asarray(full.filter(params).filtered_marginal_probabilities)

    # Filtered probs on the TRUNCATED series ret_all[:t+1], frozen params.
    t = len(ret_all) - 40
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trunc = MarkovRegression(
            ret_all[: t + 1], k_regimes=2, trend="c", switching_variance=True
        )
        fp_trunc = np.asarray(trunc.filter(params).filtered_marginal_probabilities)

    # Causality: the truncated run reproduces the full run on its whole overlap,
    # and exactly at the boundary t (the value that would feed feature row t).
    assert fp_trunc.shape == (t + 1, 2)
    assert np.max(np.abs(fp_trunc[t] - fp_full[t])) < 1e-6
    assert np.max(np.abs(fp_trunc - fp_full[: t + 1])) < 1e-6


def test_filtered_shape_nobs_2(data) -> None:
    """``filtered_marginal_probabilities`` is exactly ``(nobs, 2)``."""
    ohlcv, _ = data
    rv = _ret_vol(ohlcv, NORMAL)
    train = _train_slice(rv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_train = MarkovRegression(
            train["ret"].to_numpy(dtype=float),
            k_regimes=2,
            trend="c",
            switching_variance=True,
        )
        params = np.asarray(m_train.fit(maxiter=100, disp=False).params, dtype=float)
        full = MarkovRegression(
            rv["ret"].to_numpy(dtype=float), k_regimes=2, trend="c", switching_variance=True
        )
        fp = np.asarray(full.filter(params).filtered_marginal_probabilities)
    assert fp.shape == (len(rv), 2)


# --------------------------------------------------------------------------- #
# Fit provenance -- train-only fit + frozen-from-train standardization        #
# --------------------------------------------------------------------------- #
def test_train_index_subset_of_fe_train(data) -> None:
    """Every fit date is on or before the FE-train boundary (2021-07-01)."""
    ohlcv, _ = data
    rv = _ret_vol(ohlcv, NORMAL)
    bundle = fit_regime(_train_slice(rv), instrument=NORMAL)
    assert bundle.ok
    assert len(bundle.train_index) > 0
    assert bundle.train_index.max() <= FE_TRAIN_END
    # And the bundle was fit on the train slice exactly (subset of FE-train).
    assert set(bundle.train_index) <= set(rv.index[rv.index <= FE_TRAIN_END])


def test_gmm_standardization_frozen_from_train_not_full(data) -> None:
    """The FROZEN GMM ``(ret, vol)`` mean/std equal the TRAIN stats, not the
    full-series stats -- the load-bearing leakage guarantee for the scaler."""
    ohlcv, _ = data
    rv = _ret_vol(ohlcv, NORMAL)
    train = _train_slice(rv)
    bundle = fit_regime(train, instrument=NORMAL)
    assert bundle.ok

    train_mean = train[["ret", "vol"]].to_numpy(dtype=float).mean(axis=0)
    train_std_raw = train[["ret", "vol"]].to_numpy(dtype=float).std(axis=0)
    train_std = np.where(train_std_raw == 0.0, 1.0, train_std_raw)
    full_mean = rv[["ret", "vol"]].to_numpy(dtype=float).mean(axis=0)

    # Frozen stats == TRAIN stats.
    np.testing.assert_allclose(bundle.gmm_feat_mean, train_mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(bundle.gmm_feat_std, train_std, rtol=0, atol=1e-12)
    # ... and are NOT the full-series stats (train is a strict subset of full,
    # so the means must differ for this real series).
    assert len(train) < len(rv)
    assert not np.allclose(bundle.gmm_feat_mean, full_mean)


# --------------------------------------------------------------------------- #
# End-to-end transform: columns, causality, alignment                         #
# --------------------------------------------------------------------------- #
def test_transform_columns_and_alignment(data) -> None:
    """The transform emits the four canonical F3 columns, aligned to the input
    index, with high-vol probabilities in ``[0, 1]``."""
    ohlcv, _ = data
    rv = _ret_vol(ohlcv, NORMAL)
    bundle = fit_regime(_train_slice(rv), instrument=NORMAL)
    out = transform_regime(bundle, rv)

    assert list(out.columns) == [
        "f3_gmm_prob_highvol",
        "f3_markov_prob_highvol",
        "f3_markov_switch_prob",
        "f3_regime_dwell",
    ]
    assert out.index.equals(rv.index)

    g = out["f3_gmm_prob_highvol"].to_numpy()
    mk = out["f3_markov_prob_highvol"].to_numpy()
    assert np.nanmin(g) >= -1e-9 and np.nanmax(g) <= 1 + 1e-9
    assert np.nanmin(mk) >= -1e-9 and np.nanmax(mk) <= 1 + 1e-9
    # switch prob is a |difference| in [0, 1]; dwell is a positive trailing count.
    sw = out["f3_markov_switch_prob"].to_numpy()
    assert np.nanmin(sw) >= -1e-9 and np.nanmax(sw) <= 1 + 1e-9
    assert np.nanmin(out["f3_regime_dwell"].to_numpy()) >= 1.0


def test_transform_truncation_invariant_endtoend(data) -> None:
    """End-to-end causality: transforming the full series and transforming the
    truncated series ``rv[:t+1]`` (same frozen bundle) agree at ``t`` for the
    causal columns (GMM prob and filtered Markov prob)."""
    ohlcv, _ = data
    rv = _ret_vol(ohlcv, NORMAL)
    bundle = fit_regime(_train_slice(rv), instrument=NORMAL)

    full = transform_regime(bundle, rv)
    t = len(rv) - 30
    trunc = transform_regime(bundle, rv.iloc[: t + 1])

    for col in ("f3_gmm_prob_highvol", "f3_markov_prob_highvol"):
        a = full[col].to_numpy()[t]
        b = trunc[col].to_numpy()[t]
        assert abs(a - b) < 1e-6, f"{col} not truncation-invariant at t"


# --------------------------------------------------------------------------- #
# Degenerate robustness                                                       #
# --------------------------------------------------------------------------- #
def test_degenerate_instrument_does_not_raise(data) -> None:
    """ng1s (signal never +1) must fit/transform without raising. Its price
    series is dense, so the regime fit succeeds and yields finite posteriors."""
    ohlcv, _ = data
    rv = _ret_vol(ohlcv, DEGENERATE)
    bundle = fit_regime(_train_slice(rv), instrument=DEGENERATE)
    out = transform_regime(bundle, rv)  # must not raise
    assert list(out.columns)[0] == "f3_gmm_prob_highvol"
    assert len(out) == len(rv)


def test_fit_failure_sets_ok_false_and_structural_nan() -> None:
    """A forced fit-failure (too-few rows) yields ``ok=False`` and an all-NaN
    transform that is NOT forward-filled (every cell stays NaN)."""
    # Five rows is far below the 30-row floor -> deliberate fit failure.
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    tiny = pd.DataFrame({"ret": [0.0, 0.01, -0.01, 0.0, 0.02], "vol": [0.1] * 5}, index=idx)
    with pytest.warns(UserWarning):
        bundle = fit_regime(tiny, instrument="degenerate_synth")
    assert bundle.ok is False
    assert bundle.gmm is None
    assert bundle.gmm_highvol_comp == -1
    assert bundle.markov_highvol_regime == -1

    # Transform over a longer series: all structural NaN, no ffill.
    long_idx = pd.date_range("2020-01-01", periods=50, freq="B")
    rv = pd.DataFrame(
        {"ret": np.random.default_rng(0).normal(0, 0.01, 50), "vol": np.full(50, 0.1)},
        index=long_idx,
    )
    with pytest.warns(UserWarning):
        out = transform_regime(bundle, rv)
    assert out.shape == (50, 4)
    assert out.isna().all().all(), "fit-failure output must be entirely structural NaN"
    # Explicitly assert NOT forward-filled (a ffill of all-NaN stays NaN, but a
    # ffill of a partially-filled series would not -- guard the invariant).
    assert out.equals(out.ffill())  # ffill of all-NaN is a no-op == still NaN
    assert out.isna().all().all()


# --------------------------------------------------------------------------- #
# Source hygiene: no characterize import                                      #
# --------------------------------------------------------------------------- #
def test_module_does_not_import_characterize() -> None:
    """F3 must re-implement a fresh causal path; importing the smoothed,
    signal-era-fit ``characterize`` module would reintroduce non-causality."""
    import stml.metamodel.regime_features as mod

    src = Path(mod.__file__).read_text()
    # Line-level check: no import statement may reference characterize.
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert "characterize" not in stripped, f"illegal import: {line!r}"
