"""Tests for regime features (Stage 2, RED-first).

Two blocks:
1. The headline net-new **online EWMA 2-state Gaussian HMM** (commitment #8, nlr-cw §4;
   Nystrup-Madsen-Lindström 2017). It is causal/fit-free: parameters at t are an
   exponentially-weighted recursion over observations <= t, so the feature is
   right-edge truncation-invariant (no batch fit, no per-fold seam artifacts) AND its
   emission means/variances drift over time (the time-varying contribution).
2. A light reuse check of stml's STATIC regime blocks (F3 GMM/Markov, F17 3-state HMM)
   fit on a contiguous prefix and causally transformed (supplementary features).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alken_metamodel.regime import (
    HMM_WARMUP,
    assemble_regime_features,
    ewma_hmm_features,
    static_regime_features,
)

EWMA_COLS = [
    "ewma_hmm_prob_highvol",
    "ewma_hmm_state",
    "ewma_hmm_var_hi",
    "ewma_hmm_var_lo",
    "ewma_hmm_switch_prob",
]


def _two_regime_close(n_calm: int = 200, n_vol: int = 200, seed: int = 0) -> pd.Series:
    """Calm (sigma=0.005) then volatile (sigma=0.03) log-return regimes -> price series."""
    rng = np.random.default_rng(seed)
    r = np.concatenate(
        [rng.normal(0.0, 0.005, n_calm), rng.normal(0.0, 0.030, n_vol)]
    )
    close = 100.0 * np.exp(np.cumsum(r))
    dates = pd.bdate_range("2018-01-01", periods=len(r))
    return pd.Series(close, index=dates, name="close")


def _ohlcv_from_close(close: pd.Series, instrument: str = "tst1s") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": close.index,
            "instrument": instrument,
            "open": close.to_numpy(),
            "high": close.to_numpy() * 1.001,
            "low": close.to_numpy() * 0.999,
            "close": close.to_numpy(),
            "volume": 10_000.0,
            "open_interest": 50_000.0,
        }
    )


# --- EWMA HMM: validity + causality ----------------------------------------

def test_ewma_hmm_columns_and_valid_probabilities():
    out = ewma_hmm_features(_two_regime_close(seed=1))
    assert list(out.columns) == EWMA_COLS
    p = out["ewma_hmm_prob_highvol"].dropna()
    assert ((p >= 0.0) & (p <= 1.0)).all()
    # high-vol state is the larger-variance state by construction
    fin = out.dropna(subset=["ewma_hmm_var_hi", "ewma_hmm_var_lo"])
    assert (fin["ewma_hmm_var_hi"] >= fin["ewma_hmm_var_lo"]).all()


def test_ewma_hmm_warmup_is_nan():
    out = ewma_hmm_features(_two_regime_close(seed=1))
    assert out["ewma_hmm_prob_highvol"].iloc[:HMM_WARMUP].isna().all()
    assert out["ewma_hmm_prob_highvol"].iloc[-1] == out["ewma_hmm_prob_highvol"].iloc[-1]  # not NaN


def test_ewma_hmm_right_edge_truncation_invariance():
    """prob/var at t are identical on close[:t+1] and close[:T] (causal recursion)."""
    close = _two_regime_close(seed=3)
    full = ewma_hmm_features(close)
    t_iloc = 350  # deep in the volatile segment, well past warm-up
    t = close.index[t_iloc]
    trunc = ewma_hmm_features(close.iloc[: t_iloc + 1])
    for col in ["ewma_hmm_prob_highvol", "ewma_hmm_var_hi", "ewma_hmm_var_lo"]:
        np.testing.assert_allclose(
            float(full.loc[t, col]), float(trunc.loc[t, col]), rtol=1e-9, atol=1e-12
        )


def test_ewma_hmm_is_deterministic():
    close = _two_regime_close(seed=4)
    a = ewma_hmm_features(close)
    b = ewma_hmm_features(close)
    pd.testing.assert_frame_equal(a, b)


# --- EWMA HMM: it actually detects the regime + adapts ----------------------

def test_ewma_hmm_detects_volatility_regime():
    out = ewma_hmm_features(_two_regime_close(n_calm=200, n_vol=200, seed=5))
    prob = out["ewma_hmm_prob_highvol"]
    calm = prob.iloc[70:195].mean()       # post-warmup calm
    volatile = prob.iloc[260:395].mean()  # adapted volatile
    assert calm < 0.45
    assert volatile > 0.55
    assert volatile - calm > 0.25


def test_ewma_hmm_variance_is_time_varying():
    """The EWMA emission variance tracks the regime — the §4 time-varying contribution."""
    out = ewma_hmm_features(_two_regime_close(n_calm=200, n_vol=200, seed=6))
    var_hi = out["ewma_hmm_var_hi"]
    calm = var_hi.iloc[70:195].mean()
    volatile = var_hi.iloc[260:395].mean()
    assert volatile > 3.0 * calm  # variance genuinely adapts upward in the volatile regime


# --- static stml regime block (supplementary reuse) ------------------------

def test_static_regime_features_columns_and_bounds():
    close = _two_regime_close(n_calm=180, n_vol=160, seed=7)
    ohlcv = _ohlcv_from_close(close)
    fit_end = close.index[230]  # contiguous prefix for the static fit
    out = static_regime_features(ohlcv, fit_end=fit_end)
    for col in ["f3_gmm_prob_highvol", "f3_markov_prob_highvol", "f17_hmm_state_hi"]:
        assert col in out.columns
    for col in ["f3_gmm_prob_highvol", "f17_hmm_state_hi"]:
        fin = out[col].dropna()
        if len(fin):
            assert ((fin >= -1e-9) & (fin <= 1.0 + 1e-9)).all()


def test_assemble_regime_features_merges_both_blocks():
    close = _two_regime_close(n_calm=180, n_vol=160, seed=8)
    ohlcv = _ohlcv_from_close(close)
    fit_end = close.index[230]
    out = assemble_regime_features(ohlcv, fit_end=fit_end)
    assert "ewma_hmm_prob_highvol" in out.columns      # net-new block
    assert "f3_gmm_prob_highvol" in out.columns         # stml static block
    assert "f17_hmm_state_hi" in out.columns            # F17 3-state HMM IS already wired
    assert isinstance(out.index, pd.DatetimeIndex)


def test_f17_fit_uses_only_pre_fit_end_data():
    """S1.8-b conformance: F17's HMM is fit on the ``<= fit_end`` prefix only, then causally
    transformed — so appending post-fit_end data leaves the f17 score on a common date
    unchanged (the discipline that keeps the OOS deliverable, fit_end < predict, leak-free)."""
    close = _two_regime_close(n_calm=220, n_vol=220, seed=12)
    ohlcv = _ohlcv_from_close(close)
    fit_end = close.index[230]
    t = close.index[300]  # a date present in both, strictly after fit_end
    short = static_regime_features(ohlcv[ohlcv["date"] <= close.index[330]], fit_end=fit_end)
    full = static_regime_features(ohlcv, fit_end=fit_end)
    for col in ["f17_hmm_state_hi", "f17_hmm_state_lo", "f17_hmm_state_argmax"]:
        np.testing.assert_allclose(
            float(full.loc[t, col]), float(short.loc[t, col]), rtol=1e-9, atol=1e-12
        )
