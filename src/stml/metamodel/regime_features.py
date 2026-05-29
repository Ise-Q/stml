"""
regime_features.py
==================
F3 -- filtered, strictly-causal volatility-regime posteriors (per-instrument,
TF-class fitted features) for the triple-barrier metamodel.

Two regime models are fit on the **FE-train** partition only and then
transformed *causally* over each instrument's full per-instrument series:

* a **Gaussian mixture** (``sklearn`` ``GaussianMixture``, 2 components,
  ``covariance_type="full"``) on standardized ``(ret, vol)`` features. The
  standardization statistics are FROZEN from FE-train and applied forward, so a
  ``predict_proba`` at date ``t`` depends only on ``(ret, vol)_t`` -- both of
  which are themselves trailing quantities -- never on any future row.
* a **Markov-switching** model (``statsmodels`` ``MarkovRegression``,
  ``k_regimes=2``, ``trend="c"``, ``switching_variance=True``) fit on the
  FE-train return series. The transform uses the model-level FILTERED
  (one-sided) marginal probabilities, ``MarkovRegression(full_ret).filter(
  train_params).filtered_marginal_probabilities``, which are causal by
  construction: the filtered probability at ``t`` is a function of returns up to
  and including ``t`` only. (The *smoothed* Markov probabilities look ahead and
  are therefore NOT used here.) In both models the high-vol regime/component --
  the larger fitted
  variance (Markov) or the larger mean raw vol (GMM) -- is recorded AT FIT TIME.

Leakage contract (see ``.omc/scratch/CONTRACT_FE.md`` Sections 0 and 3)
-----------------------------------------------------------------------
* Fit on FE-train only (``train_index`` recorded; all dates ``<= 2021-07-01``).
* All standardization stats (the GMM ``(ret, vol)`` mean/std) are FROZEN from
  FE-train -- never recomputed on the full series.
* The transform is causal: filtered (never smoothed) Markov probabilities, and
  GMM ``predict_proba`` on standardized rows ``<= t``.
* Pooled transforms run per-instrument-series, never on a concatenated panel.
* On any fit failure the bundle is marked ``ok=False`` and every output column
  is a STRUCTURAL NaN (logged via :mod:`warnings`); structural NaNs are NEVER
  forward-filled or ``fillna(0)``-ed.

This module implements a fresh causal fit/transform path: it uses *filtered*
(never smoothed) Markov probabilities, fits on FE-train only, and freezes all
standardization stats from FE-train -- so it is leakage-safe for features.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

__all__ = [
    "RegimeBundle",
    "fit_regime",
    "transform_regime",
]

# FE-train partition boundary (inclusive); see CONTRACT_FE Section 0 rule 2.
FE_TRAIN_END = pd.Timestamp("2021-07-01")

# Markov fit iteration cap (statsmodels EM); matches characterize.regime.
_MARKOV_MAXITER = 100

# Output columns, in canonical order.
_OUTPUT_COLS = [
    "f3_gmm_prob_highvol",
    "f3_markov_prob_highvol",
    "f3_markov_switch_prob",
    "f3_regime_dwell",
]


@dataclass
class RegimeBundle:
    """Frozen FE-train regime artifacts for one instrument.

    Attributes
    ----------
    gmm : sklearn.mixture.GaussianMixture or None
        The fitted Gaussian mixture (``None`` when the fit failed).
    gmm_feat_mean, gmm_feat_std : np.ndarray
        FROZEN FE-train mean/std of the ``(ret, vol)`` feature columns, used to
        standardize at transform time. Zero-std entries are replaced by ``1.0``
        so the standardization is well defined.
    gmm_highvol_comp : int
        Index of the GMM component with the larger mean *raw* vol on FE-train
        (the high-vol regime); ``-1`` when the fit failed.
    markov_params : np.ndarray
        The FE-train ``MarkovRegression`` parameter vector (frozen), applied via
        the model-level ``.filter`` at transform time. Empty when the fit failed.
    markov_highvol_regime : int
        Index of the Markov regime with the larger fitted residual variance (the
        high-vol regime); ``-1`` when the fit failed.
    train_index : pd.DatetimeIndex
        The FE-train dates the models were fit on (``train_index <= FE-train``).
    n_eff_gate : int
        Optional effective-sample gate value for this instrument (defaults to
        ``-1`` / "not supplied"); recorded for provenance, never used to skip the
        fit.
    instrument : str
        Instrument identifier (for logging / provenance).
    ok : bool
        ``True`` iff BOTH model fits succeeded; ``False`` triggers all-NaN
        structural output at transform time.
    """

    gmm: object
    gmm_feat_mean: np.ndarray
    gmm_feat_std: np.ndarray
    gmm_highvol_comp: int
    markov_params: np.ndarray
    markov_highvol_regime: int
    train_index: pd.DatetimeIndex
    n_eff_gate: int
    instrument: str
    ok: bool


def _failed_bundle(
    train_index: pd.DatetimeIndex,
    instrument: str,
    n_eff_gate: int,
    reason: str,
) -> RegimeBundle:
    """Build an ``ok=False`` bundle and log *why* (never raises).

    A failed bundle still records ``train_index`` (for the provenance assertion)
    but carries empty/sentinel model artifacts, so :func:`transform_regime`
    emits structural NaNs.
    """
    warnings.warn(
        f"[regime_features] {instrument}: regime fit failed ({reason}); "
        "bundle.ok=False, features will be structural NaN.",
        stacklevel=2,
    )
    return RegimeBundle(
        gmm=None,
        gmm_feat_mean=np.full(2, np.nan),
        gmm_feat_std=np.full(2, np.nan),
        gmm_highvol_comp=-1,
        markov_params=np.empty(0),
        markov_highvol_regime=-1,
        train_index=train_index,
        n_eff_gate=int(n_eff_gate),
        instrument=instrument,
        ok=False,
    )


def fit_regime(
    train_ret_vol: pd.DataFrame,
    seed: int = 0,
    n_components: int = 2,
    instrument: str = "",
    n_eff_gate: int = -1,
) -> RegimeBundle:
    """Fit the GMM + Markov regime models on a single instrument's FE-train data.

    Parameters
    ----------
    train_ret_vol : pd.DataFrame
        Date-indexed FE-train rows with columns ``["ret", "vol"]`` (already
        restricted to dates ``<= FE-train``; rows with NaN ``ret``/``vol`` are
        dropped here before fitting). The GMM is fit on the FROZEN-standardized
        ``(ret, vol)`` features; the Markov model is fit on the ``ret`` column.
    seed : int, default 0
        ``random_state`` for the Gaussian mixture (determinism).
    n_components : int, default 2
        Number of GMM components / Markov regimes (the contract pins 2).
    instrument : str, default ""
        Instrument identifier, stored on the bundle for logging / provenance.
    n_eff_gate : int, default -1
        Optional effective-sample gate value to record on the bundle.

    Returns
    -------
    RegimeBundle
        ``ok=True`` with frozen artifacts when both fits succeed; otherwise an
        ``ok=False`` bundle (logged) whose transform yields structural NaNs.

    Notes
    -----
    The high-vol GMM component is the one with the larger mean *raw* vol on
    FE-train; the high-vol Markov regime is the one with the larger fitted
    residual variance (``params[-n_components:]``). Both are recorded so the
    transform never re-derives a regime label from out-of-sample data.
    Every model fit is wrapped so this function never raises.
    """
    feat = train_ret_vol.loc[:, ["ret", "vol"]].dropna().sort_index()
    train_index = pd.DatetimeIndex(feat.index)

    # Defensive: too few observations for an honest 2-regime fit.
    if len(feat) < 30:
        return _failed_bundle(
            train_index, instrument, n_eff_gate, f"only {len(feat)} train rows"
        )

    x = feat.to_numpy(dtype=float)
    raw_vol = feat["vol"].to_numpy(dtype=float)

    # FROZEN train (ret, vol) standardization stats.
    feat_mean = x.mean(axis=0)
    feat_std = x.std(axis=0)
    feat_std = np.where(feat_std == 0.0, 1.0, feat_std)

    # --- GMM on FROZEN-standardized (ret, vol) -------------------------------
    try:
        from sklearn.mixture import GaussianMixture

        x_std = (x - feat_mean) / feat_std
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=seed,
            max_iter=200,
        )
        labels = gmm.fit_predict(x_std)
        # High-vol component = larger mean RAW vol on FE-train.
        comp_vol = {
            c: float(raw_vol[labels == c].mean())
            for c in np.unique(labels)
            if (labels == c).any()
        }
        if not comp_vol:
            raise ValueError("GMM produced no populated components")
        gmm_highvol_comp = int(max(comp_vol, key=comp_vol.__getitem__))
    except Exception as exc:  # noqa: BLE001 - model fit must never raise
        return _failed_bundle(
            train_index, instrument, n_eff_gate, f"GMM {type(exc).__name__}: {exc}"
        )

    # --- Markov-switching on FE-train returns --------------------------------
    try:
        with warnings.catch_warnings():
            # Suppress ConvergenceWarning / RuntimeWarning during EM.
            warnings.simplefilter("ignore")
            ret_train = feat["ret"].to_numpy(dtype=float)
            m_train = MarkovRegression(
                ret_train,
                k_regimes=n_components,
                trend="c",
                switching_variance=True,
            )
            res_train = m_train.fit(maxiter=_MARKOV_MAXITER, disp=False)
        markov_params = np.asarray(res_train.params, dtype=float)
        # The switching-variance params are the LAST k entries of the vector
        # (statsmodels orders them as sigma2[0..k-1]); larger => high-vol.
        sigma2 = markov_params[-n_components:]
        if not np.all(np.isfinite(sigma2)):
            raise ValueError("non-finite Markov variance params")
        markov_highvol_regime = int(np.argmax(sigma2))
    except Exception as exc:  # noqa: BLE001 - statsmodels can fail to converge
        return _failed_bundle(
            train_index, instrument, n_eff_gate, f"Markov {type(exc).__name__}: {exc}"
        )

    return RegimeBundle(
        gmm=gmm,
        gmm_feat_mean=feat_mean,
        gmm_feat_std=feat_std,
        gmm_highvol_comp=gmm_highvol_comp,
        markov_params=markov_params,
        markov_highvol_regime=markov_highvol_regime,
        train_index=train_index,
        n_eff_gate=int(n_eff_gate),
        instrument=instrument,
        ok=True,
    )


def _dwell_since_change(regime_seq: np.ndarray) -> np.ndarray:
    """Trailing count of days since the regime label last changed.

    Causal by construction: position ``i`` depends only on ``regime_seq[:i+1]``.
    The first observation has dwell ``1`` (it is its own run so far); a value
    equal to the previous increments the count, a change resets it to ``1``.
    NaN labels reset the counter and carry forward as NaN dwell.
    """
    n = regime_seq.shape[0]
    dwell = np.full(n, np.nan)
    count = 0
    prev = np.nan
    for i in range(n):
        cur = regime_seq[i]
        if not np.isfinite(cur):
            count = 0
            prev = np.nan
            continue
        if not np.isfinite(prev) or cur != prev:
            count = 1
        else:
            count += 1
        dwell[i] = float(count)
        prev = cur
    return dwell


def _nan_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """All-NaN structural output frame with the canonical columns."""
    return pd.DataFrame(
        {col: np.full(len(index), np.nan) for col in _OUTPUT_COLS},
        index=index,
    )


def transform_regime(bundle: RegimeBundle, ret_vol_all: pd.DataFrame) -> pd.DataFrame:
    """Causally transform a full per-instrument ``(ret, vol)`` series to F3 cols.

    Parameters
    ----------
    bundle : RegimeBundle
        The frozen FE-train artifacts from :func:`fit_regime`.
    ret_vol_all : pd.DataFrame
        Date-indexed FULL per-instrument series with columns ``["ret", "vol"]``
        (so the Markov ``.filter`` runs over all observations). Rows where a
        feature is NaN yield NaN posteriors for that row (structural, never
        filled).

    Returns
    -------
    pd.DataFrame
        Date-indexed (aligned to ``ret_vol_all.index``) with columns
        ``f3_gmm_prob_highvol``, ``f3_markov_prob_highvol``,
        ``f3_markov_switch_prob`` (``|Δ|`` of the trailing filtered high-vol
        prob), and ``f3_regime_dwell`` (trailing days since the argmax regime
        last changed). When ``bundle.ok`` is ``False`` every column is a
        structural NaN (logged); structural NaNs are NEVER forward-filled.

    Notes
    -----
    Causality. The GMM term uses ``predict_proba`` on rows standardized with the
    FROZEN train mean/std, so row ``t`` depends only on ``(ret, vol)_t``. The
    Markov term builds a model on the FULL return array and applies the frozen
    train params via the model-level ``.filter`` -- the FILTERED (one-sided)
    marginal probabilities are a function of returns ``<= t`` only and are
    therefore truncation-invariant (proven in ``test_regime_features.py``).
    """
    index = pd.DatetimeIndex(ret_vol_all.index)

    if not bundle.ok:
        warnings.warn(
            f"[regime_features] {bundle.instrument or '?'}: bundle.ok=False; "
            "emitting structural-NaN regime features.",
            stacklevel=2,
        )
        return _nan_frame(index)

    rv = ret_vol_all.loc[:, ["ret", "vol"]].sort_index()
    index = pd.DatetimeIndex(rv.index)

    # --- GMM filtered high-vol posterior (causal predict_proba) --------------
    gmm_prob = np.full(len(rv), np.nan)
    x = rv.to_numpy(dtype=float)
    valid = np.isfinite(x).all(axis=1)
    if valid.any():
        x_std = (x[valid] - bundle.gmm_feat_mean) / bundle.gmm_feat_std
        proba = bundle.gmm.predict_proba(x_std)
        gmm_prob[valid] = proba[:, bundle.gmm_highvol_comp]

    # --- Markov FILTERED (one-sided/causal) high-vol posterior ---------------
    # statsmodels MarkovRegression requires a finite series; the released window
    # has dense returns per instrument, but guard the (rare) NaN by filtering on
    # the finite subseries and scattering back to the original positions.
    markov_prob = np.full(len(rv), np.nan)
    ret_all = rv["ret"].to_numpy(dtype=float)
    ret_valid = np.isfinite(ret_all)
    if ret_valid.any():
        ret_fit = ret_all[ret_valid]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_full = MarkovRegression(
                ret_fit,
                k_regimes=2,
                trend="c",
                switching_variance=True,
            )
            res = m_full.filter(bundle.markov_params)
            fp = np.asarray(res.filtered_marginal_probabilities, dtype=float)
        # Contract: filtered marginal probabilities are shape (nobs, 2).
        assert fp.shape == (ret_fit.shape[0], 2), (
            f"filtered_marginal_probabilities shape {fp.shape} "
            f"!= ({ret_fit.shape[0]}, 2)"
        )
        markov_prob[ret_valid] = fp[:, bundle.markov_highvol_regime]

    # --- Trailing switch probability = |Δ filtered high-vol prob| ------------
    # diff() is a trailing operator (row t uses t and t-1), so this is causal;
    # the first row has no predecessor and is left NaN.
    markov_switch = np.full(len(rv), np.nan)
    if len(markov_prob) > 1:
        markov_switch[1:] = np.abs(np.diff(markov_prob))

    # --- Trailing regime dwell (days since argmax regime last changed) -------
    # The argmax of the two filtered probabilities is the causal regime call at
    # t; dwell counts the trailing run length up to and including t.
    regime_seq = np.where(np.isfinite(markov_prob), (markov_prob >= 0.5).astype(float), np.nan)
    dwell = _dwell_since_change(regime_seq)

    return pd.DataFrame(
        {
            "f3_gmm_prob_highvol": gmm_prob,
            "f3_markov_prob_highvol": markov_prob,
            "f3_markov_switch_prob": markov_switch,
            "f3_regime_dwell": dwell,
        },
        index=index,
    )
