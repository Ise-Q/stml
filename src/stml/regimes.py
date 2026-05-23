"""
regimes.py
==========
Latent-variable regime features (feature group G6) — the assignment's explicit
"Latent variable models (GMM, HMM)" callout and the Lecture 3 ("Latent Variable
Models in Financial Asset Regime Detection") showpiece.

Two regime feature families:

  - **HMM regimes** (Hidden Markov Model with Gaussian emissions): for each
    instrument, fit an ``n_states``-state HMM on a 2-D observation vector
    (log returns + rolling vol) using training data ONLY. Then compute
    **filtered** (forward-only) state posteriors at every date — strictly
    causal by construction. Yields ``n_states`` features per (date, instrument).

  - **GMM cluster membership**: per-instrument Gaussian mixture clustering of
    a feature snapshot (3-D: vol, momentum, autocorrelation). Fit on
    training data, then ``predict_proba`` at every date. Soft cluster
    membership ⇒ ``n_components`` features.

Causality is enforced by:
  1. Both models are fit on a slice ``data[date < boundary]``.
  2. The forward / posterior at date ``t`` uses only ``data[date <= t]``.
  3. HMM smoothed (forward+backward) probabilities are explicitly NOT used;
     the forward-only pass is implemented from scratch here.

This module is the most novel piece of the feature stack and the one most at
risk of silent leakage. The unit tests in :mod:`tests.test_regimes` lock down
the causal invariant.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _instrument_obs(
    ohlcv_long: pd.DataFrame,
    instrument: str,
    vol_window: int = 21,
) -> pd.DataFrame:
    """Build the HMM observation matrix for one instrument.

    Columns:
      ``ret``  = daily log return
      ``vol``  = rolling-``vol_window`` realised vol (annualised, log-units)

    Both are computed on the instrument's native dense series. The first
    ``vol_window`` rows have NaN vol — dropped.
    """
    s = (
        ohlcv_long.loc[ohlcv_long["instrument"] == instrument]
        .set_index("date")["close"]
        .sort_index()
    )
    s = s[~s.index.duplicated(keep="last")]
    ret = np.log(s).diff()
    vol = ret.rolling(vol_window).std() * np.sqrt(252)
    obs = pd.concat([ret.rename("ret"), vol.rename("vol")], axis=1).dropna()
    return obs


def _instrument_gmm_snapshot(
    ohlcv_long: pd.DataFrame,
    instrument: str,
) -> pd.DataFrame:
    """A 3-D "market state" snapshot for GMM clustering per (date, instrument).

    Columns: 21d vol (annualised), 21d momentum, 21d autocorrelation.
    """
    s = (
        ohlcv_long.loc[ohlcv_long["instrument"] == instrument]
        .set_index("date")["close"]
        .sort_index()
    )
    s = s[~s.index.duplicated(keep="last")]
    ret = np.log(s).diff()
    vol = (ret.rolling(21).std() * np.sqrt(252)).rename("gmm_vol")
    mom = (ret.rolling(21).sum()).rename("gmm_mom")
    autoc = ret.rolling(21).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if pd.Series(x).std() > 0 else 0.0,
        raw=True,
    ).rename("gmm_autoc")
    snap = pd.concat([vol, mom, autoc], axis=1).dropna()
    return snap


# --------------------------------------------------------------------------- #
# Causal HMM forward pass                                                     #
# --------------------------------------------------------------------------- #
def causal_filtered_probs(
    hmm: GaussianHMM,
    X: np.ndarray,
) -> np.ndarray:
    """Forward-only (filtered) state posteriors P(state_k | X_{0..t}).

    hmmlearn's ``predict_proba`` returns SMOOTHED posteriors that use both
    forward and backward passes — i.e. P(state_k | X_{0..T}). That is NOT
    causal: the filter at time t leaks information from t+1..T.

    This function computes the filtered posteriors directly:

        log alpha[0]   = log pi + log emission[0]
        log alpha[t]   = log emission[t] + logsumexp_k'( log alpha[t-1, k'] + log A[k', k] )

    Then normalises each row to give P(state_k | X_{0..t}). The result is
    strictly causal: ``filtered[t]`` depends only on ``X[0..t]``.

    Parameters
    ----------
    hmm : fitted ``hmmlearn.hmm.GaussianHMM``
    X : ndarray of shape (T, D)

    Returns
    -------
    ndarray of shape (T, n_states)
        Row-stochastic, each row sums to 1.
    """
    # Emission log-likelihood per (t, state).
    log_emis = hmm._compute_log_likelihood(X)  # (T, K)
    log_pi = np.log(np.clip(hmm.startprob_, 1e-12, None))
    log_A = np.log(np.clip(hmm.transmat_, 1e-12, None))  # (K, K)

    T, K = log_emis.shape
    log_alpha = np.empty((T, K))

    log_alpha[0] = log_pi + log_emis[0]
    for t in range(1, T):
        # log_alpha[t, k] = log_emis[t, k] + logsumexp_k'(log_alpha[t-1, k'] + log_A[k', k])
        log_alpha[t] = log_emis[t] + logsumexp(
            log_alpha[t - 1][:, None] + log_A, axis=0
        )

    # Normalise each row to a proper posterior.
    log_filt = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
    return np.exp(log_filt)


# --------------------------------------------------------------------------- #
# HMM per-instrument regime features                                          #
# --------------------------------------------------------------------------- #
def fit_instrument_hmm(
    obs: np.ndarray,
    n_states: int = 3,
    n_iter: int = 200,
    random_state: int = 42,
    covariance_type: str = "full",
) -> GaussianHMM:
    """Fit a Gaussian HMM on a (T, D) observation matrix."""
    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        tol=1e-3,
        init_params="stmc",
        params="stmc",
    )
    hmm.fit(obs)
    return hmm


def hmm_features_for_instrument(
    ohlcv_long: pd.DataFrame,
    instrument: str,
    boundary: pd.Timestamp,
    n_states: int = 3,
    vol_window: int = 21,
    random_state: int = 42,
) -> pd.DataFrame:
    """HMM regime features for one instrument, fully causal.

    Pipeline:
      1. Build (ret, vol) observations on the instrument's native series.
      2. Fit a Gaussian HMM **on observations with date < boundary**.
      3. Run the forward algorithm over the WHOLE series to get filtered
         posteriors P(state_k | X_{0..t}) at every date — causal because each
         filtered prob at t depends only on data up to t.
      4. Return state-ordered columns + most-likely state index.

    The state ordering is normalised by **state-mean vol** (state 0 = lowest
    vol, state K-1 = highest vol) so feature columns are interpretable across
    instruments and across runs (HMM training has a label-switching ambiguity
    — we resolve it here).

    Parameters
    ----------
    ohlcv_long, instrument : as elsewhere.
    boundary : training cutoff. Only observations strictly before this date
        are used to fit the HMM.
    n_states : default 3 (low / mid / high vol intuition).
    vol_window, random_state : see :func:`fit_instrument_hmm`.

    Returns
    -------
    pd.DataFrame indexed by date with columns:
        ``hmm_state_lo``, ``hmm_state_mid``, ``hmm_state_hi`` (probabilities),
        ``hmm_state_argmax`` (integer 0..n_states-1, the lowest-vol-aligned
        most-likely state).
    """
    obs = _instrument_obs(ohlcv_long, instrument, vol_window=vol_window)
    if obs.empty:
        return pd.DataFrame()

    train_mask = obs.index < boundary
    if train_mask.sum() < 200:  # need a reasonable training sample
        return pd.DataFrame(index=obs.index)
    train_obs = obs.loc[train_mask].values

    hmm = fit_instrument_hmm(
        train_obs,
        n_states=n_states,
        random_state=random_state,
    )

    # Filtered posteriors over the WHOLE series.
    filt = causal_filtered_probs(hmm, obs.values)  # (T, K)

    # State reordering: ascend by mean vol (column 1 of obs = realised vol).
    state_mean_vol = hmm.means_[:, 1]
    order = np.argsort(state_mean_vol)  # ascending
    filt = filt[:, order]

    # Build named columns. For n_states=3 use lo/mid/hi; else state_0..K-1.
    if n_states == 3:
        col_names = ["hmm_state_lo", "hmm_state_mid", "hmm_state_hi"]
    else:
        col_names = [f"hmm_state_{i}" for i in range(n_states)]
    df = pd.DataFrame(filt, index=obs.index, columns=col_names)
    df["hmm_state_argmax"] = np.argmax(filt, axis=1)
    return df


# --------------------------------------------------------------------------- #
# GMM cluster membership                                                       #
# --------------------------------------------------------------------------- #
def gmm_features_for_instrument(
    ohlcv_long: pd.DataFrame,
    instrument: str,
    boundary: pd.Timestamp,
    n_components: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """GMM soft-cluster membership over a 3-D market-state snapshot, causal.

    Fit on snapshot rows with date < boundary; ``predict_proba`` at every date.
    """
    snap = _instrument_gmm_snapshot(ohlcv_long, instrument)
    if snap.empty:
        return pd.DataFrame()

    train_mask = snap.index < boundary
    if train_mask.sum() < 200:
        return pd.DataFrame(index=snap.index)
    train = snap.loc[train_mask].values

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        max_iter=300,
        random_state=random_state,
        reg_covar=1e-4,
    )
    gmm.fit(train)
    # `predict_proba` on snap.values uses NO future information when applied to
    # ROW i — each row is an independent input. The snapshot itself is causal
    # (uses only data up to its date). So overall causal.
    probs = gmm.predict_proba(snap.values)

    # Reorder clusters by training-set mean vol (component[0] of means_).
    mean_vol_by_cluster = gmm.means_[:, 0]
    order = np.argsort(mean_vol_by_cluster)
    probs = probs[:, order]

    if n_components == 3:
        cols = ["gmm_cluster_lo", "gmm_cluster_mid", "gmm_cluster_hi"]
    else:
        cols = [f"gmm_cluster_{i}" for i in range(n_components)]
    out = pd.DataFrame(probs, index=snap.index, columns=cols)
    out["gmm_cluster_argmax"] = np.argmax(probs, axis=1)
    return out


# --------------------------------------------------------------------------- #
# Master: build regime features for an events frame                            #
# --------------------------------------------------------------------------- #
def compute_regime_features(
    ohlcv_long: pd.DataFrame,
    events: pd.DataFrame,
    boundary: pd.Timestamp,
    n_states: int = 3,
    n_components: int = 3,
    vol_window: int = 21,
    random_state: int = 42,
) -> pd.DataFrame:
    """Per-event regime features (HMM + GMM), aligned to ``events``.

    Returns a DataFrame indexed by ``events.index`` with columns:
        hmm_state_lo, hmm_state_mid, hmm_state_hi, hmm_state_argmax,
        gmm_cluster_lo, gmm_cluster_mid, gmm_cluster_hi, gmm_cluster_argmax.

    All values at event date ``t`` use only data with date ``<= t`` (and the
    HMM/GMM models themselves are trained on data with date ``< boundary``).
    """
    if events.empty:
        return pd.DataFrame()

    universe = sorted(events["instrument"].unique())
    per_inst_hmm: dict[str, pd.DataFrame] = {}
    per_inst_gmm: dict[str, pd.DataFrame] = {}
    for inst in universe:
        per_inst_hmm[inst] = hmm_features_for_instrument(
            ohlcv_long, inst, boundary,
            n_states=n_states, vol_window=vol_window,
            random_state=random_state,
        )
        per_inst_gmm[inst] = gmm_features_for_instrument(
            ohlcv_long, inst, boundary,
            n_components=n_components, random_state=random_state,
        )

    rows: list[dict] = []
    for ev_id, ev in events.iterrows():
        t, inst = ev["t"], ev["instrument"]
        row: dict = {}
        for src in (per_inst_hmm.get(inst, pd.DataFrame()),
                    per_inst_gmm.get(inst, pd.DataFrame())):
            if src.empty:
                continue
            if t in src.index:
                row.update(src.loc[t].to_dict())
            else:
                # Forward-fill up to t (defensive).
                tmp = src.reindex(src.index.union([t]).sort_values()).ffill()
                if t in tmp.index:
                    row.update(tmp.loc[t].to_dict())
        rows.append(row)

    return pd.DataFrame(rows, index=events.index)
