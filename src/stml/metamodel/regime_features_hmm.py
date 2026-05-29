"""
regime_features_hmm.py
======================
Family **F17 — HMM regime posteriors** (TF-class, ported from the ``Sreeram``
branch). A Gaussian Hidden Markov Model is fit per instrument on its FE-train
``(ret, vol)`` observations, then **filtered** (forward-only) state posteriors
are produced over the full series — strictly causal by construction.

This complements the existing F3 family (filtered GMM + Markov-switching
posteriors): F3 is a 2-regime GMM/Markov pair, F17 is a 3-state Gaussian HMM
whose transition matrix couples the states through time. Both are fit on the
FE-train partition only and frozen, mirroring :mod:`stml.metamodel.regime_features`.

Causality
---------
1. The HMM is fit on observations with ``date < FE-train`` only.
2. The posterior at ``t`` is the **filtered** (one-sided) posterior
   ``P(state | X_{0..t})`` computed from a from-scratch forward pass — never
   hmmlearn's smoothed ``predict_proba`` (which uses the backward pass and would
   leak ``t+1..T``).
3. States are ordered by FE-train mean vol (lo/mid/hi) so the columns are
   comparable across instruments and runs (resolves HMM label-switching).

Requires ``hmmlearn`` — install with ``uv sync --extra features-extra``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["HmmBundle", "fit_hmm", "transform_hmm", "HMM_COLUMNS"]

#: The four produced F17 columns, in order (3 lo/mid/hi probabilities + argmax).
HMM_COLUMNS: tuple[str, ...] = (
    "f17_hmm_state_lo",
    "f17_hmm_state_mid",
    "f17_hmm_state_hi",
    "f17_hmm_state_argmax",
)


@dataclass
class HmmBundle:
    """Frozen FE-train HMM artifacts for one instrument.

    Attributes
    ----------
    hmm : hmmlearn.hmm.GaussianHMM or None
        The fitted model (``None`` when the fit failed).
    order : np.ndarray
        Permutation that sorts the states by ascending FE-train mean vol, so
        column 0 = lowest-vol state ... column n-1 = highest-vol state.
    n_states : int
        Number of HMM states (the contract pins 3: lo / mid / hi).
    train_index : pd.DatetimeIndex
        The FE-train observation dates the model was fit on.
    instrument : str
        Instrument identifier (for logging / provenance).
    ok : bool
        ``True`` iff the fit succeeded; ``False`` triggers structural-NaN output.
    """

    hmm: object
    order: np.ndarray
    n_states: int
    train_index: pd.DatetimeIndex
    instrument: str
    ok: bool


def _failed_bundle(
    train_index: pd.DatetimeIndex, instrument: str, n_states: int, reason: str
) -> HmmBundle:
    """Build an ``ok=False`` bundle and log *why* (never raises)."""
    warnings.warn(
        f"[regime_features_hmm] {instrument}: HMM fit failed ({reason}); "
        "bundle.ok=False, features will be structural NaN.",
        stacklevel=2,
    )
    return HmmBundle(
        hmm=None,
        order=np.arange(n_states),
        n_states=n_states,
        train_index=train_index,
        instrument=instrument,
        ok=False,
    )


def _causal_filtered_probs(hmm: object, X: np.ndarray) -> np.ndarray:
    """Forward-only (filtered) posteriors ``P(state | X_{0..t})``.

    Computed directly from the log emission likelihoods, the start vector and
    the transition matrix::

        log a[0] = log pi + log e[0]
        log a[t] = log e[t] + logsumexp_k'( log a[t-1, k'] + log A[k', k] )

    then row-normalised. ``filtered[t]`` depends only on ``X[0..t]`` (no backward
    pass), so the feature is truncation-invariant — unlike hmmlearn's smoothed
    ``predict_proba``.
    """
    from scipy.special import logsumexp

    log_emis = hmm._compute_log_likelihood(X)  # (T, K)
    log_pi = np.log(np.clip(hmm.startprob_, 1e-12, None))
    log_A = np.log(np.clip(hmm.transmat_, 1e-12, None))  # (K, K)

    T, K = log_emis.shape
    log_alpha = np.empty((T, K))
    log_alpha[0] = log_pi + log_emis[0]
    for t in range(1, T):
        log_alpha[t] = log_emis[t] + logsumexp(
            log_alpha[t - 1][:, None] + log_A, axis=0
        )
    log_filt = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
    return np.exp(log_filt)


def fit_hmm(
    train_ret_vol: pd.DataFrame,
    seed: int = 0,
    n_states: int = 3,
    instrument: str = "",
    min_train: int = 200,
) -> HmmBundle:
    """Fit a Gaussian HMM on one instrument's FE-train ``(ret, vol)`` rows.

    Parameters
    ----------
    train_ret_vol : pd.DataFrame
        Date-indexed FE-train rows with columns ``["ret", "vol"]`` (already
        restricted to dates ``<= FE-train``; NaN rows are dropped here).
    seed : int, default 0
        ``random_state`` for the HMM EM fit (determinism).
    n_states : int, default 3
        Number of Gaussian HMM states (lo / mid / hi vol).
    instrument : str, default ""
        Instrument identifier, stored on the bundle for provenance.
    min_train : int, default 200
        Minimum FE-train observations for an honest fit; below this the bundle
        is ``ok=False`` (structural-NaN transform).

    Returns
    -------
    HmmBundle
        ``ok=True`` with the frozen model + vol-ascending state order on success;
        otherwise an ``ok=False`` bundle (logged). Never raises.
    """
    feat = train_ret_vol.loc[:, ["ret", "vol"]].dropna().sort_index()
    train_index = pd.DatetimeIndex(feat.index)
    if len(feat) < min_train:
        return _failed_bundle(
            train_index, instrument, n_states, f"only {len(feat)} train rows"
        )
    try:
        import logging

        from hmmlearn.hmm import GaussianHMM

        # hmmlearn logs non-convergence at WARNING via a logger (not warnings);
        # the EM still returns a usable frozen model, so quiet the chatter.
        logging.getLogger("hmmlearn").setLevel(logging.ERROR)

        x = feat.to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hmm = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=200,
                tol=1e-3,
                random_state=seed,
                init_params="stmc",
                params="stmc",
            )
            hmm.fit(x)
        # Order states by ascending mean vol (column 1 of the observation).
        order = np.argsort(hmm.means_[:, 1])
    except Exception as exc:  # noqa: BLE001 - model fit must never raise
        return _failed_bundle(
            train_index, instrument, n_states, f"{type(exc).__name__}: {exc}"
        )

    return HmmBundle(
        hmm=hmm,
        order=np.asarray(order),
        n_states=n_states,
        train_index=train_index,
        instrument=instrument,
        ok=True,
    )


def _nan_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """All-structural-NaN F17 frame (used on a failed/empty bundle)."""
    return pd.DataFrame(
        {c: np.full(len(index), np.nan) for c in HMM_COLUMNS}, index=index
    )


def transform_hmm(bundle: HmmBundle, ret_vol_all: pd.DataFrame) -> pd.DataFrame:
    """Causally transform a full per-instrument ``(ret, vol)`` series to F17 cols.

    Runs the from-scratch forward filter over every observation, reorders the
    state columns by the frozen vol-ascending order, and returns the lo/mid/hi
    filtered posteriors plus the argmax state, aligned to ``ret_vol_all.index``.
    Rows with NaN ``(ret, vol)`` (and warm-up before the bundle could fit) are
    structural NaN — never forward-filled. A failed bundle yields all-NaN.
    """
    index = pd.DatetimeIndex(ret_vol_all.index)
    if not bundle.ok or bundle.hmm is None:
        return _nan_frame(index)

    rv = ret_vol_all.loc[:, ["ret", "vol"]].sort_index()
    obs = rv.dropna()
    if obs.empty:
        return _nan_frame(index)

    filt = _causal_filtered_probs(bundle.hmm, obs.to_numpy(dtype=float))
    filt = filt[:, bundle.order]  # lo / mid / hi by ascending FE-train vol

    frame = pd.DataFrame(
        filt,
        index=obs.index,
        columns=["f17_hmm_state_lo", "f17_hmm_state_mid", "f17_hmm_state_hi"],
    )
    frame["f17_hmm_state_argmax"] = np.argmax(filt, axis=1).astype("float64")
    # Align back to the full input index (structural NaN where obs was dropped).
    return frame.reindex(pd.DatetimeIndex(rv.index)).reindex(index)
