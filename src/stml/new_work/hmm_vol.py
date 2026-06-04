"""hmm_vol.py — Core #1: Per-instrument volatility / turbulence HMM.

Model
-----
M = 3 Gaussian hidden states, ordered calm → moderate → turbulent.
d = 1 observation: log Garman-Klass daily realised volatility.

Observation choice
------------------
The Garman-Klass (1980) estimator σ²_GK = 0.5·(ln H/L)² – (2 ln 2 – 1)·(ln C/O)²
uses the full intraday price range (open, high, low, close) rather than only
close-to-close returns.  This roughly halves estimation variance at the same
sample size, giving a cleaner vol signal for regime detection.

Design (leakage-free)
---------------------
1. Fit each HMM exclusively on the PRE-SAMPLE history that predates the
   645-day primary_signals window (cutoff: 2020-01-03).  Parameters are then
   FROZEN — no re-estimation occurs inside the metamodel window.
2. Apply FILTERING: the forward algorithm produces p(H_t | X_1:t), which
   conditions only on observations up to and including t.  Smoothing
   (forward-backward) would condition on the full future sequence and is
   therefore invalid for backtest features.
3. To ensure the state distribution at the start of the metamodel window
   is correctly conditioned, the forward pass runs on the full sequence
   (pre-sample + metamodel), but features are only extracted for dates
   in the primary_signals window.

State ordering
--------------
After fitting, states are re-indexed by ascending mean log-vol so that
state 0 = calm, state 1 = moderate, state 2 = turbulent, consistently
across all instruments and across multiple EM restarts.

Features (per date, per instrument)
------------------------------------
    hmm_vol_p0_calm        : P(state 0 = calm      | X_1:t)
    hmm_vol_p2_turbulent   : P(state 2 = turbulent | X_1:t)
      (p1 is omitted — summing to 1 makes it redundant)
    hmm_vol_next_turbulent : one-step forecast P(turbulent at t+1) = α_t · Q[:,2]
    hmm_vol_entropy        : Shannon entropy H(α_t) — regime-ambiguity scalar

Public API
----------
    from stml.new_work.hmm_vol import run_pipeline, VOL_FEATURES, filter_forward

    features_df = run_pipeline()            # fits, filters, saves CSV
    features_df = run_pipeline(save=False)  # in-memory only
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent.parent   # …/stml
OHLCV_PATH = _REPO / "data" / "ohlcv_data.csv"
PS_PATH    = _REPO / "data" / "primary_signals.csv"
OUT_PATH   = _HERE / "features_hmm_vol.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_STATES    = 3
N_RESTARTS  = 20
N_ITER      = 300
RANDOM_SEED = 42
INSTRUMENTS = [
    "es1s", "nq1s", "fesx1s",
    "cl1s", "ho1s", "rb1s", "ng1s",
    "gc1s", "si1s", "hg1s", "pl1s",
]
VOL_FEATURES = [
    "hmm_vol_p0_calm",
    "hmm_vol_p2_turbulent",
    "hmm_vol_next_turbulent",
    "hmm_vol_entropy",
]


# ---------------------------------------------------------------------------
# Observation: Garman-Klass log volatility
# ---------------------------------------------------------------------------

def garman_klass_logvol(df: pd.DataFrame) -> pd.Series:
    """
    σ²_GK = 0.5·(ln H/L)² – (2 ln 2 – 1)·(ln C/O)²

    Returns log(max(σ²_GK, ε)) aligned with df's index.
    Clamping prevents log(0) on rare days where H=L=O=C.
    """
    ln_hl = np.log(df["high"] / df["low"])
    ln_co = np.log(df["close"] / df["open"])
    gk = 0.5 * ln_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * ln_co ** 2
    return np.log(np.maximum(gk, 1e-10))


# ---------------------------------------------------------------------------
# Gaussian emission log-probabilities (no private hmmlearn API)
# ---------------------------------------------------------------------------

def _gaussian_log_emit(
    X: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
) -> np.ndarray:
    """
    log p(x_t | H_t = m) for each (t, m).

    Parameters
    ----------
    X       : (T, d)
    means   : (M, d)          — model.means_
    covars  : (M, d) or (M, d, d)
              hmmlearn 0.3.x returns a full (M, d, d) matrix from the
              covars_ property; the compact diagonal (M, d) form is stored
              in model._covars_.  Both shapes are handled here.

    Returns
    -------
    log_emit : (T, M)
    """
    T = X.shape[0]
    M = len(means)
    log_emit = np.empty((T, M))
    for m in range(M):
        mu = means[m]                                       # (d,)
        c  = covars[m]
        var = np.diag(c) if c.ndim == 2 else c             # (d,)  σ²
        log_emit[:, m] = -0.5 * (
            np.sum(((X - mu) ** 2) / var, axis=1)
            + np.sum(np.log(2.0 * np.pi * var))
        )
    return log_emit


# ---------------------------------------------------------------------------
# Scaled forward algorithm (filtering)
# ---------------------------------------------------------------------------

def filter_forward(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Compute filtering posteriors p(H_t | X_1:t) via the scaled forward pass.

    Using the notation from Rabiner (1989) and the course session 3 notebook:

        α_t(m) ∝ p(H_t = m | X_1, …, X_t)

    Recursion:
        α_0(m) ∝ π_m · p(x_0 | H_0 = m)
        α_t(m) ∝ [Σ_{m'} α_{t-1}(m') · Q_{m'm}] · p(x_t | H_t = m)

    The scale factor (logsumexp normalisation) is applied at each step to
    prevent numerical underflow — equivalent to the c_t scaling in Rabiner.

    IMPORTANT: this is NOT smoothing.  No future observation influences α_t.

    Parameters
    ----------
    model : fitted GaussianHMM (parameters frozen after pre-sample fit)
    X     : (T, d) standardised observation sequence

    Returns
    -------
    filtered : (T, M)  each row sums to 1
    """
    T  = X.shape[0]
    M  = model.n_components
    log_emit = _gaussian_log_emit(X, model.means_, model.covars_)
    log_pi   = np.log(np.clip(model.startprob_, 1e-300, 1.0))
    log_Q    = np.log(np.clip(model.transmat_,  1e-300, 1.0))   # (M, M)

    # t = 0
    log_alpha = log_pi + log_emit[0]
    log_alpha -= logsumexp(log_alpha)

    filtered = np.empty((T, M))
    filtered[0] = np.exp(log_alpha)

    for t in range(1, T):
        # Prediction step: Σ_{m'} α_{t-1}(m') · Q_{m'm}  in log-space
        log_alpha = (
            logsumexp(log_alpha[:, None] + log_Q, axis=0)   # (M,)
            + log_emit[t]
        )
        log_alpha -= logsumexp(log_alpha)
        filtered[t] = np.exp(log_alpha)

    return filtered


# ---------------------------------------------------------------------------
# EM fitting with multiple restarts
# ---------------------------------------------------------------------------

def fit_hmm_best(
    X: np.ndarray,
    n_components: int,
    n_restarts: int  = N_RESTARTS,
    n_iter: int      = N_ITER,
    base_seed: int   = RANDOM_SEED,
) -> tuple[GaussianHMM, float]:
    """
    Fit GaussianHMM (diagonal covariance) with multiple random EM initialisations.

    Running multiple restarts guards against local optima in the Baum-Welch
    EM procedure.  The model with the highest log-likelihood is returned.

    Returns
    -------
    (best_model, best_log_likelihood_per_obs)
    """
    best_model: GaussianHMM | None = None
    best_ll = -np.inf
    for i in range(n_restarts):
        model = GaussianHMM(
            n_components    = n_components,
            covariance_type = "diag",
            n_iter          = n_iter,
            random_state    = base_seed + i,
            tol             = 1e-5,
        )
        try:
            model.fit(X)
            ll = model.score(X)   # per-observation log-likelihood
            if np.isfinite(ll) and ll > best_ll:
                best_ll    = ll
                best_model = model
        except Exception:
            pass
    return best_model, best_ll


# ---------------------------------------------------------------------------
# Deterministic state ordering
# ---------------------------------------------------------------------------

def sort_states_by_mean(model: GaussianHMM) -> GaussianHMM:
    """
    Reorder states so means are ascending: state 0 = calm, state 2 = turbulent.

    The reordering permutes startprob_, transmat_, means_, and _covars_
    consistently so the model remains valid.  _covars_ (compact diagonal form,
    shape (M, d)) is manipulated directly because the covars_ setter in
    hmmlearn 0.3.x expects the compact form while the getter returns the full
    (M, d, d) matrix.
    """
    order = np.argsort(model.means_.flatten())
    model.startprob_ = model.startprob_[order]
    model.transmat_  = model.transmat_[np.ix_(order, order)]
    model.means_     = model.means_[order]
    model._covars_   = model._covars_[order]   # compact (M, d) form
    return model


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(save: bool = True, verbose: bool = True) -> pd.DataFrame:
    """
    Full pipeline: fit per-instrument vol HMMs on pre-sample, filter forward.

    Returns a tidy DataFrame (645 dates × 11 instruments = 7,095 rows):
        date | instrument | hmm_vol_p0_calm | hmm_vol_p2_turbulent
             | hmm_vol_next_turbulent | hmm_vol_entropy
    """
    ohlcv = pd.read_csv(OHLCV_PATH, parse_dates=["date"])
    ps    = pd.read_csv(PS_PATH,    parse_dates=["date"])

    cutoff         = ps["date"].min()       # 2020-01-03
    meta_dates_set = set(ps["date"])

    records: list[dict] = []

    for inst in INSTRUMENTS:
        inst_df = (
            ohlcv[ohlcv["instrument"] == inst]
            .sort_values("date")
            .set_index("date")
        )

        log_vol   = garman_klass_logvol(inst_df).dropna()
        presample = log_vol[log_vol.index < cutoff]

        # Standardise using pre-sample statistics only
        mu    = presample.mean()
        sigma = presample.std()
        X_pre  = ((presample - mu) / sigma).values.reshape(-1, 1)
        X_full = ((log_vol - mu) / sigma).values.reshape(-1, 1)

        # 1. Fit on pre-sample, freeze parameters
        model, best_ll = fit_hmm_best(X_pre, N_STATES)
        model = sort_states_by_mean(model)

        # 2. Filter forward on full sequence (parameters frozen)
        filtered   = filter_forward(model, X_full)
        full_dates = log_vol.index
        Q          = model.transmat_
        TURB       = N_STATES - 1   # index of turbulent state after sorting

        for idx, d in enumerate(full_dates):
            if d not in meta_dates_set:
                continue
            alpha = filtered[idx]
            records.append({
                "date":                   d,
                "instrument":             inst,
                "hmm_vol_p0_calm":        float(alpha[0]),
                "hmm_vol_p2_turbulent":   float(alpha[2]),
                "hmm_vol_next_turbulent": float(alpha @ Q[:, TURB]),
                "hmm_vol_entropy":        float(
                    -np.sum(alpha * np.log(np.clip(alpha, 1e-300, 1.0)))
                ),
            })

        if verbose:
            raw_means = model.means_.flatten() * sigma + mu
            print(
                f"  {inst:7s}: ll={best_ll:8.2f} | "
                f"log-GK means (calm/mod/turb): {np.round(raw_means, 3)}"
            )

    df = (
        pd.DataFrame(records)
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )

    if save:
        df.to_csv(OUT_PATH, index=False)
        if verbose:
            print(f"\nSaved {len(df):,} rows → {OUT_PATH}")

    return df


if __name__ == "__main__":
    run_pipeline()
