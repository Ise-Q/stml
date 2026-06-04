"""hmm_macro.py — Core #2: Global macro risk-on / risk-off HMM.

Model
-----
M selected from {2, 3} via BIC + forward-chaining time-series CV.
d = 4 standardised macro features:
    f11_hy_oas_z        — high-yield OAS z-score        (credit stress ↑)
    f11_vix_level_z     — VIX level z-score             (equity fear ↑)
    f11_us_2s10s_slope  — US 2s10s yield-curve slope    (recession risk ↓)
    f11_dxy_5d_change   — 5-day DXY change              (USD flight-to-safety ↑)

State ordering
--------------
After fitting, states are re-indexed by ascending composite risk score
  risk_m = mean_VIX_z(m) + mean_HY_OAS_z(m)
so that state 0 = risk-on (low VIX, tight spreads) and the highest-indexed
state = risk-off / stress.  A single global regime is applied to all instruments.

Design (leakage-free) — same as Core #1
-----------------------------------------
1. Fit HMM exclusively on the pre-sample (before 2020-01-03).  Freeze.
2. Standardise using pre-sample mean/std only (scaler is frozen too).
3. Run the scaled forward algorithm on the full sequence (pre-sample +
   metamodel).  Extract features only for dates in the primary_signals window.
4. Forward filtering only — no smoothing.

M selection methodology
------------------------
BIC  = −2·ℓ̂ + k·log(n),   k = free params = (M−1) + M(M−1) + 2Md
       Lower BIC → preferred.

Forward-chain TS-CV: the pre-sample is split into N_CV_SPLITS+1 folds.
  Fold i: train on observations 0 … split_i, evaluate on split_i … split_{i+1}.
  Metric: mean per-observation held-out log-likelihood (higher = better).

Selection rule: pick M where BIC and CV agree; if they disagree, tie-break
toward fewer states (parsimony + longer expected dwell times).

Features (global per date, broadcast to all instruments)
---------------------------------------------------------
    hmm_macro_p0            : P(state 0 = risk-on | X_1:t)   [M-1 cols; last dropped]
    hmm_macro_next_riskoff  : one-step P(risk-off at t+1) = α_t · Q[:,M-1]
    hmm_macro_entropy       : Shannon entropy H(α_t)

For M=2 there is 1 probability column (p0); for M=3 there are 2 (p0, p1).
The highest-indexed state (risk-off) is always the dropped one (redundant
given the others sum to 1).

Public API
----------
    from stml.new_work.hmm_macro import run_pipeline, MACRO_FEATURES

    features_df, model, chosen_M, sel_table = run_pipeline()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE  = Path(__file__).parent
_REPO  = _HERE.parent.parent.parent
MACRO_PATH = _REPO / "data" / "meta" / "macro_features.csv"
PS_PATH    = _REPO / "data" / "primary_signals.csv"
OUT_PATH   = _HERE / "features_hmm_macro.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURES = [
    "f11_hy_oas_z",
    "f11_vix_level_z",
    "f11_us_2s10s_slope",
    "f11_dxy_5d_change",
]
INSTRUMENTS = [
    "es1s", "nq1s", "fesx1s",
    "cl1s", "ho1s", "rb1s", "ng1s",
    "gc1s", "si1s", "hg1s", "pl1s",
]
N_RESTARTS   = 20
N_ITER       = 300
RANDOM_SEED  = 42
N_CV_SPLITS  = 5

# Features vary by chosen M; base list for M=2
MACRO_FEATURES = ["hmm_macro_p0", "hmm_macro_next_riskoff", "hmm_macro_entropy"]


# ---------------------------------------------------------------------------
# Gaussian emission log-probabilities
# ---------------------------------------------------------------------------

def _gaussian_log_emit(
    X: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
) -> np.ndarray:
    """
    log p(x_t | H_t = m) for each (t, m).  Returns (T, M).

    Accepts covars in either compact (M, d) or full (M, d, d) form —
    hmmlearn 0.3.x returns (M, d, d) from the covars_ property.
    """
    T = X.shape[0]
    M = len(means)
    log_emit = np.empty((T, M))
    for m in range(M):
        mu  = means[m]
        c   = covars[m]
        var = np.diag(c) if c.ndim == 2 else c   # σ² diagonal
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

    Identical to the vol-model implementation.  See hmm_vol.py for full
    derivation.  Returns (T, M) array, rows sum to 1.
    """
    T  = X.shape[0]
    M  = model.n_components
    log_emit = _gaussian_log_emit(X, model.means_, model.covars_)
    log_pi   = np.log(np.clip(model.startprob_, 1e-300, 1.0))
    log_Q    = np.log(np.clip(model.transmat_,  1e-300, 1.0))

    log_alpha = log_pi + log_emit[0]
    log_alpha -= logsumexp(log_alpha)

    filtered = np.empty((T, M))
    filtered[0] = np.exp(log_alpha)

    for t in range(1, T):
        log_alpha = (
            logsumexp(log_alpha[:, None] + log_Q, axis=0)
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
    n_restarts: int = N_RESTARTS,
    n_iter: int     = N_ITER,
    base_seed: int  = RANDOM_SEED,
) -> tuple[GaussianHMM, float]:
    """Multiple-restart EM; returns (best_model, best_log_lik_per_obs)."""
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
            ll = model.score(X)
            if np.isfinite(ll) and ll > best_ll:
                best_ll    = ll
                best_model = model
        except Exception:
            pass
    return best_model, best_ll


# ---------------------------------------------------------------------------
# Model selection helpers
# ---------------------------------------------------------------------------

def _n_free_params(M: int, d: int) -> int:
    """Free parameters for GaussianHMM with diagonal covariance."""
    # (M-1) initial probs + M(M-1) transition probs + 2·M·d (means + variances)
    return (M - 1) + M * (M - 1) + 2 * M * d


def compute_bic(model: GaussianHMM, X: np.ndarray) -> float:
    """BIC = −2·ℓ̂_total + k·log(n).

    hmmlearn's score() already returns the TOTAL (sum) log-likelihood,
    not per-observation — no further scaling needed.
    """
    T  = len(X)
    k  = _n_free_params(model.n_components, X.shape[1])
    ll = model.score(X)   # total log-likelihood (not per-obs)
    return -2.0 * ll + k * np.log(T)


def ts_cv_mean_loglik(
    X_pre: np.ndarray,
    n_components: int,
    n_splits: int  = N_CV_SPLITS,
    base_seed: int = RANDOM_SEED,
) -> float:
    """
    Forward-chaining time-series CV for a single M.

    For fold i (i = 1 … n_splits):
        train : X_pre[0 : split_i]
        val   : X_pre[split_i : split_{i+1}]

    Returns the mean per-observation held-out log-likelihood.
    Higher is better (opposite of BIC).

    This is a *prospective* evaluation: each training set is strictly in
    the past relative to its validation set, mirroring how the model will
    be used in production.
    """
    T         = len(X_pre)
    min_train = max(200, T // (n_splits + 2))
    lls: list[float] = []

    for fold in range(1, n_splits + 1):
        train_end = int(T * fold / (n_splits + 1))
        val_end   = int(T * (fold + 1) / (n_splits + 1))
        X_train   = X_pre[:train_end]
        X_val     = X_pre[train_end:val_end]
        if len(X_train) < min_train or len(X_val) < 20:
            continue
        m, _ = fit_hmm_best(X_train, n_components, base_seed=base_seed)
        if m is not None:
            lls.append(m.score(X_val) / len(X_val))   # per-obs for fair cross-fold comparison

    return float(np.mean(lls)) if lls else -np.inf


def sort_states_by_risk(
    model: GaussianHMM,
    vix_idx: int,
    oas_idx: int,
) -> tuple[GaussianHMM, np.ndarray]:
    """
    Sort states by ascending composite risk score: mean_VIX_z + mean_HY_OAS_z.

    State 0 → risk-on (low VIX, tight spreads).
    State M-1 → risk-off / stress (high VIX, wide spreads).

    Returns (reordered_model, permutation_order).
    """
    risk  = model.means_[:, vix_idx] + model.means_[:, oas_idx]
    order = np.argsort(risk)
    model.startprob_ = model.startprob_[order]
    model.transmat_  = model.transmat_[np.ix_(order, order)]
    model.means_     = model.means_[order]
    model._covars_   = model._covars_[order]   # compact (M, d) form
    return model, order


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    save: bool    = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, GaussianHMM, int, pd.DataFrame]:
    """
    Full pipeline: M-select, fit macro HMM on pre-sample, filter forward.

    Returns
    -------
    features_df : tidy DataFrame (645 dates × 11 instruments)
    model       : fitted, frozen GaussianHMM
    chosen_M    : selected number of states
    sel_table   : DataFrame with BIC and CV metrics for M=2 and M=3
    """
    macro = pd.read_csv(MACRO_PATH, parse_dates=["Date"])
    ps    = pd.read_csv(PS_PATH,    parse_dates=["date"])

    cutoff         = ps["date"].min()
    meta_dates_set = set(ps["date"])

    # Drop rows with any NaN in the 4 feature columns
    macro_clean = macro[["Date"] + FEATURES].dropna().copy()
    presample   = macro_clean[macro_clean["Date"] < cutoff]
    full        = macro_clean.copy()

    # Standardise on pre-sample statistics only
    scaler = StandardScaler()
    X_pre  = scaler.fit_transform(presample[FEATURES].values)
    X_full = scaler.transform(full[FEATURES].values)

    vix_idx = FEATURES.index("f11_vix_level_z")
    oas_idx = FEATURES.index("f11_hy_oas_z")

    # ------------------------------------------------------------------
    # M selection
    # ------------------------------------------------------------------
    if verbose:
        print("M selection (pre-sample only):")
    sel_rows: list[dict] = []
    models: dict[int, tuple[GaussianHMM, float]] = {}

    for M in [2, 3]:
        m, best_ll = fit_hmm_best(X_pre, M)
        b          = compute_bic(m, X_pre)
        cv         = ts_cv_mean_loglik(X_pre, M)
        models[M]  = (m, best_ll)
        sel_rows.append({
            "M": M,
            "log_lik_total": best_ll,
            "log_lik_per_obs": best_ll / len(X_pre),
            "BIC": b,
            "CV_log_lik_per_obs": cv,
        })
        if verbose:
            print(f"  M={M}: log-lik/obs={best_ll/len(X_pre):.4f}  BIC={b:.1f}  CV_ll/obs={cv:.4f}")

    sel_table = pd.DataFrame(sel_rows).set_index("M")

    bic_favors_2 = sel_table.loc[2, "BIC"] < sel_table.loc[3, "BIC"]
    cv_favors_2  = sel_table.loc[2, "CV_log_lik_per_obs"] > sel_table.loc[3, "CV_log_lik_per_obs"]

    if bic_favors_2 and cv_favors_2:
        chosen_M = 2
        reason   = "BIC and CV both favour M=2 (parsimony)"
    elif not bic_favors_2 and not cv_favors_2:
        chosen_M = 3
        reason   = "BIC and CV both favour M=3"
    else:
        chosen_M = 2
        reason   = "BIC/CV disagree — tie-break to fewer states (parsimony + dwell)"

    if verbose:
        print(f"\n  → Chosen M={chosen_M} ({reason})")

    # ------------------------------------------------------------------
    # Final fit with chosen M
    # ------------------------------------------------------------------
    final_model, final_ll = fit_hmm_best(X_pre, chosen_M)
    final_model, _        = sort_states_by_risk(final_model, vix_idx, oas_idx)

    # Filter forward on full sequence (parameters frozen)
    filtered = filter_forward(final_model, X_full)

    full_dates  = full["Date"].values
    M           = chosen_M
    riskoff_idx = M - 1
    Q           = final_model.transmat_

    records: list[dict] = []
    for idx, d in enumerate(full_dates):
        d_ts = pd.Timestamp(d)
        if d_ts not in meta_dates_set:
            continue
        alpha = filtered[idx]
        row: dict = {"date": d_ts}
        for s in range(M - 1):                 # drop last state (redundant)
            row[f"hmm_macro_p{s}"] = float(alpha[s])
        row["hmm_macro_next_riskoff"] = float(alpha @ Q[:, riskoff_idx])
        row["hmm_macro_entropy"]      = float(
            -np.sum(alpha * np.log(np.clip(alpha, 1e-300, 1.0)))
        )
        records.append(row)

    date_df = pd.DataFrame(records)

    # Broadcast global features to each instrument
    frames = []
    for inst in INSTRUMENTS:
        df_inst = date_df.copy()
        df_inst["instrument"] = inst
        frames.append(df_inst)

    feat_cols = [c for c in date_df.columns if c != "date"]
    out = (
        pd.concat(frames)
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
    out = out[["date", "instrument"] + feat_cols]

    if save:
        out.to_csv(OUT_PATH, index=False)
        if verbose:
            print(f"\nSaved {len(out):,} rows → {OUT_PATH}")

    return out, final_model, chosen_M, sel_table


if __name__ == "__main__":
    run_pipeline()
