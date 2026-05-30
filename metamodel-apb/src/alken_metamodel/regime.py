"""Regime features for the meta-labelling metamodel (Stage 2, commitment #8 / nlr-cw §4).

Two blocks, both per-instrument and causal:

1. **Online EWMA 2-state Gaussian HMM** (net-new; Nystrup-Madsen-Lindström 2017,
   "persistent states + exponential forgetting"). A forward-filtered HMM on daily log
   returns whose emission means/variances are re-estimated *online* with a forgetting
   factor ``lam`` (EWMA of responsibility-weighted sufficient statistics). Because every
   parameter at ``t`` is a recursion over observations ``<= t``, the feature is causal /
   right-edge truncation-invariant by construction — no batch fit, no fit/transform split,
   and therefore no per-fold CPCV seam artifact (a union of non-contiguous train groups
   would fabricate fake 1-step transitions in a batch HMM). The transition matrix is a
   fixed *persistent* prior (the "penalising jumps" half of Nystrup); the means/variances
   are the time-varying part. This is the literature-faithful answer to #8 that stml's
   *static* HMM cannot provide.

2. **stml static blocks** (supplementary): F3 GMM + Markov-switching (`fit_regime` /
   `transform_regime`) and F17 3-state causal-filtered Gaussian HMM (`fit_hmm` /
   `transform_hmm`), fit on a CONTIGUOUS prefix (``fit_end``) and causally transformed on
   the history-inclusive series. Reused verbatim from stml (signatures verified, not
   assumed); contiguity avoids the seam hazard above.

Determinism: the EWMA HMM uses no RNG (deterministic method-of-moments warm-up seed); the
stml blocks take ``seed`` for their GMM/HMM EM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp

# EWMA-HMM hyperparameters (nlr-cw §4 defaults; trivially overridable).
EWMA_LAMBDA: float = 0.94      # forgetting factor for the online emission update
PERSISTENCE: float = 0.97      # fixed self-transition prior (persistent regimes)
HMM_WARMUP: int = 60           # bars used to seed the emissions; emitted as NaN
VAR_FLOOR: float = 1e-12       # variance floor (no div-by-zero in the Gaussian density)
HI_INIT_SCALE: float = 2.5     # high-vol state initial sigma = HI_INIT_SCALE * warm-up sigma

_LOG_2PI: float = float(np.log(2.0 * np.pi))

EWMA_COLUMNS = [
    "ewma_hmm_prob_highvol",
    "ewma_hmm_state",
    "ewma_hmm_var_hi",
    "ewma_hmm_var_lo",
    "ewma_hmm_switch_prob",
]


def _gaussian_loglik(r: float, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Log N(r; mu_k, var_k) for each state k (vectorised over 2 states)."""
    return -0.5 * (_LOG_2PI + np.log(var)) - 0.5 * (r - mu) ** 2 / var


def ewma_hmm_features(
    close: pd.Series,
    *,
    lam: float = EWMA_LAMBDA,
    persistence: float = PERSISTENCE,
    warmup: int = HMM_WARMUP,
    var_floor: float = VAR_FLOOR,
    hi_init_scale: float = HI_INIT_SCALE,
) -> pd.DataFrame:
    """Online EWMA 2-state Gaussian-HMM regime features on one instrument's close.

    Returns a ``close``-indexed float frame with columns ``EWMA_COLUMNS``: filtered
    high-vol probability, the hard high-vol state (prob >= 0.5), the time-varying high/low
    state variances, and the one-step change in high-vol probability. The first ``warmup``
    rows are structural NaN (no fabricated warm-up values).
    """
    close = close.sort_index()
    idx = close.index
    n = len(idx)
    r = np.log(close.to_numpy(dtype=float))
    r = np.concatenate([[np.nan], np.diff(r)])  # r[t] = log(close_t / close_{t-1})

    prob = np.full(n, np.nan)
    var_hi = np.full(n, np.nan)
    var_lo = np.full(n, np.nan)

    log_a = np.array([[np.log(persistence), np.log1p(-persistence)],
                      [np.log1p(-persistence), np.log(persistence)]])

    seed = r[1 : warmup + 1]
    seed = seed[np.isfinite(seed)]
    if seed.size < 2:  # not enough history to seed — all NaN (structural)
        return pd.DataFrame(
            {c: np.full(n, np.nan) for c in EWMA_COLUMNS}, index=idx
        )
    m = float(seed.mean())
    s = float(seed.std())
    v_lo0 = max(s * s, var_floor)
    v_hi0 = max((hi_init_scale * s) ** 2, var_floor)

    mu = np.array([m, m])
    var = np.array([v_lo0, v_hi0])
    # EWMA sufficient stats seeded so (mu, var) are exactly recovered (W=1).
    w = np.array([1.0, 1.0])
    s1 = mu * w
    s2 = (var + mu**2) * w
    log_alpha = np.log(np.array([0.5, 0.5]))

    for t in range(warmup + 1, n):
        r_t = r[t]
        if not np.isfinite(r_t):  # closed-venue gap: carry state, emit NaN
            continue
        log_b = _gaussian_loglik(r_t, mu, var)
        log_pred = logsumexp(log_alpha[:, None] + log_a, axis=0)  # predict
        log_post = log_pred + log_b
        log_alpha = log_post - logsumexp(log_post)  # filtered posterior, normalised
        gamma = np.exp(log_alpha)

        # EWMA update of the responsibility-weighted emission statistics.
        w = lam * w + (1.0 - lam) * gamma
        s1 = lam * s1 + (1.0 - lam) * gamma * r_t
        s2 = lam * s2 + (1.0 - lam) * gamma * r_t * r_t
        mu = s1 / w
        var = np.maximum(s2 / w - mu**2, var_floor)

        hi = int(np.argmax(var))  # high-vol = larger-variance state (re-labelled each t)
        prob[t] = gamma[hi]
        var_hi[t] = var[hi]
        var_lo[t] = var[1 - hi]

    out = pd.DataFrame(index=idx)
    out["ewma_hmm_prob_highvol"] = prob
    out["ewma_hmm_state"] = np.where(np.isnan(prob), np.nan, (prob >= 0.5).astype(float))
    out["ewma_hmm_var_hi"] = var_hi
    out["ewma_hmm_var_lo"] = var_lo
    out["ewma_hmm_switch_prob"] = out["ewma_hmm_prob_highvol"].diff().abs()
    return out


# --- stml static regime blocks (supplementary reuse) -----------------------

def _build_ret_vol(ohlcv_inst: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Date-indexed ``(ret, vol)`` for one instrument (mirrors stml pipeline.py:146)."""
    from stml.na_checks import native_returns, rolling_vol

    rets = native_returns(ohlcv_inst, kind="log")
    ret = rets.set_index("date")["ret"].sort_index()
    instrument = ohlcv_inst["instrument"].iloc[0]
    vol = rolling_vol(rets, instrument, window=window)
    return pd.DataFrame({"ret": ret, "vol": vol}).dropna().sort_index()


def static_regime_features(ohlcv_inst: pd.DataFrame, *, fit_end, seed: int = 0) -> pd.DataFrame:
    """F3 (GMM + Markov) and F17 (3-state HMM) regime features via reused stml functions.

    Fit on the CONTIGUOUS prefix ``(ret, vol).index <= fit_end`` (no non-contiguous seams),
    then causally transform on the full history-inclusive series and subset is left to the
    caller. Returns the stml ``f3_*`` and ``f17_*`` columns (structural NaN if a fit fails).
    """
    from stml.metamodel.regime_features import fit_regime, transform_regime
    from stml.metamodel.regime_features_hmm import fit_hmm, transform_hmm

    instrument = str(ohlcv_inst["instrument"].iloc[0])
    ret_vol = _build_ret_vol(ohlcv_inst)
    train = ret_vol[ret_vol.index <= fit_end]

    regime_bundle = fit_regime(train, seed=seed, instrument=instrument)
    hmm_bundle = fit_hmm(train, seed=seed, instrument=instrument)
    f3 = transform_regime(regime_bundle, ret_vol)
    f17 = transform_hmm(hmm_bundle, ret_vol)
    return pd.concat([f3, f17], axis=1)


def assemble_regime_features(
    ohlcv_inst: pd.DataFrame,
    *,
    fit_end,
    seed: int = 0,
    lam: float = EWMA_LAMBDA,
) -> pd.DataFrame:
    """Both regime blocks on a shared per-instrument ``DatetimeIndex``.

    The net-new EWMA HMM (causal/fit-free) plus the stml static blocks (fit on the
    ``fit_end`` prefix). The EWMA frame is computed on the instrument's full close history;
    both blocks are aligned on the price calendar.
    """
    close = ohlcv_inst.set_index("date")["close"].sort_index()
    close.index = pd.DatetimeIndex(close.index)
    ewma = ewma_hmm_features(close, lam=lam)
    static = static_regime_features(ohlcv_inst, fit_end=fit_end, seed=seed)
    return pd.concat([ewma, static.reindex(ewma.index)], axis=1)
