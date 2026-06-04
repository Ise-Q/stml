"""F15 conditional risk + F16 concept drift + F17 HMM regimes.

Lifted with attribution from:
* ``src/stml/harry/features/conditional_risk.py``  → F15 path_tortuosity, semi_vol_ratio
* ``src/stml/harry/features/concept_drift.py``     → F16 regime_alignment_score
* ``src/stml/regimes.py`` (Sreeram)                → F17 HMM filtered posteriors
* ``metamodel-apb/src/alken_metamodel/regime.py``  → EWMA HMM (causal, no CV seam)

F15 / F16 are E-class (no fit). F17 + EWMA HMM are TF-class but fit on a
contiguous prefix that ends BEFORE the modelling window, so they remain
fold-safe under purged CV (the fit doesn't move per fold).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stml.experimental.features.catalog import (
    FeatureContext,
    FeatureSpec,
    log_returns,
    register,
)


# ---------------------------------------------------------------------------
# F15 — Conditional risk family.
# ---------------------------------------------------------------------------


def _f15_path_tortuosity_20(ctx: FeatureContext) -> pd.Series:
    """Sum(|r|) / |Sum(r)| over a 20-bar window — Markowitz path-tortuosity proxy.

    Identified by the plan §2.5 as the highest-value individual feature
    (stable AND informative) in the stack.
    """
    r = log_returns(ctx.frame["close"])
    num = r.abs().rolling(20).sum()
    den = r.rolling(20).sum().abs().replace(0, np.nan)
    return num / den


def _f15_realized_semi_vol_ratio_20(ctx: FeatureContext) -> pd.Series:
    """RMS(downside r) / RMS(upside r) over 20 bars.

    > 1 → downside dominates (bearish risk); < 1 → upside dominates (bullish).
    """
    r = log_returns(ctx.frame["close"])
    upside = r.clip(lower=0.0)
    downside = -r.clip(upper=0.0)
    rms_up = np.sqrt((upside ** 2).rolling(20).mean())
    rms_down = np.sqrt((downside ** 2).rolling(20).mean())
    return rms_down / rms_up.replace(0, np.nan)


register(FeatureSpec("f15_path_tortuosity_20", "F15", _f15_path_tortuosity_20, "E", 20, "harry"))
register(FeatureSpec("f15_semi_vol_ratio_20", "F15", _f15_realized_semi_vol_ratio_20, "E", 20, "harry"))


# ---------------------------------------------------------------------------
# F16 — Concept drift score (Harry's harry/features/concept_drift.py).
#
# A rolling logistic discriminator: how distinguishable the *current* feature
# vector is from a baseline ("train era") feature vector. The score is the
# discriminator's probability that the row is "recent" rather than "train era".
#
# Per plan §3.3 we use F16 BOTH as a feature AND as an inverse training weight:
#   sample_weight_drift = 1 / (1 + 0.5 · f16_score)
#
# For the feature side, we compute a simpler proxy here: the standardised
# *change* in the rolling 60-day vector of (r, |r|, abs vol) — measures how
# different recent dynamics are from long-run average.
# ---------------------------------------------------------------------------


def _f16_regime_alignment_score(ctx: FeatureContext) -> pd.Series:
    """Proxy regime-alignment score in [0, 1]; higher = more 'recent-looking'.

    Rolling cosine distance between the standardized (5d, 21d, 63d) vol triple
    at bar t and its 252-day rolling mean. Cosine distance → larger when the
    recent vol profile diverges from the long-run profile → higher score.
    """
    r = log_returns(ctx.frame["close"])
    vol_5 = r.rolling(5).std()
    vol_21 = r.rolling(21).std()
    vol_63 = r.rolling(63).std()
    # Long-run mean vols.
    mean_5 = vol_5.rolling(252, min_periods=126).mean()
    mean_21 = vol_21.rolling(252, min_periods=126).mean()
    mean_63 = vol_63.rolling(252, min_periods=126).mean()
    # L2 distance between (vol_5, vol_21, vol_63) and its long-run mean.
    diff = np.sqrt(
        (vol_5 - mean_5) ** 2 + (vol_21 - mean_21) ** 2 + (vol_63 - mean_63) ** 2
    )
    # Normalise to [0, 1] via rolling rank (252d window).
    return diff.rolling(252, min_periods=126).rank(pct=True)


register(FeatureSpec("f16_regime_alignment_score", "F16", _f16_regime_alignment_score, "E", 252, "harry"))


# ---------------------------------------------------------------------------
# F17 — HMM regime posteriors (Sreeram's regimes.py — causal forward filter).
#
# Per-instrument 3-state Gaussian HMM on (daily log return, 21d annualised vol).
# Fit on data BEFORE the modelling-window start (boundary), then causally
# forward-filtered on the full series. Critical: hmmlearn's predict_proba is
# SMOOTHED (uses future data); we use a hand-rolled forward filter.
# ---------------------------------------------------------------------------


def _fit_and_filter_hmm(
    obs: np.ndarray, n_components: int = 3, fit_n: int | None = None, seed: int = 42
) -> np.ndarray:
    """Fit a 3-state Gaussian HMM on the prefix, then causally forward-filter.

    Parameters
    ----------
    obs : (T, d) observation matrix.
    fit_n : number of leading rows used for training. The remaining rows are
        causally filtered with frozen parameters. None → use the full obs.
    seed : RNG seed.

    Returns
    -------
    filtered : (T, K) posteriors per state per bar (sums to 1 across columns).
    """
    try:
        from hmmlearn import hmm  # lazy import
    except ImportError:
        return np.full((obs.shape[0], n_components), np.nan)

    if obs.shape[0] < 200:
        return np.full((obs.shape[0], n_components), np.nan)

    if fit_n is None:
        fit_n = obs.shape[0]

    train = obs[:fit_n]
    train = train[np.isfinite(train).all(axis=1)]
    if train.shape[0] < 200:
        return np.full((obs.shape[0], n_components), np.nan)

    rng = np.random.default_rng(seed)
    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=50,
        tol=1e-3,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    try:
        model.fit(train)
    except Exception:
        return np.full((obs.shape[0], n_components), np.nan)

    # Reorder states by mean of dim 1 (vol) — ascending.
    order = np.argsort(model.means_[:, 1])
    means = model.means_[order]
    covars = model._covars_[order] if hasattr(model, "_covars_") else None
    transmat = model.transmat_[np.ix_(order, order)]
    startprob = model.startprob_[order]

    # Causal forward filter in log-domain.
    log_alpha = np.full((obs.shape[0], n_components), -np.inf)
    log_pi = np.log(np.clip(startprob, 1e-12, None))
    log_A = np.log(np.clip(transmat, 1e-12, None))

    def _emission_log_prob(x: np.ndarray, mean: np.ndarray, covar: np.ndarray) -> float:
        """Multivariate Gaussian log-pdf."""
        d = len(mean)
        diff = x - mean
        try:
            sign, logdet = np.linalg.slogdet(covar)
            if sign <= 0 or not np.isfinite(logdet):
                return -1e9
            inv = np.linalg.pinv(covar)
            quad = diff @ inv @ diff
        except Exception:
            return -1e9
        return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)

    for t in range(obs.shape[0]):
        x = obs[t]
        if not np.isfinite(x).all():
            # Carry previous posterior on missing observation.
            if t > 0:
                log_alpha[t] = log_alpha[t - 1]
            continue
        log_emis = np.array([
            _emission_log_prob(x, means[k], covars[k] if covars is not None else np.eye(len(x)))
            for k in range(n_components)
        ])
        if t == 0:
            log_alpha[t] = log_pi + log_emis
        else:
            from scipy.special import logsumexp
            for k in range(n_components):
                log_alpha[t, k] = log_emis[k] + logsumexp(log_alpha[t - 1] + log_A[:, k])
        # Normalise (subtract logsumexp).
        from scipy.special import logsumexp
        log_alpha[t] -= logsumexp(log_alpha[t])

    return np.exp(log_alpha)


def _f17_hmm_filtered_state_lo(ctx: FeatureContext) -> pd.Series:
    """Filtered posterior P(state 0 = low-vol regime)."""
    r = log_returns(ctx.frame["close"])
    vol = r.rolling(21).std()
    obs = np.column_stack([r.values, vol.values])
    # Fit on data BEFORE the signal window starts (~ 2020-01-03).
    boundary = pd.Timestamp("2020-01-01")
    fit_n = int((ctx.frame.index < boundary).sum())
    if fit_n < 200:
        return pd.Series(np.nan, index=ctx.frame.index)
    posteriors = _fit_and_filter_hmm(obs, n_components=3, fit_n=fit_n, seed=42)
    return pd.Series(posteriors[:, 0], index=ctx.frame.index)


def _f17_hmm_filtered_state_mid(ctx: FeatureContext) -> pd.Series:
    r = log_returns(ctx.frame["close"])
    vol = r.rolling(21).std()
    obs = np.column_stack([r.values, vol.values])
    boundary = pd.Timestamp("2020-01-01")
    fit_n = int((ctx.frame.index < boundary).sum())
    if fit_n < 200:
        return pd.Series(np.nan, index=ctx.frame.index)
    posteriors = _fit_and_filter_hmm(obs, n_components=3, fit_n=fit_n, seed=42)
    return pd.Series(posteriors[:, 1], index=ctx.frame.index)


def _f17_hmm_filtered_state_hi(ctx: FeatureContext) -> pd.Series:
    r = log_returns(ctx.frame["close"])
    vol = r.rolling(21).std()
    obs = np.column_stack([r.values, vol.values])
    boundary = pd.Timestamp("2020-01-01")
    fit_n = int((ctx.frame.index < boundary).sum())
    if fit_n < 200:
        return pd.Series(np.nan, index=ctx.frame.index)
    posteriors = _fit_and_filter_hmm(obs, n_components=3, fit_n=fit_n, seed=42)
    return pd.Series(posteriors[:, 2], index=ctx.frame.index)


register(FeatureSpec("f17_hmm_state_lo", "F17", _f17_hmm_filtered_state_lo, "TF", 200, "sreeram"))
register(FeatureSpec("f17_hmm_state_mid", "F17", _f17_hmm_filtered_state_mid, "TF", 200, "sreeram"))
register(FeatureSpec("f17_hmm_state_hi", "F17", _f17_hmm_filtered_state_hi, "TF", 200, "sreeram"))


# ---------------------------------------------------------------------------
# EWMA HMM — alken's regime.py online filter (no CV seam artefact).
#
# A 2-state Gaussian HMM where emission means/variances are recursively updated
# via EWMA of responsibility-weighted sufficient statistics; the transition
# matrix is FIXED (persistent prior). Every parameter at t depends only on
# data t' ≤ t — strictly causal.
# ---------------------------------------------------------------------------


def _ewma_hmm_filter(returns: np.ndarray, *, lam: float = 0.94, persistence: float = 0.97,
                      warmup: int = 60, hi_init: float = 2.5, var_floor: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Online EWMA 2-state HMM filter; returns (p_hi, switch_prob)."""
    n = len(returns)
    p_hi = np.full(n, np.nan)
    switch = np.full(n, np.nan)

    # Warmup: bootstrap emission params from the first ``warmup`` bars.
    warm = returns[:warmup]
    warm = warm[np.isfinite(warm)]
    if len(warm) < 10:
        return p_hi, switch
    sigma_lo = float(np.std(warm))
    sigma_hi = hi_init * sigma_lo
    mu_lo = float(np.mean(warm))
    mu_hi = mu_lo

    # Initial belief.
    p_hi_t = 0.5
    a = persistence
    A = np.array([[a, 1 - a], [1 - a, a]])  # symmetric persistent

    for t in range(n):
        r = returns[t]
        if not np.isfinite(r):
            continue
        # Emission likelihoods (Gaussian).
        var_lo = max(sigma_lo ** 2, var_floor)
        var_hi = max(sigma_hi ** 2, var_floor)
        lik_lo = np.exp(-0.5 * (r - mu_lo) ** 2 / var_lo) / np.sqrt(2 * np.pi * var_lo)
        lik_hi = np.exp(-0.5 * (r - mu_hi) ** 2 / var_hi) / np.sqrt(2 * np.pi * var_hi)
        # Predict step.
        p_lo_pred = (1 - p_hi_t) * A[0, 0] + p_hi_t * A[1, 0]
        p_hi_pred = (1 - p_hi_t) * A[0, 1] + p_hi_t * A[1, 1]
        # Update step (Bayes).
        num = p_hi_pred * lik_hi
        den = p_lo_pred * lik_lo + num
        if den <= 0 or not np.isfinite(den):
            continue
        new_p_hi = num / den
        switch[t] = abs(new_p_hi - p_hi_t)
        p_hi[t] = new_p_hi
        # EWMA update of emission params (responsibility-weighted).
        w_lo = 1 - new_p_hi
        w_hi = new_p_hi
        # Welford-like EWMA on the responsibility-weighted ret.
        sigma_lo = np.sqrt((1 - lam) * w_lo * (r - mu_lo) ** 2 + lam * sigma_lo ** 2)
        sigma_hi = np.sqrt((1 - lam) * w_hi * (r - mu_hi) ** 2 + lam * sigma_hi ** 2)
        mu_lo = lam * mu_lo + (1 - lam) * w_lo * r
        mu_hi = lam * mu_hi + (1 - lam) * w_hi * r
        p_hi_t = new_p_hi
    return p_hi, switch


def _ewma_hmm_prob_highvol(ctx: FeatureContext) -> pd.Series:
    r = log_returns(ctx.frame["close"]).values
    p_hi, _ = _ewma_hmm_filter(r)
    return pd.Series(p_hi, index=ctx.frame.index)


def _ewma_hmm_switch_prob(ctx: FeatureContext) -> pd.Series:
    r = log_returns(ctx.frame["close"]).values
    _, switch = _ewma_hmm_filter(r)
    return pd.Series(switch, index=ctx.frame.index)


register(FeatureSpec("ewma_hmm_prob_highvol", "EWMA_HMM", _ewma_hmm_prob_highvol, "E", 60, "alken"))
register(FeatureSpec("ewma_hmm_switch_prob", "EWMA_HMM", _ewma_hmm_switch_prob, "E", 60, "alken"))
