"""
characterize.py
===============
Central characterization module (C1) for the primary-signal reverse-engineering
study. It answers SIX questions about the *nature* of the released
``{-1, 0, +1}`` signal, each with NUMBERS (this module is diagnostic /
exploratory, so every function returns a ``dict`` of numeric results -- floats
and ints -- never prose).

The six questions
-----------------
* **Q1 alpha type** -- is the signal momentum or mean-reversion?
  :func:`alpha_type` correlates the signal with trailing returns and with the
  distance from a moving average, and measures how often a nonzero signal
  coincides with a Donchian breakout.
* **Q2 lead/lag** -- which return does the signal predict, and at what horizon?
  :func:`lead_lag` correlates ``s_t`` with ``r_{t+h}`` for ``h in -5..+5`` and
  reports the ``best_lag``. *This empirically confirms the holding convention*
  (``best_lag == +1`` => next-day execution) and is the single most important
  output for the checkpoint.
* **Q3 regime** -- does the signal avoid high-volatility regimes?
  :func:`regime` fits a statsmodels Markov-switching model and an sklearn
  Gaussian mixture, then compares participation in the low-vol vs high-vol
  regime.
* **Q4 cross-asset** -- how correlated are the 11 signals, and do instruments
  cluster by behavior? :func:`cross_asset` reports the mean absolute
  off-diagonal signal correlation and a fingerprint clustering.
* **Q5 drift** -- do the signal's base rates change over time?
  :func:`drift` reports per-split (train/val/test) participation, long-bias and
  class fractions so a train->val->test trend is visible.
* **Q6 model-family fingerprint** -- *advisory, low-confidence.*
  :func:`model_family_fingerprint` fits shallow surrogate classifiers
  (tree / linear / forest) to GUESS the generating family. ``'inconclusive'``
  is an acceptable result; this gates nothing.

Robustness contract
--------------------
Several of the 11 instruments are degenerate: ``ng1s`` is never ``+1``,
``gc1s`` / ``ho1s`` are ~80% flat, ``cl1s`` is never ``-1`` in some windows.
**No function here may raise on any of the 11.** All model fits
(Markov-switching, GMM, surrogate classifiers) are wrapped in ``try/except``
and fall back to finite values or ``{'status': 'inconclusive', ...}``.

Reuse (see ``.omc/scratch/CONTRACT.md``)
----------------------------------------
Returns come from :func:`stml.na_checks.native_returns` (per-instrument dense
series; holiday-spanning moves are correct, never a fabricated zero); volatility
from :func:`stml.na_checks.rolling_vol`; splits and effective-sample primitives
from :mod:`stml.replication.splits`. Trailing features are computed on each
instrument's FULL OHLCV history (1990->) and then aligned to the 2020->2022
signal dates, so a 20-day lookback at the start of the signal era is real
history, not a truncated window.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from stml.na_checks import native_returns, rolling_vol
from stml.replication.splits import chronological_split

__all__ = [
    "alpha_type",
    "lead_lag",
    "regime",
    "cross_asset",
    "drift",
    "model_family_fingerprint",
    "characterize_instrument",
    "characterize_all",
]

# Canonical instrument list (matches io.INSTRUMENTS / align._INSTRUMENTS order).
_INSTRUMENTS: list[str] = [
    "es1s",
    "nq1s",
    "fesx1s",
    "cl1s",
    "ho1s",
    "rb1s",
    "ng1s",
    "gc1s",
    "si1s",
    "hg1s",
    "pl1s",
]


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #
def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    """Pearson correlation that returns ``nan`` (never raises) on degenerate
    input -- a constant series, fewer than two aligned points, or all-NaN.

    Degenerate instruments routinely produce constant signal slices (e.g. an
    all-flat window), for which a correlation is mathematically undefined; the
    caller treats ``nan`` as 'no information', not as an error.
    """
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 2:
        return float("nan")
    x = pair.iloc[:, 0].to_numpy(dtype=float)
    y = pair.iloc[:, 1].to_numpy(dtype=float)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = np.corrcoef(x, y)[0, 1]
    return float(c) if np.isfinite(c) else float("nan")


def _signal_series(signals: pd.DataFrame, instrument: str) -> pd.Series:
    """Date-indexed signal series for one instrument, sorted ascending."""
    return (
        signals[["date", instrument]]
        .set_index("date")[instrument]
        .sort_index()
        .astype(float)
    )


def _instrument_ohlcv(ohlcv: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """Date-indexed OHLCV slice for one instrument on its FULL dense history."""
    return (
        ohlcv[ohlcv["instrument"] == instrument]
        .drop_duplicates("date")
        .set_index("date")
        .sort_index()
    )


def _native_log_returns(ohlcv: pd.DataFrame, instrument: str) -> pd.Series:
    """Date-indexed native log returns for one instrument (reuses na_checks).

    Computed on the instrument's own dense series, so a holiday-spanning move is
    the correct multi-day return rather than a fabricated zero.
    """
    inst = ohlcv[ohlcv["instrument"] == instrument].drop_duplicates("date")
    rets = native_returns(inst, kind="log")
    return rets.set_index("date")["ret"].sort_index()


def _participation_rate(sig: pd.Series) -> float:
    """Fraction of nonzero (active) signal days."""
    if len(sig) == 0:
        return float("nan")
    return float((sig != 0).mean())


def _long_bias(sig: pd.Series) -> float:
    """Mean signal value in ``[-1, 1]`` (>0 long-biased, <0 short-biased)."""
    if len(sig) == 0:
        return float("nan")
    return float(sig.mean())


def _persistence(sig: pd.Series) -> float:
    """``P(s_t == s_{t-1})`` -- how sticky the signal is from day to day."""
    if len(sig) < 2:
        return float("nan")
    a = sig.to_numpy()
    return float(np.mean(a[1:] == a[:-1]))


def _class_fractions(sig: pd.Series) -> dict[str, float]:
    """Fractions of ``-1`` / ``0`` / ``+1`` in the series (keys ``frac_-1`` ...)."""
    n = len(sig)
    if n == 0:
        return {"frac_neg1": float("nan"), "frac_0": float("nan"), "frac_pos1": float("nan")}
    return {
        "frac_neg1": float((sig == -1).sum() / n),
        "frac_0": float((sig == 0).sum() / n),
        "frac_pos1": float((sig == 1).sum() / n),
    }


# --------------------------------------------------------------------------- #
# Q1. Alpha type: momentum vs mean-reversion                                  #
# --------------------------------------------------------------------------- #
def alpha_type(
    instrument: str,
    signals: pd.DataFrame,
    ohlcv: pd.DataFrame,
    trail_windows: tuple[int, ...] = (1, 5, 10, 20),
    ma_windows: tuple[int, ...] = (10, 20, 50),
    donchian_window: int = 20,
) -> dict:
    """Q1 -- is the signal momentum or mean-reversion?

    Three complementary numeric diagnostics, all measured on nonzero signal days
    only (a flat day expresses no directional view):

    1. ``trail_corr_k`` -- ``corr(s_t, trailing k-day return ending at t-1)`` for
       ``k`` in ``trail_windows``. The trailing return is built from closes up to
       ``t-1`` (``log C_{t-1} - log C_{t-1-k}``) so it uses only information
       available when the signal is formed. A *positive* correlation means the
       signal goes with the recent move (momentum); *negative* means it leans
       against it (mean-reversion).
    2. ``ma_sign_agreement_n`` -- fraction of nonzero days on which
       ``sign(s_t) == sign(close_{t-1} - SMA_n)`` for ``n`` in ``ma_windows``.
       Near 1 => trades in the direction of the distance-from-MA (momentum);
       near 0 => fades it (mean-reversion); ~0.5 => unrelated.
    3. ``breakout_coincidence`` -- fraction of nonzero days that coincide with an
       ``donchian_window``-day Donchian channel break (close above the prior
       rolling high or below the prior rolling low), and the directional version
       ``breakout_coincidence_directional`` (the break is in the SAME direction
       as the signal). High directional coincidence is a momentum/breakout
       fingerprint.

    A convenience ``momentum_score`` (mean of the trailing correlations, sign
    convention: positive => momentum) and a categorical ``alpha_label``
    (``'momentum'`` / ``'mean_reversion'`` / ``'neutral'``) summarize the verdict.

    Robust to degenerate instruments: a constant signal slice yields ``nan``
    correlations and the function still returns a fully-populated dict.
    """
    sig = _signal_series(signals, instrument)
    oi = _instrument_ohlcv(ohlcv, instrument)
    out: dict = {"instrument": instrument, "n_signal_days": int(len(sig))}

    if oi.empty or "close" not in oi.columns:
        out["status"] = "no_ohlcv"
        return out

    log_close = np.log(oi["close"].astype(float))
    nonzero = sig[sig != 0]
    out["n_nonzero"] = int(len(nonzero))

    # --- 1. Trailing-return correlations -------------------------------------
    trail_corrs: dict[str, float] = {}
    for k in trail_windows:
        trailing = log_close.shift(1) - log_close.shift(1 + k)
        trail_corrs[f"trail_corr_{k}"] = _safe_corr(nonzero, trailing.reindex(nonzero.index))
    out.update(trail_corrs)
    finite_trail = [v for v in trail_corrs.values() if np.isfinite(v)]
    momentum_score = float(np.mean(finite_trail)) if finite_trail else float("nan")
    out["momentum_score"] = momentum_score

    # --- 2. Distance-from-MA sign agreement ----------------------------------
    sign_sig = np.sign(nonzero)
    for n in ma_windows:
        sma = oi["close"].astype(float).rolling(n).mean()
        dist = (oi["close"].astype(float).shift(1) - sma.shift(1)).reindex(nonzero.index)
        valid = dist.notna() & (dist != 0)
        if valid.sum() == 0:
            out[f"ma_sign_agreement_{n}"] = float("nan")
        else:
            agree = (np.sign(dist[valid]) == sign_sig[valid]).mean()
            out[f"ma_sign_agreement_{n}"] = float(agree)

    # --- 3. Donchian breakout coincidence ------------------------------------
    high = oi["high"].astype(float) if "high" in oi.columns else oi["close"].astype(float)
    low = oi["low"].astype(float) if "low" in oi.columns else oi["close"].astype(float)
    close = oi["close"].astype(float)
    prior_high = high.rolling(donchian_window).max().shift(1)
    prior_low = low.rolling(donchian_window).min().shift(1)
    up_break = (close > prior_high).reindex(nonzero.index, fill_value=False)
    down_break = (close < prior_low).reindex(nonzero.index, fill_value=False)
    any_break = up_break | down_break
    if len(nonzero) == 0:
        out["breakout_coincidence"] = float("nan")
        out["breakout_coincidence_directional"] = float("nan")
    else:
        out["breakout_coincidence"] = float(any_break.mean())
        directional = (up_break & (sign_sig > 0)) | (down_break & (sign_sig < 0))
        out["breakout_coincidence_directional"] = float(directional.mean())

    # --- Verdict -------------------------------------------------------------
    if not np.isfinite(momentum_score):
        out["alpha_label"] = "neutral"
    elif momentum_score > 0.03:
        out["alpha_label"] = "momentum"
    elif momentum_score < -0.03:
        out["alpha_label"] = "mean_reversion"
    else:
        out["alpha_label"] = "neutral"
    return out


# --------------------------------------------------------------------------- #
# Q2. Lead/lag: which return does the signal predict? (holding convention)    #
# --------------------------------------------------------------------------- #
def lead_lag(
    instrument: str,
    signals: pd.DataFrame,
    ohlcv: pd.DataFrame,
    max_lag: int = 5,
) -> dict:
    """Q2 -- the lead/lag profile that EMPIRICALLY CONFIRMS the holding convention.

    Correlates the full signed signal ``s_t in {-1, 0, +1}`` with ``r_{t+h}``
    (the instrument's native log return ``h`` days ahead) for every ``h`` in
    ``-max_lag .. +max_lag``. The zeros are kept on purpose: they carry the
    "no position" information and add the variance that lets a single-direction
    instrument (``ng1s``, only ``0``/``-1``) still produce a defined correlation
    rather than the ``nan`` a nonzero-only slice would give. The horizon at which
    ``|corr|`` is largest is ``best_lag``:

    * ``best_lag == +1`` => the signal best predicts *tomorrow's* return =>
      ``next_day`` execution confirmed (the contract's default convention).
    * ``best_lag == 0``  => concurrent / ``same_day``.
    * ``best_lag < 0``   => the signal lags the return (a reaction, not a
      prediction) -- a red flag worth surfacing.

    Returns
    -------
    dict with
        ``lag_profile``   : ``{h: corr(s_t, r_{t+h})}`` for all ``h`` (floats).
        ``best_lag``      : the ``int`` ``h`` maximizing ``|corr|`` (0 if every
                            correlation is ``nan`` -- the degenerate fallback).
        ``best_corr``     : the signed correlation at ``best_lag``.
        ``corr_at_lag1``  : ``corr(s_t, r_{t+1})`` -- the next-day number itself.
        ``corr_at_lag0``  : ``corr(s_t, r_t)`` -- the same-day number.
        ``holding_convention`` : ``'next_day'`` / ``'same_day'`` / ``'lagging'``
                            / ``'forward'`` implied by ``best_lag``.

    Degenerate-safe: a constant signal slice makes every correlation ``nan``;
    ``best_lag`` then falls back to ``0`` and ``holding_convention`` to
    ``'inconclusive'``. Never raises.
    """
    sig = _signal_series(signals, instrument)
    ret = _native_log_returns(ohlcv, instrument)
    nonzero = sig[sig != 0]

    lag_profile: dict[int, float] = {}
    for h in range(-max_lag, max_lag + 1):
        shifted = ret.shift(-h)  # day t carries r_{t+h}
        lag_profile[h] = _safe_corr(sig, shifted.reindex(sig.index))

    finite = {h: c for h, c in lag_profile.items() if np.isfinite(c)}
    if not finite:
        best_lag = 0
        best_corr = float("nan")
        convention = "inconclusive"
    else:
        best_lag = int(max(finite, key=lambda h: abs(finite[h])))
        best_corr = float(finite[best_lag])
        if best_lag == 1:
            convention = "next_day"
        elif best_lag == 0:
            convention = "same_day"
        elif best_lag < 0:
            convention = "lagging"
        else:
            convention = "forward"

    return {
        "instrument": instrument,
        "n_nonzero": int(len(nonzero)),
        "lag_profile": {int(h): float(c) for h, c in lag_profile.items()},
        "best_lag": best_lag,
        "best_corr": best_corr,
        "corr_at_lag1": float(lag_profile.get(1, float("nan"))),
        "corr_at_lag0": float(lag_profile.get(0, float("nan"))),
        "holding_convention": convention,
    }


# --------------------------------------------------------------------------- #
# Q3. Regime: does the signal avoid high volatility?                          #
# --------------------------------------------------------------------------- #
def regime(
    instrument: str,
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    vol_window: int = 20,
    n_gmm_components: int = 2,
    markov_maxiter: int = 50,
) -> dict:
    """Q3 -- does the signal participate less in high-volatility regimes?

    Two regime models are fit on the *signal-era* returns, and the
    participation rate (fraction of nonzero signal days) is compared across the
    low- and high-volatility regimes each model implies:

    * **Markov-switching** (``statsmodels`` ``MarkovRegression``, 2 regimes,
      switching variance) on the return series. The regime with the larger fitted
      residual variance is labelled high-vol; ``participation_*_markov`` reports
      participation under each smoothed-probable regime (assigned by argmax
      smoothed probability).
    * **Gaussian mixture** (``sklearn`` ``GaussianMixture``,
      ``n_gmm_components``) on ``(return, rolling-vol)`` features. The component
      with the larger mean rolling-vol is high-vol;
      ``participation_low_vol`` / ``participation_high_vol`` are the headline
      numbers and answer the contract's "AVOIDS HIGH-VOL" question directly:
      ``participation_low_vol > participation_high_vol`` means the signal is more
      active when calm.

    A model-free ``participation_*_median_vol`` split (low vs high by the median
    of the rolling vol) is always reported as a robust fallback, plus
    ``avoids_high_vol`` = ``participation_low_vol > participation_high_vol``.

    Both model fits are wrapped in ``try/except``; on failure the corresponding
    keys carry ``nan`` and ``status`` becomes ``'inconclusive'`` -- the function
    never raises and Markov non-convergence (capped at ``markov_maxiter``
    iterations, ``ConvergenceWarning`` suppressed) is treated as a finite
    fallback, not an error.
    """
    sig = _signal_series(signals, instrument)
    ret_full = _native_log_returns(ohlcv, instrument)
    vol_full = rolling_vol(
        native_returns(ohlcv[ohlcv["instrument"] == instrument], kind="log"),
        instrument,
        window=vol_window,
    )

    # Restrict to the signal era and align return / vol / signal on common dates.
    feat = pd.DataFrame(
        {
            "signal": sig,
            "ret": ret_full.reindex(sig.index),
            "vol": vol_full.reindex(sig.index),
        }
    ).dropna()

    out: dict = {
        "instrument": instrument,
        "n_obs": int(len(feat)),
        "status": "ok",
    }
    if len(feat) < 30:
        out["status"] = "inconclusive"
        for key in (
            "participation_low_vol",
            "participation_high_vol",
            "participation_low_vol_median",
            "participation_high_vol_median",
            "participation_low_vol_markov",
            "participation_high_vol_markov",
            "avoids_high_vol",
        ):
            out[key] = float("nan")
        return out

    active = (feat["signal"] != 0).astype(float)

    # --- Model-free median-vol split (always available) ----------------------
    med = feat["vol"].median()
    low_mask = feat["vol"] <= med
    high_mask = feat["vol"] > med
    part_low_med = float(active[low_mask].mean()) if low_mask.any() else float("nan")
    part_high_med = float(active[high_mask].mean()) if high_mask.any() else float("nan")
    out["participation_low_vol_median"] = part_low_med
    out["participation_high_vol_median"] = part_high_med
    out["median_vol"] = float(med)

    # --- GMM on (return, rolling-vol) ----------------------------------------
    part_low_gmm = float("nan")
    part_high_gmm = float("nan")
    try:
        from sklearn.mixture import GaussianMixture

        X = feat[["ret", "vol"]].to_numpy(dtype=float)
        # Standardize so the two features contribute comparably.
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        Xs = (X - mu) / sd
        n_comp = min(n_gmm_components, max(2, len(feat) // 50))
        gmm = GaussianMixture(
            n_components=n_comp, covariance_type="full", random_state=0, max_iter=200
        )
        labels = gmm.fit_predict(Xs)
        # The component with the highest mean RAW vol is the high-vol regime.
        comp_vol = {c: feat["vol"].to_numpy()[labels == c].mean() for c in np.unique(labels)}
        high_comp = max(comp_vol, key=comp_vol.get)
        low_comp = min(comp_vol, key=comp_vol.get)
        a = active.to_numpy()
        part_high_gmm = float(a[labels == high_comp].mean()) if (labels == high_comp).any() else float("nan")
        part_low_gmm = float(a[labels == low_comp].mean()) if (labels == low_comp).any() else float("nan")
        out["gmm_n_components"] = int(n_comp)
    except Exception as exc:  # noqa: BLE001 - model fit must never break the run
        out["status"] = "inconclusive"
        out["gmm_error"] = type(exc).__name__
    out["participation_low_vol"] = part_low_gmm
    out["participation_high_vol"] = part_high_gmm

    # --- Markov-switching on returns -----------------------------------------
    part_low_mk = float("nan")
    part_high_mk = float("nan")
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ConvergenceWarning / RuntimeWarning
            r = feat["ret"].to_numpy(dtype=float)
            mod = MarkovRegression(r, k_regimes=2, trend="c", switching_variance=True)
            res = mod.fit(maxiter=markov_maxiter, disp=False)
            smoothed = np.asarray(res.smoothed_marginal_probabilities)
            regime_assign = smoothed.argmax(axis=1)
            # Higher fitted variance => high-vol regime.
            sigma2 = np.asarray(res.params)[-2:]  # the two variance params
            high_reg = int(np.argmax(sigma2))
            low_reg = int(np.argmin(sigma2))
            a = active.to_numpy()
            part_high_mk = float(a[regime_assign == high_reg].mean()) if (regime_assign == high_reg).any() else float("nan")
            part_low_mk = float(a[regime_assign == low_reg].mean()) if (regime_assign == low_reg).any() else float("nan")
    except Exception as exc:  # noqa: BLE001 - statsmodels can fail to converge
        out["markov_error"] = type(exc).__name__
    out["participation_low_vol_markov"] = part_low_mk
    out["participation_high_vol_markov"] = part_high_mk

    # --- Verdict (prefer GMM, fall back to median split) ---------------------
    pl = part_low_gmm if np.isfinite(part_low_gmm) else part_low_med
    ph = part_high_gmm if np.isfinite(part_high_gmm) else part_high_med
    if np.isfinite(pl) and np.isfinite(ph):
        out["avoids_high_vol"] = bool(pl > ph)
    else:
        out["avoids_high_vol"] = float("nan")
    return out


# --------------------------------------------------------------------------- #
# Q4. Cross-asset: signal correlation + behavioral clustering                 #
# --------------------------------------------------------------------------- #
def cross_asset(
    signals: pd.DataFrame,
    ohlcv: pd.DataFrame,
    instruments: list[str] | None = None,
    n_clusters: int = 3,
) -> dict:
    """Q4 -- how correlated are the 11 signals, and how do they cluster?

    Two numeric outputs:

    1. ``mean_abs_offdiag_corr`` -- the mean absolute off-diagonal of the
       pairwise signal correlation matrix across instruments (computed on the
       common signal dates, treating the signal as a numeric series). Prior EDA
       expects ``~0.11``: the signals are nearly independent across assets.
       The full ``corr_matrix`` (instrument x instrument) is returned too.
    2. ``cluster_labels`` -- a fingerprint clustering of the instruments by their
       signal-behavior features: participation rate, long-bias, persistence
       ``P(s_t = s_{t-1})``, and the Q1 momentum score (sign of trailing-return
       correlation). Standardized features are clustered with
       ``sklearn.AgglomerativeClustering`` into ``min(n_clusters, n_inst)``
       groups. ``feature_table`` returns the raw per-instrument features.

    Degenerate-safe: instruments with a constant signal contribute ``nan`` to the
    correlation (excluded pairwise) and a finite feature row (constant
    participation/long-bias still well-defined); the clustering falls back to
    a single label if it fails. Never raises.
    """
    instruments = instruments or [c for c in signals.columns if c != "date"]

    # --- 1. Pairwise signal correlation --------------------------------------
    sig_wide = signals.set_index("date")[instruments].astype(float)
    corr = sig_wide.corr()
    corr_vals = corr.to_numpy(dtype=float)
    n = corr_vals.shape[0]
    off = ~np.eye(n, dtype=bool)
    off_vals = corr_vals[off]
    finite_off = off_vals[np.isfinite(off_vals)]
    mean_abs_offdiag = float(np.mean(np.abs(finite_off))) if finite_off.size else float("nan")

    # --- 2. Behavioral fingerprint features ----------------------------------
    rows: dict[str, dict[str, float]] = {}
    for inst in instruments:
        sig = _signal_series(signals, inst)
        try:
            mom = alpha_type(inst, signals, ohlcv).get("momentum_score", float("nan"))
        except Exception:  # noqa: BLE001 - feature extraction must not break clustering
            mom = float("nan")
        rows[inst] = {
            "participation": _participation_rate(sig),
            "long_bias": _long_bias(sig),
            "persistence": _persistence(sig),
            "momentum_score": float(mom) if np.isfinite(mom) else 0.0,
        }
    feature_table = pd.DataFrame(rows).T

    cluster_labels: dict[str, int] = {}
    k = int(min(n_clusters, max(1, len(instruments))))
    try:
        from sklearn.cluster import AgglomerativeClustering

        F = feature_table.to_numpy(dtype=float)
        F = np.where(np.isfinite(F), F, np.nanmean(np.where(np.isfinite(F), F, np.nan), axis=0))
        F = np.nan_to_num(F, nan=0.0)
        mu = F.mean(axis=0)
        sd = F.std(axis=0)
        sd[sd == 0] = 1.0
        Fs = (F - mu) / sd
        if k >= 2 and len(instruments) > k:
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(Fs)
        else:
            labels = np.zeros(len(instruments), dtype=int)
        cluster_labels = {inst: int(lab) for inst, lab in zip(instruments, labels)}
    except Exception:  # noqa: BLE001 - clustering must never break the run
        cluster_labels = {inst: 0 for inst in instruments}

    return {
        "n_instruments": int(len(instruments)),
        "mean_abs_offdiag_corr": mean_abs_offdiag,
        "corr_matrix": corr,
        "cluster_labels": cluster_labels,
        "n_clusters": k,
        "feature_table": feature_table,
    }


# --------------------------------------------------------------------------- #
# Q5. Drift: per-split base rates                                             #
# --------------------------------------------------------------------------- #
def drift(instrument: str, signals: pd.DataFrame) -> dict:
    """Q5 -- do the signal's base rates drift across train/val/test?

    Using the contract's chronological 60/20/20 split (train[0:387],
    val[387:516], test[516:645]), report for each split:

    * ``participation_rate`` -- fraction of nonzero days,
    * ``long_bias`` -- mean signal in ``[-1, 1]``,
    * the ``-1`` / ``0`` / ``+1`` class fractions.

    The ``trend`` block surfaces the train->val->test movement of participation
    and long-bias (``*_train_to_test`` deltas) so drift is visible at a glance --
    e.g. ng1s participation rises across splits. Returns numeric values per
    split; never raises (an empty split yields ``nan`` rates).
    """
    sig = _signal_series(signals, instrument)
    split = chronological_split(signals["date"])

    per_split: dict[str, dict[str, float]] = {}
    for name, idx in (
        ("train", split.train_idx),
        ("val", split.val_idx),
        ("test", split.test_idx),
    ):
        s = sig.iloc[idx]
        stats = {
            "participation_rate": _participation_rate(s),
            "long_bias": _long_bias(s),
            "n": int(len(s)),
        }
        stats.update(_class_fractions(s))
        per_split[name] = stats

    def _delta(key: str) -> float:
        a = per_split["train"][key]
        b = per_split["test"][key]
        if np.isfinite(a) and np.isfinite(b):
            return float(b - a)
        return float("nan")

    return {
        "instrument": instrument,
        "train": per_split["train"],
        "val": per_split["val"],
        "test": per_split["test"],
        "trend": {
            "participation_train_to_test": _delta("participation_rate"),
            "long_bias_train_to_test": _delta("long_bias"),
            "participation_train": per_split["train"]["participation_rate"],
            "participation_val": per_split["val"]["participation_rate"],
            "participation_test": per_split["test"]["participation_rate"],
        },
    }


# --------------------------------------------------------------------------- #
# Q6. Model-family fingerprint (ADVISORY / LOW-CONFIDENCE)                     #
# --------------------------------------------------------------------------- #
def _build_feature_matrix(
    instrument: str, signals: pd.DataFrame, ohlcv: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Engineered (X, y) for the surrogate classifiers in :func:`model_family_fingerprint`.

    Features (all using information available at signal time ``t``, i.e. closes
    up to ``t-1`` or same-day OHLC): lagged 1/5/10/20-day trailing returns,
    distance from the 10/20/50-day SMA, the 20-day rolling vol, and an up/down
    Donchian breakout flag pair. ``y`` is the signal on the aligned dates.
    """
    sig = _signal_series(signals, instrument)
    oi = _instrument_ohlcv(ohlcv, instrument)
    close = oi["close"].astype(float)
    log_close = np.log(close)
    high = oi["high"].astype(float) if "high" in oi.columns else close
    low = oi["low"].astype(float) if "low" in oi.columns else close

    cols: dict[str, pd.Series] = {}
    for k in (1, 5, 10, 20):
        cols[f"trail_{k}"] = log_close.shift(1) - log_close.shift(1 + k)
    for n in (10, 20, 50):
        sma = close.rolling(n).mean()
        cols[f"dist_ma_{n}"] = close.shift(1) - sma.shift(1)
    ret = log_close.diff()
    cols["vol_20"] = ret.rolling(20).std().shift(1)
    prior_high = high.rolling(20).max().shift(1)
    prior_low = low.rolling(20).min().shift(1)
    cols["break_up"] = (close > prior_high).astype(float)
    cols["break_down"] = (close < prior_low).astype(float)

    feat = pd.DataFrame(cols).reindex(sig.index)
    aligned = feat.assign(y=sig).dropna()
    y = aligned["y"].to_numpy(dtype=int)
    X = aligned.drop(columns="y").to_numpy(dtype=float)
    return X, y, list(feat.columns)


def model_family_fingerprint(
    instrument: str,
    signals: pd.DataFrame,
    ohlcv: pd.DataFrame,
    cv_folds: int = 4,
) -> dict:
    """Q6 -- ADVISORY guess at the generating model family. GATES NOTHING.

    Fits three shallow surrogate classifiers on an engineered feature set
    (lagged returns, distance-from-MA, vol, breakout flags; see
    :func:`_build_feature_matrix`) to predict the signal, and compares their
    cross-validated accuracy to guess whether the generator looks tree-like,
    linear, or nonlinear:

    * ``tree_cv_acc``     -- shallow ``DecisionTreeClassifier`` (depth 3),
    * ``linear_cv_acc``   -- multinomial ``LogisticRegression``,
    * ``forest_cv_acc``   -- small ``RandomForestClassifier`` (50 trees, depth 4),
    * ``majority_acc``    -- the no-skill majority-class baseline (the floor any
      surrogate must clear to be informative).

    A ``label`` is GUESSED from which family wins by a margin over the others and
    over the majority baseline (``'tree_like'`` / ``'linear'`` / ``'nonlinear'``
    / ``'inconclusive'``), with a ``confidence`` in ``[0, 1]`` derived from the
    margin of the best surrogate over the majority baseline. Low confidence is
    expected and ``'inconclusive'`` is an acceptable, returnable result.

    Degenerate-safe: a single-class ``y`` (e.g. ng1s in a window with only
    ``0``/``-1``, or a tiny aligned sample) makes cross-validation impossible;
    the function then returns ``label='inconclusive'``, ``confidence=0.0`` and
    ``nan`` scores. ALL classifier fits are wrapped in ``try/except`` -- this
    function NEVER raises.
    """
    out: dict = {
        "instrument": instrument,
        "tree_cv_acc": float("nan"),
        "linear_cv_acc": float("nan"),
        "forest_cv_acc": float("nan"),
        "majority_acc": float("nan"),
        "label": "inconclusive",
        "confidence": 0.0,
        "status": "ok",
    }

    try:
        X, y, feat_names = _build_feature_matrix(instrument, signals, ohlcv)
    except Exception as exc:  # noqa: BLE001 - feature build must not raise
        out["status"] = "inconclusive"
        out["error"] = type(exc).__name__
        return out

    out["n_obs"] = int(len(y))
    out["n_features"] = int(X.shape[1]) if X.ndim == 2 else 0
    classes, counts = np.unique(y, return_counts=True)
    out["n_classes"] = int(classes.size)
    if len(y) < 40 or classes.size < 2 or counts.min() < cv_folds:
        # Too little data or single-class / too-rare class for honest CV.
        out["status"] = "inconclusive"
        if counts.size:
            out["majority_acc"] = float(counts.max() / counts.sum())
        return out

    majority_acc = float(counts.max() / counts.sum())
    out["majority_acc"] = majority_acc

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier

        folds = int(min(cv_folds, counts.min()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xs = StandardScaler().fit_transform(X)
            tree = DecisionTreeClassifier(max_depth=3, random_state=0)
            linear = LogisticRegression(max_iter=500)
            forest = RandomForestClassifier(
                n_estimators=50, max_depth=4, random_state=0, n_jobs=1
            )
            tree_acc = float(np.mean(cross_val_score(tree, Xs, y, cv=folds)))
            linear_acc = float(np.mean(cross_val_score(linear, Xs, y, cv=folds)))
            forest_acc = float(np.mean(cross_val_score(forest, Xs, y, cv=folds)))
        out["tree_cv_acc"] = tree_acc
        out["linear_cv_acc"] = linear_acc
        out["forest_cv_acc"] = forest_acc
    except Exception as exc:  # noqa: BLE001 - any CV failure -> inconclusive
        out["status"] = "inconclusive"
        out["error"] = type(exc).__name__
        return out

    # --- Low-confidence family guess -----------------------------------------
    scores = {"tree_like": tree_acc, "linear": linear_acc, "nonlinear": forest_acc}
    best_label = max(scores, key=scores.get)
    best_acc = scores[best_label]
    sorted_acc = sorted(scores.values(), reverse=True)
    margin_over_second = sorted_acc[0] - sorted_acc[1] if len(sorted_acc) > 1 else 0.0
    margin_over_majority = best_acc - majority_acc

    # Confidence: must beat the majority baseline AND separate from the runner-up.
    if margin_over_majority <= 0.0:
        out["label"] = "inconclusive"
        out["confidence"] = 0.0
    else:
        conf = float(np.clip(margin_over_majority + 0.5 * max(margin_over_second, 0.0), 0.0, 1.0))
        # A weak separation between families is still a low-confidence guess.
        out["label"] = best_label if conf >= 0.05 else "inconclusive"
        out["confidence"] = conf
    out["margin_over_majority"] = float(margin_over_majority)
    out["margin_over_second_family"] = float(margin_over_second)
    return out


# --------------------------------------------------------------------------- #
# Combiners                                                                   #
# --------------------------------------------------------------------------- #
def characterize_instrument(
    instrument: str, signals: pd.DataFrame, ohlcv: pd.DataFrame
) -> dict:
    """Run Q1, Q2, Q3, Q5 and Q6 for one instrument and return the combined dict.

    Q4 (cross-asset) is intrinsically a panel-level question, so it is reported
    once by :func:`characterize_all` rather than per instrument. Each sub-result
    is degenerate-safe, so this never raises on any of the 11 instruments.
    """
    return {
        "instrument": instrument,
        "alpha_type": alpha_type(instrument, signals, ohlcv),
        "lead_lag": lead_lag(instrument, signals, ohlcv),
        "regime": regime(instrument, ohlcv, signals),
        "drift": drift(instrument, signals),
        "model_family_fingerprint": model_family_fingerprint(instrument, signals, ohlcv),
    }


def characterize_all(
    signals: pd.DataFrame,
    ohlcv: pd.DataFrame,
    instruments: list[str] | None = None,
) -> dict:
    """Run the full C1 characterization over all (or a subset of) instruments.

    Returns a dict with a ``per_instrument`` mapping (each value is the
    :func:`characterize_instrument` result) and a single panel-level
    ``cross_asset`` (Q4) entry. Never raises on the released universe.
    """
    instruments = instruments or [c for c in signals.columns if c != "date"]
    per_instrument = {
        inst: characterize_instrument(inst, signals, ohlcv) for inst in instruments
    }
    return {
        "instruments": list(instruments),
        "per_instrument": per_instrument,
        "cross_asset": cross_asset(signals, ohlcv, instruments=instruments),
    }
