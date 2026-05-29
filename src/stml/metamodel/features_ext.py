"""
features_ext.py
===============
Extended engineered (E-class) feature families folded into the shared
feature-engineering base from the ``Harry`` and ``Sreeram`` branches. Every
function here is **engineered, no fit** — causal by truncation-invariance (the
value at ``t`` is identical on ``data[:t+1]`` and on ``data[:T]``) — and so
joins the matrix exactly like :mod:`stml.metamodel.features`.

Families added here (provenance in parentheses):

* **F2 add** ``f2_rogers_satchell_20`` — Rogers-Satchell drift-independent
  range volatility (Sreeram), the third OHLC range estimator alongside the
  existing Parkinson / Garman-Klass.
* **F5 adds** ``f5_signal_entropy_20`` / ``f5_flip_rate_60`` — Shannon entropy
  and flip-rate of the trailing primary-signal trajectory (Harry).
* **F7 adds** ``f7_rolls_spread_20`` / ``f7_kyles_lambda_20`` /
  ``f7_overnight_gap`` — Roll effective spread, Kyle's lambda price-impact, and
  the overnight log gap (Harry), with the same zero-volume-> NaN guard as the
  existing Amihud column.
* **F12 mean-reversion / path-structure & trend-quality** (Sreeram) —
  ``f12_hurst_100``, ``f12_variance_ratio_5_21``, ``f12_efficiency_ratio_21``,
  ``f12_autocorr_21``, ``f12_trend_tval_{10,21,42}``, ``f12_ma21_slope``.
* **F13 wavelet / multiscale energy** (Harry) — ``f13_mra_energy_d1..d5``
  (requires PyWavelets; ``uv sync --extra features-extra``).
* **F15 conditional risk / first-passage** (Harry) — ``f15_expected_hit_time``,
  ``f15_prob_timeout``, ``f15_path_tortuosity_20``,
  ``f15_realized_semi_vol_ratio_20`` (seeded-bootstrap, deterministic per row).

Standardization
---------------
:data:`Z_TWIN_COLUMNS` lists the **scale-dependent** E-class columns (across the
whole engineered stack, core + ext) that receive a parallel ``z_<col>`` twin:
a per-instrument causal expanding-window z-score (:func:`expanding_zscore`).
Bounded / already-normalized columns (ratios, probabilities, t-statistics,
sin/cos, [0,1] positions) get no twin. :func:`add_z_twins` builds the twins;
:mod:`stml.metamodel.catalog` registers one :class:`FeatureSpec` per twin from
the SAME list so the 1:1 coverage guard cannot drift.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from stml.metamodel.features import _ohlcv_indexed

__all__ = [
    "expanding_zscore",
    "Z_TWIN_COLUMNS",
    "add_z_twins",
    "f2_rogers_satchell",
    "f5_signal_trajectory",
    "f7_microstructure_ext",
    "f12_path_structure",
    "f13_wavelet_energy",
    "f15_conditional_risk",
    "assemble_engineered_ext",
]

_ANN = float(np.sqrt(252.0))  # annualisation factor for daily vol


# --------------------------------------------------------------------------- #
# Standardization helper + the curated z-twin column set.                     #
# --------------------------------------------------------------------------- #
def expanding_zscore(s: pd.Series, min_periods: int = 60) -> pd.Series:
    """Per-instrument causal expanding-window z-score ``(s - mean) / std``.

    Uses :meth:`pandas.Series.expanding` (mean/std over ``data[:t+1]`` only), so
    the value at ``t`` never sees a future row — split-agnostic and leakage-safe
    (Sreeram's standardization scheme). A zero/NaN expanding std yields ``NaN``
    for that row rather than a divide-by-zero. NaN before ``min_periods`` rows.
    """
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std()
    return (s - mu) / sd.replace(0.0, np.nan)


#: Scale-dependent E-class columns that receive a causal expanding-z ``z_`` twin.
#: Bounded / already-normalized columns (ratios, probabilities, t-stats,
#: sin/cos, [-1,1] / [0,1] positions, percentiles, skew/kurt) are excluded.
Z_TWIN_COLUMNS: tuple[str, ...] = (
    # F2 volatility (price/return-scale) + the new Rogers-Satchell estimator
    "f2_vol_10",
    "f2_vol_20",
    "f2_vol_60",
    "f2_vol_of_vol_20",
    "f2_parkinson_20",
    "f2_garman_klass_20",
    "f2_atr_14",
    "f2_rogers_satchell_20",
    # F6 price-unit momentum (MACD line + histogram)
    "f6_macd_12_26",
    "f6_macd_hist_12_26_9",
    # F7 microstructure (level/impact scale)
    "f7_amihud_20",
    "f7_oi_level",
    "f7_oi_change",
    "f7_kyles_lambda_20",
    "f7_rolls_spread_20",
    "f7_overnight_gap",
    # F10 OHLC ranges / open-to-open returns
    "f10_hl_range",
    "f10_hl_range_mean_20",
    "f10_oto_ret",
    "f10_oto_ret_mean_20",
    # F12 normalised MA slope (the other F12 cols are already dimensionless)
    "f12_ma21_slope",
    # F15 conditional-risk magnitudes (prob_timeout is bounded [0,1] -> no twin)
    "f15_expected_hit_time",
    "f15_path_tortuosity_20",
    "f15_realized_semi_vol_ratio_20",
)


def add_z_twins(
    frame: pd.DataFrame, min_periods: int = 60
) -> pd.DataFrame:
    """Return a date-indexed frame of ``z_<col>`` twins for the columns in
    :data:`Z_TWIN_COLUMNS` that are present in ``frame``.

    Each twin is the per-instrument causal expanding-window z-score of the raw
    column (:func:`expanding_zscore`). Columns absent from ``frame`` are skipped
    (defensive); the build asserts full coverage downstream.
    """
    out: dict[str, pd.Series] = {}
    for col in Z_TWIN_COLUMNS:
        if col in frame.columns:
            out[f"z_{col}"] = expanding_zscore(frame[col], min_periods=min_periods)
    return pd.DataFrame(out, index=frame.index)


# --------------------------------------------------------------------------- #
# F2 add — Rogers-Satchell range volatility (Sreeram).                        #
# --------------------------------------------------------------------------- #
def f2_rogers_satchell(ohlcv_inst: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Rogers-Satchell (1991) drift-independent range volatility, annualised.

    Per-bar variance contribution ``ln(h/c)·ln(h/o) + ln(l/c)·ln(l/o)`` averaged
    over a trailing ``window`` and annualised by ``sqrt(252)`` — the
    drift-robust sibling of the existing F2 Parkinson / Garman-Klass estimators
    (which assume zero drift). Trailing-only (info ``<= t``); warm-up NaN.
    """
    df = _ohlcv_indexed(ohlcv_inst)
    o, h_, l_, c_ = df["open"], df["high"], df["low"], df["close"]
    pos = (o > 0.0) & (h_ > 0.0) & (l_ > 0.0) & (c_ > 0.0)
    ln_hc = np.log((h_ / c_).where(pos))
    ln_ho = np.log((h_ / o).where(pos))
    ln_lc = np.log((l_ / c_).where(pos))
    ln_lo = np.log((l_ / o).where(pos))
    rs_bar = ln_hc * ln_ho + ln_lc * ln_lo
    rs = np.sqrt(rs_bar.rolling(window, min_periods=window).mean().clip(lower=0.0)) * _ANN
    return pd.DataFrame({f"f2_rogers_satchell_{window}": rs}, index=df.index)


# --------------------------------------------------------------------------- #
# F5 adds — signal-trajectory entropy / flip-rate (Harry).                    #
# --------------------------------------------------------------------------- #
def f5_signal_trajectory(
    signal_inst: pd.Series, entropy_window: int = 20, flip_window: int = 60
) -> pd.DataFrame:
    """Trailing structure of the primary signal: Shannon entropy and flip rate.

    * ``f5_signal_entropy_20`` — Shannon entropy (nats) of the empirical
      ``{-1,0,+1}`` PMF over the trailing ``entropy_window`` bars, in
      ``[0, log 3]``. High = the signal is hopping across states (less reliable).
    * ``f5_flip_rate_60`` — fraction of consecutive-bar value changes over the
      trailing ``flip_window``, in ``[0, 1]``. The row-0 flip is undefined (NaN).

    Both are causal (trailing windows on ``s[:t+1]``) and complement the
    existing F5 run-length / participation columns. Warm-up rows are NaN.
    """
    s = pd.Series(signal_inst).sort_index().astype("float64")

    rounded = s.round()

    def _entropy(vals: np.ndarray) -> float:
        n = len(vals)
        if n == 0:
            return float("nan")
        counts = np.array(
            [(vals == -1).sum(), (vals == 0).sum(), (vals == 1).sum()],
            dtype="float64",
        )
        probs = counts / n
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    entropy = rounded.rolling(entropy_window, min_periods=entropy_window).apply(
        _entropy, raw=True
    )

    flips = s.ne(s.shift(1)).astype("float64")
    if len(flips) > 0:
        flips.iloc[0] = np.nan  # row-0 flip undefined
    flip_rate = flips.rolling(flip_window, min_periods=flip_window).mean()

    return pd.DataFrame(
        {
            f"f5_signal_entropy_{entropy_window}": entropy,
            f"f5_flip_rate_{flip_window}": flip_rate,
        },
        index=s.index,
    )


# --------------------------------------------------------------------------- #
# F7 adds — Roll spread, Kyle's lambda, overnight gap (Harry).                #
# --------------------------------------------------------------------------- #
def f7_microstructure_ext(ohlcv_inst: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Extra microstructure features: Roll spread, Kyle's lambda, overnight gap.

    * ``f7_rolls_spread_20`` — Roll (1984) implied bid-ask spread
      ``2·sqrt(max(-Cov(Δp_t, Δp_{t-1}), 0))`` over the trailing window.
    * ``f7_kyles_lambda_20`` — Hasbrouck (2009) daily-bar Kyle's lambda
      ``mean(|ret| / sqrt(volume))``; zero-volume rows are NaN-masked (no /0).
    * ``f7_overnight_gap`` — overnight log return ``log(open_t / close_{t-1})``.

    All trailing windows close at ``t`` (info ``<= t``); the overnight gap uses
    today's open and yesterday's close. Warm-up / zero-volume rows are NaN.
    """
    df = _ohlcv_indexed(ohlcv_inst)
    close = df["close"]
    open_ = df["open"]
    volume = df["volume"].astype("float64")
    ret = np.log(close).diff()
    out: dict[str, pd.Series] = {}

    # Roll (1984) effective spread from the negative serial covariance of Δp.
    dp = close.astype("float64").diff()
    dp_lag = dp.shift(1)
    cov = (
        (dp * dp_lag).rolling(window, min_periods=window).mean()
        - dp.rolling(window, min_periods=window).mean()
        * dp_lag.rolling(window, min_periods=window).mean()
    )
    out[f"f7_rolls_spread_{window}"] = 2.0 * np.sqrt(np.clip(-cov, a_min=0.0, a_max=None))

    # Kyle's lambda |ret| / sqrt(volume), zero-volume -> NaN.
    safe_vol = volume.where(volume > 0.0)
    impact = ret.abs() / np.sqrt(safe_vol)
    out[f"f7_kyles_lambda_{window}"] = impact.rolling(window, min_periods=window).mean()

    # Overnight log gap log(open_t / close_{t-1}); non-positive -> NaN (no inf).
    prev_close = close.shift(1)
    ok = (open_ > 0.0) & (prev_close > 0.0)
    out["f7_overnight_gap"] = np.log((open_ / prev_close).where(ok))

    return pd.DataFrame(out, index=df.index)


# --------------------------------------------------------------------------- #
# F12 — mean-reversion / path-structure & trend-quality (Sreeram).            #
# --------------------------------------------------------------------------- #
def _hurst_rs(x: np.ndarray) -> float:
    """Rescaled-range Hurst exponent of a return window (>0.5 trending)."""
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 20:
        return np.nan
    sizes = np.unique(np.geomspace(8, n // 2, num=5, dtype=int))
    sizes = sizes[sizes >= 8]
    if len(sizes) < 2:
        return np.nan
    rs_vals: list[tuple[int, float]] = []
    for s in sizes:
        n_chunks = n // s
        if n_chunks < 1:
            continue
        rs_chunk: list[float] = []
        for k in range(n_chunks):
            seg = x[k * s : (k + 1) * s]
            Y = seg - seg.mean()
            Z = Y.cumsum()
            R = Z.max() - Z.min()
            S = seg.std(ddof=1)
            if S > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            rs_vals.append((s, float(np.mean(rs_chunk))))
    if len(rs_vals) < 2:
        return np.nan
    arr = np.asarray(rs_vals, dtype=float)
    slope = np.polyfit(np.log(arr[:, 0]), np.log(arr[:, 1]), 1)[0]
    return float(slope)


def _eff_ratio(x: np.ndarray) -> float:
    """Kaufman efficiency ratio |net move| / sum(|moves|) over a window."""
    net = abs(np.nansum(x))
    path = np.nansum(np.abs(x))
    return net / path if path > 0 else np.nan


def _var_ratio(x: np.ndarray, k: int) -> float:
    """Variance ratio Var(k-period) / (k · Var(1-period)); >1 trending."""
    x = x[~np.isnan(x)]
    if len(x) < k + 5:
        return np.nan
    v1 = np.var(x, ddof=1)
    if v1 <= 0:
        return np.nan
    k_ret = pd.Series(x).rolling(k).sum().dropna().to_numpy()
    if len(k_ret) < 2:
        return np.nan
    return float(np.var(k_ret, ddof=1) / (k * v1))


def _t_val_lin_r(y: np.ndarray) -> float:
    """t-statistic of the OLS slope of ``y`` on a time index (tValLinR)."""
    y = y[~np.isnan(y)]
    n = len(y)
    if n < 5:
        return np.nan
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    xx = ((x - x_mean) ** 2).sum()
    if xx == 0:
        return np.nan
    beta = ((x - x_mean) * (y - y_mean)).sum() / xx
    resid = y - (y_mean + beta * (x - x_mean))
    sse = (resid**2).sum()
    if n <= 2 or sse <= 0:
        return np.nan
    se_beta = np.sqrt((sse / (n - 2)) / xx)
    if se_beta == 0:
        return np.nan
    return float(beta / se_beta)


def f12_path_structure(
    ohlcv_inst: pd.DataFrame,
    autocorr_window: int = 21,
    efficiency_window: int = 21,
    variance_ratio_lag: int = 5,
    variance_ratio_window: int = 21,
    hurst_window: int = 100,
    trend_scan_windows: tuple[int, ...] = (10, 21, 42),
) -> pd.DataFrame:
    """Mean-reversion / path-structure and trend-quality features (Sreeram G2/G3).

    All are dimensionless / already-standardized (autocorrelation, efficiency
    ratio, variance ratio, Hurst, trend t-statistics) except ``f12_ma21_slope``
    (the sigma-normalised MA slope, which gets a z-twin). Every column uses
    trailing ``rolling`` windows (info ``<= t``); warm-up rows are NaN.

    Columns
    -------
    f12_autocorr_21 : lag-1 autocorrelation of returns (negative -> mean-revert).
    f12_efficiency_ratio_21 : Kaufman ER |net|/sum|moves| in [0,1] (1 = clean).
    f12_variance_ratio_5_21 : Var(5d)/(5·Var(1d)); >1 trending, <1 mean-revert.
    f12_hurst_100 : rescaled-range Hurst exponent (>0.5 trending).
    f12_trend_tval_10/21/42 : t-stat of the OLS slope of log-close on time.
    f12_ma21_slope : 1d log-change of the 21d MA normalised by 1d return std.
    """
    df = _ohlcv_indexed(ohlcv_inst)
    close = df["close"]
    ret = np.log(close).diff()
    out: dict[str, pd.Series] = {}

    out[f"f12_autocorr_{autocorr_window}"] = ret.rolling(autocorr_window).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if pd.Series(x).std() > 0 else np.nan,
        raw=True,
    )
    out[f"f12_efficiency_ratio_{efficiency_window}"] = ret.rolling(
        efficiency_window
    ).apply(_eff_ratio, raw=True)
    out[
        f"f12_variance_ratio_{variance_ratio_lag}_{variance_ratio_window}"
    ] = ret.rolling(variance_ratio_window).apply(
        lambda x: _var_ratio(x, variance_ratio_lag), raw=True
    )

    log_close = np.log(close)
    for w in trend_scan_windows:
        out[f"f12_trend_tval_{w}"] = log_close.rolling(w).apply(_t_val_lin_r, raw=True)

    out[f"f12_hurst_{hurst_window}"] = ret.rolling(hurst_window).apply(
        _hurst_rs, raw=True
    )

    ma21 = close.rolling(21).mean()
    out["f12_ma21_slope"] = np.log(ma21 / ma21.shift(1)) / ret.rolling(21).std()

    return pd.DataFrame(out, index=df.index)


# --------------------------------------------------------------------------- #
# F13 — wavelet multi-resolution energy bands (Harry; requires PyWavelets).   #
# --------------------------------------------------------------------------- #
def f13_wavelet_energy(
    ohlcv_inst: pd.DataFrame,
    wavelet: str = "db4",
    levels: int = 5,
    window: int = 252,
) -> pd.DataFrame:
    """Trailing-window multi-resolution-analysis (MRA) energy fractions.

    For each row ``t`` a discrete wavelet transform (``periodization`` mode) of
    the trailing ``window`` log returns gives the fraction of squared variation
    at each of the first ``levels`` detail scales (~daily ... ~quarterly). Rows
    sum to ``[0, 1]`` (the missing mass is the slow-drift approximation band).
    Strictly trailing (info ``<= t``); warm-up ``window - 1`` rows are NaN.

    Requires PyWavelets — install with ``uv sync --extra features-extra``.
    """
    try:
        import pywt  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "f13_wavelet_energy requires PyWavelets. Run "
            "'uv sync --extra features-extra'."
        ) from exc

    if window < 2**levels:
        raise ValueError(f"window must be >= 2^levels = {2**levels}, got {window}")

    df = _ohlcv_indexed(ohlcv_inst)
    ret = np.log(df["close"]).diff()
    arr = ret.to_numpy(dtype="float64")
    n = len(arr)
    out = np.full((n, levels), np.nan, dtype="float64")
    for t in range(window - 1, n):
        seg = np.array(arr[t - window + 1 : t + 1], dtype="float64")  # writable copy
        if not np.isfinite(seg).all():
            continue
        coeffs = pywt.wavedec(seg, wavelet, level=levels, mode="periodization")
        detail_energy = np.array(
            [float(np.sum(np.square(coeffs[i]))) for i in range(levels, 0, -1)],
            dtype="float64",
        )
        total = float(np.sum(np.square(seg)))
        if total > 0:
            out[t, :] = detail_energy / total
    cols = [f"f13_mra_energy_d{k}" for k in range(1, levels + 1)]
    return pd.DataFrame(out, index=df.index, columns=cols)


# --------------------------------------------------------------------------- #
# F15 — conditional risk / first-passage (Harry; seeded bootstrap).           #
# --------------------------------------------------------------------------- #
_PER_ROW_PRIME: int = 1_000_003
_TORTUOSITY_EPS: float = 1e-12


def _simulate_first_passage(
    returns: np.ndarray,
    vol: np.ndarray,
    *,
    pt_mult: float,
    sl_mult: float,
    h: int,
    window: int,
    n_sims: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row bootstrap of first passage to ``±mult·vol·sqrt(h)`` barriers.

    Returns ``(median_hit_time, prob_timeout)`` arrays. The RNG is seeded per
    row from ``seed·PRIME + t`` (``t`` the positional index), so a row's output
    is identical computed on ``data[:t+1]`` or ``data[:T]`` — truncation-safe.
    Rows with a NaN trailing window or non-positive ``vol[t]`` are left NaN.
    """
    n = len(returns)
    sqrt_h = float(np.sqrt(h))
    median_hit = np.full(n, np.nan)
    p_timeout = np.full(n, np.nan)
    for t in range(window, n):
        past = returns[t - window : t]
        if not np.isfinite(past).all():
            continue
        v_t = vol[t]
        if not np.isfinite(v_t) or v_t <= 0:
            continue
        pt = pt_mult * v_t * sqrt_h
        sl = sl_mult * v_t * sqrt_h
        rng = np.random.default_rng(seed * _PER_ROW_PRIME + t)
        samples = rng.choice(past, size=(n_sims, h), replace=True)
        cum = samples.cumsum(axis=1)
        touch = (cum >= pt) | (cum <= -sl)
        has_touch = touch.any(axis=1)
        if has_touch.any():
            first_touch = touch.argmax(axis=1) + 1
            median_hit[t] = float(np.median(first_touch[has_touch]))
        else:
            median_hit[t] = float(h + 1)
        p_timeout[t] = float(1.0 - has_touch.mean())
    return median_hit, p_timeout


def f15_conditional_risk(
    ohlcv_inst: pd.DataFrame,
    *,
    pt_mult: float = 1.0,
    sl_mult: float = 1.0,
    h: int = 10,
    window: int = 252,
    n_sims: int = 200,
    seed: int = 42,
    vol_window: int = 20,
    tort_window: int = 20,
) -> pd.DataFrame:
    """First-passage and path-shape conditional-risk features (Harry).

    The barrier sigma is the trailing **daily** return std (de-annualised by
    construction — never the annualised ``f2_vol_20``, which would make every
    path time out). Columns:

    * ``f15_expected_hit_time`` — bootstrap median first-passage time to the
      symmetric ``±mult·sigma·sqrt(h)`` barriers (timeout = ``h+1``).
    * ``f15_prob_timeout`` — bootstrap probability neither barrier is hit in ``h``.
    * ``f15_path_tortuosity_20`` — trailing ``Σ|r|/|Σr|`` (1 = monotonic, larger
      = zigzag).
    * ``f15_realized_semi_vol_ratio_20`` — upside-RMS / downside-RMS of returns.

    Causal: trailing windows + per-row-seeded bootstrap (truncation-invariant).
    """
    df = _ohlcv_indexed(ohlcv_inst)
    ret = np.log(df["close"]).diff()
    vol_daily = ret.rolling(vol_window, min_periods=vol_window).std()

    hit, pto = _simulate_first_passage(
        ret.to_numpy(dtype="float64"),
        vol_daily.to_numpy(dtype="float64"),
        pt_mult=pt_mult,
        sl_mult=sl_mult,
        h=h,
        window=window,
        n_sims=n_sims,
        seed=seed,
    )

    abs_sum = ret.abs().rolling(tort_window, min_periods=tort_window).sum()
    net = ret.rolling(tort_window, min_periods=tort_window).sum()
    tortuosity = abs_sum / (net.abs() + _TORTUOSITY_EPS)

    pos = ret.where(ret > 0, 0.0)
    neg = ret.where(ret < 0, 0.0)
    rms_pos = np.sqrt((pos**2).rolling(tort_window, min_periods=tort_window).mean())
    rms_neg = np.sqrt((neg**2).rolling(tort_window, min_periods=tort_window).mean())
    semi_ratio = rms_pos / (rms_neg + _TORTUOSITY_EPS)

    return pd.DataFrame(
        {
            "f15_expected_hit_time": pd.Series(hit, index=df.index),
            "f15_prob_timeout": pd.Series(pto, index=df.index),
            f"f15_path_tortuosity_{tort_window}": tortuosity,
            f"f15_realized_semi_vol_ratio_{tort_window}": semi_ratio,
        },
        index=df.index,
    )


# --------------------------------------------------------------------------- #
# Assembly — every extended E-class family on the instrument's date index.    #
# --------------------------------------------------------------------------- #
def assemble_engineered_ext(
    ohlcv_inst: pd.DataFrame, signal_inst: pd.Series
) -> pd.DataFrame:
    """Assemble the extended E-class families (F2-RS, F5-adds, F7-adds, F12,
    F13, F15) into one date-indexed frame, aligned on the price + signal union.

    Computed separately from :func:`stml.metamodel.features.assemble_engineered`
    so the F4 latent fit keeps its stable core-feature input; this block is
    joined alongside the fitted families in :class:`FeaturePipeline`.
    """
    f2rs = f2_rogers_satchell(ohlcv_inst)
    f7e = f7_microstructure_ext(ohlcv_inst)
    f12 = f12_path_structure(ohlcv_inst)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f13 = f13_wavelet_energy(ohlcv_inst)
        f15 = f15_conditional_risk(ohlcv_inst)

    sig = pd.Series(signal_inst).sort_index()
    f5e = f5_signal_trajectory(sig)

    price_idx = f2rs.index
    union_idx = price_idx.union(f5e.index)
    frames = [f2rs, f7e, f12, f13, f15, f5e]
    out = pd.concat([f.reindex(union_idx) for f in frames], axis=1)
    return out.sort_index()
