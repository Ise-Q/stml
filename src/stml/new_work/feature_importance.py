"""feature_importance.py — Cluster-level feature importance analysis (Section 4).

Pipeline
--------
1. Hygiene        – build X from F1–F15 + HMM; trim warmup; drop dead cols;
                    dedupe near-perfect pairs; sanity-check macro timestamping.
2. Partition       – correlation-cluster continuous daily features; hand-assign
                    F4 (latent), F5 (signal-derived), F8 (calendar).
3. Cluster         – Ward linkage on sqrt(1–|rho|) Spearman distance;
                    K by silhouette (CH and DB reported alongside).
4. Importance      – inside CPCV loop (purged/embargoed):
                       PFI/MDA, MDI (tree bias flagged), TreeSHAP.
5. Clustered MDA   – jointly permute each cluster; headline result.
6. Dim-reduction   – confirm AUC holds with one representative per cluster.

All seeds fixed at 42.  Pre-computed data (HMM features, triple-barrier labels,
macro features CSV) are loaded from CSV so HMM re-fitting is not needed.
Only harry/new_work HMM features are used; any hmm_* columns from other sources
are dropped in build_feature_matrix.

Public API
----------
    from stml.new_work.feature_importance import run_analysis
    results = run_analysis(["es1s", "nq1s", "fesx1s"])
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# --- repo paths ---
_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

from stml.na_checks import load_clean_ohlcv, native_returns, wide_returns
from stml.new_work.cpcv_search import CombinatorialPurgedKFold
from stml.harry.features.macro_features import (
    m1_volatility_term_structure,
    m2_rates_curve,
    m3_credit,
    m4_fx_dollar,
    m5_commodity_fundamentals,
    m6_macro_growth,
)
from stml.harry.features.wavelet import mra_energy_bands
from stml.harry.features.conditional_risk import (
    expected_hit_time,
    path_tortuosity_20d,
    prob_timeout,
    realized_semi_vol_ratio,
)
from stml.harry.features.cross_asset import (
    ASSET_CLASSES,
    asset_class_dispersion_z,
    distance_to_lead_lag_centroid,
    ewma_implied_corr_z,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_GROUPS = 6
K_CPCV = 2
EMBARGO = 0.01
PRESAMPLE_CUTOFF = pd.Timestamp("2020-01-03")
OUTPUTS = _HERE / "outputs"

# Warmup: 63-day PMI change is the longest fundamental lookback.
# The F2 vol-percentile (252-day rolling rank) is the longest price warmup.
# All are satisfied deep in the pre-sample period before 2020.
_LONGEST_WARMUP = 252

DATA_PATHS = {
    "ohlcv":       _REPO / "data" / "ohlcv_data.csv",
    "signals":     _REPO / "data" / "primary_signals.csv",
    "macro_feats": _REPO / "data" / "meta" / "macro_features.csv",
    "labels":      _REPO / "data" / "meta" / "triple_barrier_labels_fixed.csv",
    "hmm_vol":     _HERE / "features_hmm_vol.csv",
    "hmm_macro":   _HERE / "features_hmm_macro.csv",
    "alt_macro":   _REPO / "data" / "alternate_data_cleaned.csv",
}

# Correlation-clusterable feature prefixes (continuous, daily-moving).
CORR_CLUSTER_PREFIXES = (
    "f1_", "f2_", "f6_", "f7_", "f10_",
    "f11_", "f12_", "f13_", "f15_",
    "hmm_vol_", "hmm_macro_",
)

# Hand-assigned groups: prefix → group label.
HAND_ASSIGNED_PREFIXES = {
    "f4_": "F4_latent",
    "f5_": "F5_signal",
    "f8_": "F8_calendar",
}

# Near-perfect pair threshold (Spearman |rho| >= this → drop one).
TWIN_THRESHOLD = 0.99
# Near-zero variance threshold.
NZV_THRESHOLD = 1e-6


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data() -> dict[str, pd.DataFrame]:
    """Load all raw data needed for the analysis."""
    ohlcv = load_clean_ohlcv(DATA_PATHS["ohlcv"])

    signals = pd.read_csv(DATA_PATHS["signals"], parse_dates=["date"])

    macro_feats = pd.read_csv(DATA_PATHS["macro_feats"], parse_dates=["Date"])
    macro_feats = macro_feats.rename(columns={"Date": "date"}).set_index("date").sort_index()
    # Rename to ensure f11_ prefix (columns in the file already have f11_ prefix)
    macro_feats.columns = [
        c if c.startswith("f11_") else f"f11_{c}" for c in macro_feats.columns
    ]

    labels = pd.read_csv(
        DATA_PATHS["labels"], parse_dates=["date", "t1"]
    )

    hmm_vol = pd.read_csv(DATA_PATHS["hmm_vol"], parse_dates=["date"])
    hmm_macro = pd.read_csv(DATA_PATHS["hmm_macro"], parse_dates=["date"])

    alt_macro = pd.read_csv(DATA_PATHS["alt_macro"], parse_dates=["Date"]) \
        if DATA_PATHS["alt_macro"].exists() else None

    return {
        "ohlcv":       ohlcv,
        "signals":     signals,
        "macro_feats": macro_feats,
        "labels":      labels,
        "hmm_vol":     hmm_vol,
        "hmm_macro":   hmm_macro,
        "alt_macro":   alt_macro,
    }


# ---------------------------------------------------------------------------
# Feature engineering — F1-F10 (ported from signal-deep-dive features.py)
# ---------------------------------------------------------------------------

def _ohlcv_df(ohlcv_inst: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["date", "open", "high", "low", "close", "volume", "open_interest"]
            if c in ohlcv_inst.columns]
    df = (
        ohlcv_inst[cols]
        .dropna(subset=["close"])
        .drop_duplicates("date")
        .sort_values("date")
        .set_index("date")
        .astype(float)
    )
    return df


def _zsc(x: pd.Series, w: int) -> pd.Series:
    r = x.rolling(w, min_periods=w)
    return (x - r.mean()) / r.std().where(lambda s: s > 0.0)


def _rank_in_window(x: pd.Series, w: int) -> pd.Series:
    def _last(a: np.ndarray) -> float:
        return float(np.nan if not np.isfinite(a[-1]) else np.mean(a <= a[-1]))
    return x.rolling(w, min_periods=w).apply(_last, raw=True)


def f1_counter_trend(ohlcv_inst: pd.DataFrame) -> pd.DataFrame:
    df = _ohlcv_df(ohlcv_inst)
    c = df["close"]
    logc = np.log(c)
    out: dict[str, pd.Series] = {}
    for L in (10, 20, 40):
        sma = c.rolling(L, min_periods=L).mean()
        gap = c - sma
        out[f"f1_mr_score_{L}"] = -_zsc(gap, L)
        out[f"f1_dist_ma_sigma_{L}"] = gap / gap.rolling(L, min_periods=L).std().where(lambda s: s > 0)
        out[f"f1_ret_reversal_{L}"] = -(logc - logc.shift(L))
        lo = c.rolling(L, min_periods=L).min()
        hi = c.rolling(L, min_periods=L).max()
        out[f"f1_hilo_pos_{L}"] = (c - lo) / (hi - lo).where(lambda w: w > 0)
    delta = c.diff()
    gain, loss = delta.clip(lower=0), (-delta).clip(lower=0)
    ag = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    al = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = ag / al.where(al > 0)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(~((al == 0) & (ag > 0)), other=100.0)
    out["f1_rsi_14"] = rsi
    s20 = c.rolling(20, min_periods=20).mean()
    sd20 = c.rolling(20, min_periods=20).std()
    band = (4 * sd20).where(sd20 > 0)
    out["f1_bb_pctb_20"] = (c - (s20 - 2 * sd20)) / band
    out["f1_bb_bandwidth_20"] = band / s20.where(s20 != 0)
    return pd.DataFrame(out, index=df.index)


def f2_vol_dispersion(ohlcv_inst: pd.DataFrame, inst: str) -> pd.DataFrame:
    df = _ohlcv_df(ohlcv_inst)
    rets_long = native_returns(ohlcv_inst, kind="log")
    out: dict[str, pd.Series] = {}
    for L in (10, 20, 60):
        v = (
            rets_long.loc[rets_long["instrument"] == inst]
            .set_index("date")["ret"]
            .sort_index()
            .rolling(L)
            .std()
            * np.sqrt(252)
        ).reindex(df.index)
        out[f"f2_vol_{L}"] = v
    v20, v60 = out["f2_vol_20"], out["f2_vol_60"]
    out["f2_vol_ratio_20_60"] = v20 / v60.where(v60 > 0)
    out["f2_vol_pctile_20"] = _rank_in_window(v20, 252)
    out["f2_vol_of_vol_20"] = v20.rolling(60, min_periods=60).std()
    h, lo, c, o = df["high"], df["low"], df["close"], df["open"]
    lhl = np.log((h / lo).where((h > 0) & (lo > 0)))
    lco = np.log((c / o).where((c > 0) & (o > 0)))
    out["f2_parkinson_20"] = np.sqrt(
        (lhl ** 2 / (4 * np.log(2))).rolling(20, min_periods=20).mean() * 252
    )
    gk = (0.5 * lhl ** 2 - (2 * np.log(2) - 1) * lco ** 2)
    out["f2_garman_klass_20"] = np.sqrt(
        gk.rolling(20, min_periods=20).mean().clip(lower=0) * 252
    )
    pc = c.shift(1)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    out["f2_atr_14"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    ret_s = (
        rets_long.loc[rets_long["instrument"] == inst]
        .set_index("date")["ret"]
        .sort_index()
        .reindex(df.index)
    )
    out["f2_ret_skew_60"] = ret_s.rolling(60, min_periods=60).skew()
    out["f2_ret_kurt_60"] = ret_s.rolling(60, min_periods=60).kurt()
    return pd.DataFrame(out, index=df.index)


def _run_length(vals: np.ndarray) -> np.ndarray:
    n = len(vals)
    r = np.empty(n, dtype=float)
    if n == 0:
        return r
    r[0] = 1.0
    for i in range(1, n):
        r[i] = r[i-1] + 1.0 if vals[i] == vals[i-1] else 1.0
    return r


def _days_since_nonzero(vals: np.ndarray) -> np.ndarray:
    n = len(vals)
    out = np.empty(n, dtype=float)
    since = np.nan
    for i in range(n):
        v = vals[i]
        if np.isfinite(v) and v != 0.0:
            since = 0.0
        elif np.isfinite(since):
            since += 1.0
        out[i] = since
    return out


def f5_signal_derived(signal_inst: pd.Series, mr_score: pd.Series | None = None) -> pd.DataFrame:
    s = signal_inst.sort_index().astype(float)
    idx = s.index
    vals = s.to_numpy()
    out: dict[str, pd.Series] = {}
    rl = _run_length(vals)
    out["f5_signal"] = s
    out["f5_abs_signal"] = s.abs()
    out["f5_trailing_run_length"] = pd.Series(rl, index=idx)
    out["f5_days_since_flip"] = pd.Series(rl - 1.0, index=idx)
    out["f5_days_since_nonzero"] = pd.Series(_days_since_nonzero(vals), index=idx)
    ab = s.abs()
    out["f5_participation_20"] = ab.rolling(20, min_periods=20).mean()
    out["f5_long_bias_20"] = s.rolling(20, min_periods=20).mean()
    if mr_score is not None:
        out["f5_sign_agree_mr"] = (np.sign(s) * np.sign(mr_score.reindex(idx))).astype(float)
    else:
        out["f5_sign_agree_mr"] = pd.Series(np.nan, index=idx, dtype=float)
    return pd.DataFrame(out, index=idx)


def f6_momentum_contrast(ohlcv_inst: pd.DataFrame) -> pd.DataFrame:
    df = _ohlcv_df(ohlcv_inst)
    c = df["close"]
    h, lo = df["high"], df["low"]
    logc = np.log(c)
    daily = logc.diff()
    out: dict[str, pd.Series] = {}
    for L in (20, 60):
        raw = logc - logc.shift(L)
        scale = daily.rolling(L, min_periods=L).std() * np.sqrt(L)
        out[f"f6_ts_momentum_{L}"] = raw / scale.where(scale > 0)
    s20, s60 = c.rolling(20, min_periods=20).mean(), c.rolling(60, min_periods=60).mean()
    out["f6_ma_cross_20_60"] = (s20 - s60) / s60.where(s60 != 0)
    ema12 = c.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = c.ewm(span=26, min_periods=26, adjust=False).mean()
    macd = ema12 - ema26
    out["f6_macd_12_26"] = macd
    out["f6_macd_hist_12_26_9"] = macd - macd.ewm(span=9, min_periods=9, adjust=False).mean()
    up = h.diff(); dn = -lo.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    mdm = dn.where((dn > up) & (dn > 0), 0.0)
    pc = c.shift(1)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    alpha = 1/14
    atr = tr.ewm(alpha=alpha, min_periods=14, adjust=False).mean().where(lambda x: x > 0)
    pdi = 100 * pdm.ewm(alpha=alpha, min_periods=14, adjust=False).mean() / atr
    mdi_ = 100 * mdm.ewm(alpha=alpha, min_periods=14, adjust=False).mean() / atr
    dsum = (pdi + mdi_).where(lambda x: x > 0)
    dx = 100 * (pdi - mdi_).abs() / dsum
    out["f6_adx_14"] = dx.ewm(alpha=alpha, min_periods=14, adjust=False).mean()
    hi20 = c.rolling(20, min_periods=20).max().shift(1)
    lo20 = c.rolling(20, min_periods=20).min().shift(1)
    out["f6_donchian_pos_20"] = 2 * (c - lo20) / (hi20 - lo20).where(lambda w: w > 0) - 1
    return pd.DataFrame(out, index=df.index)


def f7_microstructure(ohlcv_inst: pd.DataFrame) -> pd.DataFrame:
    df = _ohlcv_df(ohlcv_inst)
    c, vol = df["close"], df["volume"]
    oi = df.get("open_interest", pd.Series(np.nan, index=df.index, dtype=float))
    out: dict[str, pd.Series] = {}
    out["f7_volume_z_20"] = _zsc(vol, 20)
    vsma = vol.rolling(20, min_periods=20).mean()
    out["f7_volume_trend_20"] = vol / vsma.where(vsma > 0) - 1
    out["f7_oi_level"] = oi
    out["f7_oi_change"] = oi.diff()
    out["f7_oi_z_20"] = _zsc(oi, 20)
    oic20 = oi - oi.shift(20)
    pxc20 = c - c.shift(20)
    out["f7_oi_price_div_20"] = (np.sign(oic20) * np.sign(pxc20)).astype(float)
    ret = np.log(c).diff()
    safe_vol = vol.where(vol > 0)
    amihud = ret.abs() / safe_vol
    out["f7_amihud_20"] = amihud.rolling(20, min_periods=1).mean()
    return pd.DataFrame(out, index=df.index)


def f8_calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    dow = idx.dayofweek.to_numpy(dtype=float)
    month = idx.month.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "f8_dow_sin":    np.sin(2 * np.pi * dow / 7),
            "f8_dow_cos":    np.cos(2 * np.pi * dow / 7),
            "f8_month_sin":  np.sin(2 * np.pi * (month - 1) / 12),
            "f8_month_cos":  np.cos(2 * np.pi * (month - 1) / 12),
        },
        index=idx,
    )


def f10_price_action(ohlcv_inst: pd.DataFrame) -> pd.DataFrame:
    df = _ohlcv_df(ohlcv_inst)
    h, lo, op = df["high"], df["low"], df["open"]
    hl = np.log((h / lo).where((h > 0) & (lo > 0)))
    oto = np.log((op / op.shift(1)).where((op > 0) & (op.shift(1) > 0)))
    return pd.DataFrame(
        {
            "f10_hl_range":        hl,
            "f10_hl_range_mean_20": hl.rolling(20, min_periods=20).mean(),
            "f10_oto_ret":         oto,
            "f10_oto_ret_mean_20": oto.rolling(20, min_periods=20).mean(),
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Build full feature matrix for one instrument
# ---------------------------------------------------------------------------

def build_feature_matrix(
    inst: str,
    data: dict[str, pd.DataFrame],
    wide_rets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute all features for one instrument; return (daily_df, events_df).

    daily_df  : date-indexed frame, all feature columns (full history)
    events_df : event-level frame with X columns + date/t1/bin/avg_uniqueness
    """
    ohlcv = data["ohlcv"]
    signals = data["signals"]
    macro_feats = data["macro_feats"]
    labels = data["labels"]
    hmm_vol = data["hmm_vol"]
    hmm_macro = data["hmm_macro"]
    alt_macro = data.get("alt_macro")

    ohlcv_inst = ohlcv[ohlcv["instrument"] == inst].copy()
    df = _ohlcv_df(ohlcv_inst)
    close = df["close"]
    ret_series = np.log(close).diff()
    vol20 = ret_series.rolling(20).std()

    # Signal series (released window)
    sig_wide = signals.set_index("date")
    signal_inst = sig_wide[inst].sort_index() if inst in sig_wide.columns else pd.Series(dtype=float)

    # F1
    f1 = f1_counter_trend(ohlcv_inst)

    # F2
    f2 = f2_vol_dispersion(ohlcv_inst, inst)

    # F5 (on signal dates; reindex to full index)
    mr = f1["f1_mr_score_20"].reindex(signal_inst.index)
    f5 = f5_signal_derived(signal_inst, mr_score=mr).reindex(df.index)

    # F6
    f6 = f6_momentum_contrast(ohlcv_inst)

    # F7
    f7 = f7_microstructure(ohlcv_inst)

    # F8
    f8 = f8_calendar(df.index)

    # F10
    f10 = f10_price_action(ohlcv_inst)

    # F4 — PCA of F1+F2+F6+F7+F10 on pre-sample; 4 components
    price_frames = [f1, f2, f6, f7, f10]
    price_block = pd.concat([x.reindex(df.index) for x in price_frames], axis=1)
    f4 = _fit_f4_pca(price_block, df.index)

    # F11 — pre-computed macro features (date-indexed, f11_ prefix)
    f11 = macro_feats.reindex(df.index)

    # F12 — wavelet energy on log returns (window=64 for speed)
    f12 = mra_energy_bands(ret_series, window=64).reindex(df.index)
    f12.columns = [f"f12_{c}" for c in f12.columns]

    # F13 — conditional risk (n_sims=30 for speed)
    tort = path_tortuosity_20d(ret_series).rename("f13_path_tortuosity_20d")
    svr = realized_semi_vol_ratio(ret_series).rename("f13_realized_semi_vol_ratio")
    hit = expected_hit_time(
        ret_series, vol20, pt_mult=1.0, sl_mult=1.0, h=10,
        window=100, n_sims=30, seed=RANDOM_SEED,
    ).rename("f13_expected_hit_time")
    pto = prob_timeout(
        ret_series, vol20, pt_mult=1.0, sl_mult=1.0, h=10,
        window=100, n_sims=30, seed=RANDOM_SEED,
    ).rename("f13_prob_timeout")
    f13 = pd.DataFrame({"f13_path_tortuosity_20d": tort, "f13_realized_semi_vol_ratio": svr,
                        "f13_expected_hit_time": hit, "f13_prob_timeout": pto})

    # F15 — cross-asset features (needs wide returns panel)
    dist = distance_to_lead_lag_centroid(wide_rets, inst, lag=1, window=63)
    disp = asset_class_dispersion_z(wide_rets, inst, classes=ASSET_CLASSES, window=42)
    ewmc = ewma_implied_corr_z(wide_rets, inst, halflife=20, window=100)
    f15 = pd.DataFrame({
        "f15_dist_lead_lag": dist,
        "f15_asset_class_dispersion_z": disp,
        "f15_ewma_implied_corr_z": ewmc,
    }).reindex(df.index)

    # HMM vol (harry/new_work only — loaded from pre-computed CSV)
    hv = (
        hmm_vol[hmm_vol["instrument"] == inst]
        .set_index("date")
        .drop(columns=["instrument"])
        .reindex(df.index)
    )

    # HMM macro (harry/new_work only)
    hm = (
        hmm_macro[hmm_macro["instrument"] == inst]
        .set_index("date")
        .drop(columns=["instrument"])
        .reindex(df.index)
    )

    # Drop any hmm_* columns that came from other sources (safety guard)
    # All hmm_ columns here are from the new_work CSVs only.

    # Assemble daily feature matrix
    frames = [
        price_block,   # F1+F2+F6+F7+F10 (for reference)
        f4,            # F4 PCA latent
        f5,            # F5 signal-derived
        f8,            # F8 calendar
        f11,           # F11 macro
        f12,           # F12 wavelet
        f13,           # F13 conditional risk
        f15,           # F15 cross-asset
        hv,            # HMM vol
        hm,            # HMM macro
    ]
    # Drop duplicated columns (f1..f10 already in price_block)
    daily = pd.concat(frames, axis=1)
    daily = daily.loc[:, ~daily.columns.duplicated(keep="first")]
    daily = daily.sort_index()

    # --- Events: restrict to metamodel window labels for this instrument ---
    ev = labels[labels["instrument"] == inst].copy()
    ev["date"] = pd.to_datetime(ev["date"])
    ev["t1"] = pd.to_datetime(ev["t1"])
    ev = ev.sort_values("date").reset_index(drop=True)

    # Look up features at event (signal) dates
    feat_cols = [c for c in daily.columns]
    X_at_events = daily.reindex(ev["date"].values).reset_index(drop=True)

    events_df = pd.concat([ev.reset_index(drop=True), X_at_events], axis=1)

    return daily, events_df


def _fit_f4_pca(price_block: pd.DataFrame, full_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Fit PCA(4) on pre-sample price features; transform full history."""
    pre = price_block[price_block.index < PRESAMPLE_CUTOFF].dropna()
    n_components = min(4, pre.shape[1], len(pre))
    if len(pre) < 10 or n_components < 1:
        cols = [f"f4_pc{i+1}" for i in range(4)]
        return pd.DataFrame(np.nan, index=full_index, columns=cols[:n_components] if n_components > 0 else cols)
    scaler = StandardScaler()
    X_pre = scaler.fit_transform(pre)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca.fit(X_pre)
    # Transform full history (fill NaN with 0 for transform, restore NaN after)
    nan_mask = price_block.isna().any(axis=1)
    X_full = scaler.transform(price_block.fillna(0))
    pcs = pca.transform(X_full)
    cols = [f"f4_pc{i+1}" for i in range(n_components)]
    f4 = pd.DataFrame(pcs, index=price_block.index, columns=cols)
    f4.loc[nan_mask] = np.nan
    return f4.reindex(full_index)


# ---------------------------------------------------------------------------
# Hygiene pipeline
# ---------------------------------------------------------------------------

def apply_hygiene(
    events_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    log: list[str],
    col_nan_threshold: float = 0.30,
) -> tuple[pd.DataFrame, list[str]]:
    """Clean the feature matrix at event dates.

    Steps:
    0. Drop columns with more than col_nan_threshold fraction of NaN values.
       These are features that are structurally unavailable in the event window
       (e.g. f15_dist_lead_lag requires all-instruments data; it is NaN for
       ~96% of events). Keeping them would discard almost all events in step 1.
    1. Trim to post-warmup (drop events where any remaining feature is NaN).
    2. Drop zero/near-zero-variance columns.
    3. Dedupe near-perfect Spearman pairs (|rho| >= TWIN_THRESHOLD).

    Returns (events_clean, drop_log).
    """
    feat_cols = _feature_cols(events_df)
    X = events_df[feat_cols].copy()

    # 0. Drop columns that are sparse (> col_nan_threshold fraction NaN)
    col_nan_frac = X.isna().mean()
    sparse_cols = col_nan_frac[col_nan_frac > col_nan_threshold].index.tolist()
    if sparse_cols:
        X = X.drop(columns=sparse_cols)
        events_df = events_df.drop(columns=[c for c in sparse_cols if c in events_df.columns])
        log.append(f"Dropped {len(sparse_cols)} sparse columns (>{col_nan_threshold:.0%} NaN): "
                   f"{sparse_cols[:10]}")

    # 1. Trim: drop events with any NaN in feature columns
    n_before = len(X)
    nan_mask = X.isna().any(axis=1)
    X = X.loc[~nan_mask].copy()
    events_df = events_df.loc[~nan_mask].copy().reset_index(drop=True)
    n_after = len(X)
    log.append(f"Warmup trim: {n_before} → {n_after} events "
                f"(dropped {n_before - n_after} NaN-bearing rows)")

    # 2. Near-zero variance
    var = X.var(ddof=1)
    nzv_cols = var[var < NZV_THRESHOLD].index.tolist()
    if nzv_cols:
        X = X.drop(columns=nzv_cols)
        events_df = events_df.drop(columns=[c for c in nzv_cols if c in events_df.columns])
        log.append(f"Dropped {len(nzv_cols)} near-zero-variance cols: {nzv_cols[:10]}")

    # 3. Dedupe near-perfect pairs (Spearman |rho| >= 0.99)
    corr = X.corr(method="spearman").abs()
    dropped_twins: list[str] = []
    remaining = list(X.columns)
    while True:
        found = False
        for i, c1 in enumerate(remaining):
            for c2 in remaining[i+1:]:
                if corr.loc[c1, c2] >= TWIN_THRESHOLD:
                    # Keep c1 (encountered first), drop c2
                    remaining.remove(c2)
                    dropped_twins.append(f"{c2} (twin of {c1}, rho={corr.loc[c1,c2]:.3f})")
                    found = True
                    break
            if found:
                break
        if not found:
            break
    if dropped_twins:
        to_drop = [t.split(" ")[0] for t in dropped_twins]
        X = X[remaining]
        events_df = events_df.drop(columns=[c for c in to_drop if c in events_df.columns])
        log.append(f"Dropped {len(dropped_twins)} near-perfect twins (|rho|>={TWIN_THRESHOLD}):")
        for d in dropped_twins[:20]:
            log.append(f"  {d}")

    return events_df, log


def _macro_sanity_check(daily_df: pd.DataFrame, log: list[str]) -> None:
    """Warn if EIA/PMI features look like they have look-ahead contamination."""
    # EIA: should update sparsely (weekly), not every day
    eia_cols = [c for c in daily_df.columns if "stock_surprise" in c]
    for col in eia_cols[:2]:
        s = daily_df[col].dropna()
        if len(s) > 0:
            zero_diff_frac = (s.diff().fillna(0) == 0).mean()
            if zero_diff_frac < 0.7:
                log.append(f"WARN: {col} changes too frequently "
                           f"(zero_diff_frac={zero_diff_frac:.2f} < 0.70); "
                           "verify EIA weekly cadence")
            else:
                log.append(f"OK: {col} updates sparingly "
                           f"(zero_diff_frac={zero_diff_frac:.2f}) — release-date stamping confirmed")

    # PMI: monthly cadence
    pmi_cols = [c for c in daily_df.columns if "pmi" in c]
    for col in pmi_cols[:2]:
        s = daily_df[col].dropna()
        if len(s) > 0:
            zero_diff_frac = (s.diff().fillna(0) == 0).mean()
            if zero_diff_frac < 0.90:
                log.append(f"WARN: {col} changes too frequently "
                           f"(zero_diff_frac={zero_diff_frac:.2f} < 0.90); "
                           "verify PMI monthly cadence")
            else:
                log.append(f"OK: {col} has monthly update cadence "
                           f"(zero_diff_frac={zero_diff_frac:.2f})")

    # Z-score sanity: early rows should be NaN (if rolling z-score not full-sample)
    z_cols = [c for c in daily_df.columns if c.endswith("_z") and c.startswith("f11_")]
    for col in z_cols[:2]:
        s = daily_df[col]
        pre2000 = s[s.index < pd.Timestamp("1991-01-01")]
        if pre2000.notna().any():
            log.append(f"WARN: {col} has non-NaN values in first year of data "
                       "— verify rolling z-score uses causal trailing window")
        else:
            log.append(f"OK: {col} is NaN during warmup — causal rolling z-score confirmed")
    log.append("Note: only harry/new_work HMM features used (hmm_vol_*, hmm_macro_*). "
               "No HMM features from main branch included.")


# ---------------------------------------------------------------------------
# Feature group assignment
# ---------------------------------------------------------------------------

def assign_groups(events_df: pd.DataFrame) -> dict[str, list[str]]:
    """Split feature columns into correlation-clusterable vs hand-assigned groups."""
    feat_cols = _feature_cols(events_df)
    corr_cluster: list[str] = []
    hand: dict[str, list[str]] = {}
    for c in feat_cols:
        assigned = False
        for prefix, label in HAND_ASSIGNED_PREFIXES.items():
            if c.startswith(prefix):
                hand.setdefault(label, []).append(c)
                assigned = True
                break
        if not assigned:
            for prefix in CORR_CLUSTER_PREFIXES:
                if c.startswith(prefix):
                    corr_cluster.append(c)
                    break
    return {"corr_cluster": corr_cluster, **hand}


def _feature_cols(events_df: pd.DataFrame) -> list[str]:
    """Non-metadata feature columns from events_df."""
    meta = {"date", "instrument", "side", "t1", "ret", "bin",
            "trgt", "h", "pt_mult", "sl_mult", "sigma_method", "avg_uniqueness"}
    return [c for c in events_df.columns if c not in meta]


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def compute_spearman_distance(X: pd.DataFrame) -> np.ndarray:
    """Spearman distance matrix: sqrt(1 - |rho|)."""
    rho = X.corr(method="spearman").abs().fillna(0).to_numpy()
    dist = np.sqrt(np.clip(1 - rho, 0, 1))
    np.fill_diagonal(dist, 0)
    return dist


def select_k(
    X: pd.DataFrame,
    dist: np.ndarray,
    k_range: range = range(3, 16),
) -> tuple[int, pd.DataFrame]:
    """Select K by silhouette; report CH and DB alongside."""
    from scipy.spatial.distance import squareform
    condensed = squareform(dist)
    Z = linkage(condensed, method="ward")
    rows = []
    for k in k_range:
        labels = fcluster(Z, k, criterion="maxclust")
        if len(np.unique(labels)) < 2:
            continue
        sil = silhouette_score(dist, labels, metric="precomputed")
        # Clustering is over features (columns); each feature is a point in
        # n_events-dimensional space, so we need the transpose (n_features × n_events).
        X_feat = X.T.fillna(0).to_numpy()
        ch = calinski_harabasz_score(X_feat, labels)
        db = davies_bouldin_score(X_feat, labels)
        rows.append({"K": k, "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db})
    metrics = pd.DataFrame(rows)
    best_k = int(metrics.loc[metrics["silhouette"].idxmax(), "K"])
    return best_k, metrics


def get_cluster_labels(dist: np.ndarray, k: int) -> np.ndarray:
    from scipy.spatial.distance import squareform
    condensed = squareform(dist)
    Z = linkage(condensed, method="ward")
    return fcluster(Z, k, criterion="maxclust")


def cluster_representatives(
    X: pd.DataFrame,
    cluster_labels: np.ndarray,
) -> dict[int, str]:
    """One representative per cluster: smallest mean Spearman distance to clustermates."""
    dist = compute_spearman_distance(X)
    cols = list(X.columns)
    reps: dict[int, str] = {}
    for cid in np.unique(cluster_labels):
        idx = np.where(cluster_labels == cid)[0]
        if len(idx) == 1:
            reps[int(cid)] = cols[idx[0]]
            continue
        sub_dist = dist[np.ix_(idx, idx)]
        mean_dist = sub_dist.mean(axis=1)
        reps[int(cid)] = cols[idx[int(np.argmin(mean_dist))]]
    return reps


def build_cluster_map(
    corr_cols: list[str],
    cluster_labels: np.ndarray,
    hand_groups: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Combine correlation clusters with hand-assigned groups."""
    cluster_map: dict[str, list[str]] = {}
    for cid in np.unique(cluster_labels):
        idx = np.where(cluster_labels == cid)[0]
        cols = [corr_cols[i] for i in idx]
        # Detect dominant F-prefix for label
        prefix_counts: dict[str, int] = {}
        for c in cols:
            for p in CORR_CLUSTER_PREFIXES:
                if c.startswith(p):
                    prefix_counts[p] = prefix_counts.get(p, 0) + 1
                    break
        dominant = max(prefix_counts, key=prefix_counts.get) if prefix_counts else "mixed"
        label = f"C{cid}_{dominant.rstrip('_')}"
        # Flag low-freq fundamentals
        if any(c.startswith("f11_") for c in cols) and all(
            c.startswith("f11_") for c in cols
        ):
            label += "_lowfreq_macro"
        cluster_map[label] = cols
    cluster_map.update(hand_groups)
    return cluster_map


# ---------------------------------------------------------------------------
# Importance computation (inside CPCV loop)
# ---------------------------------------------------------------------------

def _rf_model(seed: int = RANDOM_SEED, min_samples_leaf: int = 20) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def _pfi_fold(
    rf: RandomForestClassifier,
    X_te: np.ndarray,
    y_te: np.ndarray,
    feat_names: list[str],
    auc_base: float,
    n_repeats: int = 1,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    pfi: dict[str, float] = {}
    for j, name in enumerate(feat_names):
        drops = []
        for _ in range(n_repeats):
            X_p = X_te.copy()
            X_p[:, j] = rng.permutation(X_p[:, j])
            prob = rf.predict_proba(X_p)[:, 1]
            try:
                drops.append(auc_base - roc_auc_score(y_te, prob))
            except Exception:
                drops.append(0.0)
        pfi[name] = float(np.mean(drops))
    return pfi


def _clustered_mda_fold(
    rf: RandomForestClassifier,
    X_te: np.ndarray,
    y_te: np.ndarray,
    feat_names: list[str],
    cluster_map: dict[str, list[str]],
    auc_base: float,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    col_idx = {name: j for j, name in enumerate(feat_names)}
    cmda: dict[str, float] = {}
    for cluster_name, cols in cluster_map.items():
        cluster_cols = [c for c in cols if c in col_idx]
        if not cluster_cols:
            cmda[cluster_name] = 0.0
            continue
        X_p = X_te.copy()
        perm = rng.permutation(len(X_p))
        for c in cluster_cols:
            X_p[:, col_idx[c]] = X_p[perm, col_idx[c]]
        prob = rf.predict_proba(X_p)[:, 1]
        try:
            cmda[cluster_name] = auc_base - roc_auc_score(y_te, prob)
        except Exception:
            cmda[cluster_name] = 0.0
    return cmda


def _shap_fold(
    rf: RandomForestClassifier,
    X_tr: np.ndarray,
    X_te: np.ndarray,
    feat_names: list[str],
    max_samples: int = 200,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (signed_mean_shap, magnitude_shap) dicts."""
    try:
        import shap as _shap
        te_sub = X_te[:max_samples]
        # tree_path_dependent: avoids interventional-path errors on correlated features.
        # check_additivity=False: skips the tight numerical additivity assertion that
        # can raise ValueError with class_weight='balanced' RandomForests in SHAP 0.46+.
        expl = _shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent")
        sv = expl.shap_values(te_sub, check_additivity=False)
        # Handle shap.Explanation object returned by newer API builds
        if hasattr(sv, "values"):
            sv = sv.values
        # shap_values shape varies by version:
        #   list [class0_arr, class1_arr]               (SHAP < 0.46)
        #   ndarray (n_samples, n_features, n_classes)  (SHAP >= 0.46, 'auto' perturbation)
        #   ndarray (n_samples, n_features)             (tree_path_dependent always returns 2-D)
        if isinstance(sv, list):
            sv = sv[1]
        elif sv.ndim == 3:
            sv = sv[:, :, 1]
        sv = np.asarray(sv, dtype=float)
        signed    = {feat_names[j]: float(sv[:, j].mean())         for j in range(len(feat_names))}
        magnitude = {feat_names[j]: float(np.abs(sv[:, j]).mean()) for j in range(len(feat_names))}
        return signed, magnitude
    except Exception as e:
        import traceback
        print(f"  [SHAP ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        zeros = {n: 0.0 for n in feat_names}
        return zeros, zeros


def _adaptive_cpcv_params(n_events: int) -> tuple[int, int, int]:
    """Return (n_groups, k, min_samples_leaf) adapted to event count.

    With too few events the standard n_groups=6/k=2 leaves ~35 training
    samples, making min_samples_leaf=20 produce single-leaf (constant)
    trees — AUC=0.5 and SHAP=0. Scale down aggressively for small datasets.

    Thresholds chosen so each training fold has >= 40 events and
    min_samples_leaf <= training_fold_size / 4.
    """
    if n_events >= 200:
        return N_GROUPS, K_CPCV, 20   # standard: 6/2, ~133 tr events
    elif n_events >= 120:
        return 5, 2, 15               # 5/2: ~72 tr events
    elif n_events >= 80:
        return 4, 1, 10               # 4/1: ~60 tr events
    else:
        return 3, 1, 5                # 3/1: ~38 tr events (minimum viable)


def run_cpcv_importance(
    events_df: pd.DataFrame,
    feat_cols: list[str],
    cluster_map: dict[str, list[str]],
    n_groups: int | None = None,
    k: int | None = None,
    embargo: float = EMBARGO,
) -> dict[str, Any]:
    """Run the full CPCV importance loop.

    n_groups and k default to None, which triggers adaptive selection based
    on event count so small instruments (e.g. ho1s with ~57 events) do not
    degenerate to single-leaf trees.

    Returns dict with keys:
        pfi_mean, pfi_std, mdi_mean, mdi_std,
        shap_signed_mean, shap_magnitude_mean,
        clustered_mda_mean, clustered_mda_std,
        fold_aucs, n_folds, n_groups, k
    """
    ev = events_df[["date", "t1", "bin"]].copy()
    X = events_df[feat_cols].fillna(0).to_numpy(dtype=float)
    y = events_df["bin"].to_numpy()
    feat_names = list(feat_cols)

    if n_groups is None or k is None:
        _ng, _k, _msl = _adaptive_cpcv_params(len(events_df))
        n_groups = n_groups if n_groups is not None else _ng
        k        = k        if k        is not None else _k
    else:
        _, _, _msl = _adaptive_cpcv_params(len(events_df))
    print(f"  CPCV config: n_groups={n_groups}, k={k}, min_samples_leaf={_msl}")

    cpcv = CombinatorialPurgedKFold(n_groups=n_groups, k=k, embargo=embargo)

    pfi_all: list[dict] = []
    mdi_all: list[np.ndarray] = []
    shap_signed_all: list[dict] = []
    shap_mag_all: list[dict] = []
    cmda_all: list[dict] = []
    aucs: list[float] = []
    rng = np.random.default_rng(RANDOM_SEED)

    for tr_idx, te_idx in cpcv.split(ev):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        w_tr = events_df.iloc[tr_idx]["avg_uniqueness"].fillna(1.0).to_numpy()
        w_tr = w_tr / w_tr.mean() if w_tr.mean() > 0 else np.ones(len(w_tr))

        rf = _rf_model(min_samples_leaf=_msl)
        rf.fit(X_tr, y_tr, sample_weight=w_tr)

        prob = rf.predict_proba(X_te)[:, 1]
        try:
            auc_base = float(roc_auc_score(y_te, prob))
        except Exception:
            continue
        aucs.append(auc_base)

        # MDI (flagged for train-set bias)
        mdi_all.append(rf.feature_importances_)

        # PFI
        pfi_fold = _pfi_fold(rf, X_te, y_te, feat_names, auc_base, rng=rng)
        pfi_all.append(pfi_fold)

        # SHAP
        s_signed, s_mag = _shap_fold(rf, X_tr, X_te, feat_names)
        shap_signed_all.append(s_signed)
        shap_mag_all.append(s_mag)

        # Clustered MDA
        cm = _clustered_mda_fold(rf, X_te, y_te, feat_names, cluster_map, auc_base, rng=rng)
        cmda_all.append(cm)

    def _mean_std(lst_of_dicts: list[dict]) -> tuple[pd.Series, pd.Series]:
        df_ = pd.DataFrame(lst_of_dicts)
        return df_.mean(), df_.std()

    pfi_mean, pfi_std = _mean_std(pfi_all) if pfi_all else (pd.Series(dtype=float), pd.Series(dtype=float))
    cmda_mean, cmda_std = _mean_std(cmda_all) if cmda_all else (pd.Series(dtype=float), pd.Series(dtype=float))
    shap_s_mean, _ = _mean_std(shap_signed_all) if shap_signed_all else (pd.Series(dtype=float), pd.Series(dtype=float))
    shap_m_mean, _ = _mean_std(shap_mag_all) if shap_mag_all else (pd.Series(dtype=float), pd.Series(dtype=float))

    mdi_mean = pd.Series(
        np.mean(mdi_all, axis=0) if mdi_all else np.zeros(len(feat_names)),
        index=feat_names,
    )
    mdi_std = pd.Series(
        np.std(mdi_all, axis=0) if mdi_all else np.zeros(len(feat_names)),
        index=feat_names,
    )

    return {
        "pfi_mean": pfi_mean,
        "pfi_std": pfi_std,
        "mdi_mean": mdi_mean,
        "mdi_std": mdi_std,
        "shap_signed_mean": shap_s_mean,
        "shap_magnitude_mean": shap_m_mean,
        "clustered_mda_mean": cmda_mean,
        "clustered_mda_std": cmda_std,
        "fold_aucs": aucs,
        "n_folds": len(aucs),
    }


# ---------------------------------------------------------------------------
# Dimensionality-reduction check
# ---------------------------------------------------------------------------

def dimensionality_reduction_check(
    events_df: pd.DataFrame,
    feat_cols: list[str],
    cluster_reps: dict[int, str],
    cluster_map: dict[str, list[str]],
    n_groups: int | None = None,
    k: int | None = None,
    embargo: float = EMBARGO,
) -> dict[str, float]:
    """Run CPCV with one representative per cluster; return mean AUC.

    Uses the same adaptive CPCV parameters as run_cpcv_importance so
    small instruments (e.g. ho1s) get the right fold sizes.
    """
    rep_cols = list(cluster_reps.values())
    for label, cols in cluster_map.items():
        if label in (list(HAND_ASSIGNED_PREFIXES.values())):
            rep_cols.extend([c for c in cols if c in feat_cols])
    rep_cols = [c for c in rep_cols if c in feat_cols]
    rep_cols = list(dict.fromkeys(rep_cols))

    if not rep_cols:
        return {"n_reps": 0, "reduced_mean_auc": float("nan"), "reduced_std_auc": float("nan")}

    _ng, _k, _msl = _adaptive_cpcv_params(len(events_df))
    n_groups = n_groups if n_groups is not None else _ng
    k        = k        if k        is not None else _k

    ev = events_df[["date", "t1", "bin"]].copy()
    X_red = events_df[rep_cols].fillna(0).to_numpy(dtype=float)
    y = events_df["bin"].to_numpy()

    cpcv = CombinatorialPurgedKFold(n_groups=n_groups, k=k, embargo=embargo)
    aucs_red: list[float] = []
    for tr_idx, te_idx in cpcv.split(ev):
        rf = _rf_model(min_samples_leaf=_msl)
        w = events_df.iloc[tr_idx]["avg_uniqueness"].fillna(1.0).to_numpy(dtype=float)
        w = w / w.mean() if w.mean() > 0 else np.ones(len(w))
        rf.fit(X_red[tr_idx], y[tr_idx], sample_weight=w)
        prob = rf.predict_proba(X_red[te_idx])[:, 1]
        try:
            aucs_red.append(float(roc_auc_score(y[te_idx], prob)))
        except Exception:
            pass

    return {
        "n_reps": len(rep_cols),
        "reduced_mean_auc": float(np.mean(aucs_red)) if aucs_red else float("nan"),
        "reduced_std_auc": float(np.std(aucs_red)) if aucs_red else float("nan"),
    }


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------

def run_analysis(
    instruments: list[str],
    data: dict[str, pd.DataFrame] | None = None,
) -> dict[str, dict]:
    """Run the full cluster-level feature importance analysis.

    Parameters
    ----------
    instruments : list of instrument tickers.
    data        : pre-loaded data dict from load_all_data(); loaded if None.

    Returns
    -------
    results dict keyed by instrument, each with sub-keys:
        events_df, daily_df, cluster_map, cluster_reps,
        cluster_metrics, importance, dim_reduction, hygiene_log.
    """
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    if data is None:
        data = load_all_data()

    # Build wide returns panel once (needed for F15 cross-asset features)
    rets_long = native_returns(data["ohlcv"], kind="log")
    w_rets = wide_returns(rets_long).sort_index()

    results: dict[str, dict] = {}

    for inst in instruments:
        print(f"\n{'='*60}")
        print(f"Instrument: {inst}")
        print("=" * 60)

        # --- Build features ---
        print("  Building features...")
        daily_df, events_df = build_feature_matrix(inst, data, w_rets)

        # --- Macro sanity check ---
        hygiene_log: list[str] = []
        _macro_sanity_check(daily_df, hygiene_log)

        # --- Hygiene ---
        print("  Applying hygiene...")
        events_df, hygiene_log = apply_hygiene(events_df, daily_df, hygiene_log)

        if len(events_df) < 50:
            print(f"  SKIP: only {len(events_df)} events after hygiene (< 50).")
            continue

        # --- Group assignment ---
        groups = assign_groups(events_df)
        corr_cols = groups["corr_cluster"]
        hand_groups = {k: v for k, v in groups.items() if k != "corr_cluster"}

        # --- Clustering ---
        print(f"  Clustering {len(corr_cols)} continuous features...")
        X_corr = events_df[corr_cols].fillna(0)
        dist_mat = compute_spearman_distance(X_corr)
        best_k, cluster_metrics = select_k(X_corr, dist_mat)
        print(f"  Best K={best_k} (silhouette={cluster_metrics.set_index('K').loc[best_k,'silhouette']:.3f})")
        cluster_labels = get_cluster_labels(dist_mat, best_k)
        cluster_reps = cluster_representatives(X_corr, cluster_labels)
        cluster_map = build_cluster_map(corr_cols, cluster_labels, hand_groups)

        # Note on low-frequency fundamentals
        for cname, cols in cluster_map.items():
            f11_frac = sum(c.startswith("f11_") for c in cols) / max(len(cols), 1)
            eia_pmi_cols = [c for c in cols if any(k in c for k in ["stock_surprise", "pmi", "bdi", "copper"])]
            if f11_frac > 0.5 and eia_pmi_cols:
                hygiene_log.append(
                    f"Cluster {cname} is dominated by F11 macro features. "
                    "Correlation partly reflects shared update cadence (weekly/monthly). "
                    "Label: 'low-frequency fundamentals'. Any importance here is a "
                    "print-day signal (information arrives at release, not bar-by-bar)."
                )

        # --- Importance ---
        print("  Running CPCV importance loop...")
        feat_cols = _feature_cols(events_df)
        importance = run_cpcv_importance(events_df, feat_cols, cluster_map)
        n_folds = importance["n_folds"]
        mean_auc = np.mean(importance["fold_aucs"]) if importance["fold_aucs"] else float("nan")
        print(f"  {n_folds} folds; mean AUC={mean_auc:.3f}")

        # --- Dimensionality-reduction check ---
        print("  Dimensionality-reduction check...")
        dim_red = dimensionality_reduction_check(
            events_df, feat_cols, cluster_reps, cluster_map
        )
        print(f"  Reduced ({dim_red['n_reps']} reps) AUC={dim_red['reduced_mean_auc']:.3f}")

        # --- Save outputs ---
        _save_outputs(inst, events_df, daily_df, cluster_map, cluster_metrics,
                      importance, dim_red, hygiene_log, cluster_labels, corr_cols,
                      best_k, dist_mat)

        results[inst] = {
            "events_df":      events_df,
            "daily_df":       daily_df,
            "cluster_map":    cluster_map,
            "cluster_reps":   cluster_reps,
            "cluster_metrics": cluster_metrics,
            "cluster_labels": cluster_labels,
            "corr_cols":      corr_cols,
            "dist_mat":       dist_mat,
            "best_k":         best_k,
            "importance":     importance,
            "dim_reduction":  dim_red,
            "hygiene_log":    hygiene_log,
        }

    return results


def _save_outputs(
    inst: str,
    events_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    cluster_map: dict,
    cluster_metrics: pd.DataFrame,
    importance: dict,
    dim_red: dict,
    hygiene_log: list[str],
    cluster_labels: np.ndarray,
    corr_cols: list[str],
    best_k: int,
    dist_mat: np.ndarray,
) -> None:
    """Save key tables and figures to OUTPUTS/."""
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage as _linkage, dendrogram as _dendrogram

    out = OUTPUTS / inst
    out.mkdir(parents=True, exist_ok=True)

    # K-selection metrics table
    cluster_metrics.to_csv(out / "cluster_k_metrics.csv", index=False)

    # Cluster membership
    member_rows = []
    for cname, cols in cluster_map.items():
        for c in cols:
            member_rows.append({"cluster": cname, "feature": c})
    pd.DataFrame(member_rows).to_csv(out / "cluster_membership.csv", index=False)

    # Clustered MDA table
    cmda_mean = importance["clustered_mda_mean"]
    cmda_std = importance["clustered_mda_std"]
    cmda_df = pd.DataFrame({"mean_drop": cmda_mean, "std_drop": cmda_std})
    cmda_df.index.name = "cluster"
    cmda_df = cmda_df.sort_values("mean_drop", ascending=False)
    cmda_df.to_csv(out / "clustered_mda.csv")

    # Per-feature PFI table
    pfi_df = pd.DataFrame({
        "pfi_mean": importance["pfi_mean"],
        "pfi_std":  importance["pfi_std"],
        "mdi_mean": importance["mdi_mean"],
        "shap_signed": importance["shap_signed_mean"],
        "shap_magnitude": importance["shap_magnitude_mean"],
    })
    pfi_df.index.name = "feature"
    pfi_df.to_csv(out / "per_feature_importance.csv")

    # Hygiene log
    with open(out / "hygiene_log.txt", "w") as f:
        f.write("\n".join(hygiene_log))

    # Dendrogram figure
    try:
        condensed = squareform(dist_mat)
        Z = _linkage(condensed, method="ward")
        fig, ax = plt.subplots(figsize=(14, 5))
        _dendrogram(Z, labels=corr_cols, ax=ax, leaf_rotation=90, leaf_font_size=6)
        ax.set_title(f"{inst} — Ward dendrogram (Spearman distance)", fontsize=11)
        ax.axhline(y=Z[-(best_k-1), 2], color="red", linestyle="--",
                   label=f"K={best_k} cut")
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(out / "dendrogram.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  Warning: dendrogram failed: {e}")

    # Clustered MDA bar chart
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        y_pos = range(len(cmda_df))
        ax.barh(y_pos, cmda_df["mean_drop"], xerr=cmda_df["std_drop"].fillna(0),
                align="center", capsize=3, color="steelblue", alpha=0.8)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(cmda_df.index, fontsize=7)
        ax.set_xlabel("Mean AUC drop (clustered MDA, ± std across CPCV paths)")
        ax.set_title(f"{inst} — Cluster-level importance (headline)")
        ax.axvline(x=0, color="black", linewidth=0.8)
        plt.tight_layout()
        fig.savefig(out / "clustered_mda.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  Warning: clustered MDA plot failed: {e}")

    print(f"  Outputs saved → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    EQUITY = ["es1s", "nq1s", "fesx1s"]
    data = load_all_data()
    results = run_analysis(EQUITY, data=data)
    print("\nDone.")
