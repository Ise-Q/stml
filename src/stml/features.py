"""
features.py
===========
Causal feature engineering for the meta-model. Every feature at date ``t`` is
computed strictly from information available at time ``t`` (inclusive) — no
peeking forward.

This module covers feature groups G1 (volatility), G2 (trend), G3 (mean-reversion
/ noise), G5 (signal context), and G7 (calendar) at the **thin-pipeline depth**
(Stage 2a). Stage 3a extends G1-G3 with range-based vol estimators, the backward
trend-scanning t-value, Hurst, Amihud, microstructure (G4), and sector-relative
ranks. HMM/GMM regime features (G6) live in ``stml.regimes``.

Public API:
  - :func:`compute_features`     -- master function: events → per-event feature row
  - :func:`feature_groups`       -- declarative grouping of features → economic group

Conventions
-----------
- All per-instrument features are computed on the instrument's **native dense
  series** (no calendar reindexing) so that a return spanning a holiday is a
  real multi-day move, not a fabricated zero.
- Features are sampled at event dates ``t`` using ``.loc[t]`` lookup; if ``t``
  is missing from the series (e.g. mismatched calendar) we forward-fill
  causally up to ``t`` only.
- Output is a wide DataFrame keyed by event id, with columns = feature names.
- Standardisation is **per-instrument expanding-window z-score** for
  scale-dependent features. Bounded features (autocorrelation, efficiency
  ratio) are left raw. This keeps the panel poolable across very different
  instruments (oil vol ≫ equity vol in absolute terms).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 0. Feature-group registry                                                   #
# --------------------------------------------------------------------------- #
# Filled at the bottom of the file once all features are defined. Used by
# the Stage 4 cluster-level importance section (G* labels feed clustering).
FEATURE_GROUPS: dict[str, str] = {}


def _register(group: str, names: Sequence[str]) -> None:
    for n in names:
        FEATURE_GROUPS[n] = group


def feature_groups() -> dict[str, str]:
    """Return ``{feature_name: group_label}`` mapping."""
    return dict(FEATURE_GROUPS)


# --------------------------------------------------------------------------- #
# 1. Helpers                                                                  #
# --------------------------------------------------------------------------- #
def _instrument_close(ohlcv_long: pd.DataFrame, instrument: str, col: str = "close") -> pd.Series:
    """Sorted native ``col`` series for one instrument, indexed by date."""
    s = (
        ohlcv_long.loc[ohlcv_long["instrument"] == instrument]
        .set_index("date")[col]
        .sort_index()
    )
    return s[~s.index.duplicated(keep="last")]


def _log_ret(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def _expanding_zscore(s: pd.Series, min_periods: int = 60) -> pd.Series:
    """Per-instrument expanding-window z-score (causal). NaN until ``min_periods``."""
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std()
    return (s - mu) / sd.replace(0, np.nan)


def _sample_at(series: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill ``series`` up to each date, then sample at ``dates``.

    This handles cases where an event date is missing from the feature series
    (which shouldn't happen for the event's own instrument, but is defensive).
    Crucially the ffill is causal — values at ``t`` use only data ``<= t``.
    """
    aligned = series.reindex(series.index.union(dates).sort_values()).ffill()
    return aligned.reindex(dates)


# --------------------------------------------------------------------------- #
# 2. Per-instrument feature time series                                       #
# --------------------------------------------------------------------------- #
def _per_instrument_features(
    ohlcv_long: pd.DataFrame,
    instrument: str,
    vol_windows: tuple[int, ...] = (5, 21, 63),
    mom_windows: tuple[int, ...] = (5, 21, 63),
    ma_windows: tuple[int, ...] = (21, 63),
    autocorr_window: int = 21,
    efficiency_window: int = 21,
    variance_ratio_lag: int = 5,
    variance_ratio_window: int = 21,
    ewma_vol_span: int = 50,
    zscore_min_periods: int = 60,
    include_range_vol: bool = True,
    include_trend_scan: bool = True,
    include_microstructure: bool = True,
    include_hurst: bool = True,
    trend_scan_windows: tuple[int, ...] = (10, 21, 42),
    hurst_window: int = 100,
) -> pd.DataFrame:
    """Build all per-instrument time-series features for ONE instrument.

    Returns a DataFrame indexed by date with one column per feature. Every
    column at date ``t`` uses only data with index ``<= t``.
    """
    close = _instrument_close(ohlcv_long, instrument, "close")
    if close.empty:
        return pd.DataFrame()
    ret = _log_ret(close)
    ann = np.sqrt(252)

    # Also load O/H/L/V/OI for range-based vol and microstructure features.
    sub = (
        ohlcv_long.loc[ohlcv_long["instrument"] == instrument]
        .set_index("date")[["open", "high", "low", "close", "volume", "open_interest"]]
        .sort_index()
    )
    sub = sub[~sub.index.duplicated(keep="last")]
    o = sub["open"]
    h_ = sub["high"]
    l_ = sub["low"]
    c_ = sub["close"]
    v_ = sub["volume"].astype(float)
    oi_ = sub["open_interest"].astype(float)

    feats: dict[str, pd.Series] = {}

    # --- G1: Volatility / risk state -------------------------------------- #
    for w in vol_windows:
        feats[f"vol_{w}d"] = ret.rolling(w).std() * ann  # annualised
    feats[f"ewma_vol_{ewma_vol_span}"] = ret.ewm(span=ewma_vol_span,
                                                  min_periods=20,
                                                  adjust=False).std() * ann
    # Vol ratio: short / long  (>1 = vol regime expanding)
    feats["vol_ratio_5_63"] = (
        feats["vol_5d"] / feats["vol_63d"]
    ).replace([np.inf, -np.inf], np.nan)

    # Vol-of-vol (rolling std of 21d vol)
    feats["vol_of_vol_63"] = feats["vol_21d"].rolling(63).std()

    # Downside semivol (only negative returns)
    neg_ret = ret.where(ret < 0, 0.0)
    feats["semivol_21d"] = neg_ret.rolling(21).std() * ann

    # --- G2: Trend quality / momentum ------------------------------------- #
    for w in mom_windows:
        feats[f"mom_{w}d"] = ret.rolling(w).sum()
    for w in ma_windows:
        ma = close.rolling(w).mean()
        # Distance from MA in SIGMA units of the w-day move:
        #   log(close / MA) / (sigma_1d * sqrt(w))
        # Result is dimensionless ("how many w-day sigmas away from MA").
        feats[f"ma_dist_{w}d"] = (
            np.log(close / ma) / (ret.rolling(w).std() * np.sqrt(w))
        )

    # MA slope: 1d log-change in 21d MA, normalised by 1d return std (dimensionless).
    ma21 = close.rolling(21).mean()
    feats["ma21_slope"] = (
        np.log(ma21 / ma21.shift(1)) / ret.rolling(21).std()
    )

    # --- G3: Mean-reversion / noise --------------------------------------- #
    feats[f"autocorr_{autocorr_window}d"] = ret.rolling(autocorr_window).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if pd.Series(x).std() > 0 else np.nan,
        raw=True,
    )
    # Kaufman efficiency ratio: |net move| / sum(|moves|) over window
    def _eff_ratio(x: np.ndarray) -> float:
        net = abs(np.nansum(x))
        path = np.nansum(np.abs(x))
        return net / path if path > 0 else np.nan

    feats[f"efficiency_ratio_{efficiency_window}d"] = ret.rolling(
        efficiency_window
    ).apply(_eff_ratio, raw=True)

    # Variance ratio = Var(k-period return) / (k * Var(1-period return)).
    # >1 => trending; <1 => mean-reverting; =1 => random walk.
    def _var_ratio(x: np.ndarray, k: int = variance_ratio_lag) -> float:
        x = x[~np.isnan(x)]
        if len(x) < k + 5:
            return np.nan
        v1 = np.var(x, ddof=1)
        if v1 <= 0:
            return np.nan
        # k-period overlapping sums.
        s = pd.Series(x)
        k_ret = s.rolling(k).sum().dropna().values
        if len(k_ret) < 2:
            return np.nan
        vk = np.var(k_ret, ddof=1)
        return vk / (k * v1)

    feats[f"variance_ratio_{variance_ratio_lag}d_{variance_ratio_window}w"] = (
        ret.rolling(variance_ratio_window).apply(_var_ratio, raw=True)
    )

    # --- G1 extension: Range-based vol estimators (Parkinson, GK, RS) ---- #
    if include_range_vol:
        ln_hl = np.log(h_ / l_)
        ln_co = np.log(c_ / o)
        ln_hc = np.log(h_ / c_)
        ln_ho = np.log(h_ / o)
        ln_lc = np.log(l_ / c_)
        ln_lo = np.log(l_ / o)
        # Per-bar variance contributions (annualised after rolling-mean).
        park_bar = (ln_hl ** 2) / (4 * np.log(2))
        gk_bar = 0.5 * (ln_hl ** 2) - (2 * np.log(2) - 1) * (ln_co ** 2)
        rs_bar = ln_hc * ln_ho + ln_lc * ln_lo
        for w in (21,):
            feats[f"parkinson_vol_{w}d"] = np.sqrt(
                park_bar.rolling(w).mean().clip(lower=0)
            ) * ann
            feats[f"garman_klass_vol_{w}d"] = np.sqrt(
                gk_bar.rolling(w).mean().clip(lower=0)
            ) * ann
            feats[f"rogers_satchell_vol_{w}d"] = np.sqrt(
                rs_bar.rolling(w).mean().clip(lower=0)
            ) * ann

    # --- G2 extension: Backward trend-scanning t-statistic --------------- #
    # tValLinR (Programming Session 1): t-stat of slope of price-on-time
    # over a backward window. High |t| ⇒ statistically clean trend.
    if include_trend_scan:
        def _t_val_lin_r(y: np.ndarray) -> float:
            n = len(y)
            if n < 5 or np.all(np.isnan(y)):
                return np.nan
            y = y[~np.isnan(y)]
            n = len(y)
            x = np.arange(n, dtype=float)
            x_mean = x.mean()
            y_mean = y.mean()
            xy = ((x - x_mean) * (y - y_mean)).sum()
            xx = ((x - x_mean) ** 2).sum()
            if xx == 0:
                return np.nan
            beta = xy / xx
            resid = y - (y_mean + beta * (x - x_mean))
            sse = (resid ** 2).sum()
            if n <= 2 or sse <= 0:
                return np.nan
            sigma_sq = sse / (n - 2)
            se_beta = np.sqrt(sigma_sq / xx)
            if se_beta == 0:
                return np.nan
            return beta / se_beta

        log_close = np.log(close)
        for w in trend_scan_windows:
            feats[f"trend_tval_{w}d"] = log_close.rolling(w).apply(
                _t_val_lin_r, raw=True
            )

    # --- G3 extension: Hurst exponent (rolling rescaled-range) ----------- #
    # Hurst > 0.5 ⇒ trending / persistent; < 0.5 ⇒ mean-reverting; = 0.5 ⇒ random walk.
    if include_hurst:
        def _hurst_rs(x: np.ndarray) -> float:
            x = x[~np.isnan(x)]
            n = len(x)
            if n < 20:
                return np.nan
            # Use 4 chunk sizes (powers of 2 up to n) for the log-log fit.
            sizes = np.unique(np.geomspace(8, n // 2, num=5, dtype=int))
            sizes = sizes[sizes >= 8]
            if len(sizes) < 2:
                return np.nan
            rs_vals = []
            for s in sizes:
                n_chunks = n // s
                if n_chunks < 1:
                    continue
                # Compute R/S for each non-overlapping chunk and average.
                rs_chunk = []
                for k in range(n_chunks):
                    seg = x[k * s : (k + 1) * s]
                    Y = seg - seg.mean()
                    Z = Y.cumsum()
                    R = Z.max() - Z.min()
                    S = seg.std(ddof=1)
                    if S > 0:
                        rs_chunk.append(R / S)
                if rs_chunk:
                    rs_vals.append((s, np.mean(rs_chunk)))
            if len(rs_vals) < 2:
                return np.nan
            arr = np.asarray(rs_vals, dtype=float)
            log_s = np.log(arr[:, 0])
            log_rs = np.log(arr[:, 1])
            slope = np.polyfit(log_s, log_rs, 1)[0]
            return float(slope)

        feats[f"hurst_{hurst_window}d"] = ret.rolling(hurst_window).apply(
            _hurst_rs, raw=True
        )

    # --- G4: Microstructure / liquidity ---------------------------------- #
    if include_microstructure:
        # Volume z-score (per-instrument rolling 63d)
        feats["volume_z_63d"] = (
            (v_ - v_.rolling(63).mean()) / v_.rolling(63).std().replace(0, np.nan)
        )
        # Volume trend (21d log-volume slope, dimensionless)
        log_vol = np.log(v_.replace(0, np.nan))
        feats["volume_trend_21d"] = log_vol.diff(21) / 21.0
        # OI trend (21d log-OI slope)
        log_oi = np.log(oi_.replace(0, np.nan))
        feats["oi_trend_21d"] = log_oi.diff(21) / 21.0
        # Amihud illiquidity = mean(|return| / dollar-volume) over 21d
        dollar_vol = v_ * c_
        amihud = (ret.abs() / dollar_vol.replace(0, np.nan)).rolling(21).mean()
        feats["amihud_illiq_21d"] = amihud
        # High-low range relative to recent (Parkinson-like, range only)
        feats["hl_range_21d"] = (ln_hl ** 2).rolling(21).mean() if include_range_vol else (
            np.log(h_ / l_) ** 2
        ).rolling(21).mean()

    df = pd.DataFrame(feats)

    # Per-instrument expanding-window z-score for SCALE-DEPENDENT features.
    # Bounded / already-dimensionless features kept raw:
    raw_features = {
        f"autocorr_{autocorr_window}d",
        f"efficiency_ratio_{efficiency_window}d",
        f"variance_ratio_{variance_ratio_lag}d_{variance_ratio_window}w",
        "vol_ratio_5_63",
        f"hurst_{hurst_window}d",
    }
    # Trend t-values are already standardised (it's a t-statistic) — also raw.
    for w in trend_scan_windows:
        raw_features.add(f"trend_tval_{w}d")
    for col in df.columns:
        if col not in raw_features:
            df[f"z_{col}"] = _expanding_zscore(df[col], min_periods=zscore_min_periods)

    return df


# --------------------------------------------------------------------------- #
# 3. Signal-context features (G5) — per-instrument signal time series          #
# --------------------------------------------------------------------------- #
def _signal_context_features(
    signals: pd.DataFrame,
    instrument: str,
) -> pd.DataFrame:
    """Per-instrument signal-context features at every signal date.

    Returns a DataFrame indexed by signal date with columns:
      side (the signal -1/0/+1), signal_run_len (days same-side streak so far,
      inclusive of t), days_since_flip (days since last side change).
    """
    if "date" in signals.columns:
        signals = signals.set_index("date")
    if instrument not in signals.columns:
        return pd.DataFrame()
    s = signals[instrument].astype(int).sort_index()

    feats = pd.DataFrame(index=s.index)
    feats["side_signal"] = s.values  # -1, 0, +1 (kept raw)

    # Run-length: consecutive same-value streak ending at t (inclusive).
    same = (s != s.shift()).cumsum()
    feats["signal_run_len"] = s.groupby(same).cumcount() + 1

    # Days since last flip (= run_len - 1 since the flip happened
    # `run_len - 1` days ago and was a different value).
    feats["days_since_flip"] = feats["signal_run_len"] - 1

    return feats


# --------------------------------------------------------------------------- #
# 4. Calendar features (G7)                                                   #
# --------------------------------------------------------------------------- #
def _calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclical sin/cos encoding of month and day-of-week."""
    month = dates.month.values
    dow = dates.dayofweek.values
    df = pd.DataFrame({
        "month_sin": np.sin(2 * np.pi * month / 12.0),
        "month_cos": np.cos(2 * np.pi * month / 12.0),
        "dow_sin": np.sin(2 * np.pi * dow / 5.0),
        "dow_cos": np.cos(2 * np.pi * dow / 5.0),
    }, index=dates)
    return df


# --------------------------------------------------------------------------- #
# 5. Cross-sectional breadth (G5 cross)                                       #
# --------------------------------------------------------------------------- #
ASSET_CLASSES: dict[str, str] = {
    "es1s": "equity", "nq1s": "equity", "fesx1s": "equity",
    "cl1s": "energy", "ho1s": "energy", "rb1s": "energy", "ng1s": "energy",
    "gc1s": "metals", "si1s": "metals", "hg1s": "metals", "pl1s": "metals",
}


def _signal_breadth(signals: pd.DataFrame) -> pd.DataFrame:
    """Per-date breadth: fraction of each asset class with the same sign as
    each instrument's signal that day. Output: date -> {instrument:
    sector_breadth_same_sign}.
    """
    if "date" in signals.columns:
        signals = signals.set_index("date")
    out_rows = []
    for d, row in signals.iterrows():
        # Per asset class, net signal balance.
        net_by_class: dict[str, float] = {}
        for cls in set(ASSET_CLASSES.values()):
            members = [m for m, c in ASSET_CLASSES.items()
                       if c == cls and m in row.index]
            sigs = row[members].astype(int)
            if len(sigs):
                net_by_class[cls] = float(sigs.sum()) / len(sigs)
            else:
                net_by_class[cls] = 0.0
        out_rows.append({"date": d, **net_by_class})
    df = pd.DataFrame(out_rows).set_index("date")
    df.columns = [f"net_signal_{c}" for c in df.columns]
    return df


# --------------------------------------------------------------------------- #
# 6. Master: compute_features                                                  #
# --------------------------------------------------------------------------- #
def compute_features(
    ohlcv_long: pd.DataFrame,
    events: pd.DataFrame,
    signals: pd.DataFrame,
    include_groups: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5", "G7"),
    zscore_min_periods: int = 60,
) -> pd.DataFrame:
    """Compute the feature matrix for each labeled event.

    Parameters
    ----------
    ohlcv_long : pd.DataFrame
        NA-cleaned OHLCV in long form (date, instrument, ...). Use
        ``stml.io.load_clean_data``.
    events : pd.DataFrame
        Labeled events with at least columns ``t`` and ``instrument`` (e.g.
        the output of ``stml.labeling.get_meta_labels``).
    signals : pd.DataFrame
        Wide primary signals — for G5 features.
    include_groups : tuple of str
        Which feature groups to compute. Stage 2a uses (G1, G2, G3, G5, G7).
        Stage 3a will extend to add G4, G6 (via stml.regimes).
    zscore_min_periods : int, default 60
        Minimum observations before an expanding z-score is emitted (NaN
        before).

    Returns
    -------
    pd.DataFrame
        One row per event (same index as ``events``), columns = features.
        Rows with NaN-only features (because of insufficient history) are
        retained — the model layer drops or imputes.
    """
    if events.empty:
        return pd.DataFrame()
    universe = sorted(events["instrument"].unique())

    # Cache per-instrument computations.
    inst_features: dict[str, pd.DataFrame] = {}
    for inst in universe:
        f = _per_instrument_features(
            ohlcv_long, inst, zscore_min_periods=zscore_min_periods
        )
        if "G5" in include_groups:
            sigf = _signal_context_features(signals, inst)
            if not f.empty and not sigf.empty:
                f = f.join(sigf, how="left")
        inst_features[inst] = f

    # Cross-sectional breadth (G5 cross).
    if "G5" in include_groups:
        breadth = _signal_breadth(signals)
    else:
        breadth = pd.DataFrame()

    # G8 — cross-sectional / economic-intuition features
    if "G8" in include_groups:
        g8 = compute_cross_sectional_features(ohlcv_long, events, signals)
    else:
        g8 = pd.DataFrame()

    # Calendar features (G7).
    if "G7" in include_groups:
        cal_idx = pd.DatetimeIndex(sorted(set(events["t"])))
        cal = _calendar_features(cal_idx)
    else:
        cal = pd.DataFrame()

    # Per-event feature lookup.
    rows: list[dict] = []
    for ev_id, ev in events.iterrows():
        t, inst = ev["t"], ev["instrument"]
        f = inst_features.get(inst, pd.DataFrame())
        if not f.empty and t in f.index:
            row = f.loc[t].to_dict()
        elif not f.empty:
            # Forward-fill up to t.
            tmp = f.reindex(f.index.union([t]).sort_values()).ffill()
            row = tmp.loc[t].to_dict() if t in tmp.index else {}
        else:
            row = {}
        # Add G5 cross-sectional breadth.
        if not breadth.empty and t in breadth.index:
            for col, val in breadth.loc[t].items():
                row[col] = val
        # Add calendar.
        if not cal.empty and t in cal.index:
            for col, val in cal.loc[t].items():
                row[col] = val
        row["__t__"] = t
        row["__instrument__"] = inst
        rows.append(row)

    df = pd.DataFrame(rows, index=events.index)
    # Drop helper columns from the output (we'll keep events as the source of truth for t, inst).
    df = df.drop(columns=["__t__", "__instrument__"], errors="ignore")
    # Join G8 cross-sectional features if requested.
    if "G8" in include_groups and not g8.empty:
        df = df.join(g8, how="left")
    return df


# --------------------------------------------------------------------------- #
# 7. Feature-group registry (populated at import)                             #
# --------------------------------------------------------------------------- #
def compute_cross_sectional_features(
    ohlcv_long: pd.DataFrame,
    events: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-sectional / cross-asset features (G8) — economic intuition.

    For each (date, instrument) event, compute features that depend on the
    *cross-section* of instruments at that date:
      - cross_sec_mom_rank_21d : rank of instrument's 21d return within its
        asset class (0 = lowest, 1 = highest).
      - cross_sec_vol_rank_21d : similarly for 21d vol.
      - corr_to_sector_63d    : rolling 63d correlation of instrument's
        returns to the sector mean (excluding self).
      - avg_cross_asset_corr_63d : average of pairwise correlations across all
        11 instruments (crisis indicator — rises when everything correlates).
      - signal_breadth_full   : (#long - #short) / 11 across the whole panel
        at this date.
      - signal_consensus_pct  : fraction of instruments with signal in this
        instrument's direction (excluding self).
      - trend_persistence     : number of consecutive same-sign primary signals
        before this date (causal).

    All causal — uses only data up to and including the event date.
    """
    out_rows: list[dict] = []
    # Pre-compute per-instrument returns and 21d momentum/vol once.
    inst_data: dict[str, dict] = {}
    for inst in events["instrument"].unique():
        s = (
            ohlcv_long.loc[ohlcv_long["instrument"] == inst]
            .set_index("date")["close"]
            .sort_index()
        )
        s = s[~s.index.duplicated(keep="last")]
        ret = np.log(s).diff()
        mom_21 = ret.rolling(21).sum()
        vol_21 = ret.rolling(21).std() * np.sqrt(252)
        inst_data[inst] = {"ret": ret, "mom_21": mom_21, "vol_21": vol_21}

    # Build wide returns panel for correlation work.
    ret_wide_dict = {inst: d["ret"] for inst, d in inst_data.items()}
    ret_wide = pd.DataFrame(ret_wide_dict).sort_index()

    # Pre-compute signal panel.
    if "date" in signals.columns:
        sig_panel = signals.set_index("date")
    else:
        sig_panel = signals

    # Asset classes.
    instruments_by_class: dict[str, list[str]] = {}
    for inst, sec in ASSET_CLASSES.items():
        instruments_by_class.setdefault(sec, []).append(inst)

    # For each event, sample the cross-sectional features.
    for ev_id, ev in events.iterrows():
        t, inst = ev["t"], ev["instrument"]
        sec = ASSET_CLASSES.get(inst, "other")
        sec_mates = [k for k in instruments_by_class.get(sec, []) if k in inst_data]
        row: dict = {}

        # 1. cross-sectional momentum and vol rank within sector
        if len(sec_mates) > 1:
            mom_vals = {}
            vol_vals = {}
            for m in sec_mates:
                mr = inst_data[m]["mom_21"]
                vr = inst_data[m]["vol_21"]
                if t in mr.index and not pd.isna(mr.loc[t]):
                    mom_vals[m] = float(mr.loc[t])
                if t in vr.index and not pd.isna(vr.loc[t]):
                    vol_vals[m] = float(vr.loc[t])
            if inst in mom_vals and len(mom_vals) > 1:
                ranks = pd.Series(mom_vals).rank(pct=True)
                row["cross_sec_mom_rank_21d"] = float(ranks[inst])
            else:
                row["cross_sec_mom_rank_21d"] = 0.5
            if inst in vol_vals and len(vol_vals) > 1:
                ranks = pd.Series(vol_vals).rank(pct=True)
                row["cross_sec_vol_rank_21d"] = float(ranks[inst])
            else:
                row["cross_sec_vol_rank_21d"] = 0.5
        else:
            row["cross_sec_mom_rank_21d"] = 0.5
            row["cross_sec_vol_rank_21d"] = 0.5

        # 2. correlation to sector mean (rolling 63d, ending at t)
        if inst in inst_data and len(sec_mates) > 1:
            mates_no_self = [m for m in sec_mates if m != inst]
            ret_self = inst_data[inst]["ret"]
            # Sector mean return excluding self.
            sec_panel = ret_wide[mates_no_self]
            sec_mean = sec_panel.mean(axis=1)
            # 63d rolling correlation up to t (inclusive).
            try:
                ret_self_window = ret_self.loc[:t].tail(63)
                sec_mean_window = sec_mean.loc[:t].tail(63)
                joined = pd.concat([ret_self_window, sec_mean_window], axis=1).dropna()
                if len(joined) >= 30:
                    corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
                    row["corr_to_sector_63d"] = float(corr) if not pd.isna(corr) else 0.0
                else:
                    row["corr_to_sector_63d"] = 0.0
            except (KeyError, IndexError):
                row["corr_to_sector_63d"] = 0.0
        else:
            row["corr_to_sector_63d"] = 0.0

        # 3. average pairwise correlation across all instruments (crisis indicator)
        try:
            window = ret_wide.loc[:t].tail(63).dropna(axis=1, thresh=30)
            if window.shape[1] >= 3:
                corr_mat = window.corr()
                # off-diagonal mean
                mask = ~np.eye(corr_mat.shape[0], dtype=bool)
                row["avg_cross_asset_corr_63d"] = float(corr_mat.values[mask].mean())
            else:
                row["avg_cross_asset_corr_63d"] = 0.5
        except (KeyError, IndexError):
            row["avg_cross_asset_corr_63d"] = 0.5

        # 4. Signal breadth (whole panel)
        if t in sig_panel.index:
            sig_row = sig_panel.loc[t]
            if isinstance(sig_row, pd.DataFrame):
                sig_row = sig_row.iloc[0]
            n_long = float((sig_row == 1).sum())
            n_short = float((sig_row == -1).sum())
            n_active = float((sig_row != 0).sum())
            row["signal_breadth_full"] = (n_long - n_short) / max(len(sig_row), 1)
            # Signal consensus with this instrument's direction (excluding self).
            my_side = ev["side"]
            others = sig_row.drop(inst) if inst in sig_row.index else sig_row
            same_dir = float((others == my_side).sum())
            row["signal_consensus_pct"] = same_dir / max(len(others), 1)
        else:
            row["signal_breadth_full"] = 0.0
            row["signal_consensus_pct"] = 0.5

        # 5. Trend persistence — consecutive same-sign signals BEFORE t (causal)
        if inst in sig_panel.columns:
            prior_signals = sig_panel.loc[:t, inst]
            # Drop the current date so it's strictly prior.
            prior_signals = prior_signals.iloc[:-1] if len(prior_signals) > 0 else prior_signals
            if len(prior_signals):
                # Find consecutive same-sign streak ending right before t.
                target = ev["side"]
                streak = 0
                for v in prior_signals.iloc[::-1]:
                    if v == target:
                        streak += 1
                    else:
                        break
                row["trend_persistence"] = float(streak)
            else:
                row["trend_persistence"] = 0.0
        else:
            row["trend_persistence"] = 0.0

        # 6. Vol clustering: autocorr of |returns| over 21d
        if inst in inst_data:
            ret = inst_data[inst]["ret"]
            abs_ret_window = ret.abs().loc[:t].tail(21).dropna()
            if len(abs_ret_window) >= 5 and abs_ret_window.std() > 0:
                row["vol_clustering_21d"] = float(abs_ret_window.autocorr(lag=1))
            else:
                row["vol_clustering_21d"] = 0.0
        else:
            row["vol_clustering_21d"] = 0.0

        # 7. Recent shock: yesterday's |return| z-score vs 63d
        if inst in inst_data:
            ret = inst_data[inst]["ret"]
            try:
                idx = ret.index.get_loc(t)
                if idx >= 63:
                    window = ret.iloc[idx - 63:idx]
                    yesterday = ret.iloc[idx - 1] if idx >= 1 else 0.0
                    z = (abs(yesterday) - window.abs().mean()) / (window.abs().std() + 1e-12)
                    row["recent_shock_z"] = float(z) if not pd.isna(z) else 0.0
                else:
                    row["recent_shock_z"] = 0.0
            except (KeyError, IndexError):
                row["recent_shock_z"] = 0.0
        else:
            row["recent_shock_z"] = 0.0

        out_rows.append(row)

    return pd.DataFrame(out_rows, index=events.index)


_register("G1_vol", [
    "vol_5d", "vol_21d", "vol_63d", "ewma_vol_50",
    "vol_ratio_5_63", "vol_of_vol_63", "semivol_21d",
    "parkinson_vol_21d", "garman_klass_vol_21d", "rogers_satchell_vol_21d",
    "z_vol_5d", "z_vol_21d", "z_vol_63d", "z_ewma_vol_50",
    "z_vol_of_vol_63", "z_semivol_21d",
    "z_parkinson_vol_21d", "z_garman_klass_vol_21d", "z_rogers_satchell_vol_21d",
])
_register("G2_trend", [
    "mom_5d", "mom_21d", "mom_63d", "ma_dist_21d", "ma_dist_63d", "ma21_slope",
    "trend_tval_10d", "trend_tval_21d", "trend_tval_42d",
    "z_mom_5d", "z_mom_21d", "z_mom_63d",
    "z_ma_dist_21d", "z_ma_dist_63d", "z_ma21_slope",
])
_register("G3_meanrev", [
    "autocorr_21d", "efficiency_ratio_21d", "variance_ratio_5d_21w",
    "hurst_100d",
])
_register("G4_microstructure", [
    "volume_z_63d", "volume_trend_21d", "oi_trend_21d",
    "amihud_illiq_21d", "hl_range_21d",
    "z_volume_z_63d", "z_volume_trend_21d", "z_oi_trend_21d",
    "z_amihud_illiq_21d", "z_hl_range_21d",
])
_register("G5_signal", [
    "side_signal", "signal_run_len", "days_since_flip",
    "net_signal_equity", "net_signal_energy", "net_signal_metals",
])
_register("G7_calendar", [
    "month_sin", "month_cos", "dow_sin", "dow_cos",
])
_register("G8_cross_section", [
    "cross_sec_mom_rank_21d", "cross_sec_vol_rank_21d",
    "corr_to_sector_63d", "avg_cross_asset_corr_63d",
    "signal_breadth_full", "signal_consensus_pct",
    "trend_persistence", "vol_clustering_21d", "recent_shock_z",
])
