"""macro_features.py — six groups of external macro features from the lag-safe
macro panel (alternate_data_cleaned.csv / macro_panel.parquet).

CAUSALITY CONTRACT
==================
Every output at row ``t`` uses only macro data at indices ``<= t``.
Z-scores and rolling statistics use causal (trailing) windows only.
EIA surprise windows count releases (weekly cadence), not calendar days.

WARMUP CONVENTIONS
==================
Each group function takes ``window`` (z-score / rolling window) and any
group-specific parameters. Warmup = the maximum warmup across all features
in the group. Rows before warmup may contain NaN; rows from warmup onward
are finite on any fully-populated macro panel.

MACRO_INSTRUMENT_TARGETS
=========================
Documents which instruments each feature most influences. The pooled feature
matrix may expose all macro features to all instruments; importance sorting
handles relevance.

GROUPS (added incrementally)
============================
M1 — volatility / term structure: VIX, MOVE, CBOE SKEW
M2 — rates / curve: UST, Bund, TIPS, breakeven
M3 — credit: HY OAS, IG OAS
M4 — FX / dollar: DXY, EURUSD
M5 — commodity fundamentals: EIA inventory surprises, copper stocks, BDI
M6 — macro growth: ISM PMI, China PMI, global breadth

CITATIONS
=========
* CBOE VIX White Paper (2019) — VIX construction and term structure.
* Merrill Lynch MOVE Index methodology — bond vol proxy.
* CBOE SKEW Index methodology (2011) — tail-risk / skewness of SPX options.
* Estrella, A. & Hardouvelis, G. (1991) "The Term Structure as a Predictor
  of Real Economic Activity", Journal of Finance 46(2): 555–576.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "MACRO_INSTRUMENT_TARGETS",
    "m1_volatility_term_structure",
    "m2_rates_curve",
    "m3_credit",
    "m4_fx_dollar",
    "m5_commodity_fundamentals",
    "m6_macro_growth",
]

# --------------------------------------------------------------------------- #
# Instrument → primary macro features mapping (for feature catalog)           #
# --------------------------------------------------------------------------- #
#: Per-feature list of instruments the feature most directly targets.
#: The pooled feature matrix exposes all macro features to all instruments;
#: this dict is for documentation and Step-4 importance analysis only.
MACRO_INSTRUMENT_TARGETS: dict[str, list[str]] = {
    # M1 — volatility / term structure
    "vix_level_z":    ["es1s", "nq1s", "fesx1s", "gc1s", "si1s"],
    "vix_5d_change":  ["es1s", "nq1s", "fesx1s"],
    "vix_term_slope": ["es1s", "nq1s", "fesx1s", "gc1s"],
    "move_z":         ["es1s", "nq1s", "fesx1s"],
    "move_vix_ratio": ["es1s", "nq1s", "fesx1s"],
    "skew_z":         ["es1s", "nq1s", "fesx1s"],
    # M2 — rates / curve
    "us_2s10s_slope":     ["es1s", "nq1s", "fesx1s"],
    "ust_10y_5d_change":  ["es1s", "nq1s", "fesx1s"],
    "bund_10y_5d_change": ["fesx1s"],
    "ust_bund_spread":    ["fesx1s"],
    "real_yield_10y":     ["gc1s", "si1s"],
    "breakeven_10y":      ["gc1s", "si1s"],
    "be_5d_change":       ["gc1s", "si1s"],
    # M3 — credit
    "hy_oas_z":          ["es1s", "nq1s", "fesx1s"],
    "hy_oas_5d_change":  ["es1s", "nq1s", "fesx1s"],
    "ig_oas_z":          ["es1s", "nq1s", "fesx1s"],
    "hy_ig_ratio":       ["es1s", "nq1s", "fesx1s"],
    # M4 — FX / dollar
    "dxy_z":                   ["gc1s", "si1s", "hg1s", "cl1s"],
    "dxy_5d_change":           ["gc1s", "si1s", "hg1s"],
    "eurusd_5d_change":        ["fesx1s", "gc1s"],
    # M5 — commodity fundamentals
    "crude_stock_surprise":    ["cl1s", "ho1s", "rb1s"],
    "dist_stock_surprise":     ["ho1s"],
    "gasoline_stock_surprise": ["rb1s"],
    "ng_stock_surprise":       ["ng1s"],
    "copper_stock_z":          ["hg1s"],
    "baltic_dry_z":            ["cl1s", "hg1s"],
    "baltic_5d_change":        ["cl1s", "hg1s"],
    # M6 — macro growth
    "ism_pmi_level":      ["es1s", "nq1s", "fesx1s", "hg1s", "cl1s"],
    "ism_pmi_3m_change":  ["es1s", "nq1s", "fesx1s"],
    "china_pmi_level":    ["hg1s", "cl1s"],
    "global_pmi_breadth": ["es1s", "nq1s", "fesx1s"],
}


# --------------------------------------------------------------------------- #
# Private helpers                                                              #
# --------------------------------------------------------------------------- #
def _rolling_z(s: pd.Series, window: int) -> pd.Series:
    """Causal trailing-window z-score. NaN for the first ``window`` rows."""
    s = s.astype("float64")
    mu = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=0)
    return (s - mu) / sd.replace(0.0, np.nan)


def _eia_surprise(series: pd.Series, n_releases: int) -> pd.Series:
    """Causal EIA release surprise: (current_release - mean_prev_n) / std_prev_n.

    Counts *releases* (days where the series value changes), not calendar days.
    A non-release day carries forward the most recent release's surprise.

    Truncation-invariant: at time t, only releases at indices <= t are used.
    NaN until n_releases + 1 releases have been observed.
    """
    s = series.astype("float64")
    is_release = s.diff().fillna(0.0).ne(0.0) & s.notna()
    releases = s[is_release]
    if releases.empty:
        return pd.Series(np.nan, index=s.index, dtype="float64")
    # shift(1) on the release sub-series aligns each release with its
    # predecessor, so rolling(n) computes the previous n releases' stats.
    shifted = releases.shift(1)
    mu = shifted.rolling(n_releases, min_periods=n_releases).mean()
    sd = shifted.rolling(n_releases, min_periods=n_releases).std(ddof=0)
    surprise = (releases - mu) / sd.replace(0.0, np.nan)
    return surprise.reindex(s.index).ffill()


# --------------------------------------------------------------------------- #
# M1 — volatility / term structure                                            #
# --------------------------------------------------------------------------- #
def m1_volatility_term_structure(
    macro_df: pd.DataFrame,
    *,
    window: int = 252,
) -> pd.DataFrame:
    """M1: VIX, MOVE, CBOE SKEW volatility and term-structure features.

    Features
    --------
    vix_level_z      : trailing-``window``-day z-score of VIX level.
                       Spikes signal acute fear; sustained elevation signals
                       structural regime shift. Targets equity + metals.
    vix_5d_change    : VIX(t) − VIX(t−5). Direction of short-term fear.
    vix_term_slope   : VIX3M − VIX. Positive = normal contango (fear priced
                       in 3 m); negative = backwardation = acute spot stress.
                       CBOE VIX White Paper (2019).
    move_z           : z-score of MOVE index (bond vol). High MOVE → rates
                       vol spilling into equities / commodities.
    move_vix_ratio   : MOVE / VIX. Measures whether current stress is
                       equity-led (low ratio) or rates-led (high ratio).
    skew_z           : z-score of CBOE SKEW index. Elevated SKEW = option
                       market pricing crash tail risk even when VIX is low.
                       CBOE SKEW White Paper (2011).

    Parameters
    ----------
    macro_df : pd.DataFrame
        Macro panel with columns ``VIX``, ``VIX3M``, ``MOVE``, ``CBOE_SKEW``.
    window : int
        Trailing window for z-score computation (default 252 trading days).

    Warmup
    ------
    ``window`` rows (z-scores require a full rolling window).
    """
    vix   = macro_df["VIX"].astype("float64")
    vix3m = macro_df["VIX3M"].astype("float64")
    move  = macro_df["MOVE"].astype("float64")
    skew  = macro_df["CBOE_SKEW"].astype("float64")

    return pd.DataFrame(
        {
            "vix_level_z":    _rolling_z(vix, window),
            "vix_5d_change":  vix - vix.shift(5),
            "vix_term_slope": vix3m - vix,
            "move_z":         _rolling_z(move, window),
            "move_vix_ratio": move / vix,
            "skew_z":         _rolling_z(skew, window),
        },
        index=macro_df.index,
    )


# --------------------------------------------------------------------------- #
# M2 — rates / curve                                                          #
# --------------------------------------------------------------------------- #
def m2_rates_curve(
    macro_df: pd.DataFrame,
    *,
    window: int = 252,
) -> pd.DataFrame:
    """M2: US and German rates, curve slope, TIPS real yield, breakeven.

    Features
    --------
    us_2s10s_slope    : 10Y_UST − 2Y_UST. The classic yield-curve slope.
                        Inversion (negative) has historically preceded US
                        recessions (Estrella & Hardouvelis 1991). Targets
                        equity, especially cyclical futures.
    ust_10y_5d_change : 5-day change in 10Y UST yield. Captures rates
                        momentum; a sharp move forces equity repricing.
    bund_10y_5d_change: 5-day change in 10Y Bund yield. European rates
                        driver for Euro Stoxx (fesx1s).
    ust_bund_spread   : 10Y_UST − 10Y_BUND. Cross-border capital flow
                        driver for EURUSD and fesx1s; positive = USD
                        assets relatively attractive.
    real_yield_10y    : z-score of TIPS 10Y yield. Negative real yields
                        are historically bullish for gold (gc1s, si1s)
                        as an inflation hedge.
    breakeven_10y     : z-score of 10Y breakeven inflation (BE10Y).
                        Rising breakevens boost inflation-hedge demand for
                        gold and energy.
    be_5d_change      : 5-day change in BE10Y. Short-term inflation
                        regime shifts.

    Parameters
    ----------
    macro_df : pd.DataFrame
        Must contain columns: ``10Y_UST``, ``2Y_UST``, ``10Y_BUND``,
        ``TIPS10Y``, ``BE10Y``.
    window : int
        Trailing window for z-scores (default 252).

    Warmup
    ------
    ``window`` rows (z-scores for real_yield_10y and breakeven_10y).
    """
    ust10 = macro_df["10Y_UST"].astype("float64")
    ust2  = macro_df["2Y_UST"].astype("float64")
    bund  = macro_df["10Y_BUND"].astype("float64")
    tips  = macro_df["TIPS10Y"].astype("float64")
    be    = macro_df["BE10Y"].astype("float64")

    return pd.DataFrame(
        {
            "us_2s10s_slope":     ust10 - ust2,
            "ust_10y_5d_change":  ust10 - ust10.shift(5),
            "bund_10y_5d_change": bund  - bund.shift(5),
            "ust_bund_spread":    ust10 - bund,
            "real_yield_10y":     _rolling_z(tips, window),
            "breakeven_10y":      _rolling_z(be, window),
            "be_5d_change":       be - be.shift(5),
        },
        index=macro_df.index,
    )


# --------------------------------------------------------------------------- #
# M3 — credit                                                                 #
# --------------------------------------------------------------------------- #
def m3_credit(
    macro_df: pd.DataFrame,
    *,
    window: int = 252,
) -> pd.DataFrame:
    """M3: High-yield and investment-grade credit spread features.

    Features
    --------
    hy_oas_z        : z-score of HY option-adjusted spread. Elevated HY
                      spreads signal credit stress and predict equity
                      drawdowns (Feldhuetter & Lando 2008). Targets equity.
    hy_oas_5d_change: 5-day change in HY OAS. Rapid widening = credit
                      deterioration in progress.
    ig_oas_z        : z-score of IG OAS. Complement to hy_oas_z; IG is
                      less volatile but captures systemic risk earlier.
    hy_ig_ratio     : HY_OAS / IG_OAS. Pure risk-premium ratio; always
                      positive. Elevated ratio = excess compensation for
                      credit quality step-down = stress in junk market.

    Parameters
    ----------
    macro_df : pd.DataFrame
        Must contain columns: ``HY_OAS``, ``IG_OAS``.
    window : int
        Trailing window for z-scores (default 252).

    Warmup
    ------
    ``window`` rows (z-scores).
    """
    hy = macro_df["HY_OAS"].astype("float64")
    ig = macro_df["IG_OAS"].astype("float64")

    return pd.DataFrame(
        {
            "hy_oas_z":          _rolling_z(hy, window),
            "hy_oas_5d_change":  hy - hy.shift(5),
            "ig_oas_z":          _rolling_z(ig, window),
            "hy_ig_ratio":       hy / ig.replace(0.0, np.nan),
        },
        index=macro_df.index,
    )


# --------------------------------------------------------------------------- #
# Causality harness registry                                                  #
# --------------------------------------------------------------------------- #
CAUSALITY_REGISTRATIONS: list[dict] = [
    {
        "name": "m1_volatility_term_structure",
        "module": __name__,
        "func": "m1_volatility_term_structure",
        "adapter": "macro_panel",
        "kwargs": {"window": 60},
        "warmup": 60,
        "data_kind": "macro_panel",
    },
    {
        "name": "m2_rates_curve",
        "module": __name__,
        "func": "m2_rates_curve",
        "adapter": "macro_panel",
        "kwargs": {"window": 60},
        "warmup": 60,
        "data_kind": "macro_panel",
    },
    {
        "name": "m3_credit",
        "module": __name__,
        "func": "m3_credit",
        "adapter": "macro_panel",
        "kwargs": {"window": 60},
        "warmup": 60,
        "data_kind": "macro_panel",
    },
    {
        "name": "m4_fx_dollar",
        "module": __name__,
        "func": "m4_fx_dollar",
        "adapter": "macro_panel",
        "kwargs": {"window": 60},
        "warmup": 60,
        "data_kind": "macro_panel",
    },
    {
        "name": "m5_commodity_fundamentals",
        "module": __name__,
        "func": "m5_commodity_fundamentals",
        "adapter": "macro_panel",
        "kwargs": {"n_releases": 5, "window": 60},
        "warmup": 60,
        "data_kind": "macro_panel",
    },
    {
        "name": "m6_macro_growth",
        "module": __name__,
        "func": "m6_macro_growth",
        "adapter": "macro_panel",
        "kwargs": {"shift_days": 21},
        "warmup": 21,
        "data_kind": "macro_panel",
    },
]


# --------------------------------------------------------------------------- #
# M4 — FX / dollar                                                            #
# --------------------------------------------------------------------------- #
def m4_fx_dollar(
    macro_df: pd.DataFrame,
    *,
    window: int = 252,
) -> pd.DataFrame:
    """M4: Dollar index and EURUSD features.

    Features
    --------
    dxy_z           : z-score of DXY. Strong dollar suppresses commodity
                      prices (invoiced in USD) and pressures EM; weak
                      dollar is broadly bullish for gold, oil, and metals.
    dxy_5d_change   : 5-day DXY change. Short-term dollar momentum.
    eurusd_5d_change: 5-day EURUSD change. Direct driver of fesx1s (Euro-
                      denominated equity index) P&L converted to USD.

    Parameters
    ----------
    macro_df : pd.DataFrame
        Must contain columns: ``DXY``, ``EURUSD``.
    window : int
        Trailing window for z-score (default 252).

    Warmup
    ------
    ``window`` rows (dxy_z).
    """
    dxy    = macro_df["DXY"].astype("float64")
    eurusd = macro_df["EURUSD"].astype("float64")

    return pd.DataFrame(
        {
            "dxy_z":            _rolling_z(dxy, window),
            "dxy_5d_change":     dxy    - dxy.shift(5),
            "eurusd_5d_change":  eurusd - eurusd.shift(5),
        },
        index=macro_df.index,
    )


# --------------------------------------------------------------------------- #
# M5 — commodity fundamentals                                                 #
# --------------------------------------------------------------------------- #
def m5_commodity_fundamentals(
    macro_df: pd.DataFrame,
    *,
    n_releases: int = 5,
    window: int = 252,
) -> pd.DataFrame:
    """M5: EIA inventory surprises, copper stocks, Baltic Dry Index.

    Features
    --------
    crude_stock_surprise    : EIA weekly crude oil inventory surprise,
                              computed as (current_release − mean_prev_n) /
                              std_prev_n where n = ``n_releases``. Positive =
                              unexpected build (bearish cl1s); negative =
                              unexpected draw (bullish cl1s, ho1s, rb1s).
                              Window counts releases, not calendar days.
                              EIA Weekly Petroleum Status Report.
    dist_stock_surprise     : Same for distillate (diesel) stocks. Target: ho1s.
    gasoline_stock_surprise : Same for gasoline stocks. Target: rb1s.
    ng_stock_surprise       : Same for natural gas storage. Target: ng1s.
                              EIA Natural Gas Storage Report (weekly).
    copper_stock_z          : z-score of LME copper warehouse stocks.
                              Rising stocks → oversupply → bearish hg1s;
                              falling stocks → tightness → bullish hg1s.
                              "Dr. Copper" demand signal.
    baltic_dry_z            : z-score of Baltic Dry Index. Proxy for global
                              dry-bulk shipping demand = leading indicator of
                              industrial activity. High BDI → bullish cl1s, hg1s.
    baltic_5d_change        : 5-day BDI change. Short-term shipping momentum.

    Parameters
    ----------
    macro_df : pd.DataFrame
        Must contain: ``EIA_CRUDE_STOCK``, ``EIA_DIST_STOCK``,
        ``EIA_GASOLINE_STOCK``, ``EIA_NG_STOCK``, ``LME_COPPER_STOCK``,
        ``BAL_DRY_INDEX``.
    n_releases : int
        Number of prior releases for the EIA surprise denominator (default 5).
    window : int
        Trailing window for copper_stock_z and baltic_dry_z (default 252).

    Warmup
    ------
    ``window`` rows (z-scores bind; EIA surprises require only
    (n_releases + 1) releases × weekly cadence ≈ 30–40 days).
    """
    crude    = macro_df["EIA_CRUDE_STOCK"].astype("float64")
    dist     = macro_df["EIA_DIST_STOCK"].astype("float64")
    gasoline = macro_df["EIA_GASOLINE_STOCK"].astype("float64")
    ng       = macro_df["EIA_NG_STOCK"].astype("float64")
    copper   = macro_df["LME_COPPER_STOCK"].astype("float64")
    bdi      = macro_df["BAL_DRY_INDEX"].astype("float64")

    return pd.DataFrame(
        {
            "crude_stock_surprise":    _eia_surprise(crude, n_releases),
            "dist_stock_surprise":     _eia_surprise(dist, n_releases),
            "gasoline_stock_surprise": _eia_surprise(gasoline, n_releases),
            "ng_stock_surprise":       _eia_surprise(ng, n_releases),
            "copper_stock_z":          _rolling_z(copper, window),
            "baltic_dry_z":            _rolling_z(bdi, window),
            "baltic_5d_change":        bdi - bdi.shift(5),
        },
        index=macro_df.index,
    )


# --------------------------------------------------------------------------- #
# M6 — macro growth                                                           #
# --------------------------------------------------------------------------- #
def m6_macro_growth(
    macro_df: pd.DataFrame,
    *,
    shift_days: int = 63,
) -> pd.DataFrame:
    """M6: ISM PMI, China PMI, and global PMI breadth.

    Features
    --------
    ism_pmi_level     : US ISM Manufacturing PMI level (forward-filled from
                        monthly release). Values above 50 indicate expansion.
                        ISM methodology: diffusion index of 5 sub-indices.
    ism_pmi_3m_change : ISM_PMI(t) − ISM_PMI(t − ``shift_days``). Captures
                        the cyclical direction of US manufacturing momentum
                        over ~3 months. Positive → expanding faster.
    china_pmi_level   : China Caixin/Official Manufacturing PMI level
                        (forward-filled). Above 50 = expansion. Leading
                        indicator for copper (hg1s) and crude (cl1s) demand.
    global_pmi_breadth: Fraction of {US_ISM_MFG_PMI, CHINA_PMI_MFG} that
                        are above 50, ignoring NaN (mean of 1[pmi > 50]).
                        Range: {0.0, 0.5, 1.0} when both series are available.
                        A breadth of 1.0 = synchronised global expansion;
                        0.0 = synchronised contraction.

    Parameters
    ----------
    macro_df : pd.DataFrame
        Must contain: ``US_ISM_MFG_PMI``, ``CHINA_PMI_MFG``.
    shift_days : int
        Look-back horizon for ism_pmi_3m_change (default 63 trading days
        ≈ 3 calendar months).

    Warmup
    ------
    ``shift_days`` rows (ism_pmi_3m_change is NaN for the first shift_days rows).
    """
    ism   = macro_df["US_ISM_MFG_PMI"].astype("float64")
    china = macro_df["CHINA_PMI_MFG"].astype("float64")

    # Global breadth: row-wise mean of binary above-50 indicators, skip NaN
    above_50 = pd.DataFrame(
        {
            "us":    (ism > 50).astype("float64"),
            "china": (china > 50).astype("float64"),
        }
    )
    above_50.loc[china.isna(), "china"] = np.nan
    breadth = above_50.mean(axis=1, skipna=True)

    return pd.DataFrame(
        {
            "ism_pmi_level":      ism,
            "ism_pmi_3m_change":  ism - ism.shift(shift_days),
            "china_pmi_level":    china,
            "global_pmi_breadth": breadth,
        },
        index=macro_df.index,
    )
