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

CITATIONS
=========
* CBOE VIX White Paper (2019) — VIX construction and term structure.
* Merrill Lynch MOVE Index methodology — bond vol proxy.
* CBOE SKEW Index methodology (2011) — tail-risk / skewness of SPX options.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "MACRO_INSTRUMENT_TARGETS",
    "m1_volatility_term_structure",
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
]
