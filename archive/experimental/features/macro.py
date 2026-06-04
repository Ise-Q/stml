"""F11 macro features — PIT-aligned, REFORMULATED as 63-day rolling RANKS.

The plan §3.3 diagnostic finding: macro LEVELS in alken's F11 have CATASTROPHIC
train→test drift (median KS 0.358, max 1.000 — see plan §2.4 / branch_descriptions
§5.26). Reformulation: every LEVEL series becomes a 63-day rolling rank in [0, 1];
CHANGES are kept as-is. The level is NOT added as a feature.

Inputs: Harry's PIT-aligned macro parquet (``data/bloomberg/cleaned/macro_harry.parquet``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stml.experimental.features.catalog import (
    FeatureContext,
    FeatureSpec,
    register,
    rolling_rank,
    rolling_zscore,
)

# Series consumed from Harry's macro parquet → feature name + role.
# Levels go through rolling rank; changes are computed and kept as-is.
_LEVEL_SERIES = {
    # series name : feature suffix (we prepend "f11_")
    "VIX": "vix",
    "MOVE": "move",
    "DXY": "dxy",
    "CBOE_SKEW": "skew",
    "10Y_UST": "ust10",
    "2Y_UST": "ust2",
    "HY_OAS": "hy_oas",
    "IG_OAS": "ig_oas",
    "TIPS10Y": "tips10y",
    "BE10Y": "be10y",
    "10Y_BUND": "bund10",
    "EURUSD": "eurusd",
    "BAL_DRY_INDEX": "baltic_dry",
    "CHINA_PMI_MFG": "china_pmi",
    "US_ISM_MFG_PMI": "us_pmi",
    "LME_COPPER_STOCK": "lme_copper",
    # EIA LEVEL series — Harry has these, BBG file has CHANGE (complementary).
    "EIA_CRUDE_STOCK": "eia_crude_stock",
    "EIA_DIST_STOCK": "eia_dist_stock",
    "EIA_GASOLINE_STOCK": "eia_gasoline_stock",
    "EIA_NG_STOCK": "eia_ng_stock",
}

# Series for which we add a 20-day log-change feature too.
_CHANGE_SERIES = {
    "VIX": "vix",
    "MOVE": "move",
    "DXY": "dxy",
    "10Y_UST": "ust10",
    "HY_OAS": "hy_oas",
    "BE10Y": "be10y",
    "EURUSD": "eurusd",
}

# Series for which we add a 5-day diff (faster macro reaction signal).
_5D_DIFF = {
    "VIX": "vix",
    "10Y_UST": "ust10",
    "DXY": "dxy",
    "BE10Y": "be10y",
    "HY_OAS": "hy_oas",
}


def _make_rank_fn(series_name: str, window: int = 63):
    def _fn(ctx: FeatureContext) -> pd.Series:
        if series_name not in ctx.macro_harry.columns:
            return pd.Series(np.nan, index=ctx.frame.index)
        s = ctx.macro_harry[series_name].astype(float)
        s = s.reindex(ctx.frame.index, method="ffill")
        return rolling_rank(s, window)
    return _fn


def _make_chg_fn(series_name: str, periods: int):
    def _fn(ctx: FeatureContext) -> pd.Series:
        if series_name not in ctx.macro_harry.columns:
            return pd.Series(np.nan, index=ctx.frame.index)
        s = ctx.macro_harry[series_name].astype(float)
        s = s.reindex(ctx.frame.index, method="ffill")
        # Use log change for positive series; otherwise plain diff.
        if (s > 0).all() and not s.isna().all():
            return np.log(s / s.shift(periods))
        return s - s.shift(periods)
    return _fn


# Build and register everything programmatically.
for series, suffix in _LEVEL_SERIES.items():
    register(
        FeatureSpec(
            name=f"f11_{suffix}_rank63",
            family="F11",
            fn=_make_rank_fn(series, window=63),
            leakage_class="E",
            warmup_bars=63,
            source="alken+rebuild",
        )
    )

for series, suffix in _CHANGE_SERIES.items():
    register(
        FeatureSpec(
            name=f"f11_{suffix}_chg20",
            family="F11",
            fn=_make_chg_fn(series, periods=20),
            leakage_class="E",
            warmup_bars=21,
            source="alken+rebuild",
        )
    )

for series, suffix in _5D_DIFF.items():
    register(
        FeatureSpec(
            name=f"f11_{suffix}_chg5",
            family="F11",
            fn=_make_chg_fn(series, periods=5),
            leakage_class="E",
            warmup_bars=6,
            source="alken+rebuild",
        )
    )


# Two derived spread features (the plan §3.3 add-ons).
def _f11_curve_slope(ctx: FeatureContext) -> pd.Series:
    """10Y UST - 2Y UST — yield curve slope."""
    if "10Y_UST" not in ctx.macro_harry.columns or "2Y_UST" not in ctx.macro_harry.columns:
        return pd.Series(np.nan, index=ctx.frame.index)
    macro = ctx.macro_harry
    slope = (macro["10Y_UST"] - macro["2Y_UST"]).reindex(ctx.frame.index, method="ffill")
    return slope


def _f11_credit_diff(ctx: FeatureContext) -> pd.Series:
    """HY OAS - IG OAS — credit risk spread."""
    if "HY_OAS" not in ctx.macro_harry.columns or "IG_OAS" not in ctx.macro_harry.columns:
        return pd.Series(np.nan, index=ctx.frame.index)
    macro = ctx.macro_harry
    return (macro["HY_OAS"] - macro["IG_OAS"]).reindex(ctx.frame.index, method="ffill")


def _f11_vix_term_slope(ctx: FeatureContext) -> pd.Series:
    """VIX3M - VIX — VIX term structure slope."""
    if "VIX3M" not in ctx.macro_harry.columns or "VIX" not in ctx.macro_harry.columns:
        return pd.Series(np.nan, index=ctx.frame.index)
    macro = ctx.macro_harry
    return (macro["VIX3M"] - macro["VIX"]).reindex(ctx.frame.index, method="ffill")


register(FeatureSpec("f11_curve_slope", "F11", _f11_curve_slope, "E", 1, "alken"))
register(FeatureSpec("f11_credit_diff", "F11", _f11_credit_diff, "E", 1, "alken"))
register(FeatureSpec("f11_vix_term_slope", "F11", _f11_vix_term_slope, "E", 1, "alken"))
