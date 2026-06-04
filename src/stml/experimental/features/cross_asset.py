"""F9 cross-section + F21 cross-asset relative value.

F9 — the alternative/features/cross_asset.py (lead-lag, asset-class dispersion,
     EWMA implied-correlation z).
F21 — NEW, computed from existing OHLCV (no BBG dependency).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stml.experimental.config import ASSET_CLASS_MEMBERS
from stml.experimental.features.catalog import (
    FeatureContext,
    FeatureSpec,
    log_returns,
    register,
    rolling_zscore,
)


# ---------------------------------------------------------------------------
# F9 — Cross-section family.
# ---------------------------------------------------------------------------


def _f9_xsect_rank_60(ctx: FeatureContext) -> pd.Series:
    """Rank of this instrument's 60d return within its asset class, in [0, 1]."""
    members = ASSET_CLASS_MEMBERS.get(ctx.asset_class, ())
    if len(members) <= 1:
        return pd.Series(np.nan, index=ctx.frame.index)
    own_ret_60 = log_returns(ctx.frame["close"]).rolling(60).sum()
    peer_returns = []
    for inst in members:
        if inst not in ctx.universe:
            continue
        peer = ctx.universe[inst]["close"]
        peer_returns.append(np.log(peer / peer.shift(60)))
    if not peer_returns:
        return pd.Series(np.nan, index=ctx.frame.index)
    peer_df = pd.concat(peer_returns, axis=1, sort=True)
    peer_df.columns = list(range(peer_df.shape[1]))
    # Reindex to our instrument's calendar.
    peer_df_aligned = peer_df.reindex(ctx.frame.index, method="ffill")
    own_aligned = own_ret_60.reindex(ctx.frame.index)
    # Per-row rank of own value among peers (including self).
    full = pd.concat([own_aligned.rename("own"), peer_df_aligned], axis=1, sort=False)
    return full.rank(axis=1, pct=True)["own"]


def _f9_dispersion_z_60(ctx: FeatureContext) -> pd.Series:
    """Cross-sectional std of 60d returns within asset class — dispersion z-score."""
    members = ASSET_CLASS_MEMBERS.get(ctx.asset_class, ())
    if len(members) <= 1:
        return pd.Series(np.nan, index=ctx.frame.index)
    peer_returns = []
    for inst in members:
        if inst not in ctx.universe:
            continue
        peer = ctx.universe[inst]["close"]
        peer_returns.append(np.log(peer / peer.shift(60)))
    if not peer_returns:
        return pd.Series(np.nan, index=ctx.frame.index)
    peer_df = pd.concat(peer_returns, axis=1, sort=True)
    dispersion = peer_df.std(axis=1)
    # Z-score the dispersion (expanding-window).
    dispersion = dispersion.reindex(ctx.frame.index, method="ffill")
    return rolling_zscore(dispersion, window=252)


def _f9_pair_corr_mean_63(ctx: FeatureContext) -> pd.Series:
    """Mean rolling 63d pair correlation between this instrument and its class peers."""
    members = ASSET_CLASS_MEMBERS.get(ctx.asset_class, ())
    if len(members) <= 1:
        return pd.Series(np.nan, index=ctx.frame.index)
    own_r = log_returns(ctx.frame["close"])
    corrs = []
    for inst in members:
        if inst == ctx.instrument or inst not in ctx.universe:
            continue
        peer_r = log_returns(ctx.universe[inst]["close"])
        # Align to own_r index.
        peer_aligned = peer_r.reindex(own_r.index)
        corrs.append(own_r.rolling(63).corr(peer_aligned))
    if not corrs:
        return pd.Series(np.nan, index=ctx.frame.index)
    return pd.concat(corrs, axis=1, sort=False).mean(axis=1)


def _f9_implied_corr_z_252(ctx: FeatureContext) -> pd.Series:
    """EWMA implied-correlation z — proxy for crisis correlation regime.

    Compute the average pair correlation across asset-class peers via EWMA
    (halflife=20), then z-score over 252 bars.
    """
    members = ASSET_CLASS_MEMBERS.get(ctx.asset_class, ())
    if len(members) <= 1:
        return pd.Series(np.nan, index=ctx.frame.index)
    own_r = log_returns(ctx.frame["close"])
    ewma_corrs = []
    for inst in members:
        if inst == ctx.instrument or inst not in ctx.universe:
            continue
        peer_r = log_returns(ctx.universe[inst]["close"]).reindex(own_r.index)
        cov = own_r.ewm(halflife=20).cov(peer_r)
        var_a = own_r.ewm(halflife=20).var()
        var_b = peer_r.ewm(halflife=20).var()
        ewma_corrs.append(cov / np.sqrt(var_a * var_b))
    if not ewma_corrs:
        return pd.Series(np.nan, index=ctx.frame.index)
    mean_corr = pd.concat(ewma_corrs, axis=1, sort=False).mean(axis=1)
    return rolling_zscore(mean_corr, window=252)


register(FeatureSpec("f9_xsect_rank_60", "F9", _f9_xsect_rank_60, "E", 60, "alternative"))
register(FeatureSpec("f9_dispersion_z_60", "F9", _f9_dispersion_z_60, "E", 252, "alternative"))
register(FeatureSpec("f9_pair_corr_mean_63", "F9", _f9_pair_corr_mean_63, "E", 63, "alternative"))
register(FeatureSpec("f9_implied_corr_z_252", "F9", _f9_implied_corr_z_252, "E", 252, "alternative"))


# ---------------------------------------------------------------------------
# F21 — Cross-asset relative value (new, from existing OHLCV).
#
# Plan §5.2 Block D: gold/silver, copper/gold, crack 3-2-1. Computed using
# the project's back-adjusted continuous-contract closes (NOT mixed with BBG).
# ---------------------------------------------------------------------------


def _f21_gold_silver_ratio(ctx: FeatureContext) -> pd.Series:
    if "gc1s" not in ctx.universe or "si1s" not in ctx.universe:
        return pd.Series(np.nan, index=ctx.frame.index)
    gc = ctx.universe["gc1s"]["close"]
    si = ctx.universe["si1s"]["close"]
    ratio = gc / si
    return ratio.reindex(ctx.frame.index, method="ffill")


def _f21_gold_silver_z63(ctx: FeatureContext) -> pd.Series:
    return rolling_zscore(_f21_gold_silver_ratio(ctx), window=63)


def _f21_copper_gold_ratio(ctx: FeatureContext) -> pd.Series:
    if "hg1s" not in ctx.universe or "gc1s" not in ctx.universe:
        return pd.Series(np.nan, index=ctx.frame.index)
    hg = ctx.universe["hg1s"]["close"]
    gc = ctx.universe["gc1s"]["close"]
    return (hg / gc).reindex(ctx.frame.index, method="ffill")


def _f21_crack_321(ctx: FeatureContext) -> pd.Series:
    """3-2-1 crack spread: 2·RB + HO − 3·CL."""
    if not all(i in ctx.universe for i in ("rb1s", "ho1s", "cl1s")):
        return pd.Series(np.nan, index=ctx.frame.index)
    rb = ctx.universe["rb1s"]["close"]
    ho = ctx.universe["ho1s"]["close"]
    cl = ctx.universe["cl1s"]["close"]
    crack = 2 * rb + ho - 3 * cl
    return crack.reindex(ctx.frame.index, method="ffill")


register(FeatureSpec("f21_gold_silver_ratio", "F21", _f21_gold_silver_ratio, "E", 1, "new"))
register(FeatureSpec("f21_gold_silver_z63", "F21", _f21_gold_silver_z63, "E", 63, "new"))
register(FeatureSpec("f21_copper_gold_ratio", "F21", _f21_copper_gold_ratio, "E", 1, "new"))
register(FeatureSpec("f21_crack_321", "F21", _f21_crack_321, "E", 1, "new"))
