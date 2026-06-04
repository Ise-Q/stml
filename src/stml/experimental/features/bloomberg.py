"""F18 term structure + F19 options IV + F22 event flags — Bloomberg-derived.

All consume from PIT-aligned parquets emitted by
:mod:`stml.experimental.bloomberg_ingest`.

F18 — futures term structure (Block A): front + 2nd month spreads, roll yield,
      contango flag. Equity uses UX1/UX2 (VIX futures front + 2nd).
F19 — options-implied vol (Block B): ATM IV 1m/3m, IV term slope, skew, RV-IV
      spread, 252d IV percentile. PL1 substituted by GC1 IV (XPT empty).
F22 — event flags (Block E partial): EIA crude release day flag + 26-week
      z-score of weekly crude inventory change. Energy instruments only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stml.experimental.features.catalog import (
    FeatureContext,
    FeatureSpec,
    log_returns,
    register,
    rolling_zscore,
)

# ---------------------------------------------------------------------------
# Instrument → BBG ticker mapping.
# ---------------------------------------------------------------------------

# Project instrument → BBG raw front-month ticker root (no _LAST suffix).
_BBG_FRONT_MAP = {
    "cl1s": "CL1", "ho1s": "HO1", "rb1s": "XB1", "ng1s": "NG1",
    "gc1s": "GC1", "si1s": "SI1", "hg1s": "HG1", "pl1s": "PL1",
}
_BBG_SECOND_MAP = {k: v.replace("1", "2") for k, v in _BBG_FRONT_MAP.items()}

# Equity term structure → use VIX futures UX1 / UX2 as the analogue.
_EQUITY_TERM_FRONT = "UX1"
_EQUITY_TERM_SECOND = "UX2"

# Project instrument → BBG IV ticker root (used by F19).
_BBG_IV_MAP = {
    "es1s": "SPX", "nq1s": "NDX", "fesx1s": "SX5E",
    "cl1s": "CL1", "ho1s": "HO1", "rb1s": "XB1", "ng1s": "NG1",
    "gc1s": "GC1", "si1s": "SI1", "hg1s": "HG1",
    "pl1s": "GC1",  # SUBSTITUTION — methodology spec fallback (XPT data empty)
}

# Energy instruments — F22 applies only to these.
_ENERGY_INSTRUMENTS = ("cl1s", "ho1s", "rb1s", "ng1s")


def _front_second_close(ctx: FeatureContext) -> tuple[pd.Series | None, pd.Series | None]:
    """Return (front, second) BBG-raw close series aligned to the instrument's index.

    For equity → VIX-futures UX1 / UX2.
    For commodities → CL1/CL2 etc.
    Returns (None, None) if the BBG parquet doesn't have those columns.
    """
    ft = ctx.futures_term
    if ft.empty:
        return None, None

    if ctx.asset_class == "equity":
        front_col = f"{_EQUITY_TERM_FRONT}_LAST"
        second_col = f"{_EQUITY_TERM_SECOND}_LAST"
    else:
        root_front = _BBG_FRONT_MAP.get(ctx.instrument)
        root_second = _BBG_SECOND_MAP.get(ctx.instrument)
        if root_front is None or root_second is None:
            return None, None
        front_col = f"{root_front}_LAST"
        second_col = f"{root_second}_LAST"

    if front_col not in ft.columns or second_col not in ft.columns:
        return None, None
    front = ft[front_col].reindex(ctx.frame.index, method="ffill")
    second = ft[second_col].reindex(ctx.frame.index, method="ffill")
    return front, second


# ---------------------------------------------------------------------------
# F18 — Term structure family.
# ---------------------------------------------------------------------------


def _f18_term_spread(ctx: FeatureContext) -> pd.Series:
    """(F2 - F1) / F1 — contango (>0) vs backwardation (<0)."""
    front, second = _front_second_close(ctx)
    if front is None or second is None:
        return pd.Series(np.nan, index=ctx.frame.index)
    return (second - front) / front.replace(0, np.nan)


def _f18_term_spread_z63(ctx: FeatureContext) -> pd.Series:
    return rolling_zscore(_f18_term_spread(ctx), window=63)


def _f18_contango_flag(ctx: FeatureContext) -> pd.Series:
    """1 if term spread positive (contango), 0 if backwardation."""
    return (_f18_term_spread(ctx) > 0).astype(float)


def _f18_term_chg5(ctx: FeatureContext) -> pd.Series:
    """5-day diff of the term spread — momentum of term structure."""
    return _f18_term_spread(ctx).diff(5)


def _f18_roll_yield(ctx: FeatureContext) -> pd.Series:
    """-term_spread × annualisation factor = approx annualised roll yield."""
    return -_f18_term_spread(ctx) * 12.0  # rough — monthly roll → annual


# DROPPED (parsimony — see s4 cluster importance): register(FeatureSpec("f18_term_spread", "F18", _f18_term_spread, "E", 1, "new"))
# DROPPED (parsimony — see s4 cluster importance): register(FeatureSpec("f18_term_spread_z63", "F18", _f18_term_spread_z63, "E", 63, "new"))
# DROPPED (parsimony — see s4 cluster importance): register(FeatureSpec("f18_contango_flag", "F18", _f18_contango_flag, "E", 1, "new"))
# DROPPED (parsimony — see s4 cluster importance): register(FeatureSpec("f18_term_chg5", "F18", _f18_term_chg5, "E", 5, "new"))
# DROPPED (parsimony — see s4 cluster importance): register(FeatureSpec("f18_roll_yield", "F18", _f18_roll_yield, "E", 1, "new"))


# ---------------------------------------------------------------------------
# F19 — Options-implied volatility family.
# ---------------------------------------------------------------------------


def _iv_series(ctx: FeatureContext, field: str) -> pd.Series:
    """Pull an IV series for this instrument's BBG IV underlying."""
    root = _BBG_IV_MAP.get(ctx.instrument)
    if root is None:
        return pd.Series(np.nan, index=ctx.frame.index)
    col = f"{root}_{field}"
    if ctx.options_iv.empty or col not in ctx.options_iv.columns:
        return pd.Series(np.nan, index=ctx.frame.index)
    return ctx.options_iv[col].reindex(ctx.frame.index, method="ffill")


def _f19_atm_iv_1m(ctx: FeatureContext) -> pd.Series:
    return _iv_series(ctx, "IV1M_ATM")


def _f19_atm_iv_3m(ctx: FeatureContext) -> pd.Series:
    return _iv_series(ctx, "IV3M_ATM")


def _f19_iv_term_slope(ctx: FeatureContext) -> pd.Series:
    """3m IV - 1m IV — IV term structure (positive = upward sloping curve)."""
    return _iv_series(ctx, "IV3M_ATM") - _iv_series(ctx, "IV1M_ATM")


def _f19_iv_rv_spread_20(ctx: FeatureContext) -> pd.Series:
    """1m ATM IV − 20d realised vol (annualised) — VRP (variance risk premium) proxy.

    Both in annualised %. The BBG IV is typically in %, the realised vol from
    log_returns is in fractional units, so the BBG side is divided by 100 to
    align. (Empirically the BBG ATM IV in 2020 for SPX is ~30 → 0.30 in
    fractional terms — and we multiply by sqrt(252) on realised. Adjust as
    needed via the rescale.)
    """
    iv = _iv_series(ctx, "IV1M_ATM") / 100.0  # IV in BBG is %, convert to fractional
    rv = log_returns(ctx.frame["close"]).rolling(20).std() * np.sqrt(252.0)
    return iv - rv


def _f19_skew_1m(ctx: FeatureContext) -> pd.Series:
    """90% MNY IV − 110% MNY IV — skew (positive = put expensive vs call).

    NOTE methodology spec / validation report: this is a moneyness-based proxy for
    25-delta skew (90% MNY ≈ 25Δ put for typical equity vols; not strict).
    """
    return _iv_series(ctx, "IV1M_90MNY") - _iv_series(ctx, "IV1M_110MNY")


def _f19_iv_pctile_252(ctx: FeatureContext) -> pd.Series:
    """252-day rolling percentile of ATM 1m IV (drift-safe)."""
    iv = _iv_series(ctx, "IV1M_ATM")
    return iv.rolling(252, min_periods=126).rank(pct=True)


# DROPPED (parsimony — see s4 cluster importance): register(FeatureSpec("f19_atm_iv_1m", "F19", _f19_atm_iv_1m, "E", 1, "new"))
# DROPPED (parsimony — see s4 cluster importance): register(FeatureSpec("f19_atm_iv_3m", "F19", _f19_atm_iv_3m, "E", 1, "new"))
# DROPPED (parsimony — see s4 cluster importance): register(FeatureSpec("f19_iv_term_slope", "F19", _f19_iv_term_slope, "E", 1, "new"))
# DROPPED (parsimony — see s4 cluster importance): register(FeatureSpec("f19_iv_rv_spread_20", "F19", _f19_iv_rv_spread_20, "E", 20, "new"))
# DROPPED (parsimony — see s4 cluster importance): register(FeatureSpec("f19_skew_1m", "F19", _f19_skew_1m, "E", 1, "new"))
# DROPPED (parsimony — see s4 cluster importance): register(FeatureSpec("f19_iv_pctile_252", "F19", _f19_iv_pctile_252, "E", 126, "new"))


# ---------------------------------------------------------------------------
# F22 — Event flags (partial; only EIA crude release + change z-score).
# ---------------------------------------------------------------------------


def _f22_eia_release_flag(ctx: FeatureContext) -> pd.Series:
    """1 on EIA crude release day, 0 elsewhere — energy instruments only."""
    if ctx.instrument not in _ENERGY_INSTRUMENTS:
        return pd.Series(np.nan, index=ctx.frame.index)
    if ctx.eia_release_flag.empty:
        return pd.Series(np.nan, index=ctx.frame.index)
    return ctx.eia_release_flag["EIA_CRUDE_RELEASE_FLAG"].reindex(
        ctx.frame.index, method="ffill"
    ).fillna(0).astype(float)


def _f22_eia_crude_change_z26w(ctx: FeatureContext) -> pd.Series:
    """26-week (~130 business day) z-score of weekly crude inventory change."""
    if ctx.instrument not in _ENERGY_INSTRUMENTS:
        return pd.Series(np.nan, index=ctx.frame.index)
    if ctx.eia_crude.empty:
        return pd.Series(np.nan, index=ctx.frame.index)
    chg = ctx.eia_crude["EIA_CRUDE_CHANGE_KB"].reindex(ctx.frame.index, method="ffill")
    return rolling_zscore(chg, window=130)


register(FeatureSpec("f22_eia_release_flag", "F22", _f22_eia_release_flag, "E", 1, "new"))
register(FeatureSpec("f22_eia_crude_change_z26w", "F22", _f22_eia_crude_change_z26w, "E", 130, "new"))
