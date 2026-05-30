"""Point-in-time-lagged macro block for §1/§3 (S1.7).

``additional_data.xlsx`` ships 22 macro series as alternating ``(date, value)`` column pairs,
each carrying **observation dates only** — no release dates. A naive join would let a trade-day
feature read a number that was not yet *published* (EIA inventories are released ~Wednesday for
the prior week; PMIs ~a month after the reference month). So each series is assigned a
conservative **publication lag** (aqms-python L2 "lags"; standard release calendars), its
observation index is shifted forward by that lag to an *availability* date, and the value is
then forward-filled onto the trade calendar. Trade date ``t`` therefore sees only the most
recent value released on or before ``t`` — point-in-time correct and truncation-invariant.

The derived features are the theory-of-storage / cross-asset drivers the brief's §3 asks for
(energy inventories; gold↔real-rates/USD; copper↔China PMI; equity↔VIX term-slope/credit), all
stationary changes or neutral-centred levels. The block is additive enrichment — documented in
the write-up Limitations — and is fit-free (causal transforms only), so it needs no ``fit_end``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from stml.io import _find_repo_root

#: Conservative publication lag (calendar days) from observation date to public availability.
PUBLICATION_LAG: dict[str, int] = {
    # daily market series: one-day implementation lag (close known EOD, traded next day)
    "VIX": 1, "VIX3M": 1, "MOVE": 1, "DXY": 1, "CBOE_SKEW": 1,
    "2Y_UST": 1, "10Y_UST": 1, "10Y_BUND": 1, "EURUSD": 1, "BAL_DRY_INDEX": 1,
    "HY_OAS": 1, "IG_OAS": 1, "TIPS10Y": 1, "BE10Y": 1, "LME_COPPER_STOCK": 1,
    # EIA weekly petroleum/gas status report: ~5 days after the reference week
    "EIA_CRUDE_STOCK": 5, "EIA_DIST_STOCK": 5, "EIA_GASOLINE_STOCK": 5, "EIA_NG_STOCK": 5,
    # monthly PMIs: published the following month (~30 days after the reference month-end)
    "US_ISM_MFG_PMI": 30, "CHINA_PMI_MFG": 30, "GERMANY_PMI_MFG": 30,
}

_CHG_WINDOW = 20  # ~4 weeks of trade days, enough to span ~4 weekly EIA updates


def _macro_path(data_dir: str | Path | None = None) -> Path:
    if data_dir is None:
        data_dir = _find_repo_root(Path.cwd().resolve()) / "data"
    return Path(data_dir) / "additional_data.xlsx"


def load_macro_series(data_dir: str | Path | None = None) -> dict[str, pd.Series]:
    """Parse the alternating ``(date, value)`` pairs into one clean Series per macro series."""
    df = pd.read_excel(_macro_path(data_dir), sheet_name="Sheet1", header=0)
    cols = list(df.columns)
    out: dict[str, pd.Series] = {}
    for i in range(1, len(cols), 2):  # value columns sit at odd positions, dates at i-1
        name = str(cols[i])
        idx = pd.DatetimeIndex(df.iloc[:, i - 1].to_numpy())
        s = pd.Series(df.iloc[:, i].to_numpy(), index=idx, name=name).dropna()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        out[name] = s
    return out


def pit_align(series: pd.Series, trade_dates, lag_days: int) -> pd.Series:
    """Forward-fill a macro series onto ``trade_dates`` using only data released by each date.

    The observation index is shifted forward by ``lag_days`` (observation -> availability), so a
    forward-fill at trade date ``t`` returns the last value whose availability is on or before
    ``t`` — never a same-period number that had not yet been published.
    """
    avail = series.dropna().sort_index()
    avail = avail[~avail.index.duplicated(keep="last")]
    avail.index = avail.index + pd.Timedelta(days=lag_days)
    avail = avail[~avail.index.duplicated(keep="last")].sort_index()
    return avail.reindex(pd.DatetimeIndex(trade_dates), method="ffill")


def _lag(name: str) -> int:
    return PUBLICATION_LAG.get(name, 1)


def _log_change(level: pd.Series, window: int = _CHG_WINDOW) -> pd.Series:
    """Past-looking log change of a positive level series (causal)."""
    safe = level.where(level > 0)
    return np.log(safe).diff(window)


def macro_features(
    trade_dates,
    *,
    series: dict[str, pd.Series] | None = None,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Assemble the PIT-lagged macro feature block on ``trade_dates`` (theory-of-storage + macro).

    Pass ``series`` to inject a custom raw-series dict (testing); otherwise it is loaded from
    ``additional_data.xlsx``. Missing series simply yield absent columns. Truncation-invariant:
    every feature at ``t`` depends only on data released on or before ``t``.
    """
    raw = series if series is not None else load_macro_series(data_dir)
    dates = pd.DatetimeIndex(trade_dates)

    def aligned(name: str) -> pd.Series | None:
        s = raw.get(name)
        return None if s is None else pit_align(s, dates, _lag(name))

    feats: dict[str, pd.Series] = {}

    def put(col: str, value: pd.Series | None) -> None:
        if value is not None:
            feats[col] = value

    vix, vix3m = aligned("VIX"), aligned("VIX3M")
    put("macro_vix_level", vix)
    if vix is not None and vix3m is not None:
        put("macro_vix_term_slope", vix3m - vix)
    put("macro_move", aligned("MOVE"))

    dxy = aligned("DXY")
    if dxy is not None:
        put("macro_dxy_chg20", _log_change(dxy))

    hy, ig = aligned("HY_OAS"), aligned("IG_OAS")
    put("macro_hy_oas", hy)
    put("macro_ig_oas", ig)
    if hy is not None and ig is not None:
        put("macro_credit_slope", hy - ig)

    put("macro_real_rate", aligned("TIPS10Y"))
    put("macro_breakeven", aligned("BE10Y"))

    for raw_name, col in (
        ("EIA_CRUDE_STOCK", "macro_eia_crude_chg"),
        ("EIA_DIST_STOCK", "macro_eia_dist_chg"),
        ("EIA_GASOLINE_STOCK", "macro_eia_gasoline_chg"),
        ("EIA_NG_STOCK", "macro_eia_ng_chg"),
    ):
        s = aligned(raw_name)
        if s is not None:
            put(col, _log_change(s))

    china = aligned("CHINA_PMI_MFG")
    if china is not None:
        put("macro_china_pmi", china - 50.0)  # deviation from the expansion/contraction line
    us_pmi = aligned("US_ISM_MFG_PMI")
    if us_pmi is not None:
        put("macro_us_pmi", us_pmi - 50.0)
    copper = aligned("LME_COPPER_STOCK")
    if copper is not None:
        put("macro_copper_stock_chg", _log_change(copper))

    return pd.DataFrame(feats, index=dates)
