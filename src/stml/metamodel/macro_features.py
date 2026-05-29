"""
macro_features.py
=================
Cross-asset macro context features (F11) for the triple-barrier metamodel.

This is a TF-class (fitted) feature family built **only** from
``data/additional_data.xlsx``. Like the F4 latent stack it carries a frozen
FE-train standardizer (:class:`MacroBundle`), but the quantity it standardizes
is a small, curated panel of macro series rather than the engineered-feature
matrix. The family is **global**: the same macro level / momentum values are
broadcast identically to every instrument on a given date (the join happens in
:class:`stml.metamodel.pipeline.FeaturePipeline`).

Curation (spec ``.omc/specs/deep-interview-macro-features.md``)
---------------------------------------------------------------
* **12 standalone series** (:data:`KEEP`): VIX, MOVE, DXY, 10Y_UST, 2Y_UST,
  HY_OAS, BE10Y, TIPS10Y, EIA_CRUDE_STOCK, EIA_NG_STOCK, US_ISM_MFG_PMI,
  CHINA_PMI_MFG.
* **3 spreads** (:data:`SPREADS`) from 2 extra spread-only inputs
  (:data:`SPREAD_INPUTS` = VIX3M, IG_OAS): ``vix_term = VIX3M - VIX``,
  ``curve_slope = 10Y_UST - 2Y_UST``, ``credit_diff = HY_OAS - IG_OAS``.
* The other 10 series in the sheet are dropped (:data:`DROPPED`); VIX3M and
  IG_OAS appear only inside spreads, never as standalone features.

Per series the family emits a PIT-applied **level** plus two **momentum**
(``chg{h}``) columns, ``12*3 + 3*3 = 45`` columns in total, all leakage-class
**TF**. The column names are produced by one traversal,
:func:`macro_feature_columns`, which is *also* the traversal the catalog uses,
so produced columns and catalog entries cannot drift.

Leakage / publication-lag contract (CONTRACT_FE Section 0 -- the graded crux)
-----------------------------------------------------------------------------
Every macro value at trade date ``t`` uses only information whose **availability
date** is ``<= t`` (an as-of / point-in-time merge), and the z-score stats are
frozen from the FE-train partition (``<= 2021-07-01``) and applied forward.

* **Stamp-grid cadence recovery** (NOT value-collapse, which would silently
  drop genuine re-releases and back-date them -> lookahead). Native cadence is
  recovered by resampling onto a fixed grid:
  - ``daily`` series keep every observation date; availability = the obs date
    (close, EOD ``t``); lag 0.
  - ``weekly_eia`` (EIA crude / NG): the value changes on the **Friday**
    week-ending date, recovered by ``resample("W-FRI").last()``; availability =
    Friday stamp **+ 6 calendar days** (a conservative buffer past the true
    mid-week EIA release, ~Wed crude / ~Thu NG, so availability never precedes
    the real release).
  - ``monthly_pmi`` (US ISM / China): the value changes at **month-end**,
    recovered by ``resample("ME").last()``; availability = the **next business
    day** strictly after the month-end stamp (the ISM/Caixin release proxy).
* **Momentum is populated from the first trade date.** Because every curated
  series starts years before the 2020-01-03 signal window, the momentum
  ``chg{h}`` columns must not be all-NaN at the window start. To achieve that
  the applied-level panel is built on a **business-day grid buffered backward**
  past the first trade date (:data:`_GRID_BUFFER_DAYS`); the ``diff(h)`` is
  computed on that regular grid and the result is then sliced to the trade
  dates. The horizon ``h`` is therefore ``h`` business days. The buffer is
  dropped before the FE-train fit, so it never leaks into the frozen stats.
* **Truncation-invariance.** An observation whose stamp is ``> T`` has
  availability ``>= stamp > T``, so it can never alter an as-of value on a trade
  date ``<= T``; truncating the input observations at ``T`` leaves every F11
  value (level and ``chg{h}``) on dates ``< T`` unchanged. Both
  :func:`build_applied_panel` and :func:`assemble_macro_raw` accept a
  pre-loaded ``raw`` dict so this is directly testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_MACRO_PATH",
    "KEEP",
    "SPREAD_INPUTS",
    "SPREADS",
    "DROPPED",
    "MOMENTUM",
    "MacroBundle",
    "load_macro_raw",
    "compute_availability",
    "build_applied_panel",
    "assemble_macro_raw",
    "macro_feature_columns",
    "fit_macro",
    "transform_macro",
]

#: Default macro source workbook (paired-column layout, see :func:`load_macro_raw`).
DEFAULT_MACRO_PATH = "data/additional_data.xlsx"

# Release classes -- drive the publication-lag rule and the momentum horizons.
_DAILY = "daily"
_WEEKLY_EIA = "weekly_eia"
_MONTHLY_PMI = "monthly_pmi"

#: Backward business-day buffer (in calendar days) prepended to the applied
#: panel so the ``chg{h}`` momentum columns are populated on the first trade
#: date. ~200 calendar days ~= 140 business days, comfortably > the 63d horizon.
_GRID_BUFFER_DAYS = 200

#: The 12 standalone kept series: ``name -> (release_class, what_it_captures)``.
KEEP: dict[str, tuple[str, str]] = {
    "VIX": (_DAILY, "CBOE VIX 30-day implied-volatility index (equity fear gauge)."),
    "MOVE": (_DAILY, "ICE BofA MOVE index (Treasury-option implied volatility)."),
    "DXY": (_DAILY, "Trade-weighted US dollar index (USD strength)."),
    "10Y_UST": (_DAILY, "10-year US Treasury benchmark yield."),
    "2Y_UST": (_DAILY, "2-year US Treasury benchmark yield."),
    "HY_OAS": (_DAILY, "ICE BofA US high-yield option-adjusted credit spread."),
    "BE10Y": (_DAILY, "10-year breakeven inflation rate (nominal minus real)."),
    "TIPS10Y": (_DAILY, "10-year TIPS real yield."),
    "EIA_CRUDE_STOCK": (_WEEKLY_EIA, "EIA weekly US commercial crude-oil inventories."),
    "EIA_NG_STOCK": (_WEEKLY_EIA, "EIA weekly US working natural-gas in storage."),
    "US_ISM_MFG_PMI": (_MONTHLY_PMI, "US ISM manufacturing PMI (diffusion index)."),
    "CHINA_PMI_MFG": (_MONTHLY_PMI, "China Caixin manufacturing PMI (diffusion index)."),
}

#: Spread-only inputs -- read from the sheet but never returned as standalone
#: features (they appear only inside :data:`SPREADS`).
SPREAD_INPUTS: tuple[str, ...] = ("VIX3M", "IG_OAS")

#: The 3 spreads: ``name -> (minuend, subtrahend, release_class, captures)``.
SPREADS: dict[str, tuple[str, str, str, str]] = {
    "vix_term": (
        "VIX3M",
        "VIX",
        _DAILY,
        "VIX term structure (3M minus spot): equity-vol contango/backwardation.",
    ),
    "curve_slope": (
        "10Y_UST",
        "2Y_UST",
        _DAILY,
        "Treasury 2s10s curve slope (10Y minus 2Y yield).",
    ),
    "credit_diff": (
        "HY_OAS",
        "IG_OAS",
        _DAILY,
        "High-yield minus investment-grade OAS differential (credit stress).",
    ),
}

#: The 10 dropped sheet series (zero standalone columns). VIX3M / IG_OAS are
#: dropped *as standalone* but still read as :data:`SPREAD_INPUTS`.
DROPPED: tuple[str, ...] = (
    "CBOE_SKEW",
    "10Y_BUND",
    "EURUSD",
    "BAL_DRY_INDEX",
    "LME_COPPER_STOCK",
    "EIA_DIST_STOCK",
    "EIA_GASOLINE_STOCK",
    "GERMANY_PMI_MFG",
)

#: Momentum horizons (in business days) per release class. Monthly uses the
#: longer (21, 63) so the change is not ~0 between releases (a dead column).
MOMENTUM: dict[str, tuple[int, int]] = {
    _DAILY: (5, 20),
    _WEEKLY_EIA: (5, 20),
    _MONTHLY_PMI: (21, 63),
}


def _slug(name: str) -> str:
    """Column-name slug for a series (lower-cased sheet name)."""
    return name.lower()


def _all_series_classes() -> dict[str, str]:
    """``series_name -> release_class`` for the 12 KEEP + 2 SPREAD_INPUTS."""
    classes = {name: rcls for name, (rcls, _) in KEEP.items()}
    classes.update({name: _DAILY for name in SPREAD_INPUTS})
    return classes


# --------------------------------------------------------------------------- #
# Raw ingest + cadence recovery                                               #
# --------------------------------------------------------------------------- #
def load_macro_raw(path: str = DEFAULT_MACRO_PATH) -> dict[str, pd.Series]:
    """Parse the paired-column workbook into stamp-indexed value series.

    The workbook is a paired-column panel: each value column at position ``j``
    is preceded by its own date column at position ``j-1`` (an ``Unnamed``
    header), pre-forward-filled to (business- or calendar-) daily. Only the 12
    :data:`KEEP` series and the 2 :data:`SPREAD_INPUTS` are read; the 10
    :data:`DROPPED` series (including the empty ``GERMANY_PMI_MFG``) are never
    touched.

    Native cadence is recovered by a **stamp grid**, not by collapsing
    consecutive-equal values (which would drop genuine re-releases and
    back-date them):

    * ``daily`` -> the observation dates as-is;
    * ``weekly_eia`` -> ``resample("W-FRI").last()`` (Friday week-ending stamp);
    * ``monthly_pmi`` -> ``resample("ME").last()`` (month-end stamp).

    Parameters
    ----------
    path : str, default :data:`DEFAULT_MACRO_PATH`
        Path to ``additional_data.xlsx`` (read with the ``openpyxl`` engine).

    Returns
    -------
    dict[str, pd.Series]
        ``series_name -> value Series`` indexed by the recovered **stamp** date
        (sorted, de-duplicated), one entry per :data:`KEEP` key and per
        :data:`SPREAD_INPUTS` member.
    """
    frame = pd.read_excel(path, engine="openpyxl", header=0)
    cols = list(frame.columns)

    def _obs(name: str) -> pd.Series:
        """Date-indexed observation series for one value column (date at j-1)."""
        j = cols.index(name)
        dates = pd.to_datetime(frame.iloc[:, j - 1], errors="coerce")
        values = pd.to_numeric(frame.iloc[:, j], errors="coerce")
        s = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates))
        s = s[s.index.notna() & s.notna()].sort_index()
        return s[~s.index.duplicated(keep="last")]

    out: dict[str, pd.Series] = {}
    for name, rcls in _all_series_classes().items():
        obs = _obs(name)
        if rcls == _WEEKLY_EIA:
            stamped = obs.resample("W-FRI").last().dropna()
        elif rcls == _MONTHLY_PMI:
            stamped = obs.resample("ME").last().dropna()
        else:
            stamped = obs
        out[name] = stamped
    return out


def compute_availability(stamp_date: pd.Timestamp, release_class: str) -> pd.Timestamp:
    """Map a recovered stamp date to its point-in-time **availability** date.

    Applies the per-class publication lag:

    * ``daily`` -> the stamp itself (close, EOD ``t``; lag 0);
    * ``weekly_eia`` -> stamp ``+ 6 calendar days`` (Friday + 6 = next Thursday,
      a conservative buffer past the real ~Wed/~Thu EIA release);
    * ``monthly_pmi`` -> the **next business day** strictly after the month-end
      stamp (e.g. ``2020-01-31`` Friday -> ``2020-02-03`` Monday).

    Parameters
    ----------
    stamp_date : pd.Timestamp
        The recovered reference / stamp date from :func:`load_macro_raw`.
    release_class : str
        One of ``"daily"``, ``"weekly_eia"``, ``"monthly_pmi"``.

    Returns
    -------
    pd.Timestamp
        The earliest date on which the value is usable.
    """
    stamp = pd.Timestamp(stamp_date)
    if release_class == _DAILY:
        return stamp
    if release_class == _WEEKLY_EIA:
        return stamp + pd.Timedelta(days=6)
    if release_class == _MONTHLY_PMI:
        return stamp + pd.tseries.offsets.BDay(1)
    raise ValueError(f"unknown release_class {release_class!r}")


# --------------------------------------------------------------------------- #
# Point-in-time applied panel                                                 #
# --------------------------------------------------------------------------- #
def build_applied_panel(
    trade_dates: object,
    path: str = DEFAULT_MACRO_PATH,
    raw: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """As-of merge the macro levels onto a backward-buffered business-day grid.

    For each kept series and spread input the stamp-indexed observations are
    re-stamped with their :func:`compute_availability` date and then carried
    forward (``reindex(grid, method="ffill")`` = as-of-backward) onto a
    business-day grid spanning ``[min(trade_dates) - buffer, max(trade_dates)]``.
    The backward buffer (:data:`_GRID_BUFFER_DAYS`) gives the downstream
    ``diff(h)`` enough history to be populated on the first trade date.

    Parameters
    ----------
    trade_dates : iterable of datetime-like
        The trade dates the matrix is built on (the nonzero-signal union).
    path : str, default :data:`DEFAULT_MACRO_PATH`
        Workbook path (used only when ``raw`` is ``None``).
    raw : dict[str, pd.Series], optional
        Pre-loaded stamp-indexed series from :func:`load_macro_raw`. Injecting a
        truncated ``raw`` is how truncation-invariance is tested.

    Returns
    -------
    pd.DataFrame
        Business-day-grid-indexed frame of PIT-applied **levels**, one column
        per kept series and spread input.
    """
    trade_idx = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(trade_dates))))
    series = load_macro_raw(path) if raw is None else raw

    grid = pd.bdate_range(
        start=trade_idx.min() - pd.Timedelta(days=_GRID_BUFFER_DAYS),
        end=trade_idx.max(),
    )
    grid = grid.union(trade_idx)

    applied: dict[str, pd.Series] = {}
    for name, rcls in _all_series_classes().items():
        obs = series[name]
        avail = pd.DatetimeIndex(
            [compute_availability(d, rcls) for d in obs.index]
        )
        by_avail = pd.Series(obs.to_numpy(dtype=float), index=avail).sort_index()
        by_avail = by_avail[~by_avail.index.duplicated(keep="last")]
        applied[name] = by_avail.reindex(grid, method="ffill")

    return pd.DataFrame(applied, index=grid)


def macro_feature_columns() -> list[str]:
    """The 45 produced F11 column names in canonical order (the one traversal).

    This is the single source of truth shared by :func:`assemble_macro_raw`
    (which produces the columns) and the catalog registration (which documents
    them), so the produced set and the catalog set cannot drift.

    Returns
    -------
    list[str]
        ``f11_<slug>_level`` / ``_chg{h1}`` / ``_chg{h2}`` per standalone series,
        then ``f11_spread_<name>_level`` / ``_chg{h1}`` / ``_chg{h2}`` per spread.
    """
    cols: list[str] = []
    for name, (rcls, _) in KEEP.items():
        slug = _slug(name)
        h1, h2 = MOMENTUM[rcls]
        cols += [f"f11_{slug}_level", f"f11_{slug}_chg{h1}", f"f11_{slug}_chg{h2}"]
    for sname, (_, _, rcls, _) in SPREADS.items():
        h1, h2 = MOMENTUM[rcls]
        cols += [
            f"f11_spread_{sname}_level",
            f"f11_spread_{sname}_chg{h1}",
            f"f11_spread_{sname}_chg{h2}",
        ]
    return cols


def assemble_macro_raw(
    trade_dates: object,
    path: str = DEFAULT_MACRO_PATH,
    raw: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Build the raw (pre-standardization) F11 features on the trade dates.

    From the PIT-applied levels (:func:`build_applied_panel`) this derives, per
    standalone series, the ``level`` plus two ``diff(h)`` momentum columns, and
    per spread the (minuend - subtrahend) ``level`` plus its two momentum
    columns. The ``diff(h)`` is computed on the buffered business-day grid and
    only then sliced to ``trade_dates``, so the momentum columns are populated
    on the first trade date. Column order is fixed by
    :func:`macro_feature_columns`.

    Parameters
    ----------
    trade_dates : iterable of datetime-like
        The trade dates the matrix is built on.
    path : str, default :data:`DEFAULT_MACRO_PATH`
        Workbook path (used only when ``raw`` is ``None``).
    raw : dict[str, pd.Series], optional
        Pre-loaded series (see :func:`build_applied_panel`).

    Returns
    -------
    pd.DataFrame
        Date-indexed (one row per trade date) frame with exactly the 45
        :func:`macro_feature_columns`, all raw / unstandardized levels.
    """
    trade_idx = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(trade_dates))))
    applied = build_applied_panel(trade_idx, path=path, raw=raw)

    out: dict[str, pd.Series] = {}
    for name, (rcls, _) in KEEP.items():
        slug = _slug(name)
        h1, h2 = MOMENTUM[rcls]
        level = applied[name]
        out[f"f11_{slug}_level"] = level
        out[f"f11_{slug}_chg{h1}"] = level.diff(h1)
        out[f"f11_{slug}_chg{h2}"] = level.diff(h2)
    for sname, (minuend, subtrahend, rcls, _) in SPREADS.items():
        h1, h2 = MOMENTUM[rcls]
        level = applied[minuend] - applied[subtrahend]
        out[f"f11_spread_{sname}_level"] = level
        out[f"f11_spread_{sname}_chg{h1}"] = level.diff(h1)
        out[f"f11_spread_{sname}_chg{h2}"] = level.diff(h2)

    frame = pd.DataFrame(out, index=applied.index)[macro_feature_columns()]
    return frame.reindex(trade_idx)


# --------------------------------------------------------------------------- #
# Frozen FE-train standardizer bundle                                         #
# --------------------------------------------------------------------------- #
@dataclass
class MacroBundle:
    """Frozen FE-train z-score artifacts for the F11 macro family.

    Attributes
    ----------
    mean_, std_ : np.ndarray
        Per-column FE-train mean / std (length ``len(feature_cols)``). A zero or
        non-finite std is replaced by ``1.0`` (the regime-GMM std guard).
    feature_cols : list[str]
        Frozen column order; transform inputs are reindexed to exactly this.
    train_index : pd.Index
        The FE-train trade dates the stats were fit on (one row per trade date).
    lag_config : dict
        JSON-serialisable description of the per-class publication-lag policy,
        series classes and momentum horizons (recorded into the provenance).
    """

    mean_: np.ndarray
    std_: np.ndarray
    feature_cols: list[str]
    train_index: pd.Index
    lag_config: dict


def _lag_config() -> dict:
    """The publication-lag / cadence policy recorded on every bundle."""
    return {
        "daily": "availability = observation date (close, EOD t); lag 0 days",
        "weekly_eia": "Friday W-FRI stamp + 6 calendar days",
        "monthly_pmi": "month-end (ME) stamp + 1 business day",
        "momentum_basis": "diff(h) on PIT-applied level over a business-day grid",
        "series_classes": {name: rcls for name, (rcls, _) in KEEP.items()},
        "spread_classes": {name: rcls for name, (_, _, rcls, _) in SPREADS.items()},
        "momentum_horizons": {rcls: list(h) for rcls, h in MOMENTUM.items()},
    }


def fit_macro(raw_train: pd.DataFrame) -> MacroBundle:
    """Freeze the per-column z-score stats from the FE-train macro frame.

    Parameters
    ----------
    raw_train : pd.DataFrame
        The **date-deduplicated** (one row per FE-train trade date) raw macro
        frame restricted to dates ``<= fe_train_end``. Must NOT be the
        instrument-stacked panel (that would 11x-inflate the stats).

    Returns
    -------
    MacroBundle
        Frozen ``mean_`` / ``std_`` (std==0 or non-finite -> 1.0),
        ``feature_cols``, ``train_index`` and ``lag_config``.
    """
    feature_cols = list(raw_train.columns)
    x = raw_train.to_numpy(dtype=float)

    mean_ = np.nanmean(x, axis=0)
    std_ = np.nanstd(x, axis=0)
    mean_ = np.where(np.isfinite(mean_), mean_, 0.0)
    std_ = np.where((std_ == 0.0) | ~np.isfinite(std_), 1.0, std_)

    return MacroBundle(
        mean_=mean_,
        std_=std_,
        feature_cols=feature_cols,
        train_index=raw_train.index,
        lag_config=_lag_config(),
    )


def transform_macro(bundle: MacroBundle, raw_full: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen FE-train z-score to the full raw macro frame.

    Reindexes ``raw_full`` to the frozen ``feature_cols`` and standardizes with
    the frozen ``mean_`` / ``std_``. NaN cells are preserved (none are expected
    at the window start -- see the module docstring on momentum population).

    Parameters
    ----------
    bundle : MacroBundle
        The frozen bundle from :func:`fit_macro`.
    raw_full : pd.DataFrame
        Date-indexed raw macro frame (e.g. :func:`assemble_macro_raw` over all
        trade dates).

    Returns
    -------
    pd.DataFrame
        Date-indexed standardized frame with columns ``bundle.feature_cols``.
    """
    mat = raw_full.reindex(columns=bundle.feature_cols).to_numpy(dtype=float)
    standardized = (mat - bundle.mean_) / bundle.std_
    return pd.DataFrame(
        standardized, index=raw_full.index, columns=bundle.feature_cols
    )
