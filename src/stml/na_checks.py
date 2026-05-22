"""
na_checks.py
============
Rigorous missing-data (NA) diagnosis, cleaning, and calendar-aware panel
statistics for the 11-instrument futures universe in ``data/ohlcv_data.csv``.

The module supersedes the earlier, un-importable ``na-checks.py`` (a hyphen is
not a legal Python identifier) and merges three previously separate concerns:

  1. **Why a row is missing** -- a self-contained NYMEX/COMEX/CME-equity + Eurex
     holiday calendar (1990-2022) plus the ad-hoc closures (9/11, Reagan, Ford,
     Sandy, Bush). See :func:`build_calendar_info`. External calendar libraries
     (``pandas_market_calendars`` / ``exchange_calendars``) are used **only as
     an optional cross-check** when installed; the hand-coded tables are the
     authoritative source because the libraries are unreliable before ~2000 and
     most instruments here start in 1990. See ``refs/missing-data-report.md``.

  2. **Which rows are spurious** -- weekend rows (calendar-impossible) are
     dropped; zero-volume *weekday* rows are KEPT (they carry a valid settle
     price; volume was simply not recorded) but flagged. See :func:`clean_long`
     and :func:`detect_anomalous_rows`.

  3. **How to compute stats on a ragged panel** -- returns are computed on each
     instrument's own dense series (no fabricated gaps); cross-sectional
     correlation is pairwise-complete then repaired to PSD; rolling pairwise
     correlation aligns on the intersection of the two trading calendars. See
     :func:`native_returns`, :func:`corr_max_info`, :func:`rolling_pair_corr`.

Quick start
-----------
    from stml.na_checks import load_clean_ohlcv, native_returns, wide_returns
    long = load_clean_ohlcv()                 # raw load + artifact removal
    rets = native_returns(long)               # per-instrument log returns
    W    = wide_returns(rets)                  # date x instrument, structural NaNs only

Or, end to end with metadata + report inputs:
    python -m stml.na_checks                   # writes diagnostics/ CSVs
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Optional calendar libraries -- cross-check only; never required.            #
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised only when the optional dep is installed
    import pandas_market_calendars as mcal

    HAVE_MCAL = True
except ImportError:
    mcal = None
    HAVE_MCAL = False


# --------------------------------------------------------------------------- #
# SECTION 0. Universe configuration                                           #
# --------------------------------------------------------------------------- #
INSTRUMENTS: list[str] = [
    "cl1s", "es1s", "fesx1s", "gc1s", "hg1s",
    "ho1s", "ng1s", "nq1s", "pl1s", "rb1s", "si1s",
]

# Each instrument -> primary calendar code + a venue group used by the
# cross-sectional diagnostic. CME_Equity (ES/NQ) and the unified CMES group
# (NYMEX energies + COMEX metals) and XEUR (Eurex / FESX) follow the research
# note refs/missing-holidays.md.
INSTRUMENT_MAP: dict[str, dict[str, str]] = {
    "cl1s":   {"calendar": "CMES",       "venue": "NYMEX"},   # WTI crude
    "ho1s":   {"calendar": "CMES",       "venue": "NYMEX"},   # heating oil
    "rb1s":   {"calendar": "CMES",       "venue": "NYMEX"},   # RBOB gasoline
    "ng1s":   {"calendar": "CMES",       "venue": "NYMEX"},   # Henry Hub natgas
    "gc1s":   {"calendar": "CMES",       "venue": "COMEX"},   # gold
    "si1s":   {"calendar": "CMES",       "venue": "COMEX"},   # silver
    "hg1s":   {"calendar": "CMES",       "venue": "COMEX"},   # copper
    "pl1s":   {"calendar": "CMES",       "venue": "COMEX"},   # platinum
    "es1s":   {"calendar": "CME_Equity", "venue": "CME_EQ"},  # E-mini S&P 500
    "nq1s":   {"calendar": "CME_Equity", "venue": "CME_EQ"},  # E-mini Nasdaq-100
    "fesx1s": {"calendar": "XEUR",       "venue": "EUREX"},   # Euro STOXX 50
}

# US-venue instruments (everything except FESX). Used for scope classification.
US_INSTRUMENTS = [t for t, m in INSTRUMENT_MAP.items() if m["venue"] != "EUREX"]
EUREX_INSTRUMENTS = [t for t, m in INSTRUMENT_MAP.items() if m["venue"] == "EUREX"]
CME_EQ_INSTRUMENTS = [t for t, m in INSTRUMENT_MAP.items() if m["venue"] == "CME_EQ"]

# Library calendars are only reliable from ~2000; before that we rely solely on
# the hand-coded tables below.
LIBRARY_RELIABLE_START = pd.Timestamp("2000-01-01")


# --------------------------------------------------------------------------- #
# SECTION 1. Hand-coded holiday tables (authoritative for 1990-2022)          #
# --------------------------------------------------------------------------- #
def _observed(d: date) -> date:
    """US federal observation rule: Saturday -> prior Friday, Sunday -> Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _easter(year: int) -> date:
    """Anonymous Gregorian (Meeus/Jones/Butcher) Easter algorithm. Avoids a
    hard dependency on dateutil.easter while staying exact for 1583-4099."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th `weekday` (Mon=0) of `month`. n>0 from start, n<0 from end."""
    if n > 0:
        d = date(year, month, 1)
        offset = (weekday - d.weekday()) % 7
        return d + timedelta(days=offset + 7 * (n - 1))
    # last/-2nd etc.
    if month == 12:
        d = date(year, 12, 31)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    offset = (d.weekday() - weekday) % 7
    return d - timedelta(days=offset - 7 * (n + 1))


# Ad-hoc, non-recurring full-day US-futures closures actually observed as
# missing rows in data/ohlcv_data.csv. (date, label, scope) where scope is:
#   "us_all"     -> closed across all US venues (commodities + CME equity)
#   "cme_equity" -> only ES/NQ closed; NYMEX/COMEX traded electronically
# These were verified empirically against the data, which is why several dates
# the literature flags (see ADHOC_EVENTS_NO_CLOSURE) are intentionally absent.
US_FUTURES_ADHOC_CLOSURES: list[tuple[str, str, str]] = [
    ("1994-04-27", "Nixon state funeral", "us_all"),
    ("2001-09-12", "9/11 aftermath - markets closed", "us_all"),
    ("2001-09-13", "9/11 aftermath - markets closed", "us_all"),
    ("2001-09-14", "9/11 - CME equity still closed (NYMEX/COMEX reopened via ACCESS)", "cme_equity"),
    ("2004-06-11", "Reagan state funeral", "us_all"),
    ("2007-01-02", "Ford state funeral (CME equity closed; NYMEX/COMEX electronic open)", "cme_equity"),
]

# Famous market events that the literature associates with closures but which
# this vendor's continuous series records WITH full rows (no missing data).
# Kept for documentation; NOT added to any holiday set.
ADHOC_EVENTS_NO_CLOSURE: list[tuple[str, str]] = [
    ("2001-09-11", "9/11 attacks - market opened then halted; pre-attack rows present"),
    ("2012-10-29", "Hurricane Sandy - all 11 instruments have rows (electronic/backfilled)"),
    ("2012-10-30", "Hurricane Sandy - all 11 instruments have rows (electronic/backfilled)"),
    ("2018-12-05", "George H.W. Bush funeral - all 11 instruments have rows"),
]


def build_us_futures_holidays(y0: int, y1: int, calendar: str = "CMES") -> dict[pd.Timestamp, str]:
    """Recurring CME/NYMEX/COMEX full-day closures, with historical caveats:
    MLK Day observed by exchanges only from 1998; Juneteenth only from 2022.

    Ad-hoc closures are filtered by ``calendar``: the unified ``CMES`` (energies
    + metals) calendar receives only ``us_all`` events because NYMEX/COMEX kept
    trading electronically on the CME-equity-only closures (Ford funeral,
    9/11 reopening), whereas ``CME_Equity`` (ES/NQ) receives both.
    """
    out: dict[pd.Timestamp, str] = {}
    for y in range(y0, y1 + 1):
        out[pd.Timestamp(_observed(date(y, 1, 1)))] = "New Year's Day"
        if y >= 1998:
            out[pd.Timestamp(_nth_weekday(y, 1, 0, 3))] = "Martin Luther King Jr. Day"
        out[pd.Timestamp(_nth_weekday(y, 2, 0, 3))] = "Presidents' Day"
        out[pd.Timestamp(_easter(y) - timedelta(days=2))] = "Good Friday"
        out[pd.Timestamp(_nth_weekday(y, 5, 0, -1))] = "Memorial Day"
        if y >= 2022:
            out[pd.Timestamp(_observed(date(y, 6, 19)))] = "Juneteenth"
        out[pd.Timestamp(_observed(date(y, 7, 4)))] = "Independence Day"
        out[pd.Timestamp(_nth_weekday(y, 9, 0, 1))] = "Labor Day"
        out[pd.Timestamp(_nth_weekday(y, 11, 3, 4))] = "Thanksgiving Day"
        out[pd.Timestamp(_observed(date(y, 12, 25)))] = "Christmas Day"
    allowed = {"us_all"} if calendar == "CMES" else {"us_all", "cme_equity"}
    for d_str, label, scope in US_FUTURES_ADHOC_CLOSURES:
        if scope in allowed:
            out[pd.Timestamp(d_str)] = label
    return out


def build_eurex_holidays(y0: int, y1: int) -> dict[pd.Timestamp, str]:
    """Recurring Eurex (XEUR / FESX) full-day closures. Dec 24 & Dec 31 are
    early closes (handled by :func:`build_early_closes`), not full closures."""
    out: dict[pd.Timestamp, str] = {}
    for y in range(y0, y1 + 1):
        e = _easter(y)
        out[pd.Timestamp(date(y, 1, 1))] = "New Year's Day (Eurex)"
        out[pd.Timestamp(e - timedelta(days=2))] = "Good Friday (Eurex)"
        out[pd.Timestamp(e + timedelta(days=1))] = "Easter Monday (Eurex)"
        out[pd.Timestamp(date(y, 5, 1))] = "Labour Day (Eurex)"
        out[pd.Timestamp(date(y, 12, 25))] = "Christmas Day (Eurex)"
        out[pd.Timestamp(date(y, 12, 26))] = "Boxing Day (Eurex)"
        # Whit Monday: observed ~1998-2007, then dropped; one-off return in 2015.
        if 1998 <= y <= 2007 or y == 2015:
            out[pd.Timestamp(e + timedelta(days=50))] = "Whit Monday (Eurex)"
    return out


def build_early_closes(y0: int, y1: int, eurex: bool = False) -> dict[pd.Timestamp, str]:
    """Half-day / bridge sessions where this vendor *sometimes* drops the row.

    These are NOT full closures: the market is open for an abbreviated session,
    so a missing row here is a benign vendor choice -- never a data glitch. The
    rules below were derived empirically from the observed gaps (see
    ``refs/missing-data-report.md``) and match the documented CME/Eurex
    early-close schedule.

    ``eurex=True`` returns only the year-end early closes (Dec 24 / Dec 31) for
    every year -- these are the dates the FESX feed drops (in practice ~post-2003).
    Emitting them in earlier years is harmless: an early-close label is only
    attached when a row is actually missing on that date. ``eurex=False`` adds
    the US-futures Thanksgiving and Independence-Day half-days and the year-end /
    new-year bridge days.
    """
    out: dict[pd.Timestamp, str] = {}
    for y in range(y0, y1 + 1):
        # Year-end early closes (both venues).
        if date(y, 12, 24).weekday() < 5:
            out[pd.Timestamp(date(y, 12, 24))] = "Christmas Eve (early close)"
        if date(y, 12, 31).weekday() < 5:
            out[pd.Timestamp(date(y, 12, 31))] = "New Year's Eve (early close)"
        if eurex:
            continue
        # --- US futures only below ---
        # Day after Thanksgiving (Black Friday, 1pm ET).
        tg = _nth_weekday(y, 11, 3, 4)
        out[pd.Timestamp(tg + timedelta(days=1))] = "Day after Thanksgiving (early close)"
        # July 3 (pre-Independence Day) when a weekday.
        if date(y, 7, 3).weekday() < 5:
            out[pd.Timestamp(date(y, 7, 3))] = "July 3 (pre-Independence Day early close)"
        # July 5 Friday bridge when July 4 is a Thursday.
        if date(y, 7, 4).weekday() == 3:
            out[pd.Timestamp(date(y, 7, 5))] = "July 5 (Independence Day Friday bridge)"
        # Dec 26 Friday bridge when Christmas is a Thursday.
        if date(y, 12, 25).weekday() == 3:
            out[pd.Timestamp(date(y, 12, 26))] = "Dec 26 (Christmas Friday bridge)"
        # New-year bridge: Jan 2 Friday when Jan 1 is a Thursday; Jan 3 Monday
        # when Jan 1 is a Saturday (first business day after the long weekend).
        if date(y, 1, 1).weekday() == 3:
            out[pd.Timestamp(date(y, 1, 2))] = "Jan 2 (New Year Friday bridge)"
        if date(y, 1, 1).weekday() == 5:
            out[pd.Timestamp(date(y, 1, 3))] = "Jan 3 (New Year Monday bridge)"
    return out


# --------------------------------------------------------------------------- #
# SECTION 2. Per-calendar session + holiday assembly                          #
# --------------------------------------------------------------------------- #
@dataclass
class CalendarInfo:
    """Expected sessions, holiday labels, and early-close labels for a calendar.

    ``sessions`` excludes full holidays but INCLUDES early-close days (the
    market is open for an abbreviated session, so a row is expected -- though
    this vendor sometimes drops it).
    """

    name: str
    holidays: dict[pd.Timestamp, str]
    early_closes: dict[pd.Timestamp, str]
    sessions: pd.DatetimeIndex


def build_calendar_info(
    cal_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cross_check: bool = True,
) -> CalendarInfo:
    """Build the expected session index + holiday/early-close maps for a calendar.

    Sessions = business days in ``[start, end]`` minus the hand-coded full
    holidays. If ``pandas_market_calendars`` is installed and ``cross_check`` is
    True, the library's session set (>= 2000) is compared and discrepancies are
    warned about -- but the hand-coded result is always what is returned, for
    reproducibility independent of library version drift.
    """
    if cal_name in ("CMES", "CME_Equity"):
        hols = build_us_futures_holidays(start.year, end.year, calendar=cal_name)
        early = build_early_closes(start.year, end.year, eurex=False)
    elif cal_name == "XEUR":
        hols = build_eurex_holidays(start.year, end.year)
        early = build_early_closes(start.year, end.year, eurex=True)
    else:
        raise ValueError(f"Unknown calendar code {cal_name!r}")

    all_bdays = pd.bdate_range(start, end)
    holiday_set = set(hols.keys())
    sessions = all_bdays[~all_bdays.isin(holiday_set)]

    if cross_check and HAVE_MCAL and end >= LIBRARY_RELIABLE_START:
        _cross_check_with_library(cal_name, sessions, start, end)

    return CalendarInfo(name=cal_name, holidays=hols, early_closes=early, sessions=sessions)


def _cross_check_with_library(
    cal_name: str,
    sessions: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:  # pragma: no cover - only runs when optional dep present
    """Warn (do not raise) when the hand-coded sessions disagree with the
    library over the window the library is reliable for (>= 2000)."""
    try:
        cal = mcal.get_calendar(cal_name)
        lib_start = max(start, LIBRARY_RELIABLE_START)
        sched = cal.schedule(
            start_date=lib_start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
        )
        lib_sessions = pd.DatetimeIndex(sched.index.normalize())
        ours = sessions[sessions >= lib_start]
        only_lib = lib_sessions.difference(ours)
        only_ours = ours.difference(lib_sessions)
        if len(only_lib) or len(only_ours):
            warnings.warn(
                f"[{cal_name}] hand-coded vs library session mismatch "
                f"(>= {lib_start.date()}): {len(only_lib)} library-only, "
                f"{len(only_ours)} hand-coded-only. Hand-coded is authoritative.",
                stacklevel=2,
            )
    except Exception as exc:  # noqa: BLE001 - cross-check must never break the run
        warnings.warn(f"Library cross-check skipped for {cal_name}: {exc}", stacklevel=2)


def build_all_calendars(
    start: pd.Timestamp, end: pd.Timestamp, cross_check: bool = True
) -> dict[str, CalendarInfo]:
    """Build a CalendarInfo for every distinct calendar code in the universe."""
    codes = sorted({m["calendar"] for m in INSTRUMENT_MAP.values()})
    return {c: build_calendar_info(c, start, end, cross_check=cross_check) for c in codes}


# --------------------------------------------------------------------------- #
# SECTION 3. Data loading                                                     #
# --------------------------------------------------------------------------- #
def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data").is_dir() and (p / "pyproject.toml").is_file():
            return p
    raise FileNotFoundError(
        f"Could not locate stml repo root (data/ + pyproject.toml) from {start}"
    )


def load_raw_ohlcv(path: str | Path | None = None) -> pd.DataFrame:
    """Load ``ohlcv_data.csv`` (long/tidy), parse dates, force numeric, sort.

    If ``path`` is None, walk up from the cwd to the repo root so this works
    from any notebook depth.
    """
    if path is None:
        path = _find_repo_root(Path.cwd().resolve()) / "data" / "ohlcv_data.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for c in ["open", "high", "low", "close", "volume", "open_interest"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["instrument", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# SECTION 4. Anomaly detection (rows that exist but should be questioned)      #
# --------------------------------------------------------------------------- #
def detect_anomalous_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Flag suspicious *present* rows. Returns long frame with a ``flag`` column.

    Flags
    -----
    weekend_row       : date is Sat/Sun -- calendar-impossible -> DROP.
    nonfinite_ohlc    : any of O/H/L/C is NaN -> price unusable.
    bad_ohlc_bounds   : high < low, or non-positive price -> corrupt.
    zero_volume_weekday : volume == 0 on a weekday. KEEP the price (valid
                          settle), but do not trust volume-derived features.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    ohlc = ["open", "high", "low", "close"]

    is_weekend = df["date"].dt.dayofweek >= 5
    is_nonfinite = df[ohlc].isna().any(axis=1)
    is_bad_bounds = (
        (df["high"] < df["low"])
        | (df[ohlc] <= 0).any(axis=1)
    ).fillna(False)
    is_zero_vol_wd = (~is_weekend) & df["volume"].fillna(0).eq(0)

    frames = []
    for mask, flag in [
        (is_weekend, "weekend_row"),
        (is_nonfinite, "nonfinite_ohlc"),
        (is_bad_bounds, "bad_ohlc_bounds"),
        (is_zero_vol_wd, "zero_volume_weekday"),
    ]:
        if mask.any():
            frames.append(df.loc[mask].assign(flag=flag))
    if not frames:
        return pd.DataFrame(columns=[*df.columns, "flag"])
    return pd.concat(frames, ignore_index=True).sort_values(["date", "instrument", "flag"])


# --------------------------------------------------------------------------- #
# SECTION 5. Cleaning -- drop only calendar-impossible rows                   #
# --------------------------------------------------------------------------- #
def clean_long(df: pd.DataFrame, drop_zero_volume: bool = False) -> pd.DataFrame:
    """Return a cleaned long frame.

    By default this drops ONLY rows that cannot be legitimate observations:
      * weekend rows (Sat/Sun are never sessions on any venue here),
      * rows with non-finite or non-positive OHLC, or high < low.

    Zero-volume *weekday* rows are intentionally **kept**: in this dataset all
    765 of them carry a valid intraday OHLC (a real settle), the volume field
    was simply not recorded. Dropping them would delete real prices and
    fabricate within-series gaps -- the opposite of what we want. Set
    ``drop_zero_volume=True`` only if you have verified those rows are
    interpolated artifacts for your subset.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    ohlc = ["open", "high", "low", "close"]

    is_weekend = df["date"].dt.dayofweek >= 5
    is_nonfinite = df[ohlc].isna().any(axis=1)
    is_bad_bounds = ((df["high"] < df["low"]) | (df[ohlc] <= 0).any(axis=1)).fillna(False)
    bad = is_weekend | is_nonfinite | is_bad_bounds
    if drop_zero_volume:
        bad = bad | ((~is_weekend) & df["volume"].fillna(0).eq(0))

    return (
        df.loc[~bad]
        .drop_duplicates(["instrument", "date"])
        .sort_values(["instrument", "date"])
        .reset_index(drop=True)
    )


def load_clean_ohlcv(
    path: str | Path | None = None, drop_zero_volume: bool = False
) -> pd.DataFrame:
    """Convenience: :func:`load_raw_ohlcv` followed by :func:`clean_long`."""
    return clean_long(load_raw_ohlcv(path), drop_zero_volume=drop_zero_volume)


# --------------------------------------------------------------------------- #
# SECTION 6. Per-instrument missing-date diagnostic                           #
# --------------------------------------------------------------------------- #
def instrument_inceptions(df: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """First observed date per instrument (its active-span start)."""
    return df.groupby("instrument")["date"].min().to_dict()


def diagnose_instrument(df_inst: pd.DataFrame, cal_info: CalendarInfo) -> pd.DataFrame:
    """Every expected session inside an instrument's active span with no row.

    Full holidays are already excluded from ``cal_info.sessions``, so a missing
    session is classified as either ``early_close`` (a known abbreviated session
    the vendor dropped -- benign) or ``unexplained`` (a genuine gap to escalate).
    """
    inception = df_inst["date"].min()
    last = df_inst["date"].max()
    expected = cal_info.sessions[
        (cal_info.sessions >= inception) & (cal_info.sessions <= last)
    ]
    present = pd.DatetimeIndex(df_inst["date"].unique())
    missing = expected.difference(present)

    rows = []
    for d in missing:
        early = cal_info.early_closes.get(d)
        rows.append(
            {
                "date": d,
                "classification": "early_close" if early else "unexplained",
                "label": early or "",
            }
        )
    return pd.DataFrame(rows, columns=["date", "classification", "label"])


# --------------------------------------------------------------------------- #
# SECTION 7. Cross-sectional presence matrix + scope classification           #
# --------------------------------------------------------------------------- #
def build_presence_matrix(
    df: pd.DataFrame,
    instruments: list[str] | None = None,
    date_start: pd.Timestamp | None = None,
    date_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """date x instrument boolean matrix over all weekdays (True = row present)."""
    instruments = instruments or INSTRUMENTS
    date_start = date_start or df["date"].min()
    date_end = date_end or df["date"].max()
    all_weekdays = pd.bdate_range(date_start, date_end)
    pivot = (
        df.assign(present=True)
        .pivot_table(
            index="date", columns="instrument", values="present",
            aggfunc="any", fill_value=False,
        )
        .reindex(all_weekdays, fill_value=False)
    )
    for inst in instruments:
        if inst not in pivot.columns:
            pivot[inst] = False
    return pivot[instruments]


def _scope_from_missing_set(missing: set[str], n_active: int) -> tuple[str, str]:
    """Map a set of missing instruments to (scope, venue_group)."""
    if not missing:
        return "none", "NONE"
    if len(missing) == n_active:
        return "global", "ALL"
    if missing <= set(EUREX_INSTRUMENTS):
        return "exchange_specific", "EUREX"
    if missing <= set(CME_EQ_INSTRUMENTS):
        return "exchange_specific", "CME_EQ"
    if missing <= set(US_INSTRUMENTS):  # US venues out, Eurex trades
        return "exchange_specific", "US_FUTURES"
    return "mixed", "MIXED"


def classify_missing_dates(
    presence: pd.DataFrame,
    inceptions: dict[str, pd.Timestamp],
    cals: dict[str, CalendarInfo],
) -> pd.DataFrame:
    """For every date with >=1 *active* instrument missing, return its category
    (``full_holiday`` / ``early_close`` / ``glitch``), cross-sectional scope
    (``global`` / ``exchange_specific`` / ``mixed``), and the affected members.
    """
    out = []
    for d, row in presence.iterrows():
        active = [k for k, inc in inceptions.items() if inc <= d]
        if not active:
            continue
        active_row = row[active]
        missing = set(active_row.index[~active_row.values])
        if not missing:
            continue
        scope, venue_group = _scope_from_missing_set(missing, len(active))

        # Prefer a full-holiday label; else an early-close label; else glitch.
        label, category = "", "glitch"
        for inst in sorted(missing):
            cal = cals[INSTRUMENT_MAP[inst]["calendar"]]
            if d in cal.holidays:
                label, category = cal.holidays[d], "full_holiday"
                break
            if d in cal.early_closes and category != "early_close":
                label, category = cal.early_closes[d], "early_close"

        out.append(
            {
                "date": d,
                "category": category,
                "scope": scope,
                "venue_group": venue_group,
                "label": label,
                "n_missing": len(missing),
                "n_active": len(active),
                "affected_instruments": ",".join(sorted(missing)),
            }
        )
    cols = [
        "date", "category", "scope", "venue_group", "label",
        "n_missing", "n_active", "affected_instruments",
    ]
    return pd.DataFrame(out, columns=cols).sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# SECTION 8. Deliverable metadata + full diagnostic driver                    #
# --------------------------------------------------------------------------- #
def holiday_venue_scope(d: pd.Timestamp, cals: dict[str, CalendarInfo]) -> tuple[str, str]:
    """Intrinsic scope of a holiday, independent of which instruments were yet
    listed: which venue calendars observe it. Returns (venue_scope, venues_closed).

    This is the principled answer to 'global vs exchange-specific': a date is
    ``global`` only if NYMEX/COMEX, CME-equity AND Eurex all close. (Contrast
    the empirical ``scope`` column, which is relative to the *active* universe
    and so labels pre-1998 US holidays 'global' merely because FESX did not yet
    exist.)
    """
    closed = []
    if d in cals["CMES"].holidays:
        closed.append("NYMEX/COMEX")
    if d in cals["CME_Equity"].holidays:
        closed.append("CME_EQ")
    if d in cals["XEUR"].holidays:
        closed.append("EUREX")
    cset = set(closed)
    if cset == {"NYMEX/COMEX", "CME_EQ", "EUREX"}:
        scope = "global"
    elif cset == {"NYMEX/COMEX", "CME_EQ"}:
        scope = "us_futures"
    elif cset == {"CME_EQ"}:
        scope = "cme_equity"
    elif cset == {"EUREX"}:
        scope = "eurex"
    elif cset == {"NYMEX/COMEX"}:
        scope = "nymex_comex"
    else:
        scope = "other"
    return scope, "+".join(closed)


def build_missing_holiday_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """The headline deliverable: one row per *full-holiday* date that halted
    trading, with global-vs-exchange-specific scope and the affected members.

    Columns
    -------
    venue_scope     : intrinsic scope (global / us_futures / eurex / cme_equity)
                      from which venue calendars observe the holiday.
    venues_closed   : the venue groups that close, e.g. ``NYMEX/COMEX+CME_EQ``.
    observed_scope  : empirical scope relative to the *active* universe on that
                      date (global / exchange_specific / mixed).
    affected_instruments : the universe members that actually had no row.

    Early-close half-days and unexplained glitches are excluded here; they are
    returned by :func:`build_other_missing_metadata`.
    """
    df = clean_long(df)
    cals = build_all_calendars(df["date"].min(), df["date"].max())
    inceptions = instrument_inceptions(df)
    presence = build_presence_matrix(df)
    classified = classify_missing_dates(presence, inceptions, cals)

    meta = classified[classified["category"] == "full_holiday"].copy()
    adhoc_dates = {pd.Timestamp(d) for d, _l, _s in US_FUTURES_ADHOC_CLOSURES}
    meta["holiday_type"] = np.where(meta["date"].isin(adhoc_dates), "adhoc", "recurring")
    meta["year"] = meta["date"].dt.year
    scopes = meta["date"].apply(lambda d: holiday_venue_scope(d, cals))
    meta["venue_scope"] = [s for s, _v in scopes]
    meta["venues_closed"] = [v for _s, v in scopes]
    meta = meta.rename(columns={"label": "holiday_name", "scope": "observed_scope"})
    cols = [
        "date", "year", "holiday_name", "holiday_type", "venue_scope",
        "venues_closed", "observed_scope", "n_missing", "n_active",
        "affected_instruments",
    ]
    return meta[cols].reset_index(drop=True)


def build_other_missing_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """The 'other non-trivial missing data' deliverable: every missing-date row
    that is NOT a full holiday -- i.e. dropped early-close half-days and the
    residual unexplained single-instrument glitches -- with scope and members.
    """
    df = clean_long(df)
    cals = build_all_calendars(df["date"].min(), df["date"].max())
    inceptions = instrument_inceptions(df)
    presence = build_presence_matrix(df)
    classified = classify_missing_dates(presence, inceptions, cals)

    other = classified[classified["category"] != "full_holiday"].copy()
    other["year"] = other["date"].dt.year
    cols = [
        "date", "year", "category", "label", "scope", "venue_group",
        "n_missing", "n_active", "affected_instruments",
    ]
    return other[cols].reset_index(drop=True)


def run_diagnostics(
    df: pd.DataFrame | None = None, out_dir: str | Path = "diagnostics"
) -> dict[str, pd.DataFrame]:
    """Run the full diagnostic suite and write CSVs. Returns the frames too."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw_ohlcv() if df is None else df.copy()

    anomalies = detect_anomalous_rows(raw)
    clean = clean_long(raw)
    cals = build_all_calendars(clean["date"].min(), clean["date"].max())
    inceptions = instrument_inceptions(clean)

    # Per-instrument missing dates.
    per_inst, summary = [], []
    for inst, meta in INSTRUMENT_MAP.items():
        di = clean[clean["instrument"] == inst]
        if di.empty:
            continue
        miss = diagnose_instrument(di, cals[meta["calendar"]])
        miss["instrument"] = inst
        miss["venue"] = meta["venue"]
        per_inst.append(miss)
        summary.append(
            {
                "instrument": inst,
                "venue": meta["venue"],
                "calendar": meta["calendar"],
                "inception": inceptions[inst].date(),
                "n_present": di["date"].nunique(),
                "n_missing_total": len(miss),
                "n_missing_early_close": int((miss["classification"] == "early_close").sum()),
                "n_missing_unexplained": int((miss["classification"] == "unexplained").sum()),
            }
        )
    per_inst_df = pd.concat(per_inst, ignore_index=True) if per_inst else pd.DataFrame()
    summary_df = pd.DataFrame(summary)

    presence = build_presence_matrix(clean)
    classified = classify_missing_dates(presence, inceptions, cals)
    holiday_meta = build_missing_holiday_metadata(raw)
    other_meta = build_other_missing_metadata(raw)
    unexplained = per_inst_df[per_inst_df["classification"] == "unexplained"].copy()

    outputs = {
        "anomalous_rows": anomalies,
        "missing_dates_per_instrument": per_inst_df,
        "summary_per_instrument": summary_df,
        "missing_dates_classified": classified,
        "missing_holidays_metadata": holiday_meta,
        "other_missing_metadata": other_meta,
        "unexplained_missing": unexplained,
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    return outputs


# --------------------------------------------------------------------------- #
# SECTION 9. Panel statistics on the ragged calendar                          #
# --------------------------------------------------------------------------- #
def native_returns(
    df: pd.DataFrame, price_col: str = "close", kind: str = "log"
) -> pd.DataFrame:
    """Per-instrument returns on each instrument's OWN dense series, so a return
    spanning a holiday is the correct multi-day move -- never a fabricated zero.
    """
    df = df.sort_values(["instrument", "date"]).copy()
    g = df.groupby("instrument", group_keys=False)[price_col]
    if kind == "log":
        df["ret"] = g.transform(lambda s: np.log(s).diff())
    elif kind == "simple":
        df["ret"] = g.transform(lambda s: s.pct_change())
    else:
        raise ValueError("kind must be 'log' or 'simple'")
    return df.dropna(subset=["ret"])


def wide_returns(df_ret: pd.DataFrame) -> pd.DataFrame:
    """Pivot native returns to date x instrument. Remaining NaNs are STRUCTURAL
    (pre-inception or other-venue holiday) -- never ffill/fillna(0) them."""
    return df_ret.pivot(index="date", columns="instrument", values="ret").sort_index()


def rolling_vol(
    df_ret: pd.DataFrame, instrument: str, window: int = 20, ann: float = 252.0
) -> pd.Series:
    """Annualised rolling vol on the NATIVE series (window = trading days)."""
    s = (
        df_ret.loc[df_ret["instrument"] == instrument]
        .set_index("date")["ret"]
        .sort_index()
    )
    return s.rolling(window).std() * np.sqrt(ann)


def rolling_mean(df_ret: pd.DataFrame, instrument: str, window: int = 20) -> pd.Series:
    s = (
        df_ret.loc[df_ret["instrument"] == instrument]
        .set_index("date")["ret"]
        .sort_index()
    )
    return s.rolling(window).mean()


def rolling_vol_panel(
    df_ret: pd.DataFrame,
    window: int = 60,
    ann: float = 252.0,
    instruments: list[str] | None = None,
) -> pd.DataFrame:
    """Rolling vol for every instrument, each computed on its OWN series then
    aligned into one wide frame. This is the correct fix for the notebook's
    'gappy rolling vol' problem: rolling on a unioned wide frame voids any
    window that contains another venue's holiday."""
    instruments = instruments or sorted(df_ret["instrument"].unique())
    cols = {inst: rolling_vol(df_ret, inst, window=window, ann=ann) for inst in instruments}
    return pd.DataFrame(cols).sort_index()


def corr_max_info(wide_ret: pd.DataFrame, min_periods: int = 252) -> pd.DataFrame:
    """Pairwise-complete correlation (max data per pair), repaired to PSD.
    The trap is calling .dropna() first (listwise) -- that truncates every pair
    to the shortest-history instrument."""
    C = wide_ret.corr(min_periods=min_periods)
    return nearest_psd_corr(C)


def nearest_psd_corr(C: pd.DataFrame) -> pd.DataFrame:
    """Clip negative eigenvalues to a small floor, renormalise to unit diagonal.
    Pairwise-estimated matrices are often slightly non-PSD; allocators need PSD."""
    cols = C.columns
    A = C.to_numpy(dtype=float)
    A = np.where(np.isnan(A), 0.0, A)
    np.fill_diagonal(A, 1.0)
    A = (A + A.T) / 2.0
    w, V = np.linalg.eigh(A)
    A_psd = V @ np.diag(np.clip(w, 1e-8, None)) @ V.T
    d = np.sqrt(np.diag(A_psd))
    A_psd = A_psd / np.outer(d, d)
    return pd.DataFrame(A_psd, index=cols, columns=cols)


def cov_ledoit_wolf(wide_ret: pd.DataFrame, min_obs: int = 252) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage covariance (PSD by construction). Requires a
    complete block, so it uses the largest common window (rows where ALL
    columns are present)."""
    from sklearn.covariance import LedoitWolf

    block = wide_ret.dropna()
    if len(block) < min_obs:
        raise ValueError(f"Common window only {len(block)} rows < {min_obs}.")
    lw = LedoitWolf().fit(block.to_numpy())
    return pd.DataFrame(lw.covariance_, index=block.columns, columns=block.columns)


def rolling_pair_corr(
    wide_ret: pd.DataFrame, a: str, b: str, window: int = 120
) -> pd.Series:
    """Rolling correlation of two instruments on the INTERSECTION of their
    trading days. Aligning first means a single other-venue holiday does not
    void the whole window (the bug behind the empty rolling-corr plot)."""
    pair = wide_ret[[a, b]].dropna()
    return pair[a].rolling(window).corr(pair[b])


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main() -> None:
    raw = load_raw_ohlcv()
    print(f"Loaded {len(raw):,} rows; instruments={sorted(raw['instrument'].unique())}")
    print(f"Range: {raw['date'].min().date()} -> {raw['date'].max().date()}")
    out = run_diagnostics(raw)
    print("\n[anomalies]")
    print(out["anomalous_rows"]["flag"].value_counts().to_string())
    print("\n[per-instrument summary]")
    print(out["summary_per_instrument"].to_string(index=False))
    print("\n[cross-sectional category x scope]")
    cls = out["missing_dates_classified"]
    print(cls.groupby(["category", "scope"]).size().to_string())
    print(f"\n[unexplained glitches] total = {len(out['unexplained_missing'])}")
    print(
        out["unexplained_missing"][["date", "instrument", "label"]].to_string(index=False)
        if len(out["unexplained_missing"]) <= 30
        else out["unexplained_missing"]["instrument"].value_counts().to_string()
    )
    print(f"\n[missing-holiday metadata] {len(out['missing_holidays_metadata'])} rows")
    print(f"[other-missing metadata]   {len(out['other_missing_metadata'])} rows")
    print("Wrote CSVs to ./diagnostics/")


if __name__ == "__main__":
    main()
