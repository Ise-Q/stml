# Missing-Data Findings — `ohlcv_data.csv` (1990-01-02 → 2022-06-30)

> Companion to [`refs/missing-holidays.md`](missing-holidays.md) (the research note).
> All numbers below are produced by [`src/stml/na_checks.py`](../src/stml/na_checks.py)
> and persisted as CSVs under [`data/meta/`](../data/meta). Regenerate with
> `uv run python -m stml.na_checks` (writes to `data/meta/` via `run_diagnostics`).

## TL;DR

- **Every weekday gap in this dataset is explained.** Of all missing
  instrument-days inside each instrument's active span: **0 are full holidays**
  (correctly absent), **the vendor-dropped early-close half-days account for the
  bulk**, and exactly **7 single-instrument residual glitches** remain — each
  with a plausible micro-cause. No multi-instrument vendor outage exists.
- **Holidays that halted trading: 339 dated closures.** Intrinsic venue scope:
  **86 global** (all venues), **197 US-futures-only** (FESX trades through),
  **54 Eurex-only** (US trades through), **2 CME-equity-only**. See
  [`data/meta/missing_holidays_metadata.csv`](../data/meta/missing_holidays_metadata.csv).
- **Two corrections to the research note's a-priori predictions, found in the data:**
  1. **Early-close days are dropped, not kept.** The note assumed Christmas Eve,
     the day after Thanksgiving, etc. keep a (low-volume) row. This vendor
     *often drops the row entirely* on those abbreviated sessions. They are
     benign, not glitches, and are classified as `early_close`.
  2. **Hurricane Sandy (2012-10-29/30) and the Bush funeral (2018-12-05) did
     NOT cause missing rows here** — all 11 instruments have rows. The note
     predicted ES/NQ would be missing. 9/11 *itself* (2001-09-11) likewise has
     rows for all 11 (the market opened, then halted).
- **Do not drop zero-volume rows.** All **765** zero-volume weekday rows carry a
  valid intraday OHLC (a real settle); volume was simply not recorded. Dropping
  them — as a naive cleaner would — deletes real prices and fabricates
  within-series gaps. **Only the 3 Sunday `2005-05-08` rows are dropped.**
- **For rolling stats / correlation on a ragged panel:** compute returns on each
  instrument's *own* dense series, keep structural NaNs after pivoting (never
  ffill/zero-fill them), use pairwise-complete correlation repaired to PSD, and
  align rolling pairwise correlation on the *intersection* of the two calendars.

---

## 1. Method

Two complementary diagnostics, both in `stml.na_checks`:

1. **Calendar-based, per instrument.** A self-contained, hand-coded holiday
   calendar for NYMEX/COMEX + CME-equity (`build_us_futures_holidays`) and Eurex
   (`build_eurex_holidays`), plus the ad-hoc closures, covering **1990-2022**.
   Expected sessions = business days − full holidays. A missing expected session
   is either an `early_close` (a known abbreviated session this vendor drops) or
   `unexplained`. The hand-coded tables are **authoritative** because the
   `exchange_calendars` / `pandas_market_calendars` libraries are unreliable
   before ~2000 and most instruments here begin in 1990. Those libraries, if
   installed (`uv sync --group calendars`), are used only as a non-fatal
   cross-check.

2. **Cross-sectional (the note's presence-matrix algorithm).** For each date,
   look at which *active* instruments have no row. If all active are missing →
   `global`; if the missing set is a subset of one venue group → `exchange_specific`;
   otherwise `mixed`. This is the basis of `missing_dates_classified.csv`.

Instrument → calendar/venue mapping (`INSTRUMENT_MAP`):

| Venue group | Calendar | Instruments |
|---|---|---|
| NYMEX (energy) | `CMES` | cl1s, ho1s, rb1s, ng1s |
| COMEX (metals) | `CMES` | gc1s, si1s, hg1s, pl1s |
| CME equity | `CME_Equity` | es1s, nq1s |
| Eurex | `XEUR` | fesx1s |

## 2. Inventory & per-instrument missing counts

From [`data/meta/summary_per_instrument.csv`](../data/meta/summary_per_instrument.csv):

| Instrument | Venue | Inception | Rows present | Missing (early-close) | Missing (unexplained) |
|---|---|---|---:|---:|---:|
| cl1s | NYMEX | 1990-01-02 | 8171 | 25 | 0 |
| ho1s | NYMEX | 1990-01-02 | 8169 | 25 | **1** |
| rb1s | NYMEX | 1990-01-02 | 8170 | 25 | 0 |
| ng1s | NYMEX | 1990-04-04 | 8104 | 25 | **1** |
| gc1s | COMEX | 1990-01-02 | 8170 | 24 | **1** |
| si1s | COMEX | 1990-01-02 | 8171 | 24 | 0 |
| hg1s | COMEX | 1990-01-02 | 8171 | 24 | **1** |
| pl1s | COMEX | 1990-01-02 | 8169 | 25 | **1** |
| es1s | CME equity | 1997-09-09 | 6296 | 0 | 0 |
| nq1s | CME equity | 1999-06-21 | 5845 | 0 | 0 |
| fesx1s | Eurex | 1998-06-30 | 6108 | 36 | **2** |

ES/NQ have **zero** missing expected sessions — their CME-equity calendar is a
clean fit. Inception dates match the project brief (ES 1997, FESX 1998, NQ 1999).

## 3. Missing holidays (the 339 trading halts)

Headline metadata: [`data/meta/missing_holidays_metadata.csv`](../data/meta/missing_holidays_metadata.csv).
Each row carries both an **intrinsic** `venue_scope` (which venue calendars
observe the holiday, independent of inception) and an **empirical**
`observed_scope` (relative to the active universe that date), plus the exact
`affected_instruments`.

### 3.1 Global holidays — all venues close (86 dates)
Closed on NYMEX/COMEX **and** CME-equity **and** Eurex: **New Year's Day, Good
Friday, Christmas Day**, and the years where **US Memorial Day coincides with
Eurex Whit Monday** (e.g. 2015-05-25 — flagged by the research note). On these
dates all 11 active instruments are absent (8 pre-FESX-era).

> Note on Good Friday: it is a *global* full closure in most years, but in ~5
> years (1999, 2010, 2012, 2015, 2021) **ES/NQ ran an abbreviated session and
> DO have rows** while everything else is closed — these surface as
> `observed_scope = mixed`. This is a real, datable exception to "no trading on
> Good Friday."

### 3.2 US-futures-only holidays — FESX trades through (197 dates)
NYMEX/COMEX + CME-equity close, Eurex open: **MLK Day** (from 1998), **Presidents'
Day, Memorial Day, Independence Day, Labor Day, Thanksgiving**, and **Juneteenth**
(first observed 2022 — `2022-06-20` is a full US closure here; only FESX trades).
`affected_instruments` = the 10 US members (or 8 before ES/NQ inception).

### 3.3 Eurex-only holidays — US trades through (54 dates)
Only FESX is absent: **Easter Monday, Labour Day (May 1), Boxing Day (Dec 26),
Whit Monday** (observed ~1998-2007 and the 2015 one-off). This is the single
biggest source of "missing for one instrument, present for the rest" in the panel.

### 3.4 Ad-hoc / non-recurring closures
**Closures that DID drop rows** (in `missing_holidays_metadata.csv`, `holiday_type=adhoc`):

| Date(s) | Event | Scope in data | Affected |
|---|---|---|---|
| 1994-04-27 | Nixon state funeral | US (all active = 8 commodities) | all commodities |
| 2001-09-12, 09-13 | 9/11 aftermath | US-only (FESX present — Eurex open) | all US |
| 2001-09-14 | 9/11 — NYMEX/COMEX reopened via ACCESS | CME-equity-only | es1s, nq1s |
| 2004-06-11 | Reagan state funeral | US-only (FESX present) | all US |
| 2007-01-02 | Ford funeral — CME equity closed, NYMEX/COMEX electronic open | CME-equity-only | es1s, nq1s |

**Events the literature flags but which did NOT cause missing rows here**
(`ADHOC_EVENTS_NO_CLOSURE` in code — verified all 11 instruments present):

- **2001-09-11** — market opened then halted; pre-attack rows exist for all 11.
- **2012-10-29 / 10-30 (Hurricane Sandy)** — all 11 present (electronic/backfilled).
- **2018-12-05 (Bush funeral)** — all 11 present.

These are intentionally **not** added to any holiday set, so the session calendar
does not falsely exclude dates that carry real data.

## 4. Other non-trivial missing data (61 dates)

[`data/meta/other_missing_metadata.csv`](../data/meta/other_missing_metadata.csv).

### 4.1 Dropped early-close half-days (54 dates) — benign
The vendor *inconsistently* drops the row on abbreviated sessions. By label:
Christmas Eve (16), New Year's Eve (16), day after Thanksgiving (14), July-4
bridge days (6: July 3 / July 5), New-Year bridge days (2: Jan 2 / Jan 3).

A clear **regime shift**:
- **Pre-2003**: the entire commodity complex (8 instruments) is dropped on these
  half-days — a market-wide vendor choice.
- **2003 onward**: only **fesx1s** is dropped (29 of the 54 rows), on its Dec 24
  / Dec 31 early closes, while the US instruments retain their abbreviated-session
  rows.

These are **not** errors. Treat them as legitimate non-sessions: a return that
spans them is the correct two-day move.

### 4.2 Residual unexplained glitches (7 dates) — single-instrument
Each is one instrument missing a single weekday with no holiday/early-close
explanation. Candidate causes (for escalation/awareness, not auto-repair):

| Date | Instrument | Likely cause |
|---|---|---|
| 1990-09-04 (Tue) | gc1s | Day after Labor Day; isolated single-name gap |
| 1991-04-19 (Fri) | hg1s | Isolated single-name gap |
| 1992-09-08 (Tue) | ho1s | Day after Labor Day; isolated single-name gap |
| 1993-06-24 (Thu) | pl1s | Platinum was very thin in 1993 (vol ~0.5-2.8k); prior row had 0 vol/OI → likely a genuine no-trade |
| 1996-01-08 (Mon) | ng1s | Coincides with the Jan 1996 US-Northeast blizzard → plausible NYMEX energy disruption |
| 1999-05-13 (Thu) | fesx1s | Early FESX life (listed 1998-06); thin/early-history gap |
| 1999-06-03 (Thu) | fesx1s | Early FESX life; thin/early-history gap |

All 7 are isolated and single-instrument — exactly the residue the research note
predicts ("single digits per instrument"). None indicate a systemic vendor outage.

## 5. Anomalous rows that ARE present (768 flagged)

[`data/meta/anomalous_rows.csv`](../data/meta/anomalous_rows.csv).

- **`weekend_row` (3): DROP.** `2005-05-08` (a Sunday) for gc1s, hg1s, si1s.
  Calendar-impossible (COMEX never trades Sunday); the gc1s/si1s values look
  interpolated (zero volume). This is the only true spurious-presence artifact.
- **`zero_volume_weekday` (765): KEEP.** Every one has a valid intraday OHLC
  (high ≥ low > 0, real ranges — not flat carry-forwards). The volume field was
  simply not recorded (common for 1990s-2000s continuous contracts; concentrated
  in gc/hg). **Dropping these would delete 765 real prices and fabricate gaps.**
  Keep the price; just don't trust volume-derived features on these rows.
- **`nonfinite_ohlc` / `bad_ohlc_bounds`: 0.** No NaN prices, no high<low, no
  non-positive prices, no duplicate `(instrument, date)` rows.

## 6. How to handle the NAs

`clean_long(df)` (the default in `load_clean_ohlcv`) implements the policy:

1. **Drop** weekend rows and any non-finite / non-positive / high<low OHLC.
   (Here: just the 3 Sunday rows.)
2. **Keep** zero-volume weekday rows. (`drop_zero_volume=True` is available but
   should not be used on this dataset.)
3. **Do not** reindex to a calendar grid and forward-fill / zero-fill. Holidays,
   early closes and other-venue closures are *meaningful* non-sessions.

After cleaning, the only NaNs that can appear are **structural**, introduced when
you pivot to a shared date index:
- **pre-inception** (e.g. nq1s before 1999-06-21), and
- **other-venue holidays** (e.g. fesx1s on US Thanksgiving, cl1s on Easter Monday).

Structural NaNs are correct and must be preserved, never filled. The wide return
panel's NaN fractions reflect exactly this: ~2% for the 1990 commodities
(other-venue holidays) and 24-30% for es/nq/fesx (later inception).

## 7. Rolling statistics & pairwise correlation on full-length data

The notebook's original gappy charts came from rolling/correlating on a
**union-of-dates** wide pivot: a single other-venue holiday inserts a NaN that
voids the entire rolling window (or the whole `.dropna()` block). The fix, all in
`stml.na_checks`:

1. **Returns on the native series** — `native_returns(long)` computes each
   instrument's return on its *own* dense, sorted series via groupby. A return
   spanning a holiday is the correct multi-day move, never a fabricated zero.
2. **Single-name rolling stats on the native series** — `rolling_vol` /
   `rolling_mean` / `rolling_vol_panel`. The window counts **trading days**, so
   there is no NaN dilution. `rolling_vol_panel` runs per-instrument then aligns
   into one frame — the correct version of the notebook's "rolling vol" chart.
3. **Cross-sectional correlation = pairwise-complete, then PSD** —
   `corr_max_info(W, min_periods=252)` uses `DataFrame.corr` (pairwise-complete:
   each pair uses every jointly-observed day) and repairs the result to the
   nearest positive-semi-definite correlation matrix (`nearest_psd_corr`) so it
   is safe for allocation routines. **Never** call `.dropna()` (listwise) first —
   that truncates every pair to the shortest-history instrument (nq1s from 1999).
   For a stable covariance on the common window, `cov_ledoit_wolf` is provided.
4. **Rolling pairwise correlation on the intersection** — `rolling_pair_corr(W, a, b)`
   drops to the days **both** instruments traded *before* rolling, so one
   other-venue holiday no longer empties the window. (gc1s/si1s yields 8050
   values vs. the original empty plot.)

## 8. Generated artifacts

| File | Rows | Contents |
|---|---:|---|
| `data/meta/missing_holidays_metadata.csv` | 339 | **Headline:** each holiday halt — venue_scope, venues_closed, observed_scope, affected_instruments |
| `data/meta/other_missing_metadata.csv` | 61 | Early-close half-days + the 7 glitches |
| `data/meta/missing_dates_classified.csv` | 400 | Every missing-instrument date: category × scope |
| `data/meta/missing_dates_per_instrument.csv` | 240 | Per-instrument early_close / unexplained |
| `data/meta/summary_per_instrument.csv` | 11 | Inception, n_present, missing breakdown |
| `data/meta/anomalous_rows.csv` | 768 | Present-but-suspect rows (3 weekend + 765 zero-volume) |
| `data/meta/unexplained_missing.csv` | 7 | The residual glitches only |

## 9. Reproduction

```bash
uv pip install -e . --no-deps          # make `stml` importable (src layout)
uv run python -m stml.na_checks        # regenerates data/meta/ + prints summary
# optional external-calendar cross-check:
uv sync --group calendars
```

```python
from stml.io import load_clean_data, load_returns_panel
ohlcv_clean, signals = load_clean_data()      # NA-handled OHLCV
W = load_returns_panel()                       # date x instrument log returns (structural NaNs only)
```
