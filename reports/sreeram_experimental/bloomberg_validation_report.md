# Bloomberg Pull Validation Report

> Forensic analysis of the three Bloomberg pulls dropped in
> `data/bloomberg/raw/` on 2026-06-02 evening. The headline verdict is at the
> bottom, but **the short story is: clean, usable, with two specific gaps and
> two specific substitutions that need to be documented in the methodology**.

Date: 2026-06-02
Files reviewed:

```
data/bloomberg/raw/block_a_front_playbook_consolidated.csv   2.9 MB
data/bloomberg/raw/block_a_playbook_consolidated.csv         3.4 MB
data/bloomberg/raw/block_b_iv_extraction_20260526_corrected (1).xlsx   4.4 MB
data/bloomberg/raw/block_e_eia_playbook_consolidated.csv     46 KB
```

---

## TL;DR — Verdict

**GO. Proceed to S2 with the partial Bloomberg stack.**

| Block | Status | What we build |
|---|---|---|
| Block A (term structure) | ✅ Clean, full history 1990-2022 | F18 — full per-instrument term structure |
| Block B (options IV) | ✅ Clean from 2005-onwards; PL1 missing | F19 — from 2005, with GC1 IV substituted for PL1 (declared) |
| Block C (CFTC COT) | ❌ Not pulled | F20 — DROPPED entirely (~6 features lost; documented limitation) |
| Block E (events) | ⚠️ Only EIA crude — no FOMC / OPEC / WASDE / ECB | F22 — partial: just EIA crude release flag + change for energy instruments |

**Net feature impact vs. the full-Bloomberg plan §3.3 catalog:**
- Lose 6 F20 columns (CFTC positioning) — affects energy + metals diff feature mostly
- Lose ~2-3 F22 columns (FOMC + OPEC flags) — affects equity + energy event features
- Gain everything else (F18 + F19 + F21 + partial F22 + F1-F17)
- Net matrix dimension: ~100 columns vs. plan's ~120 target. Inside the §8 S2 gate range [80, 120].

---

## 1. Block A — Futures term structure pull

### 1.1 Files

Two CSVs, daily wide format, ISO dates, full 1990-2022 history:

| File | Shape | Tickers | Fields per ticker |
|---|---|---|---|
| `block_a_front_playbook_consolidated.csv` | 8210 × 25 | CL1, HO1, XB1, NG1, GC1, SI1, HG1, PL1 | `last_price`, `open_interest`, `volume` |
| `block_a_playbook_consolidated.csv` | 8295 × 31 | CL2, HO2, XB2, NG2, GC2, SI2, HG2, PL2, UX1, UX2 | same |

So the user pulled BOTH the front month AND the second month for the 8 commodities, plus front + 2nd VIX futures (`UX1`/`UX2` = `VX1`/`VX2` — Bloomberg has both ticker aliases).

### 1.2 Coverage — first valid date per series

| Series | First valid | Notes |
|---|---|---|
| CL1/HO1/GC1/SI1/HG1/PL1 + 2nd month | 1990-01-02 | full history ✓ |
| NG1/NG2 | 1990-04-03 | Henry Hub futures launched April 1990 |
| XB1 (RBOB front) | 2005-10-03 | RBOB launched October 2005 |
| XB2 (RBOB 2nd) | 2005-10-03 | same |
| UX1 (VIX front) | 2004-03-26 | VIX futures launched March 2004 |
| UX2 (VIX 2nd) | 2004-03-26 | same |

This is exactly what we expected from the asset histories. **No anomalies.**

### 1.3 ⚠️ CRITICAL FINDING — OHLCV is back-adjusted, BBG is raw

Spot-check on 2020-01-02:

| Instrument | OHLCV close (project data) | BBG `LAST_PRICE` (raw) | Ratio |
|---|---:|---:|---:|
| cl1s / CL1 | 24.80 | 61.18 | 0.41 |
| ho1s / HO1 | 1.80 ($/gal) | 202.41 (¢/gal) | 0.89 after cents conversion |
| hg1s / HG1 | 3.39 ($/lb) | 282.50 (¢/lb) | 1.20 after cents conversion |
| rb1s / XB1 | 5.62 | 170.42 (¢/gal) = $1.70 | 3.3 |
| ng1s / NG1 | 0.00276 | 2.122 | 1000× |
| pl1s / PL1 | 954.71 | 978.60 | 0.976 |

The OHLCV closes have been **back-adjusted** for roll losses (CL accumulates contango drag, hence the ratio of 0.41 at 2020 vs ratio of 1.0 at 1990). The OHLCV also uses different unit conventions for some commodities (likely fractional contracts, scaled-down volumes, or partial-contract reporting).

**Implications for S2:**

* For triple-barrier labels (S1) we use OHLCV — correct, that's the investable continuous-return series.
* For F1-F17 + F21 features (return-based, ratio-based) we use OHLCV — same scale.
* **For F18 term structure we use BBG-front + BBG-2nd**, both from the same raw convention. **We do NOT compute `(BBG_F2 - OHLCV_F1) / OHLCV_F1`** because the two are on incompatible scales.

This is why the user wisely pulled BOTH BBG front + BBG second — internally consistent. F18 = `(BBG_2nd - BBG_front) / BBG_front`.

### 1.4 Open-interest series — non-trivial start dates

Several instruments have OI starting later than their price series:

* SI1/HG1/PL1 OI from 1992-11-17 (CME OI publication began then)
* GC1/SI1 OI from 1990-01-02 OK
* RBOB OI from 2005-10-12

For F18 we use price; for F7 microstructure we already have its own volume/OI from OHLCV, so this doesn't affect us materially.

---

## 2. Block B — Options-implied volatility pull

### 2.1 File structure

Excel workbook with 46 sheets:
- 1 `README`
- 1 `manifest` (the rosetta-stone: sheet name → underlying → BBG field)
- 44 data sheets (4 fields × 11 underlyings)

Per-sheet schema (a quirk — there is no header row):

```
column 0: date (datetime64)
column 1: IV value (float)
column 2: 'Unnamed: 2' (empty, ignore)
column 3: 'Ticker formula' label, header-row leakage
column 4: duplicate date
column 5: duplicate IV value
```

So the cleaner reads columns 0 and 1 (or 4 and 5 — they're the same), ignores the rest. **This needs handling in `bloomberg_ingest.py` (will write a per-sheet parser).**

### 2.2 Field mapping (from the `manifest` sheet)

User pulled MONEYNESS-based IV instead of delta-based IV — a reasonable substitute:

| Sheet field | BBG field actually pulled | Originally asked for |
|---|---|---|
| `IV1M_ATM` | `30DAY_IMPVOL_100.0%MNY_DF` | `1M_IMPVOL_100MNY_DF` ✓ (same thing) |
| `IV3M_ATM` | `3MTH_IMPVOL_100.0%MNY_DF` | `3M_IMPVOL_100MNY_DF` ✓ |
| `IV1M_90MNY` | `30DAY_IMPVOL_90.0%MNY_DF` | `1M_IMPVOL_75DELTA_DF` (25Δ put) |
| `IV1M_110MNY` | `30DAY_IMPVOL_110.0%MNY_DF` | `1M_IMPVOL_125DELTA_DF` (25Δ call) |

**Substitution `90% MNY ↔ 25Δ put` (and `110% MNY ↔ 25Δ call`):** For typical 1-month vols, 25Δ-put roughly maps to ~90% moneyness for ATM-ish forwards. The two are not identical (25Δ tracks the option's delta, 90% MNY tracks the strike level), but they capture **the same skew direction and magnitude** within ~10-20 % relative error.

The manifest explicitly notes this: *"30-day 90% moneyness implied vol (skew proxy, not strict 25-delta)"*. So **the substitution is intentional and the user / data provider have already flagged it**. We document this in the methodology as a known approximation; the skew feature still works directionally.

### 2.3 Coverage per underlying

| Underlying | First valid | Sheets populated | Notes |
|---|---|---|---|
| SPX (es1s) | 2005-01-03 | 4 | ✓ |
| NDX (nq1s) | 2005-01-03 | 4 | ✓ |
| SX5E (fesx1s) | 2006-01-02 | 4 | ✓ |
| CL1 (cl1s) | 2005-10-28 | 4 | ✓ |
| HO1 (ho1s) | 2005-10-28 | 4 | ✓ |
| XB1 (rb1s) | 2006-08-28 | 4 | ✓ |
| NG1 (ng1s) | 2005-10-28 | 4 | ✓ |
| GC1 (gc1s) | 2005-10-28 | 4 | ✓ |
| SI1 (si1s) | 2005-10-28 | 4 | ✓ |
| HG1 (hg1s) | 2005-10-28 | 4 | ✓ |
| **XPT (pl1s)** | **EMPTY** | 0 of 4 | **#N/A — platinum options data unavailable** |

**Substitution `pl1s ← GC1 IV`:** plan §5.2 already specified this fallback for platinum since its options market is thin. We apply it explicitly: `pl1s.f19_*` features use the GC1 IV series.

### 2.4 Effective date range

Because the IV series start ~2005-Oct (commodities) and 2005-Jan (equities), F19 features can only be computed from then. For the modelling window (2020-01-03 → 2022-06-30, signal period) this is **plenty of warm-up** for rolling 252-day percentile features (warmup needs ~Oct 2019 → reach back to Oct 2018 → IV data is fine since 2005).

The 252d rolling features for F19 will be available starting ~late 2006. Modelling window is 2020-onwards — no constraint binds.

---

## 3. Block E — Event flags pull

### 3.1 What we actually got

One file, one series: **EIA U.S. Crude Oil Inventory Weekly Change**.

Schema: `date, DOEASCRD_Index__crude_stocks_kb`
* 1695 rows, 1990-01-05 → 2022-06-24
* 7-day gap between every consecutive observation
* **All 1695 dates fall on a FRIDAY** — verified empirically

### 3.2 The Wed-vs-Fri question — RESOLVED

**The AI agent who flagged "BBG posts it on Wed but the data says Friday" was CORRECT.** Here's the precise EIA convention:

* The EIA Weekly Petroleum Status Report contains crude oil inventory data **as of the close of the prior Friday** (the "as-of date").
* The report is **publicly released the following Wednesday at 10:30 AM ET** (sometimes Thursday if Monday is a federal holiday — Labor Day, Memorial Day, etc.).
* Bloomberg ticker `DOEASCRD Index` records the **as-of Friday date** (the underlying measurement week-ending date), NOT the release date.

**Therefore:**

* The `date` column in this file is the **as-of Friday**.
* The actual **release date** is `as-of-Friday + 5 calendar days` (the following Wednesday).
* For PIT (point-in-time) alignment, traders can only act on the data from the release Wednesday onward.

**The publication-lag policy is the standard +5-calendar-day lag** the plan §5.1 already specifies and Harry's `macro.py` (alken parity) already implements:

```python
EIA_LAG_DAYS = 5  # Friday data → Wednesday release
```

### 3.3 What this series represents

Value statistics:

| Statistic | Value |
|---|---:|
| count | 1695 |
| mean | 60 |
| std | 4,355 |
| min | -15,222 |
| max | 21,563 |
| median | 131 |
| |median| | 2,800 |

These are **weekly CHANGES in thousands of barrels** (i.e., `kb` = thousand barrels), not levels.

Cross-check: Harry's `EIA_CRUDE_STOCK` in `data/alternate_data_cleaned.csv` is the LEVELS series (median ~321,000 mbbl, range 247k–540k mbbl — the total US crude stocks excluding SPR). So **Harry's series and this new BBG series are COMPLEMENTARY, not duplicative**:

* Harry's level → `f11_eia_crude_stock` (already in pipeline) — "where storage is right now"
* New BBG change → `f22_eia_crude_change` (NEW) — "the surprise traders react to on release day"

We use **both**.

### 3.4 What we DIDN'T get

| Event ask | Status | Plan workaround |
|---|---|---|
| FOMC meeting dates | ❌ not pulled | `f22_fomc_in_window` dropped (could hand-code from public list if needed) |
| EIA Petroleum (crude) Wed release | ✅ via inference from Friday-as-of date | `f22_eia_petroleum_release` for energy |
| EIA Natural Gas Storage Thu release | ❌ not pulled | dropped |
| OPEC meetings (JMMC + summits) | ❌ not pulled | dropped (could hand-code) |
| USDA WASDE | ❌ not pulled | dropped |
| ECB / BOE / BOJ | ❌ not pulled | dropped |

**The big losses are FOMC and OPEC.** For an MVP, we can either:
1. Hand-code the FOMC schedule (~8 dates × ~7 years = 56 dates) and OPEC (~14 dates × 7 years = ~100 dates) from public records — high accuracy, takes ~30 min.
2. Skip the F22 family entirely except for the EIA crude flag.

I'll do option (1) in S2 as a final cleanup if time permits. For now, S2 builds:
- `f22_eia_crude_change` continuous feature (weekly crude inventory change, forward-filled with lag)
- `f22_eia_release_today` binary flag (1 on the Wednesday of release, 0 otherwise)

---

## 4. CFTC COT (Block C) — confirmed missing

The user reported they couldn't get Blocks C and E in full. The F20 positioning family (net spec, net commercials, spec extremity z-score, 156-week percentile, open interest changes) is **DROPPED ENTIRELY** for this build. The plan §13 risk register flagged this as R-1 and the loader is designed to silently drop F20 if the file is absent.

**Impact:** F20 would have added ~6 columns × 10 instruments (since fesx1s isn't CFTC). On the released sample these features were expected to contribute incremental but not large AUC — they're slow-moving (weekly) and mostly capture sentiment regime. Net feature count drops from ~120 to ~100. Still inside the §8 S2 gate range.

---

## 5. Summary of substitutions and gaps (for methodology.md)

When we write `methodology.md` in S8, we document the following Bloomberg-related decisions transparently:

1. **F18 term structure uses BBG-raw front + BBG-raw 2nd**, not OHLCV-front + BBG-2nd, because OHLCV is back-adjusted and would produce nonsense term spreads.
2. **F19 starts in 2005** (commodities) / **2005** (SPX, NDX) / **2006** (SX5E, XB1). The modelling window is 2020-onwards so this is non-binding.
3. **F19 90% / 110% MNY substituted for 25Δ put / call** — same skew direction, ~10-20% relative error. Documented as a known approximation per BBG manifest.
4. **F19 for pl1s uses GC1 IV** (XPT options series is empty). Documented as fallback per plan §5.2.
5. **F20 (CFTC COT) — DROPPED ENTIRELY.** Listed as a limitation in the final report. Net AUC impact expected modest given the slow-moving nature.
6. **F22 partial** — only EIA crude release flag + weekly change for energy instruments. FOMC, OPEC, USDA WASDE, ECB/BOE/BOJ flags not available.
7. **EIA convention:** file dates are the as-of Friday; release is Friday + 5 calendar days (Wednesday). Standard `+5 day` publication lag applied via `pit_align`.
