# Bloomberg Pull List — Sreeram_experimental S2

> Single source of truth for what to extract from the Bloomberg terminal so the
> S2 feature stack can land. Companion to ``plan.md §5``.
>
> Date: 2026-06-02
> Author: Sreeram

## What is already in the repo (DO NOT re-pull)

Harry's `data/alternate_data_cleaned.csv` (8478 rows, **1990-01-02 → 2022-06-30**)
already contains 21 daily macro series — the M2/M3/M4/M5/M6 layers of the
plan §3.3 feature stack:

| Group | Already-pulled series |
|---|---|
| Rates / curve | `10Y_BUND`, `10Y_UST`, `2Y_UST`, `TIPS10Y`, `BE10Y` |
| Credit | `HY_OAS`, `IG_OAS` |
| Vol indices | `VIX`, `VIX3M`, `MOVE`, `CBOE_SKEW` |
| FX | `DXY`, `EURUSD` |
| Commodity fundamentals | `BAL_DRY_INDEX`, `LME_COPPER_STOCK`, `EIA_CRUDE_STOCK`, `EIA_DIST_STOCK`, `EIA_GASOLINE_STOCK`, `EIA_NG_STOCK` |
| Macro growth | `CHINA_PMI_MFG`, `US_ISM_MFG_PMI` |

These are read directly by the bloomberg loader on S2 start — **you do not need
to pull them again**.

## What is NEW (please pull)

Four blocks. Save each to a CSV (or parquet) under
`data/bloomberg/raw/`. Filename suggestion alongside each block. Date format
`YYYY-MM-DD`. NaN cells encoded as empty or `NaN`.

---

### Block A — Futures term structure (F18 family)

**Output file:** `data/bloomberg/raw/futures_term_structure.csv`
**Format:** one row per business day, columns = `date` + each ticker's `PX_LAST`.
**Date range:** full available history → 2022-06-30 (the longer the better;
VIX futures `VX1/VX2` only exist from ~2004).

| Ticker | Source name | Used for |
|---|---|---|
| `CL2 Comdty` | WTI 2nd-month continuous | `cl1s` term spread (front = `cl1s` from OHLCV) |
| `HO2 Comdty` | Heating Oil 2nd month | `ho1s` term spread |
| `XB2 Comdty` *(or `RB2`)* | RBOB Gasoline 2nd month | `rb1s` term spread |
| `NG2 Comdty` | Natural Gas 2nd month | `ng1s` term spread |
| `GC2 Comdty` | Gold 2nd month | `gc1s` term spread |
| `SI2 Comdty` | Silver 2nd month | `si1s` term spread |
| `HG2 Comdty` | Copper 2nd month | `hg1s` term spread |
| `PL2 Comdty` | Platinum 2nd month | `pl1s` term spread |
| `VX1 Index` | VIX front-month futures | Equity vol term structure |
| `VX2 Index` | VIX 2nd-month futures | Equity vol term structure |

**BBG command pattern (BDH formula):**
```
=BDH("CL2 Comdty", "PX_LAST", "1/2/1990", "6/30/2022")
```

10 series in total. ~7000 daily rows each.

---

### Block B — Options-implied volatility (F19 family)

**Output file:** `data/bloomberg/raw/options_iv.csv`
**Format:** one row per business day, columns = `date` + one column per
`<TICKER>_<FIELD>`. So `SPX_1M_IMPVOL_100MNY_DF`, `SPX_3M_IMPVOL_100MNY_DF`, etc.
**Date range:** full available history → 2022-06-30 (IV surface fields begin
in the late 1990s for major indices; commodity options IV typically from 2005+).

**Fields (4 per underlying):**

| Field | Meaning |
|---|---|
| `1M_IMPVOL_100MNY_DF` | 1-month at-the-money implied volatility |
| `3M_IMPVOL_100MNY_DF` | 3-month at-the-money implied volatility |
| `1M_IMPVOL_75DELTA_DF` | 1-month 25-delta PUT implied volatility |
| `1M_IMPVOL_125DELTA_DF` | 1-month 25-delta CALL implied volatility |

**Underlyings (11):**

| Asset class | Underlying ticker | For instrument |
|---|---|---|
| Equity | `SPX Index` | `es1s` |
| Equity | `NDX Index` | `nq1s` |
| Equity | `SX5E Index` | `fesx1s` |
| Energy | `CL1 Comdty` | `cl1s` |
| Energy | `HO1 Comdty` | `ho1s` (may be thin — try; if mostly NaN, drop the column and we substitute `CL1` IV in the loader) |
| Energy | `XB1 Comdty` | `rb1s` (same caveat — substitute `CL1` if thin) |
| Energy | `NG1 Comdty` | `ng1s` |
| Metals | `GC1 Comdty` | `gc1s` |
| Metals | `SI1 Comdty` | `si1s` |
| Metals | `HG1 Comdty` | `hg1s` |
| Metals | `PL1 Comdty` | `pl1s` (may be thin — substitute `GC1` if so) |

**BBG command pattern:**
```
=BDH("SPX Index", "1M_IMPVOL_100MNY_DF", "1/2/1995", "6/30/2022")
```

44 series total (4 × 11). If a particular underlying / field combination
returns `#N/A`, leave the column empty and we'll handle the fallback at the
loader.

---

### Block C — CFTC Commitment of Traders positioning (F20 family)

**Output file:** `data/bloomberg/raw/cftc_cot.csv`
**Format:** one row per **weekly** observation (Tuesday data, Friday published).
Columns = `date` + one column per `<TICKER>_<FIELD>`.
**Date range:** full available history → 2022-06-30 (CFTC weekly going back
to 1992 for many commodities).

**Fields (5 per underlying):**

| Field | Meaning |
|---|---|
| `CFTC_NONCOMM_LONG_FUT_ONLY` | Non-commercial (spec) longs |
| `CFTC_NONCOMM_SHORT_FUT_ONLY` | Non-commercial shorts |
| `CFTC_COMM_LONG_FUT_ONLY` | Commercial longs |
| `CFTC_COMM_SHORT_FUT_ONLY` | Commercial shorts |
| `CFTC_OPEN_INT` | Total open interest |

**Underlyings (10 — skip `fesx1s`, Eurex not CFTC; we'll use ICE COT or skip the
feature for fesx):**

| BBG ticker | For instrument |
|---|---|
| `CL1 Comdty` | `cl1s` |
| `HO1 Comdty` | `ho1s` |
| `XB1 Comdty` | `rb1s` |
| `NG1 Comdty` | `ng1s` |
| `GC1 Comdty` | `gc1s` |
| `SI1 Comdty` | `si1s` |
| `HG1 Comdty` | `hg1s` |
| `PL1 Comdty` | `pl1s` |
| `ES1 Index` | `es1s` |
| `NQ1 Index` | `nq1s` |

50 series total (5 × 10). Output is weekly; the loader forward-fills onto
the trade calendar between releases.

---

### Block E — Event flags (F22 family)

**Output file:** `data/bloomberg/raw/event_dates.csv`
**Format:** two columns — `date`, `event_type`. One row per event.
**Date range:** 2005-01-01 → 2022-06-30 (older history not needed; the rolling
features warm up in ~2015).

**Events to pull (use BBG's `ECRL <GO>` Economic Release Calendar to filter):**

| Event type | Frequency | Notes |
|---|---|---|
| `fomc` | ~8/year | FOMC meeting dates (the close of each two-day meeting) |
| `eia_petroleum` | weekly Wed | EIA Petroleum Status Report release |
| `eia_natgas` | weekly Thu | EIA Natural Gas Storage Report release |
| `opec_jmmc` | varies | OPEC Joint Ministerial Monitoring Committee meetings |
| `opec_summit` | ~2/year | Full OPEC summit meetings |
| `usda_wasde` | monthly | World Agricultural Supply and Demand Estimates |
| `ecb` | ~8/year | ECB Governing Council decisions |
| `boe` | ~8/year | Bank of England Monetary Policy Committee |
| `boj` | ~8/year | Bank of Japan Monetary Policy Meeting |

If pulling individual event types is tedious, even just the FOMC + EIA Petroleum
+ OPEC sets is enough — the others are nice-to-have for the F22 family but the
former three drive the headline gates.

---

## Cleaning / publication-lag policy

Once you've pulled the four files (Block A / B / C / E) and dropped them in
`data/bloomberg/raw/`, the loader applies the **publication lags** documented
in plan §5.1 + alken's `macro.py`:

| Series type | Lag |
|---|---|
| Daily market series (Block A, Block B) | **+1 day** (close known EOD, traded next day) |
| Weekly CFTC (Block C) | **+3 calendar days** (Tuesday data, Friday published) |
| Weekly EIA / monthly events (Block E) | event-date observable on day-of |
| Macro releases via Harry's CSV | already applied in his CSV (he used +1d daily, +5d EIA, +30d PMI) |

So the loader shifts each series forward by its lag, then forward-fills onto
the daily trade calendar each instrument uses. The result is point-in-time
correct: trade date `t` sees only what was released by `t`.

---

## What happens after the pull

Once the four CSVs are in `data/bloomberg/raw/`, I run S2:

1. `bloomberg_ingest.py` — applies publication lags, forward-fills to trade
   calendar, writes per-series parquets to `data/bloomberg/cleaned/`.
2. Build F18 / F19 / F20 / F21 (from existing OHLCV) / F22 feature columns.
3. KS drift filter across train (≤ 2021-10-06) vs the embargoed test slice.
4. Macro reformulation: rolling 63-day rank for LEVELS (per plan §3.3 — fixes
   the catastrophic train/test drift in macro levels).
5. Persist `data/sreeram_experimental_features.parquet`.
6. Emit `results/sreeram_experimental/feature_drift_audit.csv` so we know which
   features survived the filter and why.

Estimated S2 wall-clock once data arrives: 1-2 hours.

## Fallback if Bloomberg is unavailable

The plan §5.5 fallback: build with F1-F17 only. Pipeline still runs, deliverable
still produced, but we lose the ~0.05-0.10 AUC headroom that the new families
provide. Trigger by passing `--no-bloomberg` to the eventual S2 CLI.
