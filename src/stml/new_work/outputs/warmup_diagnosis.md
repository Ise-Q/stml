# Warmup Diagnosis — Feature Importance Pipeline

**Date:** 2026-06-01  
**Scope:** Read-only investigation. No code was modified.

---

## 1. Executive Summary

- **Windowing bug verdict — HYPOTHESIS PARTIALLY WRONG.** The original hypothesis was that OHLCV-based rolling features are computed on a frame starting at the metamodel window (2020-01-03), causing warmup to eat real labelled events. This is **false for OHLCV-derived features**: `_ohlcv_df()` and `build_feature_matrix()` use the full raw OHLCV series (back to the 1990s), so `f2_vol_pctile_20` (252-day), `f13_expected_hit_time` (100-day), and all other price-based features are fully absorbed by pre-window history. Zero NaN at event dates for those features.

- **Actual root cause identified.** The binding constraint is `f5_participation_60` — a 60-day rolling mean of `|signal|` — where the **signal series itself only begins at 2020-01-03** (PRESAMPLE_CUTOFF). No pre-window signal history exists, so the 60-day warmup is structurally irreducible. The 60th signal observation falls on **2020-03-26**, making every event before that date NaN for this feature.

- **Events recoverable.** Excluding `f5_participation_60` alone recovers 38 additional events per equity instrument. Excluding all three slow F5 features (`participation_60`, `participation_20`, `long_bias_20`) recovers all 56 dropped events for equities (e.g. es1s: 509 → 565, a 10% gain). `ng1s` loses 0 events because its first label (2020-05-05) is already past the warmup horizon.

- **Leakage verdict — CLEAN.** `f13_expected_hit_time` and `f13_prob_timeout` use only past prices and past volatility. They do not consume triple-barrier labels, realized outcomes, or future price bars. The Monte Carlo simulation draws from the trailing window of returns and projects forward synthetically. No leakage.

- **No other bugs found** with material impact. One cosmetic inconsistency: `f2_vol_dispersion` calls `rolling(L).std()` without `min_periods=L` for the 10/20/60-day vol computation, but pandas `.std()` requires ≥2 observations anyway, so the effective warmup is identical.

---

## 2. Timeline Table

| Data Source | Start Date | End Date | Notes |
|---|---|---|---|
| OHLCV — es1s | 1997-09-09 | 2022-06-30 | 6,296 rows |
| OHLCV — nq1s | 1999-06-21 | 2022-06-30 | 5,845 rows |
| OHLCV — fesx1s | 1998-06-30 | 2022-06-30 | 6,108 rows |
| OHLCV — gc1s | 1990-01-02 | 2022-06-30 | 8,171 rows |
| OHLCV — si1s | 1990-01-02 | 2022-06-30 | 8,172 rows |
| OHLCV — hg1s | 1990-01-02 | 2022-06-30 | 8,172 rows |
| OHLCV — pl1s | 1990-01-02 | 2022-06-30 | 8,169 rows |
| OHLCV — cl1s | 1990-01-02 | 2022-06-30 | 8,171 rows |
| OHLCV — ho1s | 1990-01-02 | 2022-06-30 | 8,169 rows |
| OHLCV — rb1s | 1990-01-02 | 2022-06-30 | 8,170 rows |
| OHLCV — ng1s | 1990-04-04 | 2022-06-30 | 8,104 rows |
| macro_features.csv (f11) | 1990-01-02 | 2022-06-30 | 8,478 rows; some columns start later (see §7) |
| alternate_data_cleaned.csv | 1990-01-02 | 2022-06-30 | 8,478 rows |
| primary_signals.csv | **2020-01-03** | 2022-06-30 | 645 rows — metamodel window only |
| triple_barrier_labels — es1s | 2020-01-03 | 2022-06-15 | 565 events |
| triple_barrier_labels — nq1s | 2020-01-03 | 2022-06-15 | 594 events |
| triple_barrier_labels — fesx1s | 2020-01-03 | 2022-06-16 | 627 events |
| triple_barrier_labels — gc1s | **2020-03-16** | 2022-06-15 | 162 events |
| triple_barrier_labels — si1s | 2020-01-03 | 2022-06-15 | 568 events |
| triple_barrier_labels — hg1s | 2020-01-03 | 2022-06-15 | 618 events |
| triple_barrier_labels — pl1s | 2020-01-03 | 2022-06-15 | 548 events |
| triple_barrier_labels — cl1s | 2020-01-07 | 2022-06-15 | 412 events |
| triple_barrier_labels — ho1s | 2020-01-21 | 2022-06-02 | 63 events |
| triple_barrier_labels — rb1s | 2020-01-03 | 2022-06-15 | 618 events |
| triple_barrier_labels — ng1s | **2020-05-05** | 2022-06-08 | 120 events |
| HMM vol (features_hmm_vol.csv) | 2020-01-03 | 2022-06-30 | metamodel window only |
| HMM macro (features_hmm_macro.csv) | 2020-01-03 | 2022-06-30 | metamodel window only |

---

## 3. Gap Analysis Table

Gap = (first label date) − (OHLCV start date). This is the pre-window price history available to absorb rolling feature warmup.

| Instrument | OHLCV Start | First Label | Gap (cal days) | Gap (trading rows) |
|---|---|---|---|---|
| es1s | 1997-09-09 | 2020-01-03 | 8,151 | 5,667 |
| nq1s | 1999-06-21 | 2020-01-03 | 7,501 | 5,216 |
| fesx1s | 1998-06-30 | 2020-01-03 | 7,857 | 5,471 |
| gc1s | 1990-01-02 | 2020-03-16 | 11,031 | 7,591 |
| si1s | 1990-01-02 | 2020-01-03 | 10,958 | 7,543 |
| hg1s | 1990-01-02 | 2020-01-03 | 10,958 | 7,543 |
| pl1s | 1990-01-02 | 2020-01-03 | 10,958 | 7,541 |
| cl1s | 1990-01-02 | 2020-01-07 | 10,962 | 7,545 |
| ho1s | 1990-01-02 | 2020-01-21 | 10,976 | 7,552 |
| rb1s | 1990-01-02 | 2020-01-03 | 10,958 | 7,542 |
| ng1s | 1990-04-04 | 2020-05-05 | 10,989 | 7,560 |

All instruments have 5,200–7,600 trading rows of pre-window price history. Any rolling feature with warmup ≤ 252 days is fully absorbed for all instruments. The 252-day warmup for `f2_vol_pctile_20` and the 100-day warmup for `f13` features are **not binding** at event dates.

---

## 4. Warmup-by-Feature Ranking Table

Top 15 features by leading-NaN count in `daily_df` for **es1s** (daily_df spans 1997-09-09 to 2022-06-30, 6,296 rows):

| Rank | Feature | Leading NaN (daily_df) | Source | Binding at Event Dates? |
|---|---|---|---|---|
| 1 | f5_participation_60 | 5,724 | primary_signals.csv (2020+) | **YES — 56 NaN at events** |
| 2 | f5_participation_20 | 5,685 | primary_signals.csv (2020+) | YES — 18 NaN (subsumed by rank 1) |
| 3 | f5_long_bias_20 | 5,685 | primary_signals.csv (2020+) | YES — 18 NaN (subsumed by rank 1) |
| 4 | f5_signal | 5,667 | primary_signals.csv (2020+) | No — 0 NaN at events |
| 5 | f5_abs_signal | 5,667 | primary_signals.csv (2020+) | No — 0 NaN at events |
| 6 | f5_trailing_run_length | 5,667 | primary_signals.csv (2020+) | No — 0 NaN at events |
| 7 | f5_days_since_flip | 5,667 | primary_signals.csv (2020+) | No — 0 NaN at events |
| 8 | f5_days_since_nonzero | 5,667 | primary_signals.csv (2020+) | No — 0 NaN at events |
| 9 | f5_sign_agree_mr | 5,667 | primary_signals.csv (2020+) | No — 0 NaN at events |
| 10 | hmm_vol_p0_calm | 5,667 | HMM CSVs (2020+) | No — 0 NaN at events |
| 11 | hmm_vol_p2_turbulent | 5,667 | HMM CSVs (2020+) | No — 0 NaN at events |
| 12 | hmm_vol_next_turbulent | 5,667 | HMM CSVs (2020+) | No — 0 NaN at events |
| 13 | hmm_vol_entropy | 5,667 | HMM CSVs (2020+) | No — 0 NaN at events |
| 14 | hmm_macro_p0 | 5,667 | HMM CSVs (2020+) | No — 0 NaN at events |
| 15 | hmm_macro_next_riskoff | 5,667 | HMM CSVs (2020+) | No — 0 NaN at events |

**Why ranks 4–15 are not binding despite huge leading-NaN counts:** These features have NaN in `daily_df` only for pre-2020 rows (no signal/HMM data before 2020). The reindex to event dates (all post-2020) finds valid values for all events. The critical difference for rank 1–3 is the additional rolling warmup *on top of* the signal inception date.

Notable features investigated but **not binding**:
- `f2_vol_pctile_20`: 271 leading NaN (252-day warmup on OHLCV), but 0 NaN at event dates — fully absorbed by 5,667 pre-window rows.
- `f13_expected_hit_time`: 101 leading NaN (100-day window), 0 NaN at event dates — absorbed by pre-window OHLCV history.
- `f11_china_pmi_level`: 1,862 leading NaN, but 0 NaN at event dates (PMI coverage starts 2005-01-31, well before 2020).
- `f15_dist_lead_lag`: 96.5% NaN at event dates — but correctly removed by Step 0 (sparse column drop), not Step 1.

---

## 5. Listwise Deletion Analysis

### es1s (565 raw events)

**After Step 0** (drop `f15_dist_lead_lag`, 96.5% NaN): 565 events remain, 3 columns still have NaN.

| Feature | NaN at events | Events uniquely forced by this feature | If excluded: survivors |
|---|---|---|---|
| f5_participation_60 | 56 (9.9%) | 38 | 547 |
| f5_participation_20 | 18 (3.2%) | 0 (subsumed) | 509 |
| f5_long_bias_20 | 18 (3.2%) | 0 (subsumed) | 509 |

**Cumulative exclusion:**
| Features excluded | Survivors |
|---|---|
| None (current behaviour) | 509 |
| f5_participation_60 | 547 (+38) |
| f5_participation_60 + f5_participation_20 | 547 |
| All three slow F5 features | **565** (all events recovered) |

### ho1s (63 raw events)

**After Step 0** (drop `f15_dist_lead_lag` + `f15_asset_class_dispersion_z`): 63 events remain.

| Feature | NaN at events | If excluded: survivors |
|---|---|---|
| f5_participation_60 | 6 (9.5%) | 61 |
| f5_participation_20 | 2 (3.2%) | 57 |
| f5_long_bias_20 | 2 (3.2%) | 57 |

Current behaviour: 57 survivors. Excluding all three: 63 survivors (+6, +9.5%).

**Key finding:** For ho1s, the deficit is primarily the small total label count (63), not warmup. Warmup accounts for only 6 of the 63 events (~9.5%) and those 6 are from the first few weeks of the window. The prompt's hypothesis of ~6 warmup losses for ho1s is confirmed exactly.

---

## 6. Code Verdict on Feature Windowing

**Verdict: Path (a) — features ARE computed on the full raw OHLCV series, then aligned to event dates.**

### Key code lines from `build_feature_matrix()` (feature_importance.py):

```python
# Line 417-421
ohlcv_inst = ohlcv[ohlcv["instrument"] == inst].copy()  # full history
df = _ohlcv_df(ohlcv_inst)                               # no date filter
close = df["close"]
ret_series = np.log(close).diff()                         # computed on full OHLCV
vol20 = ret_series.rolling(20).std()                      # computed on full OHLCV
```

```python
# _ohlcv_df (lines 164-175) — no date filter applied:
def _ohlcv_df(ohlcv_inst: pd.DataFrame) -> pd.DataFrame:
    df = (ohlcv_inst[cols]
          .dropna(subset=["close"])          # remove invalid prices only
          .drop_duplicates("date")
          .sort_values("date")
          .set_index("date")
          .astype(float))
    return df
```

```python
# Line 530: events aligned via reindex, not slice
X_at_events = daily.reindex(ev["date"].values).reset_index(drop=True)
```

**OHLCV-based features are computed on the full series.** The warmup hypothesis (that a date-slice causes the problem) is incorrect for f2, f13, f6, f7, f10, f1, and f4. These features start computing from the OHLCV inception date (1990s or 1997–1999), and by 2020 they have thousands of rows of history.

**The actual bug is in the signal-derived F5 group:**

```python
# Lines 424-435
sig_wide = signals.set_index("date")
signal_inst = sig_wide[inst].sort_index()   # starts 2020-01-03 — no pre-window data
mr = f1["f1_mr_score_20"].reindex(signal_inst.index)
f5 = f5_signal_derived(signal_inst, mr_score=mr).reindex(df.index)
```

```python
# In f5_signal_derived (lines 299-300):
out["f5_participation_20"] = ab.rolling(20, min_periods=20).mean()  # 20-day warmup on signal
out["f5_participation_60"] = ab.rolling(60, min_periods=60).mean()  # 60-day warmup on signal
out["f5_long_bias_20"]     = s.rolling(20, min_periods=20).mean()   # 20-day warmup on signal
```

`primary_signals.csv` starts at 2020-01-03. The 60-day rolling requires 60 signal observations. The 60th observation falls on 2020-03-26. Every labelled event between 2020-01-03 and 2020-03-25 therefore has `f5_participation_60 = NaN`, and the listwise deletion in Step 1 drops the entire row.

---

## 7. Source Coverage Table for Slow Features

| Feature | Source Start | Leading NaN (daily_df) | Warmup Root Cause | Recoverable? |
|---|---|---|---|---|
| f5_participation_60 | 2020-01-03 (signal inception) | 5,724 (es1s) | 60-day rolling on signal that starts at window start | **No — irreducible** |
| f5_participation_20 | 2020-01-03 | 5,685 | 20-day rolling on signal | **No — irreducible** |
| f5_long_bias_20 | 2020-01-03 | 5,685 | 20-day rolling on signal | **No — irreducible** |
| f11_china_pmi_level | 2005-01-31 | 1,862 (macro_feats) | PMI series only starts 2005 | No (but 0 NaN at event dates — not binding) |
| f11_vix_term_slope | 2002-01-02 | 1,087 | VIX term structure data from 2002 | No (0 NaN at events — not binding) |
| f2_vol_pctile_20 | 1997-09-09 + 272 rows | 271 | 252-day rank on 20-day vol | Yes — fully recovered (0 NaN at events) |
| f13_expected_hit_time | 1997-09-09 + 100 rows | 101 | 100-day bootstrap window | Yes — fully recovered (0 NaN at events) |
| f13_prob_timeout | 1997-09-09 + 100 rows | 101 | 100-day bootstrap window | Yes — fully recovered (0 NaN at events) |

**Conclusion:** The only irreducible warmup is from the three F5 rolling features, because the signal itself has no history before 2020. All OHLCV-based features (f2, f13, f4, etc.) are fully recovered by the deep pre-window price history.

---

## 8. Event Deficit Attribution Table

| Instrument | Raw Labels | Sparse Col Drops (Step 0) | Warmup Drop (Step 1) | Surviving | Primary Cause of Warmup Loss |
|---|---|---|---|---|---|
| es1s | 565 | 1 col (f15_dist_lead_lag) | **56** | 509 | f5_participation_60 (60-day warmup on signal) |
| nq1s | 594 | 1 col | **56** | 538 | f5_participation_60 |
| fesx1s | 627 | 1 col | **59** | 568 | f5_participation_60 (Eurex has more active days → more early labels) |
| gc1s | 162 | 2 cols | **5** | 157 | f5_participation_60 (gc1s first label 2020-03-16, near the 60-day threshold) |
| si1s | 568 | 2 cols | **53** | 515 | f5_participation_60 |
| hg1s | 618 | 2 cols | **57** | 561 | f5_participation_60 |
| pl1s | 548 | 2 cols | **52** | 496 | f5_participation_60 |
| cl1s | 412 | 2 cols | **30** | 382 | f5_participation_60 (cl1s first label 2020-01-07, some early labels miss window) |
| ho1s | 63 | 2 cols | **6** | 57 | f5_participation_60; deficit mostly from few labels, not warmup |
| rb1s | 618 | 2 cols | **57** | 561 | f5_participation_60 |
| ng1s | 120 | 2 cols | **0** | 120 | None — first label 2020-05-05 is past the 60-day warmup horizon |

**Cause classification:**
- Equities (es1s, nq1s, fesx1s, si1s, hg1s, pl1s, rb1s): primarily warmup-trim loss (~53–59 events, ~9–10% of labels).
- Energies (cl1s): warmup-trim loss (30 events, ~7%), reduced because label inception is slightly later.
- ho1s: **both** — only 63 raw labels and 6 of those lost to warmup. The deficit is primarily label scarcity from sparse signal activity.
- ng1s: **no warmup loss** — all 120 events survive.
- gc1s: minimal warmup loss (5 events) because its first label (2020-03-16) is near the 60-day cutoff.

**Prompt hypothesis check:** "equities lose ~56 to warmup" — confirmed (56 for es1s, nq1s; 59 for fesx1s). "ho1s loses ~6 to warmup but has only 63 labels in total" — confirmed exactly (6 warmup drops, 57 survivors).

---

## 9. Bug Findings

### Minor: `rolling(L)` without `min_periods=L` in `f2_vol_dispersion`

In `f2_vol_dispersion` (lines 227–232), the vol computation for L in (10, 20, 60) uses:

```python
v = (rets_long.loc[rets_long["instrument"] == inst]
     .set_index("date")["ret"]
     .sort_index()
     .rolling(L)      # <-- no min_periods=L
     .std()
     * np.sqrt(252))
```

`rolling(L)` defaults to `min_periods=1`. However, pandas `.std()` with a single observation returns NaN by convention (ddof=1 needs ≥2 points). In practice the first row of `v` is NaN (1 obs), the second is NaN (ddof=1 with 2 obs: technically defined but only 1 diff), and behaviour matches `min_periods=L` closely. **No material correctness impact** — verified that this produces the same NaN count as explicit `min_periods`. No look-ahead is introduced. Documented as a style inconsistency.

### f2_vol_dispersion instrument filter — redundant but correct

In `f2_vol_dispersion`, `native_returns(ohlcv_inst, ...)` is called with the already-filtered `ohlcv_inst`, and then filtered again with `.loc[rets_long["instrument"] == inst]`. The double filter is redundant but harmless — it returns the correct single-instrument series either way.

### f15_dist_lead_lag: structural sparsity from calendar mismatch (not a bug, but notable)

`distance_to_lead_lag_centroid` computes a 63-day rolling RMS on a `diff_sq` series derived from `wide_rets`. The wide returns panel has ~2.3% NaN in 2020 (US holidays where es1s/nq1s etc. are absent). With `rolling(63, min_periods=63)`, any window spanning even one holiday (expected roughly every 43 trading days on average) produces NaN. This results in ~95–97% NaN at event dates for all instruments, correctly handled by Step 0 (sparse column drop). Not a bug, but reducing `min_periods` or switching to `min_periods=int(0.8*window)` would recover this feature.

### No other bugs found

- No accidental global `dropna` before events are assembled.
- No features recomputed per CPCV fold — all features are computed once in `build_feature_matrix` and the resulting `events_df` is passed to the importance loop.
- No look-ahead/non-causal windowing found (all rolling windows look backward; `centroid.shift(lag)` in `distance_to_lead_lag_centroid` correctly uses the lagged peer mean).
- No mid-series structural gaps in OHLCV close prices for any instrument.

---

## 10. Leakage Verdict on f13 Features

**VERDICT: CLEAN — no leakage.**

### f13_expected_hit_time and f13_prob_timeout

**File:** `src/stml/harry/features/conditional_risk.py`

**(a) Inputs consumed:**

```python
def expected_hit_time(
    returns: pd.Series,    # log returns of this instrument (from OHLCV)
    vol: pd.Series,        # rolling 20-day vol (from OHLCV)
    *,
    pt_mult: float = 1.0,
    sl_mult: float = 1.0,
    h: int = 10,           # forward horizon in days (Monte Carlo projection length)
    window: int = 252,     # trailing return history for bootstrap pool
    n_sims: int = 200,
    seed: int = 42,
) -> pd.Series:
```

Only `returns` and `vol` are consumed. Both are derived exclusively from OHLCV (past prices only).

**(b) Label/future-bar dependency:**

None. The function signature accepts no labels, no `bin` column, no `t1` column, and no `events_df`. It is called in `build_feature_matrix` as:

```python
hit = expected_hit_time(
    ret_series, vol20, pt_mult=1.0, sl_mult=1.0, h=10,
    window=100, n_sims=30, seed=RANDOM_SEED,
)
```

where `ret_series = np.log(close).diff()` and `vol20 = ret_series.rolling(20).std()` — both from the full OHLCV series.

**(c) Forward estimate vs backward statistic:**

Forward estimate. The core loop:

```python
for t in range(window, n):
    past = r_arr[t - window : t]       # trailing 'window' past returns
    v_t = v_arr[t]                     # current (past) vol
    pt = pt_mult * v_t * sqrt_h        # barrier level (based on past vol)
    rng = np.random.default_rng(seed * _PER_ROW_PRIME + t)
    samples = rng.choice(past, size=(n_sims, h), replace=True)  # bootstrap from past
    cum = samples.cumsum(axis=1)       # synthetic forward paths
    touch_mask = (cum >= pt) | (cum <= -sl)
    median_hit[t] = ...                # median first passage across synthetic paths
```

The `h`-bar forward paths are **synthetic** (bootstrapped from past returns), not realized future prices. No future bars are read.

**(d) Signal-gated vs every-bar:**

Computed on every bar of the OHLCV series (the `for t in range(window, n)` loop iterates over all rows). Results are then aligned to event dates via `reindex(ev["date"].values)`.

**(e) Seed design preserves causality:**

Each row `t` uses `seed * 1_000_003 + t` as its RNG seed. This means the output at row `t` is identical whether computed on `data[:t+1]` or `data[:T]` for any `T ≥ t+1` — no path-dependent leakage from later rows.

**f13_path_tortuosity_20d and f13_realized_semi_vol_ratio:** Simpler 20-day trailing statistics on returns only. Both use `rolling(window, min_periods=window).sum()` / `.mean()` patterns. Fully causal.

---

## 11. Proposed Fix

**Problem:** `f5_participation_60` applies a 60-day rolling window to a signal series that starts at 2020-01-03, causing the first 60 signal observations (≈ 56 event dates for equities) to be NaN. The listwise deletion in Step 1 of `apply_hygiene` then drops these entire rows.

**Options (in order of recommendation):**

**Option A — Use `min_periods=1` for the rolling participation features (lowest-cost fix):**

In `f5_signal_derived` (feature_importance.py, lines 299–300), change:
```python
out["f5_participation_20"] = ab.rolling(20, min_periods=20).mean()
out["f5_participation_60"] = ab.rolling(60, min_periods=60).mean()
out["f5_long_bias_20"]     = s.rolling(20, min_periods=20).mean()
```
to:
```python
out["f5_participation_20"] = ab.rolling(20, min_periods=1).mean()
out["f5_participation_60"] = ab.rolling(60, min_periods=1).mean()
out["f5_long_bias_20"]     = s.rolling(20, min_periods=1).mean()
```

Effect: recovers all 56 dropped events per equity instrument. The participation estimate for early events will be computed from fewer observations (e.g. from 1 observation on the first day), so early values are noisier. This is a semantic change: `f5_participation_60` becomes "average signal activity over available history up to 60 days" rather than "strict 60-day average". Acceptable in practice since these are feature inputs, not targets.

**Option B — Drop `f5_participation_60` from the feature set:**

Remove `f5_participation_60` from `f5_signal_derived`. This recovers 38 of the 56 dropped events (the 38 events where `f5_participation_60` is the sole cause). The remaining 18 events (dropped by `f5_participation_20` and `f5_long_bias_20`) require also removing those features or applying Option A to them.

**Option C — Accept the warmup (do nothing):**

Current behaviour drops 52–59 events per equity instrument (~9–10%). These are the earliest events in the metamodel window (Jan–Mar 2020), before the signal had 60 days of history. There may be an argument that these early-period events are lower quality (signal just launched, initial distribution unknown). The remaining 509–568 events per instrument are sufficient for CPCV.

**Awaiting approval before any implementation.**

---

*Report generated by read-only diagnostic investigation. Source: `src/stml/new_work/feature_importance.py`, `src/stml/harry/features/conditional_risk.py`, `src/stml/harry/features/cross_asset.py`, `src/stml/na_checks.py`.*
