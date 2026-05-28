# 03-6 — Macro Feature Pack (Step 3.5b)

> Module: [`src/stml/harry/features/macro_features.py`](../../src/stml/harry/features/macro_features.py)  
> Tests: [`tests/harry/test_macro_features.py`](../../tests/harry/test_macro_features.py) — 49 unit tests  
> Causality harness: 6 registrations in `CAUSALITY_REGISTRATIONS`, picked up automatically by [`tests/harry/test_causality.py`](../../tests/harry/test_causality.py).  
> Data source: `data/alternate_data_cleaned.csv` (lag-safe macro panel, 1990–2022, 21 series).

---

## 1. Overview

Six groups of external macro features that condition the meta-model on the
macro/financial environment at decision time. All features are **causal**:
at row `t` only data with index ≤ `t` is used. Z-scores and rolling
statistics use trailing windows only. EIA inventory surprise windows count
**releases** (weekly cadence), not calendar days.

The pooled feature matrix exposes all macro features to all instruments;
`MACRO_INSTRUMENT_TARGETS` documents the primary targets per feature, but
importance sorting at Step 4 handles relevance.

---

## 2. Feature table — all 31 features

| # | Group | Feature | Source col(s) | Warmup | Primary targets |
|--:|---|---|---|--:|---|
| 1 | **M1** | `vix_level_z` | `VIX` | 252† | es1s, nq1s, fesx1s, gc1s, si1s |
| 2 | M1 | `vix_5d_change` | `VIX` | 5 | es1s, nq1s, fesx1s |
| 3 | M1 | `vix_term_slope` | `VIX3M − VIX` | 0 | es1s, nq1s, fesx1s, gc1s |
| 4 | M1 | `move_z` | `MOVE` | 252† | es1s, nq1s, fesx1s |
| 5 | M1 | `move_vix_ratio` | `MOVE / VIX` | 0 | es1s, nq1s, fesx1s |
| 6 | M1 | `skew_z` | `CBOE_SKEW` | 252† | es1s, nq1s, fesx1s |
| 7 | **M2** | `us_2s10s_slope` | `10Y_UST − 2Y_UST` | 0 | es1s, nq1s, fesx1s |
| 8 | M2 | `ust_10y_5d_change` | `10Y_UST` | 5 | es1s, nq1s, fesx1s |
| 9 | M2 | `bund_10y_5d_change` | `10Y_BUND` | 5 | fesx1s |
| 10 | M2 | `ust_bund_spread` | `10Y_UST − 10Y_BUND` | 0 | fesx1s |
| 11 | M2 | `real_yield_10y` | `TIPS10Y` | 252† | gc1s, si1s |
| 12 | M2 | `breakeven_10y` | `BE10Y` | 252† | gc1s, si1s |
| 13 | M2 | `be_5d_change` | `BE10Y` | 5 | gc1s, si1s |
| 14 | **M3** | `hy_oas_z` | `HY_OAS` | 252† | es1s, nq1s, fesx1s |
| 15 | M3 | `hy_oas_5d_change` | `HY_OAS` | 5 | es1s, nq1s, fesx1s |
| 16 | M3 | `ig_oas_z` | `IG_OAS` | 252† | es1s, nq1s, fesx1s |
| 17 | M3 | `hy_ig_ratio` | `HY_OAS / IG_OAS` | 0 | es1s, nq1s, fesx1s |
| 18 | **M4** | `dxy_z` | `DXY` | 252† | gc1s, si1s, hg1s, cl1s |
| 19 | M4 | `dxy_5d_change` | `DXY` | 5 | gc1s, si1s, hg1s |
| 20 | M4 | `eurusd_5d_change` | `EURUSD` | 5 | fesx1s, gc1s |
| 21 | **M5** | `crude_stock_surprise` | `EIA_CRUDE_STOCK` | ~30‡ | cl1s, ho1s, rb1s |
| 22 | M5 | `dist_stock_surprise` | `EIA_DIST_STOCK` | ~30‡ | ho1s |
| 23 | M5 | `gasoline_stock_surprise` | `EIA_GASOLINE_STOCK` | ~30‡ | rb1s |
| 24 | M5 | `ng_stock_surprise` | `EIA_NG_STOCK` | ~30‡ | ng1s |
| 25 | M5 | `copper_stock_z` | `LME_COPPER_STOCK` | 252† | hg1s |
| 26 | M5 | `baltic_dry_z` | `BAL_DRY_INDEX` | 252† | cl1s, hg1s |
| 27 | M5 | `baltic_5d_change` | `BAL_DRY_INDEX` | 5 | cl1s, hg1s |
| 28 | **M6** | `ism_pmi_level` | `US_ISM_MFG_PMI` | 0 | es1s, nq1s, fesx1s, hg1s, cl1s |
| 29 | M6 | `ism_pmi_3m_change` | `US_ISM_MFG_PMI` | 63 | es1s, nq1s, fesx1s |
| 30 | M6 | `china_pmi_level` | `CHINA_PMI_MFG` | 0 | hg1s, cl1s |
| 31 | M6 | `global_pmi_breadth` | `US_ISM + CHINA_PMI` | 0 | es1s, nq1s, fesx1s |

† Production default of 252 trading days; harness tests at 60 for speed.  
‡ Warmup = (n_releases + 1) × weekly_cadence ≈ 30–40 days with default n_releases=5.

---

## 3. Per-group economic intuition and citations

### M1 — Volatility / term structure

**Source series:** VIX (spot 30-day implied vol), VIX3M (3-month implied
vol), MOVE (bond vol index), CBOE SKEW.

The VIX level z-score captures the current fear regime relative to recent
history; sustained elevation (z > 2) historically coincides with equity
drawdowns and gold safe-haven inflows. The **VIX term slope** (VIX3M − VIX)
discriminates between *spot stress* (backwardation, negative slope =
near-term panic) and *normal* contango, where the market is calm now but
pricing in future uncertainty. MOVE z-score captures rates-market stress,
which leads equity vol and commodity repricing. The SKEW index measures the
price of OTM puts relative to calls on the SPX; an elevated SKEW while VIX
is low signals "the market is calm on the surface but traders are buying
tail-risk protection" — a precursor to a sharp move.

**Citations:**  
- CBOE VIX White Paper (2019) — term-structure construction.  
- Merrill Lynch MOVE Index methodology.  
- CBOE SKEW White Paper (2011).

---

### M2 — Rates / curve

**Source series:** 10Y UST, 2Y UST, 10Y Bund, TIPS 10Y, 10Y Breakeven.

The **2s10s slope** is the most studied recession predictor in macro
finance: every US recession since 1970 was preceded by an inversion
(Estrella & Hardouvelis 1991). For the meta-model, a steeply positive slope
favours equities (growth optimism); inversion = risk-off. The **UST–Bund
spread** drives EURUSD and makes fesx1s (Euro-denominated) more expensive
in USD terms. **Real yield** z-score is the gold/silver demand driver:
negative real yields = holding gold has zero opportunity cost. **Breakeven**
(nominal − real) captures inflation expectations; rising breakevens are
bullish for gold and energy.

**Citations:**  
- Estrella, A. & Hardouvelis, G. (1991) "The Term Structure as a Predictor
  of Real Economic Activity", *Journal of Finance* 46(2): 555–576.

---

### M3 — Credit

**Source series:** HY OAS (ICE BofA high-yield option-adjusted spread),
IG OAS (investment-grade OAS).

High-yield spreads widen before equity selloffs because: (a) the credit
market is more structurally informed; (b) HY defaults are correlated with
equity drawdowns. `hy_oas_z` is a slow-moving regime indicator; `hy_oas_5d_change`
captures the direction of current credit stress. The `hy_ig_ratio` (always
positive) measures the risk premium for stepping down the credit quality
ladder. A ratio well above its recent mean signals elevated junk-market
stress even when the absolute spread level is not extreme.

**Citations:**  
- Feldhuetter, P. & Lando, D. (2008) "Decomposing Swap Spreads", *Journal
  of Financial Economics* 88(2): 375–405 — credit spread predictability.

---

### M4 — FX / dollar

**Source series:** DXY (dollar index), EURUSD.

The US dollar has a structural inverse relationship with commodity prices
(invoiced in USD) and broad-based EM risk assets. A strong dollar
suppresses gold, oil, and metals; a weak dollar is the single largest macro
tailwind for the commodity complex. The `dxy_z` captures the current dollar
regime; `dxy_5d_change` captures momentum. `eurusd_5d_change` is a direct
P&L driver for fesx1s (Euro Stoxx futures, Euro-denominated equity index)
when converted to USD.

---

### M5 — Commodity fundamentals

**Source series:** EIA crude oil / distillate / gasoline / natural gas weekly
storage, LME copper warehouse stocks, Baltic Dry Index (BDI).

**EIA release surprises** are the most direct measure of energy
supply/demand imbalance at the sub-weekly cadence available in our panel. The
surprise formula is causal and release-count-based:

```
surprise_t = (Q_t − mean(Q_{t−1}, …, Q_{t−n})) / std(Q_{t−1}, …, Q_{t−n})
```

where `n = n_releases` (default 5 prior releases), and the rolling mean/std
operates over the release sub-series (not calendar days). An unexpected
inventory *build* (positive surprise) is bearish for the underlying
commodity; an unexpected *draw* is bullish.

**Copper stocks** (LME) are the `Dr. Copper` demand signal: rising stocks =
oversupply = bearish hg1s; falling stocks = tightness = bullish hg1s. The
z-score captures deviations from the rolling norm.

**Baltic Dry Index** is a global shipping demand proxy (dry-bulk: iron ore,
coal, grain). High BDI → strong demand for industrial raw materials → leading
indicator for global growth, bullish for energy and metals.

**Citations:**  
- EIA Weekly Petroleum Status Report.  
- EIA Natural Gas Storage Report.  
- Baltic Exchange BDI methodology.

---

### M6 — Macro growth

**Source series:** US ISM Manufacturing PMI (`US_ISM_MFG_PMI`), China
Caixin/Official Manufacturing PMI (`CHINA_PMI_MFG`).

The ISM PMI is a diffusion index of 5 sub-indices (new orders, production,
employment, supplier deliveries, inventories). A value above 50 indicates
expansion; below 50 = contraction. The **3m-change** feature
(`ism_pmi_3m_change`) captures cyclical momentum: improving PMI at any
absolute level suggests improving conditions. China PMI is a key demand
driver for copper (hg1s) and crude oil (cl1s) because China accounts for
~50% of global copper consumption.

`global_pmi_breadth` is the share of PMI indicators above 50 across US and
China. Range: {0.0, 0.5, 1.0}:
- **0.0** = synchronised global manufacturing contraction (risk-off, bearish
  equities + commodities).
- **0.5** = divergence (one economy expanding, one contracting).
- **1.0** = synchronised global expansion (risk-on, bullish equities + metals
  + energy).

**Citations:**  
- ISM Manufacturing PMI methodology (Institute for Supply Management).  
- Caixin China Manufacturing PMI methodology.

---

## 4. Causality contract

All features satisfy the universal harness contract:

1. **Truncation-invariance**: `feature(panel.iloc[:t+1]).iloc[t]` equals
   `feature(panel).iloc[t]` at `t ∈ {100, 200, 400}` on a 500-row synthetic
   macro panel.
2. **Shape preservation**: `len(feature(panel)) == len(panel)`.
3. **No NaN / Inf past warmup**: after the declared warmup window, every
   output value is finite on a fully-populated synthetic panel.

The EIA surprise features count **releases** (days where the EIA value
changes), not calendar days. `_eia_surprise` uses `shift(1)` on the
release sub-series so the rolling statistics at release `k` use only
releases `[k−n, …, k−1]` — causal. Non-release days forward-fill the last
release's surprise, which is also causal.

---

## 5. Warmup summary by group

| Group | Max warmup (production default) | Harness test window |
|---|--:|--:|
| M1 | 252 rows | 60 rows |
| M2 | 252 rows | 60 rows |
| M3 | 252 rows | 60 rows |
| M4 | 252 rows | 60 rows |
| M5 | 252 rows (z-score bound) | 60 rows |
| M6 | 63 rows | 21 rows |

The binding warmup for Step 4 feature matrix construction is 252 rows
(consistent with existing features in the pack).

---

## 6. Reproduction

```bash
uv run pytest tests/harry/test_macro_features.py -v   # 49 unit tests
uv run pytest tests/harry/test_causality.py -v         # 6 × 3 = 18 harness tests
uv run pytest tests/harry/ -q                          # full suite (239 tests)
```
