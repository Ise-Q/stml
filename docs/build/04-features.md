# Stage 3a — Full Feature Library

> Module: [`src/stml/features.py`](../../src/stml/features.py) (58 features
> at this depth; 66 with G6 regimes via [`05-regimes.md`](05-regimes.md))
> Tests: existing tests in `tests/test_labeling.py` cover label invariants;
> feature-specific tests are smoke-checked in the master pipeline.

## What changed vs Stage 2a

| Group | Stage 2a | Stage 3a |
|---|---:|---:|
| G1 Volatility | 13 | **19** (+ Parkinson, Garman-Klass, Rogers-Satchell, all z-scored) |
| G2 Trend | 12 | **15** (+ backward trend-scanning t-value at 10/21/42d) |
| G3 Mean-rev | 3 | **4** (+ Hurst exponent) |
| G4 Microstructure | 0 | **10** (volume z, volume/OI trends, Amihud, range) |
| G5 Signal context | 6 | 6 (unchanged) |
| G7 Calendar | 4 | 4 (unchanged) |
| **Total non-G6** | **38** | **58** |

## Per-group economic rationale

### G1 — Volatility / risk state (19 features)

**Idea:** the meta-model's primary job is to identify regimes where trend
following pays vs. gets chopped. Vol state is the single strongest regime
signal in markets.

| Feature | What it captures |
|---|---|
| `vol_{5,21,63}d` | Realised vol across short / med / long horizons |
| `ewma_vol_50` | Smooth EWMA vol (less noisy than rolling) |
| `vol_ratio_5_63` | Short/long vol — >1 ⇒ regime expanding, danger zone for trend |
| `vol_of_vol_63` | Vol-of-vol — stability of the vol regime |
| `semivol_21d` | Downside-only vol — captures crash risk asymmetry |
| `parkinson_vol_21d` | OHLC-range estimator (Parkinson 1980) — 5× more efficient than close-to-close at the same window |
| `garman_klass_vol_21d` | Uses both OH and OC — 7× more efficient than Parkinson |
| `rogers_satchell_vol_21d` | Drift-independent estimator — works even when there's a trend |
| `z_*` | Per-instrument expanding-window z-scores — make scales comparable across the 11 instruments |

**Range estimators matter because** they capture *intra-bar* variance that
close-to-close vol misses. On COVID-crash days the daily range was huge but
close-to-close could understate it. The three range estimators have different
robustness to drift — including all three gives the model redundancy.

### G2 — Trend quality / momentum (15 features)

| Feature | What it captures |
|---|---|
| `mom_{5,21,63}d` | Past return over the window |
| `ma_dist_{21,63}d` | `log(close/MA) / (sigma_1d · √w)` — distance from MA in sigma units |
| `ma21_slope` | Dimensionless slope of the 21d MA |
| `trend_tval_{10,21,42}d` | **t-statistic of the slope of `log(close)` on a linear time index** over a backward window (Programming Session 1's `tValLinR`) |
| `z_*` | Expanding-window z-scores |

**Why trend t-value?** Raw momentum is just net return — it doesn't say
*how clean* the trend was. A 5% move that came as a single jump-day plus
random noise is very different from a smooth 5% drift. The t-statistic
penalises noisy paths and rewards monotone ones. Programming Session 1 uses
exactly this for trend-scanning labels; here we use it as a *feature* (the
backward analogue).

### G3 — Mean-reversion / path noise (4 features)

| Feature | What it captures |
|---|---|
| `autocorr_21d` | Lag-1 return autocorrelation — negative ⇒ mean-reverting |
| `efficiency_ratio_21d` | Kaufman: `|net move| / sum(|moves|)` ∈ [0, 1] |
| `variance_ratio_5d_21w` | `Var(5d ret) / (5 · Var(1d ret))` — >1 trending, <1 mean-reverting |
| `hurst_100d` | Rescaled-range Hurst exponent over 100d — >0.5 trending, <0.5 mean-reverting |

**Why four measures of the same thing?** Each has different sensitivity to
regime, window size, and noise. They are correlated but not collinear — the
cluster importance section (Stage 4) will collapse them into a "G3 mean-rev"
cluster and report cluster-level contribution.

### G4 — Microstructure / liquidity (10 features)

| Feature | What it captures |
|---|---|
| `volume_z_63d` | Standardised volume — high vol = elevated participation |
| `volume_trend_21d` | 21d log-volume slope — accumulation or distribution |
| `oi_trend_21d` | 21d open-interest slope — position-building or unwinding |
| `amihud_illiq_21d` | Amihud (2002): `mean(|ret| / dollar_volume)` — price impact per unit traded |
| `hl_range_21d` | 21d mean squared log-range — pathological vol indicator |
| `z_*` | Per-instrument expanding z-scores |

**Why microstructure matters for the meta-model:** trend signals are most
reliable when *backed by* volume and open-interest. A move on declining volume
is suspect (distribution / lack of conviction); a move on rising OI is
high-conviction. Amihud illiquidity flags when the market is structurally
fragile (low liquidity = wider spreads + greater slippage = trend less
tradeable).

### G5 — Signal context (6 features, unchanged from Stage 2a)

`side_signal`, `signal_run_len`, `days_since_flip`, plus the three
asset-class net-signal balance features.

**Cross-sectional breadth** (`net_signal_equity/energy/metals`) is the
single most useful "regime agreement" indicator. If energy is universally
long, that's a structural regime call; if it's split 50/50, the signal is
noisier and the meta-model should be more cautious.

### G7 — Calendar (4 features, unchanged from Stage 2a)

Cyclical sin/cos encoding of month and day-of-week. Unlikely to be a major
driver but cheap and standard (Programming Session 4 uses exactly this
encoding for COVID forecasting).

## Feature scale sanity check (real data, 4984 events)

```
                           mean   std    5%    50%    95%
parkinson_vol_21d         0.33   0.27  0.12  0.27   0.74
garman_klass_vol_21d      0.33   0.28  0.12  0.28   0.73
rogers_satchell_vol_21d   0.33   0.29  0.12  0.28   0.72
trend_tval_21d            1.66   6.02 -7.78  1.55  11.98
hurst_100d                0.60   0.08  0.47  0.60   0.74
amihud_illiq_21d          ~0     ~0    ~0    ~0     ~0       (tiny — by design)
volume_z_63d              0.04   1.08 -1.34 -0.14   2.10
oi_trend_21d             -0.001  0.017-0.030 -0.000 0.026
```

- The three range vol estimators agree to within a few percent ⇒ they capture
  the same first-order signal but their *differences* may be informative.
- `trend_tval_21d` has heavy tails (|t| up to ~12 in the 5th/95th percentiles)
  — that's expected for trend strength: most periods are noisy, occasional
  windows are very statistically significant.
- `hurst_100d` averaging 0.6 (>0.5) is consistent with the panel having
  modest persistence — these are trend-followable markets in aggregate.
- `amihud_illiq_21d` is mathematically tiny (futures have huge dollar
  volume); the *z-scored* version `z_amihud_illiq_21d` is what the model uses.
- `volume_z_63d` is well-behaved standard-normal-ish.

No NaN or inf values in the full 58 × 4984 matrix (besides 94 cells from
Amihud / OI in early-history dates).

## Standardisation policy

- **Per-instrument expanding-window z-scores** for all scale-dependent
  features (`min_periods=60`). Causal by construction. Crucial for pooling:
  oil's "0.04 daily vol" and S&P's "0.01 daily vol" both map to
  similar z-scores after expanding standardisation.
- **Raw** (already dimensionless): autocorrelation ∈ [-1, 1], efficiency
  ratio ∈ [0, 1], variance ratio, Hurst, vol ratio, trend t-statistic.
- **Both raw and z-scored** kept for the rest, so the cluster importance
  (Stage 4) can see whether the raw or standardised version is more useful
  in each cluster.

## Known limitations

1. **Amihud is sensitive to the volume==0 zero-volume rows** (the NA report
   flags 765 such rows). We currently use `replace(0, np.nan)` on dollar
   volume to avoid div-by-zero, which means those rows get NaN Amihud and
   the rolling mean covers them. A more careful treatment would use a robust
   estimator that explicitly ignores zero-vol rows.
2. **Hurst exponent on 100d is noisy** — typical SE is ±0.1 with this sample
   size. We use it more as a regime signal than a precise persistence
   measurement.
3. **Trend t-value scan is fixed-window**, not the full Lecture-1 trend-scan
   that picks the window that maximises |t|. We use three fixed windows
   (10/21/42d) — adding the argmax-window variant is a Stage 4 ablation.
