# Signal Characterization (C1 Checkpoint)

Reverse-engineering the primary trading signal `s_t in {-1, 0, +1}` for the released futures universe. All numbers below are produced by `stml.replication.characterize` on the cleaned data (`stml.io.load_clean_data`); this report only renders them.

Instruments analysed (11): `es1s`, `nq1s`, `fesx1s`, `cl1s`, `ho1s`, `rb1s`, `ng1s`, `gc1s`, `si1s`, `hg1s`, `pl1s`.

## Convention verdict

`corr_at_lag1` (corr of `s_t` with next-day return `r_t+1`) is **positive for all 11 instruments** -> the default `next_day` PnL convention is **empirically supported**. No instrument shows a non-positive next-day correlation.

Note the distinction surfaced per instrument below: a non-zero *construction* lag reflects a signal BUILT from trailing returns; the SIGN of that correlation gives the style (negative => mean-reversion / counter-trend, positive => momentum), independent of the forward PnL convention. 10 of 11 instruments load negatively (short-horizon mean-reversion / counter-trend).

The headline convention uses the FORWARD-restricted argmax (`best_forward_lag`, over h>=0); the `lead_lag` module's global `best_lag` over h in [-5, +5] is dominated by the (negative) construction relationship by design and is NOT the convention verdict.

| inst | corr@lag+1 | next_day? | best_fwd_lag | best_constr_lag |
|------|-----------:|:---------:|-------------:|----------------:|
| `es1s` | 0.103 | yes | 0 | -2 |
| `nq1s` | 0.114 | yes | 0 | -2 |
| `fesx1s` | 0.047 | yes | 0 | -1 |
| `cl1s` | 0.121 | yes | 1 | -5 |
| `ho1s` | 0.036 | yes | 0 | -1 |
| `rb1s` | 0.065 | yes | 0 | -1 |
| `ng1s` | 0.093 | yes | 1 | -2 |
| `gc1s` | 0.129 | yes | 1 | -4 |
| `si1s` | 0.057 | yes | 0 | -1 |
| `hg1s` | 0.122 | yes | 0 | -1 |
| `pl1s` | 0.071 | yes | 0 | -2 |

## Q4 -- cross-asset structure (panel-level)

Mean |off-diagonal| signal correlation = **0.091** (~0.11 expected: the 11 signals are nearly independent across assets).

Behavioral fingerprint clusters (participation / long-bias / persistence / momentum):
  - cluster 0: `ho1s`, `ng1s`, `gc1s`
  - cluster 1: `es1s`, `nq1s`, `cl1s`, `rb1s`, `pl1s`
  - cluster 2: `fesx1s`, `si1s`, `hg1s`

## Effective sample size (n_eff) and asset-class pooling map

`n_eff` is computed on the POST-embargo validation window (`n_eff(signal.iloc[embargoed_val(signal, split)])`). An instrument is **standalone** when its post-embargo val `n_eff >= FLOOR`, else it is folded into asset-class pooling (`pool:<class>`).

**FLOOR = 10** is a documented checkpoint parameter and is reviewable -- the user may change it. Lowering it admits more standalone instruments; raising it pools more.

| inst | class | post-embargo val n_eff | decision |
|------|-------|-----------------------:|----------|
| `es1s` | equity | 35 | standalone |
| `nq1s` | equity | 20 | standalone |
| `fesx1s` | equity | 25 | standalone |
| `cl1s` | energy | 9 | pool:energy |
| `ho1s` | energy | 9 | pool:energy |
| `rb1s` | energy | 13 | standalone |
| `ng1s` | energy | 2 | pool:energy |
| `gc1s` | metals | 11 | standalone |
| `si1s` | metals | 19 | standalone |
| `hg1s` | metals | 29 | standalone |
| `pl1s` | metals | 26 | standalone |

Sub-floor instruments (n_eff < 10, pooled): `cl1s`, `ho1s`, `ng1s`.

Pooled per-class post-embargo val n_eff:
  - equity: 80
  - energy: 33
  - metals: 85

## Per-instrument characterization (Q1, Q2, Q3, Q5, Q6)

### `es1s` (equity)

**Q1 -- alpha type (momentum vs mean-reversion).** `alpha_label` = **mean_reversion**, `momentum_score` = -0.199 (mean trailing-return corr; >0 momentum, <0 mean-reversion).
  - trailing-return corr: trail_1=-0.153 · trail_5=-0.289 · trail_10=-0.206 · trail_20=-0.149
  - distance-from-MA sign agreement: MA10=0.478 · MA20=0.522 · MA50=0.621 (fraction trading WITH the MA distance)
  - breakout coincidence: any=0.174, directional=0.089 (Donchian-20 break on a nonzero day)

**Q2 -- lead/lag and holding convention.**
  - lag profile `corr(s_t, r_t+h)`: h-5:-0.05, h-4:-0.11, h-3:-0.11, h-2:-0.18, h-1:-0.16, h+0:-0.14, h+1:0.10, h+2:0.04, h+3:0.05, h+4:0.03, h+5:-0.03
  - *forward (PnL) convention*: `corr_at_lag1` (h=+1) = **0.103** -> next_day confirmed; `best_forward_lag` (argmax|corr| over h in 0..+5) = 0 (corr -0.140).
  - *construction lag*: `best_construction_lag` (argmax|corr| over h in -5..-1) = -2 (corr -0.176) -- the signal is built from TRAILING returns; a NEGATIVE loading => mean-reversion / counter-trend construction (independent of the forward PnL convention).

**Q3 -- regime (does it avoid high vol?).** `participation_low_vol` = 0.917, `participation_high_vol` = 0.885 (GMM); median-vol split low=0.921 / high=0.908; **avoids_high_vol = yes** (status: ok). This verdict is instrument-specific.

**Q5 -- drift (per-split base rates).**

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 387 | 0.891 | 0.499 | 0.196 | 0.109 | 0.695 |
  | val | 129 | 0.860 | 0.566 | 0.147 | 0.140 | 0.713 |
  | test | 129 | 0.922 | 0.550 | 0.186 | 0.078 | 0.736 |

  - train->test trend: participation 0.031, long_bias 0.052.

**Q6 -- model-family fingerprint (ADVISORY).** label = **inconclusive (gates nothing)**, confidence = 0.019; CV acc tree=0.725 / linear=0.719 / forest=0.738 vs majority=0.725.

### `nq1s` (equity)

**Q1 -- alpha type (momentum vs mean-reversion).** `alpha_label` = **mean_reversion**, `momentum_score` = -0.101 (mean trailing-return corr; >0 momentum, <0 mean-reversion).
  - trailing-return corr: trail_1=-0.048 · trail_5=-0.147 · trail_10=-0.094 · trail_20=-0.113
  - distance-from-MA sign agreement: MA10=0.487 · MA20=0.507 · MA50=0.502 (fraction trading WITH the MA distance)
  - breakout coincidence: any=0.184, directional=0.076 (Donchian-20 break on a nonzero day)

**Q2 -- lead/lag and holding convention.**
  - lag profile `corr(s_t, r_t+h)`: h-5:-0.06, h-4:-0.04, h-3:-0.04, h-2:-0.11, h-1:-0.05, h+0:-0.17, h+1:0.11, h+2:0.05, h+3:0.06, h+4:0.09, h+5:0.01
  - *forward (PnL) convention*: `corr_at_lag1` (h=+1) = **0.114** -> next_day confirmed; `best_forward_lag` (argmax|corr| over h in 0..+5) = 0 (corr -0.166).
  - *construction lag*: `best_construction_lag` (argmax|corr| over h in -5..-1) = -2 (corr -0.114) -- the signal is built from TRAILING returns; a NEGATIVE loading => mean-reversion / counter-trend construction (independent of the forward PnL convention).

**Q3 -- regime (does it avoid high vol?).** `participation_low_vol` = 0.958, `participation_high_vol` = 0.974 (GMM); median-vol split low=0.952 / high=0.968; **avoids_high_vol = no** (status: ok). This verdict is instrument-specific.

**Q5 -- drift (per-split base rates).**

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 387 | 0.922 | 0.204 | 0.359 | 0.078 | 0.563 |
  | val | 129 | 0.961 | 0.667 | 0.147 | 0.039 | 0.814 |
  | test | 129 | 0.953 | 0.271 | 0.341 | 0.047 | 0.612 |

  - train->test trend: participation 0.031, long_bias 0.067.

**Q6 -- model-family fingerprint (ADVISORY).** label = **inconclusive (gates nothing)**, confidence = 0.000; CV acc tree=0.530 / linear=0.590 / forest=0.609 vs majority=0.639.

### `fesx1s` (equity)

**Q1 -- alpha type (momentum vs mean-reversion).** `alpha_label` = **mean_reversion**, `momentum_score` = -0.192 (mean trailing-return corr; >0 momentum, <0 mean-reversion).
  - trailing-return corr: trail_1=-0.194 · trail_5=-0.198 · trail_10=-0.192 · trail_20=-0.182
  - distance-from-MA sign agreement: MA10=0.384 · MA20=0.410 · MA50=0.433 (fraction trading WITH the MA distance)
  - breakout coincidence: any=0.138, directional=0.033 (Donchian-20 break on a nonzero day)

**Q2 -- lead/lag and holding convention.**
  - lag profile `corr(s_t, r_t+h)`: h-5:-0.04, h-4:-0.07, h-3:-0.03, h-2:-0.14, h-1:-0.19, h+0:-0.18, h+1:0.05, h+2:0.03, h+3:-0.02, h+4:0.03, h+5:-0.03
  - *forward (PnL) convention*: `corr_at_lag1` (h=+1) = **0.047** -> next_day confirmed; `best_forward_lag` (argmax|corr| over h in 0..+5) = 0 (corr -0.179).
  - *construction lag*: `best_construction_lag` (argmax|corr| over h in -5..-1) = -1 (corr -0.194) -- the signal is built from TRAILING returns; a NEGATIVE loading => mean-reversion / counter-trend construction (independent of the forward PnL convention).

**Q3 -- regime (does it avoid high vol?).** `participation_low_vol` = 1.000, `participation_high_vol` = 1.000 (GMM); median-vol split low=1.000 / high=1.000; **avoids_high_vol = no** (status: ok). This verdict is instrument-specific.

**Q5 -- drift (per-split base rates).**

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 387 | 0.984 | -0.106 | 0.545 | 0.016 | 0.439 |
  | val | 129 | 1.000 | 0.070 | 0.465 | 0.000 | 0.535 |
  | test | 129 | 0.984 | -0.256 | 0.620 | 0.016 | 0.364 |

  - train->test trend: participation 0.000, long_bias -0.150.

**Q6 -- model-family fingerprint (ADVISORY).** label = **inconclusive (gates nothing)**, confidence = 0.011; CV acc tree=0.538 / linear=0.554 / forest=0.527 vs majority=0.551.

### `cl1s` (energy)

**Q1 -- alpha type (momentum vs mean-reversion).** `alpha_label` = **mean_reversion**, `momentum_score` = -0.063 (mean trailing-return corr; >0 momentum, <0 mean-reversion).
  - trailing-return corr: trail_1=-0.060 · trail_5=-0.051 · trail_10=-0.026 · trail_20=-0.114
  - distance-from-MA sign agreement: MA10=0.633 · MA20=0.630 · MA50=0.616 (fraction trading WITH the MA distance)
  - breakout coincidence: any=0.145, directional=0.090 (Donchian-20 break on a nonzero day)

**Q2 -- lead/lag and holding convention.**
  - lag profile `corr(s_t, r_t+h)`: h-5:0.03, h-4:-0.01, h-3:0.03, h-2:-0.00, h-1:0.00, h+0:0.00, h+1:0.12, h+2:0.08, h+3:0.07, h+4:0.08, h+5:0.09
  - *forward (PnL) convention*: `corr_at_lag1` (h=+1) = **0.121** -> next_day confirmed; `best_forward_lag` (argmax|corr| over h in 0..+5) = 1 (corr 0.121).
  - *construction lag*: `best_construction_lag` (argmax|corr| over h in -5..-1) = -5 (corr 0.028) -- the signal is built from TRAILING returns; a POSITIVE loading => momentum / trend-following construction (independent of the forward PnL convention).

**Q3 -- regime (does it avoid high vol?).** `participation_low_vol` = 0.708, `participation_high_vol` = 0.462 (GMM); median-vol split low=0.672 / high=0.672; **avoids_high_vol = yes** (status: ok). This verdict is instrument-specific.

**Q5 -- drift (per-split base rates).**

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 387 | 0.592 | 0.457 | 0.067 | 0.408 | 0.525 |
  | val | 129 | 0.806 | 0.806 | 0.000 | 0.194 | 0.806 |
  | test | 129 | 0.690 | 0.535 | 0.078 | 0.310 | 0.612 |

  - train->test trend: participation 0.098, long_bias 0.078.

**Q6 -- model-family fingerprint (ADVISORY).** label = **inconclusive (gates nothing)**, confidence = 0.000; CV acc tree=0.546 / linear=0.554 / forest=0.522 vs majority=0.615.

### `ho1s` (energy)

**Q1 -- alpha type (momentum vs mean-reversion).** `alpha_label` = **mean_reversion**, `momentum_score` = -0.304 (mean trailing-return corr; >0 momentum, <0 mean-reversion).
  - trailing-return corr: trail_1=-0.150 · trail_5=-0.302 · trail_10=-0.367 · trail_20=-0.397
  - distance-from-MA sign agreement: MA10=0.270 · MA20=0.238 · MA50=0.270 (fraction trading WITH the MA distance)
  - breakout coincidence: any=0.175, directional=0.016 (Donchian-20 break on a nonzero day)

**Q2 -- lead/lag and holding convention.**
  - lag profile `corr(s_t, r_t+h)`: h-5:-0.02, h-4:-0.03, h-3:-0.04, h-2:-0.09, h-1:-0.11, h+0:-0.15, h+1:0.04, h+2:0.02, h+3:0.03, h+4:0.06, h+5:-0.04
  - *forward (PnL) convention*: `corr_at_lag1` (h=+1) = **0.036** -> next_day confirmed; `best_forward_lag` (argmax|corr| over h in 0..+5) = 0 (corr -0.155).
  - *construction lag*: `best_construction_lag` (argmax|corr| over h in -5..-1) = -1 (corr -0.114) -- the signal is built from TRAILING returns; a NEGATIVE loading => mean-reversion / counter-trend construction (independent of the forward PnL convention).

**Q3 -- regime (does it avoid high vol?).** `participation_low_vol` = 0.091, `participation_high_vol` = 0.137 (GMM); median-vol split low=0.080 / high=0.121; **avoids_high_vol = no** (status: ok). This verdict is instrument-specific.

**Q5 -- drift (per-split base rates).**

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 387 | 0.088 | 0.047 | 0.021 | 0.912 | 0.067 |
  | val | 129 | 0.209 | 0.209 | 0.000 | 0.791 | 0.209 |
  | test | 129 | 0.016 | -0.016 | 0.016 | 0.984 | 0.000 |

  - train->test trend: participation -0.072, long_bias -0.062.

**Q6 -- model-family fingerprint (ADVISORY).** label = **inconclusive (gates nothing)**, confidence = 0.000; CV acc tree=0.826 / linear=0.779 / forest=0.865 vs majority=0.900.

### `rb1s` (energy)

**Q1 -- alpha type (momentum vs mean-reversion).** `alpha_label` = **mean_reversion**, `momentum_score` = -0.074 (mean trailing-return corr; >0 momentum, <0 mean-reversion).
  - trailing-return corr: trail_1=-0.102 · trail_5=-0.144 · trail_10=-0.064 · trail_20=0.015
  - distance-from-MA sign agreement: MA10=0.424 · MA20=0.452 · MA50=0.490 (fraction trading WITH the MA distance)
  - breakout coincidence: any=0.153, directional=0.053 (Donchian-20 break on a nonzero day)

**Q2 -- lead/lag and holding convention.**
  - lag profile `corr(s_t, r_t+h)`: h-5:-0.06, h-4:-0.05, h-3:-0.04, h-2:-0.07, h-1:-0.10, h+0:-0.07, h+1:0.07, h+2:-0.01, h+3:0.02, h+4:-0.01, h+5:-0.03
  - *forward (PnL) convention*: `corr_at_lag1` (h=+1) = **0.065** -> next_day confirmed; `best_forward_lag` (argmax|corr| over h in 0..+5) = 0 (corr -0.070).
  - *construction lag*: `best_construction_lag` (argmax|corr| over h in -5..-1) = -1 (corr -0.102) -- the signal is built from TRAILING returns; a NEGATIVE loading => mean-reversion / counter-trend construction (independent of the forward PnL convention).

**Q3 -- regime (does it avoid high vol?).** `participation_low_vol` = 1.000, `participation_high_vol` = 1.000 (GMM); median-vol split low=1.000 / high=1.000; **avoids_high_vol = no** (status: ok). This verdict is instrument-specific.

**Q5 -- drift (per-split base rates).**

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 387 | 0.974 | 0.044 | 0.465 | 0.026 | 0.509 |
  | val | 129 | 0.977 | 0.667 | 0.155 | 0.023 | 0.822 |
  | test | 129 | 0.969 | 0.023 | 0.473 | 0.031 | 0.496 |

  - train->test trend: participation -0.005, long_bias -0.021.

**Q6 -- model-family fingerprint (ADVISORY).** label = **linear (advisory only, low confidence)**, confidence = 0.053; CV acc tree=0.611 / linear=0.629 / forest=0.608 vs majority=0.584.

### `ng1s` (energy)

**Q1 -- alpha type (momentum vs mean-reversion).** `alpha_label` = **neutral**, `momentum_score` = n/a (mean trailing-return corr; >0 momentum, <0 mean-reversion).
  - trailing-return corr: trail_1=n/a · trail_5=n/a · trail_10=n/a · trail_20=n/a
  - distance-from-MA sign agreement: MA10=0.492 · MA20=0.460 · MA50=0.371 (fraction trading WITH the MA distance)
  - breakout coincidence: any=0.097, directional=0.040 (Donchian-20 break on a nonzero day)

**Q2 -- lead/lag and holding convention.**
  - lag profile `corr(s_t, r_t+h)`: h-5:0.02, h-4:0.02, h-3:-0.02, h-2:-0.06, h-1:-0.04, h+0:-0.00, h+1:0.09, h+2:0.06, h+3:0.06, h+4:0.02, h+5:0.06
  - *forward (PnL) convention*: `corr_at_lag1` (h=+1) = **0.093** -> next_day confirmed; `best_forward_lag` (argmax|corr| over h in 0..+5) = 1 (corr 0.093).
  - *construction lag*: `best_construction_lag` (argmax|corr| over h in -5..-1) = -2 (corr -0.058) -- the signal is built from TRAILING returns; a NEGATIVE loading => mean-reversion / counter-trend construction (independent of the forward PnL convention).

**Q3 -- regime (does it avoid high vol?).** `participation_low_vol` = 0.012, `participation_high_vol` = 0.393 (GMM); median-vol split low=0.006 / high=0.389; **avoids_high_vol = no** (status: ok). This verdict is instrument-specific.

**Q5 -- drift (per-split base rates).**

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 387 | 0.072 | -0.072 | 0.072 | 0.928 | 0.000 |
  | val | 129 | 0.310 | -0.310 | 0.310 | 0.690 | 0.000 |
  | test | 129 | 0.434 | -0.434 | 0.434 | 0.566 | 0.000 |

  - train->test trend: participation 0.362, long_bias -0.362.

**Q6 -- model-family fingerprint (ADVISORY).** label = **inconclusive (gates nothing)**, confidence = 0.019; CV acc tree=0.795 / linear=0.807 / forest=0.817 vs majority=0.803.

### `gc1s` (metals)

**Q1 -- alpha type (momentum vs mean-reversion).** `alpha_label` = **mean_reversion**, `momentum_score` = -0.274 (mean trailing-return corr; >0 momentum, <0 mean-reversion).
  - trailing-return corr: trail_1=-0.112 · trail_5=-0.336 · trail_10=-0.289 · trail_20=-0.361
  - distance-from-MA sign agreement: MA10=0.423 · MA20=0.357 · MA50=0.298 (fraction trading WITH the MA distance)
  - breakout coincidence: any=0.173, directional=0.071 (Donchian-20 break on a nonzero day)

**Q2 -- lead/lag and holding convention.**
  - lag profile `corr(s_t, r_t+h)`: h-5:-0.09, h-4:-0.15, h-3:-0.11, h-2:-0.08, h-1:-0.09, h+0:-0.02, h+1:0.13, h+2:0.06, h+3:0.04, h+4:0.02, h+5:0.02
  - *forward (PnL) convention*: `corr_at_lag1` (h=+1) = **0.129** -> next_day confirmed; `best_forward_lag` (argmax|corr| over h in 0..+5) = 1 (corr 0.129).
  - *construction lag*: `best_construction_lag` (argmax|corr| over h in -5..-1) = -4 (corr -0.145) -- the signal is built from TRAILING returns; a NEGATIVE loading => mean-reversion / counter-trend construction (independent of the forward PnL convention).

**Q3 -- regime (does it avoid high vol?).** `participation_low_vol` = 0.247, `participation_high_vol` = 0.355 (GMM); median-vol split low=0.255 / high=0.280; **avoids_high_vol = no** (status: ok). This verdict is instrument-specific.

**Q5 -- drift (per-split base rates).**

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 387 | 0.305 | 0.150 | 0.078 | 0.695 | 0.227 |
  | val | 129 | 0.155 | 0.124 | 0.016 | 0.845 | 0.140 |
  | test | 129 | 0.233 | 0.233 | 0.000 | 0.767 | 0.233 |

  - train->test trend: participation -0.072, long_bias 0.083.

**Q6 -- model-family fingerprint (ADVISORY).** label = **inconclusive (gates nothing)**, confidence = 0.025; CV acc tree=0.740 / linear=0.755 / forest=0.748 vs majority=0.732.

### `si1s` (metals)

**Q1 -- alpha type (momentum vs mean-reversion).** `alpha_label` = **mean_reversion**, `momentum_score` = -0.289 (mean trailing-return corr; >0 momentum, <0 mean-reversion).
  - trailing-return corr: trail_1=-0.152 · trail_5=-0.259 · trail_10=-0.385 · trail_20=-0.358
  - distance-from-MA sign agreement: MA10=0.348 · MA20=0.304 · MA50=0.308 (fraction trading WITH the MA distance)
  - breakout coincidence: any=0.111, directional=0.021 (Donchian-20 break on a nonzero day)

**Q2 -- lead/lag and holding convention.**
  - lag profile `corr(s_t, r_t+h)`: h-5:-0.12, h-4:-0.08, h-3:-0.10, h-2:-0.11, h-1:-0.14, h+0:-0.16, h+1:0.06, h+2:-0.02, h+3:-0.04, h+4:0.03, h+5:0.00
  - *forward (PnL) convention*: `corr_at_lag1` (h=+1) = **0.057** -> next_day confirmed; `best_forward_lag` (argmax|corr| over h in 0..+5) = 0 (corr -0.158).
  - *construction lag*: `best_construction_lag` (argmax|corr| over h in -5..-1) = -1 (corr -0.143) -- the signal is built from TRAILING returns; a NEGATIVE loading => mean-reversion / counter-trend construction (independent of the forward PnL convention).

**Q3 -- regime (does it avoid high vol?).** `participation_low_vol` = 0.933, `participation_high_vol` = 0.888 (GMM); median-vol split low=0.936 / high=0.904; **avoids_high_vol = yes** (status: ok). This verdict is instrument-specific.

**Q5 -- drift (per-split base rates).**

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 387 | 0.879 | -0.016 | 0.447 | 0.121 | 0.432 |
  | val | 129 | 0.938 | 0.225 | 0.357 | 0.062 | 0.581 |
  | test | 129 | 0.907 | 0.085 | 0.411 | 0.093 | 0.496 |

  - train->test trend: participation 0.028, long_bias 0.101.

**Q6 -- model-family fingerprint (ADVISORY).** label = **nonlinear (advisory only, low confidence)**, confidence = 0.112; CV acc tree=0.592 / linear=0.592 / forest=0.597 vs majority=0.487.

### `hg1s` (metals)

**Q1 -- alpha type (momentum vs mean-reversion).** `alpha_label` = **mean_reversion**, `momentum_score` = -0.126 (mean trailing-return corr; >0 momentum, <0 mean-reversion).
  - trailing-return corr: trail_1=-0.108 · trail_5=-0.168 · trail_10=-0.104 · trail_20=-0.125
  - distance-from-MA sign agreement: MA10=0.404 · MA20=0.422 · MA50=0.511 (fraction trading WITH the MA distance)
  - breakout coincidence: any=0.126, directional=0.049 (Donchian-20 break on a nonzero day)

**Q2 -- lead/lag and holding convention.**
  - lag profile `corr(s_t, r_t+h)`: h-5:-0.09, h-4:-0.05, h-3:-0.04, h-2:-0.08, h-1:-0.11, h+0:-0.18, h+1:0.12, h+2:0.03, h+3:0.02, h+4:0.02, h+5:0.02
  - *forward (PnL) convention*: `corr_at_lag1` (h=+1) = **0.122** -> next_day confirmed; `best_forward_lag` (argmax|corr| over h in 0..+5) = 0 (corr -0.179).
  - *construction lag*: `best_construction_lag` (argmax|corr| over h in -5..-1) = -1 (corr -0.108) -- the signal is built from TRAILING returns; a NEGATIVE loading => mean-reversion / counter-trend construction (independent of the forward PnL convention).

**Q3 -- regime (does it avoid high vol?).** `participation_low_vol` = 1.000, `participation_high_vol` = 1.000 (GMM); median-vol split low=1.000 / high=1.000; **avoids_high_vol = no** (status: ok). This verdict is instrument-specific.

**Q5 -- drift (per-split base rates).**

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 387 | 0.974 | -0.085 | 0.530 | 0.026 | 0.444 |
  | val | 129 | 0.977 | -0.109 | 0.543 | 0.023 | 0.434 |
  | test | 129 | 0.969 | 0.287 | 0.341 | 0.031 | 0.628 |

  - train->test trend: participation -0.005, long_bias 0.372.

**Q6 -- model-family fingerprint (ADVISORY).** label = **linear (advisory only, low confidence)**, confidence = 0.061; CV acc tree=0.562 / linear=0.567 / forest=0.506 vs majority=0.508.

### `pl1s` (metals)

**Q1 -- alpha type (momentum vs mean-reversion).** `alpha_label` = **mean_reversion**, `momentum_score` = -0.297 (mean trailing-return corr; >0 momentum, <0 mean-reversion).
  - trailing-return corr: trail_1=-0.144 · trail_5=-0.276 · trail_10=-0.317 · trail_20=-0.453
  - distance-from-MA sign agreement: MA10=0.384 · MA20=0.339 · MA50=0.338 (fraction trading WITH the MA distance)
  - breakout coincidence: any=0.090, directional=0.018 (Donchian-20 break on a nonzero day)

**Q2 -- lead/lag and holding convention.**
  - lag profile `corr(s_t, r_t+h)`: h-5:-0.05, h-4:-0.09, h-3:-0.15, h-2:-0.19, h-1:-0.15, h+0:-0.19, h+1:0.07, h+2:0.05, h+3:0.06, h+4:0.02, h+5:0.02
  - *forward (PnL) convention*: `corr_at_lag1` (h=+1) = **0.071** -> next_day confirmed; `best_forward_lag` (argmax|corr| over h in 0..+5) = 0 (corr -0.187).
  - *construction lag*: `best_construction_lag` (argmax|corr| over h in -5..-1) = -2 (corr -0.187) -- the signal is built from TRAILING returns; a NEGATIVE loading => mean-reversion / counter-trend construction (independent of the forward PnL convention).

**Q3 -- regime (does it avoid high vol?).** `participation_low_vol` = 0.887, `participation_high_vol` = 0.891 (GMM); median-vol split low=0.904 / high=0.869; **avoids_high_vol = no** (status: ok). This verdict is instrument-specific.

**Q5 -- drift (per-split base rates).**

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 387 | 0.876 | 0.307 | 0.284 | 0.124 | 0.592 |
  | val | 129 | 0.876 | 0.705 | 0.085 | 0.124 | 0.791 |
  | test | 129 | 0.814 | 0.581 | 0.116 | 0.186 | 0.698 |

  - train->test trend: participation -0.062, long_bias 0.274.

**Q6 -- model-family fingerprint (ADVISORY).** label = **nonlinear (advisory only, low confidence)**, confidence = 0.083; CV acc tree=0.739 / linear=0.739 / forest=0.748 vs majority=0.670.

## Summary table (all instruments)

| inst | class | alpha | momentum_score | corr@lag+1 | avoids_high_vol | fingerprint (conf) | n_eff | decision |
|------|-------|-------|---------------:|-----------:|:---------------:|--------------------|------:|----------|
| `es1s` | equity | mean_reversion | -0.199 | 0.103 | yes | inconclusive (0.019) | 35 | standalone |
| `nq1s` | equity | mean_reversion | -0.101 | 0.114 | no | inconclusive (0.000) | 20 | standalone |
| `fesx1s` | equity | mean_reversion | -0.192 | 0.047 | no | inconclusive (0.011) | 25 | standalone |
| `cl1s` | energy | mean_reversion | -0.063 | 0.121 | yes | inconclusive (0.000) | 9 | pool:energy |
| `ho1s` | energy | mean_reversion | -0.304 | 0.036 | no | inconclusive (0.000) | 9 | pool:energy |
| `rb1s` | energy | mean_reversion | -0.074 | 0.065 | no | linear (0.053) | 13 | standalone |
| `ng1s` | energy | neutral | n/a | 0.093 | no | inconclusive (0.019) | 2 | pool:energy |
| `gc1s` | metals | mean_reversion | -0.274 | 0.129 | no | inconclusive (0.025) | 11 | standalone |
| `si1s` | metals | mean_reversion | -0.289 | 0.057 | yes | nonlinear (0.112) | 19 | standalone |
| `hg1s` | metals | mean_reversion | -0.126 | 0.122 | no | linear (0.061) | 29 | standalone |
| `pl1s` | metals | mean_reversion | -0.297 | 0.071 | no | nonlinear (0.083) | 26 | standalone |

## Train-only threshold calibration

`results/jj/thresholds.json` is calibrated on the **train** split only (2020-01-03 .. 2021-07-01); no val/test data is used (the frozen anti-leakage commitment). For each instrument the four naive baselines (always_flat, majority_class, stratified_random(seed=0), persistence) are scored on the TRAIN signal. Each per-metric cutoff is `max(chance baselines) + 0.05` (the documented margin) -- the strongest of the no-skill references {always_flat, majority_class, stratified_random}. A replica must clear these cutoffs to count as skillful.

Cutoffs are set off no-skill baselines (flat / majority / stratified-random). Persistence (`s_t = s_{t-1}`) is a strong but trivially-laggable predictor and is shown separately as an UPPER REFERENCE, not the replication pass bar -- whether a replica must also beat persistence is a checkpoint decision for the user.

Suggested chance-level cutoffs vs the persistence reference (kappa / mcc / macro_f1 / ordinal_skill[vs_flat]):

| inst | kappa cut | mcc cut | macro_f1 cut | ordinal_skill cut | | persist kappa | persist mcc | persist macro_f1 | persist ordinal_skill |
|------|----------:|--------:|-------------:|------------------:|--|--------------:|------------:|-----------------:|---------------------:|
| `es1s` | 0.050 | 0.050 | 0.370 | 0.050 | | 0.487 | 0.487 | 0.578 | 0.585 |
| `nq1s` | 0.050 | 0.050 | 0.381 | 0.050 | | 0.506 | 0.506 | 0.590 | 0.552 |
| `fesx1s` | 0.063 | 0.064 | 0.383 | 0.064 | | 0.529 | 0.529 | 0.564 | 0.542 |
| `cl1s` | 0.067 | 0.067 | 0.416 | 0.070 | | 0.697 | 0.697 | 0.822 | 0.731 |
| `ho1s` | 0.050 | 0.050 | 0.368 | 0.050 | | 0.398 | 0.398 | 0.565 | 0.408 |
| `rb1s` | 0.146 | 0.147 | 0.439 | 0.148 | | 0.754 | 0.754 | 0.597 | 0.795 |
| `ng1s` | 0.050 | 0.050 | 0.531 | 0.050 | | 0.615 | 0.615 | 0.808 | 0.615 |
| `gc1s` | 0.050 | 0.050 | 0.337 | 0.050 | | 0.595 | 0.595 | 0.746 | 0.624 |
| `si1s` | 0.070 | 0.070 | 0.385 | 0.080 | | 0.440 | 0.440 | 0.550 | 0.520 |
| `hg1s` | 0.050 | 0.050 | 0.376 | 0.050 | | 0.624 | 0.624 | 0.550 | 0.659 |
| `pl1s` | 0.050 | 0.050 | 0.369 | 0.050 | | 0.553 | 0.553 | 0.625 | 0.649 |

