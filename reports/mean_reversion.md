# Replication family: `mean_reversion`

Counter-trend mean reversion (C1 prior-best). score_t = `-zscore(close - SMA_L)`: far above the moving average leans short, far below leans long. Iterated FIRST because the released signal is predominantly short-horizon counter-trend.

## Parameter space

- `lookback`: [5, 10, 20, 40]
- `z_window`: [20, 40, 60]
- `deadband`: [0.0, 0.25, 0.5, 1.0, 1.5]

Search tier(s) used: `tpe` (TPE above the n_eff FLOOR, exhaustive grid below it).

**Family verdict:** passes G1-G4 on **4** of 9 cell(s).

## Per-cell train/val panel

| cell | tier | n_eff | n_configs | val kappa | val ordinal_skill | composite | G1 | G2 | G3 | G4 | passed |
|------|------|------:|----------:|----------:|------------------:|----------:|:--:|:--:|:--:|:--:|:------:|
| `es1s` | tpe | 35 | 64 | 0.121 | 0.135 | 0.128 | yes | yes | yes | yes | yes |
| `nq1s` | tpe | 20 | 64 | -0.162 | -0.164 | -0.163 | no | no | no | no | no |
| `fesx1s` | tpe | 25 | 64 | 0.186 | 0.186 | 0.186 | yes | yes | yes | yes | yes |
| `rb1s` | tpe | 13 | 64 | 0.043 | 0.043 | 0.043 | no | no | no | yes | no |
| `gc1s` | tpe | 11 | 64 | 0.048 | 0.074 | 0.061 | no | yes | no | yes | no |
| `si1s` | tpe | 19 | 64 | 0.301 | 0.423 | 0.362 | yes | yes | yes | yes | yes |
| `hg1s` | tpe | 29 | 64 | 0.283 | 0.283 | 0.283 | yes | yes | yes | yes | yes |
| `pl1s` | tpe | 26 | 64 | 0.166 | 0.182 | 0.174 | yes | no | yes | yes | no |
| `pool:energy` | tpe | 20 | 64 | 0.015 | 0.027 | 0.021 | no | no | no | yes | no |

## Per-cell detail (base rates, perturbation, gate diagnostics)

### `es1s`

- best params: `{'lookback': 40, 'z_window': 40, 'deadband': 0.0}`
- post-embargo val n_eff: 35 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 378 | 0.913 | 0.511 | 0.201 | 0.087 | 0.712 |
  | val | 126 | 0.881 | 0.579 | 0.151 | 0.119 | 0.730 |

  G3 perturbation (val composite over 4 +/-1-step neighbours): min=0.073, max=0.106, std=0.014 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa 0.121 vs cutoff 0.078, ordinal_skill 0.135 vs cutoff 0.078 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.222, skill_val 0.121 (required >= 0.111, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa 0.121, ordinal_skill 0.135, increment_corr 0.330 (all must be > 0).

### `nq1s`

- best params: `{'lookback': 40, 'z_window': 40, 'deadband': 0.0}`
- post-embargo val n_eff: 20 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 378 | 0.944 | 0.209 | 0.368 | 0.056 | 0.577 |
  | val | 126 | 0.984 | 0.683 | 0.151 | 0.016 | 0.833 |

  G3 perturbation (val composite over 4 +/-1-step neighbours): min=-0.232, max=0.011, std=0.099 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa -0.162 vs cutoff 0.078, ordinal_skill -0.164 vs cutoff 0.078 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.189, skill_val -0.162 (required >= 0.094, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa -0.162, ordinal_skill -0.164, increment_corr 0.068 (all must be > 0).

### `fesx1s`

- best params: `{'lookback': 10, 'z_window': 20, 'deadband': 0.0}`
- post-embargo val n_eff: 25 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 381 | 1.000 | -0.108 | 0.554 | 0.000 | 0.446 |
  | val | 129 | 1.000 | 0.070 | 0.465 | 0.000 | 0.535 |

  G3 perturbation (val composite over 4 +/-1-step neighbours): min=0.148, max=0.309, std=0.063 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa 0.186 vs cutoff 0.092, ordinal_skill 0.186 vs cutoff 0.093 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.229, skill_val 0.186 (required >= 0.114, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa 0.186, ordinal_skill 0.186, increment_corr 0.314 (all must be > 0).

### `rb1s`

- best params: `{'lookback': 40, 'z_window': 40, 'deadband': 0.0}`
- post-embargo val n_eff: 13 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 1.000 | 0.045 | 0.477 | 0.000 | 0.523 |
  | val | 126 | 1.000 | 0.683 | 0.159 | 0.000 | 0.841 |

  G3 perturbation (val composite over 4 +/-1-step neighbours): min=0.012, max=0.060, std=0.020 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa 0.043 vs cutoff 0.175, ordinal_skill 0.043 vs cutoff 0.176 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.204, skill_val 0.043 (required >= 0.102, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa 0.043, ordinal_skill 0.043, increment_corr 0.463 (all must be > 0).

### `gc1s`

- best params: `{'lookback': 10, 'z_window': 60, 'deadband': 1.0}`
- post-embargo val n_eff: 11 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 0.313 | 0.154 | 0.080 | 0.687 | 0.233 |
  | val | 126 | 0.159 | 0.127 | 0.016 | 0.841 | 0.143 |

  G3 perturbation (val composite over 5 +/-1-step neighbours): min=-0.025, max=0.146, std=0.066 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa 0.048 vs cutoff 0.078, ordinal_skill 0.074 vs cutoff 0.078 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.075, skill_val 0.048 (required >= 0.037, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa 0.048, ordinal_skill 0.074, increment_corr 0.257 (all must be > 0).

### `si1s`

- best params: `{'lookback': 40, 'z_window': 40, 'deadband': 1.0}`
- post-embargo val n_eff: 19 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 0.902 | -0.016 | 0.459 | 0.098 | 0.443 |
  | val | 126 | 0.960 | 0.230 | 0.365 | 0.040 | 0.595 |

  G3 perturbation (val composite over 5 +/-1-step neighbours): min=0.224, max=0.418, std=0.063 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa 0.301 vs cutoff 0.098, ordinal_skill 0.423 vs cutoff 0.108 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.159, skill_val 0.301 (required >= 0.079, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa 0.301, ordinal_skill 0.423, increment_corr 0.559 (all must be > 0).

### `hg1s`

- best params: `{'lookback': 10, 'z_window': 60, 'deadband': 0.0}`
- post-embargo val n_eff: 29 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 1.000 | -0.088 | 0.544 | 0.000 | 0.456 |
  | val | 126 | 1.000 | -0.111 | 0.556 | 0.000 | 0.444 |

  G3 perturbation (val composite over 4 +/-1-step neighbours): min=0.150, max=0.325, std=0.065 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa 0.283 vs cutoff 0.078, ordinal_skill 0.283 vs cutoff 0.079 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.353, skill_val 0.283 (required >= 0.176, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa 0.283, ordinal_skill 0.283, increment_corr 0.158 (all must be > 0).

### `pl1s`

- best params: `{'lookback': 40, 'z_window': 60, 'deadband': 0.0}`
- post-embargo val n_eff: 26 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 0.899 | 0.316 | 0.292 | 0.101 | 0.607 |
  | val | 126 | 0.897 | 0.722 | 0.087 | 0.103 | 0.810 |

  G3 perturbation (val composite over 3 +/-1-step neighbours): min=0.121, max=0.147, std=0.011 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa 0.166 vs cutoff 0.078, ordinal_skill 0.182 vs cutoff 0.078 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.473, skill_val 0.166 (required >= 0.236, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa 0.166, ordinal_skill 0.182, increment_corr 0.144 (all must be > 0).

### `pool:energy`

- best params: `{'lookback': 5, 'z_window': 20, 'deadband': 1.5}`
- post-embargo val n_eff: 20 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 1131 | 0.257 | 0.148 | 0.055 | 0.743 | 0.202 |
  | val | 378 | 0.452 | 0.241 | 0.106 | 0.548 | 0.347 |

  G3 perturbation (val composite over 3 +/-1-step neighbours): min=-0.019, max=0.012, std=0.013 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa 0.015 vs cutoff 0.082, ordinal_skill 0.027 vs cutoff 0.083 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.058, skill_val 0.015 (required >= 0.029, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa 0.015, ordinal_skill 0.027, increment_corr 0.233 (all must be > 0).

