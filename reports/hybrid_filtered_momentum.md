# Replication family: `hybrid_filtered_momentum`

Fast momentum kept only when a slower trailing trend agrees in sign; disagreement forces flat.

## Parameter space

- `lookback`: [5, 10, 20]
- `filter_window`: [40, 60, 120]
- `vol_window`: [20, 40]
- `deadband`: [0.0, 0.25, 0.5, 1.0]

Search tier(s) used: `tpe` (TPE above the n_eff FLOOR, exhaustive grid below it).

**Family verdict:** passes G1-G4 on **0** of 9 cell(s).

## Per-cell train/val panel

| cell | tier | n_eff | n_configs | val kappa | val ordinal_skill | composite | G1 | G2 | G3 | G4 | passed |
|------|------|------:|----------:|----------:|------------------:|----------:|:--:|:--:|:--:|:--:|:------:|
| `es1s` | tpe | 35 | 64 | -0.029 | -0.063 | -0.046 | no | no | no | no | no |
| `nq1s` | tpe | 20 | 64 | 0.005 | 0.038 | 0.022 | no | yes | no | yes | no |
| `fesx1s` | tpe | 25 | 64 | -0.014 | -0.025 | -0.019 | no | no | no | no | no |
| `rb1s` | tpe | 13 | 64 | -0.017 | -0.032 | -0.024 | no | no | no | no | no |
| `gc1s` | tpe | 11 | 64 | -0.043 | -0.105 | -0.074 | no | no | no | no | no |
| `si1s` | tpe | 19 | 64 | -0.051 | -0.106 | -0.079 | no | no | no | no | no |
| `hg1s` | tpe | 29 | 64 | -0.091 | -0.135 | -0.113 | no | no | no | no | no |
| `pl1s` | tpe | 26 | 64 | -0.029 | -0.030 | -0.030 | no | no | no | no | no |
| `pool:energy` | tpe | 20 | 64 | -0.124 | -0.141 | -0.132 | no | no | no | no | no |

## Per-cell detail (base rates, perturbation, gate diagnostics)

### `es1s`

- best params: `{'lookback': 20, 'filter_window': 60, 'vol_window': 20, 'deadband': 0.0}`
- post-embargo val n_eff: 35 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 378 | 0.913 | 0.511 | 0.201 | 0.087 | 0.712 |
  | val | 126 | 0.881 | 0.579 | 0.151 | 0.119 | 0.730 |

  G3 perturbation (val composite over 5 +/-1-step neighbours): min=-0.091, max=0.016, std=0.035 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa -0.029 vs cutoff 0.078, ordinal_skill -0.063 vs cutoff 0.078 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.066, skill_val -0.029 (required >= 0.033, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa -0.029, ordinal_skill -0.063, increment_corr 0.443 (all must be > 0).

### `nq1s`

- best params: `{'lookback': 5, 'filter_window': 120, 'vol_window': 40, 'deadband': 1.0}`
- post-embargo val n_eff: 20 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 378 | 0.944 | 0.209 | 0.368 | 0.056 | 0.577 |
  | val | 126 | 0.984 | 0.683 | 0.151 | 0.016 | 0.833 |

  G3 perturbation (val composite over 4 +/-1-step neighbours): min=-0.001, max=0.070, std=0.026 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa 0.005 vs cutoff 0.078, ordinal_skill 0.038 vs cutoff 0.078 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train -0.035, skill_val 0.005 (required >= -0.017, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa 0.005, ordinal_skill 0.038, increment_corr 0.340 (all must be > 0).

### `fesx1s`

- best params: `{'lookback': 5, 'filter_window': 60, 'vol_window': 40, 'deadband': 1.0}`
- post-embargo val n_eff: 25 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 381 | 1.000 | -0.108 | 0.554 | 0.000 | 0.446 |
  | val | 129 | 1.000 | 0.070 | 0.465 | 0.000 | 0.535 |

  G3 perturbation (val composite over 5 +/-1-step neighbours): min=-0.109, max=0.001, std=0.042 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa -0.014 vs cutoff 0.092, ordinal_skill -0.025 vs cutoff 0.093 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train -0.043, skill_val -0.014 (required >= -0.022, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa -0.014, ordinal_skill -0.025, increment_corr -0.053 (all must be > 0).

### `rb1s`

- best params: `{'lookback': 20, 'filter_window': 120, 'vol_window': 40, 'deadband': 1.0}`
- post-embargo val n_eff: 13 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 1.000 | 0.045 | 0.477 | 0.000 | 0.523 |
  | val | 126 | 1.000 | 0.683 | 0.159 | 0.000 | 0.841 |

  G3 perturbation (val composite over 4 +/-1-step neighbours): min=-0.062, max=-0.008, std=0.019 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa -0.017 vs cutoff 0.175, ordinal_skill -0.032 vs cutoff 0.176 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train -0.020, skill_val -0.017 (required >= -0.010, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa -0.017, ordinal_skill -0.032, increment_corr 0.161 (all must be > 0).

### `gc1s`

- best params: `{'lookback': 10, 'filter_window': 120, 'vol_window': 20, 'deadband': 0.5}`
- post-embargo val n_eff: 11 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 0.313 | 0.154 | 0.080 | 0.687 | 0.233 |
  | val | 126 | 0.159 | 0.127 | 0.016 | 0.841 | 0.143 |

  G3 perturbation (val composite over 6 +/-1-step neighbours): min=-0.090, max=-0.016, std=0.027 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa -0.043 vs cutoff 0.078, ordinal_skill -0.105 vs cutoff 0.078 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.054, skill_val -0.043 (required >= 0.027, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa -0.043, ordinal_skill -0.105, increment_corr -0.425 (all must be > 0).

### `si1s`

- best params: `{'lookback': 5, 'filter_window': 120, 'vol_window': 40, 'deadband': 1.0}`
- post-embargo val n_eff: 19 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 0.902 | -0.016 | 0.459 | 0.098 | 0.443 |
  | val | 126 | 0.960 | 0.230 | 0.365 | 0.040 | 0.595 |

  G3 perturbation (val composite over 4 +/-1-step neighbours): min=-0.100, max=-0.075, std=0.009 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa -0.051 vs cutoff 0.098, ordinal_skill -0.106 vs cutoff 0.108 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train -0.023, skill_val -0.051 (required >= -0.011, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa -0.051, ordinal_skill -0.106, increment_corr -0.485 (all must be > 0).

### `hg1s`

- best params: `{'lookback': 20, 'filter_window': 60, 'vol_window': 20, 'deadband': 0.0}`
- post-embargo val n_eff: 29 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 1.000 | -0.088 | 0.544 | 0.000 | 0.456 |
  | val | 126 | 1.000 | -0.111 | 0.556 | 0.000 | 0.444 |

  G3 perturbation (val composite over 5 +/-1-step neighbours): min=-0.132, max=-0.075, std=0.020 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa -0.091 vs cutoff 0.078, ordinal_skill -0.135 vs cutoff 0.079 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.091, skill_val -0.091 (required >= 0.046, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa -0.091, ordinal_skill -0.135, increment_corr 0.056 (all must be > 0).

### `pl1s`

- best params: `{'lookback': 5, 'filter_window': 120, 'vol_window': 20, 'deadband': 1.0}`
- post-embargo val n_eff: 26 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 0.899 | 0.316 | 0.292 | 0.101 | 0.607 |
  | val | 126 | 0.897 | 0.722 | 0.087 | 0.103 | 0.810 |

  G3 perturbation (val composite over 4 +/-1-step neighbours): min=-0.037, max=-0.022, std=0.006 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa -0.029 vs cutoff 0.078, ordinal_skill -0.030 vs cutoff 0.078 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train -0.037, skill_val -0.029 (required >= -0.018, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa -0.029, ordinal_skill -0.030, increment_corr -0.397 (all must be > 0).

### `pool:energy`

- best params: `{'lookback': 10, 'filter_window': 120, 'vol_window': 40, 'deadband': 0.0}`
- post-embargo val n_eff: 20 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 1131 | 0.257 | 0.148 | 0.055 | 0.743 | 0.202 |
  | val | 378 | 0.452 | 0.241 | 0.106 | 0.548 | 0.347 |

  G3 perturbation (val composite over 5 +/-1-step neighbours): min=-0.219, max=-0.132, std=0.035 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa -0.124 vs cutoff 0.082, ordinal_skill -0.141 vs cutoff 0.083 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.008, skill_val -0.124 (required >= 0.004, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa -0.124, ordinal_skill -0.141, increment_corr 0.110 (all must be > 0).

