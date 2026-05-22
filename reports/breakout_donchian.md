# Replication family: `breakout_donchian`

Donchian-channel position: where the close sits in the trailing N-day band (band excludes today); a breach scores beyond +/-1.

## Parameter space

- `channel`: [10, 20, 40, 55]
- `deadband`: [0.0, 0.5, 0.8, 1.0]

Search tier(s) used: `tpe` (TPE above the n_eff FLOOR, exhaustive grid below it).

**Family verdict:** passes G1-G4 on **0** of 9 cell(s).

## Per-cell train/val panel

| cell | tier | n_eff | n_configs | val kappa | val ordinal_skill | composite | G1 | G2 | G3 | G4 | passed |
|------|------|------:|----------:|----------:|------------------:|----------:|:--:|:--:|:--:|:--:|:------:|
| `es1s` | tpe | 35 | 64 | -0.092 | -0.112 | -0.102 | no | no | no | no | no |
| `nq1s` | tpe | 20 | 64 | -0.012 | 0.007 | -0.003 | no | no | no | no | no |
| `fesx1s` | tpe | 25 | 64 | -0.052 | -0.097 | -0.074 | no | no | no | no | no |
| `rb1s` | tpe | 13 | 64 | -0.099 | -0.150 | -0.124 | no | no | no | no | no |
| `gc1s` | tpe | 11 | 64 | -0.093 | -0.125 | -0.109 | no | no | no | no | no |
| `si1s` | tpe | 19 | 64 | -0.026 | -0.058 | -0.042 | no | no | no | no | no |
| `hg1s` | tpe | 29 | 64 | 0.022 | 0.022 | 0.022 | no | no | no | yes | no |
| `pl1s` | tpe | 26 | 64 | -0.078 | -0.113 | -0.095 | no | no | no | no | no |
| `pool:energy` | tpe | 20 | 64 | 0.049 | 0.018 | 0.033 | no | yes | no | no | no |

## Per-cell detail (base rates, perturbation, gate diagnostics)

### `es1s`

- best params: `{'channel': 55, 'deadband': 0.0}`
- post-embargo val n_eff: 35 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 378 | 0.913 | 0.511 | 0.201 | 0.087 | 0.712 |
  | val | 126 | 0.881 | 0.579 | 0.151 | 0.119 | 0.730 |

  G3 perturbation (val composite over 2 +/-1-step neighbours): min=-0.094, max=-0.058, std=0.018 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa -0.092 vs cutoff 0.078, ordinal_skill -0.112 vs cutoff 0.078 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train -0.024, skill_val -0.092 (required >= -0.012, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa -0.092, ordinal_skill -0.112, increment_corr 0.427 (all must be > 0).

### `nq1s`

- best params: `{'channel': 55, 'deadband': 1.0}`
- post-embargo val n_eff: 20 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 378 | 0.944 | 0.209 | 0.368 | 0.056 | 0.577 |
  | val | 126 | 0.984 | 0.683 | 0.151 | 0.016 | 0.833 |

  G3 perturbation (val composite over 2 +/-1-step neighbours): min=-0.026, max=-0.009, std=0.009 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa -0.012 vs cutoff 0.078, ordinal_skill 0.007 vs cutoff 0.078 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train -0.070, skill_val -0.012 (required >= -0.035, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa -0.012, ordinal_skill 0.007, increment_corr 0.118 (all must be > 0).

### `fesx1s`

- best params: `{'channel': 55, 'deadband': 1.0}`
- post-embargo val n_eff: 25 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 381 | 1.000 | -0.108 | 0.554 | 0.000 | 0.446 |
  | val | 129 | 1.000 | 0.070 | 0.465 | 0.000 | 0.535 |

  G3 perturbation (val composite over 2 +/-1-step neighbours): min=-0.125, max=-0.073, std=0.026 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa -0.052 vs cutoff 0.092, ordinal_skill -0.097 vs cutoff 0.093 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train -0.066, skill_val -0.052 (required >= -0.033, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa -0.052, ordinal_skill -0.097, increment_corr -0.083 (all must be > 0).

### `rb1s`

- best params: `{'channel': 55, 'deadband': 0.5}`
- post-embargo val n_eff: 13 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 1.000 | 0.045 | 0.477 | 0.000 | 0.523 |
  | val | 126 | 1.000 | 0.683 | 0.159 | 0.000 | 0.841 |

  G3 perturbation (val composite over 3 +/-1-step neighbours): min=-0.219, max=-0.057, std=0.068 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa -0.099 vs cutoff 0.175, ordinal_skill -0.150 vs cutoff 0.176 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train -0.039, skill_val -0.099 (required >= -0.020, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa -0.099, ordinal_skill -0.150, increment_corr -0.126 (all must be > 0).

### `gc1s`

- best params: `{'channel': 10, 'deadband': 0.5}`
- post-embargo val n_eff: 11 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 0.313 | 0.154 | 0.080 | 0.687 | 0.233 |
  | val | 126 | 0.159 | 0.127 | 0.016 | 0.841 | 0.143 |

  G3 perturbation (val composite over 3 +/-1-step neighbours): min=-0.115, max=-0.044, std=0.030 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa -0.093 vs cutoff 0.078, ordinal_skill -0.125 vs cutoff 0.078 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.062, skill_val -0.093 (required >= 0.031, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa -0.093, ordinal_skill -0.125, increment_corr -0.295 (all must be > 0).

### `si1s`

- best params: `{'channel': 55, 'deadband': 1.0}`
- post-embargo val n_eff: 19 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 0.902 | -0.016 | 0.459 | 0.098 | 0.443 |
  | val | 126 | 0.960 | 0.230 | 0.365 | 0.040 | 0.595 |

  G3 perturbation (val composite over 2 +/-1-step neighbours): min=-0.132, max=-0.063, std=0.035 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa -0.026 vs cutoff 0.098, ordinal_skill -0.058 vs cutoff 0.108 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train -0.019, skill_val -0.026 (required >= -0.009, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa -0.026, ordinal_skill -0.058, increment_corr -0.329 (all must be > 0).

### `hg1s`

- best params: `{'channel': 55, 'deadband': 0.0}`
- post-embargo val n_eff: 29 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 1.000 | -0.088 | 0.544 | 0.000 | 0.456 |
  | val | 126 | 1.000 | -0.111 | 0.556 | 0.000 | 0.444 |

  G3 perturbation (val composite over 2 +/-1-step neighbours): min=-0.011, max=0.029, std=0.020 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa 0.022 vs cutoff 0.078, ordinal_skill 0.022 vs cutoff 0.079 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.133, skill_val 0.022 (required >= 0.066, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa 0.022, ordinal_skill 0.022, increment_corr 0.161 (all must be > 0).

### `pl1s`

- best params: `{'channel': 40, 'deadband': 1.0}`
- post-embargo val n_eff: 26 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 0.899 | 0.316 | 0.292 | 0.101 | 0.607 |
  | val | 126 | 0.897 | 0.722 | 0.087 | 0.103 | 0.810 |

  G3 perturbation (val composite over 3 +/-1-step neighbours): min=-0.141, max=-0.063, std=0.032 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa -0.078 vs cutoff 0.078, ordinal_skill -0.113 vs cutoff 0.078 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train -0.050, skill_val -0.078 (required >= -0.025, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa -0.078, ordinal_skill -0.113, increment_corr -0.405 (all must be > 0).

### `pool:energy`

- best params: `{'channel': 20, 'deadband': 0.0}`
- post-embargo val n_eff: 20 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 1131 | 0.257 | 0.148 | 0.055 | 0.743 | 0.202 |
  | val | 378 | 0.452 | 0.241 | 0.106 | 0.548 | 0.347 |

  G3 perturbation (val composite over 3 +/-1-step neighbours): min=-0.050, max=0.051, std=0.048 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa 0.049 vs cutoff 0.082, ordinal_skill 0.018 vs cutoff 0.083 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.025, skill_val 0.049 (required >= 0.013, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa 0.049, ordinal_skill 0.018, increment_corr -0.070 (all must be > 0).

