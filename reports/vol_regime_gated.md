# Replication family: `vol_regime_gated`

A base directional score (mean-reversion or momentum) that participates only inside a trailing vol regime (high/low quantile gate).

## Parameter space

- `base`: ['mean_reversion', 'ts_momentum']
- `lookback`: [10, 20, 40]
- `z_window`: [20, 40]
- `vol_window`: [20, 40]
- `regime`: ['high', 'low']
- `vol_quantile`: [0.3, 0.5, 0.7]
- `q_window`: [120]
- `deadband`: [0.0, 0.5, 1.0]

Search tier(s) used: `tpe` (TPE above the n_eff FLOOR, exhaustive grid below it).

**Family verdict:** passes G1-G4 on **2** of 9 cell(s).

## Per-cell train/val panel

| cell | tier | n_eff | n_configs | val kappa | val ordinal_skill | composite | G1 | G2 | G3 | G4 | passed |
|------|------|------:|----------:|----------:|------------------:|----------:|:--:|:--:|:--:|:--:|:------:|
| `es1s` | tpe | 35 | 64 | 0.179 | 0.220 | 0.199 | yes | yes | yes | yes | yes |
| `nq1s` | tpe | 20 | 64 | -0.065 | -0.072 | -0.068 | no | no | no | no | no |
| `fesx1s` | tpe | 25 | 64 | 0.061 | 0.095 | 0.078 | no | no | no | yes | no |
| `rb1s` | tpe | 13 | 64 | 0.006 | 0.009 | 0.007 | no | no | no | yes | no |
| `gc1s` | tpe | 11 | 64 | 0.478 | 0.480 | 0.479 | yes | yes | no | yes | no |
| `si1s` | tpe | 19 | 64 | 0.237 | 0.367 | 0.302 | yes | yes | yes | yes | yes |
| `hg1s` | tpe | 29 | 64 | -0.021 | -0.024 | -0.023 | no | no | no | no | no |
| `pl1s` | tpe | 26 | 64 | 0.108 | 0.075 | 0.091 | no | no | no | yes | no |
| `pool:energy` | tpe | 20 | 64 | 0.182 | 0.127 | 0.155 | yes | yes | yes | no | no |

## Per-cell detail (base rates, perturbation, gate diagnostics)

### `es1s`

- best params: `{'base': 'mean_reversion', 'lookback': 40, 'z_window': 40, 'vol_window': 40, 'regime': 'low', 'vol_quantile': 0.7, 'q_window': 120, 'deadband': 0.0}`
- post-embargo val n_eff: 35 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 378 | 0.913 | 0.511 | 0.201 | 0.087 | 0.712 |
  | val | 126 | 0.881 | 0.579 | 0.151 | 0.119 | 0.730 |

  G3 perturbation (val composite over 5 +/-1-step neighbours): min=0.101, max=0.188, std=0.033 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa 0.179 vs cutoff 0.078, ordinal_skill 0.220 vs cutoff 0.078 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.176, skill_val 0.179 (required >= 0.088, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa 0.179, ordinal_skill 0.220, increment_corr 0.313 (all must be > 0).

### `nq1s`

- best params: `{'base': 'mean_reversion', 'lookback': 40, 'z_window': 40, 'vol_window': 40, 'regime': 'low', 'vol_quantile': 0.7, 'q_window': 120, 'deadband': 0.0}`
- post-embargo val n_eff: 20 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 378 | 0.944 | 0.209 | 0.368 | 0.056 | 0.577 |
  | val | 126 | 0.984 | 0.683 | 0.151 | 0.016 | 0.833 |

  G3 perturbation (val composite over 5 +/-1-step neighbours): min=-0.141, max=-0.056, std=0.038 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa -0.065 vs cutoff 0.078, ordinal_skill -0.072 vs cutoff 0.078 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.151, skill_val -0.065 (required >= 0.076, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa -0.065, ordinal_skill -0.072, increment_corr 0.042 (all must be > 0).

### `fesx1s`

- best params: `{'base': 'mean_reversion', 'lookback': 10, 'z_window': 20, 'vol_window': 40, 'regime': 'low', 'vol_quantile': 0.7, 'q_window': 120, 'deadband': 0.0}`
- post-embargo val n_eff: 25 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 381 | 1.000 | -0.108 | 0.554 | 0.000 | 0.446 |
  | val | 129 | 1.000 | 0.070 | 0.465 | 0.000 | 0.535 |

  G3 perturbation (val composite over 5 +/-1-step neighbours): min=0.032, max=0.121, std=0.034 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa 0.061 vs cutoff 0.092, ordinal_skill 0.095 vs cutoff 0.093 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.151, skill_val 0.061 (required >= 0.075, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa 0.061, ordinal_skill 0.095, increment_corr 0.335 (all must be > 0).

### `rb1s`

- best params: `{'base': 'mean_reversion', 'lookback': 40, 'z_window': 40, 'vol_window': 20, 'regime': 'low', 'vol_quantile': 0.7, 'q_window': 120, 'deadband': 0.0}`
- post-embargo val n_eff: 13 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 1.000 | 0.045 | 0.477 | 0.000 | 0.523 |
  | val | 126 | 1.000 | 0.683 | 0.159 | 0.000 | 0.841 |

  G3 perturbation (val composite over 5 +/-1-step neighbours): min=-0.014, max=0.046, std=0.027 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa 0.006 vs cutoff 0.175, ordinal_skill 0.009 vs cutoff 0.176 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.191, skill_val 0.006 (required >= 0.095, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa 0.006, ordinal_skill 0.009, increment_corr 0.024 (all must be > 0).

### `gc1s`

- best params: `{'base': 'mean_reversion', 'lookback': 20, 'z_window': 20, 'vol_window': 40, 'regime': 'high', 'vol_quantile': 0.7, 'q_window': 120, 'deadband': 0.5}`
- post-embargo val n_eff: 11 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 0.313 | 0.154 | 0.080 | 0.687 | 0.233 |
  | val | 126 | 0.159 | 0.127 | 0.016 | 0.841 | 0.143 |

  G3 perturbation (val composite over 7 +/-1-step neighbours): min=-0.016, max=0.529, std=0.166 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa 0.478 vs cutoff 0.078, ordinal_skill 0.480 vs cutoff 0.078 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.102, skill_val 0.478 (required >= 0.051, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa 0.478, ordinal_skill 0.480, increment_corr 0.695 (all must be > 0).

### `si1s`

- best params: `{'base': 'mean_reversion', 'lookback': 20, 'z_window': 40, 'vol_window': 20, 'regime': 'high', 'vol_quantile': 0.3, 'q_window': 120, 'deadband': 0.5}`
- post-embargo val n_eff: 19 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 0.902 | -0.016 | 0.459 | 0.098 | 0.443 |
  | val | 126 | 0.960 | 0.230 | 0.365 | 0.040 | 0.595 |

  G3 perturbation (val composite over 7 +/-1-step neighbours): min=0.210, max=0.306, std=0.033 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa 0.237 vs cutoff 0.098, ordinal_skill 0.367 vs cutoff 0.108 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.122, skill_val 0.237 (required >= 0.061, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa 0.237, ordinal_skill 0.367, increment_corr 0.542 (all must be > 0).

### `hg1s`

- best params: `{'base': 'mean_reversion', 'lookback': 40, 'z_window': 40, 'vol_window': 40, 'regime': 'high', 'vol_quantile': 0.3, 'q_window': 120, 'deadband': 0.0}`
- post-embargo val n_eff: 29 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 1.000 | -0.088 | 0.544 | 0.000 | 0.456 |
  | val | 126 | 1.000 | -0.111 | 0.556 | 0.000 | 0.444 |

  G3 perturbation (val composite over 5 +/-1-step neighbours): min=-0.116, max=0.119, std=0.097 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa -0.021 vs cutoff 0.078, ordinal_skill -0.024 vs cutoff 0.079 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.206, skill_val -0.021 (required >= 0.103, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa -0.021, ordinal_skill -0.024, increment_corr 0.030 (all must be > 0).

### `pl1s`

- best params: `{'base': 'mean_reversion', 'lookback': 40, 'z_window': 40, 'vol_window': 40, 'regime': 'low', 'vol_quantile': 0.7, 'q_window': 120, 'deadband': 0.0}`
- post-embargo val n_eff: 26 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 0.899 | 0.316 | 0.292 | 0.101 | 0.607 |
  | val | 126 | 0.897 | 0.722 | 0.087 | 0.103 | 0.810 |

  G3 perturbation (val composite over 5 +/-1-step neighbours): min=0.045, max=0.079, std=0.013 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa 0.108 vs cutoff 0.078, ordinal_skill 0.075 vs cutoff 0.078 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.227, skill_val 0.108 (required >= 0.114, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa 0.108, ordinal_skill 0.075, increment_corr 0.227 (all must be > 0).

### `pool:energy`

- best params: `{'base': 'ts_momentum', 'lookback': 20, 'z_window': 40, 'vol_window': 20, 'regime': 'high', 'vol_quantile': 0.7, 'q_window': 120, 'deadband': 0.0}`
- post-embargo val n_eff: 20 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 1131 | 0.257 | 0.148 | 0.055 | 0.743 | 0.202 |
  | val | 378 | 0.452 | 0.241 | 0.106 | 0.548 | 0.347 |

  G3 perturbation (val composite over 6 +/-1-step neighbours): min=0.065, max=0.193, std=0.048 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa 0.182 vs cutoff 0.082, ordinal_skill 0.127 vs cutoff 0.083 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.074, skill_val 0.182 (required >= 0.037, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa 0.182, ordinal_skill 0.127, increment_corr -0.147 (all must be > 0).

