# Replication family: `xsect_rank`

Cross-sectional rank: long the top / short the bottom of the universe each day. SCOPED as an expected-negative diagnostic -- with cross-asset mean |corr| = 0.09 a single panel ranking should struggle to replicate near-independent instruments, so the 6th family exists to document the (lack of) cross-asset structure rather than to add a likely pass. Where it nonetheless clears the gates, the summary reports that as a pass and flags it as the weakest / most-surprising replicator.

## Parameter space

- `lookback`: [10, 20, 40, 60]
- `top_frac`: [0.2, 0.3, 0.4]
- `score`: ['momentum', 'reversal']

Search tier(s) used: `tpe` (TPE above the n_eff FLOOR, exhaustive grid below it).

**Family verdict:** passes G1-G4 on **1** of 9 cell(s).

## Per-cell train/val panel

| cell | tier | n_eff | n_configs | val kappa | val ordinal_skill | composite | G1 | G2 | G3 | G4 | passed |
|------|------|------:|----------:|----------:|------------------:|----------:|:--:|:--:|:--:|:--:|:------:|
| `es1s` | tpe | 35 | 64 | 0.032 | 0.083 | 0.057 | no | yes | no | no | no |
| `nq1s` | tpe | 20 | 64 | 0.078 | 0.087 | 0.083 | no | no | yes | no | no |
| `fesx1s` | tpe | 25 | 64 | -0.099 | -0.148 | -0.123 | no | no | no | no | no |
| `rb1s` | tpe | 13 | 64 | -0.013 | -0.016 | -0.014 | no | no | no | no | no |
| `gc1s` | tpe | 11 | 64 | -0.167 | -0.157 | -0.162 | no | no | no | no | no |
| `si1s` | tpe | 19 | 64 | 0.079 | 0.107 | 0.093 | no | yes | no | yes | no |
| `hg1s` | tpe | 29 | 64 | -0.024 | -0.033 | -0.029 | no | no | no | no | no |
| `pl1s` | tpe | 26 | 64 | 0.279 | 0.346 | 0.313 | yes | yes | yes | yes | yes |
| `pool:energy` | tpe | 20 | 64 | -0.011 | 0.018 | 0.003 | no | no | no | no | no |

## Per-cell detail (base rates, perturbation, gate diagnostics)

### `es1s`

- best params: `{'lookback': 40, 'top_frac': 0.4, 'score': 'reversal'}`
- post-embargo val n_eff: 35 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 378 | 0.913 | 0.511 | 0.201 | 0.087 | 0.712 |
  | val | 126 | 0.881 | 0.579 | 0.151 | 0.119 | 0.730 |

  G3 perturbation (val composite over 3 +/-1-step neighbours): min=0.016, max=0.047, std=0.012 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa 0.032 vs cutoff 0.078, ordinal_skill 0.083 vs cutoff 0.078 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.035, skill_val 0.032 (required >= 0.017, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa 0.032, ordinal_skill 0.083, increment_corr -0.353 (all must be > 0).

### `nq1s`

- best params: `{'lookback': 60, 'top_frac': 0.4, 'score': 'reversal'}`
- post-embargo val n_eff: 20 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 378 | 0.944 | 0.209 | 0.368 | 0.056 | 0.577 |
  | val | 126 | 0.984 | 0.683 | 0.151 | 0.016 | 0.833 |

  G3 perturbation (val composite over 2 +/-1-step neighbours): min=0.059, max=0.096, std=0.018 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa 0.078 vs cutoff 0.078, ordinal_skill 0.087 vs cutoff 0.078 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.196, skill_val 0.078 (required >= 0.098, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa 0.078, ordinal_skill 0.087, increment_corr -0.210 (all must be > 0).

### `fesx1s`

- best params: `{'lookback': 40, 'top_frac': 0.4, 'score': 'reversal'}`
- post-embargo val n_eff: 25 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 381 | 1.000 | -0.108 | 0.554 | 0.000 | 0.446 |
  | val | 129 | 1.000 | 0.070 | 0.465 | 0.000 | 0.535 |

  G3 perturbation (val composite over 3 +/-1-step neighbours): min=-0.057, max=0.116, std=0.072 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa -0.099 vs cutoff 0.092, ordinal_skill -0.148 vs cutoff 0.093 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.039, skill_val -0.099 (required >= 0.019, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa -0.099, ordinal_skill -0.148, increment_corr -0.117 (all must be > 0).

### `rb1s`

- best params: `{'lookback': 10, 'top_frac': 0.4, 'score': 'reversal'}`
- post-embargo val n_eff: 13 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 1.000 | 0.045 | 0.477 | 0.000 | 0.523 |
  | val | 126 | 1.000 | 0.683 | 0.159 | 0.000 | 0.841 |

  G3 perturbation (val composite over 2 +/-1-step neighbours): min=0.018, max=0.048, std=0.015 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa -0.013 vs cutoff 0.175, ordinal_skill -0.016 vs cutoff 0.176 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.200, skill_val -0.013 (required >= 0.100, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa -0.013, ordinal_skill -0.016, increment_corr 0.178 (all must be > 0).

### `gc1s`

- best params: `{'lookback': 40, 'top_frac': 0.2, 'score': 'reversal'}`
- post-embargo val n_eff: 11 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 0.313 | 0.154 | 0.080 | 0.687 | 0.233 |
  | val | 126 | 0.159 | 0.127 | 0.016 | 0.841 | 0.143 |

  G3 perturbation (val composite over 3 +/-1-step neighbours): min=-0.100, max=0.045, std=0.060 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa -0.167 vs cutoff 0.078, ordinal_skill -0.157 vs cutoff 0.078 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.201, skill_val -0.167 (required >= 0.101, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa -0.167, ordinal_skill -0.157, increment_corr -0.003 (all must be > 0).

### `si1s`

- best params: `{'lookback': 40, 'top_frac': 0.4, 'score': 'reversal'}`
- post-embargo val n_eff: 19 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 0.902 | -0.016 | 0.459 | 0.098 | 0.443 |
  | val | 126 | 0.960 | 0.230 | 0.365 | 0.040 | 0.595 |

  G3 perturbation (val composite over 3 +/-1-step neighbours): min=0.033, max=0.244, std=0.086 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa 0.079 vs cutoff 0.098, ordinal_skill 0.107 vs cutoff 0.108 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.154, skill_val 0.079 (required >= 0.077, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa 0.079, ordinal_skill 0.107, increment_corr 0.435 (all must be > 0).

### `hg1s`

- best params: `{'lookback': 20, 'top_frac': 0.4, 'score': 'reversal'}`
- post-embargo val n_eff: 29 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 1.000 | -0.088 | 0.544 | 0.000 | 0.456 |
  | val | 126 | 1.000 | -0.111 | 0.556 | 0.000 | 0.444 |

  G3 perturbation (val composite over 3 +/-1-step neighbours): min=-0.042, max=0.017, std=0.028 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa -0.024 vs cutoff 0.078, ordinal_skill -0.033 vs cutoff 0.079 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.140, skill_val -0.024 (required >= 0.070, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa -0.024, ordinal_skill -0.033, increment_corr 0.038 (all must be > 0).

### `pl1s`

- best params: `{'lookback': 20, 'top_frac': 0.4, 'score': 'reversal'}`
- post-embargo val n_eff: 26 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 377 | 0.899 | 0.316 | 0.292 | 0.101 | 0.607 |
  | val | 126 | 0.897 | 0.722 | 0.087 | 0.103 | 0.810 |

  G3 perturbation (val composite over 3 +/-1-step neighbours): min=0.143, max=0.226, std=0.034 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa 0.279 vs cutoff 0.078, ordinal_skill 0.346 vs cutoff 0.078 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.291, skill_val 0.279 (required >= 0.146, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa 0.279, ordinal_skill 0.346, increment_corr 0.691 (all must be > 0).

### `pool:energy`

- best params: `{'lookback': 60, 'top_frac': 0.2, 'score': 'reversal'}`
- post-embargo val n_eff: 20 (standalone)

  Per-split base rates (the per-split chance reference G2 respects):

  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | train | 1131 | 0.257 | 0.148 | 0.055 | 0.743 | 0.202 |
  | val | 378 | 0.452 | 0.241 | 0.106 | 0.548 | 0.347 |

  G3 perturbation (val composite over 2 +/-1-step neighbours): min=-0.038, max=0.036, std=0.037 (plateau std tol = 0.15).

  - G1 (beats baseline + multiplicity): kappa -0.011 vs cutoff 0.082, ordinal_skill 0.018 vs cutoff 0.083 (margin 0.028, n_configs 64).
  - G2 (drift-aware generalization): skill_train 0.038, skill_val -0.011 (required >= 0.019, gen_frac 0.500).
  - G4 (multi-metric consistency): kappa -0.011, ordinal_skill 0.018, increment_corr -0.013 (all must be > 0).

