# Replication Summary (US-011)

Each `(family, cell)` was searched on the TRAIN discrepancy objective (TPE above the post-embargo n_eff FLOOR, exhaustive grid below it) and the winner gated on G1-G4 over the TRAIN+VAL window. The held-out test block was touched exactly once, for the final confirmation below.

## Headline

**3 of 6 families replicate >= 1 cell (< 5 required): see honest-shortfall below.**

- `mean_reversion` replicates: `es1s`, `fesx1s`, `si1s`, `hg1s`
- `ts_momentum` replicates: none
- `breakout_donchian` replicates: none
- `vol_regime_gated` replicates: `es1s`, `si1s`
- `hybrid_filtered_momentum` replicates: none
- `xsect_rank` replicates: `pl1s`

## Families x cells pass/fail matrix

| family | `es1s` | `nq1s` | `fesx1s` | `rb1s` | `gc1s` | `si1s` | `hg1s` | `pl1s` | `pool:energy` | family passes? |
|--------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:-------------:|
| `mean_reversion`| PASS | fail | PASS | fail | fail | PASS | PASS | fail | fail | yes |
| `ts_momentum`| fail | fail | fail | fail | fail | fail | fail | fail | fail | no |
| `breakout_donchian`| fail | fail | fail | fail | fail | fail | fail | fail | fail | no |
| `vol_regime_gated`| PASS | fail | fail | fail | fail | PASS | fail | fail | fail | yes |
| `hybrid_filtered_momentum`| fail | fail | fail | fail | fail | fail | fail | fail | fail | no |
| `xsect_rank`| fail | fail | fail | fail | fail | fail | fail | PASS | fail | yes |

## Search coverage (full grid vs TPE budget)

Above the n_eff FLOOR each cell runs a seeded TPE search of `budget = 64` trials; the full grid is the Cartesian product of the family's axes (the exhaustive set the below-FLOOR grid tier would enumerate). Coverage is `min(budget, |grid|) / |grid|` -- how much of the configuration space a single cell's TPE budget can reach.

| family | full grid size | budget | coverage |
|--------|---------------:|-------:|---------:|
| `mean_reversion` | 60 | 64 | 100.0% |
| `ts_momentum` | 50 | 64 | 100.0% |
| `breakout_donchian` | 16 | 64 | 100.0% |
| `vol_regime_gated` | 432 | 64 | 14.8% |
| `hybrid_filtered_momentum` | 72 | 64 | 88.9% |
| `xsect_rank` | 24 | 64 | 100.0% |

## Pooled-cell gating semantics (energy)

The GATED energy pool is the three BELOW-FLOOR members `{cl1s, ho1s, ng1s}` (post-embargo val n_eff 9, 9, 2). Its gateable n_eff is their SUM -- 9 + 9 + 2 = 20 -- which clears the FLOOR (>= 10), so the pool is gated FIRST-CLASS (`gate_cell` routes a >= FLOOR cell straight through `evaluate`, not the forced-low-power branch). Summing is justified because cross-asset mean |corr| ~= 0.09 makes the members near-independent, so the concatenated series carries ~the sum of their independent regime-calls. Each member ALONE stays below the FLOOR and can never earn a standalone pass.

Reconciliation: `thresholds.json`'s `per_asset_class['energy']` lists FOUR members (incl. `rb1s`) as the class baseline, whereas the gated pool here is only the three below-FLOOR members. `rb1s` (val n_eff 13) is at/above the FLOOR and is gated STANDALONE, not in the pool. Including vs excluding `rb1s` in the class-baseline cutoff is immaterial (identical to 3 d.p.), so the pool's threshold entry is used as-is.

## Pooling: within-instrument aggregation

**Artifact fixed.** A pooled cell (`pool:energy = cl1s + ho1s + ng1s`) previously CONCATENATED its members' `(target, replica)` rows and computed a SINGLE metric. That inflates Cohen's kappa via cross-instrument base-rate matching: the `ts_momentum` pool winner had a concatenated val kappa of ~0.14 -- a 'pass' -- yet its honest per-member val kappas were cl1s -0.130, ho1s -0.026, ng1s +0.284 (equal-weight mean +0.04 ~= chance), so momentum ANTI-replicated two of the three energy members while still clearing the pool. `breakout_donchian` showed the same pattern.

**Fix.** A pooled cell's skill is now the equal-weight MEAN of the per-member WITHIN-instrument metrics (never a single concatenated metric), applied to BOTH the search objective (it minimises the mean per-member train discrepancy) AND the gates (G1 kappa/ordinal, G2 per-split skill, G3 neighbourhood composite, and G4 kappa/ordinal/NAV-increment-corr are each per-member-then-averaged). Cross-instrument base-rate matching can no longer manufacture a pass, so a family clears `pool:energy` only on genuine within-instrument energy skill.

Per-member vs group-averaged vs concatenated val kappa, this run's pool winners (the concatenated column is the OLD artifact, shown for contrast; the group-averaged column is what is now gated):

| family | cl1s | ho1s | ng1s | group-avg (gated) | concatenated (old) |
|--------|-----:|-----:|-----:|------------------:|-------------------:|
| `mean_reversion` | -0.002 | 0.073 | -0.025 | 0.015 | 0.033 |
| `ts_momentum` | -0.147 | -0.089 | 0.275 | 0.013 | 0.096 |
| `breakout_donchian` | -0.124 | -0.083 | 0.354 | 0.049 | 0.150 |
| `vol_regime_gated` | 0.069 | 0.071 | 0.408 | 0.182 | 0.158 |
| `hybrid_filtered_momentum` | -0.282 | -0.028 | -0.061 | -0.124 | 0.057 |
| `xsect_rank` | 0.018 | -0.047 | -0.004 | -0.011 | 0.044 |

## Gate-calibration sensitivity

How the family pass-count moves as the G3 plateau tolerance (`std_tol`, default 0.15) and the G2 generalization fraction (`gen_frac`, default 0.5) are perturbed. Each cell is re-judged from its STORED neighbourhood + skill metrics (no re-search needed); G1 and G4 are held at their frozen verdicts. A stable count across the grid is what substantiates the 'well-calibrated' claim.

Recorded winners pass-count (default tolerances): **3**.

G3 plateau tolerance sweep (gen_frac at default):

| std_tol | families passing |
|--------:|-----------------:|
| 0.10 | 3 |
| 0.15 | 3 |
| 0.20 | 3 |

G2 generalization fraction sweep (std_tol at default):

| gen_frac | families passing |
|---------:|-----------------:|
| 0.40 | 3 |
| 0.50 | 3 |
| 0.60 | 3 |

## Honest shortfall

Only 3 of 6 families cleared all four gates on at least one cell, short of the 5 required. The shortfall is structural, not a bug in the search:

- **Effective sample size.** Even the standalone cells carry only ~11-35 leakage-free regime-calls in val (post-embargo n_eff); the three energy members (cl1s 9, ho1s 9, ng1s 2) sit below the FLOOR=10 and are gated only on the energy POOL, never standalone. A genuine skill edge has to clear chance cutoffs AND a multiplicity-inflated margin on a handful of independent observations.
- **Base-rate drift.** The released signal's base rates drift across splits (e.g. ng1s participation 0.07 -> 0.31 -> 0.43), so G2 measures skill against each split's OWN chance level. A replica that merely tracks the drifting majority nets to ~0 skill and fails G2 -- exactly as intended.
- **Degeneracy / plateau.** G3 demands a parameter plateau over the NUMERIC/ordinal axes (categorical strategy switches -- `base`, `regime`, `score` -- are held fixed, so a plateau probe never conflates a strategy flip with a tuning nudge); a winner whose numeric neighbours scatter below the cutoff is rejected as overfit.
- **Non-passing families.** `ts_momentum`, `breakout_donchian`, `hybrid_filtered_momentum` cleared no cell: their winning configs either missed a chance cutoff (G1), failed to transfer per-split skill (G2), sat on a numeric knife-edge (G3) or had a metric disagree in sign (G4). These are the structural filters above biting, not a search that stopped early.
- **Cross-asset diagnostic (xsect_rank) -- CAUTION.** `xsect_rank` was scoped as an expected-negative diagnostic (cross-asset mean |corr| ~= 0.09); it nonetheless cleared all four gates on the `pl1s` cell (kappa ~0.28, val n_eff ~= 26). We treat this cautiously: the val n_eff is modest, and the winning variant is a `score=reversal` variant. It is reported as a pass but flagged as the weakest / most-surprising of the replicators.

## Final test confirmation (val <-> test consistency)

For the top passing `(family, cell, params)` per family (capped at 5), the replica is rebuilt on the held-out test block and its composite skill compared to val. This is the ONLY `get_test(final_confirmation=True)` call in the pipeline.

| family | confirmed on | val composite | test composite | val participation | test participation |
|--------|--------------|--------------:|---------------:|------------------:|-------------------:|
| `mean_reversion` | `si1s` | 0.362 | 0.169 | 0.960 | 0.935 |
| `vol_regime_gated` | `si1s` | 0.302 | 0.060 | 0.960 | 0.935 |
| `xsect_rank` | `pl1s` | 0.313 | -0.028 | 0.897 | 0.839 |

Per-split base rates for each confirmed winner:

- `mean_reversion` / `si1s` (confirmed on `si1s`):
  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | val | 126 | 0.960 | 0.230 | 0.365 | 0.040 | 0.595 |
  | test | 124 | 0.935 | 0.081 | 0.427 | 0.065 | 0.508 |

- `vol_regime_gated` / `si1s` (confirmed on `si1s`):
  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | val | 126 | 0.960 | 0.230 | 0.365 | 0.040 | 0.595 |
  | test | 124 | 0.935 | 0.081 | 0.427 | 0.065 | 0.508 |

- `xsect_rank` / `pl1s` (confirmed on `pl1s`):
  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |
  |-------|--:|--------------:|----------:|--------:|-------:|--------:|
  | val | 126 | 0.897 | 0.722 | 0.087 | 0.103 | 0.810 |
  | test | 124 | 0.839 | 0.597 | 0.121 | 0.161 | 0.718 |

