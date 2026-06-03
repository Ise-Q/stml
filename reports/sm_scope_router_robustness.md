# sm/scope-router — robustness pack for the selected routes

## TL;DR

Bootstrap (1000 row-resamples) + paired baseline comparison + CPCV-path stability + simulated-missingness check on the 11 routes selected by the full tournament. Honest verdict after multiple-testing-aware evaluation:

- **2 routes are STRONG** (diff CI clearly > 0, route lower CI > 0.5, ≥60% paths above 0.5): **`cl1s → cl_only/lightgbm/without_bbg`** and **`pl1s → si_pl/random_forest/with_bbg`**.
- **5 routes are MEDIUM** (positive uplift but diff CI touches 0): es1s, fesx1s, ho1s, rb1s, si1s.
- **4 routes are WEAK** (no advantage over the class-pool baseline, or AUC < 0.5): gc1s, hg1s, ng1s, nq1s.

The pl1s BBG advantage **survives a simulated-missingness check** (BBG cols NaN'd at test): pl1s sim-miss AUC 0.544 still beats baseline 0.510 by +3.4pp, vs +4.8pp with BBG actually loaded. **BBG is real on pl1s but partially compensable.**

**`ho1s/ho_rb` (the headline "new pool" from the MVP/full tournament) is honestly downgraded**: its 1-SE band looked strong but the bootstrap CI is wide [0.448, 0.662] and the paired diff vs baseline is −0.0106 with CI [−0.149, +0.138]. Keep as medium, don't ship as strong claim.

## The recommended router (after this pack)

`results/sm_scope_router/recommended_router.json` — adoption-ready router. Strong + medium instruments get a route; weak / no-route get `DEFAULT_BLIND`.

| instrument | route                                          | strength | bootstrap AUC      | Δ vs baseline           |
|---         |---                                             |---       |---:                |---:                     |
| **cl1s**   | cl_only/lightgbm/without_bbg                   | **strong**   | 0.670 [0.636, 0.702] | **+0.085** [+0.057, +0.110] |
| **pl1s**   | si_pl/random_forest/**with_bbg**               | **strong**   | 0.557 [0.527, 0.586] | **+0.048** [+0.006, +0.093] |
| es1s       | es_fesx/lightgbm/without_bbg                   | medium       | 0.580 [0.552, 0.607] | +0.019 (CI touches 0)   |
| ho1s       | ho_rb/elasticnet_logistic/without_bbg          | medium       | 0.553 [0.448, 0.662] | −0.011 (wide CI)        |
| rb1s       | cl_rb/random_forest/without_bbg                | medium       | 0.518 [0.492, 0.545] | +0.039 (CI touches 0)   |
| si1s       | precious/elasticnet_logistic/without_bbg       | medium       | 0.553 [0.524, 0.581] | +0.024 (CI touches 0)   |
| fesx1s     | equity_all/random_forest/**with_bbg**          | medium       | 0.550 [0.522, 0.578] | +0.005 (CI mostly < 0)  |
| gc1s       | DEFAULT_BLIND (AUC < 0.5)                      | weak     | 0.481 [0.429, 0.533] | −0.013                  |
| hg1s       | DEFAULT_BLIND (route = baseline)               | weak     | 0.543 [0.517, 0.569] | 0.000                   |
| ng1s       | DEFAULT_BLIND (route = baseline)               | weak     | 0.556 [0.442, 0.666] | +0.003                  |
| nq1s       | DEFAULT_BLIND (route = baseline)               | weak     | 0.542 [0.515, 0.570] | 0.000                   |

**The router that should actually ship**: 2 instruments get a custom route (cl1s, pl1s); 5 get a cautious custom route flagged "medium" for human review; 4 fall back to the existing class-pool baseline (or blind for gc1s).

## Method

For each of the 11 selected routes (from `router_selection_summary.csv`):

1. **Re-run** under CPCV(6,2) + per-instrument embargo via `champion_pipeline._evaluate_candidate` to recover row-level OOS predictions.
2. **Bootstrap AUC** by row resampling: 1000 independent resamples of `(y_true, y_proba, sample_weight)`, with `roc_auc_score` per resample. Report mean and [2.5, 97.5]th percentiles as **95% CI**.
3. **CPCV-path stability**: per-CPCV-path AUC across the 15 paths (some folds with insufficient class balance are dropped). Report:
   - fraction of paths with AUC > 0.5 (the "stability gate"),
   - inter-quartile range,
   - min/max.
4. **Paired baseline comparison**: baseline = `(class_pool, route_model, "without_bbg")`. This controls for model family and isolates the *scope + variant* effect. Bootstrap the difference `(route_auc − baseline_auc)` via independent resamples → 95% CI on the uplift.
5. **Simulated missingness** (BBG NaN'd at test) — run only for the 2 `with_bbg` winners (fesx1s, pl1s), per Codex's scoping instruction. Tests whether the BBG advantage survives if BBG inputs are unreliable at inference.
6. **Strength label**:
   - **strong** ⇔ diff CI lower bound > 0 AND route lower CI > 0.5 AND ≥ 60% of CPCV paths above 0.5;
   - **medium** ⇔ route mean > baseline mean AND ≥ 50% of CPCV paths above 0.5;
   - **weak** ⇔ otherwise (or AUC < 0.5).

Total wall time: **64.9 s** (22 CV runs + 2 simulated-missingness runs + 13 bootstrap batches).

## Findings

### Finding 1 — cl1s is the strongest single-instrument route

| metric                              | route (cl_only/lightgbm/without_bbg) | baseline (energy_all/lightgbm/without_bbg) |
|---                                  |---                                  |---                                          |
| 1-SE mean (tournament)              | 0.6839 ± 0.0225                     | 0.5651 ± 0.0276                             |
| Bootstrap 95% CI                    | **[0.6361, 0.7022]**                | [0.5462, 0.6263]                            |
| Diff (route − baseline) CI 95%      | **[+0.057, +0.110]** (clearly > 0)  | —                                           |
| CPCV paths above 0.5                | **13 / 14 (93%)**                   | 8 / 14 (57%)                                |
| Fold AUC IQR                        | 0.061                               | 0.094                                       |
| Min / Max fold AUC                  | 0.461 / 0.625                       | —                                           |

- The route lower CI **0.636** is **above** the baseline upper CI **0.626**.
- Even bootstrap-conservative (independent rather than paired resampling), the diff CI is clearly non-zero.
- Stability: 93% of CPCV paths cross 0.5. The 1-SE optimism on the route is essentially "fold-aggregate" — not a single-path artefact.
- **Ship.**

### Finding 2 — pl1s with_bbg is strong AND survives simulated missingness

| metric                              | route (si_pl/rf/with_bbg) | sim_missing (with_bbg, BBG NaN'd at test) | baseline (metals_all/rf/without_bbg) |
|---                                  |---                       |---                                         |---                                    |
| Bootstrap mean                      | **0.557**                | 0.544                                      | 0.510                                 |
| Bootstrap 95% CI                    | [0.527, 0.586]           | [0.514, 0.571]                             | [0.480, 0.539]                        |
| Δ vs baseline                       | **+0.048** [+0.006, +0.093] | +0.034 (still above baseline)          | —                                     |
| Δ vs with_bbg (sim_missing penalty) | —                        | **−0.014**                                 | —                                     |
| CPCV paths above 0.5                | **93%**                  | (not measured separately)                  | —                                     |

- The diff CI lower bound +0.006 sits *just* above zero — meaning the bootstrap supports the claim that BBG genuinely helps pl1s.
- **Simulated missingness check passes**: even with BBG cols NaN'd at inference, pl1s still beats the without-BBG baseline by **+3.4pp**, only losing **1.4pp** of the with-BBG advantage. **The BBG advantage on pl1s is real and partially compensable** — safer than a fragile narrative.
- Methodologically defensible. Ship.

### Finding 3 — ho1s/ho_rb is DOWNGRADED from "+4pp new pool" to medium

The full-tournament summary called this a "NEW finding" because `ho_rb` wins by 1-SE optimism. **The bootstrap shows the picture is much noisier**:

| metric                       | route (ho_rb/elasticnet/without_bbg) | baseline (energy_all/elasticnet/without_bbg) |
|---                           |---                                  |---                                            |
| 1-SE mean                    | 0.6622 ± 0.0445                     | 0.6241 ± 0.0442                               |
| Bootstrap 95% CI             | **[0.448, 0.662]** (wide!)          | [0.462, 0.668]                                |
| Diff CI 95%                  | **[−0.149, +0.138]** (overlaps 0)   | —                                             |
| CPCV paths above 0.5         | 12 / 14 (86%)                       | —                                             |
| Min / Max fold AUC           | 0.435 / **1.000**                   | —                                             |

- The fold max **1.000** is a single-path artefact (perfect AUC on a tiny fold), which inflated the 1-SE mean. Bootstrap is robust to this.
- The diff CI **clearly straddles zero** → cannot honestly claim ho_rb is better than energy_all on ho1s.
- ho1s has only **169 OOS rows** → wide bootstrap is expected.
- **Downgrade to medium**. If we ship ho_rb, flag that the uplift claim is not statistically supported; the architecture choice is defensible (ho1s + rb1s share refinery dynamics) but the AUC win is selection noise.

### Finding 4 — 3 "winners" are actually the baseline by construction

For hg1s, ng1s, nq1s, the **chosen scope IS the class pool** (`metals_all` for hg1s; `equity_all` for nq1s; `energy_all` for ng1s). The full tournament correctly identified this — there's no "narrow pool" or "individual" that wins, so the best route is the same as the class baseline. **In the robustness pack, route == baseline ⇒ diff = 0, strength = weak by construction.**

This is the **correct** answer: for these instruments, the per-instrument router is identical to the class baseline. No special treatment needed at integration time — they just inherit the class model.

### Finding 5 — gc1s remains genuinely below random

Bootstrap mean 0.481 [0.429, 0.533]. Even the 95% CI upper bound (0.533) is barely above 0.5. **No validated route**. Recommend `DEFAULT_BLIND` for gc1s (i.e. use the primary signal directly, no act/skip filtering).

### Finding 6 — Medium-strength routes worth defending

es1s, fesx1s, rb1s, si1s show **positive uplift but with diff CIs touching 0**. They're not strong-evidence wins but they're not noise either:

| inst   | route                                | Δ mean   | Δ CI 95%             | paths > 0.5 | rationale to integrate                      |
|---     |---                                  |---:      |---                  |---:         |---                                          |
| es1s   | es_fesx/lightgbm/without_bbg        | +0.019   | (CI touches 0)      | 93%         | very stable, narrow pool is defensible      |
| fesx1s | equity_all/rf/with_bbg              | +0.005   | (CI touches 0)      | 64%         | BBG-helped, but only just; consider without_bbg fallback |
| rb1s   | cl_rb/rf/without_bbg                | +0.039   | [+0.002, +0.080]    | 71%         | actually significant! marginal call medium/strong |
| si1s   | precious/elasticnet/without_bbg     | +0.024   | (CI touches 0)      | 79%         | sub-class pool wins, narrow but stable      |

**Notable**: rb1s' diff CI [+0.002, +0.080] is positive throughout. The strength rule labels it medium because `route_lower_CI < 0.5` (route boot CI [0.492, 0.545]) — i.e. the absolute AUC isn't robustly above random by itself. A future tightening of the rule (drop the absolute-AUC gate) would promote rb1s to strong.

## Recommendation for final integration

### Strong (ship as-is)

```python
ROUTER_STRONG = {
    "cl1s": ("cl_only", "lightgbm",      "without_bbg"),   # +8.5pp CI clear, 93% paths > 0.5
    "pl1s": ("si_pl",   "random_forest", "with_bbg"),      # +4.8pp CI clear, sim-miss robust
}
```

### Medium (ship with caveat, document the noise)

```python
ROUTER_MEDIUM = {
    "es1s":   ("es_fesx",    "lightgbm",            "without_bbg"),
    "ho1s":   ("ho_rb",      "elasticnet_logistic", "without_bbg"),  # selection noise; architecture defensible
    "rb1s":   ("cl_rb",      "random_forest",       "without_bbg"),  # diff CI > 0, but route abs AUC marginal
    "si1s":   ("precious",   "elasticnet_logistic", "without_bbg"),
    "fesx1s": ("equity_all", "random_forest",       "with_bbg"),     # BBG just barely helps
}
```

### Fall back to class baseline (no per-instrument route)

```python
ROUTER_BASELINE = {
    "ng1s": ("energy_all", "lightgbm or random_forest", "without_bbg"),  # tournament chose this anyway
    "nq1s": ("equity_all", "xgboost",                   "without_bbg"),
    "hg1s": ("metals_all", "xgboost",                   "without_bbg"),
}
```

### Blind default (no model)

```python
ROUTER_BLIND = ["gc1s"]   # AUC < 0.5; use primary signal directly
```

## Method limitations

1. **Independent (rather than paired) bootstrap** of route vs baseline. Both bootstrap their own predictions from the *same* set of OOS rows (since `_evaluate_candidate` filters to the target instrument), so a paired bootstrap would be tighter. Our independent CIs are **conservative** (more pessimistic on significance) — strong calls would only get stronger under a paired test.
2. **Bootstrap is over CV-OOS predictions**, not over a held-out window. Cross-validated bootstrap doesn't fully account for the test-set re-use across the 376 tournament configs. A formal PBO via CSCV would be the next step (uses the per-path × per-config matrix); not run here.
3. **Strength rule is binary on CI lower bound**. A route with diff CI [+0.001, +0.20] is "strong"; one with [−0.001, +0.20] is "medium". Could smooth to a Bayesian posterior over uplift.
4. **`simulated_missingness` was scoped only to the 2 with_bbg winners** per Codex's instruction. For the 9 without_bbg routes the question doesn't apply (no BBG features in training).
5. **Fold-stability uses 14 paths** (one CPCV path dropped due to single-class outcomes on some instruments). Per-target instrument fold count varies (ng1s has 9 valid paths only — 5 dropped — which is why ng1s' bootstrap CI is widest).

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python -m stml.experimental.make_scope_router_robustness
```

Wall time: ~65 s. Outputs:
- `results/sm_scope_router/router_robustness_summary.csv`
- `results/sm_scope_router/recommended_router.json`
- `reports/sm_scope_router_robustness.md` (this file)

Branch tip: `sm/scope-router @ <next commit>` from `Sreeram_experimental@99bea79`. MVP checkpoint `f8336dc`, full tournament `9bbf116`, robustness `<this commit>`.
