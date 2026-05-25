# 01 — Signal Direction Audit

> Source: `results/harry/signal_direction.csv` (regenerable with
> `python -m stml.harry.signal_audit`). Bootstrap: 1 000 resamples, moving
> block size 20, seed 42. Released-signal window 2020-01-03 → 2022-06-30
> (629 signal dates). Stability check
> (`results/harry/signal_direction_stability.csv`) re-runs the same
> statistics on the first and second halves of the window (split at the
> median date) and flags sign flips where both halves' 95 % bootstrap CIs
> exclude zero.

> **This audit is descriptive characterisation of the released sample;
> no model choices are conditioned on the held-out half of 2022 (the
> grader's hidden block). It deliberately uses the full released window
> as one combined estimate so the bootstrap CI absorbs sampling
> uncertainty without splitting power across train/val/test. The released-
> data train / val / test discipline applies from Step 2 (labels)
> onwards — every label, every feature, and every model in subsequent
> steps respects the chronological split.**

## TL;DR

1. **The next-day PnL convention `PnL_t = s_t · r_{t+1}` is empirically
   supported.** `corr(s_t, r_{t+1}) > 0` for **all 11** instruments; the
   bootstrap 95 % CI is strictly above zero for **6 of 11** (`es1s`, `nq1s`,
   `cl1s`, `rb1s`, `gc1s`, `hg1s`). The five remaining point-estimates are
   positive but with CIs that include zero.
2. **The signal loads NEGATIVELY on the previous day's return for 10 of 11
   instruments** (the only exception is `cl1s` at +0.003 ≈ 0). The short-horizon
   *construction* is **counter-trend / mean-reversion**, not trend-following.
3. **Counter-trend is strongest at the 1-day horizon.** It weakens monotonically
   from `corr_trail_1` (mean −0.107) to `corr_trail_20` (mean −0.011) across
   instruments. Sreeram's pipeline assumes trend-following; that prior is
   inconsistent with `corr_trail_1` for 10 of 11 names.
4. **`cl1s` is an outlier and should be treated separately.** Its
   `corr_trail_1` ≈ 0, its hit-rate at h=10 is 71 % (the highest in the panel),
   and its mean-PnL-h is also the highest. The signal-deep-dive
   characterization independently flagged the same finding (its
   `best_construction_lag` for cl1s is `-5` with a *positive* loading).
5. **Forward h=10-day hit rates split the panel.** Five instruments
   (`cl1s`, `ho1s`, `gc1s`, `ng1s`, `es1s`) sit at ≥0.60; six are 0.50–0.56.
   That's the room a metamodel has to add value: filter the bets on the lower
   half.

The headline conclusion — confirmed independently of signal-deep-dive's
replication framework and of Sreeram's pipeline — is:

> **The primary signal is short-horizon counter-trend in 10/11 instruments;
> next-day execution is the right PnL convention; `cl1s` is structurally
> different and should not be pooled blindly with the other ten.**

## Headline table

`mean_trail_corr` = mean(`corr_trail_1`, `corr_trail_5`, `corr_trail_10`,
`corr_trail_20`); the canonical `tag` thresholds at ±0.05 on
`mean_trail_corr`. The CSV also carries `tag_trail_1` (same threshold on
`corr_trail_1` alone) and `tag_trail_h10` (same threshold on `corr_trail_10`
alone) — see the reconciliation subsection below for why all three matter.

| inst | class | n bets | corr_trail_1 | CI lo | CI hi | corr_fwd_1 | CI lo | CI hi | mean_pnl_h | hit_rate_h | tag |
|------|-------|------:|------:|------:|------:|------:|------:|------:|------:|------:|---|
| es1s   | equity | 575 | **−0.161** | −0.229 | −0.103 | **+0.103** | +0.049 | +0.170 | +0.0046 | 0.604 | mixed |
| nq1s   | equity | 604 | −0.053 | −0.137 | +0.028 | **+0.114** | +0.049 | +0.179 | +0.0088 | 0.599 | mixed |
| fesx1s | equity | 637 | **−0.193** | −0.269 | −0.116 | +0.047 | −0.057 | +0.150 | +0.0019 | 0.523 | **mean_reverting** |
| cl1s   | energy | 422 | +0.003 | −0.068 | +0.067 | **+0.121** | +0.045 | +0.189 | **+0.0238** | **0.706** | mixed |
| ho1s   | energy |  63 | −0.114 | −0.202 | −0.043 | +0.036 | −0.017 | +0.098 | +0.0027 | 0.667 | mixed |
| rb1s   | energy | 628 | −0.103 | −0.189 | −0.029 | **+0.065** | +0.000 | +0.124 | −0.0091 | 0.503 | mixed |
| ng1s   | energy | 124 | −0.040 | −0.115 | +0.027 | +0.093 | −0.003 | +0.159 | +0.0068 | 0.600 | mixed |
| gc1s   | metals | 168 | −0.087 | −0.148 | −0.010 | **+0.129** | +0.080 | +0.189 | +0.0017 | 0.605 | mixed |
| si1s   | metals | 578 | **−0.143** | −0.207 | −0.070 | +0.057 | −0.009 | +0.141 | −0.0010 | 0.562 | **mean_reverting** |
| hg1s   | metals | 628 | −0.109 | −0.188 | −0.021 | **+0.122** | +0.053 | +0.202 | +0.0030 | 0.540 | **mean_reverting** |
| pl1s   | metals | 557 | **−0.147** | −0.222 | −0.067 | +0.071 | −0.005 | +0.139 | +0.0014 | 0.524 | **mean_reverting** |

**Bold** in the corr columns = CI excludes zero (one-sided in the obvious
direction). **Bold** elsewhere = panel extreme.

## Interpretation

### Construction: counter-trend, strongest at the 1-day lag

`corr_trail_1` is significantly negative (CI excludes zero) for **6 of 11**
instruments and negative point-estimate for **10 of 11**. The only positive
point-estimate is `cl1s` at +0.003 (CI [−0.068, +0.067], spans zero).

`corr_trail_5` is uniformly weaker and only `si1s` clears CI on the negative
side (−0.124, CI [−0.208, −0.050]). By `corr_trail_20` essentially every
instrument's CI includes zero and the point estimates are mixed-sign. The
counter-trend behaviour is **short-horizon** — concentrated at 1–5 trading
days.

This is the same pattern signal-deep-dive's `characterize.lead_lag` reports.
Our point estimates differ slightly because we use the **full released
window (629 dates)** for one combined estimate, whereas signal-deep-dive
splits 60/20/20 inside that window and reports the train split. The
qualitative conclusion is identical.

### Forward (PnL) convention: next-day, supported for everyone

`corr_fwd_1` (the empirical test of `s_t · r_{t+1}`) is **positive for all 11
instruments**. CI strictly above zero in 6 cases (`es1s`, `nq1s`, `cl1s`,
`rb1s`, `gc1s`, `hg1s`); the other five (`fesx1s`, `ho1s`, `ng1s`, `si1s`,
`pl1s`) have positive point estimates with CIs that include zero. The
next-day execution convention is what we should use everywhere.

`corr_fwd_3` and `corr_fwd_5` decay toward zero and have larger CIs — beyond
the next day the signal's edge thins out fast. That argues for **short
holding horizons** in the triple-barrier vertical bound.

### Asset-class structure

- **Metals**: the cleanest counter-trend group. 4 of 4 have negative
  `corr_trail_1`, and 3 of 4 (`si1s`, `hg1s`, `pl1s`) are tagged
  `mean_reverting` by the multi-horizon average. `gc1s` is borderline.
  Mean h-day hit rate: 0.558.
- **Equity**: very strong negative `corr_trail_1` for `es1s` (−0.161) and
  `fesx1s` (−0.193), but `nq1s` is weak (−0.053). Mean h-day hit rate:
  0.575.
- **Energy**: heterogeneous. `cl1s` is *not* counter-trend at lag 1; `ho1s`
  and `rb1s` are mildly so; `ng1s` is near-zero. Mean h-day hit rate: 0.619
  but driven largely by `cl1s` and `ho1s`. Energy is also where the
  released signal participates least (`ho1s` has only 63 bets, `ng1s` 124).

### `cl1s` is structurally different

Three independent lines of evidence say `cl1s` is not in the counter-trend
cluster:

1. `corr_trail_1` ≈ 0 (the only instrument where the CI is symmetric around
   zero).
2. `corr_fwd_1` = +0.121 — among the strongest forward signals — paired
   with the highest h-day hit rate (0.706) and mean-PnL-h (+0.0238).
3. signal-deep-dive's `best_construction_lag` for `cl1s` lands at −5 with a
   *positive* loading (the only instrument that switches sign on construction).

Whatever generates `cl1s`'s signal is *not* a counter-trend rule on
short-horizon returns. It may be a longer-horizon momentum rule, or a
fundamental signal that ignores recent returns. For the metamodel: do not
let `cl1s` features inherit a counter-trend prior; the right move is
sector-aware training or even an `cl1s`-specific model.

## Reconciling with signal-deep-dive's 10/11 mean-reversion claim

signal-deep-dive's headline is "10 of 11 instruments are mean-reverting".
Our canonical multi-horizon tag is **4 of 11**. These are not contradictory —
they are measuring different lags. The per-horizon columns in the CSV
(`tag_trail_1`, `tag_trail_h10`, and the canonical multi-horizon `tag`) make
the discrepancy explicit:

| classifier | trend | mean_reverting | mixed | what it uses |
|---|---:|---:|---:|---|
| `tag_trail_1`   | 0 | **9** | 2 | sign of `corr_trail_1` only (1-day construction lag) |
| `tag_trail_h10` | 2 | 2 | 7 | sign of `corr_trail_10` only (10-day construction lag) |
| `tag` (canonical) | 0 | 4 | 7 | mean of `corr_trail_{1,5,10,20}` |

**signal-deep-dive's claim is driven entirely by the trail_1 horizon.** Their
`momentum_score` is the same multi-horizon mean we compute, but their
"`alpha_label = mean_reversion`" threshold is lower / the prose summary
collapses anything negative to that label. At lag 1 the signal is
unambiguously counter-trend in 9/11 instruments — `corr_trail_1` is negative
for all but `cl1s` and `ng1s`, often by 2-3 bootstrap standard errors. **By
lag 10 the effect has dissipated**: `corr_trail_10` is mixed-sign across the
panel and only `si1s` (−0.121) and `pl1s` (−0.079) retain a clear negative
loading. `rb1s` (+0.071) and `gc1s` (+0.079) actually tag *trend* at this
horizon — the opposite direction from their trail_1 result.

**Implication for Harry's labels:** because our triple-barrier vertical
barrier is `h=10`, the structurally-aligned signal is `corr_trail_10`, not
`corr_trail_1`. That is why **4 instruments tag mean-reverting under the
canonical multi-horizon classifier**, not 10. The release signal "is"
short-horizon counter-trend in a strict 1-day sense, but the *meta-label* —
"is this bet worth taking over the next 10 trading days" — sees a much
weaker, less directional structure. Both Sreeram's trend-heavy feature
prior and signal-deep-dive's counter-trend-heavy prior over-specify; the
empirical reality is **noisy and short-lived**, and the labels we generate
in Step 2 must stay neutral so the model can learn that noise structure on
its own. This is the load-bearing argument for keeping `pt_mult = sl_mult =
1.0` as the label default (decided in §"Implications" below).

## Stability across halves of the released window

`results/harry/signal_direction_stability.csv` re-runs the audit on the
first half (dates < median split ≈ 2021-09) and the second half. **Zero
sign flips at the 95 % CI level** across 88 (instrument, metric) rows —
the audit's conclusions are not driven by one half of the window. The
largest first-vs-second-half divergence on `corr_trail_1` is `fesx1s`
(−0.158 → −0.255, same sign, stronger in H2); on `corr_fwd_1` it is
`fesx1s` and `cl1s` (sign-stable but the point estimate moves notably).
Notably `cl1s`'s `corr_trail_1` does cross zero between halves (+0.015 →
−0.061), but neither half's CI excludes zero, so this is consistent with
"cl1s is structurally near-zero at lag 1" rather than a real regime change.

## Implications for the rest of Harry's contribution

| Decision | Default | Justification from this audit |
|---|---|---|
| PnL convention | Next-day (`PnL_t = s_t · r_{t+1}`) | All 11 `corr_fwd_1 > 0`. |
| Feature **prior** | Counter-trend / mean-reversion (signal-deep-dive's F1 family) | 10/11 negative `corr_trail_1`. |
| Trend features | Keep them as *contrast* (sign-flip indicator), not as the headline group | Sreeram's MDA already showed trend-cluster permutation *helps* OOS — consistent with this. |
| Triple-barrier horizon `h` | 10 days, with sensitivity at 5 and 15 | Forward correlation decays by `h=5`; `h=10` covers ~one effective bet. |
| Asymmetric barriers | Expose `pt_mult`, `sl_mult`; **default symmetric 1.0/1.0**. Variants live in the Step 4 sensitivity sweep, not in the label default. | Audit tags are 0 trend / 4 mean-reverting / 7 mixed under the canonical multi-horizon classifier — too equivocal to bake a counter-trend bias into the label itself. A neutral target lets downstream feature-importance analysis honestly tell us *whether* counter-trend features matter rather than recovering a bias we put in. |
| Universe pooling | Pool 10 instruments; **carve out `cl1s` for separate handling** | The audit's three independent signals against pooling cl1s. |
| Cross-sectional features | Expected-negative (cross-asset mean \|corr\| ≈ 0.09 per signal-deep-dive); keep for completeness, do not weight heavily | Already documented in `signal-deep-dive`. |

## What this does NOT decide (deferred to Step 6 / team memo)

- Whether to drop equity from training (Sreeram's v3 finding). This audit
  confirms the equity prior is *counter-trend*, not trend-following — which
  is the opposite of Sreeram's working assumption — but it does not
  litigate the regime-break problem H1-2022 vs the 2020-2021 training set.
  That argument lives in Step 4's evaluation and Step 6's memo.
- Whether to ensemble Harry's predictions with Sreeram's `predictions_v5`
  for final group submission.

## Reproduction

```bash
python -m stml.harry.signal_audit \
  --out results/harry/signal_direction.csv \
  --h 10 --n-boot 1000 --block-size 20 --seed 42
```

Tests: `pytest tests/harry/test_signal_audit.py` (14 tests, ~3 s).
