# Significance + deflation + directional skill — H1-2022 OOS

Source data: `results/submission/strategy_daily_net_returns.csv` (n = 179 periods).

## §3.8 PRIMARY — Sharpe significance

| Statistic | Value | Reading |
|---|---:|---|
| n (periods) | 179 | OOS sample size |
| Per-period Sharpe SR | 0.2015 | uncorrected for n |
| Annualised Sharpe (×√252) | 3.1995 | use only if Ljung-Box OK |
| **t = SR·√n** | **2.6965** | distinguishable from 0 at 5% |
| Studentised stationary block-bootstrap 95% CI (PRIMARY) | [0.0658, 0.3377] per period | EXCLUDES 0 → significant |
| Bootstrap block length (Politis-White) | 4.88 | data-driven |
| Lo/Opdyke analytic 95% CI | [0.0622, 0.3409] per period | parametric cross-check |
| PSR(SR* = 0) | 0.9977 | > 0.95 deployment threshold |
| MinTRL (95% PSR) | 61 periods | < n → certified |
| Ljung-Box Q(10) | 7.069, p = 0.7189 | OK — IID-like, √252 annualisation valid |

## §3.8 deflation ladder

| Rung | n_trials | DSR |
|---|---:|---:|
| N_eff | 2 | 0.9964 |
| N_raw | 120 | 0.9824 |
| 2·N_raw | 240 | 0.9794 |
| 4·N_raw | 480 | 0.9762 |

* CSCV-PBO combinations at n_blocks=16: **12,870** (corrects the long-propagated 12,780 typo).
* MinBTL @ target Sharpe = ann Sharpe: 0.7 periods.
* Expected max Sharpe of N_eff = 2: 0.0104.
* Expected max Sharpe of N_raw = 120: 0.0519.

## §3.8 directional skill

| Test | Value | Reading |
|---|---:|---|
| **Pesaran-Timmermann (PRIMARY)** | S = 3.4844, p = 0.0002 | positive directional skill |
| Treynor-Mazuy γ | 0.0033 (t = 1.05) | no significant convexity |
| Henriksson-Merton hit rate (proxy) | 0.5499 (z = 3.66, p = 0.0001) | BASE-RATE SENSITIVE — read with caveat |

## Five-lens verdict

Lens 1 — AUC (per class, §8 S3): equity 0.554 | energy 0.602 | metals 0.554.
Lens 2 — cluster MDA (§8 S5): equity 0.022 PASS | energy 0.036 PASS | metals 0.016 CHECK.
Lens 3 — Sharpe significance (this section): per-period bootstrap CI EXCLUDES 0, t = 2.70, PSR(0) = 1.00.
Lens 4 — deflation: DSR at N_eff = 0.996, at 4·N_raw = 0.976.
Lens 5 — Pesaran-Timmermann: S = 3.48, p = 0.00.