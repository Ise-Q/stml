# Significance + deflation + directional skill — H1-2022 OOS

Source data: `results/submission/strategy_daily_net_returns.csv` (n = 129 periods).

## §3.8 PRIMARY — Sharpe significance

| Statistic | Value | Reading |
|---|---:|---|
| n (periods) | 129 | OOS sample size |
| Per-period Sharpe SR | 0.1982 | uncorrected for n |
| Annualised Sharpe (×√252) | 3.1466 | use only if Ljung-Box OK |
| **t = SR·√n** | **2.2513** | distinguishable from 0 at 5% |
| Studentised stationary block-bootstrap 95% CI (PRIMARY) | [0.0218, 0.3560] per period | EXCLUDES 0 → significant |
| Bootstrap block length (Politis-White) | 4.47 | data-driven |
| Lo/Opdyke analytic 95% CI | [0.0277, 0.3688] per period | parametric cross-check |
| PSR(SR* = 0) | 0.9886 | > 0.95 deployment threshold |
| MinTRL (95% PSR) | 68 periods | < n → certified |
| Ljung-Box Q(10) | 13.273, p = 0.2088 | OK — IID-like, √252 annualisation valid |

## §3.8 deflation ladder

| Rung | n_trials | DSR |
|---|---:|---:|
| N_eff | 2 | 0.9846 |
| N_raw | 120 | 0.9537 |
| 2·N_raw | 240 | 0.9483 |
| 4·N_raw | 480 | 0.9429 |

* CSCV-PBO combinations at n_blocks=16: **12,870** (corrects the long-propagated 12,780 typo).
* MinBTL @ target Sharpe = ann Sharpe: 0.7 periods.
* Expected max Sharpe of N_eff = 2: 0.0104.
* Expected max Sharpe of N_raw = 120: 0.0519.

## §3.8 directional skill

| Test | Value | Reading |
|---|---:|---|
| **Pesaran-Timmermann (PRIMARY)** | S = 4.5987, p = 0.0000 | positive directional skill |
| Treynor-Mazuy γ | -0.2530 (t = -4.05) | convex timing; check S5.12 scale-aggregation |
| Henriksson-Merton hit rate (proxy) | 0.5710 (z = 4.38, p = 0.0000) | BASE-RATE SENSITIVE — read with caveat |

## Five-lens verdict

Lens 1 — AUC (per class, §8 S3): equity 0.554 | energy 0.602 | metals 0.554.
Lens 2 — cluster MDA (§8 S5): equity 0.022 PASS | energy 0.036 PASS | metals 0.016 CHECK.
Lens 3 — Sharpe significance (this section): per-period bootstrap CI EXCLUDES 0, t = 2.25, PSR(0) = 0.99.
Lens 4 — deflation: DSR at N_eff = 0.985, at 4·N_raw = 0.943.
Lens 5 — Pesaran-Timmermann: S = 4.60, p = 0.00.