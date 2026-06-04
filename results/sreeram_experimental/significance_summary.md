# Significance + deflation + directional skill — H1-2022 OOS

Source data: `results/sreeram_experimental/strategy_daily_net_returns.csv` (n = 180 periods).

## §3.8 PRIMARY — Sharpe significance

| Statistic | Value | Reading |
|---|---:|---|
| n (periods) | 180 | OOS sample size |
| Per-period Sharpe SR | 0.0971 | uncorrected for n |
| Annualised Sharpe (×√252) | 1.5417 | use only if Ljung-Box OK |
| **t = SR·√n** | **1.3030** | NOT significant at 5% |
| Studentised stationary block-bootstrap 95% CI (PRIMARY) | [-0.0429, 0.2063] per period | CONTAINS 0 |
| Bootstrap block length (Politis-White) | 9.78 | data-driven |
| Lo/Opdyke analytic 95% CI | [-0.0551, 0.2493] per period | parametric cross-check |
| PSR(SR* = 0) | 0.8945 | below 0.95 — insufficient |
| MinTRL (95% PSR) | 311 periods | ~1.7× too short to certify |
| Ljung-Box Q(10) | 19.480, p = 0.0346 | rejects IID — √252 OVERSTATES |

## §3.8 deflation ladder

| Rung | n_trials | DSR |
|---|---:|---:|
| N_eff | 3 | 0.7585 |
| N_raw | 120 | 0.3374 |
| 2·N_raw | 240 | 0.2850 |
| 4·N_raw | 480 | 0.2398 |

* CSCV-PBO combinations at n_blocks=16: **12,870** (corrects the long-propagated 12,780 typo).
* MinBTL @ target Sharpe = ann Sharpe: 2.8 periods.
* Expected max Sharpe of N_eff = 3: 0.0426.
* Expected max Sharpe of N_raw = 120: 0.1297.

## §3.8 directional skill

| Test | Value | Reading |
|---|---:|---|
| **Pesaran-Timmermann (PRIMARY)** | S = nan, p = nan | NO positive directional skill |
| Treynor-Mazuy γ | -0.0791 (t = -2.88) | convex timing; check S5.12 scale-aggregation |
| Henriksson-Merton hit rate (proxy) | 0.5557 (z = 4.13, p = 0.0000) | BASE-RATE SENSITIVE — read with caveat |

## Five-lens verdict

Lens 1 — AUC (per class, §8 S3): equity 0.554 | energy 0.602 | metals 0.554 (vs alken 0.579/0.525/0.530).
Lens 2 — cluster MDA (§8 S5): equity 0.022 PASS | energy 0.036 PASS | metals 0.016 CHECK.
Lens 3 — Sharpe significance (this section): per-period bootstrap CI CONTAINS 0, t = 1.30, PSR(0) = 0.89.
Lens 4 — deflation: DSR at N_eff = 0.758, at 4·N_raw = 0.240.
Lens 5 — Pesaran-Timmermann: S = nan, p = nan.