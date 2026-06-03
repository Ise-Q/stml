# Experiments and decisions: how the evidence shaped the metamodel

**Module:** T3.03 — Systematic Trading Strategies with Machine Learning (Alken team challenge).
**Companion to:** the methodology set (`methodology-full.md`, `-report.md`, `-brief.md`), which carries the full bibliography.
**Stance:** methodology, not performance.

---

## Abstract

This report records the diagnostic experiments run behind the metamodel and, for each, the decision its result drove. The experiments were deliberately *firewalled* from the locked deliverable: they could justify what was shipped, confirm what was rejected, or correct an interpretation, but they were never allowed to retune the frozen labelling and sizing configuration on the out-of-sample data. Two refinements that looked attractive in point estimate — a per-instrument Kelly shrinkage and a barrier re-tuning — were declined once tested properly, and one favourable timing statistic was traced to an aggregation artefact. The experiments converge on a single conclusion, reached from five independent directions: there is insufficient evidence of a deployable act/skip edge on the provided primary signal.

## Introduction

A meta-labelling layer can only add value where the primary signal already carries filterable skill (López de Prado, 2018), and the provided primary is a short-horizon mean-reversion call whose own predictive content is slight. The design discipline that follows from this is an *adoption rule*: a refinement enters the shipped deliverable only if it clears a material, leakage-safe and significance-backed bar. Everything else is run as a diagnostic — informative about *where* and *why* the edge is thin, but firewalled from the locked configuration so that the out-of-sample window is never rehearsed. Experiment identifiers (EX.* for exploratory probes, S* for section follow-ups, X* for cross-cutting checks) are retained below as tags so each decision is traceable to its evidence. All figures are drawn from the committed experiment outputs and the deterministic emit log.

## Experiment, result and decision

| Experiment | Result | Decision it drove |
|---|---|---|
| **EX.1** edge decomposition | CPCV paths above 0.50: Equity 15/15, Metals 13/15, **Energy 6/15** (stripped roster) | read *where* the edge is real; Energy ≈ coin-flip — diagnostic framing, no config change |
| **EX.3** barrier surface | purged-CV AUC unstable across (k, T_max): **0.46–0.59**, best cell only 0.589 | **firewall held** — the locked barriers were *not* tuned to this surface |
| **EX.4 → S3.9** calibration | Platt beats isotonic in every class; selected-model ECE cut sharply (Energy 0.140→0.100, Equity 0.055→0.027, Metals 0.210→0.001), AUC unchanged | **ship per-class Platt** on both the deliverable probability and the Kelly stake |
| **EX.5** primary signal | directional hit-rates 0.51–0.69, information coefficients small (gc1s 0.66 / IC 0.21; ng1s undefined) | sets the achievable ceiling; the thin meta-edge is structural |
| **S4.7/4.8** cluster importance | top permutation accuracy 0.025 (Equity) else ≈ 0, despite large in-sample SHAP (Metals 0.71) | a high SHAP is *not* edge; permutation accuracy is the out-of-sample check |
| **S5.11/5.12** market timing | pooled Treynor–Mazuy γ +1.18 → **−0.0031** after standardise-then-repool | trust the scale-invariant Pesaran–Timmermann test, not the pooled regression |
| **S6.7** backtest | pooled barrier-exact net Sharpe **1.31** | report it, but lead with significance, not the point estimate |
| **S6.9** holding model | exit rule alone moves every book (Equity +0.54 → +0.86) | the ordering is a backtest-construction finding, not classification skill |
| **S6.14** significance | **t = 0.93** (n = 127); bootstrap 95% CI **[−0.04, +0.19]** contains zero | the **primary** strategy verdict: insufficient evidence |
| **S6.8** deflation | deflated-Sharpe ladder fails 0.95 at every trial count; PBO ≈ 0.35 | corroboration, demoted below the significance test |
| **S6.15 / EX.6** sizing | leakage-safe κᵢ shrinkage +92% in point estimate, but paired bootstrap CI contains zero | **revert to flat κ = 0.25**; byte-identical re-emit |
| **S6.16** embargo | pooled Sharpe 1.55 → **1.31** under the stricter per-instrument embargo | a stricter control lowering the number confirms no real edge |
| **X.8 / X.11** census and loading | 141/140/141 columns per class; data loaded via the shared base policy | documented; no bespoke cleaning, no frozen-matrix leak |

## Labelling and barriers

The triple-barrier configuration was frozen before the out-of-sample window. EX.3 tested whether that was leaving value on the table by sweeping the barrier half-width and vertical horizon for a representative Energy contract: the purged-CV AUC moved between 0.46 and 0.59 with no coherent ridge, the best cell reaching only 0.589. An unstable surface with no stable optimum is exactly what a thin-edge problem produces, and tuning to it on the modelling data would be a soft form of overfitting. The decision was therefore to leave the locked barriers untouched — the firewall held, by choice rather than by omission.

## Features and importance

The pooled panel was measured rather than asserted at 141, 140 and 141 columns for Energy, Equity and Metals (X.8). Cluster-level importance (S4) was scored on a Mantegna correlation metric (Mantegna, 1999) by impurity, purged permutation accuracy and SHAP (Lundberg et al., 2020; López de Prado, 2020). The result is the section's lesson: impurity and SHAP are in-sample attribution and always distribute the whole of a fitted model's importance, so they say only which features the model leaned on; the purged permutation accuracy is the out-of-sample check, and it is near zero on every cluster but one (Equity's, at 0.025), even where the SHAP attribution is large (Metals' top cluster reaches 0.71). The decision was interpretive — a high SHAP must not be reported as edge — and it set how the feature-importance section reads.

## Models and calibration

The five-estimator horse-race selects by the mean out-of-fold AUC across the fifteen combinatorial paths on the modelling sample, returning 0.57 for Equity (XGBoost), 0.54 for Energy (a multilayer perceptron) and 0.53 for Metals (elastic-net logistic) — each marginally above the no-skill line. A separate robustness probe (EX.1), run on a stripped tree-and-linear roster, asks instead how many of the fifteen paths beat 0.50: Equity survives all fifteen, Metals thirteen, Energy only six. The two figures answer different questions and are kept distinct; together they locate the edge in Equity and expose Energy as close to a coin toss. Because the sizing stage consumes a probability directly, calibration was not optional. EX.4 established on synthetic data that Platt scaling beats isotonic at this sample size, and S3.9 confirmed on the selected models that a per-class Platt map, fitted train-only, cuts the expected calibration error materially while — being monotone — leaving the AUC untouched (Gramegna & Giudici, 2021). Calibration was therefore shipped, applied to both the deliverable probabilities and the Kelly stake.

## Evaluation and market timing

EX.5 characterised the provided signal to fix the ceiling: directional hit-rates run from 0.51 to 0.69, with information coefficients that are small throughout (the strongest well-covered name, gc1s, reaches 0.66 with an IC of 0.21; ng1s is undefined). A decent base signal leaves little headroom for a secondary filter. Market-timing skill was then tested directly on the acted trades, and the choice of test mattered. The Pesaran–Timmermann (1992) directional-accuracy statistic, which conditions on the base rates, is negative or insignificant in every book and negative pooled (−2.31). One statistic appeared to dissent — the *pooled* Treynor–Mazuy coefficient was positive and nominally significant (γ = +1.18) — but S5.11 attributed this to aggregation across sub-portfolios of heterogeneous scale (a regression form of Simpson's paradox; Robinson, 1950) compounded by the mechanical, option-like convexity of a stop-and-barrier exit (Jagannathan & Korajczyk, 1986; Treynor & Mazuy, 1966). S5.12 then proved the point in the data: standardising each sleeve to a common scale and re-pooling collapses the coefficient from +1.18 to −0.0031. The decision was to treat the scale-invariant Pesaran–Timmermann test as primary and the pooled Treynor–Mazuy coefficient as an artefact, not evidence of timing.

## Strategy and sizing

The barrier-exact, cost-aware backtest (S6.7) returns a pooled net Sharpe of 1.31, with Energy at 1.86, Equity at 0.86 and Metals at zero. Two experiments show that this number is about mechanism, not skill. S6.9 holds the positions, models, features and calibration fixed and varies only the exit rule: moving from a fixed ten-day hold to the actual barrier first-touch alone shifts every book (Equity from +0.54 to +0.86, Metals from −0.21 to zero), so the ordering is driven by the convex exit — winners ride to the profit barrier, losers are cut at the stop — rather than by classification. S6.15 and its leakage-safe follow-up EX.6 then tested whether a per-instrument signal-to-noise Kelly shrinkage (Kelly, 1956; Carver, 2015) would help: estimated correctly on the modelling sample, its certainty-equivalent gain is a large +92% in point estimate, but the paired bootstrap confidence interval on that gain contains zero. Under the two-part adoption rule — material gain *and* an interval excluding zero — it fails the second condition, so the deliverable reverted to a flat quarter-Kelly and re-emits byte-identically. S6.16 supplies the robustness coda: tightening the embargo to a per-instrument basis lowered the pooled Sharpe from 1.55 to 1.31, the direction a stricter, more honest control should move a number that no real edge underpins.

## Significance and deflation

The strategy section leads with significance (S6.14), not with the Sharpe. On the roughly 128-day window the pooled per-period Sharpe of 0.083 gives a *t*-statistic of 0.93, and the primary inference — a studentised stationary block-bootstrap 95% confidence interval of [−0.04, +0.19] (Lo, 2002) — contains zero, as does the analytic band; the probability of a positive Sharpe is 0.82 and the minimum track-record length, near 399 days, is roughly three times what is available. Selection-bias deflation (S6.8) corroborates without leading: the deflated-Sharpe ladder stays below 0.95 at every plausible trial count (pooled 0.61 falling to 0.20), the probability of backtest overfitting is around 0.35, and the minimum backtest length runs to years (Bailey & López de Prado, 2014; Bailey et al., 2017). Because the deflation statistics rest on noisily estimated higher moments at this sample length, they were deliberately demoted to corroboration rather than reported as load-bearing point probabilities.

## Final results and the five-lens convergence

None of the headline numbers survives as evidence of skill. The pooled book posts a net Sharpe of 1.31 (annualised volatility 7.5%, maximum drawdown −2.6%, gross +8.3% to net +4.9%; per class Energy 1.86, Equity 0.86, Metals zero), but five mutually independent lenses converge on the same null: the out-of-sample AUC is near 0.50; the cluster permutation accuracy is near zero throughout; the Sharpe carries a *t*-statistic of 0.93 with a confidence interval containing zero; the deflated-Sharpe ladder fails its threshold at every trial count; and the Pesaran–Timmermann statistic is negative everywhere. The convergence is predicted, not coincidental. Grinold's fundamental law, IR = IC·√BR (Grinold, 1989), makes it structural: with the primary's information coefficient near zero, the achievable information ratio is near zero whatever the breadth or sizing, and a secondary filter cannot manufacture skill the primary lacks. The positive backtested Sharpe is the footprint of the convex barrier exit, the volatility target and diversification. The honest conclusion is insufficient evidence of a deployable edge — reached by declining the tempting refinements as firmly as by testing the headline number.

## Per-instrument results and the shipped deliverable

Classification quality is uneven across the eleven instruments, and reporting it per instrument before any aggregate is what keeps the evaluation honest. The table below gives each instrument's purged out-of-sample AUC on the modelling sample (with its modelling-label count) alongside its coverage on the Jan–Jun 2022 deliverable window. The two columns are distinct: the AUC is a modelling-sample ranking metric, whereas the rows and the information coefficient are the realised 2022 window, and that metamodel-prediction IC is a different quantity from the primary-signal IC reported in EX.5.

| Class | Instrument | Purged-OOS AUC | Labels | OOS rows (2022) | OOS IC |
|---|---|---|---|---|---|
| Equity | es1s | 0.62 | 457 | 118 | +0.01 |
| Equity | nq1s | 0.59 | 482 | 122 | +0.17 |
| Equity | fesx1s | 0.57 | 510 | 127 | −0.04 |
| Energy | cl1s | 0.56 | 334 | 88 | −0.03 |
| Energy | ng1s † | 0.52 | 68 | 56 | +0.19 |
| Energy | rb1s | 0.49 | 504 | 124 | −0.02 |
| Energy | ho1s † | 0.43 | 61 | 2 | undefined |
| Metals | hg1s | 0.57 | 504 | 124 | −0.06 |
| Metals | si1s | 0.51 | 462 | 116 | −0.06 |
| Metals | pl1s | 0.49 | 453 | 104 | +0.02 |
| Metals | gc1s † | 0.45 | 138 | 30 | −0.23 |

† Thin coverage (fewer than sixty 2022 rows or an undefined IC): ho1s, ng1s and gc1s; their per-instrument figures are small-sample noise. Only Equity sits uniformly above the no-skill line; Metals and Energy straddle it, with the apparent Energy exceptions (ng1s 0.52) resting on the thinnest samples. At class level the calibrated Brier scores are Equity 0.249, Energy 0.265 and Metals 0.341, with precision 0.61, 0.62 and 0.58 — the calibration the Brier reflects is what lets the Kelly stage size on the probability directly.

The shipped deliverable is 1,011 daily predictions across the eleven instruments over Jan–Jun 2022, with calibrated probabilities confined to [0.40, 0.69]; the raw, uncalibrated probabilities span a far wider [0.04, 0.98], so the calibration map is doing real work. The accompanying fractional-Kelly weights apply the p̂ ≥ 0.55 confidence floor, so 36% of them (361 of 1,011) carry zero weight and the largest single position is 0.13 of capital — the book sizes off a thin high-confidence slice. All three thin-coverage names still emit rather than abstain, flagged in the coverage file. Both the prediction and weight files re-emit byte-identically under the seeded command, and the prediction window is a configuration field, so the hidden Jul–Dec 2022 half is run by changing one value.

## References (key methods)

The full bibliography is in `methodology-full.md`; the works cited here are listed below.

Bailey, D.H. and López de Prado, M. (2014) 'The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and non-normality', *Journal of Portfolio Management*, 40(5), pp. 94–107.

Bailey, D.H., Borwein, J.M., López de Prado, M. and Zhu, Q.J. (2017) 'The probability of backtest overfitting', *Journal of Computational Finance*, 20(4), pp. 39–69.

Carver, R. (2015) *Systematic trading: a unique new method for designing trading and investing systems*. Petersfield, Harriman House.

Gramegna, A. and Giudici, P. (2021) 'SHAP and LIME: an evaluation of discriminative power in credit risk', *Frontiers in Artificial Intelligence*, 4, 752558.

Grinold, R.C. (1989) 'The fundamental law of active management', *Journal of Portfolio Management*, 15(3), pp. 30–37.

Jagannathan, R. and Korajczyk, R.A. (1986) 'Assessing the market timing performance of managed portfolios', *Journal of Business*, 59(2), pp. 217–235.

Kelly, J.L. (1956) 'A new interpretation of information rate', *Bell System Technical Journal*, 35(4), pp. 917–926.

Lo, A.W. (2002) 'The statistics of Sharpe ratios', *Financial Analysts Journal*, 58(4), pp. 36–52.

López de Prado, M. (2018) *Advances in financial machine learning*. Hoboken, Wiley.

López de Prado, M. (2020) *Machine learning for asset managers*. Cambridge, Cambridge University Press.

Lundberg, S.M., Erion, G., Chen, H., DeGrave, A., Prutkin, J.M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N. and Lee, S.I. (2020) 'From local explanations to global understanding with explainable AI for trees', *Nature Machine Intelligence*, 2(1), pp. 56–67.

Mantegna, R.N. (1999) 'Hierarchical structure in financial markets', *European Physical Journal B*, 11(1), pp. 193–197.

Pesaran, M.H. and Timmermann, A. (1992) 'A simple nonparametric test of predictive performance', *Journal of Business & Economic Statistics*, 10(4), pp. 461–465.

Robinson, W.S. (1950) 'Ecological correlations and the behavior of individuals', *American Sociological Review*, 15(3), pp. 351–357.

Treynor, J.L. and Mazuy, K.K. (1966) 'Can mutual funds outguess the market?', *Harvard Business Review*, 44(4), pp. 131–136.
