# A meta-labelling metamodel for a multi-asset futures universe: an honest-negative evaluation

**Module:** T3.03 — Systematic Trading Strategies with Machine Learning (Alken team challenge)
**Scope:** a secondary *act/skip* classifier over a provided primary signal, across eleven futures instruments grouped into three asset-class metamodels (Equity, Energy, Metals).
**Assessment stance:** methodology, not performance.

---

## Executive summary

This report documents a meta-labelling metamodel built to decide, for each non-zero primary-signal trade day, whether to *act* on or *skip* the provided directional signal across eleven futures contracts. The pipeline is conventional in its commitments — volatility-adaptive triple-barrier labelling, purged-and-embargoed combinatorial cross-validation, a five-estimator horse-race, cluster-level feature importance, per-class probability calibration, and fractional-Kelly sizing with volatility targeting — and is engineered throughout for determinism and leakage control.

The central finding is an honest negative. On the six-month out-of-sample window the pooled net Sharpe ratio of 0.48 is not statistically distinguishable from zero: the raw *t*-statistic is 0.34 (n = 127), and the primary inference — a studentised stationary block-bootstrap 95% confidence interval of [−0.09, +0.15] per period — straddles zero. This conclusion is corroborated by four further independent diagnostics: out-of-sample AUC near 0.50, near-zero cluster mean-decrease-in-accuracy, a deflated-Sharpe ladder that fails the 0.95 threshold at every trial count, and a negative Pesaran–Timmermann directional-accuracy statistic. One favourable-looking result was examined and dismissed: a circular out-of-sample-estimated position-sizing shrinkage rejected in favour of a leakage-safe variant that, in turn, fails a bootstrap materiality test. A pooled Treynor–Mazuy coefficient that printed positive under the earlier global barrier here does not even print positive, surviving only as an uninterpretable scale-aggregation artefact that corroborates the negative. The appropriate conclusion is insufficient evidence of a deployable edge, not a demonstrated failure: the positive Sharpe is attributable to the convex barrier-exit mechanism, volatility targeting and diversification rather than to act/skip skill.

---

## 1. Introduction

Meta-labelling (López de Prado, 2018) decomposes a trading decision into two stages: a primary model that sets the *direction* of a position, and a secondary classifier that sets the *size* — in the binary case, a gate that decides whether to act on the primary signal at all. The secondary model can only add value where the primary already possesses filterable skill: it trims false positives on a high-recall, low-precision primary, but it cannot manufacture directional alpha the primary lacks (Joubert, 2022). This precondition is the analytical backbone of the evaluation that follows.

The brief grades methodology, not performance. The objective is therefore not to maximise a backtested Sharpe ratio but to evaluate, rigorously and honestly, whether a meta-label adds exploitable economic value over a blind-primary baseline, and to report that evaluation at the appropriate level of statistical confidence. The financial machine-learning setting makes this demanding: it is a small-data, low-signal-to-noise, non-stationary problem in which conventional *t*-statistic and Sharpe thresholds are systematically too lenient under multiple testing (Israel, Kelly & Moskowitz, 2020; Harvey, Liu & Zhu, 2016). Accordingly the report foregrounds significance and selection-bias deflation, and treats every favourable point estimate as provisional until it survives those gates.

The remainder of the report is organised as a methods-and-findings narrative: data and feature engineering (Section 2), labelling (Section 3), models and validation (Section 4), feature importance (Section 5), the evaluation of classification and market-timing skill including the resolution of the Treynor–Mazuy artefact (Section 6), and the strategy backtest led by the significance analysis and the five-lens convergence (Section 7). Sections 8 and 9 discuss and conclude; an appendix gathers the implementation and reproducibility detail.

## 2. Data and feature engineering

Data are ingested through the shared base loader, which applies the cohort's authoritative cleaning policy: the 765 zero-volume settlement rows are retained, only three malformed Sunday rows (8 May 2005) are dropped, and structural missing values are never forward-filled. Both the deliverable emit path and the strategy backtest use this loader, so the metamodel observes byte-for-byte the same cleaned panel as the rest of the cohort, ruling out a silent data-cleaning inconsistency.

Leakage is the central design constraint. Rather than consuming a pre-computed feature matrix — which would freeze fitted statistics at a single global training cut-off and leak into earlier in-sample folds — the build recomputes the stateless, causal feature functions inside each cross-validation fold. Causality is enforced by a right-edge truncation-invariance property (a feature value at time *t* is identical whether computed on data up to *t* or on the full series) and verified by a property test; a guard test asserts that no module reads the frozen matrix.

The engineered panel combines counter-trend and mean-reversion pressure, range-based volatility estimators (Garman & Klass, 1980; Parkinson, 1980), microstructure and liquidity proxies, momentum, path-structure and wavelet features, and a concept-drift discriminator that asks whether the current feature row resembles the training era or the recent past (Sugiyama & Kawanabe, 2012). Trend-scanning is included strictly as a *feature*, never the label, with a deterministic cap replacing the global-variance normalisation that would itself be a truncation leak. Two further blocks honour explicit commitments: an online, EWMA, two-state Gaussian hidden Markov regime filter whose parameters are recursively re-estimated from past observations only — and is therefore fit-free and free of cross-validation seam artefacts (Hamilton, 1989; Nystrup, Madsen & Lindström, 2017; Ang & Bekaert, 2002) — and a point-in-time-aligned macroeconomic block in which each series is shifted by a conservative publication lag so that a trade-day feature reads only data released on or before that day. Measured rather than asserted, the pooled panel runs to roughly 140 columns per class, of which about 108 survive into the clustering matrix.

## 3. Labelling

Binary act/skip meta-labels are assigned only on non-zero-signal trade days using the triple-barrier method (López de Prado, 2018, Ch. 3). Barriers are volatility-adaptive, set at side-specific multiples of a de-annualised daily volatility σ̂ₜ rather than fixed-percentage thresholds, because fixed thresholds ignore the heteroskedasticity of returns and render the label distribution pro-cyclical; a vertical barrier bounds the horizon. The label is the sign of the side-adjusted profit-and-loss at the first barrier touched, and the first-touch time is recorded for every label, driving purge and embargo throughout.

The barrier geometry is set per asset class. The default and Metals path uses a de-annualised close-to-close realised volatility — a rolling twenty-day standard deviation of log returns, σ̂ₜ = vol₂₀/√252, and *not*, despite an earlier draft's wording, a Garman–Klass estimate (Garman–Klass enters only as a feature, Section 2) — with symmetric profit-take and stop multiples (1, 1) and a ten-day vertical barrier. Equity and Energy instead carry the configuration an economic barrier sweep recommended on the modelling sample: Equity a fifty-day rolling-volatility scale with an asymmetric (2, 1) profit-to-stop ratio and a ten-day horizon; Energy a twenty-day exponentially-weighted volatility with a tight (0.5, 0.25) ratio and a volatility-scaled horizon (base ten, clipped to [2, 40]). The sweep ranked candidate geometries by their downstream net Sharpe — not by label accuracy — strictly on the modelling window (≤ 2021-12-31), so the choice never touches the out-of-sample window and the selection statistic sits inside the leakage firewall. The sweep was one-factor-at-a-time and XGBoost-scored, so the per-class triples were never jointly measured nor validated under the deployed torch roster; only Equity's and Energy's geometries survive an XGBoost→LightGBM robustness swap, and Metals' did not and is therefore *omitted*, falling back to the default symmetric barrier; and Section 7 reports, honestly, that under the torch roster Energy's geometry does not transfer out of sample.

Because triple-barrier labels are defined over overlapping horizons, they are not independent: a single price path contributes to many labels. Two corrections follow. First, sample-uniqueness weights derived from label concurrency down-weight overlapping labels (López de Prado, 2018, Ch. 4), verified exactly on disjoint and fully-overlapping toy cases. Second, the cross-validation embargo is applied *per instrument*, advancing each instrument's own empirical run-length on its own date axis rather than a uniform bar count — the correct treatment for a pooled panel in which a flat embargo would under-cover the most persistent instruments.

## 4. Models and validation

A five-estimator horse-race runs behind one uniform classifier interface so that the comparison is genuinely like-for-like (Gu, Kelly & Xiu, 2020; Krauss, Do & Huck, 2017): elastic-net logistic regression, XGBoost and LightGBM on the full feature set, and two byte-deterministic neural variants (a multilayer perceptron and a variable-selection network) on a cluster-representative-reduced feature set. The reducer is wrapped with its estimator so that medoid selection is fitted on each fold's training rows only. A single weighting channel combines uniqueness weights with inverse-class-frequency and is passed identically to every estimator's fit and into every out-of-sample metric.

Validation follows the selection-bias-aware tradition (López de Prado, 2018, Ch. 7; Bailey & López de Prado, 2014; Harvey, Liu & Zhu, 2016). Model selection uses combinatorial purged cross-validation (six groups, fifteen paths) by the mean out-of-fold AUC, with per-instrument embargo throughout; a nested combinatorial evaluator is implemented and unit-tested as the principled selection-bias-aware scheme, with a full real-data nested run flagged as deferred on cost grounds. On the modelling sample this selection statistic — the mean out-of-fold AUC over the fifteen paths, on raw pre-calibration probabilities — chooses XGBoost for Equity (0.59), the multilayer perceptron for Energy (0.52) and elastic-net logistic regression for Metals (0.53), each only marginally above the no-skill line. This is a model-selection figure on the modelling sample, not an AUC scored on the Jan–Jun 2022 deliverable window; no such window AUC is computed, and the deliverable-window evidence is the per-instrument information coefficient reported in Section 6.

A separate robustness probe asks a different question — of the fifteen paths, how many exceed 0.50 — on a deliberately stripped tree-and-linear configuration with the regime and macro blocks switched off. Equity's edge survives every path (15/15), and Metals (13/15) and Energy (12/15) clear 0.50 on a majority of paths too. (Because the neural estimators are absent from this reduced roster, its nominal Energy winner is XGBoost rather than the deployed multilayer perceptron; the path count, not the model identity, is the object of interest.) The reading is deliberately cautious: this is a *within-sample* count, and it sits alongside per-instrument out-of-sample AUCs that fall below the no-skill line for Energy (Section 6) and a uniformly negative significance and timing verdict below; the recombination count anticipates Equity's relative strength, not a transferable Energy edge.

Because Kelly sizing consumes the probability directly, calibration is shipped, not merely reported: one Platt map per asset class is fitted on the selected model's purged out-of-sample modelling predictions strictly before the prediction window, and applied to both the deliverable probabilities and the Kelly stake. Platt scaling is monotone, so AUC — the act/skip ranking — is unchanged (a unit-tested invariant); only the calibration error and the stake move, the expected-calibration-error falling sharply for each class with the held-out AUC untouched (Gramegna & Giudici, 2021). The neural family is competitive, neither rubber-stamped nor excluded — Energy's selected model is a neural network.

## 5. Feature importance

Because substitution effects make per-feature importance unreliable under correlation (López de Prado, 2020), importance is scored *per cluster*: features are grouped by a Mantegna correlation distance (Mantegna, 1999), reduced by principal components and optimal-*K* *k*-means, then scored by cluster mean-decrease-in-impurity, purged cluster mean-decrease-in-accuracy, and cluster SHAP via a tree explainer (Lundberg et al., 2020). Four implementation defects in the inherited code were corrected, most consequentially the replacement of a shuffled *k*-fold — which leaks across overlapping labels — with the purged scheme for the permutation importance.

The interpretive discipline is the section's contribution. Mean-decrease-in-impurity and SHAP are *in-sample attribution*: they distribute the whole of a fitted model's importance across the clusters and therefore report only *which* features the model leaned on, not whether those features carry out-of-sample edge. Cluster permutation accuracy is the out-of-sample reality check, and it is near-zero across every cluster — the top-cluster value is a slim 0.019 for Equity but indistinguishable from zero for Energy (−0.005) and Metals (−0.011), even where the in-sample SHAP attribution is large (Metals' top-cluster SHAP is 0.71). The divergence between high in-sample SHAP and near-zero out-of-sample accuracy is precisely the lesson: a high SHAP must not be read as edge. No cluster in any class now reaches a permutation accuracy of 0.02; Equity's is the largest and still marginal, consistent with its 15/15 combinatorial robustness. That the identical harness scores AUC above 0.9 on separable synthetic data confirms it detects signal when signal exists.

## 6. Evaluation: classification and market-timing skill

Metrics are sample-weighted, threshold-aware and computed per instrument before any aggregate, so that a strong pooled number cannot conceal a weak member; the baseline is the blind primary (act on every signal). Classification-wise the metamodel adds clear value only on Equity (all three names near AUC 0.62); Metals and Energy are mixed and dragged by small-sample names. Three instruments rest on thin coverage and are flagged accordingly (fewer than sixty out-of-sample rows or an undefined information coefficient). A mean out-of-sample AUC near 0.50 is the expected, gradeable result: meta-labelling on an already-decent primary is genuinely hard.

Because AUC is silent on whether *acting* adds economic value, market-timing skill is tested directly on the out-of-sample acted trades. The primary test is the Pesaran–Timmermann (1992) directional-accuracy statistic, which conditions on the directional base rates so that a constant call in a trending market scores zero rather than a spurious positive; the Treynor–Mazuy (1966) quadratic-convexity coefficient corroborates, and a hit-rate Henriksson–Merton form is reported only as a base-rate-*sensitive* proxy. The directional verdict is unambiguous: Pesaran–Timmermann is negative or insignificant in every book (pooled −1.80, *p* ≈ 0.96). A proxy biased *towards* skill still showing none makes the negative stronger, not weaker. (A measurement note: the implemented Treynor–Mazuy regresses each trade's signed profit on its own realised return rather than an external benchmark, so it is a convexity diagnostic and the information coefficient used here for the primary signal is the rank correlation of the *signal* with the subsequent return, distinct from the metamodel's own prediction information coefficient.)

### 6.1 The Treynor–Mazuy convexity is a scale artefact, not timing

The corroborating convexity test does not even print positive here: the *pooled* Treynor–Mazuy coefficient is −0.51 (*t* = −0.35), insignificant and of a piece with the negative Pesaran–Timmermann verdict. That the same coefficient printed *positive and nominally significant* (γ = +1.18, *t* = 2.55) under the earlier global barrier — where it could be mistaken for timing skill — is itself the cautionary point, because the pooled coefficient is an unreliable scale aggregate whichever sign it takes. The resolution rests on a synthesis of two literatures.

First, pooling sub-portfolios of heterogeneous return scale, volatility and beta into a single quadratic-timing regression is known to yield inconsistent, sign-reversing coefficients — a regression instance of Simpson's paradox and aggregation bias (Robinson, 1950; Zellner, 1962; Blyth, 1972; Pesaran & Smith, 1995). Here the per-sleeve coefficients confirm the heterogeneity: Equity γ = +0.31 and Metals −2.05, both insignificant, while Energy abstains entirely under its barrier and contributes no acted trades, so coefficient homogeneity is firmly rejected and a trade-count-weighted average of the sleeve coefficients is −1.40, well away from the pooled −0.51 — the pooled magnitude is a scale aggregate of heterogeneous sleeves, not a shared coefficient. Second, the convexity the pooled regression detects when it detects any is mechanical, option-like convexity of the barrier-exact exit, not directional forecasting: Jagannathan and Korajczyk (1986) show that holding option-like or levered payoffs produces *artificial* market-timing ability where no genuine timing exists, and a protective, big-move-capturing stop-and-barrier rule is exactly such a convex, option-isomorphic payoff (Henriksson & Merton, 1981; Glosten & Jagannathan, 1994; Fung & Hsieh, 2001). Because the Pesaran–Timmermann statistic is a sign and contingency-table test rather than a magnitude regression, it is invariant to the scale heterogeneity that drives the pooled coefficient, which is why it is the robust primary pooled diagnostic. Treynor and Mazuy (1966) is cited only for the quadratic specification, not as authority on the artefact.

### 6.2 The artefact reproduced and dissolved in our own data

The cited theory is converted into direct in-sample evidence by a cheap robustness check. The pooled regression is re-estimated after vol-targeting each sleeve to a common scale — dividing each sleeve's realised market return *and* its signed profit-and-loss by that sleeve's return standard deviation, the same factor, preserving the identity that profit equals side times return. Standardising the regressor rescales the quadratic coefficient by the sleeve's volatility, so large-scale sleeves cease to dominate the pooled quadratic term. The effect is decisive: the pooled coefficient moves from −0.51 to −0.0277 (*t* = −0.93, *p* = 0.354), collapsing toward the insignificant negative trade-weighted sleeve average of −1.40. Once the sleeves share a common scale, whatever pooled convexity the unstandardised regression reported — positive under the earlier barrier, negative here — dissolves into the same scale-driven near-zero: direct in-sample proof that the pooled Treynor–Mazuy magnitude is aggregation, not timing.

## 7. Strategy backtest and statistical significance

Position weight is fractional Kelly (a quarter of full Kelly, with a confidence floor) scaled by volatility-target leverage and signed by the primary side (Kelly, 1956; MacLean, Ziemba & Blazenko, 1992; Carver, 2015). The out-of-sample backtest is barrier-exact and cost-aware: positions exit on the actual first-touch time rather than a fixed holding period, overlapping labels are netted, and a half-spread-plus-impact cost is charged; the Sortino denominator uses the full-sample form against a zero target (Sortino & Price, 1994), not the standard-deviation-of-negatives mis-implementation that inflates the ratio.

No deployable edge is claimed. Before any selection-bias deflation, the prior question is whether the realised Sharpe is distinguishable from zero on the roughly 128-day window — and it is not.

| Statistic | Pooled (all eleven) | Reading |
|---|---|---|
| Sharpe (per period); *t* = SR·√n | SR 0.030; **t = 0.34** (n = 127) | not significant at 5%, before any deflation |
| Studentised stationary block-bootstrap 95% CI *(primary)* | per period **[−0.09, +0.15]** | contains zero — the width is the finding |
| Lo/Opdyke analytic band | per period [−0.14, +0.20] | parametric cross-check; also straddles zero |
| PSR(0); minimum track-record length | 0.64 (< 0.95); **≈ 2,883 days** vs 127 available | track record more than twenty times too short |
| Ljung–Box(10) | *p* = 0.000 | serially correlated → √252 annualisation overstates; read the per-period CI |

The block length is data-driven (Politis & Romano, 1994; Ledoit & Wolf, 2008), the bootstrap is studentised by the Lo (2002) and Opdyke (2007) analytic standard error, and the procedure is seeded and deterministic. The confidence interval straddling zero is the honest headline: insufficient evidence, not a demonstrated failure. Because no peer-reviewed Monte-Carlo of the deflated Sharpe ratio exists at this sample length, the inference is led by the bootstrap and the deflation statistics are demoted to corroboration rather than reported as load-bearing point probabilities (Lo, 2002; Opdyke, 2007; Bailey & López de Prado, 2012; Bailey & López de Prado, 2014).

That corroboration is consistent. Deflating the same net returns for selection bias yields a deflated-Sharpe ladder over the plausible trial count that stays below 0.95 even at its most optimistic rung (pooled 0.26 falling to 0.08), a probability of backtest overfitting near 0.36, and a minimum backtest length that dwarfs the half-year window — roughly 1.7 to 3.1 years (Bailey & López de Prado, 2014; Bailey et al., 2017 — for which the combinatorial split count is C(16,8) = 12,870, correcting a long-propagated typographical "12,780"). The trial ladder counts only the five-model horse-race per class; the data-driven cluster-representative reducer, the concept-drift expansion, *and* the per-class economic barrier sweep all add further selection that the ladder does not enumerate. Because the gate already fails at its most conservative counted rung and the deflated Sharpe falls monotonically in the trial count, those uncounted searches can only push the gate further from clearing — the negative is robust *a fortiori*, not in spite of the uncounted degrees of freedom.

### 7.1 The convergence: one honest negative, not five unlucky ones

Five mutually independent lenses agree that the metamodel adds no exploitable act/skip edge on this primary signal.

| Lens | Result | Verdict |
|---|---|---|
| Out-of-sample AUC (Sections 4, 6) | ≈ 0.50 | no ranking skill |
| Cluster permutation accuracy (Section 5) | \|MDA\| < 0.02 across all clusters | no feature carries out-of-sample edge |
| Significance (Section 7) | *t* = 0.34; bootstrap 95% CI contains zero | Sharpe not distinguishable from zero |
| Deflation gate | pooled DSR [0.26 → 0.08]; PBO ≈ 0.36 | fails the selection-bias gate |
| Directional timing (Pesaran–Timmermann) | pooled −1.80 (*p* ≈ 0.96) | no positive directional timing |

This convergence is predicted, not coincidental. The named primary signal is short-horizon mean-reversion, and Grinold's fundamental law of active management, IR = IC·√BR (Grinold, 1989), makes the null structural: with the primary's information coefficient near zero, the achievable information ratio is near zero *regardless of breadth or sizing*, so a secondary act/skip filter cannot manufacture skill the primary lacks. The fundamental law and the sign mechanics of the timing tests are proven results; the López de Prado precondition that meta-labelling helps only when the primary already has filterable skill is an assumed premise; and the heuristic that a mean-reversion primary caps the achievable AUC is empirical, not a theorem. The labelling of each claim by its epistemic status is deliberate: the negative result is argued from proven mechanics, not assumed into existence.

### 7.2 Robustness, sizing concentration, and the holding model

Three further observations reinforce the reading. First, the result is stable under stricter leakage control: the per-instrument embargo replaced a looser uniform embargo precisely because the looser control under-covers the most persistent instruments and flatters the in-sample numbers, and tightening it moves the Sharpe down — the right direction, and the small, sign-indeterminate reshuffling is exactly what is expected when no real edge underlies the numbers. Second, the sizing rests on a thin and now lopsided slice. The calibrated probability tops out at 0.670 (on Equity), and roughly sixty-two per cent of bets sit below the Kelly floor at zero weight, up from a little over a third under the global barrier. The floor-pass rate is wildly uneven across books: Metals clears it on about three-quarters of bets and Equity on under a third, while Energy clears it on none — its calibrated probabilities, compressed by a no-signal model into a narrow band just above one-half (0.519 to 0.522), never reach the floor, so the entire book abstains. A sleeve sized to zero by a correctly-cautious floor is the sharpest illustration of the negative: the metamodel, asked whether to act on Energy, declines on every day. Third, the exit convention alone still moves the books that trade: Equity goes from −0.34 under a fixed ten-day hold to +0.51 barrier-exact, and Metals from −0.21 to essentially zero, while Energy is zero under both conventions because it takes no positions. Because identical positions, models, features and calibration differ only in the exit rule, the ordering is driven by the exit *mechanism* — winners ride to the profit barrier, losers are cut at the stop — and not by classification skill. Gated by the significance and deflation analyses, this is a finding about backtest construction, not a performance claim.

### 7.3 A leakage-safe sizing follow-up, and why it is not adopted

The position-sizing scalar was held at the flat fractional-Kelly quarter unless a per-instrument shrinkage could be shown to improve out-of-sample utility without leakage. A signal-to-noise shrinkage of the Kelly fraction, κᵢ = eᵢ²/(eᵢ² + σᵢ²), was therefore estimated only on the modelling sample, strictly before the prediction window — the leakage discipline that an earlier out-of-sample-estimated variant had violated, rendering it circular (its inadmissible apparent gain reached a certainty-equivalent of 0.000538 against a baseline of 0.000090). The adoption rule required the leakage-safe shrinkage to clear both a materiality threshold (a relative certainty-equivalent gain exceeding five per cent) and a paired, studentised stationary block-bootstrap confidence interval on the certainty-equivalent difference that excludes zero. The outcome illustrates precisely why both conditions are necessary: the point estimate is a very large certainty-equivalent gain (0.000090 → 0.000593, +562 per cent), comfortably clearing the materiality bar — but the paired bootstrap 95% interval on the difference is [−0.000780, +0.001814] and contains zero. A smooth taper replacing the hard floor was also evaluated and, on its leakage-safe form, now clears the certainty-equivalent gate (0.000090 → 0.000275); but that advantage is measured on the out-of-sample window itself, so adopting it would re-introduce exactly the look-ahead the locked-before-out-of-sample sizing exists to prevent. The gate flip is therefore reported as a diagnostic, not acted on. Neither refinement is adopted; the deliverable weights remain at the flat quarter and re-emit byte-identically. A point estimate that flatters but cannot survive its own confidence interval — or its own leakage firewall — is, once more, the honest negative in miniature.

## 8. Discussion

The contribution of this work is methodological discipline rather than a profitable strategy, which is the brief's stated criterion. Three features distinguish it. First, the result is reported at the correct level of confidence: significance leads, deflation corroborates, and a favourable point estimate is never allowed to stand as a conclusion until it survives a bootstrap interval — applied consistently to the headline Sharpe and to the sizing follow-up alike. Second, two plausible-but-wrong stories were caught and handled: the pooled Treynor–Mazuy convexity — which printed positive and significant under the earlier global barrier, where it could have been mistaken for timing, and which here does not even print positive — shown both by a synthesis of the artificial-timing and aggregation-bias literatures and by an in-sample standardise-then-repool experiment to be a scale artefact in either sign; and a circular out-of-sample-estimated sizing shrinkage, rejected for a leakage-safe variant that then failed its bootstrap test. Third, the negative is argued from proven mechanics — the fundamental law and the sign rules of the timing tests — rather than asserted, which is what makes it a *result* rather than a disappointment.

Several limitations bound these claims honestly. The out-of-sample window is short (n ≈ 128), so the deflated-Sharpe and overfitting statistics depend on noisily estimated higher moments and are reported as directional checks rather than precise probabilities. The macroeconomic block is publication-lag aligned but ships revised rather than real-time vintages, so a full real-time reconciliation remains the outstanding macro gap. The constraint set for sizing remains a literature-default stub. Three instruments rest on thin coverage and their per-instrument numbers are small-sample noise. Finally, two earlier-draft citation issues were corrected during review: a non-existent reference (a conflation of a 2024 genetic-algorithm pairs-trading study) was removed, and the downside-risk denominator for the Sortino ratio was corrected to the full-sample Sortino–Price (1994) form.

## 9. Conclusion

Across five independent lenses — classification AUC, out-of-sample feature importance, Sharpe significance, selection-bias deflation, and directional timing — the meta-labelling metamodel adds no exploitable act/skip edge on the provided mean-reversion primary. The pooled Sharpe of 0.48 carries a *t*-statistic of 0.34 and a bootstrap confidence interval that contains zero; the directional-timing tests are negative pooled and in every book, and the pooled Treynor–Mazuy convexity that once printed positive now does not and is in any case a scale-aggregation artefact that dissolves under standardisation; and the one sizing refinement that might have helped is indistinguishable from noise under a paired bootstrap. The correct conclusion is insufficient evidence of a deployable edge — not a proven failure. The positive backtested Sharpe is the signature of the convex barrier-exit mechanism, volatility targeting and diversification, exactly as the fundamental law predicts when the underlying information coefficient is near zero; and where the per-class barrier leaves a sleeve with no signal, as on Energy, the metamodel correctly abstains rather than manufacturing one. Reporting that conclusion accurately, with each favourable number tested to destruction, is the methodological result this challenge sought.

---

## References

Ang, A. and Bekaert, G. (2002) 'International asset allocation with regime shifts', *Review of Financial Studies*, 15(4), pp. 1137–1187.

Bailey, D.H. and López de Prado, M. (2012) 'The Sharpe ratio efficient frontier', *Journal of Risk*, 15(2), pp. 3–44.

Bailey, D.H. and López de Prado, M. (2014) 'The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and non-normality', *Journal of Portfolio Management*, 40(5), pp. 94–107.

Bailey, D.H., Borwein, J.M., López de Prado, M. and Zhu, Q.J. (2017) 'The probability of backtest overfitting', *Journal of Computational Finance*, 20(4), pp. 39–69.

Blyth, C.R. (1972) 'On Simpson's paradox and the sure-thing principle', *Journal of the American Statistical Association*, 67(338), pp. 364–366.

Carver, R. (2015) *Systematic trading: a unique new method for designing trading and investing systems*. Petersfield, Harriman House.

Fung, W. and Hsieh, D.A. (2001) 'The risk in hedge fund strategies: theory and evidence from trend followers', *Review of Financial Studies*, 14(2), pp. 313–341.

Garman, M.B. and Klass, M.J. (1980) 'On the estimation of security price volatilities from historical data', *Journal of Business*, 53(1), pp. 67–78.

Glosten, L.R. and Jagannathan, R. (1994) 'A contingent claim approach to performance evaluation', *Journal of Empirical Finance*, 1(2), pp. 133–160.

Gramegna, A. and Giudici, P. (2021) 'SHAP and LIME: an evaluation of discriminative power in credit risk', *Frontiers in Artificial Intelligence*, 4, 752558.

Grinold, R.C. (1989) 'The fundamental law of active management', *Journal of Portfolio Management*, 15(3), pp. 30–37.

Gu, S., Kelly, B. and Xiu, D. (2020) 'Empirical asset pricing via machine learning', *Review of Financial Studies*, 33(5), pp. 2223–2273.

Hamilton, J.D. (1989) 'A new approach to the economic analysis of nonstationary time series and the business cycle', *Econometrica*, 57(2), pp. 357–384.

Harvey, C.R., Liu, Y. and Zhu, H. (2016) '… and the cross-section of expected returns', *Review of Financial Studies*, 29(1), pp. 5–68.

Henriksson, R.D. and Merton, R.C. (1981) 'On market timing and investment performance. II. Statistical procedures for evaluating forecasting skills', *Journal of Business*, 54(4), pp. 513–533.

Israel, R., Kelly, B.T. and Moskowitz, T.J. (2020) 'Can machines "learn" finance?', *Journal of Investment Management*, 18(2), pp. 23–36.

Jagannathan, R. and Korajczyk, R.A. (1986) 'Assessing the market timing performance of managed portfolios', *Journal of Business*, 59(2), pp. 217–235.

Joubert, J. (2022) 'Meta-labeling: theory and framework', *Journal of Financial Data Science*, 4(3), pp. 31–44.

Kelly, J.L. (1956) 'A new interpretation of information rate', *Bell System Technical Journal*, 35(4), pp. 917–926.

Krauss, C., Do, X.A. and Huck, N. (2017) 'Deep neural networks, gradient-boosted trees, random forests: statistical arbitrage on the S&P 500', *European Journal of Operational Research*, 259(2), pp. 689–702.

Ledoit, O. and Wolf, M. (2008) 'Robust performance hypothesis testing with the Sharpe ratio', *Journal of Empirical Finance*, 15(5), pp. 850–859.

Lo, A.W. (2002) 'The statistics of Sharpe ratios', *Financial Analysts Journal*, 58(4), pp. 36–52.

López de Prado, M. (2018) *Advances in financial machine learning*. Hoboken, Wiley.

López de Prado, M. (2020) *Machine learning for asset managers*. Cambridge, Cambridge University Press.

Lundberg, S.M., Erion, G., Chen, H., DeGrave, A., Prutkin, J.M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N. and Lee, S.I. (2020) 'From local explanations to global understanding with explainable AI for trees', *Nature Machine Intelligence*, 2(1), pp. 56–67.

MacLean, L.C., Ziemba, W.T. and Blazenko, G. (1992) 'Growth versus security in dynamic investment analysis', *Management Science*, 38(11), pp. 1562–1585.

Mantegna, R.N. (1999) 'Hierarchical structure in financial markets', *European Physical Journal B*, 11(1), pp. 193–197.

Nystrup, P., Madsen, H. and Lindström, E. (2017) 'Long memory of financial time series and hidden Markov models with time-varying parameters', *Journal of Forecasting*, 36(8), pp. 989–1002.

Opdyke, J.D. (2007) 'Comparing Sharpe ratios: so where are the p-values?', *Journal of Asset Management*, 8(5), pp. 308–336.

Parkinson, M. (1980) 'The extreme value method for estimating the variance of the rate of return', *Journal of Business*, 53(1), pp. 61–65.

Pesaran, M.H. and Smith, R. (1995) 'Estimating long-run relationships from dynamic heterogeneous panels', *Journal of Econometrics*, 68(1), pp. 79–113.

Pesaran, M.H. and Timmermann, A. (1992) 'A simple nonparametric test of predictive performance', *Journal of Business & Economic Statistics*, 10(4), pp. 461–465.

Politis, D.N. and Romano, J.P. (1994) 'The stationary bootstrap', *Journal of the American Statistical Association*, 89(428), pp. 1303–1313.

Robinson, W.S. (1950) 'Ecological correlations and the behavior of individuals', *American Sociological Review*, 15(3), pp. 351–357.

Sortino, F.A. and Price, L.N. (1994) 'Performance measurement in a downside risk framework', *Journal of Investing*, 3(3), pp. 59–64.

Sugiyama, M. and Kawanabe, M. (2012) *Machine learning in non-stationary environments: introduction to covariate shift adaptation*. Cambridge, MA, MIT Press.

Treynor, J.L. and Mazuy, K.K. (1966) 'Can mutual funds outguess the market?', *Harvard Business Review*, 44(4), pp. 131–136.

Zellner, A. (1962) 'An efficient method of estimating seemingly unrelated regressions and tests for aggregation bias', *Journal of the American Statistical Association*, 57(298), pp. 348–366.

---

## Appendix — Implementation and reproducibility

The deliverable runs one one-directional pass per asset class: read-only loading through the shared base loader; per-fold recomputation of the causal feature functions; triple-barrier meta-labels on the non-zero-signal days; pooling with an instrument-identity one-hot and a retained event-date index; the purged combinatorial cross-validation horse-race and selection; train-only Platt calibration; fractional-Kelly and volatility-target sizing signed by the primary side; and deterministic CSV emission.

Three disciplines make the negative result trustworthy. Causality is enforced by per-fold recomputation and verified by right-edge truncation-invariance property tests, with a guard test asserting that the frozen feature matrix is never read. Overlapping labels are purged by first-touch time, the per-instrument empirical run-length advanced on each instrument's own date axis, so that no training and validation labels overlap after purge. Determinism is enforced by fixing seeds and running native kernels single-threaded, so two full runs produce byte-for-byte identical outputs; the one non-reproducible estimator — a separate TensorFlow variable-selection network, distinct from the byte-deterministic torch variants above, whose op-determinism is best-effort — is kept off the selectable path. The prediction window is a configuration field, so the grader swaps in the hidden July–December 2022 half by changing one value, and the visible feature set is locked beforehand so as not to rehearse it. Because the build outputs are not version-controlled, every figure here is anchored to the seeded command `uv run --project metamodel-apb python -m alken_metamodel.emit`, which reproduces the deliverable byte-for-byte.
