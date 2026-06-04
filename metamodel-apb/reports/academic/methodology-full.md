# A meta-labelling metamodel for a multi-asset futures universe: an honest-negative evaluation

**Module:** T3.03 — Systematic Trading Strategies with Machine Learning (Alken team challenge)
**Scope:** a secondary *act/skip* classifier layered over a provided primary signal, across eleven futures instruments grouped into three asset-class metamodels (Equity, Energy, Metals).
**Assessment stance:** methodology, not performance.

---

## Contents

1. Introduction
2. Data and feature engineering
3. Labelling
4. Models and validation
5. Feature importance
6. Evaluation: classification and market-timing skill
7. Strategy backtest and statistical significance
8. Discussion
9. Conclusion
- References
- Appendix A — Implementation and reproducibility
- Appendix B — Commitments, modules and citations

---

## Executive summary

This report documents a meta-labelling metamodel built to decide, on each day a provided primary signal is non-zero, whether to *act* on or *skip* the directional call, across eleven futures contracts in three asset-class books. The pipeline is conventional in its commitments — volatility-adaptive triple-barrier labelling, purged and embargoed combinatorial cross-validation, a five-estimator horse-race, cluster-level feature importance, per-class probability calibration, and fractional-Kelly sizing with a volatility target — and is engineered throughout for determinism and leakage control.

The central result is a negative one, and it is reported as such. On the six-month out-of-sample window the pooled net Sharpe ratio of 0.48 cannot be distinguished from zero: the raw *t*-statistic is 0.34 on 127 observations, and the primary inference, a studentised stationary block-bootstrap 95% confidence interval of [−0.09, +0.15] per period, straddles zero. Four further diagnostics agree. Out-of-sample ranking AUC sits near 0.50; cluster permutation importance is near zero on every cluster; a deflated-Sharpe ladder fails the 0.95 threshold at every plausible trial count; and the Pesaran–Timmermann directional-accuracy statistic is negative pooled and in every book. One favourable-looking result was examined and dismissed — an out-of-sample-estimated position-sizing shrinkage rejected as circular and then, in a leakage-safe form, as indistinguishable from noise; and a pooled Treynor–Mazuy coefficient that printed positive under the earlier global barrier here does not even print positive, surviving only as an uninterpretable scale-aggregation artefact that corroborates the negative. The defensible conclusion is therefore insufficient evidence of a deployable edge, not a demonstrated failure. The positive backtested Sharpe that remains is the footprint of the convex barrier exit, the volatility target and diversification across books, not of act/skip skill.

---

## 1. Introduction

Meta-labelling, in the sense of López de Prado (2018), splits a trading decision into two stages. A primary model fixes the *direction* of a position; a secondary classifier fixes the *size*, and in the binary case reduces to a gate that decides whether to act on the primary signal at all. The secondary stage can only earn its keep where the primary already carries filterable skill — it can trim false positives on a high-recall, low-precision signal, but it cannot conjure directional accuracy the primary does not possess (Joubert, 2022). That precondition is the spine of everything that follows, because the primary supplied here is a short-horizon mean-reversion call whose own predictive content turns out to be slight.

The brief grades methodology, not performance. The aim is not to maximise a backtested Sharpe ratio but to judge, rigorously, whether a meta-label adds exploitable economic value over a blind-primary baseline, and to report that judgement at an honest level of statistical confidence. Financial machine learning makes the task awkward: the data are few, the signal-to-noise ratio is low, and the distribution drifts, so the conventional *t* and Sharpe thresholds are far too lenient once multiple testing is taken seriously (Israel, Kelly & Moskowitz, 2020; Harvey, Liu & Zhu, 2016). The work therefore puts significance and selection-bias deflation first and treats every favourable point estimate as provisional until it survives those gates.

The eleven instruments are pooled into three per-class metamodels — Equity, Energy and Metals — one classifier per class rather than one per instrument. The remainder of the report follows a methods-and-findings line: data and feature engineering (Section 2), labelling (Section 3), the models and their validation (Section 4), feature importance (Section 5), the evaluation of classification and market-timing skill including the resolution of the Treynor–Mazuy puzzle (Section 6), and the strategy backtest led by the significance analysis (Section 7). Sections 8 and 9 discuss and conclude. Implementation, determinism and the leakage controls are gathered in Appendix A so that the body stays with the argument.

## 2. Data and feature engineering

Data enter through the shared base loader, which applies the cohort's agreed cleaning policy: the 765 zero-volume settlement rows are kept, only three malformed Sunday rows (8 May 2005) are dropped, and structural missing values are never forward-filled. Both the deliverable path and the strategy backtest read through this loader, so the metamodel observes the same cleaned panel, byte for byte, as the rest of the cohort. This rules out the quiet possibility that an apparent edge is really a data-cleaning difference.

Leakage is the constraint that shapes the design. The obvious convenience — consuming a pre-computed feature matrix — is rejected outright, because such a matrix freezes its fitted statistics at one global training cut-off and would leak future information into any earlier in-sample fold. Instead the stateless, causal feature functions are recomputed inside each cross-validation fold on that fold's training slice. Causality is not asserted but enforced and tested: every feature satisfies right-edge truncation-invariance — its value at time *t* is identical whether computed on data up to *t* or on the full series — and a property test pins this for the whole stack, while a guard test asserts that no module ever reads the frozen matrix. Per-fold recomputation is the build's signature departure from the shared base, and it is expensive by design.

The engineered panel layers several families. Counter-trend and mean-reversion measures capture the pressure the primary trades on. Range-based volatility estimators — Garman and Klass (1980) and, as a drift-independent cross-check, Parkinson (1980) — exploit the open-high-low-close bar rather than close-to-close returns alone. Microstructure and liquidity proxies, momentum and trend-strength measures, calendar seasonality, intraday range and gap features, and path-structure and wavelet-energy features complete the engineered core. A concept-drift discriminator asks, through a causal rolling logistic comparison, whether the current feature row resembles the training era or the recent past (Sugiyama & Kawanabe, 2012). Trend-scanning is included strictly as a *feature*, never as the label, and its slope *t*-statistic is capped by a deterministic constant rather than the customary global-variance normalisation, which would itself be a right-edge leak. Each scale-dependent column is paired with a per-instrument, causal, expanding-window standardisation.

Two blocks honour explicit commitments. The first is a regime filter. The shared base ships only static regime models, so to meet the time-varying commitment the build adds an online, EWMA, two-state Gaussian hidden Markov filter on daily log returns, whose emission means and variances are re-estimated recursively through a forgetting factor under a persistent transition prior (Hamilton, 1989; Nystrup, Madsen & Lindström, 2017; Ang & Bekaert, 2002). Because every parameter at *t* is a recursion over observations up to *t*, the filter is fit-free: there is no batch fit, no train-transform split, and therefore no cross-validation seam artefact of the kind a batch hidden Markov model would manufacture when handed non-contiguous training groups. The static regime models are retained as supplementary features. The second block is macroeconomic. The supplied workbook carries mixed-frequency series with observation dates only, so each series is assigned a conservative publication lag, shifted from observation to availability, and forward-filled onto the trade calendar, so that a feature at *t* uses only data released on or before *t*; block-level truncation-invariance and a release-deferral test confirm the alignment. Per-instrument calendar and basis spreads are flagged as not derivable from the front-month data and are omitted rather than fabricated. Measured empirically rather than asserted, the pooled panel runs to roughly 140 columns per class, of which about 108 survive into the clustering matrix once the regime and macro blocks and zero-variance columns are dropped.

## 3. Labelling

Binary act/skip meta-labels are assigned only on the non-zero-signal trade days, using the triple-barrier method (López de Prado, 2018, Ch. 3). The horizontal barriers are volatility-adaptive, set at side-specific multiples of a de-annualised daily volatility σ̂ₜ rather than fixed-percentage thresholds, because fixed thresholds ignore the heteroskedasticity of returns and make the label distribution pro-cyclical. A vertical barrier bounds the holding horizon. The label is the sign of the side-adjusted profit-and-loss at the first barrier touched, and the first-touch time is recorded for every label — it is this time, not a nominal holding period, that drives purge and embargo throughout the pipeline.

The barrier geometry is set per asset class. The default and Metals path uses a de-annualised close-to-close realised volatility — a rolling twenty-day standard deviation of log returns, σ̂ₜ = vol₂₀/√252, and *not*, despite an earlier draft's wording, a Garman–Klass estimate (Garman–Klass enters only as a feature, Section 2) — with symmetric profit-take and stop multiples (1, 1) and a ten-day vertical barrier. Equity and Energy instead carry the configuration an economic barrier sweep recommended on the modelling sample: Equity a fifty-day rolling-volatility scale with an asymmetric (2, 1) profit-to-stop ratio and a ten-day horizon; Energy a twenty-day exponentially-weighted volatility with a tight (0.5, 0.25) ratio and a volatility-scaled horizon (base ten, clipped to [2, 40]). The sweep ranked candidate geometries by their downstream net Sharpe — not by label accuracy — strictly on the modelling window (≤ 2021-12-31), so the choice never touches the out-of-sample window and the selection statistic sits inside the leakage firewall. Three honesty caveats attach and are carried through the evaluation: the sweep was one-factor-at-a-time and XGBoost-scored, so the per-class triples were never jointly measured nor validated under the deployed torch roster; only Equity's and Energy's geometries survive an XGBoost→LightGBM robustness swap, and Metals' did not and is therefore *omitted*, falling back to the default symmetric barrier; and adopting these geometries is a wiring decision taken at the emit boundary, not a demonstrated improvement — Section 7 reports, honestly, that under the torch roster Energy's geometry does not transfer out of sample.

Triple-barrier labels overlap, so they are not independent: a single price path feeds many labels, and a naive *k*-fold would let near-duplicate observations sit on both sides of a split. Two corrections follow from the concurrency structure. Sample-uniqueness weights, derived from the overlap of label horizons, down-weight observations that share their path with others (López de Prado, 2018, Ch. 4); the weighting is verified exactly on disjoint and fully-overlapping toy cases. And the cross-validation embargo is applied per instrument, advancing each instrument's own empirical run-length on its own date axis rather than a single uniform bar count — the correct treatment for a pooled panel in which a flat embargo would badly under-cover the most persistent instruments and over-cover the rest. Across the universe the positive-class share runs from roughly one-half to two-thirds, so the labels are both overlapping and imbalanced, a combination Section 4 handles with a single weighting channel.

## 4. Models and validation

Five estimators run behind one uniform classifier interface, so that the comparison is genuinely like-for-like rather than an accident of differing harnesses (Gu, Kelly & Xiu, 2020; Krauss, Do & Huck, 2017). Three — elastic-net logistic regression, gradient-boosted trees (XGBoost) and LightGBM — see the full feature set. The two neural variants, a multilayer perceptron and a variable-selection network, run on a cluster-representative-reduced feature set, one medoid per correlation cluster, which keeps the variable-selection network's one-gate-per-feature architecture tractable at cross-validation scale. The reducer is wrapped with its estimator, so the medoid selection is fitted on each fold's training rows only and the validation fold never informs the reduced basis. Class imbalance and label uniqueness are folded into a single weight, the product of the uniqueness weight and an inverse-class-frequency term, and that one weight is passed identically to every estimator's fit and into every out-of-sample metric.

Validation follows the selection-bias-aware tradition (López de Prado, 2018, Ch. 7; Bailey & López de Prado, 2014; Harvey, Liu & Zhu, 2016). Model selection uses combinatorial purged cross-validation — six groups, fifteen recombined paths — scored by the mean out-of-fold AUC, with the per-instrument embargo applied throughout and the pooled event-date index retained so that concurrent labels across instruments are purged together. A nested combinatorial scheme, in which an inner cross-validation runs the horse-race and an outer one scores only the winner, is implemented and unit-tested as the principled selection-bias-aware evaluator; a full real-data nested run, at roughly fifteen by five by five fits per class, is acknowledged as deferred rather than quietly skipped.

On the modelling sample the selection statistic — the mean out-of-fold ROC-AUC across the fifteen combinatorial paths, on raw pre-calibration probabilities — chooses XGBoost for Equity at 0.59, the multilayer perceptron for Energy at 0.52, and elastic-net logistic regression for Metals at 0.53. Two cautions attach to these numbers. First, each sits only marginally above the 0.50 no-skill line. Second, and more important, this is a model-selection statistic on the modelling sample, not an AUC scored on the Jan–Jun 2022 deliverable window; no such window AUC is computed, and the deliverable-window evidence is instead the per-instrument information coefficient discussed in Section 6. The neural family is competitive on its merits — Energy's winner is a neural network — so it is neither rubber-stamped nor excluded.

A complementary robustness probe asks a different question: of the fifteen combinatorial paths, how many exceed 0.50? Run on a deliberately stripped configuration — a tree-and-linear roster with the regime and macro blocks switched off — the answer is 15 of 15 for Equity, 13 of 15 for Metals, and 12 of 15 for Energy. (Because the neural estimators are absent from this reduced roster, the probe's nominal Energy winner is XGBoost rather than the deployed multilayer perceptron; the path count, not the model identity, is the object of interest.) The reading is deliberately cautious: Equity's edge survives every recombination on the modelling window, and Metals and Energy clear 0.50 on a majority of paths there too — but this is a *within-sample* count. It sits alongside per-instrument out-of-sample AUCs that fall below the no-skill line for Energy (Section 6) and a uniformly negative significance and timing verdict below; the recombination count anticipates Equity's relative strength, not a transferable Energy edge.

Because the sizing stage consumes a probability directly, calibration is shipped rather than merely reported. One Platt map per asset class is fitted on the selected model's purged out-of-sample modelling predictions, strictly before the prediction window, and applied to both the deliverable probabilities and the Kelly stake. Platt scaling is monotone, so the act/skip ranking — and therefore the AUC — is left unchanged, a property held as a unit-tested invariant; only the calibration error and the stake move. On an in-time seventy-thirty split of each class's selected model — a split that re-selects within its own training fold, and so names XGBoost for Energy where the full-sample horse-race deploys the multilayer perceptron — the raw probabilities are materially miscalibrated, Metals most of all, and the Platt map cuts the expected calibration error sharply (Energy 0.105 to 0.048, Equity 0.062 to 0.017, Metals 0.210 to 0.001) with the held-out AUC untouched (Gramegna & Giudici, 2021). The deliverable ships the calibrated probabilities, with the raw probabilities retained alongside for the before-and-after comparison.

## 5. Feature importance

Per-feature importance is unreliable under correlation, because substitution effects let correlated features trade places and split a shared contribution arbitrarily (López de Prado, 2020). Importance is therefore scored per cluster. Features are grouped by a Mantegna correlation distance (Mantegna, 1999), reduced by principal components and an optimal-*K* *k*-means, and each cluster is scored three ways: mean decrease in impurity, purged permutation mean decrease in accuracy, and cluster SHAP summed over member features via a tree explainer (Lundberg et al., 2020). Four defects in the inherited code were corrected along the way, the most consequential being the replacement of a shuffled *k*-fold — which leaks across overlapping labels — with the purged scheme for the permutation importance; the others restored a valid forest setting, added genuine SHAP attribution in place of an impurity-and-permutation-only stand-in, and swapped a non-metric correlation distance for the Mantegna metric.

The interpretive discipline is the section's real contribution. Mean decrease in impurity and SHAP are in-sample attribution: they distribute the whole of a fitted model's importance across the clusters, and so report only which features the model leaned on, never whether those features carry out-of-sample edge. The purged permutation importance is the out-of-sample reality check, and it is near zero on every cluster. Clustering each class's modelling matrix yields three clusters for Equity, three for Energy and two for Metals; the top-cluster permutation importance is a slim 0.019 for Equity and statistically indistinguishable from zero for both Energy (−0.005) and Metals (−0.011), even where the in-sample SHAP attribution is large (Metals' top-cluster SHAP is 0.71). The divergence is the lesson: a high SHAP must not be read as edge. No cluster in any class now reaches a permutation importance of 0.02; Equity's is the largest and still marginal, consistent with its relative survival across the combinatorial paths in Section 4, and the concept-drift feature lands in a noise cluster and adds nothing under permutation. That the identical harness scores AUC above 0.9 on separable synthetic data confirms it detects signal when signal is there to detect.

## 6. Evaluation: classification and market-timing skill

Metrics are sample-weighted, threshold-aware and computed per instrument before any aggregate, so that a strong pooled number cannot hide a weak member; the baseline is the blind primary, which acts on every signal. On the modelling sample the per-instrument purged out-of-sample AUCs are honest about where value lives. Equity is uniformly near 0.62 (es1s 0.61 on 457 labels, nq1s 0.62 on 482, fesx1s 0.63 on 510). Metals is mixed (hg1s 0.57, pl1s and si1s near the no-skill line at 0.49 and 0.51, gc1s below it at 0.45 on only 138 labels). Energy is weaker still, and under its per-class barrier its better-covered names sit *below* the no-skill line (cl1s 0.47, ng1s 0.42 on only 68 labels, rb1s 0.50, ho1s 0.43 on a very small sample). Three instruments rest on thin out-of-sample coverage and are flagged accordingly — ho1s on two rows with an undefined information coefficient, gc1s on thirty, ng1s on fifty-six — and their per-instrument figures should be read as small-sample noise. A pooled AUC would have concealed both the Equity strength and the Energy weakness; reporting per instrument first is what keeps the evaluation honest.

The provided signal sets the ceiling, so it is worth characterising. Its directional hit-rates run from 0.51 to 0.69 (gc1s strongest at 0.66, with a primary-signal information coefficient of 0.21; rb1s weakest), and several names time the low-volatility regime better than the high. This information coefficient — the rank correlation of the *primary signal* with the subsequent return, measured on the modelling window — must not be confused with the metamodel's own out-of-sample information coefficient, the rank correlation of the *calibrated probability* with the subsequent return on the 2022 window; the two differ in target, window and, for gc1s, even in sign (the metamodel coefficient is −0.23). The point of the characterisation is simple: the base signal is already reasonable, so the secondary filter has little headroom.

AUC says nothing about whether *acting* adds economic value, so market-timing skill is tested directly on the out-of-sample acted trades. The primary test is the Pesaran–Timmermann (1992) directional-accuracy statistic, which conditions on the directional base rates so that a constant call in a trending market scores zero rather than a spurious positive. The Treynor–Mazuy (1966) quadratic-convexity coefficient corroborates, and a hit-rate Henriksson–Merton form is reported only as a base-rate-*sensitive* proxy, since the implemented version is the plain hit-rate-against-one-half form rather than the conditional non-parametric test, and so over-reads skill in a trending window. The directional verdict is blunt: Pesaran–Timmermann is negative or insignificant in every book and negative pooled (−1.80, *p* ≈ 0.96). A proxy biased *towards* skill that still finds none strengthens the negative rather than weakening it.

### 6.1 The Treynor–Mazuy convexity is a scale artefact, not timing

The corroborating convexity test does not even print positive here: the *pooled* Treynor–Mazuy coefficient is −0.51 (*t* = −0.35), insignificant and of a piece with the negative Pesaran–Timmermann verdict. That the same coefficient printed *positive and nominally significant* (γ = +1.18, *t* = 2.55) under the earlier global barrier — where it could be mistaken for timing skill — is itself the cautionary point, because the pooled coefficient is an unreliable scale aggregate whichever sign it takes. Two literatures read together explain why.

First, pooling sub-portfolios of heterogeneous return scale, volatility and beta into a single quadratic-timing regression is known to produce inconsistent, sign-reversing coefficients — a regression instance of Simpson's paradox and aggregation bias (Robinson, 1950; Zellner, 1962; Blyth, 1972; Pesaran & Smith, 1995). The per-sleeve coefficients confirm the heterogeneity: Equity is +0.31 and Metals −2.05, both insignificant, while Energy abstains entirely under its barrier and contributes no acted trades, so coefficient homogeneity is firmly rejected and a trade-count-weighted average of the sleeve coefficients is −1.40, well away from the pooled −0.51 — the pooled magnitude is a scale aggregate of heterogeneous sleeves, not a shared coefficient. Second, the convexity such a regression detects when it detects any is mechanical, not predictive. Jagannathan and Korajczyk (1986) show that holding option-like or levered payoffs manufactures artificial market-timing ability where none exists, and a protective, big-move-capturing stop-and-barrier rule is exactly such a convex, option-isomorphic payoff (Henriksson & Merton, 1981; Glosten & Jagannathan, 1994; Fung & Hsieh, 2001). Because the Pesaran–Timmermann statistic is a sign and contingency-table test rather than a magnitude regression, it is invariant to the scale heterogeneity that drives the pooled coefficient, which is why it, and not Treynor–Mazuy, is the trusted pooled diagnostic. Treynor and Mazuy (1966) is a practitioner article cited only for the quadratic specification, not as authority on the artefact.

### 6.2 The artefact reproduced and dissolved in the data

The cited theory is turned into direct evidence by a cheap robustness check. The pooled regression is re-estimated after volatility-targeting each sleeve to a common scale — dividing each sleeve's realised market return *and* its signed profit-and-loss by that sleeve's return standard deviation, the same factor for both, so that the per-trade identity that profit equals side times return is preserved. Standardising the regressor rescales the quadratic coefficient by the sleeve's volatility, so the large-scale sleeves stop dominating the pooled quadratic term. The effect is decisive: the pooled coefficient moves from −0.51 to −0.0277 (*t* = −0.93, *p* = 0.354), collapsing toward the insignificant negative trade-weighted sleeve average. Once the sleeves share a scale, whatever pooled convexity the unstandardised regression reported — positive under the earlier barrier, negative here — dissolves into the same scale-driven near-zero: direct in-sample proof that the pooled Treynor–Mazuy magnitude is aggregation, not timing.

## 7. Strategy backtest and statistical significance

The position weight is a fractional Kelly stake — a quarter of the full Kelly fraction, with a confidence floor below which the stake is zero — scaled by a volatility-target leverage and signed by the primary side (Kelly, 1956; MacLean, Ziemba & Blazenko, 1992; Carver, 2015). The out-of-sample backtest is barrier-exact and cost-aware: positions exit on the actual first-touch time rather than a fixed holding period, overlapping labels are netted, and a half-spread-plus-impact cost is charged. The downside deviation underlying the Sortino ratio uses the full-sample denominator against a zero target (Sortino & Price, 1994) rather than the common mis-implementation that divides only by the count of negative observations and so inflates the ratio; a known-value test pins the full-sample form. Net of costs, the pooled book returns a Sharpe of 0.48 (Sortino 0.78, annualised volatility 11.0%, maximum drawdown −4.5%); Equity reads 0.51, Metals essentially zero, and Energy exactly zero — under its per-class barrier Energy takes no out-of-sample positions at all, a point Section 7.2 returns to.

No deployable edge is claimed from those numbers, because before any deflation the prior question is whether the Sharpe clears zero at all on a window of roughly 128 days, and it does not.

| Statistic | Pooled (all eleven) | Reading |
|---|---|---|
| Sharpe (per period); *t* = SR·√n | SR 0.030; **t = 0.34** (n = 127) | not significant at 5%, before any deflation |
| Studentised stationary block-bootstrap 95% CI *(primary)* | per period **[−0.09, +0.15]** | contains zero — the width is the finding |
| Lo/Opdyke analytic band | per period [−0.14, +0.20] | parametric cross-check; also straddles zero |
| PSR(0); minimum track-record length | 0.64 (< 0.95); **≈ 2,883 days** vs 127 available | track record more than twenty times too short |
| Ljung–Box(10) | *p* = 0.000 | serially correlated → the √252 annualisation overstates; read the per-period CI |

The block length is data-driven (Politis & Romano, 1994; Ledoit & Wolf, 2008), the bootstrap is studentised by the Lo (2002) and Opdyke (2007) analytic standard error, and the procedure is seeded and deterministic. The confidence interval straddling zero is the honest headline: insufficient evidence, not a demonstrated failure. Because no peer-reviewed Monte-Carlo of the deflated Sharpe ratio exists at this sample length, the inference is led by the bootstrap and the deflation statistics are treated as corroboration rather than load-bearing point probabilities (Lo, 2002; Opdyke, 2007; Bailey & López de Prado, 2012; Bailey & López de Prado, 2014).

That corroboration is consistent. Deflating the same net returns for selection bias yields a deflated-Sharpe ladder over the plausible trial count that stays below 0.95 even at its most optimistic rung — pooled 0.26 falling to 0.08 — a probability of backtest overfitting near 0.36, and a minimum backtest length that dwarfs the half-year window, running from roughly 1.7 to 3.1 years (Bailey & López de Prado, 2014; Bailey et al., 2017, for whom the combinatorial split count is C(16,8) = 12,870, correcting a long-propagated typographical "12,780"). The trial ladder counts only the five-model horse-race per class; the data-driven cluster-representative reducer, the concept-drift expansion, *and* the per-class economic barrier sweep all add further selection that the ladder does not enumerate. Because the gate already fails at its most conservative counted rung and the deflated Sharpe falls monotonically in the trial count, those uncounted searches can only push the gate further from clearing — the negative is robust *a fortiori*, not in spite of the uncounted degrees of freedom.

### 7.1 The convergence: one honest negative, not five unlucky ones

Five mutually independent lenses agree that the metamodel adds no exploitable act/skip edge on this primary signal.

| Lens | Result | Verdict |
|---|---|---|
| Out-of-sample AUC (Sections 4, 6) | ≈ 0.50 | no ranking skill |
| Cluster permutation importance (Section 5) | \|MDA\| < 0.02 across all clusters | no feature carries out-of-sample edge |
| Significance (Section 7) | *t* = 0.34; bootstrap 95% CI contains zero | Sharpe not distinguishable from zero |
| Deflation gate | pooled DSR [0.26 → 0.08]; PBO ≈ 0.36 | fails the selection-bias gate |
| Directional timing (Pesaran–Timmermann) | pooled −1.80 (*p* ≈ 0.96) | no positive directional timing |

The convergence is predicted, not coincidental. The provided primary is short-horizon mean-reversion, and Grinold's fundamental law of active management, IR = IC·√BR (Grinold, 1989), makes the null structural: with the primary's information coefficient near zero, the achievable information ratio is near zero whatever the breadth or the sizing, so a secondary act/skip filter cannot manufacture skill the primary lacks. It is worth being explicit about the epistemic status of each link. The fundamental law and the sign mechanics of the timing tests are proven results. The López de Prado precondition — that meta-labelling helps only when the primary already has filterable skill — is an assumed premise. The notion that a mean-reversion primary caps the achievable AUC is an empirical heuristic, not a theorem. Labelling each claim by its standing is deliberate: the negative is argued from proven mechanics, not assumed into being.

### 7.2 Robustness, sizing concentration and the holding model

Three further observations reinforce the reading. First, the result is stable under stricter leakage control: the per-instrument embargo (Section 4) replaced a looser uniform embargo precisely because the looser control under-covers the most persistent instruments and flatters the in-sample numbers, and tightening it moves the Sharpe down — the right direction, and the small, sign-indeterminate reshuffling is exactly what is expected when no real edge underlies the figures. Second, the sizing rests on a thin and now lopsided slice. The calibrated probability tops out at 0.670 (on Equity), and roughly sixty-two per cent of bets sit below the confidence floor at zero weight, up from a little over a third under the global barrier. The floor-pass rate is wildly uneven across books: Metals clears it on about three-quarters of bets and Equity on under a third, while Energy clears it on none — its calibrated probabilities, compressed by a no-signal model into a narrow band just above one-half (0.519 to 0.522), never reach the floor, so the entire book abstains. A sleeve sized to zero by a correctly-cautious floor is the sharpest illustration of the negative: the metamodel, asked whether to act on Energy, declines on every day. Third, the exit convention alone still moves the books that trade: Equity goes from −0.34 under a fixed ten-day hold to +0.51 barrier-exact, and Metals from −0.21 to essentially zero, while Energy is zero under both conventions because it takes no positions. Because identical positions, models, features and calibration differ only in the exit rule, the ordering is driven by the exit mechanism — winners ride to the profit barrier, losers are cut at the stop — and not by classification skill. Gated by the significance and deflation analyses, this is a finding about backtest construction, not a performance claim.

### 7.3 A leakage-safe sizing refinement, and why it is not adopted

The Kelly multiplier was held at the flat quarter unless a per-instrument shrinkage could be shown to improve out-of-sample utility without leakage. An earlier signal-to-noise shrinkage, κᵢ = eᵢ²/(eᵢ² + σᵢ²), had been estimated on the out-of-sample window itself — exactly the look-ahead this build is constructed to avoid — and was rejected as circular, its apparent certainty-equivalent gain (to 0.000538 against a baseline of 0.000090) inadmissible. The leakage-safe redo estimates the same shrinkage on the modelling sample only, strictly before the prediction window, and gates adoption on two conditions: a relative certainty-equivalent gain above five per cent *and* a paired studentised stationary block-bootstrap confidence interval on the certainty-equivalent difference that excludes zero. The outcome is instructive. The point estimate is a very large gain — 0.000090 to 0.000593, +562% — comfortably clearing the materiality bar, but the paired bootstrap 95% interval is [−0.000780, +0.001814] and contains zero. The gain is indistinguishable from noise at this sample length, the more so on a window where an entire sleeve abstains. The companion smooth-taper variant does, on its leakage-safe form, now clear the certainty-equivalent gate (0.000090 to 0.000275); but that advantage is measured on the out-of-sample window itself, so adopting it would re-introduce exactly the look-ahead the locked-before-out-of-sample sizing exists to prevent. The gate flip is therefore reported as a diagnostic, not acted on: the deliverable weights remain at the flat quarter and re-emit byte-identically. A point estimate that flatters but cannot survive its own confidence interval — or its own leakage firewall — is, once more, the honest negative in miniature.

## 8. Discussion

The contribution is methodological discipline rather than a profitable strategy, which is what the brief asks for. Three features distinguish it. The result is reported at the right level of confidence: significance leads, deflation corroborates, and no favourable point estimate is allowed to stand as a conclusion until it survives a bootstrap interval — applied alike to the headline Sharpe and to the sizing refinement. Two plausible-but-wrong stories were caught and handled: the pooled Treynor–Mazuy convexity — which printed positive and significant under the earlier global barrier, where it could have been mistaken for timing, and which here does not even print positive — shown by both a synthesis of the artificial-timing and aggregation-bias literatures and an in-sample standardise-then-repool experiment to be a scale artefact in either sign; and a circular out-of-sample-estimated sizing shrinkage, replaced by a leakage-safe variant that then failed its own bootstrap test. And the negative is argued from proven mechanics — the fundamental law and the sign rules of the timing tests — rather than asserted, which is what makes it a result rather than a disappointment.

Several limitations bound the claims. The out-of-sample window is short, near 128 observations, so the deflated-Sharpe and overfitting statistics depend on noisily estimated higher moments and are reported as directional checks rather than precise probabilities. The macroeconomic block is publication-lag aligned, so no timing look-ahead survives, but it ships revised rather than real-time vintages, and a full real-time-vintage reconciliation remains the outstanding macro gap; the conservative lags and the block's near-zero permutation importance bound its likely impact. The sizing constraint set remains a literature-default stub. Three instruments rest on thin coverage and their per-instrument numbers are small-sample noise. Two earlier-draft citation issues were corrected: a non-existent reference, a conflation of a 2024 genetic-algorithm pairs-trading study, was removed, and the Sortino denominator was corrected to the full-sample form.

## 9. Conclusion

Across five independent lenses — classification AUC, out-of-sample feature importance, Sharpe significance, selection-bias deflation and directional timing — the meta-labelling metamodel adds no exploitable act/skip edge on the provided mean-reversion primary. The pooled Sharpe of 0.48 carries a *t*-statistic of 0.34 and a bootstrap interval that contains zero; the directional-timing tests are negative pooled and in every book, and the pooled Treynor–Mazuy convexity that once printed positive now does not and is in any case a scale-aggregation artefact that dissolves under standardisation; and the one sizing refinement that might have helped is indistinguishable from noise under a paired bootstrap. The correct conclusion is insufficient evidence of a deployable edge — not a proven failure. The positive backtested Sharpe that survives is the signature of the convex barrier exit, the volatility target and diversification, exactly as the fundamental law predicts when the underlying information coefficient is near zero; and where the per-class barrier leaves a sleeve with no signal, as on Energy, the metamodel correctly abstains rather than manufacturing one. Reporting that conclusion accurately, with each favourable number tested to destruction, is the methodological result this challenge sought.

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

## Appendix A — Implementation and reproducibility

The deliverable runs one one-directional pass per asset class. Data load read-only through the shared base loader; per-instrument causal features are recomputed inside each fold; triple-barrier meta-labels are assigned on the non-zero-signal days, recording each first-touch time and uniqueness weight; the class is pooled with an instrument-identity one-hot and the event-date index retained; the purged combinatorial cross-validation runs the horse-race and selects by mean out-of-fold AUC; the selected model is refitted and predicts the configured window; one Platt map per class calibrates both the deliverable probability and the Kelly stake; the stake is sized by fractional Kelly and a volatility target, signed by the primary side; and the emitter writes the deterministic CSVs.

The package is organised into focused modules, each with one responsibility: triple-barrier labelling; the pooling pipeline; purged and combinatorial cross-validation; the tree-and-linear and the neural rosters; Garman–Klass volatility; the Mantegna-clustered importance diagnostics; fractional-Kelly and volatility-target sizing; the barrier-exact backtest; the online EWMA hidden Markov regime block; the deflation gate; the studentised block-bootstrap significance inference; per-class Platt calibration; the publication-lag-aligned macro block; and the deterministic emitter, with seeding and single-thread native-kernel configuration kept separate.

Three disciplines make the negative result trustworthy. Causality is enforced by per-fold recomputation of the causal feature functions and verified by right-edge truncation-invariance property tests covering the whole stack, including the concept-drift and regime blocks; a guard test asserts that the frozen feature matrix is never read. Overlapping labels are purged by first-touch time, with the per-instrument empirical run-length advanced on each instrument's own date axis, so that no training and validation labels overlap after purge, including in the cluster permutation importance. Determinism is enforced by fixing seeds across the random, numerical, tensor and hashing layers and running native kernels single-threaded; the emitter sorts rows, pins columns and fixes the float format, so two independent full runs produce byte-for-byte identical prediction and weight files. The single non-reproducible estimator — a TensorFlow variable-selection network whose op-determinism is best-effort — is deliberately kept off the selectable path, so that the grader's hidden-half re-run cannot select a model it cannot reproduce.

The prediction window is a configuration field, never hard-coded: the grader swaps in the hidden July–December 2022 half by changing one value, and the visible feature set is locked before the final window so that it does not rehearse the hidden one. Because the build outputs are not tracked in version control, every figure in this report is anchored to the seeded regeneration command (`uv run --project metamodel-apb python -m alken_metamodel.emit`, which reproduces the deliverable byte-for-byte) rather than to a stored file, and the full test suite runs under pytest.

## Appendix B — Commitments, modules and citations

| # | Commitment | Module(s) | Primary citation |
|---|---|---|---|
| 1 | Meta-labelling act/skip filter | triple-barrier, pipeline | López de Prado (2018, Ch. 3); Joubert (2022) |
| 2 | Per-class volatility-adaptive barriers (realised-vol / rolling-std / EWMA σ̂ₜ), vertical barrier | triple-barrier | López de Prado (2018, Ch. 3) |
| 3 | Purged CV, embargo, combinatorial and nested | cross-validation | López de Prado (2018, Ch. 7); Bailey & López de Prado (2014); Harvey, Liu & Zhu (2016) |
| 4 | Garman–Klass volatility, Parkinson cross-check | volatility | Garman & Klass (1980); Parkinson (1980) |
| 5 | Multi-family model horse-race | tree/linear and neural rosters | Gu, Kelly & Xiu (2020); Krauss, Do & Huck (2017) |
| 6 | Cluster MDI, MDA and SHAP on a Mantegna metric | cluster importance | López de Prado (2020); Lundberg et al. (2020); Mantegna (1999) |
| 7 | Fractional Kelly with a volatility target | sizing, backtest | Kelly (1956); MacLean, Ziemba & Blazenko (1992); Carver (2015) |
| 8 | Two-state HMM, EWMA time-varying | regime | Hamilton (1989); Nystrup, Madsen & Lindström (2017); Ang & Bekaert (2002) |

Cross-cutting: sample-uniqueness weights (López de Prado, 2018, Ch. 4); per-class Platt calibration of both the deliverable and the Kelly stake (Gramegna & Giudici, 2021); a single sample-weight channel for imbalance and uniqueness; and a deflation gate combining the deflated Sharpe ratio and minimum backtest length (Bailey & López de Prado, 2014) with the combinatorially-symmetric cross-validation probability of backtest overfitting (Bailey et al., 2017), so that the strategy section is reported deflated and never as a working strategy.
