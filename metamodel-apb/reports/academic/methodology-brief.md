# A meta-labelling metamodel for a multi-asset futures universe: a brief

**Module:** T3.03 — Systematic Trading Strategies with Machine Learning (Alken team challenge). **Scope:** a secondary *act/skip* classifier over a provided primary signal, across eleven futures instruments in three asset-class books (Equity, Energy, Metals). **Assessment stance:** methodology, not performance.

---

## Abstract

A meta-labelling metamodel was built to decide, on each day a provided primary signal is non-zero, whether to act on or skip the directional call. This brief states the verdict first and then explains why it is the structurally expected one. On the six-month out-of-sample window the pooled net Sharpe ratio of 1.31 is not distinguishable from zero (*t* = 0.93, n = 127; studentised block-bootstrap 95% interval [−0.04, +0.19], containing zero), and four further independent diagnostics agree. The honest conclusion is insufficient evidence of a deployable edge — not a proven failure. The positive backtested Sharpe is the footprint of a convex barrier exit, volatility targeting and diversification, not of act/skip skill.

## The verdict

Meta-labelling separates the *direction* of a trade, set by the primary, from its *size*, set by a secondary gate; the gate can only help where the primary already has filterable skill (López de Prado, 2018; Joubert, 2022). The primary here is short-horizon mean-reversion, and its own predictive content is slight — so the structural expectation, before any backtest, is that a secondary filter cannot manufacture an edge. It does not.

Five mutually independent lenses converge on the same null.

| Lens | Result | Verdict |
|---|---|---|
| Out-of-sample AUC | ≈ 0.50 (selection-mean 0.57 / 0.54 / 0.53) | no ranking skill |
| Cluster permutation accuracy | \|MDA\| < 0.02 across clusters bar one | no feature carries out-of-sample edge |
| Sharpe significance | *t* = 0.93; bootstrap 95% CI contains zero | Sharpe not distinguishable from zero |
| Selection-bias deflation | pooled deflated Sharpe [0.61 → 0.20]; PBO ≈ 0.35 | fails the deflation gate |
| Directional timing (Pesaran–Timmermann) | pooled −2.31 (*p* ≈ 0.99) | no positive directional timing |

That this is one honest negative rather than five unlucky tests is the point. Grinold's fundamental law, IR = IC·√BR (Grinold, 1989), makes it structural: with the primary's information coefficient near zero, the achievable information ratio is near zero whatever the breadth or the sizing. The fundamental law and the sign mechanics of the timing tests are proven; the precondition that meta-labelling needs filterable primary skill is an assumed premise; the notion that a mean-reversion primary caps the achievable AUC is an empirical heuristic. The negative is argued from the proven parts, not asserted.

Classification bears this out instrument by instrument. Only Equity adds clear value — its three names sit near an AUC of 0.60 — while Metals and Energy are mixed and weighed down by small-sample contracts (gc1s, ho1s and ng1s, each flagged for thin out-of-sample coverage of fewer than sixty rows or an undefined information coefficient). Reporting per instrument before any aggregate is deliberate: a single pooled AUC would have concealed both the Equity strength and the Energy weakness, and the metamodel's own out-of-sample information coefficient — the rank correlation of its calibrated probability with the subsequent return — is itself near zero across the universe.

## Method in brief

Labels are binary act/skip meta-labels, assigned on the non-zero-signal days by the triple-barrier method (López de Prado, 2018): symmetric, volatility-adaptive barriers at ±*k*·σ̂ₜ, with σ̂ₜ a de-annualised Garman–Klass volatility (Garman & Klass, 1980), and a vertical time barrier. Each label records its first-touch time, which drives purge and embargo, and sample-uniqueness weights down-weight the overlapping horizons that make the labels non-independent. The embargo is applied per instrument, on each instrument's own date axis.

Leakage is the governing constraint. The causal feature functions are recomputed inside every cross-validation fold rather than read from a pre-computed matrix that would freeze fitted statistics and leak across the training cut-off; right-edge truncation-invariance is enforced and tested. The feature set spans counter-trend, range-volatility, microstructure, momentum, path-structure and concept-drift families, plus a fit-free online EWMA hidden Markov regime filter (Hamilton, 1989; Nystrup, Madsen & Lindström, 2017) and a publication-lag-aligned macroeconomic block.

Five estimators — elastic-net logistic regression, XGBoost, LightGBM and two byte-deterministic neural variants — run behind one interface and are compared under combinatorial purged cross-validation, selected by the mean out-of-fold AUC on the modelling sample (Gu, Kelly & Xiu, 2020). Because the sizing stage consumes a probability directly, a per-class Platt map is fitted train-only and applied to both the deliverable probability and the stake; being monotone, it leaves the AUC unchanged and only corrects the calibration (Gramegna & Giudici, 2021). Feature importance is scored per cluster, on a Mantegna correlation metric (Mantegna, 1999), by impurity, purged permutation accuracy and SHAP (López de Prado, 2020; Lundberg et al., 2020); the divergence between large in-sample SHAP and near-zero out-of-sample permutation accuracy is the section's lesson — a high SHAP is not edge. Positions are sized by fractional Kelly with a volatility target, signed by the primary side (Kelly, 1956; Carver, 2015), and the backtest is barrier-exact and cost-aware, exiting on the actual first-touch time with the Sortino ratio computed on the full-sample denominator (Sortino & Price, 1994). The setting throughout is small-data and low-signal, where conventional thresholds are too lenient under multiple testing (Israel, Kelly & Moskowitz, 2020; Harvey, Liu & Zhu, 2016).

## Two favourable results, examined and dismissed

A rigorous negative has to survive its own apparent counter-evidence. Two favourable-looking results were examined and both dissolved.

The first is the *pooled* Treynor–Mazuy convexity coefficient, positive and nominally significant (γ = +1.18, *t* = 2.55), which in isolation could be read as market-timing skill. It is not. Pooling sub-portfolios of heterogeneous scale and volatility into one quadratic-timing regression is a known route to inconsistent, sign-reversing coefficients — a regression instance of Simpson's paradox and aggregation bias (Robinson, 1950) — and indeed the per-sleeve coefficients (Equity −4.51, significant; Energy +0.81 and Metals −2.05, insignificant) reverse the pooled sign, averaging −1.835. What convexity the regression does detect is the mechanical, option-like convexity of a protective stop-and-barrier exit, which manufactures artificial timing where none exists (Jagannathan & Korajczyk, 1986; Treynor & Mazuy, 1966). Standardising each sleeve to a common scale and re-pooling collapses the coefficient from +1.18 to −0.0031 (*t* = −0.14) — direct in-sample proof that it was aggregation, not timing. This is why the scale-invariant Pesaran–Timmermann directional test (Pesaran & Timmermann, 1992), which is negative everywhere, is the trusted diagnostic.

The second is a per-instrument Kelly-fraction shrinkage. Estimated on the out-of-sample window it is circular; re-estimated leakage-safely on the modelling sample its point gain is large (a certainty-equivalent improvement of 92 per cent), but the paired bootstrap interval on that gain contains zero, so it is indistinguishable from noise and not adopted. The deliverable weights stay at the flat fractional-Kelly quarter and re-emit byte-identically.

## Significance, deflation and limitations

The headline Sharpe is read significance-first. The pooled net Sharpe of 1.31 gives *t* = 0.93 on 127 observations, and the primary inference — a studentised stationary block-bootstrap 95% interval of [−0.04, +0.19] per period — contains zero, as does the Lo/Opdyke analytic band; the probability of a positive Sharpe ratio is 0.82, below the 0.95 bar, and the minimum track-record length of roughly 399 days dwarfs the 127 available (Lo, 2002). Deflating the same returns for selection bias corroborates without leading: the deflated-Sharpe ladder stays below 0.95 at every trial count (pooled 0.61 falling to 0.20), the probability of backtest overfitting is near 0.35, and the minimum backtest length runs to years (Bailey & López de Prado, 2014; Bailey et al., 2017).

Across books the net Sharpe is 1.86 for Energy, 0.86 for Equity and essentially zero for Metals, and net of a half-spread-plus-impact cost the pooled book turns a gross 8.3 per cent into 4.9 per cent. That the exit convention alone reorders these — Equity moves from +0.54 under a fixed ten-day hold to +0.86 when positions exit on the actual barrier first-touch — confirms that the ordering is driven by the convex exit mechanism, winners riding to the profit barrier while losers are cut at the stop, rather than by any classification skill.

The claims are bounded honestly. The out-of-sample window is short, so the deflation statistics rest on noisily estimated higher moments and are read as directional checks. The macroeconomic block is publication-lag aligned but ships revised rather than real-time vintages. Three instruments rest on thin coverage and their per-instrument numbers are small-sample noise. The sizing constraint set is a literature-default stub.

## Conclusion

The meta-labelling metamodel adds no exploitable act/skip edge on the provided mean-reversion primary, and five independent lenses say so in unison. The pooled Sharpe is statistically indistinguishable from zero, the one positive timing statistic is a scale artefact that dissolves under standardisation, and the one sizing refinement that might have helped cannot survive its own confidence interval. The correct conclusion is insufficient evidence of a deployable edge, not a proven failure — and the value of the work is that it can say so credibly, because every favourable number was tested to destruction.

---

## References

Bailey, D.H. and López de Prado, M. (2014) 'The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and non-normality', *Journal of Portfolio Management*, 40(5), pp. 94–107.

Bailey, D.H., Borwein, J.M., López de Prado, M. and Zhu, Q.J. (2017) 'The probability of backtest overfitting', *Journal of Computational Finance*, 20(4), pp. 39–69.

Carver, R. (2015) *Systematic trading: a unique new method for designing trading and investing systems*. Petersfield, Harriman House.

Garman, M.B. and Klass, M.J. (1980) 'On the estimation of security price volatilities from historical data', *Journal of Business*, 53(1), pp. 67–78.

Gramegna, A. and Giudici, P. (2021) 'SHAP and LIME: an evaluation of discriminative power in credit risk', *Frontiers in Artificial Intelligence*, 4, 752558.

Grinold, R.C. (1989) 'The fundamental law of active management', *Journal of Portfolio Management*, 15(3), pp. 30–37.

Gu, S., Kelly, B. and Xiu, D. (2020) 'Empirical asset pricing via machine learning', *Review of Financial Studies*, 33(5), pp. 2223–2273.

Hamilton, J.D. (1989) 'A new approach to the economic analysis of nonstationary time series and the business cycle', *Econometrica*, 57(2), pp. 357–384.

Harvey, C.R., Liu, Y. and Zhu, H. (2016) '… and the cross-section of expected returns', *Review of Financial Studies*, 29(1), pp. 5–68.

Israel, R., Kelly, B.T. and Moskowitz, T.J. (2020) 'Can machines "learn" finance?', *Journal of Investment Management*, 18(2), pp. 23–36.

Jagannathan, R. and Korajczyk, R.A. (1986) 'Assessing the market timing performance of managed portfolios', *Journal of Business*, 59(2), pp. 217–235.

Joubert, J. (2022) 'Meta-labeling: theory and framework', *Journal of Financial Data Science*, 4(3), pp. 31–44.

Kelly, J.L. (1956) 'A new interpretation of information rate', *Bell System Technical Journal*, 35(4), pp. 917–926.

Lo, A.W. (2002) 'The statistics of Sharpe ratios', *Financial Analysts Journal*, 58(4), pp. 36–52.

López de Prado, M. (2018) *Advances in financial machine learning*. Hoboken, Wiley.

López de Prado, M. (2020) *Machine learning for asset managers*. Cambridge, Cambridge University Press.

Lundberg, S.M., Erion, G., Chen, H., DeGrave, A., Prutkin, J.M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N. and Lee, S.I. (2020) 'From local explanations to global understanding with explainable AI for trees', *Nature Machine Intelligence*, 2(1), pp. 56–67.

Mantegna, R.N. (1999) 'Hierarchical structure in financial markets', *European Physical Journal B*, 11(1), pp. 193–197.

Nystrup, P., Madsen, H. and Lindström, E. (2017) 'Long memory of financial time series and hidden Markov models with time-varying parameters', *Journal of Forecasting*, 36(8), pp. 989–1002.

Pesaran, M.H. and Timmermann, A. (1992) 'A simple nonparametric test of predictive performance', *Journal of Business & Economic Statistics*, 10(4), pp. 461–465.

Robinson, W.S. (1950) 'Ecological correlations and the behavior of individuals', *American Sociological Review*, 15(3), pp. 351–357.

Sortino, F.A. and Price, L.N. (1994) 'Performance measurement in a downside risk framework', *Journal of Investing*, 3(3), pp. 59–64.

Treynor, J.L. and Mazuy, K.K. (1966) 'Can mutual funds outguess the market?', *Harvard Business Review*, 44(4), pp. 131–136.

---

*Reproducibility.* The pipeline is deterministic: seeds are fixed across the random, numerical, tensor and hashing layers, native kernels run single-threaded, and the emitter sorts rows, pins columns and fixes the float format, so two full runs produce byte-for-byte identical outputs. Leakage is controlled by per-fold causal recomputation (with a guard test against the frozen feature matrix) and purge-and-embargo on first-touch time. The prediction window is a configuration field, so the grader swaps in the hidden July–December 2022 half by changing one value; every figure here is anchored to the seeded command `uv run --project metamodel-apb python -m alken_metamodel.emit`, which reproduces the deliverable byte-for-byte. The fuller treatment is in the companion report and comprehensive methodology.
