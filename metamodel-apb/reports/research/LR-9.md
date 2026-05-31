## Key Findings (summary)

1. **The §5 resolution is literature-backed, not ad hoc.** Two independent, canonical results combine to support it: (a) option-like/convex payoffs generate spurious positive Treynor–Mazuy gamma without forecasting skill (Jagannathan and Korajczyk, 1986; Henriksson and Merton, 1981; Glosten and Jagannathan, 1994; Fung and Hsieh, 2001); and (b) pooling heterogeneous units produces inconsistent, sign-reversing coefficients — Simpson's paradox / aggregation bias (Blyth, 1972; Robinson, 1950; Pesaran and Smith, 1995; Zellner, 1962).
2. **Jagannathan and Korajczyk (1986) is the anchor.** Their abstract states verbatim that "investing in options or levered securities will show spurious market timing" and coins "artificial timing." A barrier-exact stop-loss is an option-like convex payoff, so its positive pooled gamma is the textbook artificial-timing signature, not directional skill.
3. **The pooled-vs-sleeve contradiction is a named phenomenon.** With sleeve coefficients spanning −4.51 to +0.81, coefficient homogeneity is decisively violated, so pooled TM is misspecified (the Zellner aggregation test would reject pooling). The positive pooled gamma is manufactured by between-sleeve dispersion in scale/volatility/beta interacting with the squared-return regressor.
4. **PT is the robust primary pooled test.** As a sign/contingency-table test of directional accuracy, it is invariant to the magnitude/scale heterogeneity that drives the TM artefact. Pooled PT = −2.31 (p ≈ 0.99) agrees with the predominantly negative/insignificant per-sleeve TM coefficients: no genuine timing skill.

## Recommendations (staged, with thresholds)

1. **Lead with the disaggregated result.** Report TM/HM **per sleeve** as the primary table (mean-group philosophy, Pesaran and Smith, 1995) and present pooled gamma only as a flagged artefact. *Threshold that would change this:* if a Zellner/Chow coefficient-equality test across sleeves **failed to reject** homogeneity, pooling would be legitimate and the pooled gamma could be reported as primary — but the −4.51-to-+0.81 spread guarantees rejection.
2. **Make PT the primary pooled timing test**, reported alongside the full per-sleeve TM table. Frame PT and per-sleeve TM as *agreeing* (no timing); the pooled gamma is the outlier explained by aggregation + mechanical convexity. Do not phrase PT as "overruling" a valid TM result.
3. **If a single aggregate TM number is required**, scale-/volatility-normalise sleeve returns before pooling (or volatility-target to a common scale), or estimate a panel with sleeve fixed effects and sleeve-specific slopes. Add Ferson–Schadt (1996) conditioning variables to absorb public-information-driven nonlinearity.
4. **Attribute the mechanism explicitly to Jagannathan and Korajczyk (1986)** for the convexity-without-skill claim and to the Simpson/aggregation literature for the sign reversal — as a *synthesis of two results*, since no single paper makes the combined claim in one sentence.
5. **Run a confirmatory robustness check** (cheap to add): re-estimate pooled TM after standardising each sleeve; if the positive gamma collapses toward the (negative) sleeve-weighted average, that is direct in-sample proof of the aggregation artefact and should be cited in §5.

## Caveats

- **Treynor and Mazuy (1966) is a practitioner-magazine article (HBR) with no aggregation caveat**; cite it only as the origin of the quadratic specification, not as authority on the artefact.
- **No single finance paper states "pooling heterogeneous sleeves manufactures TM convexity"** in those words; the argument is a defensible synthesis of finance (J–K; Glosten–Jagannathan) and econometrics/statistics (Blyth; Robinson; Pesaran–Smith; Zellner). Present it as such.
- **The Jagannathan–Korajczyk full-text page numbers for the worked option example were not independently verified** (paywalled); the abstract wording is safe to quote, but cite a library copy for the worked example. The *Market Timing* survey and Hübner (2012, a non-peer-reviewed working paper) were used only to confirm the mechanism — do not quote them as primary.
- **PT measures directional/sign predictability**, a narrower construct than TM convexity; it can be undefined when all signs coincide, and its power depends on sample size and up/down balance. These do not undermine its use as the robust primary diagnostic but should be stated.
- One aggregator mis-listed PT (1992) as pp. 561–65; the correct pagination is **461–465**.

## Bottom line for §5 (drop-in citable claim)

> The positive pooled Treynor–Mazuy coefficient (γ = +1.18, t = 2.55) is not evidence of market-timing skill but an aggregation artefact compounded by mechanical convexity. First, pooling sub-portfolios with heterogeneous return scales, volatilities and betas into a single quadratic-timing regression is known to yield inconsistent, sign-reversing coefficients — a regression instance of Simpson's paradox and aggregation bias (Blyth, 1972; Robinson, 1950; Pesaran and Smith, 1995; Zellner, 1962) — and here the per-sleeve coefficients (equity γ = −4.51, significant; energy +0.81 and metals −2.05, both insignificant) reject coefficient homogeneity and confirm the reversal. Second, the convexity that the pooled regression detects is the mechanical, option-like convexity of the barrier-exact exit, not directional forecasting: Jagannathan and Korajczyk (1986) show that "investing in options or levered securities will show spurious market timing" — "artificial timing ability when no true timing ability exists" — and the protective, big-move-capturing profile of a stop-loss/barrier rule is exactly such a convex, option-isomorphic payoff (Henriksson and Merton, 1981; Glosten and Jagannathan, 1994; Fung and Hsieh, 2001). Accordingly, market-timing skill is assessed primarily through the Pesaran and Timmermann (1992) directional-accuracy test, which — being a sign/contingency-table statistic rather than a magnitude regression — is invariant to the scale heterogeneity driving the pooled-gamma artefact; its result (pooled statistic −2.31, p ≈ 0.99) confirms the absence of timing skill, consistent with the disaggregated TM evidence.

## Reference list (Harvard style)

Blyth, C. R. (1972) 'On Simpson's paradox and the sure-thing principle', *Journal of the American Statistical Association*, 67(338), pp. 364–366.

Ferson, W. E. and Schadt, R. W. (1996) 'Measuring fund strategy and performance in changing economic conditions', *Journal of Finance*, 51(2), pp. 425–461.

Fung, W. and Hsieh, D. A. (2001) 'The risk in hedge fund strategies: theory and evidence from trend followers', *Review of Financial Studies*, 14(2), pp. 313–341.

Glosten, L. R. and Jagannathan, R. (1994) 'A contingent claim approach to performance evaluation', *Journal of Empirical Finance*, 1(2), pp. 133–160.

Goetzmann, W. N., Ingersoll, J. and Ivković, Z. (2000) 'Monthly measurement of daily timers', *Journal of Financial and Quantitative Analysis*, 35(3), pp. 257–290.

Goetzmann, W. N., Ingersoll, J. E., Spiegel, M. and Welch, I. (2007) 'Portfolio performance manipulation and manipulation-proof performance measures', *Review of Financial Studies*, 20(5), pp. 1503–1546.

Grinblatt, M. and Titman, S. (1994) 'A study of monthly mutual fund returns and performance evaluation techniques', *Journal of Financial and Quantitative Analysis*, 29(3), pp. 419–444.

Henriksson, R. D. and Merton, R. C. (1981) 'On market timing and investment performance. II. Statistical procedures for evaluating forecasting skills', *Journal of Business*, 54(4), pp. 513–533.

Jagannathan, R. and Korajczyk, R. A. (1986) 'Assessing the market timing performance of managed portfolios', *Journal of Business*, 59(2), pp. 217–235.

Pesaran, M. H. and Smith, R. (1995) 'Estimating long-run relationships from dynamic heterogeneous panels', *Journal of Econometrics*, 68(1), pp. 79–113.

Pesaran, M. H. and Timmermann, A. (1992) 'A simple nonparametric test of predictive performance', *Journal of Business & Economic Statistics*, 10(4), pp. 461–465.

Robinson, W. S. (1950) 'Ecological correlations and the behavior of individuals', *American Sociological Review*, 15(3), pp. 351–357.

Treynor, J. L. and Mazuy, K. K. (1966) 'Can mutual funds outguess the market?', *Harvard Business Review*, 44(4), pp. 131–136.

Zellner, A. (1962) 'An efficient method of estimating seemingly unrelated regressions and tests for aggregation bias', *Journal of the American Statistical Association*, 57(298), pp. 348–368.

*Supporting (cite original, not these): Pesaran, M. H. and Timmermann, A. (1994) 'A generalization of the non-parametric Henriksson–Merton test of market timing', Economics Letters, 44(1–2), pp. 1–7; Simpson, E. H. (1951) 'The interpretation of interaction in contingency tables', Journal of the Royal Statistical Society, Series B, 13, pp. 238–241.*