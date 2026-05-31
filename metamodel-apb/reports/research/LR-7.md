## Caveats, Source Verification & Bibliography

**Strength-of-evidence classification.**
- **Well-established theory (high confidence, peer-reviewed, primary-verified):** Kelly (1956); the growth-vs-security trade-off of fractional Kelly (MacLean, Ziemba & Blazenko 1992); the calibration/refinement orthogonality (DeGroot & Fienberg 1983); proper scoring rules and "sharpness subject to calibration" (Gneiting & Raftery 2007; Gneiting, Balabdaoui & Raftery 2007); Platt scaling (Platt 1999) and ECE (Guo et al. 2017); the *proven* result that under estimation uncertainty it is always optimal to shrink the Kelly bet, with shrinkage k\* = ((b+1)p−1)²/[((b+1)p−1)² + (b+1)²σ²] increasing in σ (Baker & McHale 2013, Theorem 1, verified against the primary text); and the volatility-targeting results of Moreira & Muir (2017) and Harvey et al. (2018).
- **Standard practitioner convention (not theorems):** quarter-/half-Kelly as default fractions and the 25% volatility target (Carver 2015; MacLean, Thorp & Ziemba 2010); bet-sizing from predicted probabilities (López de Prado 2018).
- **Our own synthesis/recommendation (clearly labelled as such):** that calibration corrects the conditional *mean* while fractional-Kelly corrects estimation *variance*, so composing them is coherent but κ should target post-calibration residual variance; that the hard floor is a coarse, dominated approximation to Kelly's own continuous zero; and the staged uncertainty-scaled κ_i design. These are reasoned extensions of the cited results, not findings lifted from a single paper.

**A corrected figure (per source verification).** The Chopra & Ziemba (1993) result is frequently misquoted as a flat "20×". The primary statement is that errors in means are *over ten times* as damaging as errors in variances and *over twenty times* as damaging as errors in covariances (two distinct ratios, ~11:2:1 to 20:2:1 depending on risk tolerance), with the relative impact of mean-errors rising at higher risk tolerance. The takeaway for LR-7 is unchanged and if anything strengthened: the *probability/edge* input dominates sizing error, which is exactly why an uncertainty-scaled κ matters more than the κ level. For Moreira & Muir (2017), the magnitude is concrete: a mean-variance investor restricted to the market portfolio can raise lifetime utility by ~65% through volatility timing.

**Sources to verify independently / interpretive flags.**
- **Baker & McHale (2013) "always shrink" result carries a stipulation:** the fully general proof assumes the bettor can *lay* (bet both sides); without it, the proof of k\* < 1 for any unbiased sampling distribution is weakened (the authors found no real-world counterexample). For non-logarithmic risk-averse utilities, rare "bet swelling" can occur at very favourable odds. For the log-utility/Kelly case used here, shrinkage is strict. There is also a Bayesian-vs-frequentist interpretive debate (the result is framed frequentistically); flag, not error.
- **Chu, Wu & Swartz** appears as a 2018 *Journal of Quantitative Analysis in Sports* article (14(1)); some references cite an earlier working-paper year. Verify the exact volume/year for the final bibliography.
- **DeGroot & Fienberg (1983)** appears variously as pp. 12–22 of *The Statistician* vol. 32 (issue 1–2); the companion "Assessing Probability Assessors" is the 1982 book chapter. Both are real; cite the 1983 *Statistician* paper for the calibration/refinement result.
- **Harvey et al. (2018)** is widely cited from the SSRN/working-paper version; the version of record is *Journal of Portfolio Management* 45(1):14–33 (Fall 2018). Note the later look-ahead-bias critique (Liu, Tang & Zhou 2019; "Conditional Volatility Targeting") — relevant if the overlay's *Sharpe* benefit (not its tail benefit) is claimed in-sample.
- **Carver (2015)** Chapter 9 quotation is verified verbatim; the 25% maximum vol target is for semi-automatic traders and is *halved* for negative-skew systems — a relevant guardrail if the strategy's return distribution is negatively skewed.

**Verified bibliography (Harvard author–date).**

Baker, R.D. & McHale, I.G. (2013) 'Optimal Betting Under Parameter Uncertainty: Improving the Kelly Criterion', *Decision Analysis*, 10(3), pp. 189–199. doi:10.1287/deca.2013.0271.

Carver, R. (2015) *Systematic Trading: A Unique New Method for Designing Trading and Investing Systems*. Petersfield: Harriman House. [Ch. 9, 'Volatility Targeting'.]

Cenesizoglu, T. & Timmermann, A. (2012) 'Do return prediction models add economic value?', *Journal of Banking & Finance*, 36(11), pp. 2974–2987.

Chopra, V.K. & Ziemba, W.T. (1993) 'The Effect of Errors in Means, Variances, and Covariances on Optimal Portfolio Choice', *Journal of Portfolio Management*, 19(2), pp. 6–11.

Chu, D., Wu, Y. & Swartz, T.B. (2018) 'Modified Kelly criteria', *Journal of Quantitative Analysis in Sports*, 14(1), pp. 1–11. doi:10.1515/jqas-2017-0122.

DeGroot, M.H. & Fienberg, S.E. (1983) 'The Comparison and Evaluation of Forecasters', *The Statistician (Journal of the Royal Statistical Society, Series D)*, 32(1–2), pp. 12–22. doi:10.2307/2987588.

De March, H. & Lehalle, C.-A. (2018) *Optimal trading using signals*. arXiv:1811.03718.

Gneiting, T., Balabdaoui, F. & Raftery, A.E. (2007) 'Probabilistic forecasts, calibration and sharpness', *Journal of the Royal Statistical Society, Series B*, 69(2), pp. 243–268. doi:10.1111/j.1467-9868.2007.00587.x.

Gneiting, T. & Raftery, A.E. (2007) 'Strictly Proper Scoring Rules, Prediction, and Estimation', *Journal of the American Statistical Association*, 102(477), pp. 359–378.

Granger, C.W.J. & Pesaran, M.H. (2000) 'Economic and statistical measures of forecast accuracy', *Journal of Forecasting*, 19(7), pp. 537–560.

Guo, C., Pleiss, G., Sun, Y. & Weinberger, K.Q. (2017) 'On Calibration of Modern Neural Networks', *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70, pp. 1321–1330. arXiv:1706.04599.

Harvey, C.R., Hoyle, E., Korgaonkar, R., Rattray, S., Sargaison, M. & Van Hemert, O. (2018) 'The Impact of Volatility Targeting', *Journal of Portfolio Management*, 45(1), pp. 14–33.

Kelly, J.L., Jr. (1956) 'A New Interpretation of Information Rate', *Bell System Technical Journal*, 35(4), pp. 917–926. doi:10.1002/j.1538-7305.1956.tb03809.x.

López de Prado, M. (2018) *Advances in Financial Machine Learning*. Hoboken, NJ: Wiley. [Ch. 10, 'Bet Sizing'.]

MacLean, L.C., Thorp, E.O. & Ziemba, W.T. (2010) 'Long-term capital growth: the good and bad properties of the Kelly and fractional Kelly capital growth criteria', *Quantitative Finance*, 10(7), pp. 681–687. [See also MacLean, Thorp & Ziemba (eds.) (2011) *The Kelly Capital Growth Investment Criterion*. Singapore: World Scientific.]

MacLean, L.C., Ziemba, W.T. & Blazenko, G. (1992) 'Growth Versus Security in Dynamic Investment Analysis', *Management Science*, 38(11), pp. 1562–1585.

Moreira, A. & Muir, T. (2017) 'Volatility-Managed Portfolios', *Journal of Finance*, 72(4), pp. 1611–1644. doi:10.1111/jofi.12513.

Platt, J.C. (1999) 'Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods', in Smola, A. et al. (eds.) *Advances in Large Margin Classifiers*. Cambridge, MA: MIT Press, pp. 61–74.

*Supplementary (drawdown/security-constrained growth, for the security-vs-growth strand):* Browne, S. (1997) 'Survival and Growth with a Liability: Optimal Portfolio Strategies in Continuous Time', *Mathematics of Operations Research*, 22(2), pp. 468–493.

---

**One-paragraph bottom line.** Platt calibration (settled) and the fractional-Kelly + vol-target stack are each well-founded, and the strategy's thin high-confidence slice is the *expected* signature of calibrated, growth-optimal sizing rather than a defect. The flat κ = 0.25 quarter-Kelly is a defensible conservative heuristic that you may keep if you document its limitation. But the literature most directly on point — Kelly under estimation risk — recommends two improvements that are better-justified than the status quo: (1) replace the hard p = 0.55 floor with a smooth taper (the Kelly map already tapers continuously to zero), and (2) make the shrinkage multiplier κ_i an explicit, per-instrument function of the *post-calibration residual* uncertainty, κ_i ≈ e_i²/(e_i² + σ_i²) (Baker & McHale 2013; Chu, Wu & Swartz 2018), which directly fixes the uneven per-instrument pass rates without double-penalising the over-confidence that calibration already removed. Evaluate both changes against certainty-equivalent return; if CER does not improve out-of-sample, the simpler flat heuristic stands.