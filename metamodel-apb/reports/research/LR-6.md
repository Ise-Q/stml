# Deflation and Sharpe-Ratio Inference on Short OOS Windows (T ≈ 128): A Methodological Review

**Bottom line up front:** At T ≈ 128 daily out-of-sample observations, the Deflated Sharpe Ratio (DSR) and CSCV-PBO should **not** be presented as load-bearing point statistics — they should be reported with explicit small-sample caveats and *supplemented* by a robust, studentised stationary-bootstrap Sharpe confidence interval plus a Probabilistic Sharpe Ratio / Minimum Track Record Length (MinTRL) statement. The arithmetic is decisive and supports an honest-negative conclusion: under standard IID inference, an annualised Sharpe of 1.36–1.55 over 128 daily returns has a t-statistic of only ≈ 0.97–1.10 — *not significant at 5% even before any deflation for multiple trials*. The deflation gate is directionally correct (it pushes toward rejection), but its specific numbers at this T are dominated by estimation noise in the skewness/kurtosis plug-ins and by the asymptotic-normality assumption breaking down, so they belong in the write-up as corroborating evidence rather than as a precise computed probability.

## TL;DR
- **Trust / caveat / supplement?** *Supplement and caveat.* DSR and CSCV-PBO are conceptually the right tools and should be shown, but at T ≈ 128 their plug-in moments (γ̂₃, γ̂₄) are too noisy and the CLT approximation too weak for the numbers to stand alone. Pair them with a studentised stationary-block bootstrap CI (Ledoit & Wolf 2008) and a PSR(0)/MinTRL statement (Bailey & López de Prado 2012).
- **The recommended small-T Sharpe CI** is the studentised stationary/circular block bootstrap of Ledoit & Wolf (2008) — callable in Python via `arch.bootstrap.StationaryBootstrap` / `CircularBlockBootstrap` with `conf_int(..., method="studentized")`. The analytic Lo (2002) IID SE and the Mertens (2002)/Opdyke (2007) skew-kurtosis SE should be reported alongside as transparent sanity bands, but the bootstrap is the workhorse because the analytic SEs under-cover at small, fat-tailed samples.
- **Honest framing:** a ~6-month deployment test (≈128 days) is *below* the MinTRL needed to call even a Sharpe of ~1.4 significantly positive at 95% (≈ 1.5–3 years of daily data), so the correct claim is "not yet statistically distinguishable from zero after accounting for selection," not "the strategy failed."

---

## Key Findings

- The four-moment Sharpe SE is confirmed correct and reduces to Lo's IID form. **SE(SR̂) ≈ √([1 − γ₃·SR̂ + ((γ₄−1)/4)·SR̂²]/T)** with γ₄ = *raw* (Pearson) kurtosis (=3 for Gaussian), giving (γ₄−1)/4 = 0.5 and recovering **SE(SR̂) ≈ √((1 + 0.5·SR̂²)/T)**. Skewness term is negative; kurtosis term positive.
- At T = 128, the dominant problem is **sample size, not non-normality**: the IID 95% CI for an annualised Sharpe of 1.36 is ≈ [−1.39, 4.11] and for 1.55 ≈ [−1.21, 4.31]; the skew-kurtosis adjustment (illustratively γ₃=−0.5, γ₄=6) widens the band only ~2–3%.
- The DSR inherits the noisy four-moment denominator and is best presented as a *directional, sensitivity-banded* corroboration, not a precise probability. **No dedicated peer-reviewed Monte-Carlo stress-test of DSR at T ≈ 128 exists** — a genuine literature gap.
- CSCV-PBO needs S ≥ 16 to be meaningful, but at T = 128 the OOS half is fixed at T/2 = 64 observations regardless of S, and at S=16 each block holds only 8 daily returns — at the very edge of usability and prone to the "noisy OOS ranking → uninformative PBO ≈ 0.5" failure mode.
- The PSR is the natural single-trial bridge (DSR = PSR at the deflated benchmark); MinTRL quantifies that ~128 days is roughly 2–3× too short; a Student-t Bayesian Sharpe posterior is the most robust route to a "P(SR>0)" number at small T.

---

## Details

### Q1 — Sharpe Confidence Intervals at Small T (Lo, Mertens/Opdyke, Ledoit–Wolf)

**The four-moment Sharpe standard error is correct and reduces cleanly to Lo's IID form.** Under stationary/ergodic returns (Mertens 2002; Christie 2005; Opdyke 2007):

  SR̂ ~ N( SR , [ 1 − γ₃·SR + ((γ₄ − 1)/4)·SR² ] / T )

so the estimated standard error is

  **SE(SR̂) ≈ √( [ 1 − γ₃·SR̂ + ((γ₄ − 1)/4)·SR̂² ] / T )** ,

where **γ₃ is skewness and γ₄ is the *raw* (Pearson) kurtosis (γ₄ = 3 for a Gaussian)**. Setting γ₃ = 0, γ₄ = 3 gives (γ₄ − 1)/4 = 0.5, recovering Lo (2002) exactly:

  **SE(SR̂) ≈ √( (1 + 0.5·SR̂²) / T )** .

Sign conventions are confirmed against primary sources: the skewness term is **negative** (negative skew *widens* the interval); the kurtosis term is positive (fat tails widen it). Mertens' original note writes the same variance with *excess* kurtosis and an explicit constant, `1 + ½SR² − γ₃·SR + ((γ₄,excess)/4)·SR²`; this is algebraically identical because (γ₄,raw − 1)/4 = ½ + γ₄,excess/4. Either is acceptable provided the convention is stated. Some implementations (Bailey & López de Prado's PSR code, PerformanceAnalytics) use Bessel-corrected √(T − 1); at T = 128 the difference vs √T is ~0.4%.

**Annualising the SE.** Annualise the point estimate and its SE *together* by the same factor. Cleanest route: **compute the SE in native (daily) units using the daily SR̂ and daily T, then multiply the whole interval by √q (q ≈ 252).** The "1" in the numerator is frequency-dependent through SR̂, so you cannot annualise SR̂ first and then plug into the daily formula. Lo (2002) warns that **√q annualisation is valid only under IID**; with serial correlation use η(q) = q / √(q·σ² + 2σ²·Σ(q−k)ρ_k). Ignoring positive autocorrelation can overstate an annualised Sharpe by ~65% (Lo's hedge-fund example). Check first-lag autocorrelation / Ljung–Box.

**Reliability at T ≈ 128.** All three analytic routes (Lo IID; Mertens/Opdyke skew-kurtosis; Bao 2009 finite-sample expansion) are *asymptotic* and rest on the CLT for SR̂. The estimator is also upward-biased in finite samples (Miller & Gehr 1978; Bao 2009). At T = 128 with fat tails/skew the analytic SEs are known to be *liberal* (intervals too narrow), exactly as Ledoit & Wolf (2008) document for HAC inference at small/moderate samples. Report analytic SEs as reference bands; treat the studentised bootstrap as primary.

**Worked illustrative numbers (illustrative — substitute your own moments).** For T = 128 daily:
- Annualised Sharpe 1.36 → daily SR̂ ≈ 1.36/√252 ≈ 0.0857; **t = √128 × 0.0857 ≈ 0.97**. Annualised 1.55 → daily SR̂ ≈ 0.0976; **t ≈ 1.10**. Neither clears 1.645 (one-sided 5%) or 1.96 (two-sided): *on 128 days these Sharpes are not significantly different from zero even under IID-normal.*
- IID (Lo) SE: daily SE ≈ √((1+0.5·0.0857²)/128) ≈ 0.0885; annualised ×√252 ≈ **1.405**. Sharpe 1.36 ± 1.96×1.405 → **95% CI ≈ [−1.39, 4.11]**. For 1.55: ann. SE ≈ 1.406; **95% CI ≈ [−1.21, 4.31]**.
- Skew-kurtosis (Mertens/Opdyke) SE, illustrative γ₃ = −0.5, γ₄ = 6: numerator = 1 − (−0.5)(0.0857) + ((6−1)/4)(0.0857²) ≈ 1.0521; SE_daily ≈ √(1.0521/128) ≈ 0.0907; annualised ≈ **1.439** (~2–3% wider). With γ₄ = 10, γ₃ = −1, numerator ≈ 1.15 (~7% wider). **Non-normality matters but is dwarfed by the sheer width driven by small T.**

### Q1 (cont.) — The Recommended Bootstrap Procedure and Python Implementation

**The studentised stationary/circular block bootstrap (Ledoit & Wolf 2008, *Journal of Empirical Finance* 15(5):850–859; peer-reviewed, with R/MATLAB code) is the small-sample workhorse.** They show the Jobson–Korkie/Memmel and Lo IID approaches are invalid under heavy tails or time-series structure, and that a **studentised time-series (stationary/circular block) bootstrap CI has materially better finite-sample coverage.** Their construction is for the *difference* of two Sharpes but the same machinery applies to a single Sharpe.

**Procedure (single-Sharpe CI):**
1. Compute SR̂ and an SE estimate ŝê (Mertens/Opdyke SE or an HAC/kernel SE).
2. Draw B (5,000–10,000) **block** resamples preserving serial dependence: stationary bootstrap (Politis–Romano, geometric block length, mean b) or circular block bootstrap; choose b for the autocorrelation horizon (e.g. 5–20 for daily; `optimal_block_length` can suggest it).
3. For each resample compute t*_b = (SR̂*_b − SR̂)/ŝê*_b, recomputing ŝê*_b within each resample.
4. The 1−α studentised CI is [ SR̂ − ŝê·q*_{1−α/2}, SR̂ − ŝê·q*_{α/2} ] from empirical quantiles of t*_b. Declare SR > 0 if the interval excludes 0.

```python
from arch.bootstrap import StationaryBootstrap, optimal_block_length
import numpy as np
def sharpe(x):
    return np.array([x.mean() / x.std(ddof=1)])
b = optimal_block_length(returns)['stationary'].iloc[0]
bs = StationaryBootstrap(b, returns)
ci = bs.conf_int(sharpe, reps=10000, method='studentized')   # also 'bca','percentile','basic'
sr_ann = sharpe(returns)[0] * np.sqrt(252)
```
`arch` (Kevin Sheppard) provides `IIDBootstrap`, `StationaryBootstrap`, `CircularBlockBootstrap`, `MovingBlockBootstrap`, with `method` ∈ {`percentile`, `basic`, `studentized`, `bca`}; **studentized** and **BCa** give higher-order accuracy. arch's docs warn that squared returns are persistent so the IID bootstrap is a poor choice for strategy returns — use a time-series bootstrap. `recombinator` is an alternative. `quantstats`/`empyrical` compute the *point* Sharpe only (no non-normal CI), so the CI must come from `arch`; Lo's autocorrelation-adjusted SE is in `PerformanceAnalytics` (`se.LoSharpe`). **At T ≈ 128 the honest expectation is a very wide interval that contains zero — that width is the finding.**

### Q2 — Is the DSR Reliable at ~6 Months / ~128 Observations?

The DSR (Bailey & López de Prado, *Journal of Portfolio Management* 40(5):94–107, 2014; peer-reviewed) is

  DSR = Z( [ (SR̂ − SR̂₀)·√(T − 1) ] / √( 1 − γ̂₃·SR̂ + ((γ̂₄ − 1)/4)·SR̂² ) ),

with SR̂₀ = √V[{SR̂ₙ}]·( (1−γ)·Z⁻¹[1 − 1/N] + γ·Z⁻¹[1 − 1/(N·e)] ), γ ≈ 0.5772, N the effectively-independent trials, V[{SR̂ₙ}] the trial-Sharpe variance. **The denominator is exactly the Mertens/Opdyke four-moment SE** — and that is where small-T fragility bites: sample skewness and kurtosis are high-variance, biased at n = 128, and for heavy tails the fourth moment is dominated by a few tail points. Bailey & López de Prado themselves note "estimating moments beyond the third, and particularly the fourth moment, requires longer sample lengths," and that the PSR/MinTRL equations rest on an asymptotic distribution for which "CLT is typically assumed to hold for samples in excess of 30 observations … the moments inputted … must be computed on longer series for CLT to hold."

Two distinct weaknesses: (1) *plug-in moment noise* — a plausible swing in γ̂₄ from 5 to 9 moves the DSR non-trivially, so the point value is not stable to within its own estimation error; the PSR paper recommends inputting a lower-bound skew and upper-bound kurtosis, i.e. a **sensitivity band**; (2) *two roles for T* — the DSR's T is the OOS path (~128), giving a small √(T−1) = √127 ≈ 11.3 multiplier, so even a respectable excess (SR̂ − SR̂₀) yields a modest z and the DSR sits far from 1.0.

**No short-window correction is published.** The closest peer-reviewed guidance is indirect: MinTRL (Q4) shows ~128 days is too short, and Bao (2009, *J. Financial Econometrics*) shows the four-moment SE is itself susceptible to higher-moment estimation error in small samples. **Verdict:** present DSR as directional ("DSR ≪ 0.95, consistent with not rejecting the null"), with a moment-sensitivity band; if a single number is shown, round and caveat it.

### Q3 — CSCV-PBO with Few Blocks: Minimum S, Reporting, Failure Modes

In CSCV (Bailey, Borwein, López de Prado & Zhu, *Journal of Computational Finance*, 2017; peer-reviewed), the (T × N) matrix is split into S equal time-blocks, forming all C(S, S/2) ways of choosing S/2 IS blocks with the complement OOS. Canonical settings are **S = 8** (C(8,4) = 70) and **S = 16** (the paper states "12,780 combinations"; the exact binomial coefficient C(16,8) = **12,870** — the "12,780" is a long-propagated typo worth verifying against the published table). The peer-reviewed Bayesian re-analysis (*Risks* 9(1):18, 2021) recommends S "at least 16," because the logit distribution PBO is read from needs enough combinations to be stable. S must be even; below S = 8, PBO is not meaningful.

**The binding constraint at T ≈ 128 is the OOS path, not S.** Each combination devotes T/2 = 64 OOS observations regardless of S:

| S | combinations C(S,S/2) | obs/block (T/S) | total OOS obs (T/2) |
|---|---|---|---|
| 8 | 70 | 16 | 64 |
| 10 | 252 | ~13 | 64 |
| 16 | 12,870 (paper: "12,780") | 8 | 64 |

Larger S buys more combinations but shorter blocks (8 returns at S=16 → near-noise rankings). This triggers the documented failure mode: **when all N configurations have similar, noisy OOS performance, the IS winner's OOS rank is effectively random, driving PBO toward ~0.5 (uninformative) regardless of true overfitting.** The CSCV paper warns hold-out/short-sample validation is inadequate below ~1,000 observations.

**How to report PBO at small S:** (1) show the full logit distribution/histogram plus degradation and dominance plots, not just a scalar; (2) report PBO as a range across S ∈ {8,10,12,16} and note sensitivity; (3) state per-block OOS length explicitly; (4) treat high PBO as evidence of overfitting risk but low PBO at short T as *weak* evidence of robustness — PBO is blind to regime shifts, look-ahead bias, and data leakage. At T ≈ 128 it is better framed as a governance artifact than a decision statistic.

### Q4 — PSR, MinTRL/MinBTL, and Bayesian Alternatives at Small T

**PSR (Bailey & López de Prado 2012, *Journal of Risk* 15(2); peer-reviewed) is distinct from DSR and is the right primitive.**

  PSR(SR*) = Z( [ (SR̂ − SR*)·√(T − 1) ] / √( 1 − γ̂₃·SR̂ + ((γ̂₄ − 1)/4)·SR̂² ) ).

DSR is *exactly* PSR evaluated at SR* = SR̂₀ (the deflated benchmark). PSR(0) isolates the sample-length question from the (separately uncertain) N and V[{SR̂ₙ}]. With t ≈ 0.97–1.10, PSR(0) for a Sharpe of ~1.36–1.55 over 128 days lands around ~0.83–0.86 under normality (lower with negative skew/fat tails) — **below the 0.95 threshold**, before any deflation.

**MinTRL** makes it quantitative (in number of observations):

  MinTRL = 1 + ( 1 − γ̂₃·SR̂ + ((γ̂₄ − 1)/4)·SR̂² ) · ( z_{1−α} / (SR̂ − SR*) )² .

Bailey & López de Prado's calibration: rejecting H₀: SR = 0 at 95% needs **roughly 2–3 years of daily data (~500–750 obs)** for typical strategies (≈2 years once annualised Sharpe > ~1.15). For SR* = 0 and near-normal moments, MinTRL ≈ 1 + (z_{0.95}/SR̂_daily)² ≈ 1 + (1.645/0.0857)² ≈ **369 daily obs** for the Sharpe-1.36 case, ≈ **285** for 1.55 — both ~2–3× the 128 days available; negative skew/fat tails push these higher. This is the most defensible peer-reviewed way to state "too short." (Distinguish from **MinBTL**, Bailey et al. 2014 *Notices of the AMS*: MinBTL ≲ 2·ln(N)/E[max_N]², bounding how many trials N you may run on a given backtest length before a spurious Sharpe of 1 is essentially guaranteed.)

**Bayesian / shrinkage alternatives.** A Bayesian Sharpe posterior is legitimate and better-behaved at small T because it (a) propagates full parameter uncertainty, (b) handles fat tails via a Student-t likelihood, and (c) yields P(SR > 0 | data) directly. The standard implementation is the Bayesian Sharpe / Kruschke "BEST" two-group model (Student-t likelihood, estimated ν) in `pyfolio`'s Bayesian tearsheet and the Packt/López-de-Prado "ML for Algorithmic Trading" notebook (`PyMC`/`arviz`); it returns posteriors over the Sharpe and the IS-vs-OOS difference. For the *selection* problem there is recent **preprint-stage** work (the *Risks* 2021 Bayesian PBO paper; "Sharpe under selection/thresholding" working notes, e.g. Mulligan QMF 2024) placing a prior over the cross-section of trial Sharpes and shrinking the selected one toward the null — a Bayesian analogue of the DSR not hinging on a plug-in γ̂₄. **Recommendation:** for a "P(true Sharpe > 0)" number, prefer a Student-t Bayesian posterior over a point DSR, report the credible interval, and flag selection-aware Bayesian variants as preprints to verify.

---

## Recommendations

1. **Report all layers in order of assumption-strength.** (a) Raw annualised Sharpe + t-stat; (b) studentised stationary/circular-block bootstrap 95% CI via `arch` (primary inference); (c) analytic Lo and Mertens/Opdyke SE bands as reference; (d) PSR(0) and MinTRL; (e) DSR and CSCV-PBO as caveated, sensitivity-banded corroboration. Do **not** let DSR/PBO carry the argument alone.
2. **State the deflation gate's conclusion as directional.** If DSR < 0.95 and PBO is materially > 0 (or uninformative because the OOS leg is only 64 points), report "consistent with not rejecting the null / overfitting risk not excluded," not a precise p-value.
3. **Show DSR and PBO as ranges, not points.** Recompute DSR over a plausible (γ̂₃, γ̂₄) range or over bootstrap-resampled moments; report PBO across S ∈ {8,10,12,16} with per-block OOS length stated.
4. **Benchmarks that would change the recommendation:** (i) once OOS reaches **MinTRL (~300–750 daily obs / ~1.5–3 yrs)**, analytic and bootstrap CIs become usable as primary evidence and DSR can be quoted as a point statistic; (ii) once each CSCV block holds **≥ ~50–100 OOS obs with S ≥ 16** (T ≳ 800–1,600), PBO becomes meaningful as a scalar; (iii) if first-lag autocorrelation is significant (Ljung–Box), switch from √q annualisation to Lo's η(q) scaling before any of the above.
5. **Prefer a Student-t Bayesian Sharpe posterior** if a "P(true Sharpe > 0)" number is required at small T — more robust to the fat tails that destabilise the DSR denominator, and it yields an honest credible interval.

### A model reporting paragraph (adapt verbatim)
> *"Over the ~6-month deployment window (T ≈ 128 daily returns), the strategy realised an annualised Sharpe of [1.55], a t-statistic of √T·SR̂_daily ≈ [1.10] — not significantly different from zero at the 5% level even under the most favourable IID-normal assumption; the studentised stationary-block bootstrap 95% CI (Ledoit & Wolf 2008) is [−1.2, 4.3] and comfortably contains zero. Accounting for the [N] configurations explored, the Deflated Sharpe Ratio (Bailey & López de Prado 2014) is [0.xx] — below 0.95 — but at this sample length the DSR depends on noisily estimated skewness and kurtosis and on an asymptotic-normality approximation that is weak at T ≈ 128, so we report it as a directional check rather than a precise probability (sensitivity range over plausible higher moments: [0.aa–0.bb]). The Minimum Track Record Length needed to call a Sharpe of this magnitude significantly positive at 95% is ≈ [285–370] daily observations — roughly 2–3× the data available. We therefore conclude that this window is too short to distinguish the strategy's risk-adjusted performance from zero after correcting for selection; this is an honest 'insufficient evidence' result, not a demonstration of either skill or failure."*

---

## Caveats and Sources to Verify Before a Formal Write-Up

- **Mertens (2002) is an unpublished working note** (self-hosted), *not* peer-reviewed — cite **Opdyke (2007, *Journal of Asset Management* 8(5):308–336, peer-reviewed)** for the stationary/ergodic four-moment SE, with **Lo (2002, *Financial Analysts Journal* 58(4):36–52)** and **Bailey & López de Prado (2012, *Journal of Risk* 15(2))** as the published anchors. Christie (2005) is a Macquarie working paper.
- **The "12,780 combinations for S = 16" in the CSCV paper appears to be a typo**; exact value C(16,8) = **12,870**. Verify against the published *Journal of Computational Finance* (2017) table.
- **No dedicated peer-reviewed Monte-Carlo study stress-testing DSR specifically at T ≈ 128 was located** — a genuine literature gap; do not cite a blog as if it were such a study. The "S ≥ 16" recommendation comes from the peer-reviewed *Risks* (2021) Bayesian-PBO paper.
- **Preprint/secondary flags (reliable for formulas/code, not as authority):** gmarti/marti.ai and Balaena-Quant/Medium DSR write-ups; QuantPy and Portfolio-Optimizer blog derivations; the Two Sigma technical report (Riondato 2018, a survey/working report); the Benhamou arXiv note (1808.04233); and "Sharpe under selection" Bayesian working papers (e.g. Mulligan QMF 2024). The CSCV/DSR/PSR papers exist as both SSRN preprints and peer-reviewed journal articles — cite the journal versions (J. Comp. Finance 2017; J. Portfolio Management 2014; J. Risk 2012) in a formal write-up.
- **Annualisation/autocorrelation:** confirm IID before √q annualisation; if first-lag autocorrelation is significant, Lo's η(q) scaling is required and the reported Sharpe may be inflated.