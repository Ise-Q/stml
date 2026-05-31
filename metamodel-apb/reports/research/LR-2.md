## S6.7 — Backtest Specification & Metric Definitions for a Triple-Barrier / Meta-Labelled Futures Strategy

### TL;DR
- **Retire latest-signal-wins.** It never realises the prior trade's first-touch (PT/SL/vertical) outcome, instead exiting at the *next signal's arrival* — a path-dependency bias that breaks the ML-target↔P&L correspondence, truncates would-be stops/profit-takes, and makes results depend on signal cadence. Replace it with an **event-driven, barrier-exact simulator** (mlfinlab `get_events`/`get_bins` semantics: exit at the earliest of upper +k·σ, lower −k·σ, or vertical T_max), with high/low intrabar touch detection and a conservative "stop-first" tie-break for both-barrier bars.
- **Net concurrent meta-bets by averaging active bets** (López de Prado AFML Ch.10 `avg_active_signals` + `discrete_signal`), not by summing — summing produces phantom 150–200% leverage; averaging bounds per-instrument exposure to ±1. Feed the resulting netted position path into vectorbt (`from_orders`, multi-asset with contract multipliers) for sweeps and cross-check the final candidate in an event-driven engine (backtrader).
- **Cost = ½·bid-ask spread per side + Grinold–Kahn square-root impact `c·σ·√(Q/ADV)`**; gate the strategy on **Deflated Sharpe Ratio** (with explicit effective-trial count N from clustering the research-config grid), **CSCV-based PBO**, and **MinBTL**.

---

### S6.7 Backtest Specification — Barrier-Exact Exits & Position Netting

**Bottom line:** Replace "latest-signal-wins" with an **event-driven, barrier-exact simulator whose concurrent meta-bets are netted into a single position path by averaging active bets**, per López de Prado's `getEvents`/`getBins` and `avgActiveSignals` logic [1][2][3]. The current model is structurally biased and must be retired.

**(a) Bias of latest-signal-wins.** When a new event overwrites the open position, the prior trade's first-touch outcome (PT/SL/vertical) is never realised; the realised holding period is set by the *arrival of the next signal*, not by the barrier the label was built on. This breaks the train/backtest correspondence (the ML target is "which barrier is hit first," but the P&L reflects something else), systematically truncates losing trades that would have hit a stop and clips winners that would have hit a profit-take, and makes results depend on signal-arrival cadence rather than market path. It is a **path-dependency/look-ahead hazard**: exits are decided by future signal timing.

**Barrier-exact exit rule (implement exactly):** For each event *i* with entry price *p₀*, target volatility *σᵢ*, multiples [*pt*, *sl*], and vertical time *t1ᵢ = entry + T_max*: walk bars forward; the exit time is the **first** bar at which the path touches the upper barrier *p₀(1+pt·σᵢ)*, the lower barrier *p₀(1−sl·σᵢ)*, or *t1ᵢ* — exactly mlfinlab's `get_events` (returns the earliest touch time) and `get_bins` (signs the return and assigns the label) [1][4].

**Intrabar resolution.** Using only close prices understates barrier touches; using bar high/low is realistic but creates the **both-barriers-in-one-bar ambiguity** (high≥PT and low≤SL in the same bar). mlfinlab/AFML resolve touch *timing* at bar granularity and do not resolve sub-bar ordering [1][4]. The defensible S6.7 convention: (i) detect touches on high/low; (ii) when both barriers fall inside one bar, **assume the stop-loss is hit first** (worst-case, removes optimistic bias); (iii) record this as a flagged "ambiguous-bar" exit for diagnostics; (iv) where intrabar data exist, drop to a finer bar to disambiguate. Fill at the barrier price (plus costs, below), not the close — except the vertical exit, which fills at bar close/next open.

**(b) Concurrency & netting.** Because triple-barrier spans overlap, multiple bets are open simultaneously on one instrument and across instruments — the same overlap that makes labels non-IID (AFML Ch.4: number of concurrent labels, average uniqueness, sequential bootstrap) [2][5]. Do **not** sum overlapping bets; net them with López de Prado's **average-active-signals** method (AFML Ch.10) [3][6]. Implement `avg_active_signals` (mlfinlab `bet_sizing.ch10_snippets`) [6]:
1. Map each event's meta-probability *p* to a signed size in [−1,1] via `get_signal` (z-score *(p−1/K)/√(p(1−p))* through the normal CDF, times the side) [3][6].
2. Build evaluation timestamps = union of all bet **start** times and all **t1** barrier-touch times (output ~2× input length) [6].
3. At each timestamp, a bet is **active** iff start ≤ *t* and (*t* < *t1* or *t1* is NaT); net target = **arithmetic mean of active-bet sizes**; 0 (flat) if none [3][6].
4. Apply `discrete_signal(step_size)`: round to a step grid, clip to [−1,1] to suppress overtrading from continuous drift [3][6].

This bounds aggregate per-instrument exposure to ±1 by construction and yields one position path the engine can cost and mark-to-market. Run per instrument; size instruments by inverse-vol / volatility-target weights at the portfolio layer (consistent with Harvey et al. 2018).

**Engine choice.** `vectorbt`'s `Portfolio.from_signals` supports stop-loss/take-profit (`sl_stop`/`tp_stop`) and time exits and auto-cleans signals, but its native model is one-position-per-column and does **not** average concurrent bets — so compute the netted path *first* (steps 1–4) and feed it via `from_orders`/target-size, multi-column for the multi-asset universe (docs show ES×NQ futures with contract multipliers) [7][8]. `backtrader` is event-driven and can encode barrier exits in `next()` with bracket orders, but netting still requires the same pre-aggregation; its `TradeAnalyzer`/`SQN` then report per-trade stats [9][10]. **Recommendation: vectorbt for vectorised sweeps + an event-driven backtrader cross-check on the final candidate**, both consuming the same barrier-exact, netted position path.

---

### Metric Definitions (Sortino, Turnover, Average Holding Period)

**(i) Sortino ratio.** Sortino & van der Meer (1991) and Sortino & Price (1994) replace total volatility with **downside (target) semideviation** against a **Minimum Acceptable Return (MAR), τ** [11][12]:

  Sortino = (E[R] − τ) / DD, where **DD = √( (1/N) Σ min(Rₜ − τ, 0)² )**.

The denominator sums **squared shortfalls below τ over the full sample N** (returns at/above τ contribute zero) — *not* the standard deviation of only the negative observations, a common error (quantstats' `rolling_sortino` was flagged for exactly this) [13][14]. Difference vs Sharpe: Sharpe penalises all volatility symmetrically; Sortino penalises only below-target dispersion, so it better fits the asymmetric payoff of a triple-barrier strategy. **S6.7 convention:** τ = 0 (per-period) for futures excess returns; annualise numerator ×*A*, denominator ×√*A*, *A* = periods/year (252 daily). Report MAR explicitly.

**(ii) Turnover.** Three conventions: (a) **weight-based** = ½·Σ|wᵢ,ₜ − wᵢ,ₜ₋₁| per period (one-way); (b) **notional** = traded notional / AUM; (c) **Grinold–Kahn** annualised rate of trading used to amortise round-trip cost, with **holding period ≈ 1/turnover** [15][16]. **Recommendation:** for a leveraged, notional-based futures meta-strategy, weights are ill-defined, so use **(b) one-way notional turnover** = (Σ|Δ position notionalₜ|)/(2·AUM), annualised — it maps directly onto the cost model and Grinold–Kahn amortisation. State it is one-way (÷2) to avoid double-counting buys+sells.

**(iii) Average holding period.** (a) **Trade-based** = mean bars/days between entry and barrier-touch exit (directly from the barrier-exact exit records and `t1`); (b) **inventory-based** = avg |position| / avg |trade size| per period, related to turnover via **holding period ≈ 1/turnover** [15][16]. mlfinlab's `backtest_statistics` provides an "average holding period from a series of positions" (inventory definition) [17]. **Recommendation: report both** — trade-based as the primary economic figure (it equals realised barrier dwell time), inventory-based as the turnover cross-check; flag material divergence (a sign of frequent flips/flattenings).

---

### Transaction-Cost Model (Half-Spread + Grinold–Kahn Square-Root Impact)

Model per-side cost of liquid futures as **bid-ask half-spread + Grinold–Kahn square-root market impact** [18][19]:

  **Cost_per_side = ½·s·|Q|·P + c·σ·√(|Q| / ADV)·|Q|·P**

*Q* = contracts, *P* = price, *s* = bid-ask spread, *σ* = daily return volatility, *ADV* = average daily volume (contracts), *c* = dimensionless impact coefficient "of order one" [18][19][20]. The second term is the **square-root law**: per-unit impact ∝ *σ·√(Q/ADV)*, the Grinold–Kahn "sigma-root-liquidity" form. Note: **Almgren, Thum, Hauptmann & Li (2005), "Direct Estimation of Equity Market Impact," used almost 700,000 US stock orders executed by Citigroup desks over the 19 months Dec 2001–Jun 2003 and explicitly *rejected* the pure square-root temporary-impact model in favour of a 3/5 (≈0.6) power law** [19][20] — so treat ½ as a baseline and test the exponent. Contrast: **Kyle (1985)** posits *linear* impact (price move = λ·order flow); the square-root law is the concave large-order refinement and is the recommended form here [21][19]. **Almgren & Chriss (2000), "Optimal Execution of Portfolio Transactions," J. Risk 3(2):5–39**, supplies the canonical permanent-plus-temporary linear-impact decomposition if intra-day slicing is later modelled [22].

**Calibration for liquid futures (not equities):**
- **Spread *s*:** observed top-of-book spreads per contract (fractions of a tick for ES, Bund, Brent); half-spread per side — the dominant cost at the small participation rates typical here.
- **σ:** the same volatility used to set the barriers (label↔cost consistency).
- **ADV:** per-contract; keep per-rebalance *Q* a small fraction of ADV so impact stays sub-dominant.
- **c:** calibrate from execution TCA if available; else set *c* ≈ 1 and run sensitivity over *c* ∈ [0.5, 2] and the impact exponent over [0.5, 0.6]. Do **not** import equity *c*/exponent values uncritically (equity microstructure is out of scope).
- Add explicit **roll costs** (calendar-spread spread) at each roll, plus exchange/clearing fees.

**Pitfalls:** Sortino is sensitive to MAR/τ and to √-annualisation under autocorrelation (Lo 2002: naive √-scaling is wrong under serial correlation) [23]; turnover convention ambiguity (fix one-way notional); holding-period definitions diverge under flips; equity-derived cost coefficients can flip net Sharpe sign. Treat the impact term as a *lower bound* on real slippage.

---

### Deflated Sharpe Ratio & Probability of Backtest Overfitting (Single Backtest)

**(i) Deflated Sharpe Ratio (DSR).** Bailey & López de Prado (2014, *JPM* 40(5):94–107) [24][25]:

  **DSR = Z[ ( (SR − SR₀)·√(T−1) ) / √(1 − γ₃·SR + ((γ₄−1)/4)·SR²) ]**

*T* = return observations, *γ₃* = skewness, *γ₄* = kurtosis (the Mertens/Lo non-normality correction to the SR standard error) [24][23]. The expected-maximum benchmark under the null (true SR=0):

  **SR₀ = √V[SR̂ₙ] · ( (1−γ)·Z⁻¹[1 − 1/N] + γ·Z⁻¹[1 − 1/(N·e)] )**

*V[SR̂ₙ]* = **variance of Sharpe ratios across the N trials**, *N* = independent trials, *γ* = Euler–Mascheroni ≈0.5772, *Z⁻¹* = inverse normal CDF [24][25]. SR₀ rises with both *N* and trial-SR dispersion, so wide brute-force search faces a higher hurdle. mlfinlab's `backtest_statistics` implements DSR/PSR directly [17].

**Choosing N for a single reported backtest (the key difficulty).** *N* is the **effective number of independent trials**, not the single final run:
- **Upper bound = full config grid** (windows × thresholds × barrier multiples × feature sets) tried during research; it is always safer to overestimate *N* [26].
- **Effective N via clustering:** trials are correlated, so López de Prado (2018/2020) recommends clustering the trial-return matrix (e.g., the **ONC** algorithm) and setting *N* = number of clusters [25][27].
- **Discipline:** log every backtest's returns so *N* and *V[SR̂]* are measured, not guessed.

**(ii) PBO via CSCV.** Bailey, Borwein, López de Prado & Zhu (2017, *J. Computational Finance* 20(4)) [28][29]: assemble the *T×N* matrix *M* of per-config returns; split into **S** equal time-contiguous submatrices (no shuffling); form all **C(S, S/2)** combinations splitting into in-sample (IS) and out-of-sample (OOS); per combination, pick the best-IS-Sharpe config, find its **relative rank** ω OOS, compute **logit λ = ln(ω/(1−ω))**. **PBO = fraction of combinations with λ ≤ 0** (best-IS config lands in the bottom OOS half) [28][29][30]. For a *single* strategy, the "N configs" are the research-process variants; CSCV on that pool quantifies whether IS selection generalises.

**(iii) Minimum Backtest Length (MinBTL).** Bailey, Borwein, López de Prado & Zhu (2014, *Notices AMS* 61(5):458–471), Theorem 3.1 [31][32]:

  **MinBTL ≈ ( 2·ln(N) ) / E[max SR]²** (years).

Per the paper verbatim: "if only 5 years of data are available, no more than 45 independent model configurations should be tried, or we are almost guaranteed to produce strategies with an annualized Sharpe ratio IS of 1, but an expected Sharpe ratio OOS of zero" [31][32].

**Recency.** López de Prado, Lipton & Zoonekynd (2025, "How to Use the Sharpe Ratio," SSRN 5520741) derive "a closed-form approximation to the sampling distribution of the Sharpe ratio estimator when returns are jointly non-Normal and serially correlated," identify five common errors, and introduce a **new hybrid FWER–FDR framework** (Bayesian FDR + family-wise control) for false positives, with replication code at github.com/zoonek/2025-sharpe-ratio; they reiterate that DSR's variance-across-trials term penalises brute-force search and advocate a theory-first research process [33][34]. **S6.7 should report PSR, DSR (with stated N and V[SR̂]), MinTRL/MinBTL, and a CSCV-PBO figure as a standard deployment gate.**

---

### Confidence Levels
- **Barrier-exact exits via `getEvents`/`getBins` (a):** *High* — primary (AFML Ch.3) + official mlfinlab source [1][4].
- **Latest-signal-wins bias (a):** *High (reasoned from primary)* — direct target↔P&L mismatch; the intrabar both-touch "stop-first" tie-break is *practitioner-consensus*, not a theorem [1][4].
- **Average-active-bets netting (b):** *High* — AFML Ch.10 Snippets 10.2/10.3 and mlfinlab `bet_sizing` reproduce identically [3][6].
- **Engine handling (b):** *High* for documented features; *Medium* that neither vectorbt nor backtrader natively averages concurrent bets (pre-aggregation required) [7][8][9].
- **Sortino target-semideviation formula (c-i):** *High* — Sortino & van der Meer (1991), Sortino & Price (1994) [11][12]; full-N denominator frequently mis-implemented [13].
- **Turnover / holding-period (c-ii,iii):** *Medium* — genuinely inconsistent definitions; one-way notional recommendation is reasoned judgement aligned with Grinold–Kahn [15][16][17].
- **Half-spread + GK √-impact (c-iv):** *High* on functional form [18][19]; *Medium* on calibration — a literal page-level GK equation quote was not retrievable online, and Almgren et al. (2005) rejected the pure ½-power for a ~0.6 exponent, so test the exponent [19][20].
- **DSR / SR₀ / N-estimation (d-i):** *High* on formulae [24][25]; *Medium* on effective-N (clustering/ONC is recommended but inherently approximate for a single backtest) [26][27].
- **CSCV-PBO and MinBTL (d-ii,iii):** *High* — primary journal sources [28][31][32].
- **2025 Sharpe paper (d):** *High* on existence/scope; *Medium* on applicability detail [33][34].

---

### References

*Net-new (not in nlr-cw-v1.md):*
- [4] Hudson & Thames, *mlfinlab* — `labeling.py` (`get_events`/`get_bins`, Snippet 3.x); DeepWiki "Triple Barrier Method." github.com/hudson-and-thames/mlfinlab; deepwiki.com/quantopian/mlfinlab.
- [5] Hudson & Thames, *mlfinlab* — `sampling/concurrent.py` (number of concurrent labels, average uniqueness).
- [6] Hudson & Thames, *mlfinlab* — `bet_sizing/ch10_snippets.py` & `bet_sizing.py` (`get_signal`, `avg_active_signals`, `mp_avg_active_signals`, `discrete_signal`). readthedocs / github.
- [7] vectorbt documentation — `Portfolio.from_signals`/`from_orders`, stop/TP/time exits, stats (Sortino). vectorbt.dev.
- [8] vectorbt(.pro) documentation — multi-asset futures with contract multipliers (ES/NQ); position accumulation.
- [9] backtrader documentation — Analyzers (SharpeRatio, SQN, TradeAnalyzer). backtrader.com/docu/analyzers.
- [10] backtrader documentation — SharpeRatio source/annualisation (RATEFACTORS 252/52/12). backtrader analyzers reference.
- [11] Sortino, F.A. & van der Meer, R. (1991) "Downside Risk: Capturing What's at Stake in Investment Situations," *J. Portfolio Management* 17(4):27–31. doi:10.3905/jpm.1991.409343.
- [12] Sortino, F.A. & Price, L.N. (1994) "Performance Measurement in a Downside Risk Framework," *J. Investing* 3(3):59–64.
- [13] quantstats — GitHub issue #78 (rolling_sortino downside-deviation denominator). github.com/ranaroussi/quantstats.
- [14] quantstats / empyrical — `sortino_ratio` (downside deviation via clip-to-zero over full sample). Red Rock Capital Sortino whitepaper.
- [15] Grinold, R.C. & Kahn, R.N. (2000) *Active Portfolio Management*, 2nd ed., Ch.16 "Transactions Costs, Turnover, and Trading" (annualised cost = round-trip cost ÷ holding period; holding period ≈ 1/turnover). McGraw-Hill.
- [16] Grinold, R.C. & Kahn (2000) — "sigma-root-liquidity" impact model: impact ∝ σ·√(Q/V), coefficient of order one.
- [17] Hudson & Thames, *mlfinlab* — `backtest_statistics` & `bet_sizing` (PSR, DSR, MinTRL, average holding period). readthedocs.
- [18] Grinold & Kahn (2000) — square-root market-impact functional form (as [16]).
- [19] Almgren, R., Thum, C., Hauptmann, E. & Li, H. (2005) "Direct Estimation of Equity Market Impact," *Risk* 18(7):58–62 (≈700k Citigroup orders, Dec 2001–Jun 2003; rejects pure ½-power for ~3/5 power).
- [20] Empirical square-root / market-impact literature (Bouchaud, Tóth et al.) corroborating impact ∝ σ·√(Q/V).
- [21] Kyle, A.S. (1985) "Continuous Auctions and Insider Trading," *Econometrica* 53(6):1315–1335 (linear λ impact — contrast).
- [22] Almgren, R. & Chriss, N. (2000) "Optimal Execution of Portfolio Transactions," *Journal of Risk* 3(2):5–39 (permanent+temporary impact decomposition).
- [23] Lo, A.W. (2002) "The Statistics of Sharpe Ratios," *Financial Analysts Journal* 58(4):36–52 (SR standard error; non-normality; √-annualisation caveat under serial correlation). Mertens (2002) non-normal SR standard error.
- [26] Liana Ling, "Deflated Sharpe Ratio" & "Probability of Backtest Overfitting (PBO)," Balaena Quant Insights (practitioner guidance on choosing/overestimating N) — *practitioner source*.
- [27] López de Prado, M. & Lewis, M.J. (2018/2019) "Detection of False Investment Strategies Using Unsupervised Learning Methods" (ONC clustering for effective N).
- [28] Bailey, D.H., Borwein, J.M., López de Prado, M. & Zhu, Q.J. (2017) "The Probability of Backtest Overfitting," *Journal of Computational Finance* 20(4):39–69 (CSCV; SSRN 2326253).
- [29] Bailey et al. CSCV algorithm (Algorithm 2.3) — davidhbailey.com/dhbpapers/backtest-prob.pdf.
- [30] CSCV/PBO logit-rank computation (λ = ln(ω/(1−ω)); PBO = P[λ≤0]) — primary paper [28].
- [33] López de Prado, M., Lipton, A. & Zoonekynd, V. (2025) "How to Use the Sharpe Ratio," SSRN doi:10.2139/ssrn.5520741 (*already in nlr-cw-v1.md*; re-listed for the closed-form non-normal/serially-correlated SR distribution and hybrid FWER–FDR).
- [34] ADIA Lab / Rebellion Research summaries of [33] (five Sharpe-ratio errors; PSR/MinTRL/DSR/oFDR).

*Already in nlr-cw-v1.md (re-listed only where load-bearing):*
- [1] López de Prado, M. (2018) *Advances in Financial Machine Learning*, Wiley — Ch.3 (triple-barrier, getEvents/getBins), Ch.4 (concurrency), Ch.10 (bet sizing), Ch.14–15 (backtest statistics).
- [2] López de Prado (2018) Ch.4 — overlapping outcomes / number of concurrent labels / average uniqueness.
- [3] López de Prado (2018) Ch.10 — Snippets 10.1–10.3 (`getSignal`, `avgActiveSignals`, `discreteSignal`).
- [24] Bailey, D.H. & López de Prado, M. (2014) "The Deflated Sharpe Ratio," *J. Portfolio Management* 40(5):94–107 (SSRN 2460551).
- [25] López de Prado (2018/2020) — DSR, expected-maximum-Sharpe SR₀, effective-N clustering.
- [31] Bailey, Borwein, López de Prado & Zhu (2014) "Pseudo-Mathematics and Financial Charlatanism," *Notices of the AMS* 61(5):458–471 (MinBTL, Theorem 3.1; doi:10.1090/noti1105).
- [32] Bailey et al. (2014) MinBTL figure (45 configs / 5 years) — davidhbailey.com/dhbpapers/backtest-pseudo.pdf.