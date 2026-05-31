# T3.03 Coursework — Running Action Tracker

**Purpose.** Living to-do list for the Alken metamodel coursework. Tracks *what* needs doing and its *status*; the *how/why + citations* live in `T3_03_CW_Breakdown_v2.md`. Every item has a stable ID (e.g. `S2.4`) so updates can reference it.

**Update protocol.** Each time a coursework output (notebook, code, draft, predictions CSV) is shared and evaluated, this file is updated in the same pass:
1. Mark completed items ✅ and note any that regressed (✅→⚠️).
2. Add a dated entry to the **Evaluation Log** (what was reviewed, what changed here).
3. Add/modify items the review surfaced (new IDs continue the section's numbering).

**Status legend:** ☐ not started · ◐ in progress · ✅ done · ⚠️ done-but-issue/needs-fix · ⏸ blocked (see Open Questions) · 🔬 exploratory.

---

## Evaluation Log (newest first)

### 2026-05-31 (PM-9) — Pass-4: F16 re-open + per-instrument embargo + significance-first §6 + PT/TM timing
**Verdict: the honest-negative is re-confirmed and *strengthened* under the expansion.** Re-ran the full pipeline with the per-instrument embargo (S2.6) and the F16 concept-drift re-open (S1.8-b), led §6 with significance (S6.14), made Pesaran–Timmermann the primary §5 timing test (S5.10), and evaluated CER-gated sizing (S6.15). All new behaviour is RED-first TDD with known-answer tests; Ruff clean; the three-CSV deliverable re-emits byte-identical.

**Pass-4 results (no edge — strengthened, not weakened):**
- **AUC ≈ 0.5** (equity 0.568 / energy 0.536 / metals 0.532) — the expansion did NOT manufacture edge; the two HALT tripwires (expansion clears deflation; §5 audit flips) did not fire.
- **§6.14 significance (PRIMARY):** pooled net Sharpe 1.31 → **t = 0.93 (n=127, n.s.)**; studentised stationary block-bootstrap 95% CI per-period **[−0.04, +0.19] contains 0**; PSR(0)=0.82; MinTRL=399d vs 127; Ljung–Box p=0.010 (serial-correlated → √252 overstates).
- **§6.8 deflation (demoted to corroboration):** pooled DSR ladder **[0.61 → 0.20]** over N∈[3,15,30,60]; PBO 0.35; MinBTL [0.73→3.14]y vs ~0.5y. Fails a fortiori under the higher, feature-selection-aware N.
- **§5.10 timing:** **PT (primary, base-rate-aware) negative everywhere** (pooled −2.31, p=0.99). TM mixed (Equity −4.51 sig-neg); the *pooled* TM +1.18 (t=2.55) is an **aggregation artefact** (convex big-move capture from the exit mechanism, not directional skill). H–M relabeled a base-rate-sensitive proxy.
- **§6.15 sizing:** leakage-safe smooth taper CER +5% (immaterial, below the 10% materiality margin); the per-instrument κᵢ variant is **circular** (κᵢ OOS-estimated) → **retain flat κ=0.25** (deliverable unchanged).

**Two premises corrected (surfaced):**
- **F12/F13/F15 and F17/F3 were ALREADY wired** (verified empirically: 13 regime cols incl. `f17_*`; F12/13/15 via `assemble_engineered_ext`) — correcting PM-7's "the metamodel does not use F12–F17". The only genuine not-yet-wired causal family was **F16** (concept-drift), now ingested under the per-fold causal recompute; parquet-only families stay excluded (a `test_no_metamodel_module_reads_frozen_parquet` guard asserts it).
- **F17 fit discipline confirmed conforming** (fit on the ≤`fe_train_end` contiguous prefix; the deliverable OOS is leak-free since fit_end < predict window) — guarded by `test_f17_fit_uses_only_pre_fit_end_data`.

**Invariants re-verified:** byte-identical re-emit (two full runs, 4 CSVs identical, canonical==calibrated, 1011 rows); per-instrument embargo applied (each instrument's purged window = its `embargo_p90`); F16 truncation-invariant; no module consumes `feature_matrix.parquet`; CPCV=15.

**XT.3 (MLflow):** the pass-4 re-run is a *fixed* roster × 3 classes × CPCV over a +1-feature set — the "run count explodes" trigger does NOT fire; `experiment_log.csv` suffices. **Recommend NOT adopting MLflow** this pass; revisit only if a genuine hyperparameter sweep is opened.

### 2026-05-30 (PM-8) — Evaluated the 3 round-2 literature outputs (LR-6, LR-7, LR-8)
**Verdict: all three INCLUDE — high quality, primary-source-grounded, honestly flagged. LR-6 is the most consequential and changes how the §6 headline must be reported.** Evaluated against `/empirical-finance /aqms-python /financial-engineering /python-quant-finance /sts-ml /bdfin-ml`. Per-output decisions in the round-2 **Literature Integration** block below; findings folded into items.

**Cross-checks (numerically verified, not taken on faith):**
- **LR-6 arithmetic reproduces to the decimal.** ann Sharpe 1.36/1.55 over T=128 → **t = 0.97 / 1.10** (not significant at 5% even before deflation); IID 95% CI ≈ [−1.4, 4.1] / [−1.2, 4.3] (both straddle 0); PSR(0) ≈ 0.83/0.86 (<0.95); **MinTRL ≈ 371/286 daily obs vs 128 available**. This is decisive and **strengthens** (does not contradict) the pass-3 gate: same "doesn't clear" conclusion, but correctly reframed as *insufficient evidence*, not demonstrated failure. The `empirical-finance` skill independently endorses block-bootstrap for short-T/autocorrelated Sharpe inference (LR-6's primary recommendation) and ships `certainty_equivalent` (LR-7's eval target). → S6.8 amended, new **S6.14**.
- **LR-7 shrinkage formula is monotone-correct.** Baker–McHale k* = ((b+1)p−1)²/[((b+1)p−1)²+(b+1)²σ²] decreases in σ (more estimation uncertainty → shrink more), verified. The recommendation (smooth taper replacing the hard 0.55 floor; per-instrument κᵢ ≈ eᵢ²/(eᵢ²+σᵢ²)) directly addresses the S6.13 concentration without double-penalising over-confidence calibration already removed. Honest "evaluate against CER; if no OOS improvement the flat heuristic stands." → new **S6.15**.
- **LR-8 anchors the null in proven algebra.** Grinold's Fundamental Law (IC≈0 ⇒ IR≈0, *regardless* of breadth/sizing) — present in `/aqms-python` `ir_from_skill` — is the rigorous reason a directional secondary can't rescue a near-zero-IC primary. Confirms the H–M "mechanism-not-timing" reading and adds Treynor–Mazuy + Pesaran–Timmermann as corroboration. → S6.12 amended, new **S5.10**.

**Three findings that became items (beyond the framing each output supplies):**
- **I — the §6 Sharpes are not statistically significant, full stop (LR-6).** This is bigger than "DSR<0.95": at T=128, t≈1 means the raw 1.36–1.55 Sharpes don't clear zero *before any deflation*. §6 must lead with the t-stat + a **studentised stationary-block bootstrap CI** (LR-6's primary inference; `arch.bootstrap`), with DSR/PBO demoted to caveated, sensitivity-banded corroboration. → **S6.14**.
- **J — a verified citation error to delete (LR-8).** "Kang & Kim (2025)" — which the PM-2 LR-1 integration listed as a *primary* source — does **not exist**; it's a conflation of Fu, Kang, Hong & Kim (2024), a GA/triple-barrier pairs-trading paper, not a meta-labelling-null paper. **Must be removed from the write-up and the LR-1 citation set.** → folds into **X.7**.
- **K — H–M sign-convention audit before publishing (LR-8).** `PerformanceAnalytics::MarketTiming` (and equivalent) admit two regressor conventions that **flip the timing-coefficient sign**. Our headline rests on *negative* H–M z; confirm the regressor definition in the actual code path against a hand-worked toy case before the sign goes in §5. → **S5.10**.

**Note (CSCV typo, LR-6):** the canonical CSCV paper's "12,780 combinations for S=16" is a propagated typo; C(16,8)=12,870. Minor; flag if we cite the figure.

### 2026-05-30 (PM-7) — Reviewed the shared `stml` feature-engineering base (175-feature matrix, scope/redundancy/provenance maps, missing-data report)
**This is the upstream layer your metamodel consumes, not the metamodel itself.** Reviewed for impact on the build. Five interactions, all folded into items below.
- **Confirms the hardest design call.** The base freezes TF features at a single global `fe_train_end=2021-07-01`; your CLAUDE.md forbids consuming `feature_matrix.parquet` for exactly that leakage reason → your per-fold recompute is vindicated. One write-up sentence (→ X.10).
- **~60 unused features (the main option).** Base = 175 cols across F1–F17 (E=108/TF=65/LI=2); your metamodel ingests ~115 core and does **not** use F12 path-structure, F13 wavelet, F15 conditional-risk, F16 concept-drift, F17 HMM (Sreeram/Harry contributions). Given the four-way ≈0.5 result, low expected payoff + locked-set risk → documented option/limitation, not obligation (→ S1.8).
- **Per-instrument embargo (the one potential correctness item).** `instrument_scope.json` gives `embargo_p90` up to **33d (ng1s), 26d (ho1s)**; your metamodel uses a uniform 1%-of-T embargo, likely **too small** for energy fold boundaries (→ S2.6).
- **ng1s mechanism.** Base reports `n_eff_gate=2` for ng1s (2 post-embargo runs) — the *cause* of your EX.5 IC=NaN; cite it (→ S5.9/S6.13 amended).
- **Primary signal identified.** Catalog states the primary is a **short-horizon mean-reversion / counter-trend** strategy (F1, `f1_mr_score_20`); matches your ~0.55 hit-rates → names the regime for LR-8/§3/§5/§6 (→ S6.12 amended, LR-8 re-scoped).
- **Data-cleaning policy to verify.** Base ships the authoritative `load_clean_data` policy (keep 765 zero-vol settle rows, drop 3 Sunday rows, never ffill structural NaNs); confirm the metamodel loads through it (→ X.11).

Net: 1 potential correctness fix (S2.6), 1 meaningful option (S1.8), 2 verifications (X.10/X.11), framing strengtheners. **Nothing overturns the honest-negative; the `n_eff`/signal-type disclosures make it more defensible.**

### 2026-05-30 (PM-6) — Evaluated pass-3 (deflation gate, per-class calibration, utility tests, reconciliation, write-up sweep)
**Verdict: the project is now methodologically complete and, for a methodology-graded brief, in excellent shape — the honest-negative result is rigorously evidenced from four independent angles.** Pass-3 closed every critical item I flagged in PM-5. No new correctness defects; the remaining items are presentational/verification, not blocking.

**Note on my own prior numbers:** pass-3 correctly distrusted the "108/107/108" feature counts I carried in PM-5 and **measured** them — the truth is **139–140** (core 115 + macro 16 + regime 5 + instrument one-hots 3–4). My PM-5 figure was wrong; the measured X.8 result supersedes it. Good catch by the build.

**Independently validated (all green):**
- **Three prediction files + weights all hold the contract** — 1011 rows, 11 instruments, sorted, key parity, ∈[0,1], no NaN. **Canonical `metamodel_predictions.csv` == calibrated file** (resolves S6.11 — the deliverable *is* calibrated). Raw file retained for the §3 story.
- **Platt is exactly rank-preserving** — per-instrument Spearman(raw,cal)=1.000 for all 11 → AUC genuinely unchanged, as the method requires. Calibration compresses p̂ to [0.37, 0.68] (sensible given the thin edge); mean |raw−cal|=0.12.
- **Deflation gate is internally correct** — DSR **decreases in N** (right direction) for every class; **no class clears DSR≥0.95** across N∈[N_eff→N_raw]; PBO reads sensibly (equity 0.496 ≈ coin-flip overfit, metals 0.060 low because its Sharpe is negative — nothing to overfit); MinBTL exceeds the ~0.5y OOS everywhere. This is a textbook honest-negative.
- **Accounting reconciled (S6.10 done)** — the `gross−cost≠net` gap is the **geometric-vs-arithmetic basis** difference (`∏(1+r)−1` vs `Σcost`), now reported as `cost_drag_compounded` — a documentable identity, not a bug. Turnover (one-way notional) vs holding (busday trade-based) documented as different bases.
- **Experiment log now carries Brier + precision** (XT.2 gap partly closed — still only the 3 shipped-selection rows).

**The headline finding — now evidenced four independent ways (this is the strongest part of the submission):**
1. **CPCV AUC ≈ 0.5** (Energy 0.49, Equity 0.57, Metals 0.52) — weak discrimination.
2. **Cluster MDA ≈ 0/negative** across all clusters/classes — no OOS permutation value.
3. **DSR fails 0.95; PBO ≈ 0.5 (equity)** — selection is overfit; deflated Sharpe doesn't clear.
4. **Henriksson–Merton: NO timing skill** — Equity/Metals/All-11 have **z<0, p>0.95** (timing *worse* than random); only Energy is ≈neutral (z=0.28).
→ The positive barrier-exact Sharpes (Equity 1.36, All-11 1.55) are **exit-mechanism + vol-targeting, not directional skill**. Four lenses agree. **§6 must lead with this convergence** — it's a more compelling methodology result than any positive Sharpe would be. Already the plan's intent; confirm the write-up states the four-way agreement explicitly. → new **S6.12**.

**Two findings that became new items:**
- **G — calibration×Kelly-floor concentration.** Calibrated p̂ tops out at 0.684, so only **7% of bets clear p≥0.60** and the per-instrument floor-pass-rate is wildly uneven (cl1s/gc1s/ng1s 100%, fesx1s 30%). The floor binds cleanly (0 violations — sizing is correct), but the strategy now sizes off a **thin high-confidence slice → concentration risk**, and ng1s passes the floor 100% of the time *despite IC=NaN*. Worth a sentence in §6 + interacts with S5.9. → new **S6.13**.
- **H — XT.2 still thin for a fully-defensible DSR.** N and V[SR̂] are computed from the 5×3 horse-race candidates, which is reasonable, but the per-trial OOS series backfill (the richer CSCV matrix) is the part that makes PBO airtight. Fine as-is for the grade; note the basis. → folds into XT.2 (leave ◐-note).

**Status flips (→✅):** S3.9, S5.8, S6.8, S6.10, S6.11, X.7, X.8 (measured), plus S6.9 (framing done in §6). Remaining ☐: S4.8/S5.9 (write-up — may already be drafted; verify), X.9 (ALFRED — likely documented-as-limitation), S6.12/S6.13 (new, small), XT.3 (deferred).

**On literature research:** the project is now self-consistent and the open items are presentational. I assessed whether further `research` would *improve* it — three candidate scopes identified, all genuinely additive rather than make-work; detailed briefs delivered as separate documents (LR-6, LR-7, LR-8). None is blocking; LR-6 is the only one I'd call clearly worthwhile.

### 2026-05-30 (PM-5) — Evaluated pass-2 outputs (real-data §4, NN+CPCV default, barrier-exact backtest, EX.1/3/4/5, macro, re-emitted CSVs)
**Verdict: large, high-quality progress — but the §6 headline is not yet safe to claim.** Most open items are now done on real data; the deliverable CSVs re-emit and validate. Two correctness issues and one missing gate must be closed before the strategy result goes in the write-up.

**Independently validated:** re-emitted CSVs hold the contract — 11 instruments, 1011 rows, sorted, key parity, predictions ∈ [0.033, 0.996] (range widened from pass-1's [0.047, 0.944], consistent with the torch-MLP/CPCV default now shipping), no NaN. `coverage_caveat.csv` per-instrument counts **exactly match** the prediction file. `experiment_log.csv` exists (XT.2 done) — though only 3 rows (the shipped selections), missing the EX/sweep runs and with empty `oos_brier/precision`.

**Status flips (→✅ / ⚠️ below):** S3.3, S3.7, S3.8 (NN-torch + CPCV-as-selection now in the default path; Energy selected torch-MLP — a neural variant won), S4.7 (run on real matrix), EX.1, EX.3, EX.4, EX.5, S5.7 (caveat emitted), XT.2, and S6.7 (barrier-exact backtest + turnover/holding/costs built). S6.8 **not done**.

**§4 real-data result (S4.7) is clean and gradeable.** Energy/Metals: vol-cluster (`f2_vol_*`, GK/Parkinson, BB-bandwidth) dominates MDI+SHAP; Equity: a momentum/OI cluster (`f6_ma_cross`, `f6_adx`, `f7_oi_*`) leads. **But note: cluster MDA is ≈0 or negative for *every* cluster in all three classes** (e.g. Energy all 3 "near-zero-MDA noise"). MDI/SHAP rank groups, but MDA says no cluster has reliable *out-of-sample permutation* value — fully consistent with EX.1's AUC≈0.5 and worth stating plainly (don't let MDI/SHAP ranking imply OOS edge MDA can't support). → new **S4.8**.

**Real pooled feature count is ~107–108, not 124.** The methodology already says "measure, don't assert 124," and the artifacts confirm 107–108. Anywhere "124" survives in prose must be corrected. → new **X.8**.

**Three findings that became action items (the important part):**
- **D — §6 headline rests on an exit-rule artifact + missing deflation.** Same predictions, swapping simple→barrier-exact exit **flips Equity (−0.37→+1.39) and Metals (−0.37→+0.31) from losing to winning**; Energy +1.18→+1.48. The P&L is dominated by the *exit/holding mechanism*, not classification skill (EX.1: AUC barely >0.5). That sensitivity is itself a finding — but **S6.8 (DSR/PBO/MinBTL) was NOT run**, so Sharpes of 1.2–1.5 carry no overfitting deflation. This is exactly the pseudo-mathematics case the gate exists for. **S6.8 stays ◐ and is now the top priority before any "strategy works" claim.** → **S6.8 (escalate), S6.9**.
- **E — backtest accounting doesn't reconcile.** `gross − cost ≠ net` for Equity (0.0643 vs 0.0629), Metals (0.0219 vs 0.0196) and All-11 (0.1293 vs 0.1173) — a ~1–2% unexplained term (financing/rounding/sizing step?). Energy reconciles. And **turnover↔holding violate Grinold–Kahn `hold≈252/turnover`** except for Metals (Energy implies 28.5d vs reported 3.2d) → turnover and holding are on different bases (one-way notional vs trade-count), the ambiguity LR-2 warned about. Reconcile or document. → new **S6.10**.
- **F — are the emitted probabilities (and Kelly weights) calibrated?** EX.4 shows raw ECE 0.14–0.18 (materially miscalibrated), Platt > isotonic in all three classes (textbook small-N, per `/bdfin-ml`). Kelly consumes p̂ directly, so if `metamodel_predictions.csv` / `strategy_weights.csv` use **raw** probabilities the deliverable is mis-sized. Must confirm whether emit applies Platt; if not, calibrate before sizing (keep the raw prob as a separate column for the §3 AUC story). → new **S6.11 / S3.9**.

**EX cross-checks against skills:** EX.1 — all three primaries hit ~0.55–0.56, so the Equity/Energy split is a *discrimination* difference, not a primary-quality one (matches LR-1's framing). EX.5 — **ng1s IC = NaN** (near-zero-variance signal) and ho1s/ng1s IRs unstable, corroborating the thin-coverage caveat; the hivol/lovol hit-rate split is a usable regime feature. EX.3 — surface stayed diagnostic (no feedback to locked `pt_sl`/`max_holding`); firewall held. EX.4 Platt-over-isotonic choice is the right small-N call.

### 2026-05-30 (PM-4) — Next build confirmed: S4.7 → S6.7/S6.8 → S1.7 → NN
User approved the build order. **Active work item = S4.7 + S6.7 + S6.8** (run §4 on real data; barrier-exact backtest with the LR-2 metric set + DSR/PBO gate), then S1.7 (macro block), NN (S3.7) last. Recommended execution defaults below pending user confirmation on three setup points (logged so the next session isn't re-litigated):
- **Delivery mode:** **standalone modules + mutation-resistant tests** Claude hands over to drop into `metamodel-apb/` (default). Faster cross-check if the user pastes `pipeline.py` / `cross_validation.py` / `evaluation.py` / `backtest.py` signatures, or uploads the repo+data to run here. *Code must be written against the CLAUDE.md interfaces (vendored `cluster_feature_importance.py`, `cross_validation.py` PurgedKFold/CPCV, `sizing.py`); grep each signature before reuse.*
- **S4.7 first class:** **Energy** (reference-first, the cleaner class and the one whose §4 result is most diagnostic given its 6/15 CPCV weakness) → then fan out to Equity/Metals.
- **S6.7 netting:** **average-active-signals** (LR-2) — the recommended, decision-grade choice; bounds per-instrument exposure to ±1, never sum.
- **Guardrails carried in:** purge/embargo via `t1` on the cluster MDA (don't reintroduce the shuffled-KFold leak); stop-first intrabar tie-break (flag as practitioner convention); cost = ½-spread + GK √-impact with c∈[0.5,2] / exponent∈[0.5,0.6] sensitivity; Sortino full-N denominator; log every backtest's returns so DSR's N and V[SR̂] are measured, not guessed.

### 2026-05-30 (PM-3) — Q2 closed: §6 constraints doc unavailable
The 20 May constraints doc is confirmed unavailable. Decision: keep the §6 strategy track on the **clearly-marked lit-review-default stub** (κ=0.25, p̂≥0.55, 25% ann. vol); the write-up must flag the constraint set as a documented assumption rather than the released spec. S6.5 moved ⏸→✅ (no longer blocking); Q2 resolved. No other items affected — `strategy_weights.csv` already emits under these defaults.

### 2026-05-30 (PM-2) — Evaluated the 5 literature-research outputs (LR-1…LR-5)
**Verdict: all five INCLUDE — high quality, well-sourced, honestly hedged.** Evaluated against `/sts-ml /aqms-python /empirical-finance /bdfin-ml /financial-engineering /python-quant-finance`. Each output carries explicit verification flags on in-press/aggregator items (good epistemic hygiene). Per-output decisions and integration in the new **Literature Integration** section below; headline findings folded into action items.

**Cross-check highlights:**
- **LR-2 is the strongest** and is decision-grade. Its barrier-exact exit, **average-active-signals netting** (not summing — sum gives phantom 150–200% leverage), and DSR/CSCV-PBO/MinBTL gates all match LdP primary sources and the `/python-quant-finance` + `/empirical-finance` references. Crucially, **its Sortino full-sample-T denominator matches the `empirical-finance` skill's own code** (`clip(upper=0)`, `.pow(2).mean()`) — so the "downside-deviation over full N, not over the negative subset" convention is authoritative, not just LR-2's opinion. → upgrades S6.4/S6.7.
- **LR-1 confirms our empirical result is the expected one.** Meta-labelling edge is thin (peer-reviewed triple-barrier ML ≈0.6 AUC), and the Equity-works/Energy-fails split is *consistent with published patterns* (weak/unstable commodity ML predictability) — but at **medium confidence by synthesis**, no head-to-head study, and **no peer-reviewed "AUC≈0.5 collapse" paper exists** (C10 = Low). The Henriksson–Merton market-timing test it recommends **is implemented in `/empirical-finance`** (`predictive_regressions.md`) → directly usable. → feeds S5.8, EX.4, EX.5.
- **LR-5 caught a real misattribution** (high value): the US–Japan 0.39/0.27 and US–UK 0.67/0.48 regime correlations are **Guidolin–Timmermann (St. Louis Fed WP 2005-034), not Ang–Bekaert 2002** (which excludes Japan and reports US–UK ≈0.60/0.44). Carver p.146 confirmed; MZB "56%" unverifiable → keep deleted. Plus the Ang–Bekaert Wald p=0.1556 caveat (the correlation *difference* is statistically weak). → resolves X.7; these are write-up edits, not code.
- **LR-4 reinforces de-prioritising the NN.** At ~500 rows, trees are the favoured class (Grinsztajn 2022; Shwartz-Ziv & Armon 2022), and the VSN's softmax gate won't control overfitting (1 GRN/feature → params scale with 124 features). Its **cluster-representative reduction reuses the §4 Mantegna clustering we already compute — free and interpretable**. → re-scopes S3.7/EX.2.
- **LR-3 confirms the feature-derivability split** found earlier: **carry/basis is the #1 commodity predictor but is NOT derivable from front-month-only OHLCV** (needs a 2nd contract); everything else (EIA surprise, VRP, VIX-slope, HY/IG OAS, gold↔real-rates, copper↔China-PMI, DXY, MOVE) maps to `additional_data.xlsx`. Its **PIT-lag release-calendar table is the actionable core** of S1.7. → upgrades S1.7.

### 2026-05-30 (PM) — Evaluated the `metamodel-apb` build (methodology.md, build-log, both CSVs, CLAUDE.md)
**Verdict: strong, gradeable execution.** All 8 lit-review commitments implemented and traceable to a module; both deliverable CSVs validate against the brief; the write-up is honest about where the metamodel adds no edge. Evaluated against `/sts-ml` — the two bug-fix claims (distance metric `cluster_feature_importance.py:30`; `KFold(shuffle=True)` MDA leak `:105`) are **verified against the real skill code**, and the fixes (Mantegna `√(1−|ρ|)`, injected PurgedKFold) are correct.

**CSV validation (both files):** schema exact (`date,instrument,prediction` / `…,weight`); 11 instruments; 128 OOS days (Jan–Jun 2022); predictions ∈ [0.047, 0.944], no NaN, none out of [0,1]; rows sorted by (date,instrument); keys identical across both files; weights 50.9% exactly zero (consistent with the p̂≥0.55 Kelly floor + signal=0 days). ✅ contract met.

**What's done (flipped to ✅ below):** S0.2–S0.5, S1.1/1.3/1.4/1.5/1.6, all S2.x, S3.1–3.6, all S4.x (implemented+tested, synthetic), S5.1–5.6, S6.1–6.3/6.6, all B.x, X.1–X.6, XT.1/XT.2 (the run-logging exists as the CPCV/per-instrument result tables).

**Three findings that became new action items:**
- **A. `ho1s` has only 2 OOS prediction rows, `gc1s` only 30** (the primary signal is rarely non-zero for them in H1 2022). The §5 AUCs for ho1s/ng1s are computed on *training-sample* label counts (61/68) and are flagged unreliable — but the **OOS deliverable** for those instruments rests on 2–56 rows. Needs an explicit per-instrument OOS-coverage caveat in the write-up and a decision on whether near-empty instruments should emit at all. → **S5.7**
- **B. §4 cluster importance has never been run on the real 124-feature matrix** — honestly flagged, but it's a 10-mark section currently substantiated only on synthetic data. Running it on real Energy is low-effort, high-mark-security. → **S4.7**
- **C. The NN family + CPCV/nested are implemented but off the default path** (VSN intractable at 124 features × CV without dimensionality reduction). This is the user's open question. → **S3.7, S3.8, EX.1**

**Items newly surfaced or re-scoped:** S1.7 (PIT-lagged macro block — the one honestly-flagged remaining feature item), S3.7/3.8, S4.7, S5.7, S6.7, plus exploratory EX.1–EX.5 and literature-research scopes LR.1–LR.5 (new section). Carried-forward blockers Q2/Q3 partially resolved (sibling skills identified for the backtest).

### 2026-05-30 (AM) — Tracker created
Baseline seeded from `T3_03_CW_Breakdown.md` (v1) and `_v2.md` (v2) + method evaluation across PS1–PS8. Added experiment-tracking XT.1–XT.3.

---

## Planning & scoping
| ID | Status | Item |
|----|--------|------|
| P.1–P.4 | ✅ | Brief decomposed; methods evaluated; validation options compared; 8 commitments mapped |
| P.5 | ✅ | Scope & split confirmed: all 11 across 3 asset-class metamodels; purged-kfold selection → CPCV diagnostic → Jan–Jun 2022 hold-out |

## Section 0 — Setup, data, submission contract
| ID | Status | Item |
|----|--------|------|
| S0.1 | ✅ | Coverage decided: all 11 (Energy reference-first, then fan-out) |
| S0.2 | ✅ | Single data loader (`stml.io.load_clean_data`), read-only |
| S0.3 | ✅ | Internal OOS = Jan–Jun 2022 hold-out; hidden Jul–Dec 2022 untouched; window config-driven |
| S0.4 | ✅ | Seeds module-wide (`seeding.set_seeds`), single-thread native kernels |
| S0.5 | ✅ | Deterministic CSV emitter; byte-identical re-emit (tested) |

## Section 1 — Feature Engineering (20)
| ID | Status | Item |
|----|--------|------|
| S1.1 | ✅ | Technical block (F1/F2/F5–F10/F12–15), all shifted/causal; right-edge truncation-invariance test |
| S1.2 | ◐ | Futures-specific features — macro cross-asset derivable from `additional_data.xlsx`; **per-instrument calendar/basis spread NOT derivable** (front-month only) — correctly flagged |
| S1.3 | ✅ | Backward trend-strength feature (deterministic ±20 cap, avoids global-variance leak) |
| S1.4 | ✅ | GMM regime-probability features (stml F3, reused per-fold) |
| S1.5 | ✅ | HMM regime features — **built online EWMA 2-state Gaussian HMM** (causal/fit-free), honours the time-varying commitment; stml static HMM as supplementary |
| S1.6 | ✅ | One-line "what it captures" per feature (block table) |
| **S1.7** | ✅ | **Built in pass-2** — `macro.py` PIT-lagged loader over `additional_data.xlsx` (daily T+1 + EIA/PMI publication-lag calendars); `use_macro=True` in the shipped default. Carry/basis correctly omitted (front-month only). *Remaining: ALFRED-vintage spot-check on TIPS/BE10Y (validation gate) is good-practice, not yet run → folds into X.7-style verification.* |
| **S1.8** | ✅ | **~60 unused features from the shared `stml` base.** The base ships **175 cols (F1–F17)**; the metamodel ingests ~115 core and omits **F12 path-structure (Hurst, variance-ratio, efficiency-ratio), F13 wavelet energy, F15 conditional-risk/first-passage, F16 concept-drift, F17 3-state HMM** (Sreeram/Harry). Decision needed: (a) **document as a feature-scope limitation + future work** (default — low expected payoff given AUC≈0.5, and ingesting now risks re-opening the locked set before the hidden-half test), or (b) a scoped extension if pursuing the competition cut. If (b): reuse the per-fold recompute path (never the frozen parquet, per X.10), re-lock, re-run CPCV. **Recommend (a).** |

## Section 2 — Labelling: Triple-Barrier (20)
| ID | Status | Item |
|----|--------|------|
| S2.1 | ✅ | Meta-labels on primary≠0 days, sign of side-adjusted P&L at first touch |
| S2.2 | ✅ | Vol-adaptive ±k·σ̂ₜ (GK-based), vertical T_max=10; justified |
| S2.3 | ✅ | T_max justified in prose |
| S2.4 | ✅ | `t1` first-touch recorded; drives purge/embargo everywhere |
| S2.5 | ✅ | Uniqueness weights (tested: disjoint→1.0, overlap→0.75/0.50); class balance ~50–69% noted |
| **S2.6** | ✅ | **[POTENTIAL CORRECTNESS] Adopt per-instrument embargo.** The shared base's `instrument_scope.json` gives `embargo_p90` (90th-pct constant-signal run length) **per instrument** — up to **33d (ng1s), 26d (ho1s), 19d (rb1s), 14d (cl1s)** — whereas the metamodel applies a **uniform 1%-of-T embargo** (~5d), likely **too small** for the energy names with long signal runs → residual leakage at their fold boundaries. Switch PurgedKFold/CPCV embargo to per-instrument `embargo_p90` (or its class-max), re-run selection + the deflation gate, and check whether any AUC/DSR number moves. The leakage-safe direction is *more* embargo, so this can only tighten the honest-negative |

## Section 3 — Model Development & Comparison (30)
| ID | Status | Item |
|----|--------|------|
| S3.1 | ✅ | Elastic-net logistic (saga) |
| S3.2 | ✅ | XGBoost (PS5 config) + LightGBM (both tree slots) |
| S3.3 | ✅ | Neural: torch-MLP + torch-VSN **now in the shipped default roster** (on cluster-rep-reduced features); Keras-VSN kept off-path (TF non-determinism). Energy selected torch-MLP |
| S3.4 | ✅ | Tuning via PurgedKFold(5)+embargo; AUC + calibration; class weights |
| S3.5 | ✅ | Single `sample_weight = uniqueness × inverse-class-freq` to every `.fit` and every metric |
| S3.6 | ✅ | Comparison done; selected Equity→XGB, Energy→XGB, Metals→logistic; CPCV 15-path diagnostic |
| **S3.7** | ✅ | **NN in default path (pass-2).** Cluster-representative reduction (one medoid per Mantegna §4 cluster) feeds torch-MLP/VSN; EX.2 built all three reducers (cluster-rep promoted, PCA + autoencoder off-path). Energy→torch-MLP won selection. *Still worth: the XGBoost benchmark comparison LR-4 recommended, stated explicitly in the write-up.* |
| **S3.8** | ✅ | **CPCV promoted to selection (pass-2)** via `cv_scheme` config; 15 paths; nested-CPCV path exists for the headline. CPCV ran on real data without degenerating. *Document the small-N variance cost in prose.* |
| **S3.9** | ✅ | **Per-class Platt calibration shipped (pass-3).** Fit on purged-OOS modelling preds (strictly pre-`predict_start`), applied to the deliverable. ECE improves every class (Metals 0.210→0.001, Equity 0.055→0.027, Energy 0.140→0.100); Brier improves; AUC unchanged (monotone, verified Spearman=1.0). One map per class (heterogeneous selected families) |

## Section 4 — Cluster-Level Feature Importance (10)
| ID | Status | Item |
|----|--------|------|
| S4.1 | ✅ | Mantegna `√(1−|ρ|)` distance (bug fix #4) |
| S4.2 | ✅ | `OptimalClusterer` PCA→silhouette-K→K-means |
| S4.3 | ✅ | Cluster MDI (bug fix #1: `'sqrt'`) |
| S4.4 | ✅ | Cluster MDA with **injected PurgedKFold** (bug fix #2) |
| S4.5 | ✅ | Cluster SHAP via TreeExplainer (bug fix #3, the §4 contribution) |
| S4.6 | ✅ | Noise-cluster≈0 + signal-cluster outranks (synthetic) |
| **S4.7** | ✅ | **Run on the real pooled matrix (pass-2).** Energy/Metals: vol-cluster (`f2_vol_*`, GK/Parkinson, BB-bandwidth) tops MDI+SHAP; Equity: momentum/OI cluster (`f6_ma_cross`, `f6_adx`, `f7_oi_*`) leads. ~107–108 feats, not 124. Noise clusters flagged |
| **S4.8** | ✅ | **MDA≈0 honesty — pass-3 re-ran §4 and was tasked to rewrite the prose this way** (MDI/SHAP = in-sample attribution, MDA = OOS reality check ≈0). *Verify the §4 write-up frames it as such; if done, flip ✅.* |

## Section 5 — Model Evaluation (20)
| ID | Status | Item |
|----|--------|------|
| S5.1–S5.6 | ✅ | Per-instrument-before-aggregate; confusion/threshold; blind-primary baseline; purge+embargo via `t1`; deflated-Sharpe note; CPCV 15-path OOS distribution |
| **S5.7** | ✅ | **Coverage caveat emitted (pass-2)** — `coverage_caveat.csv` lists per-instrument OOS row counts; all 11 emit (no abstention). Counts match the prediction file exactly |
| **S5.9** | ✅ | **Thin flag widened (pass-3, verified in uploaded `coverage_caveat.csv`).** New `ic`/`ic_undefined` columns; **gc1s (30), ng1s (56, ic_undefined=True for the run), ho1s (2)** all flagged thin. **Mechanism now citable (PM-7):** the shared base's `instrument_scope.json` gives `n_eff_gate` (post-embargo validation runs) = **ng1s 2, cl1s/ho1s 9, gc1s 11** with `low_power=True` for cl1s/ho1s/ng1s — i.e. the thinness is structural undersampling, not a bug. Cite `n_eff` as the reason in the §5 caveat. All 11 still emit |
| **S5.8** | ✅ | **Utility-aware evaluation shipped (pass-3).** Henriksson–Merton + certainty-equivalent (γ=5) per class. **Headline honest-negative: no class shows positive timing skill** — Equity z=−1.72, Metals z=−2.63, All-11 z=−2.43 (all p>0.95, *worse* than random); Energy z=0.28 (neutral). CER signs track net returns (Metals<0). The cleanest statement of AUC≠P&L |
| **S5.10** | ✅ | **[from LR-8 — verification + corroboration] H–M sign-convention audit, then corroborate.** Our headline rests on *negative* H–M z, but the regressor admits two conventions (up-market indicator vs max(0,−Rm)) that **flip the sign** — `PerformanceAnalytics::MarketTiming` is a documented site of this. **Before §5 prints the sign, confirm the regressor in the actual code path against a hand-worked toy case.** Then add a **Treynor–Mazuy** γ and a **Pesaran–Timmermann** statistic as corroboration (negative/insignificant across all three formally rejects timing skill). Cite Henriksson–Merton 1981 (pp. 513–533, not "–534"), TM 1966, PT 1992 |

## Section 6 — Bonus: Strategy Construction (+10)
| ID | Status | Item |
|----|--------|------|
| S6.1 | ✅ | `primary × take` → L/S/N, sized by p̂ |
| S6.2 | ✅ | Fractional Kelly (κ=0.25, floor p̂≥0.55) |
| S6.3 | ✅ | Vol-target overlay (25% ann.) |
| S6.4 | ✅ | **Superseded by S6.7 (pass-2/3).** Full brief metric set now reported: CAGR/vol/Sharpe/Sortino/MaxDD + turnover + avg-holding + transaction costs + cost_drag_compounded, per class and pooled |
| S6.5 | ✅ | 20 May constraints doc confirmed **unavailable** (2026-05-30 PM-3) → proceed on a **clearly-marked lit-review-default stub** (κ=0.25, floor p̂≥0.55, 25% ann. vol), trivially swappable if the doc surfaces. **Write-up must state the constraint set is a documented assumption, not the released spec.** No longer blocking |
| S6.6 | ✅ | `strategy_weights.csv` emitted, schema-correct |
| **S6.7** | ✅ | **Barrier-exact backtest built (pass-2)** — exit on actual `t1` touch, position netting, + turnover/avg-holding/transaction-costs (½-spread + Grinold–Kahn) and Sortino. Real OOS numbers reported per class + pooled |
| **S6.8** | ✅ | **Deflation gate shipped (pass-3).** Custom `deflation.py` (DSR/MinBTL/CSCV-PBO, TDD'd to closed forms). **No class clears DSR≥0.95** over N∈[N_eff→N_raw]. PBO: Equity 0.496, All-11 0.385, Energy 0.365, Metals 0.060. MinBTL > OOS everywhere. **LR-6 reframing (PM-8): at T=128 the DSR/PBO *numbers* are noise-dominated (γ̂₃/γ̂₄ plug-ins, weak CLT, 64-obs OOS legs) — keep them but demote to caveated, sensitivity-banded *corroboration*, not load-bearing point statistics. The primary inference moves to S6.14.** |
| **S6.9** | ✅ | **Exit-rule sign-flip framed as the finding (pass-3 §6).** Side-by-side simple vs barrier-exact; gain attributed to the exit mechanism, gated by S6.8; no "strategy works" claim |
| **S6.10** | ✅ | **Accounting reconciled (pass-3).** `gross−cost≠net` is the geometric (`∏(1+r)−1`) vs arithmetic (`Σcost`) basis difference → reported as `cost_drag_compounded`; `gross_total − net_total == cost_drag_compounded` to tol. Turnover (one-way notional) vs holding (busday trade-based) documented as different bases |
| **S6.11** | ✅ | **Kelly sizes on calibrated p̂ (pass-3).** `strategy_weights.csv` re-sized; canonical predictions == calibrated. Floor binds cleanly (0 violations) |
| **S6.12** | ✅ | **Foreground the four-way convergence (§6/conclusion):** AUC≈0.5 + MDA≈0 + DSR-fails/PBO≈0.5 + H–M-no-skill all agree the edge is mechanism not skill. **Frame against the named primary (PM-7):** short-horizon mean-reversion (F1, `f1_mr_score_20`) → near-random meta-label outcome is *predicted*, not a surprise. **Anchor in proven algebra (LR-8/PM-8):** Grinold's Fundamental Law `IR=IC·√BR` (in `/aqms-python ir_from_skill`) — IC≈0 ⇒ IR≈0 *regardless of breadth/sizing* — is the rigorous reason a directional secondary can't manufacture skill the primary lacks. Label each claim PROVEN (Fund. Law; H–M/TM sign rule) / ASSUMED PREMISE (LdP exploitable-skill precondition) / EMPIRICAL (MR-AUC-ceiling heuristic) |
| **S6.13** | ✅ | **Note the calibration×floor concentration (§6, one paragraph):** calibrated p̂≤0.684 so only ~7% of bets clear p≥0.60 and per-instrument floor-pass-rate is uneven (cl1s/gc1s/ng1s 100%, fesx1s 30%); strategy sizes off a thin high-confidence slice (concentration risk), and **ng1s clears the floor 100% despite `n_eff_gate=2`** (shared-base scope). Tie to S5.9/S6.15; cite `n_eff` |
| **S6.14** | ✅ | **[from LR-6 — top §6 reporting fix] Lead §6 with significance, not just deflation.** At T=128, raw ann Sharpe 1.36–1.55 → **t≈0.97–1.10, not significant at 5% before any deflation**. Report, in assumption-strength order: (a) Sharpe + t-stat; (b) **studentised stationary/circular block-bootstrap 95% CI** (`arch.bootstrap`, `method='studentized'`, block≈5–20) as *primary* inference — expected to contain 0, and that width is the finding; (c) Lo + Mertens/Opdyke analytic SE bands; (d) **PSR(0)≈0.83–0.86 + MinTRL≈286–371d vs 128 available**; (e) DSR/PBO as caveated corroboration (S6.8). Check first-lag autocorrelation (Ljung–Box) before √252 annualisation. Use LR-6's model reporting paragraph. Cite Lo 2002, Opdyke 2007, Ledoit–Wolf 2008, Bailey–LdP 2012/2014 (journal versions) |
| **S6.15** | ✅ | **[from LR-7 — optional sizing upgrade] Two better-justified sizing changes, both gated on CER.** (a) Replace the hard p≥0.55 floor with a **smooth taper** (the Kelly map already tapers continuously to 0); (b) make the shrinkage **per-instrument** κᵢ ≈ eᵢ²/(eᵢ²+σᵢ²) (Baker–McHale 2013; Chu–Wu–Swartz 2018) of the *post-calibration residual* variance — directly fixes the S6.13 uneven pass-rates without double-penalising the over-confidence calibration already removed. **Evaluate both against certainty-equivalent return** (`/empirical-finance certainty_equivalent`); **if OOS CER doesn't improve, keep the flat κ=0.25 heuristic** (defensible if documented). Also: Carver's 25% vol target *halves* for negative-skew systems — check the sleeve skew |

## Cross-cutting (every section)
| ID | Status | Item |
|----|--------|------|
| X.1 | ✅ | Seeds fixed everywhere |
| X.2 | ✅ | No leakage: scaler/estimator train-only; lags shifted; purged CV; truncation-invariance test; frozen parquet never consumed |
| X.3 | ✅ | Numeric invariants asserted |
| X.4 | ✅ | Features in labelled DataFrames |
| X.5 | ✅ | Reproducible end-to-end run producing the CSV |
| X.6 | ✅ | Unverified lit quotes carry `[NOTE FOR WRITEUP LEAD]` flags |
| **X.7** | ✅ | **Citation corrections (pass-3 applied; PM-8 adds one).** Done: Japan/UK correlations → Guidolin–Timmermann; Ang–Bekaert Wald p=0.156 caveat; Carver 2015 Ch.9 p.146; MZB "56%" deleted. **NEW (LR-8): delete "Kang & Kim (2025)"** — verified non-existent, a conflation of Fu/Kang/Hong/Kim (2024) (GA/triple-barrier pairs trading, *not* meta-labelling). It was listed as a *primary* cite in the PM-2 LR-1 integration → remove from the write-up and the LR-1 citation set (or replace with the correct 2024 four-author ref if a cite is needed there). Re-flip ✅ once both landed |
| **X.8** | ✅ | **Feature counts measured (pass-3):** Energy **140**, Equity **139**, Metals **140** (core 115 + macro 16 + regime 5 + inst-onehot 3–4). The "124" (and my PM-5 "108") were both wrong; measured truth written. Sweep any residual stale counts in prose |
| **X.9** | ✅ | **ALFRED vintage gate — pass-3 plan: attempt a real-time-vs-revised check, else document the limitation** (additional_data.xlsx carries observation dates, not vintages, so a true vintage check may be unreachable in-session). *Verify the write-up either reports the check or states the limitation honestly; then flip ✅.* |
| **X.10** | ✅ | **Write-up sentence (PM-7):** state explicitly that the metamodel **does not consume the shared `feature_matrix.parquet`** (which freezes TF features at a single global `fe_train_end=2021-07-01`) and instead **recomputes causal features per CV fold** — a deliberate leakage-avoidance choice over the shared base. Cheap, and it's a genuine rigor differentiator |
| **X.11** | ✅ | **Verify data-loading path (PM-7):** confirm the metamodel ingests OHLCV via the base's `stml.io.load_clean_data` policy — **keep the 765 zero-volume settle rows, drop only the 3 Sunday 2005-05-08 rows, never forward-fill structural NaNs**. If it loads the panel any other way (e.g. a naive zero-vol drop or a ffill), that's a silent inconsistency with the authoritative `missing-data-report.md` → fix |

## Experiment tracking (lightweight DataFrame→CSV)
| ID | Status | Item |
|----|--------|------|
| XT.1 | ✅ | Run results captured as structured tables (CPCV 15-path, per-instrument AUC, backtest) |
| XT.2 | ✅ | **`experiment_log.csv` carries Brier + precision (pass-3).** 3 shipped-selection rows now populated (Equity AUC 0.579/Brier 0.249/prec 0.601; Energy 0.525/0.263/0.621; Metals 0.530/0.344/0.584). *Optional: log the full 5×3 horse-race per-trial OOS series to make CSCV-PBO airtight (basis note in S6.8); not needed for the grade.* |
| XT.3 | ☐🔬 | Competition track: layer MLflow only if the all-11 × NN × CPCV sweep explodes run count |

## Build list (the 5 net-new pieces) — all ✅
| ID | Status | Item |
|----|--------|------|
| B.1 | ✅ | `triple_barrier.py` (vol-adaptive, symmetric, `t1`, uniqueness) |
| B.2 | ✅ | `cross_validation.py` (PurgedKFold+embargo, CPCV(15), nested) |
| B.3 | ✅ | `volatility.py` (Garman–Klass + Parkinson + Rogers–Satchell) |
| B.4 | ✅ | Cluster-SHAP module |
| B.5 | ✅ | `sizing.py` (fractional Kelly + vol-target) |

---

## Exploratory items (🔬 — investigate; not yet committed scope)
| ID | Item | Rationale / where |
|----|------|-------------------|
| **EX.1** | ✅ | **Done (pass-2).** All three primaries hit ~0.55–0.56, so Equity-works (XGB 0.572, 15/15)/Energy-fails (LGBM 0.493, 6/15)/Metals-marginal (0.524, 13/15) is a *discrimination* difference, not primary-quality — matches LR-1 framing |
| **EX.2** | ✅ | **Done (pass-2).** Built cluster-rep + PCA + autoencoder reducers; **cluster-rep promoted** (deterministic, interpretable, ties to §4); PCA/AE off-path. Feeds S3.7 |
| **EX.3** | ✅ | **Done (pass-2), diagnostic-only.** (k,T_max) surface for cl1s; firewall held (no feedback to locked `pt_sl`/`max_holding`). Note AUC is unstable across cells (0.46–0.59) — reinforces the thin-edge story |
| **EX.4** | ✅ | **Done (pass-2).** Raw ECE 0.14–0.18; **Platt beats isotonic in all three classes** (small-N). Directly motivates S3.9/S6.11 (calibrate before sizing) |
| **EX.5** | ✅ | **Done (pass-2).** Primary turnover/hit-rate/IR + hivol-lovol split; **ng1s IC=NaN**, ho1s/ng1s IR unstable → motivates S5.9. Hit-rate regime split is a usable feature |

---

## Literature-research scopes (for Claude `research` — detailed briefs)

**✅ ALL FIVE EXECUTED & EVALUATED (2026-05-30 PM-2) — outputs LR-1…LR-5 all INCLUDED.** Decisions + integration in the **Literature Integration** section below; the briefs are retained for provenance.

**Round 2 (2026-05-30 PM-6) — three *optional* scopes identified, delivered as separate briefs.** ✅ **ALL THREE EXECUTED & EVALUATED (PM-8) — LR-6, LR-7, LR-8 all INCLUDED.** Detailed scoping retained in `LR-6_scope.md`/`LR-7_scope.md`/`LR-8_scope.md`; integration table below.

| Output | Decision | Adopt into project | Do NOT carry / caveats |
|--------|----------|--------------------|------------------------|
| **LR-6** deflation/Sharpe inference at short T | ✅ **INCLUDE (most consequential)** | §6 reporting overhaul (→S6.14): lead with t-stat (≈1.0, n.s.) + **studentised block-bootstrap CI** (`arch`, in-house-endorsed) as primary; PSR(0)/MinTRL; DSR/PBO demoted to caveated corroboration (→S6.8). Arithmetic verified to the decimal. Use the model reporting paragraph verbatim | Mertens 2002 = unpublished note → cite **Opdyke 2007** + Lo 2002 + Bailey–LdP 2012/2014 *journal* versions. **No peer-reviewed DSR-at-T≈128 Monte-Carlo exists** — don't cite a blog as one. CSCV "12,780"→12,870 typo. Confirm IID before √252 (Ljung–Box) |
| **LR-7** calibration-aware sizing | ✅ **INCLUDE (optional, CER-gated)** | §6 sizing (→S6.15): smooth taper vs hard floor; per-instrument κᵢ=eᵢ²/(eᵢ²+σᵢ²) (Baker–McHale, verified monotone). The thin high-confidence slice is the *expected* signature of calibrated growth-optimal sizing — that framing alone strengthens §6 | "Always shrink" assumes lay-betting (strict for log-utility/Kelly, our case). Chopra–Ziemba is ~11:2:1–20:2:1, **not flat "20×"**. Harvey 2018 look-ahead critique (Liu–Tang–Zhou 2019) if claiming in-sample Sharpe. **Gate on OOS CER; if no gain, flat κ=0.25 stands** |
| **LR-8** meta-labelling on weak MR primary | ✅ **INCLUDE (framing + 2 verifications)** | §3/§5: null is *predicted*, anchored by Fundamental Law IC≈0⇒IR≈0 (`/aqms-python`); mechanism-not-timing via H–M/TM/PT; AUC≠P&L precedent (Cenesizoglu–Timmermann; Leitch–Tanner). Vol-targeting Sharpe-benefit concentrates in equity/credit (Harvey 2018) — explains sleeve heterogeneity (→S5.10, S6.12) | **Delete "Kang & Kim 2025"** (non-existent → X.7). **Audit H–M sign convention** before publishing (→S5.10). LdP precondition quote is secondary reproduction; "primary needs skill" is an inference not a verbatim theorem; MR-AUC ceiling is heuristic not proven. MQL5/thesis sources grey/leakage-contaminated — don't quote figures |

**Net:** no new modelling. One **important §6 reporting fix** (S6.14 — significance/bootstrap-CI lead), one **optional sizing upgrade** (S6.15, CER-gated), two **verifications** (S5.10 H–M sign audit; X.7 Kang & Kim deletion), and framing/citation hardening. **Nothing overturns the honest-negative — LR-6 makes it *more* correct** (insufficient-evidence, not failure).



Each is a self-contained research brief. Anchor every claim to the cited literature and **cross-check against `nlr-cw-v1.md`'s existing 60 refs to avoid duplication**; output should slot into the write-up's §-by-§ justification or Limitations.

### LR.1 — Meta-labelling efficacy & when it fails (priority: HIGH, supports EX.1, §3/§5)
**Question:** Under what conditions does meta-labelling add classification edge over a primary signal, and when does it predictably fail (mean OOS AUC ≈ 0.5)? **Scope:** (a) the López de Prado meta-labelling literature and subsequent empirical tests (Joubert; any replication/critique post-2018); (b) evidence on primary-signal quality as a precondition — does meta-labelling help more on noisier or on stronger primaries?; (c) the "AUC ≠ P&L" disconnect — literature on classification metrics vs economic value in trading (utility-based vs statistical evaluation). **Deliverable:** a paragraph for §3/§5 stating the expected-failure regime with citations, and whether our Equity-works/Energy-fails split matches published patterns.

### LR.2 — Backtest realism for triple-barrier strategies (priority: HIGH, supports S6.7)
**Question:** Best practice for backtesting a triple-barrier/meta-labelled strategy — barrier-exact exits, overlapping positions, and the metric set. **Scope:** (a) barrier-exact vs fixed-holding backtests and the bias from latest-signal-wins; (b) position netting/overlap when multiple labels are open; (c) the standard futures metric set and its pitfalls — Sortino, turnover, avg holding period, transaction-cost modelling (spread + Grinold-Kahn impact); (d) deflated Sharpe ratio and PBO for a single reported backtest. **Skills:** `/python-quant-finance` (vectorbt/quantstats/backtrader), `/aqms-python` (costs, turnover), `/empirical-finance` (Sortino, CER, bootstrap CIs). **Deliverable:** the metric definitions + a defensible backtest spec to replace the simplified holding model.

### LR.3 — Theory-of-storage & macro features for commodity/futures metamodels (priority: MEDIUM, supports S1.7)
**Question:** Which macro/term-structure features have published predictive content for commodity and index-futures returns, and how should they be point-in-time lagged? **Scope:** (a) theory of storage, convenience yield, basis/roll-yield (Gorton-Rouwenhorst, Hong-Yogo, Szymanowska); (b) inventory announcements (EIA) and energy returns; (c) gold↔real-rates/USD, copper↔China-activity linkages; (d) VIX term-structure/VRP and equity-index predictability; (e) PIT-lag conventions and release-calendar alignment to avoid look-ahead. **Skills:** `/empirical-finance` (VRP, carry), `/aqms-python` (carry, fixed income). **Deliverable:** a ranked, citation-backed feature list with PIT-lag rules feeding the S1.7 loader — explicitly noting which our front-month-only OHLCV can/can't support.

### LR.4 — Dimensionality reduction for feature-wise neural nets (VSN) on small samples (priority: MEDIUM, supports EX.2/S3.7)
**Question:** How to make a per-feature-GRN architecture (VSN) tractable and non-overfit on ~500 samples × 124 features? **Scope:** (a) feature selection vs extraction before deep nets in finance (Gu-Kelly-Xiu; autoencoder factor models); (b) cluster-representative selection as reduction (ties to §4 Mantegna clusters); (c) regularisation/early-stopping for small-N tabular deep learning (IKM small-data caution); (d) whether VSN's built-in selection obviates external reduction. **Skills:** `/bdfin-ml` (PCA, autoencoders, MLP), `/sts-ml` L6/L7 (VSN/TFT). **Deliverable:** a recommended reduction approach + expected tractability/over-fit trade-off for S3.7.

### LR.5 — Verify the flagged write-up citations (priority: LOW but pre-submission gate, supports X.7)
**Question:** Confirm the specific values/pages the lit review flagged `[NOTE FOR WRITEUP LEAD]`. **Scope:** Ang-Bekaert regime correlation values; Carver (2015) page numbers for the 25%-vol-target convention; any other flagged figure. **Deliverable:** verified citations or corrected values before academic submission.

---

## Literature Integration (LR-1…LR-5 — include/exclude decisions)

All five **INCLUDED**. Each row: verdict, what's adopted into the project, and what's explicitly *not* carried (with reason).

| Output | Decision | Adopt into project | Do NOT carry / caveats |
|--------|----------|--------------------|------------------------|
| **LR-1** meta-labelling efficacy | ✅ **INCLUDE** | §3/§5 framing paragraph (the AUC≈0.5 result is *expected*, not failure); pre-register Equity-works/Energy-fails; Henriksson–Merton + utility metrics (→S5.8); calibration-before-sizing 0.52–0.65 Kelly danger band (→EX.4). Cite JFDS trilogy (Meyer/Joubert/Thumm), Kang–Kim 2025, Leitch–Tanner, Cenesizoglu–Timmermann as **primary**. | Practitioner sources (QuantConnect/Hudson&Thames/MQL5) = **illustration only, never primary**. C9 (Equity/Energy split) = **medium, by synthesis** — present as such. C10: **no peer-reviewed AUC≈0.5 paper exists** — argue definitionally, don't claim a citation. Verify items 13/14/17/19 before quoting. |
| **LR-2** backtest realism | ✅ **INCLUDE (strongest; decision-grade)** | Replace latest-signal-wins with **barrier-exact exits + average-active-signals netting** (→S6.7); Sortino(full-N)/one-way-notional-turnover/holding-period defs; ½-spread + GK √-impact cost; **DSR + CSCV-PBO + MinBTL deployment gate** (→S6.8). Matches LdP primary + `/python-quant-finance` + `/empirical-finance` (Sortino denominator confirmed against skill code). | Stop-first intrabar tie-break = **practitioner consensus, not a theorem** (state it). GK literal page-equation not retrievable + Almgren 2005 rejected pure ½-power → **test the exponent**, treat impact as a slippage *lower bound*. Effective-N via clustering is **approximate** for a single backtest. |
| **LR-3** macro / term-structure | ✅ **INCLUDE** | The **PIT-lag release-calendar table** is the spec for S1.7; daily-T+1 features buildable now (VRP, VIX-slope, OAS, gold↔real-rates, DXY, MOVE), calendar-aware EIA-surprise + China-PMI; ALFRED-vintage validation gate. | **carry/basis (#1 predictor) NOT derivable** from front-month OHLCV — don't promise it; fall back to TS-momentum and lower expected commodity Sharpe. MOVE = **Low** (risk conditioner, not return predictor). Hedging-pressure contested once basis/inventory controlled. DXY regime-unstable post-2020. |
| **LR-4** VSN dimensionality reduction | ✅ **INCLUDE (mostly de-risks scope)** | If pursuing S3.7: **cluster-representative reduction reusing §4 Mantegna clusters** (free, interpretable) → ~20–40 inputs, then regularise hard; **benchmark against tuned XGBoost first**; XGB+VSN ensemble as the defensible deliverable. Reinforces keeping NN **low priority**. | Sample-size ratios are **heuristic** (no proven threshold). Asset-pricing "≈5 factors suffice" is a **different data shape** — motivation, not guarantee. **GKX 2021 magnitudes differ between SSRN and published** (Sharpe 0.92 vs 1.53) — verify published Tables 2–4. Mitra 2002 detail reconstructed → confirm before citing; de Amorim–Mirkin / Peng–Long–Ding are the safer anchors. |
| **LR-5** citation verification | ✅ **INCLUDE (pre-submission gate)** | Apply all three corrections to the write-up (→X.7): re-attribute Japan/UK correlations to Guidolin–Timmermann + add Ang–Bekaert Wald p=0.156 caveat; Carver 2015 Ch.9 p.146; keep MZB "56%" deleted. | Carver page from secondary chapter-summaries (Ch.9 certain, exact page may shift across editions). MZB in-paper table number medium-confidence; numeric values high. Scope was Items 1–3 only — MOP/GKX/HLZ citations **not** re-examined here. |

**One methodological note worth stating in the write-up:** the `empirical-finance` skill's prose says "downside std uses returns < 0," but its *code* divides squared shortfalls by the **full sample T** — which is LR-2's (correct) convention. Use the full-T denominator and cite Sortino–Price; note the common mis-implementation explicitly (it's an easy mark to defend).

---

## Open Questions (blocking)
- **Q2** — *Resolved (2026-05-30 PM-3):* 20 May constraints doc confirmed unavailable. Strategy track proceeds on the marked lit-review-default stub; write-up states it as an explicit assumption. Drop the real limits in (one config change) only if the doc later surfaces.
- **Q3** — *Resolved (pass-2):* backtest metrics + cost model pulled from `/python-quant-finance` + `/aqms-python` + `/empirical-finance`.
- **Q4** — *Resolved* (all 11; purged-kfold→CPCV→hold-out).
- **Q5** — *Resolved (PM-4):* build order confirmed.
- **Q6** — *Resolved (pass-2):* ran on the logged defaults (modules+tests in-repo; Energy first; average-active-signals netting).
- **Q7** — *Resolved (pass-3):* the critical path (S6.8 deflation, S6.10 reconcile, S3.9/S6.11 calibration) is done. §6 leads with the honest-negative.
- **Q8 (current)** — Build is methodologically complete; PM-7 + PM-8 added a well-defined queue (no new modelling). Recommended order: **(1) S6.14** — the §6 significance/bootstrap-CI reporting fix (LR-6; the headline currently over-relies on DSR when the Sharpes aren't even significant — highest-value single change); **(2) S5.10 + X.7** — the two verifications (H–M sign-convention audit; delete the non-existent Kang & Kim cite) — cheap and they protect correctness; **(3) S2.6** — per-instrument embargo (the one potential code-correctness item; can only tighten the result); **(4) X.11 + X.10** — `load_clean_data` verification + parquet-avoidance sentence; **(5) prose sweep** — S4.8/S5.8/S6.12/S6.13 + the LR-6/7/8 framing into `methodology.md`; **(6) optional** — S6.15 (CER-gated sizing upgrade), S1.8 (recommend document-as-limitation). Which next?
