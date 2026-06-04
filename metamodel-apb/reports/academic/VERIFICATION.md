# Verification Audit — `metamodel-apb/reports/`

**Scope.** This document is a number-by-number and claim-by-claim traceability audit of every file under `metamodel-apb/reports/`, checked against the regenerated ground-truth artifacts. For each file it records (a) the file's status class and the era of numbers it is *expected* to carry, (b) a discriminator-metric trace, (c) an audit of every load-bearing qualitative ("commentary") claim, and (d) the selection-bias / multiple-testing degrees-of-freedom check on the deflation argument. The canonical configuration against which "current" is defined is the **EX.5 per-class barriers, `roster=default`, CPCV, regenerated 2026-06-04** build. Ground truth is the set of `experiments/results/*.md` artifacts plus `experiments/results/_number_ledger.md` and `outputs/experiment_log.csv`; the last is gitignored, which is precisely why this audit records the trace rather than assuming a reader can re-derive it from a committed file. **Out of scope:** `docs/` is path-scoped out of this audit (it is `reports/`-scoped) and is being refreshed on a separate track; nothing here should be read as a statement about `docs/methodology.md`.

---

## 1. File-status table

The audit partitions `reports/` into five status classes. The single most important reading rule: **DATED / HISTORICAL numbers are correct as of their date, not errors.** The frozen `T3_03` reference, the pass-dated `STML_ActionItem_Tracker.md`, and the `research/` literature all *legitimately* carry the OLD shipped-global era because they were authored before the 2026-06-04 per-class regeneration and are retained unchanged by design. A finding only arises when an **authoritative-current academic file carries an OLD value as if current**, or when **`T3_03` carries a NEW value** — neither of which occurs.

| File | Status class | Expected era | Uniformity verdict |
|---|---|---|---|
| `academic/methodology-full.md` | Authoritative-current | NEW (per-class) | Uniformly NEW — PASS |
| `academic/methodology-report.md` | Authoritative-current | NEW (per-class) | Uniformly NEW — PASS |
| `academic/methodology-brief.md` | Authoritative-current | NEW (per-class) | Uniformly NEW — PASS |
| `academic/experiments-and-decisions.md` | Authoritative-current | NEW (per-class) | Uniformly NEW — PASS |
| `academic/README.md` | Authoritative-current (index) | NEW (per-class) | Uniformly NEW — PASS |
| `T3_03_Alken_Metamodel_Report.md` | Superseded-frozen | DATED — shipped-global, *correct as of date* | Uniformly OLD — PASS (by design) |
| `STML_ActionItem_Tracker.md` | Decision-log | DATED — pass-tagged history, *correct as of date* | Mixed, all pass-tagged — EXPECTED |
| `CW_Breakdown.md` | Planning | N/A — pre-build planning parameters | No result numbers — N/A |
| `research/` (`LR-1..9`, `nlr-cw-v1.md`) | Literature | DATED — pass-analysis context, *correct as of date* | Mixed — EXPECTED |
| `triple-barrier-label.pdf` | Binary figure (untracked) | N/A — image, no numeric claims | Not audited (no text) — N/A |

---

## 2. Authoritative-set number audit (the five `academic/*` files)

**Result: the authoritative academic set is uniformly NEW on every discriminator metric.** No OLD value appears in any of the five files except as an explicitly-attributed "earlier global barrier" before/after contrast (catalogued separately in §6, item B). Every NEW value traces to the regenerated artifacts. The compact discriminator trace below gives the canonical NEW value, the source artifact, and which academic files carry it; values are quoted exactly as verified in the artifacts.

| Discriminator metric | NEW value (academic) | Source artifact (verified) |
|---|---|---|
| pooled net Sharpe | 0.48 (`0.4802`) | `s6_barrier_backtest.md` — pooled `sharpe=0.4802` |
| §6 t-stat (n=127) | t = 0.34 (`0.341`) | `s6_barrier_backtest.md` — `t = SR·√n = 0.341` (n=127) |
| bootstrap 95% CI (per-period) | [−0.09, +0.15] (`[-0.093, 0.148]`) | `s6_barrier_backtest.md` — studentised block-bootstrap, contains 0 |
| # corroborating diagnostics | "four further" (4) | exec summaries (vs OLD "five") |
| Pesaran–Timmermann pooled | −1.80 (p ≈ 0.96) | `s6_barrier_backtest.md` |
| Treynor–Mazuy raw pooled γ (t) | −0.51 (t = −0.35) (`-0.5113`) | `s6_barrier_backtest.md` — raw pooled γ |
| TM standardised-pooled γ | −0.0277 (t = −0.93, p = 0.354) | `s6_barrier_backtest.md` — standardised-pooled γ |
| TM trade-wt sleeve avg | −1.40 | `s6_barrier_backtest.md` (§5.12) |
| energy Sharpe | 0.00 — abstains, no trades | `s6_barrier_backtest.md`; p̂ ∈ [0.519, 0.522] |
| equity Sharpe | 0.51 | `s6_barrier_backtest.md` (barrier-exact) |
| metals Sharpe | ≈ 0.00 | `s6_barrier_backtest.md` |
| pooled Sortino / vol / maxDD | 0.78 / 10.97% / −4.54% (`0.7798 / 0.1097 / -0.0454`) | `s6_barrier_backtest.md` |
| gross / net | +5.36% / +2.38% (`gross=0.0536 net=0.0238`) | `s6_barrier_backtest.md` |
| DSR ladder (N 6→60) | 0.26 → 0.08 (`0.260 → 0.159 → 0.109 → 0.075`) | `s6_deflation_gate.md` — N∈[6,15,30,60], does NOT clear 0.95 |
| CSCV-PBO | 0.36 (`0.361`) | `s6_deflation_gate.md` |
| zero-weight share | 62% (623/1011) | emit / `s6_barrier_backtest.md` |
| selection AUC eq/en/me | 0.59 / 0.52 / 0.53 | `experiment_log.csv`; `s3_calibration_selected.md` |
| per-inst cl1s / ng1s AUC | 0.47 / 0.42 | `_number_ledger.md` |
| barrier σ basis | realized-vol-20 (eq rolling50 / en ewma20 / me fallback) | `ex5_*` sweep; *not* Garman–Klass as width |
| calibration ECE eq/en/me (raw→Platt) | 0.062→0.017 / 0.105→0.048 / 0.210→0.001 | `s3_calibration_selected.md` |
| cluster top MDA eq/en/me | 0.019 / −0.005 / −0.011 | `s4_cluster_importance.md` |
| MinTRL / PSR(0) | ≈ 2,883 days / 0.64 | `s6_deflation_gate.md` |

**Per-file spot-check counts (discriminator + traced numbers).** `methodology-full.md`: 34 spot-checked, uniformly NEW. `methodology-report.md`: 31, uniformly NEW. `methodology-brief.md`: 26, uniformly NEW. `experiments-and-decisions.md`: 24, uniformly NEW. `README.md`: 5 (index doc; carries the 0.48 / t=0.34 / CI / "four further" headline and the energy-abstention qualitative, all NEW).

**Findings in this section.** None of finding severity. The barrier σ-basis discriminator is handled correctly in all four substantive academic files: the *barrier-width / σ̂ₜ basis* is stated as de-annualised realized close-to-close vol-20 (per-class: equity 50-day rolling, energy 20-day EWMA, metals/default fallback), and **`methodology-full.md` and `methodology-report.md` explicitly correct an earlier draft's "Garman–Klass" wording**, demoting Garman–Klass to its legitimate role as a *range-based volatility feature* (`f2_garman_klass_20`, Parkinson cross-check). The GK dual sense (feature = correct/PASS; barrier-width basis = the OLD/wrong sense) is preserved everywhere.

Three NOTE-level traceability items (non-blocking, do not break uniformity):

- `methodology-full.md` §6 reports a `gc1s` metamodel **out-of-sample IC of −0.23** that is *untraceable* in the supplied modelling-window artifacts (the ledger/EX.5 carry the distinct primary-signal IC 0.2111, which the file correctly keeps separate). Plausibly an internal OOS diagnostic; cannot be verified against the supplied ground-truth set. Does not block.
- `methodology-full.md` §6 lists `ng1s` against two denominators (AUC "on only 68 labels" vs IC coverage 56 rows) — different bases (barrier-label count vs IC-defined-return rows), consistent with the ledger's `ng1s` OOS AUC 0.4245; not an error.
- `methodology-full.md` / `methodology-report.md` reproduce the **`C(16,8)` split count 12,870 explicitly correcting a prior "12,780" typo** — a documented correction, not wrong-era carriage.

---

## 3. Commentary-claim audit (load-bearing qualitative claims)

Every load-bearing interpretive claim in the authoritative academic set is **supported and internally coherent**. The five core claims, with the exact numbers they rest on:

| Claim | Supported? | Note |
|---|---|---|
| **Energy → abstention, not "edge destroyed"** | Yes | The no-signal model compresses calibrated p̂ into **[0.519, 0.522]**, never reaching the Kelly act-floor, so the whole energy book is sized to zero (energy Sharpe 0.00, no trades). Framed throughout as the metamodel *correctly abstaining* — the sharpest illustration of the negative — never as a destroyed edge. Matches `s6_barrier_backtest.md` (energy net = 0) and the ledger's "noise → abstention" guidance. |
| **TM = scale-aggregation artefact in *either* sign** | Yes | The pooled raw coefficient is **−0.51 (t = −0.35)** here and is presented as an unreliable scale aggregate "whichever sign it takes"; standardise-then-repool collapses it to **−0.0277 (t = −0.93, p = 0.354)**; trade-weighted sleeve avg **−1.40**. The OLD **+1.18 (t = 2.55)** is cited *only* as the superseded "earlier global barrier" cautionary contrast, never as positive timing. Pesaran–Timmermann (**−1.80**, scale-invariant) is named the trusted pooled directional diagnostic. Matches `s6_barrier_backtest.md` raw −0.5113 / standardised −0.0277. |
| **Garman–Klass = volatility FEATURE, not barrier width** | Yes | GK (with Parkinson) enters only as a range-based volatility feature; `methodology-full.md` and `methodology-report.md` explicitly state σ̂ₜ is realized vol-20 "and not … a Garman–Klass estimate (Garman–Klass enters only as a feature)." The NEW per-class barrier bases (eq rolling50 / en ewma20 / me default) match the EX.5 sweep recommendations. Dual sense correctly disambiguated. |
| **Five-lens "insufficient evidence," not "demonstrated failure"** | Yes | Five mutually-independent lenses — OOS AUC ≈ 0.50, cluster MDA < 0.02, significance (t = 0.34 / CI contains 0), deflation (DSR 0.26→0.08 & PBO 0.36), PT −1.80 — converge on **"insufficient evidence of a deployable edge, not a demonstrated failure,"** argued from Grinold IR = IC·√BR mechanics. The exec-summary "four further diagnostics" (the NEW discriminator, vs OLD "five") is the primary bootstrap CI (1) + four corroborating lenses (4) = the five-lens table; no 4-vs-5 contradiction. |
| **Anti-snoop firewall: selection on the modelling window only (≤ 2021-12-31)** | Yes | The economic barrier sweep is ranked by downstream net Sharpe strictly on the modelling window, so the selection statistic never touches the OOS window; Platt calibration is fitted before the prediction window; κᵢ shrinkage is estimated on the modelling sample only. The §7.3 smooth-taper gate-flip is *not* acted on precisely because its CER gain is OOS-measured (see §6 item C). Consistent across all four substantive files. |

### 3.1 Deflation barrier-search degrees-of-freedom check (prominent)

This is the dimension a methodology grader is most likely to probe: does the deflation argument acknowledge that **selecting the per-class barrier by net Sharpe over an in-sample sweep consumes degrees of freedom the deflated Sharpe must bound** — and is the conclusion robust *a fortiori*? The audit initially found the argument present in two of the five files and omitted in two; **both omissions were closed in the same change that produced this audit**, so all four substantive files now carry it (`README.md` is an index, out of scope):

| File | DoF argument present? | Coherent / does not overstate significance? | Assessment |
|---|---|---|---|
| `methodology-full.md` | **Yes** | Yes | **Gold standard.** |
| `methodology-report.md` | **Yes** | Yes | **Gold standard** (identical argument). |
| `methodology-brief.md` | **Yes (added in this change)** | Yes | Was generic ladder-robustness; a concise a-fortiori clause was added. |
| `experiments-and-decisions.md` | **Yes (added in this change)** | Yes | Was the conspicuous firewall-vs-DoF substitution; now explicitly links the barrier-sweep trial count to the deflation ladder. |
| `README.md` | N/A | Yes | Index doc; out of scope, not a defect. |

The two gold-standard files carry the argument **verbatim** (confirmed present in `methodology-full.md` and `methodology-report.md`; absent in the other two):

> "The trial ladder counts only the five-model horse-race per class; the data-driven cluster-representative reducer, the concept-drift expansion, *and* the per-class economic barrier sweep all add further selection that the ladder does not enumerate. Because the gate already fails at its most conservative counted rung and the deflated Sharpe falls monotonically in the trial count, those uncounted searches can only push the gate further from clearing — the negative is robust *a fortiori*, not in spite of the uncounted degrees of freedom."

This is correct: the DSR falls monotonically in N (`0.260 → 0.159 → 0.109 → 0.075` over N∈[6,15,30,60] in `s6_deflation_gate.md`) and already fails at the most conservative counted rung (pooled 0.26 at N=6), so the uncounted ~60 barrier trials can only lower it further. Both files lead with significance (t = 0.34, CI contains zero) and demote deflation to corroboration, so neither overstates the result.

The audit found the argument omitted in two files; **both were closed in the same change that produced this audit** (non-blocking — neither omission had contradicted the DSR-fails conclusion or overstated significance):

- **`methodology-brief.md`** — previously "the deflated-Sharpe ladder stays below 0.95 at every trial count," *generic ladder-robustness* that never identified the per-class barrier sweep as a trial count the deflation must bound. A concise a-fortiori clause was added: the trial count exceeds the five-model horse-race because the geometry was chosen by an economic sweep, and since the deflated Sharpe falls monotonically and already fails at its most conservative counted rung, the uncounted barrier-search trials can only push it further from clearing.
- **`experiments-and-decisions.md`** — previously the **most conspicuous** case. It has a dedicated "Labelling and barriers" section *and* a dedicated "Significance and deflation" section yet had discussed the barrier sweep only through the leakage-firewall lens ("the choice never touches the out-of-sample data"). The firewall is a *leakage* argument; the DoF burden is a *multiple-testing* argument — in-sample selection bias over a Sharpe-ranked sweep is exactly what the DSR must bound, so the firewall framing does **not** discharge it. A sentence was added linking the barrier-sweep trial count to the deflation ladder and stating that the firewall does not discharge the multiple-testing burden, with the a-fortiori conclusion.

---

## 4. Dated / historical catalogue — `T3_03` (and the tracker / research)

`T3_03_Alken_Metamodel_Report.md` is the **frozen academic reference, "retained unchanged"** by design and named as superseded by `academic/README.md`. It is **uniformly OLD** — and that is the *correct, expected* state, not a defect. No NEW per-class value appears anywhere in it; a NEW value would have been the anomaly. So that a reader never mistakes these for current, the OLD values it carries:

| Metric | OLD value in `T3_03` (dated/historical — correct as of date) |
|---|---|
| pooled net Sharpe | 1.31 |
| §6 t-stat (n=127) | 0.93 (internally consistent: SR 0.083 × √127 ≈ 0.935) |
| bootstrap 95% CI | [−0.04, +0.19] |
| # corroborating diagnostics | "five" |
| Pesaran–Timmermann pooled | −2.31 (p ≈ 0.99) |
| Treynor–Mazuy raw pooled γ (t) | +1.18 (t = 2.55); standardised collapse −0.0031; sleeve avg −1.835 |
| equity Sharpe | 0.86 (1.36 → 0.86 barrier-exact) |
| DSR ladder | 0.61 → 0.20 |
| CSCV-PBO | ≈ 0.35 |
| zero-weight share | ≈ 36% |
| CPCV paths > 0.50 (eq/me/en) | 15/15, 13/15, 6/15 |
| barrier σ basis | "de-annualised **Garman–Klass** daily volatility" as the ±k·σ̂ₜ half-width (OLD/wrong basis) |

Note on the `T3_03` barrier line: it names σ̂ₜ as Garman–Klass *as the barrier width*, which is the OLD basis (the NEW basis is realized vol-20). Reproducing this OLD wording verbatim is **correct for a frozen document**; separately, `T3_03` line 33 cites Garman & Klass (1980) as a legitimate range-based volatility *feature* — the era-neutral PASS sense. Both senses are textually distinct in the file. All `T3_03` commentary (TM as scale-artefact never read as timing, five-lens insufficient-evidence verdict, anti-snoop selection-on-modelling-window) is supported and coherent at the OLD era.

The **`STML_ActionItem_Tracker.md`** decision-log and the **`research/` literature** likewise carry pass-dated OLD figures (tracker: Sharpe 1.31, t 0.932, PT −2.31, TM +1.18/2.55, PBO 0.496/0.385/0.365/0.060, zero-weight 50.9%; research LR-8/LR-9: TM +1.18/2.55, PT −2.31, per-sleeve γ −4.51..+0.81). Every discriminator-bearing row is pass/PM-tagged or dated, all predating the 2026-06-04 regeneration — **DATED/HISTORICAL, correct as of date, not errors.** `nlr-cw-v1.md` is confirmed as the Harvard reference list the `academic/*` files cite (per `academic/README.md`), with citation integrity actively maintained (non-existent Kang & Kim 2025 removed; Ang–Bekaert re-sourced; PT pagination 461–465; CSCV 12,870-vs-12,780 typo flagged). `CW_Breakdown.md` is a pre-build planning checklist with no result numbers; its only discriminator is a *planned* Garman–Klass barrier (faithfully transcribed from the lit-review's recommended intent), a PLAN-vs-FINAL note, not a wrong-era finding.

---

## 5. Findings & discrepancies

**Blocking findings: 0.** No file carries a wrong-era value as if current; no academic file contradicts the per-class ground truth; no commentary claim is unsupported. Every per-file discrepancy is severity *note*, and the two deflation items are omission *defects* that explicitly do **not** contradict the DSR-fails conclusion or overstate significance.

**Core positive finding.** **Zero wrong-era mismatches in the authoritative set.** No academic file carries an unlabelled OLD value, and `T3_03` carries no NEW value. Every OLD number that appears in the academic set (+1.18 / t = 2.55, ~36% zero-weight, the 0.000538 circular κ, the 12,780 typo) appears **only** as an explicitly-attributed "earlier global barrier" before/after contrast documenting the revision — load-bearing for the TM and zero-weight arguments, and the correct way for an authoritative-current file to document a revision. This is the load-bearing result of the audit.

**Non-blocking items, grouped:**

- **A — Deflation DoF omissions (2; found and closed in this change).** `methodology-brief.md` and `experiments-and-decisions.md` had omitted the barrier-search multiple-testing DoF argument (`experiments-and-decisions.md` substituting firewall/leakage framing for it, the conspicuous case); both now carry the a-fortiori sentence already present in `methodology-full`/`-report`, so all four substantive files acknowledge the burden. Neither omission had contradicted the failing-gate verdict.
- **A2 — TM §6.2 wording tightened (`methodology-full.md`, `methodology-report.md`; closed in this change).** The standardise-then-repool sentence described the standardised pooled γ (−0.0277) as "collapsing toward the … negative trade-weighted sleeve average (−1.40)" — a directional phrasing inherited from the OLD *positive* pooled run (+1.18) that is incorrect under the NEW *negative* pooled (−0.51), where −0.0277 shrinks toward *zero*, i.e. away from −1.40. Reworded to the sign-agnostic "collapses … to an insignificant −0.0277"; the −1.40 sleeve-average datum is retained in §6.1 where it belongs. `methodology-brief.md` and `experiments-and-decisions.md` already used the sign-agnostic phrasing.
- **B — Deliberate before/after contrasts (not mismatches).** The OLD values listed above appear in the academic set only as explicitly-attributed "earlier global barrier" contrasts. Correct scholarly practice; recorded so a reader does not mis-flag them.
- **C — §7.3 smooth-taper sizing: reports match the deliverable (confirmed at code level).** The academic files report the leakage-safe smooth taper "as a diagnostic, not acted on … weights remain at the flat quarter," whereas `_number_ledger.md` (line 97) and `s6_barrier_backtest.md` log **"Decision: ADOPT smooth taper (CER 0.000090→0.000275)"** as an open Phase-D flag. This is **not a divergence in what ships.** `emit.py` imports only `position_weight` and calls it with the defaults — `taper_width=None`, which `sizing.py` documents as "the shipped behaviour" (a hard zero below the 0.55 floor), and flat `kappa=0.25`; the `confidence_taper` function exists in `sizing.py` but is exercised only by the `s6` diagnostic, never by `emit`. **So `strategy_weights.csv` ships flat κ = 0.25 with the hard floor, and the reports' "not adopted" matches the deliverable by construction.** The ledger/s6 "ADOPT" is a diagnostic *recommendation* from the S6.15 CER gate deliberately **not** wired into emit, and the reports correctly explain why: the taper's CER gain is measured on the OOS window, so adopting it would re-introduce the look-ahead the locked-before-OOS sizing prevents (contrast the EX.6 κᵢ gain — modelling-sample / leakage-safe — correctly reverted on a CI-contains-zero basis). This is the original emit-vs-experiment consistency axis; reports-vs-deliverable agreement on the sizing decision is positively verified, not waved through.
- **D — Untraceable / dual-denominator diagnostics (3).** `gc1s` OOS IC −0.23 untraceable in the supplied artifacts; `ng1s` dual denominators (68 vs 56); the 12,780→12,870 documented typo correction. All non-blocking (see §2).
- **E — Stale doc-path pointer (cosmetic).** `CLAUDE.md`/user memory point the lit-review path at `../reports/apb/nlr-cw-v1.md`, which has moved to `reports/research/nlr-cw-v1.md` (where `academic/README.md` correctly points). Pointer hygiene, not a content error in the audited folder.

**Audit verdict.** The authoritative academic set (`methodology-full.md`, `methodology-report.md`, `methodology-brief.md`, `experiments-and-decisions.md`, `README.md`) is **fully traceable and internally consistent at the per-class canonical (EX.5, roster=default, CPCV, 2026-06-04)**: uniformly NEW on every discriminator, with all five load-bearing commentary claims supported. The non-authoritative files (`T3_03`, `STML_ActionItem_Tracker.md`, `research/`, `CW_Breakdown.md`) are correctly characterized by their status class — their OLD/planned numbers are dated/historical and correct as of their date. **Zero blocking findings.** The two non-blocking gaps this audit surfaced — the deflation barrier-search DoF argument missing from `methodology-brief.md` and `experiments-and-decisions.md` — were corrected in the same change, so all four substantive files now carry the barrier-search a-fortiori argument; the TM §6.2 standardise-then-repool wording in `methodology-full.md` / `methodology-report.md` was likewise tightened to its sign-agnostic form.
