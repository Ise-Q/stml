# Alken Meta-Labelling Metamodel — Summary

---

## 1. What it is

`alken_metamodel` is a secondary **act/skip** meta-label classifier layered over a *provided* primary trading signal. The primary signal — a `{−1, 0, +1}` directional call supplied to the team — is treated as given; the metamodel's only job is to decide, on each day the primary is non-zero, whether to **act** on that call or **skip** it. It does not predict direction. This is meta-labelling in the sense of López de Prado (2018, Ch. 3): a binary filter trained on the primary's realised outcomes.

The universe is **11 futures instruments grouped into 3 asset-class metamodels** — one classifier per class, not per instrument (`ASSET_CLASS_MAP` in the parent package `src/stml/metamodel/scope.py`):

- **Equity (3):** `es1s`, `nq1s`, `fesx1s`
- **Energy (4):** `cl1s`, `ho1s`, `rb1s`, `ng1s`
- **Metals (4):** `gc1s`, `si1s`, `hg1s`, `pl1s`

It is a nested [`uv`](https://docs.astral.sh/uv/) subproject inside the `stml` repository: it imports `stml` as an editable path dependency and reads the released data **read-only**, writing only to its own gitignored `outputs/` folder (`methodology.md` §0).

## 2. Grading stance — the verdict, stated first

The course brief grades **methodology, not performance** (`refs/project-instructions.md`), and the metamodel is reported in that spirit. The honest finding is **insufficient evidence of a deployable edge** — *not* a proven failure, but not a skill claim either. The strategy's pooled out-of-sample Sharpe is **positive but statistically indistinguishable from zero**, and five independent diagnostics converge on the same null (§5, "What the evaluation found"). A reader should take from this document that the metamodel is **carefully engineered and leakage-disciplined**, and that this discipline is precisely what lets it report, credibly, that it has *not* beaten the blind-primary baseline. Where a positive number appears, its caveat appears in the same breath. This ordering — stance before any performance figure — is deliberate and is the document's central commitment (`methodology.md` Limitations, LR-6).

## 3. The pipeline (one-directional data flow)

The deliverable runs a single, one-directional pass per asset class (`methodology.md` §0; `README.md` §4):

> **load** (read-only OHLCV + signals via `stml.io.load_clean_data()`, PIT-lagged macro, per-instrument embargo scope) → **per-fold causal feature recompute** → **triple-barrier meta-labels on `signal≠0` days** (vol-adaptive `±k·σ̂ₜ` horizontal barriers + a vertical `T_max`, with a **per-class barrier geometry** — see §4, commitment 2 — recording each label's first-touch time `t1` and uniqueness weight) → **pooled purged Combinatorial CV horse-race** (CPCV, 15 purged combinatorial paths) → **select** by mean OOS AUC → **per-class Platt calibration** (fit train-only) → **size** (fractional Kelly × volatility target, signed by the primary side) → **emit** (deterministic, byte-identical CSVs).

The single most important design choice sits at the feature step. The metamodel **recomputes stml's causal feature *functions* inside each CV fold** on that fold's train slice; it **deliberately does not consume** the frozen `results/feature_matrix.parquet`. That parquet freezes fitted statistics at one global `fe_train_end = 2021-07-01`, which would leak future information into any in-sample fold dated before that point — the exact failure this build is constructed to avoid (`methodology.md` §0). The frozen matrix is an **anti-input**, and a guard test (`test_no_metamodel_module_reads_frozen_parquet`) asserts that no module reads it. Per-fold recompute is the project's signature differentiator from the shared base.

## 4. Module map and the eight commitments

The package is **24 primary modules** (plus four verbatim `_vendor/` scripts). The 13 that carry the pipeline:

| Module | Role |
| :--- | :--- |
| `triple_barrier.py` | Vol-adaptive `±k·σ̂ₜ` + vertical `T_max` meta-labels on `signal≠0` days; records `t1` (first-touch) and uniqueness weights |
| `pipeline.py` | Pools each class (+ instrument-id one-hot), keeps the event-date index for cross-instrument purge, orchestrates per-fold recompute, loads `embargo_p90` |
| `cross_validation.py` | Purged CV + embargo, CPCV (15 paths), and the nested-CPCV evaluator |
| `models.py` | Tree/linear horse-race: elastic-net logistic, XGBoost, LightGBM |
| `neural.py` | Byte-deterministic `torch`-MLP and `torch`-VSN variants (on a cluster-representative-reduced feature set) |
| `volatility.py` | Garman–Klass range volatility *feature* (with a Parkinson cross-check); the barrier width itself scales off a de-annualised realised vol (`f2_vol_20 / √252`), per-class, **not** Garman–Klass |
| `cluster_importance.py` | Mantegna-clustered MDI + MDA + SHAP importance (§4 diagnostics, off the critical emit path) |
| `sizing.py` | Fractional-Kelly × vol-target leverage, signed by the primary side |
| `backtest.py` | Barrier-exact §6 backtest of the sized weights |
| `regime.py` | Online EWMA 2-state HMM + stml static GMM / Markov-switching / 3-state HMM regime features |
| `deflation.py` | Deflated-Sharpe + MinBTL + CSCV-PBO deflation gate |
| `significance.py` | Studentised stationary block-bootstrap + t-statistic — the S6.14 significance-first inference |
| `emit.py` | Deterministic, byte-identical CSV emission of the calibrated deliverable |

Supporting modules include `calibration.py` (Platt), `macro.py` (the 22-series PIT-lagged macro block), `features.py`, `evaluation.py`, `experiment_log.py`, `seeding.py` / `_env.py` (determinism), `cost_model.py`, `dim_reduction.py`, and `signal_analysis.py`.

Each methodological choice is tied to a literature commitment and citation (`methodology.md` "Commitments → modules → citations", the project's Definition of Done):

| # | Commitment | Module(s) | Primary citation |
| :---: | :--- | :--- | :--- |
| 1 | Meta-labelling act/skip filter | `triple_barrier.py`, `pipeline.py` | López de Prado (2018, Ch. 3); Joubert (2022) |
| 2 | Per-class vol-adaptive `±k·σ̂ₜ` + vertical `T_max` (equity rolling-50 / (2,1) / h10; energy EWMA-20 / (0.5,0.25) / vol-scaled h(10,[2,40]); metals realised-vol-20 / (1,1) / h10), economic-Sharpe-selected on the modelling window | `triple_barrier.py` | López de Prado (2018, Ch. 3) |
| 3 | Purged CV + embargo + CPCV + nested | `cross_validation.py` | López de Prado (Ch. 7/12); Bailey & López de Prado (2014); Harvey, Liu & Zhu (2016) |
| 4 | Garman–Klass volatility (+ Parkinson check) | `volatility.py` | Garman & Klass (1980); Korkusuz (2023) |
| 5 | Multi-family model horse-race | `models.py`, `neural.py` | Gu, Kelly & Xiu (2020); Krauss (2017) |
| 6 | Cluster MDI + MDA + SHAP, Mantegna | `cluster_importance.py` | López de Prado (2020, Ch. 6); Lundberg et al. (2020); Mantegna (1999) |
| 7 | Fractional Kelly + vol-target | `sizing.py`, `backtest.py` | Kelly (1956); MacLean, Ziemba & Blazenko (1992); Carver (2015) |
| 8 | 2-state HMM, EWMA time-varying | `regime.py` | Hamilton (1989); Nystrup et al. (2017); Ang & Timmermann (2012) |

Cross-cutting commitments: uniqueness weights (López de Prado, Ch. 4); per-class **Platt calibration** of both the deliverable and the Kelly stake (Gramegna & Giudici, 2021); a single sample-weight channel for class imbalance and label uniqueness; and a **deflation gate** — Deflated Sharpe + MinBTL (Bailey & López de Prado, 2014) and CSCV-PBO (Bailey, Borwein, López de Prado & Zhu, 2017) — so §6 is reported deflated and never as "the strategy works."

## 5. What the evaluation found

**Model selection (the deployed path).** Selection is by mean OOS AUC over the 15 purged CPCV paths, across a five-estimator roster (elastic-net logistic, XGBoost, LightGBM, `torch`-MLP, `torch`-VSN, with the two neural variants on a cluster-representative-reduced feature set). On the **deployed selection path** the horse-race picks **XGBoost for Equity (mean AUC 0.5897), `torch`-MLP for Energy (0.5233 — a neural variant won its class) and elastic-net logistic for Metals (0.5324)** — none far above the 0.5 no-skill line (`methodology.md` §0, §3). These three are what `emit` refits and ships.

This must not be confused with the **EX.1 modelling-sample robustness probe** (`methodology.md` EX.1), which asks a *different* question — for each class, what fraction of the 15 combinatorial paths beat AUC 0.5 — and reports a *best model per class* on that sample: **Equity LightGBM 0.5923, 15/15 paths (edge robust); Metals elasticnet 0.5241, 13/15 (marginal-but-positive); Energy XGBoost 0.5266, 12/15**. EX.1's best-model column is a diagnostic of *where the edge is real on the modelling window*, not the deployed selection — note Energy's EX.1 winner (XGBoost) differs from its deployed model (`torch`-MLP). Read together: Equity's edge survives every combinatorial path; Energy and Metals clear 0.5 on a *majority* of paths within-sample too, but this is a within-sample count — Energy's per-instrument out-of-sample AUCs fall *below* the no-skill line, so the recombination count anticipates Equity's relative strength, not a transferable Energy edge.

**The §6 strategy verdict (the primary result).** Over the pooled out-of-sample period the net strategy posts an **annualised Sharpe of 0.48** — equivalently a **per-period Sharpe of ≈0.030 over n = 127 periods** — which gives **t = SR·√n = 0.34**, not significant at the 5% level; the primary inference, a **studentised stationary block-bootstrap 95% CI of [−0.09, +0.15] (per-period), contains zero** (`methodology.md` S6.14, the primary §6 result, LR-6). The honest reading is *insufficient evidence of a deployable edge*. The **deflation gate corroborates** this conclusion — it does not establish it: the pooled Deflated Sharpe ladder degrades from 0.26 down to 0.08 across the plausible trial count, PBO is 0.36, and MinBTL runs from ~1.7 up to ~3.1 years against only ~0.5 years of OOS data (`methodology.md` Limitations). What positive Sharpe there is comes from **diversification, volatility-targeting, and the barrier-exit asymmetry — not act/skip skill**. Per sleeve, Equity reads 0.51, Metals essentially zero, and **Energy exactly 0.0000 — it takes no out-of-sample positions at all**: its old shipped-barrier 1.86 Sharpe was never an edge (insignificant *t*, fails deflation, negative directional timing) but noise that printed positive over 127 days, and under the modelling-committed per-class barrier the same no-signal book emits a near-constant calibrated `p̂ ∈ [0.519, 0.522]`, all below the Kelly act-floor, so the metamodel **correctly abstains** rather than manufacturing a signal. The abstention is fragility consistent with no-edge, not an edge destroyed.

**Five-lens convergence (`methodology.md` §6.12).** The null is not one unlucky test but five independent lenses agreeing: (1) §3/§5 OOS AUC ≈ 0.50; (2) §4 cluster MDA near zero (|MDA| < 0.02 out-of-sample); (3) the §6.14 significance test (t = 0.34, CI contains 0); (4) the §6.8 deflation diagnostics (DSR, PBO); and (5) negative directional timing in §5 (Pesaran–Timmermann negative pooled, −1.80).

**The Treynor–Mazuy wrinkle, resolved.** The pooled Treynor–Mazuy convexity does not even print positive here: the pooled coefficient is **γ = −0.51 (t = −0.35)**, insignificant and of a piece with the negative directional verdict. That the same coefficient printed *positive and nominally significant* (**γ = +1.18, t = 2.55**) under the *earlier global barrier* — where it could have been mistaken for market-timing skill — is itself the cautionary point: the pooled coefficient is an unreliable scale aggregate **in either sign**. The Pesaran–Timmermann timing test — the scale-invariant directional test — is **negative (pooled −1.80, p ≈ 0.96)**, and the Sharpe is insignificant. The per-sleeve coefficients confirm the heterogeneity (Equity +0.31, Metals −2.05, both insignificant; Energy abstains under its barrier and contributes no acted trades), and their trade-count-weighted average is **−1.40, well away from the pooled −0.51 — the pooled magnitude is a scale aggregate of heterogeneous sleeves, not a shared coefficient**. Standardising each sleeve to a common scale and re-pooling collapses the pooled coefficient to **−0.0277 (t = −0.93, p = 0.354)**: whatever convexity the unstandardised regression reports — positive under the earlier barrier, negative here — dissolves into the same scale-driven near-zero. Convexity *without* directional accuracy is the signature of the **stop/barrier-exit asymmetry** (winners ride to the vertical barrier, losers are cut at the horizontal stop), **not timing**; the pooled γ is a scale-aggregation artefact (`methodology.md` S5.11/S5.12), not a tradeable signal.

## 6. Discipline and reproducibility

Four load-bearing disciplines make the negative result trustworthy (`methodology.md` §0, "Determinism & leakage discipline"; `README.md` §6):

1. **Causal recompute per fold.** stml feature *functions* are re-run on each fold's train slice; the frozen `feature_matrix.parquet` is never consumed. Proven by **right-edge truncation-invariance** property tests — a feature's value at `t` is identical whether computed on `data[:t+1]` or the full series — covering the whole stack including the F16 concept-drift family and the EWMA-HMM regime block.
2. **Purge + embargo on triple-barrier `t1`.** Overlapping meta-labels make standard k-fold invalid; labels are purged by first-touch time and a **per-instrument `embargo_p90`** is advanced on each instrument's own date axis (zero train/test `t1`-overlap after purge), including in the §4 cluster MDA.
3. **Config-driven prediction window.** The window is a `PipelineConfig` field, never hardcoded; the grader swaps in the **hidden Jul–Dec 2022 half by changing one value**. The visible feature set is locked before the final OOS window to avoid rehearsing the hidden half.
4. **Determinism.** Seeds are fixed across `random` / `numpy` / `torch` / `tensorflow` / `PYTHONHASHSEED`; native kernels run single-threaded; the CSV emitter sorts rows, pins columns, and fixes float format → **byte-identical re-emit**, verified on the pass-4 shipped path (two independent full runs produced byte-for-byte identical `metamodel_predictions.csv`, `…_raw.csv` and `strategy_weights.csv`, 1011 rows each). The non-reproducible Keras-VSN is deliberately kept *off* the selectable path so the hidden-half re-run cannot select a model it cannot reproduce.

Data loading conforms byte-for-byte to the shared base via `stml.io.load_clean_data()` (X.11): keep the 765 zero-volume settle rows, drop only the 3 Sunday 2005-05-08 rows, never forward-fill structural NaNs — ruling out a silent data-cleaning inconsistency with the rest of the cohort.

## 7. Shipped vs deferred, and honest limitations

**Shipped this run.** Per-class **Platt calibration** (S3.9) is now in the deliverable — train-only fit sharply cuts held-out ECE (Energy 0.105→0.048, Equity 0.062→0.017, Metals 0.210→0.001) with AUC unchanged, and the metamodel ships the calibrated `p̂`. The full methodology stack (five-estimator CPCV selection, EWMA-HMM + static regime blocks, PIT-lagged macro, per-instrument embargo, F16 drift) is on the default `emit` path.

**Deferred / stubbed (flag, not fix).**

- **Nested-CPCV real-data run.** The selection-bias-aware evaluator is implemented and unit-tested, but a full real-data run (≈ 15×5×5 fits per class) is **not executed here** — the standing selection-side gap.
- **§6 constraint set** remains a literature-default **stub** (the constraints doc is absent); only the holding model, cost model, and deflation gate were upgraded.

**Honest limitations.**

- **No deployable edge is demonstrated (the headline, LR-6)** — restated here so it cannot be missed: the §6 result is *insufficient evidence*, with the significance test (not the deflation gate) as the primary verdict.
- **Thin coverage.** `ho1s` has only **2 OOS rows** (information coefficient undefined — a Spearman IC on 2 points is meaningless), `gc1s` has 30, and `ng1s` has 56. All three are flagged (`< 60` rows or undefined IC) in `coverage_caveat.csv`; their per-instrument numbers are small-sample noise and should be read as such wherever they appear.
- **Macro vintages (X.9).** The macro series are publication-lag (PIT) aligned, so there is no *timing* look-ahead, but the workbook ships **revised/final values, not real-time vintages** — a *revision* look-ahead (e.g. a later-restated TIPS10Y/BE10Y print) is not excluded. A full ALFRED real-time-vintage reconciliation is the remaining macro gap; conservative publication lags and the macro block's near-zero §4 MDA bound its impact.
- **Late Equity history.** `es1s` (1997), `fesx1s` (1998) and `nq1s` (1999) start late, leaving thin pre-2020 history for fitted features.
- **Write-up citations (X.7), corrected.** The Japan/UK regime correlations are attributed to **Guidolin & Timmermann (2005)** (St. Louis Fed WP 2005-034), with the **Ang–Bekaert Wald-test caveat (p = 0.156)** noted — and the Ang–Bekaert Table-2 correlation *values* are deliberately not quoted (Ang–Bekaert is cited only for the Wald caveat). The Kelly/vol-target convention cites **Carver (2015, Ch. 9, p. 146)**, marked **pending print-edition confirmation** rather than asserted as settled. The MacLean–Ziemba–Blazenko fractional-Kelly figures keep **≈50% / ≈75%**; the unsupported "56%" was removed. A non-existent "Kang & Kim (2025)" anchor (a conflation of Fu, Kang, Hong & Kim 2024 — a GA/pairs-trading study, not meta-labelling) is not used and never appeared in the methodology.

---

### How to read this alongside the source

This summary compresses `methodology.md` §0 (architecture), §1 (features), §2 (labelling), §3 (models + S3.9 calibration), §4 (cluster importance), §5 (evaluation, incl. S5.11/S5.12 timing) and §6 (strategy, incl. S6.12 convergence and S6.14 significance), plus the Commitments table, the Determinism & leakage section, and the Limitations (X.7/X.9). For derivations, the deflation gate internals, and the per-instrument tables, see those sections directly. Diagnostics in `experiments/` (and their gitignored `experiments/results/` output) are reproducibility probes that feed **nothing** back into the locked deliverable.

### References (Harvard)

The authoritative, page-exact reference list (8 commitments, 60 references) is the literature review `reports/apb/nlr-cw-v1.md`; `methodology.md` carries the in-context attributions. The entries below give the Harvard author–year–title–venue forms for the works cited above; consult `nlr-cw-v1.md` for exact pagination.

Ang, A. & Bekaert, G. (2002) 'International asset allocation with regime shifts', *Review of Financial Studies* (cited here only for the Wald-test caveat, p = 0.156).
Ang, A. & Timmermann, A. (2012) 'Regime changes and financial markets', *Annual Review of Financial Economics*.
Bailey, D.H. & López de Prado, M. (2014) 'The Deflated Sharpe Ratio', *Journal of Portfolio Management*.
Bailey, D.H., Borwein, J.M., López de Prado, M. & Zhu, Q.J. (2017) 'The probability of backtest overfitting', *Journal of Computational Finance*.
Carver, R. (2015) *Systematic Trading*. Harriman House (Ch. 9, p. 146 — pending print-edition confirmation).
Garman, M.B. & Klass, M.J. (1980) 'On the estimation of security price volatilities from historical data', *Journal of Business*.
Gramegna, A. & Giudici, P. (2021) — calibration / explainability (see `nlr-cw-v1.md`).
Gu, S., Kelly, B. & Xiu, D. (2020) 'Empirical asset pricing via machine learning', *Review of Financial Studies*.
Guidolin, M. & Timmermann, A. (2005) *Economic implications of bull and bear regimes in UK stock and bond returns*. Federal Reserve Bank of St. Louis Working Paper 2005-034.
Hamilton, J.D. (1989) 'A new approach to the economic analysis of nonstationary time series and the business cycle', *Econometrica*.
Harvey, C.R., Liu, Y. & Zhu, H. (2016) '… and the cross-section of expected returns', *Review of Financial Studies*.
Kelly, J.L. (1956) 'A new interpretation of information rate', *Bell System Technical Journal*.
López de Prado, M. (2018) *Advances in Financial Machine Learning*. Hoboken: Wiley.
López de Prado, M. (2020) *Machine Learning for Asset Managers*. Cambridge: Cambridge University Press.
Lundberg, S.M. et al. (2020) 'From local explanations to global understanding with explainable AI for trees', *Nature Machine Intelligence*.
MacLean, L.C., Ziemba, W.T. & Blazenko, G. (1992) 'Growth versus security in dynamic investment analysis', *Management Science*.
Mantegna, R.N. (1999) 'Hierarchical structure in financial markets', *European Physical Journal B*.
Nystrup, P., Madsen, H. & Lindström, E. (2017) hidden Markov models with time-varying parameters (see `nlr-cw-v1.md`).
Pesaran, M.H. & Timmermann, A. (1992) 'A simple nonparametric test of predictive performance', *Journal of Business & Economic Statistics*.
Treynor, J.L. & Mazuy, K.K. (1966) 'Can mutual funds outguess the market?', *Harvard Business Review*.
