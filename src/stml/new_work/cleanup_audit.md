# Cleanup Audit — Harry Branch
**Date:** 2026-06-04  
**Branch:** Harry (clean working tree confirmed; only untracked files)  
**Scope:** Every tracked file outside `.venv/`

---

## Method

1. `git ls-files` inventory — last-commit date and author via `git log --follow`.  
2. Python AST import graph — `grep`/`head` on all `.py` modules.  
3. Notebook code-cell extraction (JSON parse) for imports and path literals.  
4. Data reference scan — `grep` for `read_csv/read_parquet/to_csv/savefig/open/np.load/joblib.load` across all `.py` and `.ipynb` files.  
5. Entry points defined as: every `.ipynb`; every `.py` with `def main()` or `if __name__`; every `tests/` file; the MUST-KEEP set.  
6. Transitive reachability from entry points via both import and data-reference edges.

---

## MUST-KEEP Confirmation

| Item | Location | Status |
|------|----------|--------|
| 7 feature modules (concept_drift, conditional_risk, cross_asset, information_theoretic, macro_features, microstructure_fixed, signal_trajectory, wavelet) | `src/stml/harry/features/` | ✅ Present, all LIVE |
| HMM vol/turbulence module + notebook | `src/stml/new_work/hmm_vol.py`, `notebooks/harry/04-hmm-vol-turbulence.ipynb` | ✅ Present, LIVE |
| HMM macro risk-on/off module + notebook | `src/stml/new_work/hmm_macro.py`, `notebooks/harry/05-hmm-macro-riskoff.ipynb` | ✅ Present, LIVE |
| `triple_barrier_labels.csv` (canonical, teammate input) | `data/meta/triple_barrier_labels.csv` | ✅ Present; see §Duplicates |
| `events.csv` | `results/harry/events.csv` | ✅ Present; Harry-generated, read by `03-features-sanity.ipynb` |
| `refs/programming-session-sol/` | `refs/` | ✅ Present; classified REFERENCE |

> **Note on `.pt` weight files:** `outputs/tft_weights*.pt` and `outputs/vsn_lstm_weights*.pt` are **untracked** (show as `??` in `git status`). They exist on disk but are not in git. Methods C-vsn-lstm and D-tft in `evaluate.py` / `weights.py` require them. Pipeline **gap** — commit or document before submission.

---

## Classification Table

Classification key:
- **LIVE-CORE** — imported or data-loaded by a live entry point (transitively reachable).  
- **LIVE-ENTRY** — a notebook, standalone script, or test that is itself a deliverable.  
- **ORPHAN** — not reachable from any Harry entry point; no recognised purpose in Harry's pipeline.  
- **SUPERSEDED** — duplicate or older version of a live file.  
- **ARTIFACT** — generated output (regenerable from live code). Read-back artifacts noted.  
- **REFERENCE** — `refs/` solutions or generator scripts whose product is already committed.  
- **UNSURE** — needs your decision.

---

### A. Repository Config & Packaging

| Path | Type | Class | Imported-by / Notes | Last commit | Stage | Rec | Reason |
|------|------|-------|---------------------|-------------|-------|-----|--------|
| `.gitattributes` | config | LIVE | repo | 2026-05-20 Jay | — | keep | repo config |
| `.gitignore` | config | LIVE | repo | 2026-05-22 Ise-Q | — | keep | repo config |
| `pyproject.toml` | config | LIVE | package definition | varies | — | keep | package entry |
| `README.md` | doc | LIVE | — | 2026-05-22 Ise-Q | — | keep | project docs |

---

### B. Root-Level Data Files (potential duplicates)

| Path | Type | Class | Notes | Last commit | Rec | Reason |
|------|------|-------|-------|-------------|-----|--------|
| `alternate_data_cleaned.csv` | data | **SUPERSEDED** | Identical to `data/alternate_data_cleaned.csv` (diff = 0). Live code reads the `data/` version exclusively. | 2026-05-30 Harry | archive | exact duplicate of `data/alternate_data_cleaned.csv` |
| `DATA_SYS_PASTED.xlsx` | data | **ARTIFACT** | Excel source that was cleaned into `alternate_data_cleaned.csv`. Product already committed; file is large (2.3 MB). | 2026-05-30 Harry | archive | source already processed; large binary |
| `cleaning_alternate_data.ipynb` | notebook | **REFERENCE** | Cleaning script that produced `alternate_data_cleaned.csv`. The output CSV is the canonical artifact; notebook is provenance record only. | 2026-05-30 Harry | archive | product already committed; keep only for provenance |

---

### C. Data Directory

| Path | Type | Class | Loaded by | Last commit | Stage | Rec | Reason |
|------|------|-------|-----------|-------------|-------|-----|--------|
| `data/ohlcv_data.csv` | data | **LIVE-CORE** | `stml.io`; foundation | 2026-05-20 Jay | all | keep | primary price data |
| `data/primary_signals.csv` | data | **LIVE-CORE** | `stml.io`, hmm_vol.py, hmm_macro.py | 2026-05-20 Jay | 1 | keep | primary signals |
| `data/alternate_data_cleaned.csv` | data | **LIVE-CORE** | `feature_importance.py` (alt_macro path) | 2026-05-28 harrybrowne123 | 1 | keep | macro features source |
| `data/triple_barrier_labels.csv` | data | **SUPERSEDED** | Identical to `data/meta/triple_barrier_labels.csv` (diff = 0). `feature_importance.py` reads the `meta/` version. | 2026-06-04 Harry | 2 | archive | exact duplicate; `data/meta/` is canonical |
| `data/meta/triple_barrier_labels.csv` | data | **LIVE-CORE** | `feature_importance.py` line 98; external input from teammate. `labels_hash.json` points here. | 2026-06-04 Harry | 2 | keep | canonical team input — never archive |
| `data/meta/triple_barrier_labels_fixed.csv` | data | **ARTIFACT** | Generated by `notebooks/triple_barrier.ipynb` (fixed GARCH config). NOT loaded by live pipeline scripts. | 2026-05-30 Harry | 2 | archive | superseded by canonical labels; not read by pipeline |
| `data/meta/triple_barrier_labels_optimised.csv` | data | **ARTIFACT** | Generated by `notebooks/triple_barrier.ipynb` (CPCV-optimised config). NOT loaded by live pipeline scripts. | 2026-06-04 Harry | 2 | archive | superseded by canonical labels; not read by pipeline |
| `data/meta/labels_hash.json` | data | **LIVE** | Integrity guard for `triple_barrier_labels.csv` | 2026-06-03 Harry | 2 | keep | drift guard |
| `data/meta/macro_features.csv` | data | **LIVE-CORE** | `hmm_macro.py` (MACRO_PATH) | 2026-05-30 Harry | 1 | keep | macro features for HMM |
| `data/meta/anomalous_rows.csv` | data | **ARTIFACT** | Missing-data diagnostic output | 2026-05-22 Ise-Q | — | archive | diagnostic only |
| `data/meta/missing_dates_classified.csv` | data | **ARTIFACT** | Missing-data diagnostic | 2026-05-22 Ise-Q | — | archive | diagnostic only |
| `data/meta/missing_dates_per_instrument.csv` | data | **ARTIFACT** | Missing-data diagnostic | 2026-05-22 Ise-Q | — | archive | diagnostic only |
| `data/meta/missing_holidays_metadata.csv` | data | **ARTIFACT** | Holiday metadata | 2026-05-22 Ise-Q | — | keep | per memory note on holidays |
| `data/meta/other_missing_metadata.csv` | data | **ARTIFACT** | Missing-data diagnostic | 2026-05-22 Ise-Q | — | archive | diagnostic only |
| `data/meta/summary_per_instrument.csv` | data | **ARTIFACT** | Panel summary | 2026-05-22 Ise-Q | — | archive | diagnostic only |
| `data/meta/unexplained_missing.csv` | data | **ARTIFACT** | Unexplained gaps | 2026-05-22 Ise-Q | — | archive | diagnostic only |
| `data/oof_meta_probabilities.csv` | data | **LIVE-CORE** | `weights.py` (load_oof_probabilities), `strategy_construction.ipynb` | 2026-06-04 Harry | 3→6 | keep | OOF calibrated probabilities for SOPS/sizing |
| `data/oof_meta_probabilities.csv.version1` | data | **SUPERSEDED** | 5788-line diff vs `.csv`; older version of OOF probabilities; no code reads `.version1`. | 2026-06-04 Harry | — | archive | superseded by `.csv` |

---

### D. Notebooks — Harry's Pipeline

| Path | Type | Class | Imports / data refs | Last commit | Stage | Rec | Reason |
|------|------|-------|---------------------|-------------|-------|-----|--------|
| `notebooks/harry/03-features-sanity.ipynb` | notebook | **LIVE-ENTRY** | All 7 harry feature modules, `stml.io`, `results/harry/events.csv` | 2026-05-30 Harry | 3 | keep | Step 3 feature verification |
| `notebooks/harry/04-hmm-vol-turbulence.ipynb` | notebook | **LIVE-ENTRY** | `hmm_vol.py` logic inline (standalone demonstration), `data/ohlcv_data.csv` | 2026-05-30 Harry | 1 | keep | MUST-KEEP HMM vol notebook |
| `notebooks/harry/05-hmm-macro-riskoff.ipynb` | notebook | **LIVE-ENTRY** | `hmm_macro.py` logic inline, `data/meta/macro_features.csv` | 2026-05-30 Harry | 1 | keep | MUST-KEEP HMM macro notebook |
| `notebooks/harry/.gitkeep` | config | LIVE | placeholder | 2026-05-25 Harry | — | keep | directory marker |
| `src/stml/new_work/strategy_construction.ipynb` | notebook | **LIVE-ENTRY** | `stml.new_work.weights`, `targeting`, `evaluate`, `stml.io`, `outputs/model_comparison/{class}/cpcv_results.csv`, `metamodel_predictions.csv` | 2026-06-03 Harry | 7,8 | keep | primary deliverable |
| `src/stml/new_work/model_finalisation.ipynb` | notebook | **LIVE-ENTRY** | `outputs/finalisation/variant_results.json`, `outputs/importance/*/cluster_membership.csv` | 2026-06-03 Harry | 5 | keep | Step 5 dim-reduction / variant lock |
| `src/stml/new_work/feature_importance_analysis.ipynb` | notebook | **LIVE-ENTRY** | `outputs/importance/*/` CSVs and PNGs | 2026-06-03 Harry | 4 | keep | Step 4 importance visualisation |
| `src/stml/new_work/equity_model_comparison.ipynb` | notebook | **LIVE-ENTRY** | Reads `outputs/model_comparison/{equity,*}/` CSVs | 2026-06-03 Harry | 3 | keep | Model selection evidence |
| `src/stml/new_work/energy_model_comparison.ipynb` | notebook | **LIVE-ENTRY** | Reads `outputs/model_comparison/{energy,*}/` CSVs | 2026-06-03 Harry | 3 | keep | Model selection evidence |
| `src/stml/new_work/metals_model_comparison.ipynb` | notebook | **LIVE-ENTRY** | Reads `outputs/model_comparison/{metals,*}/` CSVs | 2026-06-03 Harry | 3 | keep | Model selection evidence |
| `src/stml/new_work/equity_importance.ipynb` | notebook | **LIVE-ENTRY** | Generated; reads `outputs/importance/{es1s,nq1s,fesx1s}/` | 2026-06-03 Harry | 4 | keep | Equity importance presentation |
| `src/stml/new_work/energy_importance.ipynb` | notebook | **LIVE-ENTRY** | Generated; reads `outputs/importance/{cl1s,ho1s,rb1s,ng1s}/` | 2026-06-03 Harry | 4 | keep | Energy importance presentation |
| `src/stml/new_work/metals_importance.ipynb` | notebook | **LIVE-ENTRY** | Generated; reads `outputs/importance/{gc1s,si1s,pl1s,hg1s}/` | 2026-06-03 Harry | 4 | keep | Metals importance presentation |

---

### E. Notebooks — Not Harry's Pipeline

| Path | Type | Class | Notes | Last commit | Rec | Reason |
|------|------|-------|-------|-------------|-----|--------|
| `notebooks/triple_barrier.ipynb` | notebook | **REFERENCE** | Generates `data/meta/triple_barrier_labels_{fixed,optimised}.csv`. The canonical external input (`data/meta/triple_barrier_labels.csv`) is NOT the output of this notebook — the live pipeline reads the team CSV, not the CPCV-optimised one. Notebook is a derivation record; its outputs are unused by pipeline. | 2026-06-03 Harry | archive | labeling provenance record; outputs not used |
| `notebooks/agent_eda.ipynb` | notebook | **ORPHAN** | Team EDA by Ise-Q; not imported or referenced by Harry's pipeline. | 2026-05-22 Ise-Q | archive | not Harry's; not reachable |
| `notebooks/jay/EDA.ipynb` | notebook | **REFERENCE** | Jay's EDA; 1480 bytes skeleton | 2026-05-20 Jay | keep | teammate artefact; keep as-is |
| `notebooks/jay/.gitkeep` | config | LIVE | placeholder | 2026-05-20 Jay | keep | directory marker |

---

### F. Python Modules — `src/stml/harry/`

| Path | Type | Class | Imported-by | Last commit | Stage | Rec | Reason |
|------|------|-------|-------------|-------------|-------|-----|--------|
| `src/stml/harry/__init__.py` | py | **LIVE** | package marker | 2026-05-30 Harry | — | keep | |
| `src/stml/harry/features/__init__.py` | py | **LIVE** | package marker | 2026-05-25 Harry | — | keep | |
| `src/stml/harry/features/concept_drift.py` | py | **LIVE-CORE** | `03-features-sanity.ipynb` | 2026-05-30 Harry | 1 | keep | regime-alignment feature |
| `src/stml/harry/features/conditional_risk.py` | py | **LIVE-CORE** | `feature_importance.py`, `03-features-sanity.ipynb` | 2026-05-30 Harry | 1 | keep | first-passage-time features |
| `src/stml/harry/features/cross_asset.py` | py | **LIVE-CORE** | `feature_importance.py`, `03-features-sanity.ipynb` | 2026-05-30 Harry | 1 | keep | cross-asset correlation features |
| `src/stml/harry/features/information_theoretic.py` | py | **LIVE-CORE** | `03-features-sanity.ipynb` | 2026-05-30 Harry | 1 | keep | MI / transfer-entropy features |
| `src/stml/harry/features/macro_features.py` | py | **LIVE-CORE** | `feature_importance.py`, `03-features-sanity.ipynb` | 2026-05-30 Harry | 1 | keep | M1–M6 macro groups |
| `src/stml/harry/features/microstructure_fixed.py` | py | **LIVE-CORE** | `03-features-sanity.ipynb` | 2026-05-30 Harry | 1 | keep | Amihud / Roll / Kyle microstructure |
| `src/stml/harry/features/signal_trajectory.py` | py | **LIVE-CORE** | `03-features-sanity.ipynb` | 2026-05-30 Harry | 1 | keep | signal run-length / entropy features |
| `src/stml/harry/features/wavelet.py` | py | **LIVE-CORE** | `feature_importance.py`, `03-features-sanity.ipynb` | 2026-05-30 Harry | 1 | keep | MRA energy bands |
| `src/stml/harry/labels.py` | py | **LIVE-CORE** | `persist_events.py`, `03-features-sanity.ipynb` | 2026-05-25 Harry | 2 | keep | Harry's labelling with next-day execution fix |
| `src/stml/harry/persist_events.py` | py | **LIVE-ENTRY** | Generates `results/harry/events.csv` + meta JSON | 2026-05-25 Harry | 2 | keep | events checkpoint script |
| `src/stml/harry/signal_audit.py` | py | **LIVE-ENTRY** | Generates `results/harry/signal_direction.csv` | 2026-05-25 Harry | 1 | keep | Step 1 signal audit |

---

### G. Python Modules — `src/stml/new_work/`

| Path | Type | Class | Imported-by / imports | Last commit | Stage | Rec | Reason |
|------|------|-------|----------------------|-------------|-------|-----|--------|
| `src/stml/new_work/__init__.py` | py | **LIVE** | imports hmm_vol, hmm_macro | 2026-05-30 harrybrowne123 | — | keep | |
| `src/stml/new_work/config.py` | py | **LIVE-CORE** | imports: split_config; imported by: data.py, weights.py, targeting.py, evaluate.py, run_locked_strategy.py, gen_oof_probas.py, train_cp5.py | 2026-06-03 Harry | all | keep | central path/parameter config |
| `src/stml/new_work/split_config.py` | py | **LIVE-CORE** | imported by config.py, model_comparison.py, equity/energy/metals_model_comparison.py, regenerate_predictions.py | 2026-06-03 Harry | 3 | keep | GLOBAL_CUT = 2021-10-06 |
| `src/stml/new_work/data.py` | py | **LIVE-CORE** | imports: stml.io, config; imported by: strategy_construction.ipynb, train_cp5.py | 2026-06-03 Harry | all | keep | data loaders |
| `src/stml/new_work/hmm_vol.py` | py | **LIVE-ENTRY** | generates `features_hmm_vol.csv`; imported by `__init__.py` | 2026-05-30 Harry | 1 | keep | MUST-KEEP HMM vol module |
| `src/stml/new_work/hmm_macro.py` | py | **LIVE-ENTRY** | generates `features_hmm_macro.csv`; imported by `__init__.py` | 2026-05-30 Harry | 1 | keep | MUST-KEEP HMM macro module |
| `src/stml/new_work/features_hmm_vol.csv` | data | **LIVE-CORE** | loaded by `feature_importance.py` (DATA_PATHS\["hmm_vol"\]) | 2026-05-30 Harry | 1 | keep | HMM vol regime features |
| `src/stml/new_work/features_hmm_macro.csv` | data | **LIVE-CORE** | loaded by `feature_importance.py` (DATA_PATHS\["hmm_macro"\]) | 2026-05-30 Harry | 1 | keep | HMM macro regime features |
| `src/stml/new_work/triple_barrier.py` | py | **UNSURE** | imported by `cpcv_search.py` (for `_avg_uniqueness`) and `notebooks/triple_barrier.ipynb`; but per your brief "treat TB code as NOT mine". It provides a utility function used by live pipeline. | 2026-06-03 Harry | 2 | **decide** | See §UNSURE |
| `src/stml/new_work/cpcv_search.py` | py | **LIVE-CORE** | imports: triple_barrier; imported by: model_comparison.py, equity/energy/metals_model_comparison.py, champion_importance.py, feature_importance.py, model_finalisation.py, run_locked_strategy.py, regenerate_predictions.py | 2026-06-03 Harry | 3,4,5 | keep | CPCV CV infrastructure |
| `src/stml/new_work/feature_importance.py` | py | **LIVE-CORE** | imports: harry features, cpcv_search; imported by: model_comparison.py, equity/energy/metals_model_comparison.py, champion_importance.py, run_locked_strategy.py, regenerate_predictions.py | 2026-06-03 Harry | 4 | keep | feature matrix builder + clustering |
| `src/stml/new_work/model_comparison.py` | py | **LIVE-CORE** | imports: feature_importance, cpcv_search; imported by: champion_importance.py, equity/energy/metals_model_comparison.py, model_finalisation.py, run_locked_strategy.py | 2026-06-03 Harry | 3 | keep | initial model zoo (generates selection_table.csv) |
| `src/stml/new_work/equity_model_comparison.py` | py | **LIVE-CORE** | imports: model_comparison, feature_importance, split_config; standalone `--main`; read by: regenerate_predictions.py | 2026-06-03 Harry | 3 | keep | Phase 2 equity comparison |
| `src/stml/new_work/energy_model_comparison.py` | py | **LIVE-CORE** | imports: model_comparison, feature_importance, split_config | 2026-06-03 Harry | 3 | keep | Phase 2 energy comparison |
| `src/stml/new_work/metals_model_comparison.py` | py | **LIVE-CORE** | imports: model_comparison, feature_importance, split_config | 2026-06-03 Harry | 3 | keep | Phase 2 metals comparison |
| `src/stml/new_work/champion_importance.py` | py | **LIVE-CORE** | imports: cpcv_search, feature_importance, model_comparison; standalone `--main` | 2026-06-03 Harry | 4 | keep | clustered MDA/MDI/SHAP pipeline |
| `src/stml/new_work/model_finalisation.py` | py | **LIVE-CORE** | imports: cpcv_search, feature_importance, model_comparison; used by model_finalisation.ipynb | 2026-06-03 Harry | 5 | keep | variant lock + dim-reduction |
| `src/stml/new_work/run_locked_strategy.py` | py | **LIVE-CORE** | imports: cpcv_search, feature_importance, model_comparison, energy_model_comparison, split_config, experimental.{calibration,sizing,volatility,backtest}; generates `metamodel_predictions.csv`, OOF probs | 2026-06-03 Harry | 3→6 | keep | locked champion re-fit + OOS prediction |
| `src/stml/new_work/regenerate_predictions.py` | py | **LIVE-ENTRY** | imports: equity/energy/metals_model_comparison; standalone script that regenerates OOF+OOS from locked champions | 2026-06-03 Harry | 3→6 | keep | used in cc29e29 to regenerate locked predictions |
| `src/stml/new_work/gen_oof_probas.py` | py | **LIVE-ENTRY** | imports: run_locked_strategy, experimental.calibration; generates `data/oof_meta_probabilities.csv` | 2026-06-03 Harry | 6 | keep | OOF calibration script |
| `src/stml/new_work/weights.py` | py | **LIVE-CORE** | imports: config, targeting, experimental.sizing; loads `metamodel_predictions.csv`, `oof_meta_probabilities.csv`, `.pt` weights; imported by: evaluate.py, strategy_construction.ipynb | 2026-06-03 Harry | 7 | keep | weight generation for all methods |
| `src/stml/new_work/targeting.py` | py | **LIVE-CORE** | imports: config, vol; imported by: weights.py, evaluate.py | 2026-06-03 Harry | 7 | keep | vol-targeted position sizing |
| `src/stml/new_work/vol.py` | py | **LIVE-CORE** | imported by: targeting.py | 2026-06-03 Harry | 7 | keep | vol estimators (ewma, yang-zhang, garch, gjr) |
| `src/stml/new_work/evaluate.py` | py | **LIVE-ENTRY** | imports: weights, targeting, experimental.{backtest,significance}; generates `results/strategy_eval/*` | 2026-06-03 Harry | 8 | keep | full eval pipeline |
| `src/stml/new_work/train_cp5.py` | py | **LIVE-ENTRY** | imports: config, split_config, data, models; trains VSN+LSTM + TFT; generates `.pt` weight files | 2026-06-03 Harry | — | keep | neural model training (weights currently untracked!) |
| `src/stml/new_work/models/__init__.py` | py | **LIVE-CORE** | imported by train_cp5.py | 2026-06-03 Harry | — | keep | |
| `src/stml/new_work/models/common.py` | py | **LIVE-CORE** | imported by models/tft.py, vsn_lstm.py; TrainConfig | 2026-06-03 Harry | — | keep | |
| `src/stml/new_work/models/tft.py` | py | **LIVE-CORE** | imported by train_cp5.py, weights.py | 2026-06-03 Harry | — | keep | TFT architecture |
| `src/stml/new_work/models/vsn_lstm.py` | py | **LIVE-CORE** | imported by train_cp5.py, weights.py | 2026-06-03 Harry | — | keep | VSN+LSTM architecture |
| `src/stml/new_work/reconciliation.py` | py | **UNSURE** | standalone script; reads `model_comparison/{group}/{model}/oos_predictions.csv`; writes `selection_table_v2.csv`, `reconciliation_report.md`; NOT imported by any other module | 2026-06-03 Harry | 3 | **decide** | See §UNSURE |
| `src/stml/new_work/_gen_energy_comparison_nb.py` | py | **REFERENCE** | Notebook generator that produced `energy_model_comparison.ipynb`; the notebook is already committed | 2026-06-03 Harry | — | archive | product committed; generator is provenance only |
| `src/stml/new_work/_gen_metals_comparison_nb.py` | py | **REFERENCE** | Notebook generator that produced `metals_model_comparison.ipynb` | 2026-06-03 Harry | — | archive | product committed; generator is provenance only |
| `src/stml/new_work/_gen_importance_notebooks.py` | py | **REFERENCE** | Notebook generator that produced equity/energy/metals_importance.ipynb | 2026-06-03 Harry | — | archive | product committed; generator is provenance only |
| `src/stml/new_work/code-workspace.code-workspace` | config | **UNSURE** | VS Code workspace file; no pipeline role | 2026-06-03 Harry | — | **decide** | See §UNSURE |

---

### H. Python Modules — `src/stml/experimental/` (Utility layer)

These modules are imported by Harry's new_work pipeline and are LIVE-CORE infrastructure.

| Path | Class | Imported-by | Stage | Rec |
|------|-------|-------------|-------|-----|
| `src/stml/experimental/__init__.py` | **LIVE-CORE** | side-effect (env setup) | all | keep |
| `src/stml/experimental/_env.py` | **LIVE-CORE** | `__init__.py` | all | keep |
| `src/stml/experimental/backtest.py` | **LIVE-CORE** | `evaluate.py`, `run_locked_strategy.py` | 8 | keep |
| `src/stml/experimental/calibration.py` | **LIVE-CORE** | `run_locked_strategy.py`, `gen_oof_probas.py` | 6 | keep |
| `src/stml/experimental/cost_model.py` | **LIVE-CORE** | `backtest.py` | 8 | keep |
| `src/stml/experimental/significance.py` | **LIVE-CORE** | `evaluate.py` | 8 | keep |
| `src/stml/experimental/sizing.py` | **LIVE-CORE** | `run_locked_strategy.py`, `weights.py` | 7 | keep |
| `src/stml/experimental/volatility.py` | **LIVE-CORE** | `run_locked_strategy.py` | 7 | keep |
| `src/stml/experimental/config.py` | **LIVE-CORE** | `data_loader.py`, config constants | — | keep |
| `src/stml/experimental/seeding.py` | **UNSURE** | not imported by Harry's pipeline; may be used by experimental tests | — | **decide** |

---

### I. Python Modules — `src/stml/experimental/` (Sreeram's Pipeline — ORPHAN from Harry's perspective)

These modules are only referenced by `make_*.py` scripts which belong to Sreeram's experimental pipeline. None are imported by Harry's `new_work/` code.

| Path | Class | Notes | Rec |
|------|-------|-------|-----|
| `src/stml/experimental/champion_pipeline.py` | **ORPHAN** | Sreeram's champion selection pipeline | archive |
| `src/stml/experimental/cv.py` | **ORPHAN** | Sreeram's CPCV implementation (different from new_work/cpcv_search.py) | archive |
| `src/stml/experimental/data_loader.py` | **ORPHAN** | Reads `sreeram_experimental_*.parquet`; only used by make_*.py | archive |
| `src/stml/experimental/deflation.py` | **ORPHAN** | DSR / backtest overfitting; used by make_significance.py | archive |
| `src/stml/experimental/dim_reduction.py` | **ORPHAN** | ClusterRepSelector; not imported by Harry's pipeline | archive |
| `src/stml/experimental/emit.py` | **ORPHAN** | Writes `outputs/strategy_weights.csv` etc; only used by make_deliverables.py | archive |
| `src/stml/experimental/evaluation.py` | **ORPHAN** | CVResult; used by make_*.py scripts | archive |
| `src/stml/experimental/importance.py` | **ORPHAN** | Sreeram's clustered importance; used by make_importance.py | archive |
| `src/stml/experimental/labels.py` | **ORPHAN** | Sreeram's triple-barrier labeller; used by make_labels.py | archive |
| `src/stml/experimental/make_baseline.py` | **ORPHAN** | Sreeram pipeline entry point | archive |
| `src/stml/experimental/make_champions.py` | **ORPHAN** | Sreeram pipeline entry point | archive |
| `src/stml/experimental/make_deliverables.py` | **ORPHAN** | Sreeram pipeline entry point | archive |
| `src/stml/experimental/make_features.py` | **ORPHAN** | Sreeram pipeline entry point | archive |
| `src/stml/experimental/make_final_comparison.py` | **ORPHAN** | Sreeram pipeline entry point | archive |
| `src/stml/experimental/make_importance_deep.py` | **ORPHAN** | Sreeram pipeline entry point | archive |
| `src/stml/experimental/make_importance.py` | **ORPHAN** | Sreeram pipeline entry point | archive |
| `src/stml/experimental/make_labels_harry_spec.py` | **ORPHAN** | Sreeram's harry-spec labeller variant | archive |
| `src/stml/experimental/make_labels.py` | **ORPHAN** | Sreeram's labelling entry point; reads Jay's `data/triple_barrier_labels.csv` | archive |
| `src/stml/experimental/make_nn_strategy.py` | **ORPHAN** | Sreeram's NN strategy entry point | archive |
| `src/stml/experimental/make_scope.py` | **ORPHAN** | Sreeram's scope/embargo computation | archive |
| `src/stml/experimental/make_significance.py` | **ORPHAN** | Sreeram's significance entry point; reads `results/sreeram_experimental/strategy_daily_net_returns.csv` written by run_locked_strategy.py | archive |
| `src/stml/experimental/models.py` | **ORPHAN** | Model roster; only used by Sreeram's pipeline | archive |
| `src/stml/experimental/multitask.py` | **ORPHAN** | Multi-task head; only used by Sreeram's pipeline | archive |
| `src/stml/experimental/nn_dataset.py` | **ORPHAN** | NN dataset builder; used by make_nn_strategy.py | archive |
| `src/stml/experimental/nn_portfolio.py` | **ORPHAN** | NN portfolio; used by make_nn_strategy.py | archive |
| `src/stml/experimental/pipeline.py` | **ORPHAN** | Sreeram's pipeline runner | archive |
| `src/stml/experimental/signal_analysis.py` | **ORPHAN** | Signal analysis utilities; not imported by Harry's pipeline | archive |
| `src/stml/experimental/threshold.py` | **ORPHAN** | Threshold utilities; not imported by Harry's pipeline | archive |
| `src/stml/experimental/bloomberg_ingest.py` | **ORPHAN** | Bloomberg data ingestion; reads raw CSVs from `data/` raw dir, writes parquet. Not used by Harry's pipeline. | archive |
| `src/stml/experimental/features/__init__.py` | **ORPHAN** | Sreeram's feature registry | archive |
| `src/stml/experimental/features/bloomberg.py` | **ORPHAN** | Sreeram's Bloomberg features | archive |
| `src/stml/experimental/features/catalog.py` | **ORPHAN** | Feature catalog | archive |
| `src/stml/experimental/features/closed_form.py` | **ORPHAN** | Price features | archive |
| `src/stml/experimental/features/cross_asset.py` | **ORPHAN** | Cross-asset features (distinct from `harry/features/cross_asset.py`) | archive |
| `src/stml/experimental/features/macro.py` | **ORPHAN** | Macro features (distinct from `harry/features/macro_features.py`) | archive |
| `src/stml/experimental/features/risk_drift_regime.py` | **ORPHAN** | Regime features | archive |

---

### J. Python Modules — `src/stml/` (Shared Foundation)

| Path | Class | Imported-by | Rec |
|------|-------|-------------|-----|
| `src/stml/__init__.py` | **LIVE-CORE** | package root | keep |
| `src/stml/io.py` | **LIVE-CORE** | `data.py`, `feature_importance.py`, `evaluate.py`, `harry/` modules | keep |
| `src/stml/na_checks.py` | **LIVE-CORE** | `feature_importance.py`, `equity/energy/metals_model_comparison.py` | keep |

---

### K. Tests

| Path | Class | Notes | Rec |
|------|-------|-------|-----|
| `tests/harry/__init__.py` | **LIVE** | | keep |
| `tests/harry/conftest.py` | **LIVE** | | keep |
| `tests/harry/test_causality.py` | **LIVE-ENTRY** | Universal causality harness for all 7 feature modules | keep |
| `tests/harry/test_concept_drift.py` | **LIVE-ENTRY** | | keep |
| `tests/harry/test_conditional_risk.py` | **LIVE-ENTRY** | | keep |
| `tests/harry/test_cross_asset.py` | **LIVE-ENTRY** | | keep |
| `tests/harry/test_events_consistency.py` | **LIVE-ENTRY** | drift-guard test; reads `results/harry/events.csv` | keep |
| `tests/harry/test_information_theoretic.py` | **LIVE-ENTRY** | | keep |
| `tests/harry/test_labels.py` | **LIVE-ENTRY** | | keep |
| `tests/harry/test_macro_features.py` | **LIVE-ENTRY** | | keep |
| `tests/harry/test_microstructure_fixed.py` | **LIVE-ENTRY** | | keep |
| `tests/harry/test_signal_audit.py` | **LIVE-ENTRY** | | keep |
| `tests/harry/test_signal_trajectory.py` | **LIVE-ENTRY** | | keep |
| `tests/harry/test_wavelet.py` | **LIVE-ENTRY** | | keep |
| `tests/experimental/__init__.py` | **ORPHAN** | Tests for Sreeram's pipeline | archive |
| `tests/experimental/conftest.py` | **ORPHAN** | | archive |
| `tests/experimental/test_bloomberg_ingest.py` | **ORPHAN** | | archive |
| `tests/experimental/test_cv.py` | **ORPHAN** | | archive |
| `tests/experimental/test_data_loader.py` | **ORPHAN** | | archive |
| `tests/experimental/test_evaluation.py` | **ORPHAN** | | archive |
| `tests/experimental/test_features.py` | **ORPHAN** | | archive |
| `tests/experimental/test_importance.py` | **ORPHAN** | | archive |
| `tests/experimental/test_labels.py` | **ORPHAN** | | archive |
| `tests/experimental/test_make_labels_jay.py` | **ORPHAN** | | archive |
| `tests/experimental/test_methodology_guards.py` | **ORPHAN** | | archive |
| `tests/experimental/test_models.py` | **ORPHAN** | | archive |
| `tests/experimental/test_multitask.py` | **ORPHAN** | | archive |
| `tests/experimental/test_nn_portfolio.py` | **ORPHAN** | | archive |
| `tests/experimental/test_pipeline.py` | **ORPHAN** | | archive |
| `tests/experimental/test_s6.py` | **ORPHAN** | | archive |
| `tests/experimental/test_s7.py` | **ORPHAN** | | archive |
| `tests/experimental/test_scaffold.py` | **ORPHAN** | | archive |
| `tests/experimental/test_volatility.py` | **ORPHAN** | | archive |

---

### L. Reports

| Path | Class | Notes | Rec |
|------|-------|-------|-----|
| `reports/harry/00-context.md` | **LIVE** | Harry's pipeline context document | keep |
| `reports/harry/01-signal-direction.md` | **LIVE** | Step 1 writeup | keep |
| `reports/harry/02-labels.md` | **LIVE** | Step 2 writeup | keep |
| `reports/harry/03-features.md` | **LIVE** | Step 3 writeup | keep |
| `reports/harry/03-6-macro-features.md` | **LIVE** | Macro features detail | keep |
| `reports/harry/SETUP.md` | **LIVE** | Environment setup guide | keep |
| `reports/missing-data-report.md` | **REFERENCE** | Team missing-data writeup | keep |
| `reports/.gitkeep` | LIVE | placeholder | keep |

---

### M. Refs

| Path | Class | Notes | Rec |
|------|-------|-------|-----|
| `refs/programming-session-sol/Solution_Programming_Session_*.ipynb` (×8) | **REFERENCE** | Course solutions; NOT submission material | keep (as-is; separate from submission) |
| `refs/missing-holidays.md` | **REFERENCE** | Holiday handling reference | keep |
| `refs/project-instructions.md` | **REFERENCE** | Project brief | keep |

---

### N. Results Directory

| Path | Class | Notes | Rec |
|------|-------|-------|-----|
| `results/harry/events.csv` | **LIVE-CORE** | Read by `03-features-sanity.ipynb` and `test_events_consistency.py` | keep |
| `results/harry/events.meta.json` | **LIVE-CORE** | Drift-guard JSON for events.csv | keep |
| `results/harry/signal_direction.csv` | **LIVE-CORE** | Generated by `signal_audit.py`; referenced by reports | keep |
| `results/harry/signal_direction_stability.csv` | **ARTIFACT** | Additional output of signal_audit.py; not read back by any pipeline | archive |
| `results/harry/.gitkeep` | LIVE | placeholder | keep |
| `results/jj/.gitkeep` | LIVE | teammate placeholder | keep |
| `results/sreeram_experimental/oos_events_with_predictions.csv` | **ARTIFACT** | Generated by `run_locked_strategy.py`; not read by Harry's live pipeline | archive |
| `results/sreeram_experimental/strategy_daily_net_returns.csv` | **ARTIFACT** | Generated by `run_locked_strategy.py`; read by `make_significance.py` (Sreeram's script) only | archive |
| `results/sreeram_experimental/deflation_ladder.csv` | **ARTIFACT** | Generated by `make_significance.py` (Sreeram's pipeline) | archive |
| `results/sreeram_experimental/significance_summary.csv` | **ARTIFACT** | Generated by `make_significance.py` | archive |
| `results/sreeram_experimental/significance_summary.md` | **ARTIFACT** | Generated by `make_significance.py` | archive |
| `results/strategy_eval/eval_summary.csv` | **ARTIFACT** | Generated by `evaluate.py`; regenerable | archive |
| `results/strategy_eval/eval_summary.md` | **ARTIFACT** | Generated by `evaluate.py` | archive |
| `results/strategy_eval/model_comparison_cpcv.png` | **ARTIFACT** | | archive |
| `results/strategy_eval/cumulative_returns.png` | **ARTIFACT** | | archive |
| `results/strategy_eval/net_returns_A.csv` | **ARTIFACT** | | archive |
| `results/strategy_eval/net_returns_B_aon.csv` | **ARTIFACT** | | archive |
| `results/strategy_eval/net_returns_B_mc.csv` | **ARTIFACT** | | archive |
| `results/strategy_eval/net_returns_B_ncdf.csv` | **ARTIFACT** | | archive |
| `results/strategy_eval/net_returns_B_sops.csv` | **ARTIFACT** | | archive |
| `results/strategy_eval/net_returns_C_vsn_lstm.csv` | **ARTIFACT** | | archive |
| `results/strategy_eval/net_returns_D_tft.csv` | **ARTIFACT** | | archive |
| `results/strategy_eval/per_asset_cumulative_returns.png` | **ARTIFACT** | | archive |
| `results/strategy_eval/per_instrument_contribution.png` | **ARTIFACT** | Modified in working tree | archive |
| `results/strategy_eval/vol_comparison.csv` | **ARTIFACT** | | archive |
| `results/.DS_Store` | junk | macOS metadata | delete-candidate |

---

### O. `new_work/outputs/` — Grouped Summary

The `outputs/` tree has **675 tracked files**. The majority are regenerable computation artifacts.

**LIVE-ARTIFACTS (read by live pipeline — do NOT archive):**

| Group | Path prefix | Read by | Notes |
|-------|------------|---------|-------|
| Event cache | `outputs/model_comparison/_cache/*.parquet` (12 files) | `equity/energy/metals_model_comparison.py`, `champion_importance.py`, `model_finalisation.py`, `run_locked_strategy.py`, `regenerate_predictions.py` | Critical — removing breaks re-runs |
| Class CPCV results | `outputs/model_comparison/{equity,energy,metals}/cpcv_results.csv` | `strategy_construction.ipynb` | Phase 1 results |
| Class OOS results | `outputs/model_comparison/{equity,energy,metals}/oos_results.csv` | `strategy_construction.ipynb` | Phase 2 results |
| Locked picks | `outputs/model_comparison/{equity,energy,metals}/locked_picks.csv` | `equity/energy/metals_model_comparison.py` | Variant lock decisions |
| Selection table | `outputs/model_comparison/selection_table.csv` | `champion_importance.py`, `equity/energy/metals_model_comparison.py` | Champion table |
| Importance / membership | `outputs/importance/*/cluster_membership.csv` (11 files) | `champion_importance.py`, `equity/energy/metals_model_comparison.py`, `model_finalisation.py` | Feature clusters |
| Full importance data | `outputs/importance/*/clustered_mda_full.csv` (11 files) | Importance notebooks | MDA results |
| Within-cluster detail | `outputs/importance/*/within_cluster_*.csv` (~80 files) | `model_finalisation.py` | Cluster breakdown |
| Finalisation | `outputs/finalisation/{final_models.csv,variant_results.json,variant_features.json}` | `model_finalisation.ipynb` | Locked variant metadata |
| OOF/OOS predictions | `outputs/metamodel_predictions.csv` | `weights.py`, `strategy_construction.ipynb` | Live OOS probabilities |

**SUPERSEDED (old per-instrument importance — top-level `outputs/[inst]/`):**

The directories `outputs/{cl1s,es1s,fesx1s,gc1s,hg1s,ho1s,ng1s,nq1s,pl1s,rb1s,si1s}/` (~77 files) were written by an earlier run of `feature_importance.py`'s `run_analysis()` function (which used `OUTPUTS = _HERE / "outputs"` without the `importance/` subdirectory). All live code now reads from `outputs/importance/[inst]/` (written by `champion_importance.py`). **No live file reads from the top-level `outputs/[inst]/` paths.** These are SUPERSEDED.

**OTHER ARTIFACTS (regenerable):**

| Path | Notes | Rec |
|------|-------|-----|
| `outputs/model_comparison/master_results.csv` | Summary from model_comparison.py; not read by live pipeline | archive |
| `outputs/model_comparison/selection_table_v2.csv` | From `reconciliation.py`; live pipeline reads `selection_table.csv` | UNSURE (see below) |
| `outputs/model_comparison/reconciliation_report.md` | From `reconciliation.py` | archive |
| `outputs/model_comparison/{inst}/{model}/*` (all per-instrument model subdirs; ~200 files) | From initial `model_comparison.py` run; read only by `reconciliation.py` | archive |
| `outputs/strategy_weights.csv` | Generated by `run_locked_strategy.py`; config defines a different path for the new strategy | archive |
| `outputs/net_returns_{tft,vsn_lstm}.csv` | Older evaluate.py run; superseded by `results/strategy_eval/` | archive |
| `outputs/warmup_diagnosis.md` | Diagnostic log | archive |
| `outputs/main_findings.csv` | From `feature_importance.py` aggregate run | archive |
| `outputs/main_findings_top5_clusters.png` | Aggregate importance chart | archive |
| `outputs/k_selection_metrics.png` | Cluster K selection plot | archive |
| `outputs/clustered_mda_equity.png` | Standalone chart | archive |
| All other `outputs/importance/*/` charts/logs | regenerable | archive |

---

## Pipeline Map

| Stage | Purpose | LIVE files | Gaps / Competing files |
|-------|---------|-----------|------------------------|
| 1. Features | Build F1–F15 + HMM regime features | `harry/features/` (7 modules), `hmm_vol.py`, `hmm_macro.py`, `features_hmm_{vol,macro}.csv`, `data/alternate_data_cleaned.csv`, `data/meta/macro_features.csv` | — |
| 2. Labels | External team input | `data/meta/triple_barrier_labels.csv` (canonical); `harry/labels.py` + `results/harry/events.csv` (Harry's next-day version for Step 2–3 sanity only) | `data/triple_barrier_labels.csv` (root duplicate); `data/meta/triple_barrier_labels_{fixed,optimised}.csv` (CPCV variants not used by live pipeline) |
| 3. Model zoo | CPCV model selection | `model_comparison.py`, `cpcv_search.py`, `feature_importance.py`, equity/energy/metals_model_comparison.py; `outputs/model_comparison/_cache/*.parquet`, `selection_table.csv`, `equity/cpcv_results.csv`, `locked_picks.csv`, `oos_results.csv` | Old per-inst subdirs in `outputs/model_comparison/[inst]/` (superseded by equity/energy/metals subdirs) |
| 4. Feature importance | MDI/MDA/SHAP per cluster | `champion_importance.py`, `feature_importance.py`; `outputs/importance/*/` | `outputs/[inst]/` (old location, superseded) |
| 5. Pruned/reduced models | Dim-reduction test | `model_finalisation.py`, `outputs/finalisation/` | — |
| 6. Calibration | Platt calibration | `experimental/calibration.py`, `gen_oof_probas.py`, `run_locked_strategy.py`; `data/oof_meta_probabilities.csv` | `data/oof_meta_probabilities.csv.version1` (old backup) |
| 7. Strategy weights | Vol-targeted weights for all methods | `weights.py`, `targeting.py`, `vol.py`, `config.py`, `outputs/metamodel_predictions.csv`; `.pt` weight files (untracked!) | — |
| 8. Evaluation / OOS | Performance metrics, drift | `evaluate.py`, `experimental/{backtest,significance,cost_model,sizing,volatility}.py`; `results/strategy_eval/` | `results/sreeram_experimental/` (Sreeram's parallel eval, written by `run_locked_strategy.py` to Sreeram's dir) |

---

## Summary Counts

| Classification | Count (approx) |
|----------------|---------------|
| LIVE-CORE | ~80 |
| LIVE-ENTRY | ~40 |
| ORPHAN | ~60 |
| SUPERSEDED | ~92 (77 old importance outputs + a handful of data duplicates) |
| ARTIFACT | ~480 (majority of outputs/, results/) |
| REFERENCE | ~15 |
| UNSURE | 6 |
| **Total tracked (excl. .venv)** | **~770** |

---

## UNSURE — Files Requiring Your Decision

| # | Path | Why unsure | Options |
|---|------|-----------|---------|
| 1 | `src/stml/new_work/triple_barrier.py` | Your brief says "treat TB code as NOT mine". But this file provides `_avg_uniqueness` (imported by `cpcv_search.py`) and `label_signals_fixed` (imported by `notebooks/triple_barrier.ipynb`). Removing it breaks the CPCV harness. | **A:** Keep as-is (it's infrastructure, even if not your original labeling work). **B:** Archive the notebook-facing labeling functions; keep only `_avg_uniqueness` in a utility module. |
| 2 | `src/stml/new_work/reconciliation.py` | Standalone script that re-analyses model_comparison outputs and writes `selection_table_v2.csv`. Not imported by any live pipeline module. Purpose: diagnostic reconciliation of per-instrument fold stats vs group-level selection. | **A:** Keep (analytical record). **B:** Archive (not needed for submission, selection_table.csv is canonical). |
| 3 | `src/stml/new_work/outputs/model_comparison/selection_table_v2.csv` | Produced by `reconciliation.py`. Live code reads `selection_table.csv`, not `_v2`. Was the v2 meant to replace v1? | **A:** Archive (v1 is the live canonical). **B:** Decide whether v2 should replace v1 first. |
| 4 | `src/stml/new_work/code-workspace.code-workspace` | VS Code workspace config in the source tree. Not a pipeline artifact. Commit was accidental (workspace files are usually gitignored). | **A:** Delete from tracking (`git rm --cached`). **B:** Keep (harmless). |
| 5 | `src/stml/experimental/seeding.py` | Not imported by Harry's pipeline. May be used by Sreeram's `tests/experimental/` tests. If experimental tests are archived, this can follow. | **A:** Archive (follows experimental tests). **B:** Keep with experimental infrastructure. |
| 6 | `notebooks/triple_barrier.ipynb` | Harry's CPCV labelling derivation notebook. The pipeline reads the team's `data/meta/triple_barrier_labels.csv`, not this notebook's outputs. It is useful provenance but not a submission deliverable. | **A:** Archive (outputs not used by live pipeline). **B:** Keep as evidence of Step 2 exploration. |

---

## Top Archive / Delete Candidates

### High-confidence archive (exact duplicates / clearly superseded)
1. `alternate_data_cleaned.csv` (root) — exact duplicate of `data/alternate_data_cleaned.csv`
2. `data/triple_barrier_labels.csv` (root of data/) — exact duplicate of `data/meta/triple_barrier_labels.csv`
3. `data/oof_meta_probabilities.csv.version1` — superseded OOF backup
4. All 77 files in `outputs/{cl1s,es1s,fesx1s,gc1s,hg1s,ho1s,ng1s,nq1s,pl1s,rb1s,si1s}/` (old importance location)

### High-confidence archive (Sreeram's pipeline — not Harry's submission)
5. All 37 files in `src/stml/experimental/` that are ORPHAN (Sreeram's `make_*.py`, feature catalog, etc.)
6. All 19 files in `tests/experimental/` (Sreeram's tests)

### Archive-candidate data diagnostics
7. `data/meta/{anomalous_rows,missing_dates_classified,missing_dates_per_instrument,other_missing_metadata,summary_per_instrument,unexplained_missing}.csv` — 6 diagnostic files

### Archive-candidate results
8. All `results/sreeram_experimental/` files (5 files)
9. All `results/strategy_eval/` files (11 files; regenerable by `evaluate.py`)

### Delete candidate (junk)
10. `results/.DS_Store`
11. `src/stml/new_work/code-workspace.code-workspace` (if you decide to remove it)

### Large binary to verify
12. `DATA_SYS_PASTED.xlsx` (2.3 MB) — source for alternate_data_cleaned.csv, already in git history. Recommend `git rm --cached` and add to `.gitignore` to avoid blob churn.

---

## Proposed `git mv` Plan

**Review this carefully before running.** Commands below relocate files to `archive/` subdirectories, preserving history. No `rm` commands are included.

```bash
# === EXACT DUPLICATES ===
git mv alternate_data_cleaned.csv                         archive/data_duplicates/alternate_data_cleaned.csv
git mv data/triple_barrier_labels.csv                     archive/data_duplicates/triple_barrier_labels.csv
git mv data/oof_meta_probabilities.csv.version1           archive/data_duplicates/oof_meta_probabilities.version1.csv

# === ROOT-LEVEL DATA ARTIFACTS ===
git mv DATA_SYS_PASTED.xlsx                               archive/raw_inputs/DATA_SYS_PASTED.xlsx
git mv cleaning_alternate_data.ipynb                      archive/raw_inputs/cleaning_alternate_data.ipynb

# === DATA META DIAGNOSTICS ===
git mv data/meta/anomalous_rows.csv                       archive/data_diagnostics/anomalous_rows.csv
git mv data/meta/missing_dates_classified.csv             archive/data_diagnostics/missing_dates_classified.csv
git mv data/meta/missing_dates_per_instrument.csv         archive/data_diagnostics/missing_dates_per_instrument.csv
git mv data/meta/other_missing_metadata.csv               archive/data_diagnostics/other_missing_metadata.csv
git mv data/meta/summary_per_instrument.csv               archive/data_diagnostics/summary_per_instrument.csv
git mv data/meta/unexplained_missing.csv                  archive/data_diagnostics/unexplained_missing.csv

# === TRIPLE BARRIER LABEL VARIANTS (not used by live pipeline) ===
git mv data/meta/triple_barrier_labels_fixed.csv          archive/label_variants/triple_barrier_labels_fixed.csv
git mv data/meta/triple_barrier_labels_optimised.csv      archive/label_variants/triple_barrier_labels_optimised.csv
git mv notebooks/triple_barrier.ipynb                     archive/label_variants/triple_barrier.ipynb

# === NOTEBOOK GENERATORS (products already committed) ===
git mv src/stml/new_work/_gen_energy_comparison_nb.py     archive/notebook_generators/_gen_energy_comparison_nb.py
git mv src/stml/new_work/_gen_metals_comparison_nb.py     archive/notebook_generators/_gen_metals_comparison_nb.py
git mv src/stml/new_work/_gen_importance_notebooks.py     archive/notebook_generators/_gen_importance_notebooks.py

# === SREERAM'S EXPERIMENTAL PIPELINE — Python modules ===
git mv src/stml/experimental/bloomberg_ingest.py          archive/experimental/bloomberg_ingest.py
git mv src/stml/experimental/champion_pipeline.py         archive/experimental/champion_pipeline.py
git mv src/stml/experimental/cv.py                        archive/experimental/cv.py
git mv src/stml/experimental/data_loader.py               archive/experimental/data_loader.py
git mv src/stml/experimental/deflation.py                 archive/experimental/deflation.py
git mv src/stml/experimental/dim_reduction.py             archive/experimental/dim_reduction.py
git mv src/stml/experimental/emit.py                      archive/experimental/emit.py
git mv src/stml/experimental/evaluation.py                archive/experimental/evaluation.py
git mv src/stml/experimental/importance.py                archive/experimental/importance.py
git mv src/stml/experimental/labels.py                    archive/experimental/labels.py
git mv src/stml/experimental/make_baseline.py             archive/experimental/make_baseline.py
git mv src/stml/experimental/make_champions.py            archive/experimental/make_champions.py
git mv src/stml/experimental/make_deliverables.py         archive/experimental/make_deliverables.py
git mv src/stml/experimental/make_features.py             archive/experimental/make_features.py
git mv src/stml/experimental/make_final_comparison.py     archive/experimental/make_final_comparison.py
git mv src/stml/experimental/make_importance_deep.py      archive/experimental/make_importance_deep.py
git mv src/stml/experimental/make_importance.py           archive/experimental/make_importance.py
git mv src/stml/experimental/make_labels_harry_spec.py    archive/experimental/make_labels_harry_spec.py
git mv src/stml/experimental/make_labels.py               archive/experimental/make_labels.py
git mv src/stml/experimental/make_nn_strategy.py          archive/experimental/make_nn_strategy.py
git mv src/stml/experimental/make_scope.py                archive/experimental/make_scope.py
git mv src/stml/experimental/make_significance.py         archive/experimental/make_significance.py
git mv src/stml/experimental/models.py                    archive/experimental/models.py
git mv src/stml/experimental/multitask.py                 archive/experimental/multitask.py
git mv src/stml/experimental/nn_dataset.py                archive/experimental/nn_dataset.py
git mv src/stml/experimental/nn_portfolio.py              archive/experimental/nn_portfolio.py
git mv src/stml/experimental/pipeline.py                  archive/experimental/pipeline.py
git mv src/stml/experimental/seeding.py                   archive/experimental/seeding.py
git mv src/stml/experimental/signal_analysis.py           archive/experimental/signal_analysis.py
git mv src/stml/experimental/threshold.py                 archive/experimental/threshold.py
git mv src/stml/experimental/features/__init__.py         archive/experimental/features/__init__.py
git mv src/stml/experimental/features/bloomberg.py        archive/experimental/features/bloomberg.py
git mv src/stml/experimental/features/catalog.py          archive/experimental/features/catalog.py
git mv src/stml/experimental/features/closed_form.py      archive/experimental/features/closed_form.py
git mv src/stml/experimental/features/cross_asset.py      archive/experimental/features/cross_asset.py
git mv src/stml/experimental/features/macro.py            archive/experimental/features/macro.py
git mv src/stml/experimental/features/risk_drift_regime.py archive/experimental/features/risk_drift_regime.py

# ⚠ WARNING: archiving these experimental modules will break the experimental/__init__.py
# import chain if anything still imports them. Verify no outstanding imports before running.

# === SREERAM'S EXPERIMENTAL TESTS ===
git mv tests/experimental/__init__.py                     archive/tests_experimental/__init__.py
git mv tests/experimental/conftest.py                     archive/tests_experimental/conftest.py
git mv tests/experimental/test_bloomberg_ingest.py        archive/tests_experimental/test_bloomberg_ingest.py
git mv tests/experimental/test_cv.py                      archive/tests_experimental/test_cv.py
git mv tests/experimental/test_data_loader.py             archive/tests_experimental/test_data_loader.py
git mv tests/experimental/test_evaluation.py              archive/tests_experimental/test_evaluation.py
git mv tests/experimental/test_features.py                archive/tests_experimental/test_features.py
git mv tests/experimental/test_importance.py              archive/tests_experimental/test_importance.py
git mv tests/experimental/test_labels.py                  archive/tests_experimental/test_labels.py
git mv tests/experimental/test_make_labels_jay.py         archive/tests_experimental/test_make_labels_jay.py
git mv tests/experimental/test_methodology_guards.py      archive/tests_experimental/test_methodology_guards.py
git mv tests/experimental/test_models.py                  archive/tests_experimental/test_models.py
git mv tests/experimental/test_multitask.py               archive/tests_experimental/test_multitask.py
git mv tests/experimental/test_nn_portfolio.py            archive/tests_experimental/test_nn_portfolio.py
git mv tests/experimental/test_pipeline.py                archive/tests_experimental/test_pipeline.py
git mv tests/experimental/test_s6.py                      archive/tests_experimental/test_s6.py
git mv tests/experimental/test_s7.py                      archive/tests_experimental/test_s7.py
git mv tests/experimental/test_scaffold.py                archive/tests_experimental/test_scaffold.py
git mv tests/experimental/test_volatility.py              archive/tests_experimental/test_volatility.py

# === SREERAM'S EXPERIMENTAL RESULTS ===
git mv results/sreeram_experimental/deflation_ladder.csv          archive/results_sreeram/deflation_ladder.csv
git mv results/sreeram_experimental/oos_events_with_predictions.csv archive/results_sreeram/oos_events_with_predictions.csv
git mv results/sreeram_experimental/significance_summary.csv      archive/results_sreeram/significance_summary.csv
git mv results/sreeram_experimental/significance_summary.md       archive/results_sreeram/significance_summary.md
git mv results/sreeram_experimental/strategy_daily_net_returns.csv archive/results_sreeram/strategy_daily_net_returns.csv

# === RESULTS/STRATEGY_EVAL (all regenerable from evaluate.py) ===
# Optional: keep if you want to snapshot the current eval; archive if you want clean slate.
git mv results/strategy_eval/eval_summary.csv             archive/strategy_eval/eval_summary.csv
git mv results/strategy_eval/eval_summary.md              archive/strategy_eval/eval_summary.md
git mv results/strategy_eval/cumulative_returns.png       archive/strategy_eval/cumulative_returns.png
git mv results/strategy_eval/model_comparison_cpcv.png    archive/strategy_eval/model_comparison_cpcv.png
git mv results/strategy_eval/net_returns_A.csv            archive/strategy_eval/net_returns_A.csv
git mv results/strategy_eval/net_returns_B_aon.csv        archive/strategy_eval/net_returns_B_aon.csv
git mv results/strategy_eval/net_returns_B_mc.csv         archive/strategy_eval/net_returns_B_mc.csv
git mv results/strategy_eval/net_returns_B_ncdf.csv       archive/strategy_eval/net_returns_B_ncdf.csv
git mv results/strategy_eval/net_returns_B_sops.csv       archive/strategy_eval/net_returns_B_sops.csv
git mv results/strategy_eval/net_returns_C_vsn_lstm.csv   archive/strategy_eval/net_returns_C_vsn_lstm.csv
git mv results/strategy_eval/net_returns_D_tft.csv        archive/strategy_eval/net_returns_D_tft.csv
git mv results/strategy_eval/per_asset_cumulative_returns.png archive/strategy_eval/per_asset_cumulative_returns.png
git mv results/strategy_eval/per_instrument_contribution.png  archive/strategy_eval/per_instrument_contribution.png
git mv results/strategy_eval/vol_comparison.csv           archive/strategy_eval/vol_comparison.csv
git mv results/harry/signal_direction_stability.csv       archive/results_harry/signal_direction_stability.csv

# === OLD IMPORTANCE OUTPUTS (superseded by outputs/importance/[inst]/) ===
# 77 files across 11 per-instrument top-level dirs.
# Run as a glob batch after verifying no live code reads them:
for inst in cl1s es1s fesx1s gc1s hg1s ho1s ng1s nq1s pl1s rb1s si1s; do
  git mv "src/stml/new_work/outputs/${inst}" "archive/old_importance_outputs/${inst}"
done

# === OLD MODEL_COMPARISON OUTPUTS (per-instrument/per-pool from model_comparison.py) ===
# ~200 files. Only reconciliation.py reads them; if reconciliation.py is archived too, safe to move.
# Pool/instrument subdirs: cl1s, energy_all, energy_cl_ho, es1s, fesx1s, gc1s, hg1s,
#                          nq1s, pl1s, precious, rb1s, si1s
for group in cl1s energy_all energy_cl_ho es1s fesx1s gc1s hg1s nq1s pl1s precious rb1s si1s; do
  git mv "src/stml/new_work/outputs/model_comparison/${group}" \
         "archive/old_model_comparison/${group}"
done
git mv src/stml/new_work/outputs/model_comparison/master_results.csv \
       archive/old_model_comparison/master_results.csv
git mv src/stml/new_work/outputs/model_comparison/reconciliation_report.md \
       archive/old_model_comparison/reconciliation_report.md

# === MISC OUTPUTS ARTIFACTS ===
git mv src/stml/new_work/outputs/strategy_weights.csv          archive/misc_outputs/strategy_weights.csv
git mv src/stml/new_work/outputs/net_returns_tft.csv           archive/misc_outputs/net_returns_tft.csv
git mv src/stml/new_work/outputs/net_returns_vsn_lstm.csv      archive/misc_outputs/net_returns_vsn_lstm.csv
git mv src/stml/new_work/outputs/warmup_diagnosis.md           archive/misc_outputs/warmup_diagnosis.md
git mv src/stml/new_work/outputs/main_findings.csv             archive/misc_outputs/main_findings.csv
git mv src/stml/new_work/outputs/main_findings_top5_clusters.png archive/misc_outputs/main_findings_top5_clusters.png
git mv src/stml/new_work/outputs/k_selection_metrics.png       archive/misc_outputs/k_selection_metrics.png
git mv src/stml/new_work/outputs/clustered_mda_equity.png      archive/misc_outputs/clustered_mda_equity.png

# === UNSURE — run only after deciding (see §UNSURE) ===
# git mv src/stml/new_work/reconciliation.py                   archive/unsure/reconciliation.py
# git mv src/stml/new_work/outputs/model_comparison/selection_table_v2.csv archive/unsure/selection_table_v2.csv
# git rm --cached src/stml/new_work/code-workspace.code-workspace
# echo "src/stml/new_work/*.code-workspace" >> .gitignore
```

> **Before running any block:** run `git status` and `git diff --stat HEAD` to confirm the working tree is clean. Then execute the moves in a single commit. Do not `rm` anything — the moves preserve history and are reversible.

---

## Outstanding Pipeline Gap

`outputs/tft_weights*.pt` and `outputs/vsn_lstm_weights*.pt` are **untracked** (not in git). `weights.py` needs them for Methods C-vsn-lstm and D-tft. Either:

- Add to `.gitattributes` as LFS and `git add` them, **or**
- Accept that Methods C/D require re-running `train_cp5.py` after clone.

Document whichever choice you make in `reports/harry/SETUP.md` before submission.
