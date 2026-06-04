# Submission cleanup recommendation

**Status: recommendation only — nothing in this document has been moved or deleted.**
It refreshes the file-by-file audit in `src/stml/new_work/cleanup_audit.md` into an actionable,
tiered plan for trimming the repository before final submission. Execute it yourself after
review; verify every "duplicate" with `cmp`/`diff` before removing, since deletion is the one
irreversible step here.

The live pipeline is `src/stml/harry/` + `src/stml/new_work/` (with a thin set of utility modules
borrowed from `src/stml/experimental/`). The orphaned Sreeram pipeline, superseded outputs, and
exact duplicates are the bulk of what can go. The three new
`notebooks/submission/{equity,energy,metals}.ipynb` consolidate and supersede the six per-class
slice notebooks.

> Suggested mechanism: `git mv <path> archive/<path>` to preserve history for the ARCHIVE tier,
> `git rm` for the REMOVE tier. An `archive/` tree keeps everything recoverable without `git`
> archaeology while taking it out of the working surface a marker sees.

---

## KEEP — load-bearing for the submission (do not touch)

| Path | Why |
|---|---|
| `src/stml/{io,na_checks,__init__}.py` | Data spine; every notebook loads through it. |
| `src/stml/harry/**` | Live feature engineering, labelling, signal audit. |
| `src/stml/new_work/**` (live modules) | Strategy construction, HMM regimes, model comparison, importance. |
| `src/stml/experimental/{backtest,calibration,cost_model,significance,sizing,volatility,config,_env,__init__}.py` | The 9 utility modules **imported by** `new_work` (verified: zero other `experimental/*` modules are imported by live code). |
| `data/{ohlcv_data,primary_signals,alternate_data_cleaned}.csv`, `data/meta/{triple_barrier_labels,macro_features,labels_hash}.*` | Inputs and canonical labels. |
| `src/stml/new_work/outputs/importance/**` | Per-instrument importance figures/CSVs embedded by the submission notebooks. |
| `src/stml/new_work/outputs/model_comparison/{equity,energy,metals,_cache}/**` | Per-class CPCV/OOS results + event caches the notebooks load. |
| `src/stml/new_work/outputs/{metamodel_predictions.csv,finalisation/**}` | Locked OOS predictions and finalisation metadata. |
| `results/strategy_eval/**` | **Keep for submission** — the notebooks embed its figures and load `eval_summary.csv`. Regenerable, but it is the committed evidence of the evaluation. |
| `results/harry/**` | Signal-audit and event checkpoints. |
| `reports/**` | Methodology the notebooks cite (`00-context`…`03-6-macro-features`, `missing-data-report`). |
| `notebooks/submission/**` | The new deliverable (3 notebooks + `exports/*.{html,pdf}`). |

---

## ARCHIVE — superseded or orphaned (safe; verified zero live imports)

| Path | Superseded by / reason | Risk |
|---|---|---|
| `src/stml/experimental/*` **except the 9 utils above** (≈37 modules: `make_*.py`, `champion_pipeline`, `cv`, `data_loader`, `deflation`, `dim_reduction`, `emit`, `evaluation`, `importance`, `labels`, `models`, `multitask`, `nn_*`, `pipeline`, `signal_analysis`, `threshold`, `bloomberg_ingest`, `features/*`) | Sreeram's parallel pipeline; not reachable from any live entry point. | None — confirmed unimported. |
| `tests/experimental/**` (≈19 files) | Tests for the orphaned pipeline. | None. |
| `results/sreeram_experimental/**` | Outputs of the orphaned pipeline; no live consumer. | None. |
| `src/stml/new_work/outputs/{cl1s,es1s,fesx1s,gc1s,hg1s,ho1s,ng1s,nq1s,pl1s,rb1s,si1s}/` | Old per-instrument importance location; superseded by `outputs/importance/<inst>/`. | None — notebooks read `importance/`. |
| `src/stml/new_work/outputs/model_comparison/{cl1s,…,precious,energy_all,energy_cl_ho}/` (per-instrument/pool dirs) | Superseded by the per-class `{equity,energy,metals}/` rollups. | Low — only `reconciliation.py` (UNSURE) reads them. |
| `src/stml/new_work/_gen_*.py` (incl. the new `_gen_final_notebooks.py`) | Generators whose products are committed. | Low — keep if you want to regenerate notebooks; otherwise archive. |
| `DATA_SYS_PASTED.xlsx`, `cleaning_alternate_data.ipynb` | Cleaning provenance; product already in `data/`. | None. |
| `data/meta/triple_barrier_labels_{fixed,optimised}.csv`, `data/meta/*missing*`, `anomalous_rows.csv` | Label variants + data-quality diagnostics not read by the live pipeline. | None. |
| `notebooks/agent_eda.ipynb` | Orphan EDA, not part of this submission. | None. |
| **`src/stml/new_work/{equity,energy,metals}_{model_comparison,importance}.ipynb`** (6 slice notebooks) | **Consolidated into `notebooks/submission/*`.** Archive the `.ipynb`; **keep their underlying `outputs/`**, which the new notebooks still read. | None. |

---

## REMOVE — duplicates and junk (verify with `diff` first; then `git rm`)

| Path | Reason | Check before removing |
|---|---|---|
| `alternate_data_cleaned.csv` (repo root) | Reported byte-identical to `data/alternate_data_cleaned.csv`. | `cmp alternate_data_cleaned.csv data/alternate_data_cleaned.csv` |
| `data/triple_barrier_labels.csv` (root of `data/`) | Reported byte-identical to `data/meta/triple_barrier_labels.csv` (the canonical copy). | `cmp data/triple_barrier_labels.csv data/meta/triple_barrier_labels.csv` |
| `data/oof_meta_probabilities.csv.version1` | Old backup of `data/oof_meta_probabilities.csv` (floating-point-only delta). | `diff` the two. |
| `results/.DS_Store` | macOS metadata. | — |

---

## DECIDE — six items needing your call (from `cleanup_audit.md` §UNSURE)

| Item | Note | Recommended default |
|---|---|---|
| `src/stml/new_work/triple_barrier.py` | Supplies `_avg_uniqueness` to `cpcv_search.py` and `feature_importance.py`; the labelling functions are provenance. | **Keep** (it is live infrastructure). |
| `src/stml/new_work/reconciliation.py` + `outputs/model_comparison/selection_table_v2.csv` | Post-hoc analysis; live code reads `selection_table.csv`, not `_v2`. Not imported anywhere. | **Archive** both. |
| `src/stml/new_work/code-workspace.code-workspace` | Accidentally committed editor config. | `git rm --cached` and add to `.gitignore`. |
| `src/stml/experimental/seeding.py` | Used only by the orphaned `tests/experimental/`. | **Archive** with the experimental pipeline. |
| `notebooks/triple_barrier.ipynb` | Labelling provenance; the canonical labels are `data/meta/triple_barrier_labels.csv`, not this notebook's output. | **Archive** (keep as Step-2 evidence if preferred). |

---

## GAP — untracked neural-model weights

`outputs/{vsn_lstm,tft}_weights*.pt` are **untracked** and absent on disk. The Method C
(VSN-LSTM) and D (TFT) strategy results are already committed as
`results/strategy_eval/net_returns_{C_vsn_lstm,D_tft}.csv`, so the *evaluation* reproduces from
committed data, but the *weights* do not. Choose one and record it in `reports/harry/SETUP.md`:

1. Commit the `.pt` files via git-LFS (`.gitattributes`), or
2. Document that reproducing Methods C/D requires re-running `src/stml/new_work/train_cp5.py`
   after clone.

---

## A note on the labelling provenance (resolved)

Two triple-barrier label sets exist. `data/meta/triple_barrier_labels.csv` (the team labels)
uses `h = 1`, `pt = sl = 0.25`; `results/harry/events.csv` with `reports/harry/02-labels.md`
documents an earlier `h = 10` exploration. They carry the same number of events (one per signal
date) but different labels.

The submitted metamodel uses the **team `h = 1` labels**: every locked out-of-sample row in
`outputs/metamodel_predictions.csv` matches `data/meta/triple_barrier_labels.csv` exactly
(1373/1373 rows agree on the binary label after joining on instrument and date). The submission
notebooks now compute the Labelling section from that file and describe `events.csv` /
`02-labels.md` as the superseded exploration. No action is required beyond keeping
`reports/harry/02-labels.md` as a record; consider adding a one-line "superseded by the team
`h = 1` labels" header to it for clarity.
