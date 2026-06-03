# `metamodel-apb` README Documentation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three READMEs to the `metamodel-apb/` subproject — a hub `README.md`, a `src/alken_metamodel/README.md` module map, and an `experiments/README.md` probe index — making every external input and output explicit.

**Architecture:** Documentation only; no code changes. The hub is the front door (what / quickstart / **Inputs** / **Outputs** / I/O diagram / layout / links); the two subfolder READMEs go deep where the hub can't. All facts are verified against the code in this plan — tables below are the source of truth, copy them verbatim.

**Tech Stack:** Markdown (GFM), markdownlint (`metamodel-apb/.markdownlint.jsonc`, only `MD013` disabled). House style mirrors the repo-root `/Users/aiden.p.berendes/GitHub/stml/README.md` (numbered `##` sections, GFM tables, bash fences, `[text](path)` links). `sts-ml` framing: meta-labelling / act-skip / 11 instruments-3 asset-class metamodels / methodology-not-performance.

**Branch:** `model/alken-metamodel` (already a model branch — never `main`, per `metamodel-apb/CLAUDE.md`).

---

## Verified reference data (copy into the READMEs verbatim — no placeholders)

### Inputs — consumed from OUTSIDE `metamodel-apb/` (paths relative to repo root)

| Input | Path | Read via | Purpose |
| :---: | :---: | :---: | :---: |
| OHLCV (long) | `data/ohlcv_data.csv` | `stml.io.load_clean_data()` (`emit.py:187`) | OHLCV + open-interest for the 11 instruments |
| Primary signals (wide) | `data/primary_signals.csv` | `stml.io.load_clean_data()` (`emit.py:187`) | the {−1, 0, +1} primary signal being meta-labelled |
| Macro workbook | `data/additional_data.xlsx` | `macro.py` `load_macro_series()` (Sheet1, 22 series) | PIT-lagged macro block (shipped path: ON; `--no-macro` disables) |
| Per-instrument scope | `results/instrument_scope.json` | `pipeline.py` `load_embargo_days()` | per-instrument `embargo_p90` (+ `n_eff_gate`) for purge/embargo |
| `stml` package | editable path dep (`pyproject.toml`: `stml = {path = "..", editable = true}`) | `import stml.*` | causal feature **functions**, recomputed per fold |
| Course brief | `refs/project-instructions.md` | referenced in docs/comments only — not read by code | grading stance |
| Literature review | `reports/apb/nlr-cw-v1.md` | referenced in docs/comments only — not read by code | 8 commitments / 60 refs |

**stml functions imported** (sub-list under the table): `stml.io` (`load_clean_data`, `load_returns_panel`, `_find_repo_root`); `stml.metamodel.features` (`assemble_engineered`); `stml.metamodel.features_ext` (`assemble_engineered_ext`, `add_z_twins`); `stml.metamodel.drift_features` (`regime_alignment_score`, F16); `stml.metamodel.regime_features` (`fit_regime` / `transform_regime`, F3); `stml.metamodel.regime_features_hmm` (`fit_hmm` / `transform_hmm`, F17); `stml.metamodel.scope` (`ASSET_CLASS_MAP`); `stml.na_checks` (`native_returns`, `rolling_vol`).

**Anti-input call-out (leakage guard):** `results/feature_matrix.parquet` is **deliberately NOT consumed** — it freezes fitted stats at one global `fe_train_end`, which would leak into in-sample folds before that date. All stml feature functions are recomputed per fold instead; enforced by `tests/test_features.py::test_no_metamodel_module_reads_frozen_parquet`.

### Outputs — produced by `metamodel-apb/` (all in `outputs/`, gitignored)

| Output | Columns | Role |
| :---: | :---: | :---: |
| `outputs/metamodel_predictions.csv` | `date, instrument, prediction` | **deliverable** — calibrated P(act) ∈ [0, 1] |
| `outputs/metamodel_predictions_calibrated.csv` | `date, instrument, prediction` | byte-identical alias of the calibrated deliverable |
| `outputs/metamodel_predictions_raw.csv` | `date, instrument, prediction` | uncalibrated p̂ (pre-Platt; §3 Brier/ECE story) |
| `outputs/strategy_weights.csv` | `date, instrument, weight` | §6 bonus — fractional-Kelly × vol-target leverage |
| `outputs/experiment_log.csv` | `run_id, asset_class, roster, cv_scheme, reducer, use_macro, best_model, oos_auc, oos_brier, oos_precision, notes` | one row per asset class, deterministically rewritten each emit run (`emit.py:213` unlinks first) |
| `outputs/coverage_caveat.csv` | `instrument, n_oos_rows, ic, ic_undefined, thin` | thin-coverage flags (S5.9): `< 60` OOS rows or undefined IC |

Plus: `experiments/*.py` write diagnostics (markdown + `s6_net_returns.csv`) to the gitignored `experiments/results/`, regenerated per run — they feed nothing back into the locked deliverable. Determinism: rows sorted by `(date, instrument)`, columns pinned, `%.10f`, `\n` → byte-identical re-emit.

### Shipped-path defaults vs library defaults (state both — this is a real, easy-to-get-wrong nuance)

- **`emit` shipped deliverable path** (`emit.py:169-196`): `roster="default"` (5-estimator incl. torch NN), `cv_scheme="cpcv"`, **`use_macro=True`, `per_instrument_embargo=True`, `use_drift=True` (F16)** — all ON; disable with `--no-macro` / `--no-per-instrument-embargo` / `--no-drift`. Prediction window `--predict-start`/`--predict-end` default `2022-01-01`…`2022-06-30`, config-driven (grader swaps the hidden Jul–Dec 2022 half).
- **`PipelineConfig` library defaults** (`pipeline.py:80-99`, conservative): `roster="tree_linear"`, `cv_scheme="purged"`, `use_macro=False`, `per_instrument_embargo=False`, `use_drift=False`.

### Quickstart commands (from `CLAUDE.md` + `pyproject.toml`)

```bash
uv sync   --directory metamodel-apb                                  # install (Python 3.12 pinned)
uv run    --directory metamodel-apb pytest                           # full test suite
uv run    --directory metamodel-apb ruff check --no-fix src/ tests/  # lint (global fix=true caveat)
uv run    --directory metamodel-apb python -m alken_metamodel.emit   # emit deliverable CSVs -> outputs/
```

---

## Task 1: Hub README — `metamodel-apb/README.md`

**Files:**
- Create: `metamodel-apb/README.md`

- [ ] **Step 1: Write the hub README** with these numbered `##` sections, in order:
  1. **Title + what-this-is** (1 para): secondary act/skip meta-label classifier over a provided primary signal; 11 futures instruments / 3 asset-class metamodels (Equity/Energy/Metals); nested `uv` subproject importing `stml` as an editable path dep; **graded on methodology, not performance**. One honest results pointer: "no deployable edge is demonstrated — see [`docs/methodology.md`](docs/methodology.md) §5/§6"; do **not** quote Sharpes as claims.
  2. **Quickstart** — the four-command bash block above, then the `emit` CLI flags line (`--asset-classes`, `--predict-start`/`--predict-end`, `--outdir`, `--roster`, `--cv-scheme`, `--no-macro`, `--no-per-instrument-embargo`, `--no-drift`) and the shipped-path-vs-library-defaults note above.
  3. **Inputs (from outside the folder)** — the verified Inputs table verbatim, then the stml-functions sub-list, then the anti-input leakage call-out (one short paragraph).
  4. **Outputs** — the verified Outputs table verbatim, plus the `experiments/results/` one-liner and the determinism sentence.
  5. **I/O boundary diagram** — a fenced ```text block (simpler than `docs/methodology.md §0`), e.g.:

     ```text
     data/ohlcv_data.csv ─┐
     data/primary_signals.csv ─┤
     data/additional_data.xlsx ─┤  load_clean_data / macro / scope
     results/instrument_scope.json ─┤        │
     stml.* causal feature functions ─┘        ▼
                                    [ metamodel-apb pipeline: features → labels → CPCV → select → calibrate → size ]
                                               │
                                               ▼
                            outputs/{metamodel_predictions,…_raw,strategy_weights,experiment_log,coverage_caveat}.csv
     ✗ results/feature_matrix.parquet  — deliberately NOT read (leakage guard)
     ```
  6. **Repo layout** — one-line-per-entry table for `src/` (→ link `src/alken_metamodel/README.md`), `tests/` (mirrors src, RED-first; name guards: truncation-invariance, per-instrument embargo, no-frozen-parquet, emit-determinism, load-path/X.11), `docs/` (`methodology.md`, `plans/`), `experiments/` (→ link `experiments/README.md`), `reports/`, `_vendor/` (verbatim sts-ml; Stage-3 fixes logged in `_vendor/__init__.py`), `pyproject.toml`, `CLAUDE.md`, `.markdownlint.jsonc`.
  7. **Methodology, determinism & further reading** — links to [`docs/methodology.md`](docs/methodology.md), [`docs/plans/2026-05-30-metamodel-build.md`](docs/plans/2026-05-30-metamodel-build.md), [`CLAUDE.md`](CLAUDE.md), `reports/`; restate the four pillars in two sentences (causal-recompute-per-fold; purge+embargo on triple-barrier `t1`; config-driven window; byte-identical re-emit).

- [ ] **Step 2: Validate links resolve** — every relative `[text](path)` target exists:

```bash
cd /Users/aiden.p.berendes/GitHub/stml/metamodel-apb
grep -oE '\]\(([^)]+)\)' README.md | sed -E 's/\]\(([^)]+)\)/\1/' | grep -vE '^https?:' | while read -r p; do [ -e "$p" ] && echo "OK  $p" || echo "MISSING  $p"; done
```
Expected: every line `OK` (no `MISSING`). Fix any miss before continuing.

- [ ] **Step 3: Validate table/render sanity** — header/separator column counts match, no bare separators:

```bash
cd /Users/aiden.p.berendes/GitHub/stml/metamodel-apb
grep -nE '^\|-{2,}' README.md && echo "BARE SEPARATOR FOUND" || echo "no bare separators OK"
```
Expected: `no bare separators OK`.

## Task 2: Source module map — `metamodel-apb/src/alken_metamodel/README.md`

**Files:**
- Create: `metamodel-apb/src/alken_metamodel/README.md`

- [ ] **Step 1: Write the module map**, ordered by pipeline data-flow (not alphabetical), grouped with a one-line purpose per module (use each module's actual docstring first line — confirm by reading the file's top line):
  - **Data & features:** `features.py`, `regime.py`, `macro.py`, `volatility.py`, `triple_barrier.py`
  - **Modelling & validation:** `cross_validation.py`, `models.py`, `neural.py`, `dim_reduction.py`, `evaluation.py`, `calibration.py`, `cluster_importance.py`, `signal_analysis.py`
  - **Strategy / §6:** `sizing.py`, `cost_model.py`, `backtest.py`, `deflation.py`, `significance.py`
  - **Orchestration & infra:** `pipeline.py`, `emit.py`, `experiment_log.py`, `seeding.py`, `_env.py`, `__init__.py`
  - **`_vendor/`:** `vsn.py`, `cluster_feature_importance.py` (Stage-3 fixes #2 PurgedKFold + #4 Mantegna √ here), `trend_scanning.py`, `regression_metrics.py` — note fixes #1 `max_features='sqrt'` and #3 cluster SHAP live in `cluster_importance.py`, per `_vendor/__init__.py`.
  - Top note: entry point `python -m alken_metamodel.emit` → `emit.main()`; two-line fold-safety contract (causal functions recomputed per fold; never the frozen parquet). Cross-link to `../../README.md` and `../../docs/methodology.md`.

- [ ] **Step 2: Confirm every listed module exists** (catch typos/renames):

```bash
cd /Users/aiden.p.berendes/GitHub/stml/metamodel-apb/src/alken_metamodel
for m in features regime macro volatility triple_barrier cross_validation models neural dim_reduction evaluation calibration cluster_importance signal_analysis sizing cost_model backtest deflation significance pipeline emit experiment_log seeding _env __init__; do [ -f "$m.py" ] && echo "OK  $m.py" || echo "MISSING  $m.py"; done
ls _vendor/*.py
```
Expected: every `OK`; `_vendor/` lists `vsn.py cluster_feature_importance.py trend_scanning.py regression_metrics.py __init__.py`.

- [ ] **Step 3: Link/render sanity** (same two checks as Task 1 Steps 2–3, on this file).

## Task 3: Experiments index — `metamodel-apb/experiments/README.md`

**Files:**
- Create: `metamodel-apb/experiments/README.md`

- [ ] **Step 1: Write the probe index.** Lead paragraph: `experiments/` holds **diagnostic / reproducibility probes that are NOT part of the locked deliverable** — they build real pooled panels and write markdown + CSV to the gitignored `experiments/results/`, feeding no config back into `emit`. Table mapping script → methodology section → artifact (confirm each script's exact output filename by reading its write path while writing — do not guess filenames):

  | Script | Supports | Output (in `results/`) |
  | :---: | :---: | :---: |
  | `ex1_edge_decomposition.py` | EX.1 CPCV path robustness | markdown |
  | `ex3_barrier_surface.py` | barrier-hyperparameter surface | markdown |
  | `ex4_calibration.py` | EX.4 calibration diagnostics | markdown |
  | `ex5_signal_characterisation.py` | EX.5 primary-signal ceiling | markdown |
  | `s3_calibration_selected.py` | S3.9 calibration on the selected model | markdown |
  | `s4_cluster_importance.py` | §4 cluster importance | markdown |
  | `s6_barrier_backtest.py` | §6 barrier-exact backtest | markdown + `s6_net_returns.csv` |
  | `s6_deflation_gate.py` | S6.8 deflation gate | markdown |
  | `x8_feature_counts.py` | X.8 feature census | markdown |
  | `_common.py` | shared `modelling_panel` / `imputed_X` helpers | (no artifact) |

  Then: how to run one — `uv run --directory metamodel-apb python experiments/<name>.py` — and that `experiments/results/` is gitignored. Cross-link to `../README.md`.

- [ ] **Step 2: Confirm scripts + gitignore** :

```bash
cd /Users/aiden.p.berendes/GitHub/stml/metamodel-apb
ls experiments/*.py
grep -nE 'experiments/results' .gitignore && echo "results gitignored OK"
```
Expected: the listed scripts present; `results gitignored OK`.

- [ ] **Step 3: Link/render sanity** (same checks as Task 1 Steps 2–3, on this file).

## Task 4: Final verification, formatting pass & commit

- [ ] **Step 1: markdownlint clean** (MD013 off per config) — if a linter is available:

```bash
cd /Users/aiden.p.berendes/GitHub/stml/metamodel-apb
npx --yes markdownlint-cli2 README.md src/alken_metamodel/README.md experiments/README.md 2>/dev/null || echo "markdownlint-cli2 not installed — skip (rely on render-sanity checks above)"
```
Expected: no errors, or the skip message.

- [ ] **Step 2: Optional formatting pass** — run the `markdown-formatter` agent over the three new files for table/heading consistency with the repo (compact, centered tables to match `docs/methodology.md`). Preserve all content; cosmetic only.

- [ ] **Step 3: Accuracy self-review** — re-read each README against the verified reference data above; confirm no path/column/flag/import drifted, the macro/embargo/drift "shipped ON vs library OFF" nuance is stated, and `experiment_log.csv` is described as "rewritten per run" not "append-only".

- [ ] **Step 4: Commit** (on `model/alken-metamodel`; confirm with the user first per "commit only when asked"):

```bash
cd /Users/aiden.p.berendes/GitHub/stml
git add metamodel-apb/README.md metamodel-apb/src/alken_metamodel/README.md metamodel-apb/experiments/README.md
git commit -m "docs(metamodel-apb): add hub + src + experiments READMEs (external I/O boundary)"
```

---

## Self-review (spec coverage)

- Hub README covers what / quickstart / **Inputs** / **Outputs** / I/O diagram / layout / links → Task 1. ✓
- "All inputs from outside the folder" → Inputs table + stml-functions sub-list + anti-input → Task 1 Step 1.3. ✓
- "Outputs" → Outputs table + diagnostics + determinism → Task 1 Step 1.4. ✓
- Subfolder READMEs where they earn it (src, experiments) → Tasks 2, 3. ✓
- `tests/` & `docs/` covered by hub layout → Task 1 Step 1.6. ✓
- No code changes; data/refs/_vendor untouched → documentation-only throughout. ✓
- Placeholder scan: tables are verbatim verified data; the only "confirm while writing" notes (docstring first lines, exact `results/` artifact filenames) are explicit verification steps, not vague TODOs. ✓
