# `metamodel-apb` — Alken Meta-Labelling Metamodel (T3.03)

A secondary **act/skip** meta-label classifier layered over a provided primary trading signal, for
**11 futures instruments across 3 asset-class metamodels** — Equity (`es1s`, `nq1s`, `fesx1s`),
Energy (`cl1s`, `ho1s`, `rb1s`, `ng1s`), and Metals (`gc1s`, `si1s`, `hg1s`, `pl1s`). It is a nested
[`uv`](https://docs.astral.sh/uv/) subproject inside the `stml` repo: it imports `stml` as an
editable path dependency and reads the released data **read-only**.

**The grade is methodology, not performance.** No deployable edge is demonstrated — the honest
finding is *insufficient evidence of edge*. See [`docs/methodology.md`](docs/methodology.md) §5–§6
for the full evaluation and why a positive backtest Sharpe is *not* a claim of skill here.

## 1. Quickstart

Run from the **repo root** with `--directory` (Python 3.12, pinned):

```bash
uv sync   --directory metamodel-apb                                  # install deps
uv run    --directory metamodel-apb pytest                           # full test suite
uv run    --directory metamodel-apb ruff check --no-fix src/ tests/  # lint (see caveat below)
uv run    --directory metamodel-apb python -m alken_metamodel.emit   # emit deliverable CSVs -> outputs/
```

`emit` CLI flags: `--asset-classes equity energy metals`, `--predict-start` / `--predict-end`
(the prediction window is **config-driven**, never hardcoded — the grader swaps in the hidden
Jul–Dec 2022 half by changing one value), `--outdir`, `--roster`, `--cv-scheme`, `--no-macro`,
`--no-per-instrument-embargo`, `--no-drift`.

**Shipped vs library defaults.** The `emit` deliverable path turns the full methodology on —
`roster="default"` (the five-estimator horse-race incl. the torch NN variants), `cv_scheme="cpcv"`,
and **macro, per-instrument embargo, and the F16 drift feature all ON** (disable each with its
`--no-*` flag). The bare `PipelineConfig` library defaults are deliberately conservative:
`roster="tree_linear"`, `cv_scheme="purged"`, and macro / embargo / drift all off.

> **Ruff caveat:** this machine's global `~/.config/ruff/ruff.toml` sets `fix = true`, so a bare
> `ruff check` rewrites files. Always verify with `ruff check --no-fix`.

## 2. Inputs — consumed from outside `metamodel-apb/`

All paths are relative to the **repo root** (`metamodel-apb/`'s parent). The subproject reads these
**read-only** and never writes outside its own folder.

| Input | Path | Read via | Purpose |
| :---: | :---: | :---: | :---: |
| OHLCV (long) | `data/ohlcv_data.csv` | `stml.io.load_clean_data()` (`emit.py`) | OHLCV + open-interest for the 11 instruments |
| Primary signals (wide) | `data/primary_signals.csv` | `stml.io.load_clean_data()` (`emit.py`) | the {−1, 0, +1} primary signal being meta-labelled |
| Macro workbook | `data/additional_data.xlsx` | `macro.py` → `load_macro_series()` (Sheet1, 22 series) | PIT-lagged macro block (shipped path ON; `--no-macro` disables) |
| Per-instrument scope | `results/instrument_scope.json` | `pipeline.py` → `load_embargo_days()` | per-instrument `embargo_p90` (+ `n_eff_gate`) for purge / embargo |
| `stml` package | editable path dep (`pyproject.toml`: `stml = { path = "..", editable = true }`) | `import stml.*` | causal feature **functions**, recomputed per fold (see below) |
| Course brief | `refs/project-instructions.md` | referenced in docs/comments only — *not* read by code | grading stance (methodology, not performance) |
| Literature review | `reports/apb/nlr-cw-v1.md` | referenced in docs/comments only — *not* read by code | 8 commitments / 60 references |

**`stml` functions imported** — the real cross-folder dependency is *functions*, not cached data:

- `stml.io` — `load_clean_data`, `load_returns_panel`, `_find_repo_root`
- `stml.metamodel.features` — `assemble_engineered` (F1 / F2 / F5 / F6 / F7 / F8 / F10)
- `stml.metamodel.features_ext` — `assemble_engineered_ext` (F2-RS / F12 / F13 / F15), `add_z_twins`
- `stml.metamodel.drift_features` — `regime_alignment_score` (F16 concept-drift)
- `stml.metamodel.regime_features` — `fit_regime` / `transform_regime` (F3 GMM + Markov-switching)
- `stml.metamodel.regime_features_hmm` — `fit_hmm` / `transform_hmm` (F17 3-state HMM)
- `stml.metamodel.scope` — `ASSET_CLASS_MAP` (the 11-instrument universe)
- `stml.na_checks` — `native_returns`, `rolling_vol`

> **Anti-input (leakage guard).** The frozen `results/feature_matrix.parquet` is **deliberately NOT
> consumed.** It freezes fitted statistics at a single global `fe_train_end`, which would leak future
> information into in-sample folds before that date. Instead every stml feature *function* is
> **recomputed inside each CV fold** on the fold-train slice. A guard test
> (`tests/test_features.py::test_no_metamodel_module_reads_frozen_parquet`) asserts that no module
> reads the parquet. This per-fold recompute is the project's signature design choice.

## 3. Outputs

Everything is written to `outputs/` (gitignored) by `alken_metamodel.emit`. Emission is
**deterministic** — rows sorted by `(date, instrument)`, columns pinned, floats formatted `%.10f`,
`\n` line endings → byte-identical re-emit.

| Output | Columns | Role |
| :---: | :---: | :---: |
| `outputs/metamodel_predictions.csv` | `date, instrument, prediction` | **deliverable** — calibrated P(act) ∈ [0, 1] |
| `outputs/metamodel_predictions_calibrated.csv` | `date, instrument, prediction` | byte-identical alias of the calibrated deliverable |
| `outputs/metamodel_predictions_raw.csv` | `date, instrument, prediction` | uncalibrated p̂ (pre-Platt; the §3 Brier/ECE-improvement story) |
| `outputs/strategy_weights.csv` | `date, instrument, weight` | §6 bonus — fractional-Kelly × vol-target leverage, signed by the primary side |
| `outputs/experiment_log.csv` | `run_id, asset_class, roster, cv_scheme, reducer, use_macro, best_model, oos_auc, oos_brier, oos_precision, notes` | one row per asset class, deterministically rewritten each emit run |
| `outputs/coverage_caveat.csv` | `instrument, n_oos_rows, ic, ic_undefined, thin` | thin-coverage flags (S5.9): `< 60` OOS rows or undefined IC |

**Diagnostics (not deliverables).** `experiments/*.py` write markdown + CSV to the gitignored
`experiments/results/` (e.g. `s6_barrier_backtest.md`, `s6_net_returns.csv`); they are regenerated
per run and feed nothing back into the locked deliverable. See
[`experiments/README.md`](experiments/README.md).

## 4. Data-flow boundary

```text
  data/ohlcv_data.csv ───────────┐
  data/primary_signals.csv ──────┤ stml.io.load_clean_data()
  data/additional_data.xlsx ─────┤ macro.py            ┌────────────────────────────────────────────┐
  results/instrument_scope.json ─┤ pipeline.py         │  metamodel-apb pipeline                     │
  stml.* causal feature fns ─────┘ (recomputed/fold) ─►│  features → triple-barrier labels → purged  │
                                                       │  CPCV horse-race → SELECT → calibrate → size │
                                                       └───────────────────────┬──────────────────────┘
                                                                               ▼
        outputs/  metamodel_predictions[.|_calibrated|_raw].csv · strategy_weights.csv
                  experiment_log.csv · coverage_caveat.csv          (all gitignored, byte-identical re-emit)

  ✗ results/feature_matrix.parquet — deliberately NOT read (leakage guard; see §2)
```

A fuller, annotated version of this flow is in [`docs/methodology.md`](docs/methodology.md) §0.

## 5. Repository layout

| Path | Role |
| :---: | :---: |
| [`src/alken_metamodel/`](src/alken_metamodel/README.md) | the package — 24 modules (see its README for the module map) |
| `tests/` | pytest suite mirroring `src/`, RED-first; key guards: truncation-invariance, per-instrument embargo, no-frozen-parquet, emit determinism, load-path (X.11) |
| `docs/methodology.md` | full §1–§6 methodology narrative |
| `docs/plans/` | dated build/design plans (`2026-05-30-metamodel-build.md` + passes 2–5) |
| [`experiments/`](experiments/README.md) | diagnostic / reproducibility probes (gitignored outputs; see its README) |
| `reports/` | coursework report + action-item trackers |
| `src/alken_metamodel/_vendor/` | verbatim sts-ml scripts; Stage-3 bug fixes logged in `_vendor/__init__.py` |
| `pyproject.toml` | package + dependencies (Python 3.12; `stml` editable path dep) |
| `CLAUDE.md` | non-negotiable rules for AI agents working in this subproject |
| `.markdownlint.jsonc` | markdown lint config (`MD013` line-length disabled) |
| `outputs/` | emitted deliverable CSVs (gitignored; created by `emit`) |

## 6. Methodology, determinism & further reading

The method is justified choice-by-choice against the literature review (8 commitments, 60
references). The four load-bearing disciplines:

1. **Causal recompute per fold** — stml feature *functions* re-run on each fold's train slice; the
   frozen feature matrix is never consumed (§2).
2. **Purge + embargo on triple-barrier `t1`** — overlapping meta-labels are purged by first-touch
   time, with a per-instrument `embargo_p90` advanced on each instrument's own date axis.
3. **Config-driven prediction window** — never hardcoded, so the hidden Jul–Dec 2022 half is a
   one-value swap.
4. **Determinism** — seeds pinned, native kernels single-threaded, CSV emitter sorted/pinned →
   byte-identical re-emit.

Further reading: [`docs/methodology.md`](docs/methodology.md) (full §1–§6 + bonus),
[`docs/plans/2026-05-30-metamodel-build.md`](docs/plans/2026-05-30-metamodel-build.md) (build
sequence + 8 commitments), [`CLAUDE.md`](CLAUDE.md) (rules), and `reports/` (the coursework report).
