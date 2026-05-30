# CLAUDE.md — `metamodel-apb`

This file provides guidance to Claude Code (claude.ai/code) when working in this subproject.

## What this is

`alken_metamodel` is the **meta-labelling metamodel** for the T3.03 Alken coursework: a
secondary *act/skip* classifier over a provided primary signal, for 11 futures instruments
across three asset-class metamodels (Equity/Energy/Metals). It is a **nested uv subproject
inside the `stml` repo**, importing stml as an editable path dependency and reading the
released data read-only.

**The grade is methodology, not performance** (`../refs/project-instructions.md`). Every
methodological choice is justified against the literature review `../reports/apb/nlr-cw-v1.md`
(8 commitments, 60 refs). Full design + build sequence: `docs/plans/2026-05-30-metamodel-build.md`.

## Commands (run from the repo root with `--directory`)

```bash
uv sync   --directory metamodel-apb              # install (Python 3.12, pinned)
uv run    --directory metamodel-apb pytest       # tests
uv run    --directory metamodel-apb ruff check --no-fix src/ tests/   # lint (see ruff caveat)
uv run    --directory metamodel-apb ruff format src/ tests/
uv run    --directory metamodel-apb python -m alken_metamodel.emit    # emit the deliverable CSVs
```

**Ruff caveat:** this machine's global `~/.config/ruff/ruff.toml` sets `fix = true`, so a bare
`ruff check` rewrites files. Always verify with `ruff check --no-fix`.

## Non-negotiable rules

- **Leakage / fold-safety.** Reuse stml's *causal* feature **functions** (`stml.metamodel.features.assemble_engineered`, `f2_vol_dispersion`, regime `fit_*`/`transform_*`) **recomputed inside each CV fold** on the fold-train slice; fitted regime/HMM/latent blocks are re-fit on fold-train only. **Never consume `../results/feature_matrix.parquet`** — it freezes fitted stats at a single global `fe_train_end`, which leaks into in-sample folds before that date.
- **Triple-barrier labels overlap** → standard k-fold is invalid. Track `t1` (first-touch) per label; purge + embargo on `t1` everywhere, including cluster MDA.
- **Lock the feature set before final OOS.** No data-snooping on Jan–Jun 2022 (it rehearses the hidden Jul–Dec 2022 half).
- **Determinism.** Call `alken_metamodel.seeding.set_seeds()` at every entry point. Scalers/estimators fit on **train only**; every lag/rolling feature **shifted**. `emit` sorts rows, pins column order, fixes float format → byte-identical re-emit. The **prediction window is config-driven**, never hardcoded to Jan–Jun 2022 (the grader swaps in the hidden half).
- **Data is read-only.** Released data lives in `../data/` (loaded via `stml.io.load_clean_data()`); never edit it. Migrations of method live in code.
- **Branch discipline.** Never commit to `main` (stml convention); work on a model branch, open a draft PR.

## What not to touch

`../data/` (released inputs), `../reports/apb/nlr-cw-v1.md` (the lit review), `../refs/` (course
materials), and the `src/alken_metamodel/_vendor/` scripts except for the logged Stage-3 bug
fixes (see `_vendor/__init__.py`).

## Layout

`src/alken_metamodel/{volatility,triple_barrier,cross_validation,sizing,features,regime,cluster_importance,models,evaluation,pipeline,emit}.py`,
`_vendor/` (verbatim sts-ml scripts), `tests/` (mirrors src, RED-first), `outputs/` (the two CSVs, gitignored).
