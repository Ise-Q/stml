# Build Notes — Technical Documentation

> **Audience:** the team. **Not** the graded report.
>
> These notes document *how* and *why* we built each component — design decisions,
> parameter choices, economic and ML rationale, test outcomes, known limitations.
> The graded report (`reports/` — TBD) will be written later from a subset of this
> material plus narrative analysis.

## Index

| File | Stage | What it covers |
|---|---|---|
| [`01-labeling.md`](01-labeling.md) | Stage 1a | Triple-barrier method, t1, uniqueness weights, parameter choices |
| [`02-cv.md`](02-cv.md) | Stage 1b | Purged K-fold, embargo, walk-forward; no-leakage guarantee |
| [`03-thin-pipeline.md`](03-thin-pipeline.md) | Stage 2 | End-to-end thin pipeline, baseline model, first `predictions.csv` |
| [`04-features.md`](04-features.md) | Stage 3a | Full feature library (G1–G5, G7); economic rationale per group |
| [`05-regimes.md`](05-regimes.md) | Stage 3b | HMM/GMM regime features, strict causal fitting protocol |
| [`06-pipeline-v1.md`](06-pipeline-v1.md) | Stage 3c | Full-feature pipeline, v0 vs v1 comparison |

## Conventions

- **All features are computed strictly causally.** Any feature that uses information
  from time > t to inform the value at time t is a bug. The grader reruns the code
  on hidden Jul–Dec 2022 data; non-causal code will either break or be visibly wrong.
- **Train/predict boundary is a parameter** of the master pipeline. For our submission
  it is `2022-01-01` (predict H1 2022). On rerun the grader sets it to `2022-07-01`.
- **Module location:** shared code in `src/stml/` (importable as `from stml.X import Y`).
- **Tests:** `tests/` directory; run via `uv run pytest tests/`.
- **Personal artifacts** (figures, intermediate CSVs, scratch notebooks) under
  `notebooks/sreeram/` and `results/sreeram/` per the repo workflow.

## Design principles

1. **Methodology > performance.** The rubric awards 0 marks for raw performance;
   100 for methodology. Every decision is justified on methodological grounds first,
   economic grounds second, and convenience never.
2. **Causality is non-negotiable.** Every transform that depends on time-ordered data
   has an explicit `t` boundary parameter and is unit-tested for no-peeking.
3. **Panel-first.** We pool across all 11 instruments rather than fit 11 separate models;
   per-instrument behavior is carried by features and evaluated in breakdowns.
4. **Course-aligned.** Methods used are explicitly taught in lectures 1–4 and
   programming sessions 1–8. Departures are documented and justified.
5. **Tests before claims.** Any non-trivial computational claim in the report must be
   backed by a unit test or a sanity-check cell in a notebook.
