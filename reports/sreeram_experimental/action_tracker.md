# Sreeram_experimental — Action Tracker

> Chronological log of every build session. Alken pattern (their
> `STML_ActionItem_Tracker.md`) — one PM-N entry per session, with the date,
> what was done, gates that passed, and what's next. Per plan R10, this gets
> appended every session that touches the experimental package.

---

## PM-1 — 2026-06-02 evening — S0 scaffold + S1 labels (Bloomberg-free)

**Goal of session.** Execute plan §8 Stages 0 and 1 in full, gate everything,
hand the user the precise Bloomberg pull list.

**Done.**

1. **Branch hygiene** — cleaned the working tree (was inheriting 100+ leftover
   files from the prior Sreeram-rebase). After `git clean -fdx`, the branch
   carries only the orphan baseline (`branch_descriptions.md` +
   `reports/sreeram_experimental/plan.md`).

2. **S0.a — Shared spine import.** `chore(s0)` commit `6ea00fd` brings in
   `data/ohlcv_data.csv`, `data/primary_signals.csv`, `data/meta/*.csv`,
   `src/stml/{__init__,io,na_checks}.py`, `pyproject.toml`, `uv.lock`,
   `README.md`, `refs/*`, `reports/missing-data-report.md`, `reports/README.md`,
   `.gitignore`, `.gitattributes` from `origin/main` — byte-identical, R8
   read-only.

3. **S0.b — Experimental scaffold.** `feat(s0)` commit `68eee39` adds
   `src/stml/experimental/{__init__,_env,seeding,config}.py`,
   `tests/experimental/{__init__,conftest,test_scaffold}.py`,
   notebooks/sreeram_experimental/ + results/sreeram_experimental/ +
   data/bloomberg/{raw,cleaned}/ directories. Extends pyproject.toml with
   `arch>=7.0`, `lightgbm>=4.0`, `hmmlearn>=0.3` (base deps); moved `torch` and
   `shap` to optional extras `multitask` / `importance` because their wheels
   collide with macOS x86_64 + py3.12 in the base build. Registered
   `pytest.mark.slow`.

4. **S0.c — Acceptance gates.** All three pass:
   - `uv sync` succeeds (128 packages, 0 conflicts).
   - `python -c "from stml.experimental import _env, seeding, config"` → `OK: 42`.
   - `pytest tests/experimental/ -q` → 7/7 passed.

5. **S1.a-d — Labels stack.** `feat(s1)` commit `39d6415`:
   - `src/stml/experimental/data_loader.py` (per-instrument frames, trading-day
     offset on the instrument's own calendar)
   - `src/stml/experimental/volatility.py` (GK / Parkinson / RS closed forms +
     rolling annualised σ̂ + EWMA daily σ̂ + GARCH(1,1) one-step-ahead daily σ̂
     via arch + cumulative-h GARCH variant for experiments)
   - `src/stml/experimental/labels.py` (triple-barrier with **t+1 entry** —
     Harry's load-bearing fix — AFML Ch.4 uniqueness weights computed via
     vectorised diff/cumsum, drop policy on right-edge truncation)
   - `src/stml/experimental/make_labels.py` (CLI runner)
   - Tests: 14 cases for labels (Harry's t+1 vs t entry-opposite-label
     5-row example; uniqueness on disjoint + adjacent events; truncation
     invariance; schema; long/short PT directional semantics; vertical timeout;
     NaN sigma drop; right-edge drop). 13 cases for volatility (closed forms,
     rolling warmup, EWMA + GARCH truncation invariance, GARCH config). 4
     cases for data_loader.

6. **S1.e-f — Real-data run + acceptance gate.**
   `uv run python -m stml.experimental.make_labels --quiet` produces
   `data/sreeram_experimental_events.parquet` (4886 rows × 10 cols) +
   `results/sreeram_experimental/label_outcome_audit.csv`. Wall-clock ~80s
   (GARCH fits 11 instruments × ~3000 bars truncated).

   **Plan §8 S1 acceptance gates:**
   - [PASS] event count = 4886 (target 4886 ±50, **delta = +0**). Per-instrument
     breakdown matches Harry's events.csv EXACTLY:
     cl1s 411 · es1s 564 · fesx1s 626 · gc1s 161 · hg1s 617 · ho1s 63 ·
     ng1s 120 · nq1s 593 · pl1s 547 · rb1s 617 · si1s 567.
   - [PASS] per-instrument vertical fraction max = 0.160 (< 0.65 target).
     Pooled vertical fraction 12.5 % — the tighter pt=sl=0.5 barriers resolve
     reliably (vs Harry's pt=1.5/sl=1.0 ~30-40% vertical or alken's pt=sl=1.0
     ~50-65% vertical).
   - Pooled: PT 47.8% / SL 39.7% / Vert 12.5%; pos rate 0.547;
     mean uniqueness 0.20 (N_eff ≈ 975).

7. **S2 gate-document — Bloomberg pull list.** Commit `be2a046` adds
   `reports/sreeram_experimental/bloomberg_pull_list.md` with the precise
   ticker / field / format spec for the four Blocks (A: futures 2nd month +
   VIX1/2; B: ATM/skew IV; C: CFTC COT; E: event flags). Documents what's
   already in Harry's CSV (21 macro series) so the user doesn't double-pull.

**Test totals end of session.** 38 tests (36 fast + 2 slow), 0 failures, on
`stml.experimental`. Plan §9 RED-first discipline observed throughout — every
behaviour test exists before / alongside the implementation.

**Total commits on branch.** 4:
- `5dbeb70` init(Sreeram_experimental): clean baseline
- `6ea00fd` chore(s0): import shared spine + data
- `68eee39` feat(s0): scaffold experimental package
- `39d6415` feat(s1): data loader + GARCH σ̂ + triple-barrier labels
- `be2a046` docs(s1): commit Bloomberg pull list

**Gating step.** S2 (feature stack including F18-F22 Bloomberg families) needs
the Bloomberg pull. List handed to user via `bloomberg_pull_list.md`.

**Next session.** When the user reports BBG files in `data/bloomberg/raw/`,
start S2:
- `bloomberg_ingest.py` (PIT alignment + lag + cleaned parquets)
- F1-F17 features (lifted from Sreeram + Harry, drift-filtered)
- F18-F22 NEW families
- Drift filter (KS) + macro reformulation (levels → 63d rolling rank)
- `data/sreeram_experimental_features.parquet`
- `feature_drift_audit.csv`

Wall-clock estimate once data arrives: 1-2 days for S2.
