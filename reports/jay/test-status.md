# Test-Triage Status — PR `signal-deep-dive-clean` → `main`

> Branch: `signal-deep-dive-clean` · Date: 2026-06-11
> Plan: [`.omc/plans/branch-cleanup-main-merge.md`](../../.omc/plans/branch-cleanup-main-merge.md) (findings F8–F10)

---

## 1. Headline position

**`tests/model/` is 100% green — that is the merge gate.**

Full-suite baseline: **9 failed, 512 passed** (recorded on `uv run pytest -q`, ~5 m 21 s). Every
one of the 9 failures is **pre-existing and inherited from the frozen FE/harry layer**. Zero are
in `tests/model/`. This branch changes no existing code (it is a hygiene-only clean), so the
count is expected to be identical on the clean branch.

---

## 2. Classification of the 9 failures

### 2a. pywt read-only buffer — 7 failures (environment/dependency)

| File | Tests |
|---|---|
| `tests/harry/test_wavelet.py` | 4 tests (exact names in that file) |
| `tests/harry/test_causality.py` `[mra_energy_bands]` | 3 parametrized: `test_truncation_invariance`, `test_shape_preservation`, `test_no_nan_or_inf_after_warmup` |

**Error:** `pywt.wavedec` raises "buffer source array is read-only" under pywt 1.9.0 / numpy 2.x.

**Class: environment/dependency.** Both packages are locked in `uv.lock` at HEAD. This predates the
branch, is not triggered by any code on it, and does not affect the model layer (pywt is only used
in the frozen FE feature `features_ext.py`). Declined fix (S1) for this clean: adding `.copy()` in
`src/stml/metamodel/features_ext.py` would touch the frozen FE layer. Logged as a follow-up.

### 2b. Stale macro artifact — 1 failure (stale data-artifact drift)

| Test | `tests/test_macro_features.py::test_macro_artifact_row_aligned` |
|---|---|
| Error | `KeyError` — committed macro artifact is out of sync with the current feature matrix |
| Class | Pre-existing data-artifact drift; not introduced by this branch |

Declined fix (S2): regenerating the macro artifact is a separate, deliberate task outside the scope
of a hygiene clean. Logged as a follow-up.

### 2c. BLAS summation-order non-determinism — 1 failure (environment)

| Test | `tests/test_build_determinism.py::test_rebuild_is_frame_equal` |
|---|---|
| Error | 10/6,687 fitted-transform elements exceed `atol=1e-10`; max diff 1.6e-10 |
| Affected columns | `f4_cluster_dist` (Euclidean-to-centroid float reduction) + an F17 HMM posterior |
| Class | Environment (BLAS sum-order); fails on every standalone run |

The tolerance of `1e-10` is marginally too tight for last-ULP float-sum non-associativity. The
test's own docstring (`tests/test_build_determinism.py:21`) attributes the band to this cause. The
deterministic columns reproduce bit-exact; only fitted-transform float reductions drift by ~1.6e-10.
This is not a real determinism break. Declined fix (S3) for this clean: loosening `atol` to ~1e-9
would touch a frozen-adjacent test this branch does not own. Logged as a follow-up.

---

## 3. Lint debt

`ruff check src/` and `ruff check tests/model/` → **"All checks passed!"** The model package and its
tests are fully ruff-clean (plan finding F10).

**3 pre-existing F401 (unused-import) errors in frozen-adjacent test files:**

| File | Line |
|---|---|
| `tests/harry/test_labels.py` | 13 |
| `tests/harry/test_signal_audit.py` | 10 |
| `tests/test_drift_features.py` | 14 |

**Out of scope for this clean.** The frozen-layer discipline covers the FE/harry test files as well
as the source: this branch does not own them. Fix offered as explicit opt-in follow-up (S4: `ruff
--fix` on the three files, genuinely zero-risk).

---

## 4. Declined opt-in follow-ups

| ID | What | Why declined for this clean |
|---|---|---|
| S1 | Add `.copy()` before `pywt.wavedec` in `src/stml/metamodel/features_ext.py` — would green the 7 pywt reds | Touches the frozen FE layer |
| S2 | Regenerate the committed macro artifact to fix `test_macro_artifact_row_aligned` | Deliberate separate task; out of scope for hygiene |
| S3 | Loosen `atol` in `test_build_determinism.py` from `1e-10` to ~`1e-9` | Touches a frozen-adjacent test this branch doesn't own |
| S4 | `ruff --fix` the 3 F401s in frozen-adjacent test files | Touches test files outside this branch's ownership boundary |

All four are **safe and actionable** — logged here so the next session can pick them up without
re-investigation.

---

## 5. `model/hmm_features_v2.py` — intentional experimental orphan

`src/stml/model/hmm_features_v2.py` is **by design not part of the default pipeline.** It is not
re-exported from `model/__init__.py` and has no callers in production code. Its sole importer is
its own test (`tests/model/test_hmm_features_v2.py`). This is a label-aware HMM variant
(11-dimensional observation including past resolved-label columns `rhr`/`rer`/`rs`; states ordered
by hit-rate) that is intentionally distinct from the frozen F17 feature (`ret,vol`-only, label-free).
It cannot live in the FE layer without inverting the layering. **Not dead code — experimental module,
kept for future promotion.** Document in the PR description so reviewers don't flag it.
