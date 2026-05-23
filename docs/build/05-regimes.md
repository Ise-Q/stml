# Stage 3b — HMM / GMM Regime Features (G6) — the Showpiece

> Module: [`src/stml/regimes.py`](../../src/stml/regimes.py)
> Tests: [`tests/test_regimes.py`](../../tests/test_regimes.py) — 10 tests
> Course refs: Lecture 2 (GMM/EM), Lecture 3 (HMM regime detection, the
> entire lecture on this topic), Programming Session 3 (HMM implementation).
> Assignment quote: *"Latent variable models (GMM, HMM)"* explicitly listed
> as in-scope feature engineering.

## Why this matters

The assignment names GMM and HMM by hand. Lecture 3 is *entirely* about HMM
turbulence-regime detection. Programming Session 3 walks through fitting a
discrete HMM. This is the most distinctive single contribution we can make to
the feature engineering rubric — and the one most at risk of silent leakage
if implemented carelessly.

## The features (8 columns per event)

```
hmm_state_lo      P(low-vol regime  | data up to t)        ∈ [0, 1]
hmm_state_mid     P(mid-vol regime  | data up to t)        ∈ [0, 1]
hmm_state_hi      P(high-vol regime | data up to t)        ∈ [0, 1]
hmm_state_argmax  most-likely state ∈ {0, 1, 2}             integer
gmm_cluster_lo    P(low-vol cluster  | snapshot at t)       ∈ [0, 1]
gmm_cluster_mid   P(mid-vol cluster  | snapshot at t)       ∈ [0, 1]
gmm_cluster_hi    P(high-vol cluster | snapshot at t)       ∈ [0, 1]
gmm_cluster_argmax most-likely cluster                       integer
```

## The HMM pipeline

```
                                        (boundary, e.g. 2022-01-01)
                                                  │
                            ┌─────────────────────┴───────────────────┐
                            │                                          │
       Training observations: ret_t + vol_t      Inference observations: same X
       (date < boundary)                          (whole history)
                            │                                          │
                            ▼                                          ▼
                  fit_instrument_hmm                  causal_filtered_probs
                  GaussianHMM(n_states=3)             (manual forward pass —
                  covariance_type='full'              NEVER hmmlearn.predict_proba,
                  EM via hmmlearn                     which uses smoothed posteriors)
                            │                                          │
                            ▼                                          ▼
                    π, A, μ_k, Σ_k                    P(state_k | X_{0..t}) at every t
                                                                       │
                                                                       ▼
                                                      Reorder columns by mean vol:
                                                      state 0 = lowest, K-1 = highest
                                                      ⇒ hmm_state_lo / mid / hi
```

### Per instrument, separately

We fit **one HMM per instrument**. Reason: regime structure differs across
asset classes (oil's high-vol regime is structurally different from gold's),
and per-instrument fitting captures that. A pooled-across-panel HMM would
have to compromise between very different signal-to-noise structures.

### Strictly causal — the key invariant

Two threats:

1. **Future data in the training fit.** Trivially fixed by filtering on
   `date < boundary` before calling `fit`.

2. **Future data in the inference.** This is the subtle one. ``hmmlearn``'s
   `predict_proba` returns **smoothed** posteriors P(state_k | X_{0..T}) — they
   use the full sequence, including future observations. That means the
   posterior at time t depends on data t+1, t+2, …, T. *Catastrophic leakage.*

   Our fix: implement the forward algorithm ourselves
   (`causal_filtered_probs`), which yields P(state_k | X_{0..t}) — the
   filtered posterior. Filtered ≠ smoothed in general, and our test
   `test_filtered_differs_from_smoothed` proves it: on the toy panel the
   max difference is ~0.76 (i.e. smoothed assigns nearly opposite probabilities
   to the same row when it can see the future).

The forward algorithm:

```
log α[0, k]  = log π_k + log p(x_0 | state_k)
log α[t, k]  = log p(x_t | state_k) + logsumexp_{k'}( log α[t-1, k'] + log A[k', k] )
filtered[t, k] = α[t, k] / Σ_{k'} α[t, k']
```

By construction `filtered[t]` reads only `x_0, …, x_t` — never the future.
Our test `test_filtered_no_peeking` confirms: filtered probabilities computed
on `X[:1000]` are bit-identical to filtered probabilities computed on
`X[:1400]` then truncated to `[:1000]`.

### State reordering

Gaussian HMM has a **label-switching** identifiability problem: the EM
algorithm can converge to permuted versions of the same model. The "high
vol" state might be index 0 in one run and index 2 in another, breaking
column semantics.

We resolve it by sorting states **by training-set mean vol** (the second
column of `means_`). After reordering, `hmm_state_lo` is always the
lowest-vol regime, `hmm_state_hi` always the highest. Feature semantics are
stable across instruments and across reruns.

## The GMM pipeline

GMM has no temporal dimension — each row is an independent input. Causality
is automatic if the *snapshot* at time t uses only data up to t, and the
GMM was fit only on training-window snapshots.

For each instrument:

1. Build a 3-D snapshot per date: `(21d vol, 21d momentum, 21d autocorrelation)`.
   Each component is computed on the instrument's native series using only
   data up to that date.
2. Fit `GaussianMixture(n_components=3)` on snapshots with date < boundary.
3. `predict_proba` on the full snapshot series ⇒ soft cluster membership.
4. Reorder clusters by training-set mean vol (same identifiability fix).

GMM gives a more "instantaneous" regime signal than the HMM (which carries
state persistence via the transition matrix). Pairing the two captures both
the **current** regime guess (GMM) and the **persistent** regime belief that
weights recent observations against past evidence (HMM).

## Real-data sanity check — CL during the COVID crash

```
            hmm_state_lo  hmm_state_mid  hmm_state_hi  hmm_state_argmax
2020-03-09        0.0           0.0            1.0             2
2020-03-10        0.0           0.0            1.0             2
...
2020-03-20        0.0           0.0            1.0             2

            gmm_cluster_lo  gmm_cluster_mid  gmm_cluster_hi  gmm_cluster_argmax
2020-03-09        0.0           0.001          0.999             2
...
2020-03-20        0.0           0.000          1.000             2
```

Both models — independently — assign probability ~1.0 to the **high-vol
state/cluster** on every single day of the March 2020 COVID crash. This is a
strong sanity check that the regime features are picking up real economic
structure.

State distribution across the panel (4984 events):

```
hmm_state_argmax
0 (low-vol)   1220   24%
1 (mid-vol)   2021   41%
2 (high-vol)  1743   35%
```

The mid-vol regime is the modal state; the high-vol regime is the most
common during the 2020–2022 signal window, consistent with that window
containing the COVID crash and the 2022 energy/inflation regime.

## Test coverage (10 tests, all passing in 4.4s)

| Test | What it locks down |
|---|---|
| `test_filtered_no_peeking` | **THE invariant.** `filtered[t]` from `X[:t+1]` equals `filtered[t]` from `X[:T]`, T ≫ t, to 12 decimal places. |
| `test_rows_sum_to_one` | Filtered probabilities are well-defined posteriors. |
| `test_filtered_differs_from_smoothed` | Proof that we are **not** accidentally using smoothed posteriors (max diff > 0.001). |
| `test_states_are_vol_ordered` | After reordering by mean vol, the high-vol state activates more during high-vol regimes. |
| `test_training_only_uses_pre_boundary_data` | Different boundaries → different fitted HMMs → different filtered probs in the overlap region. |
| `test_too_little_data_returns_empty` | Graceful handling when training sample is too small. |
| `test_*_shape_and_columns` | Output schemas. |
| `test_*_sums_to_one` | Posterior properties. |
| `test_returns_one_row_per_event` | Master function aligns to events. |

## Known limitations

1. **Per-instrument HMM ⇒ 11 separate models.** Each must converge, which on
   thin-data instruments (ng1s, ho1s — small signal windows but full price
   history is fine) is robust enough. The EM can land in different local
   optima on different seeds — we fix `random_state=42` for reproducibility.
2. **3 states is a heuristic.** We could pick `n_states` via BIC per
   instrument. Stage 4 candidate ablation.
3. **Observation vector is just (return, vol).** Adding e.g. range or volume
   might help — but adds dimensionality without much extra signal in our
   experiments (planned ablation).
4. **HMM is refit at each rerun.** When the grader reruns with
   `boundary=2022-07-01`, the HMM is retrained on training data that now
   includes H1-2022. This is the right behaviour — the model adapts to the
   new training cutoff — but it means the filtered probabilities on
   2020–2021 dates will be subtly different in the submission vs the
   rerun. This is correct, intended, and documented.
