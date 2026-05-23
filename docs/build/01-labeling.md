# Stage 1a — Triple-Barrier Labeling

> Module: [`src/stml/labeling.py`](../../src/stml/labeling.py)
> Tests: [`tests/test_labeling.py`](../../tests/test_labeling.py) — 22 tests
> Course refs: Lecture 1 §"Labeling"; AFML Ch. 3 (López de Prado)

## What this module does

For every event ``(date t, instrument, side ∈ {−1, +1})`` where the primary
signal is non-zero, it produces:

1. A **first-touch time** `t1` ∈ `[t, t + h]` (the earliest of profit-take /
   stop-loss / vertical barrier).
2. A **realised signed return** at `t1` (return in the bet's direction).
3. A **binary meta-label** ∈ `{0, 1}`: `1` iff the bet was profitable
   (`ret > 0`).
4. **Sample-uniqueness weights** (AFML Ch. 4 concurrency).

`t1` is the input to purged CV in [`02-cv.md`](02-cv.md).

## Design choices and why

### 1. Volatility scaling: EWMA daily-return std, span = 100

Barriers are set at `±m · σ_t · √h`, where `σ_t` is the EWMA daily volatility
of log returns up to and **including** date `t`.

- **Why EWMA, not rolling?** EWMA adapts smoothly without sharp transitions
  when a high-vol day enters/exits the window — better for risk-state targeting.
  AFML uses EWMA (`getDailyVol`); the course's Lecture 1 also uses it.
- **Why span 100?** Captures ~6 months of market context — long enough to be
  stable, short enough to track the regime. Tested with sensitivity at 50 / 200
  (planned in Stage 4).
- **Causal by construction:** `ewm` with `adjust=False` is one-pass and uses no
  future information. Verified by a "truncate-and-recompute" test
  ([`test_causality_no_peeking`](../../tests/test_labeling.py)).
- **Log returns, not pct change.** Consistent with the rest of the project
  (`stml.na_checks.native_returns` uses log). Log-returns are additive across
  time, which matters when computing cumulative path returns later.

### 2. Vertical barrier: h = 10 trading days

- The EDA shows the primary signal's mean run-length is 5–10+ days, so `h = 10`
  ≈ the bet's natural holding period.
- Long enough for a vol-scaled move (`σ · √10` ≈ 3× the daily vol) to resolve.
- Short enough to keep label/feature overlap manageable (uniqueness weights
  give 73% effective sample size at this h — see real-data numbers below).
- **Sensitivity testing planned** at h ∈ {5, 10, 15} in Stage 4.

### 3. Horizontal barriers: symmetric, `pt_mult = sl_mult = 1.0`

Upper barrier at `+1.0 · σ_t · √h` ≈ one-σ of the h-day move in the bet
direction. Lower at `−1.0 · σ_t · √h`.

- **Why symmetric?** A symmetric barrier band means the label measures pure
  directional correctness; an asymmetric band would encode a risk-management
  preference into the label itself, conflating "was the direction right" with
  "was the risk asymmetry right". For a binary "is the primary signal worth
  taking" question, symmetric is the cleanest target.
- **Why ~1σ?** A ±1σ band over `h` days is economically meaningful: it
  comfortably exceeds round-trip transaction costs on liquid futures
  (typically 1–3 bps per side) and is a reasonable conviction threshold
  ("did the bet make a real move, not noise").
- **Asymmetric variants tested in Stage 4** (e.g. pt=1.5, sl=1.0 — risk-on)
  as a robustness check.

### 4. Time-out resolution: by sign of `ret`

When neither PT nor SL is breached, `t1` = the vertical barrier and the label
is `1 if ret > 0 else 0`. This unifies the labelling rule (label is always
`sign(ret)`) and is the canonical meta-labeling convention.

Alternative considered and rejected: label time-outs as `0` (no PT touch =
unsuccessful). That would penalize slow profits unfairly and bias the
distribution toward 0.

### 5. Barrier touch on closes only

We check close-to-close, not intra-bar high/low.

- AFML's standard treatment.
- Avoids the order-of-touch ambiguity within a single bar (if both PT and SL
  ranges are crossed intra-day, which came first?).
- Documented in code as a possible refinement; not used here.

### 6. Sample-uniqueness weights, per instrument (AFML Ch. 4)

For each event ``i`` with span `[t_i, t1_i]`:

```
concurrency(u) = #{j : t_j <= u <= t1_j, same instrument as i}
uniqueness(i)  = mean over u in [t_i, t1_i] of (1 / concurrency(u))
weight(i)      = uniqueness(i) / mean(uniqueness)        # normalize mean = 1
```

- **Why per-instrument?** Two events on different instruments don't share a
  price path — their labels are independent at the path level. Calendar-time
  concurrency is what we purge on in CV, which is a different concern.
- **Effective sample size** = `(Σw)² / Σw²`. On our real data with `h=10` it
  is ~3634 vs N=4975 (≈ 73%) — a meaningful concentration that downweights
  the dense-concurrent-bet regions.
- Weights feed `sample_weight` in sklearn / xgboost training, and indirectly
  affect feature-importance estimates.

## Real-data sanity check

Running `get_meta_labels(ohlcv, signals, h=10, pt_mult=1.0, sl_mult=1.0)`:

```
Total labeled events: 4975  (across 11 instruments)

Per-instrument summary:
            n_events  label_1_share  pt_share  sl_share  vertical_share
cl1s             421         0.69      0.20      0.13       0.67
es1s             574         0.58      0.23      0.21       0.56
fesx1s           636         0.53      0.21      0.14       0.65
gc1s             167         0.60      0.20      0.10       0.70
hg1s             627         0.54      0.23      0.19       0.57
ho1s              63         0.65      0.27      0.16       0.57
ng1s             124         0.58      0.35      0.15       0.51
nq1s             603         0.59      0.27      0.18       0.56
pl1s             556         0.51      0.22      0.18       0.60
rb1s             627         0.51      0.20      0.16       0.63
si1s             577         0.56      0.19      0.19       0.62

Overall: label_1 = 0.563, pt = 0.22, sl = 0.17, vertical = 0.61
```

**Reads economically:**

- The primary signal is informative — overall label balance is **56.3%
  positive** (above the 50% no-skill baseline) and the PT/SL ratio of 22/17
  also favours profit-taking.
- **`cl1s` (crude oil) leads at 69%** — consistent with the EDA's observation
  that crude had clean trends in this window (COVID crash → 2022 oil spike).
- **`pl1s` and `rb1s` are the weakest at 51%** — choppy assets where trend
  signals get whipsawed. The metamodel's job is to identify which of *these*
  signals are still worth taking.
- ~61% of events are vertical (time-outs) — the barriers are not too tight;
  most bets resolve gradually rather than via stop-out.
- Uniqueness weights produce effective N ≈ 3634 vs 4975 actual — there's
  meaningful overlap in label spans (with `h=10`, up to 10 events on the
  same instrument can be concurrent).

## Test coverage (22 tests, all passing)

- **Causality**: re-running vol on truncated data agrees with original up to
  the truncation point — proves no peeking.
- **Triple barrier on constructed paths**: rising → PT hits, falling → SL hits,
  short on falling → PT hits (signed-return semantics), flat → vertical with
  zero ret, both barriers reachable → earliest wins, end-of-data → NaN gracefully,
  NaN sigma → vertical fallback, invalid side → ValueError.
- **End-to-end on synthetic 2-instrument panel**: shape + columns correct,
  `label = (ret > 0)`, `t1 ≥ t`, up-trending instrument has higher label_1
  share than down-trending one (both long bets).
- **Uniqueness weights**: disjoint events → all weight 1; fully overlapping
  events → raw weight 0.5 each, normalized = 1; cross-instrument events don't
  affect each other; normalisation produces mean = 1.
- **Fixed-horizon baseline** (rejected): label equals `(ret > threshold)` by
  construction, ret matches manual `log(close_{t+h}/close_t)`.

## Known limitations / refinements

1. **Concurrency uses calendar days, not trading days.** Weekends inside a
   span count as additional points with the same concurrency as Friday. For
   the typical h = 10 this changes weights by < 1%; a trading-day version
   would be a clean refinement.
2. **Close-only barrier touch.** As above, intra-bar touches are not detected.
3. **`sigma_at_t` is unconditional EWMA** — does not use the day's range or
   intra-day path. A range-based estimator (Parkinson, Garman-Klass) for vol
   is implemented in Stage 3a and could be substituted here as an ablation.
4. **No down-weighting of large-h labels** when fewer events resolve. We do
   weight by uniqueness, which partly captures this.
