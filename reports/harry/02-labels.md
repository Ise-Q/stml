# 02 — Triple-Barrier Labels with Next-Day-Execution Fix

> Module: [`src/stml/harry/labels.py`](../../src/stml/harry/labels.py)
> Tests: [`tests/harry/test_labels.py`](../../tests/harry/test_labels.py) — 19 tests
> Output (real data): [`results/harry/events.csv`](../../results/harry/events.csv)
> Course refs: Lecture 1 §"Labeling"; AFML Ch. 3 (triple-barrier), Ch. 4
> (sample uniqueness).

## 1. What this module produces

For every non-zero primary-signal day in the released window, the module
emits one event row. The output schema is:

```
instrument          ticker (cl1s, es1s, …, pl1s)
t_signal            date the primary signal was observed (close of t)
t_start             entry date  = t_signal + 1 trading day  (the fix)
t_end               resolution date (PT touch / SL touch / vertical)
side                ∈ {-1, +1}; events with signal = 0 are skipped
ret                 side * (log(close[t_end]) - log(close[t_start]))
label               1 if ret > 0 else 0
uniqueness_weight   AFML Ch.4 uniqueness in [0, 1] per event
sigma               EWMA daily-return std at t_signal (barrier scale)
```

On the real released window (2020-01-03 → 2022-06-30) this produces
**4 886 events** across the 11 instruments. Overall ``label_1`` share is
**0.556** — the primary signal is informative on average. Per-instrument
shares range from 0.510 (``hg1s``) to 0.698 (``cl1s``); ``cl1s`` is the
strongest instrument by both this and the Step 1 audit's forward hit-rate.

The effective sample size (sum of uniqueness weights) is **644**,
about 13 % of the raw count of 4 886 — events overlap heavily because
``h = 10`` is longer than the typical inter-signal gap on most
instruments. This is exactly the AFML Ch. 4 motivation for uniqueness
weighting.

## 2. The labelling formula

For each event at signal date ``t`` with side ``s_t ∈ {-1, +1}``:

```
sigma_t      = EWMA daily-log-return std up to and INCLUDING bar t,
               span = vol_span (default 100). Computed strictly causally
               via pandas .ewm(adjust=False).std() on log returns.
sqrt_h       = sqrt(h) where h is the holding horizon in trading days.

# Entry — the load-bearing fix:
entry_close  = close at bar  t + 1  (NOT close at t).

# Barrier widths in log-return units, in the direction of the bet:
pt_width     = pt_mult * sigma_t * sqrt(h)
sl_width     = sl_mult * sigma_t * sqrt(h)

# Touch logic — close-to-close, scanning bars  t + 1  through  t + 1 + h:
for u in (t + 2, t + 3, …, t + 1 + h):
    log_dist = log(close[u]) - log(entry_close)
    signed_dist = s_t * log_dist           # +ve toward profit
    if signed_dist >=  pt_width: PT touch at u → break
    if signed_dist <= -sl_width: SL touch at u → break

t_end = u_first_touch  if a barrier touched
        t + 1 + h      otherwise (vertical barrier)

ret   = s_t * (log(close[t_end]) - log(entry_close))
label = 1 if ret > 0 else 0
```

The entry bar (``t + 1``) is by construction at log-distance zero from
itself; it is excluded from the touch scan.

## 3. Decision 1 — Entry at ``t + 1``, not ``t``

This is the load-bearing change versus Sreeram's labeller. The Step 1
audit (``reports/harry/01-signal-direction.md``) empirically confirmed
``corr(s_t, r_{t+1}) > 0`` for all 11 instruments and ``corr(s_t, r_t)
< 0`` for 10 of 11 — the signal is observed at the close of ``t`` but
the next bar's return (``r_{t+1}``) is what it predicts. Acting at the
close of ``t`` would mean executing on information from the same bar
that produced the signal, which is the lookahead.

### 3.1 Worked 5-row example

Construct ``closes = [100, 90, 110, 115, 120]`` at consecutive trading
days; signal ``s_0 = +1`` at day 0; ``h = 3``; barrier thresholds (log
units) ``pt = sl = 0.05`` (call this case "fixed-threshold-0.05" — the
EWMA-vol-scaled formula reduces to a constant threshold under appropriate
``sigma_t``).

**OLD convention (entry at ``t = 0``, window = ``closes[0:4]``):**

```
entry_close = 100
log-distances:  [0, log(0.9)  = -0.105, log(1.1)  = +0.095, log(1.15) = +0.140]
signed-dist:    [0,           -0.105,           +0.095,           +0.140]
PT mask (≥ +0.05):  [F, F, T, T]
SL mask (≤ -0.05):  [F, T, F, F]
first touch         day 1   (SL)
ret_old = -0.105    label_old = 0
```

**NEW convention (entry at ``t + 1 = 1``, window = ``closes[1:5]``):**

```
entry_close = 90
log-distances:  [0, log(110/90) = +0.201, log(115/90) = +0.245, log(120/90) = +0.288]
signed-dist:    [0,             +0.201,             +0.245,             +0.288]
PT mask (≥ +0.05):  [F, T, T, T]
SL mask (≤ -0.05):  [F, F, F, F]
first touch         day 1   (PT)
ret_new = +0.201    label_new = 1
```

Same input, opposite label. This is encoded as
``test_off_by_one_fix_changes_label_5_row_ohlc`` in
``tests/harry/test_labels.py``, with the same hand-computed expected
values.

### 3.2 What this changes in practice

Sreeram's ``labeling.py`` resolves over ``[t, t+h]``: the return between
``t`` and ``t+1`` is inside the held window even though it is determined
by the same close that produced the signal. On the real released window
this is a small percentage of events per the audit's `corr_contemp_0`
values (typically −0.15 to −0.20), but it systematically biases the
realised return of every event by the contemporaneous correlation — a
non-trivial source of bias for the rubric's marking criteria.

## 4. Decision 2 — Symmetric default, asymmetric exposed

Defaults are ``pt_mult = sl_mult = 1.0``. The audit produced **0 trend /
4 mean-reverting / 7 mixed** sign labels under the canonical multi-horizon
classifier — too equivocal to bake a counter-trend bias into the label
itself. The triple-barrier *target* must stay neutral so downstream
feature-importance analysis can honestly tell us *whether* counter-trend
features matter rather than recovering a bias we put in. Asymmetric
variants will be a Step 4 sensitivity sweep, not the default.

The ``pt_mult`` / ``sl_mult`` parameters are independently configurable
in the API. The test ``test_asymmetric_barriers_change_label_distribution``
constructs three events with injected sigma and verifies the
mechanically-correct direction:

| config (pt / sl) | barrier widths | event A | event B | event C | label rate |
|---|---|---:|---:|---:|---:|
| tight  0.5 / 1.0 | ±1.58 % / ±3.16 % | 1 | 1 | 1 | 1.000 |
| sym    1.0 / 1.0 | ±3.16 % / ±3.16 % | 1 | 1 | 0 | 0.667 |
| wide   2.0 / 1.0 | ±6.32 % / ±3.16 % | 0 | 1 | 0 | 0.333 |

(Event A: round-trip +5 % → −5 %. Event B: monotone +1 %/day. Event C:
+2 % then declining to −5 %.) **Wider PT produces FEWER ``+1`` labels**
— the direction the Step-2 spec example had reversed. The
per-event monotonicity property ("wider PT can only demote labels, never
promote") is also tested directly across a synthetic random-walk panel.

## 5. Decision 3 — Trading-day concurrency for uniqueness weights

Sample-uniqueness weights follow AFML Ch. 4
(``getIndMatrix`` + ``getAvgUniqueness``) but the bar index used is the
instrument's **cleaned trading-day index** — that is, the close series
after ``dropna()``. Calendar-day rows in the input panel that are NaN
(weekends inserted by reindexing, missing-holiday flags) are dropped
before concurrency is computed. A three-day weekend between two events
does not inflate or shrink the computed concurrency.

The formula:

```
concurrency[u]  =  #{events i : t_start_i ≤ u ≤ t_end_i}   for u in bar_index
uniqueness_i    =  mean over u in [t_start_i, t_end_i] of  (1 / concurrency[u])
```

The implementation uses the cumulative-sum trick on a delta array (start
positions get +1, ``end + 1`` positions get -1, ``cumsum`` gives the
concurrency at every bar) — O(n_events + n_bars) instead of
O(n_events × avg_span).

The test ``test_trading_day_concurrency_invariant_to_weekend_padding``
verifies trading-day invariance: building the OHLCV + signals panel on
business days alone and on a calendar-day grid with NaN-padded weekends
produces *identical* uniqueness weights, ``t_end`` values, ``label``
values, and ``ret`` values.

## 6. Per-instrument numbers on the real released window

```
                n_events  label_1   uniq_mean   uniq_min   sigma_mean   long_share
cl1s                 411    0.698       0.147      0.091       0.0394        0.912
es1s                 564    0.578       0.129      0.091       0.0133        0.791
fesx1s               626    0.516       0.114      0.091       0.0145        0.457
gc1s                 161    0.615       0.271      0.100       0.0121        0.801
hg1s                 617    0.514       0.112      0.091       0.0148        0.483
ho1s                  63    0.651       0.375      0.123       0.0257        0.841
ng1s                 120    0.558       0.200      0.091       0.0450        0.000
nq1s                 593    0.597       0.121      0.091       0.0163        0.669
pl1s                 547    0.516       0.127      0.091       0.0220        0.751
rb1s                 617    0.510       0.114      0.091       0.0332        0.580
si1s                 567    0.541       0.121      0.091       0.0226        0.524
```

### Reads

- **`cl1s` leads at 0.698 label-1 share** — matches the Step 1 audit's
  0.71 forward hit-rate. Crude is the strongest single instrument on this
  metric, and the audit's three-way evidence already flagged it as a
  structural outlier.
- **`ng1s` is single-direction (`long_share = 0.000`)** — every non-zero
  signal is short. This is the same finding signal-deep-dive's
  characterization flagged independently.
- **`ho1s` has only 63 events** — the thinnest signal in the panel. The
  audit's effective-sample-size warning applies here: per-instrument
  claims about ``ho1s`` are statistically weak.
- **Uniqueness minima are pinned at 1/11 = 0.091** for most instruments —
  consistent with a worst-case full-overlap concurrency of 11 (all of an
  instrument's events overlapping at one bar). The instruments with high
  ``uniq_mean`` (``ho1s`` 0.375, ``gc1s`` 0.271) are the ones with sparse
  signals → little overlap.

## 7. Tests (19 total, all passing in <1 s)

| Group | Tests |
|---|---|
| Config validation | `test_triple_barrier_config_validates_inputs`, `test_triple_barrier_config_sqrt_h` |
| EWMA causality | `test_ewma_daily_vol_truncation_invariance` |
| Off-by-one fix | `test_off_by_one_fix_changes_label_5_row_ohlc`, `test_get_meta_labels_uses_t_plus_one_entry_integration` |
| Causality (full pipeline) | `test_truncation_invariance_for_resolved_events` |
| Asymmetric barriers | `test_asymmetric_barriers_change_label_distribution`, `test_wider_pt_can_only_lower_label_rate_per_event`, `test_asymmetric_barriers_short_side_consistency` |
| Uniqueness weights | `test_uniqueness_in_zero_one_and_sum_bounded`, `test_uniqueness_disjoint_events_all_one`, `test_uniqueness_two_fully_overlapping_events_get_half` |
| Trading-day concurrency | `test_trading_day_concurrency_invariant_to_weekend_padding` |
| Touch semantics | `test_resolve_event_vertical_when_no_touch_label_by_sign`, `test_resolve_event_first_touch_wins_among_pt_and_sl`, `test_resolve_event_short_side_pt_means_price_down`, `test_resolve_event_invalid_inputs` |
| Output schema | `test_get_meta_labels_output_schema_and_dtypes`, `test_get_meta_labels_empty_when_no_signals` |

## 8. Known limitations

1. **Concurrency exclusivity at the event boundary.** AFML's
   ``getIndMatrix`` includes both endpoints; Harry's implementation does
   the same. Two events with ``t_end_1 = t_start_2`` are jointly
   concurrent at that shared bar. This is the AFML default and matches
   the test ``test_uniqueness_two_fully_overlapping_events_get_half``.
2. **EWMA span is fixed per call.** ``vol_span = 100`` is the AFML
   default and the same value Sreeram uses. A per-instrument-tuned span
   would be a Step 4 sensitivity sweep.
3. **Close-to-close touches only.** Intra-bar high/low excursions are
   not checked. AFML's standard treatment, and consistent with
   ``stml.io.load_clean_data`` returning daily bars. If the rubric
   rewards intra-bar resolution this would need an extension.
4. **Vertical-barrier labelling by sign of return.** When neither PT
   nor SL touches, the event is labelled by the sign of the realised
   return. This is the canonical meta-labeling convention (AFML Ch. 3)
   and unifies the label rule across resolution types: ``label =
   (ret > 0)``.
5. **The events frame includes** ``t_signal`` **and** ``sigma`` **even
   though the Step-2 spec listed only** ``t_start, t_end, ret, label,
   side, uniqueness_weight``. The two extra columns are needed for
   chronological splitting in Step 4 (``t_signal`` owns the event for
   embargo / purge) and for diagnostics (``sigma`` lets us reconstruct
   the barrier widths). They do not change downstream behaviour.

## 9. What Steps 3–4 will consume

- The events frame from this module is the *target* of every model in
  Step 4. The pipeline calls ``get_meta_labels`` with the default
  configuration, splits chronologically by ``t_signal``, purges with
  embargo equal to ``h`` trading days (AFML Ch. 7), and feeds
  ``X = harry features at t_signal`` against ``y = label``.
- The ``uniqueness_weight`` column is passed as ``sample_weight`` to
  every model fit and to the cluster-importance permutation routine.
- The ``sigma`` column is reused by feature ``f1_dist_ma_sigma_*`` in
  Step 3 (counter-trend distance-from-MA in sigma units) — already
  defined in signal-deep-dive's catalog and vendored by Harry.
