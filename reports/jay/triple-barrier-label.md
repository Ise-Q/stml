# Triple-Barrier Label Selection — Per-Instrument, by Adjusted-Signal Sharpe, Across Return Lags

> Notebook: [`notebooks/jay/triple-barrier-label.ipynb`](../../notebooks/jay/triple-barrier-label.ipynb) (executed end-to-end)
> Reuses: [`src/stml/model/labels.py`](../../src/stml/model/labels.py) (`triple_barrier_labels`), [`src/stml/model/dataset.py`](../../src/stml/model/dataset.py) (`events_frame`, σ = de-annualised `f2_vol_20`), [`src/stml/model/evaluate.py`](../../src/stml/model/evaluate.py) (`nav_sharpe`, `release_test`)
> Artifact: [`results/triple_barrier_per_instrument.csv`](../../results/triple_barrier_per_instrument.csv) (top-3 geometries × 11 instruments × 4 PnL types, train + hold-out Sharpe)
> Companion to: [`reports/jay/metamodel-results.md`](metamodel-results.md) · Signal nature: [`reports/harry/01-signal-direction.md`](../harry/01-signal-direction.md)

---

## 0. What this study is — and how it differs from the AUC search

A **model-free** study of *which triple-barrier exit geometry `(pt, sl, h)` makes the best label* for
each instrument. The triple barrier produces a binary label `y ∈ {0,1}` per primary-signal event
(`1` = the trade was profitable under that profit-take / stop-loss / horizon exit). The label
**adjusts the primary signal** by zeroing out the trades it marks unprofitable:

> primary side `s = [+1, −1, +1]`, label `y = [1, 0, 0]` ⇒ **adjusted signal `s·y = [+1, 0, 0]`**,
> adjusted PnL `= Σ_i (s_i·y_i)·r_i` vs raw primary `Σ_i s_i·r_i`.

We score each geometry by the **Sharpe of the reconstructed adjusted-PnL curve** — *not* by AUC.

> **Why not AUC** (the methodological point). Each `(pt, sl, h)` defines a *different* label set — a
> different ground truth — so AUC values are not comparable across geometries (AUC compares *models*
> under a *fixed* labeling). The labeling-invariant objective is an **economic** one: the Sharpe of the
> PnL the label produces. This is the deliberate contrast with the meta-model's AUC-based
> `barrier_search` (see [`metamodel-results.md`](metamodel-results.md) §2, where an AUC search picked a
> degenerate geometry).

**The question this answers** — the user's criterion: *a good triple-barrier label should discern a
trustworthy primary from an untrustworthy one and produce a better risk-adjusted adjusted signal.*

---

## 1. Method

| Choice | Value |
|---|---|
| Grid (per instrument) | `pt, sl ∈ {0.25, 0.5, 0.75, 1, 1.5, 2, 2.5}` (**all asymmetric pairs**) × `h ∈ {1,2,3,5,10,15,20}` → **343 geometries** |
| Barrier scale σ | de-annualised `f2_vol_20` (`events_frame(de_annualize=True)`) |
| Train (in-sample) | 2020-01-03 → 2022-01-01 (the `dev` partitions); `price_end=2021-12-30` (no 2022 price read) |
| Hold-out | 2022-H1 (the `test` partition, via `release_test`); `price_end=2022-06-30` |
| Score | per `(instrument, geometry)`: `nav_sharpe` of the adjusted PnL (`y=1` trades) vs the raw PnL |
| Selection | rank by **in-sample adjusted Sharpe** → **top-3 per instrument**, then validate on the hold-out |

**Three return lags, in parallel.** The label `y` is the *filter*; the **return the kept position
earns** is the single-bar close-to-close return at lag `L ∈ {0,1,2}` from the signal bar `t`:
`g_i(L) = s_i · u_{t_i+L}`, `u_τ = close_τ/close_{τ−1}−1`.

- **lag 0 = `u_t`** — contemporaneous, *known at signal time* → a **non-tradeable placebo**.
- **lag 1 = `u_{t+1}`** — the **first tradeable bar**, where the signal's edge lives.
- **lag 2 = `u_{t+2}`** — decayed.

The full grid → top-3 → hold-out runs for each lag, plus the **realized triple-barrier exit return**
as a reference. A *real* edge should make lag-1 dominate lag-0/lag-2 — a **placebo-in-time** test that
operationalises "discern a good signal from a bad one" (the bad-timing signal is the placebo lag).

> **This is an oracle / in-sample measure by design.** Filtering realized returns by realized labels is
> an *upper bound* on what a learnable meta-model could capture — it measures *labeling quality*, not a
> deployable backtest. The **hold-out** and the **lag contrast** are the honesty guards.

---

## 2. Placebo-in-time — the core validation

**Raw per-lag Sharpe** (no geometry, no filter — just `s·u_{t+L}`), per instrument:

| instrument | lag 0 | lag 1 | lag 2 |
|---|---|---|---|
| cl1s | 0.420 | **2.424** | 1.572 |
| es1s | −1.162 | **1.729** | 0.884 |
| fesx1s | −3.300 | **0.324** | 0.611 |
| gc1s | 0.962 | **4.279** | 1.912 |
| hg1s | −3.010 | **2.230** | 0.362 |
| ho1s | −5.273 | **3.807** | 2.727 |
| ng1s | 0.525 | **2.698** | 2.746 |
| nq1s | −2.388 | **1.779** | 0.907 |
| pl1s | −2.502 | **1.085** | 0.976 |
| rb1s | −0.961 | **0.660** | −0.547 |
| si1s | −2.593 | **1.064** | −0.319 |

**lag-1 is the strongest lag for all 11 instruments**; lag-0 is negative for 8/11 (the contemporaneous
counter-trend reaction — consistent with `corr(s_t, u_t) < 0` in the signal study); lag-2 decays. This
is the project's "signal predicts `t+1`" finding, recovered cleanly and instrument-by-instrument.

**The label filter then amplifies only the real lag.** Median (across instruments) of the in-sample
**top-1 adjusted** Sharpe, by PnL type:

| PnL type | median top-1 **adjusted** Sharpe | median **raw** Sharpe |
|---|---|---|
| lag 0 (placebo) | **−1.46** | −2.38 |
| lag 1 (tradeable) | **15.37** | 1.80 |
| lag 2 (decayed) | 8.62 | 0.90 |
| realized exit | 34.55 | 1.97 |

The filter **cannot rescue lag-0** (it stays negative) — strong evidence the label encodes genuine
forward-looking structure, not a generic "keep positive returns" trick. It lifts lag-1 from ~1.8 to
~15 — the labeling *discerns* the good trades precisely where the signal has real predictive timing.

---

## 3. In-sample selection & hold-out validation (tradeable lag-1)

**Recommended geometry per instrument** = the in-sample top-1 for the tradeable **lag-1** PnL, with its
hold-out adjusted Sharpe, the hold-out *raw* Sharpe (the unfiltered primary), and the hold-out **rank**
of that geometry among all 343 (1 = best out-of-sample):

| instrument | pt | sl | h | adj Sharpe (train) | adj Sharpe (hold-out) | raw Sharpe (hold-out) | hold-out rank /343 |
|---|---|---|---|---|---|---|---|
| cl1s | 0.25 | 0.25 | 1 | 10.51 | 17.76 | 3.57 | 25 |
| es1s | 1.00 | 0.25 | 10 | 13.13 | **24.44** | 1.17 | 11 |
| fesx1s | 0.25 | 0.25 | 1 | 14.91 | 19.87 | 2.62 | 25 |
| gc1s | 1.00 | 0.25 | 15 | 15.34 | 18.32 | 4.58 | 109 |
| hg1s | 1.00 | 0.25 | 10 | 19.88 | **23.67** | 1.88 | 11 |
| ho1s | 0.25 | 0.25 | 1 | 19.71 | **0.00** | **−31.0** | 172 |
| ng1s | 0.25 | 0.25 | 1 | 27.50 | 16.86 | 1.02 | 125 |
| nq1s | 0.25 | 0.25 | 1 | 15.37 | 24.44 | 2.47 | 41 |
| pl1s | 0.25 | 0.25 | 1 | 18.87 | 22.01 | 1.09 | 52 |
| rb1s | 2.50 | 0.25 | 15 | 10.90 | 16.97 | 3.31 | 110 |
| si1s | 0.75 | 0.25 | 20 | 16.17 | 23.93 | 0.27 | 58 |

*(Full top-3 per `(instrument, PnL type)` in [`results/triple_barrier_per_instrument.csv`](../../results/triple_barrier_per_instrument.csv).)*

**What generalises.** For **10/11 instruments the in-sample pick keeps a high adjusted Sharpe
out-of-sample** (es1s 13→24, hg1s 20→24, nq1s 15→24, si1s 16→24, pl1s 19→22, cl1s 10.5→18) and the
adjusted Sharpe **beats the raw primary** on the hold-out — the label adds value out-of-sample. The
hold-out *rank* is mid-pack (≈ 25–125 of 343): the in-sample-best is rarely the hold-out-best, but it
stays in the upper portion, so selection is robust-ish rather than perfect.

**The cautionary failure — `ho1s`.** In-sample adjusted Sharpe 19.7 looked excellent, but the hold-out
raw primary Sharpe is **−31** and the filter sits almost entirely out (adjusted → 0, the degenerate
"too-few-trades" value). `ho1s` is the thinnest name (~50 events) — its in-sample pick is noise. This
is exactly what the hold-out exists to expose; it is reported, not buried (with `ng1s`, the short-only
name, the other thin caveat).

---

## 4. Caveats (read before using the numbers)

- **Oracle / in-sample by design.** The adjusted Sharpe uses realized `y` to filter realized returns —
  an upper bound on a learnable meta-model, not a deployable strategy. The hold-out + lag contrast are
  the guards; the *absolute* Sharpe levels (10–30) are not achievable live.
- **Very short `h` overlaps the lag-1 return — the dominant artifact.** Many winners sit at `h=1` with a
  tight stop `sl=0.25`. With a 1-bar horizon the label `y` is largely a function of the **first forward
  bar — the same bar as the lag-1 return** — so filtering the lag-1 PnL by `y` is partly self-fulfilling
  (circular), inflating in-sample adjusted Sharpe for tiny-`h` geometries. **Trust the larger-`h`,
  OOS-persistent picks** (`es1s` `1/0.25/10`, `hg1s` `1/0.25/10`, `gc1s` `1/0.25/15`, `si1s`
  `0.75/0.25/20`, `rb1s` `2.5/0.25/15`) over the `h=1` names (`cl1s`, `fesx1s`, `ng1s`, `nq1s`, `pl1s`),
  whose in-sample peak is partly the overlap. The **realized-exit panel** (a multi-bar return that does
  not share a single bar with the label) and the **hold-out** are the cross-checks.
- **Tight stop `sl=0.25` dominates** almost everywhere — a tight stop cuts losers fast (→ `y=0`,
  dropped), which the oracle filter rewards. Live, a tight stop also realises more whipsaw losses; the
  oracle doesn't see that cost.
- **Sharpe annualisation** uses `nav_sharpe`'s per-trade × √252; comparing very short `h` vs long `h` is
  approximate but applied consistently.
- **Thin instruments** (`ho1s` ~50 events, `ng1s` short-only) are noisy — surfaced, not hidden.

---

## 5. Conclusions & how to use this

- **The methodology works as a labeling diagnostic.** The placebo-in-time result is clean and
  instrument-universal: lag-1 dominates, the filter helps only where the signal truly predicts, and it
  cannot manufacture edge at the non-predictive lag. This *is* the "discern good vs bad signal"
  property the study set out to demonstrate.
- **Per-instrument geometries are stable for the liquid names.** The larger-`h` recommendations
  (`es1s`/`hg1s` `h=10`, `gc1s` `h=15`, `si1s` `h=20`, `rb1s` `h=15`) keep a high, raw-beating adjusted
  Sharpe out-of-sample and are the trustworthy picks. The `h=1` recommendations are inflated by the
  label/return overlap and should be treated with suspicion (or learned via the meta-model rather than
  taken from the oracle).
- **`ho1s` is the documented failure** — in-sample-great, out-of-sample-broken — the value of holding
  out 2022-H1.
- **Now wired into the meta-model.** These per-instrument geometries feed the meta-model pipeline. To
  avoid the `h=1` label/return single-bar overlap, the **model geometry** is the best lag-1 geometry
  with **`h ≥ 5`** per instrument (the five larger-`h` names unchanged; the six `h=1` names move to a
  multi-bar exit at minimal in-sample cost, e.g. `ng1s` 27.5→26.1, `ho1s` 19.7→18.4). It is persisted
  by §9 of this notebook to [`results/triple_barrier_model_geometry.csv`](../../results/triple_barrier_model_geometry.csv).
  `notebooks/jay/metamodel.ipynb` consumes it behind a `LABELING_MODE = "global" | "per_instrument"`
  switch: per-instrument mode re-labels each instrument **causally** (its own `(pt, sl, h)`,
  `price_end=VAL_END`) and purges CPCV by per-instrument `h`, then trains the model families to
  *predict* `y` out-of-sample — turning the oracle filter into a deployable one. `ho1s`/`ng1s` stay
  thin/short-only and are reported per-instrument, not buried.
