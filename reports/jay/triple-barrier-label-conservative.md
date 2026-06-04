# Triple-Barrier Label Selection — **Conservative** Scheme, and an Honest Lenient-vs-Conservative Verdict

> Notebook: [`notebooks/jay/triple-barrier-label-conservative.ipynb`](../../notebooks/jay/triple-barrier-label-conservative.ipynb) (executed end-to-end)
> Reuses: [`src/stml/model/labels.py`](../../src/stml/model/labels.py) (`triple_barrier_labels` with `vertical_zero=True`, `load_instrument_geometry`), [`src/stml/model/dataset.py`](../../src/stml/model/dataset.py) (`events_frame`, σ = de-annualised `f2_vol_20`), [`src/stml/model/evaluate.py`](../../src/stml/model/evaluate.py) (`nav_sharpe`, `release_test`)
> Artifacts: [`results/triple_barrier_per_instrument_conservative.csv`](../../results/triple_barrier_per_instrument_conservative.csv) (conservative top-3 × floor-passing instruments × 4 PnL types) · [`results/triple_barrier_lenient_vs_conservative.csv`](../../results/triple_barrier_lenient_vs_conservative.csv) (per-instrument master comparison)
> Companion to (and direct contrast with): [`reports/jay/triple-barrier-label.md`](triple-barrier-label.md) — the **lenient** study this re-examines.
> **`data/triple_barrier_labels.csv` is intentionally NOT modified** — this is an analysis-only study.

---

## 0. What this study is — and why "conservative"

A stricter re-run of the model-free triple-barrier study. Same machinery (a geometry `(pt, sl, h)`
produces a binary label `y∈{0,1}`; the label **adjusts** the primary signal `adjusted = s·y`; we score by
the **Sharpe of the reconstructed adjusted-PnL** per instrument, *not* by AUC), but with a deliberately
**conservative** labeling envelope. The lenient study's **own §4 caveats** flag its picks as
oracle-inflated: 6 of 11 per-instrument top-1 geometries sit at `pt=sl=0.25, h=1`, and **all 11** use the
hair-trigger stop `sl=0.25`. Three named leniency sources motivate this pass:

1. **`pt` too small** — at `pt=0.25` the profit barrier is touched almost trivially.
2. **`h=1` too short** — a 1-bar label overlaps the *same bar* as the lag-1 return it is scored against ⇒
   the adjusted Sharpe is partly **circular** (self-fulfilling) in-sample.
3. **timeout = `sign(return)` too lenient** — a trade that merely drifts to the horizon and ends barely
   positive counts as a "win".

**The conservative scheme — one mechanism per leniency:**

| Lever | Lenient | **Conservative** | Fixes |
|---|---|---|---|
| `pt` (×σ) | `{0.25 … 2.5}` | **`{1, 1.5, 2, 2.5, 3}`** | leniency 1 |
| `sl` (×σ) | `{0.25 … 2.5}` | **`{0.5, 0.75, 1, 1.5, 2, 2.5}`** (drop `0.25`) | targets the tight-stop oracle bias (§4) |
| `h` (bars) | `{1,2,3,5,10,15,20}` | **`{5, 10, 15, 20}`** | leniency 2 — kills the h=1/lag-1 overlap |
| timeout rule | `sign(ret)` | **`vertical_zero=True`** ⇒ timeout→`0` | leniency 3 |

⇒ **5×6×4 = 120 geometries** per instrument. Under `vertical_zero=True` (`labels.py` line 131) a label is
`y=1` **iff the profit barrier is touched first** (PT-touch→1; SL-touch→0; timeout→0) — a strict,
economically meaningful "win". A **degeneracy floor** (`pos_rate ∈ [0.15, 0.85]` and `n_kept ≥ 15`,
in-sample) rejects geometries whose strict label is too one-sided / too thin to trust. Everything else
(per-instrument selection by lag-1 adjusted Sharpe, the 3-lag placebo, the 2022-H1 hold-out) is unchanged,
so the two studies are **directly comparable**.

> The headline going in was "does tightening the barriers deflate the suspiciously-high adjusted Sharpe?"
> The honest answer below is **no — and *why* not is the interesting result.**

---

## 1. Method

- **Universe / split.** 11 instruments; train = 2020-01-03 → 2022-01-01 (`DEV_PARTITIONS`,
  `price_end=2021-12-30`); hold-out = 2022-H1 (`test` partition via `release_test(final_confirmation=True)`,
  `price_end=2022-06-30`). σ = de-annualised `f2_vol_20`.
- **Scoring.** For every `(instrument × geometry × pnl_type)`: `raw = nav_sharpe(all trades)`,
  `adj = nav_sharpe(y==1 trades)`. `pnl_type ∈ {exit, lag0, lag1, lag2}`, where `lagL` scores the kept
  position by the single-bar return `g_i(L)=s_i·u_{t+L}` (lag-1 = first tradeable bar) and `exit` scores it
  by the realized first-touch return.
- **Selection.** Among **floor-passing** geometries, rank by in-sample **lag-1** `adj_sharpe`
  (deterministic `(h,pt,sl)`-ascending tie-break, matching `load_instrument_geometry`); take top-1 / top-3.
- **Comparison (apples-to-apples).** The lenient *geometry* is loaded from the lenient study CSV
  (`load_instrument_geometry(..., pnl_type="lag1")`); **every Sharpe is recomputed here** under the same
  scoring, so both arms sit on identical axes. We also run a **`vertical_zero` ablation** (lenient geometry,
  *only* the timeout rule flipped) to isolate the timeout-rule effect from the grid effect.

---

## 2. Placebo-in-time still holds

The geometry-free raw per-lag Sharpe (signal only, no label) is unchanged from the lenient study —
**lag-1 strongest for all 11 instruments**, lag-0 negative for 8/11 — confirming the signal predicts the
*next* bar ([`01-signal-direction.md`](../harry/01-signal-direction.md)). Crucially the **conservative**
label still respects this: the conservative in-sample top-1 adjusted Sharpe by lag (median over
floor-passing instruments) is

| lag | 0 (placebo) | 1 (tradeable) | 2 (decayed) | exit (circular) |
|---|---|---|---|---|
| median top-1 adj Sharpe | **−0.66** | **15.86** | 10.15 | 36.49 |

The filter **amplifies lag-1** and **cannot rescue lag-0** (stays negative) — evidence the strict label
encodes genuine *forward* timing structure, not a generic "keep the positive returns" trick. (The `exit`
column is the maximally-circular upper bound — see §4.)

---

## 3. In-sample selection & hold-out validation

**Per-instrument: lenient vs conservative** (lag-1; in-sample `adj_tr`, hold-out `adj_ho`; conservative
hold-out rank within the 120-grid; conservative trades taken on the hold-out `n_ho`):

| instrument | lenient (pt/sl/h) | **conservative (pt/sl/h)** | pos-rate L→C | adj_tr L→C | adj_ho L→C | cons ho-rank /120 | cons n_ho |
|---|---|---|---|---|---|---|---|
| cl1s   | 0.25/0.25/1  | **1.5/0.5/5**  | 0.59→0.34 | 10.5→9.9  | 17.8→17.2 | 4  | 38 |
| es1s   | 1.0/0.25/10  | **2.5/0.5/5**  | 0.43→0.19 | 13.1→13.0 | 24.4→22.5 | 2  | 20 |
| fesx1s | 0.25/0.25/1  | **2.5/0.5/5**  | 0.53→0.16 | 14.9→14.3 | 19.9→20.1 | 1  | 23 |
| gc1s   | 1.0/0.25/15  | **2.5/0.5/5**  | 0.48→0.19 | 15.3→15.6 | 18.3→23.8 | 21 | **5** |
| hg1s   | 1.0/0.25/10  | **2.0/0.5/5**  | 0.41→0.26 | 19.9→18.7 | 23.7→18.9 | 9  | 39 |
| ho1s   | 0.25/0.25/1  | **2.0/0.5/20** | 0.69→0.38 | 19.7→18.4 | **0.0→0.0** | 60 | **0** |
| ng1s   | 0.25/0.25/1  | **3.0/0.5/15** | 0.50→0.22 | 27.5→23.2 | 16.9→26.2 | 13 | **4** |
| nq1s   | 0.25/0.25/1  | **2.5/0.5/5**  | 0.54→0.20 | 15.4→15.9 | 24.4→26.4 | 4  | 24 |
| pl1s   | 0.25/0.25/1  | **2.0/0.5/5**  | 0.54→0.24 | 18.9→17.6 | 22.0→17.8 | 7  | 29 |
| rb1s   | 2.5/0.25/15  | **2.5/0.5/5**  | 0.24→0.17 | 10.9→10.1 | 17.0→18.9 | 4  | 32 |
| si1s   | 0.75/0.25/20 | **2.5/0.5/5**  | 0.41→0.15 | 16.2→15.9 | 28.1→28.1 | 1  | 23 |
| **median (floor-pass)** | | | **0.50→0.20** | **15.37→15.86** | **19.87→20.08** | | |

**Reading it:** the in-sample lag-1 adjusted Sharpe **barely moves** (15.37→15.86 median; per-name deltas
within ±1.5), and the hold-out is **a wash** (19.87→20.08). What changes sharply is **selectivity**
(positive rate halves, 0.50→0.20) and **the chosen geometry — for all 11 instruments**. Note the
conservative in-sample pick also tends to rank near the top *out-of-sample* (8/11 within rank 9/120) — the
geometry choice is reasonably stable — but `gc1s`/`ng1s` ride only **4–5** hold-out trades and `ho1s` rides
**zero**, so those OOS Sharpes are noise (see §4).

**The `vertical_zero` ablation** (lenient geometry, only the timeout rule flipped) does **not** deflate the
Sharpe — it *raises* it where it bites (median 15.37→**17.65**; +6.0 ng1s, +4.1 ho1s, +3.0 fesx1s; exactly
0 for the `h≥10` lenient names, whose tight `sl=0.25` almost never lets a trade time out). Demoting marginal
timeouts to `0` *improves* the oracle-kept set rather than shrinking the headline number.

---

## 4. Caveats (read before using the numbers)

- **The adjusted Sharpe is an oracle measure — intrinsic to the construction, not the geometry.** Filtering
  *realized* returns by *realized* labels is an upper bound on a learnable meta-model, not a backtest. That
  is *why* tightening `pt`/`sl`/`h` left the level near 15–20: the inflation is the oracle, not the lenient
  barriers. The hold-out + the lag-0 placebo are the only guards; the absolute levels are **not** achievable
  live.
- **The exit panel is maximally circular.** The label is *defined from* the exit return (`y=1 ⟺` a `+pt·σ`
  profit-touch), so the kept exit returns are all `≈ +pt·σ` by construction. Its Sharpe is the highest of
  all (median ~33–36; **ng1s 91**) and is **mechanical, not edge** — we report it only as the clearest
  illustration of the oracle point.
- **The oracle's boundary-seeking behaviour survives conservatism.** The search still pins to the *tightest
  allowed* stop (**11/11 pick `sl=0.5`**) and the *shortest allowed* horizon (**8/11 pick `h=5`**) — the
  same behaviour that gave `sl=0.25`, `h=1` under the lenient grid. Removing the hair-trigger and the 1-bar
  horizon **moved the boundary, not the preference.** The lenient report's *"tight stop dominates the
  oracle"* caveat is therefore **structural** — only `pt` genuinely entered the interior (1.5–3.0, median
  2.5).
- **Thin names still fail.** `ho1s` (~50 events) clears the in-sample floor but takes **0 trades on the
  hold-out** (raw hold-out Sharpe −31); `gc1s`/`ng1s` ride only 4–5. Surfaced, not hidden — exactly what the
  floor and hold-out exist to expose.

---

## 5. Conclusions & how to use this

1. **Tightening did not deflate the in-sample Sharpe (≈15→16), and the hold-out is a wash (≈20).** The
   suspiciously-high adjusted Sharpe of the lenient study was **not primarily a labeling artifact of the
   lenient *geometry*** — it is intrinsic to the oracle/adjusted-signal construction. This is the central,
   slightly humbling finding.
2. **What conservatism genuinely buys is a cleaner *mechanism*, not a better *number*.** At `h ≥ 5` the
   scored lag-1 bar is no longer the label bar, so the ~16 Sharpe is earned **without** the h=1 single-bar
   circularity — and it lands at the same level the lenient pick reached *with* that circularity. The strict
   label is also far more selective (positive rate halves) and encodes a stronger economic prior (a real
   `+pt·σ` move, not a drift).
3. **Recommendation for the meta-model.** Prefer the **conservative per-instrument geometry as the labeling
   target** — not because it scores higher (it does not) but because it is *less circular*, *more
   selective*, and *more economically meaningful*. Use the 10 floor-passing geometries (`cl1s 1.5/0.5/5`,
   `es1s 2.5/0.5/5`, `fesx1s 2.5/0.5/5`, `gc1s 2.5/0.5/5`, `hg1s 2.0/0.5/5`, `ng1s 3.0/0.5/15`,
   `nq1s 2.5/0.5/5`, `pl1s 2.0/0.5/5`, `rb1s 2.5/0.5/5`, `si1s 2.5/0.5/5`); **exclude `ho1s`** (structurally
   too thin under any strict label).
4. **The only honest test remains downstream.** Both the lenient and conservative adjusted Sharpes are
   oracle upper bounds; neither is deployable. The real question is whether a meta-model can **predict** the
   (now cleaner) conservative `y` out-of-sample, using out-of-fold probabilities (CLAUDE.md §6). This study
   supplies the *target*; the meta-model layer supplies the test.
