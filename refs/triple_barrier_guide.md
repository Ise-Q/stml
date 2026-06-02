# The Triple-Barrier Method: A Practical Guide

*Labeling financial observations the way a trader actually experiences them — with profit targets, stop-losses, and holding-period limits.*

---

## 1. Why we need it

The naive way to label a financial observation is the **fixed-time horizon** method: look at the return over the next `h` bars and label it `+1`, `0`, or `-1` against some fixed threshold `τ`. This is what most of the ML literature does, and it is almost always wrong for trading:

- **The threshold ignores volatility.** A `τ = 1%` threshold means very different things during a quiet overnight session (`σ ≈ 0.01%`) versus the market open (`σ ≈ 1%`). You end up labeling almost everything `0` even when returns were predictable and statistically significant.
- **It ignores the path.** Every real strategy has a stop-loss. A position that ends the horizon at `+2%` but dipped to `-10%` along the way would have been stopped out in reality. Labeling it `+1` trains your model on trades you could never have held.
- **Time bars have poor statistical properties** (heteroscedastic, serially correlated returns).

The triple-barrier method fixes all three by labeling each observation according to **which of three barriers is touched first**.

---

## 2. The three barriers

For each observation starting at time `t_{i,0}`, we set:

| Barrier | Type | Label if touched first |
|---|---|---|
| Upper (profit-taking) | horizontal, `+pt · σ` | `+1` |
| Lower (stop-loss) | horizontal, `−sl · σ` | `−1` |
| Vertical (expiration) | time, after `h` bars | `0` or `sign(return)` |

The two horizontal barriers are a **dynamic function of estimated volatility** `σ` (so they self-adjust to market conditions), and the vertical barrier is a maximum holding period. The label is determined by the **first** barrier the price path touches.

```
price
  │        ┌─────────────────┐  ← upper barrier (+pt·σ)  → label +1
  │     ╱╲╱                  │
  │  ──╱──────╲──────────────│  ← entry price
  │           ╲╱╲            │
  │              ╲___________│  ← lower barrier (−sl·σ)  → label −1
  └────────────────────────────→ time
   t_{i,0}              t_{i,0}+h  ← vertical barrier      → label 0 / sign(ret)
```

Key properties:

- **Path-dependent.** To label an observation you must examine the *entire* price path over `[t_{i,0}, t_{i,0}+h]`, not just the endpoints.
- **Asymmetric allowed.** `pt` and `sl` need not be equal (e.g. `ptSl=[1, 2]` = take profit at `1σ`, stop out at `2σ`).
- **Barriers can be disabled.** Setting a factor to `0` removes that horizontal barrier. `ptSl=[0, 2]` is pure stop-loss (useful for mean-reversion strategies).
- **Caveat.** Barrier crossing is a discrete event sampled at bar boundaries, so it can miss a transition by a small margin. Finer bars reduce this.

---

## 3. The full pipeline

The method is the last step in a chain. The standard López de Prado workflow is:

```
prices ──▶ CUSUM filter ──▶ event timestamps (tEvents)
                                    │
daily volatility (getDailyVol) ─────┤──▶ getEvents() ──▶ first-touch times
                                    │         │
vertical barrier (numDays) ─────────┘         ▼
                                          getBins() ──▶ labels {−1,0,1}
```

We'll build each piece.

---

## 4. Implementation

These are clean, Python-3, well-commented versions of the canonical functions. They keep the original logic but fix the Python-2-isms (`xrange`, `.iteritems()`) and add docstrings.

### 4.1 Setup and a synthetic price series

```python
import numpy as np
import pandas as pd

# Reproducible synthetic minute-bar price series so the guide runs standalone.
rng = np.random.default_rng(42)
n = 50_000
idx = pd.date_range("2023-01-01", periods=n, freq="min")
returns = rng.normal(0, 0.0005, n)
close = pd.Series(100 * np.exp(np.cumsum(returns)), index=idx)
```

### 4.2 Dynamic volatility target (Snippet 3.1)

The barrier widths scale with this. It is a daily-horizon EWMA of returns, reindexed onto the price index.

```python
def get_daily_vol(close, span0=100):
    """Exponentially-weighted daily volatility, reindexed to `close`.

    For each timestamp, find the price ~1 day earlier, compute the return,
    then take an EWMA standard deviation of those returns.
    """
    # locate the bar one day before each timestamp
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(
        close.index[df0 - 1],
        index=close.index[close.shape[0] - df0.shape[0]:],
    )
    # daily returns
    ret = close.loc[df0.index] / close.loc[df0.values].values - 1
    # EWMA std of returns
    return ret.ewm(span=span0).std()
```

### 4.3 Event sampling — the CUSUM filter (Snippet 2.4)

We don't label *every* bar; we label bars where something "happened." The symmetric CUSUM filter samples a bar whenever cumulative up- or down-moves exceed a threshold `h`, then resets. This avoids the Bollinger-band flaw of firing repeatedly while hovering at a level.

```python
def get_t_events(g_raw, h):
    """Symmetric CUSUM filter. Returns timestamps where cumulative move > h."""
    t_events, s_pos, s_neg = [], 0.0, 0.0
    diff = g_raw.diff().dropna()
    for i in diff.index:
        s_pos = max(0.0, s_pos + diff.loc[i])
        s_neg = min(0.0, s_neg + diff.loc[i])
        if s_neg < -h:
            s_neg = 0.0
            t_events.append(i)
        elif s_pos > h:
            s_pos = 0.0
            t_events.append(i)
    return pd.DatetimeIndex(t_events)
```

### 4.4 The core — first barrier touch (Snippet 3.2)

This is the heart of the method. For each event it walks the price path and records when (if ever) the profit-taking and stop-loss barriers are first hit.

```python
def apply_pt_sl_on_t1(close, events, pt_sl, molecule):
    """For each event, find the first time the PT or SL barrier is touched.

    Parameters
    ----------
    close : pd.Series of prices.
    events : DataFrame with columns:
        't1'   : vertical-barrier timestamp (NaT = none),
        'trgt' : unit width of the horizontal barriers (the volatility target),
        'side' : position side (+1 long / -1 short).
    pt_sl : [pt_factor, sl_factor]; multiplies 'trgt'. 0 disables that barrier.
    molecule : subset of event indices to process (for parallelism).
    """
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)

    # profit-taking barrier (upper)
    pt = pt_sl[0] * events_['trgt'] if pt_sl[0] > 0 else pd.Series(index=events.index, dtype=float)
    # stop-loss barrier (lower, negative)
    sl = -pt_sl[1] * events_['trgt'] if pt_sl[1] > 0 else pd.Series(index=events.index, dtype=float)

    for loc, t1 in events_['t1'].fillna(close.index[-1]).items():
        path = close[loc:t1]                                  # price path in window
        path_ret = (path / close[loc] - 1) * events_.at[loc, 'side']  # side-adjusted returns
        out.loc[loc, 'sl'] = path_ret[path_ret < sl[loc]].index.min()  # first SL touch
        out.loc[loc, 'pt'] = path_ret[path_ret > pt[loc]].index.min()  # first PT touch
    return out
```

### 4.5 Orchestration — `get_events` (Snippets 3.3 & 3.6)

Ties it together and computes the **first** touch across all three barriers. Includes the `side` argument for meta-labeling (Section 6).

```python
def get_events(close, t_events, pt_sl, trgt, min_ret, t1=False, side=None):
    """Find the time of the first barrier touch for each event.

    pt_sl : if side is None -> scalar (symmetric barriers, learning side+size);
            if side is given -> [pt, sl] (asymmetric allowed, meta-labeling).
    min_ret : minimum target return to bother running a triple barrier.
    t1 : Series of vertical-barrier timestamps, or False to disable.
    """
    # 1) target, filtered by minimum return
    trgt = trgt.loc[trgt.index.intersection(t_events)]
    trgt = trgt[trgt > min_ret]

    # 2) vertical barrier
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=t_events)

    # 3) assemble events object
    if side is None:
        side_ = pd.Series(1.0, index=trgt.index)
        pt_sl_ = [pt_sl, pt_sl]
    else:
        side_ = side.loc[trgt.index]
        pt_sl_ = pt_sl[:2]

    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])

    # 4) find first touch (single-threaded; swap for mpPandasObj to parallelize)
    df0 = apply_pt_sl_on_t1(close, events, pt_sl_, events.index)
    events['t1'] = df0.dropna(how='all').min(axis=1)   # earliest of {pt, sl, t1}

    if side is None:
        events = events.drop('side', axis=1)
    return events
```

### 4.6 Vertical barrier helper (Snippet 3.4)

```python
def add_vertical_barrier(close, t_events, num_days=1):
    """Timestamp `num_days` after each event (the expiration / time-out barrier)."""
    t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
    t1 = t1[t1 < close.shape[0]]
    return pd.Series(close.index[t1], index=t_events[:t1.shape[0]])
```

### 4.7 Producing the labels — `get_bins` (Snippets 3.5 & 3.7)

```python
def get_bins(events, close):
    """Compute each event's realized return and label.

    Case 1 ('side' absent): bin in {-1, 0, 1}, labeled by sign of return.
    Case 2 ('side' present): bin in {0, 1}, labeled by P&L (meta-labeling).
    """
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')

    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index].values - 1
    if 'side' in events_:
        out['ret'] *= events_['side']          # side-adjusted P&L
    out['bin'] = np.sign(out['ret'])
    if 'side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0     # meta-label: take the bet or pass
    return out
```

> **Tip (Exercise 3.3):** to label vertical-barrier touches as `0` instead of `sign(ret)`, after computing `out['bin']` set `out.loc[t1_touched, 'bin'] = 0`, where `t1_touched` is the set of events whose first touch equals their vertical barrier. Whether `0` or `sign(ret)` works better is empirical — test both.

### 4.8 Running the full pipeline

```python
# 1) volatility target
vol = get_daily_vol(close, span0=100)

# 2) sample events with CUSUM (threshold = mean daily vol is a reasonable start)
t_events = get_t_events(close, h=vol.mean())

# 3) vertical barriers, 1 day out
t1 = add_vertical_barrier(close, t_events, num_days=1)

# 4) first-touch times with symmetric 1σ barriers
events = get_events(close, t_events, pt_sl=1.0, trgt=vol, min_ret=0.0, t1=t1)

# 5) labels
labels = get_bins(events, close)
print(labels['bin'].value_counts())
```

### 4.9 Drop rare labels (Snippet 3.8)

Some classes can be too rare for sklearn to handle well. Drop labels below a frequency floor, iteratively:

```python
def drop_labels(events, min_pct=0.05):
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > min_pct or df0.shape[0] < 3:
            break
        events = events[events['bin'] != df0.idxmin()]
    return events
```

---

## 5. The parameters, and what each one does

| Parameter | Where | Effect | Trade-off |
|---|---|---|---|
| `pt`, `sl` (`ptSl`) | `get_events` | Width of profit / loss barriers in units of σ | Wide → more `0` (time-outs), fewer but cleaner directional labels; narrow → noisier, more frequent touches |
| `h` / `num_days` | vertical barrier | Max holding period | Longer → fewer time-outs, more path risk, more label leakage between overlapping events |
| `span0` | `get_daily_vol` | Responsiveness of the σ estimate | Short → reactive but jittery barriers; long → smooth but stale |
| `h` (CUSUM threshold) | `get_t_events` | How selective sampling is | High → fewer, more "significant" events; low → more events, more noise |
| `min_ret` | `get_events` | Skip events with tiny targets | Filters low-information observations |

---

## 6. Meta-labeling (the most important use)

This is where the method earns its keep. Suppose you already have a **primary model** that decides the *side* (long/short) — a moving-average crossover, a Bollinger-band rule, or another ML model. You don't want a second model to re-learn the side; you want it to learn the **size**, including the option of *no bet*.

1. Run the primary model to get `side ∈ {−1, +1}` for each event.
2. Apply the triple barrier **with that side** (`get_events(..., side=side, pt_sl=[pt, sl])`). Because the side is known, barriers can be asymmetric and `get_bins` labels each trade `1` (would have hit PT before SL → worth taking) or `0` (not worth taking).
3. Train a **secondary (meta) model** on features to predict `{0, 1}` — purely whether to act on the primary signal.
4. Use the meta-model's predicted probability to **size** the bet; the side stays fixed by the primary model.

The payoff: meta-labeling improves the **F1 score** by raising precision (filtering out false positives) without forcing the primary model to sacrifice recall. It's a clean separation of "which way" from "how much / whether at all."

```python
# Sketch: meta-labeling with a primary side signal
side = primary_model_side(close, t_events)          # your rule/model -> {-1,+1}
events = get_events(close, t_events, pt_sl=[1, 2],   # asymmetric: PT 1σ, SL 2σ
                    trgt=vol, min_ret=0.0, t1=t1, side=side)
meta_labels = get_bins(events, close)                # bin in {0, 1}
# X = your features aligned to events.index;  y = meta_labels['bin']
# clf = RandomForestClassifier(...).fit(X, y)
```

---

## 7. Optimizing the parameters

There is no single "correct" `ptSl` — it depends on the instrument, the strategy, and your costs. Treat barrier configuration as a hyperparameter search, but with **finance-specific guardrails** so you don't fool yourself.

### 7.1 Define the right objective
Do **not** optimize for classification accuracy. A labeler that produces 95% `0`-labels scores high accuracy and is useless. Optimize for a downstream, economically meaningful metric:

- For the **labels themselves**: class balance + how well a baseline model trained on them generalizes.
- For the **strategy**: out-of-sample Sharpe ratio (or, better, the **Deflated Sharpe Ratio**, which corrects for the number of configurations you tried — directly relevant because grid search inflates the best in-sample Sharpe).

### 7.2 Grid search the barrier geometry
A practical grid:

```python
from itertools import product

pt_grid   = [0.5, 1.0, 1.5, 2.0, 3.0]
sl_grid   = [0.5, 1.0, 1.5, 2.0, 3.0]
hold_grid = [1, 2, 5, 10]          # num_days for the vertical barrier
span_grid = [50, 100, 200]         # vol estimator span

results = []
for pt, sl, days, span in product(pt_grid, sl_grid, hold_grid, span_grid):
    vol_s = get_daily_vol(close, span0=span)
    t1_s  = add_vertical_barrier(close, t_events, num_days=days)
    ev    = get_events(close, t_events, pt_sl=[pt, sl], trgt=vol_s,
                       min_ret=0.0, t1=t1_s, side=side)   # use side for meta-labeling
    lab   = get_bins(ev, close)
    # score = your purged-CV out-of-sample metric on a model trained with `lab`
    results.append((pt, sl, days, span, score(lab)))
```

### 7.3 Validate with finance-aware cross-validation (critical)
Triple-barrier labels **overlap in time** — event `i`'s window can extend into event `j`'s. Standard k-fold CV leaks information across that overlap and will massively overstate performance. Use:

- **Purging:** drop training observations whose label windows overlap the test set.
- **Embargoing:** additionally drop training observations immediately *after* the test set to block serial-correlation leakage.
- **Combinatorial Purged Cross-Validation (CPCV):** generates many backtest paths so your parameter choice isn't a function of one arbitrary train/test split.

(These are Chapters 7 and 12 in your course — they are not optional add-ons; barrier optimization without them is how people overfit.)

### 7.4 Guard against backtest overfitting
- The more `(pt, sl, h, span)` combinations you try, the more likely the best one is luck. Track the number of trials and **deflate** your Sharpe accordingly.
- Sanity-check the winner on **synthetic data** and on a truly held-out period.
- Prefer **robust plateaus** over sharp peaks: pick a configuration surrounded by other good configurations, not an isolated spike — a spike usually means overfit.

### 7.5 Sensible starting points
- **Symmetric, learning side + size:** `ptSl=[1, 1]`, `num_days=1`.
- **Trend-following meta-label:** `ptSl=[1, 2]` (let winners run a bit, cut losers tighter), `num_days=1`.
- **Mean-reversion meta-label:** `ptSl=[0, 2]` (no upper barrier, pure stop-loss), `num_days=1`.

---

## 8. Common pitfalls checklist

- [ ] **Volatility-scaled barriers**, not fixed thresholds.
- [ ] **Volume/dollar bars** rather than time bars where possible (better homoscedasticity).
- [ ] **Event-based sampling** (CUSUM) instead of labeling every bar.
- [ ] **Purged + embargoed CV** whenever you train or tune on these labels.
- [ ] **Don't optimize accuracy** — optimize a deflated, economically meaningful metric.
- [ ] **Account for overlapping labels** with sample-uniqueness weighting (Chapter 4) when training.
- [ ] **Watch the discrete-crossing caveat** — finer bars reduce missed transitions.

---

*References: M. López de Prado, "Advances in Financial Machine Learning" (Wiley, 2018), Ch. 2–4, 7, 12, 15; ICBS Lecture 1 (H. Madmoun).*
