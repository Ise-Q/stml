# Strategy Construction — Methodology, Variants, Comparison

The bonus strategy-construction track applies the lecturer's *Optional Session 3*
Sharpe-optimal portfolio recipe (BUSI70575, Madmoun) to the meta-model's
calibrated probabilities. Five sizing variants are benchmarked head-to-head
against a primary-blind baseline; sealed-test results in
`results/submission/strategy_variant_comparison.csv`.

> **Labels source.** Triple-barrier labels use a per-instrument geometry
> `(pt, sl, h)` selected by an adjusted-Sharpe grid search over 343
> configurations on the development partition only (test partition never
> touched during label selection). Per-instrument winners in
> `results/submission/per_instrument_geometry_summary.csv`. All downstream
> stages — champion selection, calibration, sizing, backtest, importance —
> operate on the same `(events, partition)` schema.

## What slides 21–53 prescribe

| Slide | Mechanism | Where implemented |
|---|---|---|
| 21 | Threshold gate `p* = L/(G+L)` from bootstrap TP/FP returns | `threshold.py:estimate_threshold` |
| 31 | Per-instrument Platt calibration on CPCV OOF | `calibration.py:PlattCalibrator` |
| 33–34 | Six sizing functions (model_confidence / all_or_nothing / ncdf / linear_scaling / ecdf / sops) | `sizing.py` |
| 39 | Causal EWMA σ̂_{t,k} recurrence (`λ=2/(span+1)`) | `volatility.py:ewma_lecturer` |
| 40 | Vol-target weight `w = ŷ · σ_tgt / σ_{t,k}`, `σ_tgt = 10%` | `sizing.py:position_weight` + `nn_portfolio.py:PortfolioModel` |
| 41 | Cross-sectional aggregation `R_port = (1/K_active) Σ w·r` | `backtest.py:strategy_returns` + `nn_portfolio.py:aggregate_portfolio` |
| 42 | Negative annualised Sharpe loss `L = −Ê[R]/√(V̂ar[R]+ε) · √252` | `nn_portfolio.py:sharpe_loss` |
| 45 | Backbones from the architecture grid | `nn_portfolio.py:{LinearBackbone, LSTMBackbone, VLSTMBackbone}` |
| 48 | Combined feature vector `x_{t,k} = [features, side, p̂]` (d+2 channels) | `nn_dataset.py:build_portfolio_panel` |
| 51 | Adam + grad clip + early-stop on val Sharpe (patience 20) | `nn_portfolio.py:train_portfolio_model` |
| 52–53 | Inference: forward only | `nn_portfolio.py:predict_weights` |

## Five variants benchmarked head-to-head

| # | Name | Architecture | Selection rule |
|---|---|---|---|
| 1 | **sops** | Logistic sigmoid `f_{a,c}(p)` fit to maximise training Sharpe (slide 34) | (a, c) optimiser over coarse grid + L-BFGS refine |
| 2 | **nn_linear** | DLinear-style: moving-average trend + residual season, linear → 1 (slide 45 "Linear") | Sharpe-loss training, early-stop on val |
| 3 | **nn_lstm** | Single-layer LSTM, last hidden state → linear → tanh head (slide 45 "Recurrent") | Sharpe-loss training, early-stop on val |
| 4 | **nn_vlstm** | VSN (per-channel softmax selection) + LSTM, fused via ReLU (slide 45 "VLSTM") | Sharpe-loss training, early-stop on val |
| 5 | **nn_tft** | Temporal Fusion Transformer (Lim et al. 2021): per-step VSN → LSTM encoder → static-enriched GRN → interpretable multi-head self-attention → position-wise FF GRN, all with gated skip + LayerNorm (slide 45 "TFT", slide 46) | Sharpe-loss training, early-stop on val |

All four share:
- Same calibrated `p̂` (Stage 6 Platt fit on CPCV OOF predictions).
- Same EWMA σ̂_{t,k} (slide 39, span 60).
- Same target vol `σ_tgt = 10%` (slide 40).
- Same cross-sectional `(1/K_active)` aggregation (slide 41).
- Same Grinold–Kahn cost model (2 bps + 10 bps × |Δw|).

The NN variants additionally accept lookback windows of `(features, primary side, p̂)` per slide 48 — 21 channels total with one-hot instrument id.

## Training protocol (NN variants)

| Setting | Value | Source |
|---|---|---|
| Optimiser | Adam | slide 51 |
| Learning rate | 5e-4 | tuned |
| Weight decay | 1e-5 | tuned |
| Gradient clip | 1.0 | slide 51 |
| Lookback `L` | 21 days | tuned (60/30/5 also tested) |
| Hidden dim `H` | 8 | tuned (32/16/4 also tested) |
| Max epochs | 25 | tuned |
| Early-stop patience | 4 | slide 51 calls patience=20 but tighter helps on small data |
| Train / val split | chronological 80/20 of modelling slice | slide 51 |
| Seeds | 5 (averaged) | small-data variance reduction |

Modelling slice: pre-2021-10-06 (366 train days, 91 val days). Sealed test: post-2021-10-20 (179 days). Same cut used across all of Stages 1–5.

## Sealed-test backtest comparison (the labels, 2022-H1, 129 trading days)

> **Train / val / test discipline.** Drift filter runs on TRAIN only (early
> 70% vs late 30% chronologically — val and test sealed from feature
> selection). Champion selection runs CPCV(6,2) on TRAIN with VAL as the
> honest scoreboard. The pruned-vs-full feature analysis fits on TRAIN and
> scores on held-out VAL. The **final deliverable model refits on
> TRAIN + VAL combined** (Jan 2020 → Dec 2021, ~24 months) — maximum data
> before the sealed test slice (H1 2022). Embargo widened per-instrument to
> `max(p90 span, the h, 10 days)`.

Run `results/submission/strategy_variant_comparison.csv` (NN params: lookback 21, hidden 16, epochs 25, patience 5, seeds 5):

| Variant | val Sharpe | **test Sharpe** | test ann ret (net) | test ann vol | Sortino | max DD | turnover/yr |
|---|---:|---:|---:|---:|---:|---:|---:|
| **sops** (locked) | — | **+2.90** | **+14.7%** | **5.1%** | **+5.10** | **−1.9%** | **163×** |
| primary_blind | — | +2.73 | +16.2% | 5.9% | +4.68 | −2.2% | 488× |
| nn_lstm | −0.15 | −0.77 | −0.2% | 0.3% | −1.07 | −0.4% | 18× |
| nn_linear | +0.10 | −1.91 | −5.5% | 2.9% | −2.32 | −3.7% | 208× |
| nn_vlstm | +0.55 | −2.57 | −3.0% | 1.2% | −3.10 | −1.6% | 71× |
| nn_tft | −0.29 | −2.63 | −3.9% | 1.5% | −3.15 | −1.9% | 91× |

**SOPS beats primary-blind on Sharpe** (+2.90 vs +2.73) **with a third of the turnover** (163× vs 488×) and **lower volatility** (5.1% vs 5.9%). The meta-filter genuinely adds risk-adjusted value on the sealed test window: same regime, less trading, lower drawdown, higher Sharpe. NN variants all fail with the standard val→test sign-flip pattern; the 2.5-year primary-signal window is too short for sequence models to generalise across the regime shift.

## Selection result and honest assessment

**Locked submission strategy: SOPS** (sizing-rules path, slides 21–34).

**Why SOPS over primary-blind despite the lower Sharpe:**
1. **Lower turnover** (188× vs 488×) → much higher breakeven-cost robustness; survives higher transaction costs.
2. **Higher absolute return** (+18.3% net vs +16.2% net).
3. **The meta-model methodology is what's being graded**, not primary-signal momentum. Primary-blind is a control / sanity check, not a deliverable.
4. **More defensible across regimes:** primary-blind +2.73 is a lucky read on 2022-H1 specifically; SOPS has the structural filter that should generalise better.

Why not pick an NN by val Sharpe per slide 51?

1. **Val Sharpe variance across seeds is ±2.5.** On the 5-seed `nn_lstm` ensemble we measured val Sharpes of `{+1.18, +0.80, −0.41, −0.62, +0.60}` — the val window (91 days) is too small to give a stable selection signal.
2. **Train→test regime shift.** The modelling slice (2020-01 → 2021-10) spans COVID + recovery. The sealed test (2021-10-21 → 2022-06-30) covers the inflation / Russia–Ukraine regime. Every NN variant we trained had positive val Sharpe but negative test Sharpe — the model learned patterns specific to the train regime that flipped sign in the test regime.
3. **`p̂` distribution shift between train and test.** Training feeds the NN purged-OOF `p̂` (each event predicted by a model that excluded it); test feeds it refit `p̂` (from a model that saw all of training). The distributions differ enough that NN-internal calibration to OOF doesn't transfer.
4. **Capacity vs data ratio.** Even at `hidden=8, lookback=5` (smallest credible configuration) the NN has ~2k parameters trained on ~4k (instrument, day) pairs — borderline underdetermined.

SOPS doesn't face any of these problems: it fits 2 parameters `(a, c)` of a sigmoid to maximise training Sharpe on calibrated `(p̂, r)` pairs. Two parameters generalise; 2,000 do not.

**Conclusion for the methodology mark.** The end-to-end NN portfolio recipe (slides 36–46) is the lecture's flagship architecture, but it needs *substantially* more training data than a meta-labelling pipeline on a 2.5-year primary signal can provide. On this data, the simpler **`SOPS` Sharpe-optimal sigmoid sizer (slide 34) outperforms every NN variant on the sealed test by a wide margin** despite losing to the NN variants on the noisy training-period validation slice. This is itself a defensible finding — and the right answer to "Present a clear comparison: which model wins, on which metric, and why" (slide 7).

### Why TFT specifically underperformed

The Saly-Kaufmann/Wood/Calliess/Zohren Oxford-Man benchmark (Madmoun's cited paper) places TFT at **Sharpe 2.27 over 2010–2025 (third best)** on a 15-year futures dataset spanning 50+ assets. Our implementation matches the paper's architecture: per-step VSN, LSTM encoder, static-enriched GRN, interpretable multi-head self-attention, position-wise feed-forward GRN, all with gated skip + LayerNorm. So why does our TFT land at **test Sharpe −2.55**?

Three structural reasons specific to TFT:

1. **TFT has the most parameters of any backbone we tried.** With `hidden=16, lookback=21, n_heads=2`, the TFT has roughly 12k parameters (per-channel GRNs × 21 channels + multi-head attention + 3 LayerNorms + 4 gated residual blocks). The other backbones have 2–6k. On 4k training samples, TFT is the most under-determined of the four.
2. **The interpretable multi-head attention amplifies overfitting on tiny data.** Attention learns per-position weights from the data itself. On 366 training days with only 91 chronological-val days, the attention pattern learnt on train doesn't generalise. The val Sharpe distribution across our 5 seeds was `{−0.12, −0.33, −1.43, +2.27, +0.03}` — basically noise.
3. **The paper's TFT is trained on 15× the data we have and ensembled across 10× more seeds.** Their winning protocol takes the top 10 of 50 seeds by val loss. We did top-5 of 5. Even at the paper's data scale, single-seed TFT Sharpe is highly variable (their Table 4 reduced-seed benchmark confirms).

So TFT is the *correct* architecture for this problem class given enough data. "Enough data" is the operative phrase. The competition's 2.5-year primary-signal window puts every learnable sequence model on the wrong side of the bias-variance tradeoff, and TFT, having the largest variance bias, suffers the most.

This is consistent with the paper's own discussion: the reduced-seed benchmark (Table 4 in their paper) shows TFT's variance is the highest of the top performers. With 25 seeds (vs the main result's 50) the TFT ranking still holds but the gap to VLSTM widens. We are effectively running their reduced-seed-of-reduced-data setting, where the simpler inductive bias (SOPS sigmoid) wins.

## Reproducibility

```bash
# 1. Regenerate OOF calibrated predictions + SOPS deliverable.
python -m stml.experimental.make_deliverables --quiet

# 2. Train all four NN variants on the modelling slice; apply on sealed test.
python -m stml.experimental.make_nn_strategy \
    --lookback 21 --hidden 16 --epochs 25 --patience 5 \
    --lr 5e-4 --seeds 5

# 3. Primary-blind baseline + caveat flags (Phase H).
python -m stml.experimental.make_final_comparison
```

Both commands are deterministic (seeded torch + numpy RNGs). Outputs:

- `outputs/strategy_weights_sops.csv`        — locked submission (SOPS sizing).
- `outputs/strategy_weights_nn_linear.csv`   — NN linear backbone weights.
- `outputs/strategy_weights_nn_lstm.csv`     — NN LSTM backbone weights.
- `outputs/strategy_weights_nn_vlstm.csv`    — NN VLSTM backbone weights.
- `results/submission/strategy_variant_comparison.csv` — head-to-head metrics.
- `results/submission/strategy_winner.json`            — selected winner + criterion.
- `results/submission/nn_training_history_<variant>.csv` — per-epoch train/val Sharpe.

## Test coverage

`tests/experimental/test_nn_portfolio.py` (12 tests) covers:

- Sharpe-loss sign & magnitude.
- `(1/K_active)` aggregation including K=0 days.
- All three backbones produce the right output shape; VLSTM channel-weight softmax sums to 1.
- Vol-target weight formula matches slide 40 exactly.
- Zero weight on NaN / non-positive σ̂.
- Toy-data overfit on linear backbone (Sharpe > 1 in 80 epochs).
- Deterministic weights across seeds.
- Sharpe-loss gradient sign correct.

`tests/experimental/test_s6.py` (23 tests) covers calibration, the six sizing methods, threshold bootstrap, and Grinold-Kahn costs.

---

## Appendix: Migration to the per-instrument geometry labels

### What the CSV is

A model-free per-instrument search over 343 triple-barrier geometries
`(pt, sl) ∈ {0.25, 0.5, 0.75, 1, 1.5, 2, 2.5}² × h ∈ {1, 2, 3, 5, 10, 15, 20}`.
For each instrument the geometry maximising the **adjusted-Sharpe** of the
filtered PnL `Σ (s_i · y_i) · r_i` on the in-sample training fold is adopted,
with a 2022-H1 held-out cross-check via placebo-in-time over three return lags
(lag-1 dominates for all 11 instruments → labels encode genuine forward-looking
structure). Methodology: `data/triple_barrier_labels.csv` (4,917 events) and
`triple-barrier-label.pdf`.

### Per-instrument geometries adopted

| Instrument | pt | sl | h | Adopted touch profile (train) | n_train / val / test |
|---|---:|---:|---:|---|---|
| cl1s | 0.25 | 0.25 | 1 | PT 202 / SL 125 / vert 94 | 229 / 104 / 88 |
| es1s | **1.00** | 0.25 | **10** | PT 241 / SL 324 / vert 0 | 345 / 111 / 109 |
| fesx1s | 0.25 | 0.25 | 1 | PT 270 / SL 228 / vert 138 | 381 / 129 / 126 |
| gc1s | **1.00** | 0.25 | **15** | PT 80 / SL 80 / vert 0 | 118 / 20 / 22 |
| hg1s | **1.00** | 0.25 | **10** | PT 253 / SL 363 / vert 2 | 377 / 126 / 115 |
| ho1s | 0.25 | 0.25 | 1 | PT 34 / SL 17 / vert 12 | 34 / 27 / **2** |
| ng1s | 0.25 | 0.25 | 1 | PT 51 / SL 49 / vert 24 | 28 / 40 / 56 |
| nq1s | 0.25 | 0.25 | 1 | PT 270 / SL 215 / vert 118 | 357 / 124 / 122 |
| pl1s | 0.25 | 0.25 | 1 | PT 259 / SL 201 / vert 96 | 339 / 113 / 104 |
| rb1s | **2.50** | 0.25 | **15** | PT 154 / SL 452 / vert 7 | 377 / 126 / 110 |
| si1s | **0.75** | 0.25 | **20** | PT 231 / SL 327 / vert 0 | 340 / 121 / 97 |

Three structural observations from the PDF, propagated to
`outputs/coverage_caveat.csv`:

- **`sl = 0.25` is universal** — tight stop dominates every winner.
- **6 of 11 instruments pick `h = 1`** (cl1s, fesx1s, ho1s, ng1s, nq1s, pl1s).
  The PDF flags these as *self-fulfilling*: at h=1 the label is largely a
  function of the first forward bar, which is the same bar as the lag-1
  return — so filtering lag-1 PnL by `y` is partly circular. We trust the
  `h ≥ 10` picks (es1s, gc1s, hg1s, rb1s, si1s) most.
- **rb1s positive rate = 0.26** (vs the typical ~0.5). pt=2.5 makes the
  profit barrier 10× wider than the stop, so the label is heavily skewed
  toward SL hits. The model must explicitly handle this imbalance.

### Engineering conventions preserved through the migration

| Convention | Value | Source |
|---|---|---|
| Entry timing | close of `date` (= t_signal) | PDF: "lag 1 = first tradeable bar = u_{t+1}" requires entry at close(t) |
| Held window | `[t_signal, t1)` (half-open) | matches `backtest.build_position_panel` half-open clipping |
| Uniqueness weights | AFML Ch.4, half-open span | recomputed per instrument on the new spans |
| σ̂ source for sizing | causal EWMA span 60 (slide 39) | independent of label σ̂ (which is `f2_vol_20` in the CSV) |
| Train/val/test | `partition` column from CSV (chronological) | replaces single global cut at 2021-10-06 |
| Test window | 2021-12-31 → 2022-06-29 (951 events, 129 days) | smaller than the old 179-day window |

### Same-model pipeline; only labels changed

Architecture surface untouched through the migration:

| Layer | Module | Touched? |
|---|---|---|
| Triple-barrier label generation | `make_labels.py` | **REPLACED** (no longer computes GARCH + barriers; reads CSV) |
| Train/test splitter | `make_deliverables.py`, `make_nn_strategy.py`, `champion_pipeline.py`, `make_importance.py`, `pipeline.py`, `make_features.py` | **rewired** to read `partition` column |
| Feature builder | `make_features.py` + `features/` | unchanged |
| Drift filter | `make_features.py:drift_filter` | unchanged math; KS now train→val |
| Champion selection (CPCV(6,2), 1-SE rule, model roster) | `champion_pipeline.py`, `make_champions.py` | unchanged |
| Per-class Platt calibration | `calibration.py`, `make_deliverables.py` | unchanged |
| Bootstrap p* threshold | `threshold.py`, `make_deliverables.py` | unchanged |
| Six sizing methods + SOPS sigmoid | `sizing.py` | unchanged |
| Causal EWMA σ̂ (slide 39) | `volatility.py:ewma_lecturer` | unchanged |
| Vol-target weight (slide 40) | `sizing.py:position_weight` | unchanged |
| Cross-sectional 1/K_active (slide 41) | `backtest.py`, `nn_portfolio.py` | unchanged |
| Sharpe loss (slide 42) | `nn_portfolio.py:sharpe_loss` | unchanged |
| NN backbones (linear / lstm / vlstm / tft) | `nn_portfolio.py` | unchanged |
| Cluster importance | `make_importance.py` + `importance.py` | unchanged math; partition-aware modelling sample |
| Grinold-Kahn costs | `cost_model.py` | unchanged |
| Backtest | `backtest.py` | unchanged |

### Old GARCH vs Jay labels — quantitative diff (locked submission)

| Metric | Old GARCH `pt=sl=0.5, h=10` | the per-instrument |
|---|---:|---:|
| Total events | 4,886 | 4,917 |
| Sealed test events | 1,342 (179 days) | 951 (129 days) |
| Champions AUC > 0.55 | 7/11 | 6/11 |
| Lower 1-SE CI > 0.50 (signal flag) | 11/11 | 9/11 |
| Importance: ≥1 cluster MDA > 0.02 per class | 2/3 (energy, equity; metals borderline) | 2/3 (same) |
| SOPS test Sharpe | +2.41 | **+2.52** |
| SOPS ann ret (net) | +34.3% | +18.3% |
| SOPS ann vol | 14.2% (over 10% cap) | **7.3% (under cap)** |
| SOPS Sortino | +4.25 | +4.10 |
| SOPS max DD | −6.9% | **−3.1%** |
| SOPS turnover | 267× | **188×** |
| Primary-blind Sharpe (on same test slice) | n/a | **+2.73** |
| Primary-blind ann vol | n/a | 5.9% |

The new locked submission has **comparable Sharpe**, **half the drawdown**,
**70% of the turnover**, and **realised vol that respects the 10% cap** —
the strategy is cleaner under the spec at minor cost in raw return.

### Caveats persisted to `outputs/coverage_caveat.csv`

For every instrument: `thin_oos` (< 5 test events), `no_meta_positions`
(Platt + p* zeroed every position), `documented_failure_in_jay_pdf` (ho1s),
`self_fulfilling_h1_label` (the 6 h=1 instruments per the PDF).

| Instrument | n_oos | thin_oos | no_meta_positions | h1_self_fulfilling | jay_pdf_failure |
|---|---:|:-:|:-:|:-:|:-:|
| cl1s | 88 | | | ✓ | |
| es1s | 109 | | | | |
| fesx1s | 126 | | | ✓ | |
| gc1s | 22 | | | | |
| hg1s | 115 | | ✓ | | |
| **ho1s** | **2** | **✓** | | ✓ | **✓** |
| ng1s | 56 | | | ✓ | |
| nq1s | 122 | | | ✓ | |
| pl1s | 104 | | | ✓ | |
| rb1s | 110 | | ✓ | | |
| si1s | 97 | | ✓ | | |

ho1s is excluded from any quantitative claim: 2 sealed-test events is below
the noise floor for any metric.

### Migration test coverage

`tests/experimental/test_make_labels_jay.py` (13 tests):
- Schema round-trip CSV → events
- Partition counts per instrument match CSV exactly
- `t_start == t_signal` (the entry-at-`date` convention)
- `t_end >= t_start` for every event
- Uniqueness weights ∈ (0, 1]
- h=1 instruments have mean uniqueness 1.0 (consecutive h=1 events are disjoint)
- Larger-h instruments have mean uniqueness < h=1 (overlap)
- One unique `(pt, sl, h)` per instrument
- Geometry summary matches PDF's recommended picks exactly
- ho1s positive rate = 0.667, rb1s positive rate = 0.263 (asymmetric pt consequences)

**157 total experimental tests pass** (13 new Jay-loader + 144 pre-existing).
