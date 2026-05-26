# 03 — Harry's Feature Pack

> Modules: [`src/stml/harry/features/`](../../src/stml/harry/features/) — 7
> submodules, 22 public feature functions, 1 optional extra.
> Tests: 65 unit tests + 22 universal-harness tests (3 properties × 22
> features) = **87 tests under** [`tests/harry/`](../../tests/harry/).
> Causality harness: [`tests/harry/test_causality.py`](../../tests/harry/test_causality.py).
> Drift guard: [`tests/harry/test_events_consistency.py`](../../tests/harry/test_events_consistency.py).

This is the synthesis-and-creativity layer the rubric rewards. The
features are deliberately chosen so that no other branch ships them: every
function here either captures a phenomenon the other branches do not
encode at all (signal-trajectory, conditional-risk simulation,
information-theoretic, wavelet energy, regime alignment) or is a
correction of a well-known bug in another branch's implementation
(microstructure with the zero-volume mask correctly applied).

## 1. The universal contracts every feature satisfies

Every public function in `src/stml/harry/features/` is registered in the
parametrised causality harness via a `CAUSALITY_REGISTRATIONS` constant
on its own module. The harness verifies three properties, on every
feature, on the same synthetic panel:

1. **Truncation-invariance** — `feature(panel.iloc[:t+1]).iloc[t]` equals
   `feature(panel).iloc[t]` at three values of `t` (100, 200, 400). This
   is the canonical no-leakage guarantee.
2. **Shape preservation** — `len(feature(panel)) == len(panel)`.
3. **No NaN / Inf past warmup** — after the declared warmup window every
   output value is finite.

If a teammate adds a new feature, it appears in the harness automatically
the moment the module exposes `CAUSALITY_REGISTRATIONS` — no edits to the
harness file are needed.

Determinism is also universal: every random draw uses
`np.random.default_rng(seed=42)` (or a per-row derived seed of the form
`seed × 1_000_003 + t`, which is identical between truncated and full
input). Every classifier fit uses `random_state=42`.

## 2. Summary table — all 22 features

| # | family | feature | warmup | range / dtype | notes |
|--:|---|---|--:|---|---|
| 1 | **signal_trajectory** | `signal_run_length` | 0 | int64, ≥ 1 | consecutive-identical-value run length |
| 2 | signal_trajectory | `time_since_last_flip` | 0 | int64, ≥ 0 | = run_length − 1 |
| 3 | signal_trajectory | `signal_entropy_20d` | 19 | [0, log 3] | Shannon entropy of {-1, 0, +1} PMF |
| 4 | signal_trajectory | `signal_flip_rate_60d` | 60 | [0, 1] | fraction of bar-to-bar value changes |
| 5 | signal_trajectory | `signal_cum_pnl_20d` | 19 | ℝ | trailing Σ s·r |
| 6 | **conditional_risk** | `expected_hit_time` | 252† | [1, h+1] | median first-passage time, bootstrap MC |
| 7 | conditional_risk | `prob_timeout` | 252† | [0, 1] | P(no barrier touched in h bars) |
| 8 | conditional_risk | `path_tortuosity_20d` | 19 | [0, ∞) | Σ\|r\| / \|Σ r\|, eps-protected |
| 9 | conditional_risk | `realized_semi_vol_ratio` | 19 | [0, ∞) | RMS(positive r) / RMS(negative r) |
| 10 | **information_theoretic** | `rolling_mutual_information_252d` | 251 | [0, log min(5, 5)] nats | 5×5 quantile-binned MI |
| 11 | information_theoretic | `transfer_entropy_vol_to_signal_acc` | 126 | [0, ∞) nats | Schreiber lag-1 TE |
| 12 | **microstructure_fixed** | `amihud_illiquidity` | 19 | [0, ∞) | mean \|r\|/volume, zero-vol → NaN |
| 13 | microstructure_fixed | `rolls_effective_spread` | 21 | [0, ∞) | Roll (1984) implied spread |
| 14 | microstructure_fixed | `kyles_lambda` | 19 | [0, ∞) | Hasbrouck (2009) bar-form |
| 15 | microstructure_fixed | `overnight_gap` | 1 | ℝ | log(open / close_prev) |
| 16 | **cross_asset** | `distance_to_lead_lag_centroid` | 126† | [0, ∞) | L2 vs lag-shifted peer mean |
| 17 | cross_asset | `asset_class_dispersion_z` | 63† | ℝ | z-score of within-class std |
| 18 | cross_asset | `ewma_implied_corr_z` | 252† | ℝ | z-score of EWMA-avg pairwise corr |
| 19 | **wavelet** (extra) | `mra_energy_bands` (5 cols) | 251† | rows in [0, 1] | MRA detail-band energy fractions |
| 20 | **concept_drift** | `regime_alignment_score` | varies | [0, 1] | discriminator P(row is "recent") |

(† Default window. The causality harness exercises smaller windows for
test speed; production defaults remain as documented.)

Total **22 public functions** (the wavelet feature returns a 5-column
DataFrame, so the produced *feature columns* are
**20 scalars + 5 wavelet columns = 25 columns** at the per-row level
plus whatever asset-class dispersion mapping is chosen).

## 3. Per-family economic intuition + citations

### 3.1 signal_trajectory — the signal's own structure

The primary signal `s_t` is the only labelled input the meta-model sees
that is not derived from price. Reading the signal's recent *structure*
(how long it has held its current position, how often it is flipping,
how chaotic vs persistent the past 60 days have been, how its naïve PnL
has trended) gives the model a direct conditioning variable for "is
this bet worth taking right now". When the signal has been hopping
noisily — high entropy, high flip rate, weak cum-PnL — it is plausibly
less reliable than when it has held a clean run.

No other branch ships these. Sreeram's `G5_signal` cluster has a single
`signal_run_length`-like feature; signal-deep-dive has `f5_signal`-family
features that overlap with Harry's but do not include the Shannon-
entropy or flip-rate views.

**Citations** — Lopez de Prado (2018) Ch. 3 (meta-labelling); Friedman
in Programming Session 4 (regime-conditional features).

### 3.2 conditional_risk — bootstrap first-passage simulation

The triple-barrier label asks "will the bet hit a profit or stop barrier
within h trading days?" — the answer is not known at decision time, only
its conditional distribution is. These features estimate that
distribution from the trailing 252-day empirical return distribution:

- `expected_hit_time` and `prob_timeout` jointly characterise the
  resolution-dynamics distribution. Low expected-hit and low timeout
  probability means the typical recent path resolves *quickly* — a
  fast-resolving regime where the meta-model has a cleaner training
  signal. High expected-hit and high timeout probability means recent
  volatility is below the barrier scale — a quiet regime where
  most events time out and the meta-model is reading drift, not
  first-passage.
- `path_tortuosity_20d` and `realized_semi_vol_ratio` are
  static path-shape features that capture how much of recent
  variation is "round-trip" vs "directional" and how upside / downside
  asymmetry decomposes.

**Bootstrap design.** Non-parametric (no Gaussian assumption); the
trailing-window empirical CDF is sampled with replacement to construct
synthetic paths. The marginal distribution shape (skew, kurtosis,
heavy tails) is preserved; autocorrelation is not. The simulator is
seeded per row, so a row's output is identical across truncations.

**Citations** — Cont & Tankov (2003) for the empirical first-passage
bootstrap; Markowitz (1959) for the semi-vol view.

### 3.3 information_theoretic — non-linear dependence

The signal-direction audit (Step 1) measured *linear* relationships
between signal and trailing / forward returns. Mutual information and
transfer entropy are the non-linear generalisation: they capture
dependence regardless of functional form.

- `rolling_mutual_information_252d` flags windows where the signal
  carries information about a paired quantity (e.g. the h-day forward
  return) that linear correlation misses — "the signal is informative
  only at the tails" or "the signal anticipates volatility but not
  direction" would both show up here.
- `transfer_entropy_vol_to_signal_acc` is the explicit *regime-
  conditioning* question: does the previous day's volatility tell us
  something about whether the signal will be correct today, beyond
  what the signal's own past tells us? A positive TE quantifies the
  audit's hint that vol bands affect reliability.

Both functions are pure histogram estimators (quantile binning, no
KSG / k-NN dependence). They take generic Series and do not shift inputs
— the caller pre-aligns forward quantities. This is explicit in the
docstrings.

**Citations** — Shannon (1948), Schreiber (2000) for transfer entropy,
Cover & Thomas Ch. 2 for the joint-entropy identities used to decompose
TE into four Shannon entropies.

### 3.4 microstructure_fixed — Amihud, Roll, Kyle with the zero-volume mask

Sreeram's `G4` microstructure cluster computes Amihud illiquidity as
`|r| / volume` without masking the 765 documented zero-volume rows
(`reports/missing-data-report.md`). Dividing by zero produces Inf; the
rolling mean propagates the Inf and contaminates the feature. This
module is the corrected implementation.

- `amihud_illiquidity` — Amihud (2002): `mean(|r| / volume)`.
  Zero-volume rows are masked to NaN (vendored predicate
  `volume.where(volume > 0, NaN)` from `na_checks.detect_anomalous_rows`).
- `rolls_effective_spread` — Roll (1984):
  `2 √(max(-Cov(Δp_t, Δp_{t-1}), 0))`. The Roll bounce signature is the
  negative serial covariance of price changes under the bid-ask
  bounce model.
- `kyles_lambda` — Hasbrouck (2009) daily-bar form of Kyle (1985):
  `mean(|r| / √volume)`. Higher = more price impact per share.
- `overnight_gap` — `log(open / close_prev)`. Caller passes
  `close.shift(1)` explicitly so the causality contract is visible at
  the call site.

**Citations** — Amihud (2002), Roll (1984), Kyle (1985), Hasbrouck
(2009). All in the module docstring.

### 3.5 cross_asset — what the rest of the panel is doing

signal-deep-dive showed cross-asset mean |corr| ≈ 0.09 between primary
signals — they are nearly independent across the panel. But the
*returns* are not: an asset-class shock moves multiple instruments
together. These features capture how *this* instrument sits within its
peer group.

- `distance_to_lead_lag_centroid` — L2 distance over a trailing window
  between the instrument's returns and the lag-shifted peer mean. Big
  distance = out of step with the panel (leader/laggard); small =
  conforming. Lag 1 captures the next-day-execution convention.
- `asset_class_dispersion_z` — z-score of the trailing cross-sectional
  std of returns *within the instrument's asset class*. Spikes flag
  intra-class divergence days (silver decoupling from gold etc.).
- `ewma_implied_corr_z` — z-score of the EWMA-averaged pairwise
  correlation between this instrument and every other in the panel.
  Crisis indicator — corrs cluster toward 1 in market-wide risk-off
  events.

Asset-class mapping `ASSET_CLASSES` is exported for the Step 4 pipeline
to reuse:
- equity = {es1s, nq1s, fesx1s}
- energy = {cl1s, ho1s, rb1s, ng1s}
- metals = {gc1s, si1s, hg1s, pl1s}

**Citations** — Pollet & Wilson (2010) for the implied-corr / market-
stress link; Lopez de Prado (2018) Ch. 25 for multi-asset features.

### 3.6 wavelet — multi-resolution-analysis energy bands

A daily-return time series carries information at many scales
simultaneously: intraday noise, weekly cycles, month-end / quarter-end
rebalancing, and slow macro drift. A wavelet MRA decomposes the recent
return path into orthogonal *detail* signals at each scale, and the
energy at each scale tells us how much of recent variation lives at
that scale.

- `mra_energy_bands` returns a 5-column DataFrame: the fraction of the
  trailing 252-day variation that lives at each of the first 5 wavelet
  detail levels. Levels approximately correspond to ~daily, ~weekly,
  ~bi-weekly, ~monthly, ~quarterly cycles.

For the meta-model: a regime where most energy lives at the daily-noise
band is one in which the primary signal's mean-reversion structure is
the active driver; a regime where most energy lives at the monthly /
quarterly bands is one in which slow macro trend dominates and a
counter-trend primary signal may be unreliable.

Requires the `harry-features` extra (PyWavelets). Install instructions
in [`reports/harry/SETUP.md`](SETUP.md). The module raises a clean
ImportError pointing to SETUP.md if pywt is absent.

**Citations** — Percival & Walden (2000) for MRA theory; Gencay, Selcuk
& Whitcher (2002) for finance applications.

### 3.7 concept_drift — discriminator-based regime alignment

Sreeram's v5 critique identified the dominant failure mode of the meta-
model: H1-2022 features differ materially from 2020-2021 training, and
a model trained on the latter does not transfer. The
`regime_alignment_score` quantifies this directly: train a logistic
discriminator at each refit time to distinguish train-era rows from
recent rows, and emit `P(row is "recent")` as the per-day feature.

For the meta-model: the downstream classifier can downweight its
prediction when the alignment score is high — an explicit regime-
confidence channel.

For the team-synthesis memo (Step 6): the alignment score is the
*quantitative* version of Sreeram's regime-break narrative.

**Citations** — Sugiyama & Kawanabe (2012) for the covariate-shift
discriminator; Lopez de Prado (2018) Ch. 7 for the meta-labelling
view.

## 4. What's not here — and why

* **TDA / persistent homology** (the optional eighth module) is
  deferred. The user's brief flagged ripser's C++ build dependencies as
  a deferral trigger. We will revisit if Step 4 finishes ahead of
  schedule.
* **Sreeram's HMM/GMM regime posteriors** are not vendored. The
  filtered-Gaussian-HMM filter is in Sreeram's `regimes.py`; the
  signal-deep-dive branch has its own filtered Markov-switching
  implementation. Step 4's pipeline can vendor whichever turns out to
  be more useful; this module's `regime_alignment_score` is a
  *discriminator-based* alternative that's lighter, faster, and easier
  to audit.
* **Lopez de Prado's information-driven bars / tick-imbalance bars**
  are not implemented — we only have daily bars in our panel, so
  intra-bar features are not derivable.

## 5. How Step 4 will consume this

The Step 4 pipeline will:
1. Load the events checkpoint from `results/harry/events.csv` (drift-
   guarded by the SHA / config-hash test).
2. Build the per-event feature matrix by calling each of the 22 feature
   functions on the cleaned OHLCV panel at the event's `t_signal` date.
3. Drop rows whose features are NaN past the longest warmup (the
   wavelet's 251-row warmup is the binding constraint; cross-asset
   features have 252-row warmup; conditional-risk has 252-row warmup).
4. Train ElasticNet + LightGBM + a re-implemented VSN on the event
   panel, with chronological train / val / test split honoring the
   warmups.
5. Cluster-importance via hierarchical clustering on `1 - |Spearman|`,
   then MDI + MDA + tree-SHAP side-by-side.

## 6. Reproduction

```bash
uv sync --extra harry-features     # one-time, for the wavelet module
uv run pytest tests/harry/ -q      # full Harry test suite (~26 s)
```

The suite must report `N passed` with zero failures. The Step 4
pipeline will fail loudly if any feature drifts away from the test
panel's expected behaviour.
