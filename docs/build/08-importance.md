# Stage 5a — Cluster-Level Feature Importance

> Module: [`src/stml/importance.py`](../../src/stml/importance.py)
> Course refs: Programming Session 2 (clustered feature importance);
> AFML Ch. 6 (clustered MDI/MDA); AFML Ch. 8 (MDI/MDA).
> Rubric: **10 marks**.

## The motivation

Single-feature importance (MDI on a tree, permutation on anything) is
*biased downward* when features are correlated. Two near-duplicate features
each get half the credit, so neither looks important — even though the
underlying *information* is highly important. The fix is to attribute
importance to **clusters of correlated features**, not features individually.
This is the AFML Ch. 6 / Session 2 recipe.

## Recipe

1. **Distance matrix**: `1 − |Spearman correlation|`. Spearman is rank-based
   ⇒ a feature and its z-scored version sit at distance ≈ 0.
2. **Hierarchical clustering** with `linkage='average'` on the distance matrix.
3. **K chosen by silhouette** on the 1-|Spearman| distance metric, scanned
   over `K ∈ {3, ..., 12}`.
4. **Clustered MDI**: sum each tree's MDI within a cluster ⇒ cluster's MDI
   share. In-sample, comes free from XGBoost's `feature_importances_`.
5. **Clustered MDA**: out-of-sample. For each cluster, permute the *entire
   block* of cluster members with the same row permutation (preserves intra-
   cluster correlation structure), measure the OOS log-loss drop, repeat 5×.

## Real-data output on our 66 features

**Silhouette picked K = 10 clusters.** Composition (most informative ones):

```
Cluster 9 (30 features) — the "vol/regime mega-cluster":
  all 19 G1 vol features (vol_5/21/63d, EWMA, range estimators, vol_of_vol,
  semivol, ratios) + 3 G4 microstructure (amihud, hl_range)
  + 8 G6 regime features (HMM lo/mid/hi, GMM lo/mid/hi, argmax states).

Cluster 8 (16 features) — the "trend mega-cluster":
  most G2 trend features (momentum, MA distance, MA slope, trend t-values)
  + side_signal + net_signal_metals.

Cluster 7 (mean-reversion small):  autocorr_21d, variance_ratio_5d_21w
Cluster 3 (mean-reversion):        days_since_flip, signal_run_len
Cluster 4 (microstructure):        volume_z_63d, volume_trend_21d
Cluster 1 (microstructure):        oi_trend_21d, z_oi_trend_21d
Cluster 5 (signal-context):        hurst_100d, z_amihud_illiq_21d
Cluster 2 (signal-context):        month_sin, net_signal_energy
Cluster 0 (calendar):              dow_sin, dow_cos
Cluster 6 (other):                 gmm_cluster_mid, ...
```

**Reads:** the data-driven clusters *recover the economic groups* — the
features I designated G1 (vol) and G6 (regimes) cluster together, because
the HMM/GMM regime probabilities are literally derived from vol observations.
G2 (trend) clusters together. G4 (microstructure) splits into volume vs OI.
This is reassuring — the engineered economic groupings are robust to a
data-driven sanity check.

## Importance rankings (with XGBoost)

### Clustered MDI (in-sample, XGBoost gain)
```
cluster  mdi_share  dominant_group        n_features
9        0.48       G1_vol (vol+regime)       30
8        0.19       G2_trend                  16
6        0.11       G5_signal                  4
4        0.06       G4_microstructure          4
5        0.04       G5_signal                  2
3        0.04       G3_meanrev                 2
2        0.03       G5_signal                  2
7        0.03       G3_meanrev                 2
1        0.02       G4_microstructure          2
0        0.01       G7_calendar                2
```

**Cluster 9 (vol/regime) = 48% of total MDI.** Almost half the in-sample
"learning power" of the model comes from volatility + regime features.
That's a huge concentration, and it validates the economic thesis: the
meta-model is fundamentally a regime filter.

### Clustered MDA (out-of-sample log-loss drop)
```
cluster  mda     std     rank   dominant_group
9        0.012   0.008    1     vol/regime           ←  same #1
7        0.004   0.001    2     mean-reversion (small)
3        0.003   0.001    3     mean-reversion
4        0.001   0.000    4     microstructure-volume
2        0.001   0.001    5     signal-context
8       -0.003   0.003    8     TREND ← negative OOS!
5       -0.006   0.001    9     signal-context
6       -0.008   0.002   10     other
```

**Two key disagreements between MDI and MDA:**

1. **Trend (Cluster 8) has high MDI (19%) but NEGATIVE MDA OOS.** The model
   learned to use trend features in-sample, but permuting them out OOS
   actually *helps* — meaning the model was *over-relying* on trend signals
   that don't transfer. This is the equity-regime-break story in the
   importance numbers.

2. **Vol/regime (Cluster 9) is #1 by BOTH metrics.** Strongest, most robust
   signal in the model — works in-sample AND OOS.

## Cross-check with MLP permutation importance

Running permutation importance on the MLP gives directionally the same
result: vol/regime cluster #1, trend cluster prominent in-sample, with the
same in-sample / OOS divergence on trend.

## Implications for the report

The cluster importance section can lead with three claims, each backed by a
specific number:

1. **Vol + regime features dominate** (48% MDI, top MDA). The HMM/GMM
   showpiece is genuinely doing work, not decoration.
2. **Trend features overfit in-sample, transfer poorly OOS** (high MDI,
   negative MDA). Directly maps to the equity regime-break finding.
3. **Data-driven clusters recover economic groups**. The engineered taxonomy
   (G1-G7) was the right one — sanity check passed.

These three observations sit comfortably inside the 10-mark
cluster-importance section and tie into the broader story.

## Limitations

- **Clustered MDA is computationally expensive** (5 permutations × 10
  clusters × OOS forward pass). We use `n_repeats=5`; AFML recommends 10+.
- **Hierarchical clustering with average linkage** can produce slightly
  unbalanced clusters (we ended up with one cluster of 30 features). For
  finer-grained analysis one could split mega-clusters or use a fixed K.
- **The two largest clusters are very heterogeneous**: cluster 9 mixes
  pure vol features with HMM regime probabilities (which *use* vol but are
  semantically different). A future refinement is to constrain cluster
  membership by economic group.
