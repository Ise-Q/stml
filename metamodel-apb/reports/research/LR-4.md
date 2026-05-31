## Recommendations

**Staged, concrete next steps for EX.2 → S3.7:**

1. **Reduce 124 → ~20–40 inputs before the VSN (do this first; lowest risk).** Reuse the Section-4 correlation/hierarchical clustering you already compute: cut the dendrogram to yield ~20–40 clusters and keep **one representative per cluster**. Choose the representative by max-information-compression-index relative to the cluster centroid (Mitra et al. 2002; de Amorim & Mirkin 2015) or by highest target relevance / lowest redundancy (mRMR; Peng, Long & Ding 2005). This preserves raw-feature interpretability for the VSN and is computationally free given the clustering already exists. *Benchmark that would change this:* if cross-validated performance on the reduced set materially lags the full set, the clusters are dropping signal — widen to ~50–60 representatives before abandoning selection.

2. **As an alternative/complement, test extraction.** If per-feature interpretability is *not* the priority, extract ~5–15 latent factors (PCA, an IPCA-style supervised projection à la Kelly, Pruitt & Su 2019, or a small bottleneck autoencoder à la Hinton & Salakhutdinov 2006 / Gu, Kelly & Xiu 2021). The asset-pricing evidence (≈5 factors suffice; IPCA's ten characteristics drive ~100% of accuracy; Feng–Giglio–Xiu's redundancy result) says this loses little information. Use extraction when maximal compression matters more than naming individual drivers.

3. **Regardless of reduction route, apply a tuned regularisation cocktail** (Kadra et al. 2021): hidden-unit dropout ≈0.5 with lower input dropout (Srivastava et al. 2014); **AdamW** decoupled weight decay (Loshchilov & Hutter 2019) so the learning-rate and weight-decay searches are separable; validation-based **early stopping** with moderate patience (Prechelt 1998); retain the GRN's **layer normalisation** (Ba, Kiros & Hinton 2016) for stability under the small batches a 500-row set forces.

4. **Do not rely on the VSN's softmax gate to control overfitting.** If you want selection learned end-to-end, replace or augment the soft gate with a **sparsity-inducing mechanism that zeroes features and their parameters** — Stochastic Gates (Yamada et al. 2020), L0 regularisation (Louizos, Welling & Kingma 2018), or LassoNet (Lemhadri et al. 2021, which yields a full sparsity path so you can pick the feature count empirically).

5. **Set the realistic baseline.** Before claiming the VSN works, beat a well-tuned XGBoost/random forest on the same split (Shwartz-Ziv & Armon 2022; Grinsztajn et al. 2022). At ~500 samples this is the model class most likely to win; an ensemble of the tree model and the (reduced-input, regularised) VSN is a defensible final deliverable.

## Caveats

- **Sample-size rules of thumb are heuristic.** No primary source reviewed states a single proven parameters-to-samples ratio. The defensible thresholds are qualitative: ~500 samples is firmly inside the "trees-favoured, regularise-aggressively" regime (Grinsztajn et al. find trees still lead at ~10K), and the binding constraint for a per-feature GRN is parameter count, which scales with feature count irrespective of the softmax.
- **Asset-pricing extraction evidence is from a different data shape.** Gu–Kelly–Xiu (2021), Kelly–Pruitt–Su (2019), Chen–Pelger–Zhu (2024) and Feng–Giglio–Xiu (2020) reduce the *cross-sectional characteristic dimension* over tens of thousands of stock-months for a pricing/SDF objective — not a 500-row predictive table. The "≈5 factors suffice" finding transfers as motivation, not as a guarantee of the same factor count for your task.
- **Exact GKX (2021) magnitudes carry a version caveat.** The K=5 / two-hidden-layer ("CA2") preferred specification is consistently confirmed across independent replications, but the precise Sharpe/R² values differ between the SSRN working paper and the published *Journal of Econometrics* article (e.g., preferred value-weighted Sharpe reported as ≈0.92 in the working paper vs. 1.53 in the published conclusion). Verify the published Tables 2–4 directly before quoting a single figure.
- **The Yao et al. (2019) finance clustering example is an imperfect match.** It clusters volatility *lags/components* and selects/aggregates them, rather than literally picking one raw feature per correlation cluster; treat it as supporting analogy, not exact precedent.
- **Mitra et al. (2002) detail** was reconstructed from secondary descriptions within this search session and should be confirmed against the IEEE TPAMI primary text (DOI 10.1109/34.990133) before formal citation; the de Amorim & Mirkin (2015) and Peng–Long–Ding (2005) primary sources were directly verified and are the safer anchors for the cluster-representative step.
- **Net-new vs. already-cited.** All numbered sources in sections (a)–(d) are net-new relative to nlr-cw-v1.md except where explicitly flagged as already-cited anchors (Gu-Kelly-Xiu 2020 RFS; Lim et al. 2021 TFT/VSN; Mantegna 1999; López de Prado 2018/2020; Israel-Kelly-Moskowitz 2020; Borisov et al. 2022). Out-of-scope large-sample CV/CV-NLP architectures were excluded as instructed.

## Net-New Source Checklist (full citations)

- Gu, Kelly & Xiu (2021), *J. Econometrics* 222(1):429–450, DOI 10.1016/j.jeconom.2020.07.009 — autoencoder, K≈5, OOS Sharpe 1.53 vs IPCA 0.96.
- Kelly, Pruitt & Su (2019), *JFE* 134(3):501–524, DOI 10.1016/j.jfineco.2019.05.001 — IPCA; 5 factors; only 10 characteristics significant at 1%, ~100% of accuracy.
- Chen, Pelger & Zhu (2024), *Management Science* 70(2):714–750, DOI 10.1287/mnsc.2023.4695 (arXiv:1904.00745) — deep SDF, few extracted states/factors.
- Feng, Giglio & Xiu (2020), *Journal of Finance* 75(3):1327–1370, DOI 10.1111/jofi.12883 — most new factors redundant (selection/shrinkage).
- Hinton & Salakhutdinov (2006), *Science* 313(5786):504–507, DOI 10.1126/science.1127647 — autoencoder as nonlinear PCA generalisation.
- Peng, Long & Ding (2005), *IEEE TPAMI* 27(8):1226–1238, DOI 10.1109/TPAMI.2005.159 — mRMR.
- de Amorim & Mirkin (2015), *KICSS*, AISC vol. 364, pp. 465–475, DOI 10.1007/978-3-319-19090-7_35 — cluster features, keep representative per cluster (max-information-compression-index).
- Mitra, Murthy & Pal (2002), *IEEE TPAMI* 24(3):301–312, DOI 10.1109/34.990133 — unsupervised representative-feature selection.
- Yao, Izzeldin & Li (2019), *Int. J. Forecasting* 35(4):1318–1331, DOI 10.1016/j.ijforecast.2019.04.017 — cluster HAR; finance clustering-then-representative.
- Grinsztajn, Oyallon & Varoquaux (2022), *NeurIPS Datasets & Benchmarks* (arXiv:2207.08815) — trees win to ~10K; MLPs not robust to uninformative features.
- Shwartz-Ziv & Armon (2022), *Information Fusion* 81:84–90, DOI 10.1016/j.inffus.2021.11.011 — XGBoost beats deep tabular models, less tuning.
- Kadra, Lindauer, Hutter & Grabocka (2021), *NeurIPS* (arXiv:2106.11189) — regularisation cocktail; first to show well-regularised MLPs substantially outperform XGBoost in a fair large-scale study.
- Srivastava et al. (2014), *JMLR* 15(56):1929–1958 — dropout; p≈0.5 hidden, lower on inputs.
- Loshchilov & Hutter (2019), *ICLR* (arXiv:1711.05101) — AdamW decoupled weight decay.
- Prechelt (1998), *Neural Networks: Tricks of the Trade*, LNCS 1524:55–69, DOI 10.1007/3-540-49430-8_3 — early stopping; ~4% generalisation gain vs ~4× time.
- Ba, Kiros & Hinton (2016), arXiv:1607.06450 — layer normalization (batch-size-independent stability).
- Yamada, Lindenbaum, Negahban & Kluger (2020), *ICML*, PMLR 119:10648–10659 (arXiv:1810.04247) — Stochastic Gates (ℓ0-relaxation hard feature selection).
- Lemhadri, Ruan, Abraham & Tibshirani (2021), *JMLR* 22(127):1–29 (AISTATS 2021, PMLR 130:10–18; arXiv:1907.12207) — LassoNet feature sparsity path.
- Louizos, Welling & Kingma (2018), *ICLR* (arXiv:1712.01312) — L0 regularisation via hard-concrete gates.

**Single most important takeaway:** A per-feature GRN/VSN instantiates one GRN per input feature, so parameter count and overfitting risk scale with the 124 features no matter how peaked the softmax gate becomes; reduce to ~20–40 inputs (cluster-representatives or extracted factors) before the VSN, regularise heavily (dropout + AdamW + early stopping, with the GRN's layer norm retained), and benchmark against a tuned tree model — the soft gate alone will not save a 500×124 fit.