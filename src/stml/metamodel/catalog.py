"""
catalog.py
==========
The graded **feature catalog** for the triple-barrier metamodel
feature-engineering layer (US-FE-008). This module is the documentation
artifact of the layer: a single :class:`FeatureSpec` registry
(:data:`CATALOG`) that documents *every* produced feature column with its
family, what it captures, its lookback horizon, its leakage class and a
reuse pointer back to the primitive it mirrors / the fit it comes from.

Three companions consume the registry:

* :func:`assert_coverage` -- the 1:1-coverage guard. It raises if any
  non-meta column in a produced feature matrix lacks a :class:`FeatureSpec`
  **or** if any :class:`FeatureSpec` has no matching column (no orphans).
* :func:`render_catalog` -- writes a grouped-by-family markdown table to
  ``reports/feature-catalog.md`` (the deliverable), carrying the four
  required prose annotations (F1 highest-value, F9 expected-negative, F5
  trailing-run causality, F7 Amihud zero-volume guard) and, when supplied,
  an AE-vs-PCA(k=4) reconstruction-MSE table per asset class.
* :data:`CATALOG` itself -- imported by :mod:`stml.metamodel` and the build
  CLI so the catalog renders from one source of truth.

Leakage classes (CONTRACT_FE Section 3)
---------------------------------------
* **E** -- engineered, no fit; causality proven by *truncation-invariance*
  (the value at ``t`` is identical on ``data[:t+1]`` and on ``data[:T]``).
* **TF** -- fitted (GMM / Markov / PCA / KMeans / autoencoder / scaler); fit
  on the FE-train partition only and transformed causally with frozen params
  (the F3 regime and F4 latent families).
* **LI** -- the **label-interface** subset the deferred triple-barrier label
  consumes: the 20-day volatility ``f2_vol_20`` (the barrier ``sigma``) and
  ``f5_trailing_run_length`` (López-de-Prado-style run membership). These are
  engineered + causal like the E-class, but flagged separately because they
  are the FE -> label hand-off interface.

Meta columns
------------
The four provenance / identity columns
``{date, instrument, partition, fe_train_end_date}`` are **not** features and
carry no :class:`FeatureSpec`; :data:`META_COLS` enumerates them and the
coverage / render functions exclude them.
"""

from __future__ import annotations

from dataclasses import dataclass

from stml.metamodel.macro_features import (
    KEEP as _MACRO_KEEP,
    MOMENTUM as _MACRO_MOMENTUM,
    SPREADS as _MACRO_SPREADS,
)

__all__ = [
    "FeatureSpec",
    "CATALOG",
    "META_COLS",
    "assert_coverage",
    "render_catalog",
]

# Provenance / identity columns -- never features (CONTRACT_FE Section 3).
META_COLS: frozenset[str] = frozenset(
    {"date", "instrument", "partition", "fe_train_end_date"}
)

# Human-readable family titles for the grouped markdown render, in the order
# the families appear in the produced matrix (engineered first, then the
# fitted TF families, then cross-sectional).
_FAMILY_TITLES: dict[str, str] = {
    "F1": "F1 — Counter-trend / mean-reversion (C1 highest-value family)",
    "F2": "F2 — Volatility & dispersion",
    "F6": "F6 — Momentum & trend-contrast",
    "F7": "F7 — Microstructure (volume / open-interest)",
    "F10": "F10 — OHLC price-action returns (range / open-to-open)",
    "F5": "F5 — Signal-derived (trailing run structure)",
    "F8": "F8 — Calendar (deterministic sin/cos)",
    "F3": "F3 — Regime posteriors (filtered GMM + Markov, fitted)",
    "F4": "F4 — Latent structure (PCA / KMeans / autoencoder, fitted)",
    "F9": "F9 — Cross-sectional (rank / pair-correlation / cross-asset positioning)",
    "F11": "F11 — Cross-asset macro context (PIT publication-lagged, FE-train z-scored, fitted)",
    "F12": "F12 — Mean-reversion / path-structure & trend-quality (from Sreeram)",
    "F13": "F13 — Wavelet / multiscale energy (from Harry)",
    "F15": "F15 — Conditional risk / first-passage (from Harry)",
    "F16": "F16 — Concept-drift / regime-alignment (from Harry, fitted)",
    "F17": "F17 — HMM regime posteriors (filtered Gaussian HMM, fitted; from Sreeram)",
}


@dataclass(frozen=True)
class FeatureSpec:
    """Documentation record for one produced feature column.

    Parameters
    ----------
    name : str
        The exact produced column name (e.g. ``"f1_mr_score_20"``).
    family : str
        The family tag (``"F1"``..``"F9"``) the column belongs to.
    what_it_captures : str
        A concise, accurate description of the quantity the column measures.
    lookback : str
        The trailing horizon, e.g. ``"20d"``, ``"expanding"``, ``"n/a"`` (for
        a memoryless / single-row feature), or ``"1y"``.
    leakage_class : str
        One of ``"E"`` (engineered, no fit), ``"TF"`` (fitted on FE-train,
        frozen transform) or ``"LI"`` (label-interface subset).
    reuse_pointer : str
        Where the column's logic comes from: a reused primitive
        (e.g. ``"na_checks.rolling_vol"``), the mirrored archetype
        (e.g. ``"features.f1_counter_trend"``), or the fit scope
        (e.g. ``"fit: FE-train per-instrument"``).
    """

    name: str
    family: str
    what_it_captures: str
    lookback: str
    leakage_class: str
    reuse_pointer: str


def _spec(
    name: str,
    family: str,
    what_it_captures: str,
    lookback: str,
    leakage_class: str,
    reuse_pointer: str,
) -> tuple[str, FeatureSpec]:
    """Build a ``(name, FeatureSpec)`` pair for the :data:`CATALOG` dict."""
    return name, FeatureSpec(
        name=name,
        family=family,
        what_it_captures=what_it_captures,
        lookback=lookback,
        leakage_class=leakage_class,
        reuse_pointer=reuse_pointer,
    )


# --------------------------------------------------------------------------- #
# The registry — one FeatureSpec per produced feature column.                 #
# Order mirrors the produced matrix (features.assemble_engineered: F1,F2,F6,  #
# F7,F10,F5,F8; then F3, F4, F9). Leakage classes per CONTRACT_FE Section 3:  #
# F3_*/F4_* = TF; f2_vol_20 (sigma) + f5_trailing_run_length = LI; rest = E.  #
# --------------------------------------------------------------------------- #
CATALOG: dict[str, FeatureSpec] = dict(
    [
        # ----- F1 counter-trend / mean-reversion (15) — C1 highest value ----- #
        _spec(
            "f1_mr_score_10",
            "F1",
            "Counter-trend score -zscore_10(close - SMA_10): close far above its "
            "10d average leans short, far below leans long.",
            "10d",
            "E",
            "features.f1_counter_trend",
        ),
        _spec(
            "f1_mr_score_20",
            "F1",
            "C1 prime counter-trend score -zscore_20(close - SMA_20); the "
            "highest-value mean-reversion replicator.",
            "20d",
            "E",
            "features.f1_counter_trend",
        ),
        _spec(
            "f1_mr_score_40",
            "F1",
            "Counter-trend score -zscore_40(close - SMA_40) at the slow horizon.",
            "40d",
            "E",
            "features.f1_counter_trend",
        ),
        _spec(
            "f1_dist_ma_sigma_10",
            "F1",
            "Gap (close - SMA_10) expressed in trailing-sigma units of the gap.",
            "10d",
            "E",
            "features._zscore",
        ),
        _spec(
            "f1_dist_ma_sigma_20",
            "F1",
            "Gap (close - SMA_20) expressed in trailing-sigma units of the gap.",
            "20d",
            "E",
            "features._zscore",
        ),
        _spec(
            "f1_dist_ma_sigma_40",
            "F1",
            "Gap (close - SMA_40) expressed in trailing-sigma units of the gap.",
            "40d",
            "E",
            "features._zscore",
        ),
        _spec(
            "f1_ret_reversal_10",
            "F1",
            "Negated trailing 10d log return: a reversal (counter-trend) lean.",
            "10d",
            "E",
            "features.f1_counter_trend",
        ),
        _spec(
            "f1_ret_reversal_20",
            "F1",
            "Negated trailing 20d log return: a reversal (counter-trend) lean.",
            "20d",
            "E",
            "features.f1_counter_trend",
        ),
        _spec(
            "f1_ret_reversal_40",
            "F1",
            "Negated trailing 40d log return: a reversal (counter-trend) lean.",
            "40d",
            "E",
            "features.f1_counter_trend",
        ),
        _spec(
            "f1_hilo_pos_10",
            "F1",
            "Position of the close within its trailing 10d [min, max] range, [0,1].",
            "10d",
            "E",
            "features.f1_counter_trend",
        ),
        _spec(
            "f1_hilo_pos_20",
            "F1",
            "Position of the close within its trailing 20d [min, max] range, [0,1].",
            "20d",
            "E",
            "features.f1_counter_trend",
        ),
        _spec(
            "f1_hilo_pos_40",
            "F1",
            "Position of the close within its trailing 40d [min, max] range, [0,1].",
            "40d",
            "E",
            "features.f1_counter_trend",
        ),
        _spec(
            "f1_rsi_14",
            "F1",
            "Wilder RSI(14) on closes (overbought/oversold momentum), in [0,100].",
            "14d",
            "E",
            "features.f1_counter_trend",
        ),
        _spec(
            "f1_bb_pctb_20",
            "F1",
            "Bollinger %b (close - lower)/(upper - lower) on the 20d/2sigma band.",
            "20d",
            "E",
            "features.f1_counter_trend",
        ),
        _spec(
            "f1_bb_bandwidth_20",
            "F1",
            "Bollinger bandwidth (upper - lower)/SMA_20: relative band width.",
            "20d",
            "E",
            "features.f1_counter_trend",
        ),
        # ----- F2 volatility & dispersion (11); f2_vol_20 = LI sigma --------- #
        _spec(
            "f2_vol_10",
            "F2",
            "Annualised trailing 10d realised volatility of log returns.",
            "10d",
            "E",
            "na_checks.rolling_vol",
        ),
        _spec(
            "f2_vol_20",
            "F2",
            "Annualised trailing 20d realised volatility — the label-interface "
            "sigma the deferred triple-barrier label consumes.",
            "20d",
            "LI",
            "na_checks.rolling_vol",
        ),
        _spec(
            "f2_vol_60",
            "F2",
            "Annualised trailing 60d realised volatility of log returns.",
            "60d",
            "E",
            "na_checks.rolling_vol",
        ),
        _spec(
            "f2_vol_ratio_20_60",
            "F2",
            "Short-vs-long vol regime: f2_vol_20 / f2_vol_60.",
            "20d/60d",
            "E",
            "features.f2_vol_dispersion",
        ),
        _spec(
            "f2_vol_pctile_20",
            "F2",
            "Trailing 1y (252d) percentile of f2_vol_20: how high vol is vs its "
            "own recent history.",
            "1y",
            "E",
            "features._rolling_percentile",
        ),
        _spec(
            "f2_vol_of_vol_20",
            "F2",
            "Vol-of-vol: trailing 60d std of f2_vol_20 (volatility instability).",
            "60d",
            "E",
            "features.f2_vol_dispersion",
        ),
        _spec(
            "f2_parkinson_20",
            "F2",
            "Parkinson high-low range volatility over 20d, annualised.",
            "20d",
            "E",
            "features.f2_vol_dispersion",
        ),
        _spec(
            "f2_garman_klass_20",
            "F2",
            "Garman-Klass OHLC volatility over 20d, annualised.",
            "20d",
            "E",
            "features.f2_vol_dispersion",
        ),
        _spec(
            "f2_atr_14",
            "F2",
            "Wilder Average True Range over 14d (price-unit volatility).",
            "14d",
            "E",
            "features.f2_vol_dispersion",
        ),
        _spec(
            "f2_ret_skew_60",
            "F2",
            "Trailing 60d skewness of daily log returns (return asymmetry).",
            "60d",
            "E",
            "features.f2_vol_dispersion",
        ),
        _spec(
            "f2_ret_kurt_60",
            "F2",
            "Trailing 60d excess kurtosis of daily log returns (tail heaviness).",
            "60d",
            "E",
            "features.f2_vol_dispersion",
        ),
        # ----- F6 momentum & trend-contrast (7) ------------------------------ #
        _spec(
            "f6_ts_momentum_20",
            "F6",
            "Vol-scaled trailing 20d log return (time-series momentum score).",
            "20d",
            "E",
            "features.f6_momentum_contrast",
        ),
        _spec(
            "f6_ts_momentum_60",
            "F6",
            "Vol-scaled trailing 60d log return (time-series momentum score).",
            "60d",
            "E",
            "features.f6_momentum_contrast",
        ),
        _spec(
            "f6_ma_cross_20_60",
            "F6",
            "Fast-vs-slow MA cross (SMA_20 - SMA_60)/SMA_60 (trend direction).",
            "20d/60d",
            "E",
            "features.f6_momentum_contrast",
        ),
        _spec(
            "f6_macd_12_26",
            "F6",
            "MACD line EMA_12 - EMA_26 on closes (price-unit momentum).",
            "12d/26d",
            "E",
            "features.f6_momentum_contrast",
        ),
        _spec(
            "f6_macd_hist_12_26_9",
            "F6",
            "MACD histogram MACD - EMA_9(MACD): momentum acceleration.",
            "12d/26d/9d",
            "E",
            "features.f6_momentum_contrast",
        ),
        _spec(
            "f6_adx_14",
            "F6",
            "Wilder ADX(14) trend strength (regardless of direction), in [0,100].",
            "14d",
            "E",
            "features.f6_momentum_contrast",
        ),
        _spec(
            "f6_donchian_pos_20",
            "F6",
            "Close position in the prior 20d Donchian channel (breakout), [-1,1].",
            "20d",
            "E",
            "features.f6_momentum_contrast",
        ),
        # ----- F7 microstructure (7); Amihud zero-volume->NaN guard ---------- #
        _spec(
            "f7_volume_z_20",
            "F7",
            "Trailing 20d z-score of traded volume (volume surprise).",
            "20d",
            "E",
            "features._zscore",
        ),
        _spec(
            "f7_volume_trend_20",
            "F7",
            "Volume trend volume/SMA_20(volume) - 1 (relative volume).",
            "20d",
            "E",
            "features.f7_microstructure",
        ),
        _spec(
            "f7_oi_level",
            "F7",
            "Raw open interest level (structural NaN where OI is missing).",
            "n/a",
            "E",
            "features.f7_microstructure",
        ),
        _spec(
            "f7_oi_change",
            "F7",
            "One-day change in open interest (position flow).",
            "1d",
            "E",
            "features.f7_microstructure",
        ),
        _spec(
            "f7_oi_z_20",
            "F7",
            "Trailing 20d z-score of open interest.",
            "20d",
            "E",
            "features._zscore",
        ),
        _spec(
            "f7_oi_price_div_20",
            "F7",
            "Sign agreement of trailing 20d OI change vs price change "
            "(+1 aligned, -1 divergent, 0 flat).",
            "20d",
            "E",
            "features.f7_microstructure",
        ),
        _spec(
            "f7_amihud_20",
            "F7",
            "Trailing 20d mean Amihud illiquidity |ret|/volume; zero-volume rows "
            "are NaN-guarded (no divide-by-zero).",
            "20d",
            "E",
            "features.f7_microstructure",
        ),
        # ----- F10 OHLC price-action returns (4) — raw range / open-to-open -- #
        _spec(
            "f10_hl_range",
            "F10",
            "Daily intraday high-low log range log(high/low): the per-bar "
            "trading range (raw daily form of the f2_parkinson_20 input).",
            "1d",
            "E",
            "features.f10_price_action",
        ),
        _spec(
            "f10_hl_range_mean_20",
            "F10",
            "Trailing 20d mean of the daily high-low log range (typical recent "
            "intraday range).",
            "20d",
            "E",
            "features.f10_price_action",
        ),
        _spec(
            "f10_oto_ret",
            "F10",
            "Daily open-to-open log return log(open_t/open_{t-1}); the "
            "open-anchored, overnight-inclusive counterpart to the close return.",
            "1d",
            "E",
            "features.f10_price_action",
        ),
        _spec(
            "f10_oto_ret_mean_20",
            "F10",
            "Trailing 20d mean of the open-to-open log return (open-anchored "
            "drift).",
            "20d",
            "E",
            "features.f10_price_action",
        ),
        # ----- F5 signal-derived (9); f5_trailing_run_length = LI ------------ #
        _spec(
            "f5_signal",
            "F5",
            "The released signal s_t in {-1, 0, +1}.",
            "n/a",
            "E",
            "features.f5_signal_derived",
        ),
        _spec(
            "f5_abs_signal",
            "F5",
            "|s_t| participation indicator (1 on a participating day, else 0).",
            "n/a",
            "E",
            "features.f5_signal_derived",
        ),
        _spec(
            "f5_trailing_run_length",
            "F5",
            "Length of the current constant-signal run ending at t, computed on "
            "s[:t+1] (causal); the label's run-membership interface.",
            "expanding",
            "LI",
            "features._trailing_run_length",
        ),
        _spec(
            "f5_days_since_flip",
            "F5",
            "Trailing days since the signal last changed (run length - 1), "
            "computed on s[:t+1] (causal).",
            "expanding",
            "E",
            "features._days_since_flip",
        ),
        _spec(
            "f5_days_since_nonzero",
            "F5",
            "Trailing days since the last participating (nonzero) signal.",
            "expanding",
            "E",
            "features._days_since_nonzero",
        ),
        _spec(
            "f5_participation_20",
            "F5",
            "Trailing 20d mean |s| (share of participating days).",
            "20d",
            "E",
            "features.f5_signal_derived",
        ),
        _spec(
            "f5_participation_60",
            "F5",
            "Trailing 60d mean |s| (share of participating days).",
            "60d",
            "E",
            "features.f5_signal_derived",
        ),
        _spec(
            "f5_long_bias_20",
            "F5",
            "Trailing 20d mean s (net long/short tilt of the signal).",
            "20d",
            "E",
            "features.f5_signal_derived",
        ),
        _spec(
            "f5_sign_agree_mr",
            "F5",
            "Sign agreement of s_t with the C1 counter-trend score f1_mr_score_20 "
            "(+1 agree, -1 disagree, 0 flat/NaN).",
            "n/a",
            "E",
            "features.f5_signal_derived",
        ),
        # ----- F8 calendar (4) — deterministic ------------------------------- #
        _spec(
            "f8_dow_sin",
            "F8",
            "Sine encoding of day-of-week on a 7-cycle (cyclical calendar).",
            "n/a",
            "E",
            "features.f8_calendar",
        ),
        _spec(
            "f8_dow_cos",
            "F8",
            "Cosine encoding of day-of-week on a 7-cycle (cyclical calendar).",
            "n/a",
            "E",
            "features.f8_calendar",
        ),
        _spec(
            "f8_month_sin",
            "F8",
            "Sine encoding of month on a 12-cycle (seasonality).",
            "n/a",
            "E",
            "features.f8_calendar",
        ),
        _spec(
            "f8_month_cos",
            "F8",
            "Cosine encoding of month on a 12-cycle (seasonality).",
            "n/a",
            "E",
            "features.f8_calendar",
        ),
        # ----- F3 regime posteriors (4) — TF, fit FE-train per-instrument ---- #
        _spec(
            "f3_gmm_prob_highvol",
            "F3",
            "Filtered GMM posterior probability of the high-vol component on the "
            "frozen-standardized (ret, vol) at t (predict_proba on <= t).",
            "expanding",
            "TF",
            "fit: FE-train per-instrument",
        ),
        _spec(
            "f3_markov_prob_highvol",
            "F3",
            "Filtered (one-sided) Markov-switching posterior of the high-vol "
            "regime; uses returns <= t only (never smoothed).",
            "expanding",
            "TF",
            "fit: FE-train per-instrument",
        ),
        _spec(
            "f3_markov_switch_prob",
            "F3",
            "Trailing |delta| of the filtered high-vol probability (regime-switch "
            "intensity at t).",
            "1d",
            "TF",
            "fit: FE-train per-instrument",
        ),
        _spec(
            "f3_regime_dwell",
            "F3",
            "Trailing days since the argmax (filtered) regime call last changed "
            "(regime persistence).",
            "expanding",
            "TF",
            "fit: FE-train per-instrument",
        ),
        # ----- F4 latent structure (11) — TF, fit FE-train class-pooled ------ #
        _spec(
            "f4_pc1",
            "F4",
            "1st principal component of the scaled engineered-feature vector "
            "(class-pooled PCA).",
            "n/a",
            "TF",
            "fit: FE-train class-pooled",
        ),
        _spec(
            "f4_pc2",
            "F4",
            "2nd principal component of the scaled engineered-feature vector.",
            "n/a",
            "TF",
            "fit: FE-train class-pooled",
        ),
        _spec(
            "f4_pc3",
            "F4",
            "3rd principal component of the scaled engineered-feature vector.",
            "n/a",
            "TF",
            "fit: FE-train class-pooled",
        ),
        _spec(
            "f4_pc4",
            "F4",
            "4th principal component of the scaled engineered-feature vector.",
            "n/a",
            "TF",
            "fit: FE-train class-pooled",
        ),
        _spec(
            "f4_cluster_id",
            "F4",
            "Assigned KMeans (k=4) cluster id of the scaled feature vector "
            "(class-pooled regime archetype). NOMINAL, not ordinal: a "
            "downstream model should one-hot this, not treat id 3 > id 1.",
            "n/a",
            "TF",
            "fit: FE-train class-pooled",
        ),
        _spec(
            "f4_cluster_dist",
            "F4",
            "Euclidean distance from the scaled vector to its assigned KMeans "
            "centroid (how atypical today is).",
            "n/a",
            "TF",
            "fit: FE-train class-pooled",
        ),
        _spec(
            "f4_ae_code1",
            "F4",
            "1st autoencoder bottleneck code (nonlinear latent coordinate).",
            "n/a",
            "TF",
            "fit: FE-train class-pooled",
        ),
        _spec(
            "f4_ae_code2",
            "F4",
            "2nd autoencoder bottleneck code (nonlinear latent coordinate).",
            "n/a",
            "TF",
            "fit: FE-train class-pooled",
        ),
        _spec(
            "f4_ae_code3",
            "F4",
            "3rd autoencoder bottleneck code (nonlinear latent coordinate).",
            "n/a",
            "TF",
            "fit: FE-train class-pooled",
        ),
        _spec(
            "f4_ae_code4",
            "F4",
            "4th autoencoder bottleneck code (nonlinear latent coordinate).",
            "n/a",
            "TF",
            "fit: FE-train class-pooled",
        ),
        _spec(
            "f4_ae_recon_err",
            "F4",
            "Per-row autoencoder reconstruction MSE (nonlinear novelty score).",
            "n/a",
            "TF",
            "fit: FE-train class-pooled",
        ),
        # ----- F9 cross-sectional (3) — expected-negative pair corr ---------- #
        _spec(
            "f9_xsect_rank",
            "F9",
            "Cross-sectional rank of the trailing 20d reversal score across the "
            "per-day finite-score universe, normalised to [-1, 1].",
            "20d",
            "E",
            "xsection.xsection_features",
        ),
        _spec(
            "f9_xsection_universe_size",
            "F9",
            "Number of instruments with a finite trailing score on day t "
            "(ragged multi-venue calendar; ~24/645 days < 11).",
            "20d",
            "E",
            "xsection.xsection_features",
        ),
        _spec(
            "f9_pair_corr_mean",
            "F9",
            "Mean rolling pairwise correlation to asset-class peers; documented "
            "expected-negative (cross-asset mean |corr| ~= 0.09).",
            "120d",
            "E",
            "na_checks.rolling_pair_corr",
        ),
    ]
)


# --------------------------------------------------------------------------- #
# F11 cross-asset macro context (45) — TF, generated from the SAME traversal   #
# macro_features.macro_feature_columns uses, so the catalog entry set and the  #
# produced-column set cannot drift (test_macro_catalog_roundtrip pins this).   #
# --------------------------------------------------------------------------- #
_MACRO_REUSE = "fit: FE-train frozen z-score (macro_features)"


def _macro_specs() -> list[tuple[str, FeatureSpec]]:
    """Build the F11 ``(name, FeatureSpec)`` pairs from the macro MACRO_SPEC.

    Mirrors :func:`stml.metamodel.macro_features.macro_feature_columns`: per
    standalone series and per spread, a PIT-applied ``level`` plus two
    ``chg{h}`` momentum columns (``h`` business days). All are leakage-class
    ``TF`` (FE-train-frozen standardizer).
    """
    specs: list[tuple[str, FeatureSpec]] = []
    for name, (rcls, captures) in _MACRO_KEEP.items():
        slug = name.lower()
        h1, h2 = _MACRO_MOMENTUM[rcls]
        specs.append(
            _spec(
                f"f11_{slug}_level",
                "F11",
                f"Point-in-time-applied level. {captures}",
                "PIT level",
                "TF",
                _MACRO_REUSE,
            )
        )
        for h in (h1, h2):
            specs.append(
                _spec(
                    f"f11_{slug}_chg{h}",
                    "F11",
                    f"{h}-business-day change of the PIT-applied level. {captures}",
                    f"{h}d",
                    "TF",
                    _MACRO_REUSE,
                )
            )
    for sname, (_, _, rcls, captures) in _MACRO_SPREADS.items():
        h1, h2 = _MACRO_MOMENTUM[rcls]
        specs.append(
            _spec(
                f"f11_spread_{sname}_level",
                "F11",
                f"Point-in-time-applied spread level. {captures}",
                "PIT level",
                "TF",
                _MACRO_REUSE,
            )
        )
        for h in (h1, h2):
            specs.append(
                _spec(
                    f"f11_spread_{sname}_chg{h}",
                    "F11",
                    f"{h}-business-day change of the PIT-applied spread. {captures}",
                    f"{h}d",
                    "TF",
                    _MACRO_REUSE,
                )
            )
    return specs


CATALOG.update(dict(_macro_specs()))


# --------------------------------------------------------------------------- #
# Extended families folded in from the Harry / Sreeram branches.              #
# F2-add (Rogers-Satchell), F5-adds (entropy/flip-rate), F7-adds (Roll/Kyle/  #
# overnight), F9-adds (cross-asset positioning), F12 (path-structure), F13    #
# (wavelet), F15 (conditional risk), F16 (drift, TF), F17 (HMM, TF). All      #
# E-class unless noted; provenance recorded in the reuse pointer.             #
# --------------------------------------------------------------------------- #
def _ext_specs() -> list[tuple[str, FeatureSpec]]:
    """Build the ``(name, FeatureSpec)`` pairs for the extended families."""
    return [
        # ----- F2 add — Rogers-Satchell range volatility (Sreeram) ----------- #
        _spec(
            "f2_rogers_satchell_20",
            "F2",
            "Rogers-Satchell drift-independent OHLC range volatility over 20d, "
            "annualised (the drift-robust sibling of Parkinson / Garman-Klass).",
            "20d",
            "E",
            "features_ext.f2_rogers_satchell",
        ),
        # ----- F5 adds — signal-trajectory entropy / flip-rate (Harry) ------- #
        _spec(
            "f5_signal_entropy_20",
            "F5",
            "Shannon entropy (nats) of the trailing 20d {-1,0,+1} signal PMF in "
            "[0, log 3]; high = the signal is hopping across states.",
            "20d",
            "E",
            "features_ext.f5_signal_trajectory",
        ),
        _spec(
            "f5_flip_rate_60",
            "F5",
            "Fraction of consecutive-bar signal value changes over the trailing "
            "60d, in [0, 1] (signal instability).",
            "60d",
            "E",
            "features_ext.f5_signal_trajectory",
        ),
        # ----- F7 adds — Roll spread / Kyle's lambda / overnight gap (Harry) - #
        _spec(
            "f7_rolls_spread_20",
            "F7",
            "Roll (1984) implied bid-ask spread 2·sqrt(max(-Cov(Δp_t,Δp_{t-1}),0)) "
            "over 20d (negative serial covariance = bounce signature).",
            "20d",
            "E",
            "features_ext.f7_microstructure_ext",
        ),
        _spec(
            "f7_kyles_lambda_20",
            "F7",
            "Hasbrouck (2009) daily-bar Kyle's lambda mean(|ret|/sqrt(volume)) "
            "over 20d (price impact per share); zero-volume rows NaN-guarded.",
            "20d",
            "E",
            "features_ext.f7_microstructure_ext",
        ),
        _spec(
            "f7_overnight_gap",
            "F7",
            "Overnight log return log(open_t / close_{t-1}) (the gap a "
            "close-to-close return blends away).",
            "1d",
            "E",
            "features_ext.f7_microstructure_ext",
        ),
        # ----- F9 adds — cross-asset positioning (Harry) --------------------- #
        _spec(
            "f9_dist_lead_lag_centroid",
            "F9",
            "L2 distance over a trailing 126d window between the instrument's "
            "returns and the 1-day-lagged mean of the rest of the panel "
            "(small = tracks the lagged panel, large = out-of-step leader/laggard).",
            "126d",
            "E",
            "xsection._f9_lead_lag_centroid",
        ),
        _spec(
            "f9_asset_class_dispersion_z",
            "F9",
            "Z-score of the trailing 63d cross-sectional return std within the "
            "instrument's asset class (intra-class divergence spikes).",
            "63d",
            "E",
            "xsection._f9_asset_class_dispersion_z",
        ),
        _spec(
            "f9_ewma_implied_corr_z",
            "F9",
            "Z-score of the EWMA(halflife 20) mean pairwise correlation with the "
            "rest of the panel over 252d (a market-stress / crisis spike).",
            "252d",
            "E",
            "xsection._f9_ewma_implied_corr_z",
        ),
        # ----- F12 mean-reversion / path-structure & trend-quality (Sreeram) - #
        _spec(
            "f12_autocorr_21",
            "F12",
            "Lag-1 autocorrelation of returns over 21d (negative = mean-reverting, "
            "positive = trending).",
            "21d",
            "E",
            "features_ext.f12_path_structure",
        ),
        _spec(
            "f12_efficiency_ratio_21",
            "F12",
            "Kaufman efficiency ratio |net move|/sum(|moves|) over 21d in [0,1] "
            "(1 = clean directional move, 0 = pure noise).",
            "21d",
            "E",
            "features_ext.f12_path_structure",
        ),
        _spec(
            "f12_variance_ratio_5_21",
            "F12",
            "Variance ratio Var(5d)/(5·Var(1d)) over 21d; >1 trending, <1 "
            "mean-reverting, =1 random walk.",
            "21d",
            "E",
            "features_ext.f12_path_structure",
        ),
        _spec(
            "f12_trend_tval_10",
            "F12",
            "t-statistic of the OLS slope of log-close on a time index over 10d "
            "(tValLinR); high |t| = a statistically clean trend.",
            "10d",
            "E",
            "features_ext.f12_path_structure",
        ),
        _spec(
            "f12_trend_tval_21",
            "F12",
            "t-statistic of the OLS slope of log-close on a time index over 21d "
            "(tValLinR); high |t| = a statistically clean trend.",
            "21d",
            "E",
            "features_ext.f12_path_structure",
        ),
        _spec(
            "f12_trend_tval_42",
            "F12",
            "t-statistic of the OLS slope of log-close on a time index over 42d "
            "(tValLinR); high |t| = a statistically clean trend.",
            "42d",
            "E",
            "features_ext.f12_path_structure",
        ),
        _spec(
            "f12_hurst_100",
            "F12",
            "Rescaled-range Hurst exponent of returns over 100d (>0.5 persistent/"
            "trending, <0.5 anti-persistent/mean-reverting, =0.5 random walk).",
            "100d",
            "E",
            "features_ext.f12_path_structure",
        ),
        _spec(
            "f12_ma21_slope",
            "F12",
            "1-day log change of the 21d moving average normalised by the 1d "
            "return std (dimensionless trend slope).",
            "21d",
            "E",
            "features_ext.f12_path_structure",
        ),
        # ----- F13 wavelet / multiscale energy (Harry) ----------------------- #
        *[
            _spec(
                f"f13_mra_energy_d{k}",
                "F13",
                f"Fraction of the trailing 252d return variation at wavelet detail "
                f"level D{k} (db4 MRA; level D{k} ~ "
                f"{['daily', 'weekly', 'bi-weekly', 'monthly', 'quarterly'][k - 1]} "
                f"scale). Rows sum to <= 1.",
                "1y",
                "E",
                "features_ext.f13_wavelet_energy",
            )
            for k in range(1, 6)
        ],
        # ----- F15 conditional risk / first-passage (Harry) ------------------ #
        _spec(
            "f15_expected_hit_time",
            "F15",
            "Bootstrap median first-passage time to symmetric ±sigma·sqrt(h) "
            "barriers (h=10) from the trailing 252d return distribution; timeout "
            "= h+1. Low = fast-resolving regime.",
            "1y",
            "E",
            "features_ext.f15_conditional_risk",
        ),
        _spec(
            "f15_prob_timeout",
            "F15",
            "Bootstrap probability that neither barrier is touched within h=10 "
            "bars (high = recent vol below the barrier scale), in [0, 1].",
            "1y",
            "E",
            "features_ext.f15_conditional_risk",
        ),
        _spec(
            "f15_path_tortuosity_20",
            "F15",
            "Trailing 20d sum|r| / |sum r| (1 = monotonic path, larger = zigzag); "
            "trend signals are more reliable on lower-tortuosity paths.",
            "20d",
            "E",
            "features_ext.f15_conditional_risk",
        ),
        _spec(
            "f15_realized_semi_vol_ratio_20",
            "F15",
            "Upside-RMS / downside-RMS of trailing 20d returns (>1 = recent "
            "up-moves larger than down-moves; recent return asymmetry).",
            "20d",
            "E",
            "features_ext.f15_conditional_risk",
        ),
        # ----- F16 concept-drift / regime-alignment (Harry, TF) -------------- #
        _spec(
            "f16_regime_alignment_score",
            "F16",
            "Rolling logistic-discriminator P(today looks 'recent' vs the FE-train "
            "era) in [0,1]; an explicit regime/covariate-shift confidence channel. "
            "Structural NaN on FE-train-era rows (before the first refit).",
            "expanding",
            "TF",
            "fit: rolling FE-train-vs-recent discriminator (drift_features)",
        ),
        # ----- F17 HMM regime posteriors (Sreeram, TF) ----------------------- #
        _spec(
            "f17_hmm_state_lo",
            "F17",
            "Filtered (causal, forward-only) posterior of the low-vol state of a "
            "3-state Gaussian HMM on (ret, vol); fit on FE-train, frozen.",
            "expanding",
            "TF",
            "fit: FE-train per-instrument (regime_features_hmm)",
        ),
        _spec(
            "f17_hmm_state_mid",
            "F17",
            "Filtered (causal) posterior of the mid-vol HMM state on (ret, vol).",
            "expanding",
            "TF",
            "fit: FE-train per-instrument (regime_features_hmm)",
        ),
        _spec(
            "f17_hmm_state_hi",
            "F17",
            "Filtered (causal) posterior of the high-vol HMM state on (ret, vol).",
            "expanding",
            "TF",
            "fit: FE-train per-instrument (regime_features_hmm)",
        ),
        _spec(
            "f17_hmm_state_argmax",
            "F17",
            "Most-likely HMM state at t (0=lo,1=mid,2=hi by FE-train mean vol). "
            "NOMINAL, not ordinal — a downstream model should one-hot it.",
            "expanding",
            "TF",
            "fit: FE-train per-instrument (regime_features_hmm)",
        ),
    ]


CATALOG.update(dict(_ext_specs()))


# --------------------------------------------------------------------------- #
# Standardization twins: one expanding-z `z_<col>` spec per scale-dependent   #
# E-class column (the SAME list features_ext.add_z_twins emits), so the 1:1   #
# coverage guard cannot drift. Each twin inherits its base column's family.   #
# --------------------------------------------------------------------------- #
def _z_twin_specs() -> list[tuple[str, FeatureSpec]]:
    """Build a ``z_<col>`` :class:`FeatureSpec` per :data:`Z_TWIN_COLUMNS`."""
    from stml.metamodel.features_ext import Z_TWIN_COLUMNS

    specs: list[tuple[str, FeatureSpec]] = []
    for base in Z_TWIN_COLUMNS:
        base_spec = CATALOG[base]  # base must already be registered
        specs.append(
            _spec(
                f"z_{base}",
                base_spec.family,
                f"Per-instrument causal expanding-window z-score of `{base}` "
                f"(split-agnostic standardization twin).",
                "expanding",
                "E",
                "features_ext.expanding_zscore",
            )
        )
    return specs


CATALOG.update(dict(_z_twin_specs()))


# --------------------------------------------------------------------------- #
# Coverage guard (AC-1): exact 1:1 spec <-> produced-column correspondence.   #
# --------------------------------------------------------------------------- #
def _feature_columns(matrix_columns: object) -> list[str]:
    """Return the non-meta columns of ``matrix_columns`` in order.

    Parameters
    ----------
    matrix_columns : iterable of str
        Columns of a produced feature matrix (e.g. ``matrix.columns``).

    Returns
    -------
    list[str]
        The columns with the four :data:`META_COLS` removed.
    """
    return [c for c in matrix_columns if c not in META_COLS]


def assert_coverage(matrix_columns: object) -> None:
    """Assert exact 1:1 coverage between :data:`CATALOG` and the matrix columns.

    Every non-meta column of the produced matrix must have a
    :class:`FeatureSpec` in :data:`CATALOG`, and every :class:`FeatureSpec`
    must have a matching produced column. The check is symmetric: a missing
    spec and an orphan spec both fail (AC-1).

    Parameters
    ----------
    matrix_columns : iterable of str
        Columns of a produced feature matrix (the four meta columns are
        excluded automatically).

    Raises
    ------
    AssertionError
        If any non-meta column lacks a :class:`FeatureSpec`, or if any
        :class:`FeatureSpec` has no matching column. The message lists both
        offending sets.
    """
    feature_cols = _feature_columns(matrix_columns)
    cols = set(feature_cols)
    specs = set(CATALOG)

    missing = sorted(cols - specs)  # produced columns with no spec
    orphans = sorted(specs - cols)  # specs with no produced column

    if missing or orphans:
        raise AssertionError(
            "Feature catalog coverage failure (expected exact 1:1):\n"
            f"  columns missing a CATALOG entry ({len(missing)}): {missing}\n"
            f"  orphan CATALOG entries with no column ({len(orphans)}): {orphans}"
        )

    # Each spec's recorded name must equal its dict key (no silent mislabel).
    mislabelled = sorted(
        name for name, spec in CATALOG.items() if spec.name != name
    )
    if mislabelled:
        raise AssertionError(
            f"CATALOG keys disagree with FeatureSpec.name: {mislabelled}"
        )


# --------------------------------------------------------------------------- #
# Markdown render (the deliverable) with the four required annotations.       #
# --------------------------------------------------------------------------- #
# The four required prose annotations (CONTRACT_FE Section 3). Keywords the
# rendered file is asserted to contain: "mean-reversion", "0.09",
# "run_length_p90", "Amihud".
_ANNOTATIONS: list[str] = [
    "**F1 is the highest-value family (per C1).** The signal is a short-horizon "
    "mean-reversion / counter-trend strategy, so the F1 counter-trend family — "
    "led by `f1_mr_score_20` — is expected to carry the most predictive value.",
    "**F9 `f9_pair_corr_mean` is expected-negative.** The cross-asset mean "
    "absolute correlation is approximately 0.09; the cross-sectional features "
    "are retained precisely because they capture near-independent structure the "
    "per-instrument time-series features miss.",
    "**F5 `f5_trailing_run_length` / `f5_days_since_flip` are CAUSAL trailing "
    "features** computed on `s[:t+1]` (an expanding scan). They are deliberately "
    "DISTINCT from the full-period `splits.run_length_p90`, which measures the "
    "entire released window and is reserved for embargo sizing only (it would "
    "leak the future if used as a feature).",
    "**F7 `f7_amihud_20` uses a zero-volume -> NaN guard.** A zero-volume day "
    "contributes NaN to the trailing Amihud illiquidity mean rather than "
    "dividing by zero; the resulting structural NaNs are never forward-filled.",
    "**F11 macro context is point-in-time publication-lagged (per-class).** "
    "Daily market series (VIX, MOVE, DXY, yields, OAS, breakevens) are available "
    "at their own EOD close (lag 0); weekly EIA inventories are stamped on the "
    "Friday week-ending date and made available 6 calendar days later (a "
    "conservative buffer past the real ~Wed/~Thu release); monthly PMIs are "
    "stamped at month-end and made available on the next business day (the "
    "ISM/Caixin release). A 12-series + 3-spread subset was curated aggressively "
    "from the 22-series workbook (10 series dropped). The z-score is fit on the "
    "FE-train slice of the macro panel after it is reindexed onto the equity "
    "trade-date cadence (so the standardizer reflects the trade calendar, by "
    "design), then frozen forward; momentum is the h-business-day change of the "
    "PIT-applied level (measured on a business-day grid, which diverges from the "
    "equity trading calendar only on market holidays). F11 values are broadcast "
    "identically to all 11 "
    "instruments, so the redundancy map shows F11 internal clustering by "
    "construction.",
    "**Provenance — this is a curated union of three branches.** F1–F11 are the "
    "`signal-deep-dive` base. Folded in (novel-only, de-duplicated): from "
    "**Sreeram** the F12 mean-reversion / path-structure & trend-quality family "
    "(Hurst exponent, variance ratio, Kaufman efficiency ratio, return "
    "autocorrelation, backward trend-scanning t-values, sigma-normalised MA "
    "slope), the F2 Rogers-Satchell range-vol estimator, and the F17 filtered "
    "Gaussian-HMM regime posteriors; from **Harry** the F13 wavelet multiscale "
    "energy bands, the F15 conditional-risk / first-passage family (bootstrap "
    "expected hit time, timeout probability, path tortuosity, semi-vol ratio), "
    "the F16 concept-drift regime-alignment score, the F5 signal entropy / "
    "flip-rate, the F7 Roll spread / Kyle's lambda / overnight gap, and the F9 "
    "cross-asset positioning columns (lead-lag centroid distance, within-class "
    "dispersion z, EWMA implied-correlation z). Duplicated columns (e.g. both "
    "branches' rolling vol / momentum / calendar / signal run-length / GMM "
    "clustering) were dropped in favour of the existing F1–F11 implementations.",
    "**Two standardization regimes.** (1) The fitted TF families "
    "(F3/F4/F11/F16/F17) are standardized at fit time on the FE-train partition "
    "and frozen. (2) Every scale-dependent E-class column carries a parallel "
    "`z_<col>` twin: a per-instrument *causal expanding-window z-score* "
    "(`expanding(min_periods=60)`), which is split-agnostic — it bakes in no "
    "train/test cutoff, so each downstream branch keeps its own split. Bounded / "
    "already-normalized columns (ratios, probabilities, t-statistics, sin/cos, "
    "[-1,1] / [0,1] positions, percentiles, Hurst, wavelet energy fractions) get "
    "no twin. Modellers can use the raw column, its `z_` twin, or both.",
    "**F15 conditional-risk uses DAILY (de-annualised) vol as the barrier "
    "sigma.** The bootstrap first-passage barriers are `±mult·sigma·sqrt(h)` "
    "where sigma is the trailing 20d daily-return std — never the annualised "
    "`f2_vol_20`, which would push almost every simulated path to time out.",
    "**No labels.** This matrix is keyed by `(date, instrument)` over the "
    "nonzero-signal released window with NO triple-barrier (or other) label "
    "column; each downstream branch attaches its own labels for modelling.",
]


def _family_order(feature_cols: list[str]) -> list[str]:
    """Family tags present in ``feature_cols``, in the canonical render order.

    The canonical order follows :data:`_FAMILY_TITLES`; any family tag seen in
    the data but absent from the title map is appended (sorted) so an
    unexpected family is still rendered rather than silently dropped.
    """
    present = {CATALOG[c].family for c in feature_cols if c in CATALOG}
    ordered = [f for f in _FAMILY_TITLES if f in present]
    extra = sorted(present - set(ordered))
    return ordered + extra


def _recon_mse_table(recon_mse: dict) -> list[str]:
    """Render the AE-vs-PCA(k=4) reconstruction-MSE table per asset class.

    Parameters
    ----------
    recon_mse : dict
        Mapping ``asset_class -> {"pca_k4": float, "ae_k4": float}`` (the
        :attr:`LatentBundle.recon_mse` of each class bundle).

    Returns
    -------
    list[str]
        Markdown lines (heading + table). One row per asset class, with the
        lower (better) reconstructor flagged.
    """
    lines = [
        "## Latent reconstruction — Autoencoder vs PCA (k=4)",
        "",
        "Train-block reconstruction MSE on the scaled pooled FE-train feature "
        "matrix, per asset class. The AE is retained as a feature regardless of "
        "which reconstructor is lower (no performance gate).",
        "",
        "| Asset class | PCA(k=4) MSE | AE(k=4) MSE | Lower |",
        "| --- | --- | --- | --- |",
    ]
    for asset_class in sorted(recon_mse):
        stats = recon_mse[asset_class] or {}
        pca = stats.get("pca_k4")
        ae = stats.get("ae_k4")
        if isinstance(pca, (int, float)) and isinstance(ae, (int, float)):
            lower = "AE" if ae < pca else "PCA"
            pca_s = f"{pca:.6g}"
            ae_s = f"{ae:.6g}"
        else:
            lower = "n/a"
            pca_s = "n/a" if pca is None else f"{pca}"
            ae_s = "n/a" if ae is None else f"{ae}"
        lines.append(f"| {asset_class} | {pca_s} | {ae_s} | {lower} |")
    lines.append("")
    return lines


def render_catalog(
    matrix_columns: object,
    recon_mse: dict | None = None,
    path: str = "reports/feature-catalog.md",
) -> None:
    """Render the grouped-by-family feature catalog to a markdown file.

    Writes a markdown document with: a header, the leakage-class legend, the
    four required prose annotations (F1 highest-value, F9 expected-negative,
    F5 trailing-run causality, F7 Amihud zero-volume guard), one table per
    family (name / what it captures / lookback / leakage class / reuse
    pointer) covering every produced feature column, and — when ``recon_mse``
    is supplied — an AE-vs-PCA(k=4) reconstruction-MSE table per asset class.

    Parameters
    ----------
    matrix_columns : iterable of str
        Columns of the produced feature matrix; the four meta columns are
        excluded and every remaining column is rendered (and must have a spec).
    recon_mse : dict, optional
        Mapping ``asset_class -> {"pca_k4": float, "ae_k4": float}``. When
        ``None`` the reconstruction-MSE section is omitted.
    path : str, default "reports/feature-catalog.md"
        Destination markdown path (parent directories must exist).

    Raises
    ------
    AssertionError
        Propagated from :func:`assert_coverage` if the columns and the catalog
        are not in exact 1:1 correspondence.
    """
    # Render only a covered matrix — guarantees every column has a spec.
    assert_coverage(matrix_columns)
    feature_cols = _feature_columns(matrix_columns)
    by_name = {c: CATALOG[c] for c in feature_cols}

    counts = {"E": 0, "TF": 0, "LI": 0}
    for spec in by_name.values():
        counts[spec.leakage_class] = counts.get(spec.leakage_class, 0) + 1

    lines: list[str] = [
        "# Feature Catalog — Triple-Barrier Metamodel",
        "",
        "Auto-generated by `stml.metamodel.catalog.render_catalog`. One row per "
        "produced feature column; the four provenance columns "
        "(`date`, `instrument`, `partition`, `fe_train_end_date`) are metadata, "
        "not features.",
        "",
        f"**{len(feature_cols)} feature columns** across "
        f"{len(_family_order(feature_cols))} families "
        f"(E={counts.get('E', 0)} engineered, TF={counts.get('TF', 0)} fitted, "
        f"LI={counts.get('LI', 0)} label-interface).",
        "",
        "## Leakage classes",
        "",
        "- **E** — engineered, no fit; causal by truncation-invariance "
        "(value at `t` identical on `data[:t+1]` and `data[:T]`).",
        "- **TF** — fitted (GMM / Markov / PCA / KMeans / autoencoder / scaler) "
        "on the FE-train partition only (ends 2021-07-01), transformed causally "
        "with frozen params.",
        "- **LI** — the label-interface subset the deferred triple-barrier label "
        "consumes (`f2_vol_20` = barrier sigma; `f5_trailing_run_length` = run "
        "membership); engineered + causal, flagged as the FE -> label hand-off.",
        "",
        "## Methodology notes",
        "",
    ]
    for note in _ANNOTATIONS:
        lines.append(f"- {note}")
    lines.append("")

    # One grouped table per family, in canonical render order.
    for family in _family_order(feature_cols):
        title = _FAMILY_TITLES.get(family, family)
        fam_cols = [c for c in feature_cols if by_name[c].family == family]
        lines.append(f"## {title}")
        lines.append("")
        lines.append(
            "| Feature | What it captures | Lookback | Leakage | Reuse pointer |"
        )
        lines.append("| --- | --- | --- | --- | --- |")
        for col in fam_cols:
            spec = by_name[col]
            lines.append(
                f"| `{spec.name}` | {spec.what_it_captures} | {spec.lookback} | "
                f"{spec.leakage_class} | `{spec.reuse_pointer}` |"
            )
        lines.append("")

    if recon_mse:
        lines.extend(_recon_mse_table(recon_mse))

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
