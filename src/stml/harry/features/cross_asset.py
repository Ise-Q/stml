"""cross_asset.py — features that read this instrument against the panel.

ECONOMIC INTUITION
==================
The signal-deep-dive replication study showed cross-asset mean |corr|
≈ 0.09 between primary signals — they are nearly independent across the
panel. But the *returns* are not independent: an asset-class shock
(equity sell-off, energy spike, metals rotation) moves multiple
instruments together. These features measure how *this* instrument sits
within its peer group:

* ``distance_to_lead_lag_centroid``  — L2 distance over a 126-day
  trailing window between the instrument's returns and a peer-mean
  series shifted by ``lag`` days. Big distances flag the instrument as
  out-of-step with the panel (a "leader" or "laggard"); small distances
  flag conformity. Lag 1 captures the next-day-execution convention's
  natural horizon.
* ``asset_class_dispersion_z``       — z-score of the trailing
  cross-sectional standard deviation of returns within the instrument's
  *asset class*. Spikes flag intra-class divergence days (e.g. silver
  decoupling from gold); flat values flag uniform regime.
* ``ewma_implied_corr_z``            — z-score of the EWMA-smoothed
  average pairwise correlation between this instrument and every other
  in the panel. A "crisis indicator" — pairwise correlations cluster
  toward 1 in market-wide risk-off events.

These features explicitly *complement* the per-instrument time-series
features: they encode what the rest of the panel is doing today, which
the time-series view cannot see.

CAUSALITY CONTRACT
==================
Every output at row ``t`` uses only panel data at indices ``<= t``.
EWMA is one-pass with ``adjust=False``; cross-sectional aggregations
are contemporaneous; rolling z-scores look backward only.

WARMUP WINDOWS
==============
* ``distance_to_lead_lag_centroid``  : ``window + lag - 1`` rows.
* ``asset_class_dispersion_z``       : ``window`` rows (rolling-z needs a
  full window of dispersion samples).
* ``ewma_implied_corr_z``            : ``window`` rows.

CITATIONS
=========
* Pollet, J. & Wilson, M. (2010) "Average Correlation and Stock Market
  Returns", Journal of Financial Economics 96: 364–380 — the
  implied-corr / market-stress link.
* Lopez de Prado (2018) "Advances in Financial Machine Learning",
  Chapter 25 (multi-asset features).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "ASSET_CLASSES",
    "distance_to_lead_lag_centroid",
    "asset_class_dispersion_z",
    "ewma_implied_corr_z",
]

#: Default asset-class membership for the 11-instrument universe — passed
#: to :func:`asset_class_dispersion_z` if no override is provided.
ASSET_CLASSES: dict[str, str] = {
    "es1s": "equity", "nq1s": "equity", "fesx1s": "equity",
    "cl1s": "energy", "ho1s": "energy", "rb1s": "energy", "ng1s": "energy",
    "gc1s": "metals", "si1s": "metals", "hg1s": "metals", "pl1s": "metals",
}


def _validate(returns_panel: pd.DataFrame, instrument: str) -> None:
    if instrument not in returns_panel.columns:
        raise KeyError(
            f"instrument {instrument!r} not a column of returns_panel"
        )


def distance_to_lead_lag_centroid(
    returns_panel: pd.DataFrame,
    instrument: str,
    *,
    lag: int = 1,
    window: int = 126,
) -> pd.Series:
    """L2 distance over a trailing window between ``r[instrument]_t`` and
    ``mean_peers(r[peer]_{t - lag})``.

    At each row ``t``::

        centroid_t = mean over peers of r[peer]_{t - lag}
        dist_t     = sqrt( mean over u in [t-window+1, t] of
                            (r[instrument]_u - centroid_u)^2 )

    Output is non-negative; NaN for the first ``window + lag - 1`` rows.
    A small value means the instrument tracks the lagged peer mean
    closely; a large value flags it as out-of-step.
    """
    _validate(returns_panel, instrument)
    if lag < 0:
        raise ValueError(f"lag must be >= 0, got {lag}")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    inst_r = returns_panel[instrument].astype("float64")
    peer_cols = [c for c in returns_panel.columns if c != instrument]
    if not peer_cols:
        return pd.Series(
            np.nan, index=returns_panel.index,
            name=f"dist_lead_lag_centroid_lag{lag}_w{window}",
        )
    centroid = returns_panel[peer_cols].mean(axis=1).shift(lag)
    diff_sq = (inst_r - centroid) ** 2
    out = np.sqrt(
        diff_sq.rolling(window, min_periods=window).mean()
    )
    return out.rename(f"dist_lead_lag_centroid_lag{lag}_w{window}")


def asset_class_dispersion_z(
    returns_panel: pd.DataFrame,
    instrument: str,
    classes: dict[str, str] | None = None,
    *,
    window: int = 63,
) -> pd.Series:
    """Z-score of the trailing cross-sectional std of returns within the
    instrument's asset class.

    At each row ``t``::

        peers       = {i : classes[i] == classes[instrument]}
        dispersion_t = std over peers of r[peer]_t       (cross-section)
        z_t         = (dispersion_t - rolling_mean(dispersion, window))
                       / rolling_std(dispersion, window)

    ``classes`` defaults to :data:`ASSET_CLASSES`. If the asset class has
    fewer than two members the result is NaN everywhere (no within-class
    cross-section). NaN for the first ``window`` rows.
    """
    _validate(returns_panel, instrument)
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    cls_map = ASSET_CLASSES if classes is None else dict(classes)
    if instrument not in cls_map:
        raise KeyError(f"instrument {instrument!r} not in classes mapping")
    my_class = cls_map[instrument]
    peers = [
        c for c in returns_panel.columns
        if cls_map.get(c) == my_class
    ]
    if len(peers) < 2:
        return pd.Series(
            np.nan, index=returns_panel.index,
            name=f"asset_class_dispersion_z_w{window}",
        )
    dispersion = returns_panel[peers].std(axis=1, ddof=0).astype("float64")
    mu = dispersion.rolling(window, min_periods=window).mean()
    sd = dispersion.rolling(window, min_periods=window).std(ddof=0)
    z = (dispersion - mu) / sd.replace(0.0, np.nan)
    return z.rename(f"asset_class_dispersion_z_w{window}")


def ewma_implied_corr_z(
    returns_panel: pd.DataFrame,
    instrument: str,
    *,
    halflife: int = 20,
    window: int = 252,
) -> pd.Series:
    """Z-score of EWMA-smoothed average pairwise correlation between this
    instrument and every other in the panel.

    At each row ``t``::

        corr_t      = mean over peers of  ewm-corr_h(r[instrument], r[peer])
                                          evaluated at t (one-sided, causal)
        z_t         = (corr_t - rolling_mean(corr, window))
                       / rolling_std(corr, window)

    A positive z-score means the instrument is more strongly correlated
    with the rest of the panel than its recent history; a "crisis spike"
    signature. NaN for the first ``window`` rows.
    """
    _validate(returns_panel, instrument)
    if halflife < 1:
        raise ValueError(f"halflife must be >= 1, got {halflife}")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    inst_r = returns_panel[instrument].astype("float64")
    peer_cols = [c for c in returns_panel.columns if c != instrument]
    if not peer_cols:
        return pd.Series(
            np.nan, index=returns_panel.index,
            name=f"ewma_implied_corr_z_hl{halflife}_w{window}",
        )
    pair_corrs: list[pd.Series] = []
    for peer in peer_cols:
        c = inst_r.ewm(halflife=halflife, adjust=False).corr(
            returns_panel[peer].astype("float64")
        )
        pair_corrs.append(c.astype("float64"))
    implied = pd.concat(pair_corrs, axis=1).mean(axis=1)
    mu = implied.rolling(window, min_periods=window).mean()
    sd = implied.rolling(window, min_periods=window).std(ddof=0)
    z = (implied - mu) / sd.replace(0.0, np.nan)
    return z.rename(f"ewma_implied_corr_z_hl{halflife}_w{window}")


# --------------------------------------------------------------------------- #
# Causality harness registry                                                  #
# --------------------------------------------------------------------------- #
# Smaller windows here keep the parametrised harness fast; production
# defaults of 126 / 63 / 252 remain in the function signatures.
CAUSALITY_REGISTRATIONS: list[dict] = [
    {
        "name": "distance_to_lead_lag_centroid",
        "module": __name__,
        "func": "distance_to_lead_lag_centroid",
        "adapter": "panel",
        "kwargs": {"instrument": "es1s", "lag": 1, "window": 60},
        "warmup": 60,  # window + lag - 1 = 60 + 1 - 1
        "data_kind": "returns_panel",
    },
    {
        "name": "asset_class_dispersion_z",
        "module": __name__,
        "func": "asset_class_dispersion_z",
        "adapter": "panel",
        "kwargs": {
            "instrument": "es1s",
            "classes": dict(ASSET_CLASSES),
            "window": 60,
        },
        "warmup": 60,
        "data_kind": "returns_panel",
    },
    {
        "name": "ewma_implied_corr_z",
        "module": __name__,
        "func": "ewma_implied_corr_z",
        "adapter": "panel",
        "kwargs": {"instrument": "es1s", "halflife": 20, "window": 100},
        "warmup": 100,
        "data_kind": "returns_panel",
    },
]
