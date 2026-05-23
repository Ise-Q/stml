"""
xsection.py
===========
Cross-sectional feature group F9 for the metamodel feature-engineering layer.

This module produces three date-indexed features for a single instrument by
operating on the full universe panel, all strictly look-ahead-free (truncation-
invariant): the instrument's daily cross-sectional rank of its trailing log
return, the universe size on each day, and the rolling pairwise correlation with
its asset-class peers.

The signal studied in C1 is mean-reversion, so the default ``score="reversal"``
negates the trailing return before ranking (highest-reversal = highest negative
return = likeliest to bounce); this matches :func:`archetypes.generate_panel`
with ``score="reversal"``.

Expected-negative diagnostic
-----------------------------
``f9_pair_corr_mean`` is documented as *expected-negative* for the feature
catalog: the cross-asset mean |corr| is approximately 0.09 per C1, meaning that
even within a class the rolling pair correlations are low.  This is not a bug
— the cross-sectional rank was chosen precisely because it captures
cross-instrument structure that pure time-series features miss; its low
pairwise correlation proves the feature is nearly independent of the others.

Leakage rule (CONTRACT §0, rule 1)
------------------------------------
All operations use trailing windows only:

* ``logc.shift(lookback)`` — strictly prior close, shift >= 0.
* Day-by-day ranking: for date t, only the row at t is used (already built
  from info <= t in the wide-return matrix above).
* ``rolling_pair_corr`` is right-aligned in :mod:`stml.na_checks`.

Truncation-invariance assertion: for any t, the feature values computed from
``ohlcv_all[ohlcv_all["date"] <= t]`` equal those computed on the full panel.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from stml.na_checks import native_returns, rolling_pair_corr, wide_returns

__all__ = ["xsection_features", "ASSET_CLASS_PEERS"]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Asset-class peer map (CONTRACT §2, D5 scope table)
# EQ: {es1s, nq1s, fesx1s}
# EN: {cl1s, ho1s, rb1s, ng1s}
# ME: {gc1s, si1s, hg1s, pl1s}
# ---------------------------------------------------------------------------
ASSET_CLASS_PEERS: dict[str, list[str]] = {
    "es1s":   ["es1s", "nq1s", "fesx1s"],
    "nq1s":   ["es1s", "nq1s", "fesx1s"],
    "fesx1s": ["es1s", "nq1s", "fesx1s"],
    "cl1s":   ["cl1s", "ho1s", "rb1s", "ng1s"],
    "ho1s":   ["cl1s", "ho1s", "rb1s", "ng1s"],
    "rb1s":   ["cl1s", "ho1s", "rb1s", "ng1s"],
    "ng1s":   ["cl1s", "ho1s", "rb1s", "ng1s"],
    "gc1s":   ["gc1s", "si1s", "hg1s", "pl1s"],
    "si1s":   ["gc1s", "si1s", "hg1s", "pl1s"],
    "hg1s":   ["gc1s", "si1s", "hg1s", "pl1s"],
    "pl1s":   ["gc1s", "si1s", "hg1s", "pl1s"],
}


def _build_wide_trailing_return(ohlcv_all: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Wide ``date x instrument`` trailing ``lookback``-day log return.

    Each instrument's log return is computed on its own dense calendar series,
    then all series are joined into a wide frame.  Remaining NaNs are structural
    (warm-up or other-venue holiday) and are left as NaN — they simply do not
    rank that day.

    Parameters
    ----------
    ohlcv_all : pd.DataFrame
        Long OHLCV for the whole universe.  Must contain columns ``instrument``,
        ``date``, and ``close``.
    lookback : int
        Number of trading days for the trailing return window (on each
        instrument's own dense calendar).

    Returns
    -------
    pd.DataFrame
        Wide frame, date-indexed, one column per instrument; values are the
        ``lookback``-day trailing log returns (strictly look-ahead-free).
    """
    parts: list[pd.Series] = []
    for inst, grp in ohlcv_all.groupby("instrument"):
        close = (
            grp[["date", "close"]]
            .dropna(subset=["close"])
            .drop_duplicates("date")
            .sort_values("date")
            .set_index("date")["close"]
            .astype(float)
        )
        logc = np.log(close)
        trailing = logc - logc.shift(lookback)  # info <= t, trailing
        parts.append(trailing.rename(inst))

    wide = pd.concat(parts, axis=1, sort=True).sort_index()  # sort=True: align date index
    return wide


def xsection_features(
    ohlcv_all: pd.DataFrame,
    instrument: str,
    lookback: int = 20,
    score: str = "reversal",
    peers: list[str] | None = None,
    pair_window: int = 120,
) -> pd.DataFrame:
    """Compute cross-sectional F9 features for a single instrument.

    All features are date-indexed over the instrument's own trading calendar
    and are strictly look-ahead-free (truncation-invariant): the value at any
    date t is determined solely by closes at dates <= t.

    Parameters
    ----------
    ohlcv_all : pd.DataFrame
        Long OHLCV for the **whole universe** (all 11 instruments).  Columns
        must include ``date``, ``instrument``, ``close``.  The full history
        should be passed so rolling warm-up uses real bars.
    instrument : str
        The target instrument to produce features for (e.g. ``"es1s"``).
    lookback : int, optional
        Trailing window in trading days for the per-instrument log return used
        in the cross-sectional rank.  Default 20.
    score : {'reversal', 'momentum'}, optional
        ``"reversal"`` (default, per C1 signal finding): the trailing return is
        **negated** before ranking, so the instrument with the largest negative
        return ranks first (highest reversal potential).
        ``"momentum"``: higher positive return ranks first.
    peers : list[str] or None, optional
        Instruments to use for the rolling pairwise correlation in
        ``f9_pair_corr_mean``.  If None, uses the asset-class peer list from
        :data:`ASSET_CLASS_PEERS`.  The target instrument itself is excluded
        from the correlation (self-correlation would always be 1.0).
    pair_window : int, optional
        Rolling window (in days on the intersection calendar) for the pairwise
        correlation.  Default 120.

    Returns
    -------
    pd.DataFrame
        Date-indexed (same index as the instrument's close series) with columns:

        ``f9_xsect_rank``
            Cross-sectional rank of the instrument's trailing ``lookback``-day
            log return (negated when ``score="reversal"``) among all universe
            members with a finite score on that day.  Normalised to **[-1, 1]**:
            rank 0 (lowest) maps to -1.0, rank n-1 (highest) maps to +1.0.
            A rank of 0.0 means the instrument is exactly at the median.
            Days where the instrument itself has no finite score → NaN
            (structural, never filled).

        ``f9_xsection_universe_size``
            Number of instruments with a finite trailing score on that day.
            Integer-valued in [1, 11].  Varies across days due to the ragged
            multi-venue calendar (~24 days in the 645-day released window have
            fewer than 11 members present).

        ``f9_pair_corr_mean``
            Mean of the rolling pairwise correlation between ``instrument`` and
            each of its asset-class peers (excluding itself), computed on the
            INTERSECTION of their trading calendars.  Expected near 0 (≈ 0.09
            mean |corr| per C1) — documented as expected-negative diagnostic.
            NaN until the rolling window is filled; bounded in [-1, 1] where finite.

    Notes
    -----
    **Normalisation convention ([-1, 1]):**
    Given ``n`` instruments with finite scores on a day and this instrument's
    0-based rank ``r`` (0 = worst score, n-1 = best), the normalised rank is::

        f9_xsect_rank = 2 * r / (n - 1) - 1   when n >= 2
        f9_xsect_rank = 0.0                     when n == 1 (sole member)

    This is consistent with the Donchian position formula in
    :func:`archetypes._score_breakout_donchian` (2*(pos - lo)/(hi - lo) - 1).

    **Truncation-invariance proof:**
    The trailing return at date t depends only on close[t] and close[t-lookback].
    Ranking is performed row-by-row using only the value at t.  Neither step
    looks ahead, so truncating ohlcv_all to dates <= t reproduces identical
    feature values at t (for all t with >= lookback bars of history).
    """
    if instrument not in ohlcv_all["instrument"].unique():
        raise ValueError(f"Instrument {instrument!r} not found in ohlcv_all.")

    if score not in ("reversal", "momentum"):
        raise ValueError(f"score must be 'reversal' or 'momentum', got {score!r}")

    # ------------------------------------------------------------------
    # Build wide trailing returns (look-ahead-free: value at t uses info <= t)
    # ------------------------------------------------------------------
    wide = _build_wide_trailing_return(ohlcv_all, lookback)

    if instrument not in wide.columns:
        raise ValueError(
            f"Instrument {instrument!r} produced no return series — check ohlcv_all."
        )

    # Negate for mean-reversion (highest -return = most oversold = highest rank)
    wide_scored = -wide if score == "reversal" else wide.copy()

    # ------------------------------------------------------------------
    # F9a: cross-sectional rank normalised to [-1, 1]
    # F9b: universe size (finite-score count per day)
    # ------------------------------------------------------------------
    arr = wide_scored.to_numpy(dtype=float)
    cols = list(wide_scored.columns)
    inst_col = cols.index(instrument)

    n_rows = arr.shape[0]
    xsect_rank = np.full(n_rows, np.nan)
    universe_size = np.zeros(n_rows, dtype=np.int32)

    for r in range(n_rows):
        row = arr[r]
        finite_mask = np.isfinite(row)
        n = int(finite_mask.sum())
        universe_size[r] = n

        if n == 0:
            continue

        if not finite_mask[inst_col]:
            # This instrument has no finite score today — structural NaN
            continue

        # Rank among finite instruments (0-based, ascending: 0 = lowest score)
        finite_idx = np.where(finite_mask)[0]
        finite_vals = row[finite_idx]
        # argsort of finite_vals gives ascending order indices into finite_idx
        order = finite_idx[np.argsort(finite_vals, kind="stable")]
        # rank of our instrument within the finite set
        pos = int(np.where(order == inst_col)[0][0])

        if n == 1:
            xsect_rank[r] = 0.0
        else:
            # Normalise to [-1, 1]: 0/(n-1) -> -1, (n-1)/(n-1) -> +1
            xsect_rank[r] = 2.0 * pos / (n - 1) - 1.0

    rank_series = pd.Series(xsect_rank, index=wide_scored.index, name="f9_xsect_rank")
    size_series = pd.Series(universe_size, index=wide_scored.index, name="f9_xsection_universe_size")

    # Re-index to the instrument's own close calendar (drop dates where the
    # instrument was never in the panel at all — structural absence)
    inst_dates = (
        ohlcv_all.loc[ohlcv_all["instrument"] == instrument, "date"]
        .dropna()
        .sort_values()
        .unique()
    )
    inst_index = pd.DatetimeIndex(inst_dates)

    rank_series = rank_series.reindex(inst_index)
    size_series = size_series.reindex(inst_index)

    # ------------------------------------------------------------------
    # F9c: mean rolling pair-correlation with asset-class peers
    # ------------------------------------------------------------------
    if peers is None:
        peer_list = ASSET_CLASS_PEERS.get(instrument, [])
    else:
        peer_list = list(peers)

    # Exclude self-correlation
    other_peers = [p for p in peer_list if p != instrument]

    # Build wide returns for pair-corr (use 1-day returns, not lookback-returns)
    rets = native_returns(ohlcv_all, kind="log")
    wide_ret = wide_returns(rets)

    corr_series_list: list[pd.Series] = []
    for peer in other_peers:
        if peer not in wide_ret.columns:
            log.warning("Peer %r not found in wide returns; skipping.", peer)
            continue
        if instrument not in wide_ret.columns:
            log.warning("Instrument %r not found in wide returns; skipping pair-corr.", instrument)
            break
        c = rolling_pair_corr(wide_ret, instrument, peer, window=pair_window)
        corr_series_list.append(c)

    if corr_series_list:
        # Mean across peers — pairwise-complete for each pair separately then averaged
        corr_df = pd.concat(corr_series_list, axis=1, sort=True)
        pair_corr_mean = corr_df.mean(axis=1, skipna=True)
    else:
        # No peers available (degenerate case) — all NaN
        pair_corr_mean = pd.Series(np.nan, index=wide_ret.index, dtype=float)

    pair_corr_mean.name = "f9_pair_corr_mean"
    pair_corr_mean = pair_corr_mean.reindex(inst_index)

    # ------------------------------------------------------------------
    # Assemble output frame
    # ------------------------------------------------------------------
    out = pd.DataFrame(
        {
            "f9_xsect_rank": rank_series,
            "f9_xsection_universe_size": size_series,
            "f9_pair_corr_mean": pair_corr_mean,
        },
        index=inst_index,
    )
    out.index.name = "date"
    return out
