"""
nav.py
======
Cumulative log-NAV construction and replica-vs-target discrepancy metrics.

The PnL convention follows the contract in ``.omc/scratch/CONTRACT.md``:
``PnL_t = s_t * r_{t+1}`` where ``r_{t+1}`` is already attached to signal day
``t`` by :func:`stml.replication.align.align_instrument` (next_day convention).
**Do not shift** the aligned return series again here.

Public API
----------
- :func:`nav_series`       -- cumulative log-NAV from a signal and aligned returns.
- :func:`nav_discrepancy`  -- scalar metrics comparing a replica to a target signal.
- :func:`nav_from_raw`     -- convenience: align then compute NAV in one call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["nav_series", "nav_discrepancy", "nav_from_raw"]


def nav_series(
    signal: pd.Series,
    aligned_ret: pd.Series,
) -> pd.Series:
    """Compute the cumulative log-NAV for a signal against aligned returns.

    Parameters
    ----------
    signal : pd.Series
        Per-day position signal in ``{-1, 0, 1}`` (or any float).  Index must
        be date-like and must overlap with ``aligned_ret``.
    aligned_ret : pd.Series
        Per-day *already-aligned* log-returns.  For the ``next_day`` convention
        this series already carries ``r_{t+1}`` on day ``t`` (produced by
        :func:`stml.replication.align.align_instrument`).  **Do not shift again.**

    Returns
    -------
    pd.Series
        Cumulative log-NAV indexed on the *common* dates of ``signal`` and
        ``aligned_ret``.  Defined as ``cumsum(signal_t * aligned_ret_t)``.

    Notes
    -----
    The two inputs are first inner-joined on their index so that dates present
    in one but not the other are silently excluded.  This is consistent with
    the alignment contract: structural NaNs remain structural.
    """
    # Align on the intersection of the two indices
    common = signal.index.intersection(aligned_ret.index)
    sig = signal.loc[common]
    ret = aligned_ret.loc[common]

    pnl = sig * ret
    return pnl.cumsum()


def nav_discrepancy(
    replica_signal: pd.Series,
    target_signal: pd.Series,
    aligned_ret: pd.Series,
) -> dict[str, float]:
    """Compute scalar discrepancy metrics between a replica and a target NAV curve.

    All three inputs are first aligned on their common dates before any metric
    is computed.

    Parameters
    ----------
    replica_signal : pd.Series
        Candidate signal series (values in ``{-1, 0, 1}`` typically).
    target_signal : pd.Series
        Ground-truth signal series.
    aligned_ret : pd.Series
        Per-day aligned log-returns (next_day convention already applied; no
        further shifting).

    Returns
    -------
    dict with keys:

    ``cumnav_ssd_norm``
        Sum of squared differences between the two cumulative log-NAV curves,
        divided by the number of common dates (normalised by series length so
        the metric is comparable across instruments with different histories).
    ``cumnav_ssd_norm_vs_flat``
        Same SSD but comparing the *target* cumulative log-NAV against a flat
        (all-zeros) NAV.  Serves as a reference scale: a perfect replica gives
        ``cumnav_ssd_norm / cumnav_ssd_norm_vs_flat ≈ 0``.  Normalised by the
        same series length.
    ``tracking_error_ann``
        Annualised standard deviation of per-step PnL differences
        ``(pnl_replica_t − pnl_target_t)``.  Computed as
        ``std(pnl_diff) * sqrt(252)``.
    ``increment_corr``
        Pearson correlation of per-step PnL increments
        ``(signal_replica_t * ret_t)`` vs ``(signal_target_t * ret_t)``.
        Returns ``nan`` if either series is constant.
    ``retained_fraction``
        Fraction of dates common to *all three* inputs relative to the length
        of ``aligned_ret`` (i.e. fraction of available return days used after
        three-way alignment).
    """
    # Three-way intersection
    common = (
        replica_signal.index
        .intersection(target_signal.index)
        .intersection(aligned_ret.index)
    )
    n = len(common)

    sig_r = replica_signal.loc[common]
    sig_t = target_signal.loc[common]
    ret = aligned_ret.loc[common]

    pnl_r = sig_r * ret
    pnl_t = sig_t * ret

    cum_r = pnl_r.cumsum()
    cum_t = pnl_t.cumsum()

    # Flat / zero NAV reference
    flat_nav = pd.Series(0.0, index=common)

    ssd_norm = float(((cum_r - cum_t) ** 2).sum() / n) if n > 0 else float("nan")
    ssd_norm_vs_flat = float(((cum_t - flat_nav) ** 2).sum() / n) if n > 0 else float("nan")

    pnl_diff = pnl_r - pnl_t
    tracking_error_ann = float(pnl_diff.std() * np.sqrt(252)) if n > 1 else float("nan")

    if n > 1 and pnl_r.std() > 0 and pnl_t.std() > 0:
        increment_corr = float(pnl_r.corr(pnl_t))
    else:
        increment_corr = float("nan")

    retained_fraction = n / len(aligned_ret) if len(aligned_ret) > 0 else float("nan")

    return {
        "cumnav_ssd_norm": ssd_norm,
        "cumnav_ssd_norm_vs_flat": ssd_norm_vs_flat,
        "tracking_error_ann": tracking_error_ann,
        "increment_corr": increment_corr,
        "retained_fraction": retained_fraction,
    }


def nav_from_raw(
    signals_wide: pd.DataFrame,
    ohlcv_long: pd.DataFrame,
    instrument: str,
    convention: str = "next_day",
) -> pd.Series:
    """Convenience: align an instrument then return its cumulative log-NAV.

    This helper calls :func:`stml.replication.align.align_instrument` internally
    so callers do not need to re-implement alignment logic.

    Parameters
    ----------
    signals_wide : pd.DataFrame
        Wide signals frame -- columns are ``date`` plus one column per
        instrument containing values in ``{-1, 0, 1}``.
    ohlcv_long : pd.DataFrame
        Long OHLCV frame (artifact rows already removed by
        :func:`stml.io.load_clean_data`).
    instrument : str
        Instrument to compute NAV for.
    convention : str
        Holding convention forwarded to :func:`~stml.replication.align.align_instrument`.

    Returns
    -------
    pd.Series
        Cumulative log-NAV indexed by date.
    """
    from stml.replication.align import align_instrument

    result = align_instrument(signals_wide, ohlcv_long, instrument, convention=convention)
    frame = result.frame.set_index("date")
    signal = frame["signal"]
    ret = frame["ret"]
    return nav_series(signal, ret)
