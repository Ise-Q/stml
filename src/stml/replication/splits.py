"""
splits.py
=========
Chronological train/val/test partitioning plus the leakage controls that make
the validation set a *statistically honest* gate for the primary-signal
replication study (US-002).

Two ideas drive this module:

  1. **No shuffling, ever.** The signal is a time series, so the only valid
     split is contiguous and time-ordered: ``train`` is the earliest block,
     ``test`` the latest. A 60/20/20 split on the 645-day released panel gives
     ``train[0:387]``, ``val[387:516]``, ``test[516:645]`` (the boundaries are
     ``int(n*0.6)`` and ``int(n*0.8)``). See :func:`chronological_split`.

  2. **Embargo the val edges so no constant-signal run straddles a boundary.**
     The primary signal is piecewise-constant: it holds a single value for a
     run of days, then flips. A run that begins in ``train`` and continues into
     ``val`` (or spans the val/test boundary) leaks the train regime into the
     evaluation window. :func:`embargoed_val` removes ``embargo`` rows at *both*
     val edges; the default ``embargo`` is the 90th-percentile run length
     measured on the **full released period** of that instrument's signal
     (:func:`run_length_p90`), which is long enough that no typical run can
     bridge a boundary. Sizing the embargo from the whole sample is a
     *structural* gap choice (standard purge/embargo practice): it leaks no
     return/label information into the search -- only performance and threshold
     *calibration* must stay strictly train-only. Train-only p90 is also
     unstable on low-run instruments (ng1s's train block holds a few 80+ day
     COVID-era flat runs, inflating its train p90 to ~77 and vacuuming the val
     window); the full-period p90 (~33) is the value the work plan verified.

Effective sample size
---------------------
The primary signal does not give ``len(series)`` independent observations: a
40-day flat run is *one* regime-call, not 40. :func:`n_eff` counts the maximal
constant-value runs, i.e. the number of independent regime decisions. This is
the denominator that any skill statistic on the val set must respect.

**Implementation note #1 (gating uses the POST-embargo window).** The
``n_eff`` that gates an instrument's validation result is computed on the
*embargoed* val window, never the raw one::

    gateable_n_eff = n_eff(signal.iloc[embargoed_val(signal, split)])

Embargoing removes whole runs at each edge, so the post-embargo ``n_eff`` is
strictly ``<=`` the raw val ``n_eff`` (e.g. on the released panel ng1s drops
5 -> ~2, ho1s 15 -> ~8, cl1s 11 -> ~9). ng1s, the most persistent instrument,
retains only ~2 leakage-free regime-calls in val -- below any sane gating
floor -- so it is reported as low-power and folded into asset-class pooling
rather than given a standalone verdict. As a defensive guard, if an (explicit)
embargo is so wide that ``2 * embargo >= len(val)``, ``embargoed_val`` returns
an empty array, which callers must treat as "no leakage-free regime-call
available" rather than an error.

The released split is INSIDE the public data; it is not the coursework's hidden
Jul-Dec 2022 set. :func:`get_test` is a deliberate tripwire so the test block is
never touched during model development.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd


class Split(NamedTuple):
    """A contiguous, time-ordered chronological partition of a date axis.

    The ``*_idx`` fields are integer positions into the original date array
    (suitable for ``.iloc`` / fancy indexing); the ``*_dates`` fields are the
    corresponding ``DatetimeIndex`` slices. The three index blocks are
    contiguous, non-overlapping, strictly increasing, and their union is the
    full ``range(0, n)``.
    """

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex


def chronological_split(
    dates: pd.DatetimeIndex | pd.Series | np.ndarray | list,
    fracs: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> Split:
    """Split a date axis into contiguous train/val/test blocks -- NO shuffle.

    Cut points are ``int(n * fracs[0])`` and ``int(n * (fracs[0] + fracs[1]))``,
    so a 645-day axis under the default ``(0.6, 0.2, 0.2)`` yields
    ``train[0:387]``, ``val[387:516]``, ``test[516:645]``. Because the cuts use
    ``int`` (floor), the test block absorbs any rounding remainder and the three
    blocks always tile ``range(0, n)`` exactly.

    Parameters
    ----------
    dates : index-like of timestamps, assumed already in ascending order.
    fracs : (train, val, test) fractions; the first two drive the cut points.

    Raises
    ------
    ValueError : if ``fracs`` has the wrong length, is non-positive, or does not
        sum to 1, or if ``dates`` is empty.
    """
    if len(fracs) != 3:
        raise ValueError(f"fracs must have 3 entries, got {len(fracs)}")
    if any(f <= 0 for f in fracs):
        raise ValueError(f"fracs must be strictly positive, got {fracs}")
    if not np.isclose(sum(fracs), 1.0):
        raise ValueError(f"fracs must sum to 1.0, got {sum(fracs)}")

    idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    n = len(idx)
    if n == 0:
        raise ValueError("dates is empty")

    cut_train = int(n * fracs[0])
    cut_val = int(n * (fracs[0] + fracs[1]))

    train_idx = np.arange(0, cut_train)
    val_idx = np.arange(cut_train, cut_val)
    test_idx = np.arange(cut_val, n)

    return Split(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_dates=idx[train_idx],
        val_dates=idx[val_idx],
        test_dates=idx[test_idx],
    )


def _run_lengths(signal_series: pd.Series | np.ndarray | list) -> np.ndarray:
    """Lengths of the maximal constant-value runs in ``signal_series``.

    A run is a maximal block of equal adjacent values. ``[1, 1, 0, 0, 0, 1]``
    has runs of length ``[2, 3, 1]``. NaNs are compared by identity, so a run is
    broken at any value change including to/from NaN. Returns an empty array for
    an empty input.
    """
    a = np.asarray(getattr(signal_series, "to_numpy", lambda: signal_series)())
    if a.size == 0:
        return np.empty(0, dtype=int)
    change_at = np.flatnonzero(a[1:] != a[:-1]) + 1
    bounds = np.concatenate(([0], change_at, [a.size]))
    return np.diff(bounds)


def run_length_p90(signal_series: pd.Series | np.ndarray | list) -> int:
    """90th-percentile length of the maximal constant-value runs in the series.

    Uses :func:`numpy.percentile` (linear interpolation) on the run-length
    distribution, floored to an ``int``. This is the natural embargo width: a
    window of ``run_length_p90`` days is wider than ~90% of the signal's flat
    runs, so embargoing that many rows at a boundary removes essentially every
    run that could straddle it. Returns 0 for an empty series.
    """
    lengths = _run_lengths(signal_series)
    if lengths.size == 0:
        return 0
    return int(np.percentile(lengths, 90))


def n_eff(signal_series: pd.Series | np.ndarray | list) -> int:
    """Effective sample size = number of constant-value runs in the series.

    A piecewise-constant signal yields one *independent* regime-call per run,
    not one per row: a 40-day flat stretch is a single decision. This count
    equals ``1 + (number of adjacent value changes)`` and is the honest
    denominator for any skill statistic on the window. Returns 0 for an empty
    series.

    Per implementation note #1, the *gateable* ``n_eff`` for an instrument's
    validation set is this function applied to the POST-embargo window::

        gateable = n_eff(signal.iloc[embargoed_val(signal, split)])
    """
    a = np.asarray(getattr(signal_series, "to_numpy", lambda: signal_series)())
    if a.size == 0:
        return 0
    return 1 + int(np.count_nonzero(a[1:] != a[:-1]))


def embargoed_val(
    signal_inst_series: pd.Series | np.ndarray | list,
    split: Split,
    embargo: int | None = None,
) -> np.ndarray:
    """Val positions remaining after embargoing ``embargo`` rows at BOTH edges.

    The returned integer positions index into the same axis as ``split`` (the
    full date array), so they can be fed straight to ``signal.iloc[...]``. The
    block is ``[val_start + embargo, val_end - embargo)``: removing ``embargo``
    rows adjacent to the train/val boundary and ``embargo`` rows adjacent to the
    val/test boundary guarantees no retained run can straddle either boundary.

    Parameters
    ----------
    signal_inst_series : the full (whole-axis) signal series for one instrument.
        Only its TRAIN portion is consulted, and only when ``embargo is None``.
    split : the :class:`Split` whose ``val_idx`` / ``train_idx`` define the
        windows.
    embargo : rows to drop at each edge. If ``None`` (default) it is
        :func:`run_length_p90` of the **full released period** of this
        instrument's signal -- a structural gap that scales to how long this
        instrument tends to hold a constant value (full-period sizing is stable
        and leaks no answer; see the module docstring).

    Returns
    -------
    np.ndarray : the post-embargo val positions, in increasing order. **Empty**
        when ``2 * embargo >= len(val)`` (a defensively-handled edge case for an
        oversized explicit embargo); callers must treat an empty window as "no
        leakage-free regime-call available", not as an error.

    Raises
    ------
    ValueError : if an explicit ``embargo`` is negative.
    """
    if embargo is not None and embargo < 0:
        raise ValueError(f"embargo must be non-negative, got {embargo}")

    a = np.asarray(
        getattr(signal_inst_series, "to_numpy", lambda: signal_inst_series)()
    )
    if embargo is None:
        # Structural gap-sizing from the FULL released period (see module
        # docstring): stable, and leaks no return/label info into the search
        # (only performance/threshold *calibration* must stay train-only).
        embargo = run_length_p90(a)

    val_start = int(split.val_idx[0])
    val_end = int(split.val_idx[-1]) + 1  # exclusive
    lo = val_start + embargo
    hi = val_end - embargo
    if lo >= hi:
        return np.empty(0, dtype=int)
    return np.arange(lo, hi)


def get_test(split: Split, final_confirmation: bool = False) -> np.ndarray:
    """Return the test positions -- but only behind an explicit confirmation.

    The test block is the final, untouchable evaluation set; accidentally
    reading it during development is the classic leakage failure. This tripwire
    forces an explicit ``final_confirmation=True`` so test access is always a
    deliberate, auditable act.

    Raises
    ------
    RuntimeError : unless ``final_confirmation`` is True.
    """
    if not final_confirmation:
        raise RuntimeError(
            "Refusing to expose the test set. The test block is the final "
            "evaluation set and must not be touched during model development. "
            "Pass final_confirmation=True only for the one-time final report."
        )
    return split.test_idx
