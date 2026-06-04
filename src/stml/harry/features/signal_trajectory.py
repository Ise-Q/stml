"""signal_trajectory.py — features derived from the primary signal's
own trajectory through time.

ECONOMIC INTUITION
==================
The primary signal ``s_t`` is the only labelled input the meta-model
sees that is not derived from price. A meta-model that can read the
signal's recent STRUCTURE — how long it has held its current position,
how often it has been flipping, how noisy vs persistent the trailing 60
days have been, how its naïve PnL has been trending — gets a direct
conditioning variable for "is this bet worth taking right now". When
the signal has been hopping noisily (high entropy, high flip rate, weak
cumulative PnL) it is plausibly less reliable than when it has held a
clean run. None of these features are computed by Sreeram's or
signal-deep-dive's branch.

CAUSALITY CONTRACT
==================
Every output at row ``t`` uses only values of ``s`` (and ``r``, where
applicable) at indices ``<= t``. The first row's flip is undefined (no
prior); it is set to NaN so the trailing flip-rate window correctly
excludes that bar from its count.

WARMUP WINDOWS
==============
* ``signal_run_length``    : 0   (defined from row 0).
* ``time_since_last_flip`` : 0   (defined from row 0; first row = 0).
* ``signal_entropy_20d``   : 19  (needs 20 trailing values).
* ``signal_flip_rate_60d`` : 60  (needs 60 well-defined consecutive flips).
* ``signal_cum_pnl_20d``   : 19  (needs 20 trailing products).

CITATIONS
=========
The signal-as-conditioning-variable idea has many lineages (Lopez de
Prado's meta-labelling chapter, AFML Ch. 3; Friedman's "regime-conditional"
features in Programming Session 4). The specific feature set here is
designed for this coursework's piecewise-constant 3-state signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "signal_run_length",
    "time_since_last_flip",
    "signal_entropy_20d",
    "signal_flip_rate_60d",
    "signal_cum_pnl_20d",
]


# --------------------------------------------------------------------------- #
# Trajectory features                                                          #
# --------------------------------------------------------------------------- #
def signal_run_length(s: pd.Series) -> pd.Series:
    """Length of the consecutive-identical-value run of ``s`` ending at ``t``.

    For ``s = [1, 1, 0, 0, 0, -1]`` the result is ``[1, 2, 1, 2, 3, 1]``.
    The first row is always 1: it starts a fresh run (NaN-shift makes the
    inequality True, which is the desired behaviour). Integer-valued,
    ``>= 1`` everywhere, ``int64`` dtype.
    """
    s = s.astype("float64")
    change = s.ne(s.shift(1))
    group_id = change.cumsum()
    counts = s.groupby(group_id).cumcount() + 1
    return counts.astype("int64").rename("signal_run_length")


def time_since_last_flip(s: pd.Series) -> pd.Series:
    """Number of bars since ``s`` last changed value.

    Equals ``signal_run_length(s) - 1`` by construction. The first row is
    0 — there is no prior bar to flip from. Integer-valued, ``>= 0``.
    """
    return (signal_run_length(s) - 1).rename("time_since_last_flip")


def signal_entropy_20d(s: pd.Series, window: int = 20) -> pd.Series:
    """Rolling Shannon entropy (natural log) of the empirical
    ``{-1, 0, +1}`` PMF over the trailing ``window`` bars.

    Output is in ``[0, log(3)] ≈ [0, 1.0986]``. NaN for the first
    ``window - 1`` rows. ``signal`` values are rounded to integers
    before tallying so floating-point noise (e.g. from upstream
    multiplication by 1.0) cannot leak extra categories into the PMF.
    """
    s = s.astype("float64").round()

    def _entropy(vals: np.ndarray) -> float:
        n = len(vals)
        if n == 0:
            return float("nan")
        # Empirical PMF over the three categories.
        n_neg = float((vals == -1).sum())
        n_zero = float((vals == 0).sum())
        n_pos = float((vals == 1).sum())
        probs = np.array([n_neg, n_zero, n_pos]) / n
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    return (
        s.rolling(window, min_periods=window)
        .apply(_entropy, raw=True)
        .rename(f"signal_entropy_{window}d")
    )


def signal_flip_rate_60d(s: pd.Series, window: int = 60) -> pd.Series:
    """Fraction of consecutive-bar value changes in the trailing window.

    A "flip" is any ``s[u] != s[u-1]`` — ``0 -> 1``, ``1 -> 0``,
    ``1 -> -1`` etc. The row-0 flip is undefined (no prior bar); it is
    set to NaN so the rolling mean correctly excludes warmup rows.
    Output is in ``[0, 1]`` and NaN for the first ``window`` rows
    (``rolling(min_periods=window)`` requires all values non-NaN).
    """
    s = s.astype("float64")
    flips = s.ne(s.shift(1)).astype("float64")
    flips.iloc[0] = np.nan  # row 0 flip is undefined
    return (
        flips.rolling(window, min_periods=window)
        .mean()
        .rename(f"signal_flip_rate_{window}d")
    )


def signal_cum_pnl_20d(
    s: pd.Series, r: pd.Series, window: int = 20
) -> pd.Series:
    """Trailing ``window``-day cumulative product ``Σ s_u · r_u``.

    Contemporaneous-product proxy for "follow the primary blindly". Both
    ``s`` and ``r`` are observed by time ``t`` so the feature is
    strictly causal. The next-day-execution convention says ``PnL_t =
    s_t · r_{t+1}``; this feature uses the simpler contemporaneous
    product to give the model a single "recent track record" channel
    rather than a leakage-prone forward-looking estimator. Unbounded;
    NaN for the first ``window - 1`` rows.
    """
    aligned = s.astype("float64") * r.astype("float64")
    return (
        aligned.rolling(window, min_periods=window)
        .sum()
        .rename(f"signal_cum_pnl_{window}d")
    )


# --------------------------------------------------------------------------- #
# Causality harness registry                                                   #
# --------------------------------------------------------------------------- #
CAUSALITY_REGISTRATIONS: list[dict] = [
    {
        "name": "signal_run_length",
        "module": __name__,
        "func": "signal_run_length",
        "adapter": "signal",
        "kwargs": {},
        "warmup": 0,
        "data_kind": "single_instrument",
    },
    {
        "name": "time_since_last_flip",
        "module": __name__,
        "func": "time_since_last_flip",
        "adapter": "signal",
        "kwargs": {},
        "warmup": 0,
        "data_kind": "single_instrument",
    },
    {
        "name": "signal_entropy_20d",
        "module": __name__,
        "func": "signal_entropy_20d",
        "adapter": "signal",
        "kwargs": {},
        "warmup": 19,
        "data_kind": "single_instrument",
    },
    {
        "name": "signal_flip_rate_60d",
        "module": __name__,
        "func": "signal_flip_rate_60d",
        "adapter": "signal",
        "kwargs": {},
        "warmup": 60,
        "data_kind": "single_instrument",
    },
    {
        "name": "signal_cum_pnl_20d",
        "module": __name__,
        "func": "signal_cum_pnl_20d",
        "adapter": "signal_returns",
        "kwargs": {},
        "warmup": 19,
        "data_kind": "single_instrument",
    },
]
