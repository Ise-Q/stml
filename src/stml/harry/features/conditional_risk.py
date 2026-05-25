"""conditional_risk.py — first-passage-time and path-shape features.

ECONOMIC INTUITION
==================
The triple-barrier label asks "will the bet hit a profit or stop barrier
within h trading days?" — but the *answer* is not known at decision
time, only its distribution is. These features estimate the conditional
distribution of resolution dynamics from the recent return history:

* ``expected_hit_time``  — empirical median first-passage time to the
  symmetric ``±pt_mult · σ_t · √h`` barriers, bootstrapped from the
  trailing 252-day return distribution. Low values mean the typical
  recent path hits a barrier quickly; high values mean paths tend to
  time out.
* ``prob_timeout``       — probability that NEITHER barrier is touched
  within ``h`` bars, same bootstrap. A complementary view: high values
  mean recent volatility is below the barrier scale.
* ``path_tortuosity_20d``    — ``Σ|r_u| / |Σ r_u|`` over the trailing
  20 days. Higher = more zigzag (high absolute-variation per unit net
  move). Trend signals are more reliable on lower-tortuosity paths.
* ``realized_semi_vol_ratio`` — RMS of positive returns divided by RMS
  of negative returns over a trailing window. Captures recent
  asymmetry: a ratio > 1 means recent up-moves are larger than down-moves
  (skewed regime), < 1 means the opposite.

None of these are computed by Sreeram's or signal-deep-dive's branches.

CAUSALITY CONTRACT
==================
Every output at row ``t`` uses only data at indices ``<= t``. The
bootstrap simulators seed their RNG from the row index ``t`` itself
(``seed × 1_000_003 + t``), so a row's output is identical when computed
on ``data[:t+1]`` and on ``data[:T]`` for any ``T >= t+1`` — verified by
the universal causality harness.

WARMUP WINDOWS
==============
* ``expected_hit_time``       : ``window`` rows (default 252).
* ``prob_timeout``            : ``window`` rows (default 252).
* ``path_tortuosity_20d``     : ``window - 1`` rows (default 19).
* ``realized_semi_vol_ratio`` : ``window - 1`` rows (default 19).

CITATIONS
=========
* Cont, R. & Tankov, P. (2003) "Financial Modelling with Jump Processes"
  — non-parametric bootstrap of first-passage statistics.
* Markowitz, H. (1959) "Portfolio Selection" — downside / semi-vol.

A note on bootstrap design: the empirical CDF of the trailing 252 returns
is sampled WITH REPLACEMENT to construct synthetic paths. This preserves
the *marginal* distribution shape (skew, kurtosis, heavy tails) without
imposing a Gaussian prior. We do NOT preserve autocorrelation in the
simulated paths — a block bootstrap would be more faithful but adds
complexity; for the feature engineering use case the marginal-only
bootstrap is a documented approximation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "expected_hit_time",
    "prob_timeout",
    "path_tortuosity_20d",
    "realized_semi_vol_ratio",
]

# Default Monte Carlo / window sizes — chosen to be fast enough for the
# Step 4 pipeline while still giving a stable simulation answer.
_DEFAULT_WINDOW: int = 252
_DEFAULT_N_SIMS: int = 200
_DEFAULT_SEED: int = 42
_PER_ROW_PRIME: int = 1_000_003  # arbitrary large prime used to mix the per-row seed


# --------------------------------------------------------------------------- #
# First-passage simulation                                                    #
# --------------------------------------------------------------------------- #
def _simulate_first_passage(
    returns: pd.Series,
    vol: pd.Series,
    *,
    pt_mult: float,
    sl_mult: float,
    h: int,
    window: int = _DEFAULT_WINDOW,
    n_sims: int = _DEFAULT_N_SIMS,
    seed: int = _DEFAULT_SEED,
) -> tuple[pd.Series, pd.Series]:
    """Per-row bootstrap simulation of first passage to symmetric barriers.

    Returns ``(median_hit_time, prob_timeout)`` as two Series aligned
    with ``returns``. The simulator is seeded per row so the result at
    any row ``t`` depends only on data at indices ``<= t``.

    For row ``t``:
      1. Take the trailing ``window`` returns ``r[t - window : t]``.
      2. Resample ``n_sims`` paths of length ``h`` with replacement.
      3. For each path, compute the cumulative log-return and detect the
         first crossing of ``±pt_mult · vol[t] · √h``.
      4. Median first-passage time across paths that touched any barrier;
         probability that none touched.

    Rows with NaN or non-positive ``vol[t]``, or with any NaN in the
    trailing window, are left at NaN.
    """
    if h < 1:
        raise ValueError(f"h must be >= 1, got {h}")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if n_sims < 1:
        raise ValueError(f"n_sims must be >= 1, got {n_sims}")

    r_arr = returns.astype("float64").to_numpy()
    v_arr = vol.astype("float64").to_numpy()
    n = len(returns)
    sqrt_h = float(np.sqrt(h))

    median_hit = np.full(n, np.nan)
    p_timeout = np.full(n, np.nan)

    for t in range(window, n):
        past = r_arr[t - window : t]
        if not np.isfinite(past).all():
            continue
        v_t = v_arr[t]
        if not np.isfinite(v_t) or v_t <= 0:
            continue
        pt = pt_mult * v_t * sqrt_h
        sl = sl_mult * v_t * sqrt_h
        rng = np.random.default_rng(seed * _PER_ROW_PRIME + t)
        samples = rng.choice(past, size=(n_sims, h), replace=True)
        cum = samples.cumsum(axis=1)
        touch_mask = (cum >= pt) | (cum <= -sl)
        has_touch = touch_mask.any(axis=1)
        if has_touch.any():
            first_touch = touch_mask.argmax(axis=1) + 1  # 1-indexed day of first touch
            median_hit[t] = float(np.median(first_touch[has_touch]))
        else:
            # No simulated path touched — define hit time as the timeout horizon.
            median_hit[t] = float(h + 1)
        p_timeout[t] = float(1.0 - has_touch.mean())

    hit_series = pd.Series(median_hit, index=returns.index, name="expected_hit_time")
    pto_series = pd.Series(p_timeout, index=returns.index, name="prob_timeout")
    return hit_series, pto_series


def expected_hit_time(
    returns: pd.Series,
    vol: pd.Series,
    *,
    pt_mult: float = 1.0,
    sl_mult: float = 1.0,
    h: int = 10,
    window: int = _DEFAULT_WINDOW,
    n_sims: int = _DEFAULT_N_SIMS,
    seed: int = _DEFAULT_SEED,
) -> pd.Series:
    """Empirical median first-passage time to ``±mult · vol · √h`` barriers.

    See :func:`_simulate_first_passage` for the algorithm. NaN for the
    first ``window`` rows.
    """
    hit, _ = _simulate_first_passage(
        returns, vol,
        pt_mult=pt_mult, sl_mult=sl_mult, h=h,
        window=window, n_sims=n_sims, seed=seed,
    )
    return hit


def prob_timeout(
    returns: pd.Series,
    vol: pd.Series,
    *,
    pt_mult: float = 1.0,
    sl_mult: float = 1.0,
    h: int = 10,
    window: int = _DEFAULT_WINDOW,
    n_sims: int = _DEFAULT_N_SIMS,
    seed: int = _DEFAULT_SEED,
) -> pd.Series:
    """Empirical probability that neither barrier is touched within ``h``.

    See :func:`_simulate_first_passage`. Output in ``[0, 1]``; NaN for
    the first ``window`` rows.
    """
    _, pto = _simulate_first_passage(
        returns, vol,
        pt_mult=pt_mult, sl_mult=sl_mult, h=h,
        window=window, n_sims=n_sims, seed=seed,
    )
    return pto


# --------------------------------------------------------------------------- #
# Path-shape features                                                          #
# --------------------------------------------------------------------------- #
_TORTUOSITY_EPS: float = 1e-12


def path_tortuosity_20d(r: pd.Series, window: int = 20) -> pd.Series:
    """Trailing ``Σ |r_u| / |Σ r_u|`` over the trailing ``window`` bars.

    Captures how much "wandering" there has been per unit of net move.
    Range ``[1, ∞)``: 1 is a perfectly monotonic path, larger values are
    zigzag-ier. A small epsilon protects against ``Σ r_u = 0``; the
    output is always finite. NaN for the first ``window - 1`` rows.
    """
    r = r.astype("float64")
    abs_sum = r.abs().rolling(window, min_periods=window).sum()
    net = r.rolling(window, min_periods=window).sum()
    out = abs_sum / (net.abs() + _TORTUOSITY_EPS)
    return out.rename(f"path_tortuosity_{window}d")


def realized_semi_vol_ratio(r: pd.Series, window: int = 20) -> pd.Series:
    """Upside-RMS / downside-RMS of returns over a trailing window.

    ``upside_rms² = mean(max(r, 0)²)``; ``downside_rms² = mean(min(r, 0)²)``.
    Ratio is non-negative; a value > 1 means recent up-moves are larger
    (in RMS) than down-moves, < 1 means the opposite. NaN for the first
    ``window - 1`` rows; a tiny ``eps`` in the denominator prevents
    Inf when all trailing returns share a single sign.
    """
    r = r.astype("float64")
    pos = r.where(r > 0, 0.0)
    neg = r.where(r < 0, 0.0)
    rms_pos = np.sqrt((pos ** 2).rolling(window, min_periods=window).mean())
    rms_neg = np.sqrt((neg ** 2).rolling(window, min_periods=window).mean())
    return (rms_pos / (rms_neg + _TORTUOSITY_EPS)).rename(
        f"realized_semi_vol_ratio_{window}d"
    )


# --------------------------------------------------------------------------- #
# Causality harness registry                                                  #
# --------------------------------------------------------------------------- #
# n_sims / window kept small in the registry so the causality-harness pass
# (truncation × full × multiple t values) stays fast. The defaults of 200
# sims and 252 days remain in place for real-data calls.
_HARNESS_KW: dict = {
    "pt_mult": 1.0,
    "sl_mult": 1.0,
    "h": 10,
    "window": 80,
    "n_sims": 40,
    "seed": 42,
}

CAUSALITY_REGISTRATIONS: list[dict] = [
    {
        "name": "expected_hit_time",
        "module": __name__,
        "func": "expected_hit_time",
        "adapter": "returns_vol",
        "kwargs": dict(_HARNESS_KW),
        "warmup": 80,
        "data_kind": "single_instrument",
    },
    {
        "name": "prob_timeout",
        "module": __name__,
        "func": "prob_timeout",
        "adapter": "returns_vol",
        "kwargs": dict(_HARNESS_KW),
        "warmup": 80,
        "data_kind": "single_instrument",
    },
    {
        "name": "path_tortuosity_20d",
        "module": __name__,
        "func": "path_tortuosity_20d",
        "adapter": "returns",
        "kwargs": {},
        "warmup": 19,
        "data_kind": "single_instrument",
    },
    {
        "name": "realized_semi_vol_ratio",
        "module": __name__,
        "func": "realized_semi_vol_ratio",
        "adapter": "returns",
        "kwargs": {},
        "warmup": 19,
        "data_kind": "single_instrument",
    },
]
