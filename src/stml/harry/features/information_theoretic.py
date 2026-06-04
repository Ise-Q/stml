"""information_theoretic.py — non-parametric mutual information and
transfer entropy features.

ECONOMIC INTUITION
==================
The signal-direction audit (Step 1) measured *linear* relationships
between the primary signal and trailing / forward returns. Information-
theoretic quantities are the non-linear generalisation: they capture
dependence regardless of functional form. Two features:

* ``rolling_mutual_information_252d(s, y)`` — the empirical mutual
  information between the signal and another quantity ``y`` (e.g. the
  h-day forward return), binned into 5-quantile cells on each side and
  estimated over a trailing 252-day window. Non-zero MI is a sign that
  the signal carries information about ``y`` that linear correlation may
  miss — for instance, "the signal is informative only at the tails".

* ``transfer_entropy_vol_to_signal_acc(vol, s, r_fwd)`` — Schreiber-style
  transfer entropy from realised volatility to the binary signal-accuracy
  indicator ``sign(s · r_fwd) > 0``, lag 1. Asks: "does the previous
  day's vol tell us something about whether the signal will be correct
  TODAY beyond what the signal's own past told us?" A positive TE is
  the regime-conditioning the audit hinted at (vol bands affect signal
  reliability).

CAUSALITY CONTRACT
==================
Every output at row ``t`` uses only data at indices ``<= t``. Forward-
looking inputs (e.g. h-day forward returns) must be pre-shifted by the
CALLER so that the value at row ``t`` represents information observable
at time ``t``. The functions themselves take generic Series and do not
attempt to detect or re-align forward-looking inputs — that contract is
explicit in the docstrings.

WARMUP WINDOWS
==============
* ``rolling_mutual_information_252d``         : ``window`` rows (default 252).
* ``transfer_entropy_vol_to_signal_acc``      : ``window`` rows (default 126).

CITATIONS
=========
* Shannon, C. E. (1948) "A Mathematical Theory of Communication" —
  mutual information.
* Schreiber, T. (2000) "Measuring Information Transfer", Phys. Rev. Lett.
  85: 461 — transfer entropy with one-lag conditioning.
* Cover & Thomas, "Elements of Information Theory" (2nd ed.) — joint /
  conditional entropy identities used to decompose TE into four
  Shannon entropies.

A note on the histogram estimator: quantile binning (rather than fixed-
width binning) is robust to outliers and adapts to the local return
distribution; it is the standard non-parametric MI estimator. We
intentionally do NOT use the Kraskov–Stögbauer–Grassberger k-NN
estimator (better but heavier and harder to make truncation-invariant
across pandas indices).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "rolling_mutual_information_252d",
    "transfer_entropy_vol_to_signal_acc",
]

_DEFAULT_MI_WINDOW: int = 252
_DEFAULT_TE_WINDOW: int = 126
_DEFAULT_BINS_MI: int = 5
_DEFAULT_BINS_TE: int = 3


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _bin_quantiles(x: np.ndarray, n_bins: int) -> np.ndarray | None:
    """Quantile-bin ``x`` into at most ``n_bins`` integer labels.

    Returns ``None`` if the data is degenerate (zero unique values).
    Otherwise returns an integer array of the same length as ``x`` with
    labels in ``[0, k)`` for some ``k <= n_bins``.
    """
    if len(x) == 0:
        return None
    if np.unique(x).size < 2:
        # Degenerate window — all values identical. Treat as a single
        # category so MI is 0 by construction.
        return np.zeros(len(x), dtype=np.int64)
    try:
        bins = pd.qcut(x, n_bins, duplicates="drop", labels=False)
    except ValueError:
        return None
    arr = np.asarray(bins, dtype=np.int64)
    return arr


def _shannon_entropy(probs: np.ndarray) -> float:
    """Natural-log Shannon entropy of a probability vector."""
    p = probs[probs > 0]
    return float(-np.sum(p * np.log(p)))


def _mutual_information(x_bins: np.ndarray, y_bins: np.ndarray) -> float:
    """MI in nats between two integer-labelled categorical variables.

    Computed via the joint histogram and the closed-form identity
    ``MI = H(X) + H(Y) - H(X, Y)``.
    """
    n = len(x_bins)
    if n == 0:
        return 0.0
    n_x = int(x_bins.max()) + 1
    n_y = int(y_bins.max()) + 1
    joint = np.zeros((n_x, n_y), dtype=np.float64)
    np.add.at(joint, (x_bins, y_bins), 1.0)
    joint /= joint.sum()
    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)
    return (
        _shannon_entropy(p_x)
        + _shannon_entropy(p_y)
        - _shannon_entropy(joint.flatten())
    )


# --------------------------------------------------------------------------- #
# Rolling mutual information                                                   #
# --------------------------------------------------------------------------- #
def rolling_mutual_information_252d(
    x: pd.Series,
    y: pd.Series,
    *,
    window: int = _DEFAULT_MI_WINDOW,
    n_bins: int = _DEFAULT_BINS_MI,
) -> pd.Series:
    """Rolling non-parametric mutual information ``MI(x; y)`` in nats.

    At time ``t`` uses the trailing-window observation pairs
    ``[(x_u, y_u) : u in [t-window+1, t]]``. NaN pairs are dropped before
    binning; each side is quantile-binned into at most ``n_bins`` bins
    (ties collapse bins via ``duplicates="drop"``). Output ``>= 0`` in
    nats; NaN for the first ``window - 1`` rows or when too few pairs
    remain after dropping NaN.

    The function does NOT shift either input. To compute MI between the
    signal at ``t`` and the h-day FORWARD return, the caller must pass
    ``r_fwd_h.shift(0)`` where ``r_fwd_h_u = sum(r_{u+1..u+h})`` is
    already aligned so that the value at row ``u`` is observable at
    time ``u`` (i.e. shifted back by ``h``).
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")

    x_arr = x.astype("float64").to_numpy()
    y_arr = y.astype("float64").to_numpy()
    n = len(x)
    out = np.full(n, np.nan)
    min_count = max(2 * n_bins, 10)

    for t in range(window - 1, n):
        xw = x_arr[t - window + 1 : t + 1]
        yw = y_arr[t - window + 1 : t + 1]
        mask = np.isfinite(xw) & np.isfinite(yw)
        if mask.sum() < min_count:
            continue
        x_bins = _bin_quantiles(xw[mask], n_bins)
        y_bins = _bin_quantiles(yw[mask], n_bins)
        if x_bins is None or y_bins is None:
            continue
        out[t] = _mutual_information(x_bins, y_bins)
    return pd.Series(
        out, index=x.index, name=f"rolling_mutual_information_{window}d"
    )


# --------------------------------------------------------------------------- #
# Transfer entropy (vol → signal-accuracy)                                    #
# --------------------------------------------------------------------------- #
def transfer_entropy_vol_to_signal_acc(
    vol: pd.Series,
    s: pd.Series,
    r_fwd: pd.Series,
    *,
    window: int = _DEFAULT_TE_WINDOW,
    n_bins: int = _DEFAULT_BINS_TE,
) -> pd.Series:
    """Schreiber-style transfer entropy from vol to signal accuracy, lag 1.

    The signal-accuracy series ``acc_u = (s_u · r_fwd_u > 0)`` is treated
    as a binary process (1 = signal correct, 0 = wrong / no-bet). The
    transfer entropy at lag 1 is::

        TE(vol -> acc)_1 = H(acc_t | acc_{t-1}) - H(acc_t | acc_{t-1}, vol_{t-1})

    estimated by 3D histograms over the trailing ``window`` triples
    ``(acc_t, acc_{t-1}, vol_{t-1})``. Vol is quantile-binned into
    ``n_bins`` cells (default 3); ``acc`` is already binary so it
    contributes 2 bins. Output is in nats, ``>= 0``; NaN for the first
    ``window`` rows.

    As with :func:`rolling_mutual_information_252d`, this function does
    not shift inputs. Caller is responsible for pre-aligning forward
    quantities (e.g. ``r_fwd``) to the observation index.
    """
    if window < 4:
        raise ValueError(f"window must be >= 4, got {window}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")

    s_arr = s.astype("float64").to_numpy()
    r_arr = r_fwd.astype("float64").to_numpy()
    v_arr = vol.astype("float64").to_numpy()
    n = len(s)
    out = np.full(n, np.nan)
    min_count = max(2 * n_bins * 4, 20)

    for t in range(window, n):
        # Trailing window of acc and vol; lag-1 conditioning means we need
        # acc_{u-1} and vol_{u-1} for every u in the window, so we need
        # `window + 1` consecutive rows.
        u_lo = t - window
        u_hi = t  # inclusive
        acc_curr = (s_arr[u_lo + 1 : u_hi + 1] * r_arr[u_lo + 1 : u_hi + 1] > 0).astype(np.int64)
        acc_prev = (s_arr[u_lo : u_hi] * r_arr[u_lo : u_hi] > 0).astype(np.int64)
        vol_prev = v_arr[u_lo : u_hi]
        mask = (
            np.isfinite(vol_prev)
            & np.isfinite(s_arr[u_lo + 1 : u_hi + 1])
            & np.isfinite(r_arr[u_lo + 1 : u_hi + 1])
            & np.isfinite(s_arr[u_lo : u_hi])
            & np.isfinite(r_arr[u_lo : u_hi])
        )
        if mask.sum() < min_count:
            continue
        ac = acc_curr[mask]
        ap = acc_prev[mask]
        vp = vol_prev[mask]
        v_bins = _bin_quantiles(vp, n_bins)
        if v_bins is None:
            continue
        n_v = int(v_bins.max()) + 1

        # 3-D histogram (acc_t, acc_{t-1}, vol_{t-1}).
        joint3 = np.zeros((2, 2, n_v), dtype=np.float64)
        np.add.at(joint3, (ac, ap, v_bins), 1.0)
        total = joint3.sum()
        if total == 0:
            continue
        joint3 /= total

        # Marginals via tensor sums.
        p_acc_prev = joint3.sum(axis=(0, 2))                # shape (2,)
        p_acc_acc_prev = joint3.sum(axis=2)                  # (acc_t, acc_{t-1}); shape (2, 2)
        p_acc_prev_vol_prev = joint3.sum(axis=0)            # (acc_{t-1}, vol_{t-1}); shape (2, n_v)

        # TE = H(acc_t, acc_{t-1}) - H(acc_{t-1})
        #    - H(acc_t, acc_{t-1}, vol_{t-1}) + H(acc_{t-1}, vol_{t-1})
        te = (
            _shannon_entropy(p_acc_acc_prev.flatten())
            - _shannon_entropy(p_acc_prev)
            - _shannon_entropy(joint3.flatten())
            + _shannon_entropy(p_acc_prev_vol_prev.flatten())
        )
        # TE is non-negative in theory; clip tiny negative noise from FP.
        out[t] = max(float(te), 0.0)
    return pd.Series(
        out, index=s.index, name=f"transfer_entropy_vol_to_acc_{window}d"
    )


# --------------------------------------------------------------------------- #
# Causality harness registry                                                  #
# --------------------------------------------------------------------------- #
# Small windows in the harness so the parametrised tests stay fast.
CAUSALITY_REGISTRATIONS: list[dict] = [
    {
        "name": "rolling_mutual_information_252d",
        "module": __name__,
        "func": "rolling_mutual_information_252d",
        "adapter": "signal_returns",
        "kwargs": {"window": 100, "n_bins": 5},
        "warmup": 99,
        "data_kind": "single_instrument",
    },
    {
        "name": "transfer_entropy_vol_to_signal_acc",
        "module": __name__,
        "func": "transfer_entropy_vol_to_signal_acc",
        "adapter": "vol_signal_return",
        "kwargs": {"window": 100, "n_bins": 3},
        "warmup": 100,
        "data_kind": "single_instrument",
    },
]
