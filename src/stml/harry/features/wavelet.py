"""wavelet.py — multi-resolution-analysis (MRA) energy bands.

ECONOMIC INTUITION
==================
A daily-return time series carries information at many time scales
simultaneously: intraday noise, day-of-week effects, weekly cycles,
month-end / quarter-end rebalancing, and slow macro drift. A wavelet
multi-resolution analysis decomposes the recent return path into
orthogonal *detail* signals at each scale, and the *energy* (sum of
squared coefficients) at each scale tells us how much of recent
variation lives at that scale.

For the meta-model: a regime where most energy lives at the daily-noise
band is one in which the primary signal's mean-reversion structure
(short-horizon) is the active driver. A regime where most energy lives
at the monthly / quarterly bands is one in which a slow macro trend
dominates — and a counter-trend primary signal may be unreliable.

* ``mra_energy_bands(r, wavelet='db4', levels=5, window=252)`` — returns
  a 5-column DataFrame: the fraction of recent (trailing 252-day) total
  variation that lives at each of the first 5 detail levels. Levels
  approximately correspond to ~daily / ~weekly / ~bi-weekly / ~monthly /
  ~quarterly cycles.

CAUSALITY CONTRACT
==================
The wavelet transform is computed inside a strictly trailing window
``r.iloc[t-window+1 : t+1]``. Output at row ``t`` uses only data at
indices ``<= t``. Verified by the universal causality harness.

WARMUP WINDOW
=============
``window - 1`` rows (default 251).

CITATIONS
=========
* Percival, D. B. & Walden, A. T. (2000) "Wavelet Methods for Time
  Series Analysis", Cambridge — multi-resolution decomposition theory.
* Gencay, R., Selcuk, F. & Whitcher, B. (2002) "An Introduction to
  Wavelets and Other Filtering Methods in Finance and Economics" — the
  finance applications of MRA energy.

REQUIRES
========
``pywavelets``. Install with ``uv sync --extra harry-features`` (see
``reports/harry/SETUP.md``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pywt  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "stml.harry.features.wavelet requires PyWavelets. "
        "Run 'uv sync --extra harry-features' (see reports/harry/SETUP.md)."
    ) from exc

__all__ = ["mra_energy_bands"]

_DEFAULT_WAVELET: str = "db4"
_DEFAULT_LEVELS: int = 5
_DEFAULT_WINDOW: int = 252


def _energy_fractions(values: np.ndarray, wavelet: str, levels: int) -> np.ndarray:
    """Run a single-pass DWT and return the fraction of energy at each
    detail level (cD1, cD2, …, cD_levels). The approximation coefficient
    cA_levels is intentionally excluded — the resulting vector sums to
    *at most* 1 (the missing mass is the slow-drift residual).

    ``values`` must be a 1-D finite array of length ``>= 2^levels``.
    """
    # ``wavedec`` returns [cA_n, cD_n, cD_{n-1}, ..., cD_1].
    # np.array() ensures a writable copy — pywt raises ValueError on read-only buffers
    # produced by NumPy array slices in newer NumPy/pywt versions.
    coeffs = pywt.wavedec(np.array(values, dtype=np.float64), wavelet, level=levels, mode="periodization")
    # Energy of each detail level. The output order from wavedec is
    # [cA_n, cD_n, cD_{n-1}, ..., cD_1]; we want cD_1, cD_2, ..., cD_n.
    detail_energies = [
        float(np.sum(np.square(coeffs[i]))) for i in range(levels, 0, -1)
    ]
    detail_energies_arr = np.array(detail_energies, dtype=np.float64)
    total = float(np.sum(np.square(values)))
    if total <= 0:
        return np.full(levels, np.nan)
    return detail_energies_arr / total


def mra_energy_bands(
    r: pd.Series,
    *,
    wavelet: str = _DEFAULT_WAVELET,
    levels: int = _DEFAULT_LEVELS,
    window: int = _DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Rolling MRA energy fractions per detail level.

    Returns a DataFrame with one column per detail level
    (``mra_energy_D1`` through ``mra_energy_D{levels}``). At each row
    ``t``, the values are the fraction of squared variation in the
    trailing ``window`` returns that lies at each scale, computed via a
    multi-level discrete wavelet transform with ``periodization`` mode
    (avoids boundary artifacts on a finite window).

    Output rows are non-negative; the row's sum is in ``[0, 1]``. NaN
    for the first ``window - 1`` rows.
    """
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")
    if window < 2 ** levels:
        raise ValueError(
            f"window must be >= 2^levels = {2 ** levels} for level={levels}, got {window}"
        )

    arr = r.astype("float64").to_numpy()
    n = len(arr)
    out = np.full((n, levels), np.nan, dtype=np.float64)
    for t in range(window - 1, n):
        seg = arr[t - window + 1 : t + 1]
        if not np.isfinite(seg).all():
            continue
        out[t, :] = _energy_fractions(seg, wavelet, levels)
    columns = [f"mra_energy_D{k}" for k in range(1, levels + 1)]
    return pd.DataFrame(out, index=r.index, columns=columns)


# --------------------------------------------------------------------------- #
# Causality harness registry                                                  #
# --------------------------------------------------------------------------- #
# Smaller window in the harness so the parametrised tests stay fast; the
# default of 252 remains in place for real-data calls.
CAUSALITY_REGISTRATIONS: list[dict] = [
    {
        "name": "mra_energy_bands",
        "module": __name__,
        "func": "mra_energy_bands",
        "adapter": "returns",
        "kwargs": {"wavelet": "db4", "levels": 5, "window": 64},
        "warmup": 63,
        "data_kind": "single_instrument",
    },
]
