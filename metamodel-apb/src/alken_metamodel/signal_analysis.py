"""Primary-signal characterisation for EX.5 (/aqms-python L3/L4).

The metamodel is a *secondary* act/skip filter, so its ceiling is set by the primary signal it
sits on. This module measures the provided signal in its own right: turnover (flip frequency),
hit-rate (directional accuracy on non-zero days), the information coefficient (rank correlation
with the forward return), and the fundamental-law information ratio ``IR = IC·√breadth`` (Grinold
1989). Together they contextualise how much edge is even available for the metamodel to refine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def signal_hit_rate(signal, forward_returns) -> float:
    """Fraction of non-zero-signal days whose sign matches the forward return (NaN if none)."""
    s = np.asarray(signal, dtype=float)
    r = np.asarray(forward_returns, dtype=float)
    active = s != 0
    if not active.any():
        return float("nan")
    return float((np.sign(s[active]) == np.sign(r[active])).mean())


def signal_turnover(signal) -> float:
    """Mean number of unit position changes per period = mean(|Δsignal|)/2."""
    s = pd.Series(np.asarray(signal, dtype=float))
    return float(s.diff().abs().div(2.0).dropna().mean())


def information_coefficient(signal, forward_returns, *, method: str = "spearman") -> float:
    """Rank (Spearman) correlation between the signal and the forward return — the IC."""
    s = np.asarray(signal, dtype=float)
    r = np.asarray(forward_returns, dtype=float)
    if method != "spearman":
        raise ValueError(f"unsupported IC method: {method}")
    return float(spearmanr(s, r).correlation)


def information_ratio(ic: float, breadth: float) -> float:
    """Fundamental law of active management: IR = IC·√breadth (Grinold 1989)."""
    return float(ic * np.sqrt(breadth))
