"""Per-instrument feature assembly for the meta-labelling metamodel (Stage 2).

This adapter wires the leakage-safe **causal** feature *functions* that stml already
ships (proven E-class: every value at ``t`` is identical on ``data[:t+1]`` and on the full
series) into the meta-model, and adds a backward **trend-scanning feature** (López de Prado
trend scanning is used as a *feature here, never as the label* — the label is the
triple-barrier outcome). No fitted state lives in this module; the only families that fit
parameters (regime/HMM) live in ``regime.py``.

Reused stml functions (signatures verified against source, not assumed):
- ``stml.metamodel.features.assemble_engineered`` -> F1/F2/F5/F6/F7/F8/F10 (≈37 cols).
- ``stml.metamodel.features_ext.assemble_engineered_ext`` -> F2-RS/F7-adds/F12/F13/F15/F5-adds
  (F13 wavelets require ``pywavelets``).
- ``stml.metamodel.features_ext.add_z_twins`` -> per-instrument causal expanding-window
  ``z_<col>`` twins for the 24 scale-dependent columns.

**Fold-safety contract (the crux).** Every ``assemble_*`` is *stateless*, so the leakage
concern is purely the input window: compute on each instrument's FULL fixed-start history
and then **right-slice the OUTPUT** to the fold dates (``right_slice``). Never left-truncate
the input — it would break (a) rolling warm-up, (b) the start-sensitive expanding z-twins,
and (c) ``f15``'s positional-seed bootstrap (whose truncation-invariance is right-edge only).
This is enforced as a property test (``tests/test_features.py``): ``feature[t]`` is identical
whether computed on ``data[:t+1]`` or ``data[:T]``.

The backward trend feature reuses the trend-scanning *algorithm* of
``_vendor/trend_scanning.py`` (``trend_labels(look_forward=False)``) but computes the OLS
slope t-statistic in closed form (``_segment_tval``, validated equal to the vendored
``tValLinR``) and applies a **deterministic ±cap** instead of ``trend_labels``' global-variance
cap — that global cap depends on the whole series and is itself a right-edge truncation leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from stml.metamodel.features import assemble_engineered
from stml.metamodel.features_ext import add_z_twins, assemble_engineered_ext

SQRT_252: float = float(np.sqrt(252.0))
TREND_TVAL_CAP: float = 20.0
DEFAULT_TREND_SPAN: tuple[int, int] = (5, 25)


# --- backward trend-scanning feature ---------------------------------------

def _segment_tval(y: np.ndarray) -> float:
    """OLS slope t-statistic of ``y`` regressed on integer time 0..L-1 (closed form).

    Identical to the vendored ``tValLinR`` (validated by test) but ~100x faster: no
    statsmodels object per call, which matters because trend scanning runs per fold ×
    per instrument × full history in the pipeline.
    """
    n = y.shape[0]
    if n <= 2:
        return 0.0
    x = np.arange(n, dtype=float)
    dx = x - x.mean()
    sxx = float((dx * dx).sum())
    slope = float((dx * (y - y.mean())).sum()) / sxx
    resid = y - (y.mean() + slope * dx)
    ssr = float((resid * resid).sum())
    se = np.sqrt((ssr / (n - 2)) / sxx)
    if se == 0.0:
        return 0.0 if slope == 0.0 else float(np.sign(slope) * np.inf)
    return slope / se


def backward_trend_feature(
    close: pd.Series,
    span: tuple[int, int] = DEFAULT_TREND_SPAN,
    cap: float = TREND_TVAL_CAP,
) -> pd.DataFrame:
    """Backward trend-scanning feature (``look_forward=False``), causal by construction.

    For each date ``t`` with at least ``span[1]`` bars of history, scan backward windows
    ``[t-h, t]`` for ``h`` in ``range(*span)``, take the window with the largest |t|, and
    record its (capped) t-value, sign, and window size. Warm-up rows are NaN (never
    fabricated). Returns columns ``trend_tval_back``, ``trend_sign_back``, ``trend_window_back``.
    """
    close = close.sort_index()
    idx = close.index
    vals = close.to_numpy(dtype=float)
    min_w, max_w = span
    n = len(idx)
    tval = np.full(n, np.nan)
    win = np.full(n, np.nan)
    for i in range(n):
        if i < max_w:  # not enough backward history (mirrors trend_labels' guard)
            continue
        best_t = 0.0
        best_h = np.nan
        for h in range(min_w, max_w):
            t = _segment_tval(vals[i - h : i + 1])  # backward window, inclusive of i
            if abs(t) > abs(best_t):
                best_t, best_h = t, float(h)
        tval[i] = best_t
        win[i] = best_h
    return pd.DataFrame(
        {
            "trend_tval_back": np.clip(tval, -cap, cap),
            "trend_sign_back": np.sign(tval),
            "trend_window_back": win,
        },
        index=idx,
    )


# --- full per-instrument assembly ------------------------------------------

def _close_series(ohlcv_inst: pd.DataFrame) -> pd.Series:
    s = ohlcv_inst.set_index("date")["close"].sort_index()
    s.index = pd.DatetimeIndex(s.index)
    return s.astype(float)


def assemble_instrument_features(
    ohlcv_inst: pd.DataFrame,
    signal_inst: pd.Series,
    *,
    trend_span: tuple[int, int] = DEFAULT_TREND_SPAN,
) -> pd.DataFrame:
    """Assemble the full causal feature stack for ONE instrument over its full history.

    Parameters
    ----------
    ohlcv_inst : long OHLCV for one instrument (cols ``date, instrument, open, high, low,
        close, volume`` (+optional ``open_interest``)); pass the FULL fixed-start history.
    signal_inst : date-indexed primary signal Series in {-1, 0, +1}.

    Returns a per-instrument ``DatetimeIndex`` float frame = stml core (F1/F2/F5/F6/F7/F8/F10)
    + ext (F2-RS/F7-adds/F12/F13/F15/F5-adds) + z-twins + backward trend feature. UNFILTERED
    (includes zero-signal days) and unkeyed — use ``filter_signal_days`` / ``attach_instrument``
    downstream, and ``right_slice`` to enforce the fold boundary.
    """
    core = assemble_engineered(ohlcv_inst, signal_inst)
    ext = assemble_engineered_ext(ohlcv_inst, signal_inst)
    stack = pd.concat([core, ext], axis=1)
    z = add_z_twins(stack)
    trend = backward_trend_feature(_close_series(ohlcv_inst), span=trend_span)
    feats = pd.concat([stack, z, trend.reindex(stack.index)], axis=1)
    return feats.astype(float)


# --- fold-safety + label-interface helpers ---------------------------------

def right_slice(features: pd.DataFrame, end) -> pd.DataFrame:
    """Enforce the fold boundary on the OUTPUT only (keep the left inception anchor)."""
    return features.loc[:end].copy()


def filter_signal_days(features: pd.DataFrame, signal_inst: pd.Series) -> pd.DataFrame:
    """Keep only the non-zero-signal trade days (the meta-label event days)."""
    nonzero = signal_inst.index[signal_inst.to_numpy() != 0]
    keep = features.index.intersection(nonzero)
    return features.loc[keep]


def attach_instrument(features: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """Re-attach the instrument key (the stml assemblers drop it)."""
    out = features.copy()
    out.insert(0, "instrument", instrument)
    return out


def daily_barrier_sigma(features: pd.DataFrame) -> pd.Series:
    """De-annualise ``f2_vol_20`` (annualised in stml) into the daily triple-barrier sigma."""
    return features["f2_vol_20"] / SQRT_252
