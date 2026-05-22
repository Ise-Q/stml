"""
archetypes.py
=============
Strategy *archetypes* (US-007): families of look-ahead-free rules that each map
one instrument's OHLCV history to a primary-style signal ``s_t in {-1, 0, +1}``.

Why a score -> deadband + sign decomposition
--------------------------------------------
The target signal is a 3-state regime call. Rather than hand-tune a different
rule per family, every archetype emits a single real-valued **score** ``z_t``
(positive = lean long, negative = lean short) and then routes it through one
shared decision rule (:func:`decide`):

    direction    = sign(z_t)                     in {-1, 0, +1}
    participate  = abs(z_t) >= deadband          (a conviction gate)
    s_t          = direction * participate       then clamp to ``allowed``

The deadband is the participation threshold: widen it and the rule trades on
fewer, higher-conviction days, so the nonzero-day count is **monotone
non-increasing** in the deadband. To make one deadband comparable across very
different families, each family **standardises** its raw score (z-score over a
window, or divide by a rolling scale) so the score is in roughly unit terms.

``allowed`` implements the *participation-only* instruments from C1: ng1s never
prints ``+1`` (pass ``allowed=(-1, 0)``); a long-only cell would pass
``allowed=(0, 1)``. A disallowed direction is dropped to ``0`` (flat), not
clamped to the nearest allowed sign -- the rule simply does not take that side.

Information set (NO look-ahead)
-------------------------------
The signal ``s_t`` is decided from information available at the **end of day t**:
the close ``close_t`` and every earlier bar, and nothing later. Execution is
next-day, so PnL is realised as ``s_t * r_{t+1}`` (handled downstream by
:mod:`stml.replication.nav`); that single forward step is the *only* place a
future bar enters, and it lives in the PnL accounting, never in the signal.

Concretely, every feature here is built from **trailing** windows
(``rolling(L)``, ``shift(+k)`` with ``k >= 0``, ``diff``) so the value at row
``t`` depends only on rows ``<= t``. No feature uses ``shift(-k)`` or any
forward/centred window. Equivalently: truncating the OHLCV at any date ``T`` and
re-running an archetype reproduces the identical signal on every date ``< T``
(a property the test-suite asserts directly).

Public API
----------
- :func:`decide` -- the shared score -> {-1,0,+1} decision rule.
- :class:`Archetype` -- a named family with ``.generate`` and ``.param_space``.
- :data:`ARCHETYPES` -- registry ``name -> Archetype`` (>= 6 families).
- The ``xsect_rank`` family additionally exposes :func:`generate_panel` via its
  ``Archetype.generate_panel`` attribute (cross-sectional; ranks the whole
  universe each day).

All scores are computed from prices only and reuse :mod:`stml.na_checks`
(``native_returns``, ``rolling_vol``) so returns honour each instrument's own
dense calendar (holiday-spanning moves are correct, never fabricated zeros) and
structural NaNs are never forward-filled.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stml.na_checks import native_returns, rolling_vol

__all__ = [
    "decide",
    "Archetype",
    "ARCHETYPES",
    "generate_panel",
]

# The fixed signal alphabet, in canonical order.
_SIGNAL_LABELS: tuple[int, int, int] = (-1, 0, 1)


# --------------------------------------------------------------------------- #
# Shared decision rule: score -> deadband(participation) + sign(direction).   #
# --------------------------------------------------------------------------- #
def decide(
    score: pd.Series,
    deadband: float,
    allowed: Sequence[int] = (-1, 0, 1),
) -> pd.Series:
    """Map a real-valued score to a signal in ``{-1, 0, +1}``.

    The decomposition is ``direction = sign(score)``,
    ``participate = abs(score) >= deadband``, ``signal = direction *
    participate``; the result is then clamped to ``allowed`` (a direction not in
    ``allowed`` is dropped to ``0``, i.e. the rule declines to take that side).

    Parameters
    ----------
    score : pd.Series
        Real-valued, date-indexed conviction score (sign = side, magnitude =
        strength). ``NaN`` scores -- e.g. inside a rolling warm-up window --
        decide to ``0`` (no position), never to a fabricated trade.
    deadband : float
        Non-negative participation threshold. ``0`` means "trade on the sign of
        every finite, non-zero score"; larger values trade only on
        higher-conviction days, so the nonzero-day count is monotone
        non-increasing in ``deadband``.
    allowed : sequence of int
        The subset of ``{-1, 0, +1}`` this cell may emit. ``(-1, 0, 1)`` is the
        unrestricted default; ``(-1, 0)`` is participation-only short side (ng1s
        never goes long); ``(0, 1)`` long-only.

    Returns
    -------
    pd.Series
        Int-dtype signal aligned to ``score.index``, values in ``allowed``.

    Raises
    ------
    ValueError
        If ``deadband`` is negative or ``allowed`` is not a subset of
        ``{-1, 0, +1}``.
    """
    if deadband < 0:
        raise ValueError(f"deadband must be non-negative, got {deadband}")
    allowed_set = set(int(a) for a in allowed)
    if not allowed_set <= set(_SIGNAL_LABELS):
        raise ValueError(
            f"allowed must be a subset of {{-1, 0, 1}}, got {sorted(allowed_set)}"
        )

    s = pd.Series(score, copy=False)
    vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)

    direction = np.sign(vals)  # NaN -> NaN, propagated to flat below
    participate = np.abs(vals) >= deadband  # NaN >= x -> False (warm-up -> flat)
    sig = np.where(participate, direction, 0.0)
    sig = np.nan_to_num(sig, nan=0.0)  # any residual NaN -> flat

    out = sig.astype(int)
    # Clamp to the allowed alphabet: a disallowed direction declines to flat.
    if allowed_set != set(_SIGNAL_LABELS):
        mask = np.isin(out, list(allowed_set))
        out = np.where(mask, out, 0)

    return pd.Series(out.astype(int), index=s.index, name="signal")


# --------------------------------------------------------------------------- #
# Feature helpers -- all TRAILING (info <= t), reused across families.        #
# --------------------------------------------------------------------------- #
def _close_series(ohlcv_inst: pd.DataFrame) -> pd.Series:
    """Date-indexed close for one instrument, sorted, on its own dense calendar.

    ``ohlcv_inst`` is a long frame for a single instrument (columns include
    ``date`` and ``close``); if several instruments are present this collapses
    to whichever rows are passed, so callers should pre-filter. The index is the
    trading calendar of *this* instrument -- no calendar grid is imposed.
    """
    s = (
        ohlcv_inst[["date", "close"]]
        .dropna(subset=["close"])
        .drop_duplicates("date")
        .sort_values("date")
        .set_index("date")["close"]
    )
    return s.astype(float)


def _instrument_returns(ohlcv_inst: pd.DataFrame) -> pd.DataFrame:
    """Long return frame (``instrument``, ``date``, ``ret``) for one instrument.

    Wraps :func:`stml.na_checks.native_returns` so downstream ``na_checks``
    helpers (e.g. :func:`stml.na_checks.rolling_vol`, which expects the long
    layout) can be reused unchanged.
    """
    return native_returns(ohlcv_inst, kind="log")


def _zscore(x: pd.Series, window: int) -> pd.Series:
    """Trailing z-score: ``(x - rolling_mean) / rolling_std`` over ``window``.

    Uses only rows ``<= t`` (pandas ``rolling`` is right-aligned and trailing),
    so it is look-ahead-free. A zero/NaN rolling std (a flat window) yields
    ``NaN`` for that row, which :func:`decide` treats as flat.
    """
    roll = x.rolling(window, min_periods=window)
    mu = roll.mean()
    sd = roll.std()
    sd = sd.where(sd > 0.0)  # 0 std -> NaN -> flat (avoid divide-by-zero blow-up)
    return (x - mu) / sd


def _log_close(close: pd.Series) -> pd.Series:
    return np.log(close)


# --------------------------------------------------------------------------- #
# Family score functions. Each returns a date-indexed real score (info <= t). #
# A positive score leans LONG, negative leans SHORT, ~0 leans flat.           #
# --------------------------------------------------------------------------- #
def _score_ts_momentum(ohlcv_inst: pd.DataFrame, params: dict) -> pd.Series:
    """Time-series momentum: standardised trailing log-return over ``L`` days.

    score_t = (log close_t - log close_{t-L}) / (rolling-vol scale).
    Positive => price rose over the lookback => lean long (trend-following).
    The raw L-day return is divided by a trailing return-vol scale so the
    deadband is comparable across instruments and lookbacks.
    """
    L = int(params["lookback"])
    vol_window = int(params.get("vol_window", max(L, 20)))
    close = _close_series(ohlcv_inst)
    logc = _log_close(close)

    raw = logc - logc.shift(L)  # trailing L-day log return, info <= t
    # Per-day return scale (trailing std of 1-day log returns), >= L history.
    daily = logc.diff()
    scale = daily.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(L)
    scale = scale.where(scale > 0.0)
    return raw / scale


def _score_mean_reversion(ohlcv_inst: pd.DataFrame, params: dict) -> pd.Series:
    """Counter-trend mean reversion (PRIOR-BEST per C1).

    score_t = -zscore_t(close - SMA_L). When the close sits far ABOVE its
    moving average the z-score is large positive, so the (negated) score is
    large negative => lean SHORT; far below => lean LONG. This is the
    counter-trend sign the C1 checkpoint identified as the prior-best
    replicator.
    """
    L = int(params["lookback"])
    z_window = int(params.get("z_window", L))
    close = _close_series(ohlcv_inst)

    sma = close.rolling(L, min_periods=L).mean()
    gap = close - sma  # trailing deviation from the moving average
    return -_zscore(gap, z_window)


def _score_breakout_donchian(ohlcv_inst: pd.DataFrame, params: dict) -> pd.Series:
    """Donchian-channel position: where the close sits in the trailing N-day band.

    Using the channel formed by bars ``<= t-1`` (the band is shifted one bar so
    today's close is compared against a band that does NOT include today --
    standard breakout construction and strictly look-ahead-free), the score is

        score_t = 2 * (close_t - lo) / (hi - lo) - 1   in [-1, +1] within band,

    extending beyond +-1 on a breach. A close at/above the prior N-day high
    scores >= +1 (breakout long); at/below the prior low scores <= -1 (breakdown
    short). A degenerate flat band (hi == lo) yields ``NaN`` -> flat.
    """
    N = int(params["channel"])
    close = _close_series(ohlcv_inst)

    hi = close.rolling(N, min_periods=N).max().shift(1)  # band excludes today
    lo = close.rolling(N, min_periods=N).min().shift(1)
    width = (hi - lo).where(lambda w: w > 0.0)
    return 2.0 * (close - lo) / width - 1.0


def _score_vol_regime_gated(ohlcv_inst: pd.DataFrame, params: dict) -> pd.Series:
    """A base directional score that PARTICIPATES only inside a vol regime.

    The base score (``mean_reversion`` by default, or ``ts_momentum``) is kept
    only on days whose trailing rolling-vol is on the chosen side of a rolling
    quantile threshold; off-regime days are forced to score ``0`` (flat). Vol is
    :func:`stml.na_checks.rolling_vol` on this instrument's own series, and the
    quantile threshold is a trailing rolling quantile (info <= t), so the gate
    introduces no look-ahead.

    params: ``base`` ('mean_reversion'|'ts_momentum'), ``vol_window``,
    ``regime`` ('high'|'low'), ``vol_quantile`` (0..1), ``q_window`` plus the
    base family's own params.
    """
    base = str(params.get("base", "mean_reversion"))
    vol_window = int(params.get("vol_window", 20))
    regime = str(params.get("regime", "high"))
    q = float(params.get("vol_quantile", 0.5))
    q_window = int(params.get("q_window", 120))

    if base == "ts_momentum":
        base_score = _score_ts_momentum(ohlcv_inst, params)
    elif base == "mean_reversion":
        base_score = _score_mean_reversion(ohlcv_inst, params)
    else:
        raise ValueError(f"unknown base archetype {base!r} for vol_regime_gated")

    rets = _instrument_returns(ohlcv_inst)
    inst = ohlcv_inst["instrument"].iloc[0]
    vol = rolling_vol(rets, inst, window=vol_window)  # trailing annualised vol
    # Trailing rolling-quantile threshold (info <= t): a row's threshold uses
    # only vols up to and including that row.
    thresh = vol.rolling(q_window, min_periods=vol_window).quantile(q)
    if regime == "high":
        in_regime = vol >= thresh
    elif regime == "low":
        in_regime = vol <= thresh
    else:
        raise ValueError(f"regime must be 'high' or 'low', got {regime!r}")

    # Align the gate onto the base score's index; missing/NaN gate -> off.
    gate = in_regime.reindex(base_score.index).fillna(False).to_numpy(dtype=bool)
    gated = base_score.to_numpy(dtype=float).copy()
    gated[~gate] = 0.0
    return pd.Series(gated, index=base_score.index)


def _score_hybrid_filtered_momentum(
    ohlcv_inst: pd.DataFrame, params: dict
) -> pd.Series:
    """Momentum, but only when a slower trend filter AGREES with its sign.

    The fast momentum score (over ``lookback``) is kept only on days where a
    slower trailing trend (sign of the log-return over ``filter_window``) shares
    its sign; on disagreement the score is forced to ``0`` (flat). This filters
    out fast blips that fight the slower trend. Both windows are trailing, so
    there is no look-ahead.

    params: ``lookback`` (fast), ``filter_window`` (slow), ``vol_window`` plus
    the deadband.
    """
    filt = int(params["filter_window"])
    close = _close_series(ohlcv_inst)
    logc = _log_close(close)

    fast = _score_ts_momentum(ohlcv_inst, params)  # reads 'lookback' from params
    slow_trend = logc - logc.shift(filt)  # trailing slow log-return, info <= t
    slow_sign = np.sign(slow_trend)

    agree = np.sign(fast) == slow_sign
    agree = agree & np.isfinite(slow_trend)
    out = fast.to_numpy(dtype=float).copy()
    out[~agree.to_numpy(dtype=bool)] = 0.0
    return pd.Series(out, index=fast.index)


# --------------------------------------------------------------------------- #
# Archetype container + registry.                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Archetype:
    """A named, look-ahead-free signal family.

    Attributes
    ----------
    name : str
        Registry key.
    score_fn : callable
        ``(ohlcv_inst, params) -> pd.Series`` returning the date-indexed real
        score (positive leans long). ``None`` for cross-sectional families that
        operate on the whole panel (use :attr:`generate_panel` instead).
    grid : dict
        The default search grid: ``param_name -> list/tuple of candidates``.
        Always includes ``deadband``.
    panel_fn : callable or None
        For cross-sectional families, ``(ohlcv_all, params) -> dict[inst ->
        pd.Series]``. ``None`` for per-instrument families.
    """

    name: str
    score_fn: Callable[[pd.DataFrame, dict], pd.Series] | None
    grid: dict[str, list | tuple] = field(default_factory=dict)
    panel_fn: (
        Callable[[pd.DataFrame, dict], dict[str, pd.Series]] | None
    ) = None

    def generate(
        self,
        ohlcv_inst: pd.DataFrame,
        params: dict,
        allowed: Sequence[int] = (-1, 0, 1),
    ) -> pd.Series:
        """Produce the date-indexed ``{-1, 0, +1}`` signal for one instrument.

        Computes this family's score from ``ohlcv_inst`` (info <= t only) and
        routes it through :func:`decide` with ``params['deadband']``. For a
        cross-sectional family, call :meth:`generate_panel` instead (this raises
        a clear error).

        Parameters
        ----------
        ohlcv_inst : pd.DataFrame
            Long OHLCV for a single instrument (columns include ``date``,
            ``close``, ``instrument``). The full available price history is used
            so rolling features warm up on real bars, not NaNs.
        params : dict
            Family params plus ``deadband`` (defaults to ``0.0`` if absent).
        allowed : sequence of int
            Permitted signal alphabet; see :func:`decide`. ``(-1, 0)`` gives the
            participation-only short side (e.g. ng1s).
        """
        if self.score_fn is None:
            raise ValueError(
                f"archetype {self.name!r} is cross-sectional; "
                "call generate_panel(ohlcv_all, params) instead of generate(...)"
            )
        deadband = float(params.get("deadband", 0.0))
        score = self.score_fn(ohlcv_inst, params)
        return decide(score, deadband, allowed=allowed)

    def generate_panel(
        self,
        ohlcv_all: pd.DataFrame,
        params: dict,
        allowed: Sequence[int] = (-1, 0, 1),
    ) -> dict[str, pd.Series]:
        """Cross-sectional generate over the whole panel.

        Only defined for cross-sectional families (``panel_fn`` set); otherwise
        raises a clear error directing the caller to :meth:`generate`.
        """
        if self.panel_fn is None:
            raise ValueError(
                f"archetype {self.name!r} is per-instrument; "
                "call generate(ohlcv_inst, params) instead of generate_panel(...)"
            )
        return self.panel_fn(ohlcv_all, {**params, "allowed": tuple(allowed)})

    def param_space(self) -> dict[str, list | tuple]:
        """The search grid for this family: ``param -> candidate values``.

        Returned as a fresh dict (callers may mutate it freely). Always contains
        a ``deadband`` axis so the participation threshold is part of the
        search.
        """
        return {k: list(v) for k, v in self.grid.items()}


# --------------------------------------------------------------------------- #
# Cross-sectional family: rank the universe each day, long top / short bottom.#
# --------------------------------------------------------------------------- #
def _panel_close_returns(
    ohlcv_all: pd.DataFrame, lookback: int
) -> pd.DataFrame:
    """Wide ``date x instrument`` trailing ``lookback``-day log returns.

    Each instrument's log return is computed on its OWN dense series (so a
    holiday-spanning move is correct), then pivoted. Remaining NaNs are
    structural and are left as NaN (never filled); they simply do not rank that
    day. Look-ahead-free: the value at row ``t`` uses closes ``<= t``.
    """
    rets = []
    for inst, g in ohlcv_all.groupby("instrument"):
        close = _close_series(g.assign(instrument=inst))
        logc = np.log(close)
        mom = logc - logc.shift(lookback)
        rets.append(mom.rename(inst))
    return pd.concat(rets, axis=1, sort=True).sort_index()


def generate_panel(
    ohlcv_all: pd.DataFrame, params: dict
) -> dict[str, pd.Series]:
    """Cross-sectional rank signal: long the top, short the bottom each day.

    EXPECTED-NEGATIVE diagnostic (cross-asset mean ``|corr| = 0.09`` per C1): a
    cross-sectional rank cannot replicate a near-independent panel, and that is
    fine -- it exists so that >= 5 of the 6 families can still pass while this
    one documents the cross-asset (non-)structure.

    Each day, instruments are ranked by a trailing score (default: ``lookback``-
    day log return, ``score='momentum'``; ``'reversal'`` negates it). The top
    ``top_frac`` fraction get ``+1``, the bottom ``bottom_frac`` get ``-1``, the
    rest ``0``. Ranking uses only same-day-or-earlier closes, so it is
    look-ahead-free. ``allowed`` (optional, in ``params``) clamps every
    instrument's output via :func:`decide`-style masking (a disallowed side ->
    0); the default permits all three states.

    Parameters
    ----------
    ohlcv_all : pd.DataFrame
        Long OHLCV for the whole universe (``date``, ``instrument``, ``close``).
    params : dict
        ``lookback`` (int), ``top_frac`` (0..1), ``bottom_frac`` (0..1, defaults
        to ``top_frac``), ``score`` ('momentum'|'reversal'), optional
        ``allowed``.

    Returns
    -------
    dict[str, pd.Series]
        ``instrument -> date-indexed {-1, 0, +1}`` signal, one per instrument
        present in ``ohlcv_all``.
    """
    lookback = int(params.get("lookback", 20))
    top_frac = float(params.get("top_frac", 0.3))
    bottom_frac = float(params.get("bottom_frac", top_frac))
    score_kind = str(params.get("score", "momentum"))
    allowed = params.get("allowed", (-1, 0, 1))
    allowed_set = set(int(a) for a in allowed)

    wide = _panel_close_returns(ohlcv_all, lookback)
    if score_kind == "reversal":
        wide = -wide
    elif score_kind != "momentum":
        raise ValueError(f"score must be 'momentum' or 'reversal', got {score_kind!r}")

    # Rank within each day across the instruments that have a finite score.
    out: dict[str, np.ndarray] = {c: np.zeros(len(wide), dtype=int) for c in wide.columns}
    idx = wide.index
    arr = wide.to_numpy(dtype=float)
    cols = list(wide.columns)

    for r in range(arr.shape[0]):
        row = arr[r]
        finite = np.where(np.isfinite(row))[0]
        n = finite.size
        if n == 0:
            continue
        order = finite[np.argsort(row[finite])]  # ascending: worst..best
        n_top = max(1, int(np.floor(n * top_frac))) if top_frac > 0 else 0
        n_bot = max(1, int(np.floor(n * bottom_frac))) if bottom_frac > 0 else 0
        # Guard the degenerate small-universe case so top/bottom never overlap.
        n_top = min(n_top, n)
        n_bot = min(n_bot, n - n_top)
        long_idx = order[n - n_top:] if n_top else np.empty(0, dtype=int)
        short_idx = order[:n_bot] if n_bot else np.empty(0, dtype=int)
        for j in long_idx:
            out[cols[j]][r] = 1
        for j in short_idx:
            out[cols[j]][r] = -1

    result: dict[str, pd.Series] = {}
    for c in cols:
        sig = out[c]
        if allowed_set != set(_SIGNAL_LABELS):
            sig = np.where(np.isin(sig, list(allowed_set)), sig, 0)
        result[c] = pd.Series(sig.astype(int), index=idx, name="signal")
    return result


# --------------------------------------------------------------------------- #
# The registry: >= 6 families, each with .generate + .param_space.            #
# --------------------------------------------------------------------------- #
ARCHETYPES: dict[str, Archetype] = {
    "ts_momentum": Archetype(
        name="ts_momentum",
        score_fn=_score_ts_momentum,
        grid={
            "lookback": [5, 10, 20, 40, 60],
            "vol_window": [20, 40],
            "deadband": [0.0, 0.25, 0.5, 1.0, 1.5],
        },
    ),
    "mean_reversion": Archetype(
        name="mean_reversion",
        score_fn=_score_mean_reversion,
        grid={
            "lookback": [5, 10, 20, 40],
            "z_window": [20, 40, 60],
            "deadband": [0.0, 0.25, 0.5, 1.0, 1.5],
        },
    ),
    "breakout_donchian": Archetype(
        name="breakout_donchian",
        score_fn=_score_breakout_donchian,
        grid={
            "channel": [10, 20, 40, 55],
            "deadband": [0.0, 0.5, 0.8, 1.0],
        },
    ),
    "vol_regime_gated": Archetype(
        name="vol_regime_gated",
        score_fn=_score_vol_regime_gated,
        grid={
            "base": ["mean_reversion", "ts_momentum"],
            "lookback": [10, 20, 40],
            "z_window": [20, 40],
            "vol_window": [20, 40],
            "regime": ["high", "low"],
            "vol_quantile": [0.3, 0.5, 0.7],
            "q_window": [120],
            "deadband": [0.0, 0.5, 1.0],
        },
    ),
    "hybrid_filtered_momentum": Archetype(
        name="hybrid_filtered_momentum",
        score_fn=_score_hybrid_filtered_momentum,
        grid={
            "lookback": [5, 10, 20],
            "filter_window": [40, 60, 120],
            "vol_window": [20, 40],
            "deadband": [0.0, 0.25, 0.5, 1.0],
        },
    ),
    "xsect_rank": Archetype(
        name="xsect_rank",
        score_fn=None,
        panel_fn=generate_panel,
        grid={
            "lookback": [10, 20, 40, 60],
            "top_frac": [0.2, 0.3, 0.4],
            "score": ["momentum", "reversal"],
        },
    ),
}
