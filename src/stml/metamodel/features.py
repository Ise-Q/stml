"""
features.py
===========
Engineered (**E-class**) feature families for the triple-barrier metamodel
(US-FE-002). Each family maps one instrument's full OHLCV history (and, for F5,
its released signal series) to a date-indexed block of features that are
**look-ahead-free by construction**.

Leakage contract (the crux — graded)
-------------------------------------
Every engineered value at date ``t`` is built only from rows ``<= t`` using
**trailing** operations: right-aligned ``rolling(L)``, ``shift(+k)`` with
``k >= 0``, ``diff``, and cumulative-from-left scans. None of these families
uses ``shift(-k)``, a centred/forward window, or a full-series percentile/rank
that peeks ahead. The operational consequence — asserted directly by
``tests/test_features_leakage.py`` — is **truncation-invariance**: truncating
the inputs at any date ``T`` and recomputing reproduces the identical value on
every date ``< T``.

Two families carry the highest leakage risk and so are pinned explicitly:

* **F1 ``f1_mr_score_20``** is the C1 counter-trend mean-reversion score
  ``score_t = -zscore_t(close - SMA_L)`` (the highest-value replicator: a close
  far ABOVE its average leans short).
* **F5 ``f5_trailing_run_length`` / ``f5_days_since_flip``** are computed from
  ``signal_inst.iloc[:i + 1]`` ONLY (an expanding, cumulative-from-left scan),
  **never** from :func:`stml.metamodel.splits.run_length_p90` (which measures
  the full released period and therefore leaks the future into the past). The
  embargo-sizing ``run_length_p90`` and these trailing run-length features are
  deliberately distinct quantities; the catalog documents the distinction.

Structural NaNs (pre-warm-up rows, a gapped input, missing volume) are **never**
forward-filled or ``fillna(0)``-ed; they propagate as ``NaN`` exactly as the
upstream :mod:`stml.na_checks` helpers leave them. Returns and rolling vol are
reused from :mod:`stml.na_checks` so they honour each instrument's own dense
calendar (holiday-spanning moves are correct, never fabricated zeros).

Inputs
------
``ohlcv_inst`` is a long OHLCV frame for ONE instrument over its FULL history
(columns include ``date, open, high, low, close, volume, open_interest`` and an
``instrument`` column). The full history is passed so rolling windows warm up on
real pre-2020 bars rather than NaNs. ``signal_inst`` is a date-indexed signal
series ``s_t in {-1, 0, +1}`` over the 645-day released window.

Public API
----------
- :func:`f1_counter_trend`     — counter-trend / mean-reversion family.
- :func:`f2_vol_dispersion`    — volatility & dispersion family (label-input σ).
- :func:`f5_signal_derived`    — signal-derived family (trailing run structure).
- :func:`f6_momentum_contrast` — momentum & trend-contrast family.
- :func:`f7_microstructure`    — volume / open-interest microstructure family.
- :func:`f8_calendar`          — deterministic calendar (sin/cos) family.
- :func:`f10_price_action`     — OHLC price-action returns (range / open-to-open).
- :func:`assemble_engineered`  — concat every family on the date index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stml.na_checks import native_returns, rolling_vol

__all__ = [
    "f1_counter_trend",
    "f2_vol_dispersion",
    "f5_signal_derived",
    "f6_momentum_contrast",
    "f7_microstructure",
    "f8_calendar",
    "f10_price_action",
    "assemble_engineered",
]

# Annualisation factor for trading-day vol (matches na_checks.rolling_vol).
_ANN = 252.0
# The label-input volatility window: f2_vol_20 is the σ subset the deferred
# triple-barrier label consumes, computed exactly like na_checks.rolling_vol.
_LABEL_VOL_WINDOW = 20


# --------------------------------------------------------------------------- #
# Shared trailing helpers (info <= t), mirroring archetypes.py conventions.   #
# --------------------------------------------------------------------------- #
def _ohlcv_indexed(ohlcv_inst: pd.DataFrame) -> pd.DataFrame:
    """Date-indexed, de-duplicated, sorted OHLCV for one instrument.

    Builds a clean per-instrument OHLCV frame retaining the full columns
    (open/high/low/close/volume/open_interest). The index is
    this instrument's own trading calendar — no calendar grid is imposed — so
    rolling windows span real bars and never fabricate gaps.

    Parameters
    ----------
    ohlcv_inst : pd.DataFrame
        Long OHLCV for a single instrument; must contain a ``date`` column and
        the price/volume columns. Rows with a non-finite ``close`` are dropped
        (a close is required to anchor every price feature).

    Returns
    -------
    pd.DataFrame
        Float OHLCV indexed by a sorted, unique ``DatetimeIndex``.
    """
    cols = ["date", "open", "high", "low", "close", "volume", "open_interest"]
    present = [c for c in cols if c in ohlcv_inst.columns]
    df = (
        ohlcv_inst[present]
        .dropna(subset=["close"])
        .drop_duplicates("date")
        .sort_values("date")
        .set_index("date")
    )
    return df.astype(float)


def _zscore(x: pd.Series, window: int) -> pd.Series:
    """Trailing z-score ``(x - rolling_mean) / rolling_std`` over ``window``.

    Pandas ``rolling`` is right-aligned and trailing (info ``<= t``), and a
    zero/NaN rolling std (a flat window) yields ``NaN`` for that row rather than
    a divide-by-zero blow-up.
    """
    roll = x.rolling(window, min_periods=window)
    mu = roll.mean()
    sd = roll.std()
    sd = sd.where(sd > 0.0)
    return (x - mu) / sd


def _rolling_percentile(x: pd.Series, window: int) -> pd.Series:
    """Trailing rank of the latest value within its own ``window`` (in [0, 1]).

    For each row ``t`` this is the fraction of the trailing ``window`` (rows
    ``t - window + 1 .. t``, inclusive of ``t``) whose value is ``<=`` the value
    at ``t``. Uses only same-or-earlier rows, so it is look-ahead-free; warm-up
    rows (fewer than ``window`` observations) are ``NaN``.
    """

    def _last_rank(a: np.ndarray) -> float:
        last = a[-1]
        if not np.isfinite(last):
            return np.nan
        return float(np.mean(a <= last))

    return x.rolling(window, min_periods=window).apply(_last_rank, raw=True)


def _instrument_returns(ohlcv_inst: pd.DataFrame) -> pd.DataFrame:
    """Long return frame (``instrument``, ``date``, ``ret``) for one instrument.

    Thin wrapper over :func:`stml.na_checks.native_returns` (kind ``"log"``) so
    the long-layout :mod:`stml.na_checks` helpers can be reused unchanged.
    """
    return native_returns(ohlcv_inst, kind="log")


# --------------------------------------------------------------------------- #
# F1 — Counter-trend / mean reversion (C1 highest-value family).              #
# --------------------------------------------------------------------------- #
def f1_counter_trend(ohlcv_inst: pd.DataFrame) -> pd.DataFrame:
    """Counter-trend (mean-reversion) features for one instrument.

    The prime column ``f1_mr_score_20`` is the C1 counter-trend score
    ``-zscore_20(close - SMA_20)``. A positive score (close far below its average)
    leans LONG; a negative score (close far above) leans SHORT — the C1
    counter-trend sign, identified as the highest-value replicator.

    Columns
    -------
    f1_mr_score_{L} : ``-zscore_L(close - SMA_L)`` for ``L in {10, 20, 40}``;
        ``f1_mr_score_20`` is the C1 prime.
    f1_dist_ma_sigma_{L} : ``(close - SMA_L)`` in trailing-σ units of the gap.
    f1_ret_reversal_{L} : negated trailing ``L``-day log return (reversal lean).
    f1_rsi_14 : Wilder RSI(14) on closes, in [0, 100].
    f1_bb_pctb_20 : Bollinger %b ``(close - lower) / (upper - lower)``, 20/2σ.
    f1_bb_bandwidth_20 : Bollinger bandwidth ``(upper - lower) / SMA_20``.
    f1_hilo_pos_{L} : position of close in its trailing ``L``-day [min, max].

    Parameters
    ----------
    ohlcv_inst : pd.DataFrame
        Full-history long OHLCV for one instrument.

    Returns
    -------
    pd.DataFrame
        Date-indexed float features (NaN on warm-up rows; never ffilled).
    """
    df = _ohlcv_indexed(ohlcv_inst)
    close = df["close"]
    logc = np.log(close)
    out: dict[str, pd.Series] = {}

    for L in (10, 20, 40):
        sma = close.rolling(L, min_periods=L).mean()
        gap = close - sma
        # f1_mr_score_L = -zscore_L(close - SMA_L) — mirrors archetypes.
        out[f"f1_mr_score_{L}"] = -_zscore(gap, L)
        # Distance from the MA in trailing-σ units of the gap itself.
        gap_sd = gap.rolling(L, min_periods=L).std().where(lambda s: s > 0.0)
        out[f"f1_dist_ma_sigma_{L}"] = gap / gap_sd
        # Trailing L-day return, negated => a reversal (counter-trend) lean.
        out[f"f1_ret_reversal_{L}"] = -(logc - logc.shift(L))
        # Position of the close within its trailing L-day range, in [0, 1].
        lo = close.rolling(L, min_periods=L).min()
        hi = close.rolling(L, min_periods=L).max()
        width = (hi - lo).where(lambda w: w > 0.0)
        out[f"f1_hilo_pos_{L}"] = (close - lo) / width

    # Wilder RSI(14): trailing exponential averages of up/down moves.
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.where(avg_loss > 0.0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # A flat-loss window (avg_loss == 0) with gains => RSI 100; both zero => NaN.
    rsi = rsi.where(~((avg_loss == 0.0) & (avg_gain > 0.0)), other=100.0)
    out["f1_rsi_14"] = rsi

    # Bollinger Bands (20, 2σ): %b and bandwidth.
    sma20 = close.rolling(20, min_periods=20).mean()
    sd20 = close.rolling(20, min_periods=20).std()
    upper = sma20 + 2.0 * sd20
    lower = sma20 - 2.0 * sd20
    band = (upper - lower).where(lambda w: w > 0.0)
    out["f1_bb_pctb_20"] = (close - lower) / band
    out["f1_bb_bandwidth_20"] = band / sma20.where(sma20 != 0.0)

    return pd.DataFrame(out, index=df.index)


# --------------------------------------------------------------------------- #
# F2 — Volatility & dispersion (f2_vol_20 = the label-input σ subset).        #
# --------------------------------------------------------------------------- #
def f2_vol_dispersion(ohlcv_inst: pd.DataFrame) -> pd.DataFrame:
    """Volatility and dispersion features for one instrument.

    ``f2_vol_20`` is computed by :func:`stml.na_checks.rolling_vol` (annualised
    trailing std of log returns, window 20) and is the **label-input σ subset**
    the deferred triple-barrier label consumes — surfaced here as an ordinary
    feature column so the catalog can flag its leakage class.

    Columns
    -------
    f2_vol_{L} : annualised trailing rolling vol (na_checks.rolling_vol) for
        ``L in {10, 20, 60}``; ``f2_vol_20`` is the label-input σ.
    f2_vol_ratio_20_60 : ``f2_vol_20 / f2_vol_60`` (short-vs-long vol regime).
    f2_vol_pctile_20 : trailing 1y (252-day) percentile of ``f2_vol_20``.
    f2_vol_of_vol_20 : trailing 60-day std of ``f2_vol_20`` (vol-of-vol).
    f2_parkinson_20 : Parkinson high-low range volatility (annualised).
    f2_garman_klass_20 : Garman-Klass OHLC volatility (annualised).
    f2_atr_14 : Wilder Average True Range over 14 days (price units).
    f2_ret_skew_60 / f2_ret_kurt_60 : trailing 60-day return skew / excess kurt.

    Parameters
    ----------
    ohlcv_inst : pd.DataFrame
        Full-history long OHLCV for one instrument.

    Returns
    -------
    pd.DataFrame
        Date-indexed float features (NaN on warm-up rows; never ffilled).
    """
    df = _ohlcv_indexed(ohlcv_inst)
    rets = _instrument_returns(ohlcv_inst)
    inst = ohlcv_inst["instrument"].iloc[0]
    out: dict[str, pd.Series] = {}

    # Annualised rolling vol from na_checks (native dense series), reindexed
    # onto this family's date index. f2_vol_20 is the label-input σ subset.
    for L in (10, _LABEL_VOL_WINDOW, 60):
        out[f"f2_vol_{L}"] = rolling_vol(rets, inst, window=L).reindex(df.index)

    v20 = out[f"f2_vol_{_LABEL_VOL_WINDOW}"]
    v60 = out["f2_vol_60"]
    out["f2_vol_ratio_20_60"] = v20 / v60.where(v60 > 0.0)
    out["f2_vol_pctile_20"] = _rolling_percentile(v20, 252)
    out["f2_vol_of_vol_20"] = v20.rolling(60, min_periods=60).std()

    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    # Parkinson (high-low) volatility, annualised.
    log_hl = np.log((high / low).where((high > 0.0) & (low > 0.0)))
    park_var = (log_hl**2) / (4.0 * np.log(2.0))
    out["f2_parkinson_20"] = np.sqrt(
        park_var.rolling(20, min_periods=20).mean() * _ANN
    )

    # Garman-Klass volatility, annualised.
    log_co = np.log((close / df["open"]).where((close > 0.0) & (df["open"] > 0.0)))
    gk_var = 0.5 * (log_hl**2) - (2.0 * np.log(2.0) - 1.0) * (log_co**2)
    out["f2_garman_klass_20"] = np.sqrt(
        gk_var.rolling(20, min_periods=20).mean().clip(lower=0.0) * _ANN
    )

    # Wilder ATR(14): EW mean of the true range.
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["f2_atr_14"] = tr.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()

    # Trailing return distribution shape on this instrument's native returns.
    ret_s = (
        rets.loc[rets["instrument"] == inst]
        .set_index("date")["ret"]
        .sort_index()
        .reindex(df.index)
    )
    out["f2_ret_skew_60"] = ret_s.rolling(60, min_periods=60).skew()
    out["f2_ret_kurt_60"] = ret_s.rolling(60, min_periods=60).kurt()

    return pd.DataFrame(out, index=df.index)


# --------------------------------------------------------------------------- #
# F5 — Signal-derived (TRAILING run structure on s[:t+1] ONLY).              #
# --------------------------------------------------------------------------- #
def _trailing_run_length(values: np.ndarray) -> np.ndarray:
    """Length of the current constant-value run ending at each position.

    Cumulative-from-left scan: position ``i`` reports how many consecutive
    rows ``.. i`` (inclusive) share ``values[i]``, i.e. it depends only on rows
    ``<= i``. ``[1, 1, 0, 0, 0, 1] -> [1, 2, 1, 2, 3, 1]``. A NaN value starts /
    continues its own run by identity (``NaN != NaN``), so a run breaks at any
    change including to/from NaN.
    """
    n = values.size
    run = np.empty(n, dtype=float)
    if n == 0:
        return run
    run[0] = 1.0
    for i in range(1, n):
        same = values[i] == values[i - 1]
        run[i] = run[i - 1] + 1.0 if same else 1.0
    return run


def _days_since_flip(values: np.ndarray) -> np.ndarray:
    """Trailing days since the value last changed (the current run length - 1).

    Cumulative-from-left: ``0`` on the first row and on any row whose value
    differs from the previous row (a flip just happened), otherwise one more
    than the previous row. Depends only on rows ``<= i``.
    """
    return _trailing_run_length(values) - 1.0


def _days_since_nonzero(values: np.ndarray) -> np.ndarray:
    """Trailing days since the last non-zero (participating) signal.

    ``0`` on a non-zero row; otherwise one more than the previous row's count.
    Before the first non-zero observation the count is ``NaN`` (there is no
    prior participation to measure from). Cumulative-from-left (info ``<= i``).
    """
    n = values.size
    out = np.empty(n, dtype=float)
    since = np.nan
    for i in range(n):
        v = values[i]
        if np.isfinite(v) and v != 0.0:
            since = 0.0
        elif np.isfinite(since):
            since = since + 1.0
        out[i] = since
    return out


def f5_signal_derived(
    signal_inst: pd.Series, mr_score: pd.Series | None = None
) -> pd.DataFrame:
    """Signal-derived features from the released signal series (trailing only).

    Every column is a **trailing / cumulative-from-left** function of
    ``signal_inst.iloc[:i + 1]`` — never of the full released period. In
    particular ``f5_trailing_run_length`` and ``f5_days_since_flip`` are
    expanding scans (MUST-FIX-2): they are deliberately distinct from
    :func:`stml.metamodel.splits.run_length_p90` (a full-period statistic used
    only for embargo sizing, which would leak the future if used as a feature).

    Columns
    -------
    f5_signal : the released signal ``s_t in {-1, 0, +1}``.
    f5_abs_signal : ``|s_t|`` (participation indicator).
    f5_trailing_run_length : length of the current constant-``s`` run ending at
        ``t`` (cumulative-from-left; HIGH-RISK MUST-FIX-2 column).
    f5_days_since_flip : trailing days since ``s`` last changed (run length - 1;
        HIGH-RISK MUST-FIX-2 column).
    f5_days_since_nonzero : trailing days since the last participating signal.
    f5_participation_20 / f5_participation_60 : trailing mean ``|s|`` (share of
        participating days) over 20 / 60 days.
    f5_long_bias_20 : trailing mean ``s`` over 20 days (net long/short tilt).
    f5_sign_agree_mr : sign agreement of ``s_t`` with ``mr_score`` (``+1`` agree,
        ``-1`` disagree, ``0`` when either is flat/NaN); all-NaN if ``mr_score``
        is not supplied.

    Parameters
    ----------
    signal_inst : pd.Series
        Date-indexed signal ``s_t`` over the released window.
    mr_score : pd.Series, optional
        Date-indexed counter-trend score (e.g. ``f1_mr_score_20``) for the
        sign-agreement column. When ``None`` the agreement column is all-NaN.

    Returns
    -------
    pd.DataFrame
        Date-indexed float features (never ffilled).
    """
    s = pd.Series(signal_inst).sort_index()
    idx = s.index
    vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    out: dict[str, pd.Series] = {}

    out["f5_signal"] = s.astype(float)
    out["f5_abs_signal"] = s.abs().astype(float)
    out["f5_trailing_run_length"] = pd.Series(
        _trailing_run_length(vals), index=idx
    )
    out["f5_days_since_flip"] = pd.Series(_days_since_flip(vals), index=idx)
    out["f5_days_since_nonzero"] = pd.Series(_days_since_nonzero(vals), index=idx)

    abs_s = s.abs().astype(float)
    out["f5_participation_20"] = abs_s.rolling(20, min_periods=20).mean()
    out["f5_participation_60"] = abs_s.rolling(60, min_periods=60).mean()
    out["f5_long_bias_20"] = s.astype(float).rolling(20, min_periods=20).mean()

    if mr_score is not None:
        mr = pd.Series(mr_score).reindex(idx)
        agree = np.sign(s.astype(float)) * np.sign(mr)
        out["f5_sign_agree_mr"] = agree.astype(float)
    else:
        out["f5_sign_agree_mr"] = pd.Series(np.nan, index=idx, dtype=float)

    return pd.DataFrame(out, index=idx)


# --------------------------------------------------------------------------- #
# F6 — Momentum & trend contrast (mirrors ts_momentum / breakout_donchian).   #
# --------------------------------------------------------------------------- #
def f6_momentum_contrast(ohlcv_inst: pd.DataFrame) -> pd.DataFrame:
    """Momentum and trend-contrast features for one instrument.

    ``f6_ts_momentum_20`` is the trailing
    ``L``-day log return divided by a trailing return-vol scale
    (``daily_std * sqrt(L)``). ``f6_donchian_pos_20`` is the close's
    position in the prior ``N``-day channel (band excludes today), in [-1, +1]
    within band and beyond on a breach.

    Columns
    -------
    f6_ts_momentum_{L} : vol-scaled trailing ``L``-day log return for
        ``L in {20, 60}`` (ts_momentum score).
    f6_ma_cross_20_60 : ``(SMA_20 - SMA_60) / SMA_60`` (fast-vs-slow MA cross).
    f6_macd_12_26 : MACD line ``EMA_12 - EMA_26`` on closes (price units).
    f6_macd_hist_12_26_9 : MACD histogram ``MACD - EMA_9(MACD)``.
    f6_adx_14 : Wilder ADX(14) trend strength, in [0, 100].
    f6_donchian_pos_20 : close position in the prior 20-day Donchian channel.

    Parameters
    ----------
    ohlcv_inst : pd.DataFrame
        Full-history long OHLCV for one instrument.

    Returns
    -------
    pd.DataFrame
        Date-indexed float features (NaN on warm-up rows; never ffilled).
    """
    df = _ohlcv_indexed(ohlcv_inst)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    logc = np.log(close)
    daily = logc.diff()
    out: dict[str, pd.Series] = {}

    for L in (20, 60):
        raw = logc - logc.shift(L)
        scale = daily.rolling(L, min_periods=L).std() * np.sqrt(L)
        scale = scale.where(scale > 0.0)
        out[f"f6_ts_momentum_{L}"] = raw / scale

    sma20 = close.rolling(20, min_periods=20).mean()
    sma60 = close.rolling(60, min_periods=60).mean()
    out["f6_ma_cross_20_60"] = (sma20 - sma60) / sma60.where(sma60 != 0.0)

    ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
    macd = ema12 - ema26
    out["f6_macd_12_26"] = macd
    signal_line = macd.ewm(span=9, min_periods=9, adjust=False).mean()
    out["f6_macd_hist_12_26_9"] = macd - signal_line

    # Wilder ADX(14): directional movement smoothed by Wilder's EW average.
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), other=0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), other=0.0)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    alpha = 1.0 / 14.0
    atr = tr.ewm(alpha=alpha, min_periods=14, adjust=False).mean()
    atr_pos = atr.where(atr > 0.0)
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, min_periods=14, adjust=False).mean() / atr_pos
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, min_periods=14, adjust=False).mean() / atr_pos
    di_sum = (plus_di + minus_di).where(lambda x: x > 0.0)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    out["f6_adx_14"] = dx.ewm(alpha=alpha, min_periods=14, adjust=False).mean()

    # Donchian position in the prior 20-day band (band excludes today).
    hi = close.rolling(20, min_periods=20).max().shift(1)
    lo = close.rolling(20, min_periods=20).min().shift(1)
    width = (hi - lo).where(lambda w: w > 0.0)
    out["f6_donchian_pos_20"] = 2.0 * (close - lo) / width - 1.0

    return pd.DataFrame(out, index=df.index)


# --------------------------------------------------------------------------- #
# F7 — Microstructure (volume / open-interest; Amihud zero-volume->NaN).      #
# --------------------------------------------------------------------------- #
def f7_microstructure(ohlcv_inst: pd.DataFrame) -> pd.DataFrame:
    """Microstructure (volume / open-interest) features for one instrument.

    ``f7_amihud_20`` is the trailing 20-day mean of the Amihud illiquidity ratio
    ``|ret| / volume`` with a **zero-volume -> NaN guard** (the contract's
    minor-fix): a zero-volume day contributes ``NaN`` rather than dividing by
    zero. Open-interest columns are left as **structural NaN** wherever OI is
    missing (some instruments have no OI) — never ffilled or zero-filled.

    Columns
    -------
    f7_volume_z_20 : trailing z-score of volume (20-day).
    f7_volume_trend_20 : ``volume / SMA_20(volume) - 1`` (volume trend).
    f7_oi_level : raw open interest (structural NaN where missing).
    f7_oi_change : 1-day change in open interest.
    f7_oi_z_20 : trailing z-score of open interest (20-day).
    f7_oi_price_div_20 : sign divergence between trailing OI change and price
        change (``+1`` aligned, ``-1`` divergent, ``0`` either flat).
    f7_amihud_20 : trailing 20-day mean Amihud ``|ret| / volume`` (zero-volume
        rows -> NaN, never divide-by-zero).

    Parameters
    ----------
    ohlcv_inst : pd.DataFrame
        Full-history long OHLCV for one instrument.

    Returns
    -------
    pd.DataFrame
        Date-indexed float features (structural NaN preserved; never ffilled).
    """
    df = _ohlcv_indexed(ohlcv_inst)
    close = df["close"]
    volume = df["volume"]
    oi = df["open_interest"] if "open_interest" in df.columns else pd.Series(
        np.nan, index=df.index, dtype=float
    )
    out: dict[str, pd.Series] = {}

    out["f7_volume_z_20"] = _zscore(volume, 20)
    vol_sma = volume.rolling(20, min_periods=20).mean()
    out["f7_volume_trend_20"] = volume / vol_sma.where(vol_sma > 0.0) - 1.0

    out["f7_oi_level"] = oi
    oi_change = oi.diff()
    out["f7_oi_change"] = oi_change
    out["f7_oi_z_20"] = _zscore(oi, 20)

    # OI-price divergence over a trailing 20-day window: sign of the OI change
    # vs the sign of the price change. Zero on either side -> 0 (no signal).
    oi_chg_20 = oi - oi.shift(20)
    px_chg_20 = close - close.shift(20)
    out["f7_oi_price_div_20"] = (np.sign(oi_chg_20) * np.sign(px_chg_20)).astype(float)

    # Amihud illiquidity |ret| / volume with a zero-volume -> NaN guard.
    ret = np.log(close).diff()
    safe_vol = volume.where(volume > 0.0)  # zero/negative volume -> NaN (no /0)
    amihud = ret.abs() / safe_vol
    out["f7_amihud_20"] = amihud.rolling(20, min_periods=1).mean()

    return pd.DataFrame(out, index=df.index)


# --------------------------------------------------------------------------- #
# F10 — OHLC price-action returns (intraday range / open-to-open).            #
# --------------------------------------------------------------------------- #
def f10_price_action(ohlcv_inst: pd.DataFrame) -> pd.DataFrame:
    """OHLC price-action return features for one instrument.

    Two raw OHLC-derived returns the other families only consume in aggregate
    form, surfaced here as first-class features. Both use **log** returns to
    match the layer's house convention (:func:`stml.na_checks.native_returns`
    ``kind="log"``), each as a single-day value plus its trailing 20-day mean:

    * the intraday **high-low log range** ``log(high / low)`` — the per-bar
      trading range (the same quantity :func:`f2_vol_dispersion`'s Parkinson vol
      aggregates over 20 days, exposed here as the raw daily range and a typical
      recent range); and
    * the **open-to-open log return** ``log(open_t / open_{t-1})`` — the
      open-anchored counterpart to the close-based returns used by F1/F6,
      capturing the overnight-inclusive move a close-to-close return blends away.

    Leakage contract
    ----------------
    Look-ahead-free by construction: the daily range uses only the same-day
    high/low (info ``<= t``); the open-to-open return uses today's and
    yesterday's open (``shift(+1)``, info ``<= t``); the 20-day means are
    right-aligned trailing ``rolling(20)`` averages. Non-positive prices map to
    ``NaN`` (never an ``inf`` — mirrors the Parkinson guard in
    :func:`f2_vol_dispersion`), and warm-up rows are ``NaN`` and never ffilled.

    Columns
    -------
    f10_hl_range : daily intraday high-low log range ``log(high / low)``.
    f10_hl_range_mean_20 : trailing 20d mean of ``f10_hl_range`` (typical range).
    f10_oto_ret : daily open-to-open log return ``log(open_t / open_{t-1})``.
    f10_oto_ret_mean_20 : trailing 20d mean of ``f10_oto_ret`` (open-drift).

    Parameters
    ----------
    ohlcv_inst : pd.DataFrame
        Full-history long OHLCV for one instrument.

    Returns
    -------
    pd.DataFrame
        Date-indexed float features (NaN on warm-up rows; never ffilled).
    """
    df = _ohlcv_indexed(ohlcv_inst)
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    out: dict[str, pd.Series] = {}

    # Intraday high-low log range; non-positive high/low -> NaN (never inf),
    # matching the f2_parkinson_20 guard so a bad bar cannot blow up.
    hl_range = np.log((high / low).where((high > 0.0) & (low > 0.0)))
    out["f10_hl_range"] = hl_range
    out["f10_hl_range_mean_20"] = hl_range.rolling(20, min_periods=20).mean()

    # Open-to-open log return; needs both today's and yesterday's open > 0,
    # else NaN. The first row is NaN by construction (no prior open).
    prev_open = open_.shift(1)
    oto = np.log((open_ / prev_open).where((open_ > 0.0) & (prev_open > 0.0)))
    out["f10_oto_ret"] = oto
    out["f10_oto_ret_mean_20"] = oto.rolling(20, min_periods=20).mean()

    return pd.DataFrame(out, index=df.index)


# --------------------------------------------------------------------------- #
# F8 — Calendar (deterministic sin/cos of day-of-week and month).            #
# --------------------------------------------------------------------------- #
def f8_calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic calendar features (sin/cos of day-of-week and month).

    Purely a function of the date index — trivially look-ahead-free (no data
    enters at all). Cyclical sin/cos encodings keep the wrap-around continuous
    (Friday is adjacent to Monday; December to January).

    Columns
    -------
    f8_dow_sin / f8_dow_cos : day-of-week (Mon=0..Sun=6) on a 7-cycle.
    f8_month_sin / f8_month_cos : month (1..12) on a 12-cycle.

    Parameters
    ----------
    index : pd.DatetimeIndex
        The date index to encode.

    Returns
    -------
    pd.DataFrame
        Date-indexed float calendar features.
    """
    idx = pd.DatetimeIndex(index)
    dow = idx.dayofweek.to_numpy(dtype=float)
    month = idx.month.to_numpy(dtype=float)
    out = {
        "f8_dow_sin": np.sin(2.0 * np.pi * dow / 7.0),
        "f8_dow_cos": np.cos(2.0 * np.pi * dow / 7.0),
        "f8_month_sin": np.sin(2.0 * np.pi * (month - 1.0) / 12.0),
        "f8_month_cos": np.cos(2.0 * np.pi * (month - 1.0) / 12.0),
    }
    return pd.DataFrame(out, index=idx)


# --------------------------------------------------------------------------- #
# Assembly — concat every family aligned on the date index.                   #
# --------------------------------------------------------------------------- #
def assemble_engineered(
    ohlcv_inst: pd.DataFrame, signal_inst: pd.Series
) -> pd.DataFrame:
    """Assemble every engineered family into one date-indexed feature frame.

    Each family is computed on its natural index (price families on the
    instrument's full OHLCV calendar; F5 on the released signal calendar; F8 on
    the union date index) and concatenated column-wise on the date axis. The
    F1 ``f1_mr_score_20`` column is threaded into :func:`f5_signal_derived` so
    the sign-agreement column compares the signal against the C1 counter-trend
    score. No alignment fabricates rows; structural NaNs are preserved.

    Parameters
    ----------
    ohlcv_inst : pd.DataFrame
        Full-history long OHLCV for one instrument.
    signal_inst : pd.Series
        Date-indexed signal ``s_t`` over the released window.

    Returns
    -------
    pd.DataFrame
        One date-indexed float frame with every engineered feature column,
        family-prefixed, sorted by date.
    """
    f1 = f1_counter_trend(ohlcv_inst)
    f2 = f2_vol_dispersion(ohlcv_inst)
    f6 = f6_momentum_contrast(ohlcv_inst)
    f7 = f7_microstructure(ohlcv_inst)
    f10 = f10_price_action(ohlcv_inst)

    sig = pd.Series(signal_inst).sort_index()
    mr_score = f1["f1_mr_score_20"].reindex(sig.index)
    f5 = f5_signal_derived(sig, mr_score=mr_score)

    price_idx = f1.index
    union_idx = price_idx.union(f5.index)
    f8 = f8_calendar(pd.DatetimeIndex(union_idx))

    frames = [f1, f2, f6, f7, f10, f5, f8]
    out = pd.concat([f.reindex(union_idx) for f in frames], axis=1)
    return out.sort_index()
