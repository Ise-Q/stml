"""Dataset assembly for the NN portfolio model.

Reads:

* The cleaned OHLCV panel + signals (``data_loader.load_panel``).
* The per-event feature parquet (built by ``make_features``).
* Per-instrument purged-OOF calibrated meta-probabilities
  (built inside ``make_deliverables`` as the Platt-calibrated OOF).

Produces a :class:`stml.experimental.nn_portfolio.PortfolioPanel` ready for
training: per-instrument, per-day lookback windows of channels
``(features, primary side, calibrated p̂_ff)`` with causal EWMA σ̂ and next-day
returns aligned.

Lookback windows are causal: ``X_{t,k}`` uses ``[t-L+1, t]`` -- never
``> t``. ``r_{t+1,k}`` is the asset's own next-day log-return realised at
``t+1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stml.experimental.config import INSTRUMENTS
from stml.experimental.data_loader import load_panel, per_instrument_frames
from stml.experimental.nn_portfolio import PortfolioPanel
from stml.experimental.volatility import ewma_lecturer


# Default channel selection used by the NN backbones. We pick a compact set
# of features that survived the drift filter and span the main feature
# families (path / vol / momentum / regime) without bloating the input dim.
# Plus ALWAYS the primary side and calibrated p̂.
DEFAULT_FEATURE_COLS: tuple[str, ...] = (
    "f12_variance_ratio_5_21",
    "f12_autocorr_21",
    "f5_long_bias_20",
    "f15_path_tortuosity_20",
    "f5_trailing_run_length",
    "f6_ts_momentum_20",
    "f2_vol_20",
    "f2_vol_60",
    "f19_iv_pctile_252",
    "ewma_hmm_prob_highvol",
)


def _per_instrument_daily_features(
    ohlcv_inst: pd.DataFrame, feature_cols: tuple[str, ...] = DEFAULT_FEATURE_COLS,
) -> pd.DataFrame:
    """Return a (date × feature) frame for one instrument.

    Features are recomputed inline -- we deliberately keep this lightweight
    (10 hand-picked, all causal) so the NN sees a compact, regularised
    feature view rather than the full 80-column F1-F22 stack.
    """
    out = pd.DataFrame(index=ohlcv_inst.index)
    close = ohlcv_inst["close"].astype(float)
    log_ret = np.log(close / close.shift(1))

    # F12 path features.
    out["f12_variance_ratio_5_21"] = _variance_ratio(log_ret, 5, 21)
    out["f12_autocorr_21"] = log_ret.rolling(21).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False,
    )

    # F5 signal-based features need the signal — handled outside this function.
    # We populate placeholders that are later joined per-instrument.

    # F15 path tortuosity (simple proxy: sum |Δlog-r| / |Σ log-r|).
    rolling20 = log_ret.rolling(20)
    out["f15_path_tortuosity_20"] = rolling20.apply(
        lambda x: np.sum(np.abs(np.diff(x))) / (np.abs(np.sum(x)) + 1e-9), raw=True,
    )

    # F6 momentum (20-day return).
    out["f6_ts_momentum_20"] = log_ret.rolling(20).sum()

    # F2 vol (20/60-day std).
    out["f2_vol_20"] = log_ret.rolling(20).std()
    out["f2_vol_60"] = log_ret.rolling(60).std()

    return out


def _variance_ratio(log_ret: pd.Series, short: int, long: int) -> pd.Series:
    """Lo-MacKinlay variance ratio (short/long horizons)."""
    var_short = log_ret.rolling(short).var()
    var_long = log_ret.rolling(long).var()
    return var_long / (short / long * var_short + 1e-12)


def _signal_features_for_instrument(
    signal: pd.Series, dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build trailing run length + long bias features for one instrument."""
    s = signal.reindex(dates).fillna(0).astype(int)
    out = pd.DataFrame(index=dates)
    # Long bias = rolling mean of (signal == +1).
    is_long = (s == 1).astype(float)
    out["f5_long_bias_20"] = is_long.rolling(20).mean()
    # Trailing run length: consecutive same-sign days.
    run = np.zeros(len(s))
    for i in range(1, len(s)):
        if s.iloc[i] != 0 and s.iloc[i] == s.iloc[i - 1]:
            run[i] = run[i - 1] + 1
        elif s.iloc[i] != 0:
            run[i] = 1
        else:
            run[i] = 0
    out["f5_trailing_run_length"] = run
    return out


def build_portfolio_panel(
    *,
    p_hat_oof: pd.DataFrame,  # columns: date, instrument, calibrated_proba
    lookback: int = 60,
    span_sigma: int = 60,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    include_inst_id: bool = True,
) -> PortfolioPanel:
    """Build a :class:`PortfolioPanel` for the NN portfolio training.

    Parameters
    ----------
    p_hat_oof
        Calibrated meta-probabilities for the dates / instruments where the
        meta-model has been run (typically: training OOF predictions for
        training; sealed-test predictions for inference). Long format with
        columns ``date``, ``instrument``, ``calibrated_proba``. Forward-
        filled between event days within each instrument.
    lookback
        Window length ``L`` (slide 37). Default 60 days.
    span_sigma
        EWMA span for σ̂ (slide 39). Default 60 days.
    start, end
        Optional date filters on the panel.
    include_inst_id
        Add a one-hot instrument-id channel so a shared backbone can
        specialise (slide 38 "ticker embedding"). Adds ``K`` channels.
    """
    ohlcv, signals = load_panel()
    panel = per_instrument_frames(ohlcv, signals)

    # Determine the joint date range -- union of all instruments' dates.
    all_dates = sorted(set().union(*[df.index for df in panel.values()]))
    all_dates = pd.DatetimeIndex(all_dates)
    if start is not None:
        all_dates = all_dates[all_dates >= pd.Timestamp(start)]
    if end is not None:
        all_dates = all_dates[all_dates <= pd.Timestamp(end)]

    # Per-instrument feature matrix and σ̂.
    inst_features: dict[str, pd.DataFrame] = {}
    inst_sigma: dict[str, pd.Series] = {}
    inst_return: dict[str, pd.Series] = {}

    p_hat_oof = p_hat_oof.copy()
    p_hat_oof["date"] = pd.to_datetime(p_hat_oof["date"])
    p_hat_oof = p_hat_oof.sort_values(["instrument", "date"])

    for inst in INSTRUMENTS:
        df = panel.get(inst)
        if df is None or df.empty:
            continue
        df = df.reindex(all_dates)
        feats = _per_instrument_daily_features(df)
        sig_feats = _signal_features_for_instrument(df["signal"], all_dates)
        feats = pd.concat([feats, sig_feats], axis=1)

        # Primary side channel.
        feats["primary_side"] = df["signal"].fillna(0).astype(float)

        # Calibrated p̂ -- forward-filled from event days; 0.5 before first event.
        ph_sub = p_hat_oof.loc[p_hat_oof["instrument"] == inst].set_index("date")
        ph_series = ph_sub["calibrated_proba"].reindex(all_dates).ffill().fillna(0.5)
        feats["p_hat_cal"] = ph_series.astype(float)

        inst_features[inst] = feats

        log_ret = np.log(df["close"].astype(float) / df["close"].astype(float).shift(1))
        # Causal σ̂ from EWMA (slide 39).
        inst_sigma[inst] = ewma_lecturer(log_ret.fillna(0.0), span=span_sigma).reindex(all_dates)
        # Next-day return (used as r_{t+1}) — shift returns BACKWARD by 1.
        inst_return[inst] = log_ret.shift(-1).reindex(all_dates)

    if not inst_features:
        raise RuntimeError("no instruments produced features")

    instruments_with_data = [i for i in INSTRUMENTS if i in inst_features]
    K = len(instruments_with_data)
    T = len(all_dates)
    feature_names = list(inst_features[instruments_with_data[0]].columns)
    n_feat = len(feature_names)

    # Optional one-hot inst id (slide 38).
    if include_inst_id:
        feature_names = feature_names + [f"is_{i}" for i in instruments_with_data]
    C = len(feature_names)

    # Assemble (T, K, n_feat) panel.
    F = np.full((T, K, n_feat), np.nan, dtype=np.float32)
    for ki, inst in enumerate(instruments_with_data):
        F[:, ki, :] = inst_features[inst].to_numpy(dtype=np.float32)
    sigma_d = np.full((T, K), np.nan, dtype=np.float32)
    r_next = np.full((T, K), np.nan, dtype=np.float32)
    for ki, inst in enumerate(instruments_with_data):
        sigma_d[:, ki] = inst_sigma[inst].to_numpy(dtype=np.float32)
        r_next[:, ki] = inst_return[inst].to_numpy(dtype=np.float32)

    # Build lookback windows. Drop the first L-1 days (warm-up) by setting
    # mask to False there.
    L = int(lookback)
    if include_inst_id:
        X_arr = np.full((T, K, L, C), 0.0, dtype=np.float32)
        for ki in range(K):
            for t in range(T):
                lo = max(0, t - L + 1)
                seg = F[lo:t + 1, ki, :]
                # Pad with zeros at the front if t < L-1.
                pad = L - seg.shape[0]
                if pad > 0:
                    seg = np.vstack([np.zeros((pad, n_feat), dtype=np.float32), seg])
                X_arr[t, ki, :, :n_feat] = seg
                # One-hot inst id stamped on every step.
                X_arr[t, ki, :, n_feat + ki] = 1.0
    else:
        X_arr = np.full((T, K, L, C), 0.0, dtype=np.float32)
        for ki in range(K):
            for t in range(T):
                lo = max(0, t - L + 1)
                seg = F[lo:t + 1, ki, :]
                pad = L - seg.shape[0]
                if pad > 0:
                    seg = np.vstack([np.zeros((pad, n_feat), dtype=np.float32), seg])
                X_arr[t, ki, :, :] = seg

    # Replace NaN in features with 0 (caller-supplied features include only
    # rolling stats; NaN-at-start is the natural warm-up).
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Mask: valid only when σ̂ available, r_{t+1} available, and t >= L-1.
    mask = np.zeros((T, K), dtype=bool)
    for ki in range(K):
        valid = (np.isfinite(sigma_d[:, ki]) & (sigma_d[:, ki] > 0)
                  & np.isfinite(r_next[:, ki]))
        if L > 1:
            valid[:L - 1] = False
        mask[:, ki] = valid

    return PortfolioPanel(
        dates=all_dates, instruments=instruments_with_data, X=X_arr,
        sigma_daily=sigma_d, r_next=r_next, mask=mask,
        feature_names=feature_names,
    )
