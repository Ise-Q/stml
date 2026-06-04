"""Per-event conviction and vol-targeted weights for each strategy method.

Seven methods:
    A         — Benchmark: ŷ = side (primary-only, no meta-filter).
    B-mc      — model_confidence: ŷ = side · p̂  when p̂ > 0.5, else 0.
    B-aon     — all_or_nothing:  ŷ = side · 1(p̂ > 0.5).
    B-ncdf    — NCDF:            ŷ = side · Φ((p̂ − 0.5) / √(p̂(1−p̂))).
    B-sops    — SOPS:            ŷ = side · f_{a*,c*}(p̂); sigmoid fitted on
                                  training OOF (p̂, r) pairs via Sharpe optimisation.
    C-vsn-lstm — VSN+LSTM neural model; loads saved weights from outputs/.
    D-tft      — Temporal Fusion Transformer; loads saved weights from outputs/.

All methods share the same vol-targeting layer:
    w_{t,k} = ŷ_{t,k} · σ_tgt / σ̂_{t,k}    clipped to [−max_leverage, +max_leverage]

The output of each `make_weights_*` function is a pair:
    (events_df, weights_series)
ready to pass directly to `stml.experimental.backtest.barrier_backtest`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from stml.new_work import config
from stml.new_work.targeting import build_vol_panel
from stml.experimental.sizing import (
    SOPSFit,
    fit_sops,
    model_confidence,
    all_or_nothing,
    ncdf,
)

MethodName = Literal["A", "B-mc", "B-aon", "B-ncdf", "B-sops", "C-vsn-lstm", "D-tft"]

_OUTPUTS_DIR = Path(__file__).parent / "outputs"


# ---------------------------------------------------------------------------
# Neural model loaders (architecture inferred from saved state-dict shapes)
# ---------------------------------------------------------------------------


def _load_vsn_lstm_model():
    """Load VSN+LSTM checkpoint (d_model=16, embed_dim=4, EWMA vol features)."""
    import torch
    from stml.new_work.models.vsn_lstm import VSNLSTMModel, N_FEATURES, N_INSTRUMENTS
    # Prefer cp5 (EWMA-feature trained); fall back to original for compatibility.
    for name in ("vsn_lstm_weights_cp5.pt", "vsn_lstm_weights.pt"):
        path = _OUTPUTS_DIR / name
        if path.exists():
            break
    else:
        raise FileNotFoundError(
            f"VSN+LSTM weights not found in {_OUTPUTS_DIR}. "
            "Run train_cp5.py first."
        )
    model = VSNLSTMModel(
        n_features=N_FEATURES, n_instruments=N_INSTRUMENTS,
        d_model=16, embed_dim=4, n_lstm_layers=1, dropout=0.1,
    )
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    return model


def _load_tft_model():
    """Load TFT checkpoint (d_model=16, n_heads=4, embed_dim=4, EWMA vol features)."""
    import torch
    from stml.new_work.models.tft import TFTModel, N_FEATURES, N_INSTRUMENTS
    for name in ("tft_weights_cp5.pt", "tft_weights.pt"):
        path = _OUTPUTS_DIR / name
        if path.exists():
            break
    else:
        raise FileNotFoundError(
            f"TFT weights not found in {_OUTPUTS_DIR}. "
            "Run train_cp5.py first."
        )
    model = TFTModel(
        n_features=N_FEATURES, n_instruments=N_INSTRUMENTS,
        d_model=16, n_heads=4, embed_dim=4, n_lstm_layers=1, dropout=0.1,
    )
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# OOS event loader
# ---------------------------------------------------------------------------


def load_oos_events(path: Path | None = None) -> pd.DataFrame:
    """Load OOS events from metamodel_predictions.csv.

    Adds t_start / t_end columns required by barrier_backtest.
    Returns sorted by (instrument, date).
    """
    p = Path(path or config.OOS_PROBA_PATH)
    if not p.exists():
        raise FileNotFoundError(f"OOS predictions not found at {p}")
    df = pd.read_csv(p, parse_dates=["date", "t1"])
    df = df.rename(columns={"t1": "t_end"})
    df["t_start"] = df["date"]
    df = df.sort_values(["instrument", "date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Conviction functions — map (events, [oof_df]) → signed conviction ŷ ∈ [-1, 1]
# ---------------------------------------------------------------------------


def conviction_benchmark(events: pd.DataFrame) -> pd.Series:
    """Method A: ŷ = side (primary only, no meta).

    Always follows the primary signal at full conviction.
    """
    return events["side"].astype(float).rename("y_hat")


def conviction_model_confidence(events: pd.DataFrame) -> pd.Series:
    """Method B-mc: ŷ = side · p̂  for p̂ > 0.5, else 0."""
    p = events["calibrated_proba"].values
    g = model_confidence(p)  # g = p̂ · 1(p̂ > 0.5)
    return pd.Series(events["side"].values * g, index=events.index, name="y_hat")


def conviction_all_or_nothing(events: pd.DataFrame) -> pd.Series:
    """Method B-aon: ŷ = side · 1(p̂ > 0.5)."""
    p = events["calibrated_proba"].values
    g = all_or_nothing(p)    # 0 or 1
    return pd.Series(events["side"].values * g, index=events.index, name="y_hat")


def conviction_ncdf(events: pd.DataFrame) -> pd.Series:
    """Method B-ncdf: ŷ = side · Φ((p̂ − 0.5) / √(p̂(1−p̂)))."""
    p = events["calibrated_proba"].values
    g = ncdf(p)
    return pd.Series(events["side"].values * g, index=events.index, name="y_hat")


def conviction_sops(events: pd.DataFrame, sops_fit: SOPSFit) -> pd.Series:
    """Method B-sops: ŷ = side · f_{a*,c*}(p̂)."""
    p = events["calibrated_proba"].values
    g = sops_fit.transform(p)    # logistic sigmoid, gated at 0.5
    return pd.Series(events["side"].values * g, index=events.index, name="y_hat")


# ---------------------------------------------------------------------------
# SOPS fitting from OOF training data
# ---------------------------------------------------------------------------


def fit_sops_from_oof(
    oof_df: pd.DataFrame,
    *,
    inst: str | None = None,
    min_events: int = 20,
) -> SOPSFit:
    """Fit the SOPS sigmoid on OOF training (p̂, r) pairs.

    Parameters
    ----------
    oof_df
        DataFrame from data/oof_meta_probabilities.csv with columns:
        instrument, p_hat_oof, ret.
    inst
        If provided, use only that instrument's OOF data. If None, pool all
        instruments (recommended — larger sample, more stable fit).
    min_events
        Minimum events required. Falls back to all-or-nothing sigmoid if fewer.
    """
    df = oof_df if inst is None else oof_df[oof_df["instrument"] == inst]
    df = df.dropna(subset=["p_hat_oof", "ret"])
    if len(df) < min_events:
        print(f"  SOPS: too few events ({len(df)} < {min_events}), using default sigmoid")
        return SOPSFit(a=20.0, c=10.0, train_sharpe=0.0)

    p = df["p_hat_oof"].values
    r = df["ret"].values
    result = fit_sops(p, r)
    return result


# ---------------------------------------------------------------------------
# Vol lookup helper
# ---------------------------------------------------------------------------


def _vol_at_events(
    events: pd.DataFrame,
    vol_panel: pd.DataFrame,
) -> pd.Series:
    """Look up ex-ante annualised σ̂ at each event's entry date.

    Uses .asof() per instrument so the last available vol on or before
    t_start is returned (handles calendar misalignment).
    """
    result = np.full(len(events), np.nan)
    for inst in events["instrument"].unique():
        if inst not in vol_panel.columns:
            continue
        mask = (events["instrument"] == inst).values
        inst_vol = vol_panel[inst].dropna().sort_index()
        entry_dates = pd.DatetimeIndex(events.loc[mask, "t_start"])
        # asof for each date: last value on or before the date.
        looked_up = np.array([
            float(inst_vol.asof(dt)) if dt >= inst_vol.index[0] else np.nan
            for dt in entry_dates
        ])
        result[mask] = looked_up
    return pd.Series(result, index=events.index, name="ann_sigma")


# ---------------------------------------------------------------------------
# Weight assembly — conviction × vol-targeting → per-event weight
# ---------------------------------------------------------------------------


def apply_vol_targeting_to_conviction(
    y_hat: pd.Series,
    ann_sigma: pd.Series,
    *,
    sigma_tgt: float = config.SIGMA_TGT,
    max_leverage: float = config.MAX_LEVERAGE,
) -> pd.Series:
    """Apply vol-targeting formula: w = ŷ · σ_tgt / σ̂, clipped."""
    raw = y_hat * sigma_tgt / ann_sigma
    # Zero where conviction is zero (preserve NaN from missing vol → becomes 0).
    raw = raw.where(y_hat != 0.0, other=0.0)
    # Clip; NaN where vol is NaN.
    return raw.clip(lower=-max_leverage, upper=max_leverage).fillna(0.0).rename("weight")


# ---------------------------------------------------------------------------
# Top-level: make weights for a given method
# ---------------------------------------------------------------------------


def make_weights(
    method: MethodName,
    *,
    oos_events: pd.DataFrame | None = None,
    vol_panel: pd.DataFrame | None = None,
    oof_df: pd.DataFrame | None = None,
    ohlcv: pd.DataFrame | None = None,
    signals: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build (events_df, weights_series) ready for barrier_backtest.

    Parameters
    ----------
    method
        One of: A, B-mc, B-aon, B-ncdf, B-sops, C-vsn-lstm, D-tft.
    oos_events
        OOS event DataFrame from load_oos_events(). Loaded from disk if None.
    vol_panel
        Wide date × instrument annualised vol panel. Computed if None.
    oof_df
        OOF training probability DataFrame. Required for B-sops. Loaded from
        config.OOF_PROBA_PATH if None; raises FileNotFoundError if missing.
    ohlcv
        Long-format OHLCV. Required for neural methods and vol_panel
        construction if vol_panel is None. Loaded lazily if needed.
    signals
        Primary signal DataFrame. Required for neural methods. Loaded
        lazily if needed.

    Returns
    -------
    events : pd.DataFrame
        OOS events with t_start, t_end, instrument columns.
    weights : pd.Series
        Per-event weight, same index as events.
    """
    # Load dependencies lazily.
    if oos_events is None:
        oos_events = load_oos_events()

    if vol_panel is None:
        if ohlcv is None:
            from stml.io import load_data
            ohlcv, _ = load_data()
        vol_panel = build_vol_panel(
            ohlcv,
            method=config.VOL_METHOD,
            window=config.LOOKBACK_L,
            span=config.EWMA_SPAN,
        )

    # Neural methods delegate fully to their own weight builders.
    if method == "C-vsn-lstm":
        if ohlcv is None:
            from stml.io import load_data
            ohlcv, _ = load_data()
        if signals is None:
            from stml.new_work.data import load_signals
            signals = load_signals()
        from stml.new_work.models.vsn_lstm import make_weights_vsn_lstm
        model = _load_vsn_lstm_model()
        return make_weights_vsn_lstm(model, oos_events, vol_panel, ohlcv, signals)

    if method == "D-tft":
        if ohlcv is None:
            from stml.io import load_data
            ohlcv, _ = load_data()
        if signals is None:
            from stml.new_work.data import load_signals
            signals = load_signals()
        from stml.new_work.models.tft import make_weights_tft
        model = _load_tft_model()
        return make_weights_tft(model, oos_events, vol_panel, ohlcv, signals)

    # Conviction — rule-based methods.
    if method == "A":
        y_hat = conviction_benchmark(oos_events)
    elif method == "B-mc":
        y_hat = conviction_model_confidence(oos_events)
    elif method == "B-aon":
        y_hat = conviction_all_or_nothing(oos_events)
    elif method == "B-ncdf":
        y_hat = conviction_ncdf(oos_events)
    elif method == "B-sops":
        if oof_df is None:
            from stml.new_work.data import load_oof_probabilities
            oof_df = load_oof_probabilities()
        sops_fit = fit_sops_from_oof(oof_df)
        print(f"  SOPS fit: a={sops_fit.a:.3f}, c={sops_fit.c:.3f}, "
              f"train_sharpe={sops_fit.train_sharpe:.4f}")
        y_hat = conviction_sops(oos_events, sops_fit)
    else:
        raise ValueError(f"Unknown method {method!r}")

    # Vol lookup at each event's entry date.
    ann_sigma = _vol_at_events(oos_events, vol_panel)

    n_missing = ann_sigma.isna().sum()
    if n_missing:
        print(f"  [{method}] {n_missing} events missing vol → weight=0")

    # Weight assembly.
    weights = apply_vol_targeting_to_conviction(y_hat, ann_sigma)

    n_active = (weights.abs() > 0).sum()
    print(f"  [{method}] events={len(oos_events)} | active (|w|>0)={n_active} | "
          f"mean |w|={weights[weights != 0].abs().mean():.4f}")

    return oos_events.copy(), weights
