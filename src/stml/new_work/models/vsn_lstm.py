"""VSN + LSTM strategy model — Checkpoint 3.

Architecture:
    Input  : (batch, lookback, n_features)  [log_ret, log_hl, log_co, side, p̂]
    VSN    : n_features × per-feature GRNs + softmax selection → (batch, lookback, d_model)
    LSTM   : (batch, lookback, d_model) → last hidden → (batch, d_model)
    Ticker : TickerEmbedding(inst_id) → (batch, embed_dim)
    Head   : Linear(d_model + embed_dim, 1) → Tanh → ŷ ∈ (−1, +1)

Training loss: PooledSharpeLoss on the mini-batch {ŷ_i · ret_i}.
Causality: all features at time t use only data up to t; OOF p̂ is used
in training, OOS calibrated p̂ at test time.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent.parent.parent.parent  # src/stml/new_work/models → repo root
sys.path.insert(0, str(_REPO / "src"))

from stml.new_work import config
from stml.new_work.data import build_feature_window, load_ohlcv_for_instrument, load_signals
from stml.new_work.targeting import build_vol_panel, apply_vol_targeting
from stml.new_work.vol import ewma_close as _ewma_close
from stml.new_work.models.common import (
    GRN,
    TanhHead,
    TickerEmbedding,
    TrainConfig,
    TrainHistory,
    fit,
    get_device,
)

INST_TO_IDX: dict[str, int] = {inst: i for i, inst in enumerate(config.INSTRUMENTS)}
N_INSTRUMENTS = len(config.INSTRUMENTS)

# Feature columns used by build_feature_window (d=3 + side + p_hat = 5 total).
FEATURE_COLS = ["log_ret", "log_hl", "log_co"]
N_FEATURES = len(FEATURE_COLS) + 2   # + side + p_hat channels


# ---------------------------------------------------------------------------
# Variable Selection Network
# ---------------------------------------------------------------------------


class VariableSelectionNetwork(nn.Module):
    """Soft feature selection via per-feature GRNs and a softmax gating network.

    Reference: Lim et al. (2021) TFT, §3.2 Variable selection networks.

    Input  : (batch, time, n_features)  — one scalar per feature
    Output : (batch, time, d_model), (batch, time, n_features) selection weights
    """

    def __init__(
        self,
        n_features: int,
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # One GRN per feature: scalar (1) → d_model.
        self.feature_grns = nn.ModuleList([
            GRN(1, d_model, d_model, dropout=dropout)
            for _ in range(n_features)
        ])
        # Selection GRN: concatenated processed features → n_features weights.
        self.select_grn = GRN(n_features * d_model, d_model, n_features, dropout=dropout)
        self.n_features = n_features
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F)
        B, T, F = x.shape

        # Per-feature transformation: (B, T, 1) → (B, T, d_model) each.
        processed = torch.stack(
            [self.feature_grns[i](x[..., i : i + 1]) for i in range(F)],
            dim=2,
        )  # (B, T, F, d_model)

        # Flatten for selection weights: (B, T, F * d_model).
        flat = processed.reshape(B, T, F * self.d_model)
        weights = torch.softmax(self.select_grn(flat), dim=-1)  # (B, T, F)

        # Weighted sum over features: (B, T, d_model).
        # weights: (B, T, F, 1) × processed: (B, T, F, d_model) → sum over F
        out = (weights.unsqueeze(-1) * processed).sum(dim=2)
        return out, weights


# ---------------------------------------------------------------------------
# Full VSN + LSTM model
# ---------------------------------------------------------------------------


class VSNLSTMModel(nn.Module):
    """Variable Selection Network followed by LSTM encoder with tanh output head.

    Parameters
    ----------
    n_features
        Number of input features per time step (default 5: log_ret, log_hl,
        log_co, side, p̂).
    n_instruments
        Size of the ticker embedding table.
    d_model
        Hidden dimension shared by VSN and LSTM.
    embed_dim
        Dimension of the instrument embedding.
    n_lstm_layers
        Number of stacked LSTM layers.
    dropout
        Dropout probability applied in GRNs.
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        n_instruments: int = N_INSTRUMENTS,
        d_model: int = 32,
        embed_dim: int = 4,
        n_lstm_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ticker_embed = TickerEmbedding(n_instruments, embed_dim)
        self.vsn = VariableSelectionNetwork(n_features, d_model, dropout)
        self.lstm = nn.LSTM(
            d_model, d_model, n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.head = TanhHead(d_model + embed_dim)

    def forward(self, x: torch.Tensor, ticker_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            (batch, lookback, n_features) float32.
        ticker_ids
            (batch,) int64 instrument indices.

        Returns
        -------
        ŷ : (batch,) float32 ∈ (−1, +1).
        """
        emb = self.ticker_embed(ticker_ids)       # (B, embed_dim)
        vsn_out, _ = self.vsn(x)                  # (B, T, d_model)
        lstm_out, _ = self.lstm(vsn_out)           # (B, T, d_model)
        last = lstm_out[:, -1, :]                  # (B, d_model)  — final hidden state
        combined = torch.cat([last, emb], dim=-1)  # (B, d_model + embed_dim)
        return self.head(combined)                 # (B,)


# ---------------------------------------------------------------------------
# Training data builder
# ---------------------------------------------------------------------------


def _build_training_arrays(
    oof_df: pd.DataFrame,
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    lookback: int = config.LOOKBACK_L,
    boundary: pd.Timestamp = config.BOUNDARY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build pooled (X, ret, ticker_ids) arrays for all instruments.

    Returns
    -------
    X          : (N, lookback, N_FEATURES)  float32
    ret        : (N,)                        float32
    ticker_ids : (N,)                        int64
    dates      : list[str]                   for diagnostics
    """
    all_X:   list[np.ndarray] = []
    all_ret: list[np.ndarray] = []
    all_tid: list[np.ndarray] = []
    all_dates: list[str]      = []

    for inst in config.INSTRUMENTS:
        if inst not in INST_TO_IDX:
            continue

        # OOF data for this instrument — must be within training period.
        inst_oof = oof_df[
            (oof_df["instrument"] == inst) & (oof_df["date"] < boundary)
        ].copy()
        if len(inst_oof) < lookback + 1:
            continue

        # OHLCV for this instrument.
        inst_ohlc = ohlcv[ohlcv["instrument"] == inst].copy()
        if inst_ohlc.empty:
            continue
        inst_ohlc = inst_ohlc.set_index("date").sort_index()
        inst_ohlc.index = pd.to_datetime(inst_ohlc.index)

        # EWMA vol series for this instrument (causal, annualised).
        inst_vol = _ewma_close(inst_ohlc["close"], span=config.EWMA_SPAN)

        # Build feature windows using OOF probabilities; vol-normalise features.
        X, _, dates = build_feature_window(
            inst_ohlc,
            signals,
            inst_oof,
            inst,
            lookback=lookback,
            feature_cols=FEATURE_COLS,
            vol_series=inst_vol,
        )
        if len(X) == 0:
            continue

        # Join ret from OOF df on (instrument, date).
        oof_idx = inst_oof.set_index("date")["ret"]
        rets = np.array([
            float(oof_idx.loc[dt]) if dt in oof_idx.index else np.nan
            for dt in dates
        ], dtype=np.float32)

        valid = np.isfinite(rets) & np.all(np.isfinite(X.reshape(len(X), -1)), axis=1)
        if valid.sum() == 0:
            continue

        X    = X[valid].astype(np.float32)
        rets = rets[valid]
        tid  = np.full(len(X), INST_TO_IDX[inst], dtype=np.int64)

        all_X.append(X)
        all_ret.append(rets)
        all_tid.append(tid)
        all_dates.extend([str(d) for d in np.array(dates)[valid]])

        print(f"    {inst:8s}: {len(X):4d} windows  "
              f"ret_mean={rets.mean():.5f}  "
              f"p̂_mean={inst_oof['p_hat_oof'].mean():.4f}")

    if not all_X:
        raise RuntimeError("No training windows could be built — check OOF artifact")

    X_all   = np.concatenate(all_X,   axis=0)
    ret_all = np.concatenate(all_ret, axis=0)
    tid_all = np.concatenate(all_tid, axis=0)

    # Sort chronologically across all instruments for a clean val split.
    order   = np.argsort(all_dates)
    return X_all[order], ret_all[order], tid_all[order], [all_dates[i] for i in order]


# ---------------------------------------------------------------------------
# Test-time feature window builder (uses OOS calibrated_proba)
# ---------------------------------------------------------------------------


def _build_test_arrays(
    oos_events: pd.DataFrame,
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    lookback: int = config.LOOKBACK_L,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, ticker_ids, row_order) for OOS events.

    The OOS events DataFrame must have columns:
        instrument, date (= t_start), calibrated_proba, side

    Returns arrays aligned to oos_events.index; rows where a feature window
    cannot be built are filled with zeros (weight will be zero via the
    vol-targeting layer since ŷ ≈ 0 for near-zero-feature inputs).
    """
    N = len(oos_events)
    X_out   = np.zeros((N, lookback, N_FEATURES), dtype=np.float32)
    tid_out = np.zeros(N, dtype=np.int64)
    built   = np.zeros(N, dtype=bool)

    for inst in oos_events["instrument"].unique():
        inst_mask = (oos_events["instrument"] == inst).values
        inst_rows = oos_events[inst_mask].reset_index(drop=True)

        # Construct a synthetic oof_probas DataFrame from OOS calibrated_proba.
        inst_oof = pd.DataFrame({
            "date":        pd.to_datetime(inst_rows["t_start"]),
            "instrument":  inst,
            "p_hat_oof":   inst_rows["calibrated_proba"].values,
        })

        inst_ohlc = ohlcv[ohlcv["instrument"] == inst].copy()
        if inst_ohlc.empty:
            continue
        inst_ohlc = inst_ohlc.set_index("date").sort_index()
        inst_ohlc.index = pd.to_datetime(inst_ohlc.index)

        inst_vol = _ewma_close(inst_ohlc["close"], span=config.EWMA_SPAN)

        X, _, dates = build_feature_window(
            inst_ohlc, signals, inst_oof, inst,
            lookback=lookback, feature_cols=FEATURE_COLS,
            vol_series=inst_vol,
        )
        if len(X) == 0:
            continue

        # Map back to global row indices.
        date_to_global: dict[pd.Timestamp, int] = {
            pd.Timestamp(row["t_start"]): i
            for i, (_, row) in enumerate(oos_events.iterrows())
            if row["instrument"] == inst
        }
        for local_i, dt in enumerate(dates):
            gi = date_to_global.get(dt)
            if gi is not None:
                X_out[gi]   = X[local_i].astype(np.float32)
                tid_out[gi] = INST_TO_IDX.get(inst, 0)
                built[gi]   = True

    n_built = built.sum()
    n_missed = (~built).sum()
    if n_missed:
        print(f"  test windows: {n_built} built, {n_missed} missing (will be ŷ≈0)")

    return X_out, tid_out, built


# ---------------------------------------------------------------------------
# End-to-end training entry point
# ---------------------------------------------------------------------------


def train_vsn_lstm(
    oof_df: pd.DataFrame,
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    cfg: Optional[TrainConfig] = None,
    d_model: int = 32,
    embed_dim: int = 4,
    n_lstm_layers: int = 1,
    dropout: float = 0.1,
    lookback: int = config.LOOKBACK_L,
    verbose: bool = True,
) -> tuple[VSNLSTMModel, TrainHistory]:
    """Train the VSN+LSTM model on all available training events.

    Returns
    -------
    model
        Trained VSNLSTMModel (on CPU, best checkpoint restored).
    history
        Training history (train/val Sharpe per epoch).
    """
    if cfg is None:
        cfg = TrainConfig()

    if verbose:
        print(f"\nBuilding training windows (lookback={lookback}) …")

    X, ret, ticker_ids, _ = _build_training_arrays(
        oof_df, ohlcv, signals,
        lookback=lookback,
        boundary=config.BOUNDARY,
    )

    if verbose:
        print(f"  Pooled training set: {len(X)} windows across "
              f"{len(np.unique(ticker_ids))} instruments")
        pos_ret = (ret > 0).mean()
        print(f"  Return stats: mean={ret.mean():.5f}  std={ret.std():.5f}  "
              f"pos_frac={pos_ret:.3f}")

    model = VSNLSTMModel(
        n_features=N_FEATURES,
        n_instruments=N_INSTRUMENTS,
        d_model=d_model,
        embed_dim=embed_dim,
        n_lstm_layers=n_lstm_layers,
        dropout=dropout,
    )

    if verbose:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model parameters: {n_params:,}")

    model, hist = fit(model, X, ret, ticker_ids, cfg, verbose=verbose)
    return model, hist


# ---------------------------------------------------------------------------
# Weight generation for the backtest
# ---------------------------------------------------------------------------


def make_weights_vsn_lstm(
    model: VSNLSTMModel,
    oos_events: pd.DataFrame,
    vol_panel: pd.DataFrame,
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    lookback: int = config.LOOKBACK_L,
    sigma_tgt: float = config.SIGMA_TGT,
    max_leverage: float = config.MAX_LEVERAGE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Produce (events_df, weights_series) for barrier_backtest.

    Runs forward pass on OOS windows, then applies vol-targeting.
    """
    print("  Building OOS feature windows for VSN+LSTM …")
    X_test, tid_test, built = _build_test_arrays(
        oos_events, ohlcv, signals, lookback=lookback
    )

    device = torch.device("cpu")    # inference on CPU
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_test, dtype=torch.float32).to(device)
        tt = torch.tensor(tid_test, dtype=torch.long).to(device)
        y_hat_np = model(Xt, tt).numpy()  # (N,)

    # Zero out windows that couldn't be built.
    y_hat_np[~built] = 0.0

    y_hat = pd.Series(y_hat_np, index=oos_events.index, name="y_hat")

    from stml.new_work.weights import _vol_at_events, apply_vol_targeting_to_conviction
    ann_sigma = _vol_at_events(oos_events, vol_panel)
    weights = apply_vol_targeting_to_conviction(
        y_hat, ann_sigma, sigma_tgt=sigma_tgt, max_leverage=max_leverage
    )

    n_active = (weights.abs() > 0).sum()
    mean_w   = weights[weights != 0].abs().mean() if n_active else 0.0
    print(f"  [VSN+LSTM] active={n_active}/{len(oos_events)} | "
          f"mean |w|={mean_w:.4f} | "
          f"ŷ range=[{y_hat_np.min():.3f}, {y_hat_np.max():.3f}]")

    return oos_events.copy(), weights
