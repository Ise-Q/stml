"""Temporal Fusion Transformer strategy model — Checkpoint 4.

A streamlined TFT (Lim et al. 2021) adapted for the many-to-one sizing task:
one lookback window → single ŷ ∈ (−1, +1).

Architecture:
    Input   : (batch, lookback, n_features)  [log_ret, log_hl, log_co, side, p̂]
    Static  : TickerEmbedding(inst_id) → context vector (batch, embed_dim)
    VSN     : feature selection with static context → (batch, lookback, d_model)
    LSTM    : encoder over selected features → (batch, lookback, d_model)
    GLU skip: gate LSTM output, add VSN skip → (batch, lookback, d_model)
    IMHA    : interpretable multi-head self-attention → (batch, lookback, d_model)
    GLU skip: gate attention output → (batch, lookback, d_model)
    GRN pw  : position-wise feedforward → (batch, lookback, d_model)
    GLU skip: gate feedforward output → (batch, lookback, d_model)
    Aggregate: last time step → (batch, d_model)
    Head    : Linear(d_model + embed_dim) → Tanh → ŷ

Key TFT features retained vs. VSN+LSTM:
    • Interpretable multi-head attention (shared V matrix, heads inspectable)
    • Three levels of gated skip connections (GRN-based GLU gates)
    • Static covariate (ticker embedding) injected as context into the VSN
      selection gate AND the first post-LSTM GLU skip

Reference: Lim et al. (2021). Temporal Fusion Transformers for interpretable
  multi-horizon time series forecasting. International Journal of Forecasting.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent.parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

from stml.new_work import config
from stml.new_work.models.common import GRN, TanhHead, TickerEmbedding, TrainConfig, TrainHistory, fit
from stml.new_work.models.vsn_lstm import (
    INST_TO_IDX,
    N_FEATURES,
    N_INSTRUMENTS,
    FEATURE_COLS,
    _build_training_arrays,
    _build_test_arrays,
)


# ---------------------------------------------------------------------------
# Gated Linear Unit skip connection (TFT §3, eq. 6-8)
# ---------------------------------------------------------------------------


class GatedSkip(nn.Module):
    """GLU-gated residual: output = LayerNorm(x + sigmoid(W1·h) ⊙ W2·h).

    Used after each major sub-layer (LSTM, attention, GRN) to suppress
    sub-layer contributions when they are not needed.
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.gate  = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.norm  = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(h)) * self.value(h)
        return self.norm(residual + self.drop(g))


# ---------------------------------------------------------------------------
# VSN with static-context injection (extends vsn_lstm.VariableSelectionNetwork)
# ---------------------------------------------------------------------------


class ContextVSN(nn.Module):
    """Variable Selection Network that accepts an optional static context vector.

    The context (ticker embedding) modulates the feature selection gate so
    the model can learn instrument-specific feature importance.

    Input : (batch, time, n_features), context (batch, context_dim)
    Output: (batch, time, d_model), (batch, time, n_features) selection weights
    """

    def __init__(
        self,
        n_features: int,
        d_model: int,
        context_dim: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_grns = nn.ModuleList([
            GRN(1, d_model, d_model, dropout=dropout)
            for _ in range(n_features)
        ])
        # Selection GRN takes context from ticker embedding via context_dim.
        self.select_grn = GRN(
            n_features * d_model, d_model, n_features,
            context_dim=context_dim, dropout=dropout,
        )
        self.n_features = n_features
        self.d_model    = d_model

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, F = x.shape
        processed = torch.stack(
            [self.feature_grns[i](x[..., i : i + 1]) for i in range(F)],
            dim=2,
        )  # (B, T, F, d_model)
        flat = processed.reshape(B, T, F * self.d_model)

        # Broadcast context (B, context_dim) → (B, T, context_dim) for the GRN.
        ctx = context.unsqueeze(1).expand(-1, T, -1) if context is not None else None
        weights = torch.softmax(self.select_grn(flat, ctx), dim=-1)  # (B, T, F)
        out = (weights.unsqueeze(-1) * processed).sum(dim=2)          # (B, T, d_model)
        return out, weights


# ---------------------------------------------------------------------------
# Interpretable Multi-Head Self-Attention (TFT §3, eq. 11-14)
# ---------------------------------------------------------------------------


class InterpretableMultiHeadAttention(nn.Module):
    """Shared-value multi-head attention making per-head weights comparable.

    Shares a single value projection W_V across all heads so that attention
    weights can be averaged and interpreted as a single importance map.

        A_h  = softmax(Q_h K_h^T / sqrt(d_k))
        β    = (1/H) Σ_h A_h · (X W_V)        [shared V]
        out  = W_O · β

    Returns (output, mean_attention_weights) where mean_attn: (B, T, T)
    can be examined to understand which past time steps are most attended.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_k     = d_model // n_heads
        self.n_heads = n_heads
        self.W_Q     = nn.Linear(d_model, d_model, bias=False)
        self.W_K     = nn.Linear(d_model, d_model, bias=False)
        self.W_V     = nn.Linear(d_model, d_model, bias=False)   # shared across heads
        self.W_O     = nn.Linear(d_model, d_model, bias=False)
        self.drop    = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        H, dk = self.n_heads, self.d_k

        # Per-head Q and K: (B, H, T, dk).
        Q = self.W_Q(x).view(B, T, H, dk).transpose(1, 2)
        K = self.W_K(x).view(B, T, H, dk).transpose(1, 2)
        # Shared V: (B, T, D) → broadcasted across heads as (B, 1, T, D).
        V = self.W_V(x).unsqueeze(1)

        # Attention scores: (B, H, T, T).
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)
        A = torch.softmax(scores, dim=-1)
        A = self.drop(A)

        # Weighted values: (B, H, T, D) → mean over heads → (B, T, D).
        out = torch.matmul(A, V).mean(dim=1)
        out = self.W_O(out)

        return out, A.mean(dim=1)   # (B, T, D), (B, T, T)


# ---------------------------------------------------------------------------
# Full TFT model
# ---------------------------------------------------------------------------


class TFTModel(nn.Module):
    """Temporal Fusion Transformer for single-step conviction sizing.

    Parameters
    ----------
    n_features
        Input features per time step (default 5).
    n_instruments
        Ticker embedding table size.
    d_model
        Core hidden dimension shared across VSN, LSTM, attention.
    n_heads
        Attention heads (must divide d_model).
    embed_dim
        Ticker embedding dimension (static context).
    n_lstm_layers
        Stacked LSTM layers in the encoder.
    dropout
        Dropout applied in GRNs and attention.
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        n_instruments: int = N_INSTRUMENTS,
        d_model: int = 32,
        n_heads: int = 4,
        embed_dim: int = 8,
        n_lstm_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ticker_embed = TickerEmbedding(n_instruments, embed_dim)

        # VSN with static context from ticker embedding.
        self.vsn = ContextVSN(n_features, d_model, context_dim=embed_dim, dropout=dropout)

        # LSTM encoder.
        self.lstm = nn.LSTM(
            d_model, d_model, n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )

        # Gated skip after LSTM (context-conditioned).
        self.lstm_skip = GatedSkip(d_model, dropout)

        # Interpretable multi-head self-attention.
        self.attn = InterpretableMultiHeadAttention(d_model, n_heads, dropout)

        # Gated skip after attention.
        self.attn_skip = GatedSkip(d_model, dropout)

        # Position-wise GRN feedforward.
        self.ff_grn = GRN(d_model, d_model * 2, d_model, dropout=dropout)

        # Gated skip after feedforward.
        self.ff_skip = GatedSkip(d_model, dropout)

        # Output head: last time step + ticker embed → tanh.
        self.head = TanhHead(d_model + embed_dim)

    def forward(
        self, x: torch.Tensor, ticker_ids: torch.Tensor
    ) -> torch.Tensor:
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
        emb = self.ticker_embed(ticker_ids)          # (B, embed_dim)

        # 1. VSN with static context.
        vsn_out, _sel = self.vsn(x, context=emb)     # (B, L, d_model)

        # 2. LSTM encoder.
        lstm_out, _ = self.lstm(vsn_out)             # (B, L, d_model)

        # 3. Gate LSTM output; skip from VSN.
        h = self.lstm_skip(lstm_out, vsn_out)        # (B, L, d_model)

        # 4. Interpretable self-attention.
        attn_out, _ = self.attn(h)                   # (B, L, d_model)

        # 5. Gate attention output; skip from pre-attention.
        h = self.attn_skip(attn_out, h)              # (B, L, d_model)

        # 6. Position-wise GRN feedforward.
        ff_out = self.ff_grn(h)                      # (B, L, d_model)

        # 7. Gate feedforward; skip from pre-feedforward.
        h = self.ff_skip(ff_out, h)                  # (B, L, d_model)

        # 8. Aggregate: last time step only.
        last = h[:, -1, :]                           # (B, d_model)

        # 9. Concat ticker embed and project to ŷ.
        combined = torch.cat([last, emb], dim=-1)    # (B, d_model + embed_dim)
        return self.head(combined)                   # (B,)

    def attention_weights(
        self, x: torch.Tensor, ticker_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (ŷ, attn_weights) without grad — for interpretability analysis."""
        emb = self.ticker_embed(ticker_ids)
        vsn_out, sel = self.vsn(x, context=emb)
        lstm_out, _ = self.lstm(vsn_out)
        h = self.lstm_skip(lstm_out, vsn_out)
        attn_out, A = self.attn(h)
        h = self.attn_skip(attn_out, h)
        ff_out = self.ff_grn(h)
        h = self.ff_skip(ff_out, h)
        last = h[:, -1, :]
        combined = torch.cat([last, emb], dim=-1)
        return self.head(combined), A   # (B,), (B, T, T)


# ---------------------------------------------------------------------------
# End-to-end training
# ---------------------------------------------------------------------------


def train_tft(
    oof_df: pd.DataFrame,
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    cfg: Optional[TrainConfig] = None,
    d_model: int = 32,
    n_heads: int = 4,
    embed_dim: int = 8,
    n_lstm_layers: int = 1,
    dropout: float = 0.1,
    lookback: int = config.LOOKBACK_L,
    verbose: bool = True,
) -> tuple[TFTModel, TrainHistory]:
    """Train the TFT on all available training events.

    Reuses the same data pipeline as VSN+LSTM (`_build_training_arrays`)
    so the two models are trained on identical inputs — a fair comparison.

    Returns
    -------
    model
        Best-checkpoint TFTModel (on CPU).
    history
        Train/val Sharpe per epoch.
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

    model = TFTModel(
        n_features=N_FEATURES,
        n_instruments=N_INSTRUMENTS,
        d_model=d_model,
        n_heads=n_heads,
        embed_dim=embed_dim,
        n_lstm_layers=n_lstm_layers,
        dropout=dropout,
    )

    if verbose:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  TFT parameters: {n_params:,}")

    model, hist = fit(model, X, ret, ticker_ids, cfg, verbose=verbose)
    return model, hist


# ---------------------------------------------------------------------------
# Weight generation for backtest
# ---------------------------------------------------------------------------


def make_weights_tft(
    model: TFTModel,
    oos_events: pd.DataFrame,
    vol_panel: pd.DataFrame,
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    lookback: int = config.LOOKBACK_L,
    sigma_tgt: float = config.SIGMA_TGT,
    max_leverage: float = config.MAX_LEVERAGE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Produce (events_df, weights_series) for barrier_backtest."""
    print("  Building OOS feature windows for TFT …")
    X_test, tid_test, built = _build_test_arrays(
        oos_events, ohlcv, signals, lookback=lookback
    )

    model.eval()
    with torch.no_grad():
        Xt  = torch.tensor(X_test, dtype=torch.float32)
        tt  = torch.tensor(tid_test, dtype=torch.long)
        y_hat_np = model(Xt, tt).numpy()

    y_hat_np[~built] = 0.0

    y_hat = pd.Series(y_hat_np, index=oos_events.index, name="y_hat")

    from stml.new_work.weights import _vol_at_events, apply_vol_targeting_to_conviction
    ann_sigma = _vol_at_events(oos_events, vol_panel)
    weights   = apply_vol_targeting_to_conviction(
        y_hat, ann_sigma, sigma_tgt=sigma_tgt, max_leverage=max_leverage
    )

    n_active = (weights.abs() > 0).sum()
    mean_w   = weights[weights != 0].abs().mean() if n_active else 0.0
    print(f"  [TFT] active={n_active}/{len(oos_events)} | "
          f"mean |w|={mean_w:.4f} | "
          f"ŷ range=[{y_hat_np.min():.3f}, {y_hat_np.max():.3f}]")

    return oos_events.copy(), weights
