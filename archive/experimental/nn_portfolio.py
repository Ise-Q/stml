"""End-to-end NN portfolio construction -- Madmoun Optional Session 3 §4.

Implements the lecturer's "Portfolio Construction with Neural Networks" section
(slides 36-46) plus the "Bridge: Meta-Signals as Features" (slides 47-53) on top
of our meta-labelling pipeline.

For each instrument :math:`k` and day :math:`t`:

  * Inputs: lookback window :math:`X_{t,k} \\in \\mathbb R^{L \\times (d+2)}`
    of ``(features, primary side, calibrated meta-prob p̂)`` (slide 48 channel
    layout).
  * Backbone :math:`g_\\phi`: maps the window to a temporal state
    :math:`h_{t,k} \\in \\mathbb R^H`. Three choices in this module:
      - :class:`LinearBackbone` -- DLinear-style (trend / season decomposition).
      - :class:`LSTMBackbone`   -- canonical recurrent baseline.
      - :class:`VLSTMBackbone`  -- VSN (variable selection net) + LSTM
        (TFT's interpretability building block, slide 45 "VLSTM").
  * Projection head: :math:`\\hat y_{t,k} = \\tanh(w_{\\text{lin}}^\\top h_{t,k}
    + b_{\\text{lin}}) \\in [-1, +1]` (slide 38).
  * Volatility targeting (slide 40):
    :math:`w_{t,k} = \\hat y_{t,k} \\cdot \\sigma_{\\text{tgt}} /
    \\sigma_{t,k}^{\\text{ann}}` with ``σ_tgt = 10 %`` annualised and
    ``σ_{t,k} = σ_{t,k}^{\\text{daily}} \\sqrt{252}``.
  * Cross-sectional aggregation (slide 41):
    :math:`R^{\\text{port}}_{t+1} = (1/K_{\\text{active}}) \\sum_k w_{t,k}
    r_{t+1,k}`.
  * Loss (slide 42): negative annualised Sharpe of the pooled path
    :math:`L(\\theta) = - \\sqrt{252} \\cdot \\hat{\\mathbb E}[R^{\\text{port}}]
    / (\\sqrt{\\hat{\\text{Var}}[R^{\\text{port}}] + \\epsilon})`.
  * Training: Adam + gradient clipping + early-stop on validation Sharpe
    (patience 20, slide 51).
  * Inference: forward only (slide 52-53).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.optim import Adam
    _HAS_TORCH = True
except Exception:  # pragma: no cover -- torch always present in this env
    _HAS_TORCH = False


def _t2np(t: "torch.Tensor") -> np.ndarray:
    """torch.Tensor -> numpy array, avoiding the broken numpy<->torch buffer
    bridge on torch-2.2.x with numpy 2.x. Uses ``.tolist()`` as the carrier.
    """
    return np.asarray(t.detach().cpu().tolist())


BackboneName = Literal["linear", "lstm", "vlstm", "tft"]


# ---------------------------------------------------------------------------
# Loss + aggregation primitives (slides 41-42).
# ---------------------------------------------------------------------------


def aggregate_portfolio(
    weights: "torch.Tensor", forward_returns: "torch.Tensor"
) -> "torch.Tensor":
    """Slide 41 cross-sectional 1/K_active aggregation.

    ``weights`` and ``forward_returns`` are (T, K) tensors aligned so column
    ``k`` is instrument ``k``. Returns a (T,) series of per-day portfolio
    returns. K_active(t) is the number of instruments with a *non-zero* weight
    at ``t``; days with zero active are zero-return.
    """
    w = weights
    r = forward_returns
    raw = (w * r).sum(dim=1)
    k_active = (w.abs() > 1e-9).float().sum(dim=1).clamp(min=1.0)
    out = raw / k_active
    no_active = (w.abs().sum(dim=1) <= 1e-9)
    return out.masked_fill(no_active, 0.0)


def sharpe_loss(
    port_returns: "torch.Tensor", *, eps: float = 1e-8, ann: float = 252.0
) -> "torch.Tensor":
    """Slide 42 -- negative differentiable annualised Sharpe.

    ``L(θ) = - Ê[R] / √(V̂ar[R] + ε) · √ann``. Minimising this maximises
    expected return while penalising variance.
    """
    mean = port_returns.mean()
    var = port_returns.var(unbiased=False)
    return -(mean / torch.sqrt(var + eps)) * math.sqrt(ann)


# ---------------------------------------------------------------------------
# Backbones (slide 45 architecture grid -- we ship 3).
# ---------------------------------------------------------------------------


class LinearBackbone(nn.Module):
    """DLinear-style: decompose lookback into trend + season, project to H.

    A direct linear map ``flatten(X) -> H`` would have too many parameters on
    long windows. DLinear (Zeng et al. 2022) factors the input into a moving-
    average trend and a residual, projects each linearly, and sums.
    Used as a "trivially small" baseline -- the simplest credible
    architecture in the lecture's grid (slide 45 "Linear" column).
    """

    def __init__(self, lookback: int, n_channels: int, hidden_dim: int,
                 ma_kernel: int = 5):
        super().__init__()
        self.L = lookback
        self.C = n_channels
        self.H = hidden_dim
        self.ma_kernel = ma_kernel
        # Moving-average pool for trend extraction.
        # Use a fixed averaging kernel implemented as a conv.
        self.trend_proj = nn.Linear(n_channels * lookback, hidden_dim)
        self.season_proj = nn.Linear(n_channels * lookback, hidden_dim)

    def _moving_average(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (B, L, C). Average over a sliding window along L.
        k = self.ma_kernel
        x_t = x.transpose(1, 2)  # (B, C, L)
        pad = (k - 1) // 2
        x_pad = nn.functional.pad(x_t, (pad, pad), mode="replicate")
        kernel = torch.ones(1, 1, k, device=x.device) / float(k)
        # Apply per-channel: depthwise conv.
        kernel = kernel.expand(self.C, 1, k)
        trend = nn.functional.conv1d(x_pad, kernel, groups=self.C)
        return trend.transpose(1, 2)  # (B, L, C)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (B, L, C)
        trend = self._moving_average(x)
        season = x - trend
        h_t = self.trend_proj(trend.reshape(trend.size(0), -1))
        h_s = self.season_proj(season.reshape(season.size(0), -1))
        return h_t + h_s


class LSTMBackbone(nn.Module):
    """Canonical recurrent baseline -- a single LSTM, take last hidden state."""

    def __init__(self, lookback: int, n_channels: int, hidden_dim: int,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.L = lookback
        self.C = n_channels
        self.H = hidden_dim
        self.lstm = nn.LSTM(
            input_size=n_channels, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (B, L, C)
        out, (h_n, _) = self.lstm(x)
        return h_n[-1]  # (B, H) -- last layer's last hidden state


class _VariableSelectionNetwork(nn.Module):
    """TFT's per-feature gating (Lim et al. 2021 §5.1).

    For each input channel, compute a softmax-normalised weight from the
    flattened lookback, then take a convex combination of per-channel
    embeddings. This learns which channels matter most.
    """

    def __init__(self, lookback: int, n_channels: int, embed_dim: int):
        super().__init__()
        self.C = n_channels
        self.L = lookback
        self.E = embed_dim
        # Per-channel embedding from the L-window.
        self.channel_embed = nn.Linear(lookback, embed_dim)
        # Selection scores from the flattened input.
        self.score_net = nn.Sequential(
            nn.Linear(n_channels * lookback, n_channels * 2),
            nn.ELU(),
            nn.Linear(n_channels * 2, n_channels),
        )

    def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        # x: (B, L, C) -> per-channel embeddings of shape (B, C, E).
        B = x.size(0)
        x_per_channel = x.transpose(1, 2)  # (B, C, L)
        embeds = self.channel_embed(x_per_channel)  # (B, C, E)
        # Selection weights from flat input.
        scores = self.score_net(x.reshape(B, -1))   # (B, C)
        weights = torch.softmax(scores, dim=1)        # (B, C)
        selected = (embeds * weights.unsqueeze(-1)).sum(dim=1)  # (B, E)
        return selected, weights


# ---------------------------------------------------------------------------
# TFT building blocks (Lim et al. 2021).
# ---------------------------------------------------------------------------


class _GatedLinearUnit(nn.Module):
    """Gated Linear Unit -- GLU(x) = (W_1 x + b_1) ⊙ σ(W_2 x + b_2).

    Lim et al. §5.1: the gating mechanism that controls residual contribution
    everywhere in the TFT. When σ(W_2 x + b_2) ≈ 0 the residual is skipped
    entirely; when ≈ 1 the linear projection passes through.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.lin(x) * torch.sigmoid(self.gate(x))


class _GatedResidualNetwork(nn.Module):
    """Gated Residual Network -- the TFT's basic computational block (§5.1).

        η_1 = ELU(W_2 a + W_3 c + b_2)        (context optional)
        η_2 = W_1 η_1 + b_1
        GRN(a, c) = LayerNorm( a + GLU(η_2) )

    Allows the model to skip the block entirely when not useful (GLU gate -> 0).
    """

    def __init__(self, input_dim: int, hidden_dim: int | None = None,
                 output_dim: int | None = None, context_dim: int | None = None,
                 dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        output_dim = output_dim or input_dim
        self.skip = (input_dim != output_dim)
        self.skip_proj = nn.Linear(input_dim, output_dim) if self.skip else None
        self.lin_a = nn.Linear(input_dim, hidden_dim)
        self.lin_c = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.lin_eta = nn.Linear(hidden_dim, output_dim)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.glu = _GatedLinearUnit(output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, a: "torch.Tensor",
                 c: "torch.Tensor | None" = None) -> "torch.Tensor":
        residual = self.skip_proj(a) if self.skip else a
        eta1 = self.lin_a(a)
        if c is not None and self.lin_c is not None:
            c_proj = self.lin_c(c)
            # Broadcast across a's middle dims if needed.
            while c_proj.dim() < eta1.dim():
                c_proj = c_proj.unsqueeze(1)
            eta1 = eta1 + c_proj
        eta1 = self.elu(eta1)
        eta1 = self.dropout(eta1)
        eta2 = self.lin_eta(eta1)
        return self.norm(residual + self.glu(eta2))


class _PerStepVSN(nn.Module):
    """Per-step variable selection -- TFT §5.1, applied at every time step.

    At each step t, compute per-channel selection weights from the
    concatenated channels at that step (+ optional static context), and
    return a convex combination of per-channel GRN-embedded values.
    Returns (selected sequence (B, L, E), per-step channel weights (B, L, C)).
    """

    def __init__(self, n_channels: int, embed_dim: int,
                 context_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.C = n_channels
        self.E = embed_dim
        # Per-channel GRN: each channel passes through its own GRN to produce
        # a per-step embedding.
        self.channel_grns = nn.ModuleList([
            _GatedResidualNetwork(input_dim=1, hidden_dim=embed_dim,
                                    output_dim=embed_dim, dropout=dropout)
            for _ in range(n_channels)
        ])
        # Selection GRN: maps concatenated raw values (and optional static
        # context) to C softmax-able scores.
        self.select_grn = _GatedResidualNetwork(
            input_dim=n_channels, hidden_dim=max(n_channels, embed_dim),
            output_dim=n_channels, context_dim=context_dim, dropout=dropout,
        )

    def forward(
        self, x: "torch.Tensor", static_context: "torch.Tensor | None" = None,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        # x: (B, L, C); static_context: (B, context_dim) if any.
        B, L, C = x.shape
        # Per-channel embedding via GRN.
        emb = []
        for ci, grn in enumerate(self.channel_grns):
            xi = x[:, :, ci:ci + 1]  # (B, L, 1)
            emb.append(grn(xi))      # (B, L, E)
        embeds = torch.stack(emb, dim=2)  # (B, L, C, E)
        # Selection weights per time-step.
        scores = self.select_grn(x, static_context)  # (B, L, C)
        weights = torch.softmax(scores, dim=2)         # (B, L, C)
        selected = (embeds * weights.unsqueeze(-1)).sum(dim=2)  # (B, L, E)
        return selected, weights


class _InterpretableMultiHeadAttention(nn.Module):
    """TFT's interpretable multi-head attention -- §5.3.

    Standard self-attention except the V projection is SHARED across heads
    so the per-head attention weights can be averaged into a single
    interpretable attention pattern.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_head, bias=False)  # SHARED across heads.
        self.out_proj = nn.Linear(self.d_head, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        # x: (B, L, d_model). Causal mask: position t attends only to ≤ t.
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).unsqueeze(1).expand(B, self.n_heads, L, self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal_mask = torch.triu(
            torch.full((L, L), float("-inf"), device=x.device), diagonal=1,
        )
        scores = scores + causal_mask
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        head_out = torch.matmul(attn, v)            # (B, H, L, d_head)
        # Average across heads (the "interpretable" part).
        out = head_out.mean(dim=1)                    # (B, L, d_head)
        return self.out_proj(out), attn.mean(dim=1)  # (B, L, d_model), (B, L, L)


class TFTBackbone(nn.Module):
    """Temporal Fusion Transformer (Lim et al. 2021) -- single-horizon variant.

    Architecture (slide 45 ``TFT``; paper §5.1-5.3):

      1. Per-step VSN: select important channels at each time step.
      2. LSTM encoder: capture local temporal dynamics.
      3. Gated skip + LayerNorm around the LSTM output.
      4. Static enrichment GRN: condition LSTM output on a static context
         vector (derived here from the LSTM's final hidden state -- a
         simplification of TFT's static covariate encoders).
      5. Interpretable multi-head self-attention over the enriched sequence.
      6. Gated skip + LayerNorm around the attention output.
      7. Position-wise feed-forward GRN.
      8. Final gated skip + LayerNorm.
      9. Take the last time-step's hidden state -> output dim H.

    Simplifications vs the full paper: no quantile decoder (single-horizon),
    no separate static covariate encoders for state cells / additive context
    (we derive a single static vector from the LSTM final hidden), no
    distinction between known/observed/static channel splits (everything is
    treated as time-varying input). The interpretability hooks
    (`last_channel_weights`, `last_attention`) are preserved.
    """

    def __init__(self, lookback: int, n_channels: int, hidden_dim: int,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.L = lookback
        self.C = n_channels
        self.H = hidden_dim
        self.n_heads = n_heads
        # Round H up to nearest multiple of n_heads (needed for attention).
        if hidden_dim % n_heads != 0:
            hidden_dim = ((hidden_dim // n_heads) + 1) * n_heads
        self.H = hidden_dim
        # 1. Per-step VSN.
        self.vsn = _PerStepVSN(
            n_channels=n_channels, embed_dim=hidden_dim, dropout=dropout,
        )
        # 2. LSTM encoder.
        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=1, batch_first=True,
        )
        # 3. Gated skip around LSTM.
        self.glu_lstm = _GatedLinearUnit(hidden_dim)
        self.norm_lstm = nn.LayerNorm(hidden_dim)
        # 4. Static enrichment GRN.
        self.static_grn = _GatedResidualNetwork(
            input_dim=hidden_dim, hidden_dim=hidden_dim,
            output_dim=hidden_dim, context_dim=hidden_dim, dropout=dropout,
        )
        # 5. Interpretable multi-head attention.
        self.attn = _InterpretableMultiHeadAttention(
            d_model=hidden_dim, n_heads=n_heads, dropout=dropout,
        )
        # 6. Gated skip around attention.
        self.glu_attn = _GatedLinearUnit(hidden_dim)
        self.norm_attn = nn.LayerNorm(hidden_dim)
        # 7. Position-wise feed-forward GRN.
        self.ff_grn = _GatedResidualNetwork(
            input_dim=hidden_dim, hidden_dim=hidden_dim,
            output_dim=hidden_dim, dropout=dropout,
        )
        # 8. Final gated skip.
        self.glu_final = _GatedLinearUnit(hidden_dim)
        self.norm_final = nn.LayerNorm(hidden_dim)

        # Interpretability hooks.
        self.last_channel_weights: torch.Tensor | None = None
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (B, L, C).
        # 1. Per-step VSN.
        selected, ch_weights = self.vsn(x)         # (B, L, H), (B, L, C)
        self.last_channel_weights = ch_weights.detach()
        # 2. LSTM encoder.
        lstm_out, (h_n, _) = self.lstm(selected)   # (B, L, H), (1, B, H)
        # 3. Gated skip + LayerNorm around LSTM.
        seq = self.norm_lstm(selected + self.glu_lstm(lstm_out))
        # 4. Static enrichment: use LSTM final hidden as static context.
        static_ctx = h_n.squeeze(0)                # (B, H)
        enriched = self.static_grn(seq, static_ctx)  # (B, L, H)
        # 5. Interpretable multi-head attention.
        attn_out, attn_w = self.attn(enriched)     # (B, L, H), (B, L, L)
        self.last_attention = attn_w.detach()
        # 6. Gated skip + LayerNorm around attention.
        post_attn = self.norm_attn(enriched + self.glu_attn(attn_out))
        # 7. Position-wise feed-forward GRN.
        ff = self.ff_grn(post_attn)                  # (B, L, H)
        # 8. Final gated skip.
        out = self.norm_final(post_attn + self.glu_final(ff))
        # 9. Take last time-step.
        return out[:, -1, :]                          # (B, H)


class VLSTMBackbone(nn.Module):
    """VSN + LSTM -- slide 45 "VLSTM (VSN+LSTM)".

    VSN compresses the C channels of each lookback step into a single
    embedding using a per-channel softmax selection. The resulting (B, L, E)
    sequence is then processed by an LSTM. Adds interpretability (the per-
    step channel weights) without the full TFT complexity.
    """

    def __init__(self, lookback: int, n_channels: int, hidden_dim: int,
                 embed_dim: int = 16, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.L = lookback
        self.C = n_channels
        self.H = hidden_dim
        self.E = embed_dim
        # Apply a single VSN to the WHOLE window (not per-step) -- simpler
        # and faster, captures the global "which channels matter" view.
        self.vsn = _VariableSelectionNetwork(
            lookback=lookback, n_channels=n_channels, embed_dim=embed_dim,
        )
        # Then per-step LSTM on the raw channels for sequential structure.
        self.lstm = nn.LSTM(
            input_size=n_channels, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Fuse VSN embedding with LSTM final hidden.
        self.fuse = nn.Linear(embed_dim + hidden_dim, hidden_dim)
        self.last_channel_weights: torch.Tensor | None = None

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        vsn_h, channel_weights = self.vsn(x)
        self.last_channel_weights = channel_weights.detach()
        lstm_out, (h_n, _) = self.lstm(x)
        h_lstm = h_n[-1]
        combined = torch.cat([vsn_h, h_lstm], dim=1)
        return torch.relu(self.fuse(combined))


# ---------------------------------------------------------------------------
# Full portfolio model -- backbone + tanh head + vol-target weight (slide 40).
# ---------------------------------------------------------------------------


class PortfolioModel(nn.Module):
    """Full forward: lookback windows → ŷ → vol-targeted weights.

    Treats every instrument with a SHARED backbone. The per-instrument
    distinction is encoded in the input channels (which include the
    instrument-specific features, primary side, and meta-probability), so
    no separate per-instrument head is needed -- this matches the
    lecturer's "ticker embedding added so the shared model can specialise
    per instrument" (slide 38) once a one-hot inst-id channel is added.
    """

    def __init__(
        self,
        backbone: nn.Module,
        target_vol: float = 0.10,
        max_leverage: float = 10.0,
        trading_days: float = 252.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.target_vol = float(target_vol)
        self.max_leverage = float(max_leverage)
        self.trading_days = float(trading_days)
        self.head = nn.Linear(backbone.H, 1)

    def forward(
        self, X: "torch.Tensor", sigma_daily: "torch.Tensor"
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """X: (B, L, C). sigma_daily: (B,) daily σ̂ for the asset at t.

        Returns ``(y_hat, weight)`` of shape ``(B,)``.
        """
        h = self.backbone(X)
        y = torch.tanh(self.head(h).squeeze(-1))  # (B,)
        # Replace NaN sigma with 1.0 (will be masked out below); avoid NaN
        # propagation in the leverage division.
        sigma_clean = torch.nan_to_num(sigma_daily, nan=1.0, posinf=1.0, neginf=1.0)
        sigma_ann = sigma_clean * math.sqrt(self.trading_days)
        sigma_ann_safe = torch.clamp(sigma_ann, min=1e-6)
        lev = self.target_vol / sigma_ann_safe
        lev = torch.clamp(lev, min=0.0, max=self.max_leverage)
        w = y * lev
        # If σ̂ was non-finite or non-positive, zero out the weight.
        finite_mask = torch.isfinite(sigma_daily) & (sigma_daily > 0)
        w = w * finite_mask.float()
        return y, w


# ---------------------------------------------------------------------------
# Builder.
# ---------------------------------------------------------------------------


def build_portfolio_model(
    backbone_name: BackboneName,
    *,
    lookback: int,
    n_channels: int,
    hidden_dim: int,
    target_vol: float = 0.10,
    max_leverage: float = 10.0,
    seed: int = 42,
) -> PortfolioModel:
    """Build a :class:`PortfolioModel` for one of three backbones."""
    if not _HAS_TORCH:
        raise RuntimeError("torch is not available")
    torch.manual_seed(seed)
    if backbone_name == "linear":
        backbone: nn.Module = LinearBackbone(
            lookback=lookback, n_channels=n_channels, hidden_dim=hidden_dim,
        )
    elif backbone_name == "lstm":
        backbone = LSTMBackbone(
            lookback=lookback, n_channels=n_channels, hidden_dim=hidden_dim,
        )
    elif backbone_name == "vlstm":
        backbone = VLSTMBackbone(
            lookback=lookback, n_channels=n_channels, hidden_dim=hidden_dim,
        )
    elif backbone_name == "tft":
        backbone = TFTBackbone(
            lookback=lookback, n_channels=n_channels, hidden_dim=hidden_dim,
        )
    else:
        raise ValueError(f"unknown backbone {backbone_name!r}")
    return PortfolioModel(
        backbone=backbone,
        target_vol=target_vol,
        max_leverage=max_leverage,
    )


# ---------------------------------------------------------------------------
# Dataset assembly.
# ---------------------------------------------------------------------------


@dataclass
class PortfolioPanel:
    """Causal panel for training the NN portfolio model.

    Attributes
    ----------
    dates : pd.DatetimeIndex, shape (T,)
        Trading days the panel spans.
    instruments : list[str], length K
        Instrument tickers, fixed order.
    X : np.ndarray of shape (T, K, L, C)
        Per-instrument causal lookback windows of channels
        ``(features, primary side, calibrated p̂_ff)``.
    sigma_daily : np.ndarray of shape (T, K)
        Causal EWMA σ̂_{t,k} (daily, per slide 39). NaN where unavailable.
    r_next : np.ndarray of shape (T, K)
        Per-instrument log-return realised at ``t+1`` (the asset's own
        next-day return). NaN where the instrument doesn't trade.
    mask : np.ndarray of shape (T, K)
        Boolean — True where this (t, k) is valid (window not all-NaN AND
        σ̂ available AND r_{t+1} available).
    """

    dates: pd.DatetimeIndex
    instruments: list[str]
    X: np.ndarray = field(repr=False)
    sigma_daily: np.ndarray = field(repr=False)
    r_next: np.ndarray = field(repr=False)
    mask: np.ndarray = field(repr=False)
    feature_names: list[str] = field(default_factory=list)

    @property
    def T(self) -> int: return self.X.shape[0]
    @property
    def K(self) -> int: return self.X.shape[1]
    @property
    def L(self) -> int: return self.X.shape[2]
    @property
    def C(self) -> int: return self.X.shape[3]


def chronological_split(
    panel: PortfolioPanel, *, val_frac: float = 0.20
) -> tuple[PortfolioPanel, PortfolioPanel]:
    """Chronologically last ``val_frac`` of the panel becomes the val set."""
    T = panel.T
    n_val = max(1, int(T * val_frac))
    cut = T - n_val
    def _slice(p, lo, hi):
        return PortfolioPanel(
            dates=p.dates[lo:hi], instruments=p.instruments,
            X=p.X[lo:hi], sigma_daily=p.sigma_daily[lo:hi],
            r_next=p.r_next[lo:hi], mask=p.mask[lo:hi],
            feature_names=p.feature_names,
        )
    return _slice(panel, 0, cut), _slice(panel, cut, T)


# ---------------------------------------------------------------------------
# Training and inference.
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Hyperparameters for :func:`train_portfolio_model` (slide 51)."""

    lr: float = 1e-3
    epochs: int = 200
    grad_clip: float = 1.0
    patience: int = 20  # early-stop patience on val Sharpe (slide 51)
    weight_decay: float = 1e-5
    seed: int = 42
    device: str = "cpu"  # MPS sometimes unstable for small models.
    verbose: bool = False


def _forward_panel(
    model: PortfolioModel, panel: PortfolioPanel, *, device: str
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """One forward pass over the whole panel; returns (W, port_returns).

    ``W`` is (T, K) the weight matrix. ``port_returns`` is (T,) the per-day
    portfolio return ``(1/K_active) Σ w·r``.
    """
    T, K, L, C = panel.X.shape
    X = torch.tensor(panel.X, dtype=torch.float32, device=device)
    sigma = torch.tensor(panel.sigma_daily, dtype=torch.float32, device=device)
    r_next = torch.tensor(panel.r_next, dtype=torch.float32, device=device)
    mask = torch.tensor(panel.mask, dtype=torch.float32, device=device)
    # Reshape into (T*K, L, C) for shared-backbone forward.
    X_flat = X.reshape(T * K, L, C)
    sigma_flat = sigma.reshape(T * K)
    _y, w = model(X_flat, sigma_flat)
    W = w.reshape(T, K) * mask
    R_next = torch.nan_to_num(r_next, nan=0.0) * mask
    port = aggregate_portfolio(W, R_next)
    return W, port


def train_portfolio_model(
    model: PortfolioModel,
    train_panel: PortfolioPanel,
    val_panel: PortfolioPanel | None = None,
    *,
    cfg: TrainConfig | None = None,
) -> dict:
    """Slide 51 -- Adam + grad clip + early stop on validation Sharpe."""
    cfg = cfg or TrainConfig()
    if not _HAS_TORCH:
        raise RuntimeError("torch is not available")
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = model.to(device)

    opt = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val_sharpe = -float("inf")
    best_state = None
    epochs_since_best = 0
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        _W, port = _forward_panel(model, train_panel, device=cfg.device)
        loss = sharpe_loss(port)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        train_sharpe = float(-loss.item())
        val_sharpe = float("nan")
        if val_panel is not None and val_panel.T > 5:
            model.eval()
            with torch.no_grad():
                _W_val, port_val = _forward_panel(model, val_panel, device=cfg.device)
                val_loss = sharpe_loss(port_val)
                val_sharpe = float(-val_loss.item())
        history.append({"epoch": epoch, "train_sharpe": train_sharpe,
                         "val_sharpe": val_sharpe})

        monitor = val_sharpe if val_panel is not None else train_sharpe
        if np.isfinite(monitor) and monitor > best_val_sharpe:
            best_val_sharpe = monitor
            best_state = {k: v.detach().cpu().clone()
                           for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if cfg.verbose and (epoch % 10 == 0 or epoch == cfg.epochs - 1):
            print(f"  epoch {epoch:4d}: train_SR={train_sharpe:+.3f}  "
                  f"val_SR={val_sharpe:+.3f}  best={best_val_sharpe:+.3f}  "
                  f"patience={epochs_since_best}/{cfg.patience}")

        if epochs_since_best >= cfg.patience:
            if cfg.verbose:
                print(f"  early stop at epoch {epoch}: best_val_SR={best_val_sharpe:+.3f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_sharpe": best_val_sharpe,
        "n_epochs_run": len(history),
        "history": pd.DataFrame(history),
    }


def predict_weights(
    model: PortfolioModel, panel: PortfolioPanel, *, device: str = "cpu"
) -> pd.DataFrame:
    """Slide 52-53 -- forward only. Returns (date × instrument) weight DataFrame."""
    if not _HAS_TORCH:
        raise RuntimeError("torch is not available")
    model = model.to(torch.device(device))
    model.eval()
    with torch.no_grad():
        W, _port = _forward_panel(model, panel, device=device)
    arr = _t2np(W)
    return pd.DataFrame(arr, index=panel.dates, columns=panel.instruments)
