"""Shared building blocks for all neural strategy models.

Components:
    GRN              — Gated Residual Network (Lim et al. 2021, §3.1)
    TanhHead         — Linear → Tanh final layer producing ŷ ∈ [−1, 1]
    TickerEmbedding  — Learnable instrument ID embedding
    PooledSharpeLoss — −annualised Sharpe of {ŷ_i · ret_i} pooled over a batch
    EarlyStopper     — Patience-based early stopping on validation Sharpe
    fit              — Standard epoch-loop training harness
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Device selection (MPS > CUDA > CPU)
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# GRN — Gated Residual Network
# ---------------------------------------------------------------------------


class GRN(nn.Module):
    """Gated Residual Network from TFT (Lim et al. 2021, eq. 1–4).

    GRN(a, c=None):
        h = ELU(W1 · a + b1  [+ Wc · c])    # context modulation
        h = W2 · h + b2                       # project to 2 × output_dim for GLU
        h1, h2 = split(h)
        g = sigmoid(h1) ⊙ h2                  # gating
        output = LayerNorm(skip(a) + dropout(g))
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        context_dim: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.skip = (
            nn.Linear(input_dim, output_dim, bias=False)
            if input_dim != output_dim
            else nn.Identity()
        )
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_ctx = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim * 2)
        self.norm = nn.LayerNorm(output_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, a: torch.Tensor, c: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = self.skip(a)
        h = self.fc1(a)
        if c is not None and self.fc_ctx is not None:
            h = h + self.fc_ctx(c)
        h = self.elu(h)
        h = self.fc2(h)
        h1, h2 = h.chunk(2, dim=-1)
        g = torch.sigmoid(h1) * h2
        g = self.drop(g)
        return self.norm(residual + g)


# ---------------------------------------------------------------------------
# TanhHead — Linear → Tanh producing ŷ ∈ [−1, 1]
# ---------------------------------------------------------------------------


class TanhHead(nn.Module):
    """Single linear layer followed by tanh; output ∈ (−1, +1)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x)).squeeze(-1)   # (batch,)


# ---------------------------------------------------------------------------
# TickerEmbedding — learnable instrument-specific bias
# ---------------------------------------------------------------------------


class TickerEmbedding(nn.Module):
    """Embedding table keyed by integer instrument index."""

    def __init__(self, n_instruments: int, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_instruments, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.01)

    def forward(self, ticker_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(ticker_ids)  # (batch, embed_dim)


# ---------------------------------------------------------------------------
# PooledSharpeLoss — negative annualised Sharpe of sized per-event returns
# ---------------------------------------------------------------------------


class PooledSharpeLoss(nn.Module):
    """L(θ) = −Sharpe_ann({ŷ_i · ret_i}).

    Per the spec: L = −(mean(R) / sqrt(var(R) + ε)) · sqrt(ann).
    Uses biased variance (denominator N) for gradient stability; small ε
    guards divide-by-zero on degenerate batches.
    """

    def __init__(self, eps: float = 1e-6, ann: float = 252.0) -> None:
        super().__init__()
        self.eps = eps
        self.scale = math.sqrt(ann)

    def forward(self, y_hat: torch.Tensor, ret: torch.Tensor) -> torch.Tensor:
        R = y_hat * ret
        mu = R.mean()
        var = R.var(unbiased=False)
        sharpe = mu / torch.sqrt(var + self.eps) * self.scale
        return -sharpe


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 15            # early stopping patience (epochs)
    val_frac: float = 0.20        # fraction of events for time-series val split
    seed: int = 42
    ann: float = 252.0
    eps: float = 1e-6


@dataclass
class TrainHistory:
    train_sharpe: list[float] = field(default_factory=list)
    val_sharpe: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_sharpe: float = -float("inf")


class EarlyStopper:
    def __init__(self, patience: int = 15) -> None:
        self.patience = patience
        self.best = -float("inf")
        self.counter = 0
        self.best_state: dict | None = None

    def step(self, val_sharpe: float, model: nn.Module) -> bool:
        """Return True if training should stop."""
        if val_sharpe > self.best:
            self.best = val_sharpe
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _sharpe(y_hat: torch.Tensor, ret: torch.Tensor, eps: float, ann: float) -> float:
    with torch.no_grad():
        R = y_hat * ret
        mu = R.mean().item()
        var = R.var(unbiased=False).item()
        return (mu / math.sqrt(var + eps)) * math.sqrt(ann)


def fit(
    model: nn.Module,
    X: np.ndarray,           # (N, lookback, n_features)  float32
    ret: np.ndarray,         # (N,)  float32
    ticker_ids: np.ndarray,  # (N,)  int64
    cfg: TrainConfig = TrainConfig(),
    *,
    verbose: bool = True,
) -> tuple[nn.Module, TrainHistory]:
    """Time-series-split train loop with early stopping on validation Sharpe.

    Parameters
    ----------
    X
        Feature windows array.
    ret
        Realised triple-barrier returns (signed, same sign convention as side).
    ticker_ids
        Integer instrument indices, same length as X.
    cfg
        Hyper-parameter configuration.
    """
    torch.manual_seed(cfg.seed)
    device = get_device()
    model = model.to(device)

    # Time-series split — last val_frac of events (chronological order preserved).
    N = len(X)
    split = int(N * (1.0 - cfg.val_frac))

    Xt  = torch.tensor(X[:split],          dtype=torch.float32)
    rt  = torch.tensor(ret[:split],         dtype=torch.float32)
    tid = torch.tensor(ticker_ids[:split],  dtype=torch.long)

    Xv  = torch.tensor(X[split:],          dtype=torch.float32).to(device)
    rv  = torch.tensor(ret[split:],         dtype=torch.float32).to(device)
    tidv= torch.tensor(ticker_ids[split:],  dtype=torch.long).to(device)

    dataset = TensorDataset(Xt, rt, tid)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    optim  = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = PooledSharpeLoss(eps=cfg.eps, ann=cfg.ann)
    stopper = EarlyStopper(patience=cfg.patience)
    hist    = TrainHistory()

    if verbose:
        print(f"  Device: {device}  train={split}  val={N-split}  "
              f"epochs={cfg.epochs}  bs={cfg.batch_size}")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for Xb, rb, tidb in loader:
            Xb, rb, tidb = Xb.to(device), rb.to(device), tidb.to(device)
            optim.zero_grad()
            y_hat = model(Xb, tidb)
            loss  = loss_fn(y_hat, rb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

        # Validation Sharpe (no grad).
        model.eval()
        with torch.no_grad():
            yv = model(Xv, tidv)
        val_sh  = _sharpe(yv, rv, cfg.eps, cfg.ann)

        # Quick training Sharpe on the full training set.
        Xt_d  = Xt.to(device)
        rt_d  = rt.to(device)
        tid_d = tid.to(device)
        with torch.no_grad():
            yt = model(Xt_d, tid_d)
        train_sh = _sharpe(yt, rt_d, cfg.eps, cfg.ann)

        hist.train_sharpe.append(train_sh)
        hist.val_sharpe.append(val_sh)

        if val_sh > hist.best_val_sharpe:
            hist.best_val_sharpe = val_sh
            hist.best_epoch = epoch

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:4d}/{cfg.epochs}  "
                  f"train_SR={train_sh:.4f}  val_SR={val_sh:.4f}  "
                  f"[best {hist.best_val_sharpe:.4f} @ ep{hist.best_epoch}]")

        if stopper.step(val_sh, model):
            if verbose:
                print(f"  Early stop at epoch {epoch} (patience={cfg.patience})")
            break

    stopper.restore_best(model)
    if verbose:
        print(f"  Training done. Best val Sharpe: {hist.best_val_sharpe:.4f} "
              f"@ epoch {hist.best_epoch}")
    return model.to("cpu"), hist
