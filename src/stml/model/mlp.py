"""
mlp.py
======
PyTorch feed-forward MLP meta-model (BatchNorm + Dropout) and the shared torch training loop the
VSN model (``vsn.py``) also uses.

The data is tabular -- one row per signal day, ~80 features -- so a plain MLP is the natural
neural baseline (sequence models would need fabricated lookback windows on a 2.5-year panel).
Class imbalance is handled with a positive-class weight in ``BCEWithLogitsLoss``; inputs are
median-imputed and standardised on the training rows only (via
:class:`~stml.model.dataset.Preprocessor`). Everything is seeded for determinism.

The wrapper exposes the same ``fit`` / ``predict_proba`` / ``feature_names_`` surface as the tree
models, so the Optuna objective and importance code are model-agnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from stml.model.dataset import Preprocessor


def seed_torch(seed: int) -> None:
    """Seed torch + numpy and force deterministic CPU kernels."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _pos_weight(y: np.ndarray) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return (n_neg / n_pos) if n_pos > 0 else 1.0


class MLP(nn.Module):
    """Configurable BatchNorm+Dropout MLP with a single logit output."""

    def __init__(self, n_features: int, hidden: tuple[int, ...] = (64, 32),
                 dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_torch_classifier(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    pos_weight: float,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    sample_weight: np.ndarray | None = None,
    patience: int = 25,
    seed: int = 0,
) -> nn.Module:
    """Train any logit-output torch model with class-weighted BCE + early stopping.

    A small chronological tail of the training rows is held out as an internal early-stopping
    monitor (the rows arrive already time-ordered within a fold), so training never peeks at the
    CV validation fold. Returns the model with best-monitor weights restored.
    """
    seed_torch(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    wt = torch.tensor(sample_weight if sample_weight is not None else np.ones(len(y)),
                      dtype=torch.float32)

    n = len(y)
    n_mon = max(16, int(0.15 * n))
    tr_idx = slice(0, n - n_mon)
    mon_idx = slice(n - n_mon, n)
    pw = torch.tensor([pos_weight], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    gen = torch.Generator().manual_seed(seed)
    best_state, best_mon, bad = None, np.inf, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n - n_mon, generator=gen)
        for s in range(0, len(perm), batch_size):
            b = perm[s : s + batch_size]
            xb, yb, wb = Xt[tr_idx][b], yt[tr_idx][b], wt[tr_idx][b]
            if xb.shape[0] < 2:  # BatchNorm needs >=2 rows
                continue
            opt.zero_grad()
            logits = model(xb)
            per = loss_fn(logits, yb)
            # weight positives by pos_weight, then by sample-uniqueness weights
            w = wb * torch.where(yb > 0.5, pw, torch.ones_like(yb))
            (per * w).mean().backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            mon_logits = model(Xt[mon_idx])
            mon_loss = loss_fn(mon_logits, yt[mon_idx]).mean().item()
        if mon_loss < best_mon - 1e-5:
            best_mon, bad = mon_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


class MLPModel:
    """Tabular MLP wrapper with the uniform fit/predict_proba interface."""

    def __init__(self, params: dict, seed: int = 0) -> None:
        self.params = dict(params)
        self.seed = seed
        self.prep_: Preprocessor | None = None
        self.model_: MLP | None = None
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None):
        self.feature_names_ = list(X.columns)
        self.prep_ = Preprocessor().fit(X)
        Xt = self.prep_.transform(X)
        hidden = self.params.get("hidden", (64, 32))
        self.model_ = MLP(Xt.shape[1], hidden=hidden,
                          dropout=self.params.get("dropout", 0.2))
        train_torch_classifier(
            self.model_, Xt, y,
            pos_weight=_pos_weight(y),
            epochs=self.params.get("epochs", 200),
            batch_size=self.params.get("batch_size", 64),
            lr=self.params.get("lr", 1e-3),
            weight_decay=self.params.get("weight_decay", 1e-4),
            sample_weight=sample_weight,
            seed=self.seed,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            Xt = torch.tensor(self.prep_.transform(X), dtype=torch.float32)
            return torch.sigmoid(self.model_(Xt)).numpy()


def mlp_param_space(trial) -> dict:
    """Optuna search space for the MLP."""
    n_layers = trial.suggest_int("n_layers", 1, 3)
    widths = [trial.suggest_categorical(f"units_l{i}", [16, 32, 64, 128]) for i in range(n_layers)]
    return {
        "hidden": tuple(widths),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "epochs": trial.suggest_int("epochs", 100, 300, step=50),
    }
