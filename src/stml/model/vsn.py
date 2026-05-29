"""
vsn.py
======
A Variable Selection Network (VSN)-style gated classifier in PyTorch (after Programming Session 6,
reimplemented from Keras to torch).

The VSN's appeal here is **built-in interpretability**: a softmax selection layer assigns each
input feature a weight per sample, so averaging those weights over a dataset gives a native,
model-internal feature-importance ranking -- a neural complement to SHAP/permutation. Each feature
is projected to a small embedding and passed through its own Gated Residual Network (GRN); the
sample-wise selection weights combine them before a GRN head emits the logit.

It shares the imbalance handling, preprocessing and training loop of the MLP (``mlp.py``) and the
same fit/predict_proba interface, plus ``gate_weights`` for the selection importances.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from stml.model.dataset import Preprocessor
from stml.model.mlp import _pos_weight, train_torch_classifier


class _GLU(nn.Module):
    """Gated Linear Unit: ``a * sigmoid(b)`` from a single projection."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, 2 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class _GRN(nn.Module):
    """Gated Residual Network: ELU MLP + GLU gate + skip connection + LayerNorm."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.elu = nn.ELU()
        self.drop = nn.Dropout(dropout)
        self.glu = _GLU(d_out)
        self.ln = nn.LayerNorm(d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.drop(self.elu(self.fc1(x))))
        return self.ln(self.skip(x) + self.glu(h))


class VariableSelectionNet(nn.Module):
    """Per-feature embeddings combined by a softmax selection layer; single logit output.

    Vectorised for speed: each scalar feature is projected to ``d_model`` by its own affine
    weights (a learnable ``(n_features, d_model)`` matrix) and passed through a **shared**
    feature-GRN -- a common simplification of the per-variable-GRN VSN that runs without a
    Python loop over features while preserving the interpretable selection weights.
    """

    def __init__(self, n_features: int, d_model: int = 16, d_hidden: int = 32,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        # per-feature affine projection of a scalar -> d_model: weight (n, d), bias (n, d)
        self.proj_w = nn.Parameter(torch.randn(n_features, d_model) * 0.1)
        self.proj_b = nn.Parameter(torch.zeros(n_features, d_model))
        self.feat_grn = _GRN(d_model, d_hidden, d_model, dropout)  # shared across features
        self.select_grn = _GRN(n_features, d_hidden, n_features, dropout)
        self.out_grn = _GRN(d_model, d_hidden, d_model, dropout)
        self.head = nn.Linear(d_model, 1)
        self.last_weights_: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sel = torch.softmax(self.select_grn(x), dim=-1)            # (B, n_features)
        self.last_weights_ = sel.detach()
        emb = x.unsqueeze(-1) * self.proj_w + self.proj_b          # (B, n_features, d_model)
        b, n, d = emb.shape
        emb = self.feat_grn(emb.reshape(b * n, d)).reshape(b, n, d)
        combined = (sel.unsqueeze(-1) * emb).sum(dim=1)            # (B, d_model)
        return self.head(self.out_grn(combined)).squeeze(-1)


class VSNModel:
    """VSN wrapper with the uniform fit/predict_proba interface + ``gate_weights``."""

    def __init__(self, params: dict, seed: int = 0) -> None:
        self.params = dict(params)
        self.seed = seed
        self.prep_: Preprocessor | None = None
        self.model_: VariableSelectionNet | None = None
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None):
        self.feature_names_ = list(X.columns)
        self.prep_ = Preprocessor().fit(X)
        Xt = self.prep_.transform(X)
        self.model_ = VariableSelectionNet(
            Xt.shape[1],
            d_model=self.params.get("d_model", 16),
            d_hidden=self.params.get("d_hidden", 32),
            dropout=self.params.get("dropout", 0.1),
        )
        train_torch_classifier(
            self.model_, Xt, y,
            pos_weight=_pos_weight(y),
            epochs=self.params.get("epochs", 150),
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

    def gate_weights(self, X: pd.DataFrame) -> pd.Series:
        """Mean softmax selection weight per feature over ``X`` -- native VSN importance."""
        self.model_.eval()
        with torch.no_grad():
            Xt = torch.tensor(self.prep_.transform(X), dtype=torch.float32)
            self.model_(Xt)
            w = self.model_.last_weights_.mean(dim=0).numpy()
        return pd.Series(w, index=self.feature_names_).sort_values(ascending=False)


def vsn_param_space(trial) -> dict:
    """Optuna search space for the VSN."""
    return {
        "d_model": trial.suggest_categorical("d_model", [8, 16, 32]),
        "d_hidden": trial.suggest_categorical("d_hidden", [16, 32, 64]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "epochs": trial.suggest_int("epochs", 50, 110, step=30),
    }
