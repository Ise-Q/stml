"""Neural act/skip variants for the §3 horse-race (Stage 2 enrichment).

Kept separate from ``models.py`` so the torch/tensorflow imports stay lazy — ``pipeline.py``
uses only the tree/linear roster, and pulling a second OpenMP runtime into that import path is
what caused the earlier libomp segfault.

All variants conform to the ``MetaClassifier`` protocol (``fit(X, y, sample_weight=)`` /
``predict_act_proba(X)``) so the evaluation harness treats them identically. They impute (median)
and standardise inputs like the logistic path, since neural nets need a complete, scaled matrix.

Determinism: ``set_seeds`` at every fit + full-batch training (no minibatch shuffle) + the
single-thread env make the **torch** variants byte-reproducible. (The Keras VSN, added next,
carries the documented TensorFlow op-determinism caveat.)
"""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .seeding import set_seeds


class TorchMLP:
    """A small torch MLP act/skip classifier (BCE-with-logits, weighted, full-batch Adam)."""

    def __init__(
        self,
        *,
        seed: int = 42,
        hidden: tuple[int, ...] = (64, 32),
        epochs: int = 150,
        lr: float = 1e-3,
        dropout: float = 0.1,
        weight_decay: float = 1e-4,
    ):
        self.name = "torch_mlp"
        self.seed = seed
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay
        self._net = None
        self._imputer: SimpleImputer | None = None
        self._scaler: StandardScaler | None = None

    def _transform_fit(self, x: np.ndarray) -> np.ndarray:
        self._imputer = SimpleImputer(strategy="median").fit(x)
        self._scaler = StandardScaler().fit(self._imputer.transform(x))
        return self._scaler.transform(self._imputer.transform(x))

    def _transform(self, X) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        return self._scaler.transform(self._imputer.transform(x))

    def fit(self, X, y, sample_weight=None) -> "TorchMLP":
        import torch
        from torch import nn

        set_seeds(self.seed)
        x = self._transform_fit(np.asarray(X, dtype=float)).astype("float32")
        yv = np.asarray(y, dtype=float).astype("float32")
        w = (
            np.ones_like(yv)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).astype("float32")
        )
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(yv).view(-1, 1)
        wt = torch.from_numpy(w).view(-1, 1)

        layers: list = []
        dim = x.shape[1]
        for h in self.hidden:
            layers += [nn.Linear(dim, h), nn.ReLU(), nn.Dropout(self.dropout)]
            dim = h
        layers += [nn.Linear(dim, 1)]
        self._net = nn.Sequential(*layers)

        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self._net.train()
        for _ in range(self.epochs):  # full-batch, no shuffle -> deterministic
            opt.zero_grad()
            loss = (loss_fn(self._net(xt), yt) * wt).mean()
            loss.backward()
            opt.step()
        return self

    def predict_act_proba(self, X) -> np.ndarray:
        import torch

        self._net.eval()
        with torch.no_grad():
            logits = self._net(torch.from_numpy(self._transform(X).astype("float32")))
            return torch.sigmoid(logits).numpy().ravel()

    def predict_proba(self, X) -> np.ndarray:
        p = self.predict_act_proba(X)
        return np.column_stack([1.0 - p, p])
