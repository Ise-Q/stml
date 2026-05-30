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

    def fit(self, X, y, sample_weight=None) -> TorchMLP:
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


class _NeuralBase:
    """Shared median-impute + standardise input handling for the NN variants."""

    def __init__(self):
        self._imputer: SimpleImputer | None = None
        self._scaler: StandardScaler | None = None

    def _transform_fit(self, x: np.ndarray) -> np.ndarray:
        self._imputer = SimpleImputer(strategy="median").fit(x)
        self._scaler = StandardScaler().fit(self._imputer.transform(x))
        return self._scaler.transform(self._imputer.transform(x)).astype("float32")

    def _transform(self, X) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        return self._scaler.transform(self._imputer.transform(x)).astype("float32")


class KerasVSN(_NeuralBase):
    """Variable Selection Network (Keras), reusing the vendored PS6 ``FinalModel``.

    Trained with PS6's imperative GradientTape loop (full-batch for determinism) with a
    sample_weight injected into the per-sample binary cross-entropy. All features are numerical
    (``num_categorical=0``). NOTE: TensorFlow op-determinism is best-effort — this variant may
    not be byte-reproducible across machines (documented caveat); the torch variants are.
    """

    def __init__(
        self,
        *,
        seed: int = 42,
        embedding_dim: int = 32,
        hidden_dim: int = 8,
        output_dim: int = 14,
        epochs: int = 40,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.name = "keras_vsn"
        self.seed = seed
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.epochs = epochs
        self.lr = lr
        self._model = None

    def fit(self, X, y, sample_weight=None) -> KerasVSN:
        import tensorflow as tf

        from ._vendor.vsn import FinalModel

        set_seeds(self.seed)
        x = self._transform_fit(np.asarray(X, dtype=float))
        yv = np.asarray(y, dtype=float).astype("float32")
        w = (
            np.ones_like(yv)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).astype("float32")
        )
        n_features = x.shape[1]
        self._model = FinalModel(
            self.embedding_dim, n_features, 0, [], self.hidden_dim, self.output_dim
        )
        x_num = tf.constant(x, dtype=tf.float32)
        x_cat = tf.zeros((x.shape[0], 0), dtype=tf.int32)
        yt = tf.reshape(tf.constant(yv), (-1, 1))
        wt = tf.constant(w)
        opt = tf.keras.optimizers.Adam(self.lr)
        for _ in range(self.epochs):  # full-batch imperative loop (PS6 pattern)
            with tf.GradientTape() as tape:
                pred, _ = self._model((x_num, x_cat))
                pred = tf.reshape(pred, (-1, 1))
                per_sample = tf.keras.losses.binary_crossentropy(yt, pred)  # (n,)
                loss = tf.reduce_mean(per_sample * wt)
            grads = tape.gradient(loss, self._model.trainable_variables)
            opt.apply_gradients(zip(grads, self._model.trainable_variables, strict=False))
        return self

    def predict_act_proba(self, X) -> np.ndarray:
        import tensorflow as tf

        x = self._transform(X)
        x_cat = tf.zeros((x.shape[0], 0), dtype=tf.int32)
        pred, _ = self._model((tf.constant(x, dtype=tf.float32), x_cat))
        return np.asarray(tf.reshape(pred, (-1,)), dtype=float)

    def predict_proba(self, X) -> np.ndarray:
        p = self.predict_act_proba(X)
        return np.column_stack([1.0 - p, p])


_TORCH_VSN_CACHE: dict = {}


def _torch_vsn_class():
    """Build the torch VSN module class lazily (keeps torch import out of module load)."""
    if "net" not in _TORCH_VSN_CACHE:
        import torch
        import torch.nn.functional as F
        from torch import nn

        class _GRN(nn.Module):  # Gated Residual Network (Lim et al. 2021)
            def __init__(self, in_dim, hidden, out_dim, dropout=0.1):
                super().__init__()
                self.fc1, self.fc2 = nn.Linear(in_dim, hidden), nn.Linear(hidden, hidden)
                self.drop = nn.Dropout(dropout)
                self.lin, self.gate = nn.Linear(hidden, out_dim), nn.Linear(hidden, out_dim)
                self.skip = None if in_dim == out_dim else nn.Linear(in_dim, out_dim)
                self.norm = nn.LayerNorm(out_dim)

            def forward(self, x):
                h = self.drop(self.fc2(F.elu(self.fc1(x))))
                glu = torch.sigmoid(self.gate(h)) * self.lin(h)
                return self.norm((x if self.skip is None else self.skip(x)) + glu)

        class _VSNNet(nn.Module):
            def __init__(self, n_features, emb, hidden, out_dim):
                super().__init__()
                self.n = n_features
                self.proj = nn.ModuleList([nn.Linear(1, emb) for _ in range(n_features)])
                self.grns = nn.ModuleList([_GRN(emb, hidden, out_dim) for _ in range(n_features)])
                self.flat = _GRN(out_dim * n_features, hidden, n_features)
                self.head = nn.Linear(out_dim, 1)

            def forward(self, x):  # x: (batch, n_features)
                emb = torch.stack([self.proj[i](x[:, i : i + 1]) for i in range(self.n)], dim=-1)
                feats = torch.stack([self.grns[i](emb[:, :, i]) for i in range(self.n)], dim=-1)
                weights = torch.softmax(self.flat(feats.reshape(feats.shape[0], -1)), dim=-1)
                out = (feats * weights.unsqueeze(1)).sum(dim=-1)
                return self.head(out).squeeze(-1), weights

        _TORCH_VSN_CACHE["net"] = _VSNNet
    return _TORCH_VSN_CACHE["net"]


class TorchVSN(_NeuralBase):
    """Variable Selection Network ported to torch — byte-deterministic, with softmax feature
    selection weights (the VSN interpretability), trained full-batch with weighted BCE."""

    def __init__(
        self,
        *,
        seed: int = 42,
        embedding_dim: int = 16,
        hidden_dim: int = 8,
        output_dim: int = 14,
        epochs: int = 150,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.name = "torch_vsn"
        self.seed = seed
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self._net = None

    def fit(self, X, y, sample_weight=None) -> TorchVSN:
        import torch
        from torch import nn

        set_seeds(self.seed)
        x = self._transform_fit(np.asarray(X, dtype=float))
        yv = np.asarray(y, dtype=float).astype("float32")
        w = (
            np.ones_like(yv)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).astype("float32")
        )
        net_cls = _torch_vsn_class()
        self._net = net_cls(x.shape[1], self.embedding_dim, self.hidden_dim, self.output_dim)
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(yv)
        wt = torch.from_numpy(w)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self._net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            logits, _ = self._net(xt)
            loss = (loss_fn(logits, yt) * wt).mean()
            loss.backward()
            opt.step()
        return self

    def predict_act_proba(self, X) -> np.ndarray:
        import torch

        self._net.eval()
        with torch.no_grad():
            logits, _ = self._net(torch.from_numpy(self._transform(X)))
            return torch.sigmoid(logits).numpy().ravel()

    def predict_proba(self, X) -> np.ndarray:
        p = self.predict_act_proba(X)
        return np.column_stack([1.0 - p, p])


def neural_roster(*, seed: int = 42) -> dict:
    """The three neural act/skip variants, keyed by name."""
    return {
        "torch_mlp": TorchMLP(seed=seed),
        "torch_vsn": TorchVSN(seed=seed),
        "keras_vsn": KerasVSN(seed=seed),
    }


def full_roster(*, seed: int = 42) -> dict:
    """The complete six-estimator horse-race: tree/linear + neural."""
    from .models import tree_linear_roster

    return {**tree_linear_roster(seed=seed), **neural_roster(seed=seed)}
