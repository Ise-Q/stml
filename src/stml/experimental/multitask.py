"""Multi-task neural net with per-instrument heads — methodology spec Family B + §8 S4.

Architecture (methodology spec):

    instrument id ──> Embedding(11, 8) ──┐
                                          ├──> concat ──> Linear(d+8, 64) + ReLU + Dropout(0.1)
    row features (d) ─────────────────────┘             └─> Linear(64, 32)  + ReLU + Dropout(0.1)
                                                         └─> Linear(32, 16) = shared h
                                                              ├─> Head_0: Linear(16, 1) + sigmoid
                                                              ├─> Head_1: Linear(16, 1) + sigmoid
                                                              ⋮
                                                              └─> Head_{n-1}

Joint training: ``loss = sum over rows of sample_weight × BCE(y_i, ŷ_{head[inst(i)]})``.
Only the head matching the row's instrument contributes — implemented by gathering
each row's prediction from its own head and stacking.

Determinism:
* Full-batch Adam (no minibatch shuffle).
* ``torch.manual_seed`` + ``use_deterministic_algorithms(True, warn_only=True)``.
* ``torch.set_num_threads(1)``.
* Early stopping on inner-CV val log-loss to control overfit.

Why instrument heads (over per-class XGB):
* Shared encoder learns class-level structure → less per-instrument overfit
  on the 400-600-event dense instruments.
* Per-instrument heads still specialise to each instrument's calibration scale.
* The 11-head design generalises across asset classes when used at the class
  level (3-4 heads per class model).

Per-class instance (not pooled across classes — methodology spec explicit).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="torch")  # numpy compat

# Lazy import — torch is in optional deps `multitask`.
def _torch():
    try:
        import torch  # type: ignore
        return torch
    except ImportError as e:
        raise ImportError(
            "torch is required for multitask NN. Install with: "
            "`uv sync --extra multitask`"
        ) from e


@dataclass(frozen=True)
class MultiTaskConfig:
    """Frozen hyperparameters for one training run.

    Defaults mirror methodology spec spec.
    """

    embed_dim: int = 8
    hidden_widths: tuple[int, ...] = (64, 32)
    shared_dim: int = 16
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 200
    early_stop_patience: int = 25
    seed: int = 42
    # If the inner CV val fraction; 0 = no early stop (use n_epochs).
    val_frac: float = 0.2


# ---------------------------------------------------------------------------
# Torch model definition.
# ---------------------------------------------------------------------------


def _build_model(d_feat: int, n_instruments: int, cfg: MultiTaskConfig):
    """Build the torch nn.Module per methodology spec"""
    torch = _torch()
    nn = torch.nn

    class _MultiTaskNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(n_instruments, cfg.embed_dim)
            layers = []
            in_dim = d_feat + cfg.embed_dim
            for w in cfg.hidden_widths:
                layers += [nn.Linear(in_dim, w), nn.ReLU(), nn.Dropout(cfg.dropout)]
                in_dim = w
            layers += [nn.Linear(in_dim, cfg.shared_dim)]
            self.encoder = nn.Sequential(*layers)
            self.heads = nn.ModuleList(
                [nn.Linear(cfg.shared_dim, 1) for _ in range(n_instruments)]
            )

        def forward(self, x, inst_idx):  # x: (N, d_feat), inst_idx: (N,) int
            e = self.embed(inst_idx)
            h = self.encoder(torch.cat([x, e], dim=1))  # (N, shared_dim)
            out = torch.empty(x.size(0), 1, device=x.device, dtype=x.dtype)
            # Only the matching head contributes — vectorised across heads.
            for i in range(n_instruments):
                mask = (inst_idx == i)
                if mask.any():
                    out[mask] = self.heads[i](h[mask])
            return out  # logits (N, 1)

    return _MultiTaskNN()


# ---------------------------------------------------------------------------
# Public estimator — sklearn-style interface, integrated with our
# MetaClassifier convention (fit / predict_act_proba).
# ---------------------------------------------------------------------------


class MultiTaskMetaClassifier:
    """Multi-task NN that fits / predicts via the standard MetaClassifier shape.

    Unlike :class:`stml.experimental.models.MetaClassifier`, this one is
    instrument-aware: ``fit`` accepts an ``instrument_ids`` array (int 0..K-1)
    aligned to X. The user maps ticker strings to ints once at the pipeline
    level via ``instrument_to_id``.

    Sample weights are applied per row (weighted BCE).
    Early stopping uses a held-out val fraction from the training rows
    (chronological split — last ``val_frac`` of rows by t-order kept as val).
    """

    name = "multitask_nn"

    def __init__(self, *, n_instruments: int, config: MultiTaskConfig | None = None) -> None:
        self.n_instruments = n_instruments
        self.config = config or MultiTaskConfig()
        self._model = None
        self._d_feat: int | None = None
        self._feature_names: list[str] | None = None
        # Imputation / scaling parameters fit on train.
        self._impute_means: np.ndarray | None = None
        self._scaler_mean: np.ndarray | None = None
        self._scaler_std: np.ndarray | None = None
        self._best_val_loss: float = float("inf")
        self._best_state = None
        self._epoch_at_best: int = 0
        # Class membership tracked for predict_act_proba shape.
        self.classes_ = np.array([0, 1])

    # ---- preprocessing -----------------------------------------------------

    def _preprocess_X(self, X: pd.DataFrame | np.ndarray, fit: bool = False) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if fit and self._feature_names is None:
                self._feature_names = list(X.columns)
            arr = X.values.astype(np.float32)
        else:
            arr = np.asarray(X, dtype=np.float32)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        if fit:
            self._impute_means = np.nanmean(arr, axis=0)
            # Defensive: columns that are ALL NaN -> 0.
            self._impute_means = np.where(
                np.isfinite(self._impute_means), self._impute_means, 0.0
            )
        # Impute + standardise.
        if self._impute_means is None:
            raise RuntimeError("must call fit before transform")
        means_b = np.tile(self._impute_means, (arr.shape[0], 1))
        arr = np.where(np.isnan(arr), means_b, arr)
        if fit:
            self._scaler_mean = arr.mean(axis=0)
            self._scaler_std = arr.std(axis=0) + 1e-8
        arr = (arr - self._scaler_mean) / self._scaler_std
        return arr.astype(np.float32)

    # ---- training ----------------------------------------------------------

    def fit(
        self,
        X,
        y,
        *,
        instrument_ids: np.ndarray,
        sample_weight: np.ndarray | None = None,
        t_signal_for_split: pd.Series | None = None,
    ) -> "MultiTaskMetaClassifier":
        """Fit the multi-task NN with full-batch Adam + early stopping.

        Parameters
        ----------
        X, y : feature matrix and binary labels.
        instrument_ids : ndarray of int in [0, n_instruments-1] aligned to X.
        sample_weight : optional per-row weights (uniqueness x balanced class).
        t_signal_for_split : optional Series of timestamps; used to do a
            CHRONOLOGICAL train/val split for early stopping (last ``val_frac``
            of rows by time become val). If None, last 20% positional.
        """
        torch = _torch()

        # Seed + determinism.
        torch.manual_seed(self.config.seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=True)

        X_arr = self._preprocess_X(X, fit=True)
        y_arr = np.asarray(y, dtype=np.float32)
        inst_arr = np.asarray(instrument_ids, dtype=np.int64)
        if sample_weight is None:
            sw = np.ones(len(y_arr), dtype=np.float32)
        else:
            sw = np.asarray(sample_weight, dtype=np.float32)

        # Build chrono split for early stopping.
        n = len(y_arr)
        if t_signal_for_split is not None:
            order = np.argsort(pd.to_datetime(t_signal_for_split.values))
            X_arr = X_arr[order]
            y_arr = y_arr[order]
            inst_arr = inst_arr[order]
            sw = sw[order]
        n_val = int(self.config.val_frac * n)
        if n_val < 5:
            n_val = 0
        train_slice = slice(0, n - n_val)
        val_slice = slice(n - n_val, n)

        self._d_feat = X_arr.shape[1]
        self._model = _build_model(self._d_feat, self.n_instruments, self.config)

        Xt = torch.tensor(X_arr, dtype=torch.float32)
        yt = torch.tensor(y_arr, dtype=torch.float32)
        it = torch.tensor(inst_arr, dtype=torch.int64)
        swt = torch.tensor(sw, dtype=torch.float32)

        opt = torch.optim.Adam(
            self._model.parameters(), lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        bad_epochs = 0
        for epoch in range(self.config.n_epochs):
            self._model.train()
            opt.zero_grad()
            logits = self._model(Xt[train_slice], it[train_slice]).squeeze(-1)
            loss_per_row = bce(logits, yt[train_slice])
            loss = (loss_per_row * swt[train_slice]).sum() / swt[train_slice].sum()
            loss.backward()
            opt.step()

            if n_val > 0:
                self._model.eval()
                with torch.no_grad():
                    val_logits = self._model(Xt[val_slice], it[val_slice]).squeeze(-1)
                    val_loss_rows = bce(val_logits, yt[val_slice])
                    val_loss = float(
                        (val_loss_rows * swt[val_slice]).sum() / swt[val_slice].sum()
                    )
                if val_loss < self._best_val_loss - 1e-6:
                    self._best_val_loss = val_loss
                    self._best_state = {k: v.detach().clone() for k, v in self._model.state_dict().items()}
                    self._epoch_at_best = epoch
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.config.early_stop_patience:
                        break

        # Restore best state if early-stopped.
        if self._best_state is not None:
            self._model.load_state_dict(self._best_state)
        return self

    # ---- inference ---------------------------------------------------------

    def predict_act_proba(self, X, *, instrument_ids: np.ndarray) -> np.ndarray:
        """P(class == 1) for each row, using the row's instrument head.

        Clipped to [0.01, 0.99] to bound log-loss (adopted convention with our
        sklearn MetaClassifier).
        """
        torch = _torch()
        if self._model is None:
            raise RuntimeError("must call fit before predict")
        X_arr = self._preprocess_X(X, fit=False)
        Xt = torch.tensor(X_arr, dtype=torch.float32)
        it = torch.tensor(np.asarray(instrument_ids, dtype=np.int64), dtype=torch.int64)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(Xt, it).squeeze(-1)
            # torch 2.2 on macOS x86_64 + numpy 2.x can't call .numpy() directly.
            # .tolist() bridges through Python ints/floats, lossless for floats.
            probs = np.asarray(torch.sigmoid(logits).tolist(), dtype=np.float64)
        return np.clip(probs, 0.01, 0.99)

    def predict_proba(self, X, *, instrument_ids: np.ndarray) -> np.ndarray:
        """Two-column probability matrix [P(0), P(1)] for compatibility."""
        p = self.predict_act_proba(X, instrument_ids=instrument_ids)
        return np.column_stack([1.0 - p, p])
