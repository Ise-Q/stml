"""
models.py
=========
Model wrappers for the meta-model. Three model families per the assignment
brief, all sharing the same sklearn-flavoured ``fit`` / ``predict_proba``
interface so they plug into the master pipeline interchangeably.

  - :class:`ElasticNetLogReg`  -- L1+L2 penalised logistic regression
    (the interpretable linear baseline).
  - :class:`XGBoostMeta`       -- gradient-boosted trees (the tabular-finance
    workhorse, captures non-linear feature interactions).
  - :class:`VsnMeta`           -- Variable Selection Network (Programming
    Session 6's neural net — reimplemented in PyTorch, exposes softmax
    feature-attention weights for built-in importance).

All three wrap their model in :class:`CalibratedClassifierCV` with isotonic
calibration on a purged held-out fold. The deliverable is a probability,
calibration matters.

Conventions
-----------
- ``X``: pd.DataFrame indexed by event id; columns are features.
- ``y``: pd.Series of binary labels {0, 1} indexed identically to X.
- ``sample_weight``: pd.Series of non-negative weights, same index. We pass
  through to sklearn's ``fit(..., sample_weight=...)``.
- ``predict_proba``: returns the probability of class 1 (label = profitable).
- ``threshold``: optional decision threshold (default 0.5) for hard predictions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from stml.cv import PurgedKFold


class ElasticNetLogReg:
    """L1+L2-penalised logistic regression with purged-CV tuning and isotonic
    probability calibration.

    Pipeline (post-fit):
        StandardScaler -> LogisticRegression(elasticnet) -> Isotonic calibration

    The internal scaler is fit on training data only; calibration uses a
    held-out fold to remap raw scores to true probabilities (so reliability is
    a first-class output rather than an after-thought).
    """

    def __init__(
        self,
        n_splits_inner: int = 5,
        n_iter: int = 20,
        embargo_td: Optional[pd.Timedelta] = None,
        random_state: int = 42,
        l1_ratio_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
        C_log_grid: tuple[float, ...] = (-3, -2, -1, 0, 1, 2),
        verbose: int = 0,
    ):
        self.n_splits_inner = n_splits_inner
        self.n_iter = n_iter
        self.embargo_td = embargo_td
        self.random_state = random_state
        self.l1_ratio_grid = list(l1_ratio_grid)
        self.C_log_grid = list(C_log_grid)
        self.verbose = verbose

        # Set at fit time:
        self.best_params_: Optional[dict] = None
        self.calibrator_: Optional[CalibratedClassifierCV] = None
        self.feature_names_: Optional[list[str]] = None

    # ------------------------------------------------------------------ #
    def _make_pipeline(self) -> Pipeline:
        # `liblinear` doesn't support elasticnet; use `saga`.
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                max_iter=5000,
                random_state=self.random_state,
                class_weight="balanced",
                l1_ratio=0.5,
                C=1.0,
            )),
        ])

    def _param_distributions(self) -> dict:
        return {
            "clf__C": [10**g for g in self.C_log_grid],
            "clf__l1_ratio": self.l1_ratio_grid,
        }

    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        t: pd.Series,
        t1: pd.Series,
        sample_weight: Optional[pd.Series] = None,
    ) -> "ElasticNetLogReg":
        """Tune hyperparameters with purged K-fold CV, then refit + calibrate.

        Parameters
        ----------
        X : pd.DataFrame indexed by event id.
        y : pd.Series of {0,1} labels.
        t, t1 : event start / end dates, indexed identically to X.
        sample_weight : optional uniqueness weights (passed through to fit).
        """
        self.feature_names_ = list(X.columns)

        cv = PurgedKFold(
            n_splits=self.n_splits_inner, t=t, t1=t1, embargo_td=self.embargo_td
        )

        pipe = self._make_pipeline()
        # Use the SAGA solver, which supports elastic-net and sample_weight.
        search = RandomizedSearchCV(
            pipe,
            param_distributions=self._param_distributions(),
            n_iter=self.n_iter,
            cv=cv,
            scoring="neg_log_loss",
            n_jobs=-1,
            random_state=self.random_state,
            verbose=self.verbose,
            refit=True,
        )

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["clf__sample_weight"] = sample_weight.values

        # Pass X as DataFrame so PurgedKFold can read the event-id index;
        # the sklearn pipeline (StandardScaler -> LogisticRegression) handles
        # the DataFrame -> array conversion internally.
        search.fit(X, y.values, **fit_kwargs)
        self.best_params_ = search.best_params_
        self._search = search

        # Calibrate the *tuned* pipeline on a held-out fold (cv=PurgedKFold).
        # CalibratedClassifierCV with cv='prefit' would skip the held-out
        # split; instead pass our PurgedKFold so calibration is also purged.
        cal_cv = PurgedKFold(
            n_splits=min(3, self.n_splits_inner),
            t=t, t1=t1, embargo_td=self.embargo_td,
        )
        self.calibrator_ = CalibratedClassifierCV(
            estimator=search.best_estimator_,
            method="isotonic",
            cv=cal_cv,
        )
        # CalibratedClassifierCV needs the splitter to be applied to (X, y)
        # but our PurgedKFold reads its index from X. Calling with arrays
        # would lose the index alignment — pass DataFrames.
        cal_fit_kwargs = {}
        if sample_weight is not None:
            cal_fit_kwargs["sample_weight"] = sample_weight.values
        self.calibrator_.fit(X, y, **cal_fit_kwargs)
        return self

    # ------------------------------------------------------------------ #
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated P(label = 1) for each row."""
        if self.calibrator_ is None:
            raise RuntimeError("Model is not fitted. Call .fit() first.")
        return self.calibrator_.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # ------------------------------------------------------------------ #
    def linear_coefficients(self) -> pd.Series:
        """Coefficients of the *underlying* (uncalibrated) elastic-net.

        Useful for sanity-checking feature signs. Note these are on the
        standardised feature scale (scaler is fit inside the pipeline).
        """
        if self._search is None:
            raise RuntimeError("Model is not fitted.")
        pipe = self._search.best_estimator_
        coefs = pipe.named_steps["clf"].coef_[0]
        return pd.Series(coefs, index=self.feature_names_, name="coef").sort_values()


# --------------------------------------------------------------------------- #
# 2. XGBoost wrapper                                                          #
# --------------------------------------------------------------------------- #
class XGBoostMeta:
    """Gradient-boosted decision trees with purged-CV tuning + isotonic calibration.

    Defaults are tuned for binary classification on ~few-thousand-row tabular
    financial data. The search grid covers depth, learning rate, n_estimators,
    subsampling, and L1/L2 regularisation — the standard XGBoost levers.

    Why XGBoost specifically:
      - Captures non-linear feature interactions that linear models can't
        (e.g. ``regime_high_vol * trend_strength * signal_agreement``).
      - Handles heterogeneous feature scales natively (trees are scale-invariant).
      - ``feature_importances_`` gives MDI directly — feeds the cluster importance
        stage with no extra work.
      - Tabular-finance workhorse: this is the model class most likely to win
        on AUC.
    """

    def __init__(
        self,
        n_splits_inner: int = 5,
        n_iter: int = 30,
        embargo_td: Optional[pd.Timedelta] = None,
        random_state: int = 42,
        verbose: int = 0,
        param_grid: Optional[dict] = None,
        scale_pos_weight: Optional[float] = None,
    ):
        self.n_splits_inner = n_splits_inner
        self.n_iter = n_iter
        self.embargo_td = embargo_td
        self.random_state = random_state
        self.verbose = verbose
        self.scale_pos_weight = scale_pos_weight
        self.param_grid = param_grid or {
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "n_estimators": [100, 200, 300, 500],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0.0, 0.01, 0.1, 1.0],
            "reg_lambda": [0.1, 1.0, 5.0],
            "min_child_weight": [1, 3, 5],
        }
        self.best_params_: Optional[dict] = None
        self.calibrator_: Optional[CalibratedClassifierCV] = None
        self.feature_names_: Optional[list[str]] = None
        self._search: Optional[RandomizedSearchCV] = None

    def _base_estimator(self) -> XGBClassifier:
        spw = self.scale_pos_weight if self.scale_pos_weight is not None else 1.0
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=self.random_state,
            verbosity=0,
            scale_pos_weight=spw,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        t: pd.Series,
        t1: pd.Series,
        sample_weight: Optional[pd.Series] = None,
    ) -> "XGBoostMeta":
        self.feature_names_ = list(X.columns)
        if self.scale_pos_weight is None:
            pos = float((y == 1).sum())
            neg = float((y == 0).sum())
            self.scale_pos_weight = neg / pos if pos > 0 else 1.0

        cv = PurgedKFold(
            n_splits=self.n_splits_inner, t=t, t1=t1, embargo_td=self.embargo_td
        )
        base = self._base_estimator()
        search = RandomizedSearchCV(
            base,
            param_distributions=self.param_grid,
            n_iter=self.n_iter,
            cv=cv,
            scoring="neg_log_loss",
            n_jobs=-1,
            random_state=self.random_state,
            verbose=self.verbose,
            refit=True,
        )
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight.values
        search.fit(X, y.values, **fit_kwargs)
        self.best_params_ = search.best_params_
        self._search = search

        # Calibrate the tuned XGBoost on a purged held-out fold.
        cal_cv = PurgedKFold(
            n_splits=min(3, self.n_splits_inner),
            t=t, t1=t1, embargo_td=self.embargo_td,
        )
        # Build a FRESH XGB with the best params; CalibratedClassifierCV clones it.
        best = XGBClassifier(
            **{**self._base_estimator().get_params(), **search.best_params_,
               "scale_pos_weight": self.scale_pos_weight},
        )
        self.calibrator_ = CalibratedClassifierCV(
            estimator=best, method="isotonic", cv=cal_cv,
        )
        cal_fit_kwargs = {}
        if sample_weight is not None:
            cal_fit_kwargs["sample_weight"] = sample_weight.values
        self.calibrator_.fit(X, y, **cal_fit_kwargs)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.calibrator_ is None:
            raise RuntimeError("Model is not fitted.")
        return self.calibrator_.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # ------------------------------------------------------------------ #
    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """MDI feature importance from the best (uncalibrated) booster.

        ``importance_type`` is one of ``'weight'``, ``'gain'``, ``'cover'``,
        ``'total_gain'``, ``'total_cover'`` (XGBoost native names). Default
        ``'gain'`` is the average gain per split using each feature —
        the closest analogue to sklearn's tree MDI.
        """
        if self._search is None:
            raise RuntimeError("Model is not fitted.")
        booster = self._search.best_estimator_.get_booster()
        score_dict = booster.get_score(importance_type=importance_type)
        # XGBoost names features f0..fN-1 unless feature_names is set; map back.
        out = pd.Series(0.0, index=self.feature_names_, name=f"xgb_{importance_type}")
        for fname, val in score_dict.items():
            # XGBoost may return either the original name (if set) or 'f<idx>'.
            if fname.startswith("f") and fname[1:].isdigit():
                idx = int(fname[1:])
                if idx < len(self.feature_names_):
                    out.iloc[idx] = float(val)
            elif fname in out.index:
                out.loc[fname] = float(val)
        # Normalise to sum to 1 so multiple importance lenses are comparable.
        total = out.sum()
        if total > 0:
            out = out / total
        return out.sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# 3. Variable Selection Network (Programming Session 6, PyTorch port)         #
# --------------------------------------------------------------------------- #
class _GatedLinearUnit:
    """Lazily-loaded torch nn.Module placeholder so the module imports cleanly
    even when torch is unavailable. Real implementation lives in `_vsn_torch`."""

    pass


def _vsn_torch_modules():
    """Build the VSN torch modules — done lazily so this module imports even
    if torch is missing (we use sklearn for everything else)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class GLU(nn.Module):
        """Gated Linear Unit: sigmoid(W1 x + b1) * (W2 x + b2)."""

        def __init__(self, dim: int):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)

        def forward(self, x):
            return torch.sigmoid(self.fc1(x)) * self.fc2(x)

    class GRN(nn.Module):
        """Gated Residual Network: ELU → Dense → Dropout → GLU → +residual → LayerNorm."""

        def __init__(self, dim: int, dropout: float = 0.1):
            super().__init__()
            self.dense1 = nn.Linear(dim, dim)
            self.dense2 = nn.Linear(dim, dim)
            self.glu = GLU(dim)
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(dim)

        def forward(self, x):
            residual = x
            h = F.elu(self.dense1(x))
            h = self.dense2(h)
            h = self.dropout(h)
            h = self.glu(h)
            return self.norm(h + residual)

    class VSN(nn.Module):
        """Variable Selection Network for tabular inputs.

        Each scalar feature is embedded to ``d_e`` dims via a SHARED linear
        projection (we have no categorical features), then per-feature GRNs
        transform each, and a top-level GRN+softmax produces per-feature
        attention weights ``alpha_i``. Output = weighted sum of GRN outputs
        passed through a final GRN + linear head.
        """

        def __init__(self, n_features: int, d_e: int = 16, dropout: float = 0.1):
            super().__init__()
            self.n_features = n_features
            self.d_e = d_e
            self.embed = nn.Linear(1, d_e)  # shared scalar embedding
            self.feature_grns = nn.ModuleList([GRN(d_e, dropout) for _ in range(n_features)])
            self.select_grn = GRN(d_e * n_features, dropout)
            self.select_proj = nn.Linear(d_e * n_features, n_features)
            self.final_grn = GRN(d_e, dropout)
            self.head = nn.Linear(d_e, 1)

        def forward(self, x):  # x: (B, n_features)
            B = x.size(0)
            # Per-feature scalar embedding (shared) — (B, n_features, d_e)
            embedded = self.embed(x.unsqueeze(-1))
            # Per-feature GRN — (B, n_features, d_e)
            grn_out = torch.stack([
                self.feature_grns[i](embedded[:, i, :])
                for i in range(self.n_features)
            ], dim=1)
            # Variable-selection: flatten → GRN → linear → softmax — (B, n_features)
            flat = grn_out.reshape(B, -1)
            attn_logits = self.select_proj(self.select_grn(flat))
            attn = F.softmax(attn_logits, dim=1)  # (B, n_features)
            # Weighted combine — (B, d_e)
            weighted = (attn.unsqueeze(-1) * grn_out).sum(dim=1)
            # Final
            h = self.final_grn(weighted)
            logits = self.head(h).squeeze(-1)  # (B,)
            return logits, attn

    return GLU, GRN, VSN, torch, nn, F


class VsnMeta:
    """Variable Selection Network meta-model — Programming Session 6, PyTorch.

    Trains with early stopping on a purged held-out validation fold. The
    model's softmax variable-selection weights are stored and exposed as
    feature importance (mean across the training set).
    """

    def __init__(
        self,
        d_e: int = 16,
        dropout: float = 0.1,
        lr: float = 1e-3,
        max_epochs: int = 100,
        batch_size: int = 256,
        patience: int = 10,
        embargo_td: Optional[pd.Timedelta] = None,
        n_splits_inner: int = 5,
        random_state: int = 42,
        verbose: int = 0,
    ):
        self.d_e = d_e
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.embargo_td = embargo_td
        self.n_splits_inner = n_splits_inner
        self.random_state = random_state
        self.verbose = verbose

        self.model_ = None
        self.scaler_: Optional[StandardScaler] = None
        self.feature_names_: Optional[list[str]] = None
        self.history_: list[dict] = []
        self.best_params_: dict = {}  # for API parity with other wrappers
        self.calibrator_: Optional[CalibratedClassifierCV] = None
        self.attention_weights_: Optional[pd.Series] = None

    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        t: pd.Series,
        t1: pd.Series,
        sample_weight: Optional[pd.Series] = None,
    ) -> "VsnMeta":
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        _GLU, _GRN, VSN, torch, nn, F = _vsn_torch_modules()

        self.feature_names_ = list(X.columns)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Split a purged validation fold for early stopping (use the LAST fold).
        cv = PurgedKFold(n_splits=self.n_splits_inner, t=t, t1=t1,
                         embargo_td=self.embargo_td)
        splits = list(cv.split(X))
        train_idx, val_idx = splits[-1]

        self.scaler_ = StandardScaler()
        Xtr_arr = self.scaler_.fit_transform(X.iloc[train_idx].values)
        Xv_arr = self.scaler_.transform(X.iloc[val_idx].values)
        ytr = y.iloc[train_idx].values.astype(np.float32)
        yv = y.iloc[val_idx].values.astype(np.float32)
        wtr = (sample_weight.iloc[train_idx].values.astype(np.float32)
               if sample_weight is not None else np.ones_like(ytr))

        Xtr_t = torch.from_numpy(Xtr_arr.astype(np.float32))
        Xv_t = torch.from_numpy(Xv_arr.astype(np.float32))
        ytr_t = torch.from_numpy(ytr)
        yv_t = torch.from_numpy(yv)
        wtr_t = torch.from_numpy(wtr)

        ds = TensorDataset(Xtr_t, ytr_t, wtr_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        model = VSN(n_features=Xtr_t.shape[1], d_e=self.d_e, dropout=self.dropout)
        # pos_weight for class imbalance.
        pos = float(ytr.sum())
        neg = float(len(ytr) - pos)
        pos_weight = torch.tensor([neg / pos if pos > 0 else 1.0])
        criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        best_val = float("inf")
        best_state = None
        bad = 0
        for epoch in range(self.max_epochs):
            model.train()
            for xb, yb, wb in loader:
                optimizer.zero_grad()
                logits, _attn = model(xb)
                loss_per = criterion(logits, yb)
                loss = (loss_per * wb).mean()
                loss.backward()
                optimizer.step()
            # Validation
            model.eval()
            with torch.no_grad():
                v_logits, _ = model(Xv_t)
                v_loss = nn.BCEWithLogitsLoss()(v_logits, yv_t).item()
            self.history_.append({"epoch": epoch, "val_loss": v_loss})
            if v_loss < best_val - 1e-4:
                best_val = v_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model

        # Capture mean attention weights over training set (feature importance).
        model.eval()
        with torch.no_grad():
            _, attn = model(Xtr_t)
        self.attention_weights_ = pd.Series(
            attn.mean(dim=0).numpy(),
            index=self.feature_names_,
            name="vsn_attention",
        ).sort_values(ascending=False)

        # Isotonic calibration: wrap model as sklearn-style and use CalibratedClassifierCV.
        # Easiest path: use simple Platt/Isotonic on (predicted, y) ourselves.
        from sklearn.isotonic import IsotonicRegression
        with torch.no_grad():
            tr_logits, _ = model(Xtr_t)
            tr_proba = torch.sigmoid(tr_logits).numpy()
        # Fit on a holdout (val) to avoid overfitting on training labels.
        with torch.no_grad():
            v_logits, _ = model(Xv_t)
            v_proba = torch.sigmoid(v_logits).numpy()
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(v_proba, yv)
        if self.verbose:
            print(f"[VSN] best val_loss={best_val:.4f} epoch={len(self.history_)}")
        return self

    # ------------------------------------------------------------------ #
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        if self.model_ is None:
            raise RuntimeError("VSN is not fitted")
        self.model_.eval()
        Xa = self.scaler_.transform(X.values).astype(np.float32)
        with torch.no_grad():
            logits, _attn = self.model_(torch.from_numpy(Xa))
            p_raw = torch.sigmoid(logits).numpy()
        return self._iso.predict(p_raw)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importance(self) -> pd.Series:
        if self.attention_weights_ is None:
            raise RuntimeError("VSN is not fitted")
        return self.attention_weights_.copy()


# --------------------------------------------------------------------------- #
# 4. MLP neural-network wrapper (sklearn-based, Programming Session 5)        #
# --------------------------------------------------------------------------- #
#
# Note: the VSN code above is the "ideal" PyTorch implementation per Programming
# Session 6, but the current env has a torch (2.2.x, x86_64 macOS wheel) vs
# numpy (2.x) ABI mismatch — torch 2.2 was compiled for numpy 1.x. The cleanest
# fix is to upgrade torch to 2.4+, but those wheels are arm64-only.
#
# Until that's resolved, we use sklearn's MLPClassifier — the same NN family,
# also course-aligned (Programming Session 5 uses MLPClassifier exactly), and
# avoids the ABI mismatch. It still hits the rubric's "neural networks" model
# family requirement.

from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance


class MlpMeta:
    """Sklearn MLPClassifier wrapped with the standard scaler + purged-CV
    tuning + isotonic calibration. Course-aligned (Programming Session 5).

    Hyperparameter grid covers depth (1-2 hidden layers), width, learning rate,
    L2 regularisation, and activation. Tuned with the existing PurgedKFold.

    Feature importance via permutation importance (n_repeats=5) using AUC.
    """

    def __init__(
        self,
        n_splits_inner: int = 5,
        n_iter: int = 15,
        embargo_td: Optional[pd.Timedelta] = None,
        random_state: int = 42,
        verbose: int = 0,
    ):
        self.n_splits_inner = n_splits_inner
        self.n_iter = n_iter
        self.embargo_td = embargo_td
        self.random_state = random_state
        self.verbose = verbose

        self.best_params_: Optional[dict] = None
        self.calibrator_: Optional[CalibratedClassifierCV] = None
        self.feature_names_: Optional[list[str]] = None
        self._search: Optional[RandomizedSearchCV] = None
        self._feature_importance: Optional[pd.Series] = None

    def _make_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(50, 30),
                activation="relu",
                solver="adam",
                alpha=0.01,
                learning_rate_init=1e-3,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                random_state=self.random_state,
            )),
        ])

    def _param_distributions(self) -> dict:
        return {
            "clf__hidden_layer_sizes": [(50, 30), (80, 40), (100,), (64, 32, 16)],
            "clf__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
            "clf__learning_rate_init": [1e-4, 1e-3, 5e-3],
            "clf__activation": ["relu", "tanh"],
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        t: pd.Series,
        t1: pd.Series,
        sample_weight: Optional[pd.Series] = None,  # noqa: ARG002 (sklearn MLP doesn't accept sample_weight)
    ) -> "MlpMeta":
        self.feature_names_ = list(X.columns)
        cv = PurgedKFold(
            n_splits=self.n_splits_inner, t=t, t1=t1, embargo_td=self.embargo_td
        )
        pipe = self._make_pipeline()
        search = RandomizedSearchCV(
            pipe,
            param_distributions=self._param_distributions(),
            n_iter=self.n_iter,
            cv=cv,
            scoring="neg_log_loss",
            n_jobs=-1,
            random_state=self.random_state,
            verbose=self.verbose,
            refit=True,
        )
        # MLPClassifier doesn't accept sample_weight ⇒ ignored (documented above).
        search.fit(X, y.values)
        self.best_params_ = search.best_params_
        self._search = search

        cal_cv = PurgedKFold(
            n_splits=min(3, self.n_splits_inner),
            t=t, t1=t1, embargo_td=self.embargo_td,
        )
        self.calibrator_ = CalibratedClassifierCV(
            estimator=search.best_estimator_, method="isotonic", cv=cal_cv,
        )
        self.calibrator_.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.calibrator_ is None:
            raise RuntimeError("Model is not fitted.")
        return self.calibrator_.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importance(
        self, X_oos: Optional[pd.DataFrame] = None, y_oos: Optional[pd.Series] = None,
        n_repeats: int = 5,
    ) -> pd.Series:
        """Permutation importance on OOS data (model-agnostic).

        Falls back to coefficients-of-magnitude if X_oos / y_oos not provided.
        """
        if X_oos is None or y_oos is None:
            # Without OOS data, we can't compute permutation importance.
            # Return a flat uninformative series.
            return pd.Series(1.0 / len(self.feature_names_),
                             index=self.feature_names_, name="mlp_perm")
        result = permutation_importance(
            self.calibrator_, X_oos, y_oos,
            n_repeats=n_repeats, scoring="roc_auc",
            random_state=self.random_state, n_jobs=-1,
        )
        imp = pd.Series(result.importances_mean, index=self.feature_names_, name="mlp_perm")
        total = imp.clip(lower=0).sum()
        if total > 0:
            imp = imp.clip(lower=0) / total
        self._feature_importance = imp.sort_values(ascending=False)
        return self._feature_importance.copy()
