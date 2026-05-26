"""concept_drift.py — regime-alignment score.

ECONOMIC INTUITION
==================
Sreeram's v5 critique identified the dominant failure mode of the meta-
model: the H1-2022 test regime differs materially from the 2020-2021
training distribution, and the model trained on the latter does not
transfer. We can quantify this drift directly as a feature: train a
small classifier to discriminate "train-era" rows from "recent" rows,
and ask it how recent the current row looks. A high regime-alignment
score means today's features look more like the recent past than the
training era; a low score means today still looks like the training
era.

For the meta-model, this is an EXPLICIT regime-confidence channel:
the downstream classifier can downweight its prediction when the
alignment score is high (signalling a regime that the labelled training
data may not represent). For the team-synthesis memo, the alignment
score is the quantification of Sreeram's v5 finding that the team
narrative requires.

CAUSALITY CONTRACT
==================
At each refit time ``t_r``, the discriminator is fit ONLY on data at
indices ``<= t_r``. For each row ``t`` in the next ``refit_every`` bars,
the prediction uses the classifier fit at ``t_r``. The per-row output
at ``t`` therefore depends only on data at indices ``<= t``. Verified
by the universal causality harness.

WARMUP WINDOW
=============
``train_end_pos + window`` rows (no output before the first viable
refit).

CITATIONS
=========
* Sugiyama, M. & Kawanabe, M. (2012) "Machine Learning in Non-Stationary
  Environments: Introduction to Covariate Shift Adaptation" — the
  classifier-based drift detector ("covariate-shift estimator") is the
  textbook approach used here, also known as discriminator-based
  importance weighting.
* Lopez de Prado (2018) Ch. 7 — regime detection as a meta-labelling
  conditioning channel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

__all__ = ["regime_alignment_score"]

_DEFAULT_WINDOW: int = 63
_DEFAULT_REFIT_EVERY: int = 21
_DEFAULT_SEED: int = 42
_MIN_RECENT: int = 10
_MAX_TRAIN_OVERSAMPLE: int = 5  # train set capped at this × recent size


def _resolve_train_end_pos(
    features_df: pd.DataFrame, train_end: pd.Timestamp | int | np.integer
) -> int:
    """Translate ``train_end`` (timestamp OR positional index) into a row
    position. Positional accepts ``int`` / ``np.integer``."""
    if isinstance(train_end, (int, np.integer)):
        pos = int(train_end)
        if pos < 0:
            raise ValueError(f"train_end positional must be >= 0, got {pos}")
        return pos
    ts = pd.Timestamp(train_end)
    return int(features_df.index.searchsorted(ts, side="left"))


def regime_alignment_score(
    features_df: pd.DataFrame,
    train_end: pd.Timestamp | int,
    *,
    window: int = _DEFAULT_WINDOW,
    refit_every: int = _DEFAULT_REFIT_EVERY,
    seed: int = _DEFAULT_SEED,
) -> pd.Series:
    """Per-row ``P(row is "recent")`` from a rolling logistic discriminator.

    Two classes are formed:
        train  = rows with positional index < ``train_end_pos``.
        recent = rows in ``[t_r - window, t_r)``      (sliding window past ``train_end``).

    At each refit time ``t_r`` ∈ ``[train_end_pos + window, n, refit_every)``,
    a ``StandardScaler + LogisticRegression`` (L2, ``random_state=seed``,
    ``max_iter=200``) is fitted on the union of the two classes. For each
    row ``t`` in ``[t_r, t_r + refit_every)``, the score is the predicted
    probability that ``t``'s feature row belongs to the recent class.

    Determinism: the subsample of the train pool at each refit uses an
    RNG seeded from ``seed × 1_000_003 + t_r``. Output is in ``[0, 1]``;
    NaN before the first viable refit, and NaN at any row where the
    feature vector contains NaN.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if refit_every < 1:
        raise ValueError(f"refit_every must be >= 1, got {refit_every}")

    features = features_df.sort_index()
    dates = features.index
    train_end_pos = _resolve_train_end_pos(features, train_end)

    n = len(features)
    out = np.full(n, np.nan, dtype=np.float64)

    if train_end_pos < _MIN_RECENT or train_end_pos + window > n:
        return pd.Series(out, index=dates, name="regime_alignment_score")

    train_pool = features.iloc[:train_end_pos].dropna()
    if len(train_pool) < _MIN_RECENT:
        return pd.Series(out, index=dates, name="regime_alignment_score")

    feature_cols = list(features.columns)

    import warnings

    for refit_pos in range(train_end_pos + window, n, refit_every):
        recent_block = features.iloc[refit_pos - window : refit_pos].dropna()
        if len(recent_block) < _MIN_RECENT:
            continue
        # Subsample the train pool so the classifier is not dominated by
        # millions of pre-train_end rows when training data is plentiful.
        sample_n = min(
            len(train_pool),
            max(_MAX_TRAIN_OVERSAMPLE * len(recent_block), _MIN_RECENT * 2),
        )
        rng = np.random.default_rng(seed * 1_000_003 + refit_pos)
        train_idx = rng.choice(len(train_pool), size=sample_n, replace=False)
        x_train_block = train_pool.iloc[train_idx]
        X = pd.concat([x_train_block, recent_block], axis=0)[feature_cols].to_numpy()
        y = np.concatenate(
            [np.zeros(sample_n, dtype=np.int64),
             np.ones(len(recent_block), dtype=np.int64)]
        )

        scaler = StandardScaler()
        try:
            X_scaled = scaler.fit_transform(X)
        except ValueError:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf = LogisticRegression(
                random_state=seed, max_iter=200, solver="lbfgs",
            )
            try:
                clf.fit(X_scaled, y)
            except ValueError:
                continue

        end_pos = min(refit_pos + refit_every, n)
        block_X = features.iloc[refit_pos:end_pos][feature_cols]
        finite_mask = block_X.notna().all(axis=1).to_numpy()
        if not finite_mask.any():
            continue
        block_arr = block_X.to_numpy()
        block_scaled = scaler.transform(block_arr)
        proba = clf.predict_proba(block_scaled)[:, 1]
        for i in range(end_pos - refit_pos):
            if finite_mask[i]:
                out[refit_pos + i] = float(proba[i])

    return pd.Series(out, index=dates, name="regime_alignment_score")


# --------------------------------------------------------------------------- #
# Causality harness registry                                                  #
# --------------------------------------------------------------------------- #
# Positional train_end so the test does not depend on the synthetic
# panel's exact date range. window + refit_every chosen so the first
# refit lands deep inside the harness's 400-row panel.
CAUSALITY_REGISTRATIONS: list[dict] = [
    {
        "name": "regime_alignment_score",
        "module": __name__,
        "func": "regime_alignment_score",
        "adapter": "drift_features",
        "kwargs": {
            "train_end": 150,
            "window": 30,
            "refit_every": 15,
            "seed": 42,
        },
        # First refit at train_end_pos + window = 150 + 30 = 180.
        "warmup": 180,
        "data_kind": "single_instrument",
    },
]
