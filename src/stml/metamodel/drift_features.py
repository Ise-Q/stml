"""
drift_features.py
=================
Family **F16 — concept-drift / regime-alignment** (TF-class, ported from the
``Harry`` branch). A rolling logistic discriminator is trained to separate
"FE-train-era" feature rows from "recent" feature rows; its predicted
probability that today looks *recent* is the regime-alignment score.

A high score means today's features look more like the recent past than the
FE-train era — an explicit regime-confidence channel a downstream model can use
to downweight predictions when the labelled training distribution may not
represent the current regime (Sugiyama & Kawanabe 2012, covariate-shift
discriminator).

Causality
---------
At each refit time ``t_r`` the discriminator is fit ONLY on rows ``<= t_r`` (the
FE-train pool vs the sliding recent window). For each row ``t`` in the following
``refit_every`` bars the score uses the classifier fit at ``t_r``, so the output
at ``t`` depends only on data ``<= t``. The per-refit train subsample is seeded
from ``seed·PRIME + t_r`` so the result is deterministic and truncation-stable.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

__all__ = ["regime_alignment_score", "DRIFT_COLUMN"]

DRIFT_COLUMN: str = "f16_regime_alignment_score"

_PER_ROW_PRIME: int = 1_000_003
_MIN_RECENT: int = 10
_MAX_TRAIN_OVERSAMPLE: int = 5


def _resolve_train_end_pos(
    features_df: pd.DataFrame, train_end: pd.Timestamp | int | np.integer
) -> int:
    """Translate ``train_end`` (timestamp OR positional index) into a row position."""
    if isinstance(train_end, (int, np.integer)):
        pos = int(train_end)
        if pos < 0:
            raise ValueError(f"train_end positional must be >= 0, got {pos}")
        return pos
    ts = pd.Timestamp(train_end)
    return int(features_df.index.searchsorted(ts, side="right"))


def regime_alignment_score(
    features_df: pd.DataFrame,
    train_end: pd.Timestamp | int,
    *,
    window: int = 63,
    refit_every: int = 21,
    seed: int = 42,
) -> pd.Series:
    """Per-row ``P(row is "recent")`` from a rolling logistic discriminator.

    Two classes are formed at each refit time ``t_r``:
        train  = rows with positional index < ``train_end_pos`` (FE-train era).
        recent = rows in ``[t_r - window, t_r)`` (the sliding recent window).

    A ``StandardScaler + LogisticRegression`` (L2, ``random_state=seed``) is fit
    on the union; for each row ``t`` in ``[t_r, t_r + refit_every)`` the score is
    the predicted probability that ``t``'s feature row is recent. Output is in
    ``[0, 1]``; NaN before the first viable refit and at any NaN feature row.

    Returns a single Series named :data:`DRIFT_COLUMN`.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if refit_every < 1:
        raise ValueError(f"refit_every must be >= 1, got {refit_every}")

    dates = features_df.sort_index().index
    n = len(dates)
    out = np.full(n, np.nan, dtype=np.float64)

    # Drop columns that are entirely NaN over this instrument's series (e.g. a
    # missing open-interest channel) so a single absent column does not force
    # the per-refit dropna to discard every row.
    features = features_df.sort_index().dropna(axis=1, how="all")
    if features.shape[1] == 0:
        return pd.Series(out, index=dates, name=DRIFT_COLUMN)

    train_end_pos = _resolve_train_end_pos(features, train_end)
    if train_end_pos < _MIN_RECENT or train_end_pos + window > n:
        return pd.Series(out, index=dates, name=DRIFT_COLUMN)

    train_pool = features.iloc[:train_end_pos].dropna()
    if len(train_pool) < _MIN_RECENT:
        return pd.Series(out, index=dates, name=DRIFT_COLUMN)

    feature_cols = list(features.columns)

    for refit_pos in range(train_end_pos + window, n, refit_every):
        recent_block = features.iloc[refit_pos - window : refit_pos].dropna()
        if len(recent_block) < _MIN_RECENT:
            continue
        sample_n = min(
            len(train_pool),
            max(_MAX_TRAIN_OVERSAMPLE * len(recent_block), _MIN_RECENT * 2),
        )
        rng = np.random.default_rng(seed * _PER_ROW_PRIME + refit_pos)
        train_idx = rng.choice(len(train_pool), size=sample_n, replace=False)
        x_train_block = train_pool.iloc[train_idx]
        X = pd.concat([x_train_block, recent_block], axis=0)[feature_cols].to_numpy()
        y = np.concatenate(
            [np.zeros(sample_n, dtype=np.int64), np.ones(len(recent_block), dtype=np.int64)]
        )

        scaler = StandardScaler()
        try:
            X_scaled = scaler.fit_transform(X)
        except ValueError:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf = LogisticRegression(random_state=seed, max_iter=200, solver="lbfgs")
            try:
                clf.fit(X_scaled, y)
            except ValueError:
                continue

        end_pos = min(refit_pos + refit_every, n)
        block_X = features.iloc[refit_pos:end_pos][feature_cols]
        finite_mask = block_X.notna().all(axis=1).to_numpy()
        if not finite_mask.any():
            continue
        # Score only the fully-finite rows (LogisticRegression rejects NaN);
        # NaN rows keep their structural NaN score.
        fin_positions = np.flatnonzero(finite_mask)
        block_scaled = scaler.transform(block_X.to_numpy()[fin_positions])
        proba = clf.predict_proba(block_scaled)[:, 1]
        for j, i in enumerate(fin_positions):
            out[refit_pos + int(i)] = float(proba[j])

    return pd.Series(out, index=dates, name=DRIFT_COLUMN)
