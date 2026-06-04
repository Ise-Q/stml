"""Purged + combinatorial CV for triple-barrier labels.

Plan §3.4 / §8 Stage 3 deliverable.

Three splitters:

* :class:`PurgedKFold` — sklearn-compatible k-fold with AFML Ch.7 purging on
  `t1` (first-touch time) + embargo.
* :class:`CombinatorialPurgedCV` — N groups, k test groups → C(N,k) purged
  paths. Default `(6, 2)` → 15 paths.
* :func:`nested_cpcv` — outer CPCV (selection-bias evaluator) over an inner
  CPCV (hyperparameter tuning) — gives selection-bias-aware OOS scores.

Plus **per-instrument embargo** (methodology spec): the standard 1 %
forward embargo under-covers thin instruments whose label windows can stretch
to ~33 business days on the pooled calendar. With an `embargo_days`
``{inst: int}`` map, each test block's forward embargo is applied per
instrument on each instrument's own date axis.

References
----------
López de Prado, M. (2018). *Advances in Financial Machine Learning*, Ch.7
    (purged k-fold) and Ch.12 (combinatorial purged CV).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Iterator

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def embargo_size(n: int, pct: float = 0.01) -> int:
    """Number of bars to embargo after each test block — the AFML 1 % rule."""
    return int(np.ceil(pct * n))


def _instrument_date_axes(
    t: pd.Series, instruments: pd.Series | None
) -> dict[str, pd.DatetimeIndex]:
    """Build ``{inst: sorted unique trading dates}`` for per-instrument embargo."""
    if instruments is None:
        return {}
    df = pd.DataFrame({"t": pd.to_datetime(t.values), "inst": instruments.values})
    axes: dict[str, pd.DatetimeIndex] = {}
    for inst in df["inst"].dropna().unique():
        mask = df["inst"] == inst
        ts = pd.DatetimeIndex(df.loc[mask, "t"].sort_values().unique())
        axes[inst] = ts
    return axes


def _advance_inst_days(axis: pd.DatetimeIndex, anchor: pd.Timestamp, days: int) -> pd.Timestamp:
    """Advance ``anchor`` by ``days`` bars on ``axis``; clamp to the end."""
    if len(axis) == 0:
        return anchor
    pos = int(axis.searchsorted(anchor, side="right")) - 1
    pos = max(pos, 0)
    target = min(pos + days, len(axis) - 1)
    return axis[target]


# ---------------------------------------------------------------------------
# PurgedKFold.
# ---------------------------------------------------------------------------


@dataclass
class _PurgeConfig:
    """Internal — args to the purge / embargo step."""

    t: pd.Series
    t1: pd.Series
    instruments: pd.Series | None
    embargo_days: dict[str, int] | None
    pct_embargo: float


def _purge_train(
    cfg: _PurgeConfig, train_idx: np.ndarray, test_idx: np.ndarray
) -> np.ndarray:
    """Drop train events whose `[t, t1]` overlaps the test block + embargo."""
    if len(test_idx) == 0:
        return train_idx
    t = pd.to_datetime(cfg.t.values)
    t1 = pd.to_datetime(cfg.t1.values)
    insts = cfg.instruments.values if cfg.instruments is not None else None

    test_t = t[test_idx]
    test_t1 = t1[test_idx]
    b_t0 = test_t.min()
    b_t1_max = test_t1.max()

    # Step 1 — span-overlap purge (cross-instrument): drop any train event
    # whose [t, t1] intersects [b_t0, b_t1_max].
    train_t = t[train_idx]
    train_t1 = t1[train_idx]
    keep_mask = (train_t1 < b_t0) | (train_t > b_t1_max)
    purged = train_idx[keep_mask]

    # Step 2 — forward embargo. Two paths:
    if cfg.embargo_days and insts is not None and len(cfg.embargo_days) > 0:
        # Per-instrument embargo on each instrument's own axis (methodology spec).
        axes = _instrument_date_axes(cfg.t, cfg.instruments)
        # For each instrument in the TEST block, find its own max t1; then
        # advance embargo_days on that instrument's axis; drop train events
        # of the SAME instrument starting within that window.
        test_insts = pd.Series(insts[test_idx]).dropna().unique()
        embargo_windows: dict[str, pd.Timestamp] = {}
        for inst in test_insts:
            mask_test = (insts[test_idx] == inst)
            if not mask_test.any():
                continue
            inst_max_t1 = pd.Timestamp(test_t1[mask_test].max())
            days = int(cfg.embargo_days.get(inst, 0))
            if days <= 0:
                continue
            axis = axes.get(inst, pd.DatetimeIndex([]))
            embargo_end = _advance_inst_days(axis, inst_max_t1, days)
            embargo_windows[inst] = embargo_end
        # Drop train events of the same instrument whose t falls in [max_t1, embargo_end].
        drop_mask = np.zeros(len(purged), dtype=bool)
        purged_t = t[purged]
        purged_insts = insts[purged]
        for inst, embargo_end in embargo_windows.items():
            mask_test = (insts[test_idx] == inst)
            inst_max_t1 = pd.Timestamp(test_t1[mask_test].max())
            drop_mask |= (
                (purged_insts == inst)
                & (purged_t > inst_max_t1)
                & (purged_t <= embargo_end)
            )
        purged = purged[~drop_mask]
    else:
        # Uniform 1 % embargo (AFML default).
        emb = embargo_size(len(t), cfg.pct_embargo)
        if emb > 0:
            last_test_idx = int(test_idx[-1])
            embargo_start_idx = last_test_idx + 1
            embargo_end_idx = min(embargo_start_idx + emb, len(t))
            purged = purged[~((purged >= embargo_start_idx) & (purged < embargo_end_idx))]

    return purged


class PurgedKFold:
    """Sklearn-compatible purged k-fold splitter (AFML Ch.7).

    Parameters
    ----------
    n_splits
        Number of folds. Default 5.
    t : ``pd.Series`` of event observation times (one per row of X).
    t1 : ``pd.Series`` of first-touch (label resolution) times.
    pct_embargo
        Forward embargo as a fraction of total sample length (1 % default).
    instruments
        Optional ``pd.Series`` of instrument tickers per row.
    embargo_days
        Optional ``{inst: int}`` map of per-instrument embargo days. When
        present, overrides the flat ``pct_embargo``.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        t: pd.Series | None = None,
        t1: pd.Series | None = None,
        pct_embargo: float = 0.01,
        instruments: pd.Series | None = None,
        embargo_days: dict[str, int] | None = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.t = t
        self.t1 = t1
        self.pct_embargo = pct_embargo
        self.instruments = instruments
        self.embargo_days = embargo_days

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: ARG002
        return self.n_splits

    def split(
        self, X, y=None, groups=None  # noqa: ARG002
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if self.t is None or self.t1 is None:
            raise ValueError("t and t1 are required for PurgedKFold")
        n = len(X)
        if n != len(self.t) or n != len(self.t1):
            raise ValueError("len(X), len(t), len(t1) must match")
        fold_size = n // self.n_splits
        cfg = _PurgeConfig(
            t=self.t,
            t1=self.t1,
            instruments=self.instruments,
            embargo_days=self.embargo_days,
            pct_embargo=self.pct_embargo,
        )
        for k in range(self.n_splits):
            start = k * fold_size
            end = (k + 1) * fold_size if k < self.n_splits - 1 else n
            test_idx = np.arange(start, end)
            train_idx = np.concatenate([np.arange(0, start), np.arange(end, n)])
            train_idx = _purge_train(cfg, train_idx, test_idx)
            yield train_idx, test_idx


# ---------------------------------------------------------------------------
# CombinatorialPurgedCV.
# ---------------------------------------------------------------------------


class CombinatorialPurgedCV:
    """AFML Ch.12 — N groups, k test groups, C(N,k) purged paths.

    Defaults ``(n_groups=6, n_test_groups=2)`` → 15 paths.

    The returned splits are over groups of contiguous events (sorted by `t`),
    not single events. Each `combinations(range(N), k)` choice becomes one
    test set; the remaining `(N - k)` groups become the train set, then
    purged + embargoed per :class:`PurgedKFold`.
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        *,
        t: pd.Series | None = None,
        t1: pd.Series | None = None,
        pct_embargo: float = 0.01,
        instruments: pd.Series | None = None,
        embargo_days: dict[str, int] | None = None,
    ) -> None:
        if n_groups < 2 or n_test_groups < 1 or n_test_groups >= n_groups:
            raise ValueError("require 1 <= n_test_groups < n_groups")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.t = t
        self.t1 = t1
        self.pct_embargo = pct_embargo
        self.instruments = instruments
        self.embargo_days = embargo_days

    @property
    def n_paths(self) -> int:
        return comb(self.n_groups, self.n_test_groups)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: ARG002
        return self.n_paths

    def split(
        self, X, y=None, groups=None  # noqa: ARG002
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if self.t is None or self.t1 is None:
            raise ValueError("t and t1 are required for CombinatorialPurgedCV")
        n = len(X)
        # Sort indices by t for deterministic group assignment.
        order = np.argsort(pd.to_datetime(self.t.values).astype("datetime64[ns]"))
        group_size = n // self.n_groups
        groups_idx = [
            order[k * group_size : (k + 1) * group_size if k < self.n_groups - 1 else n]
            for k in range(self.n_groups)
        ]
        cfg = _PurgeConfig(
            t=self.t,
            t1=self.t1,
            instruments=self.instruments,
            embargo_days=self.embargo_days,
            pct_embargo=self.pct_embargo,
        )
        for combo in combinations(range(self.n_groups), self.n_test_groups):
            test_idx = np.concatenate([groups_idx[g] for g in combo])
            train_idx = np.concatenate([
                groups_idx[g] for g in range(self.n_groups) if g not in combo
            ])
            test_idx = np.sort(test_idx)
            train_idx = np.sort(train_idx)
            train_idx = _purge_train(cfg, train_idx, test_idx)
            yield train_idx, test_idx


# ---------------------------------------------------------------------------
# Nested CPCV — selection-bias-aware evaluator.
# ---------------------------------------------------------------------------


def nested_cpcv(
    X: pd.DataFrame,
    *,
    t: pd.Series,
    t1: pd.Series,
    outer_groups: int = 6,
    outer_test_groups: int = 2,
    inner_groups: int = 5,
    inner_test_groups: int = 1,
    pct_embargo: float = 0.01,
    instruments: pd.Series | None = None,
    embargo_days: dict[str, int] | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, "CombinatorialPurgedCV"]]:
    """Yield ``(outer_train, outer_test, inner_cv)`` tuples.

    The inner CV is built from the OUTER-TRAIN rows ONLY (no tuning leakage —
    the inner splits never see the outer test fold). Use the inner CV for
    hyperparameter selection, refit on outer_train, score on outer_test.
    """
    outer = CombinatorialPurgedCV(
        n_groups=outer_groups,
        n_test_groups=outer_test_groups,
        t=t,
        t1=t1,
        pct_embargo=pct_embargo,
        instruments=instruments,
        embargo_days=embargo_days,
    )
    for outer_train_idx, outer_test_idx in outer.split(X):
        # Inner CV uses only outer-train indices.
        inner_t = t.iloc[outer_train_idx].reset_index(drop=True)
        inner_t1 = t1.iloc[outer_train_idx].reset_index(drop=True)
        inner_insts = (
            instruments.iloc[outer_train_idx].reset_index(drop=True)
            if instruments is not None
            else None
        )
        inner_cv = CombinatorialPurgedCV(
            n_groups=inner_groups,
            n_test_groups=inner_test_groups,
            t=inner_t,
            t1=inner_t1,
            pct_embargo=pct_embargo,
            instruments=inner_insts,
            embargo_days=embargo_days,
        )
        yield outer_train_idx, outer_test_idx, inner_cv


# ---------------------------------------------------------------------------
# Defensive leakage assertion (used by tests).
# ---------------------------------------------------------------------------


def assert_no_leakage(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    t: pd.Series,
    t1: pd.Series,
) -> None:
    """Raise AssertionError if any train event's [t, t1] overlaps test [t, t1]."""
    t_arr = pd.to_datetime(t.values)
    t1_arr = pd.to_datetime(t1.values)
    test_t = t_arr[test_idx]
    test_t1 = t1_arr[test_idx]
    train_t = t_arr[train_idx]
    train_t1 = t1_arr[train_idx]
    b_t0, b_t1 = test_t.min(), test_t1.max()
    bad = (train_t1 >= b_t0) & (train_t <= b_t1)
    if bad.any():
        n = int(bad.sum())
        raise AssertionError(
            f"PurgedKFold leak: {n} train events overlap test [{b_t0}, {b_t1}]"
        )
