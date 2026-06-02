"""
cv.py
=====
Purged + embargoed **expanding walk-forward** cross-validation for the meta-model, over the
pooled multi-instrument panel.

Triple-barrier labels overlap in time: an event at ``t`` has a label window ``[t, t+h]`` that
can reach into a neighbour's window. Plain k-fold (even plain ``TimeSeriesSplit``) leaks that
overlap across the train/validation boundary and massively overstates AUC. This splitter is the
walk-forward generalisation of :func:`stml.replication.splits.embargoed_val`, adding the
label-horizon **purge** that a pure run-embargo omits:

* **Purge (width ``h``).** Drop any train event whose label window ``[t, t+h]`` reaches the
  validation block -- it shares realised future with the validation labels.
* **Embargo (width ``embargo_p90[instrument]``).** Additionally drop train events within
  ``embargo_p90`` trading bars *before* the validation block. The primary signal is
  piecewise-constant with long runs (7-33 bar p90, per ``results/instrument_scope.json``); a run
  that bridges the boundary leaks the validation regime. This is exactly the boundary gap
  :func:`embargoed_val` opens, sized per instrument.

Both are applied **per instrument on its own trading-bar axis** (ragged exchange calendars), and
combined into a single train-side gap of ``h + embargo_p90`` bars before each validation block.
Because the window is *expanding* (validation is always the future tail), only the train side of
the boundary needs the gap -- there is no train block after validation inside a fold.

Design choices (deliberate, documented):

* **Expanding (anchored), not sliding.** The development panel is small and thin per instrument;
  sliding would starve early folds. Expanding maximises training data and matches production
  ref-on-all-history. (CPCV would give more paths but is out of scope -- the untouched test set
  is the single unbiased estimate.)
* **Fold boundaries on the global date axis**, purge/embargo per instrument. The global date cut
  also handles cross-sectional leakage (F9/F11): every train row predates the validation block.
* **The official test partition is never seen here.** Only the dev (train+val) frame is split;
  :func:`stml.replication.splits.get_test` remains the one auditable test-access tripwire.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

# Columns the dev frame must carry for splitting.
REQUIRED_COLS = ("date", "instrument", "bar_pos")


def n_eff(signal_series: pd.Series | np.ndarray | list) -> int:
    """Effective sample size = number of constant-value runs in the series.

    A piecewise-constant signal yields one *independent* regime-call per run, not one per
    row: a 40-day flat stretch is a single decision, the honest denominator for any skill
    statistic. Equals ``1 + (number of adjacent value changes)``; 0 for an empty series.

    Vendored from the now-removed ``stml.replication.splits`` so the model package is
    self-contained on ``main`` (the replication subsystem is no longer part of the tree).
    """
    a = np.asarray(getattr(signal_series, "to_numpy", lambda: signal_series)())
    if a.size == 0:
        return 0
    return 1 + int(np.count_nonzero(a[1:] != a[:-1]))


class PurgedWalkForward:
    """Expanding walk-forward splitter with per-instrument purge + embargo.

    Parameters
    ----------
    n_splits : number of (train, val) folds. The dev date axis is cut into ``n_splits + 1``
        contiguous blocks; fold ``i`` validates on block ``i+1`` and trains on blocks ``0..i``.
    h : label horizon in trading bars (the purge width). Must match the ``h`` used to build the
        labels -- a larger ``h`` both widens this gap and changes the labels.
    embargo_by_instrument : mapping ``instrument -> embargo bars`` (e.g. the ``embargo_p90`` of
        ``results/instrument_scope.json``). Missing instruments fall back to ``default_embargo``.
    default_embargo : embargo used when an instrument is absent from the mapping.
    """

    def __init__(
        self,
        n_splits: int = 4,
        *,
        h: int = 5,
        embargo_by_instrument: dict[str, int] | None = None,
        default_embargo: int = 10,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if h < 1:
            raise ValueError(f"h must be >= 1, got {h}")
        self.n_splits = n_splits
        self.h = int(h)
        self.embargo_by_instrument = dict(embargo_by_instrument or {})
        self.default_embargo = int(default_embargo)

    def _date_blocks(self, dates: pd.Series) -> list[np.ndarray]:
        """Cut the sorted unique dev dates into ``n_splits + 1`` contiguous blocks."""
        uniq = np.sort(pd.unique(dates.to_numpy()))
        return np.array_split(uniq, self.n_splits + 1)

    def split(self, df: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_pos, val_pos)`` integer-position arrays into ``df`` (``.iloc``-ready).

        ``df`` must hold :data:`REQUIRED_COLS`; it must contain ONLY development rows (train+val)
        -- never the test partition.
        """
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise KeyError(f"dev frame missing required columns: {missing}")

        df = df.reset_index(drop=True)
        dates = pd.to_datetime(df["date"])
        inst = df["instrument"].to_numpy()
        bar_pos = df["bar_pos"].to_numpy()
        blocks = self._date_blocks(dates)

        for i in range(self.n_splits):
            val_dates = blocks[i + 1]
            train_dates = np.concatenate(blocks[: i + 1])

            in_val = dates.isin(val_dates).to_numpy()
            in_train = dates.isin(train_dates).to_numpy()
            keep_train = in_train.copy()

            # Per-instrument purge + embargo on the train side of this boundary.
            for g in np.unique(inst[in_val]):
                emb = self.embargo_by_instrument.get(g, self.default_embargo)
                g_val = in_val & (inst == g)
                # val-block start position on this instrument's own trading-bar axis
                pos_v = int(bar_pos[g_val].min())
                # Keep a train event iff its label window ends strictly before the embargoed
                # boundary: bar_pos + h < pos_v - emb  <=>  bar_pos < pos_v - emb - h.
                cutoff = pos_v - emb - self.h
                g_train = in_train & (inst == g)
                drop = g_train & (bar_pos >= cutoff)
                keep_train[drop] = False

            train_pos = np.flatnonzero(keep_train)
            val_pos = np.flatnonzero(in_val)
            yield train_pos, val_pos

    def fold_n_eff(self, df: pd.DataFrame) -> list[dict]:
        """Per-fold effective sample size of the validation block (independent regime-calls).

        Uses :func:`stml.replication.splits.n_eff` on each instrument's validation signal run,
        summed across instruments -- the honest denominator for a fold's AUC. Folds/cells with
        tiny ``n_eff`` should be flagged (the caller may exclude a degenerate cell from the
        averaged CV score rather than letting a 0.5 AUC pollute the mean).
        """
        df = df.reset_index(drop=True)
        out = []
        has_side = "side" in df.columns
        for fold, (tr, va) in enumerate(self.split(df)):
            sub = df.iloc[va]
            total = 0
            for g, g_sub in sub.groupby("instrument", sort=False):
                seq = g_sub.sort_values("bar_pos")["side"] if has_side else g_sub["instrument"]
                total += n_eff(seq.to_numpy())
            out.append({"fold": fold, "n_train": int(tr.size), "n_val": int(va.size),
                        "val_n_eff": int(total)})
        return out


class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation (López de Prado, AFML Ch. 12).

    The development date axis is cut into ``n_blocks`` contiguous blocks; every
    ``C(n_blocks, k_test)`` choice of ``k_test`` blocks is used once as the test set and the
    remaining blocks (purged + embargoed) as the train set. Because each block is tested in many
    combinations, every observation receives several out-of-fold predictions -- the
    **``C(n_blocks-1, k_test-1)`` backtest paths** the guide builds (e.g. ``(10, 2)`` -> 45 models
    / 9 paths; this project defaults to ``(6, 2)`` -> 15 models / 5 paths). Multiple paths give a
    *distribution* of scores per config, which is what the two-stage HP selection
    (mean-AUC then Sharpe-spread) needs and a single walk-forward path cannot provide.

    Leakage discipline is identical to :class:`PurgedWalkForward`, but applied on **both sides** of
    every (interior) test block:

    * **Purge.** A train event at bar ``p`` (label window ``[p, p+h]``) is dropped when that window
      overlaps any test event's window ``[t, t+h]`` -- i.e. when ``|p - t| <= h`` for some test bar
      ``t`` on the same instrument. Unlike the expanding walk-forward (test is always the future
      tail, so only the earlier side needs purging), CPCV test blocks are interior, so train events
      both before *and* after a test block can leak.
    * **Embargo.** Additionally drop train events within ``embargo_p90[instrument]`` bars of a test
      block on either side -- combined with the purge into a single forbidden envelope
      ``|p - t| <= h + embargo`` around every test bar ``t`` (per instrument, on its own bar axis).

    Parameters
    ----------
    n_blocks : number of contiguous date blocks (``>= 2``).
    k_test : test blocks per combination (``1 <= k_test < n_blocks``). ``(6, 2)`` is the project
        default; ``(8, 2)`` (28 models / 7 paths) is the documented robustness upgrade.
    h : label horizon in trading bars (purge width); must match the labels' ``h``.
    embargo_by_instrument : ``instrument -> embargo bars`` (the ``embargo_p90`` of
        ``results/instrument_scope.json``); missing instruments fall back to ``default_embargo``.
    default_embargo : embargo for instruments absent from the mapping.
    """

    def __init__(
        self,
        n_blocks: int = 6,
        k_test: int = 2,
        *,
        h: int = 5,
        embargo_by_instrument: dict[str, int] | None = None,
        default_embargo: int = 10,
    ) -> None:
        if n_blocks < 2:
            raise ValueError(f"n_blocks must be >= 2, got {n_blocks}")
        if not 1 <= k_test < n_blocks:
            raise ValueError(f"k_test must satisfy 1 <= k_test < n_blocks, got {k_test}")
        if h < 1:
            raise ValueError(f"h must be >= 1, got {h}")
        self.n_blocks = int(n_blocks)
        self.k_test = int(k_test)
        self.h = int(h)
        self.embargo_by_instrument = dict(embargo_by_instrument or {})
        self.default_embargo = int(default_embargo)

    def n_splits(self) -> int:
        """Number of train/test combinations = ``C(n_blocks, k_test)`` (one model each)."""
        return comb(self.n_blocks, self.k_test)

    def n_paths(self) -> int:
        """Number of backtest paths per observation = ``C(n_blocks, k_test) * k_test / n_blocks``.

        Equivalently ``C(n_blocks - 1, k_test - 1)``: each block is a test block in exactly that
        many combinations, so every observation is predicted that many times out-of-fold.
        """
        return comb(self.n_blocks - 1, self.k_test - 1)

    def _date_blocks(self, dates: pd.Series) -> list[np.ndarray]:
        uniq = np.sort(pd.unique(dates.to_numpy()))
        return np.array_split(uniq, self.n_blocks)

    def _test_combos(self) -> list[tuple[int, ...]]:
        return list(combinations(range(self.n_blocks), self.k_test))

    def split(self, df: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_pos, test_pos)`` integer-position arrays, one per block combination.

        ``df`` must hold :data:`REQUIRED_COLS` and contain ONLY development rows (train+val) --
        never the test partition. Order is deterministic (``itertools.combinations`` order).
        """
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise KeyError(f"dev frame missing required columns: {missing}")

        df = df.reset_index(drop=True)
        dates = pd.to_datetime(df["date"])
        inst = df["instrument"].to_numpy()
        bar_pos = df["bar_pos"].to_numpy()
        blocks = self._date_blocks(dates)

        for combo in self._test_combos():
            test_dates = np.concatenate([blocks[b] for b in combo])
            in_test = dates.isin(test_dates).to_numpy()
            keep_train = ~in_test

            # Per-instrument purge + embargo: forbid train bars within (h + embargo) of any test
            # bar on the same instrument's own trading-bar axis (both sides of interior blocks).
            for g in np.unique(inst[in_test]):
                emb = self.embargo_by_instrument.get(g, self.default_embargo)
                g_mask = inst == g
                test_bars = np.sort(bar_pos[in_test & g_mask])
                if test_bars.size == 0:
                    continue
                w = self.h + emb
                g_train_idx = np.flatnonzero(keep_train & g_mask)
                if g_train_idx.size == 0:
                    continue
                pv = bar_pos[g_train_idx]
                # nearest test bar to each train bar (test_bars sorted): check both neighbours
                ins = np.searchsorted(test_bars, pv)
                left = test_bars[np.clip(ins - 1, 0, test_bars.size - 1)]
                right = test_bars[np.clip(ins, 0, test_bars.size - 1)]
                dist = np.minimum(np.abs(pv - left), np.abs(pv - right))
                keep_train[g_train_idx[dist <= w]] = False

            yield np.flatnonzero(keep_train), np.flatnonzero(in_test)

    def path_assignments(self, df: pd.DataFrame) -> list[dict]:
        """Map each ``(split index, test block) -> path id`` (guide pp. 10-14).

        For each block, the combinations that test it are ranked in split order and assigned path
        ids ``0 .. n_paths-1``; reading one path id down the blocks reconstructs one contiguous
        backtest path. Returned as a flat list of ``{split, block, path}`` records.
        """
        per_block_rank: dict[int, int] = {}
        out: list[dict] = []
        for split_idx, combo in enumerate(self._test_combos()):
            for b in combo:
                path = per_block_rank.get(b, 0)
                out.append({"split": split_idx, "block": int(b), "path": int(path)})
                per_block_rank[b] = path + 1
        return out

    def oof_predict(
        self,
        model_cls: type,
        params: dict,
        X: pd.DataFrame,
        y: np.ndarray,
        df: pd.DataFrame,
        *,
        seed: int = 42,
        weight_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Out-of-fold probabilities over all CPCV splits, aligned to ``df``'s row order.

        Each row is predicted once per combination that places it in the test set; the returned
        ``oof_proba`` is the **mean** of those predictions (``NaN`` for a row never scored). This
        is the calibration source (Platt/isotonic fit on OOF pairs) and the per-config path
        distribution for two-stage selection.

        Sample-uniqueness weights are **recomputed per purged-train fold** via ``weight_fn(tr)``
        (the train row positions into ``df``) -- never sliced from a precomputed vector, which would
        leak each fold's dropped-overlap structure. A split whose **train** lacks both classes is
        skipped (a model cannot be fit); ``n_folds_scored`` counts the surviving splits.

        Returns ``(oof_proba, n_pred_per_row, n_folds_scored)``:
        ``oof_proba`` and ``n_pred_per_row`` are length-``len(df)`` arrays; ``n_folds_scored`` is
        the number of splits that produced predictions (drops when a fold is class-skipped).
        """
        df = df.reset_index(drop=True)
        X = X.reset_index(drop=True)
        y = np.asarray(y)
        n = len(df)
        proba_sum = np.zeros(n, dtype=float)
        counts = np.zeros(n, dtype=int)
        n_folds_scored = 0
        for tr, te in self.split(df):
            if np.unique(y[tr]).size < 2:
                continue
            sw = weight_fn(tr) if weight_fn is not None else None
            model = model_cls(params, seed).fit(X.iloc[tr], y[tr], sample_weight=sw)
            proba_sum[te] += model.predict_proba(X.iloc[te])
            counts[te] += 1
            n_folds_scored += 1
        oof = np.where(counts > 0, proba_sum / np.maximum(counts, 1), np.nan)
        return oof, counts, n_folds_scored
