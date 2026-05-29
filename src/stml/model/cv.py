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

from collections.abc import Iterator

import numpy as np
import pandas as pd

from stml.replication.splits import n_eff

# Columns the dev frame must carry for splitting.
REQUIRED_COLS = ("date", "instrument", "bar_pos")


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
