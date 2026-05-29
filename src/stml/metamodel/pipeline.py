"""
pipeline.py
===========
Orchestration + leakage provenance for the triple-barrier metamodel
feature-engineering layer (US-FE-007).

:class:`FeaturePipeline` is the integration keystone that wires the five built
feature modules into one fitted, persisted, tidy-long feature matrix:

* engineered E-class families (:mod:`stml.metamodel.features`) -- no fit, proven
  causal by truncation-invariance, computed once per instrument and cached;
* filtered GMM + Markov-switching regime posteriors
  (:mod:`stml.metamodel.regime_features`, F3) -- TF-class, **fit per instrument**
  on that instrument's FE-train ``(ret, vol)`` rows, then transformed causally
  over its full series;
* unsupervised latent structure (:mod:`stml.metamodel.latent`, F4) -- TF-class,
  **fit pooled-within-asset-class** on the FE-train nonzero-signal engineered
  rows of every class member, then applied **per-instrument-series**;
* cross-sectional rank / pair-correlation (:mod:`stml.metamodel.xsection`, F9) --
  E-class, computed over the whole-universe panel;
* cross-asset macro context (:mod:`stml.metamodel.macro_features`, F11) --
  TF-class, **fit on the FE-train trade-date slice** of the point-in-time
  publication-lagged macro panel, then broadcast identically to every
  instrument on each date.

Leakage contract (CONTRACT_FE Sections 0 and 3 -- the crux, graded)
-------------------------------------------------------------------
* **Fit-on-train + frozen transform.** Every fitted object (the per-instrument
  regime GMM/Markov and the per-class PCA/KMeans/AE/scaler) is fit on the
  **FE-train partition only** (``fe_train_end="2021-07-01"``). The regime bundle
  records its ``train_index`` and freezes the ``(ret, vol)`` standardization
  stats from FE-train; the latent bundle records its pooled FE-train
  ``train_index``. Nothing is refit on the full series.
* **Pooled transforms run per-instrument-series.** The class-level latent bundle
  is applied to one instrument's engineered block at a time, never to a
  concatenated multi-instrument panel (avoids the C1 cross-instrument artifact).
* **No structural-NaN ffill.** Warm-up / fit-failure / missing-input NaNs
  propagate as ``NaN`` exactly as the upstream modules leave them; the persisted
  matrix is never forward-filled or ``fillna(0)``-ed.

The transform restricts rows to the **nonzero-signal trade-days** within the
645-day released window (``signal != 0``) and tags every row with its
chronological ``partition`` (train/val/test) and the constant ``fe_train_end_date``
provenance column. The fitted bundles (``_regime``, ``_latent``, ``scope``) are
exposed for ``tests/test_features_provenance.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stml.metamodel.features import assemble_engineered
from stml.metamodel.latent import LatentBundle, fit_latent, transform_latent
from stml.metamodel.macro_features import (
    DEFAULT_MACRO_PATH,
    MacroBundle,
    assemble_macro_raw,
    fit_macro,
    transform_macro,
)
from stml.metamodel.regime_features import RegimeBundle, fit_regime, transform_regime
from stml.metamodel.scope import ASSET_CLASS_MAP, InstrumentScope, build_scope
from stml.metamodel.xsection import xsection_features
from stml.na_checks import native_returns, rolling_vol
from stml.replication.splits import chronological_split

__all__ = ["FeaturePipeline"]

# The label-input volatility window; matches features.f2_vol_20 / na_checks.
_VOL_WINDOW = 20

# Metadata columns prepended to every feature row, in canonical order.
_META_COLS = ["date", "instrument", "partition", "fe_train_end_date"]


class FeaturePipeline:
    """Fit + transform the full metamodel feature stack with leakage provenance.

    The pipeline fits the TF-class feature groups on the FE-train partition only
    and then transforms causally, producing a single tidy-long feature matrix
    restricted to nonzero-signal trade-days and tagged with chronological
    partition / FE-train-end provenance.

    Parameters
    ----------
    fe_train_end : str, default "2021-07-01"
        ISO date of the (inclusive) FE-train partition boundary. The
        per-instrument regime models are fit on ``(ret, vol)`` rows on or before
        this date; the per-class latent models are fit on the FE-train
        nonzero-signal engineered rows (FE-train = the chronological-split train
        block, which ends exactly at this date for the released window).
    seed : int, default 0
        Determinism seed threaded into the regime GMM and the latent
        KMeans / autoencoder fits.
    macro_path : str, default ``"data/additional_data.xlsx"``
        Path to the F11 cross-asset macro workbook. The macro panel is built
        and FE-train-frozen-standardized internally (the single leakage
        keystone); a missing path fails fast at the top of :meth:`fit`.

    Attributes
    ----------
    scope : dict[str, InstrumentScope]
        D5 per-instrument scope registry (built at ``fit``).
    """

    def __init__(
        self,
        fe_train_end: str = "2021-07-01",
        seed: int = 0,
        macro_path: str = DEFAULT_MACRO_PATH,
    ) -> None:
        self.fe_train_end = fe_train_end
        self.fe_train_end_ts = pd.Timestamp(fe_train_end)
        self.seed = seed
        self.macro_path = macro_path

        # Populated by fit():
        self._regime: dict[str, RegimeBundle] = {}
        self._latent: dict[str, LatentBundle] = {}
        self._macro: MacroBundle | None = None
        self._macro_raw: pd.DataFrame | None = None
        self.scope: dict[str, InstrumentScope] = {}
        self._engineered: dict[str, pd.DataFrame] = {}
        self._ret_vol: dict[str, pd.DataFrame] = {}
        self._feature_cols: list[str] = []
        self._train_dates: pd.DatetimeIndex = pd.DatetimeIndex([])
        self._val_dates: pd.DatetimeIndex = pd.DatetimeIndex([])
        self._test_dates: pd.DatetimeIndex = pd.DatetimeIndex([])
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _signal_series(signals: pd.DataFrame, instrument: str) -> pd.Series:
        """Date-indexed signal series ``s_t`` for one instrument (sorted)."""
        return signals.set_index("date")[instrument].sort_index()

    def _build_ret_vol(self, ohlcv: pd.DataFrame, instrument: str) -> pd.DataFrame:
        """Date-indexed ``(ret, vol)`` on the instrument's FULL dense series.

        Returns (log) and rolling vol both reuse :mod:`stml.na_checks`, so a
        return spanning a holiday is the correct multi-day move and the 20-day
        vol uses real history rather than a truncated window. Rows with NaN
        ``ret`` or ``vol`` are dropped so the regime fit/transform sees a finite
        ``(ret, vol)`` series (structural warm-up NaNs are not fabricated).
        """
        inst = ohlcv[ohlcv["instrument"] == instrument]
        rets = native_returns(inst, kind="log")
        ret = rets.set_index("date")["ret"].sort_index()
        vol = rolling_vol(rets, instrument, window=_VOL_WINDOW)
        return pd.DataFrame({"ret": ret, "vol": vol}).dropna().sort_index()

    def _instruments(self, signals: pd.DataFrame) -> list[str]:
        """Universe members present in ``signals`` and the D5 class map."""
        return [
            c
            for c in signals.columns
            if c != "date" and c in ASSET_CLASS_MAP
        ]

    # ------------------------------------------------------------------ #
    # fit                                                                 #
    # ------------------------------------------------------------------ #
    def fit(self, ohlcv: pd.DataFrame, signals: pd.DataFrame) -> "FeaturePipeline":
        """Fit the TF-class feature groups on the FE-train partition.

        The chronological split of the released signal dates fixes the
        train / val / test date sets; the FE-train block ends at
        ``fe_train_end``. For each instrument the regime models are fit on its
        FE-train ``(ret, vol)`` rows; for each asset class the latent stack is
        fit on the pooled FE-train nonzero-signal engineered rows of the class
        members. The D5 scope and the per-instrument engineered features are
        also computed and cached here.

        Parameters
        ----------
        ohlcv : pd.DataFrame
            Long OHLCV for the whole universe (full history).
        signals : pd.DataFrame
            Wide signal panel: a ``date`` column plus one column per instrument
            with values in ``{-1, 0, +1}`` over the released window.

        Returns
        -------
        FeaturePipeline
            ``self`` (fitted), so calls can be chained with :meth:`transform`.

        Raises
        ------
        FileNotFoundError
            If ``macro_path`` does not resolve to a file (raised before any
            per-instrument work so the F11 dependency fails fast, not late as a
            silently all-NaN macro family).
        """
        # Fail-fast: the F11 macro workbook must resolve BEFORE any per-instrument
        # work (M3) -- otherwise a missing path surfaces late or all-NaN.
        macro_file = Path(self.macro_path)
        if not macro_file.is_file():
            raise FileNotFoundError(
                f"FeaturePipeline.fit: macro workbook not found at "
                f"{macro_file.resolve()}"
            )

        instruments = self._instruments(signals)

        # Chronological split of the released signal dates: train ends at
        # fe_train_end, val/test follow. These date sets drive the partition
        # labels and the FE-train fit window.
        split = chronological_split(signals["date"])
        self._train_dates = pd.DatetimeIndex(split.train_dates)
        self._val_dates = pd.DatetimeIndex(split.val_dates)
        self._test_dates = pd.DatetimeIndex(split.test_dates)
        train_date_set = set(self._train_dates)

        # D5 scope registry (n_eff gate per instrument, fit-scope policy).
        self.scope = build_scope(signals, ohlcv)

        # Per-instrument: cache (ret, vol) + engineered features, fit regime.
        for inst in instruments:
            sig = self._signal_series(signals, inst)
            ret_vol = self._build_ret_vol(ohlcv, inst)
            self._ret_vol[inst] = ret_vol

            ohlcv_inst = ohlcv[ohlcv["instrument"] == inst]
            engineered = assemble_engineered(ohlcv_inst, sig)
            self._engineered[inst] = engineered

            # FE-train (ret, vol) rows: dates <= fe_train_end.
            train_ret_vol = ret_vol[ret_vol.index <= self.fe_train_end_ts]
            n_eff_gate = (
                self.scope[inst].n_eff_gate if inst in self.scope else -1
            )
            self._regime[inst] = fit_regime(
                train_ret_vol,
                seed=self.seed,
                instrument=inst,
                n_eff_gate=n_eff_gate,
            )

        # Frozen engineered feature-column order (shared across instruments).
        self._feature_cols = list(self._engineered[instruments[0]].columns)

        # Per asset class: pool the FE-train nonzero-signal engineered rows of
        # the class members into one block, then fit the latent stack on it.
        for asset_class in sorted({ASSET_CLASS_MAP[i] for i in instruments}):
            members = [i for i in instruments if ASSET_CLASS_MAP[i] == asset_class]
            blocks: list[pd.DataFrame] = []
            for inst in members:
                sig = self._signal_series(signals, inst)
                eng = self._engineered[inst]
                # FE-train nonzero-signal days for this instrument.
                nz_train_dates = sig.index[
                    (sig != 0) & (sig.index.isin(train_date_set))
                ]
                block = eng.reindex(nz_train_dates)[self._feature_cols]
                blocks.append(block)

            pooled = pd.concat(blocks).sort_index()
            pooled.attrs["asset_class"] = asset_class
            self._latent[asset_class] = fit_latent(pooled, k=4, seed=self.seed)

        # F11 cross-asset macro context. Build the PIT-applied + momentum panel
        # on the nonzero-signal trade-date union (the released-window dates that
        # appear in the matrix), then freeze the z-score on the FE-train slice
        # only -- a single leakage boundary. The momentum-warm-up buffer lives
        # inside assemble_macro_raw and is dropped here by the trade-date slice,
        # so it never enters the frozen stats. The frame is one row per trade
        # date (NOT the 11x instrument-stacked panel), so the stats are not
        # instrument-inflated.
        nz_union: set[pd.Timestamp] = set()
        for inst in instruments:
            sig = self._signal_series(signals, inst)
            nz_union.update(sig.index[sig != 0])
        all_trade_dates = pd.DatetimeIndex(sorted(nz_union))
        self._macro_raw = assemble_macro_raw(all_trade_dates, self.macro_path)
        raw_train = self._macro_raw[self._macro_raw.index <= self.fe_train_end_ts]
        self._macro = fit_macro(raw_train)

        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    # transform                                                           #
    # ------------------------------------------------------------------ #
    def _partition_for(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Map each date to its chronological partition label.

        Dates in the FE-train block are ``"train"``, the validation block
        ``"val"``, the test block ``"test"``. A date outside all three sets
        (should not occur for released-window nonzero-signal rows) is labelled
        ``""``.
        """
        train_set = set(self._train_dates)
        val_set = set(self._val_dates)
        test_set = set(self._test_dates)
        labels = np.empty(len(index), dtype=object)
        for i, d in enumerate(index):
            if d in train_set:
                labels[i] = "train"
            elif d in val_set:
                labels[i] = "val"
            elif d in test_set:
                labels[i] = "test"
            else:
                labels[i] = ""
        return labels

    def transform(self, ohlcv: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """Assemble the tidy-long feature matrix (nonzero-signal trade-days).

        For each instrument the cached engineered block is joined (on its own
        date index) with the causal regime transform, the per-instrument latent
        transform (its class bundle applied to its own engineered block -- never
        a concatenated panel), and the cross-sectional features. Rows are then
        restricted to the **nonzero-signal trade-days** within the released
        window and tagged with ``instrument``, ``partition`` and the constant
        ``fe_train_end_date`` provenance column. Per-instrument frames are
        concatenated into one tidy-long matrix.

        Parameters
        ----------
        ohlcv : pd.DataFrame
            Long OHLCV for the whole universe (full history); the
            cross-sectional features rank over this panel.
        signals : pd.DataFrame
            The wide signal panel (same object passed to :meth:`fit`).

        Returns
        -------
        pd.DataFrame
            Tidy-long matrix with column order
            ``["date", "instrument", "partition", "fe_train_end_date", <features...>]``.
            ``df.attrs["fe_train_end_date"]`` is set to ``fe_train_end``.
            Structural NaNs are preserved (never forward-filled).

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if not self._fitted:
            raise RuntimeError("FeaturePipeline.transform called before fit().")

        instruments = self._instruments(signals)
        per_inst_frames: list[pd.DataFrame] = []

        # F11 macro is GLOBAL: build the FE-train-frozen-standardized panel once
        # (date-indexed) and broadcast it identically into every instrument's
        # join, so the same macro values land on every instrument on a date.
        macro_std = transform_macro(self._macro, self._macro_raw)

        for inst in instruments:
            sig = self._signal_series(signals, inst)
            engineered = self._engineered.get(inst)
            if engineered is None:
                # Defensive: an instrument seen at transform but not at fit.
                ohlcv_inst = ohlcv[ohlcv["instrument"] == inst]
                engineered = assemble_engineered(ohlcv_inst, sig)
            ret_vol = self._ret_vol.get(inst)
            if ret_vol is None:
                ret_vol = self._build_ret_vol(ohlcv, inst)

            # F3: causal regime posteriors over the FULL per-instrument series.
            regime = transform_regime(self._regime[inst], ret_vol)

            # F4: per-instrument latent transform with the class bundle.
            asset_class = ASSET_CLASS_MAP[inst]
            latent = transform_latent(
                self._latent[asset_class], engineered[self._feature_cols]
            )

            # F9: cross-sectional features over the whole-universe panel.
            xsect = xsection_features(ohlcv, inst)

            # Outer-join every block on this instrument's dates. The engineered
            # frame carries the instrument's full price+signal date union; the
            # other blocks (incl. the global macro panel) align onto it
            # (structural NaN where absent).
            joined = engineered.join([regime, latent, xsect, macro_std], how="outer")
            joined = joined.sort_index()

            # Restrict to nonzero-signal trade-days within the released window.
            nz_dates = sig.index[sig != 0]
            joined = joined.reindex(pd.DatetimeIndex(nz_dates)).sort_index()

            # Provenance / identity columns.
            joined.insert(0, "instrument", inst)
            joined.insert(1, "partition", self._partition_for(joined.index))
            joined.insert(2, "fe_train_end_date", self.fe_train_end)
            per_inst_frames.append(joined)

        out = pd.concat(per_inst_frames).reset_index()
        out = out.rename(columns={"index": "date"})
        if "date" not in out.columns and out.columns[0] != "date":
            out = out.rename(columns={out.columns[0]: "date"})

        feature_cols = [c for c in out.columns if c not in _META_COLS]
        out = out[_META_COLS + feature_cols]
        out.attrs["fe_train_end_date"] = self.fe_train_end
        return out.reset_index(drop=True)
