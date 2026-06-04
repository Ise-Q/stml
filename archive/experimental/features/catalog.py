"""Feature catalog — the registry every feature family writes to.

Plan §3.3, §8 Stage 2, R1 (one version).

Every feature is a small pure function ``f(panel, ctx) -> pd.Series`` that takes
a :class:`FeatureContext` (the per-instrument frame, the cleaned BBG parquets,
and the asset-class membership map) and returns a Series indexed by the
instrument's trading-day calendar.

Each feature self-registers a :class:`FeatureSpec` at module load via
:func:`register`, so the assembler can iterate without inspecting the call
sites. The registry is **frozen at import time**; adding a feature requires a
new module-level ``register(...)`` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from stml.experimental.config import ASSET_CLASS_MEMBERS, INSTRUMENT_TO_CLASS


# ---------------------------------------------------------------------------
# Types.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureContext:
    """Per-instrument inputs handed to each feature function.

    Attributes
    ----------
    instrument
        Instrument ticker (e.g. ``'cl1s'``).
    asset_class
        ``'equity'`` | ``'energy'`` | ``'metals'``.
    frame
        DataFrame indexed by trading date with columns
        ``[open, high, low, close, volume, open_interest, signal]``.
    universe
        ``{inst: frame}`` dict for ALL instruments — used by cross-section
        features (F9 / F21). Caller passes the same dict to every instrument's
        feature run.
    macro_harry
        Harry's PIT-aligned macro parquet (``data/bloomberg/cleaned/macro_harry.parquet``).
    futures_term
        BBG-raw front + 2nd month parquet
        (``data/bloomberg/cleaned/futures_term.parquet``).
    options_iv
        BBG IV parquet (``data/bloomberg/cleaned/options_iv.parquet``).
    eia_crude
        BBG EIA weekly crude change parquet.
    eia_release_flag
        BBG EIA release-day binary parquet.
    """

    instrument: str
    asset_class: str
    frame: pd.DataFrame
    universe: dict[str, pd.DataFrame]
    macro_harry: pd.DataFrame
    futures_term: pd.DataFrame
    options_iv: pd.DataFrame
    eia_crude: pd.DataFrame
    eia_release_flag: pd.DataFrame


FeatureFn = Callable[[FeatureContext], pd.Series]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str  # "F1", "F2", ... "F22", "EWMA_HMM"
    fn: FeatureFn
    leakage_class: str  # "E" = pure causal / "TF" = train-fitted
    warmup_bars: int  # bars before the first valid value
    source: str  # "sreeram", "harry", "alken", "bbg", "new", "shared"


# ---------------------------------------------------------------------------
# Module-level registry.
# ---------------------------------------------------------------------------


REGISTRY: list[FeatureSpec] = []


def register(spec: FeatureSpec) -> FeatureSpec:
    """Add ``spec`` to the global registry; return it (for decorator-style use)."""
    if any(s.name == spec.name for s in REGISTRY):
        raise ValueError(f"Duplicate feature name {spec.name}")
    REGISTRY.append(spec)
    return spec


def registered_names() -> list[str]:
    return [s.name for s in REGISTRY]


def family_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in REGISTRY:
        counts[s.family] = counts.get(s.family, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Assembler.
# ---------------------------------------------------------------------------


def assemble_features(
    panel: dict[str, pd.DataFrame],
    *,
    cleaned_dir: Path | None = None,
    instruments: Iterable[str] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Build the per-event feature matrix.

    Parameters
    ----------
    panel : ``{instrument: DataFrame}`` — output of
        :func:`stml.experimental.data_loader.per_instrument_frames`.
    cleaned_dir : path to ``data/bloomberg/cleaned/`` (the parquets emitted by
        :mod:`stml.experimental.bloomberg_ingest`).
    instruments : optional subset of instruments to process.
    verbose : print per-instrument progress.

    Returns
    -------
    pd.DataFrame
        Long-form, one row per (instrument, date) with columns
        ``[instrument, date, <feature_name>, ...]``. The date index is the
        intersection of each instrument's trading calendar; missing values are
        ``NaN``.
    """
    if cleaned_dir is None:
        # Default to the repo root.
        here = Path(__file__).resolve()
        for parent in [here, *here.parents]:
            if (parent / "data").is_dir() and (parent / "pyproject.toml").is_file():
                cleaned_dir = parent / "data" / "bloomberg" / "cleaned"
                break
        else:
            raise FileNotFoundError("Could not locate repo root.")

    cleaned_dir = Path(cleaned_dir)

    def _read_parquet(name: str) -> pd.DataFrame:
        p = cleaned_dir / name
        if not p.exists():
            return pd.DataFrame()
        return pd.read_parquet(p)

    macro_harry = _read_parquet("macro_harry.parquet")
    futures_term = _read_parquet("futures_term.parquet")
    options_iv = _read_parquet("options_iv.parquet")
    eia_crude = _read_parquet("eia_crude.parquet")
    eia_release_flag = _read_parquet("eia_release_flag.parquet")

    insts = list(instruments) if instruments else list(panel.keys())
    parts = []
    for inst in insts:
        if inst not in panel:
            continue
        frame = panel[inst]
        if verbose:
            print(f"  {inst}: {len(frame)} bars")
        ctx = FeatureContext(
            instrument=inst,
            asset_class=INSTRUMENT_TO_CLASS.get(inst, "unknown"),
            frame=frame,
            universe=panel,
            macro_harry=macro_harry,
            futures_term=futures_term,
            options_iv=options_iv,
            eia_crude=eia_crude,
            eia_release_flag=eia_release_flag,
        )
        cols: dict[str, pd.Series] = {}
        for spec in REGISTRY:
            try:
                series = spec.fn(ctx)
                if series is None:
                    continue
                # Reindex to the instrument's trading-day index.
                cols[spec.name] = series.reindex(frame.index)
            except Exception as exc:
                if verbose:
                    print(f"    [WARN] {spec.name} failed for {inst}: {exc}")
                cols[spec.name] = pd.Series(np.nan, index=frame.index)
        inst_frame = pd.DataFrame(cols, index=frame.index)
        inst_frame.insert(0, "instrument", inst)
        inst_frame.insert(1, "date", inst_frame.index)
        parts.append(inst_frame.reset_index(drop=True))

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out


# ---------------------------------------------------------------------------
# Small numeric helpers reused across feature modules.
# ---------------------------------------------------------------------------


def expanding_zscore(s: pd.Series, min_periods: int = 60) -> pd.Series:
    """Per-instrument **causal expanding-window z-score**.

    The mean / std at bar ``t`` use only ``s.iloc[:t+1]`` — truncation invariant.
    """
    mean = s.expanding(min_periods=min_periods).mean()
    std = s.expanding(min_periods=min_periods).std()
    return (s - mean) / std.replace(0, np.nan)


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mean = s.rolling(window=window, min_periods=window).mean()
    std = s.rolling(window=window, min_periods=window).std()
    return (s - mean) / std.replace(0, np.nan)


def rolling_rank(s: pd.Series, window: int) -> pd.Series:
    """Rolling rank in [0, 1] over the window."""
    return s.rolling(window=window, min_periods=window).rank(pct=True)


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def safe_divide(num: pd.Series, denom: pd.Series, fill: float = np.nan) -> pd.Series:
    out = num.astype(float) / denom.replace(0, np.nan).astype(float)
    return out.fillna(fill) if not np.isnan(fill) else out
