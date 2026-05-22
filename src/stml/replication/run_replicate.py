"""
run_replicate.py
================
Replication-search orchestrator (US-011) for the primary-signal study. This is
a thin CLI that *drives* the already-built, already-tested replication modules
and renders their numbers -- it reimplements nothing.

Run it::

    python -m stml.replication.run_replicate                 # all families x cells
    python -m stml.replication.run_replicate --families mean_reversion

For every ``(family, cell)`` it:

1. builds the **target** signal and aligned next-day returns via
   :func:`stml.replication.align.align_instrument` (``convention='next_day'``);
2. wires a ``generate_fn(params) -> replica`` that reindexes the archetype's
   signal onto the target index (a standalone instrument uses
   :meth:`Archetype.generate`; a pool generates each member with the SAME params
   and CONCATENATES target/replica/return/index masks across members so the
   search and gates operate on the pooled series);
3. searches the params on the TRAIN objective via
   :func:`stml.replication.search.search_cell` (TPE above the n_eff FLOOR, an
   exhaustive grid below it), recording every trial to the
   :class:`stml.replication.ledger.Ledger`;
4. builds a ``+/-1 step`` G3 neighbourhood for the winner and recomputes the VAL
   composite (:func:`stml.replication.search.composite_skill`) per neighbour;
5. gates the winner via :func:`stml.replication.gates.evaluate` (standalone) or
   :func:`stml.replication.gates.gate_cell` (pool, ``low_power=True``), recording
   the :class:`stml.replication.gates.GateResult` to the ledger.

A family "passes" if it clears G1-G4 on >= 1 cell. After all families x cells,
if >= 5 families pass the study is a success; otherwise the summary carries an
**honest-shortfall** section explaining why (effective-n, drift, degeneracy, the
expected-negative cross-asset diagnostic for ``xsect_rank``).

Artifacts written (under the repo root by default; tests pass ``out_dir``):

* ``reports/<archetype>.md`` per family -- definition, param space + tier,
  per-cell train+val panel WITH n_eff + per-split base rates, the G3
  perturbation spread, and the G1-G4 verdicts.
* ``reports/replication-summary.md`` -- the families x cells pass/fail matrix,
  the headline, the honest-shortfall section if < 5 pass, and the val<->test
  final-confirmation consistency.
* ``reports/replication-ledger.md`` -- :meth:`Ledger.render_markdown`.
* ``results/jj/top_candidates.json`` -- per passing ``(family, cell)`` the
  archetype/cell/params/tier/n_eff/val_metrics/gate booleans (single best cell
  per passing family).

Final test confirmation
-----------------------
After the search + gates are frozen, the top passing ``(family, cell, params)``
per family (capped at five) is confirmed ONCE on the held-out test block via
:func:`stml.replication.splits.get_test` with ``final_confirmation=True``. That
is the ONLY ``get_test(final_confirmation=True)`` call in the whole pipeline; the
search never touches test.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from stml.replication import gates, search
from stml.replication.align import align_instrument
from stml.replication.archetypes import ARCHETYPES, Archetype
from stml.replication.ledger import Ledger
from stml.replication.metrics import panel
from stml.replication.splits import (
    Split,
    chronological_split,
    embargoed_val,
    get_test,
    n_eff,
)

# --------------------------------------------------------------------------- #
# Cells + families (LOCKED per CONTRACT2.md / the C1 checkpoint).             #
# --------------------------------------------------------------------------- #
# Standalone cells all sit at post-embargo val n_eff >= FLOOR; the three energy
# members below the FLOOR are gated only on their asset-class POOL.
STANDALONE_CELLS: tuple[str, ...] = (
    "es1s",
    "nq1s",
    "fesx1s",
    "rb1s",
    "gc1s",
    "si1s",
    "hg1s",
    "pl1s",
)
POOL_NAME: str = "pool:energy"
POOL_MEMBERS: tuple[str, ...] = ("cl1s", "ho1s", "ng1s")
POOL_CLASS: str = "energy"

CLASS_OF: dict[str, str] = {
    "es1s": "equity",
    "nq1s": "equity",
    "fesx1s": "equity",
    "cl1s": "energy",
    "ho1s": "energy",
    "rb1s": "energy",
    "ng1s": "energy",
    "gc1s": "metals",
    "si1s": "metals",
    "hg1s": "metals",
    "pl1s": "metals",
}

# Per-instrument allowed signal alphabet (LOCKED C1 fact): ng1s is
# participation-only short (never +1); every other instrument is unrestricted.
ALLOWED_OF: dict[str, tuple[int, ...]] = {"ng1s": (-1, 0)}

# Iterate MEAN_REVERSION first -- the C1 prior-best replicator (the signal is
# counter-trend) -- then the remaining families in registry order.
FAMILY_ORDER: tuple[str, ...] = (
    "mean_reversion",
    "ts_momentum",
    "breakout_donchian",
    "vol_regime_gated",
    "hybrid_filtered_momentum",
    "xsect_rank",
)

# A family is a "replication success" for the study when it passes G1-G4 on at
# least one cell; the study as a whole succeeds when this many families do.
FAMILIES_REQUIRED: int = 5

# Cap on the number of families taken into the one-shot final test confirmation.
MAX_FINAL_FAMILIES: int = 5

# Where the artifacts land (resolved relative to the repo root).
_SUMMARY_REL = Path("reports/replication-summary.md")
_TOP_CANDIDATES_REL = Path("results/jj/top_candidates.json")
_LEDGER_REL = Path("results/jj/ledger.json")


# --------------------------------------------------------------------------- #
# Repo / IO helpers (mirror run_characterize).                                #
# --------------------------------------------------------------------------- #
def _repo_root() -> Path:
    """Walk up from this file until ``data/`` + ``pyproject.toml`` are found."""
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "data").is_dir() and (p / "pyproject.toml").is_file():
            return p
    raise FileNotFoundError(
        f"could not locate stml repo root (data/ + pyproject.toml) from {here}"
    )


def _load_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Thin indirection over :func:`stml.io.load_clean_data` (kept local)."""
    from stml.io import load_clean_data

    return load_clean_data()


def _fmt(x: float | int | None, nd: int = 3) -> str:
    """Compact fixed-point string; ``nan`` / ``None`` render as ``n/a``."""
    if x is None:
        return "n/a"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(xf):
        return "n/a"
    return f"{xf:.{nd}f}"


def _fmt_bool(x: object) -> str:
    """Render a bool gate verdict (``yes`` / ``no``), else ``n/a``."""
    if isinstance(x, (bool, np.bool_)):
        return "yes" if bool(x) else "no"
    return "n/a"


# --------------------------------------------------------------------------- #
# Per-split base rates (mirrors run_characterize / drift reporting).          #
# --------------------------------------------------------------------------- #
def _base_rates(labels: np.ndarray) -> dict[str, float]:
    """Participation / long-bias / per-state fractions of a label vector.

    ``participation`` is the nonzero fraction; ``long_bias`` is ``P(+1) -
    P(-1)``; the three ``frac_*`` are the class shares. An empty vector yields
    ``nan`` everywhere so a thin (embargoed-away) window reports honestly rather
    than dividing by zero. This is the per-split chance reference G2 is built to
    respect (the released base rate DRIFTS across splits per C1).
    """
    n = int(labels.size)
    if n == 0:
        nan = float("nan")
        return {
            "n": 0,
            "participation": nan,
            "long_bias": nan,
            "frac_neg1": nan,
            "frac_0": nan,
            "frac_pos1": nan,
        }
    neg = float((labels == -1).sum()) / n
    zero = float((labels == 0).sum()) / n
    pos = float((labels == 1).sum()) / n
    return {
        "n": n,
        "participation": float((labels != 0).sum()) / n,
        "long_bias": pos - neg,
        "frac_neg1": neg,
        "frac_0": zero,
        "frac_pos1": pos,
    }


# --------------------------------------------------------------------------- #
# Cell construction: target / aligned_ret / replica generator on a common idx.#
# --------------------------------------------------------------------------- #
class CellData:
    """Everything one cell needs to search + gate, all on a single date index.

    ``target`` (the released signal), ``aligned_ret`` and any replica are
    positionally comparable because they share ``index``; ``train_idx`` /
    ``val_idx`` are positions into that index. ``split`` is the original
    645-day :class:`Split` (gates intersect by date, so they need the full
    date axis, not the positional one).
    """

    def __init__(
        self,
        *,
        cell: str,
        target: pd.Series,
        aligned_ret: pd.Series,
        generate_fn: Callable[[dict], pd.Series],
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        n_eff_val: int,
        split: Split,
        thresholds_entry: dict,
        base_rates: dict[str, dict[str, float]],
        groups: np.ndarray | None = None,
    ) -> None:
        self.cell = cell
        self.target = target
        self.aligned_ret = aligned_ret
        self.generate_fn = generate_fn
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.n_eff_val = n_eff_val
        self.split = split
        self.thresholds_entry = thresholds_entry
        self.base_rates = base_rates
        # None for standalone; for a pool, a per-row member id aligned to the
        # synthetic index so search + gates measure WITHIN-instrument skill (the
        # mean of per-member metrics, never a single concatenated metric).
        self.groups = groups


def _released_signal(signals: pd.DataFrame, instrument: str) -> pd.Series:
    """Date-indexed released signal for one instrument (ascending, int)."""
    return (
        signals[["date", instrument]]
        .set_index("date")[instrument]
        .sort_index()
        .astype(int)
    )


def _post_embargo_n_eff(signals: pd.DataFrame, instrument: str, split: Split) -> int:
    """Post-embargo val ``n_eff`` for one instrument (the gateable count).

    Mirrors :func:`stml.replication.run_characterize._pooling_map`:
    ``n_eff(signal.iloc[embargoed_val(signal, split)])`` on the released signal.
    """
    sig = _released_signal(signals, instrument)
    return int(n_eff(sig.iloc[embargoed_val(sig, split)]))


def _positional_idx(
    index: pd.DatetimeIndex, dates: pd.DatetimeIndex
) -> np.ndarray:
    """Positions in ``index`` of the dates also present in ``dates`` (sorted).

    Alignment drops the rows with no defined next-day return, so the cell index
    is a subset of the full date axis; the split's train/val *dates* are mapped
    onto positions into that subset here, keeping the search objective on the
    leakage-free train slice.
    """
    common = index.intersection(dates)
    pos = pd.Series(np.arange(len(index)), index=index)
    return pos.loc[common].to_numpy()


def _instrument_signal(
    archetype: Archetype,
    instrument: str,
    ohlcv: pd.DataFrame,
    params: dict,
) -> pd.Series:
    """One instrument's ``{-1, 0, +1}`` signal for either family kind.

    A per-instrument family runs :meth:`Archetype.generate` on that instrument's
    OHLCV. A cross-sectional family (``xsect_rank``, ``score_fn is None``) is
    PANEL-level: :meth:`Archetype.generate_panel` ranks the WHOLE universe each
    day, and this instrument's resulting series is pulled out (per the contract,
    "evaluate each instrument's resulting series per cell"). Both honour the
    instrument's ``allowed`` alphabet -- ng1s stays short-only either way.
    """
    allowed = ALLOWED_OF.get(instrument, (-1, 0, 1))
    if archetype.score_fn is None:
        panel_signals = archetype.generate_panel(ohlcv, params, allowed=allowed)
        sig = panel_signals.get(instrument)
        if sig is None:
            return pd.Series(dtype=int)
        return sig
    ohlcv_inst = ohlcv[ohlcv["instrument"] == instrument]
    return archetype.generate(ohlcv_inst, params, allowed=allowed)


def _standalone_cell(
    *,
    instrument: str,
    archetype: Archetype,
    signals: pd.DataFrame,
    ohlcv: pd.DataFrame,
    split: Split,
    thresholds_entry: dict,
) -> CellData:
    """Build a :class:`CellData` for one standalone instrument.

    The target signal and aligned next-day return come from
    :func:`align_instrument`; the replica generator builds this instrument's
    signal via :func:`_instrument_signal` (per-instrument ``generate`` or the
    panel-level ``generate_panel`` slice for ``xsect_rank``) and reindexes onto
    the target index (a warm-up NaN -> flat).
    """
    aligned = align_instrument(signals, ohlcv, instrument, convention="next_day")
    frame = aligned.frame.set_index("date").sort_index()
    target = frame["signal"].astype(int)
    aligned_ret = frame["ret"].astype(float)
    index = pd.DatetimeIndex(target.index)

    def generate_fn(params: dict) -> pd.Series:
        sig = _instrument_signal(archetype, instrument, ohlcv, params)
        return sig.reindex(index).fillna(0).astype(int)

    train_idx = _positional_idx(index, split.train_dates)
    val_idx = _positional_idx(index, split.val_dates)
    base_rates = _split_base_rates(target, split)
    return CellData(
        cell=instrument,
        target=target,
        aligned_ret=aligned_ret,
        generate_fn=generate_fn,
        train_idx=train_idx,
        val_idx=val_idx,
        n_eff_val=_post_embargo_n_eff(signals, instrument, split),
        split=split,
        thresholds_entry=thresholds_entry,
        base_rates=base_rates,
    )


def _pool_cell(
    *,
    archetype: Archetype,
    signals: pd.DataFrame,
    ohlcv: pd.DataFrame,
    split: Split,
    thresholds_entry: dict,
) -> CellData:
    """Build a :class:`CellData` for the energy POOL by member concatenation.

    Each member is aligned independently (``next_day``); the per-member target,
    aligned return and the train/val *positional* masks are CONCATENATED so the
    search and gates operate on the pooled series exactly as the contract
    requires. The pool's ``generate_fn`` applies the SAME params to every member
    (respecting each member's ``allowed`` alphabet -- ng1s stays short-only) and
    concatenates the reindexed replicas in the same member order, so the pooled
    replica is positionally aligned to the pooled target.

    The concatenated series carry a synthetic monotone integer index, and the
    train/val masks are positions into it (per-member positions shifted by a
    running offset so members never collide). The pooled :class:`Split` stores
    those positional integers as its ``*_dates``, so the gates' date-based
    intersection -- ``index.intersection(split.*_dates)`` -- selects exactly the
    pooled rows the search used.
    """
    members = list(POOL_MEMBERS)

    targets: list[pd.Series] = []
    rets: list[pd.Series] = []
    member_indices: list[pd.DatetimeIndex] = []
    train_pos: list[np.ndarray] = []
    val_pos: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    offset = 0
    n_eff_sum = 0

    for inst in members:
        aligned = align_instrument(signals, ohlcv, inst, convention="next_day")
        frame = aligned.frame.set_index("date").sort_index()
        tgt = frame["signal"].astype(int)
        ret = frame["ret"].astype(float)
        idx = pd.DatetimeIndex(tgt.index)
        member_indices.append(idx)
        targets.append(tgt.reset_index(drop=True))
        rets.append(ret.reset_index(drop=True))
        # One group label per concatenated row (member id), in the SAME member
        # order the series are concatenated, so the search objective and gates can
        # average WITHIN-instrument metrics rather than score one concatenation.
        group_parts.append(np.full(len(idx), inst, dtype=object))
        train_pos.append(_positional_idx(idx, split.train_dates) + offset)
        val_pos.append(_positional_idx(idx, split.val_dates) + offset)
        offset += len(idx)
        # POOLED gateable n_eff = SUM of the members' post-embargo val
        # regime-calls (energy: cl1s 9 + ho1s 9 + ng1s 2 = 20). This sum is the
        # right effective sample size because the pool members are
        # near-independent: cross-asset mean |corr| ~= 0.09 (LOCKED C1 fact), so
        # the concatenated series carries ~the SUM of the members' independent
        # regime-calls rather than max/avg. That is what lifts the pool over the
        # FLOOR (>= 10) and lets gate_cell judge it first-class, even though each
        # member alone sits below the FLOOR and can never pass standalone.
        n_eff_sum += _post_embargo_n_eff(signals, inst, split)

    # A synthetic monotone integer "date" index so the concatenated series stay
    # positionally addressable AND date-sliceable by the gates without member
    # calendars colliding. The gates only need train/val membership, supplied via
    # the pooled split below.
    pooled_target = pd.concat(targets, ignore_index=True)
    pooled_ret = pd.concat(rets, ignore_index=True)
    synth_index = pd.RangeIndex(len(pooled_target))
    pooled_target.index = synth_index
    pooled_ret.index = synth_index
    # Member id per pooled row, positionally aligned to the synthetic index.
    groups = (
        np.concatenate(group_parts) if group_parts else np.empty(0, dtype=object)
    )

    train_idx = np.concatenate(train_pos) if train_pos else np.empty(0, dtype=int)
    val_idx = np.concatenate(val_pos) if val_pos else np.empty(0, dtype=int)

    # The pooled "split" carries train/val membership as positional integers
    # mapped onto the synthetic index, so gates' date-intersection selects the
    # same pooled rows the search used.
    pooled_split = Split(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=np.empty(0, dtype=int),
        train_dates=pd.Index(train_idx),
        val_dates=pd.Index(val_idx),
        test_dates=pd.Index(np.empty(0, dtype=int)),
    )

    def generate_fn(params: dict) -> pd.Series:
        parts: list[pd.Series] = []
        for inst, idx in zip(members, member_indices, strict=True):
            sig = _instrument_signal(archetype, inst, ohlcv, params)
            parts.append(sig.reindex(idx).fillna(0).astype(int).reset_index(drop=True))
        out = pd.concat(parts, ignore_index=True)
        out.index = synth_index
        return out

    base_rates = _split_base_rates_pooled(pooled_target, train_idx, val_idx)
    return CellData(
        cell=POOL_NAME,
        target=pooled_target,
        aligned_ret=pooled_ret,
        generate_fn=generate_fn,
        train_idx=train_idx,
        val_idx=val_idx,
        n_eff_val=n_eff_sum,
        split=pooled_split,
        thresholds_entry=thresholds_entry,
        base_rates=base_rates,
        groups=groups,
    )


def _split_base_rates(target: pd.Series, split: Split) -> dict[str, dict[str, float]]:
    """Per-split (train / val) base rates of a date-indexed target signal."""
    index = pd.DatetimeIndex(target.index)
    train = target.loc[index.intersection(split.train_dates)].to_numpy(dtype=int)
    val = target.loc[index.intersection(split.val_dates)].to_numpy(dtype=int)
    return {"train": _base_rates(train), "val": _base_rates(val)}


def _split_base_rates_pooled(
    target: pd.Series, train_idx: np.ndarray, val_idx: np.ndarray
) -> dict[str, dict[str, float]]:
    """Per-split base rates for a pooled target addressed by positional idx."""
    train = target.iloc[list(train_idx)].to_numpy(dtype=int)
    val = target.iloc[list(val_idx)].to_numpy(dtype=int)
    return {"train": _base_rates(train), "val": _base_rates(val)}


# --------------------------------------------------------------------------- #
# G3 perturbation neighbourhood.                                              #
# --------------------------------------------------------------------------- #
# CATEGORICAL axes name a *strategy choice*, not a tuning knob: ``base``
# (mean_reversion <-> ts_momentum) and ``score`` (momentum <-> reversal) flip the
# whole rule, and ``regime`` (high <-> low vol) flips which regime the rule even
# participates in. A G3 +/-1 step over these would compare DIFFERENT STRATEGIES,
# not perturbations of one, so they are held fixed at the winner's value. Every
# other axis (lookback, z_window, deadband, vol_window, filter_window, channel,
# vol_quantile, q_window, top_frac) is numeric/ordinal -- a grid step there is a
# genuine tuning perturbation, which is exactly the plateau G3 measures.
_CATEGORICAL_AXES: frozenset[str] = frozenset({"base", "regime", "score"})


def _is_numeric_axis(axis: str, values: list) -> bool:
    """Whether a param axis is numeric/ordinal (perturbable) vs categorical.

    An axis is categorical -- and so HELD FIXED in the G3 neighbourhood -- if it
    is named in :data:`_CATEGORICAL_AXES` (``base`` / ``regime`` / ``score``) or
    if any of its grid values is non-numeric (a robust fallback so a new
    string-valued axis is never mistaken for a tuning knob). Otherwise it is
    numeric and a +/-1 grid step is a legitimate tuning perturbation.
    """
    if axis in _CATEGORICAL_AXES:
        return False
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)


def _neighbours(best_params: dict, param_space: dict[str, list]) -> list[dict]:
    """A ``+/-1 step`` neighbourhood of ``best_params`` over the NUMERIC axes only.

    For each searched **numeric/ordinal** axis the value is moved one grid step up
    and one step down (when such a neighbour exists), holding every other axis --
    including all CATEGORICAL axes -- at the winner. The result is a plateau probe
    over *tuning*: G3 passes only if the val composite stays high and tight across
    these single-step perturbations (a robust optimum, not a spike). Duplicate or
    out-of-grid points are skipped.

    Why categorical axes are held fixed
    -----------------------------------
    ``base`` (mean_reversion <-> ts_momentum), ``score`` (momentum <-> reversal)
    and ``regime`` (high <-> low vol) are *strategy switches*, not tuning knobs:
    flipping one yields a structurally different replica, not a +/-1 perturbation
    of the same one. G3 measures **plateau-in-tuning** (is the optimum robust to a
    small parameter nudge?), so conflating a strategy switch with a perturbation
    would wrongly reject a robust tuning plateau merely because the *other*
    strategy scores differently. Categorical axes are identified via
    :func:`_is_numeric_axis` (the :data:`_CATEGORICAL_AXES` set plus a
    non-numeric-value fallback) and excluded from the step.
    """
    neighbours: list[dict] = []
    seen: set[tuple] = set()
    for axis, choices in param_space.items():
        values = list(choices)
        if not _is_numeric_axis(axis, values):
            continue  # categorical: a step here is a strategy switch, not a tune
        if axis not in best_params or best_params[axis] not in values:
            continue
        pos = values.index(best_params[axis])
        for step in (-1, 1):
            j = pos + step
            if 0 <= j < len(values):
                cand = dict(best_params)
                cand[axis] = values[j]
                key = tuple(sorted(cand.items(), key=lambda kv: kv[0]))
                if key not in seen:
                    seen.add(key)
                    neighbours.append(cand)
    return neighbours


def _perturbed_val_composites(
    cell: CellData,
    neighbours: Sequence[dict],
) -> list[float]:
    """VAL composite skill (:func:`search.composite_skill`) for each neighbour.

    Each neighbour's replica is regenerated and scored against the target on the
    val slice. An empty val slice yields no usable composite (skipped), so G3
    falls back to non-blocking when there is nothing to perturb.
    """
    val_list = list(cell.val_idx)
    if not val_list:
        return []
    tgt_val = cell.target.iloc[val_list]
    # For a pool, score each neighbour on the group-AVERAGED (within-instrument)
    # composite so G3's plateau probe matches the search objective and the gates.
    grp_val = None if cell.groups is None else np.asarray(cell.groups)[val_list]
    out: list[float] = []
    for params in neighbours:
        rep = cell.generate_fn(params)
        out.append(search.composite_skill(tgt_val, rep.iloc[val_list], grp_val))
    return out


# --------------------------------------------------------------------------- #
# Per-(family, cell) run.                                                     #
# --------------------------------------------------------------------------- #
class CellResult:
    """The frozen search + gate outcome for one ``(family, cell)``."""

    def __init__(
        self,
        *,
        family: str,
        cell: str,
        params: dict,
        tier: str,
        n_eff_val: int,
        n_configs: int,
        val_metrics: dict,
        gate_result: gates.GateResult,
        perturbed: list[float],
        base_rates: dict[str, dict[str, float]],
        pool_member_metrics: dict | None = None,
    ) -> None:
        self.family = family
        self.cell = cell
        self.params = params
        self.tier = tier
        self.n_eff_val = n_eff_val
        self.n_configs = n_configs
        self.val_metrics = val_metrics
        self.gate_result = gate_result
        self.perturbed = perturbed
        self.base_rates = base_rates
        # For a pool only: the per-member vs group-averaged-vs-concatenated val
        # kappas that document the within-instrument aggregation fix; None
        # otherwise.
        self.pool_member_metrics = pool_member_metrics

    @property
    def passed(self) -> bool:
        return bool(self.gate_result.passed)


def _composite_from_panel(vm: dict) -> float | None:
    """Composite skill from a recorded val :func:`panel` dict, or ``None``."""
    kappa = vm.get("kappa")
    osk = vm.get("ordinal_skill")
    if isinstance(osk, dict):
        osk = osk.get("vs_flat")
    if isinstance(kappa, (int, float)) and isinstance(osk, (int, float)):
        return 0.5 * (float(kappa) + float(osk))
    return None


def _pool_member_metrics(cell_data: CellData, best_replica: pd.Series) -> dict:
    """Per-member vs group-averaged vs CONCATENATED val kappa for a pool winner.

    Documents the within-instrument aggregation fix on the actual winner: for each
    pool member the val kappa is computed on that member's rows ALONE (the honest
    within-instrument number), then contrasted with their equal-weight mean (the
    gated quantity) and the single CONCATENATED-panel kappa (the old artifact that
    let cross-instrument base-rate matching manufacture a pass). Returns a dict
    ``{members: {inst: kappa}, group_avg_kappa, concatenated_kappa}``; empty val
    yields ``None`` values rather than raising.
    """
    val_list = list(cell_data.val_idx)
    tgt_val = cell_data.target.iloc[val_list]
    rep_val = best_replica.iloc[val_list]
    grp_val = np.asarray(cell_data.groups)[val_list]

    per_member: dict[str, float | None] = {}
    kappas: list[float] = []
    for inst in POOL_MEMBERS:
        mask = grp_val == inst
        if not mask.any():
            per_member[inst] = None
            continue
        k = float(panel(tgt_val[mask].to_numpy(), rep_val[mask].to_numpy())["kappa"])
        per_member[inst] = k
        kappas.append(k)
    group_avg = float(np.mean(kappas)) if kappas else None
    concat = (
        float(panel(tgt_val.to_numpy(), rep_val.to_numpy())["kappa"])
        if val_list
        else None
    )
    return {
        "members": per_member,
        "group_avg_kappa": group_avg,
        "concatenated_kappa": concat,
    }


def _run_cell(
    *,
    family: str,
    archetype: Archetype,
    cell_data: CellData,
    pool_cell_data: CellData | None,
    ledger: Ledger,
    budget: int,
    seed: int,
) -> CellResult:
    """Search + gate one ``(family, cell)`` and record the verdict to the ledger.

    The search minimises the TRAIN discrepancy (TPE above the FLOOR, grid below)
    and records every trial; the winner is then gated. A standalone cell
    (``n_eff >= FLOOR``) is gated via :func:`gates.evaluate`; the pool cell is
    routed through :func:`gates.gate_cell` so its verdict carries
    ``low_power=True`` and can never be a standalone pass for a thin member.
    """
    param_space = archetype.param_space()
    result = search.search_cell(
        archetype=family,
        cell=cell_data.cell,
        param_space=param_space,
        generate_fn=cell_data.generate_fn,
        target=cell_data.target,
        train_idx=cell_data.train_idx,
        val_idx=cell_data.val_idx,
        n_eff=cell_data.n_eff_val,
        ledger=ledger,
        budget=budget,
        seed=seed,
        groups=cell_data.groups,
    )
    best_params = result["best_params"]
    val_metrics = result["best_metrics"]
    n_configs = len(result["trials"])

    best_replica = cell_data.generate_fn(best_params)
    neighbours = _neighbours(best_params, param_space)
    perturbed = _perturbed_val_composites(cell_data, neighbours)
    perturbed_arg = perturbed if perturbed else None

    is_pool = cell_data.cell == POOL_NAME
    if is_pool:
        gate_result = gates.gate_cell(
            best_replica,
            cell_data.target,
            cell_data.aligned_ret,
            cell_data.split,
            cell_data.thresholds_entry,
            cell_data.n_eff_val,
            n_configs,
            pooled_replica_signal=best_replica,
            pooled_target_signal=cell_data.target,
            pooled_aligned_ret=cell_data.aligned_ret,
            pooled_split=cell_data.split,
            pooled_thresholds_entry=cell_data.thresholds_entry,
            pooled_n_eff=cell_data.n_eff_val,
            perturbed_metrics=perturbed_arg,
            groups=cell_data.groups,
        )
    else:
        gate_result = gates.evaluate(
            best_replica,
            cell_data.target,
            cell_data.aligned_ret,
            cell_data.split,
            cell_data.thresholds_entry,
            cell_data.n_eff_val,
            n_configs,
            perturbed_metrics=perturbed_arg,
        )

    # Record the chosen config + GateResult to the ledger (a dedicated trial so
    # the verdict is auditable alongside the search trajectory).
    vm_record = {k: v for k, v in val_metrics.items() if k != "confusion"}
    comp = _composite_from_panel(vm_record)
    if comp is not None:
        vm_record["composite_skill"] = comp
    ledger.record(
        {
            "archetype": family,
            "cell": cell_data.cell,
            "tier": result["tier"],
            "params": dict(best_params),
            "val_metrics": vm_record,
            "train_discrepancy": float(result["best_discrepancy"]),
            "gate_result": {
                "g1": gate_result.g1,
                "g2": gate_result.g2,
                "g3": gate_result.g3,
                "g4": gate_result.g4,
                "passed": gate_result.passed,
                **gate_result.details,
            },
            "motivated_by": ["final config for this cell (post-search gate)"],
        }
    )

    pool_member_metrics = (
        _pool_member_metrics(cell_data, best_replica) if is_pool else None
    )

    return CellResult(
        family=family,
        cell=cell_data.cell,
        params=best_params,
        tier=result["tier"],
        n_eff_val=cell_data.n_eff_val,
        n_configs=n_configs,
        val_metrics=val_metrics,
        gate_result=gate_result,
        perturbed=perturbed,
        base_rates=cell_data.base_rates,
        pool_member_metrics=pool_member_metrics,
    )


# --------------------------------------------------------------------------- #
# Final test confirmation (the ONE get_test call).                            #
# --------------------------------------------------------------------------- #
def _test_consistency(
    *,
    family: str,
    cell: str,
    params: dict,
    signals: pd.DataFrame,
    ohlcv: pd.DataFrame,
    split: Split,
    test_idx: np.ndarray,
    test_dates: pd.DatetimeIndex,
) -> dict:
    """Confirm one winner on the held-out TEST block (val<->test consistency).

    Builds the same archetype replica on test, scores the val and test panels
    against the target, and reports both composites alongside per-split base
    rates. The test positions (``test_idx`` / ``test_dates``) are fetched ONCE by
    the caller via :func:`stml.replication.splits.get_test` with
    ``final_confirmation=True`` -- the single deliberate test read of the whole
    pipeline -- and reused for every confirmed family (the split is identical
    across families). Pools are confirmed on their first member (the
    representative standalone series) so the call stays uniform.
    """
    instrument = POOL_MEMBERS[0] if cell == POOL_NAME else cell
    archetype = ARCHETYPES[family]

    aligned = align_instrument(signals, ohlcv, instrument, convention="next_day")
    frame = aligned.frame.set_index("date").sort_index()
    target = frame["signal"].astype(int)
    index = pd.DatetimeIndex(target.index)
    replica = (
        _instrument_signal(archetype, instrument, ohlcv, params)
        .reindex(index)
        .fillna(0)
        .astype(int)
    )

    val_dates = index.intersection(split.val_dates)
    test_d = index.intersection(test_dates)
    tgt_val = target.loc[val_dates].to_numpy(dtype=int)
    rep_val = replica.loc[val_dates].to_numpy(dtype=int)
    tgt_test = target.loc[test_d].to_numpy(dtype=int)
    rep_test = replica.loc[test_d].to_numpy(dtype=int)

    def _comp(yt: np.ndarray, yp: np.ndarray) -> float | None:
        if yt.size == 0:
            return None
        p = panel(yt, yp)
        return 0.5 * (float(p["kappa"]) + float(p["ordinal_skill"]["vs_flat"]))

    return {
        "family": family,
        "cell": cell,
        "confirmed_on": instrument,
        "n_test_positions": int(np.asarray(test_idx).size),
        "val_composite": _comp(tgt_val, rep_val),
        "test_composite": _comp(tgt_test, rep_test),
        "val_base_rates": _base_rates(tgt_val),
        "test_base_rates": _base_rates(tgt_test),
    }


# --------------------------------------------------------------------------- #
# Report rendering.                                                            #
# --------------------------------------------------------------------------- #
def _render_base_rate_row(label: str, br: dict[str, float]) -> str:
    return (
        f"| {label} | {br['n']} | {_fmt(br['participation'])} | {_fmt(br['long_bias'])} "
        f"| {_fmt(br['frac_neg1'])} | {_fmt(br['frac_0'])} | {_fmt(br['frac_pos1'])} |"
    )


def _render_family_report(
    family: str,
    archetype: Archetype,
    results: list[CellResult],
) -> str:
    """One ``reports/<archetype>.md``: definition, grid, per-cell panel, gates."""
    grid = archetype.param_space()
    lines: list[str] = []
    lines.append(f"# Replication family: `{family}`")
    lines.append("")
    lines.append(_FAMILY_DEFS.get(family, "A look-ahead-free signal family."))
    lines.append("")

    lines.append("## Parameter space")
    lines.append("")
    for axis, choices in grid.items():
        lines.append(f"- `{axis}`: {list(choices)}")
    lines.append("")
    tiers = sorted({r.tier for r in results})
    lines.append(
        f"Search tier(s) used: {', '.join(f'`{t}`' for t in tiers)} "
        "(TPE above the n_eff FLOOR, exhaustive grid below it)."
    )
    lines.append("")

    n_pass = sum(1 for r in results if r.passed)
    lines.append(
        f"**Family verdict:** passes G1-G4 on **{n_pass}** of {len(results)} cell(s)."
    )
    lines.append("")

    lines.append("## Per-cell train/val panel")
    lines.append("")
    lines.append(
        "| cell | tier | n_eff | n_configs | val kappa | val ordinal_skill | "
        "composite | G1 | G2 | G3 | G4 | passed |"
    )
    lines.append(
        "|------|------|------:|----------:|----------:|------------------:|"
        "----------:|:--:|:--:|:--:|:--:|:------:|"
    )
    for r in results:
        vm = r.val_metrics or {}
        osk = vm.get("ordinal_skill")
        osk = osk.get("vs_flat") if isinstance(osk, dict) else None
        comp = _composite_from_panel(vm)
        gr = r.gate_result
        lines.append(
            f"| `{r.cell}` | {r.tier} | {r.n_eff_val} | {r.n_configs} "
            f"| {_fmt(vm.get('kappa'))} | {_fmt(osk)} | {_fmt(comp)} "
            f"| {_fmt_bool(gr.g1)} | {_fmt_bool(gr.g2)} | {_fmt_bool(gr.g3)} "
            f"| {_fmt_bool(gr.g4)} | {_fmt_bool(gr.passed)} |"
        )
    lines.append("")

    lines.append("## Per-cell detail (base rates, perturbation, gate diagnostics)")
    lines.append("")
    for r in results:
        lines.append(f"### `{r.cell}`")
        lines.append("")
        lines.append(f"- best params: `{r.params}`")
        lines.append(
            f"- post-embargo val n_eff: {r.n_eff_val} "
            f"({'standalone' if r.n_eff_val >= gates.FLOOR else 'below FLOOR -> pooled'})"
        )
        if r.gate_result.details.get("low_power"):
            lines.append(
                "- **low-power / pooled**: gated on the asset-class pool; a thin "
                "member never earns a standalone pass."
            )
        lines.append("")
        lines.append("  Per-split base rates (the per-split chance reference G2 respects):")
        lines.append("")
        lines.append(
            "  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |"
        )
        lines.append(
            "  |-------|--:|--------------:|----------:|--------:|-------:|--------:|"
        )
        for sp in ("train", "val"):
            lines.append("  " + _render_base_rate_row(sp, r.base_rates[sp]))
        lines.append("")

        if r.perturbed:
            arr = np.asarray(r.perturbed, dtype=float)
            lines.append(
                f"  G3 perturbation (val composite over {arr.size} +/-1-step "
                f"neighbours): min={_fmt(float(arr.min()))}, "
                f"max={_fmt(float(arr.max()))}, std={_fmt(float(arr.std()))} "
                f"(plateau std tol = {gates.PLATEAU_STD_TOL})."
            )
        else:
            lines.append(
                "  G3 perturbation: no usable neighbourhood (single-point grid or "
                "empty val) -> G3 non-blocking."
            )
        lines.append("")

        d = r.gate_result.details
        g1 = d.get("g1", {})
        g2 = d.get("g2", {})
        g4 = d.get("g4", {})
        lines.append(
            f"  - G1 (beats baseline + multiplicity): kappa {_fmt(g1.get('kappa'))} "
            f"vs cutoff {_fmt(g1.get('kappa_cutoff'))}, ordinal_skill "
            f"{_fmt(g1.get('ordinal_skill_vs_flat'))} vs cutoff "
            f"{_fmt(g1.get('ordinal_skill_cutoff'))} (margin {_fmt(g1.get('margin_required'))}, "
            f"n_configs {g1.get('n_configs')})."
        )
        lines.append(
            f"  - G2 (drift-aware generalization): skill_train "
            f"{_fmt(g2.get('skill_train'))}, skill_val {_fmt(g2.get('skill_val'))} "
            f"(required >= {_fmt(g2.get('required_val_skill'))}, gen_frac "
            f"{_fmt(g2.get('gen_frac'))})."
        )
        lines.append(
            f"  - G4 (multi-metric consistency): kappa {_fmt(g4.get('kappa'))}, "
            f"ordinal_skill {_fmt(g4.get('ordinal_skill_vs_flat'))}, increment_corr "
            f"{_fmt(g4.get('increment_corr'))} (all must be > 0)."
        )
        lines.append("")

    return "\n".join(lines) + "\n"


_FAMILY_DEFS: dict[str, str] = {
    "mean_reversion": (
        "Counter-trend mean reversion (C1 prior-best). score_t = "
        "`-zscore(close - SMA_L)`: far above the moving average leans short, far "
        "below leans long. Iterated FIRST because the released signal is "
        "predominantly short-horizon counter-trend."
    ),
    "ts_momentum": (
        "Time-series momentum: score_t = standardised trailing L-day log-return. "
        "Positive (price rose) leans long. Expected weak per C1 (the signal is "
        "counter-trend)."
    ),
    "breakout_donchian": (
        "Donchian-channel position: where the close sits in the trailing N-day "
        "band (band excludes today); a breach scores beyond +/-1."
    ),
    "vol_regime_gated": (
        "A base directional score (mean-reversion or momentum) that participates "
        "only inside a trailing vol regime (high/low quantile gate)."
    ),
    "hybrid_filtered_momentum": (
        "Fast momentum kept only when a slower trailing trend agrees in sign; "
        "disagreement forces flat."
    ),
    "xsect_rank": (
        "Cross-sectional rank: long the top / short the bottom of the universe "
        "each day. SCOPED as an expected-negative diagnostic -- with cross-asset "
        "mean |corr| = 0.09 a single panel ranking should struggle to replicate "
        "near-independent instruments, so the 6th family exists to document the "
        "(lack of) cross-asset structure rather than to add a likely pass. Where "
        "it nonetheless clears the gates, the summary reports that as a pass and "
        "flags it as the weakest / most-surprising replicator."
    ),
}


def _grid_size(archetype: Archetype) -> int:
    """Full Cartesian grid size for a family (product of its axis lengths).

    This is the number of distinct configurations an exhaustive grid would
    evaluate; the TPE ``budget`` is compared against it to report search
    coverage. A family with no axes returns 1 (the empty product).
    """
    size = 1
    for choices in archetype.param_space().values():
        size *= max(len(list(choices)), 1)
    return size


def _recompute_passed(
    cell: CellResult, *, std_tol: float, gen_frac: float
) -> bool:
    """Re-derive a cell's G1-G4 pass at alternative G2/G3 tolerances.

    G1 and G4 are read from the frozen gate booleans (their thresholds are not
    being swept); G2 is recomputed as ``skill_val > 0 and skill_val >= gen_frac *
    skill_train`` and G3 as ``min_perturbed > cutoff and std_perturbed <
    std_tol`` from the stored gate details. A low-power / pooled verdict that was
    forced to ``False`` (a thin standalone member gated only on its class) stays
    ``False`` regardless of tolerances -- the FLOOR rule is not part of the
    sensitivity sweep. A G3 that was never evaluated (no neighbourhood) stays
    non-blocking (treated as passing) exactly as in :func:`gates._gate3`.
    """
    d = cell.gate_result.details
    if d.get("low_power"):
        return False

    g1 = bool(cell.gate_result.g1)
    g4 = bool(cell.gate_result.g4)

    g2d = d.get("g2", {})
    skill_train = g2d.get("skill_train")
    skill_val = g2d.get("skill_val")
    if isinstance(skill_train, (int, float)) and isinstance(skill_val, (int, float)):
        g2 = bool(skill_val > 0.0 and skill_val >= gen_frac * skill_train)
    else:
        g2 = bool(cell.gate_result.g2)

    g3d = d.get("g3", {})
    if g3d.get("not_evaluated"):
        g3 = True
    else:
        worst = g3d.get("min_perturbed")
        spread = g3d.get("std_perturbed")
        cutoff = g3d.get("cutoff")
        if all(isinstance(v, (int, float)) for v in (worst, spread, cutoff)):
            g3 = bool(worst > cutoff and spread < std_tol)
        else:
            g3 = bool(cell.gate_result.g3)

    return bool(g1 and g2 and g3 and g4)


def _sensitivity_passcount(
    results_by_family: dict[str, list[CellResult]],
    *,
    std_tol: float,
    gen_frac: float,
) -> int:
    """Number of families that pass >= 1 cell at the given G2/G3 tolerances."""
    return sum(
        1
        for results in results_by_family.values()
        if any(
            _recompute_passed(r, std_tol=std_tol, gen_frac=gen_frac) for r in results
        )
    )


def _xsect_caution_or_negative(
    results_by_family: dict[str, list[CellResult]],
) -> str:
    """The xsect_rank shortfall bullet, data-driven on its actual gate verdict.

    ``xsect_rank`` was *scoped* as an expected-negative diagnostic (cross-asset
    mean |corr| ~= 0.09). This reads its real per-cell results: if it cleared all
    four gates anywhere we must NOT claim it failed -- we emit a clearly-labelled
    CAUTION paragraph reporting the pass (with the winning cell's kappa, its
    n_eff -- labelled ``val`` for a standalone cell or ``pooled`` for the energy
    pool -- and its score variant, noting the momentum-vs-mean-reversion tension
    only when the winning variant is actually momentum) and flag it as the
    weakest/most-surprising replicator. A secondary mention of the energy pool is
    added only when the pool ALSO passed, reporting its real variant + metrics
    (no hard-coded assumption). If xsect genuinely failed, the original
    expected-negative narrative stands. Every number is pulled from the stored
    gate details / params so the prose can never drift from the matrix.
    """
    results = results_by_family.get("xsect_rank")
    if not results:
        return (
            "- **Cross-asset independence (xsect_rank).** Not run this pass; "
            "scoped as an expected-negative diagnostic (cross-asset mean |corr| "
            "= 0.09)."
        )
    passing = [r for r in results if r.passed]
    if not passing:
        return (
            "- **Cross-asset independence (xsect_rank).** With cross-asset mean "
            "|corr| = 0.09 the instruments are near-independent, so a single "
            "cross-sectional ranking cannot reproduce any one instrument's regime "
            "calls. `xsect_rank` was scoped as an EXPECTED-NEGATIVE diagnostic and "
            "indeed cleared no cell -- its failure documents the (lack of) "
            "cross-asset structure rather than a deficient search."
        )
    best = max(
        passing,
        key=lambda r: (_composite_from_panel(r.val_metrics or {}) or -math.inf),
    )
    vm = best.val_metrics or {}
    kappa = vm.get("kappa")

    def _variant_phrase(score_kind: str) -> str:
        return (
            "a momentum variant, which runs counter to the C1 mean-reversion "
            "characterisation"
            if score_kind == "momentum"
            else f"a `score={score_kind}` variant"
        )

    counter = _variant_phrase(str(best.params.get("score", "?")))
    n_eff_label = "pooled n_eff" if best.cell == POOL_NAME else "val n_eff"

    # Secondary mention: only if the energy POOL ALSO passed and isn't already
    # the cell quoted above -- and report its ACTUAL score variant (no
    # hard-coded assumption that the pool winner is momentum).
    extra = ""
    pool_pass = next(
        (r for r in passing if r.cell == POOL_NAME and r is not best), None
    )
    if pool_pass is not None:
        pk = (pool_pass.val_metrics or {}).get("kappa")
        extra = (
            f" It also cleared the pooled energy cell ({_variant_phrase(str(pool_pass.params.get('score', '?')))}, "
            f"kappa ~{_fmt(pk, 2)}, pooled n_eff ~= {pool_pass.n_eff_val}), "
            "reinforcing the caution."
        )

    return (
        "- **Cross-asset diagnostic (xsect_rank) -- CAUTION.** `xsect_rank` was "
        "scoped as an expected-negative diagnostic (cross-asset mean |corr| ~= "
        f"0.09); it nonetheless cleared all four gates on the `{best.cell}` cell "
        f"(kappa ~{_fmt(kappa, 2)}, {n_eff_label} ~= {best.n_eff_val}). We treat "
        f"this cautiously: the {n_eff_label} is modest, and the winning variant "
        f"is {counter}. It is reported as a pass but flagged as the weakest / "
        f"most-surprising of the replicators.{extra}"
    )


def _pool_within_instrument_lines(
    results_by_family: dict[str, list[CellResult]],
) -> list[str]:
    """Render the 'Pooling: within-instrument aggregation' methodological note.

    Documents the cross-instrument pooling artifact that was fixed: a pooled cell
    used to CONCATENATE members and score ONE metric, which let cross-instrument
    base-rate matching inflate kappa. The canonical proof is ts_momentum, whose
    concatenated pool kappa was 0.14 while its honest per-member val kappas
    (cl1s -0.130, ho1s -0.026, ng1s +0.284) average +0.04 -- chance, with two of
    three members ANTI-replicated. Pooled skill is now the equal-weight MEAN of the
    per-member within-instrument metrics in BOTH the search objective and every
    gate, so that artifact can no longer create a pass. Where this run actually
    produced pool winners, their real per-member-vs-group-averaged-vs-concatenated
    val kappas are tabulated so the prose can never drift from the numbers.
    """
    lines: list[str] = []
    lines.append("## Pooling: within-instrument aggregation")
    lines.append("")
    lines.append(
        "**Artifact fixed.** A pooled cell (`pool:energy = cl1s + ho1s + ng1s`) "
        "previously CONCATENATED its members' `(target, replica)` rows and computed "
        "a SINGLE metric. That inflates Cohen's kappa via cross-instrument "
        "base-rate matching: the `ts_momentum` pool winner had a concatenated val "
        "kappa of ~0.14 -- a 'pass' -- yet its honest per-member val kappas were "
        "cl1s -0.130, ho1s -0.026, ng1s +0.284 (equal-weight mean +0.04 ~= "
        "chance), so momentum ANTI-replicated two of the three energy members while "
        "still clearing the pool. `breakout_donchian` showed the same pattern."
    )
    lines.append("")
    lines.append(
        "**Fix.** A pooled cell's skill is now the equal-weight MEAN of the "
        "per-member WITHIN-instrument metrics (never a single concatenated metric), "
        "applied to BOTH the search objective (it minimises the mean per-member "
        "train discrepancy) AND the gates (G1 kappa/ordinal, G2 per-split skill, G3 "
        "neighbourhood composite, and G4 kappa/ordinal/NAV-increment-corr are each "
        "per-member-then-averaged). Cross-instrument base-rate matching can no "
        "longer manufacture a pass, so a family clears `pool:energy` only on "
        "genuine within-instrument energy skill."
    )
    lines.append("")
    # Real per-member numbers from this run's pool winners (data-driven).
    pool_rows: list[tuple[str, dict]] = []
    for fam, results in results_by_family.items():
        for r in results:
            if r.cell == POOL_NAME and r.pool_member_metrics:
                pool_rows.append((fam, r.pool_member_metrics))
    if pool_rows:
        lines.append(
            "Per-member vs group-averaged vs concatenated val kappa, this run's "
            "pool winners (the concatenated column is the OLD artifact, shown for "
            "contrast; the group-averaged column is what is now gated):"
        )
        lines.append("")
        lines.append(
            "| family | cl1s | ho1s | ng1s | group-avg (gated) | concatenated (old) |"
        )
        lines.append(
            "|--------|-----:|-----:|-----:|------------------:|-------------------:|"
        )
        for fam, pm in pool_rows:
            mem = pm.get("members", {})
            lines.append(
                f"| `{fam}` | {_fmt(mem.get('cl1s'))} | {_fmt(mem.get('ho1s'))} "
                f"| {_fmt(mem.get('ng1s'))} | {_fmt(pm.get('group_avg_kappa'))} "
                f"| {_fmt(pm.get('concatenated_kappa'))} |"
            )
        lines.append("")
    return lines


def _render_summary(
    *,
    results_by_family: dict[str, list[CellResult]],
    passing_families: list[str],
    budget: int,
    cells_in_order: list[str],
    test_consistency: list[dict],
) -> str:
    """``reports/replication-summary.md``: matrix, headline, shortfall, test."""
    lines: list[str] = []
    lines.append("# Replication Summary (US-011)")
    lines.append("")
    lines.append(
        "Each `(family, cell)` was searched on the TRAIN discrepancy objective "
        "(TPE above the post-embargo n_eff FLOOR, exhaustive grid below it) and "
        "the winner gated on G1-G4 over the TRAIN+VAL window. The held-out test "
        "block was touched exactly once, for the final confirmation below."
    )
    lines.append("")

    n_pass = len(passing_families)
    success = n_pass >= FAMILIES_REQUIRED
    lines.append("## Headline")
    lines.append("")
    if success:
        lines.append(
            f"**{n_pass} of {len(results_by_family)} families replicate >= 1 cell "
            f"(>= {FAMILIES_REQUIRED} required): SUCCESS.**"
        )
    else:
        lines.append(
            f"**{n_pass} of {len(results_by_family)} families replicate >= 1 cell "
            f"(< {FAMILIES_REQUIRED} required): see honest-shortfall below.**"
        )
    lines.append("")
    for fam in results_by_family:
        cells = [r.cell for r in results_by_family[fam] if r.passed]
        if cells:
            lines.append(f"- `{fam}` replicates: " + ", ".join(f"`{c}`" for c in cells))
        else:
            lines.append(f"- `{fam}` replicates: none")
    lines.append("")

    # --- pass/fail matrix ----------------------------------------------------
    lines.append("## Families x cells pass/fail matrix")
    lines.append("")
    header = "| family | " + " | ".join(f"`{c}`" for c in cells_in_order) + " | family passes? |"
    sep = "|--------|" + "|".join([":--:"] * len(cells_in_order)) + "|:-------------:|"
    lines.append(header)
    lines.append(sep)
    for fam, results in results_by_family.items():
        by_cell = {r.cell: r for r in results}
        row = [f"| `{fam}`"]
        for c in cells_in_order:
            r = by_cell.get(c)
            if r is None:
                row.append(" n/a ")
            else:
                row.append(" PASS " if r.passed else " fail ")
        fam_ok = "yes" if fam in passing_families else "no"
        row.append(f" {fam_ok} ")
        lines.append("|".join(row) + "|")
    lines.append("")

    # --- search coverage (grid size vs budget) -------------------------------
    lines.append("## Search coverage (full grid vs TPE budget)")
    lines.append("")
    lines.append(
        f"Above the n_eff FLOOR each cell runs a seeded TPE search of "
        f"`budget = {budget}` trials; the full grid is the Cartesian product of "
        "the family's axes (the exhaustive set the below-FLOOR grid tier would "
        "enumerate). Coverage is `min(budget, |grid|) / |grid|` -- how much of the "
        "configuration space a single cell's TPE budget can reach."
    )
    lines.append("")
    lines.append("| family | full grid size | budget | coverage |")
    lines.append("|--------|---------------:|-------:|---------:|")
    for fam in results_by_family:
        gsize = _grid_size(ARCHETYPES[fam])
        cov = min(budget, gsize) / gsize if gsize else 1.0
        lines.append(f"| `{fam}` | {gsize} | {budget} | {_fmt(100.0 * cov, 1)}% |")
    lines.append("")

    # --- pooling semantics + rb1s reconciliation -----------------------------
    lines.append("## Pooled-cell gating semantics (energy)")
    lines.append("")
    lines.append(
        "The GATED energy pool is the three BELOW-FLOOR members "
        "`{cl1s, ho1s, ng1s}` (post-embargo val n_eff 9, 9, 2). Its gateable "
        "n_eff is their SUM -- 9 + 9 + 2 = 20 -- which clears the FLOOR (>= 10), "
        "so the pool is gated FIRST-CLASS (`gate_cell` routes a >= FLOOR cell "
        "straight through `evaluate`, not the forced-low-power branch). Summing "
        "is justified because cross-asset mean |corr| ~= 0.09 makes the members "
        "near-independent, so the concatenated series carries ~the sum of their "
        "independent regime-calls. Each member ALONE stays below the FLOOR and "
        "can never earn a standalone pass."
    )
    lines.append("")
    lines.append(
        "Reconciliation: `thresholds.json`'s `per_asset_class['energy']` lists "
        "FOUR members (incl. `rb1s`) as the class baseline, whereas the gated "
        "pool here is only the three below-FLOOR members. `rb1s` (val n_eff 13) "
        "is at/above the FLOOR and is gated STANDALONE, not in the pool. "
        "Including vs excluding `rb1s` in the class-baseline cutoff is immaterial "
        "(identical to 3 d.p.), so the pool's threshold entry is used as-is."
    )
    lines.append("")

    # --- pooling: within-instrument aggregation (the artifact fixed) ----------
    lines.extend(_pool_within_instrument_lines(results_by_family))

    # --- gate-calibration sensitivity ----------------------------------------
    std_tols = (0.10, 0.15, 0.20)
    gen_fracs = (0.4, 0.5, 0.6)
    lines.append("## Gate-calibration sensitivity")
    lines.append("")
    lines.append(
        "How the family pass-count moves as the G3 plateau tolerance "
        f"(`std_tol`, default {gates.PLATEAU_STD_TOL}) and the G2 generalization "
        f"fraction (`gen_frac`, default {gates.GEN_FRAC}) are perturbed. Each cell "
        "is re-judged from its STORED neighbourhood + skill metrics (no re-search "
        "needed); G1 and G4 are held at their frozen verdicts. A stable count "
        "across the grid is what substantiates the 'well-calibrated' claim."
    )
    lines.append("")
    base_count = len(passing_families)
    lines.append(f"Recorded winners pass-count (default tolerances): **{base_count}**.")
    lines.append("")
    lines.append("G3 plateau tolerance sweep (gen_frac at default):")
    lines.append("")
    lines.append("| std_tol | families passing |")
    lines.append("|--------:|-----------------:|")
    for st in std_tols:
        cnt = _sensitivity_passcount(
            results_by_family, std_tol=st, gen_frac=gates.GEN_FRAC
        )
        lines.append(f"| {_fmt(st, 2)} | {cnt} |")
    lines.append("")
    lines.append("G2 generalization fraction sweep (std_tol at default):")
    lines.append("")
    lines.append("| gen_frac | families passing |")
    lines.append("|---------:|-----------------:|")
    for gf in gen_fracs:
        cnt = _sensitivity_passcount(
            results_by_family, std_tol=gates.PLATEAU_STD_TOL, gen_frac=gf
        )
        lines.append(f"| {_fmt(gf, 2)} | {cnt} |")
    lines.append("")

    # --- caveats + non-passing families + cross-asset diagnostic -------------
    # Rendered whether or not the study reaches the threshold: the report must
    # stay accurate about WHY any family fails and must reconcile the xsect_rank
    # narrative against its actual verdict (the heading + lead adapt to the
    # outcome, but the structural mechanisms and the xsect caution always show).
    non_passing = [f for f in results_by_family if f not in passing_families]
    if success:
        lines.append("## Caveats and non-passing families")
        lines.append("")
        lines.append(
            f"{n_pass} of {len(results_by_family)} families cleared all four gates "
            f"on at least one cell (>= {FAMILIES_REQUIRED} required), so the study "
            "succeeds. The mechanisms below explain why the remaining family does "
            "not replicate and flag the most-surprising of the passes -- the gates "
            "remain the same structural filters either way:"
        )
    else:
        lines.append("## Honest shortfall")
        lines.append("")
        lines.append(
            f"Only {n_pass} of {len(results_by_family)} families cleared all four "
            f"gates on at least one cell, short of the {FAMILIES_REQUIRED} required. "
            "The shortfall is structural, not a bug in the search:"
        )
    lines.append("")
    lines.append(
        "- **Effective sample size.** Even the standalone cells carry only "
        "~11-35 leakage-free regime-calls in val (post-embargo n_eff); the "
        "three energy members (cl1s 9, ho1s 9, ng1s 2) sit below the FLOOR=10 "
        "and are gated only on the energy POOL, never standalone. A genuine "
        "skill edge has to clear chance cutoffs AND a multiplicity-inflated "
        "margin on a handful of independent observations."
    )
    lines.append(
        "- **Base-rate drift.** The released signal's base rates drift across "
        "splits (e.g. ng1s participation 0.07 -> 0.31 -> 0.43), so G2 measures "
        "skill against each split's OWN chance level. A replica that merely "
        "tracks the drifting majority nets to ~0 skill and fails G2 -- exactly "
        "as intended."
    )
    lines.append(
        "- **Degeneracy / plateau.** G3 demands a parameter plateau over the "
        "NUMERIC/ordinal axes (categorical strategy switches -- `base`, "
        "`regime`, `score` -- are held fixed, so a plateau probe never "
        "conflates a strategy flip with a tuning nudge); a winner whose "
        "numeric neighbours scatter below the cutoff is rejected as overfit."
    )
    # Name the families that did not pass (excluding xsect_rank, which has its
    # own dedicated caution line below) so the prose is concrete, not generic.
    other_non_passing = [f for f in non_passing if f != "xsect_rank"]
    if other_non_passing:
        names = ", ".join(f"`{f}`" for f in other_non_passing)
        lines.append(
            f"- **Non-passing families.** {names} cleared no cell: their winning "
            "configs either missed a chance cutoff (G1), failed to transfer "
            "per-split skill (G2), sat on a numeric knife-edge (G3) or had a "
            "metric disagree in sign (G4). These are the structural filters above "
            "biting, not a search that stopped early."
        )
    lines.append(_xsect_caution_or_negative(results_by_family))
    lines.append("")

    # --- final test confirmation ---------------------------------------------
    lines.append("## Final test confirmation (val <-> test consistency)")
    lines.append("")
    if not test_consistency:
        lines.append(
            "No family cleared all four gates, so there is no frozen winner to "
            "confirm on the held-out test block. `get_test(final_confirmation=True)` "
            "was therefore not invoked (no leakage-free claim to validate)."
        )
        lines.append("")
    else:
        lines.append(
            "For the top passing `(family, cell, params)` per family (capped at "
            f"{MAX_FINAL_FAMILIES}), the replica is rebuilt on the held-out test "
            "block and its composite skill compared to val. This is the ONLY "
            "`get_test(final_confirmation=True)` call in the pipeline."
        )
        lines.append("")
        lines.append(
            "| family | confirmed on | val composite | test composite | "
            "val participation | test participation |"
        )
        lines.append("|--------|--------------|--------------:|---------------:|"
                     "------------------:|-------------------:|")
        for tc in test_consistency:
            lines.append(
                f"| `{tc['family']}` | `{tc['confirmed_on']}` "
                f"| {_fmt(tc['val_composite'])} | {_fmt(tc['test_composite'])} "
                f"| {_fmt(tc['val_base_rates']['participation'])} "
                f"| {_fmt(tc['test_base_rates']['participation'])} |"
            )
        lines.append("")
        lines.append("Per-split base rates for each confirmed winner:")
        lines.append("")
        for tc in test_consistency:
            lines.append(f"- `{tc['family']}` / `{tc['cell']}` (confirmed on `{tc['confirmed_on']}`):")
            lines.append(
                "  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |"
            )
            lines.append(
                "  |-------|--:|--------------:|----------:|--------:|-------:|--------:|"
            )
            lines.append("  " + _render_base_rate_row("val", tc["val_base_rates"]))
            lines.append("  " + _render_base_rate_row("test", tc["test_base_rates"]))
            lines.append("")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# top_candidates.json                                                          #
# --------------------------------------------------------------------------- #
def _top_candidates(
    results_by_family: dict[str, list[CellResult]],
    passing_families: list[str],
) -> list[dict]:
    """Best passing ``(family, cell)`` per passing family for the JSON artifact.

    The "best" passing cell for a family is the one with the highest val
    composite skill among its passing cells; each entry carries the params, tier,
    n_eff, val_metrics (confusion dropped), and the four gate booleans.
    """
    out: list[dict] = []
    for fam in passing_families:
        passing = [r for r in results_by_family[fam] if r.passed]
        if not passing:
            continue
        best = max(
            passing,
            key=lambda r: (_composite_from_panel(r.val_metrics or {}) or -math.inf),
        )
        vm = {k: v for k, v in (best.val_metrics or {}).items() if k != "confusion"}
        out.append(
            {
                "archetype": best.family,
                "cell": best.cell,
                "params": dict(best.params),
                "tier": best.tier,
                "n_eff": best.n_eff_val,
                "val_metrics": vm,
                "gates": {
                    "g1": bool(best.gate_result.g1),
                    "g2": bool(best.gate_result.g2),
                    "g3": bool(best.gate_result.g3),
                    "g4": bool(best.gate_result.g4),
                    "passed": bool(best.gate_result.passed),
                },
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Orchestration.                                                              #
# --------------------------------------------------------------------------- #
def run(
    families: list[str] | None = None,
    *,
    out_dir: str | Path | None = None,
    budget: int = 64,
    seed: int = 0,
    ledger_path: str | Path | None = None,
) -> dict[str, object]:
    """Run the replication search over every ``(family, cell)`` and write artifacts.

    Parameters
    ----------
    families : the subset of families to run; ``None`` runs all six in
        ``FAMILY_ORDER`` (mean_reversion first -- the C1 prior-best).
    out_dir : root under which ``reports/`` and ``results/jj/`` are written.
        ``None`` (default) uses the repo root, producing the canonical
        deliverables. Tests pass a temp dir so a tiny-budget smoke run never
        clobbers the real artifacts.
    budget : TPE trials per cell (ignored on the grid tier, which is exhaustive).
    seed : seeds the TPE sampler for reproducibility.
    ledger_path : where the ledger persists. ``None`` uses ``out_dir`` (or the
        repo root) so a smoke run keeps its own isolated ledger.

    Returns
    -------
    dict with the written ``paths`` and a ``summary`` block (families passed,
    success flag, per-family passing cells) for programmatic inspection / tests.
    """
    fam_list = list(families) if families else list(FAMILY_ORDER)
    for fam in fam_list:
        if fam not in ARCHETYPES:
            raise ValueError(f"unknown family {fam!r}; known: {sorted(ARCHETYPES)}")

    root = Path(out_dir) if out_dir is not None else _repo_root()
    if ledger_path is not None:
        led_path = Path(ledger_path)
    else:
        led_path = root / _LEDGER_REL
    ledger = Ledger(led_path)

    ohlcv, signals = _load_clean()
    # A one-year lookback before the 2020-2022 signal era warms every trailing
    # feature (mirrors run_characterize); the search only ever indexes the
    # train/val slices, never test.
    ohlcv = ohlcv[ohlcv["date"] >= "2019-01-01"].copy()
    thresholds = _read_thresholds(root)

    split = chronological_split(signals["date"])

    cells_in_order = [*STANDALONE_CELLS, POOL_NAME]
    results_by_family: dict[str, list[CellResult]] = {}

    for family in fam_list:
        archetype = ARCHETYPES[family]
        cell_results: list[CellResult] = []

        # Standalone instruments.
        for inst in STANDALONE_CELLS:
            cell_data = _standalone_cell(
                instrument=inst,
                archetype=archetype,
                signals=signals,
                ohlcv=ohlcv,
                split=split,
                thresholds_entry=thresholds["per_instrument"][inst],
            )
            cell_results.append(
                _run_cell(
                    family=family,
                    archetype=archetype,
                    cell_data=cell_data,
                    pool_cell_data=None,
                    ledger=ledger,
                    budget=budget,
                    seed=seed,
                )
            )

        # The one energy pool.
        pool_data = _pool_cell(
            archetype=archetype,
            signals=signals,
            ohlcv=ohlcv,
            split=split,
            thresholds_entry=thresholds["per_asset_class"][POOL_CLASS],
        )
        cell_results.append(
            _run_cell(
                family=family,
                archetype=archetype,
                cell_data=pool_data,
                pool_cell_data=pool_data,
                ledger=ledger,
                budget=budget,
                seed=seed,
            )
        )

        results_by_family[family] = cell_results

    passing_families = [
        fam for fam, rs in results_by_family.items() if any(r.passed for r in rs)
    ]

    # --- Final test confirmation: the ONE get_test call ----------------------
    # The test block is read EXACTLY ONCE here, then reused for every confirmed
    # family (the chronological split is identical across families, so a single
    # deliberate, auditable read serves them all).
    test_consistency: list[dict] = []
    final_families = passing_families[:MAX_FINAL_FAMILIES]
    if final_families:
        test_idx = get_test(split, final_confirmation=True)
        test_dates = split.test_dates
        for fam in final_families:
            passing = [r for r in results_by_family[fam] if r.passed]
            best = max(
                passing,
                key=lambda r: (_composite_from_panel(r.val_metrics or {}) or -math.inf),
            )
            test_consistency.append(
                _test_consistency(
                    family=fam,
                    cell=best.cell,
                    params=best.params,
                    signals=signals,
                    ohlcv=ohlcv,
                    split=split,
                    test_idx=test_idx,
                    test_dates=test_dates,
                )
            )

    # --- Write artifacts -----------------------------------------------------
    reports_dir = root / "reports"
    results_dir = root / "results" / "jj"
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    family_report_paths: dict[str, Path] = {}
    for family, results in results_by_family.items():
        text = _render_family_report(family, ARCHETYPES[family], results)
        p = reports_dir / f"{family}.md"
        p.write_text(text, encoding="utf-8")
        family_report_paths[family] = p

    summary_text = _render_summary(
        results_by_family=results_by_family,
        passing_families=passing_families,
        budget=budget,
        cells_in_order=cells_in_order,
        test_consistency=test_consistency,
    )
    summary_path = root / _SUMMARY_REL
    summary_path.write_text(summary_text, encoding="utf-8")

    top = _top_candidates(results_by_family, passing_families)
    top_path = root / _TOP_CANDIDATES_REL
    top_path.write_text(json.dumps(top, indent=2) + "\n", encoding="utf-8")

    # Ledger.render_markdown() writes the canonical ``reports/replication-ledger.md``
    # under the REPO ROOT unconditionally (a fixed behaviour of the locked ledger
    # module). For an isolated ``out_dir`` (a smoke run) that would clobber the
    # real artifact, so we snapshot and restore the repo-root copy around the call
    # -- the smoke run still gets its own ledger md under ``out_dir``, the real one
    # is left byte-for-byte intact.
    ledger_md_path = reports_dir / "replication-ledger.md"
    canonical_md = _repo_root() / "reports" / "replication-ledger.md"
    is_isolated = ledger_md_path.resolve() != canonical_md.resolve()
    backup = (
        canonical_md.read_bytes() if (is_isolated and canonical_md.is_file()) else None
    )
    ledger_md = ledger.render_markdown()
    if is_isolated:
        if backup is not None:
            canonical_md.write_bytes(backup)
        elif canonical_md.is_file():
            # The repo had no ledger md before this smoke run; remove the one the
            # render side-effect just created so the smoke run leaves no trace.
            canonical_md.unlink()
    ledger_md_path.write_text(ledger_md, encoding="utf-8")

    return {
        "paths": {
            "summary": summary_path,
            "top_candidates": top_path,
            "ledger_md": ledger_md_path,
            "family_reports": family_report_paths,
        },
        "summary": {
            "families_passed": passing_families,
            "n_families_passed": len(passing_families),
            "success": len(passing_families) >= FAMILIES_REQUIRED,
            "passing_cells": {
                fam: [r.cell for r in results_by_family[fam] if r.passed]
                for fam in results_by_family
            },
            "test_consistency": test_consistency,
        },
    }


def _read_thresholds(root: Path) -> dict:
    """Load ``results/jj/thresholds.json`` (the gate cutoffs).

    Falls back to the repo-root copy when ``root`` (an ``out_dir`` for a smoke
    run) has no calibrated thresholds of its own -- the cutoffs are fixed,
    train-only artifacts, not something a smoke run regenerates.
    """
    candidate = root / "results" / "jj" / "thresholds.json"
    if not candidate.is_file():
        candidate = _repo_root() / "results" / "jj" / "thresholds.json"
    return json.loads(candidate.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> dict[str, object]:
    """CLI entry point: parse args and run the replication orchestrator."""
    parser = argparse.ArgumentParser(
        prog="python -m stml.replication.run_replicate",
        description="Run the replication search over every archetype family x "
        "cell, gate the winners, and write the US-011 deliverables.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        metavar="FAMILY",
        choices=sorted(ARCHETYPES),
        help="families to run (default: all six, mean_reversion first).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        metavar="DIR",
        help="output root for reports/ and results/jj/ (default: repo root).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=64,
        metavar="N",
        help="TPE trials per cell above the n_eff FLOOR (default: 64).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="S",
        help="seed for the TPE sampler (default: 0).",
    )
    args = parser.parse_args(argv)

    result = run(
        args.families,
        out_dir=args.out_dir,
        budget=args.budget,
        seed=args.seed,
    )
    summary = result["summary"]
    paths = result["paths"]
    print(
        f"families passed: {summary['n_families_passed']} "
        f"(>= {FAMILIES_REQUIRED} required: "
        f"{'YES' if summary['success'] else 'NO'})"
    )
    for fam, cells in summary["passing_cells"].items():
        print(f"  {fam}: {', '.join(cells) if cells else 'none'}")
    print(f"wrote summary: {paths['summary']}")
    print(f"wrote top_candidates: {paths['top_candidates']}")
    print(f"wrote ledger: {paths['ledger_md']}")
    for fam, p in paths["family_reports"].items():
        print(f"wrote family report {fam}: {p}")
    return result


if __name__ == "__main__":
    main()
