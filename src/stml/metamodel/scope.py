"""scope.py
=========
D5 InstrumentScope registry for the feature-engineering metamodel layer.

Computes and persists the per-instrument scope decisions that govern how
fitted model families are applied:

* **regime features (F3 GMM + Markov)** — always ``per_instrument``.
* **latent features (F4 PCA + KMeans + AE)** — always
  ``pooled_within_class``.

The key quantity is ``n_eff_gate``: the effective sample size on the
post-embargo validation window (see :func:`stml.replication.splits.n_eff`
and :func:`stml.replication.splits.embargoed_val`).  Instruments whose
``n_eff_gate`` falls below ``FLOOR = 10`` are flagged ``low_power = True``
and treated with extra caution downstream (class-level pooling rather than
standalone verdicts).

Verified ground-truth values (CONTRACT_FE §2):
    es1s=35  nq1s=20  fesx1s=25  cl1s=9  ho1s=9  rb1s=13  ng1s=2
    gc1s=11  si1s=19  hg1s=29    pl1s=26
Low-power set: {cl1s, ho1s, ng1s}.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from stml.replication.splits import (
    chronological_split,
    embargoed_val,
    n_eff,
    run_length_p90,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FLOOR: int = 10
"""Minimum gateable n_eff for a standalone instrument verdict."""

#: Asset-class membership per CONTRACT_FE §2.
ASSET_CLASS_MAP: dict[str, str] = {
    "es1s": "EQ",
    "nq1s": "EQ",
    "fesx1s": "EQ",
    "cl1s": "EN",
    "ho1s": "EN",
    "rb1s": "EN",
    "ng1s": "EN",
    "gc1s": "ME",
    "si1s": "ME",
    "hg1s": "ME",
    "pl1s": "ME",
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class InstrumentScope:
    """Scope metadata for one instrument.

    Parameters
    ----------
    instrument : str
        Ticker symbol (e.g. ``"es1s"``).
    asset_class : str
        One of ``{"EQ", "EN", "ME"}`` per the D5 asset-class map.
    n_eff_gate : int
        Effective sample size on the post-embargo validation window.
        Used to gate statistical verdicts.
    fit_scope_regime : str
        Fitting scope for regime features (F3 GMM + Markov).
        Always ``"per_instrument"``.
    fit_scope_latent : str
        Fitting scope for latent features (F4 PCA + KMeans + AE).
        Always ``"pooled_within_class"``.
    low_power : bool
        ``True`` when ``n_eff_gate < FLOOR`` (10).  Instruments in this set
        are not given standalone verdicts; results are pooled at class level.
    embargo_p90 : int
        Per-instrument embargo width = the 90th-percentile constant-signal
        run length over the full released period
        (:func:`stml.replication.splits.run_length_p90`).  This is the
        purge/embargo a downstream cross-validation must apply at each split
        boundary so no constant-signal run straddles it; persisting it encodes
        that piece of the FE->model handoff contract (plan AC-6e).
    """

    instrument: str
    asset_class: str
    n_eff_gate: int
    fit_scope_regime: str = "per_instrument"
    fit_scope_latent: str = "pooled_within_class"
    low_power: bool = False
    embargo_p90: int = 0

    def __post_init__(self) -> None:
        # Enforce low_power from n_eff_gate regardless of caller-supplied value.
        object.__setattr__(self, "low_power", self.n_eff_gate < FLOOR)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_scope(
    signals: pd.DataFrame,
    ohlcv: pd.DataFrame | None = None,  # noqa: ARG001  (reserved for future use)
) -> dict[str, InstrumentScope]:
    """Compute the D5 scope for all instruments present in ``signals``.

    Parameters
    ----------
    signals : pd.DataFrame
        Wide signal panel with a ``"date"`` column and one column per
        instrument (values in ``{-1, 0, 1}``).  This is the object returned
        by :func:`stml.io.load_clean_data` as ``signals_wide``.
    ohlcv : pd.DataFrame or None
        Unused; reserved so downstream callers can pass the OHLCV frame
        without triggering a signature change.

    Returns
    -------
    dict[str, InstrumentScope]
        Mapping from instrument ticker to its :class:`InstrumentScope`.
        Covers exactly the instrument columns present in ``signals`` that
        also appear in :data:`ASSET_CLASS_MAP`.

    Notes
    -----
    ``n_eff_gate`` is computed as::

        split = chronological_split(signals["date"])
        emb   = embargoed_val(sig, split)   # full-period p90 embargo
        n_eff_gate = n_eff(sig.iloc[emb])

    This is the post-embargo effective sample size on the validation window,
    exactly as specified in CONTRACT_FE §3 and documented in
    :func:`stml.replication.splits.embargoed_val`.
    """
    split = chronological_split(signals["date"])

    scope: dict[str, InstrumentScope] = {}
    for inst, asset_class in ASSET_CLASS_MAP.items():
        if inst not in signals.columns:
            continue
        sig: pd.Series = signals[inst]
        emb = embargoed_val(sig, split)
        gate = n_eff(sig.iloc[emb])
        scope[inst] = InstrumentScope(
            instrument=inst,
            asset_class=asset_class,
            n_eff_gate=gate,
            fit_scope_regime="per_instrument",
            fit_scope_latent="pooled_within_class",
            low_power=gate < FLOOR,
            embargo_p90=run_length_p90(sig),
        )

    return scope


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def persist_scope(
    scope: dict[str, InstrumentScope],
    path: str | Path = "results/instrument_scope.json",
) -> None:
    """Write the scope registry to ``path`` as a JSON object.

    Parameters
    ----------
    scope : dict[str, InstrumentScope]
        Mapping returned by :func:`build_scope`.
    path : str or Path
        Destination file.  Parent directories are created as needed.
        The default ``"results/instrument_scope.json"`` is relative to the
        current working directory; callers should pass an absolute path when
        the cwd may vary (e.g. in tests, use ``tmp_path``).

    Notes
    -----
    The file is a JSON object keyed by instrument ticker; each value is the
    flat dict of :class:`InstrumentScope` fields.  This format round-trips
    without loss: ``json.loads(path.read_text())`` gives back all fields with
    correct types.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {inst: asdict(sc) for inst, sc in scope.items()}
    dest.write_text(json.dumps(payload, indent=2))
