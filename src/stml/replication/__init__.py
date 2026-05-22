"""Primary-signal reverse-engineering & replication framework.

Phase A foundation (pure, unit-tested): align, splits, baselines, metrics, nav.
Phase B characterization (C1): characterize, run_characterize.
Phase C replication search: archetypes, gates, ledger, search, run_replicate.

See the work plan at ``.omc/plans/signal-reverse-engineering-plan.md``.
"""

from stml.replication.align import AlignResult, align_instrument, align_panel
from stml.replication.archetypes import ARCHETYPES, Archetype, decide, generate_panel
from stml.replication.baselines import (
    always_flat,
    majority_class,
    persistence,
    stratified_random,
)
from stml.replication.characterize import (
    alpha_type,
    characterize_all,
    characterize_instrument,
    cross_asset,
    drift,
    lead_lag,
    model_family_fingerprint,
    regime,
)
from stml.replication.gates import GateResult, evaluate, gate_cell
from stml.replication.ledger import Ledger
from stml.replication.metrics import panel
from stml.replication.nav import nav_discrepancy, nav_from_raw, nav_series
from stml.replication.splits import (
    Split,
    chronological_split,
    embargoed_val,
    get_test,
    n_eff,
    run_length_p90,
)

__all__ = [
    # align
    "AlignResult",
    "align_instrument",
    "align_panel",
    # splits
    "Split",
    "chronological_split",
    "run_length_p90",
    "n_eff",
    "embargoed_val",
    "get_test",
    # baselines
    "always_flat",
    "majority_class",
    "stratified_random",
    "persistence",
    # metrics
    "panel",
    # nav
    "nav_series",
    "nav_discrepancy",
    "nav_from_raw",
    # characterize (C1)
    "alpha_type",
    "lead_lag",
    "regime",
    "cross_asset",
    "drift",
    "model_family_fingerprint",
    "characterize_instrument",
    "characterize_all",
    # archetypes (C2)
    "Archetype",
    "ARCHETYPES",
    "decide",
    "generate_panel",
    # gates (C3)
    "GateResult",
    "evaluate",
    "gate_cell",
    # ledger
    "Ledger",
]
