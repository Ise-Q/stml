"""`stml.experimental.features` — causal feature catalogue.

Plan §8 Stage 2 deliverable. One package; every feature self-registers in
:mod:`catalog` so the assembler can iterate uniformly.

Submodules:

    catalog          FeatureSpec / FeatureRegistry / assemble_features()
    closed_form      F1 / F2 / F5 / F6 / F7 / F8 / F10 / F12 — OHLCV+signal
    risk_drift_regime F15 (path) / F16 (drift) / F17 (HMM)
    cross_asset      F9 (cross-section) / F21 (relative value)
    macro            F11 (the macro, REFORMULATED as 63-day rolling ranks)
    bloomberg        F18 (term) / F19 (options IV) / F22 (event flags) — BBG
"""

from __future__ import annotations

from stml.experimental.features import (  # noqa: F401 — side-effect registrations
    bloomberg,
    closed_form,
    cross_asset,
    macro,
    risk_drift_regime,
)
from stml.experimental.features.catalog import (
    REGISTRY,
    FeatureSpec,
    assemble_features,
    family_counts,
    registered_names,
)

__all__ = [
    "REGISTRY",
    "FeatureSpec",
    "assemble_features",
    "family_counts",
    "registered_names",
]
