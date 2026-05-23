"""Feature-engineering layer for the triple-barrier metamodel.

Builds a rich, C1-informed, strictly leakage-safe feature set for the 11
instruments, computed at each non-zero-signal trade day using only information
``<= t`` (execution is next-day ``r_{t+1}``):

* engineered E-class features (counter-trend, vol/dispersion, signal-derived,
  momentum-contrast, microstructure, calendar) — :mod:`stml.metamodel.features`;
* filtered (causal) GMM + Markov-switching regime posteriors —
  :mod:`stml.metamodel.regime_features`;
* unsupervised latent structure (PCA, clustering, a shallow dense autoencoder
  benchmarked vs PCA) — :mod:`stml.metamodel.latent`;
* cross-sectional rank / pair-correlation — :mod:`stml.metamodel.xsection`.

Fitted artifacts (GMM/Markov/PCA/AE/scaler) are fit on the FE-train partition
only (ends 2021-07-01) and transformed causally; engineered features are proven
causal by truncation-invariance, fitted features by a fit-provenance assertion.
Scope is pinned per model type (:mod:`stml.metamodel.scope`, D5). The pipeline
(:mod:`stml.metamodel.pipeline`) assembles a persisted tidy-long matrix with
train/val/test provenance; the catalog (:mod:`stml.metamodel.catalog`) documents
every column.

See the work plan at ``.omc/plans/feature-engineering-metamodel-plan.md``.
"""

from __future__ import annotations

from stml.metamodel.build_features import (
    build_feature_matrix,
    compute_redundancy,
    main,
)
from stml.metamodel.catalog import (
    CATALOG,
    FeatureSpec,
    assert_coverage,
    render_catalog,
)
from stml.metamodel.features import (
    assemble_engineered,
    f1_counter_trend,
    f2_vol_dispersion,
    f5_signal_derived,
    f6_momentum_contrast,
    f7_microstructure,
    f8_calendar,
)
from stml.metamodel.latent import (
    LatentBundle,
    fit_latent,
    transform_latent,
)
from stml.metamodel.pipeline import FeaturePipeline
from stml.metamodel.regime_features import (
    RegimeBundle,
    fit_regime,
    transform_regime,
)
from stml.metamodel.scope import (
    FLOOR,
    InstrumentScope,
    build_scope,
    persist_scope,
)
from stml.metamodel.xsection import xsection_features

__all__ = [
    # pipeline + build API
    "FeaturePipeline",
    "build_feature_matrix",
    "compute_redundancy",
    "main",
    # engineered features (E-class)
    "assemble_engineered",
    "f1_counter_trend",
    "f2_vol_dispersion",
    "f5_signal_derived",
    "f6_momentum_contrast",
    "f7_microstructure",
    "f8_calendar",
    # regime (F3, TF)
    "RegimeBundle",
    "fit_regime",
    "transform_regime",
    # latent (F4, TF)
    "LatentBundle",
    "fit_latent",
    "transform_latent",
    # cross-section (F9)
    "xsection_features",
    # scope registry (D5)
    "FLOOR",
    "InstrumentScope",
    "build_scope",
    "persist_scope",
    # catalog (graded artifact)
    "CATALOG",
    "FeatureSpec",
    "assert_coverage",
    "render_catalog",
]
