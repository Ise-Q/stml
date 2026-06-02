"""`stml.experimental` — Sreeram_experimental build (plan.md golden record).

This package is the single home for the methodology-first meta-labelling pipeline
documented in ``reports/sreeram_experimental/plan.md``. Every cardinal rule (R1-R11)
applies. The pipeline targets per-instrument AUC 0.65-0.75 by combining

  * per-instrument modelling under a shared backbone (per-class XGBoost + multi-task NN),
  * tighter labels via GARCH(1,1) sigma-hat with CPCV-optimised barriers,
  * Bloomberg-augmented feature families (F18-F22) the OHLCV+signal stack cannot derive,

all under the same leakage and significance discipline that
``model/alken-metamodel`` established.

Submodule layout (see plan §6 for the canonical map):

    _env             single-thread native kernels (macOS libomp fix)
    seeding          set_seeds(seed=42) entrypoint
    config           PipelineConfig dataclass (frozen, single source of truth)
    data_loader      OHLCV + signals consolidator (calls stml.io)
    volatility       GK / Parkinson / RS / GARCH(1,1) sigma-hat
    labels           triple-barrier with t+1 entry + uniqueness weights
    features/        catalogued causal feature families (F1-F22)
    bloomberg_ingest BBG cleaning + PIT alignment (S2)
    cv               PurgedKFold + CPCV + nested CPCV
    models           ElasticNet + XGBoost
    multitask        multi-task NN with instrument heads
    importance       Mantegna + clustered MDI / MDA / SHAP (four bug fixes)
    calibration      per-class Platt
    sizing           fractional Kelly + vol target
    cost_model       Grinold-Kahn
    backtest         barrier-exact + cost-aware (Sortino full-T)
    significance     studentised stationary block-bootstrap + PT + MinTRL
    deflation        DSR ladder + CSCV-PBO + MinBTL
    signal_analysis  Pesaran-Timmermann + TM + H-M proxy
    pipeline         run_asset_class orchestrator
    emit             deterministic CSV writer + CLI entry point
"""

from __future__ import annotations

# Eagerly applying single-thread native-kernel pinning at IMPORT time guarantees
# the BLAS / OpenMP env vars are set before any downstream package (xgboost,
# lightgbm, torch) loads its native runtime. Plan §10 + plan §22 (alken parity).
from stml.experimental import _env  # noqa: F401 — side-effect import

__all__: list[str] = []
