"""`stml.experimental` — modelling pipeline for the BUSI70575 submission.

This package implements the meta-model on top of a primary trading signal:
triple-barrier meta-labels, per-instrument champion selection via
combinatorial purged CV, cluster-level feature importance, calibration,
position sizing, backtest, and significance batteries.

Submodule layout:

* `config.py` — frozen pipeline configuration.
* `seeding.py` — deterministic numpy / random / torch seeds.
* `data_loader.py` — OHLCV + signals + Bloomberg-augmented features.
* `labels.py`, `make_labels.py` — triple-barrier labels.
* `cv.py` — purged k-fold and combinatorial purged CV.
* `features/` — 18 feature families.
* `models.py`, `multitask.py` — model roster + multi-task NN.
* `champion_pipeline.py`, `make_champions.py` — per-instrument champion.
* `importance.py`, `make_importance*.py` — cluster-level importance.
* `calibration.py` — Platt + isotonic calibration.
* `threshold.py` — bootstrap p* gate.
* `sizing.py` — six position-sizing methods + SOPS.
* `volatility.py` — EWMA / Yang-Zhang / Parkinson / GARCH.
* `backtest.py`, `cost_model.py` — barrier-exact backtest + Grinold-Kahn costs.
* `significance.py`, `deflation.py` — PSR / MinTRL / DSR / PT.
* `emit.py`, `make_deliverables.py` — deterministic deliverable CSV writer.

Every optional heavyweight dependency (`torch`, `shap`, `xgboost`,
`lightgbm`) loads its native runtime lazily, so the base import surface
is fast and side-effect free.
"""

from . import _env  # noqa: F401  — sets thread limits before native libs load
