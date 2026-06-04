"""Central configuration for the strategy-construction package.

The single source of truth for boundary, vol parameters, and paths.
BOUNDARY is derived from split_config.GLOBAL_CUT — the same cutoff used by
the meta-model — so the sizing layer is aligned to the identical training window.
Override via env var STML_BOUNDARY (ISO date string); grader slides to 2022-07-01.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Leakage boundary — derived from the meta-model's locked cutoff, not hardcoded.
# STML_BOUNDARY env override lets the grader slide the window forward.
# ---------------------------------------------------------------------------

def _resolve_boundary() -> pd.Timestamp:
    env = os.environ.get("STML_BOUNDARY")
    if env:
        return pd.Timestamp(env)
    from stml.new_work import split_config  # noqa: PLC0415
    return split_config.GLOBAL_CUT


BOUNDARY: pd.Timestamp = _resolve_boundary()

# ---------------------------------------------------------------------------
# Portfolio construction parameters.
# ---------------------------------------------------------------------------

SIGMA_TGT: float = 0.10     # annualised vol target
MAX_LEVERAGE: float = 10.0  # hard clamp on |w|
TRADING_DAYS: float = 252.0

# Vol estimator — EWMA only.  Do not change to yang_zhang / garch / gjr.
VOL_METHOD: str = "ewma_close"

# EWMA span (bars).
EWMA_SPAN: int = 60

# Annualised vol floor applied after EWMA (prevents blow-up in vol-targeting).
VOL_FLOOR: float = 0.02

# Lookback window — used only for warm-up guard / min_periods logic.
LOOKBACK_L: int = 20

# Minimum bars before a vol estimate is emitted (warm-up guard).
VOL_MIN_PERIODS: int = 10

# ---------------------------------------------------------------------------
# Instruments.
# ---------------------------------------------------------------------------

INSTRUMENTS: list[str] = [
    "cl1s", "es1s", "fesx1s", "gc1s", "hg1s",
    "ho1s", "ng1s", "nq1s", "pl1s", "rb1s", "si1s",
]

# Equity indices that use GJR-GARCH when VOL_METHOD == "gjr".
EQUITY_INSTRUMENTS: frozenset[str] = frozenset(["es1s", "nq1s", "fesx1s"])

# ---------------------------------------------------------------------------
# Paths — resolved relative to repo root.
# ---------------------------------------------------------------------------


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"Could not locate repo root from {here}")


REPO_ROOT: Path = _find_repo_root()
DATA_DIR: Path = REPO_ROOT / "data"
NEW_WORK_DIR: Path = REPO_ROOT / "src" / "stml" / "new_work"
OUTPUTS_DIR: Path = NEW_WORK_DIR / "outputs"

# OOF training-period calibrated probabilities — NOT YET GENERATED.
# Schema: date (= t_signal), instrument, p_hat_oof
# Coverage: date < BOUNDARY only.
# Generate by running CPCV(6,2) on training events in run_locked_strategy.py.
OOF_PROBA_PATH: Path = DATA_DIR / "oof_meta_probabilities.csv"

# Sealed OOS test predictions (already generated).
OOS_PROBA_PATH: Path = OUTPUTS_DIR / "metamodel_predictions.csv"

# Output paths for strategy weights and evaluation results.
STRATEGY_WEIGHTS_PATH: Path = OUTPUTS_DIR / "strategy_weights_new.csv"
EVAL_RESULTS_PATH: Path = OUTPUTS_DIR / "strategy_eval.csv"
