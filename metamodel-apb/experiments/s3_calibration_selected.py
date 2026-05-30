"""S3.9 — calibration of the SELECTED models (raw vs Platt, leakage-safe held-out).

Pass-2's EX.4 measured ECE on LightGBM as a common representative. Pass-3 ships per-class Platt
calibration on the *actual selected* models, so we report the real before/after on those models'
purged-OOS modelling predictions: fit Platt on the first 70%, score ECE/Brier/AUC on the held-out
last 30%. Platt is monotone, so the AUC is unchanged — only Brier/ECE (and the Kelly stake) move.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import CLASSES, results_dir  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402
from stml.io import load_clean_data  # noqa: E402

from alken_metamodel.calibration import PlattCalibrator, expected_calibration_error  # noqa: E402
from alken_metamodel.cross_validation import PurgedKFold  # noqa: E402
from alken_metamodel.evaluation import oos_predictions  # noqa: E402
from alken_metamodel.models import balanced_sample_weight  # noqa: E402
from alken_metamodel.pipeline import (  # noqa: E402
    PipelineConfig,
    _roster_factory,
    build_class_panel,
    class_members,
    feature_columns,
    select_model,
)
from alken_metamodel.seeding import set_seeds  # noqa: E402


def run() -> None:
    cfg = PipelineConfig(roster="default", cv_scheme="cpcv", use_macro=True)
    ohlcv, signals = load_clean_data()
    out = ["# S3.9 — selected-model calibration (raw vs Platt, leakage-safe held-out)\n"]
    for cls in CLASSES:
        set_seeds(cfg.seed)
        pooled = build_class_panel(ohlcv, signals, class_members(cls), cfg)
        cols = feature_columns(pooled)
        X, y, t1 = pooled[cols], pooled["bin"].to_numpy(), pooled["t1"]
        mmask = np.asarray(pd.DatetimeIndex(pooled["date"]) <= cfg.modelling_end)
        sw = balanced_sample_weight(y[mmask], base=pooled["weight"].to_numpy()[mmask])
        best, _ = select_model(X[mmask], y[mmask], t1[mmask], sw, cfg)
        cv = PurgedKFold(n_splits=cfg.n_splits, t1=t1[mmask], pct_embargo=cfg.pct_embargo)
        oos = oos_predictions(
            lambda best=best: _roster_factory(cfg)(seed=cfg.seed)[best],
            X[mmask], y[mmask], cv, sample_weight=sw,
        )
        finite = np.isfinite(oos)
        pv, yv = oos[finite], y[mmask][finite]
        cut = int(0.7 * len(pv))
        ptr, ytr, pte, yte = pv[:cut], yv[:cut], pv[cut:], yv[cut:]
        pc = PlattCalibrator().fit(ptr, ytr).transform(pte)
        auc_raw = roc_auc_score(yte, pte) if len(np.unique(yte)) == 2 else float("nan")
        auc_cal = roc_auc_score(yte, pc) if len(np.unique(yte)) == 2 else float("nan")
        out.append(
            f"## {cls} — selected `{best}` (n_oos={int(finite.sum())}, held-out={len(pte)})\n"
            f"- ECE   raw={expected_calibration_error(yte, pte):.4f} -> "
            f"Platt={expected_calibration_error(yte, pc):.4f}\n"
            f"- Brier raw={brier_score_loss(yte, pte):.4f} -> "
            f"Platt={brier_score_loss(yte, pc):.4f}\n"
            f"- AUC   raw={auc_raw:.4f} vs Platt={auc_cal:.4f} (monotone -> unchanged)\n"
        )
        print(f"{cls} [{best}]: ECE {expected_calibration_error(yte, pte):.3f}->"
              f"{expected_calibration_error(yte, pc):.3f}  AUC {auc_raw:.3f}/{auc_cal:.3f}")
    (results_dir() / "s3_calibration_selected.md").write_text("\n".join(out))
    print("wrote results/s3_calibration_selected.md")


if __name__ == "__main__":
    run()
