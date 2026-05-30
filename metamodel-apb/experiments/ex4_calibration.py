"""EX.4 — calibration deep-dive (DIAGNOSTIC, purged-OOS on the modelling sample).

Kelly consumes p̂ directly, so miscalibration -> mis-sizing (part of the AUC≠P&L story). For each
asset class we take purged-OOS predictions from a representative model, measure ECE, and fit
Platt / isotonic on an in-time first-70% split and evaluate ECE on the held-out last-30% — an
honest before/after (no fitting the calibrator on the data it is scored on).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import CLASSES, imputed_modelling_X, modelling_panel, results_dir  # noqa: E402

from alken_metamodel.calibration import (  # noqa: E402
    IsotonicCalibrator,
    PlattCalibrator,
    expected_calibration_error,
    reliability_curve,
)
from alken_metamodel.cross_validation import PurgedKFold  # noqa: E402
from alken_metamodel.evaluation import oos_predictions  # noqa: E402
from alken_metamodel.models import balanced_sample_weight, tree_linear_roster  # noqa: E402
from alken_metamodel.pipeline import PipelineConfig  # noqa: E402
from alken_metamodel.seeding import set_seeds  # noqa: E402


def run() -> None:
    set_seeds(42)
    cfg = PipelineConfig(use_regime=False)
    out = ["# EX.4 — Calibration deep-dive (purged-OOS, lightgbm)\n"]
    for cls in CLASSES:
        pooled, cols, mask = modelling_panel(cls, cfg)
        X, y, t1 = imputed_modelling_X(pooled, cols, mask)
        sw = balanced_sample_weight(y, base=pooled["weight"].to_numpy()[mask])
        cv = PurgedKFold(n_splits=cfg.n_splits, t1=t1, pct_embargo=cfg.pct_embargo)
        oos = oos_predictions(
            lambda: tree_linear_roster(seed=42)["lightgbm"], X, y, cv, sample_weight=sw
        )
        finite = np.isfinite(oos)
        yv, pv = y[finite], oos[finite]
        cut = int(0.7 * len(pv))  # in-time split: fit calibrator on first 70%, score last 30%
        ptr, ytr, pte, yte = pv[:cut], yv[:cut], pv[cut:], yv[cut:]
        ece_raw = expected_calibration_error(yte, pte)
        ece_iso = expected_calibration_error(yte, IsotonicCalibrator().fit(ptr, ytr).transform(pte))
        ece_platt = expected_calibration_error(yte, PlattCalibrator().fit(ptr, ytr).transform(pte))
        curve = reliability_curve(yv, pv, n_bins=5)
        out.append(
            f"## {cls} — n_oos={int(finite.sum())}\n"
            f"- ECE raw={ece_raw:.4f}  Platt={ece_platt:.4f}  isotonic={ece_iso:.4f} "
            f"(held-out last 30%)\n"
            f"- reliability (5 bins):\n```\n{curve.round(3).to_string(index=False)}\n```\n"
        )
        print(f"{cls}: ECE raw={ece_raw:.4f} platt={ece_platt:.4f} iso={ece_iso:.4f}")
    (results_dir() / "ex4_calibration.md").write_text("\n".join(out))
    print("wrote results/ex4_calibration.md")


if __name__ == "__main__":
    run()
