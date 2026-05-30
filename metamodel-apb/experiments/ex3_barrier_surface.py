"""EX.3 — (k, T_max) barrier-width sensitivity surface (DIAGNOSTIC-ONLY, cl1s).

Grids the barrier half-width k and vertical horizon T_max, relabelling cl1s on the MODELLING
sample only, and reports the label balance + a quick purged-CV AUC per cell. Strengthens the §2
barrier justification and the deflated-metric story. **Diagnostic only** — the chosen cell never
feeds back into the locked deliverable config (that would snoop the Jan–Jun rehearsal half).
Scoped to one representative energy instrument to stay tractable.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import results_dir  # noqa: E402
from stml.io import load_clean_data  # noqa: E402

from alken_metamodel.cross_validation import PurgedKFold  # noqa: E402
from alken_metamodel.evaluation import cross_val_evaluate  # noqa: E402
from alken_metamodel.features import (  # noqa: E402
    assemble_instrument_features,
    daily_barrier_sigma,
    filter_signal_days,
)
from alken_metamodel.models import balanced_sample_weight, tree_linear_roster  # noqa: E402
from alken_metamodel.pipeline import PipelineConfig  # noqa: E402
from alken_metamodel.seeding import set_seeds  # noqa: E402
from alken_metamodel.triple_barrier import triple_barrier_labels  # noqa: E402

INSTRUMENT = "cl1s"
K_GRID = (0.5, 1.0, 1.5)
TMAX_GRID = (5, 10, 20)


def run() -> None:
    set_seeds(42)
    cfg = PipelineConfig()
    ohlcv, signals = load_clean_data()
    ohlcv_inst = ohlcv[ohlcv["instrument"] == INSTRUMENT]
    signal = signals.set_index("date")[INSTRUMENT].sort_index()
    signal.index = pd.DatetimeIndex(signal.index)
    feats = assemble_instrument_features(ohlcv_inst, signal)
    sigma = daily_barrier_sigma(feats)
    close = ohlcv_inst.set_index("date")["close"].sort_index().astype(float)
    close.index = pd.DatetimeIndex(close.index)
    feats_events = filter_signal_days(feats, signal)
    feature_cols = list(feats.columns)

    rows = []
    for k in K_GRID:
        for tmax in TMAX_GRID:
            labels = triple_barrier_labels(
                close, signal, sigma, pt_sl=(k, k), max_holding=tmax
            )
            panel = feats_events.join(labels, how="inner").dropna(subset=["bin"])
            panel = panel[panel.index <= cfg.modelling_end]
            x_raw = panel[feature_cols]
            imp = SimpleImputer(strategy="median", keep_empty_features=True)
            X = pd.DataFrame(imp.fit_transform(x_raw), index=x_raw.index, columns=feature_cols)
            X = X[[c for c in X.columns if X[c].var() > 1e-12]]
            y = panel["bin"].to_numpy()
            t1 = panel["t1"]
            sw = balanced_sample_weight(y, base=panel["weight"].to_numpy())
            cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
            res = cross_val_evaluate(
                lambda: tree_linear_roster(seed=42)["lightgbm"], X, y, cv, sample_weight=sw
            )
            rows.append(
                {
                    "k": k,
                    "t_max": tmax,
                    "n": len(y),
                    "pos_rate": round(float(y.mean()), 3),
                    "cv_auc": round(float(np.nanmean(res["auc"].to_numpy())), 4),
                }
            )
            print(f"k={k} t_max={tmax}: n={len(y)} pos={y.mean():.3f} "
                  f"auc={np.nanmean(res['auc'].to_numpy()):.4f}")
    table = pd.DataFrame(rows)
    out = [
        f"# EX.3 — (k, T_max) barrier surface for {INSTRUMENT} (diagnostic, modelling sample)\n",
        "Pos-rate and purged-CV AUC per cell. NOT used to retune the locked config.\n",
        "```\n" + table.to_string(index=False) + "\n```\n",
    ]
    (results_dir() / "ex3_barrier_surface.md").write_text("\n".join(out))
    print("wrote results/ex3_barrier_surface.md")


if __name__ == "__main__":
    run()
