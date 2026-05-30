"""X.8 — measure the real per-class feature counts (correct the stale '124').

The write-up referenced '124 features'; pass-3 measures the actual pooled column count per class on
the shipped path (causal features + EWMA-HMM regime block + PIT-macro block + instrument one-hots)
so the methodology states the truth rather than a remembered figure. Diagnostic only.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import CLASSES, results_dir  # noqa: E402
from stml.io import load_clean_data  # noqa: E402

from alken_metamodel.pipeline import (  # noqa: E402
    PipelineConfig,
    build_class_panel,
    class_members,
    feature_columns,
)


def run() -> None:
    cfg = PipelineConfig(roster="default", cv_scheme="cpcv", use_macro=True)
    ohlcv, signals = load_clean_data()
    out = ["# X.8 — measured per-class feature counts (shipped path: features + regime + macro)\n"]
    for cls in CLASSES:
        pooled = build_class_panel(ohlcv, signals, class_members(cls), cfg)
        cols = feature_columns(pooled)
        n_macro = sum(c.startswith("macro_") for c in cols)
        n_regime = sum(c.startswith(("regime_", "ewma", "hmm")) for c in cols)
        n_inst = sum(c.startswith("inst_") for c in cols)
        n_core = len(cols) - n_macro - n_regime - n_inst
        line = (
            f"## {cls}: {len(cols)} feature columns\n"
            f"- core={n_core}  macro={n_macro}  regime={n_regime}  inst_onehot={n_inst}\n"
        )
        out.append(line)
        print(f"{cls}: total={len(cols)} core={n_core} macro={n_macro} "
              f"regime={n_regime} inst={n_inst}")
    (results_dir() / "x8_feature_counts.md").write_text("\n".join(out))
    print("wrote results/x8_feature_counts.md")


if __name__ == "__main__":
    run()
