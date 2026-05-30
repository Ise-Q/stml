"""Unified experiment log (XT.2): one append-only, deterministic row per run.

The horse-race + CPCV/nested runs + the EX.* sensitivity sweeps multiply run count. Rather than
scatter result tables, every run appends one row to ``experiment_log.csv`` with a pinned schema
(``RUN_FIELDS`` first, any extra keys alphabetically after). Writes are deterministic — sorted
nowhere by row (append order is the record order), fixed float format, ``\\n`` line endings — so
the log is byte-reproducible and belongs in the reproducible-artifact set.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: Pinned leading columns; runs that omit a field get an empty cell.
RUN_FIELDS = [
    "run_id",
    "asset_class",
    "roster",
    "cv_scheme",
    "reducer",
    "use_macro",
    "n_modelling",
    "best_model",
    "oos_auc",
    "oos_brier",
    "oos_precision",
    "notes",
]

_FLOAT_FORMAT = "%.6f"


def _order_columns(columns) -> list[str]:
    extras = sorted(c for c in columns if c not in RUN_FIELDS)
    return RUN_FIELDS + extras  # always emit the full pinned schema (missing -> empty cell)


def log_run(record: dict, path: str | Path) -> pd.DataFrame:
    """Append ``record`` as one row to the experiment log at ``path`` (created if absent)."""
    path = Path(path)
    row = pd.DataFrame([record])
    combined = (
        pd.concat([pd.read_csv(path), row], ignore_index=True) if path.exists() else row
    )
    combined = combined.reindex(columns=_order_columns(combined.columns))
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False, float_format=_FLOAT_FORMAT, lineterminator="\n")
    return combined
