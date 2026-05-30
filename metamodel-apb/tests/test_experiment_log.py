"""XT.2 unified experiment log (RED-first).

The horse-race, the CPCV/nested runs and the EX.* sweeps multiply the number of runs; XT.2 keeps
one append-only ``experiment_log.csv`` with a pinned schema so every run is one comparable row.
The log must be deterministic (byte-identical for the same sequence of records) so it can live in
the reproducible-artifact set.
"""

from __future__ import annotations

import pandas as pd

from alken_metamodel.experiment_log import RUN_FIELDS, log_run


def test_log_run_creates_then_appends(tmp_path):
    p = tmp_path / "experiment_log.csv"
    log_run({"run_id": "a", "asset_class": "energy", "best_model": "torch_mlp", "oos_auc": 0.52}, p)
    log_run({"run_id": "b", "asset_class": "metals", "best_model": "lightgbm", "oos_auc": 0.55}, p)
    df = pd.read_csv(p)
    assert len(df) == 2
    assert list(df["run_id"]) == ["a", "b"]
    assert list(df.columns[: len(RUN_FIELDS)]) == RUN_FIELDS  # schema pinned, run_id first


def test_log_run_keeps_extra_keys_after_pinned_fields(tmp_path):
    p = tmp_path / "experiment_log.csv"
    log_run({"run_id": "a", "k_barrier": 1.0, "t_max": 10}, p)
    df = pd.read_csv(p)
    assert "k_barrier" in df.columns and "t_max" in df.columns
    # extras sit after the pinned fields, alphabetically
    extras = list(df.columns[len(RUN_FIELDS):])
    assert extras == sorted(extras)


def test_log_run_is_byte_deterministic(tmp_path):
    records = [
        {"run_id": "a", "asset_class": "energy", "oos_auc": 0.521, "oos_brier": 0.241},
        {"run_id": "b", "asset_class": "metals", "oos_auc": 0.553, "oos_brier": 0.233},
    ]
    for fname in ("x.csv", "y.csv"):
        for r in records:
            log_run(r, tmp_path / fname)
    assert (tmp_path / "x.csv").read_bytes() == (tmp_path / "y.csv").read_bytes()
