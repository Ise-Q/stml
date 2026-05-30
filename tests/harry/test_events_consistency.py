"""Drift guard: persisted ``events.csv`` must match ``events.meta.json``.

This test reads only the two persisted artifacts under
``results/harry/``. It does not regenerate. The role is to fail loudly
if anyone modifies the CSV without re-running
``python -m stml.harry.persist_events`` to refresh the meta JSON.

The Step 4 pipeline will consume ``events.csv`` directly; the drift-guard
keeps Step 3 / Step 4 / Step 5 honest about what they trained on.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVENTS_CSV = _REPO_ROOT / "results" / "harry" / "events.csv"
_EVENTS_META = _REPO_ROOT / "results" / "harry" / "events.meta.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def events_and_meta() -> tuple[pd.DataFrame, dict]:
    if not _EVENTS_CSV.exists():
        pytest.skip(f"missing {_EVENTS_CSV} — run python -m stml.harry.persist_events")
    if not _EVENTS_META.exists():
        pytest.skip(f"missing {_EVENTS_META} — run python -m stml.harry.persist_events")
    df = pd.read_csv(_EVENTS_CSV)
    meta = json.loads(_EVENTS_META.read_text())
    return df, meta


def test_csv_sha256_matches_meta(events_and_meta):
    _, meta = events_and_meta
    actual = _sha256(_EVENTS_CSV)
    assert actual == meta["sha256_of_csv"], (
        f"events.csv has drifted vs events.meta.json: "
        f"expected {meta['sha256_of_csv']!r}, got {actual!r}. "
        f"Re-run python -m stml.harry.persist_events to refresh."
    )


def test_n_events_matches_meta(events_and_meta):
    df, meta = events_and_meta
    assert len(df) == meta["n_events"], (
        f"events row-count drifted: csv has {len(df)}, meta says {meta['n_events']}"
    )


def test_per_instrument_balance_matches_meta(events_and_meta):
    df, meta = events_and_meta
    expected_map = meta["label_balance_per_instrument"]
    actual_instruments = set(df["instrument"].unique())
    expected_instruments = set(expected_map.keys())
    assert actual_instruments == expected_instruments, (
        f"instrument set drifted: csv has {sorted(actual_instruments)}, "
        f"meta has {sorted(expected_instruments)}"
    )
    for inst, expected in expected_map.items():
        grp = df[df["instrument"] == inst]
        assert int(len(grp)) == expected["n_events"], f"{inst}: n_events drift"
        assert int((grp["side"] == 1).sum()) == expected["n_long"], f"{inst}: n_long drift"
        assert int((grp["side"] == -1).sum()) == expected["n_short"], f"{inst}: n_short drift"
        assert int((grp["label"] == 1).sum()) == expected["n_label_1"], f"{inst}: n_label_1 drift"
        assert int((grp["label"] == 0).sum()) == expected["n_label_0"], f"{inst}: n_label_0 drift"


def test_meta_has_label_config_hash(events_and_meta):
    """The config hash must be present so Step 4's pipeline can verify the
    checkpoint matches its requested ``TripleBarrierConfig`` before
    consuming the CSV (or pass ``--regenerate-events`` to override)."""
    _, meta = events_and_meta
    assert "label_config_hash" in meta
    assert isinstance(meta["label_config_hash"], str)
    assert len(meta["label_config_hash"]) == 64  # SHA256 hex length


def test_meta_records_label_config_values(events_and_meta):
    _, meta = events_and_meta
    cfg = meta.get("label_config")
    assert cfg is not None
    for key in ("h", "pt_mult", "sl_mult", "vol_span"):
        assert key in cfg
    # Spot check defaults — committed checkpoint uses the default config.
    assert cfg["h"] == 10
    assert cfg["pt_mult"] == 1.0
    assert cfg["sl_mult"] == 1.0
    assert cfg["vol_span"] == 100
