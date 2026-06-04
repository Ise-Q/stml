"""Persist Harry's events CSV with a drift-guard JSON.

The events panel is regenerated live from OHLCV + signals each time
``get_meta_labels`` is called. To prevent silent drift between the
committed checkpoint and a re-run that uses different config or a
different code path, this module persists *two* artifacts:

* ``events.csv``       — the canonical labelled events panel.
* ``events.meta.json`` — metadata that lets a downstream test verify the
  CSV has not drifted: SHA256 of the CSV bytes, the
  :class:`TripleBarrierConfig` hash, total event count, and the
  per-instrument label balance.

The drift-guard test lives in ``tests/harry/test_events_consistency.py``
and reads only the persisted files. It does not regenerate, so it remains
deterministic across pandas / numpy versions.

CLI::

    python -m stml.harry.persist_events                 # default config
    python -m stml.harry.persist_events --pt-mult 1.5   # asymmetric variant
    python -m stml.harry.persist_events --out path/...  # custom location
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stml.harry.labels import (
    DEFAULT_H,
    DEFAULT_PT_MULT,
    DEFAULT_SL_MULT,
    DEFAULT_VOL_SPAN,
    TripleBarrierConfig,
    get_meta_labels,
)

__all__ = [
    "compute_config_hash",
    "compute_file_sha256",
    "build_meta",
    "main",
]


def compute_config_hash(cfg: TripleBarrierConfig) -> str:
    """SHA256 of a canonicalised JSON view of the TripleBarrierConfig.

    Uses ``sort_keys=True`` and a fixed numeric repr so two callers on
    different platforms produce the same hash for the same config.
    """
    payload = {
        "h": int(cfg.h),
        "pt_mult": float(cfg.pt_mult),
        "sl_mult": float(cfg.sl_mult),
        "vol_span": int(cfg.vol_span),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_file_sha256(path: Path) -> str:
    """Streaming SHA256 of the file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_meta(
    events: pd.DataFrame,
    cfg: TripleBarrierConfig,
    csv_path: Path,
) -> dict:
    """Compose the meta dict.

    ``csv_path`` must point to the already-written events.csv; its SHA256
    is computed from disk to match what a downstream reader would see.
    """
    per_inst: dict[str, dict[str, int]] = {}
    for inst, grp in events.groupby("instrument"):
        per_inst[str(inst)] = {
            "n_events": int(len(grp)),
            "n_long": int((grp["side"] == 1).sum()),
            "n_short": int((grp["side"] == -1).sum()),
            "n_label_1": int((grp["label"] == 1).sum()),
            "n_label_0": int((grp["label"] == 0).sum()),
        }
    return {
        "n_events": int(len(events)),
        "label_balance_per_instrument": per_inst,
        "sha256_of_csv": compute_file_sha256(csv_path),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label_config_hash": compute_config_hash(cfg),
        "label_config": {
            "h": int(cfg.h),
            "pt_mult": float(cfg.pt_mult),
            "sl_mult": float(cfg.sl_mult),
            "vol_span": int(cfg.vol_span),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Persist Harry's events CSV + drift-guard JSON.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/harry/events.csv"),
        help="Destination CSV. Sibling events.meta.json is written next to it.",
    )
    parser.add_argument("--h", type=int, default=DEFAULT_H)
    parser.add_argument("--pt-mult", type=float, default=DEFAULT_PT_MULT)
    parser.add_argument("--sl-mult", type=float, default=DEFAULT_SL_MULT)
    parser.add_argument("--vol-span", type=int, default=DEFAULT_VOL_SPAN)
    args = parser.parse_args(argv)

    from stml.io import load_clean_data  # imported here so unit tests skip the cost

    ohlcv, signals = load_clean_data()
    cfg = TripleBarrierConfig(
        h=args.h, pt_mult=args.pt_mult, sl_mult=args.sl_mult,
        vol_span=args.vol_span,
    )
    events = get_meta_labels(
        ohlcv,
        signals,
        h=cfg.h,
        pt_mult=cfg.pt_mult,
        sl_mult=cfg.sl_mult,
        vol_span=cfg.vol_span,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.out, index=False)

    meta = build_meta(events, cfg, args.out)
    meta_path = args.out.parent / "events.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"wrote {args.out} ({len(events)} rows)")
    print(f"wrote {meta_path}")
    print(f"label_config_hash = {meta['label_config_hash']}")
    print(f"sha256_of_csv      = {meta['sha256_of_csv']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
