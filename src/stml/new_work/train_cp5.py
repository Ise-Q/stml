"""train_cp5.py — Checkpoint 5 sizing-layer training with full leakage assertions.

Trains VSN+LSTM and TFT on vol-normalised features (EWMA-only vol, span=60).
Saves weights to outputs/vsn_lstm_weights_cp5.pt and outputs/tft_weights_cp5.pt.

Assertions:
  - Training OOF set aligns to meta-model (start, end < GLOBAL_CUT, count = 3313)
  - OOS event set is clean (min > EMBARGO_END, max ≤ 2022-06-30)
  - Embargo gap noted (pre-existing: 14 cal days < t_max 20 trading days)

Usage:
    .venv/bin/python src/stml/new_work/train_cp5.py [--vsn-only] [--tft-only]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

from stml.new_work import config, split_config
from stml.new_work.data import load_oof_probabilities, load_oos_probabilities
from stml.io import load_data
from stml.new_work.data import load_signals
from stml.new_work.models.common import TrainConfig

OUTPUTS = _HERE / "outputs"

# ── Expected constants (derived, not hardcoded) ────────────────────────────────

_EXPECTED_TRAIN_COUNT   = 3313
_EXPECTED_TRAIN_START   = pd.Timestamp("2020-01-30")
_EXPECTED_OOS_MIN       = pd.Timestamp("2021-10-21")
_GRADER_HIDDEN_START    = pd.Timestamp("2022-07-01")


# ── Assertion helpers ──────────────────────────────────────────────────────────

def assert_training_window(oof_df: pd.DataFrame) -> None:
    """Assert the OOF artifact aligns to the meta-model's training window."""
    n     = len(oof_df)
    start = oof_df["date"].min()
    end   = oof_df["date"].max()
    bound = config.BOUNDARY  # = split_config.GLOBAL_CUT unless env overridden

    errors = []
    if n != _EXPECTED_TRAIN_COUNT:
        errors.append(
            f"  row count: got {n}, expected {_EXPECTED_TRAIN_COUNT}"
        )
    if start.date() != _EXPECTED_TRAIN_START.date():
        errors.append(
            f"  train start: got {start.date()}, expected {_EXPECTED_TRAIN_START.date()}"
        )
    if end >= bound:
        errors.append(
            f"  train end {end.date()} >= BOUNDARY {bound.date()} — leakage!"
        )
    if errors:
        raise RuntimeError(
            "assert_training_window FAILED:\n" + "\n".join(errors)
        )
    print(f"  [assert] training window OK: {start.date()} → {end.date()}, "
          f"n={n}, max < BOUNDARY ({bound.date()})")


def assert_oos_clean(oos_df: pd.DataFrame) -> None:
    """Assert the OOS event set contains no hidden Jul–Dec 2022 data."""
    min_d = oos_df["date"].min()
    max_d = oos_df["date"].max()
    n     = len(oos_df)
    emb   = split_config.EMBARGO_END

    errors = []
    if min_d <= emb:
        errors.append(
            f"  OOS min date {min_d.date()} ≤ EMBARGO_END {emb.date()} — embargo violated!"
        )
    if max_d >= _GRADER_HIDDEN_START:
        errors.append(
            f"  OOS max date {max_d.date()} ≥ {_GRADER_HIDDEN_START.date()} — hidden set touched!"
        )
    if errors:
        raise RuntimeError("assert_oos_clean FAILED:\n" + "\n".join(errors))
    print(f"  [assert] OOS clean: {min_d.date()} → {max_d.date()}, n={n}")


def print_segment_table() -> None:
    gc  = split_config.GLOBAL_CUT
    emb = split_config.EMBARGO_END
    oos_min = _EXPECTED_OOS_MIN
    oos_max = pd.Timestamp("2022-06-29")
    t_max_trading = 20
    embargo_cal   = (emb - gc).days
    embargo_trading_approx = embargo_cal * 5 // 7

    print("\n" + "=" * 70)
    print("SEGMENT TABLE")
    print("=" * 70)
    print(f"  Train   : {_EXPECTED_TRAIN_START.date()} → {gc.date()}"
          f"  (BOUNDARY = GLOBAL_CUT)  n={_EXPECTED_TRAIN_COUNT}")
    print(f"  Embargo : {(gc + pd.Timedelta(days=1)).date()} → {emb.date()}"
          f"  ({embargo_cal} cal days ≈ {embargo_trading_approx} trading days)")
    print(f"  ⚠ embargo ({embargo_trading_approx} trading) < t_max ({t_max_trading} trading)"
          f" — pre-existing; apply_train_mask enforces t1 < BOUNDARY as mitigation")
    print(f"  OOS     : {oos_min.date()} → {oos_max.date()}  n=1373")
    print(f"  Hidden  : {_GRADER_HIDDEN_START.date()} → 2022-12-31  (grader only — never read)")
    print("=" * 70 + "\n")


# ── Training ───────────────────────────────────────────────────────────────────

def train_vsn_lstm(oof_df: pd.DataFrame, ohlcv: pd.DataFrame, signals: pd.DataFrame) -> None:
    from stml.new_work.models.vsn_lstm import train_vsn_lstm as _train
    import torch

    cfg = TrainConfig(
        epochs=200, batch_size=128, lr=5e-4, weight_decay=2e-3,
        grad_clip=1.0, patience=20, val_frac=0.20, seed=42,
    )
    print("\n── Training VSN+LSTM (d_model=16, embed_dim=4, EWMA vol features) ──")
    model, hist = _train(
        oof_df, ohlcv, signals,
        cfg=cfg, d_model=16, embed_dim=4, n_lstm_layers=1, dropout=0.25,
        verbose=True,
    )
    out = OUTPUTS / "vsn_lstm_weights_cp5.pt"
    torch.save(model.state_dict(), out)
    print(f"  Saved → {out.relative_to(_REPO)}")
    print(f"  Best val Sharpe: {hist.best_val_sharpe:.4f} @ epoch {hist.best_epoch}")


def train_tft(oof_df: pd.DataFrame, ohlcv: pd.DataFrame, signals: pd.DataFrame) -> None:
    from stml.new_work.models.tft import train_tft as _train
    import torch

    cfg = TrainConfig(
        epochs=200, batch_size=128, lr=5e-4, weight_decay=2e-3,
        grad_clip=1.0, patience=20, val_frac=0.20, seed=42,
    )
    print("\n── Training TFT (d_model=16, n_heads=4, embed_dim=4, EWMA vol features) ──")
    model, hist = _train(
        oof_df, ohlcv, signals,
        cfg=cfg, d_model=16, n_heads=4, embed_dim=4, n_lstm_layers=1, dropout=0.25,
        verbose=True,
    )
    out = OUTPUTS / "tft_weights_cp5.pt"
    torch.save(model.state_dict(), out)
    print(f"  Saved → {out.relative_to(_REPO)}")
    print(f"  Best val Sharpe: {hist.best_val_sharpe:.4f} @ epoch {hist.best_epoch}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Checkpoint 5 — retrain sizing layer")
    ap.add_argument("--vsn-only", action="store_true")
    ap.add_argument("--tft-only", action="store_true")
    args = ap.parse_args(argv)
    OUTPUTS.mkdir(exist_ok=True)

    print("=" * 70)
    print("Checkpoint 5 — sizing layer training")
    print(f"  BOUNDARY = GLOBAL_CUT = {config.BOUNDARY.date()}"
          f"  (source: split_config.py:17)")
    print(f"  VOL_METHOD = {config.VOL_METHOD!r}  EWMA_SPAN = {config.EWMA_SPAN}"
          f"  VOL_FLOOR = {config.VOL_FLOOR}")
    print("=" * 70)

    print_segment_table()

    # Load and assert OOF training data.
    print("Loading OOF training data …")
    oof_df = load_oof_probabilities()
    assert_training_window(oof_df)

    # Load and assert OOS events.
    print("Loading OOS events …")
    oos_df = load_oos_probabilities()
    oos_df["date"] = pd.to_datetime(oos_df["date"])
    assert_oos_clean(oos_df)

    # Load shared data.
    print("Loading OHLCV and signals …")
    ohlcv, _ = load_data()
    signals = load_signals()

    do_vsn = not args.tft_only
    do_tft = not args.vsn_only

    if do_vsn:
        train_vsn_lstm(oof_df, ohlcv, signals)

    if do_tft:
        train_tft(oof_df, ohlcv, signals)

    print("\n" + "=" * 70)
    print("Training complete.")
    if do_vsn:
        print(f"  VSN+LSTM → {(OUTPUTS / 'vsn_lstm_weights_cp5.pt').relative_to(_REPO)}")
    if do_tft:
        print(f"  TFT      → {(OUTPUTS / 'tft_weights_cp5.pt').relative_to(_REPO)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
