"""split_config.py — Single source of truth for the global train/test split.

Global cut: 2021-10-06 (70th percentile of 4,917 pooled events across 11 instruments
            in the new label file; 70th-pct date = 2021-10-04, rounded to next trading
            day 2021-10-06 for backward compatibility).
Embargo end: 2021-10-20 (cut + 10 trading days).

Train: date <= GLOBAL_CUT and t1 < GLOBAL_CUT (purge label-window crossers).
Test:  date > EMBARGO_END (SEALED — do not read during training).
"""
from __future__ import annotations

import os

import pandas as pd

GLOBAL_CUT: pd.Timestamp = pd.Timestamp(os.environ.get("STML_TRAIN_CUT", "2021-10-06"))
EMBARGO_END: pd.Timestamp = pd.Timestamp(os.environ.get("STML_EMBARGO_END", "2021-10-20"))


def apply_train_mask(events_df: pd.DataFrame) -> pd.DataFrame:
    """Train rows: date <= cut AND t1 < cut (purge boundary-crossing events)."""
    mask = (events_df["date"] <= GLOBAL_CUT) & (events_df["t1"] < GLOBAL_CUT)
    return events_df.loc[mask].copy().reset_index(drop=True)


def apply_test_mask(events_df: pd.DataFrame) -> pd.DataFrame:
    """Test rows: date > embargo_end. SEALED — do not call during training."""
    mask = events_df["date"] > EMBARGO_END
    return events_df.loc[mask].copy().reset_index(drop=True)
