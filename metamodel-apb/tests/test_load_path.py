"""X.11 — load-path conformance guard.

The metamodel must ingest OHLCV through stml's ``load_clean_data`` policy, never a
bespoke loader. This characterises the released-data policy documented in
``reports/missing-data-report.md`` §5–§6: keep the 765 zero-volume weekday settle
rows, drop the 3 Sunday ``2005-05-08`` rows, never forward-fill structural NaNs,
and carry no non-finite / bad-bounds OHLC. A regression here means the pipeline's
data contract has silently changed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from stml.io import load_clean_data


def test_load_clean_data_matches_missing_data_report():
    ohlcv, signals = load_clean_data()

    # 765 zero-volume weekday settles are KEPT (dropping them deletes real prices).
    assert int((ohlcv["volume"] == 0).sum()) == 765

    # The 3 Sunday 2005-05-08 rows (gc1s/hg1s/si1s) are DROPPED; no weekend rows.
    dow = pd.DatetimeIndex(ohlcv["date"]).dayofweek
    assert dow.max() <= 4  # Mon–Fri only
    sunday = ohlcv[ohlcv["date"] == pd.Timestamp("2005-05-08")]
    assert sunday.empty

    # No non-finite / bad-bounds OHLC; no forward-filled NaN in long form.
    ohlc = ohlcv[["open", "high", "low", "close"]].to_numpy(dtype=float)
    assert np.isfinite(ohlc).all()
    assert (ohlcv["high"].to_numpy() >= ohlcv["low"].to_numpy()).all()
    assert (ohlcv["close"].to_numpy() > 0).all()
