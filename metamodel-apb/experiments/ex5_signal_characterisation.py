"""EX.5 — primary-signal characterisation (DIAGNOSTIC, modelling window).

Characterises the PROVIDED primary signal per instrument: turnover (flip frequency), directional
hit-rate vs the next-day return, information coefficient, the fundamental-law IR = IC·√breadth,
and a high-vs-low-vol regime split. Contextualises how much edge the metamodel can even refine.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import results_dir  # noqa: E402
from stml.io import INSTRUMENTS, load_clean_data, load_returns_panel  # noqa: E402

from alken_metamodel.pipeline import PipelineConfig  # noqa: E402
from alken_metamodel.signal_analysis import (  # noqa: E402
    information_coefficient,
    information_ratio,
    signal_hit_rate,
    signal_turnover,
)


def run() -> None:
    cfg = PipelineConfig()
    ohlcv, signals = load_clean_data()
    rets = load_returns_panel(kind="log")
    sig_wide = signals.set_index("date")
    sig_wide.index = pd.DatetimeIndex(sig_wide.index)
    rows = []
    for inst in INSTRUMENTS:
        if inst not in sig_wide.columns or inst not in rets.columns:
            continue
        sig = sig_wide[inst]
        fwd = rets[inst].shift(-1)  # position held at t earns the t+1 return
        df = pd.concat({"sig": sig, "fwd": fwd}, axis=1).dropna()
        df = df[df.index <= cfg.modelling_end]
        active = df["sig"] != 0
        vol = rets[inst].rolling(20).std().reindex(df.index)
        hi = vol > vol.median()
        ic = information_coefficient(df.loc[active, "sig"], df.loc[active, "fwd"])
        rows.append(
            {
                "instrument": inst,
                "n_active": int(active.sum()),
                "turnover": round(signal_turnover(df["sig"]), 4),
                "hit_rate": round(signal_hit_rate(df["sig"], df["fwd"]), 4),
                "ic": round(ic, 4),
                "ir_ann": round(information_ratio(ic, int(active.sum())), 3),
                "hit_hivol": round(signal_hit_rate(df.loc[hi, "sig"], df.loc[hi, "fwd"]), 4),
                "hit_lovol": round(signal_hit_rate(df.loc[~hi, "sig"], df.loc[~hi, "fwd"]), 4),
            }
        )
    table = pd.DataFrame(rows)
    out = [
        "# EX.5 — Primary-signal characterisation (modelling window)\n",
        "IR = IC·√breadth with breadth = number of active bets (theoretical upper bound).\n",
        "```\n" + table.to_string(index=False) + "\n```\n",
    ]
    (results_dir() / "ex5_signal_characterisation.md").write_text("\n".join(out))
    print(table.to_string(index=False))
    print("wrote results/ex5_signal_characterisation.md")


if __name__ == "__main__":
    run()
