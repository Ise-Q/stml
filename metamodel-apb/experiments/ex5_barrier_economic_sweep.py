"""EX.5 — economically-ranked triple-barrier labelling sweep (DIAGNOSTIC-ONLY).

Sweeps barrier configurations (vol estimator, width, vertical horizon) + baselines (fixed-time-
horizon, trend scanning, always-act floor) on the MODELLING sample only, ranks them by **net
Sharpe** from the barrier-exact cost-aware backtest (NOT label accuracy), and writes a per-class
results table + recommendation. The chosen config NEVER feeds back into the locked deliverable
config — that would snoop the Jan–Jun rehearsal half (CLAUDE.md). Logic lives in
``alken_metamodel.barrier_sweep``; this is the thin runner. Run:

    uv run --project metamodel-apb python experiments/ex5_barrier_economic_sweep.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# Pin single-threaded OpenMP/BLAS before the native libs (xgboost+lightgbm) initialise — mirrors
# conftest.py; avoids the macOS libomp duplicate-runtime crash and removes kernel nondeterminism.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from _common import CLASSES, results_dir  # noqa: E402
from stml.io import load_clean_data  # noqa: E402

from alken_metamodel.barrier_sweep import (  # noqa: E402
    build_configs,
    recommend,
    run_class,
)
from alken_metamodel.experiment_log import log_run  # noqa: E402
from alken_metamodel.pipeline import PipelineConfig, load_embargo_days  # noqa: E402
from alken_metamodel.seeding import set_seeds  # noqa: E402

_ANCHOR = "anchor_shipped_1_1_h10"
_FLOOR = "always_act_floor"


def _flags(table: pd.DataFrame, rec: dict) -> list[str]:
    """Surface configs whose edge traces to imbalance / cost-fragility / floor, not signal."""
    flags = []
    if rec["best_net_sharpe_1x"] <= rec["floor_net_sharpe_1x"]:
        flags.append(
            f"HEADLINE: no labelling beats the always-act floor net of costs "
            f"(best {rec['best_net_sharpe_1x']:.3f} ≤ floor {rec['floor_net_sharpe_1x']:.3f})."
        )
    # cost fragility: winner at 1x is a loser at 2x
    top1 = table.loc[table["net_sharpe_1x"].idxmax()]
    if top1["net_sharpe_2x"] < table["net_sharpe_2x"].median():
        flags.append(
            f"COST-FRAGILE: {top1['config']} leads at 1× but drops below median at 2× cost."
        )
    # imbalance: extreme pos_rate with high precision = edge from base rate, not skill
    imb = table[(table["pos_rate"] < 0.2) | (table["pos_rate"] > 0.8)]
    for _, r in imb.iterrows():
        flags.append(
            f"IMBALANCE: {r['config']} pos_rate={r['pos_rate']:.2f} (edge may be base-rate)."
        )
    return flags


def _render(asset_class, table, rec, robust, flags, common_n) -> str:
    cols = [
        "config",
        "stage",
        "family",
        "n",
        "pos_rate",
        "pct_pt",
        "pct_sl",
        "pct_timeout",
        "auc",
        "precision_taken",
        "f1",
        "brier",
        "net_sharpe_0x",
        "net_sharpe_1x",
        "net_sharpe_2x",
        "max_dd_1x",
        "ann_turnover",
        "avg_holding",
        "adj_sharpe_lag0",
        "adj_sharpe_lag1",
        "adj_sharpe_lag2",
    ]
    cols = [c for c in cols if c in table.columns]
    body = table[cols].sort_values("net_sharpe_1x", ascending=False).to_string(index=False)
    lines = [
        f"# EX.5 — economically-ranked barrier labelling sweep: {asset_class} (DIAGNOSTIC)\n",
        f"Ranked by **net Sharpe (1× cost)** on the COMMON event set (n={common_n}); modelling "
        f"sample only (≤ modelling_end). Label accuracy (AUC/precision) reported, NOT ranked on. "
        f"NOT fed back into the locked config.\n",
        "```\n" + body + "\n```\n",
        "## Recommendation (per-stage winner, tie-broken toward simpler/more balanced)\n",
        f"- vol: `{rec['vol_winner']}`  ·  width: `{rec['width_winner']}`  ·  "
        f"horizon: `{rec['horizon_winner']}`\n",
        f"- best net Sharpe 1× = {rec['best_net_sharpe_1x']:.3f}  vs  always-act floor = "
        f"{rec['floor_net_sharpe_1x']:.3f}\n",
        "> **OFAT caveat.** Each stage varies one factor with the others at the LdP anchor "
        "(GK, (1,1), h=10); the three winners were NOT measured jointly. Given the small "
        "modelling sample, treat this as a defensible *protocol* + honestly-uncertain result, "
        "not a validated optimum. Joint-confirmation of the winner is a documented follow-up.\n",
        "> **Placebo-in-time (`adj_sharpe_lag{0,1,2}`).** ORACLE labeling-quality diagnostic "
        "(uses the realised label, NEVER ranked): per-trade Sharpe of the label-filtered signal at "
        "lag-0 (contemporaneous, untradeable), lag-1 (first tradeable bar), lag-2. A trustworthy "
        "label has **lag-1 ≫ lag-0**; a strong lag-0 (or a tiny-`h` config whose barrier overlaps "
        "its own bar) signals a circular/self-fulfilling label, not real forward edge.\n",
        "> **Do NOT import the jay/triple-barrier-label per-instrument picks into the locked "
        "config:** those were validated on the 2022-H1 hold-out, which this experiment's "
        "anti-snooping rule forbids — re-derive any per-instrument geometry ≤ modelling_end.\n",
        "## Robustness (xgb → lightgbm swap on anchor + stage winners + floor)\n",
        "```\n" + robust + "\n```\n",
        "## Flags\n",
        ("\n".join(f"- {f}" for f in flags) if flags else "- none\n"),
    ]
    return "\n".join(lines)


def run() -> None:
    set_seeds(42)
    cfg = PipelineConfig()
    ohlcv, signals = load_clean_data()
    embargo = load_embargo_days()
    all_configs = build_configs()
    log_path = results_dir() / "ex5_experiment_log.csv"

    for ac in CLASSES:
        table = run_class(ac, ohlcv, signals, cfg, embargo, model_name="xgboost")
        rec = recommend(table)
        common_n = table.attrs.get("common_n", -1)

        # robustness: re-score anchor + stage winners + floor under lightgbm
        winner_names = {
            _ANCHOR,
            _FLOOR,
            rec["vol_winner"],
            rec["width_winner"],
            rec["horizon_winner"],
        }
        subset = [c for c in all_configs if c.name in winner_names]
        table_lgb = run_class(
            ac, ohlcv, signals, cfg, embargo, configs=subset, model_name="lightgbm"
        )
        robust = (
            table_lgb[["config", "net_sharpe_1x", "precision_taken", "pos_rate"]]
            .sort_values("net_sharpe_1x", ascending=False)
            .to_string(index=False)
        )

        flags = _flags(table, rec)
        out = results_dir() / f"ex5_barrier_economic_sweep_{ac}.md"
        out.write_text(_render(ac, table, rec, robust, flags, common_n))
        table.to_csv(
            results_dir() / f"ex5_barrier_economic_sweep_{ac}.csv", index=False, lineterminator="\n"
        )
        log_run(
            {
                "experiment": "ex5_barrier_economic_sweep",
                "asset_class": ac,
                "common_n": common_n,
                "best_net_sharpe_1x": round(rec["best_net_sharpe_1x"], 4),
                "floor_net_sharpe_1x": round(rec["floor_net_sharpe_1x"], 4),
                "vol_winner": rec["vol_winner"],
                "width_winner": rec["width_winner"],
                "horizon_winner": rec["horizon_winner"],
            },
            log_path,
        )
        print(
            f"[{ac}] common_n={common_n} best1x={rec['best_net_sharpe_1x']:.3f} "
            f"floor={rec['floor_net_sharpe_1x']:.3f} -> wrote {out.name}"
        )


if __name__ == "__main__":
    run()
