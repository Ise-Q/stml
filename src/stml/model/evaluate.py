"""
evaluate.py
===========
Metrics, plots and the single test-set gate for the meta-model.

The headline metric is **ROC-AUC** (threshold-free, imbalance-robust); we also report
average-precision, Brier (calibration), and F1/precision/recall at a chosen threshold. Per the
project's leakage discipline the test partition is opened exactly once, through
:func:`release_test` -- a tripwire mirroring :func:`stml.replication.splits.get_test` that refuses
to hand back test rows without an explicit ``final_confirmation=True``.

Because the pooled AUC is dominated by the data-rich instruments, :func:`per_instrument_breakdown`
reports AUC per instrument so thin / degenerate names (ho1s, ng1s) are visible rather than hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(y: np.ndarray, proba: np.ndarray, *, threshold: float = 0.5) -> dict:
    """Standard binary metrics for one set of probabilities."""
    y = np.asarray(y).astype(int)
    pred = (proba >= threshold).astype(int)
    n_classes = np.unique(y).size
    return {
        "n": int(y.size),
        "pos_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, proba)) if n_classes == 2 else float("nan"),
        "ap": float(average_precision_score(y, proba)) if n_classes == 2 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, proba)),
    }


def per_instrument_breakdown(
    instruments: np.ndarray, y: np.ndarray, proba: np.ndarray, *, min_n: int = 10
) -> pd.DataFrame:
    """AUC / positive-rate / count per instrument; AUC is NaN where a class is missing."""
    df = pd.DataFrame({"instrument": instruments, "y": np.asarray(y).astype(int), "p": proba})
    rows = []
    for inst, g in df.groupby("instrument", sort=True):
        two = g["y"].nunique() == 2
        rows.append({
            "instrument": inst,
            "n": int(len(g)),
            "pos_rate": float(g["y"].mean()),
            "auc": float(roc_auc_score(g["y"], g["p"])) if (two and len(g) >= min_n) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("instrument").reset_index(drop=True)


def plot_roc_pr(curves: dict[str, tuple[np.ndarray, np.ndarray]], ax=None):
    """Overlay ROC curves for ``{label: (y, proba)}`` on one axis."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    for label, (y, proba) in curves.items():
        fpr, tpr, _ = roc_curve(y, proba)
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc_score(y, proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC")
    ax.legend(loc="lower right", fontsize=8)
    return ax


def plot_calibration(y: np.ndarray, proba: np.ndarray, *, n_bins: int = 10, ax=None):
    """Reliability curve: mean predicted prob vs observed frequency per bin."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(proba, bins) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            xs.append(proba[m].mean())
            ys.append(np.asarray(y)[m].mean())
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(xs, ys, "o-", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration")
    ax.legend(fontsize=8)
    return ax


def release_test(matrix: pd.DataFrame, *, final_confirmation: bool = False) -> pd.DataFrame:
    """Return the test-partition rows -- only behind an explicit confirmation (tripwire).

    Mirrors :func:`stml.replication.splits.get_test`: the test block is the one-time, untouchable
    evaluation set, so accessing it is forced to be a deliberate, auditable act. Call this exactly
    once, in the final-report cell.
    """
    if not final_confirmation:
        raise RuntimeError(
            "Refusing to expose the test partition. It is the final evaluation set and must not "
            "be touched during barrier search or model tuning. Pass final_confirmation=True only "
            "for the one-time final report."
        )
    return matrix[matrix["partition"] == "test"].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Economic evaluation utilities
# ---------------------------------------------------------------------------


def bootstrap_returns(
    train_labels: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Estimate (r_G, r_L) from training-fold event returns via bootstrap.

    Splits ``train_labels["ret"]`` into gains (ret > 0) and losses (ret <= 0), bootstraps each
    group ``n_boot`` times, and returns the mean of bootstrap means. Handles empty groups by
    returning 0.0 for the missing side. r_G >= 0, r_L <= 0.
    """
    rets = train_labels["ret"].to_numpy(dtype=float)
    gains = rets[rets > 0]
    losses = rets[rets <= 0]
    rng = np.random.default_rng(seed)

    if gains.size == 0:
        r_G = 0.0
    else:
        boot_means = np.array([rng.choice(gains, size=gains.size, replace=True).mean()
                               for _ in range(n_boot)])
        r_G = float(boot_means.mean())

    if losses.size == 0:
        r_L = 0.0
    else:
        boot_means = np.array([rng.choice(losses, size=losses.size, replace=True).mean()
                               for _ in range(n_boot)])
        r_L = float(boot_means.mean())

    return r_G, r_L


def decision_threshold(r_G: float, r_L: float) -> float:
    """EV-optimal filter threshold p* = L / (G + L) from guide Part 3.

    Take a trade only when p̂ >= p*, i.e. when expected value is positive:
    p·r_G + (1-p)·r_L > 0  =>  p* = -r_L / (r_G - r_L) = L / (G + L).
    Returns 0.5 when G + L <= 1e-12 (symmetric magnitudes or degenerate case).
    """
    G = float(r_G)
    L = abs(float(r_L))
    if (G + L) <= 1e-12:
        return 0.5
    return L / (G + L)


def _confusion_from_masks(y_true: np.ndarray, pred_mask: np.ndarray) -> dict:
    """Compute tn/fp/fn/tp + derived metrics from boolean prediction mask."""
    y = np.asarray(y_true).astype(int)
    take = np.asarray(pred_mask, dtype=bool)
    tp = int(np.sum((y == 1) & take))
    fp = int(np.sum((y == 0) & take))
    fn = int(np.sum((y == 1) & ~take))
    tn = int(np.sum((y == 0) & ~take))
    n_taken = tp + fp
    precision = tp / n_taken if n_taken > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "precision": float(precision), "recall": float(recall), "n_taken": n_taken}


def adding_zeros_eval(
    y: "np.ndarray | pd.Series",
    p: "np.ndarray | pd.Series",
    *,
    p_star: float,
) -> dict:
    """Dual-confusion-matrix comparison: primary-alone vs primary+meta filter.

    Primary-alone takes every trade (pred=1 everywhere), giving recall=1 and precision=base rate.
    Primary+meta takes trades where p̂ >= p_star. All metrics computed from boolean masks to
    handle the degenerate all-take case without sklearn edge issues.

    Returns keys: p_star, primary_alone, primary_meta (each: tn/fp/fn/tp/precision/recall/n_taken),
    precision_lift, true_positives_missed, false_positives_avoided.
    """
    y_arr = np.asarray(y).astype(int)
    p_arr = np.asarray(p, dtype=float)

    primary_mask = np.ones(len(y_arr), dtype=bool)   # always take
    meta_mask = p_arr >= p_star

    alone = _confusion_from_masks(y_arr, primary_mask)
    meta = _confusion_from_masks(y_arr, meta_mask)

    return {
        "p_star": float(p_star),
        "primary_alone": alone,
        "primary_meta": meta,
        "precision_lift": meta["precision"] - alone["precision"],
        "true_positives_missed": alone["tp"] - meta["tp"],
        "false_positives_avoided": alone["fp"] - meta["fp"],
    }


def nav_sharpe(
    events: pd.DataFrame,
    take_mask: "np.ndarray | pd.Series",
    *,
    side_returns: str = "ret",
) -> dict:
    """NAV and annualized Sharpe for a strategy that takes events where take_mask is True.

    NAV is the **cumulative P&L in return units** -- the *sum* of realized per-event returns -- the
    correct aggregation for fixed-size, non-reinvested bets. Triple-barrier events overlap in time
    (concurrent positions across instruments), so compounding a sequential ``prod(1+r)`` across them
    would spuriously explode; equal-weight additive P&L is the honest equity curve. Sharpe is
    annualized (mean/std * sqrt(252)) over taken returns only; 0.0 if fewer than 2 taken events or
    std == 0.
    """
    mask = np.asarray(take_mask, dtype=bool)
    all_rets = events[side_returns].to_numpy(dtype=float)
    realized = np.where(mask, all_rets, 0.0)

    nav = float(np.nansum(realized))

    taken_rets = all_rets[mask]
    n_taken = int(taken_rets.size)
    if n_taken < 2 or taken_rets.std() < 1e-14:
        sharpe = 0.0
    else:
        sharpe = float(taken_rets.mean() / taken_rets.std() * np.sqrt(252))

    return {"nav": nav, "sharpe": sharpe, "n_taken": n_taken}


def per_instrument_vs_baseline(
    events: pd.DataFrame,
    y: "np.ndarray | pd.Series",
    p: "np.ndarray | pd.Series",
    *,
    p_star: float,
) -> pd.DataFrame:
    """Per-instrument precision / NAV / Sharpe of meta filter vs blind primary.

    ``events`` must carry 'instrument' and 'ret'. Aggregated within each instrument
    (no pooling across instruments to avoid cross-instrument base-rate artefacts).

    Columns: instrument, n, base_rate, precision_primary, precision_meta, precision_lift,
    nav_primary, nav_meta, sharpe_primary, sharpe_meta.
    """
    y_arr = np.asarray(y).astype(int)
    p_arr = np.asarray(p, dtype=float)
    meta_mask = p_arr >= p_star
    primary_mask = np.ones(len(y_arr), dtype=bool)

    rows = []
    for inst, grp in events.groupby("instrument", sort=True):
        idx = grp.index
        y_g = y_arr[events.index.get_indexer(idx)]
        meta_g = meta_mask[events.index.get_indexer(idx)]
        prim_g = primary_mask[events.index.get_indexer(idx)]

        cm_prim = _confusion_from_masks(y_g, prim_g)
        cm_meta = _confusion_from_masks(y_g, meta_g)
        ns_prim = nav_sharpe(grp.reset_index(drop=True), prim_g)
        ns_meta = nav_sharpe(grp.reset_index(drop=True), meta_g)

        rows.append({
            "instrument": inst,
            "n": int(len(grp)),
            "base_rate": float(y_g.mean()),
            "precision_primary": cm_prim["precision"],
            "precision_meta": cm_meta["precision"],
            "precision_lift": cm_meta["precision"] - cm_prim["precision"],
            "nav_primary": ns_prim["nav"],
            "nav_meta": ns_meta["nav"],
            "sharpe_primary": ns_prim["sharpe"],
            "sharpe_meta": ns_meta["sharpe"],
        })

    return pd.DataFrame(rows).reset_index(drop=True)


def predictions_grid(
    primary_signals: pd.DataFrame,
    prob_lookup: pd.DataFrame,
    *,
    date_start,
    date_end,
) -> pd.DataFrame:
    """Build the deliverable CSV grid: one row per (date, instrument) in the window.

    ``primary_signals`` is WIDE: a 'date' column + one column per instrument holding {-1,0,+1}
    (data/primary_signals.csv format). ``prob_lookup`` is LONG with columns ['date','instrument',
    'prob'].

    Steps: melt primary_signals; filter date_start <= date <= date_end; left-join prob_lookup;
    set prediction = prob where signal != 0 else 0.0. Non-zero signals with no prob match default
    to 0.0 (conservative: no model opinion means no confidence -- treated as "do not take").

    Output columns EXACTLY ['date','instrument','prediction'], sorted by (date, instrument).
    Row count == (unique dates in window) x (instrument columns in primary_signals).
    Rerun on the hidden H2-2022 test set by swapping date_start/date_end only.
    """
    sig = primary_signals.copy()
    sig["date"] = pd.to_datetime(sig["date"])

    long = sig.melt(id_vars=["date"], var_name="instrument", value_name="signal")

    date_start_ts = pd.Timestamp(date_start)
    date_end_ts = pd.Timestamp(date_end)
    long = long[(long["date"] >= date_start_ts) & (long["date"] <= date_end_ts)].copy()

    lookup = prob_lookup.copy()
    lookup["date"] = pd.to_datetime(lookup["date"])

    merged = long.merge(lookup[["date", "instrument", "prob"]], on=["date", "instrument"],
                        how="left")

    merged["prediction"] = np.where(
        merged["signal"] != 0,
        merged["prob"].fillna(0.0),
        0.0,
    )
    merged["prediction"] = merged["prediction"].clip(0.0, 1.0)

    out = (
        merged[["date", "instrument", "prediction"]]
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
    return out
