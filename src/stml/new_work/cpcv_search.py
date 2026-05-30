"""cpcv_search.py — Combinatorial Purged CV barrier optimisation.

Entry points
------------
    from stml.new_work.cpcv_search import cpcv_grid_search, compute_pbo, label_signals_optimised

    results, fold_aucs = cpcv_grid_search(ohlcv, signals)
    pbo = compute_pbo(fold_aucs)
    is_plateau = check_plateau(results, results.iloc[0])
    out = label_signals_optimised(ohlcv, signals, results.iloc[0])

FEATURES
--------
Seven causal price-based features computed at the signal date.  These are a
*documented minimal placeholder*.  Replace with the full stml.harry.features
pipeline (M1–M6 macro, microstructure, signal_trajectory) before Step 4
production runs.  The macro/microstructure features add ~30 predictors that
the price-only set cannot capture (e.g. credit-spread regime, EIA inventory
surprises, bond-vol term structure).

    mom_5d    : log(close[t] / close[t-5])
    mom_20d   : log(close[t] / close[t-20])
    vol_20d   : trailing 20-day daily-log-return std
    vol_60d   : trailing 60-day daily-log-return std
    ret_z_60d : 60-day z-score of the 5-day log-return
    side      : primary signal direction (+1/-1)
    trgt      : GARCH h-day sigma at signal date (regime proxy)

ASSET-CLASS POOLING
-------------------
Thin instruments:
    HO (63 total events, ~44 IS): below the reliable-CPCV threshold.
    NG (120 total events, ~84 IS): borderline.
Both are included in the pooled search alongside their energy-class siblings
(CL, RB) so the joint energy pool has ~1050+ IS events.  The search runs on
ALL 11 instruments pooled; no separate per-instrument CPCV is run because the
goal is a single global config applied to all instruments.

LEAKAGE CONTROLS
----------------
- Search is on IN-SAMPLE only: first 70% of each instrument's signalled dates.
  The terminal 30% is never touched.
- CPCV purges training events whose [date, t1] label window overlaps any test
  block, then embargos ``embargo`` fraction of total span after each test block.
- Features are computed at the signal date using only data at or before that bar.
- GARCH sigmas are strictly causal (fit on returns[0..t-1]; see sigma_garch).

CPCV DETAILS
------------
n_groups=6, k=2 → C(6,2)=15 test paths.
RandomForest: max_depth=4, min_samples_leaf=20, class_weight='balanced'.
Sample weights: return-attributed AFML Ch.4 uniqueness weights.

OBJECTIVE
---------
score = mean(AUC across folds) − 0.5 × |class_balance − 0.5|

The balance term penalises barrier configs that produce near-degenerate label
distributions (where a trivial classifier can achieve high AUC by always
predicting the majority class).

PBO (PROBABILITY OF BACKTEST OVERFITTING)
-----------------------------------------
Implemented via the CSCV protocol (Bailey et al. 2016):
  For each of the 15 CPCV paths treated as OOS:
    1. IS = mean AUC on the remaining 14 paths, per config.
    2. Best IS config = argmax IS performance.
    3. OOS rank = rank of best IS config on this held-out path.
  PBO = fraction of paths where the IS-best config ranks below median OOS.

A high PBO (> 0.5) means the IS-optimal config is likely to be a random
winner, not a robust one.

GRID
----
h ∈ {5, 10, 20}, pt_mult ∈ {1.0, 1.5, 2.0}, sl_mult ∈ {1.0, 1.5, 2.0} → 27 configs.
"""

from __future__ import annotations

from itertools import combinations, product as iterproduct

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from stml.new_work.triple_barrier import (
    INSTRUMENTS,
    MIN_RET,
    _avg_uniqueness,
    _label_instrument_with_trgt,
    sigma_garch,
)

# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────

H_GRID: tuple[int, ...] = (5, 10, 20)
PT_GRID: tuple[float, ...] = (1.0, 1.5, 2.0)
SL_GRID: tuple[float, ...] = (1.0, 1.5, 2.0)
IN_SAMPLE_FRAC: float = 0.70
N_GROUPS: int = 6
K: int = 2
EMBARGO: float = 0.01
BALANCE_LAMBDA: float = 0.5
MIN_IS_EVENTS: int = 40

# Thin instruments pooled with their asset class; noted in output
_THIN = frozenset({"ho1s", "ng1s"})


# ──────────────────────────────────────────────────────────────────────────────
# Feature builder
# ──────────────────────────────────────────────────────────────────────────────

def build_event_features(
    close: pd.Series,
    events_inst: pd.DataFrame,
) -> pd.DataFrame:
    """Causal feature matrix at each event's signal date for one instrument.

    All features are computed from data at or before the signal date.
    ``events_inst`` must have columns 'date' and 'trgt' (GARCH sigma from the
    parent label call).

    NOTE: Replace with stml.harry.features (M1–M6 macro, microstructure,
    signal_trajectory) for production use.  The 7 features here are a minimal
    self-contained set that requires only the price series.

    Returns
    -------
    DataFrame indexed 0..len(events_inst)-1, columns = feature names.
    NaNs imputed to 0 (early-warmup bars before rolling windows are full).
    """
    close = close.sort_index().dropna()
    log_ret = np.log(close).diff()

    # Reindex helper: last known value at event dates
    def _at(series: pd.Series) -> np.ndarray:
        return series.reindex(pd.DatetimeIndex(events_inst["date"]), method="pad").to_numpy(dtype=np.float64)

    mom_5d  = np.log(close / close.shift(5))
    mom_20d = np.log(close / close.shift(20))
    vol_20d = log_ret.rolling(20, min_periods=10).std()
    vol_60d = log_ret.rolling(60, min_periods=30).std()

    ret_5d = log_ret.rolling(5, min_periods=3).sum()
    mu60 = ret_5d.rolling(60, min_periods=30).mean()
    sd60 = ret_5d.rolling(60, min_periods=30).std()
    ret_z_60d = (ret_5d - mu60) / sd60.replace(0.0, np.nan)

    feat = pd.DataFrame(
        {
            "mom_5d":    _at(mom_5d),
            "mom_20d":   _at(mom_20d),
            "vol_20d":   _at(vol_20d),
            "vol_60d":   _at(vol_60d),
            "ret_z_60d": _at(ret_z_60d),
            "side":      events_inst["side"].to_numpy(dtype=np.float64),
            "trgt":      events_inst["trgt"].to_numpy(dtype=np.float64),
        }
    )
    return feat.fillna(0.0)


# ──────────────────────────────────────────────────────────────────────────────
# CombinatorialPurgedKFold
# ──────────────────────────────────────────────────────────────────────────────

class CombinatorialPurgedKFold:
    """Combinatorial Purged K-Fold CV for financial meta-labels.

    López de Prado, AFML Ch. 12.

    Splits ``events`` (sorted by date) into ``n_groups`` equal-size groups by
    time order, then generates all C(n_groups, k) train/test partitions.
    For each partition:
      - Test  = events in the k chosen groups.
      - Train = all other events, MINUS:
          (a) Purge: any training event whose [date, t1] label window starts
              before and overlaps a test-group date range.
          (b) Embargo: training events that START within ``embargo`` fraction
              of the total date span AFTER a test-group end.

    With n_groups=6, k=2 this gives C(6,2)=15 paths.
    """

    def __init__(self, n_groups: int = 6, k: int = 2, embargo: float = 0.01):
        self.n_groups = n_groups
        self.k = k
        self.embargo = embargo

    def split(self, events: pd.DataFrame):
        """Yield (train_idx, test_idx) integer-array pairs.

        Parameters
        ----------
        events : DataFrame with 'date' and 't1' columns, sorted by 'date'.
                 Must be reset-indexed (0..n-1).
        """
        n = len(events)
        if n == 0:
            return

        dates  = pd.to_datetime(events["date"].values)
        t1_arr = pd.to_datetime(events["t1"].values)

        # Assign groups sequentially; last group absorbs any remainder
        gs = n // self.n_groups
        groups = np.full(n, self.n_groups - 1, dtype=int)
        for g in range(self.n_groups - 1):
            groups[g * gs : (g + 1) * gs] = g

        total_days = (dates[-1] - dates[0]) / np.timedelta64(1, "D")
        embargo_td = np.timedelta64(max(1, int(self.embargo * total_days)), "D")

        for test_gs in combinations(range(self.n_groups), self.k):
            test_mask  = np.isin(groups, list(test_gs))
            test_idx   = np.where(test_mask)[0]
            if len(test_idx) == 0:
                continue

            train_mask = ~test_mask

            for tg in test_gs:
                tg_mask = groups == tg
                if not tg_mask.any():
                    continue
                tg_start = dates[tg_mask][0]
                tg_end   = dates[tg_mask][-1]

                # Purge: train events starting BEFORE tg_start with t1 >= tg_start
                purge = (dates < tg_start) & (t1_arr >= tg_start)
                train_mask &= ~purge

                # Embargo: train events starting just after tg_end
                emb_end = tg_end + embargo_td
                emb = (dates > tg_end) & (dates <= emb_end)
                train_mask &= ~emb

            train_idx = np.where(train_mask)[0]

            # Guardrails
            if len(train_idx) < 5 or len(test_idx) < 5:
                continue
            if events.iloc[train_idx]["bin"].nunique() < 2:
                continue
            if events.iloc[test_idx]["bin"].nunique() < 2:
                continue

            yield train_idx, test_idx


# ──────────────────────────────────────────────────────────────────────────────
# Sample weights
# ──────────────────────────────────────────────────────────────────────────────

def _return_weights(events: pd.DataFrame) -> np.ndarray:
    """AFML Ch.4 return-attributed sample weights.

    w_i = avg_uniqueness_i * |ret_i|, normalised so mean = 1 (sklearn convention).
    """
    u = events["avg_uniqueness"].to_numpy(dtype=np.float64)
    r = np.abs(events["ret"].to_numpy(dtype=np.float64))
    w = u * r
    w = np.where(np.isfinite(w), w, 0.0)
    mean_w = w.mean()
    return w / mean_w if mean_w > 0 else np.ones(len(w))


# ──────────────────────────────────────────────────────────────────────────────
# Main grid search
# ──────────────────────────────────────────────────────────────────────────────

def cpcv_grid_search(
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    instruments: list[str] | None = None,
    h_grid: tuple[int, ...] = H_GRID,
    pt_grid: tuple[float, ...] = PT_GRID,
    sl_grid: tuple[float, ...] = SL_GRID,
    in_sample_frac: float = IN_SAMPLE_FRAC,
    n_groups: int = N_GROUPS,
    k: int = K,
    embargo: float = EMBARGO,
    balance_lambda: float = BALANCE_LAMBDA,
    min_is_events: int = MIN_IS_EVENTS,
    garch_refit: int = 21,
    garch_min_obs: int = 500,
    garch_max_window: int = 2000,
    rf_seed: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Run 27-point CPCV grid search over (h, pt_mult, sl_mult).

    Thin instruments (HO, NG) are included in the pooled search alongside all
    other instruments.  The search is conducted on the IS portion only
    (first ``in_sample_frac`` of each instrument's signalled dates).

    Returns
    -------
    results : pd.DataFrame sorted by score (descending).
              Columns: h, pt_mult, sl_mult, mean_auc, std_auc, n_folds,
              class_balance, avg_uniqueness, n_events_is, score.
    fold_aucs : dict mapping (h, pt_mult, sl_mult) → list[float] of per-fold
                AUCs.  Used by compute_pbo.
    """
    if instruments is None:
        instruments = list(INSTRUMENTS)

    ohlcv_l = ohlcv.copy()
    ohlcv_l["date"] = pd.to_datetime(ohlcv_l["date"])
    sigs = signals.copy()
    sigs["date"] = pd.to_datetime(sigs["date"])
    sigs = sigs.set_index("date").sort_index()

    # ── Step 1: pre-compute GARCH sigmas for each (inst, h) ──────────────────
    if verbose:
        print("Pre-computing GARCH sigmas ...")
    garch_cache: dict[tuple, pd.Series] = {}
    for inst in instruments:
        sub = ohlcv_l[ohlcv_l["instrument"] == inst][["date", "close"]]
        if sub.empty:
            continue
        close = sub.sort_values("date").set_index("date")["close"].dropna()
        for h in h_grid:
            if verbose:
                print(f"  {inst} h={h}", end=" ", flush=True)
            garch_cache[(inst, h)] = sigma_garch(
                close, h, refit=garch_refit,
                min_obs=garch_min_obs, max_window=garch_max_window,
            )
            if verbose:
                print("✓")

    # ── Step 2: pre-compute IS close series and instrument split dates ────────
    close_map: dict[str, pd.Series] = {}
    is_cutoff: dict[str, pd.Timestamp] = {}
    for inst in instruments:
        sub = ohlcv_l[ohlcv_l["instrument"] == inst][["date", "close"]]
        if sub.empty:
            continue
        close_map[inst] = sub.sort_values("date").set_index("date")["close"].dropna()
        # IS cutoff = in_sample_frac quantile of signal dates for this instrument
        inst_sig = sigs[inst].dropna() if inst in sigs.columns else pd.Series(dtype=float)
        nonzero_dates = inst_sig[inst_sig != 0].index
        if len(nonzero_dates) >= 2:
            cutoff_pos = int(len(nonzero_dates) * in_sample_frac)
            is_cutoff[inst] = nonzero_dates[min(cutoff_pos, len(nonzero_dates) - 1)]

    # ── Step 3: grid search ───────────────────────────────────────────────────
    results: list[dict] = []
    fold_aucs: dict[tuple, list[float]] = {}
    total = len(h_grid) * len(pt_grid) * len(sl_grid)

    for cfg_idx, (h, pt_mult, sl_mult) in enumerate(iterproduct(h_grid, pt_grid, sl_grid)):
        if verbose:
            print(f"Config {cfg_idx+1}/{total}: h={h} pt={pt_mult} sl={sl_mult} ...", end=" ", flush=True)

        # Collect IS events across all instruments
        ev_parts: list[pd.DataFrame] = []
        feat_parts: list[pd.DataFrame] = []

        for inst in instruments:
            if inst not in sigs.columns or inst not in close_map:
                continue
            close = close_map[inst]
            tgt = garch_cache.get((inst, h))
            if tgt is None:
                continue

            signal = sigs[inst].astype("float64")
            evts = _label_instrument_with_trgt(
                close, signal, tgt, h, pt_mult, sl_mult, min_ret=MIN_RET
            )
            if evts.empty:
                continue

            bar_index = close.sort_index().index
            evts["avg_uniqueness"] = _avg_uniqueness(evts, bar_index)
            evts["instrument"] = inst

            # Restrict to IS
            cutoff = is_cutoff.get(inst)
            if cutoff is not None:
                evts = evts[evts["date"] <= cutoff]
            if len(evts) == 0:
                continue

            feats = build_event_features(close, evts)
            ev_parts.append(evts)
            feat_parts.append(feats)

        if not ev_parts:
            if verbose:
                print("skip (no events)")
            continue

        all_evts = (
            pd.concat(ev_parts, ignore_index=True)
            .sort_values("date")
            .reset_index(drop=True)
        )
        all_feats = np.vstack([f.to_numpy() for f in feat_parts])
        # Re-sort features to match sorted events order
        sort_order = pd.concat(
            [df.assign(_idx=i) for i, df in enumerate(ev_parts)],
            ignore_index=True
        ).sort_values("date").index.tolist()
        # Build feature matrix in same sorted order as all_evts
        all_feats = np.concatenate([feat_parts[i].to_numpy() for i in range(len(feat_parts))], axis=0)
        # Rebuild sorted
        _ev_with_feat = pd.concat(
            [ev.assign(**{f"f{j}": feat_parts[i].iloc[:, j].values
                         for j in range(feat_parts[i].shape[1])})
             for i, ev in enumerate(ev_parts)],
            ignore_index=True,
        ).sort_values("date").reset_index(drop=True)
        feat_cols = [c for c in _ev_with_feat.columns if c.startswith("f")]
        all_feats = _ev_with_feat[feat_cols].to_numpy(dtype=np.float64)
        all_evts = _ev_with_feat.drop(columns=feat_cols)

        if len(all_evts) < min_is_events:
            if verbose:
                print(f"skip ({len(all_evts)} IS events < {min_is_events})")
            continue

        class_balance = float(all_evts["bin"].mean())

        # CPCV
        cpcv = CombinatorialPurgedKFold(n_groups=n_groups, k=k, embargo=embargo)
        fold_auc_list: list[float] = []

        for tr_idx, te_idx in cpcv.split(all_evts):
            X_tr, y_tr = all_feats[tr_idx], all_evts["bin"].values[tr_idx]
            X_te, y_te = all_feats[te_idx], all_evts["bin"].values[te_idx]
            w_tr = _return_weights(all_evts.iloc[tr_idx])

            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=rf_seed,
                n_jobs=-1,
            )
            rf.fit(X_tr, y_tr, sample_weight=w_tr)
            prob = rf.predict_proba(X_te)[:, 1]
            fold_auc_list.append(float(roc_auc_score(y_te, prob)))

        if not fold_auc_list:
            if verbose:
                print("skip (no valid folds)")
            continue

        mean_auc = float(np.mean(fold_auc_list))
        std_auc  = float(np.std(fold_auc_list))
        score    = mean_auc - balance_lambda * abs(class_balance - 0.5)

        key = (h, pt_mult, sl_mult)
        fold_aucs[key] = fold_auc_list

        results.append(
            {
                "h":              h,
                "pt_mult":        pt_mult,
                "sl_mult":        sl_mult,
                "mean_auc":       round(mean_auc, 4),
                "std_auc":        round(std_auc, 4),
                "n_folds":        len(fold_auc_list),
                "class_balance":  round(class_balance, 4),
                "avg_uniqueness": round(float(all_evts["avg_uniqueness"].mean()), 4),
                "n_events_is":    len(all_evts),
                "score":          round(score, 4),
            }
        )

        if verbose:
            thin_note = " [incl. HO/NG pooled]" if any(i in all_evts.get("instrument", pd.Series()).unique() for i in _THIN) else ""
            print(f"AUC={mean_auc:.4f} bal={class_balance:.3f} score={score:.4f}{thin_note}")

    if not results:
        return pd.DataFrame(), {}

    results_df = (
        pd.DataFrame(results)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    return results_df, fold_aucs


# ──────────────────────────────────────────────────────────────────────────────
# PBO via CSCV
# ──────────────────────────────────────────────────────────────────────────────

def compute_pbo(fold_aucs: dict) -> float:
    """Probability of Backtest Overfitting via the CSCV protocol.

    For each of the available CPCV paths (folds), treat that fold as OOS.
    Identify the IS-best config (highest mean AUC on all OTHER folds).
    PBO = fraction of paths where that config's OOS AUC is below the median
    OOS AUC across all configs on the same path.

    Parameters
    ----------
    fold_aucs : dict mapping config_key → list[float] of per-fold AUCs.

    Returns
    -------
    float in [0, 1].  PBO > 0.5 is a red flag for overfitting.
    """
    if not fold_aucs:
        return float("nan")

    keys = list(fold_aucs.keys())
    # Build matrix: configs × folds (pad with NaN where a config has fewer folds)
    n_folds_max = max(len(v) for v in fold_aucs.values())
    mat = np.full((len(keys), n_folds_max), np.nan)
    for i, k in enumerate(keys):
        vals = fold_aucs[k]
        mat[i, : len(vals)] = vals

    oos_below_median_count = 0
    valid_paths = 0

    for f in range(n_folds_max):
        oos_col = mat[:, f]
        # Configs that have a valid OOS value for this fold
        valid = ~np.isnan(oos_col)
        if valid.sum() < 2:
            continue

        # IS = mean of all OTHER folds
        other_folds = [j for j in range(n_folds_max) if j != f]
        is_means = np.nanmean(mat[:, other_folds], axis=1)
        is_means[~valid] = np.nan  # only consider configs with OOS value

        best_is_cfg = int(np.nanargmax(is_means))
        if np.isnan(oos_col[best_is_cfg]):
            continue

        median_oos = float(np.nanmedian(oos_col[valid]))
        if oos_col[best_is_cfg] < median_oos:
            oos_below_median_count += 1
        valid_paths += 1

    if valid_paths == 0:
        return float("nan")
    return oos_below_median_count / valid_paths


# ──────────────────────────────────────────────────────────────────────────────
# Plateau check
# ──────────────────────────────────────────────────────────────────────────────

def check_plateau(results: pd.DataFrame, best_row: pd.Series, tolerance: float = 0.01) -> bool:
    """True if the best config sits on a performance plateau.

    A plateau means >= 3 configs have scores within ``tolerance`` of the best.
    A spike means the best is isolated (< 3 neighbours close to it), which is
    a sign of overfitting / optimistic IS selection.

    Parameters
    ----------
    results   : full results DataFrame from cpcv_grid_search (sorted by score).
    best_row  : the row for the chosen best config (typically results.iloc[0]).
    tolerance : score gap within which a config counts as "close". Default 0.01.
    """
    best_score = float(best_row["score"])
    n_close = (np.abs(results["score"] - best_score) <= tolerance).sum()
    return int(n_close) >= 3


# ──────────────────────────────────────────────────────────────────────────────
# Apply chosen config to full history
# ──────────────────────────────────────────────────────────────────────────────

_OPT_OUTPUT_COLS: tuple[str, ...] = (
    "date", "instrument", "side", "t1", "ret", "bin",
    "trgt", "h", "pt_mult", "sl_mult", "sigma_method", "avg_uniqueness",
)


def label_signals_optimised(
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    best_config: pd.Series | dict,
    *,
    garch_refit: int = 21,
    garch_min_obs: int = 500,
    garch_max_window: int = 2000,
    instruments: list[str] | None = None,
) -> pd.DataFrame:
    """Apply the CPCV-chosen barrier config to the FULL signal history.

    Parameters
    ----------
    best_config : row from the results DataFrame (or dict with h, pt_mult, sl_mult).

    Returns
    -------
    DataFrame with columns matching _OPT_OUTPUT_COLS (same schema as
    label_signals_fixed).
    """
    if isinstance(best_config, dict):
        h       = int(best_config["h"])
        pt_mult = float(best_config["pt_mult"])
        sl_mult = float(best_config["sl_mult"])
    else:
        h       = int(best_config["h"])
        pt_mult = float(best_config["pt_mult"])
        sl_mult = float(best_config["sl_mult"])

    if instruments is None:
        instruments = list(INSTRUMENTS)

    ohlcv_l = ohlcv.copy()
    ohlcv_l["date"] = pd.to_datetime(ohlcv_l["date"])
    sigs = signals.copy()
    sigs["date"] = pd.to_datetime(sigs["date"])
    sigs = sigs.set_index("date").sort_index()

    try:
        import arch  # noqa: F401
        method = "garch"
    except ImportError:
        method = "ewma_fallback"

    parts: list[pd.DataFrame] = []
    for inst in instruments:
        if inst not in sigs.columns:
            continue
        sub = ohlcv_l[ohlcv_l["instrument"] == inst][["date", "close"]]
        if sub.empty:
            continue
        close = sub.sort_values("date").set_index("date")["close"].dropna()
        signal = sigs[inst].astype("float64")

        trgt_series = sigma_garch(
            close, h, refit=garch_refit,
            min_obs=garch_min_obs, max_window=garch_max_window,
        )
        evts = _label_instrument_with_trgt(
            close, signal, trgt_series, h, pt_mult, sl_mult, min_ret=MIN_RET
        )
        if evts.empty:
            continue

        bar_index = close.sort_index().index
        evts["avg_uniqueness"] = _avg_uniqueness(evts, bar_index)
        evts["instrument"] = inst
        evts["h"] = h
        evts["pt_mult"] = pt_mult
        evts["sl_mult"] = sl_mult
        evts["sigma_method"] = method
        parts.append(evts)

    if not parts:
        return pd.DataFrame(columns=list(_OPT_OUTPUT_COLS))

    out = pd.concat(parts, ignore_index=True)
    return out[list(_OPT_OUTPUT_COLS)]


__all__ = [
    "build_event_features",
    "CombinatorialPurgedKFold",
    "cpcv_grid_search",
    "compute_pbo",
    "check_plateau",
    "label_signals_optimised",
]
