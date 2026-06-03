"""model_finalisation.py — Full vs pruned/reduced/pca-reduced feature variants under CPCV.

Compares three variants for cl1s, two for es1s/ho1s/rb1s; adds ng1s exploratory
(full + pca_reduced with per-fold PCA — no leakage).
Uses development (pre-boundary) data only; holdout untouched.
Results saved to outputs/finalisation/.

Usage
-----
    python -m stml.new_work.model_finalisation          # skip already-cached variants
    python -m stml.new_work.model_finalisation --force  # rerun everything
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

from stml.new_work.cpcv_search import CombinatorialPurgedKFold
from stml.new_work.feature_importance import RANDOM_SEED
from stml.new_work.model_comparison import (
    CPCV_EMBARGO,
    CPCV_K,
    CPCV_N_GROUPS,
    _tune_fit_rf,
    _tune_fit_xgb,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUTS   = _HERE / "outputs"
IMP_BASE  = OUTPUTS / "importance"
MC_BASE   = OUTPUTS / "model_comparison"
CACHE_DIR = MC_BASE / "_cache"
FIN_OUT   = OUTPUTS / "finalisation"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

META_COLS = frozenset({
    "date", "instrument", "side", "t1", "ret", "bin",
    "trgt", "h", "pt_mult", "sl_mult", "sigma_method", "avg_uniqueness",
})

CHAMPIONS: dict[str, dict] = {
    "cl1s": {"group": "cl1s",         "model": "xgb", "target_inst": "cl1s"},
    "es1s": {"group": "es1s",         "model": "rf",  "target_inst": "es1s"},
    "ho1s": {"group": "energy_cl_ho", "model": "rf",  "target_inst": "ho1s"},
    "rb1s": {"group": "energy_all",   "model": "xgb", "target_inst": "rb1s"},
    "ng1s": {"group": "ng1s",         "model": "rf",  "target_inst": "ng1s"},
}

# Variants per instrument (in evaluation order: full first so it anchors the rule)
VARIANTS: dict[str, list[str]] = {
    "cl1s": ["full", "pruned", "reduced"],
    "es1s": ["full", "pruned"],
    "ho1s": ["full", "pruned"],
    "rb1s": ["full", "pruned"],
    "ng1s": ["full", "pca_reduced"],
}

# Variants marked exploratory: not subject to the AUC-delta decision rule;
# kept or discarded based on parsimony + domain prior rather than significance.
EXPLORATORY: dict[str, set[str]] = {
    "ho1s": {"pruned"},
    "ng1s": {"pca_reduced"},   # no confirmed signal; variant is diagnostic only
}

# ---------------------------------------------------------------------------
# NG1S PCA-reduced variant specification
# ---------------------------------------------------------------------------

# Feature clusters: read members at runtime from cluster_membership.csv;
# n_components capped at min(3, cluster_size) per cluster.
NG1S_PCA_SPEC: dict = {
    "passthrough":   ["f7_oi_z_20", "hmm_macro_entropy"],
    "pca_clusters":  [
        {"name": "C4_f11",  "n_components": 3},
        {"name": "C14_f6",  "n_components": 3},
    ],
}

# ---------------------------------------------------------------------------
# Feature list builders
# ---------------------------------------------------------------------------

def _feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def _get_variant_features(
    inst: str,
    variant: str,
    events_df: pd.DataFrame,
) -> list[str]:
    """Return the ordered feature column list for (inst, variant)."""
    all_feats     = _feat_cols(events_df)
    all_feat_set  = set(all_feats)
    mem           = pd.read_csv(IMP_BASE / inst / "cluster_membership.csv")

    if variant == "full":
        return all_feats

    def _cluster_feats(names: list[str]) -> list[str]:
        rows = mem.loc[mem["cluster"].isin(names), "feature"].tolist()
        return [f for f in rows if f in all_feat_set]

    # ── cl1s ────────────────────────────────────────────────────────────────
    if inst == "cl1s":
        if variant == "pruned":
            # All members of the two significant clusters + the next-best by MDA
            return _cluster_feats(["C4_f2", "C12_f11", "C3_f6"])

        if variant == "reduced":
            # C4_f2: single representative = top-1 by mean|SHAP| (f2_vol_60)
            c4  = pd.read_csv(IMP_BASE / inst / "within_cluster_C4_f2.csv")
            c4_rep = (
                c4.sort_values("mean_shap_mag", ascending=False)
                  .iloc[0]["feature"]
            )
            # C12_f11: top-3 by mean|SHAP|
            c12 = pd.read_csv(IMP_BASE / inst / "within_cluster_C12_f11.csv")
            top3_c12 = (
                c12.sort_values("mean_shap_mag", ascending=False)
                   .head(3)["feature"]
                   .tolist()
            )
            # C3_f6: single representative = top-1 by mean|SHAP| (f13_prob_timeout)
            c3  = pd.read_csv(IMP_BASE / inst / "within_cluster_C3_f6.csv")
            c3_rep = (
                c3.sort_values("mean_shap_mag", ascending=False)
                  .iloc[0]["feature"]
            )
            feats = [c4_rep] + top3_c12 + [c3_rep]
            # dedup, preserve order, restrict to columns present in events_df
            seen: set[str] = set()
            ordered: list[str] = []
            for f in feats:
                if f in all_feat_set and f not in seen:
                    ordered.append(f)
                    seen.add(f)
            return ordered

    # ── es1s ────────────────────────────────────────────────────────────────
    if inst == "es1s":
        if variant == "pruned":
            return _cluster_feats(["F5_signal", "C8_f11", "C2_f11"])

    # ── ho1s ────────────────────────────────────────────────────────────────
    if inst == "ho1s":
        if variant == "pruned":
            # Energy-fundamentals (C13_f11: energy stocks/Baltic/OI-divergence) +
            # vol cluster containing f2_vol_60 (C5_f2) +
            # macro risk-factor cluster (C10_f11_lowfreq_macro: VIX/rates/HY) +
            # instrument dummy — mirrors the crude (cl1s) reduced-variant logic
            return _cluster_feats(
                ["C13_f11", "C5_f2", "C10_f11_lowfreq_macro", "F_instrument"]
            )

    # ── rb1s ────────────────────────────────────────────────────────────────
    if inst == "rb1s":
        if variant == "pruned":
            # Signal + vol + macro clusters, plus F_instrument (one-hot dummies)
            return _cluster_feats(["F5_signal", "C5_f2", "C11_f11", "F_instrument"])

    # ── ng1s ────────────────────────────────────────────────────────────────
    if inst == "ng1s":
        if variant == "pca_reduced":
            # Returns CONCEPTUAL names (not raw columns) for n_features reporting.
            # The actual per-fold matrix is built in evaluate_pca_variant().
            names: list[str] = list(NG1S_PCA_SPEC["passthrough"])
            for cs in NG1S_PCA_SPEC["pca_clusters"]:
                members = mem.loc[mem["cluster"] == cs["name"], "feature"].tolist()
                members = [f for f in members if f in all_feat_set]
                n_comp  = min(cs["n_components"], len(members))
                names  += [f"{cs['name']}_pc{i+1}" for i in range(n_comp)]
            return names

    raise ValueError(f"No feature spec for inst={inst!r}, variant={variant!r}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2 or len(y_true) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def evaluate_variant(
    events_df: pd.DataFrame,
    feat_cols: list[str],
    model_type: str,
    target_inst: str,
) -> dict[str, Any]:
    """CPCV(6,2) evaluation on one feature variant.

    For pooled models, OOS metrics are computed on the target instrument's
    test-fold slice only (same convention as champion_importance.py).
    """
    ev_meta   = events_df[["date", "t1", "bin", "instrument", "avg_uniqueness"]].copy()
    X         = events_df[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y         = events_df["bin"].to_numpy(dtype=int)
    instruments = events_df["instrument"].to_numpy()
    is_pooled = events_df["instrument"].nunique() > 1

    cpcv = CombinatorialPurgedKFold(
        n_groups=CPCV_N_GROUPS, k=CPCV_K, embargo=CPCV_EMBARGO,
    )

    aucs: list[float] = []
    llosses: list[float] = []
    briers: list[float] = []
    skipped = 0

    for fold_i, (tr_idx, te_idx) in enumerate(cpcv.split(ev_meta)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]
        events_tr  = ev_meta.iloc[tr_idx].reset_index(drop=True)

        if len(np.unique(y_tr)) < 2:
            skipped += 1
            continue

        # Target-instrument mask for pooled scoring
        te_insts    = instruments[te_idx]
        target_mask = (te_insts == target_inst) if is_pooled else None

        y_tgt = y_te if target_mask is None else y_te[target_mask]
        if len(np.unique(y_tgt)) < 2 or len(y_tgt) < 2:
            skipped += 1
            continue

        try:
            if model_type == "rf":
                model, _ = _tune_fit_rf(X_tr, y_tr, events_tr)
            elif model_type == "xgb":
                model, _ = _tune_fit_xgb(X_tr, y_tr, events_tr)
            else:
                raise ValueError(f"Unknown model_type={model_type!r}")
        except Exception as exc:
            print(f"      fold {fold_i} fit failed: {exc}")
            skipped += 1
            continue

        prob_all = model.predict_proba(X_te)[:, 1]
        prob     = prob_all if target_mask is None else prob_all[target_mask]

        auc = _safe_auc(y_tgt, prob)
        if auc is None:
            skipped += 1
            continue

        aucs.append(auc)
        pc = np.clip(prob, 1e-7, 1 - 1e-7)
        llosses.append(float(log_loss(y_tgt, pc)))
        briers.append(float(brier_score_loss(y_tgt, prob)))

    n = len(aucs)
    return {
        "auc_mean":     float(np.mean(aucs))    if aucs else float("nan"),
        "auc_std":      float(np.std(aucs))     if aucs else float("nan"),
        "logloss_mean": float(np.mean(llosses)) if llosses else float("nan"),
        "logloss_std":  float(np.std(llosses))  if llosses else float("nan"),
        "brier_mean":   float(np.mean(briers))  if briers else float("nan"),
        "brier_std":    float(np.std(briers))   if briers else float("nan"),
        "n_folds":      n,
        "skipped":      skipped,
    }


# ---------------------------------------------------------------------------
# PCA-reduced variant evaluation (per-fold fit — no leakage)
# ---------------------------------------------------------------------------

def evaluate_pca_variant(
    events_df: pd.DataFrame,
    pca_spec: dict,
    model_type: str,
    target_inst: str,
) -> dict[str, Any]:
    """CPCV(6,2) evaluation where cluster features are replaced by per-fold PCA scores.

    For each CPCV fold:
      - StandardScaler and PCA are fit on TRAINING rows only for each cluster.
      - Both train and test rows are projected onto those components.
      - Passthrough features are appended raw (no scaling).
    This is strictly leak-free: no test-fold information enters the scaler or PCA.
    """
    # Resolve cluster members from cluster_membership.csv (existing, do not re-cluster)
    mem_df = pd.read_csv(IMP_BASE / target_inst / "cluster_membership.csv")
    all_feat_cols = _feat_cols(events_df)
    all_feat_set  = set(all_feat_cols)
    feat_idx      = {c: j for j, c in enumerate(all_feat_cols)}

    passthrough = pca_spec["passthrough"]
    pt_indices  = [feat_idx[c] for c in passthrough if c in feat_idx]

    cluster_specs: list[dict] = []
    for cs in pca_spec["pca_clusters"]:
        members = mem_df.loc[mem_df["cluster"] == cs["name"], "feature"].tolist()
        members = [f for f in members if f in all_feat_set]
        n_comp  = min(cs["n_components"], len(members))
        if n_comp < 1:
            print(f"      [warn] cluster {cs['name']}: no valid members, skipping")
            continue
        cluster_specs.append({
            "name":        cs["name"],
            "col_indices": [feat_idx[m] for m in members],
            "n_components": n_comp,
            "n_members":   len(members),
        })
        print(f"      {cs['name']}: {len(members)} members → {n_comp} PCs")

    ev_meta     = events_df[["date", "t1", "bin", "instrument", "avg_uniqueness"]].copy()
    X_raw       = events_df[all_feat_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y           = events_df["bin"].to_numpy(dtype=int)
    instruments = events_df["instrument"].to_numpy()
    is_pooled   = events_df["instrument"].nunique() > 1

    cpcv = CombinatorialPurgedKFold(
        n_groups=CPCV_N_GROUPS, k=CPCV_K, embargo=CPCV_EMBARGO,
    )

    aucs: list[float]    = []
    llosses: list[float] = []
    briers: list[float]  = []
    skipped = 0

    for fold_i, (tr_idx, te_idx) in enumerate(cpcv.split(ev_meta)):
        y_tr, y_te    = y[tr_idx], y[te_idx]
        events_tr     = ev_meta.iloc[tr_idx].reset_index(drop=True)

        if len(np.unique(y_tr)) < 2:
            skipped += 1
            continue

        te_insts    = instruments[te_idx]
        target_mask = (te_insts == target_inst) if is_pooled else None
        y_tgt       = y_te if target_mask is None else y_te[target_mask]
        if len(np.unique(y_tgt)) < 2 or len(y_tgt) < 2:
            skipped += 1
            continue

        # ── Build feature matrices: passthrough + per-cluster PCA scores ──
        parts_tr: list[np.ndarray] = []
        parts_te: list[np.ndarray] = []

        if pt_indices:
            parts_tr.append(X_raw[tr_idx][:, pt_indices])
            parts_te.append(X_raw[te_idx][:, pt_indices])

        for cs in cluster_specs:
            X_cl_tr = X_raw[tr_idx][:, cs["col_indices"]]
            X_cl_te = X_raw[te_idx][:, cs["col_indices"]]

            scaler  = StandardScaler()
            X_sc_tr = scaler.fit_transform(X_cl_tr)   # fit on train only
            X_sc_te = scaler.transform(X_cl_te)       # transform test with train params

            pca_fold = PCA(n_components=cs["n_components"], random_state=RANDOM_SEED)
            pca_fold.fit(X_sc_tr)                      # fit on train only
            parts_tr.append(pca_fold.transform(X_sc_tr))
            parts_te.append(pca_fold.transform(X_sc_te))

        X_tr = np.hstack(parts_tr)
        X_te = np.hstack(parts_te)

        try:
            if model_type == "rf":
                model, _ = _tune_fit_rf(X_tr, y_tr, events_tr)
            elif model_type == "xgb":
                model, _ = _tune_fit_xgb(X_tr, y_tr, events_tr)
            else:
                raise ValueError(f"Unknown model_type={model_type!r}")
        except Exception as exc:
            print(f"      fold {fold_i} fit failed: {exc}")
            skipped += 1
            continue

        prob_all = model.predict_proba(X_te)[:, 1]
        prob     = prob_all if target_mask is None else prob_all[target_mask]

        auc = _safe_auc(y_tgt, prob)
        if auc is None:
            skipped += 1
            continue

        aucs.append(auc)
        pc = np.clip(prob, 1e-7, 1 - 1e-7)
        llosses.append(float(log_loss(y_tgt, pc)))
        briers.append(float(brier_score_loss(y_tgt, prob)))

    return {
        "auc_mean":     float(np.mean(aucs))    if aucs else float("nan"),
        "auc_std":      float(np.std(aucs))     if aucs else float("nan"),
        "logloss_mean": float(np.mean(llosses)) if llosses else float("nan"),
        "logloss_std":  float(np.std(llosses))  if llosses else float("nan"),
        "brier_mean":   float(np.mean(briers))  if briers else float("nan"),
        "brier_std":    float(np.std(briers))   if briers else float("nan"),
        "n_folds":      len(aucs),
        "skipped":      skipped,
    }


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------

def apply_decision_rule(
    variant_results: dict[str, dict],
    variant_n_feats: dict[str, int],
) -> tuple[str, str]:
    """Lock the variant with fewest features whose AUC is within 1σ of full.

    If no reduced variant qualifies, lock full.
    """
    full_auc = variant_results["full"]["auc_mean"]
    full_std = variant_results["full"]["auc_std"]

    # Candidates sorted by feature count (ascending) — skip 'full'
    candidates = sorted(
        [v for v in variant_results if v != "full"],
        key=lambda v: variant_n_feats[v],
    )

    for v in candidates:
        delta = variant_results[v]["auc_mean"] - full_auc
        if abs(delta) <= full_std:
            reason = (
                f"AUC {variant_results[v]['auc_mean']:.4f} is within 1σ "
                f"({full_std:.4f}) of full ({full_auc:.4f}); "
                f"Δ={delta:+.4f}; fewest features "
                f"({variant_n_feats[v]} vs {variant_n_feats['full']})"
            )
            return v, reason

    reason = (
        f"No reduced variant within 1σ of full AUC {full_auc:.4f} "
        f"(σ={full_std:.4f}); lock full"
    )
    return "full", reason


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_finalisation(force: bool = False) -> dict[str, Any]:
    FIN_OUT.mkdir(parents=True, exist_ok=True)
    results_path = FIN_OUT / "variant_results.json"

    # Load any previously-computed results so we can skip them
    prior: dict[str, Any] = {}
    if results_path.exists() and not force:
        with open(results_path) as fh:
            prior = json.load(fh)

    all_results: dict[str, Any] = {}
    final_rows: list[dict] = []

    for inst in ["cl1s", "es1s", "ho1s", "rb1s", "ng1s"]:
        cfg = CHAMPIONS[inst]
        print(f"\n{'='*64}")
        print(f"Instrument: {inst}  "
              f"(group={cfg['group']}, model={cfg['model'].upper()})")
        print("=" * 64)

        events_df = pd.read_parquet(CACHE_DIR / f"{cfg['group']}_events.parquet")
        variants  = VARIANTS[inst]

        # Seed from prior run if available
        variant_results: dict[str, dict] = dict(
            prior.get(inst, {}).get("variant_results", {})
        )
        variant_feats: dict[str, int] = {}

        for variant in variants:
            feats = _get_variant_features(inst, variant, events_df)
            variant_feats[variant] = len(feats)

            if variant in variant_results and not force:
                res = variant_results[variant]
                print(
                    f"  [{variant:12s}] {len(feats):3d} features — [cached]  "
                    f"AUC={res['auc_mean']:.4f}±{res['auc_std']:.4f}"
                )
                continue

            print(f"  [{variant:12s}] {len(feats):3d} features — running CPCV …")

            # pca_reduced requires per-fold PCA; all others use the standard evaluator
            if variant == "pca_reduced":
                res = evaluate_pca_variant(
                    events_df, NG1S_PCA_SPEC, cfg["model"], cfg["target_inst"]
                )
            else:
                res = evaluate_variant(events_df, feats, cfg["model"], cfg["target_inst"])

            variant_results[variant] = res
            print(
                f"               AUC={res['auc_mean']:.4f}±{res['auc_std']:.4f}  "
                f"ll={res['logloss_mean']:.4f}  "
                f"Brier={res['brier_mean']:.4f}  "
                f"({res['n_folds']} folds)"
            )

        # Decision rule: exploratory variants don't participate
        explorable = EXPLORATORY.get(inst, set())
        rule_variants = {v: r for v, r in variant_results.items()
                         if v not in explorable}
        rule_feats    = {v: n for v, n in variant_feats.items()
                         if v not in explorable}
        locked, reason = apply_decision_rule(rule_variants, rule_feats)
        print(f"\n  ✓ LOCKED: {locked}  —  {reason}")
        if explorable:
            print(f"     (exploratory variants {sorted(explorable)} excluded from rule)")
        print()

        lr = variant_results[locked]
        fr = variant_results["full"]
        final_rows.append({
            "instrument":     inst,
            "locked_variant": locked,
            "model":          cfg["model"],
            "n_features":     variant_feats[locked],
            "auc_mean":       round(lr["auc_mean"], 4),
            "auc_std":        round(lr["auc_std"],  4),
            "vs_full_delta":  round(lr["auc_mean"] - fr["auc_mean"], 4),
            "exploratory":    bool(explorable),
            "reason":         reason,
        })

        all_results[inst] = {
            "variant_results": variant_results,
            "variant_feats":   variant_feats,
            "locked":          locked,
            "reason":          reason,
        }

    # ── persist ─────────────────────────────────────────────────────────────
    pd.DataFrame(final_rows).to_csv(FIN_OUT / "final_models.csv", index=False)
    print(f"Saved final_models.csv → {FIN_OUT / 'final_models.csv'}")

    with open(results_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"Saved variant_results.json → {results_path}")

    # Save per-instrument variant feature lists for notebook reference
    feat_map: dict[str, dict[str, list[str]]] = {}
    for inst, data in all_results.items():
        events_df = pd.read_parquet(
            CACHE_DIR / f"{CHAMPIONS[inst]['group']}_events.parquet"
        )
        feat_map[inst] = {}
        for v in VARIANTS[inst]:
            names = _get_variant_features(inst, v, events_df)
            feat_map[inst][v] = names
    with open(FIN_OUT / "variant_features.json", "w") as fh:
        json.dump(feat_map, fh, indent=2)
    print(f"Saved variant_features.json → {FIN_OUT / 'variant_features.json'}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model finalisation: feature-variant comparison under CPCV"
    )
    parser.add_argument("--force", action="store_true",
                        help="Rerun even if cached results exist")
    args = parser.parse_args()
    run_finalisation(force=args.force)
    print("\nDone.")
