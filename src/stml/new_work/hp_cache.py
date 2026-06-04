"""hp_cache.py — consolidated champion / hyperparameter cache for the meta-model.

Harry's CPCV pipeline selects, per instrument, a *champion* feature-group + model family
(`outputs/model_comparison/selection_table.csv`), a *locked variant* / feature subset
(`outputs/model_comparison/{class}/locked_picks.csv`), a cluster-level *importance family*
(`outputs/importance/{inst}/champion_meta.csv`), and the cluster-level weight vector
(`outputs/importance/{inst}/global_{shap,coef}_summary.csv`).

Those selections ARE "the hyperparameter set": they fully determine which model is fit on which
features per instrument. The low-level estimator hyper-parameters (RF depth, XGB lr, …) are
re-tuned per fold by Harry's inner CPCV + 1SE rule, so there is no single global estimator dict to
pin — the cache captures the *selection*, which is the expensive search result.

`build_selected_hps()` reads Harry's committed CPCV model-comparison + importance outputs and writes a
single consolidated `outputs/selected_hps.json`. `load_selected_hps()` reads it back. The submission
notebook writes this once (first run / FORCE_RECOMPUTE) and loads it on every subsequent run so the
lecturer's run reuses the saved selections instead of re-searching.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).parent
OUTPUTS = _HERE / "outputs"
MC = OUTPUTS / "model_comparison"
IMPORTANCE = OUTPUTS / "importance"
FINALISATION = OUTPUTS / "finalisation"
CACHE_PATH = OUTPUTS / "selected_hps.json"

# Static instrument -> asset-class map (matches the per-class locked_picks.csv files).
ASSET_CLASS: dict[str, str] = {
    "es1s": "equity", "nq1s": "equity", "fesx1s": "equity",
    "cl1s": "energy", "ho1s": "energy", "rb1s": "energy", "ng1s": "energy",
    "gc1s": "metals", "si1s": "metals", "pl1s": "metals", "hg1s": "metals",
}


def _split_meta() -> dict:
    """Train/test cut + embargo + seed, read from the live config modules (with fallbacks)."""
    meta = {"split_cut": "2021-10-06", "embargo_end": "2021-10-20", "random_seed": None}
    try:
        from stml.new_work import split_config
        meta["split_cut"] = str(split_config.GLOBAL_CUT.date())
        meta["embargo_end"] = str(split_config.EMBARGO_END.date())
    except Exception:  # pragma: no cover - config always present in submission
        pass
    try:
        from stml.new_work.feature_importance import RANDOM_SEED
        meta["random_seed"] = int(RANDOM_SEED)
    except Exception:  # pragma: no cover
        pass
    return meta


def _read_locked_variants() -> dict[str, str]:
    """instrument -> locked_variant, pooled from the three per-class locked_picks.csv (all 11)."""
    out: dict[str, str] = {}
    for cls in ("equity", "energy", "metals"):
        p = MC / cls / "locked_picks.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            out[str(r["inst"])] = str(r["locked_variant"])
    return out


def _read_champion_meta() -> dict[str, dict]:
    """instrument -> champion_meta row (asset_class, family, model_type, auc, signal, clusters)."""
    out: dict[str, dict] = {}
    if not IMPORTANCE.exists():
        return out
    for d in sorted(IMPORTANCE.iterdir()):
        p = d / "champion_meta.csv"
        if not p.exists():
            continue
        row = pd.read_csv(p).iloc[0].to_dict()
        out[d.name] = row
    return out


def _weight_vector_file(inst: str, family: str | None) -> str | None:
    """The importance CSV that holds this instrument's champion weight vector."""
    fam = (family or "").lower()
    shap = IMPORTANCE / inst / "global_shap_summary.csv"
    coef = IMPORTANCE / inst / "global_coef_summary.csv"
    if fam == "logistic" and coef.exists():
        return coef.name
    if shap.exists():
        return shap.name
    return coef.name if coef.exists() else None


def build_selected_hps(write: bool = True) -> dict:
    """Consolidate Harry's committed champion selections into one dict; optionally persist it."""
    # --- per-instrument champion model (all 11) ---
    sel_path = MC / "selection_table.csv"
    selection = pd.read_csv(sel_path).set_index("instrument") if sel_path.exists() else pd.DataFrame()
    locked = _read_locked_variants()
    champ_meta = _read_champion_meta()

    instruments = sorted(
        set(ASSET_CLASS)
        | set(selection.index.astype(str))
        | set(locked)
        | set(champ_meta)
    )

    records: dict[str, dict] = {}
    for inst in instruments:
        rec: dict = {"asset_class": ASSET_CLASS.get(inst)}

        if inst in selection.index:
            s = selection.loc[inst]
            rec.update(
                champion_group=str(s["best_group"]),
                champion_model=str(s["best_model"]),
                selection_auc=float(s["best_auc"]),
                selection_lower_ci=float(s["lower_ci"]),
                signal=bool(s["signal"]),
                runner_up_model=(None if pd.isna(s.get("runner_up_model")) else str(s["runner_up_model"])),
            )

        rec["locked_variant"] = locked.get(inst)

        cm = champ_meta.get(inst)
        if cm:
            rec["asset_class"] = cm.get("asset_class", rec["asset_class"])
            rec["importance_family"] = str(cm.get("family"))
            rec["importance_model_type"] = str(cm.get("model_type"))
            rec["n_sig_clusters"] = (None if pd.isna(cm.get("n_sig_clusters")) else int(cm["n_sig_clusters"]))
        rec["weight_vector_file"] = _weight_vector_file(inst, rec.get("importance_family"))

        records[inst] = rec

    by_class: dict[str, list[str]] = {}
    for inst in instruments:
        by_class.setdefault(ASSET_CLASS.get(inst, "other"), []).append(inst)

    payload = {
        "meta": {
            "description": (
                "Consolidated champion / hyperparameter selection derived from Harry's committed "
                "CPCV model-comparison and importance outputs. Saved so a re-run "
                "loads these selections instead of re-running the search."
            ),
            **_split_meta(),
            "n_instruments": len(records),
            "asset_classes": by_class,
            "sources": [
                "model_comparison/selection_table.csv",
                "model_comparison/{equity,energy,metals}/locked_picks.csv",
                "importance/{inst}/champion_meta.csv",
            ],
        },
        "instruments": records,
    }

    if write:
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def load_selected_hps() -> dict:
    """Read the consolidated selection cache written by build_selected_hps()."""
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"{CACHE_PATH} not found — run build_selected_hps() (notebook: FORCE_RECOMPUTE=True) first."
        )
    return json.loads(CACHE_PATH.read_text())


if __name__ == "__main__":
    hp = build_selected_hps(write=True)
    print(f"Wrote {CACHE_PATH} with {hp['meta']['n_instruments']} instruments.")
    for inst, rec in hp["instruments"].items():
        print(
            f"  {inst:7s} {rec.get('asset_class',''):7s} "
            f"champ={rec.get('champion_model','?')}/{rec.get('locked_variant','?')} "
            f"signal={rec.get('signal','?')} wv={rec.get('weight_vector_file','?')}"
        )
