"""`PipelineConfig` — the single source of truth for every hyperparameter.

Plan R1 (one version), R2 (edit in place), R7 (10% vol cap). When a value is
referenced more than once anywhere in the experimental package it lives here,
not as a magic number scattered through code.

The config is a *frozen* dataclass — accidental mutation at runtime would break
the determinism contract (R3 / R5). To change a value, edit this file in a
``feat(sN): ...`` commit and re-run the affected stage.

Stage tags applied to each field document **the earliest stage that consumes it**;
S0 (this file) ships only fields used by the whole pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Type aliases — keep narrow strings as discriminated literals for editor help.
# ---------------------------------------------------------------------------
AssetClass = Literal["equity", "energy", "metals"]
SigmaSource = Literal["garch", "ewma", "gk"]
SelectionRule = Literal["best_mean", "one_se"]
Roster = Literal["tree_linear", "default", "full"]
CvScheme = Literal["purged", "cpcv", "nested_cpcv"]


# ---------------------------------------------------------------------------
# Universe — mirrored from stml.io.INSTRUMENTS for visibility in the config.
# ---------------------------------------------------------------------------
INSTRUMENTS: tuple[str, ...] = (
    "cl1s", "es1s", "fesx1s", "gc1s", "hg1s",
    "ho1s", "ng1s", "nq1s", "pl1s", "rb1s", "si1s",
)

ASSET_CLASS_MEMBERS: dict[AssetClass, tuple[str, ...]] = {
    "equity": ("es1s", "nq1s", "fesx1s"),
    "energy": ("cl1s", "ho1s", "rb1s", "ng1s"),
    "metals": ("gc1s", "si1s", "hg1s", "pl1s"),
}

INSTRUMENT_TO_CLASS: dict[str, AssetClass] = {
    inst: cls for cls, members in ASSET_CLASS_MEMBERS.items() for inst in members
}


@dataclass(frozen=True)
class PipelineConfig:
    """Frozen, byte-stable config object passed through every stage.

    The defaults are the *shipped* configuration. Stages may receive copies with
    targeted fields overridden via ``dataclasses.replace(cfg, key=value)``; that
    pattern is preferred over a mutable config because it keeps the original
    around for byte-identical re-emits.
    """

    # ---------------------------------------------------------------- S0 base
    seed: int = 42

    # ----------------------------------------------------------- S1 labelling
    # Triple-barrier convention (methodology spec).
    # Entry is at t+1 (the load-bearing fix). Barriers sized as
    # ``pt_mult * sigma_t * sqrt(h)`` and ``-sl_mult * sigma_t * sqrt(h)`` in
    # log-return units. CPCV barrier search in S2 may override pt/sl/h per
    # asset class (methodology spec gates).
    pt_mult: float = 0.5
    sl_mult: float = 0.5
    max_holding: int = 10  # h, in trading days

    # Sigma source for the barrier scale. ``garch`` is the plan default
    # (forward-aware, sharper). EWMA is the fall-back if arch is unavailable.
    sigma_source: SigmaSource = "garch"
    garch_refit_every: int = 21
    garch_min_obs: int = 500
    garch_max_window: int = 2000

    # ---------------------------------------------------- S1 train/test split
    # adopted clean global cut (methodology spec). The
    # 30 % post-2021-10-20 slice is SEALED — never read during selection or
    # importance, only at the final OOS confirmation.
    global_train_cut: str = "2021-10-06"
    embargo_end: str = "2021-10-20"

    # ----------------------------------------------- S2 features (drift filter)
    drift_ks_threshold: float = 0.25
    drift_val_auc_floor: float = 0.54
    drift_val_auc_keep_above: float = 0.58
    macro_rolling_rank_window: int = 63  # methodology spec reformulation

    # ----------------------------------------------------------- S3-S4 CV / models
    cv_scheme: CvScheme = "cpcv"
    cpcv_n_groups: int = 6
    cpcv_n_test_groups: int = 2  # → C(6,2) = 15 paths
    cpcv_pct_embargo: float = 0.01  # uniform fallback; per-instrument map overrides
    inner_kfold_k: int = 4
    selection_rule: SelectionRule = "one_se"  # methodology spec — 1SE rule

    roster: Roster = "default"

    # ----------------------------------------------------------- S5 importance
    importance_max_clusters: int = 16

    # --------------------------------------------------------- S6 sizing / vol
    # Madmoun Optional Session 3 recipe (slides 32-34, 39-40).
    # sizing_method is one of: model_confidence, all_or_nothing, ncdf,
    # linear_scaling, ecdf, sops. Default SOPS — the only method that fits
    # the sigmoid shape to in-sample Sharpe directly.
    sizing_method: str = "sops"
    # Bootstrap-estimate p* per instrument; if False, use 0.5 baseline gate.
    use_pstar_threshold: bool = True
    pstar_bootstrap: int = 2000
    # Volatility targeting -- lecturer's slide 40 prescribes 10 % annualised.
    target_vol: float = 0.10
    max_leverage: float = 10.0  # defensive clamp; not in the lecture.
    # EWMA σ̂ for the position-weight denominator (slide 39).
    ewma_sigma_span: int = 60

    # ---------------------------------------------------- S6 cost model
    half_spread_bps: float = 2.0  # Grinold-Kahn, adopted convention
    impact_bps: float = 10.0
    impact_exponent: float = 1.0  # linear

    # ----------------------------------------------- I/O
    output_dir: str = "outputs"
    results_dir: str = "results/submission"

    # ------------------------------------- catalogue switches (S2)
    use_bloomberg: bool = True  # if False, F18-F22 silently dropped (R-1 fallback)
    use_drift_feature: bool = True  # F16 — methodology spec
    use_ewma_hmm: bool = True  # methodology spec — adopted convention

    # -------- "extras" reserved for stage-specific overrides without proliferation
    extras: dict[str, object] = field(default_factory=dict)
