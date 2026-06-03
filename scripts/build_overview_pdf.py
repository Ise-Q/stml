"""Build overview.pdf at branch root — comprehensive summary of Sreeram_experimental.

Uses reportlab for direct PDF generation (no LaTeX available locally).
The document covers data / features / labels / models / cluster importance /
performance matrix / decisions, then a concise decisions section.

Run via:
    uv run python scripts/build_overview_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "overview.pdf"


def styles():
    s = getSampleStyleSheet()
    s.add(
        ParagraphStyle(
            name="Heading1Sm", parent=s["Heading1"], spaceAfter=4, spaceBefore=10,
            fontSize=14, textColor=colors.HexColor("#1F2937"),
        )
    )
    s.add(
        ParagraphStyle(
            name="Heading2Sm", parent=s["Heading2"], spaceAfter=3, spaceBefore=6,
            fontSize=11, textColor=colors.HexColor("#374151"),
        )
    )
    s.add(
        ParagraphStyle(
            name="Body8", parent=s["BodyText"], fontSize=8, leading=11, spaceAfter=4,
            alignment=TA_LEFT,
        )
    )
    s.add(
        ParagraphStyle(
            name="BodyBullet", parent=s["BodyText"], fontSize=8, leading=11,
            leftIndent=12, bulletIndent=2, spaceAfter=2,
        )
    )
    s.add(
        ParagraphStyle(
            name="Mono7", parent=s["Code"], fontSize=7, leading=9,
        )
    )
    return s


def std_table(data, *, col_widths=None, header_bg="#1F2937", header_fg="#FFFFFF",
                font_size=7, body_bg="#FFFFFF"):
    t = Table(data, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(header_fg)),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
            ("TOPPADDING", (0, 0), (-1, 0), 4),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 2),
            ("TOPPADDING", (0, 1), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D1D5DB")),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor(body_bg)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
              [colors.HexColor("#FFFFFF"), colors.HexColor("#F9FAFB")]),
        ])
    )
    return t


def bullet(s, text, style="BodyBullet"):
    return Paragraph(f"• {text}", s[style])


def build():
    s = styles()
    story = []

    # ---------------------------------------------------------------- HEADER
    story.append(Paragraph("Sreeram_experimental — Branch Overview", s["Title"]))
    story.append(
        Paragraph(
            "<b>BUSI70575 (Imperial)</b> · meta-labelling metamodel · 11 instruments "
            "(equity / energy / metals) · 4886 events · 2026-06-03",
            s["Body8"],
        )
    )
    story.append(
        Paragraph(
            "<i>Purpose: assessment-of-record for which model variant to submit. "
            "Not the final academic report — concise overview of what was built, "
            "what was measured, and the decisions made.</i>",
            s["Body8"],
        )
    )
    story.append(Spacer(1, 6))

    # ============================================================== 1. DATA
    story.append(Paragraph("1. Data", s["Heading1Sm"]))
    data_rows = [
        ["Source", "What", "Range", "Use"],
        ["data/ohlcv_data.csv", "Daily OHLCV for 11 futures", "1990-01-02 → 2022-06-30 (8200 rows × 11)", "labels + F1-F17 + F21 features"],
        ["data/primary_signals.csv", "Primary signal in {-1, 0, +1}", "2020-01-03 → 2022-06-30 (645 dates)", "label trigger; trade direction"],
        ["data/bloomberg/cleaned/macro_harry.parquet", "Harry's 21 macro series (VIX, rates, credit, FX, EIA, PMI)", "1990-01-02 → 2022-06-30 daily", "F11 macro features (rank/changes)"],
        ["data/bloomberg/cleaned/futures_term.parquet", "BBG raw front (CL1..PL1) + 2nd-month (CL2..PL2) + VIX futures UX1/UX2", "1990 → 2022 (UX from 2004; XB from 2005)", "F18 term structure (NEW)"],
        ["data/bloomberg/cleaned/options_iv.parquet", "1m/3m ATM IV + 90%/110% MNY for 10 underlyings", "2005-2006 → 2022-06-30", "F19 options IV (NEW)"],
        ["data/bloomberg/cleaned/eia_crude.parquet", "EIA weekly crude inventory CHANGES (DOEASCRD)", "1990 → 2022 (Friday data, +5d release lag)", "F22 EIA flag + crude_change_z26w (NEW)"],
    ]
    story.append(std_table(data_rows, col_widths=[5.5*cm, 5.2*cm, 4.0*cm, 4.5*cm]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "<b>Hidden test:</b> H2-2022 — grader re-runs the pipeline; BBG data CANNOT "
            "be extended past 2022-06-30 (released period, plan §5.1). R-11 ablation "
            "ensures model robustness to BBG missingness at hidden-test time.",
            s["Body8"],
        )
    )
    story.append(
        Paragraph(
            "<b>R-10:</b> OHLCV is <b>ratio back-adjusted continuous futures</b>, not raw front-month. "
            "Return-R² vs raw BBG over 2020-2022: metals 0.995+, equity 0.94-0.97, "
            "ng1s 0.72 (problematic — 11% label flip).",
            s["Body8"],
        )
    )
    story.append(Spacer(1, 4))

    # ============================================================ 2. FEATURES
    story.append(Paragraph("2. Features (105 registered, 80 kept after drift filter)", s["Heading1Sm"]))
    feat_rows = [
        ["Family", "N reg.", "N kept", "Source / lift", "Notes"],
        ["F1 counter-trend", "3", "3", "Sreeram + shared", "bb_pctb, rsi_14, mr_score"],
        ["F2 vol / dispersion", "6", "6", "Sreeram + alken", "vol_20/60, vol_ratio, parkinson, GK, vol_of_vol"],
        ["F5 signal trajectory", "6", "6", "Harry signal_trajectory.py", "run_length, days_since_flip, entropy, flip_rate, long_bias, participation"],
        ["F6 momentum", "6", "5", "Sreeram", "ts_mom 20/60, MA cross, MACD + signal + hist"],
        ["F7 microstructure", "7", "7", "Harry microstructure_fixed.py", "volume_z, volume_trend, oi_level, oi_change, amihud, kyles_lambda, overnight_gap"],
        ["F8 calendar", "4", "4", "Sreeram", "dow_sin/cos, month_sin/cos"],
        ["F9 cross-section", "4", "4", "Harry cross_asset.py", "xsect_rank, dispersion_z, pair_corr_mean, implied_corr_z"],
        ["F10 price action", "4", "4", "Sreeram", "hl_range, oc_ret + 20d means"],
        ["F11 macro (REFORMULATED)", "35", "17", "Rebuilt — 63d rolling RANKS + chg5 / chg20", "<b>levels → ranks</b> per plan §3.3 (fix for catastrophic train→test drift in alken's F11)"],
        ["F12 path structure", "5", "5", "Sreeram", "variance_ratio_5_21, autocorr_21, trend_tval_21, efficiency_ratio_21, hurst_100"],
        ["F15 conditional risk", "2", "2", "Harry conditional_risk.py", "path_tortuosity_20, semi_vol_ratio_20"],
        ["F16 concept drift", "1", "1", "Harry concept_drift.py", "regime_alignment_score (proxy version)"],
        ["F17 HMM (Sreeram)", "3", "0", "Sreeram regimes.py", "All-NaN in production (hmmlearn convergence)"],
        ["F18 term structure (NEW)", "5", "4", "Bloomberg Block A", "term_spread, z63, contango_flag, term_chg5, roll_yield"],
        ["F19 options IV (NEW)", "6", "6", "Bloomberg Block B", "atm_iv_1m/3m, term_slope, iv_rv_spread, skew, pctile_252"],
        ["F21 cross-asset RV", "4", "2", "Plan new (OHLCV-derived)", "gold/silver z, copper/gold, crack 3-2-1"],
        ["F22 event flag (PARTIAL)", "2", "1", "Bloomberg Block E", "EIA crude release flag (Wed); change z26w drift-dropped"],
        ["EWMA HMM (online)", "2", "2", "alken regime.py", "Causal online filter, no CV seam"],
        ["TOTAL", "105", "80", "—", "Drift filter: drop if KS>0.25 ∧ val_AUC<0.54 OR KS>0.20 ∧ val_AUC<0.51"],
    ]
    story.append(std_table(feat_rows, col_widths=[4.0*cm, 1.2*cm, 1.2*cm, 4.5*cm, 8.2*cm]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "<b>Top 8 features by val-AUC (post-drift):</b> "
            "f12_variance_ratio_5_21 (0.590), f12_autocorr_21 (0.584), f5_long_bias_20 (0.570), "
            "f15_path_tortuosity_20 (0.567), f5_trailing_run_length (0.565), f5_days_since_flip (0.565), "
            "<b>f19_iv_pctile_252 (0.559 — NEW BBG)</b>, f5_signal_entropy_20 (0.558).",
            s["Body8"],
        )
    )
    story.append(
        Paragraph(
            "<b>R-11 BBG missingness ablation result:</b> "
            "with_bbg vs without_bbg deltas all ≤ 0.011 across classes — model is BBG-robust. "
            "All 3 asset classes ship with_bbg.",
            s["Body8"],
        )
    )
    story.append(Spacer(1, 4))

    # ============================================================== 3. LABELS
    story.append(Paragraph("3. Labels (triple-barrier, t+1 entry)", s["Heading1Sm"]))
    story.append(
        Paragraph(
            "<b>Spec:</b> pt = sl = 0.5 (symmetric tight); h = 10 trading days; "
            "entry at t+1 (Harry's load-bearing off-by-one fix); GARCH(1,1) one-step-ahead "
            "daily σ̂ (refit every 21d, min 500 obs, max window 2000); "
            "per-instrument concurrency for AFML Ch.4 uniqueness weights; "
            "events with truncated window dropped.",
            s["Body8"],
        )
    )
    story.append(
        Paragraph(
            "<b>Count:</b> 4886 events exact match to Harry's events.csv (per-instrument also exact).",
            s["Body8"],
        )
    )
    story.append(Spacer(1, 3))
    label_rows = [
        ["Instrument", "n_events", "n_long", "n_short", "pos_rate", "PT %", "SL %", "Vert %", "mean uniq"],
        ["cl1s", "411", "375", "36", "0.672", "55.9", "29.0", "15.1", "0.214"],
        ["es1s", "564", "446", "118", "0.530", "47.7", "41.3", "11.0", "0.202"],
        ["fesx1s", "626", "286", "340", "0.534", "47.0", "37.9", "15.2", "0.178"],
        ["gc1s", "161", "129", "32", "0.602", "52.2", "33.5", "14.3", "0.352"],
        ["hg1s", "617", "298", "319", "0.509", "45.5", "44.2", "10.2", "0.198"],
        ["ho1s", "63", "53", "10", "0.635", "58.7", "34.9", "6.3", "0.466"],
        ["ng1s", "120", "0", "120", "0.517", "45.0", "40.0", "15.0", "0.274"],
        ["nq1s", "593", "397", "196", "0.567", "51.8", "40.3", "7.9", "0.203"],
        ["pl1s", "547", "411", "136", "0.554", "48.6", "38.9", "12.4", "0.213"],
        ["rb1s", "617", "358", "259", "0.525", "45.7", "41.5", "12.8", "0.184"],
        ["si1s", "567", "297", "270", "0.510", "40.7", "43.2", "16.0", "0.190"],
        ["TOTAL", "4886", "3050", "1836", "0.547", "47.8", "39.7", "12.5", "—"],
    ]
    story.append(std_table(label_rows, col_widths=[2.4*cm, 1.6*cm, 1.4*cm, 1.6*cm, 1.5*cm, 1.4*cm, 1.4*cm, 1.4*cm, 1.6*cm]))
    story.append(Spacer(1, 3))
    story.append(
        Paragraph(
            "<b>Ablation NOT shipped:</b> Harry's wider asymmetric spec (pt=1.5, sl=1.0, h-day cumulative GARCH) "
            "produced 69.5% vertical, lifted ng1s to 0.800 AUC but per R-10 that's construction-artefact-driven. "
            "Artefacts at <i>data/sreeram_experimental_events_harry_spec.parquet</i> for inspection.",
            s["Body8"],
        )
    )
    story.append(Spacer(1, 4))

    # ============================================================== 4. MODELS
    story.append(Paragraph("4. Models", s["Heading1Sm"]))
    model_rows = [
        ["Family", "Used as", "Config", "Notes"],
        ["Elastic-net Logistic", "Linear baseline (rubric)", "saga solver, l1_ratio=0.5, C=1.0; standardised + median-imputed", "Won several class champions (ho1s, ng1s, si1s)"],
        ["XGBoost", "Tree baseline (rubric, Family A)", "PS5 cell-43: max_depth=4, lr=0.05, subsample=0.8, colsample=0.8, reg_α=0.1, reg_λ=1.0; <b>inner-CV 1SE tuning</b> per plan §4.19", "Won cl1s (LightGBM picked there; XGB was within 1SE)"],
        ["LightGBM", "Tree (alken parity)", "num_leaves=15, lr=0.05, subsample=0.8, deterministic=True", "Won cl1s @ individual pool"],
        ["RandomForest", "Tree-bagging", "n_estimators=200, max_depth=6, min_samples_leaf=10, <b>max_features='sqrt'</b> (PS4 bug fix)", "Won 5 of 11 champions (es1s, fesx1s, nq1s, rb1s, pl1s, hg1s)"],
        ["Ensemble simple-avg", "Defensive averaging (Sreeram v5)", "Uniform mean of 4 estimators' OOS predictions per row", "Used when no single estimator dominates"],
        ["Multi-task NN (S4)", "NN family (rubric, Family B)", "Embedding(11,8) → Linear(d+8,64)+ReLU+Dropout(0.1) → Linear(64,32) → Linear(32,16) shared h → 11 per-inst heads; full-batch Adam; early stop on chrono val", "Won gc1s @ precious; lifted gc1s 0.48→0.57"],
    ]
    story.append(std_table(model_rows, col_widths=[3.5*cm, 3.4*cm, 5.8*cm, 6.4*cm]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "<b>Champion architecture (plan §4.3 + Harry §4.11):</b> per instrument, the champion is the "
            "best (pool, model) combination chosen by CPCV-mean AUC with the 1-SE rule "
            "(most-regularised within 1 SE preferred). Pools per Harry's INSTRUMENT_REGIMES: "
            "individual or asset-class pool, plus sub-class pools (energy_cl_ho, precious) for relevant instruments.",
            s["Body8"],
        )
    )
    story.append(
        Paragraph(
            "<b>Cross-validation throughout:</b> CombinatorialPurgedCV(n_groups=6, n_test_groups=2) → 15 paths. "
            "Per-instrument embargo via results/sreeram_experimental/instrument_scope.json. "
            "Modelling sample = events with t_signal ≤ 2021-10-06. Sealed test = t_signal > 2021-10-20 "
            "(10-day embargo). NO data after embargo touches model selection.",
            s["Body8"],
        )
    )
    story.append(Spacer(1, 4))

    story.append(PageBreak())

    # =================================================== 5. CLUSTER IMPORTANCE
    story.append(Paragraph("5. Cluster importance (S5 — 4 bug fixes implemented)", s["Heading1Sm"]))
    story.append(
        Paragraph(
            "<b>Method:</b> Hygiene drop (NaN > 30%, zero-variance, |ρ| > 0.99 twin) → "
            "<b>Mantegna distance √(1-|ρ|)</b> (bug fix 4) → Ward linkage + silhouette K → "
            "RF with <b>max_features='sqrt'</b> (bug fix 1) inside CPCV(6,2) (<b>bug fix 2</b>: purging) → "
            "Cluster MDI + clustered <b>purged MDA</b> (joint permutation per cluster) + "
            "<b>TreeSHAP via XGBoost native pred_contribs</b> (bug fix 3 — no shap library needed). "
            "Cross-method Kendall τ.",
            s["Body8"],
        )
    )
    story.append(Spacer(1, 3))
    cluster_rows = [
        ["Class", "Hygiene", "K (sil.)", "Top cluster MDA", "n cluster > 0.02 MDA", "alken comparable", "Gate"],
        ["Equity", "70 / 80 kept", "16", "0.0223", "1 / 16", "0.025 (alken)", "PASS"],
        ["Energy", "73 / 80 kept", "8",  "0.0364", "1 / 8",  "0.003 (alken)", "PASS (12× alken)"],
        ["Metals", "71 / 80 kept", "16", "0.0155", "0 / 16", "-0.011 (alken)", "CHECK (we beat alken)"],
    ]
    story.append(std_table(cluster_rows, col_widths=[2.5*cm, 2.5*cm, 1.7*cm, 3.2*cm, 3.5*cm, 3.0*cm, 2.5*cm]))
    story.append(Spacer(1, 3))
    story.append(
        Paragraph(
            "<b>Energy's dominant cluster (MDA 0.036, 19 features):</b> "
            "<i>f19_atm_iv_1m, f19_atm_iv_3m, f19_iv_term_slope, f19_iv_pctile_252, "
            "f2_vol_20/60, f2_parkinson_20, f2_garman_klass_20, f2_vol_of_vol_20, "
            "f10_hl_range, f12_hurst_100, f9_dispersion_z_60, f9_pair_corr_mean_63, f9_implied_corr_z_252, "
            "f21_crack_321, f11_eia_crude_stock_rank63, f11_curve_slope, f11_vix_term_slope, "
            "f16_regime_alignment_score</i>. The F19 BBG options-IV pull (which alken did NOT have) is "
            "the empirical driver of energy outperformance.",
            s["Body8"],
        )
    )
    story.append(
        Paragraph(
            "<b>Cross-method agreement</b> (Kendall τ): SHAP ↔ MDI strong (0.72, p&lt;1e-4 on metals); "
            "SHAP ↔ MDA weak (0.15). Expected pattern per alken §5.16: MDI/SHAP are in-sample attribution, "
            "MDA is the OOS reality check.",
            s["Body8"],
        )
    )
    story.append(Spacer(1, 6))

    # ============================================== 6. PERFORMANCE MATRIX
    story.append(Paragraph("6. Performance matrix — per-class CPCV mean AUC across methodologies", s["Heading1Sm"]))
    perf_rows = [
        ["Methodology", "Equity", "Energy", "Metals", "Alken Δ", "Notes"],
        ["Alken (published)", "0.579", "0.525", "0.530", "—", "Reference baseline"],
        ["Per-class XGB only (S3.0)", "0.521", "0.556", "0.512", "−0.058 / +0.031 / −0.018", "No instrument one-hot"],
        ["+ instrument 1-hot + LGBM + ensemble (S3.1)", "0.554", "0.558", "0.525", "−0.025 / +0.033 / −0.005", "Roster of 4 + simple-avg"],
        ["+ champion arch. per inst. (S3-fix)", "0.550", "0.602", "0.533", "−0.029 / +0.077 / +0.003", "Per Harry's INSTRUMENT_REGIMES; 1SE picks"],
        ["+ multi-task NN (S4)", "0.550", "0.602", "0.554", "−0.029 / +0.077 / +0.024", "gc1s rescued 0.48→0.57"],
    ]
    story.append(std_table(perf_rows, col_widths=[6.0*cm, 1.5*cm, 1.5*cm, 1.5*cm, 3.5*cm, 4.0*cm]))
    story.append(Spacer(1, 3))
    story.append(Paragraph("Per-instrument champion AUC (with_bbg, sorted by AUC)", s["Heading2Sm"]))
    perinst_rows = [
        ["Inst", "Champion pool", "Champion model", "AUC", "± SEM", "Lower 1SE CI", "vs Harry post-refresh"],
        ["cl1s",   "cl1s",       "LightGBM",            "0.671", "0.022", "0.649", "Harry 0.675 (MATCH)"],
        ["ng1s",   "energy_all", "elastic-net log",     "0.601", "0.044", "0.557", "Harry no signal"],
        ["ho1s",   "energy_cl_ho","elastic-net log",     "0.599", "0.048", "0.551", "Harry n/a (thin)"],
        ["pl1s",   "pl1s",       "RandomForest",        "0.581", "0.019", "0.562", "Harry 0.608 (-0.027)"],
        ["gc1s",   "precious",   "multitask_nn",        "0.568", "0.029", "0.540", "Harry 0.478"],
        ["fesx1s", "equity_all", "RandomForest",        "0.557", "0.019", "0.538", "Harry 0.579 (-0.022)"],
        ["es1s",   "es1s",       "RandomForest",        "0.555", "0.017", "0.537", "Harry 0.516"],
        ["rb1s",   "rb1s",       "RandomForest",        "0.538", "0.020", "0.517", "Harry 0.551"],
        ["nq1s",   "equity_all", "RandomForest",        "0.537", "0.011", "0.526", "Harry 0.689 (-0.152) †"],
        ["si1s",   "si1s",       "elastic-net log",     "0.534", "0.011", "0.522", "Harry 0.515"],
        ["hg1s",   "metals_all", "RandomForest",        "0.533", "0.014", "0.519", "Harry 0.604 (-0.071) †"],
    ]
    story.append(std_table(perinst_rows, col_widths=[1.4*cm, 2.3*cm, 3.0*cm, 1.4*cm, 1.4*cm, 1.7*cm, 5.0*cm]))
    story.append(
        Paragraph(
            "<b>†</b> Harry's lead on nq1s/hg1s correlates with his wider GARCH labels (pt=1.5,sl=1.0); "
            "our spec is tighter per plan §3.2. Switching label spec is a §3.2 deviation; see decisions.",
            s["Body8"],
        )
    )
    story.append(Spacer(1, 3))
    story.append(Paragraph("Acceptance gates", s["Heading2Sm"]))
    gate_rows = [
        ["Gate", "Target", "Actual", "Status"],
        ["≥6 of 11 per-inst AUC > 0.55", "≥ 6", "7 (cl/ng/ho/pl/gc/fesx/es)", "PASS"],
        ["Signal flag (lower 1-SE CI > 0.50)", "≥ 6", "11 / 11", "PASS"],
        ["Per-class AUC ≥ alken + 0.02 (equity)", "≥ 0.599", "0.550", "CHECK"],
        ["Per-class AUC ≥ alken + 0.02 (energy)", "≥ 0.545", "0.602", "PASS (+0.057)"],
        ["Per-class AUC ≥ alken + 0.02 (metals)", "≥ 0.550", "0.554", "PASS (+0.004)"],
        ["R-11: BBG-missingness Δ ≤ 0.03", "All classes", "Max Δ 0.011", "PASS"],
        ["S5: ≥1 cluster > 0.02 MDA per class", "All classes", "2 of 3 (metals 0.016)", "PASS / CHECK"],
        ["S5: all 4 bug fixes present", "All", "All 4 in code + tests", "PASS"],
    ]
    story.append(std_table(gate_rows, col_widths=[8.0*cm, 3.0*cm, 4.5*cm, 2.5*cm]))
    story.append(Spacer(1, 4))

    story.append(Paragraph("S6 backtest (sealed test slice, 179 days)", s["Heading2Sm"]))
    bt_rows = [
        ["Metric", "Value", "Notes"],
        ["n days", "179", "Sealed test window (post-2021-10-20 embargo)"],
        ["Ann. return (net)", "17.3 %", "After Grinold-Kahn costs (2 bps half-spread + 10 bps impact)"],
        ["Ann. vol", "5.40 %", "Below 10 % R7 cap; slightly below [0.06, 0.10] target band"],
        ["Sharpe", "3.20", "Per-period; pre-deflation"],
        ["Sortino (full-T, Sortino-Price)", "6.03", "NOT std-of-negatives mis-impl"],
        ["Max DD", "−2.33 %", "Shallow"],
        ["Turnover (1/yr)", "16.6", "Holding ~6.6 days mean"],
        ["Total cost (period bp)", "142 bp", "Gross 19.3 % → Net 17.3 %"],
    ]
    story.append(std_table(bt_rows, col_widths=[5.0*cm, 3.0*cm, 10.0*cm]))
    story.append(Spacer(1, 3))
    story.append(
        Paragraph(
            "<b>Pre-deflation Sharpe 3.20 reads 'too good'.</b> Plan §13 R-8 + alken's five-lens framework "
            "(AUC, MDA, significance, deflation, PT) is the discipline. S7 (significance + deflation + PT) "
            "is the next stage where we test if 3.20 survives the studentised block-bootstrap CI / DSR ladder / "
            "CSCV-PBO / Pesaran-Timmermann checks. Until then the 3.20 is one-lens and not a deployable claim.",
            s["Body8"],
        )
    )

    story.append(PageBreak())

    # ============================================== 7. KEY DECISIONS
    story.append(Paragraph("7. Key decisions (bullet form)", s["Heading1Sm"]))

    story.append(Paragraph("Data + labels", s["Heading2Sm"]))
    for b in [
        "<b>Triple-barrier with t+1 entry</b> (Harry's load-bearing fix) — corr(s_t, r_{t+1}) > 0 on all 11 instruments empirically. Our 4886 events EXACT match Harry's events.csv per instrument.",
        "<b>GARCH(1,1) one-step daily σ̂</b> (refit 21d, min_obs=500, max_window=2000) — forward-aware, NOT EWMA (alken's default). Per-bar daily; barrier widths = pt·σ·√h.",
        "<b>pt=sl=0.5, h=10</b> — symmetric tight (NOT alken pt=sl=1.0 or Harry pt=1.5/sl=1.0). Resolves 87.5 % of events at PT or SL (vs alken ~50 % vertical). Plan §3.2 spec.",
        "<b>Per-instrument concurrency on each instrument's own trading-day index</b> — AFML Ch.4 uniqueness weights; calendar gaps don't inflate overlap.",
        "<b>Right-edge truncation: drop events whose [t+1, t+1+h] window isn't fully observable</b> — matches Harry's convention (4886 vs 4975 with truncation kept).",
        "<b>Harry-spec ablation tried, NOT shipped:</b> wider barriers lift ng1s to 0.800 but per R-10 that's construction-artefact (11 % label flip vs raw BBG). Artefacts retained for inspection only.",
    ]:
        story.append(bullet(s, b))

    story.append(Paragraph("Features", s["Heading2Sm"]))
    for b in [
        "<b>F11 macro REBUILT</b>: 63-day rolling RANKS instead of raw LEVELS (alken's F11 had median KS 0.36, max 1.0 — catastrophic drift). Plus chg5/chg20 changes + 3 spread features.",
        "<b>F18 / F19 / F22 NEW</b> (Bloomberg-derived): term-structure, options-implied vol, EIA event flag. F19 options IV explicitly drives energy's outperformance (~6.6 % importance, in the highest-MDA cluster).",
        "<b>F21 cross-asset RV</b> (NEW, OHLCV-derived): gold/silver, copper/gold, crack 3-2-1. No Bloomberg needed.",
        "<b>Drift filter</b> (plan §3.3): DROP if KS > 0.25 AND val_AUC < 0.54, OR if KS > 0.20 AND val_AUC < 0.51. Brings 105 registered → 80 kept; 66 of 80 have KS &lt; 0.20 (82.5 %, S2 gate PASS).",
        "<b>F17 HMM</b> (Sreeram): implementation failed (hmmlearn convergence) — emits NaN, drift-filtered out. Online EWMA HMM (alken's regime.py port) used instead — causal, no CV-seam.",
        "<b>R-11 ablation:</b> shipped <i>with_bbg</i>; without_bbg AUC delta ≤ 0.011 across classes (model is BBG-robust to H2-2022 missingness at hidden-test time).",
    ]:
        story.append(bullet(s, b))

    story.append(Paragraph("Models + selection", s["Heading2Sm"]))
    for b in [
        "<b>Four-estimator roster:</b> elastic-net logistic, XGBoost, LightGBM, RandomForest — rubric requires linear + tree + NN families; the multi-task NN serves the NN slot.",
        "<b>Champion architecture per Harry §4.11 INSTRUMENT_REGIMES:</b> each instrument has 1-3 candidate pools (individual / sub-class / asset-class). The full 4×pool grid per instrument is searched.",
        "<b>1-SE selection rule</b> (plan §4.19): most-regularised model within 1 SE of best mean AUC is the champion. Rank: logistic < RF < LightGBM < XGB < multitask. Pool size: individual &lt; sub-class &lt; class.",
        "<b>Inner-CV XGB tuning</b> (plan §3.1 + §4.19): purged inner k-fold (k=4) over a 5-config grid (max_depth × n_estimators × reg_α / reg_λ); 1-SE pick most regularised within 1 SE.",
        "<b>Instrument one-hot dummies</b> in the asset-class pool models (plan §3.1) — lets the tree learn instrument-conditioned splits without per-instrument training.",
        "<b>Multi-task NN</b> (plan §3.1 Family B): only candidate for multi-instrument pools (sharing meaningless for individual). Won gc1s precious champion (0.48 → 0.57).",
        "<b>Prediction clip [0.01, 0.99]</b> — bounds log-loss; AUC unchanged (monotone). Fixed energy logistic's blown-up log-loss (2-5 → ≤ 1.5).",
        "<b>CombinatorialPurgedCV(6, 2) → 15 paths</b> with per-instrument embargo via embargo_p90 map. Nested CPCV implemented + unit-tested but real-data run deferred (alken parity).",
    ]:
        story.append(bullet(s, b))

    story.append(Paragraph("Cluster importance (S5)", s["Heading2Sm"]))
    for b in [
        "<b>Bug fix 1: max_features='sqrt'</b> — PS4's 'auto' was removed in sklearn ≥ 1.3.",
        "<b>Bug fix 2: PurgedKFold for MDA</b> — KFold(shuffle=True) leaks across overlapping triple-barrier labels.",
        "<b>Bug fix 3: TreeSHAP via XGBoost native pred_contribs</b> — runs Tree SHAP internally; NO shap library / numba / llvmlite dependency required. The chain was blocked: shap requires numba which requires numpy &lt; 2.4, but pandas 3.0 pins numpy ≥ 2.4. XGBoost's built-in path sidesteps it.",
        "<b>Bug fix 4: Mantegna distance √(1-|ρ|)</b> — proper metric (Mantegna 1999); the original 1-|ρ| is non-metric.",
        "<b>Per-class K selection by silhouette</b> over [3, 16] — equity 16, energy 8, metals 16.",
        "<b>Energy top cluster MDA 0.036</b> (19 features, includes ALL F19 BBG IV) vs alken 0.003 — 12× alken. <b>This is where the BBG pull pays off.</b>",
        "<b>Cross-method Kendall τ:</b> SHAP↔MDI = 0.72 (strong, both in-sample); SHAP↔MDA = 0.15 (weak, MDA is OOS reality check) — matches alken §5.16's framing.",
    ]:
        story.append(bullet(s, b))

    story.append(Paragraph("Risks documented in plan §13", s["Heading2Sm"]))
    for b in [
        "<b>R-1</b> BBG pull thin / unavailable: F18/F19/F22 silently drop. <b>HANDLED</b> — pulled and kept.",
        "<b>R-8</b> Grinold ceiling: pooled IC ≈ 0.07 → AUC ceiling ≈ 0.52. <b>Per-class AUC 0.55-0.60 is at the ceiling.</b>",
        "<b>R-9</b> BBG vs OHLCV convention: confirmed in practice — OHLCV is ratio back-adjusted; F18 uses BBG raw front + BBG raw 2nd, never mixed with OHLCV.",
        "<b>R-10 (NEW) Continuous-contract artefact on target</b>, esp. ng1s (R² 0.72 vs raw, 11 % label flip). <b>Methodology must NOT claim raw-market profitability.</b> ng1s flagged in coverage_caveat.csv.",
        "<b>R-11 (NEW) H2-2022 BBG missingness</b>: simulated-missingness ablation built into pipeline. Δ ≤ 0.011 across classes → model is robust. Ship with_bbg. <b>Pulling H2-2022 BBG would VIOLATE the brief (§5.1).</b>",
        "<b>R-12 (deferred → resolved)</b> TreeSHAP bug fix 3 — resolved via XGBoost native pred_contribs, no external shap dep needed.",
    ]:
        story.append(bullet(s, b))

    story.append(Paragraph("Where the artefacts live", s["Heading2Sm"]))
    for b in [
        "<b>S1 events:</b> data/sreeram_experimental_events.parquet (4886 rows × 10 cols)",
        "<b>S2 features:</b> data/sreeram_experimental_features.parquet (4886 × 80 + schema)",
        "<b>S2 drift audit:</b> results/sreeram_experimental/feature_drift_audit.csv",
        "<b>S3 instrument scope:</b> results/sreeram_experimental/instrument_scope.json",
        "<b>S3 champions (with_bbg):</b> results/sreeram_experimental/champions_summary.csv + champions_per_pool_per_model.csv",
        "<b>S3 R-11 ablation:</b> results/sreeram_experimental/bbg_missingness_ablation.csv",
        "<b>S3 coverage caveats:</b> outputs/coverage_caveat.csv (ng1s low_coherence_vs_raw flag)",
        "<b>S5 cluster importance per class:</b> results/sreeram_experimental/importance/{equity,energy,metals}/clustered_importance.csv + cluster_membership.csv + rank_agreement.csv",
        "<b>S6 deliverables:</b> outputs/metamodel_predictions.csv (calibrated), outputs/metamodel_predictions_raw.csv, outputs/strategy_weights.csv, outputs/experiment_log.csv",
        "<b>S6 backtest metrics:</b> results/sreeram_experimental/backtest_metrics.csv; daily returns: strategy_daily_net_returns.csv + strategy_daily_gross_returns.csv",
        "<b>Plan + tracker:</b> reports/sreeram_experimental/plan.md (golden record), action_tracker.md (PM-1..PM-4 log)",
        "<b>BBG validation:</b> reports/sreeram_experimental/bloomberg_validation_report.md, bloomberg_pull_list.md",
    ]:
        story.append(bullet(s, b))

    story.append(Paragraph("Outstanding / not yet built", s["Heading2Sm"]))
    for b in [
        "<b>S7 significance + deflation + PT</b> (the other 4 lenses of alken's framework): modules <i>significance.py</i>, <i>deflation.py</i>, <i>signal_analysis.py</i> are <b>written</b> (studentised stationary block bootstrap, DSR ladder, CSCV-PBO with C(16,8)=12870, Pesaran-Timmermann, Treynor-Mazuy, Henriksson-Merton proxy) but the end-to-end significance_summary.md run is pending. Backtest Sharpe 3.20 needs to survive these gates.",
        "<b>S8 final academic report</b> with Harvard refs (alken's T3_03 equivalent): pending.",
        "<b>TorchVSN</b> (alken §5.12): not built. Multi-task NN already provides the per-instrument NN family slot.",
    ]:
        story.append(bullet(s, b))

    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "<i>Build state: 22 commits on Sreeram_experimental, all pushed to origin. "
            "97 fast tests + 3 slow tests pass. Working tree clean.</i>",
            s["Body8"],
        )
    )

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=1.4*cm, rightMargin=1.4*cm,
        topMargin=1.4*cm, bottomMargin=1.4*cm,
        title="Sreeram_experimental — Branch Overview",
        author="Sreeram",
    )
    doc.build(story)
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    build()
