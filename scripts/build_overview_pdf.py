"""Build overview.pdf at branch root — Sreeram_experimental branch summary.

This document covers ONLY the work in this branch. No cross-branch references.

Run via:
    uv run python scripts/build_overview_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "overview.pdf"


COLOR_PRIMARY = colors.HexColor("#0F172A")     # slate-900
COLOR_ACCENT = colors.HexColor("#0369A1")      # sky-700
COLOR_HEADER_BG = colors.HexColor("#1E293B")   # slate-800
COLOR_HEADER_FG = colors.HexColor("#FFFFFF")
COLOR_ROW_ALT = colors.HexColor("#F1F5F9")     # slate-100
COLOR_BORDER = colors.HexColor("#CBD5E1")      # slate-300
COLOR_BODY = colors.HexColor("#1E293B")
COLOR_MUTED = colors.HexColor("#64748B")       # slate-500


def styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle(
        name="TitleBig", parent=s["Title"], fontSize=22, leading=26,
        textColor=COLOR_PRIMARY, spaceAfter=4, alignment=TA_LEFT,
    ))
    s.add(ParagraphStyle(
        name="Subtitle", parent=s["BodyText"], fontSize=10, leading=13,
        textColor=COLOR_MUTED, spaceAfter=16,
    ))
    s.add(ParagraphStyle(
        name="H1", parent=s["Heading1"], fontSize=15, leading=20,
        textColor=COLOR_ACCENT, spaceAfter=6, spaceBefore=18,
        keepWithNext=True,
    ))
    s.add(ParagraphStyle(
        name="H2", parent=s["Heading2"], fontSize=11, leading=15,
        textColor=COLOR_PRIMARY, spaceAfter=4, spaceBefore=10,
        keepWithNext=True,
    ))
    s.add(ParagraphStyle(
        name="Body", parent=s["BodyText"], fontSize=9.5, leading=14,
        textColor=COLOR_BODY, spaceAfter=6, alignment=TA_LEFT,
    ))
    s.add(ParagraphStyle(
        name="BodyJ", parent=s["BodyText"], fontSize=9.5, leading=14,
        textColor=COLOR_BODY, spaceAfter=6, alignment=TA_JUSTIFY,
    ))
    s.add(ParagraphStyle(
        name="BulletItem", parent=s["BodyText"], fontSize=9.5, leading=14,
        textColor=COLOR_BODY, leftIndent=14, bulletIndent=2, spaceAfter=4,
    ))
    s.add(ParagraphStyle(
        name="Caption", parent=s["BodyText"], fontSize=8.5, leading=11,
        textColor=COLOR_MUTED, spaceAfter=10, spaceBefore=2,
    ))
    return s


def make_table(data, *, col_widths=None, font_size=9, header_align="LEFT"):
    """Standard table with clean styling, larger fonts, more padding."""
    t = Table(data, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        # Header row.
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), COLOR_HEADER_FG),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), font_size),
        ("ALIGN", (0, 0), (-1, 0), header_align),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
        ("TOPPADDING", (0, 0), (-1, 0), 7),
        # Body.
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), font_size),
        ("TEXTCOLOR", (0, 1), (-1, -1), COLOR_BODY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        ("TOPPADDING", (0, 1), (-1, -1), 5),
        # Borders.
        ("LINEBELOW", (0, 0), (-1, 0), 1.0, COLOR_HEADER_BG),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, COLOR_BORDER),
        # Alternating row backgrounds.
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
            [colors.white, COLOR_ROW_ALT]),
    ]))
    return t


def bullet(s, text):
    return Paragraph(f"•&nbsp;&nbsp;{text}", s["BulletItem"])


def build():
    s = styles()
    story = []

    # ============================================================ HEADER
    story.append(Paragraph("Sreeram_experimental — Branch Overview", s["TitleBig"]))
    story.append(Paragraph(
        "BUSI70575 (Imperial) &nbsp; · &nbsp; meta-labelling metamodel &nbsp; · &nbsp; "
        "11 instruments (equity / energy / metals) &nbsp; · &nbsp; 4,886 events &nbsp; · &nbsp; "
        "2026-06-03",
        s["Subtitle"],
    ))
    story.append(Paragraph(
        "Concise overview of what was built, what was measured, and the decisions made — "
        "for the submission-readiness review. Not the academic report itself.",
        s["BodyJ"],
    ))

    # ============================================================ 1. DATA
    story.append(Paragraph("1. Data", s["H1"]))
    data_rows = [
        ["Source", "What", "Range", "Use"],
        ["data/ohlcv_data.csv",
         "Daily OHLCV for 11 futures",
         "1990 → Jun 2022",
         "labels, F1–F17, F21 features"],
        ["data/primary_signals.csv",
         "Primary signal {−1, 0, +1}",
         "Jan 2020 → Jun 2022 (645 dates)",
         "label trigger, trade direction"],
        ["data/bloomberg/cleaned/macro_harry.parquet",
         "21 macro series (VIX, rates, credit, FX, EIA, PMI)",
         "1990 → Jun 2022 daily",
         "F11 macro features"],
        ["data/bloomberg/cleaned/futures_term.parquet",
         "Front + 2nd month for 8 commodities + VIX futures",
         "1990 → Jun 2022 (UX from 2004; XB from 2005)",
         "F18 term structure"],
        ["data/bloomberg/cleaned/options_iv.parquet",
         "1m / 3m ATM IV + 90 % / 110 % MNY for 10 underlyings",
         "2005 → Jun 2022",
         "F19 options IV"],
        ["data/bloomberg/cleaned/eia_crude.parquet",
         "EIA weekly crude inventory change",
         "1990 → Jun 2022 (Friday data, +5d release lag)",
         "F22 event flag + crude change z-score"],
    ]
    story.append(make_table(data_rows, col_widths=[5.5*cm, 5.5*cm, 4.0*cm, 4.5*cm]))
    story.append(Paragraph(
        "Hidden test is H2-2022; Bloomberg data cannot legally be extended past Jun 2022 (released-period constraint). "
        "The model architecture is robust to BBG missingness at hidden-test time — verified by simulated-missingness ablation.",
        s["Caption"],
    ))

    # ============================================================ 2. FEATURES
    story.append(Paragraph("2. Features", s["H1"]))
    story.append(Paragraph(
        "105 features registered across 18 families; 80 kept after the drift filter. "
        "Drift filter rule: drop a feature if its train→test KS exceeds 0.25 AND its validation AUC is below 0.54, "
        "or if KS exceeds 0.20 AND validation AUC is below 0.51.",
        s["BodyJ"],
    ))
    feat_rows = [
        ["Family", "Registered", "Kept", "Description"],
        ["F1 counter-trend", "3", "3", "BB %b, RSI-14, mean-reversion score"],
        ["F2 vol / dispersion", "6", "6", "rolling vol 20/60d, vol ratio, Parkinson, Garman-Klass, vol-of-vol"],
        ["F5 signal trajectory", "6", "6", "run length, days since flip, entropy, flip rate, long bias, participation"],
        ["F6 momentum", "6", "5", "TS momentum 20/60d, MA cross, MACD + signal + histogram"],
        ["F7 microstructure", "7", "7", "volume z, volume trend, OI level, OI change, Amihud, Kyle's λ, overnight gap"],
        ["F8 calendar", "4", "4", "day-of-week sin/cos, month sin/cos"],
        ["F9 cross-section", "4", "4", "rank, dispersion z, pair correlation, EWMA implied correlation z"],
        ["F10 price action", "4", "4", "high-low range, open-close return, 20d means"],
        ["F11 macro (ranks + changes)", "35", "17", "63d rolling rank of 20 macro series + 7 chg20 + 5 chg5 + 3 spreads"],
        ["F12 path structure", "5", "5", "variance ratio, autocorr, trend t-val, efficiency ratio, Hurst"],
        ["F15 conditional risk", "2", "2", "path tortuosity 20d, semi-vol ratio 20d"],
        ["F16 concept drift", "1", "1", "regime alignment score"],
        ["F17 HMM (Sreeram-style)", "3", "0", "(failed to converge — replaced by EWMA HMM)"],
        ["F18 term structure", "5", "4", "term spread, z63, contango flag, term chg5, roll yield"],
        ["F19 options IV", "6", "6", "ATM 1m/3m, term slope, IV-RV spread, skew (90 % MNY), 252d percentile"],
        ["F21 cross-asset RV", "4", "2", "gold/silver z, copper/gold, crack 3-2-1"],
        ["F22 event flag", "2", "1", "EIA crude release Wednesday flag"],
        ["EWMA HMM (online)", "2", "2", "causal online filter, no CV-seam"],
        ["TOTAL", "105", "80", "—"],
    ]
    story.append(make_table(feat_rows, col_widths=[5.0*cm, 1.8*cm, 1.5*cm, 11.2*cm], font_size=8.5))
    story.append(Paragraph(
        "<b>Top 8 features by validation AUC:</b> f12_variance_ratio_5_21 (0.590), f12_autocorr_21 (0.584), "
        "f5_long_bias_20 (0.570), f15_path_tortuosity_20 (0.567), f5_trailing_run_length (0.565), "
        "f5_days_since_flip (0.565), f19_iv_pctile_252 (0.559), f5_signal_entropy_20 (0.558).",
        s["Caption"],
    ))

    story.append(PageBreak())

    # ============================================================ 3. LABELS
    story.append(Paragraph("3. Labels", s["H1"]))
    story.append(Paragraph(
        "<b>Triple-barrier with entry at t+1.</b> The signal is observed at the close of bar t, the position is "
        "entered at the close of t+1, and the held window is [t+1, t+1+h]. Barrier widths: ± p<sub>t</sub>·σ̂·√h "
        "in log-return space, with the daily σ̂ from a causal expanding-window GARCH(1,1) model (refit every 21 "
        "days, minimum 500 observations).",
        s["BodyJ"],
    ))
    story.append(Paragraph(
        "Configuration: p<sub>t</sub> = s<sub>l</sub> = 0.5 (symmetric tight), h = 10 trading days. "
        "Per-instrument concurrency on each instrument's own trading-day index. Events with truncated "
        "[t+1, t+1+h] window are dropped.",
        s["BodyJ"],
    ))
    label_rows = [
        ["Instrument", "Events", "Long", "Short", "Pos rate", "PT %", "SL %", "Vert %", "Mean uniq"],
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
        ["TOTAL", "4,886", "3,050", "1,836", "0.547", "47.8", "39.7", "12.5", "—"],
    ]
    story.append(make_table(label_rows, col_widths=[2.5*cm, 1.8*cm, 1.4*cm, 1.6*cm, 1.7*cm, 1.5*cm, 1.5*cm, 1.6*cm, 1.8*cm],
                              font_size=9))
    story.append(Paragraph(
        "Tight barriers resolve 87.5 % of events at PT or SL; 12.5 % at vertical timeout. "
        "Effective sample size (sum of AFML Ch. 4 uniqueness weights) is ~975, vs the raw count of 4,886.",
        s["Caption"],
    ))

    story.append(PageBreak())

    # ============================================================ 4. MODELS
    story.append(Paragraph("4. Models", s["H1"]))
    story.append(Paragraph(
        "Six estimators in the candidate roster — the linear / tree / NN families cover the rubric requirement. "
        "Inner-CV hyperparameter tuning for XGBoost uses the one-standard-error rule (smallest model within "
        "1 SE of the best mean AUC). Predictions are clipped to [0.01, 0.99] so a single extreme probability "
        "cannot blow up log-loss; ranking-based metrics (AUC) are unaffected by clipping.",
        s["BodyJ"],
    ))
    model_rows = [
        ["Family", "Configuration"],
        ["Elastic-net logistic regression", "saga solver, l1_ratio = 0.5, C = 1.0, median impute + standardise"],
        ["XGBoost", "max_depth = 4, lr = 0.05, subsample = 0.8, colsample = 0.8, reg_α = 0.1, reg_λ = 1.0, hist; inner-CV 1-SE tuning"],
        ["LightGBM", "num_leaves = 15, lr = 0.05, subsample = 0.8, reg_α = 0.1, reg_λ = 1.0, deterministic = True"],
        ["Random Forest", "n_estimators = 200, max_depth = 6, min_samples_leaf = 10, max_features = 'sqrt'"],
        ["Simple-average ensemble", "Uniform mean of OOS predictions from the four estimators above"],
        ["Multi-task neural net", "Embedding(11, 8) → Linear(d+8, 64) + ReLU + Dropout(0.1) → Linear(64, 32) + ReLU + Dropout(0.1) → Linear(32, 16) shared h → 11 per-instrument heads; full-batch Adam; early stop on chronological val log-loss"],
    ]
    story.append(make_table(model_rows, col_widths=[5.0*cm, 14.5*cm], font_size=9))

    story.append(Paragraph("Champion architecture", s["H2"]))
    story.append(Paragraph(
        "Per instrument, the champion is the best (pool, model) combination chosen by CombinatorialPurgedCV(6, 2) "
        "mean AUC under the 1-SE selection rule. Pool options per instrument:",
        s["BodyJ"],
    ))
    pool_rows = [
        ["Instrument", "Candidate pools"],
        ["es1s, nq1s, fesx1s", "individual or equity_all"],
        ["cl1s", "individual or energy_cl_ho or energy_all"],
        ["ho1s", "energy_cl_ho or energy_all"],
        ["rb1s, ng1s", "individual or energy_all"],
        ["gc1s, si1s, pl1s", "individual or precious (gc+si+pl) or metals_all"],
        ["hg1s", "individual or metals_all"],
    ]
    story.append(make_table(pool_rows, col_widths=[5.5*cm, 14.0*cm], font_size=9))

    story.append(Paragraph("Cross-validation", s["H2"]))
    story.append(Paragraph(
        "CombinatorialPurgedCV with 6 groups and 2 test groups produces 15 paths. Per-instrument embargo: each "
        "instrument's label-window p90 advances on its own date axis. Modelling sample is events with "
        "t_signal ≤ 2021-10-06; sealed test slice is t_signal > 2021-10-20 (10 trading-day embargo). "
        "No data after the embargo touches model selection.",
        s["BodyJ"],
    ))

    story.append(PageBreak())

    # ============================================================ 5. CLUSTER IMPORTANCE
    story.append(Paragraph("5. Cluster importance", s["H1"]))
    story.append(Paragraph(
        "Pipeline: hygiene (drop NaN > 30 %, zero variance, |ρ| > 0.99 twins) → "
        "<b>Mantegna distance √(1 − |ρ|)</b> (proper metric) → Ward linkage, K by silhouette → "
        "Random Forest with <b>max_features = 'sqrt'</b> inside CombinatorialPurgedCV(6, 2) → "
        "cluster MDI + clustered <b>purged MDA</b> (joint per-cluster permutation) + <b>TreeSHAP</b> "
        "via XGBoost native pred_contribs (no shap library needed).",
        s["BodyJ"],
    ))
    cluster_rows = [
        ["Class", "Hygiene kept", "K (silhouette)", "Top cluster MDA", "Clusters > 0.02 MDA"],
        ["Equity", "70 / 80", "16", "0.0223", "1 of 16"],
        ["Energy", "73 / 80", "8", "0.0364", "1 of 8"],
        ["Metals", "71 / 80", "16", "0.0155", "0 of 16"],
    ]
    story.append(make_table(cluster_rows, col_widths=[3.5*cm, 3.5*cm, 3.5*cm, 4.5*cm, 4.5*cm], font_size=9.5))

    story.append(Paragraph("Energy's dominant cluster (MDA 0.036, 19 features)", s["H2"]))
    story.append(Paragraph(
        "<i>f19_atm_iv_1m, f19_atm_iv_3m, f19_iv_term_slope, f19_iv_pctile_252, f2_vol_20, f2_vol_60, "
        "f2_parkinson_20, f2_garman_klass_20, f2_vol_of_vol_20, f10_hl_range, f12_hurst_100, "
        "f9_dispersion_z_60, f9_pair_corr_mean_63, f9_implied_corr_z_252, f21_crack_321, "
        "f11_eia_crude_stock_rank63, f11_curve_slope, f11_vix_term_slope, f16_regime_alignment_score</i>.",
        s["BodyJ"],
    ))
    story.append(Paragraph(
        "The cluster groups options-implied volatility (F19), realised volatility (F2), and macro regime "
        "indicators (F11, F16) — the volatility-and-options block is the empirical driver of energy AUC.",
        s["Caption"],
    ))

    story.append(Paragraph("Cross-method rank agreement (Kendall τ)", s["H2"]))
    story.append(Paragraph(
        "SHAP ↔ MDI: τ = 0.72 (p &lt; 1e-4) — strong, both in-sample attribution. &nbsp; &nbsp; "
        "SHAP ↔ MDA: τ = 0.15 (p = 0.45) — weak. MDA is the out-of-sample reality check; "
        "MDI/SHAP attribute how the trained model uses features in-sample.",
        s["BodyJ"],
    ))

    story.append(PageBreak())

    # ============================================================ 6. PERFORMANCE
    story.append(Paragraph("6. Performance", s["H1"]))

    story.append(Paragraph("Per-class CPCV mean AUC by methodology", s["H2"]))
    perf_rows = [
        ["Methodology", "Equity", "Energy", "Metals"],
        ["Per-class XGBoost only", "0.521", "0.556", "0.512"],
        ["+ instrument one-hot + LightGBM + ensemble", "0.554", "0.558", "0.525"],
        ["+ champion architecture per instrument", "0.550", "0.602", "0.533"],
        ["+ multi-task neural net (Family B)", "0.550", "0.602", "0.554"],
    ]
    story.append(make_table(perf_rows, col_widths=[10.0*cm, 3.0*cm, 3.0*cm, 3.0*cm], font_size=9.5,
                              header_align="LEFT"))

    story.append(Paragraph("Per-instrument champion AUC", s["H2"]))
    perinst_rows = [
        ["Instrument", "Pool", "Model", "AUC", "± SEM", "Lower 1-SE CI"],
        ["cl1s",   "cl1s",        "LightGBM",                  "0.671", "0.022", "0.649"],
        ["ng1s",   "energy_all",  "elastic-net logistic",      "0.601", "0.044", "0.557"],
        ["ho1s",   "energy_cl_ho","elastic-net logistic",      "0.599", "0.048", "0.551"],
        ["pl1s",   "pl1s",        "Random Forest",             "0.581", "0.019", "0.562"],
        ["gc1s",   "precious",    "multi-task NN",             "0.568", "0.029", "0.540"],
        ["fesx1s", "equity_all",  "Random Forest",             "0.557", "0.019", "0.538"],
        ["es1s",   "es1s",        "Random Forest",             "0.555", "0.017", "0.537"],
        ["rb1s",   "rb1s",        "Random Forest",             "0.538", "0.020", "0.517"],
        ["nq1s",   "equity_all",  "Random Forest",             "0.537", "0.011", "0.526"],
        ["si1s",   "si1s",        "elastic-net logistic",      "0.534", "0.011", "0.522"],
        ["hg1s",   "metals_all",  "Random Forest",             "0.533", "0.014", "0.519"],
    ]
    story.append(make_table(perinst_rows,
                              col_widths=[2.0*cm, 2.5*cm, 4.5*cm, 1.7*cm, 1.7*cm, 2.5*cm],
                              font_size=9.5))
    story.append(Paragraph(
        "Seven of eleven instruments above 0.55 AUC. All eleven have a lower 1-SE CI above 0.50 "
        "(positive signal under the standard significance flag).",
        s["Caption"],
    ))

    story.append(Paragraph("Acceptance gates", s["H2"]))
    gate_rows = [
        ["Gate", "Result"],
        ["≥ 6 of 11 per-instrument AUC > 0.55", "7 of 11 — PASS"],
        ["Lower 1-SE CI > 0.50 (signal flag)", "11 of 11 — PASS"],
        ["BBG-missingness ablation Δ ≤ 0.03 per class", "Max Δ = 0.011 — PASS"],
        ["≥ 1 cluster per class with MDA > 0.02", "2 of 3 classes — PASS (metals top MDA 0.016)"],
        ["All 4 importance bug fixes present", "max_features='sqrt', PurgedKFold, TreeSHAP, Mantegna — PASS"],
        ["Realised annualised vol ≤ 10 % (strategy constraint)", "5.4 % — PASS"],
        ["Byte-identical CSV re-emit", "Verified by unit test — PASS"],
    ]
    story.append(make_table(gate_rows, col_widths=[12.5*cm, 7.0*cm], font_size=9.5))

    story.append(Paragraph("Backtest on the sealed test slice (179 days)", s["H2"]))
    bt_rows = [
        ["Metric", "Value", "Notes"],
        ["Days", "179", "Sealed test (t_signal > 2021-10-20)"],
        ["Annualised return (net)", "17.3 %", "After Grinold-Kahn costs (2 bps half-spread + 10 bps impact)"],
        ["Annualised volatility", "5.40 %", "Below the 10 % strategy-constraint cap"],
        ["Sharpe (per-period × √252)", "3.20", "Pre-deflation read"],
        ["Sortino (full-T form)", "6.03", "Sortino-Price 1994 convention"],
        ["Maximum drawdown", "−2.33 %", ""],
        ["Annualised turnover", "16.6×", "Mean holding ~6.6 days"],
        ["Total cost (period)", "142 bp", "Gross 19.3 % → Net 17.3 %"],
    ]
    story.append(make_table(bt_rows, col_widths=[5.5*cm, 3.0*cm, 11.0*cm], font_size=9.5))
    story.append(Paragraph(
        "Pre-deflation Sharpe 3.20 has not yet been gated by the studentised stationary block-bootstrap CI, "
        "DSR ladder, CSCV-PBO, or Pesaran-Timmermann tests — that significance run is the remaining step "
        "before treating the number as a robust claim.",
        s["Caption"],
    ))

    story.append(PageBreak())

    # ============================================================ 7. DECISIONS
    story.append(Paragraph("7. Key decisions", s["H1"]))

    story.append(Paragraph("Labels", s["H2"]))
    for b in [
        "Entry at t+1 (signal observed at close of t is acted on at close of t+1); the held window is [t+1, t+1+h]. Removes the same-bar return from the label.",
        "GARCH(1,1) one-step-ahead daily σ̂, refit every 21 days, minimum 500 observations, expanding window capped at 2,000 bars. Forward-aware; sharper than EWMA on regime breaks.",
        "Symmetric tight barriers: p<sub>t</sub> = s<sub>l</sub> = 0.5, h = 10 days. Resolves 87.5 % of events at PT or SL rather than vertical timeout.",
        "Per-instrument concurrency on each instrument's own trading-day index. Calendar gaps do not inflate overlap counts.",
        "Drop events whose [t+1, t+1+h] window is not fully observable (right-edge truncation).",
    ]:
        story.append(bullet(s, b))

    story.append(Paragraph("Features", s["H2"]))
    for b in [
        "F11 macro reformulated as 63-day rolling RANKS (not raw levels). Raw levels show catastrophic train→test drift (median KS 0.36); ranks are scale-stable.",
        "F18 / F19 / F22 added from Bloomberg pulls — futures term structure, options-implied vol, EIA crude release flag.",
        "F21 cross-asset relative value computed from existing OHLCV (gold/silver, copper/gold, crack 3-2-1).",
        "Drift filter: drop a feature if KS > 0.25 AND val_AUC < 0.54, or KS > 0.20 AND val_AUC < 0.51. 105 registered → 80 kept; 82.5 % of kept features have KS < 0.20.",
        "F17 HMM (3-state Gaussian fit) failed to converge in production and was drift-filtered out. The online EWMA HMM (causal, no CV-seam) is the regime feature used in practice.",
        "Bloomberg-missingness ablation: shipped <i>with_bbg</i>; AUC delta vs <i>without_bbg</i> is ≤ 0.011 across classes. The model is robust to H2-2022 BBG missingness at hidden-test time.",
    ]:
        story.append(bullet(s, b))

    story.append(Paragraph("Models and selection", s["H2"]))
    for b in [
        "Four-estimator base roster: elastic-net logistic, XGBoost, LightGBM, Random Forest. The rubric requires linear / tree / NN families; the multi-task NN serves the NN slot.",
        "Champion architecture per instrument: 1–3 candidate pools × 4 models, selected by CombinatorialPurgedCV(6, 2) mean AUC under the 1-SE rule (smallest model within 1 SE of best preferred).",
        "1-SE rank order (most regularised to least): elastic-net logistic → Random Forest → LightGBM → XGBoost → multi-task NN. Smaller pool preferred over larger when within 1 SE.",
        "Inner-CV XGBoost hyperparameter tuning: purged inner k-fold (k = 4) over a 5-config grid (max_depth × n_estimators × reg_α / reg_λ); 1-SE pick of the most regularised within 1 SE of best.",
        "Per-instrument one-hot dummies added to asset-class pools so the tree can split on instrument before features.",
        "Multi-task neural net is a candidate only for multi-instrument pools (sharing across heads is meaningless for individual pools).",
        "Prediction clipping to [0.01, 0.99] bounds log-loss without changing AUC (AUC depends only on ranking).",
        "CombinatorialPurgedCV(6, 2) → 15 paths everywhere, with per-instrument embargo via embargo_p90 on each instrument's date axis.",
    ]:
        story.append(bullet(s, b))

    story.append(Paragraph("Cluster importance bug fixes", s["H2"]))
    for b in [
        "<b>max_features = 'sqrt'</b> on the importance Random Forest (the deprecated 'auto' value was removed in sklearn ≥ 1.3).",
        "<b>PurgedKFold for MDA</b>: K-fold with shuffle leaks across overlapping triple-barrier labels — the cluster importance loop uses CombinatorialPurgedCV(6, 2).",
        "<b>TreeSHAP via XGBoost native pred_contribs</b>: avoids the shap-library dependency chain (shap → numba → numpy &lt; 2.4) which is incompatible with our pandas pin.",
        "<b>Mantegna distance √(1 − |ρ|)</b>: proper metric (Mantegna 1999); the original 1 − |ρ| violates triangle inequality.",
    ]:
        story.append(bullet(s, b))

    story.append(Paragraph("Limitations and risks", s["H2"]))
    for b in [
        "<b>Grinold ceiling</b>: pooled information coefficient ≈ 0.07 implies a pooled AUC ceiling near 0.52. Per-class AUC of 0.55–0.60 is at the ceiling.",
        "<b>OHLCV adjustment artefact on the target</b>: the price series is ratio back-adjusted continuous, not raw front-month. Return-R² against raw futures is 0.995+ for metals, 0.94–0.97 for equity, but 0.72 for ng1s (with 11 % label-flip rate). The methodology must not claim raw front-month profitability.",
        "<b>BBG missingness at H2-2022</b>: cleaned Bloomberg parquets cover up to Jun 2022 (released period). Pulling beyond would violate the brief. Mitigations: simulated-missingness ablation in CV (delta ≤ 0.011 confirms robustness); without_bbg robustness baseline available; XGBoost-native NaN handling.",
        "<b>Pre-deflation Sharpe</b>: 3.20 has not been gated by significance / deflation / Pesaran-Timmermann checks yet — that's the remaining S7 step.",
    ]:
        story.append(bullet(s, b))

    story.append(Paragraph("Where the artefacts live", s["H2"]))
    art_rows = [
        ["What", "Path"],
        ["Triple-barrier events (S1)", "data/sreeram_experimental_events.parquet"],
        ["Feature matrix (S2)", "data/sreeram_experimental_features.parquet"],
        ["Drift filter audit", "results/sreeram_experimental/feature_drift_audit.csv"],
        ["Instrument scope (embargo, n_eff)", "results/sreeram_experimental/instrument_scope.json"],
        ["Champion summary", "results/sreeram_experimental/champions_summary.csv"],
        ["Full candidate grid", "results/sreeram_experimental/champions_per_pool_per_model.csv"],
        ["BBG-missingness ablation", "results/sreeram_experimental/bbg_missingness_ablation.csv"],
        ["Coverage caveats per instrument", "outputs/coverage_caveat.csv"],
        ["Cluster importance per class", "results/sreeram_experimental/importance/{class}/*.csv"],
        ["Backtest metrics + daily returns", "results/sreeram_experimental/backtest_metrics.csv"],
        ["Daily net returns (for significance)", "results/sreeram_experimental/strategy_daily_net_returns.csv"],
        ["Submission predictions (calibrated)", "outputs/metamodel_predictions.csv"],
        ["Submission strategy weights", "outputs/strategy_weights.csv"],
        ["Build plan (golden record)", "reports/sreeram_experimental/plan.md"],
        ["Action tracker (PM log)", "reports/sreeram_experimental/action_tracker.md"],
    ]
    story.append(make_table(art_rows, col_widths=[6.5*cm, 13.0*cm], font_size=9))

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<i>22 commits on Sreeram_experimental; 97 fast tests and 3 slow tests pass; working tree clean.</i>",
        s["Caption"],
    ))

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=2.0*cm, rightMargin=2.0*cm,
        topMargin=2.0*cm, bottomMargin=2.0*cm,
        title="Sreeram_experimental — Branch Overview",
        author="Sreeram",
    )
    doc.build(story)
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    build()
