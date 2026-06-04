#!/usr/bin/env python3
"""Generate the three per-asset-class final-submission notebooks.

One notebook per asset class (equity, energy, metals) narrating the full
pipeline — data, signal, labels, features, model selection, feature importance
and strategy attribution — by *loading committed CSVs and embedding committed
PNGs*. No model is trained or retrained here; every figure and table is an
artifact already produced by the live pipeline (``src/stml/harry`` +
``src/stml/new_work``).

Run from anywhere::

    uv run python src/stml/new_work/_gen_final_notebooks.py          # write
    uv run python src/stml/new_work/_gen_final_notebooks.py --check   # validate only

Design notes
------------
* Scaffolding (``code_cell``/``md_cell``/``notebook``/``_uid``) mirrors the
  team's existing ``_gen_*.py`` generators.
* Per-class *analysis* prose lives in ``NARRATIVES`` and is hand-authored and
  distinct; only loading/rendering/embedding is parametrised. Shared
  *methodology* blocks (triple-barrier mechanics, CPCV, costs) are deliberately
  identical across the three self-contained notebooks.
* Headline numbers are loaded inside the notebook, never transcribed from
  report prose — the repo's reports have mis-described its code before (e.g. the
  barrier multipliers; see the labels section).
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (this file: src/stml/new_work/_gen_final_notebooks.py)
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[3]          # new_work → stml → src → repo
OUT = REPO / "src/stml/new_work/outputs"
RESULTS = REPO / "results"
REPORTS = REPO / "reports"
DATA = REPO / "data"
DEST = REPO / "notebooks/submission"

# --------------------------------------------------------------------------- #
# Notebook scaffolding
# --------------------------------------------------------------------------- #
def _uid() -> str:
    return str(uuid.uuid4())


def code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "id": _uid(),
        "metadata": {},
        "source": src.rstrip("\n"),
        "outputs": [],
        "execution_count": None,
    }


def md_cell(src: str) -> dict:
    return {"cell_type": "markdown", "id": _uid(), "metadata": {}, "source": src.rstrip("\n")}


def notebook(cells: list) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (stml)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "cells": cells,
    }


# --------------------------------------------------------------------------- #
# Per-class configuration
# --------------------------------------------------------------------------- #
CLASSES: dict[str, dict] = {
    "equity": {
        "title": "Equity",
        "insts": ["es1s", "nq1s", "fesx1s"],
        "names": {
            "es1s": "S&P 500 e-mini (ES)",
            "nq1s": "Nasdaq-100 e-mini (NQ)",
            "fesx1s": "Euro Stoxx 50 (FESX)",
        },
    },
    "energy": {
        "title": "Energy",
        "insts": ["cl1s", "ho1s", "rb1s", "ng1s"],
        "names": {
            "cl1s": "WTI crude (CL)",
            "ho1s": "Heating oil (HO)",
            "rb1s": "RBOB gasoline (RB)",
            "ng1s": "Henry Hub natural gas (NG)",
        },
    },
    "metals": {
        "title": "Metals",
        "insts": ["gc1s", "si1s", "hg1s", "pl1s"],
        "names": {
            "gc1s": "Gold (GC)",
            "si1s": "Silver (SI)",
            "hg1s": "Copper (HG)",
            "pl1s": "Platinum (PL)",
        },
    },
}

# --------------------------------------------------------------------------- #
# Hand-authored, class-specific narrative prose (British academic register).
# Each value is a distinct passage — NOT a shared template with the class name
# substituted. Tested for pairwise dissimilarity in
# tests/harry/test_submission_notebooks.py.
# --------------------------------------------------------------------------- #
NARRATIVES: dict[str, dict[str, str]] = {
    "equity": {
        "abstract": (
            "This notebook assembles the end-to-end metamodel for the three equity index "
            "futures — the S&P 500, Nasdaq-100 and Euro Stoxx 50 e-minis. The primary signal "
            "behaves as a short-horizon contrarian rule across the index complex: positions "
            "entered the day after a signal earn a positive forward return, yet they lean "
            "against the preceding day's move. The metamodel's job is narrower than the "
            "signal's — it predicts whether a given trigger is worth taking. Among the three, "
            "the Nasdaq carries the cleanest edge, the S&P is suggestive but rests on a thin "
            "single-shot test, and the Euro Stoxx fails to clear chance out of sample. The "
            "sections below trace that conclusion from raw prices through to the instruments' "
            "contribution to the pooled book."
        ),
        "signal": (
            "All three indices show a positive next-day relationship between the signal and "
            "the realised return, which is what makes the trigger tradeable at a one-day lag. "
            "The contrarian character is equally clear: the signal loads negatively on the "
            "trailing one-day move, so it fires after pull-backs rather than with momentum. "
            "The Nasdaq and S&P express this most strongly; the Euro Stoxx is the weakest of "
            "the three and, as the model section confirms, its edge does not survive an honest "
            "hold-out. Read the per-instrument correlations below as a description of the "
            "signal's construction, not a promise about the metamodel."
        ),
        "models": (
            "The horse race lands on a different architecture for each index, which is itself "
            "informative — there is no single equity model. A penalised logistic fit wins for "
            "the S&P, gradient boosting for the Nasdaq, and a random forest for the Euro Stoxx. "
            "The first two clear their lower confidence bound and are flagged as carrying "
            "signal; the Euro Stoxx forest sits below the 0.5 line out of sample and is not. "
            "Note the gap between the cross-validated and single-shot numbers for the S&P: a "
            "respectable development score paired with a thin test set is a hypothesis, not a "
            "verdict."
        ),
        "importance": (
            "Clustering the features before ranking them matters here because the equity "
            "predictors are correlated in blocks — momentum and range measures move together, "
            "as do the macro volatility and credit gauges. The cluster-level drops show the "
            "signal's own trajectory features and the volatility-regime block doing most of "
            "the work for the Nasdaq and S&P. For the Euro Stoxx the importance picture is "
            "diffuse, consistent with a model that has found little to hold on to."
        ),
        "discussion": (
            "The honest equity read is one clear name, one tentative name and one pass. The "
            "Nasdaq metamodel is the result worth carrying forward. The S&P is plausible but "
            "should be treated as a hypothesis until a longer test confirms it. The Euro Stoxx "
            "offers no exploitable edge under the metamodel and would dilute the book if traded "
            "on conviction. None of this contradicts the signal being real; it says the filter "
            "adds value unevenly across the complex."
        ),
    },
    "energy": {
        "abstract": (
            "This notebook builds the metamodel for the energy complex — WTI crude, heating "
            "oil, RBOB gasoline and Henry Hub natural gas. Energy is the awkward, and most "
            "rewarding, corner of the universe. The instruments do not share a single signal "
            "character: crude behaves unlike the refined products, and natural gas is thin "
            "enough that it must be modelled in a pool rather than alone. Despite that "
            "heterogeneity, energy is the engine of the pooled strategy's out-of-sample return. "
            "The sections trace how the signal, the labelling and the per-instrument model "
            "choices combine into that contribution."
        ),
        "signal": (
            "Energy breaks the tidy contrarian story seen elsewhere. Crude is the outlier: its "
            "trailing-return correlation is close to zero and its hit rate is the highest in "
            "the universe, so it is better treated as a near-momentum special case than lumped "
            "with the refined products. Heating oil and gasoline reverse more conventionally, "
            "while natural gas has so few clean bets that its statistics are directional at "
            "best. This is why the modelling stage pools the thin names rather than fitting "
            "each in isolation."
        ),
        "models": (
            "The per-instrument choices follow the signal's heterogeneity. Crude takes a "
            "standalone penalised logistic; the refined products and gas borrow strength from "
            "pooled fits (the crude–heating-oil pool and the all-energy pool) because their own "
            "event counts are too small to support an independent model. Gasoline's locked "
            "variant is deliberately minimal — a check on whether its apparent edge is anything "
            "more than the raw signal plus a credit spread. The locked picks and their "
            "out-of-sample AUCs below should be read with the pooling in mind: a pooled score "
            "is a statement about the group, not the single name."
        ),
        "importance": (
            "Because several energy names are modelled in pools, the importance plots speak to "
            "shared structure as much as to any one contract. The commodity-fundamental macro "
            "block — inventory surprises and the dollar — earns its place alongside the signal "
            "trajectory features, which is the expected fingerprint of a complex driven by "
            "physical supply as well as positioning. Crude's standalone importance picture is "
            "the most self-contained of the four."
        ),
        "discussion": (
            "Energy is where the metamodel pays for itself. The complex contributes the bulk "
            "of the pooled book's out-of-sample return, with crude and the refined products "
            "doing the heavy lifting and natural gas contributing little once the calibrated "
            "filter has spoken. The cost of that return is modelling complexity: pooling, a "
            "special case for crude and a minimal-variant guard for gasoline. The pay-off "
            "justifies the bookkeeping, but the thin-instrument caveats should travel with the "
            "result."
        ),
    },
    "metals": {
        "abstract": (
            "This notebook covers the metals — gold, silver, copper and platinum. Of the three "
            "asset classes this is the cleanest expression of the signal's contrarian "
            "construction: every metal reverses against its prior move. That tidiness at the "
            "signal level does not, however, translate into a strong metamodel edge. The most "
            "interesting tension in the whole study lives here — a feature set that looks "
            "predictive in cross-validation can still collapse to chance out of sample, and "
            "gold is the clearest example. The sections below follow that thread from the "
            "signal audit to the locked models and their honest hold-out scores."
        ),
        "signal": (
            "Metals are the textbook case. All four contracts load negatively on the trailing "
            "one-day return and positively on the forward return, so the signal is uniformly "
            "short-horizon mean reversion with no awkward exceptions of the kind crude "
            "introduces in energy. If signal cleanliness alone decided the matter, metals would "
            "be the strongest class. The model section shows why it is not: a well-behaved "
            "signal is necessary but not sufficient for the metamodel to add value."
        ),
        "models": (
            "Random forests win across the metals, and the precious names are pooled where "
            "their individual event counts are thin. The cautionary result is gold: a reduced "
            "feature set lifts its cross-validated AUC above 0.56, yet the same model drops "
            "below 0.5 out of sample. That pattern — strong in development, chance in the "
            "hold-out — is the signature of fitting to the top few importance clusters rather "
            "than to anything that generalises. The locked picks below are honest about which "
            "metals clear their confidence bound and which do not."
        ),
        "importance": (
            "The clustered importance plots for metals are worth reading against the model "
            "caveat. Where a name's apparent edge rests on two or three significant clusters, "
            "the reduced-feature model that exploits them is exactly the one that fails to "
            "generalise. Silver and gold illustrate this; copper and platinum present a "
            "steadier, if modest, picture. The dendrograms make the block structure of the "
            "metals feature set visible, which is what motivates clustering the importances in "
            "the first place."
        ),
        "discussion": (
            "Metals close the study on a deliberately sober note. The signal is the cleanest in "
            "the universe, yet the metamodel edge is the weakest: gold is effectively random "
            "out of sample, silver is marginal, and only copper and platinum offer a steady if "
            "small contribution. The lesson is methodological as much as empirical — a high "
            "cross-validated score on a reduced feature set is not evidence of an exploitable "
            "edge, and the hold-out is the only arbiter that matters."
        ),
    },
}

# --------------------------------------------------------------------------- #
# Shared methodology prose (identical across notebooks by design; each notebook
# is a self-contained submission document). Cited to the report it summarises.
# --------------------------------------------------------------------------- #
def _md_intro(cfg) -> str:
    insts = ", ".join(f"`{i}`" for i in cfg["insts"])
    return (
        f"# {cfg['title']} — metamodel pipeline\n\n"
        f"{NARRATIVES[_key(cfg)]['abstract']}\n\n"
        f"**Universe.** {insts} ({len(cfg['insts'])} contracts).\n\n"
        "**Reading guide.** The notebook runs top to bottom against committed artifacts; no "
        "model is retrained. Each stage loads its result files and embeds the figures already "
        "produced by the pipeline. Section order mirrors the marking scheme in "
        "`reports/harry/00-context.md`: data, signal, labelling, features, model selection, "
        "cluster importance, evaluation.\n\n"
        "_Source of truth._ Where a number appears it is loaded from a file in this cell's "
        "output, not transcribed from prose; the few places where the committed data and the "
        "written reports disagree are flagged inline."
    )


def _md_data() -> str:
    return (
        "## 1. Data and cleaning\n\n"
        "The price history runs from 1990, while the primary signal is released only over "
        "2020-01-03 to 2022-06-30; the sealed test period sits beyond the embargo. Loading goes "
        "through `stml.io`, and the missing-data policy — keep zero-volume weekday settles, drop "
        "weekend and out-of-bounds rows — is documented in `reports/missing-data-report.md` and "
        "implemented in `stml.na_checks`. The coverage table below is computed directly from "
        "`data/ohlcv_data.csv` for this class's contracts."
    )


def _md_labels() -> str:
    return (
        "## 3. Labelling\n\n"
        "Events use the triple-barrier method of López de Prado (2018, ch. 3) with entry at "
        "`t+1`: the signal is observed at the close of bar `t` and the position is opened the "
        "following session, which is justified by the positive next-day correlation shown above. "
        "Profit-taking and stop-loss barriers are set in units of a causal volatility estimate, "
        "a vertical barrier caps the holding period, and overlapping events are down-weighted by "
        "average uniqueness (López de Prado, 2018, ch. 4). The methodology is set out in "
        "`reports/harry/02-labels.md`.\n\n"
        "The summary below is computed from `results/harry/events.csv` (the event table that "
        "feeds the feature and model stages).\n\n"
        "<!-- TODO: verify barrier parameters against the labelling code. "
        "reports/harry/02-labels.md states pt=sl=1.0 and h=10, whereas the committed "
        "data/meta/triple_barrier_labels.csv carries pt=sl=0.25 and h=1. The two label sets "
        "differ; events.csv is used here for the per-class statistics. -->"
    )


def _md_features() -> str:
    return (
        "## 4. Features\n\n"
        "Predictors are grouped into causal families, all computed at the signal date and "
        "predictive of the next-day label, so none peeks into the future. The micro families "
        "(`reports/harry/03-features.md`) cover the signal's own trajectory, conditional risk, "
        "information-theoretic dependence, fixed microstructure proxies, cross-asset structure, "
        "an optional wavelet decomposition and a concept-drift discriminator. A macro block of "
        "thirty-one rolling, standardised series (`reports/harry/03-6-macro-features.md`) adds "
        "volatility, rates, credit, dollar, commodity-fundamental and growth context. Two hidden-"
        "Markov regime blocks — per-instrument volatility (`hmm_vol`) and a global risk-on/"
        "risk-off state (`hmm_macro`) — are fit only on pre-sample data and then frozen, so they "
        "are train-fitted rather than leaking. Features are tagged `E` (purely causal) or `TF` "
        "(train-fitted on pre-sample) accordingly."
    )


def _md_models() -> str:
    return (
        "## 5. Model selection\n\n"
        "Each instrument runs a horse race across penalised logistic regression, a random "
        "forest, gradient boosting and a small multi-task network, scored by combinatorial "
        "purged cross-validation (six groups, two held out, fifteen paths) with per-instrument "
        "purging and embargo. Feature variants (full, pruned, reduced) are compared, and the "
        "simplest variant within one cross-path standard error of the best is locked **before** "
        "the out-of-sample window is touched. The cross-validation runs on the training "
        "partition only; the sealed test (after 2021-10-20) is scored once, with the locked "
        "choice fixed."
    )


def _md_importance() -> str:
    return (
        "## 6. Feature importance\n\n"
        "Importance is read at the cluster level rather than per raw feature, because the "
        "predictors are correlated in blocks and per-feature scores would split credit "
        "arbitrarily across substitutes. Features are clustered on a rank-correlation distance; "
        "the dendrogram shows the block structure, mean-decrease-accuracy gives each cluster's "
        "drop, and a global SHAP or coefficient ranking resolves the within-cluster detail. "
        "Logistic champions are read through their coefficients, tree and network champions "
        "through SHAP."
    )


def _md_strategy() -> str:
    return (
        "## 7. Strategy — pooled portfolio and per-class attribution\n\n"
        "There is a single backtest, not one per class. The live strategy is one volatility-"
        "targeted, equally-weighted book over all eleven contracts: an EWMA(60) volatility "
        "estimate scales each position to a 10% annualised target, leverage is capped, returns "
        "lag the weights by a day, and Grinold-Kahn costs (a 2bp half-spread plus 10bp of "
        "turnover) are charged. The conventions are recorded in "
        "`results/strategy_eval/eval_summary.md`. Method A follows every signal; Method B trades "
        "only when the calibrated metamodel probability exceeds one half.\n\n"
        "The headline table and cumulative-return figures below are the **pooled** result. The "
        "per-class table that follows is an **attribution** — this class's slice of the shared "
        "book — and not a standalone per-class Sharpe; with one pooled portfolio the instruments "
        "cannot be isolated into independent backtests."
    )


def _md_references() -> str:
    return (
        "## References\n\n"
        "- López de Prado, M. (2018) *Advances in Financial Machine Learning*. Hoboken, NJ, "
        "Wiley. [Triple-barrier labelling ch. 3; average uniqueness ch. 4; purged and "
        "combinatorial cross-validation ch. 7.]\n"
        "- Project methodology notes (this repository): `reports/harry/00-context.md`, "
        "`01-signal-direction.md`, `02-labels.md`, `03-features.md`, `03-6-macro-features.md`, "
        "and `reports/missing-data-report.md`.\n"
        "- Strategy conventions: `results/strategy_eval/eval_summary.md` (StrategyWeights "
        "lecture, slides 38–43).\n"
        "- Course materials: *Systematic Trading Strategies with Machine Learning* (T3.03), "
        "lectures L1–L8."
    )


def _key(cfg) -> str:
    """Reverse-lookup the class key for a config dict (abstract embedding)."""
    for k, v in CLASSES.items():
        if v is cfg:
            return k
    raise KeyError("unknown class config")


# --------------------------------------------------------------------------- #
# Emitted SETUP / helper cell (parametrised once per class via repr)
# --------------------------------------------------------------------------- #
_SETUP_BODY = '''
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from IPython.display import Image, display, Markdown


def _repo_root() -> Path:
    here = Path.cwd().resolve()
    for q in [here, *here.parents]:
        if (q / "pyproject.toml").exists():
            return q
    return here


REPO = _repo_root()
OUT = REPO / "src/stml/new_work/outputs"
RESULTS = REPO / "results"
REPORTS = REPO / "reports"
DATA = REPO / "data"


def show(path, width=1000):
    p = Path(path)
    if p.exists():
        display(Image(str(p), width=width))
    else:
        print(f"[missing figure] {p}")


def load(path) -> pd.DataFrame:
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    print(f"[missing data] {p}")
    return pd.DataFrame()


def table(df, n=None, caption=None):
    if df is None or len(df) == 0:
        print("[empty table]")
        return
    d = df.head(n) if n else df
    sty = d.style.format(precision=4).hide(axis="index")
    if caption:
        sty = sty.set_caption(caption)
    display(sty)
'''


def setup_cell(cfg) -> dict:
    head = (
        f"CLASS = {_key(cfg)!r}\n"
        f"INSTS = {cfg['insts']!r}\n"
        f"NAMES = {cfg['names']!r}\n"
    )
    return code_cell(head + _SETUP_BODY)


# --------------------------------------------------------------------------- #
# Section code bodies (literal strings; reference CLASS / INSTS / NAMES)
# --------------------------------------------------------------------------- #
_CODE_DATA = '''
ohlcv = load(DATA / "ohlcv_data.csv")
sub = ohlcv[ohlcv["instrument"].isin(INSTS)]
cov = (sub.groupby("instrument")
          .agg(first_date=("date", "min"), last_date=("date", "max"), n_rows=("date", "size"))
          .reindex(INSTS).reset_index())
table(cov, caption=f"{CLASS.title()} — OHLCV coverage (data/ohlcv_data.csv)")
'''

_CODE_SIGNAL = '''
sd = load(RESULTS / "harry/signal_direction.csv")
if not sd.empty:
    want = ["instrument", "n_bets", "corr_fwd_1", "corr_fwd_5", "corr_trail_1", "corr_contemp_0"]
    cols = [c for c in want if c in sd.columns]
    rows = sd[sd["instrument"].isin(INSTS)][cols].set_index("instrument").reindex(INSTS).reset_index()
    table(rows, caption=f"{CLASS.title()} — signal direction (results/harry/signal_direction.csv)")
'''

_CODE_LABELS = '''
ev = load(RESULTS / "harry/events.csv")
ev = ev[ev["instrument"].isin(INSTS)]
if not ev.empty:
    g = (ev.groupby("instrument")
           .agg(n_events=("label", "size"),
                label1_share=("label", "mean"),
                mean_uniqueness=("uniqueness_weight", "mean"),
                mean_sigma=("sigma", "mean"))
           .reindex(INSTS).reset_index())
    table(g, caption=f"{CLASS.title()} — triple-barrier events (results/harry/events.csv)")
    print(f"Total events for {CLASS}: {len(ev):,}  |  label==1 share: {ev['label'].mean():.3f}")
'''

_CODE_FEATURES = '''
# The feature matrix is built by the pipeline; here we confirm the frozen HMM
# regime features are present for this class's instruments.
for fn in ["features_hmm_vol.csv", "features_hmm_macro.csv"]:
    f = load(REPO / "src/stml/new_work" / fn)
    if not f.empty and "instrument" in f.columns:
        n = f[f["instrument"].isin(INSTS)].shape[0]
        print(f"{fn}: {n:,} rows for {CLASS} instruments | columns: {list(f.columns)}")
    elif not f.empty:
        print(f"{fn}: {f.shape[0]:,} rows | columns: {list(f.columns)}")
'''

_CODE_MODELS = '''
mc = OUT / "model_comparison" / CLASS
sel = load(OUT / "model_comparison/selection_table.csv")
locked = load(mc / "locked_picks.csv")
cpcv = load(mc / "cpcv_results.csv")
oos = load(mc / "oos_results.csv")

if not sel.empty:
    keep = [c for c in ["instrument", "best_model", "best_auc", "lower_ci", "signal", "n_events"]
            if c in sel.columns]
    sc = sel[sel["instrument"].isin(INSTS)][keep].set_index("instrument").reindex(INSTS).reset_index()
    table(sc, caption=f"{CLASS.title()} — champion model and signal verdict (selection_table.csv)")

if not locked.empty:
    table(locked, caption="Locked feature variant per instrument (1SE rule, train-only)")

for inst in INSTS:
    display(Markdown(f"**{NAMES[inst]} — `{inst}` cross-validated variants**"))
    show(mc / f"{inst}_cpcv_chart.png", width=760)

show(mc / "oos_summary_chart.png", width=820)

# Out-of-sample AUC for the locked variants only.
if not oos.empty and not locked.empty:
    lk = dict(zip(locked["inst"], locked["locked_variant"]))
    mask = oos.apply(lambda r: lk.get(r["inst"]) == r["variant"], axis=1)
    keep = [c for c in ["inst", "variant", "auc", "auc_ci_lo", "auc_ci_hi", "n_test"] if c in oos.columns]
    table(oos[mask][keep], caption="Out-of-sample AUC — locked variants only")
'''

_CODE_IMPORTANCE = '''
imp = OUT / "importance"
sel = load(OUT / "model_comparison/selection_table.csv")
model_of = dict(zip(sel.get("instrument", []), sel.get("best_model", []))) if not sel.empty else {}

for inst in INSTS:
    d = imp / inst
    display(Markdown(f"### {NAMES[inst]} — `{inst}`"))
    cm = load(d / "champion_meta.csv")
    if not cm.empty:
        table(cm)
    show(d / "dendrogram.png", width=900)
    show(d / "clustered_mda_chart.png", width=820)
    mda = load(d / "clustered_mda_full.csv")
    if not mda.empty:
        table(mda.head(8), caption="Top clusters by mean decrease in accuracy")
    coef_png = d / "global_coef_chart.png"
    if str(model_of.get(inst, "")) == "logistic" and coef_png.exists():
        show(coef_png, width=820)
        table(load(d / "global_coef_summary.csv").head(15), caption="Top logistic coefficients")
    else:
        show(d / "global_shap_chart.png", width=820)
        table(load(d / "global_shap_summary.csv").head(15), caption="Top features by |SHAP|")
'''

_CODE_STRATEGY = '''
se = RESULTS / "strategy_eval"
table(load(se / "eval_summary.csv"), caption="Pooled portfolio — OOS headline (eval_summary.csv)")
show(se / "cumulative_returns.png", width=860)
show(se / "per_asset_cumulative_returns.png", width=860)
show(se / "per_instrument_contribution.png", width=860)

# --- Per-class attribution within the pooled 1/K book (with numeric guards) ---
preds = load(OUT / "metamodel_predictions.csv")
assert not preds.empty, "metamodel_predictions.csv missing"
assert preds["calibrated_proba"].between(0, 1).all(), "calibrated_proba outside [0, 1]"
assert set(INSTS) <= set(preds["instrument"].unique()), "class instruments absent from predictions"

cls = preds[preds["instrument"].isin(INSTS)].copy()
cls["taken"] = cls["calibrated_proba"] > 0.5
attr = (cls.groupby("instrument")
           .apply(lambda g: pd.Series({
               "n_events": len(g),
               "events_taken": int(g["taken"].sum()),
               "mean_calib_proba": g["calibrated_proba"].mean(),
               "label1_share": g["bin"].mean(),
               "hit_rate_taken": g.loc[g["taken"], "bin"].mean() if g["taken"].any() else float("nan"),
           }), include_groups=False)
           .reindex(INSTS).reset_index())
assert (attr["events_taken"] <= attr["n_events"]).all(), "events_taken exceeds n_events"
table(attr, caption=f"{CLASS.title()} — attribution within the pooled book (metamodel_predictions.csv)")
'''


# --------------------------------------------------------------------------- #
# Section builders
# --------------------------------------------------------------------------- #
def build(cls: str) -> dict:
    cfg = CLASSES[cls]
    nar = NARRATIVES[cls]
    cells: list = [
        md_cell(_md_intro(cfg)),
        setup_cell(cfg),
        md_cell(_md_data()),
        code_cell(_CODE_DATA),
        md_cell("## 2. Signal characterisation\n\n" + nar["signal"]),
        code_cell(_CODE_SIGNAL),
        md_cell(_md_labels()),
        code_cell(_CODE_LABELS),
        md_cell(_md_features()),
        code_cell(_CODE_FEATURES),
        md_cell(_md_models() + "\n\n" + nar["models"]),
        code_cell(_CODE_MODELS),
        md_cell(_md_importance() + "\n\n" + nar["importance"]),
        code_cell(_CODE_IMPORTANCE),
        md_cell(_md_strategy()),
        code_cell(_CODE_STRATEGY),
        md_cell("## 8. Discussion\n\n" + nar["discussion"]),
        md_cell(_md_references()),
    ]
    return notebook(cells)


# --------------------------------------------------------------------------- #
# Artifact path sets (single source of truth for the validation test)
# --------------------------------------------------------------------------- #
def required_artifacts(cls: str) -> list[Path]:
    """Paths the narrative depends on; the test asserts every one exists."""
    cfg = CLASSES[cls]
    mc = OUT / "model_comparison" / cls
    paths: list[Path] = [
        DATA / "ohlcv_data.csv",
        RESULTS / "harry/events.csv",
        OUT / "model_comparison/selection_table.csv",
        OUT / "metamodel_predictions.csv",
        RESULTS / "strategy_eval/eval_summary.csv",
        RESULTS / "strategy_eval/cumulative_returns.png",
        RESULTS / "strategy_eval/per_asset_cumulative_returns.png",
        RESULTS / "strategy_eval/per_instrument_contribution.png",
        mc / "cpcv_results.csv",
        mc / "locked_picks.csv",
        mc / "oos_results.csv",
        mc / "oos_summary_chart.png",
        REPORTS / "harry/00-context.md",
        REPORTS / "harry/01-signal-direction.md",
        REPORTS / "harry/02-labels.md",
        REPORTS / "harry/03-features.md",
        REPORTS / "harry/03-6-macro-features.md",
        REPORTS / "missing-data-report.md",
    ]
    for inst in cfg["insts"]:
        paths.append(mc / f"{inst}_cpcv_chart.png")
        d = OUT / "importance" / inst
        paths += [
            d / "champion_meta.csv",
            d / "clustered_mda_full.csv",
            d / "clustered_mda_chart.png",
            d / "dendrogram.png",
            d / "global_shap_chart.png",       # always present, even for logistic champions
            d / "global_shap_summary.csv",
        ]
    return paths


def optional_artifacts(cls: str) -> list[Path]:
    """Tolerant extras — rendered via show()/load(), never asserted."""
    cfg = CLASSES[cls]
    mc = OUT / "model_comparison" / cls
    paths: list[Path] = [
        RESULTS / "harry/signal_direction.csv",
        OUT / "model_comparison/locked_picks.csv",
    ]
    for inst in cfg["insts"]:
        paths.append(mc / f"{inst}_cpcv_chart.png")  # ng1s individual chart may be absent
        d = OUT / "importance" / inst
        paths += [d / "global_coef_chart.png", d / "global_coef_summary.csv"]
    return paths


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate per-asset-class submission notebooks.")
    ap.add_argument("--check", action="store_true", help="validate required artifacts; do not write")
    args = ap.parse_args(argv)

    if args.check:
        bad = {c: [str(p) for p in required_artifacts(c) if not p.exists()] for c in CLASSES}
        missing = {c: v for c, v in bad.items() if v}
        if missing:
            for c, v in missing.items():
                print(f"[{c}] missing {len(v)} required artifacts:")
                for p in v:
                    print(f"    {p}")
            return 1
        print("OK — all required artifacts present for equity, energy, metals.")
        return 0

    DEST.mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        nb = build(cls)
        out = DEST / f"{cls}.ipynb"
        with out.open("w", encoding="utf-8") as fh:
            json.dump(nb, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
        print(f"wrote {out.relative_to(REPO)}  ({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
