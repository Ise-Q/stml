#!/usr/bin/env python3
"""
_gen_importance_notebooks.py
Generates metals/energy/equity importance notebooks from pre-computed artifacts.
Run from the new_work directory:
    python _gen_importance_notebooks.py
"""

import json
import uuid
from pathlib import Path

NB_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Notebook scaffolding helpers
# ---------------------------------------------------------------------------

def _uid():
    return str(uuid.uuid4())

def code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "id": _uid(),
        "metadata": {},
        "source": src,
        "outputs": [],
        "execution_count": None,
    }

def md_cell(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _uid(),
        "metadata": {},
        "source": src,
    }

def notebook(cells: list) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3.13 (stml)",
                "language": "python",
                "name": "stml-py313",
            },
            "language_info": {"name": "python", "version": "3.13.0"},
        },
        "cells": cells,
    }

# ---------------------------------------------------------------------------
# Shared setup cell (identical across all three notebooks)
# ---------------------------------------------------------------------------

SETUP = """\
from pathlib import Path
import math
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from IPython.display import Image, display

SEL_PATH = Path('outputs/model_comparison/selection_table.csv')
BASE     = Path('outputs/importance')

sel = pd.read_csv(SEL_PATH).set_index('instrument')

# ── helpers ────────────────────────────────────────────────────────────────

def load(path):
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    print(f'[missing] {p}')
    return pd.DataFrame()

def show(path, width=1100):
    p = Path(path)
    if p.exists():
        display(Image(str(p), width=width))
    else:
        print(f'[missing] {p}')

def get_meta(inst):
    m = load(BASE / inst / 'champion_meta.csv')
    return m.iloc[0].to_dict() if not m.empty else {}

def cluster_summary(inst):
    mem = load(BASE / inst / 'cluster_membership.csv')
    mda = load(BASE / inst / 'clustered_mda_full.csv')
    if mem.empty or mda.empty:
        return pd.DataFrame()
    grp = (
        mem.groupby('cluster')
        .agg(
            n_members   =('feature', 'count'),
            dominant_pfx=('f_prefix', lambda x: x.value_counts().index[0]),
            purity      =('f_prefix', lambda x: round(x.value_counts().iloc[0] / len(x), 2)),
        )
        .reset_index()
    )
    return (
        grp.merge(mda[['cluster', 'mean_drop', 'std_drop', 'significant']],
                  on='cluster', how='left')
        .sort_values('mean_drop', ascending=False)
    )

def top_n_clusters(inst, n=3):
    \"\"\"Significant clusters first (MDA desc), then non-significant (MDA desc); take n.\"\"\"
    mda = load(BASE / inst / 'clustered_mda_full.csv')
    if mda.empty:
        return []
    sig = mda[mda['significant']].sort_values('mean_drop', ascending=False)
    non = mda[~mda['significant']].sort_values('mean_drop', ascending=False)
    return pd.concat([sig, non])['cluster'].head(n).tolist()

def pick_score_col(wc, family):
    if family == 'logistic' and 'mean_coef_abs' in wc.columns:
        return 'mean_coef_abs', '|coef|'
    for col, lbl in [('mean_shap_mag', '|SHAP|'),
                     ('mean_perm_imp', '|Perm|'),
                     ('mean_coef_abs', '|coef|')]:
        if col in wc.columns:
            return col, lbl
    return wc.columns[2], wc.columns[2]

def print_header(inst):
    \"\"\"Print section header from champion_meta; return meta dict.\"\"\"
    meta    = get_meta(inst)
    mda_df  = load(BASE / inst / 'clustered_mda_full.csv')
    sig     = mda_df[mda_df['significant']] if not mda_df.empty else pd.DataFrame()
    sig_str = (
        ', '.join(f"{r['cluster']} (MDA {r['mean_drop']:.4f} \\u00b1 {r['std_drop']:.4f})"
                  for _, r in sig.iterrows())
        or 'none'
    )
    family  = meta.get('family', 'tree')
    model   = meta.get('model_type', '?').upper()
    auc     = meta.get('auc_mean', float('nan'))
    std     = meta.get('auc_std',  float('nan'))
    lci     = meta.get('lower_ci', float('nan'))
    signal  = bool(meta.get('signal', False))
    grp     = meta.get('group', '?')
    n_folds = int(meta.get('n_folds', 0))
    print(f'--- {inst.upper()} | group={grp} | champion={model} | family={family} ---')
    print(f'CPCV: {n_folds} paths  AUC {auc:.4f} \\u00b1 {std:.4f}  '
          f'{"SIGNAL" if signal else "NO SIGNAL"} (lower CI {lci:.4f})')
    print(f'Significant clusters: {sig_str}')
    return meta

def show_within_cluster(inst, cluster_name, family):
    wc = load(BASE / inst / f'within_cluster_{cluster_name}.csv')
    if wc.empty:
        return
    pc1 = wc['pca_pc1_var_explained'].iloc[0] if 'pca_pc1_var_explained' in wc.columns else float('nan')
    pc2 = wc['pca_pc2_var_explained'].iloc[0] if 'pca_pc2_var_explained' in wc.columns else float('nan')
    tier = ('>= 65% single latent' if pc1 >= 0.65 else
            ('>= 40% dominant'     if pc1 >= 0.40 else '< 40% multi-dim'))
    pc2_s   = f' PC2={pc2:.1%}' if math.isfinite(pc2) else ''
    pca_lbl = f'PC1={pc1:.1%}{pc2_s} ({tier})' if math.isfinite(pc1) else 'no PCA'
    col, col_lbl = pick_score_col(wc, family)
    display(
        wc[['feature', col, 'pc1_loading']].style
        .format({col: '{:.4f}', 'pc1_loading': '{:.3f}'})
        .background_gradient(subset=[col], cmap='Blues')
        .set_caption(f'{inst} | {cluster_name} — {col_lbl} ranking + PC1 loadings  ({pca_lbl})')
    )
    show(BASE / inst / f'within_cluster_{cluster_name}.png', width=950)

def show_step1(inst):
    cs = cluster_summary(inst)
    if not cs.empty:
        rng = max(cs['mean_drop'].abs().max() * 1.1, 0.005)
        display(
            cs.style
            .format({'mean_drop': '{:.4f}', 'std_drop': '{:.4f}', 'purity': '{:.0%}'})
            .background_gradient(subset=['mean_drop'], cmap='RdYlGn', vmin=-rng, vmax=rng)
            .set_caption(f'{inst} — cluster summary')
        )
    show(BASE / inst / 'dendrogram.png', width=1100)

def show_step2(inst):
    show(BASE / inst / 'clustered_mda_chart.png', width=1050)
    cc = load(BASE / inst / 'cluster_crosscheck_table.csv')
    ra = load(BASE / inst / 'rank_agreement.csv')
    if not cc.empty:
        fmt = {'mda_mean': '{:.4f}'}
        for c in ('mdi_sum', 'shap_sum', 'coef_sum', 'perm_sum'):
            if c in cc.columns:
                fmt[c] = '{:.4f}'
        rng = max(cc['mda_mean'].abs().max() * 1.2, 0.005)
        display(
            cc.style.format(fmt)
            .background_gradient(subset=['mda_mean'], cmap='RdYlGn', vmin=-rng, vmax=rng)
            .set_caption(f'{inst} — cluster cross-check')
        )
    if not ra.empty:
        print('\\nKendall \\u03c4 rank agreement:')
        display(ra.style.format({'kendall_tau': '{:.2f}'}))

def show_step3(inst):
    meta   = get_meta(inst)
    family = meta.get('family', 'tree')
    top3   = top_n_clusters(inst, n=3)
    print(f'Step-3 clusters (significant first): {top3}')
    for c in top3:
        show_within_cluster(inst, c, family)

def show_step5(inst):
    meta   = get_meta(inst)
    family = meta.get('family', 'tree')
    if family == 'logistic':
        g = load(BASE / inst / 'global_coef_summary.csv')
        if not g.empty:
            sc  = 'coef_abs' if 'coef_abs' in g.columns else g.columns[1]
            fmt = {sc: '{:.4f}'}
            if 'coef_signed' in g.columns:
                fmt['coef_signed'] = '{:.4f}'
            display(
                g.head(25).style.format(fmt)
                .background_gradient(subset=[sc], cmap='Blues')
                .set_caption(f'{inst} — top 25 features by mean |coef|')
            )
        show(BASE / inst / 'global_coef_chart.png', width=1000)
    else:
        g = load(BASE / inst / 'global_shap_summary.csv')
        if not g.empty:
            sc  = 'shap_magnitude' if 'shap_magnitude' in g.columns else g.columns[1]
            fmt = {sc: '{:.4f}'}
            for c in ('shap_signed', 'mdi'):
                if c in g.columns:
                    fmt[c] = '{:.4f}'
            display(
                g.head(25).style.format(fmt)
                .background_gradient(subset=[sc], cmap='Blues')
                .set_caption(f'{inst} — top 25 features by mean |SHAP|')
            )
        show(BASE / inst / 'global_shap_chart.png', width=1000)

print('Setup complete.  Artifact root:', BASE.resolve())
print('Selection table:', len(sel), 'instruments')
"""


# ---------------------------------------------------------------------------
# Per-notebook intro champion table
# ---------------------------------------------------------------------------

def intro_table_code(insts: list, asset_class: str) -> str:
    return f"""\
# Intro champion table — all values from selection_table + champion_meta
INSTS = {insts!r}

rows = []
for inst in INSTS:
    m = get_meta(inst)
    s = sel.loc[inst]
    rows.append({{
        'Instrument': inst,
        'Group'     : m.get('group',      s['best_group']),
        'Family'    : m.get('family',     s['best_model']),
        'Champion'  : m.get('model_type', s['best_model']).upper(),
        'AUC'       : f"{{m.get('auc_mean', s['best_auc']):.3f}}\\u00b1{{m.get('auc_std', 0):.3f}}",
        'Lower CI'  : f"{{m.get('lower_ci', s['lower_ci']):.4f}}",
        'Signal'    : '\\u2713' if m.get('signal', bool(s['signal'])) else '\\u2717',
    }})

intro_df = pd.DataFrame(rows)
display(
    intro_df.style
    .set_caption('{asset_class} champions — source: selection_table.csv \\u2192 champion_meta.csv')
    .hide(axis='index')
)

# Warn about MLP instruments (artifacts may have been computed with a different model)
for inst in INSTS:
    s = sel.loc[inst]
    m = get_meta(inst)
    if s['best_model'] == 'mlp' and m.get('model_type', '') != 'mlp':
        print(f'\\u26a0\\ufe0f  {{inst}}: selection_table champion=MLP but champion_meta shows '
              f'{{m.get("model_type","?")}}.  Importance artifacts may be from the runner-up model.')
"""


# ---------------------------------------------------------------------------
# Per-instrument cells (called with inst name)
# ---------------------------------------------------------------------------

def inst_cells(inst: str) -> list:
    return [
        md_cell(f"---\n### {inst.upper()} — Steps 1–5"),
        code_cell(f"meta = print_header('{inst}')"),
        md_cell("#### Step 1 · Feature clusters"),
        code_cell(f"show_step1('{inst}')"),
        md_cell("#### Step 2 · Cluster-level importance"),
        code_cell(f"show_step2('{inst}')"),
        md_cell("#### Step 3 · Within-cluster breakdown (top 3, significant first)"),
        code_cell(f"show_step3('{inst}')"),
        md_cell("#### Step 5 · Global feature importance"),
        code_cell(f"show_step5('{inst}')"),
    ]


# ---------------------------------------------------------------------------
# Cross-instrument summary
# ---------------------------------------------------------------------------

def cross_summary_code(insts: list, asset_class: str) -> str:
    return f"""\
# Cross-instrument summary — all values from champion_meta + clustered_mda_full
INSTS = {insts!r}

rows = []
for inst in INSTS:
    m   = get_meta(inst)
    mda = load(BASE / inst / 'clustered_mda_full.csv')
    ra  = load(BASE / inst / 'rank_agreement.csv')
    if not m:
        continue
    sig = mda[mda['significant']] if not mda.empty else pd.DataFrame()
    rows.append({{
        'inst'          : inst,
        'group'         : m.get('group', '?'),
        'champion'      : m.get('model_type', '?').upper(),
        'family'        : m.get('family', '?'),
        'signal'        : '\\u2713' if m.get('signal', False) else '\\u2717',
        'AUC'           : f"{{m.get('auc_mean', 0):.4f}}\\u00b1{{m.get('auc_std', 0):.4f}}",
        'lower_CI'      : f"{{m.get('lower_ci', 0):.4f}}",
        'n_sig_clusters': int(len(sig)),
        'top_cluster'   : mda.iloc[0]['cluster'] if not mda.empty else '?',
        'top_mda'       : f"{{mda.iloc[0]['mean_drop']:.4f}}" if not mda.empty else '?',
        '\\u03c4_primary' : (f"{{ra['kendall_tau'].iloc[0]:.2f}}"
                           if not ra.empty else 'n/a'),
    }})

summary = pd.DataFrame(rows).set_index('inst')
display(
    summary.style
    .set_caption('{asset_class} champions — cross-instrument summary (all values computed)')
)

print('\\nTop-5 MDA per instrument:')
for inst in INSTS:
    mda = load(BASE / inst / 'clustered_mda_full.csv')
    m   = get_meta(inst)
    if mda.empty:
        continue
    tag = '' if m.get('signal', False) else ' (NO SIGNAL)'
    print(f'\\n{{inst.upper()}}{{tag}}')
    print(mda[['cluster','mean_drop','std_drop','significant']].head(5).to_string(index=False))
"""


# ---------------------------------------------------------------------------
# Assertions cell
# ---------------------------------------------------------------------------

def assertions_code(insts: list) -> str:
    return f"""\
# ── CONSISTENCY ASSERTIONS ─────────────────────────────────────────────────
# (a/b/c) champion_meta matches selection_table
# (d)     Step-3 cluster names == names in clustered_mda_full
# (e)     Step-3 cluster IDs exist in cluster_membership.csv
# (f)     AUC in champion_meta self-consistent (trivially true; verified here for completeness)

INSTS = {insts!r}
errors = []

for inst in INSTS:
    s   = sel.loc[inst]
    m   = get_meta(inst)
    mda = load(BASE / inst / 'clustered_mda_full.csv')
    mem = load(BASE / inst / 'cluster_membership.csv')

    if not m:
        errors.append(f'{{inst}}: champion_meta.csv missing or empty')
        continue

    # (a/b/c) model type
    meta_model, sel_model = m.get('model_type',''), s['best_model']
    if meta_model != sel_model:
        errors.append(
            f'{{inst}}: champion_meta model_type={{meta_model!r}} '
            f'!= selection_table best_model={{sel_model!r}}'
        )

    # AUC match
    meta_auc = round(float(m.get('auc_mean', -1)), 4)
    sel_auc  = round(float(s['best_auc']), 4)
    if abs(meta_auc - sel_auc) > 0.0001:
        errors.append(
            f'{{inst}}: champion_meta auc_mean={{meta_auc}} '
            f'!= selection_table best_auc={{sel_auc}}'
        )

    if mda.empty:
        errors.append(f'{{inst}}: clustered_mda_full.csv missing')
        continue

    # (d) Step-3 clusters come from the same MDA table
    step3   = top_n_clusters(inst, n=3)
    mda_set = set(mda['cluster'].tolist())
    for c in step3:
        if c not in mda_set:
            errors.append(f'{{inst}}: Step-3 cluster {{c!r}} not in clustered_mda_full.csv')

    # All significant clusters shown (assuming ≤ 3 significant)
    sig_set = set(mda[mda['significant']]['cluster'].tolist())
    shown   = set(step3)
    if len(sig_set) <= 3 and not sig_set.issubset(shown):
        missing = sig_set - shown
        errors.append(f'{{inst}}: significant clusters {{missing}} not shown in Step-3')

    # (e) cluster IDs exist in cluster_membership.csv
    if not mem.empty:
        mem_clusters = set(mem['cluster'].unique())
        for c in step3:
            if c not in mem_clusters:
                errors.append(f'{{inst}}: Step-3 cluster {{c!r}} not in cluster_membership.csv')

    # (f) AUC fields in champion_meta are internally consistent
    auc_m = m.get('auc_mean', float('nan'))
    lci_m = m.get('lower_ci', float('nan'))
    if not math.isnan(auc_m) and not math.isnan(lci_m) and lci_m > auc_m:
        errors.append(f'{{inst}}: lower_ci={{lci_m}} > auc_mean={{auc_m}} (impossible)')

if errors:
    print('ASSERTION FAILURES:')
    for e in errors:
        print(f'  \\u2717 {{e}}')
    raise AssertionError(f'{{len(errors)}} assertion(s) failed — see above')
else:
    print(f'\\u2713 All assertions passed for {{INSTS}}')
"""


# ---------------------------------------------------------------------------
# Notebook builders
# ---------------------------------------------------------------------------

def build_metals():
    insts = ['gc1s', 'si1s', 'pl1s', 'hg1s']
    cells = [
        md_cell("# Metals champions — feature importance (train-only)\n\n"
                "All values generated from `selection_table.csv` and per-instrument artifacts. "
                "Train data only; the 30% test set is **sealed**.\n\n"
                "> **gc1s / si1s / pl1s** share the pooled `precious` RF champion.  "
                "**hg1s** has its own `hg1s` RF champion."),
        code_cell(SETUP),
        code_cell(intro_table_code(insts, 'Metals')),
    ]
    for inst in insts:
        cells.extend(inst_cells(inst))
    cells.append(md_cell("---\n## Cross-instrument findings — metals"))
    cells.append(code_cell(cross_summary_code(insts, 'Metals')))
    cells.append(md_cell("---\n## Consistency assertions"))
    cells.append(code_cell(assertions_code(insts)))
    return notebook(cells)


def build_energy():
    insts = ['cl1s', 'ho1s', 'rb1s', 'ng1s']
    cells = [
        md_cell("# Energy champions — feature importance (train-only)\n\n"
                "All values generated from `selection_table.csv` and per-instrument artifacts. "
                "Train data only; the 30% test set is **sealed**.\n\n"
                "> **ho1s** and **ng1s** have MLP champions in `selection_table.csv`; "
                "their importance artifacts were computed with a different model family — "
                "see the `⚠️` warning in the intro table and the assertion output."),
        code_cell(SETUP),
        code_cell(intro_table_code(insts, 'Energy')),
    ]
    for inst in insts:
        cells.extend(inst_cells(inst))
    cells.append(md_cell("---\n## Cross-instrument findings — energy"))
    cells.append(code_cell(cross_summary_code(insts, 'Energy')))
    cells.append(md_cell("---\n## Consistency assertions"))
    cells.append(code_cell(assertions_code(insts)))
    return notebook(cells)


def build_equity():
    insts = ['es1s', 'nq1s', 'fesx1s']
    cells = [
        md_cell("# Equity champions — feature importance (train-only)\n\n"
                "All values generated from `selection_table.csv` and per-instrument artifacts. "
                "Train data only; the 30% test set is **sealed**."),
        code_cell(SETUP),
        code_cell(intro_table_code(insts, 'Equity')),
    ]
    for inst in insts:
        cells.extend(inst_cells(inst))
    cells.append(md_cell("---\n## Cross-instrument findings — equity"))
    cells.append(code_cell(cross_summary_code(insts, 'Equity')))
    cells.append(md_cell("---\n## Consistency assertions"))
    cells.append(code_cell(assertions_code(insts)))
    return notebook(cells)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    targets = {
        'metals_importance.ipynb' : build_metals(),
        'energy_importance.ipynb' : build_energy(),
        'equity_importance.ipynb' : build_equity(),
    }
    for fname, nb in targets.items():
        out = NB_DIR / fname
        with open(out, 'w') as f:
            json.dump(nb, f, indent=1)
        n_cells = len(nb['cells'])
        print(f'Wrote {out}  ({n_cells} cells)')
