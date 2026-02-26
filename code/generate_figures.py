#!/usr/bin/env python3
"""
Generate publication-quality figures for the BIB paper.
=======================================================
Figure 1: Pipeline schematic (text-based, manual in Illustrator)
Figure 2: Leakage benchmark — AUC/PR-AUC bar chart with 95% CI
Figure 3: LOSO per-subfamily accuracy distribution
Figure 4: BW-site significance Manhattan/heatmap
Figure 5: Family-internal ICL2/C-term divergence summary
Figure 6: OPN4 BW-site alignment vs opsin family + Gq consensus

Usage: py code/generate_figures.py
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# ===========================================================================
# Load results
# ===========================================================================
bench_df = pd.read_csv(os.path.join(RESULTS_DIR, "benchmark_results.csv"))
bw_df = pd.read_csv(os.path.join(RESULTS_DIR, "bw_site_statistics.csv"))
abl_df = pd.read_csv(os.path.join(RESULTS_DIR, "ablation_results.csv"))
perm_df = pd.read_csv(os.path.join(RESULTS_DIR, "permutation_importance.csv"))

default_bench = bench_df[bench_df['label_mode'] == 'default']

print("Generating figures...")

# ===========================================================================
# Figure 2: Leakage benchmark — Ensemble across splits
# ===========================================================================
def fig2_leakage_benchmark():
    ens = default_bench[default_bench['model'] == 'Ensemble'].copy()
    ens = ens.sort_values('split', key=lambda x: x.map({
        'Random': 0, 'SeqCluster_0.4': 1, 'SeqCluster_0.3': 2, 'Subfamily': 3
    }))

    splits = ens['split'].values
    split_labels = []
    for s in splits:
        if s == 'Random':
            split_labels.append('Random\n(leaky)')
        elif s == 'Subfamily':
            split_labels.append('Subfamily\n(no-leak)')
        elif s == 'SeqCluster_0.3':
            split_labels.append('SeqCluster\n(t=0.3)')
        elif s == 'SeqCluster_0.4':
            split_labels.append('SeqCluster\n(t=0.4)')
        else:
            split_labels.append(s)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    x = np.arange(len(splits))
    w = 0.6

    # AUC-ROC
    ax = axes[0]
    aucs = ens['AUC'].values
    auc_lo = ens['AUC_lo'].values
    auc_hi = ens['AUC_hi'].values
    yerr = np.array([aucs - auc_lo, auc_hi - aucs])
    bars = ax.bar(x, aucs, w, color=colors, edgecolor='black', linewidth=0.5)
    ax.errorbar(x, aucs, yerr=yerr, fmt='none', ecolor='black', capsize=4, linewidth=1.2)
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.7, label='Chance')
    ax.set_xticks(x)
    ax.set_xticklabels(split_labels)
    ax.set_ylabel('AUC-ROC')
    ax.set_title('A) AUC-ROC across split strategies')
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(aucs):
        ax.text(i, v + (auc_hi[i] - v) + 0.03, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax.legend(loc='lower right')

    # PR-AUC
    ax = axes[1]
    praucs = ens['PR_AUC'].values
    prauc_lo = ens['PRAUC_lo'].values
    prauc_hi = ens['PRAUC_hi'].values
    yerr = np.array([praucs - prauc_lo, prauc_hi - praucs])
    bars = ax.bar(x, praucs, w, color=colors, edgecolor='black', linewidth=0.5)
    ax.errorbar(x, praucs, yerr=yerr, fmt='none', ecolor='black', capsize=4, linewidth=1.2)
    baseline_pr = ens.iloc[0]['n_test']  # approximate
    gq_frac = 91 / 230
    ax.axhline(gq_frac, color='grey', linestyle='--', linewidth=0.8, alpha=0.7, label=f'Baseline ({gq_frac:.2f})')
    ax.set_xticks(x)
    ax.set_xticklabels(split_labels)
    ax.set_ylabel('PR-AUC')
    ax.set_title('B) PR-AUC across split strategies')
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(praucs):
        ax.text(i, v + (prauc_hi[i] - v) + 0.03, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax.legend(loc='lower right')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig2_leakage_benchmark.png")
    plt.savefig(path, dpi=600)
    plt.savefig(path.replace('.png', '.pdf'), dpi=600)
    plt.close()
    print(f"  ✓ {path}")


# ===========================================================================
# Figure 3: All models under Subfamily split
# ===========================================================================
def fig3_model_comparison():
    sf = default_bench[default_bench['split'] == 'Subfamily'].copy()
    models = sf['model'].values
    x = np.arange(len(models))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    aucs = sf['AUC'].values
    auc_lo = sf['AUC_lo'].values
    auc_hi = sf['AUC_hi'].values
    yerr_auc = np.array([aucs - auc_lo, auc_hi - aucs])

    praucs = sf['PR_AUC'].values
    prauc_lo = sf['PRAUC_lo'].values
    prauc_hi = sf['PRAUC_hi'].values
    yerr_pr = np.array([praucs - prauc_lo, prauc_hi - praucs])

    bars1 = ax.bar(x - w/2, aucs, w, color='#3498db', edgecolor='black', linewidth=0.5, label='AUC-ROC')
    ax.errorbar(x - w/2, aucs, yerr=yerr_auc, fmt='none', ecolor='black', capsize=3, linewidth=1)

    bars2 = ax.bar(x + w/2, praucs, w, color='#e67e22', edgecolor='black', linewidth=0.5, label='PR-AUC')
    ax.errorbar(x + w/2, praucs, yerr=yerr_pr, fmt='none', ecolor='black', capsize=3, linewidth=1)

    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0, fontsize=13)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Model comparison under subfamily split (no-leak)', fontsize=17)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.35)
    ax.legend(fontsize=13)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig3_model_comparison.png")
    plt.savefig(path, dpi=600)
    plt.savefig(path.replace('.png', '.pdf'), dpi=600)
    plt.close()
    print(f"  ✓ {path}")


# ===========================================================================
# Figure 4: BW-site significance (Manhattan-style)
# ===========================================================================
def fig4_bw_manhattan():
    bw = bw_df.copy()
    bw['neg_log_p'] = -np.log10(bw['p_value'].clip(lower=1e-10))
    bw['neg_log_fdr'] = -np.log10(bw['p_fdr'].clip(lower=1e-10))

    # Color by region
    region_colors = {
        'ICL2': '#e74c3c', 'TM3': '#3498db', 'TM3-DRY': '#3498db',
        'TM5': '#2ecc71', 'TM6': '#9b59b6', 'H8': '#f39c12'
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [2, 1]})

    # Panel A: -log10(p) plot
    ax = axes[0]
    x = np.arange(len(bw))
    colors = [region_colors.get(r, '#95a5a6') for r in bw['region']]
    bars = ax.bar(x, bw['neg_log_p'], color=colors, edgecolor='black', linewidth=0.3)

    # Significance lines
    ax.axhline(-np.log10(0.05), color='orange', linestyle='--', linewidth=1, label='p=0.05')
    fdr_threshold = -np.log10(bw[bw['sig_fdr']]['p_value'].max()) if bw['sig_fdr'].any() else 2
    ax.axhline(-np.log10(0.01), color='red', linestyle='--', linewidth=1, label='p=0.01')

    ax.set_xticks(x)
    ax.set_xticklabels(bw['bw'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(r'$-\log_{10}$(p-value)')
    ax.set_title('A) Statistical significance of amino acid distribution differences (Gq vs non-Gq)')

    # Legend for regions
    handles = [mpatches.Patch(color=c, label=r) for r, c in region_colors.items() if r != 'TM3-DRY']
    handles.append(plt.Line2D([0],[0], color='orange', linestyle='--', label='p=0.05'))
    handles.append(plt.Line2D([0],[0], color='red', linestyle='--', label='p=0.01'))
    ax.legend(handles=handles, loc='upper right', ncol=2, fontsize=8)

    # Mark FDR-significant
    for i, row in bw.iterrows():
        if row['sig_fdr']:
            ax.plot(i, row['neg_log_p'] + 0.15, marker=(5, 1), markersize=10, color='red', markeredgecolor='darkred', markeredgewidth=0.5, linestyle='None', zorder=5)

    # Panel B: Cramér's V
    ax2 = axes[1]
    ax2.bar(x, bw['cramers_v'], color=colors, edgecolor='black', linewidth=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bw['bw'], rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel("Cramér's V")
    ax2.set_title("B) Effect size (Cramér's V)")
    ax2.axhline(0.1, color='grey', linestyle=':', linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_bw_manhattan.png")
    plt.savefig(path, dpi=600)
    plt.savefig(path.replace('.png', '.pdf'), dpi=600)
    plt.close()
    print(f"  ✓ {path}")


# ===========================================================================
# Figure 5: Ablation study
# ===========================================================================
def fig5_ablation():
    abl = abl_df.copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(abl))
    w = 0.35

    ax.bar(x - w/2, abl['AUC'], w, color='#3498db', edgecolor='black', linewidth=0.5, label='AUC-ROC')
    ax.bar(x + w/2, abl['PR_AUC'], w, color='#e67e22', edgecolor='black', linewidth=0.5, label='PR-AUC')

    ax.set_xticks(x)
    ax.set_xticklabels(abl['features'], rotation=0, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Feature ablation under subfamily split (RF)', fontsize=17)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.legend(fontsize=13)

    for i, row in abl.iterrows():
        ax.text(i - w/2, row['AUC'] + 0.02, f"{row['AUC']:.3f}", ha='center', fontsize=11)
        ax.text(i + w/2, row['PR_AUC'] + 0.02, f"{row['PR_AUC']:.3f}", ha='center', fontsize=11)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig5_ablation.png")
    plt.savefig(path, dpi=600)
    plt.savefig(path.replace('.png', '.pdf'), dpi=600)
    plt.close()
    print(f"  ✓ {path}")


# ===========================================================================
# Figure 6: Permutation importance
# ===========================================================================
def fig6_permutation_importance():
    perm = perm_df.head(15).copy()
    perm = perm.iloc[::-1]  # reverse for horizontal bar

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(perm))
    ax.barh(y, perm['importance'], xerr=perm['std'], color='#3498db',
            edgecolor='black', linewidth=0.3, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(perm['feature'], fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.set_xlabel('Permutation Importance (ΔAUC)', fontsize=14)
    ax.set_title('Top 15 features by permutation importance\n(Subfamily split, RF)', fontsize=17)
    ax.axvline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig6_permutation_importance.png")
    plt.savefig(path, dpi=600)
    plt.savefig(path.replace('.png', '.pdf'), dpi=600)
    plt.close()
    print(f"  ✓ {path}")


# ===========================================================================
# Figure S1: Label sensitivity
# ===========================================================================
def figs1_label_sensitivity():
    ens = bench_df[bench_df['model'] == 'Ensemble'].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    splits = ['Random', 'Subfamily', 'SeqCluster_0.3', 'SeqCluster_0.4']
    x = np.arange(len(splits))
    w = 0.35

    for i, lm in enumerate(['default', 'primary_only']):
        subset = ens[ens['label_mode'] == lm]
        aucs = []
        for s in splits:
            row = subset[subset['split'] == s]
            aucs.append(row['AUC'].values[0] if len(row) > 0 else 0)
        offset = -w/2 + i * w
        color = '#3498db' if lm == 'default' else '#e67e22'
        label = 'Primary + Secondary' if lm == 'default' else 'Primary only'
        ax.bar(x + offset, aucs, w * 0.9, color=color, edgecolor='black',
               linewidth=0.5, label=label)

    ax.set_xticks(x)
    split_labels = ['Random\n(leaky)', 'Subfamily\n(no-leak)', 'SeqCluster\n(t=0.3)', 'SeqCluster\n(t=0.4)']
    ax.set_xticklabels(split_labels)
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Label sensitivity: Primary+Secondary vs Primary-only (Ensemble)')
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "figs1_label_sensitivity.png")
    plt.savefig(path)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f"  ✓ {path}")


# ===========================================================================
# Figure S2: Heatmap — all models × all splits
# ===========================================================================
def figs2_heatmap():
    pivot = default_bench.pivot_table(index='model', columns='split', values='AUC')
    col_order = ['Random', 'SeqCluster_0.4', 'SeqCluster_0.3', 'Subfamily']
    pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=0.9)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha='right')
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = 'white' if val < 0.5 or val > 0.8 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontsize=9)

    ax.set_title('AUC-ROC: Models × Split strategies')
    plt.colorbar(im, ax=ax, label='AUC-ROC')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "figs2_heatmap.png")
    plt.savefig(path)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f"  ✓ {path}")


# ===========================================================================
# Run all
# ===========================================================================
fig2_leakage_benchmark()
fig3_model_comparison()
fig4_bw_manhattan()
fig5_ablation()
fig6_permutation_importance()
figs1_label_sensitivity()
figs2_heatmap()

print(f"\n✓ All figures saved to: {FIGURES_DIR}")
print("  Main figures: fig2-fig6 (.png + .pdf)")
print("  Supplementary: figs1-figs2 (.png + .pdf)")
