#!/usr/bin/env python3
"""
Model Interpretability & Visualization Enhancement
====================================================
S5: SHAP analysis (summary, dependence, OPN4 waterfall)
S6: BW-site sequence logo (Gq vs non-Gq)
S7: Gq consensus heatmap (unsupervised validation)

Usage: py code/run_interpretability.py
"""
import os
import sys
import json
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict, Counter, OrderedDict
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import shap
import logomaker

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DATASET_FILE = os.path.join(DATA_DIR, "gpcrdb_coupling_dataset.csv")
CACHE_FILE = os.path.join(DATA_DIR, "gpcrdb_residues_cache.json")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 72)
print("Model Interpretability & Visualization Enhancement")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 72)

# ===========================================================================
# Load data
# ===========================================================================
df = pd.read_csv(DATASET_FILE)
df = df[df['coupling_description'] != 'Unknown'].reset_index(drop=True)

with open(CACHE_FILE, 'r') as f:
    residue_cache = json.load(f)

print(f"Dataset: {len(df)} GPCRs, BW cache: {len(residue_cache)} entries")

# ===========================================================================
# G protein contact sites
# ===========================================================================
GP_CONTACT_SITES = OrderedDict([
    ('34.50', 'ICL2'), ('34.51', 'ICL2'), ('34.52', 'ICL2'),
    ('34.53', 'ICL2'), ('34.54', 'ICL2'), ('34.55', 'ICL2'),
    ('34.56', 'ICL2'), ('34.57', 'ICL2'),
    ('3.49', 'TM3'), ('3.50', 'TM3-DRY'), ('3.51', 'TM3-DRY'),
    ('3.53', 'TM3'), ('3.54', 'TM3'), ('3.55', 'TM3'), ('3.56', 'TM3'),
    ('5.61', 'TM5'), ('5.64', 'TM5'), ('5.65', 'TM5'),
    ('5.67', 'TM5'), ('5.68', 'TM5'), ('5.69', 'TM5'), ('5.71', 'TM5'),
    ('6.32', 'TM6'), ('6.33', 'TM6'), ('6.36', 'TM6'), ('6.37', 'TM6'),
    ('8.47', 'H8'), ('8.48', 'H8'), ('8.49', 'H8'),
])

FDR_SIG_SITES = ['34.50', '34.53', '3.53', '5.65', '5.71']

pos_aa = set('KRH')
neg_aa = set('DE')


def get_bw_residue(residue_data, bw_label):
    for res in residue_data:
        dgn = res.get('display_generic_number', '')
        if dgn:
            bw_part = dgn.split('x')[0] if 'x' in dgn else dgn
            if bw_part == bw_label:
                return res['amino_acid']
        for alt in res.get('alternative_generic_numbers', []):
            if alt.get('scheme') == 'BW' and alt.get('label') == bw_label:
                return res['amino_acid']
    return '-'


def encode_bw_features(bw_row, all_bw_sites):
    feats = {}
    for bw in all_bw_sites:
        aa = bw_row.get(f'bw_{bw}', '-')
        feats[f'{bw}_is_pos'] = 1.0 if aa in pos_aa else 0.0
        feats[f'{bw}_is_neg'] = 1.0 if aa in neg_aa else 0.0
        feats[f'{bw}_is_hydro'] = 1.0 if aa in set('AILMFVPW') else 0.0
        feats[f'{bw}_is_arom'] = 1.0 if aa in set('FYW') else 0.0
        feats[f'{bw}_is_gap'] = 1.0 if aa == '-' else 0.0
    return feats


# ===========================================================================
# Extract BW residues for all receptors
# ===========================================================================
print("\n[Step 1] Extracting BW-site data...")

records = []
for _, row in df.iterrows():
    ename = row['entry_name']
    if ename not in residue_cache:
        continue
    rec = {
        'entry_name': ename,
        'gq_label': int(row['gq_label']),
        'family': row['family'],
        'coupling': row['coupling_description'],
    }
    for bw in GP_CONTACT_SITES:
        rec[f'bw_{bw}'] = get_bw_residue(residue_cache[ename], bw)
    records.append(rec)

bw_df = pd.DataFrame(records)
gq_mask = bw_df['gq_label'] == 1
nongq_mask = bw_df['gq_label'] == 0
print(f"  {len(bw_df)} receptors (Gq={gq_mask.sum()}, non-Gq={nongq_mask.sum()})")

# Build BW feature matrix
bw_sites_list = list(GP_CONTACT_SITES.keys())
bw_feature_names = None
X_bw, y_bw, families_bw, entry_names_bw = [], [], [], []
for _, row in bw_df.iterrows():
    feats = encode_bw_features(row, bw_sites_list)
    if bw_feature_names is None:
        bw_feature_names = sorted(feats.keys())
    X_bw.append([feats[n] for n in bw_feature_names])
    y_bw.append(row['gq_label'])
    families_bw.append(row['family'])
    entry_names_bw.append(row['entry_name'])

X_bw = np.array(X_bw, dtype=np.float32)
y_bw = np.array(y_bw)
print(f"  BW features: {X_bw.shape[1]} dims, {len(X_bw)} samples")

# ===========================================================================
# Subfamily split (inline to avoid benchmark script side-effects)
# ===========================================================================
def get_subfamily(family_slug):
    parts = family_slug.split('_')
    return '_'.join(parts[:3]) if len(parts) >= 3 else family_slug

def split_subfamily(y, families, seed=42):
    np.random.seed(seed)
    sf_map = defaultdict(list)
    for i, f in enumerate(families):
        sf_map[get_subfamily(f)].append(i)
    sfs = list(sf_map.keys())
    np.random.shuffle(sfs)
    target = int(len(y) * 0.2)
    te, tr = [], []
    n_te = 0
    for sf in sfs:
        m = sf_map[sf]
        if n_te < target:
            te.extend(m); n_te += len(m)
        else:
            tr.extend(m)
    return np.array(tr), np.array(te)

tr_idx, te_idx = split_subfamily(y_bw, families_bw)
X_tr, X_te = X_bw[tr_idx], X_bw[te_idx]
y_tr, y_te = y_bw[tr_idx], y_bw[te_idx]

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# Train RF model for SHAP
rf = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=2,
                             class_weight='balanced', random_state=RANDOM_SEED)
rf.fit(X_tr_s, y_tr)

y_prob = rf.predict_proba(X_te_s)[:, 1]
auc = roc_auc_score(y_te, y_prob)
print(f"  RF model trained (Subfamily split AUC={auc:.3f})")

# Also train on full data for SHAP analysis of all samples (incl OPN4)
scaler_full = StandardScaler()
X_bw_s = scaler_full.fit_transform(X_bw)
rf_full = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=2,
                                  class_weight='balanced', random_state=RANDOM_SEED)
rf_full.fit(X_bw_s, y_bw)
print(f"  Full RF model trained (for SHAP on all samples including OPN4)")


# ===================================================================
# S5: SHAP ANALYSIS
# ===================================================================
print(f"\n{'='*72}")
print("S5: SHAP Value Analysis")
print(f"{'='*72}")

# Use TreeExplainer for RF
explainer = shap.TreeExplainer(rf_full)
shap_values = explainer.shap_values(X_bw_s)

# For binary classification, shap_values may be:
# - list of 2 arrays [class0, class1] each (n_samples, n_features)
# - 3D array (n_samples, n_features, 2)
# We want class 1 (Gq)
if isinstance(shap_values, list):
    shap_vals_gq = shap_values[1]
elif shap_values.ndim == 3:
    shap_vals_gq = shap_values[:, :, 1]
else:
    shap_vals_gq = shap_values

print(f"  SHAP values shape (Gq class): {shap_vals_gq.shape}")

# Create readable feature names for plots
def make_readable_name(fname):
    parts = fname.split('_is_')
    if len(parts) == 2:
        bw = parts[0]
        prop = parts[1]
        prop_map = {'pos': 'positive', 'neg': 'negative', 'hydro': 'hydrophobic',
                    'arom': 'aromatic', 'gap': 'gap'}
        return f"BW {bw} ({prop_map.get(prop, prop)})"
    return fname

readable_names = [make_readable_name(n) for n in bw_feature_names]

# --- Fig S5a: SHAP Summary Plot (top 20 features) ---
print("\n  Generating SHAP summary plot...")

# Create a DataFrame for SHAP values with readable names
shap_df = pd.DataFrame(shap_vals_gq, columns=readable_names)
X_display = pd.DataFrame(X_bw_s, columns=readable_names)

# Compute mean |SHAP| and get top 20
mean_abs_shap = np.abs(shap_vals_gq).mean(axis=0)
top20_idx = np.argsort(mean_abs_shap)[::-1][:20]
top20_names = [readable_names[i] for i in top20_idx]

# Manual beeswarm-style plot
fig, ax = plt.subplots(figsize=(10, 8))
for plot_i, feat_i in enumerate(reversed(top20_idx)):
    shap_col = shap_vals_gq[:, feat_i]
    feat_col = X_bw_s[:, feat_i]
    # Normalize feature values for coloring
    fmin, fmax = feat_col.min(), feat_col.max()
    if fmax > fmin:
        feat_norm = (feat_col - fmin) / (fmax - fmin)
    else:
        feat_norm = np.zeros_like(feat_col)
    # Add jitter
    jitter = np.random.normal(0, 0.15, size=len(shap_col))
    ax.scatter(shap_col, plot_i + jitter, c=feat_norm, cmap='coolwarm',
              s=8, alpha=0.6, edgecolors='none', vmin=0, vmax=1)

ax.set_yticks(range(20))
ax.set_yticklabels(list(reversed(top20_names)), fontsize=9)
ax.set_xlabel('SHAP value (impact on Gq prediction)', fontsize=12)
ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
ax.set_title("SHAP Feature Importance for Gq Coupling Prediction\n(RF model, BW-site features)",
             fontsize=13, fontweight='bold', pad=15)

# Colorbar
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(0, 1))
sm.set_array([])
cb = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
cb.set_label('Feature value', fontsize=10)
cb.set_ticks([0, 1])
cb.set_ticklabels(['Low', 'High'])

plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "fig7_shap_summary.png")
plt.savefig(fig_path, dpi=600, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_path}")

# --- Fig S5b: SHAP bar plot (mean |SHAP|) grouped by BW site ---
print("  Generating SHAP importance by BW site...")

# Aggregate SHAP by BW position (sum of 5 physicochemical features per site)
site_shap = {}
for bw in bw_sites_list:
    site_cols = [i for i, n in enumerate(bw_feature_names) if n.startswith(f'{bw}_')]
    site_shap[bw] = np.abs(shap_vals_gq[:, site_cols]).sum(axis=1).mean()

site_shap_df = pd.DataFrame([
    {'BW_site': bw, 'region': GP_CONTACT_SITES[bw], 
     'mean_abs_SHAP': site_shap[bw],
     'is_FDR_sig': bw in FDR_SIG_SITES}
    for bw in bw_sites_list
]).sort_values('mean_abs_SHAP', ascending=True)

fig, ax = plt.subplots(figsize=(8, 10))
colors = ['#e74c3c' if sig else '#3498db' for sig in site_shap_df['is_FDR_sig']]
bars = ax.barh(range(len(site_shap_df)), site_shap_df['mean_abs_SHAP'], color=colors)
ax.set_yticks(range(len(site_shap_df)))
labels = [f"BW {row['BW_site']} ({row['region']})" for _, row in site_shap_df.iterrows()]
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Mean |SHAP value|', fontsize=12)
ax.set_title('SHAP Importance by BW Contact Site\n(Red = FDR-significant)', 
             fontsize=13, fontweight='bold')
ax.axvline(x=0, color='gray', linewidth=0.5)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', label='FDR-significant (p<0.05)'),
                   Patch(facecolor='#3498db', label='Not significant')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "fig8_shap_by_site.png")
plt.savefig(fig_path, dpi=600, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_path}")

# Save site SHAP to CSV
site_shap_df.to_csv(os.path.join(RESULTS_DIR, "shap_by_site.csv"), index=False)

# --- Fig S5c: OPN4 SHAP Waterfall ---
print("  Generating OPN4 SHAP waterfall...")

opn4_idx = None
for i, ename in enumerate(entry_names_bw):
    if ename == 'opn4_human':
        opn4_idx = i
        break

if opn4_idx is not None:
    # Get OPN4 SHAP values
    opn4_shap = shap_vals_gq[opn4_idx]
    opn4_features = X_bw_s[opn4_idx]
    
    # Get top features by |SHAP| for OPN4
    top_k = 15
    top_idx = np.argsort(np.abs(opn4_shap))[::-1][:top_k]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    vals = opn4_shap[top_idx]
    names = [readable_names[i] for i in top_idx]
    feat_vals = [opn4_features[i] for i in top_idx]
    
    # Sort by SHAP value for waterfall
    sort_order = np.argsort(vals)
    vals = vals[sort_order]
    names = [names[i] for i in sort_order]
    feat_vals = [feat_vals[i] for i in sort_order]
    
    colors_wf = ['#e74c3c' if v > 0 else '#3498db' for v in vals]
    bars = ax.barh(range(len(vals)), vals, color=colors_wf, edgecolor='white', linewidth=0.5)
    
    # Add feature value annotations
    for i, (v, fv) in enumerate(zip(vals, feat_vals)):
        label = f"= {fv:.0f}" if fv == int(fv) else f"= {fv:.2f}"
        offset = 0.002 if v >= 0 else -0.002
        ha = 'left' if v >= 0 else 'right'
        ax.text(v + offset, i, label, va='center', ha=ha, fontsize=8, color='gray')
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('SHAP value (impact on Gq prediction)', fontsize=12)
    ax.set_title('OPN4 (Melanopsin) — SHAP Feature Attribution\nWhich BW-site features drive Gq coupling prediction?',
                 fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)
    
    legend_elements = [Patch(facecolor='#e74c3c', label='Pushes toward Gq'),
                       Patch(facecolor='#3498db', label='Pushes away from Gq')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Add prediction probability
    opn4_prob = rf_full.predict_proba(X_bw_s[opn4_idx:opn4_idx+1])[0][1]
    ax.text(0.02, 0.97, f"P(Gq) = {opn4_prob:.3f}", transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig9_opn4_shap_waterfall.png")
    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")
    print(f"  OPN4 P(Gq) = {opn4_prob:.3f}")
else:
    print("  WARNING: OPN4 not found in BW data")

# --- S5d: SHAP dependence plot for top FDR sites ---
print("  Generating SHAP dependence plots for FDR-significant sites...")

fig, axes = plt.subplots(1, len(FDR_SIG_SITES), figsize=(4*len(FDR_SIG_SITES), 4))
if len(FDR_SIG_SITES) == 1:
    axes = [axes]

for ax_i, bw in enumerate(FDR_SIG_SITES):
    # Use the 'is_pos' feature for each site as the main one
    feat_name = f'{bw}_is_pos'
    if feat_name in bw_feature_names:
        feat_idx = bw_feature_names.index(feat_name)
    else:
        # fallback to first feature of this site
        feat_idx = [i for i, n in enumerate(bw_feature_names) if n.startswith(f'{bw}_')][0]
    
    ax = axes[ax_i]
    scatter = ax.scatter(X_bw_s[:, feat_idx], shap_vals_gq[:, feat_idx],
                        c=y_bw, cmap='RdYlBu_r', alpha=0.6, s=20, edgecolors='none')
    ax.set_xlabel(readable_names[feat_idx], fontsize=9)
    ax.set_ylabel('SHAP value', fontsize=9)
    ax.set_title(f'BW {bw}', fontsize=11, fontweight='bold')
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

fig.suptitle('SHAP Dependence at FDR-Significant BW Sites\n(Color: Red=Gq, Blue=non-Gq)',
             fontsize=13, fontweight='bold', y=1.05)
plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "figs4_shap_dependence.png")
plt.savefig(fig_path, dpi=600, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_path}")


# ===================================================================
# S6: BW-SITE SEQUENCE LOGO
# ===================================================================
print(f"\n{'='*72}")
print("S6: BW-Site Sequence Logo (Gq vs non-Gq)")
print(f"{'='*72}")

STANDARD_AAS = list('ACDEFGHIKLMNPQRSTVWY')

fig, axes = plt.subplots(2, len(FDR_SIG_SITES), figsize=(3.5*len(FDR_SIG_SITES), 6),
                         gridspec_kw={'hspace': 0.4, 'wspace': 0.3})

for site_i, bw in enumerate(FDR_SIG_SITES):
    col = f'bw_{bw}'
    
    for group_i, (mask, label, color_scheme) in enumerate([
        (gq_mask, 'Gq-coupled', 'classic'),
        (nongq_mask, 'Non-Gq', 'classic')
    ]):
        aas = [a for a in bw_df.loc[mask, col] if a != '-' and a in set(STANDARD_AAS)]
        
        if not aas:
            axes[group_i, site_i].text(0.5, 0.5, 'No data', ha='center', va='center')
            continue
        
        # Build frequency matrix (single position)
        aa_counts = Counter(aas)
        total = len(aas)
        
        # Create probability matrix
        prob_dict = {aa: aa_counts.get(aa, 0) / total for aa in STANDARD_AAS}
        prob_df = pd.DataFrame([prob_dict])
        
        # Convert to information content
        # IC = log2(20) + sum(p * log2(p)) for each position
        entropy = 0
        for aa in STANDARD_AAS:
            p = prob_dict[aa]
            if p > 0:
                entropy += p * np.log2(p)
        ic = np.log2(20) + entropy  # max IC = log2(20) ≈ 4.32
        
        # Weight probabilities by IC
        ic_dict = {aa: prob_dict[aa] * ic for aa in STANDARD_AAS}
        ic_df = pd.DataFrame([ic_dict])
        
        ax = axes[group_i, site_i]
        
        # Color scheme: positive=blue, negative=red, hydrophobic=black, polar=green, aromatic=purple
        color_dict = {}
        for aa in STANDARD_AAS:
            if aa in pos_aa:
                color_dict[aa] = '#2166ac'  # blue
            elif aa in neg_aa:
                color_dict[aa] = '#b2182b'  # red
            elif aa in set('AILMFVPW'):
                color_dict[aa] = '#333333'  # dark gray
            elif aa in set('FYW'):
                color_dict[aa] = '#7b3294'  # purple
            elif aa in set('STNQ'):
                color_dict[aa] = '#1b7837'  # green
            else:
                color_dict[aa] = '#888888'  # gray
        
        try:
            logo = logomaker.Logo(ic_df, ax=ax, color_scheme=color_dict,
                                 font_name='Arial', show_spines=False)
            ax.set_ylim(0, max(ic + 0.5, 1.0))
        except Exception as e:
            # Fallback: bar chart
            top_aas = sorted(prob_dict.items(), key=lambda x: -x[1])[:8]
            bars_x = range(len(top_aas))
            ax.bar(bars_x, [v for _, v in top_aas], 
                   color=[color_dict.get(a, 'gray') for a, _ in top_aas])
            ax.set_xticks(bars_x)
            ax.set_xticklabels([a for a, _ in top_aas])
        
        if group_i == 0:
            ax.set_title(f'BW {bw}\n({GP_CONTACT_SITES[bw]})', fontsize=11, fontweight='bold')
        
        if site_i == 0:
            ax.set_ylabel(f'{label}\n(n={total})', fontsize=10, fontweight='bold')
        else:
            ax.set_ylabel(f'n={total}', fontsize=9)

fig.suptitle('Amino Acid Preferences at FDR-Significant BW Positions\n'
             'Colors: Blue=positive, Red=negative, Black=hydrophobic, Green=polar',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "fig10_bw_logos.png")
plt.savefig(fig_path, dpi=600, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_path}")

# --- Alternative: Stacked bar chart (more robust) ---
print("  Generating stacked AA distribution chart...")

fig, axes = plt.subplots(2, len(FDR_SIG_SITES), figsize=(3.5*len(FDR_SIG_SITES), 7),
                         gridspec_kw={'hspace': 0.5, 'wspace': 0.35})

for site_i, bw in enumerate(FDR_SIG_SITES):
    col = f'bw_{bw}'
    
    for group_i, (mask, label) in enumerate([(gq_mask, 'Gq-coupled'), (nongq_mask, 'Non-Gq')]):
        aas = [a for a in bw_df.loc[mask, col] if a != '-' and a in set(STANDARD_AAS)]
        if not aas:
            continue
        
        aa_counts = Counter(aas)
        total = len(aas)
        
        # Get top 8 AAs
        top_aas = sorted(aa_counts.items(), key=lambda x: -x[1])[:8]
        other_count = total - sum(c for _, c in top_aas)
        
        ax = axes[group_i, site_i]
        
        aa_names = [a for a, _ in top_aas]
        aa_fracs = [c / total for _, c in top_aas]
        
        color_dict = {}
        for aa in STANDARD_AAS:
            if aa in pos_aa:
                color_dict[aa] = '#2166ac'
            elif aa in neg_aa:
                color_dict[aa] = '#b2182b'
            elif aa in set('AILMFVPW'):
                color_dict[aa] = '#4d4d4d'
            elif aa in set('STNQ'):
                color_dict[aa] = '#1b7837'
            else:
                color_dict[aa] = '#999999'
        
        bar_colors = [color_dict.get(a, 'gray') for a in aa_names]
        bars = ax.bar(range(len(aa_names)), aa_fracs, color=bar_colors, edgecolor='white', linewidth=0.5)
        
        # Add percentage labels
        for bi, (frac, name) in enumerate(zip(aa_fracs, aa_names)):
            if frac > 0.03:
                ax.text(bi, frac + 0.01, f'{frac:.0%}', ha='center', va='bottom', fontsize=7)
        
        ax.set_xticks(range(len(aa_names)))
        ax.set_xticklabels(aa_names, fontsize=10, fontweight='bold')
        ax.set_ylim(0, max(aa_fracs) * 1.25)
        ax.set_ylabel('Frequency', fontsize=9)
        
        if group_i == 0:
            ax.set_title(f'BW {bw} ({GP_CONTACT_SITES[bw]})', fontsize=11, fontweight='bold')
        
        if site_i == 0:
            ax.set_ylabel(f'{label} (n={total})\nFrequency', fontsize=10, fontweight='bold')

fig.suptitle('Amino Acid Distributions at FDR-Significant BW Positions\n'
             'Gq-coupled (top) vs Non-Gq (bottom)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "fig10b_bw_aa_distributions.png")
plt.savefig(fig_path, dpi=600, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_path}")


# ===================================================================
# S7: Gq CONSENSUS HEATMAP
# ===================================================================
print(f"\n{'='*72}")
print("S7: Gq Consensus Matching Heatmap")
print(f"{'='*72}")

# Compute Gq consensus amino acid at each BW site
gq_consensus = {}
for bw in bw_sites_list:
    gq_aas = [a for a in bw_df.loc[gq_mask, f'bw_{bw}'] if a != '-']
    if gq_aas:
        gq_consensus[bw] = Counter(gq_aas).most_common(1)[0][0]
    else:
        gq_consensus[bw] = '-'

# For each receptor, compute match score at each BW site
# 1 = matches Gq consensus, 0 = doesn't match, NaN = gap
match_matrix = []
receptor_labels = []
receptor_gq = []

for _, row in bw_df.iterrows():
    scores = []
    for bw in bw_sites_list:
        aa = row[f'bw_{bw}']
        if aa == '-':
            scores.append(np.nan)
        elif aa == gq_consensus[bw]:
            scores.append(1.0)
        else:
            scores.append(0.0)
    match_matrix.append(scores)
    
    ename = row['entry_name']
    short_name = ename.replace('_human', '').upper()
    receptor_labels.append(short_name)
    receptor_gq.append(row['gq_label'])

match_matrix = np.array(match_matrix, dtype=np.float64)
receptor_gq = np.array(receptor_gq)

print(f"  Match matrix: {match_matrix.shape}")
print(f"  Gq consensus: {dict(list(gq_consensus.items())[:5])} ...")

# Hierarchical clustering
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist

# Replace NaN with 0.5 for clustering
match_for_cluster = np.nan_to_num(match_matrix, nan=0.5)

# Cluster rows (receptors)
row_dist = pdist(match_for_cluster, metric='hamming')
row_linkage = linkage(row_dist, method='ward')
row_order = leaves_list(row_linkage)

# Cluster columns (BW sites)
col_dist = pdist(match_for_cluster.T, metric='hamming')
col_linkage = linkage(col_dist, method='ward')
col_order = leaves_list(col_linkage)

# Reorder
match_ordered = match_matrix[row_order][:, col_order]
labels_ordered = [receptor_labels[i] for i in row_order]
gq_ordered = receptor_gq[row_order]
bw_ordered = [bw_sites_list[i] for i in col_order]

# Create heatmap
fig = plt.figure(figsize=(14, 16))
gs = gridspec.GridSpec(1, 3, width_ratios=[0.3, 10, 0.5], wspace=0.02)

# Gq label sidebar
ax_side = fig.add_subplot(gs[0])
gq_colors = np.array([[1, 0.2, 0.2] if g == 1 else [0.2, 0.4, 0.8] for g in gq_ordered])
ax_side.imshow(gq_colors.reshape(-1, 1, 3), aspect='auto', interpolation='none')
ax_side.set_xticks([0])
ax_side.set_xticklabels(['Gq'], fontsize=9, rotation=90)
ax_side.set_yticks([])
ax_side.set_ylabel('')

# Main heatmap
ax_heat = fig.add_subplot(gs[1])

# Custom colormap: white(gap/NaN) -> blue(no match) -> red(match)
cmap = LinearSegmentedColormap.from_list('gq_match', ['#3498db', '#f7f7f7', '#e74c3c'])
cmap.set_bad(color='#f0f0f0')

masked_data = np.ma.array(match_ordered, mask=np.isnan(match_ordered))
im = ax_heat.imshow(masked_data, aspect='auto', cmap=cmap, interpolation='none',
                     vmin=0, vmax=1)

# Labels
ax_heat.set_xticks(range(len(bw_ordered)))
ax_heat.set_xticklabels([f'{bw}\n({GP_CONTACT_SITES[bw]})' for bw in bw_ordered],
                         fontsize=7, rotation=90)

# Only show every Nth receptor label to avoid crowding
n_labels = len(labels_ordered)
if n_labels > 50:
    step = max(1, n_labels // 30)
    tick_positions = list(range(0, n_labels, step))
    ax_heat.set_yticks(tick_positions)
    ax_heat.set_yticklabels([labels_ordered[i] for i in tick_positions], fontsize=5)
else:
    ax_heat.set_yticks(range(n_labels))
    ax_heat.set_yticklabels(labels_ordered, fontsize=5)

# Highlight OPN4
opn4_pos = None
for i, label in enumerate(labels_ordered):
    if label == 'OPN4':
        opn4_pos = i
        break

if opn4_pos is not None:
    ax_heat.axhline(y=opn4_pos - 0.5, color='lime', linewidth=1.5, linestyle='-')
    ax_heat.axhline(y=opn4_pos + 0.5, color='lime', linewidth=1.5, linestyle='-')
    ax_heat.text(-1, opn4_pos, 'OPN4 ►', ha='right', va='center', fontsize=8,
                 fontweight='bold', color='green')

# Highlight FDR significant columns
for ci, bw in enumerate(bw_ordered):
    if bw in FDR_SIG_SITES:
        ax_heat.axvline(x=ci - 0.5, color='gold', linewidth=1, linestyle='-', alpha=0.7)
        ax_heat.axvline(x=ci + 0.5, color='gold', linewidth=1, linestyle='-', alpha=0.7)

ax_heat.set_title('Gq Consensus Matching at 29 G Protein Contact Sites\n'
                   '(Hierarchical clustering; Red sidebar = Gq-coupled)',
                   fontsize=13, fontweight='bold', pad=15)

# Colorbar
ax_cb = fig.add_subplot(gs[2])
cb = plt.colorbar(im, cax=ax_cb)
cb.set_ticks([0, 0.5, 1])
cb.set_ticklabels(['Mismatch', '', 'Match'])
cb.set_label('Gq consensus match', fontsize=10)

plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "fig11_gq_consensus_heatmap.png")
plt.savefig(fig_path, dpi=600, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_path}")

# --- Compute cluster purity ---
from scipy.cluster.hierarchy import fcluster as fcluster_fn

# Cut tree into 2 clusters
cluster_labels = fcluster_fn(row_linkage, t=2, criterion='maxclust')
cluster_gq_fracs = {}
for c in [1, 2]:
    c_mask = cluster_labels == c
    gq_frac = receptor_gq[c_mask].mean()
    cluster_gq_fracs[c] = (c_mask.sum(), gq_frac)
    print(f"  Cluster {c}: {c_mask.sum()} receptors, {gq_frac:.1%} Gq")

# --- Focused heatmap: FDR-significant sites only ---
print("  Generating focused heatmap (FDR-significant sites only)...")

fdr_col_idx = [bw_sites_list.index(bw) for bw in FDR_SIG_SITES]
match_fdr = match_matrix[:, fdr_col_idx]
match_fdr_for_cluster = np.nan_to_num(match_fdr, nan=0.5)

row_dist_fdr = pdist(match_fdr_for_cluster, metric='hamming')
row_linkage_fdr = linkage(row_dist_fdr, method='ward')
row_order_fdr = leaves_list(row_linkage_fdr)

match_fdr_ordered = match_fdr[row_order_fdr]
labels_fdr_ordered = [receptor_labels[i] for i in row_order_fdr]
gq_fdr_ordered = receptor_gq[row_order_fdr]

fig, (ax_side2, ax_heat2) = plt.subplots(1, 2, figsize=(8, 14),
                                          gridspec_kw={'width_ratios': [0.3, 5], 'wspace': 0.02})

gq_colors2 = np.array([[1, 0.2, 0.2] if g == 1 else [0.2, 0.4, 0.8] for g in gq_fdr_ordered])
ax_side2.imshow(gq_colors2.reshape(-1, 1, 3), aspect='auto', interpolation='none')
ax_side2.set_xticks([0])
ax_side2.set_xticklabels(['Gq'], fontsize=9, rotation=90)
ax_side2.set_yticks([])

masked_fdr = np.ma.array(match_fdr_ordered, mask=np.isnan(match_fdr_ordered))
im2 = ax_heat2.imshow(masked_fdr, aspect='auto', cmap=cmap, interpolation='none',
                       vmin=0, vmax=1)

ax_heat2.set_xticks(range(len(FDR_SIG_SITES)))
ax_heat2.set_xticklabels([f'BW {bw}\n({GP_CONTACT_SITES[bw]})' for bw in FDR_SIG_SITES],
                          fontsize=10, fontweight='bold')

n_labels2 = len(labels_fdr_ordered)
if n_labels2 > 50:
    step2 = max(1, n_labels2 // 40)
    tick_pos2 = list(range(0, n_labels2, step2))
    ax_heat2.set_yticks(tick_pos2)
    ax_heat2.set_yticklabels([labels_fdr_ordered[i] for i in tick_pos2], fontsize=5)
else:
    ax_heat2.set_yticks(range(n_labels2))
    ax_heat2.set_yticklabels(labels_fdr_ordered, fontsize=5)

# Highlight OPN4
for i, label in enumerate(labels_fdr_ordered):
    if label == 'OPN4':
        ax_heat2.axhline(y=i - 0.5, color='lime', linewidth=2)
        ax_heat2.axhline(y=i + 0.5, color='lime', linewidth=2)
        ax_heat2.text(-0.8, i, 'OPN4 ►', ha='right', va='center', fontsize=8,
                      fontweight='bold', color='green')
        break

ax_heat2.set_title('Gq Consensus Match at 5 FDR-Significant BW Sites\n'
                    '(Red sidebar = Gq-coupled; Green = OPN4)',
                    fontsize=12, fontweight='bold', pad=10)

plt.colorbar(im2, ax=ax_heat2, shrink=0.3, label='Gq consensus match',
             ticks=[0, 1], pad=0.02)

plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "fig11b_fdr_consensus_heatmap.png")
plt.savefig(fig_path, dpi=600, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_path}")

# Cluster purity for FDR-only
cluster_labels_fdr = fcluster_fn(row_linkage_fdr, t=2, criterion='maxclust')
for c in [1, 2]:
    c_mask = cluster_labels_fdr == c
    gq_frac = receptor_gq[c_mask].mean()
    print(f"  FDR-only Cluster {c}: {c_mask.sum()} receptors, {gq_frac:.1%} Gq")


# ===================================================================
# SAVE SHAP RESULTS
# ===================================================================
print(f"\n{'='*72}")
print("Saving all SHAP results...")
print(f"{'='*72}")

# Per-feature mean |SHAP| for paper table
feature_importance = []
for i, fname in enumerate(bw_feature_names):
    feature_importance.append({
        'feature': fname,
        'readable_name': readable_names[i],
        'mean_abs_shap': np.abs(shap_vals_gq[:, i]).mean(),
        'mean_shap_gq': shap_vals_gq[y_bw == 1, i].mean(),
        'mean_shap_nongq': shap_vals_gq[y_bw == 0, i].mean(),
    })

fimp_df = pd.DataFrame(feature_importance).sort_values('mean_abs_shap', ascending=False)
fimp_csv = os.path.join(RESULTS_DIR, "shap_feature_importance.csv")
fimp_df.to_csv(fimp_csv, index=False)
print(f"  Saved: {fimp_csv}")

# OPN4 SHAP values
if opn4_idx is not None:
    opn4_records = []
    for i, fname in enumerate(bw_feature_names):
        opn4_records.append({
            'feature': fname,
            'readable_name': readable_names[i],
            'feature_value': X_bw[opn4_idx, i],
            'shap_value': shap_vals_gq[opn4_idx, i],
        })
    opn4_df = pd.DataFrame(opn4_records).sort_values('shap_value', ascending=False)
    opn4_csv = os.path.join(RESULTS_DIR, "shap_opn4.csv")
    opn4_df.to_csv(opn4_csv, index=False)
    print(f"  Saved: {opn4_csv}")


# ===================================================================
# SUMMARY
# ===================================================================
print(f"\n{'='*72}")
print("SUMMARY")
print(f"{'='*72}")
print(f"""
S5 - SHAP Analysis:
  - SHAP summary plot: figures/fig7_shap_summary.png
  - SHAP by BW site: figures/fig8_shap_by_site.png
  - OPN4 waterfall: figures/fig9_opn4_shap_waterfall.png
  - Dependence plots: figures/figs4_shap_dependence.png
  - CSV: results/shap_feature_importance.csv, results/shap_opn4.csv

S6 - Sequence Logo:
  - Logo plot: figures/fig10_bw_logos.png
  - AA distribution: figures/fig10b_bw_aa_distributions.png

S7 - Consensus Heatmap:
  - Full heatmap: figures/fig11_gq_consensus_heatmap.png
  - FDR-only heatmap: figures/fig11b_fdr_consensus_heatmap.png
  - Cluster analysis: results/shap_by_site.csv
""")

# Print top 10 SHAP features
print("Top 10 features by mean |SHAP|:")
for _, row in fimp_df.head(10).iterrows():
    print(f"  {row['readable_name']:<35s} |SHAP|={row['mean_abs_shap']:.4f}")

if opn4_idx is not None:
    print(f"\nOPN4 Gq prediction probability: {opn4_prob:.3f}")
    print("OPN4 top 5 SHAP contributions:")
    opn4_sorted = opn4_df.sort_values('shap_value', ascending=False)
    for _, row in opn4_sorted.head(5).iterrows():
        direction = "→ Gq" if row['shap_value'] > 0 else "→ non-Gq"
        print(f"  {row['readable_name']:<35s} SHAP={row['shap_value']:+.4f} {direction}")

print(f"\n{'='*72}")
print("All done!")
print(f"{'='*72}")
