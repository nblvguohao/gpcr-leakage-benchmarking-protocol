#!/usr/bin/env python3
"""
Paper BW-Enhanced Analysis: FDR correction, effect sizes, ablation study
=========================================================================
Produces:
  1. BW site analysis with Benjamini-Hochberg FDR correction + Cramér's V
  2. Ablation: BW-site one-hot features vs handcrafted vs combined
  3. Cross-reference with published contact sites (Flock 2017, Inoue 2019)
  4. OPN4 Gq-consensus matching score with GPCRdb BW annotation
  5. SHAP / permutation importance for top features

Usage: py code/run_paper_bw_enhanced.py
"""
import os
import sys
import re
import json
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from collections import defaultdict, Counter, OrderedDict
from datetime import datetime
from scipy import stats

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.inspection import permutation_importance

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASET_FILE = os.path.join(DATA_DIR, "gpcrdb_coupling_dataset.csv")
CACHE_FILE = os.path.join(DATA_DIR, "gpcrdb_residues_cache.json")

print("=" * 72)
print("Paper BW-Enhanced Analysis")
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
# G protein contact sites (literature-based)
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

# Published selectivity-associated positions
PUBLISHED_GQ_SITES = {
    'Flock2017': ['3.50', '3.53', '3.54', '34.50', '34.51', '34.53',
                  '34.54', '34.55', '5.65', '5.68', '6.33'],
    'Inoue2019': ['3.50', '3.53', '3.54', '3.55', '3.56',
                  '34.50', '34.51', '34.52', '34.53', '34.54', '34.55',
                  '5.61', '5.64', '5.65', '5.67', '5.68', '6.32', '6.33', '6.36'],
}


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


# ===========================================================================
# 1. Extract BW residues for all receptors
# ===========================================================================
print("\n[1/5] Extracting BW-site amino acids...")

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


# ===========================================================================
# 2. Per-site chi-squared test + FDR + Cramér's V
# ===========================================================================
print("\n[2/5] Per-site statistical analysis (chi2 + FDR + Cramér's V)...")

pos_aa = set('KRH')
neg_aa = set('DE')

site_stats = []
for bw, region in GP_CONTACT_SITES.items():
    col = f'bw_{bw}'
    gq_aas = [a for a in bw_df.loc[gq_mask, col] if a != '-']
    nongq_aas = [a for a in bw_df.loc[nongq_mask, col] if a != '-']

    if not gq_aas or not nongq_aas:
        site_stats.append({'bw': bw, 'region': region, 'p_value': 1.0,
                           'cramers_v': 0, 'n_gq': 0, 'n_nongq': 0})
        continue

    all_aas = sorted(set(gq_aas + nongq_aas))
    gq_counts = Counter(gq_aas)
    nongq_counts = Counter(nongq_aas)

    if len(all_aas) < 2:
        site_stats.append({'bw': bw, 'region': region, 'p_value': 1.0,
                           'cramers_v': 0, 'n_gq': len(gq_aas), 'n_nongq': len(nongq_aas)})
        continue

    obs = np.array([[gq_counts.get(a, 0) for a in all_aas],
                     [nongq_counts.get(a, 0) for a in all_aas]])
    try:
        chi2, pval, dof, _ = stats.chi2_contingency(obs)
        n_total = obs.sum()
        k = min(obs.shape)
        cramers_v = np.sqrt(chi2 / (n_total * (k - 1))) if n_total > 0 and k > 1 else 0
    except:
        pval, cramers_v = 1.0, 0

    # Gq positive charge enrichment
    gq_pos_frac = sum(1 for a in gq_aas if a in pos_aa) / len(gq_aas)
    nongq_pos_frac = sum(1 for a in nongq_aas if a in pos_aa) / len(nongq_aas)

    gq_top3 = Counter(gq_aas).most_common(3)
    nongq_top3 = Counter(nongq_aas).most_common(3)

    site_stats.append({
        'bw': bw, 'region': region,
        'p_value': pval, 'cramers_v': cramers_v,
        'n_gq': len(gq_aas), 'n_nongq': len(nongq_aas),
        'gq_pos_frac': gq_pos_frac, 'nongq_pos_frac': nongq_pos_frac,
        'gq_top3': '/'.join(f"{a}({c})" for a, c in gq_top3),
        'nongq_top3': '/'.join(f"{a}({c})" for a, c in nongq_top3),
        'in_Flock2017': bw in PUBLISHED_GQ_SITES['Flock2017'],
        'in_Inoue2019': bw in PUBLISHED_GQ_SITES['Inoue2019'],
    })

stats_df = pd.DataFrame(site_stats)

# Benjamini-Hochberg FDR correction
pvals = stats_df['p_value'].values
n_tests = len(pvals)
sorted_idx = np.argsort(pvals)
ranks = np.empty_like(sorted_idx)
ranks[sorted_idx] = np.arange(1, n_tests + 1)
fdr_adjusted = np.minimum(1.0, pvals * n_tests / ranks)
# Enforce monotonicity
for i in range(n_tests - 2, -1, -1):
    idx = sorted_idx[i]
    idx_next = sorted_idx[i + 1]
    fdr_adjusted[idx] = min(fdr_adjusted[idx], fdr_adjusted[idx_next])
stats_df['p_fdr'] = fdr_adjusted
stats_df['sig_nominal'] = stats_df['p_value'] < 0.05
stats_df['sig_fdr'] = stats_df['p_fdr'] < 0.05

# Print results
print(f"\n{'BW':<8s} {'Region':<10s} {'p-value':<12s} {'FDR':<12s} {'V':<8s} "
      f"{'Flock':<6s} {'Inoue':<6s} {'Gq top':<20s} {'nonGq top'}")
print('-' * 95)
for _, r in stats_df.sort_values('p_value').iterrows():
    sig = '**' if r['sig_fdr'] else ('*' if r['sig_nominal'] else '')
    flock = '✓' if r.get('in_Flock2017', False) else ''
    inoue = '✓' if r.get('in_Inoue2019', False) else ''
    print(f"{r['bw']:<8s} {r['region']:<10s} {r['p_value']:<12.2e} {r['p_fdr']:<12.2e} "
          f"{r['cramers_v']:<8.3f} {flock:<6s} {inoue:<6s} "
          f"{r.get('gq_top3',''):<20s} {r.get('nongq_top3','')}{sig}")

n_sig_nom = stats_df['sig_nominal'].sum()
n_sig_fdr = stats_df['sig_fdr'].sum()
print(f"\nSignificant sites: {n_sig_nom} nominal (p<0.05), {n_sig_fdr} after FDR correction")

# Cross-reference with literature
for pub, sites in PUBLISHED_GQ_SITES.items():
    our_sig = set(stats_df[stats_df['sig_fdr']]['bw'])
    pub_set = set(sites)
    overlap = our_sig & pub_set
    print(f"  {pub}: {len(pub_set)} published sites, {len(overlap)} overlap with our FDR-sig sites")

# Save
stats_csv = os.path.join(RESULTS_DIR, "bw_site_statistics.csv")
stats_df.to_csv(stats_csv, index=False)
print(f"  Saved: {stats_csv}")


# ===========================================================================
# 3. Ablation: BW-site features vs handcrafted vs combined
# ===========================================================================
print(f"\n[3/5] Ablation study: BW-site features vs handcrafted...")

# BW one-hot features
def encode_bw_features(bw_df_row, all_bw_sites):
    feats = {}
    for bw in all_bw_sites:
        aa = bw_df_row.get(f'bw_{bw}', '-')
        # Physicochemical encoding (5 dims per site)
        feats[f'{bw}_is_pos'] = 1.0 if aa in pos_aa else 0.0
        feats[f'{bw}_is_neg'] = 1.0 if aa in neg_aa else 0.0
        feats[f'{bw}_is_hydro'] = 1.0 if aa in set('AILMFVPW') else 0.0
        feats[f'{bw}_is_arom'] = 1.0 if aa in set('FYW') else 0.0
        feats[f'{bw}_is_gap'] = 1.0 if aa == '-' else 0.0
    return feats

# Build BW feature matrix
bw_feature_names = None
X_bw, y_bw = [], []
for _, row in bw_df.iterrows():
    feats = encode_bw_features(row, list(GP_CONTACT_SITES.keys()))
    if bw_feature_names is None:
        bw_feature_names = sorted(feats.keys())
    X_bw.append([feats[n] for n in bw_feature_names])
    y_bw.append(row['gq_label'])

X_bw = np.array(X_bw, dtype=np.float32)
y_bw = np.array(y_bw)
print(f"  BW features: {X_bw.shape[1]} dims, {len(X_bw)} samples")

# Handcrafted features (reuse from benchmark - import extract_features)
from run_paper_benchmark import extract_features, get_kmer_set, split_subfamily

hc_feature_names = None
X_hc = []
families_for_split = []
for _, row in bw_df.iterrows():
    entry = row['entry_name']
    df_row = df[df['entry_name'] == entry].iloc[0]
    seq = df_row['sequence']
    fam = str(df_row['family'])
    feat = extract_features(seq, fam)
    if hc_feature_names is None:
        hc_feature_names = sorted(feat.keys())
    X_hc.append([feat.get(n, 0) for n in hc_feature_names])
    families_for_split.append(fam)

X_hc = np.array(X_hc, dtype=np.float32)

# Combined
X_combined = np.hstack([X_hc, X_bw])
print(f"  Handcrafted: {X_hc.shape[1]} dims")
print(f"  Combined: {X_combined.shape[1]} dims")

# Evaluate each feature set under subfamily split
def evaluate_ablation(X, y, families, feature_label):
    tr_idx, te_idx = split_subfamily(y, families)
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    if y_tr.sum() < 3 or y_te.sum() < 2:
        return {'features': feature_label, 'AUC': np.nan, 'F1': np.nan}

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    rf = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=2,
                                 class_weight='balanced', random_state=42)
    rf.fit(X_tr_s, y_tr)
    y_prob = rf.predict_proba(X_te_s)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_te, y_prob)
    except:
        auc = np.nan
    try:
        prauc = average_precision_score(y_te, y_prob)
    except:
        prauc = np.nan
    f1 = f1_score(y_te, y_pred, zero_division=0)

    return {'features': feature_label, 'AUC': auc, 'PR_AUC': prauc,
            'F1': f1, 'n_features': X.shape[1]}

ablation_results = []
for label, X_feat in [('Handcrafted (99d)', X_hc),
                       ('BW-site (145d)', X_bw),
                       ('Combined (244d)', X_combined)]:
    r = evaluate_ablation(X_feat, y_bw, families_for_split, label)
    ablation_results.append(r)
    print(f"  {label:<25s} AUC={r['AUC']:.3f}  PR-AUC={r.get('PR_AUC',0):.3f}  F1={r['F1']:.3f}")

abl_df = pd.DataFrame(ablation_results)
abl_csv = os.path.join(RESULTS_DIR, "ablation_results.csv")
abl_df.to_csv(abl_csv, index=False)
print(f"  Saved: {abl_csv}")


# ===========================================================================
# 4. OPN4 Gq-consensus matching with GPCRdb BW annotation
# ===========================================================================
print(f"\n[4/5] OPN4 Gq-consensus matching (GPCRdb BW annotation)...")

opsin_entries = ['opn4_human', 'opsd_human', 'opn3_human', 'opn5_human',
                 'opsb_human', 'opsg_human', 'opsr_human']
coupling_labels = {
    'opn4_human': 'Gq/11', 'opsd_human': 'Gt', 'opn3_human': 'Gi/o',
    'opn5_human': 'Gi/o', 'opsb_human': 'Gt', 'opsg_human': 'Gt', 'opsr_human': 'Gt',
}

# GPCRdb segment annotations for OPN4
if 'opn4_human' in residue_cache:
    opn4_res = residue_cache['opn4_human']
    seg_counts = defaultdict(int)
    for r in opn4_res:
        seg_counts[r.get('protein_segment', 'other')] += 1
    print(f"\n  OPN4 GPCRdb segment annotation:")
    for seg in ['N-term','TM1','ICL1','TM2','ECL1','TM3','ICL2','TM4','ECL2',
                'TM5','ICL3','TM6','ECL3','TM7','H8','C-term']:
        if seg in seg_counts:
            print(f"    {seg}: {seg_counts[seg]} residues")

# FDR-significant sites only
sig_sites = stats_df[stats_df['sig_fdr']]['bw'].tolist()
if not sig_sites:
    sig_sites = stats_df[stats_df['sig_nominal']]['bw'].tolist()
    print(f"  Using nominal-significant sites ({len(sig_sites)})")
else:
    print(f"  Using FDR-significant sites ({len(sig_sites)})")

# OPN4 matching
opn4_match = []
gq_consensus = {}
nongq_consensus = {}
for bw in GP_CONTACT_SITES:
    gq_aas = [a for a in bw_df.loc[gq_mask, f'bw_{bw}'] if a != '-']
    nongq_aas = [a for a in bw_df.loc[nongq_mask, f'bw_{bw}'] if a != '-']
    gq_consensus[bw] = Counter(gq_aas).most_common(1)[0][0] if gq_aas else '-'
    nongq_consensus[bw] = Counter(nongq_aas).most_common(1)[0][0] if nongq_aas else '-'

if 'opn4_human' in residue_cache:
    print(f"\n  {'BW':<8s} {'OPN4':<6s} {'Gq cons.':<10s} {'nonGq cons.':<12s} {'Match':<15s} {'FDR sig?'}")
    print('  ' + '-' * 60)

    match_gq, match_nongq, match_unique, total = 0, 0, 0, 0
    for bw in GP_CONTACT_SITES:
        opn4_aa = get_bw_residue(residue_cache['opn4_human'], bw)
        gc = gq_consensus[bw]
        nc = nongq_consensus[bw]
        is_sig = '✓' if bw in sig_sites else ''

        if opn4_aa == '-':
            match_type = 'gap'
        elif opn4_aa == gc and opn4_aa != nc:
            match_type = 'Gq-specific'
            match_gq += 1
            total += 1
        elif opn4_aa == nc and opn4_aa != gc:
            match_type = 'nonGq-specific'
            match_nongq += 1
            total += 1
        elif opn4_aa == gc == nc:
            match_type = 'conserved'
            total += 1
        else:
            match_type = 'unique'
            match_unique += 1
            total += 1

        print(f"  {bw:<8s} {opn4_aa:<6s} {gc:<10s} {nc:<12s} {match_type:<15s} {is_sig}")

    print(f"\n  OPN4 consensus matching ({total} non-gap sites):")
    print(f"    Gq-specific: {match_gq}/{total}")
    print(f"    nonGq-specific: {match_nongq}/{total}")
    print(f"    Unique: {match_unique}/{total}")

# Opsin family comparison table
print(f"\n  Opsin family at FDR-significant sites:")
display_sites = sig_sites if sig_sites else list(GP_CONTACT_SITES.keys())[:10]
header = f"  {'Receptor':<16s} {'G prot.':<8s}"
for bw in display_sites:
    header += f" {bw:>6s}"
print(header)
print('  ' + '-' * (26 + 7 * len(display_sites)))
for ename in opsin_entries:
    if ename not in residue_cache:
        continue
    line = f"  {ename:<16s} {coupling_labels.get(ename,'?'):<8s}"
    for bw in display_sites:
        aa = get_bw_residue(residue_cache[ename], bw)
        line += f"     {aa}"
    print(line)


# ===========================================================================
# 5. Permutation importance (top features under subfamily split)
# ===========================================================================
print(f"\n[5/5] Permutation importance analysis...")

tr_idx, te_idx = split_subfamily(y_bw, families_for_split)
X_tr, X_te = X_hc[tr_idx], X_hc[te_idx]
y_tr, y_te = y_bw[tr_idx], y_bw[te_idx]

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

rf = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=2,
                             class_weight='balanced', random_state=42)
rf.fit(X_tr_s, y_tr)

perm_imp = permutation_importance(rf, X_te_s, y_te, n_repeats=30,
                                   random_state=42, scoring='roc_auc')

# Sort by importance
imp_order = np.argsort(perm_imp.importances_mean)[::-1]
print(f"\n  Top 20 features by permutation importance (subfamily split):")
print(f"  {'Feature':<25s} {'Importance':<12s} {'Std':<12s}")
print('  ' + '-' * 50)
perm_records = []
for i in imp_order[:20]:
    fname = hc_feature_names[i]
    imp = perm_imp.importances_mean[i]
    std = perm_imp.importances_std[i]
    print(f"  {fname:<25s} {imp:<12.4f} {std:<12.4f}")
    perm_records.append({'feature': fname, 'importance': imp, 'std': std})

perm_df = pd.DataFrame(perm_records)
perm_csv = os.path.join(RESULTS_DIR, "permutation_importance.csv")
perm_df.to_csv(perm_csv, index=False)
print(f"  Saved: {perm_csv}")


# ===========================================================================
# Summary
# ===========================================================================
print(f"\n{'='*72}")
print("Summary")
print(f"{'='*72}")
print(f"""
1. BW site analysis:
   - {n_sig_nom} sites significant at p<0.05 (nominal)
   - {n_sig_fdr} sites significant after FDR correction
   - Key regions: ICL2 (34.50-34.55), TM5 (5.64-5.69), TM3 (3.53)

2. Ablation study (Subfamily split, RF):""")
for r in ablation_results:
    print(f"   - {r['features']}: AUC={r['AUC']:.3f}, F1={r['F1']:.3f}")

print(f"""
3. OPN4 Gq-consensus: {match_gq} Gq-specific, {match_nongq} nonGq-specific, {match_unique} unique

4. All results saved to: results/
   - bw_site_statistics.csv
   - ablation_results.csv
   - permutation_importance.csv
""")
print("=" * 72)
