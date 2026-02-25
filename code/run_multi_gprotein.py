#!/usr/bin/env python3
"""
Multi-G protein BW-site analysis + dual-coupling analysis + larger ESM-2 test.
Addresses key limitations:
  1. Generalize BW-site framework to Gs, Gi/o, G12/13 (not just Gq)
  2. Analyze dual-coupled receptors at BW sites
  3. Test larger ESM-2 model (35M vs 8M)
"""
import os, sys, json, csv, warnings
import numpy as np
import pandas as pd
from collections import OrderedDict, Counter
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")

# BW contact sites
GP_CONTACT_SITES = [
    '34.50','34.51','34.52','34.53','34.54','34.55','34.56','34.57',
    '3.49','3.50','3.51','3.53','3.54','3.55','3.56',
    '5.61','5.64','5.65','5.67','5.68','5.69','5.71',
    '6.32','6.33','6.36','6.37',
    '8.47','8.48','8.49',
]

def load_data():
    """Load GPCR dataset with coupling labels."""
    json_file = os.path.join(DATA_DIR, "gpcrdb_coupling_dataset.json")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Parse multi-G protein labels from coupling_description
    for r in data:
        desc = r.get('coupling_description', '')
        r['couples_Gs'] = 1 if 'Gs' in desc else 0
        r['couples_Gi'] = 1 if ('Gi' in desc or 'Gt' in desc) else 0
        r['couples_Gq'] = r['gq_label']
        r['couples_G12'] = 1 if 'G12' in desc else 0
        
        # Count how many G proteins this receptor couples to
        r['n_coupling'] = r['couples_Gs'] + r['couples_Gi'] + r['couples_Gq'] + r['couples_G12']
        
        # Primary G protein
        if 'Gq' in desc and 'primary' in desc.split('Gq')[0][-20:] + 'primary':
            pass  # already handled
        r['primary_g'] = 'unknown'
        if 'Gq/11 primary' in desc or 'Gq primary' in desc:
            r['primary_g'] = 'Gq'
        elif 'Gs primary' in desc:
            r['primary_g'] = 'Gs'
        elif 'Gi/o primary' in desc or 'Gi primary' in desc:
            r['primary_g'] = 'Gi'
        elif 'Gt primary' in desc:
            r['primary_g'] = 'Gt'
        elif 'G12/13 primary' in desc:
            r['primary_g'] = 'G12'
    
    return data

def load_bw_annotations():
    """Load cached BW annotations from gpcrdb_residues_cache.json."""
    bw_file = os.path.join(DATA_DIR, "gpcrdb_residues_cache.json")
    if os.path.exists(bw_file):
        with open(bw_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def get_bw_residue(residue_data, bw_label):
    """Extract amino acid at a specific BW position from residue list."""
    for res in residue_data:
        dgn = res.get('display_generic_number', '')
        if dgn:
            bw_part = dgn.split('x')[0] if 'x' in dgn else dgn
            if bw_part == bw_label:
                return res.get('amino_acid', '-')
    return '-'

def get_aa_at_bw(entry_name, bw_annotations):
    """Get amino acid at each BW site for a receptor."""
    residue_data = bw_annotations.get(entry_name, [])
    result = {}
    for site in GP_CONTACT_SITES:
        result[site] = get_bw_residue(residue_data, site)
    return result

def run_chi2_for_coupling(data, bw_annotations, label_key, label_name):
    """Run chi-squared tests at 29 BW sites for a given coupling label."""
    pos_entries = [r for r in data if r[label_key] == 1]
    neg_entries = [r for r in data if r[label_key] == 0]
    
    results = []
    for site in GP_CONTACT_SITES:
        pos_aas = [get_aa_at_bw(r['entry_name'], bw_annotations).get(site, '-') for r in pos_entries]
        neg_aas = [get_aa_at_bw(r['entry_name'], bw_annotations).get(site, '-') for r in neg_entries]
        
        # Build contingency table
        all_aas = sorted(set(pos_aas + neg_aas))
        if len(all_aas) <= 1:
            results.append({'site': site, 'chi2': 0, 'p': 1.0, 'cramers_v': 0})
            continue
        
        table = []
        for aa in all_aas:
            table.append([pos_aas.count(aa), neg_aas.count(aa)])
        table = np.array(table)
        
        # Remove rows with all zeros
        table = table[table.sum(axis=1) > 0]
        if table.shape[0] < 2:
            results.append({'site': site, 'chi2': 0, 'p': 1.0, 'cramers_v': 0})
            continue
        
        try:
            chi2, p, dof, _ = chi2_contingency(table)
            n = table.sum()
            k = min(table.shape)
            cramers_v = np.sqrt(chi2 / (n * (k - 1))) if k > 1 else 0
        except:
            chi2, p, cramers_v = 0, 1.0, 0
        
        results.append({'site': site, 'chi2': chi2, 'p': p, 'cramers_v': cramers_v})
    
    # FDR correction
    pvals = [r['p'] for r in results]
    reject, fdr_pvals, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    for i, r in enumerate(results):
        r['fdr'] = fdr_pvals[i]
        r['significant'] = reject[i]
    
    n_sig = sum(1 for r in results if r['significant'])
    sig_sites = [r['site'] for r in results if r['significant']]
    print(f"  {label_name}: {len(pos_entries)} positive, {len(neg_entries)} negative")
    print(f"    FDR-significant sites: {n_sig}/29 -> {sig_sites}")
    
    return results

def run_multi_gprotein_analysis(data, bw_annotations):
    """Run BW-site chi-squared for all 4 G protein families."""
    print("=" * 72)
    print("ANALYSIS 1: Multi-G protein BW-site chi-squared")
    print("=" * 72)
    
    all_results = {}
    for label_key, label_name in [
        ('couples_Gq', 'Gq/11'),
        ('couples_Gs', 'Gs'),
        ('couples_Gi', 'Gi/o'),
        ('couples_G12', 'G12/13'),
    ]:
        results = run_chi2_for_coupling(data, bw_annotations, label_key, label_name)
        all_results[label_name] = results
    
    return all_results

def run_dual_coupling_analysis(data, bw_annotations):
    """Analyze BW-site patterns in dual-coupled vs single-coupled receptors."""
    print("\n" + "=" * 72)
    print("ANALYSIS 2: Dual-coupling receptor analysis")
    print("=" * 72)
    
    # Categorize receptors
    gq_only = [r for r in data if r['couples_Gq'] == 1 and r['n_coupling'] == 1]
    gq_dual = [r for r in data if r['couples_Gq'] == 1 and r['n_coupling'] > 1]
    non_gq = [r for r in data if r['couples_Gq'] == 0]
    
    print(f"  Gq-only: {len(gq_only)}")
    print(f"  Gq + other: {len(gq_dual)}")
    print(f"  Non-Gq: {len(non_gq)}")
    
    # For FDR sites, compare AA distributions
    fdr_sites = ['34.50', '34.53', '3.53', '5.65', '5.71']
    
    results = []
    for site in fdr_sites:
        gq_only_aas = [get_aa_at_bw(r['entry_name'], bw_annotations).get(site, '-') for r in gq_only]
        gq_dual_aas = [get_aa_at_bw(r['entry_name'], bw_annotations).get(site, '-') for r in gq_dual]
        non_gq_aas = [get_aa_at_bw(r['entry_name'], bw_annotations).get(site, '-') for r in non_gq]
        
        # Most common AA in each group
        gq_only_top = Counter(gq_only_aas).most_common(2)
        gq_dual_top = Counter(gq_dual_aas).most_common(2)
        non_gq_top = Counter(non_gq_aas).most_common(2)
        
        def fmt(top):
            return '/'.join(f"{aa}({c})" for aa, c in top)
        
        results.append({
            'site': site,
            'gq_only_top': fmt(gq_only_top),
            'gq_dual_top': fmt(gq_dual_top),
            'non_gq_top': fmt(non_gq_top),
        })
        print(f"  BW {site}: Gq-only={fmt(gq_only_top)}  Gq+other={fmt(gq_dual_top)}  non-Gq={fmt(non_gq_top)}")
    
    return results

def run_larger_esm2(data, bw_annotations):
    """Test larger ESM-2 model (35M) at BW sites."""
    print("\n" + "=" * 72)
    print("ANALYSIS 3: Larger ESM-2 model (esm2_t12_35M_UR50D)")
    print("=" * 72)
    
    try:
        import torch
        import esm
    except ImportError:
        print("  ESM-2 not available, skipping")
        return None
    
    # Check if cached
    cache_file = os.path.join(DATA_DIR, "esm2_35m_bw_embeddings.npz")
    if os.path.exists(cache_file):
        print("  Loading cached 35M embeddings...")
        npz = np.load(cache_file, allow_pickle=True)
        embeddings = npz['embeddings'].item()
        entry_names = npz['entry_names'].tolist()
    else:
        print("  Loading ESM-2 35M model...")
        try:
            model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        except Exception as e:
            print(f"  Failed to load 35M model: {e}")
            print("  Trying to download...")
            try:
                model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            except:
                print("  Cannot load esm2_t12_35M_UR50D, skipping")
                return None
        
        model.eval()
        batch_converter = alphabet.get_batch_converter()
        embed_dim = model.embed_dim  # 480 for 35M
        
        embeddings = {}
        entry_names = []
        
        for i, r in enumerate(data):
            entry = r['entry_name']
            seq = r['sequence']
            
            # Get BW positions
            annot = bw_annotations.get(entry, {})
            bw_indices = {}
            for site in GP_CONTACT_SITES:
                if site in annot and annot[site] != '-':
                    # Find sequence index for this BW position
                    # BW annotations store the AA, we need the index
                    pass
            
            # Extract per-token embeddings
            try:
                batch_labels, batch_strs, batch_tokens = batch_converter([(entry, seq)])
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[12])
                token_reps = results["representations"][12][0]  # [L+2, D]
                # Mean pool (exclude BOS/EOS)
                mean_emb = token_reps[1:-1].mean(dim=0).numpy()
                embeddings[entry] = mean_emb
                entry_names.append(entry)
            except Exception as e:
                if i < 3:
                    print(f"    Error for {entry}: {e}")
                continue
            
            if (i+1) % 50 == 0:
                print(f"    Processed {i+1}/{len(data)}")
        
        # Cache
        np.savez_compressed(cache_file, 
                          embeddings=embeddings, 
                          entry_names=entry_names)
        print(f"  Cached {len(embeddings)} embeddings to {cache_file}")
    
    # Now evaluate with sklearn
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    
    # Prepare feature matrix
    valid_entries = [r for r in data if r['entry_name'] in embeddings]
    X = np.array([embeddings[r['entry_name']] for r in valid_entries])
    y = np.array([r['gq_label'] for r in valid_entries])
    families = [r['family'] for r in valid_entries]
    
    print(f"  Valid entries: {len(valid_entries)}, features: {X.shape[1]}")
    
    # Subfamily split evaluation (same as paper)
    from sklearn.model_selection import GroupKFold
    
    # Use first 6 chars of family as group (subfamily level)
    groups = [f[:6] for f in families]
    unique_groups = list(set(groups))
    
    # Simple grouped evaluation
    aucs = []
    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        if len(set(y_test)) < 2:
            continue
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        rf = RandomForestClassifier(n_estimators=500, max_depth=15, 
                                     class_weight='balanced', random_state=42)
        rf.fit(X_train_s, y_train)
        
        try:
            auc = roc_auc_score(y_test, rf.predict_proba(X_test_s)[:, 1])
            aucs.append(auc)
        except:
            pass
    
    mean_auc = np.mean(aucs) if aucs else 0
    std_auc = np.std(aucs) if aucs else 0
    print(f"  ESM-2 35M mean-pool: AUC = {mean_auc:.3f} +/- {std_auc:.3f} (GroupKFold)")
    
    return {'mean_auc': mean_auc, 'std_auc': std_auc, 'n_folds': len(aucs)}

def generate_multi_gprotein_figure(all_results):
    """Generate comparison figure for multi-G protein BW-site analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('BW-Site Statistical Significance Across G Protein Families', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    gp_names = ['Gq/11', 'Gs', 'Gi/o', 'G12/13']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    for idx, (gp_name, color) in enumerate(zip(gp_names, colors)):
        ax = axes[idx // 2][idx % 2]
        results = all_results[gp_name]
        
        sites = [r['site'] for r in results]
        neg_log_p = [-np.log10(max(r['p'], 1e-20)) for r in results]
        sig = [r['significant'] for r in results]
        
        bars = ax.bar(range(len(sites)), neg_log_p, color=[color if s else '#bdc3c7' for s in sig],
                      edgecolor='white', linewidth=0.5)
        
        # FDR threshold line
        fdr_threshold = -np.log10(0.05)
        ax.axhline(y=fdr_threshold, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_title(f'{gp_name} ({sum(sig)} FDR-significant sites)', fontsize=12, fontweight='bold')
        ax.set_ylabel('$-\\log_{10}(p)$')
        ax.set_xticks(range(len(sites)))
        ax.set_xticklabels(sites, rotation=90, fontsize=7)
        
        # Annotate significant sites
        for i, (s, nlp) in enumerate(zip(sig, neg_log_p)):
            if s:
                ax.text(i, nlp + 0.1, '★', ha='center', fontsize=8, color='red')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_png = os.path.join(FIGURES_DIR, "fig14_multi_gprotein_bw.png")
    out_pdf = os.path.join(FIGURES_DIR, "fig14_multi_gprotein_bw.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")

def generate_dual_coupling_figure(data, bw_annotations):
    """Generate figure comparing BW-site patterns across coupling categories."""
    fdr_sites = ['34.50', '34.53', '3.53', '5.65', '5.71']
    
    gq_only = [r for r in data if r['couples_Gq'] == 1 and r['n_coupling'] == 1]
    gq_dual = [r for r in data if r['couples_Gq'] == 1 and r['n_coupling'] > 1]
    non_gq = [r for r in data if r['couples_Gq'] == 0]
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle('AA Distributions at FDR Sites: Gq-only vs Dual-coupled vs Non-Gq',
                 fontsize=12, fontweight='bold')
    
    for idx, site in enumerate(fdr_sites):
        ax = axes[idx]
        
        groups = {'Gq-only': gq_only, 'Gq+other': gq_dual, 'Non-Gq': non_gq}
        group_colors = {'Gq-only': '#e74c3c', 'Gq+other': '#f39c12', 'Non-Gq': '#3498db'}
        
        # Get all AAs
        all_aas = set()
        group_counts = {}
        for gname, gdata in groups.items():
            aas = [get_aa_at_bw(r['entry_name'], bw_annotations).get(site, '-') for r in gdata]
            counts = Counter(aas)
            group_counts[gname] = counts
            all_aas.update(aas)
        
        # Top 6 AAs by total frequency
        total_counts = Counter()
        for gc in group_counts.values():
            total_counts.update(gc)
        top_aas = [aa for aa, _ in total_counts.most_common(6)]
        
        x = np.arange(len(top_aas))
        width = 0.25
        
        for i, (gname, gc) in enumerate(group_counts.items()):
            total = sum(gc.values())
            freqs = [gc.get(aa, 0) / total * 100 for aa in top_aas]
            ax.bar(x + (i - 1) * width, freqs, width, label=gname, 
                   color=group_colors[gname], alpha=0.8)
        
        ax.set_title(f'BW {site}', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(top_aas, fontsize=9)
        ax.set_ylabel('Frequency (%)' if idx == 0 else '')
        if idx == 4:
            ax.legend(fontsize=7, loc='upper right')
    
    plt.tight_layout()
    out_png = os.path.join(FIGURES_DIR, "fig15_dual_coupling_bw.png")
    out_pdf = os.path.join(FIGURES_DIR, "fig15_dual_coupling_bw.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_png}")

def save_multi_gprotein_csv(all_results):
    """Save multi-G protein results to CSV."""
    out_file = os.path.join(RESULTS_DIR, "multi_gprotein_bw_stats.csv")
    rows = []
    for gp_name, results in all_results.items():
        for r in results:
            rows.append({
                'g_protein': gp_name,
                'bw_site': r['site'],
                'chi2': f"{r['chi2']:.2f}",
                'p_value': f"{r['p']:.2e}",
                'fdr': f"{r['fdr']:.2e}",
                'cramers_v': f"{r['cramers_v']:.3f}",
                'significant': r['significant'],
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")

def main():
    print("Multi-G Protein Analysis Script")
    print("Addressing paper limitations: generalization, dual-coupling, larger ESM-2")
    print("=" * 72)
    
    # Load data
    data = load_data()
    bw_annotations = load_bw_annotations()
    
    # Filter to entries with BW annotations
    valid_entries = [r['entry_name'] for r in data if r['entry_name'] in bw_annotations]
    print(f"Total receptors: {len(data)}, with BW annotations: {len(valid_entries)}")
    
    # Coupling distribution
    print("\nCoupling distribution:")
    for gp in ['Gq', 'Gs', 'Gi', 'G12']:
        key = f'couples_{gp}'
        n = sum(1 for r in data if r[key] == 1)
        print(f"  {gp}: {n} receptors")
    
    n_multi = sum(1 for r in data if r['n_coupling'] > 1)
    print(f"  Multi-coupled: {n_multi}")
    
    # Analysis 1: Multi-G protein BW-site
    all_results = run_multi_gprotein_analysis(data, bw_annotations)
    save_multi_gprotein_csv(all_results)
    generate_multi_gprotein_figure(all_results)
    
    # Analysis 2: Dual-coupling
    dual_results = run_dual_coupling_analysis(data, bw_annotations)
    generate_dual_coupling_figure(data, bw_annotations)
    
    # Analysis 3: Larger ESM-2
    esm2_results = run_larger_esm2(data, bw_annotations)
    
    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY OF NEW FINDINGS")
    print("=" * 72)
    
    print("\n1. Multi-G protein BW-site analysis:")
    for gp_name, results in all_results.items():
        n_sig = sum(1 for r in results if r['significant'])
        sig_sites = [r['site'] for r in results if r['significant']]
        print(f"   {gp_name}: {n_sig} FDR-significant sites -> {sig_sites}")
    
    print(f"\n2. Dual-coupling: {sum(1 for r in data if r['couples_Gq']==1 and r['n_coupling']==1)} Gq-only, "
          f"{sum(1 for r in data if r['couples_Gq']==1 and r['n_coupling']>1)} Gq+other")
    
    if esm2_results:
        print(f"\n3. ESM-2 35M: AUC = {esm2_results['mean_auc']:.3f} +/- {esm2_results['std_auc']:.3f}")

if __name__ == "__main__":
    main()
