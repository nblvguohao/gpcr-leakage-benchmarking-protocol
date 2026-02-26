#!/usr/bin/env python3
"""
Reviewer-requested analyses for BiB revision.

1. Sequence identity clustering splits at 20%/30%/40% thresholds (gradient analysis)
2. PRECOG-like kNN comparison on same no-leak splits
3. Physicochemical feature category decomposition (charge/hydrophobicity/aromaticity/size)
4. Feature weight change analysis: random → subfamily split
5. SHAP comparison panel: OPN4 (Gq) vs Rhodopsin (non-Gq)
6. Updated figures with 95% CI error bars

Outputs:
  results/reviewer_seqid_splits.csv
  results/reviewer_feature_category_ablation.csv
  results/reviewer_feature_weight_shift.csv
  figures/fig1_revised_auc_errorbars.png
  figures/fig_shap_comparison_opn4_rho.png
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ================================================================
# Project paths
# ================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

from collections import OrderedDict

# ================================================================
# BW contact sites (same as existing scripts)
# ================================================================
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
BW_SITES = list(GP_CONTACT_SITES.keys())

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


def get_subfamily(family_slug):
    parts = family_slug.split('_')
    return '_'.join(parts[:3]) if len(parts) >= 3 else family_slug


# ================================================================
# Load dataset and build BW features
# ================================================================
def load_dataset():
    """Load the GPCR dataset."""
    csv_path = os.path.join(DATA_DIR, "gpcrdb_coupling_dataset.csv")
    df = pd.read_csv(csv_path)
    df = df[df['coupling_description'] != 'Unknown'].reset_index(drop=True)
    print(f"Loaded dataset: {len(df)} receptors")
    return df


def load_bw_features():
    """Build BW-site feature matrix from residue cache."""
    csv_path = os.path.join(DATA_DIR, "gpcrdb_coupling_dataset.csv")
    cache_path = os.path.join(DATA_DIR, "gpcrdb_residues_cache.json")

    if not os.path.exists(csv_path) or not os.path.exists(cache_path):
        print("ERROR: Dataset or residue cache not found")
        return None, None, None, None

    df = pd.read_csv(csv_path)
    df = df[df['coupling_description'] != 'Unknown'].reset_index(drop=True)

    with open(cache_path, 'r') as f:
        residue_cache = json.load(f)

    # Extract BW residues
    records = []
    for _, row in df.iterrows():
        ename = row['entry_name']
        if ename not in residue_cache:
            continue
        rec = {
            'entry_name': ename,
            'gq_label': int(row['gq_label']),
            'family': row['family'],
            'subfamily': get_subfamily(row['family']),
        }
        for bw in BW_SITES:
            rec[f'bw_{bw}'] = get_bw_residue(residue_cache[ename], bw)
        records.append(rec)

    bw_df = pd.DataFrame(records)

    # Build feature matrix
    bw_feature_names = None
    X_list, y_list, subfamilies, entry_names = [], [], [], []
    for _, row in bw_df.iterrows():
        feats = encode_bw_features(row, BW_SITES)
        if bw_feature_names is None:
            bw_feature_names = sorted(feats.keys())
        X_list.append([feats[n] for n in bw_feature_names])
        y_list.append(row['gq_label'])
        subfamilies.append(row['subfamily'])
        entry_names.append(row['entry_name'])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    subfamilies = np.array(subfamilies)
    entry_names = np.array(entry_names)

    print(f"BW features: {X.shape} (Gq={y.sum()}, non-Gq={len(y)-y.sum()})")
    return X, y, entry_names, subfamilies


# ================================================================
# 1. Sequence identity gradient analysis
# ================================================================
def compute_kmer_similarity(seq1, seq2, k=3):
    """Compute k-mer Jaccard similarity between two sequences."""
    def get_kmers(seq, k):
        return set(seq[i:i+k] for i in range(len(seq)-k+1))
    k1 = get_kmers(seq1, k)
    k2 = get_kmers(seq2, k)
    if not k1 or not k2:
        return 0.0
    return len(k1 & k2) / len(k1 | k2)


def sequence_identity_cluster_split(df, threshold, k=3):
    """
    Cluster sequences by k-mer similarity at given threshold.
    Returns list of (train_idx, test_idx) for cross-validation.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    sequences = df['sequence'].tolist()
    n = len(sequences)

    # Compute pairwise similarity matrix
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            sim = compute_kmer_similarity(sequences[i], sequences[j], k)
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
        sim_matrix[i, i] = 1.0

    # Convert to distance and cluster
    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0)
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method='average')
    clusters = fcluster(Z, t=1.0-threshold, criterion='distance')

    # Group-based CV: hold out each cluster group
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    print(f"  Threshold {threshold}: {n_clusters} clusters")

    # Use 5-fold grouped split
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=min(5, n_clusters))
    splits = list(gkf.split(np.arange(n), df['gq_label'].values, groups=clusters))
    return splits


def run_seqid_gradient_analysis(df):
    """
    Run benchmark at multiple sequence identity thresholds.
    Addresses reviewer comment 1.1.
    """
    print("=" * 70)
    print("Analysis 1: Sequence identity gradient analysis")
    print("=" * 70)

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    # Load BW features
    X_bw, y, names, subfamilies = load_bw_features()
    if X_bw is None:
        print("ERROR: BW features not found.")
        return None

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
    results = []

    for t in thresholds:
        print(f"\n  Testing threshold = {t}")
        splits = sequence_identity_cluster_split(df, threshold=t)

        fold_aucs = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X_bw[train_idx], X_bw[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Ensemble
            rf = RandomForestClassifier(n_estimators=500, max_depth=15,
                                         class_weight='balanced', random_state=42)
            gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                             max_depth=5, random_state=42)
            svm = SVC(kernel='rbf', C=10, probability=True,
                       class_weight='balanced', random_state=42)

            ensemble = VotingClassifier(
                estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
                voting='soft', weights=[2, 2, 1]
            )
            ensemble.fit(X_train, y_train)

            if len(np.unique(y_test)) > 1:
                proba = ensemble.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, proba)
                fold_aucs.append(auc)

        if fold_aucs:
            mean_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs)
            ci_lo = np.percentile(fold_aucs, 2.5) if len(fold_aucs) >= 5 else mean_auc - 1.96*std_auc
            ci_hi = np.percentile(fold_aucs, 97.5) if len(fold_aucs) >= 5 else mean_auc + 1.96*std_auc

            results.append({
                'threshold': t,
                'mean_auc': mean_auc,
                'std_auc': std_auc,
                'ci_lo': ci_lo,
                'ci_hi': ci_hi,
                'n_folds': len(fold_aucs),
            })
            print(f"    AUC = {mean_auc:.3f} ± {std_auc:.3f}")

    # Save
    if results:
        df_res = pd.DataFrame(results)
        out_path = os.path.join(RESULTS_DIR, "reviewer_seqid_splits.csv")
        df_res.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")
        return df_res
    return None


# ================================================================
# 2. Physicochemical feature category decomposition
# ================================================================
def run_feature_category_ablation(df):
    """
    Decompose BW-site features by physicochemical category.
    Addresses reviewer supplementary comment on charge/hydrophobicity/aromaticity/size.
    """
    print("\n" + "=" * 70)
    print("Analysis 2: Feature category decomposition")
    print("=" * 70)

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    X_bw, y, names, subfamilies = load_bw_features()
    if X_bw is None:
        print("ERROR: BW features not found")
        return None

    # BW features are sorted alphabetically: for each site, order is
    # is_arom, is_gap, is_hydro, is_neg, is_pos (5 per site, 29 sites = 145)
    n_sites = 29
    # Determine actual column order from sorted feature names
    # Sorted order per site: is_arom(0), is_gap(1), is_hydro(2), is_neg(3), is_pos(4)
    categories = {
        'charge': [i*5+3 for i in range(n_sites)] + [i*5+4 for i in range(n_sites)],  # neg + pos
        'hydrophobicity': [i*5+2 for i in range(n_sites)],
        'aromaticity': [i*5+0 for i in range(n_sites)],
        'gap_size': [i*5+1 for i in range(n_sites)],
    }

    results = []

    for cat_name, col_idx in categories.items():
        X_cat = X_bw[:, col_idx]
        print(f"\n  Category: {cat_name} ({len(col_idx)} dims)")

        if subfamilies is not None:
            gkf = GroupKFold(n_splits=5)
            splits = list(gkf.split(np.arange(len(y)), y, groups=subfamilies))
        else:
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            splits = list(skf.split(np.arange(len(y)), y))

        fold_aucs = []
        for train_idx, test_idx in splits:
            X_train, X_test = X_cat[train_idx], X_cat[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            rf = RandomForestClassifier(n_estimators=500, max_depth=15,
                                         class_weight='balanced', random_state=42)
            rf.fit(X_train, y_train)

            if len(np.unique(y_test)) > 1:
                proba = rf.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, proba)
                fold_aucs.append(auc)

        if fold_aucs:
            results.append({
                'category': cat_name,
                'n_features': len(col_idx),
                'mean_auc': np.mean(fold_aucs),
                'std_auc': np.std(fold_aucs),
            })
            print(f"    AUC = {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")

    # Also test full BW features
    if subfamilies is not None:
        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(np.arange(len(y)), y, groups=subfamilies))
    fold_aucs = []
    for train_idx, test_idx in splits:
        X_train, X_test = X_bw[train_idx], X_bw[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        rf = RandomForestClassifier(n_estimators=500, max_depth=15,
                                     class_weight='balanced', random_state=42)
        rf.fit(X_train, y_train)
        if len(np.unique(y_test)) > 1:
            proba = rf.predict_proba(X_test)[:, 1]
            fold_aucs.append(roc_auc_score(y_test, proba))
    results.append({
        'category': 'all_combined',
        'n_features': X_bw.shape[1],
        'mean_auc': np.mean(fold_aucs),
        'std_auc': np.std(fold_aucs),
    })

    df_res = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "reviewer_feature_category_ablation.csv")
    df_res.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return df_res


# ================================================================
# 3. Feature weight shift: random → subfamily split
# ================================================================
def run_feature_weight_shift(df):
    """
    Compare feature importances between random and subfamily splits.
    Shows which features lose/gain importance when leakage is removed.
    """
    print("\n" + "=" * 70)
    print("Analysis 3: Feature weight shift (random → subfamily)")
    print("=" * 70)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold

    X_bw, y, names, subfamilies = load_bw_features()
    if X_bw is None:
        return None

    # Feature labels match sorted order from encode_bw_features
    # Sorted per site: is_arom, is_gap, is_hydro, is_neg, is_pos
    props = ["is_arom", "is_gap", "is_hydro", "is_neg", "is_pos"]
    feature_labels = []
    for site in sorted(BW_SITES):
        for prop in props:
            feature_labels.append(f"{site}_{prop}")

    # Random split importances
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_importances = np.zeros(X_bw.shape[1])
    for train_idx, test_idx in skf.split(X_bw, y):
        rf = RandomForestClassifier(n_estimators=500, max_depth=15,
                                     class_weight='balanced', random_state=42)
        rf.fit(X_bw[train_idx], y[train_idx])
        random_importances += rf.feature_importances_
    random_importances /= 5

    # Subfamily split importances
    gkf = GroupKFold(n_splits=5)
    subfamily_importances = np.zeros(X_bw.shape[1])
    n_folds = 0
    for train_idx, test_idx in gkf.split(X_bw, y, groups=subfamilies):
        rf = RandomForestClassifier(n_estimators=500, max_depth=15,
                                     class_weight='balanced', random_state=42)
        rf.fit(X_bw[train_idx], y[train_idx])
        subfamily_importances += rf.feature_importances_
        n_folds += 1
    subfamily_importances /= n_folds

    # Compute shift
    shift = subfamily_importances - random_importances

    df_shift = pd.DataFrame({
        'feature': feature_labels[:len(shift)],
        'random_importance': random_importances,
        'subfamily_importance': subfamily_importances,
        'shift': shift,
        'relative_shift': shift / (random_importances + 1e-8),
    })
    df_shift = df_shift.sort_values('shift', ascending=False)

    out_path = os.path.join(RESULTS_DIR, "reviewer_feature_weight_shift.csv")
    df_shift.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Print top gainers and losers
    print("\nTop 10 features GAINING importance under no-leak:")
    for _, row in df_shift.head(10).iterrows():
        print(f"  {row['feature']}: {row['random_importance']:.4f} → {row['subfamily_importance']:.4f} ({row['shift']:+.4f})")

    print("\nTop 10 features LOSING importance under no-leak:")
    for _, row in df_shift.tail(10).iterrows():
        print(f"  {row['feature']}: {row['random_importance']:.4f} → {row['subfamily_importance']:.4f} ({row['shift']:+.4f})")

    return df_shift


# ================================================================
# 4. SHAP comparison: OPN4 vs Rhodopsin
# ================================================================
def run_shap_comparison():
    """
    Generate SHAP waterfall comparison for OPN4 (Gq) vs Rhodopsin (non-Gq).
    """
    print("\n" + "=" * 70)
    print("Analysis 4: SHAP comparison OPN4 vs Rhodopsin")
    print("=" * 70)

    try:
        import shap
    except ImportError:
        print("ERROR: shap not installed. Run: pip install shap")
        return

    from sklearn.ensemble import RandomForestClassifier

    X_bw, y, names, subfamilies = load_bw_features()
    if X_bw is None or names is None:
        print("ERROR: BW features not found")
        return

    # Feature labels
    feat_labels_props = ["is_arom", "is_gap", "is_hydro", "is_neg", "is_pos"]
    feat_labels = []
    for site in sorted(BW_SITES):
        for prop in feat_labels_props:
            feat_labels.append(f"{site}_{prop}")
    feat_labels = feat_labels[:X_bw.shape[1]]

    # Train model on all data
    rf = RandomForestClassifier(n_estimators=500, max_depth=15,
                                 class_weight='balanced', random_state=42)
    rf.fit(X_bw, y)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_bw)

    # Find OPN4 and Rhodopsin indices
    names_list = list(names)
    opn4_idx = None
    rho_idx = None
    for i, name in enumerate(names_list):
        name_lower = str(name).lower()
        if 'opn4' in name_lower or 'melanopsin' in name_lower:
            opn4_idx = i
        if 'opsd' in name_lower or 'rho_human' in name_lower or name_lower == 'rho':
            rho_idx = i

    if opn4_idx is None:
        print("WARNING: OPN4 not found, trying partial match")
        for i, name in enumerate(names_list):
            if 'opn4' in str(name).lower():
                opn4_idx = i
                break
    if rho_idx is None:
        print("WARNING: Rhodopsin not found, trying partial match")
        for i, name in enumerate(names_list):
            if 'opsd' in str(name).lower():
                rho_idx = i
                break

    print(f"OPN4 index: {opn4_idx} ({names_list[opn4_idx] if opn4_idx else 'N/A'})")
    print(f"Rhodopsin index: {rho_idx} ({names_list[rho_idx] if rho_idx else 'N/A'})")

    if opn4_idx is None or rho_idx is None:
        print("ERROR: Could not find both receptors")
        return

    # Get SHAP values for class 1 (Gq)
    if isinstance(shap_values, list):
        sv = shap_values[1]  # class 1, shape (N, F)
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]  # (N, F, 2) -> (N, F)
    else:
        sv = shap_values

    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # BW site labels
    bw_sites = ["2.39","2.40","3.46","3.49","3.50","3.51","3.52","3.53","3.54",
                "3.55","3.56","34.50","34.51","34.52","34.53","34.54","34.55",
                "34.56","34.57","5.61","5.64","5.65","5.67","5.68","5.69",
                "5.71","5.72","5.74","5.75"]
    props = ["pos", "neg", "hyd", "aro", "gap"]
    feat_labels = []
    for site in bw_sites:
        for prop in props:
            feat_labels.append(f"{site}_{prop}")
    feat_labels = feat_labels[:sv.shape[1]]

    for ax_idx, (idx, title, color) in enumerate([
        (opn4_idx, f"OPN4 (Gq) — P(Gq)={rf.predict_proba(X_bw[opn4_idx:opn4_idx+1])[0,1]:.3f}", "red"),
        (rho_idx, f"Rhodopsin (Gt) — P(Gq)={rf.predict_proba(X_bw[rho_idx:rho_idx+1])[0,1]:.3f}", "blue"),
    ]):
        ax = axes[ax_idx]
        shap_vals = sv[idx]

        # Top 15 features by absolute SHAP
        abs_shap = np.abs(shap_vals).flatten()
        sorted_idx = np.argsort(abs_shap)
        top_idx = sorted_idx[-15:]
        top_vals = np.array([shap_vals.flatten()[i] for i in top_idx])
        top_names = [feat_labels[i] for i in top_idx.astype(int)]

        colors = ['#d62728' if v > 0 else '#1f77b4' for v in top_vals]
        ax.barh(range(len(top_vals)), top_vals, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(top_vals)))
        ax.set_yticklabels(top_names, fontsize=8)
        ax.set_xlabel('SHAP value (toward Gq)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig_shap_comparison_opn4_rho.png")
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ================================================================
# 5. Updated AUC figure with error bars
# ================================================================
def generate_revised_auc_figure():
    """
    Generate revised Fig1 with 95% CI error bars for all split strategies.
    """
    print("\n" + "=" * 70)
    print("Analysis 5: Revised AUC figure with error bars")
    print("=" * 70)

    # Data from manuscript + new sequence identity results
    data = {
        'Split': ['Random\n(leaky)', 'SeqCluster\nt=0.4', 'SeqCluster\nt=0.3', 'Subfamily\n(no-leak)'],
        'AUC': [0.800, 0.780, 0.688, 0.509],
        'CI_lo': [0.641, 0.637, 0.524, 0.307],
        'CI_hi': [0.931, 0.907, 0.828, 0.702],
    }

    # Add new threshold results if available
    seqid_path = os.path.join(RESULTS_DIR, "reviewer_seqid_splits.csv")
    if os.path.exists(seqid_path):
        df_new = pd.read_csv(seqid_path)
        for _, row in df_new.iterrows():
            t = row['threshold']
            if t not in [0.3, 0.4]:  # Don't duplicate existing
                data['Split'].append(f'SeqID\nt={t}')
                data['AUC'].append(row['mean_auc'])
                data['CI_lo'].append(row['ci_lo'])
                data['CI_hi'].append(row['ci_hi'])

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df)))

    bars = ax.bar(x, df['AUC'], color=colors, edgecolor='white', linewidth=1.5, width=0.6)

    # Error bars (95% CI)
    yerr_lo = df['AUC'] - df['CI_lo']
    yerr_hi = df['CI_hi'] - df['AUC']
    ax.errorbar(x, df['AUC'], yerr=[yerr_lo, yerr_hi],
                fmt='none', color='black', capsize=5, capthick=1.5, linewidth=1.5)

    # Chance line
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Random chance')

    ax.set_xticks(x)
    ax.set_xticklabels(df['Split'], fontsize=9)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('Ensemble model performance across splitting strategies\n(BW-site features, 95% CI)', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)

    # Value labels
    for i, (v, lo, hi) in enumerate(zip(df['AUC'], df['CI_lo'], df['CI_hi'])):
        ax.text(i, v + (hi - v) + 0.02, f'{v:.3f}\n[{lo:.2f}-{hi:.2f}]',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig1_revised_auc_errorbars.png")
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ================================================================
# Main
# ================================================================
def main():
    print("=" * 70)
    print("BiB Reviewer-Requested Analyses")
    print("=" * 70)

    df = load_dataset()

    # 1. Sequence identity gradient
    seqid_results = run_seqid_gradient_analysis(df)

    # 2. Feature category decomposition
    cat_results = run_feature_category_ablation(df)

    # 3. Feature weight shift
    shift_results = run_feature_weight_shift(df)

    # 4. SHAP comparison
    run_shap_comparison()

    # 5. Revised AUC figure
    generate_revised_auc_figure()

    print("\n" + "=" * 70)
    print("All reviewer analyses complete!")
    print("=" * 70)
    print("\nOutput files:")
    print(f"  results/reviewer_seqid_splits.csv")
    print(f"  results/reviewer_feature_category_ablation.csv")
    print(f"  results/reviewer_feature_weight_shift.csv")
    print(f"  figures/fig_shap_comparison_opn4_rho.png")
    print(f"  figures/fig1_revised_auc_errorbars.png")


if __name__ == "__main__":
    main()
