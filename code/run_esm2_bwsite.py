#!/usr/bin/env python3
"""
ESM-2 BW-Site Per-Residue Embedding Model
==========================================
Instead of mean-pooling ESM-2 embeddings (loses positional info),
extract per-residue embeddings at 29 G protein contact BW positions.

This creates a 29 × D matrix per receptor (D=320 for esm2_t6_8M),
which is then used for Gq coupling prediction via:
  1. Flattened MLP: concat all 29 position embeddings → 29*320 = 9280d → PCA → classifier
  2. Attention-pooled: learned attention over 29 positions → 320d → classifier
  3. Simple mean over 29 BW positions → 320d → classifier

Usage: py code/run_esm2_bwsite.py
"""
import os
import sys
import json
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DATASET_FILE = os.path.join(DATA_DIR, "gpcrdb_coupling_dataset.csv")
CACHE_FILE = os.path.join(DATA_DIR, "gpcrdb_residues_cache.json")

RANDOM_SEED = 42
N_BOOTSTRAP = 1000
np.random.seed(RANDOM_SEED)

print("=" * 72)
print("ESM-2 BW-Site Per-Residue Embedding Model")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 72)

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
BW_SITES = list(GP_CONTACT_SITES.keys())

# ===========================================================================
# Load data
# ===========================================================================
df = pd.read_csv(DATASET_FILE)
df = df[df['coupling_description'] != 'Unknown'].reset_index(drop=True)

with open(CACHE_FILE, 'r') as f:
    residue_cache = json.load(f)

print(f"Dataset: {len(df)} GPCRs, BW cache: {len(residue_cache)} entries")


def get_bw_sequence_positions(residue_data, bw_labels):
    """Map BW labels to 0-indexed sequence positions."""
    positions = {}
    for res in residue_data:
        dgn = res.get('display_generic_number', '')
        seq_num = res.get('sequence_number', None)
        if dgn and seq_num is not None:
            bw_part = dgn.split('x')[0] if 'x' in dgn else dgn
            if bw_part in bw_labels:
                positions[bw_part] = seq_num - 1  # 0-indexed
        for alt in res.get('alternative_generic_numbers', []):
            if alt.get('scheme') == 'BW' and alt.get('label') in bw_labels:
                if seq_num is not None:
                    positions[alt['label']] = seq_num - 1
    return positions


# ===========================================================================
# Split functions (inline)
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

def bootstrap_metrics(y_true, y_prob, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    aucs, praucs, f1s = [], [], []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(yt, yp))
        except:
            pass
        try:
            praucs.append(average_precision_score(yt, yp))
        except:
            pass
        ypred = (yp >= 0.5).astype(int)
        f1s.append(f1_score(yt, ypred, zero_division=0))
    def ci(arr):
        if len(arr) < 10:
            return (np.nan, np.nan)
        return (np.percentile(arr, 2.5), np.percentile(arr, 97.5))
    return {
        'AUC': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) >= 2 else np.nan,
        'AUC_lo': ci(aucs)[0], 'AUC_hi': ci(aucs)[1],
        'PR_AUC': average_precision_score(y_true, y_prob) if len(np.unique(y_true)) >= 2 else np.nan,
        'PRAUC_lo': ci(praucs)[0], 'PRAUC_hi': ci(praucs)[1],
        'F1': f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0),
        'F1_lo': ci(f1s)[0], 'F1_hi': ci(f1s)[1],
    }

# ===========================================================================
# Step 1: Get BW position mappings for all receptors
# ===========================================================================
print("\n[Step 1] Mapping BW positions to sequence indices...")

valid_entries = []
valid_seqs = []
valid_labels = []
valid_families = []
bw_pos_maps = []

for _, row in df.iterrows():
    ename = row['entry_name']
    seq = row['sequence']
    if ename not in residue_cache or not seq or len(seq) < 50:
        continue
    
    pos_map = get_bw_sequence_positions(residue_cache[ename], set(BW_SITES))
    
    # Require at least 15 of 29 BW positions to be mapped
    if len(pos_map) < 15:
        continue
    
    valid_entries.append(ename)
    valid_seqs.append(seq)
    valid_labels.append(row['gq_label'])
    valid_families.append(str(row.get('family', '')))
    bw_pos_maps.append(pos_map)

y_all = np.array(valid_labels)
print(f"  {len(valid_entries)} receptors with >=15 BW positions mapped")
print(f"  Gq={int(y_all.sum())}, non-Gq={len(y_all)-int(y_all.sum())}")

# Coverage statistics
coverages = [len(pm) for pm in bw_pos_maps]
print(f"  BW coverage: mean={np.mean(coverages):.1f}, min={np.min(coverages)}, max={np.max(coverages)} of 29")


# ===========================================================================
# Step 2: Extract ESM-2 per-residue embeddings at BW positions
# ===========================================================================
print(f"\n[Step 2] Extracting ESM-2 per-residue embeddings at BW positions...")

ESM_BW_CACHE = os.path.join(DATA_DIR, "esm2_bw_embeddings.npz")

if os.path.exists(ESM_BW_CACHE):
    print("  Loading cached BW embeddings...")
    cache_data = np.load(ESM_BW_CACHE, allow_pickle=True)
    X_bw_emb = cache_data['X_bw_emb']
    cached_entries = cache_data['entries'].tolist()
    if cached_entries == valid_entries and X_bw_emb.shape[0] == len(valid_entries):
        emb_dim = X_bw_emb.shape[2]
        print(f"  ✓ Loaded from cache: {X_bw_emb.shape} (29 sites × {emb_dim}d)")
    else:
        print("  Cache mismatch, re-extracting...")
        X_bw_emb = None
else:
    X_bw_emb = None

if X_bw_emb is None:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        model_name = "facebook/esm2_t6_8M_UR50D"
        print(f"  Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        esm_model = AutoModel.from_pretrained(model_name)
        esm_model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        esm_model = esm_model.to(device)
        print(f"  Device: {device}")
        
        n_sites = len(BW_SITES)
        all_bw_embeddings = []
        t0 = time.time()
        
        for idx in range(len(valid_seqs)):
            seq = valid_seqs[idx][:1022]  # ESM-2 max length
            pos_map = bw_pos_maps[idx]
            
            # Tokenize
            inputs = tokenizer([seq], return_tensors="pt", padding=False,
                             truncation=True, max_length=1024).to(device)
            
            with torch.no_grad():
                outputs = esm_model(**inputs)
            
            # last_hidden_state shape: (1, seq_len+2, emb_dim)
            # Positions are +1 offset due to <cls> token
            hidden = outputs.last_hidden_state[0].cpu().numpy()  # (seq_len+2, emb_dim)
            emb_dim = hidden.shape[1]
            
            # Extract embedding at each BW position
            site_embeddings = np.zeros((n_sites, emb_dim), dtype=np.float32)
            for si, bw in enumerate(BW_SITES):
                if bw in pos_map:
                    seq_pos = pos_map[bw]  # 0-indexed in sequence
                    token_pos = seq_pos + 1  # +1 for <cls> token
                    if 0 <= token_pos < hidden.shape[0] - 1:  # -1 for <eos>
                        site_embeddings[si] = hidden[token_pos]
                    # else: leave as zero (gap)
                # else: leave as zero (position not present)
            
            all_bw_embeddings.append(site_embeddings)
            
            if idx % 30 == 0:
                elapsed = time.time() - t0
                print(f"    Progress: {idx+1}/{len(valid_seqs)} ({elapsed:.0f}s)")
        
        X_bw_emb = np.array(all_bw_embeddings, dtype=np.float32)
        print(f"  ESM-2 BW embeddings: {X_bw_emb.shape} ({time.time()-t0:.0f}s)")
        
        # Cache
        np.savez_compressed(ESM_BW_CACHE, X_bw_emb=X_bw_emb, entries=np.array(valid_entries))
        print(f"  ✓ Cached to {ESM_BW_CACHE}")
        
    except Exception as e:
        print(f"  ESM-2 extraction failed: {e}")
        import traceback
        traceback.print_exc()
        print("  Exiting.")
        sys.exit(1)

n_samples, n_sites, emb_dim = X_bw_emb.shape
print(f"  Final: {n_samples} samples × {n_sites} sites × {emb_dim}d embedding")


# ===========================================================================
# Step 3: Build feature representations
# ===========================================================================
print(f"\n[Step 3] Building feature representations...")

# Strategy A: Mean over 29 BW positions → 320d
X_bw_mean = X_bw_emb.mean(axis=1)  # (n_samples, 320)
print(f"  A) BW-mean: {X_bw_mean.shape}")

# Strategy B: Flatten → PCA to reduce dimensionality
X_flat = X_bw_emb.reshape(n_samples, -1)  # (n_samples, 29*320=9280)
print(f"  B) Flattened: {X_flat.shape}")

# Strategy C: Weighted mean using FDR-significant site emphasis
FDR_SIG = {'34.50', '34.53', '3.53', '5.65', '5.71'}
weights = np.array([3.0 if bw in FDR_SIG else 1.0 for bw in BW_SITES])
weights = weights / weights.sum()
X_bw_weighted = np.zeros((n_samples, emb_dim), dtype=np.float32)
for si in range(n_sites):
    X_bw_weighted += weights[si] * X_bw_emb[:, si, :]
print(f"  C) FDR-weighted mean: {X_bw_weighted.shape}")

# Also load BW-site binary features and mean-pool ESM-2 for comparison
# BW binary (from run_interpretability)
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

X_bw_binary_list = []
for ename in valid_entries:
    feats = []
    for bw in BW_SITES:
        aa = get_bw_residue(residue_cache[ename], bw)
        feats.extend([
            1.0 if aa in pos_aa else 0.0,
            1.0 if aa in neg_aa else 0.0,
            1.0 if aa in set('AILMFVPW') else 0.0,
            1.0 if aa in set('FYW') else 0.0,
            1.0 if aa == '-' else 0.0,
        ])
    X_bw_binary_list.append(feats)

X_bw_binary = np.array(X_bw_binary_list, dtype=np.float32)
print(f"  BW-binary (baseline): {X_bw_binary.shape}")

# Load mean-pool ESM-2 from cache
ESM_MEAN_CACHE = os.path.join(DATA_DIR, "esm2_embeddings.npy")
ESM_ENTRIES_CACHE = os.path.join(DATA_DIR, "esm2_entries.json")
X_esm_mean = None
if os.path.exists(ESM_MEAN_CACHE) and os.path.exists(ESM_ENTRIES_CACHE):
    X_esm_all = np.load(ESM_MEAN_CACHE)
    with open(ESM_ENTRIES_CACHE) as f:
        esm_entries = json.load(f)
    # Map to our valid_entries order
    esm_entry_map = {e: i for i, e in enumerate(esm_entries)}
    esm_idx = [esm_entry_map.get(e, -1) for e in valid_entries]
    if all(i >= 0 for i in esm_idx):
        X_esm_mean = X_esm_all[esm_idx]
        print(f"  ESM-2 mean-pool (baseline): {X_esm_mean.shape}")
    else:
        print(f"  ESM-2 mean-pool: entry mismatch, skipping")


# ===========================================================================
# Step 4: Benchmark all representations under subfamily split
# ===========================================================================
print(f"\n[Step 4] Benchmarking under subfamily split...")

def build_ensemble():
    rf = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=2,
                                 class_weight='balanced', random_state=RANDOM_SEED)
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                      subsample=0.8, random_state=RANDOM_SEED)
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True,
               class_weight='balanced', random_state=RANDOM_SEED)
    return VotingClassifier(estimators=[('rf', rf), ('gbm', gbm), ('svm', svm)],
                             voting='soft', weights=[2, 2, 1])

def evaluate_features(X, y, families, label, use_pca=False, pca_dim=100):
    tr_idx, te_idx = split_subfamily(y, families)
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    if use_pca and X_tr_s.shape[1] > pca_dim:
        pca = PCA(n_components=min(pca_dim, X_tr_s.shape[0] - 1), random_state=RANDOM_SEED)
        X_tr_s = pca.fit_transform(X_tr_s)
        X_te_s = pca.transform(X_te_s)
        explained = pca.explained_variance_ratio_.sum()
        label_ext = f"{label} (PCA→{X_tr_s.shape[1]}d, {explained:.1%} var)"
    else:
        label_ext = label
    
    models = {
        'RF': RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=2,
                                      class_weight='balanced', random_state=RANDOM_SEED),
        'Ensemble': build_ensemble(),
    }
    
    results = []
    for mname, model in models.items():
        t0 = time.time()
        model.fit(X_tr_s, y_tr)
        y_prob = model.predict_proba(X_te_s)[:, 1]
        elapsed = time.time() - t0
        
        metrics = bootstrap_metrics(y_te, y_prob, n_boot=N_BOOTSTRAP)
        metrics['feature_set'] = label_ext
        metrics['model'] = mname
        metrics['n_features'] = X_tr_s.shape[1]
        metrics['train_time'] = elapsed
        results.append(metrics)
        
        print(f"  {label_ext:<45s} {mname:<10s} AUC={metrics['AUC']:.3f} "
              f"[{metrics['AUC_lo']:.3f}-{metrics['AUC_hi']:.3f}]  "
              f"PR={metrics['PR_AUC']:.3f}  F1={metrics['F1']:.3f}  ({elapsed:.1f}s)")
    
    return results

all_results = []

# Baseline: BW-binary (145d)
print("\n--- BW-binary (145d) ---")
all_results.extend(evaluate_features(X_bw_binary, y_all, valid_families, "BW-binary (145d)"))

# Baseline: ESM-2 mean-pool (320d)
if X_esm_mean is not None:
    print("\n--- ESM-2 mean-pool (320d) ---")
    all_results.extend(evaluate_features(X_esm_mean, y_all, valid_families, "ESM-2 mean-pool (320d)"))

# NEW: ESM-2 BW-mean (320d)
print("\n--- ESM-2 BW-mean (320d) ---")
all_results.extend(evaluate_features(X_bw_mean, y_all, valid_families, "ESM-2 BW-mean (320d)"))

# NEW: ESM-2 BW-weighted (320d)
print("\n--- ESM-2 BW-weighted (FDR, 320d) ---")
all_results.extend(evaluate_features(X_bw_weighted, y_all, valid_families, "ESM-2 BW-weighted (320d)"))

# NEW: ESM-2 BW-flat + PCA
print("\n--- ESM-2 BW-flat + PCA ---")
all_results.extend(evaluate_features(X_flat, y_all, valid_families, 
                                      "ESM-2 BW-flat", use_pca=True, pca_dim=50))

# NEW: ESM-2 BW-flat + PCA (100d)
print("\n--- ESM-2 BW-flat + PCA (100d) ---")
all_results.extend(evaluate_features(X_flat, y_all, valid_families,
                                      "ESM-2 BW-flat", use_pca=True, pca_dim=100))

# COMBINED: BW-binary + ESM-2 BW-mean
print("\n--- BW-binary + ESM-2 BW-mean (465d) ---")
X_combined = np.hstack([X_bw_binary, X_bw_mean])
all_results.extend(evaluate_features(X_combined, y_all, valid_families,
                                      "BW-binary + ESM-2 BW-mean (465d)"))

# COMBINED: BW-binary + ESM-2 BW-weighted
print("\n--- BW-binary + ESM-2 BW-weighted (465d) ---")
X_combined_w = np.hstack([X_bw_binary, X_bw_weighted])
all_results.extend(evaluate_features(X_combined_w, y_all, valid_families,
                                      "BW-binary + ESM-2 BW-weighted (465d)"))


# ===========================================================================
# Step 5: Save results
# ===========================================================================
print(f"\n{'='*72}")
print("Results Summary")
print(f"{'='*72}")

results_df = pd.DataFrame(all_results)
results_csv = os.path.join(RESULTS_DIR, "esm2_bwsite_results.csv")
results_df.to_csv(results_csv, index=False)
print(f"Saved: {results_csv}")

# Summary table
print(f"\n{'Feature Set':<50s} {'Model':<10s} {'AUC [95% CI]':<25s} {'PR-AUC':<8s} {'F1':<8s}")
print('-' * 105)
for _, r in results_df.sort_values(['model', 'AUC'], ascending=[True, False]).iterrows():
    print(f"{r['feature_set']:<50s} {r['model']:<10s} "
          f"{r['AUC']:.3f} [{r['AUC_lo']:.3f}-{r['AUC_hi']:.3f}]  "
          f"{r['PR_AUC']:.3f}    {r['F1']:.3f}")


# ===========================================================================
# Step 6: Generate comparison figure
# ===========================================================================
print(f"\n[Step 6] Generating comparison figure...")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Bar chart: AUC comparison (Ensemble model only)
ens_results = results_df[results_df['model'] == 'Ensemble'].sort_values('AUC', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))

colors = []
for _, r in ens_results.iterrows():
    fs = r['feature_set']
    if 'binary' in fs and 'ESM' not in fs:
        colors.append('#3498db')  # baseline blue
    elif 'mean-pool' in fs:
        colors.append('#95a5a6')  # gray for old ESM-2
    elif 'BW-mean' in fs and '+' not in fs:
        colors.append('#e67e22')  # orange for new
    elif 'BW-weighted' in fs and '+' not in fs:
        colors.append('#e74c3c')  # red for new weighted
    elif 'BW-flat' in fs:
        colors.append('#9b59b6')  # purple for PCA
    elif '+' in fs:
        colors.append('#2ecc71')  # green for combined
    else:
        colors.append('#34495e')

y_pos = range(len(ens_results))
bars = ax.barh(y_pos, ens_results['AUC'], color=colors, edgecolor='white', linewidth=0.5, height=0.7)

# Error bars
for i, (_, r) in enumerate(ens_results.iterrows()):
    ax.plot([r['AUC_lo'], r['AUC_hi']], [i, i], color='black', linewidth=1.5, zorder=5)
    ax.plot([r['AUC_lo']], [i], marker='|', color='black', markersize=8, zorder=5)
    ax.plot([r['AUC_hi']], [i], marker='|', color='black', markersize=8, zorder=5)

ax.set_yticks(y_pos)
ax.set_yticklabels(ens_results['feature_set'], fontsize=9)
ax.set_xlabel('AUC-ROC (Subfamily Split)', fontsize=12)
ax.set_title('ESM-2 BW-Site Embedding Comparison\n(Ensemble model, 95% Bootstrap CI)',
             fontsize=13, fontweight='bold')
ax.axvline(x=0.5, color='gray', linewidth=1, linestyle='--', label='Chance')
ax.set_xlim(0.3, 1.0)
ax.legend(loc='lower right')

plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "fig12_esm2_bwsite_comparison.png")
plt.savefig(fig_path, dpi=600, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_path}")

print(f"\n{'='*72}")
print("All done!")
print(f"{'='*72}")
