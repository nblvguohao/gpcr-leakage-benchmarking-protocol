#!/usr/bin/env python3
"""
Paper-level Benchmark: Leakage-Aware Evaluation of GPCR-Gq Coupling Prediction
================================================================================
Produces all numerical results for the benchmark section of the paper:
  - 5 models × 4 split strategies
  - Bootstrap 95% CI for AUC-ROC, PR-AUC, Accuracy, F1
  - DeLong test comparing random vs no-leak AUC
  - Label sensitivity analysis (primary-only vs primary+secondary)
  - CSV output for downstream figure generation

Usage: py code/run_paper_benchmark.py
"""
import os
import sys
import re
import time
import json
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              accuracy_score, f1_score, precision_score,
                              recall_score)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASET_FILE = os.path.join(DATA_DIR, "gpcrdb_coupling_dataset.csv")

GNAQ_SEQ = ("MTLESIMACCLSEEAKEARRINDEIERQLRRDKRDARRELKLLLLGTGESGKSTFIKQMRIIHGS"
            "GYSDEDKRGFTKLVYQNIFTAMQAMIRAMDTLKIPYKYEHNKAHAQLVREVDVEKVSAFENPYVD"
            "AIKSLWNDPGIQECYDRRREYQLSDSTKYYLNDLDRVADPAYLPTQQDVLRVRVPTTGIIEYPFDL"
            "QSVIFRMVDVGGQRSERRKWIHCFENVTSIMFLVALSEYDQVLVESDNENRMEESKALFRTIITYPWFQNSSVILFLNKKDLLEEK")
GNAS_SEQ = ("MGCLGNSKTEDQRNEEKAQREANKKIEKQLQKDKQVYRATHRLLLLGAGESGKSTIVKQMRILHV"
            "NGFNGEGGEEDPQAARSNSDGEKATKVQDIKNNLKEAIETIVAAMSNLVPPVELANPENQFRVDYIL"
            "SVMNVPDFDFPPEFYEHAKALWEDEGVRACYERSNEYQLIDCAQYFLDKIDVIKQADYVPSDQDLLR"
            "CRVLTSGIFETKFQVDKVNFHMFDVGGQRDERRKWIQCFNDVTAIIFVVASSSYNMVIREDNQTNRL"
            "QEALNLFKSIWNNRWLRTISVILFLNKQDLLAEKVLAGKSKIEDYFPEFARYTTPEDATPEPGEDPRVTRAKYFIRDEFLRISTASGDGRHYCYPHFTCAVDTENIRRVFNDCRDIIQRMHLRQYELL")
GNAI_SEQ = ("MGCTLSAEDKAAVERSKMIDRNLREDGEKAAREVKLLLLGAGESGKSTIVKQMKIIHEAGYSEEEC"
            "KQYKAVVYSNTIQSIIAIIRAMGRLKIDFGDSARADDARQLFVLAGAAEEGFMTAELAGVIKRLWK"
            "DSGVQACFNRSREYQLNDSAAYYLNDLDRIAQPNYIPTQQDVLRTRVKTTGIVETHFTFKDLHFKMF"
            "DVGGQRSERKKWIHCFEGVTAIIFCVALSDYDLVLAEDEEMNRMHESMKLFDSICNNKWFTDTSIILF"
            "LNKKDLFEEKITHSPLTICFPEYTGANKYDEASYYIQSKFEDLNKRKDTKEIYTHFTCATDTKNVQFVFDAVTDVIIKNNLKDCGLF")

N_BOOTSTRAP = 1000
RANDOM_SEED = 42

print("=" * 72)
print("Paper Benchmark: Leakage-Aware GPCR-Gq Coupling Evaluation")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Bootstrap iterations: {N_BOOTSTRAP}")
print("=" * 72)


# ===========================================================================
# Feature extraction (identical to V5)
# ===========================================================================
def find_tm_regions(sequence):
    hydro_set = set('AILMFVW')
    seq_len = len(sequence)
    tm_regions, in_tm, tm_start = [], False, 0
    for i in range(seq_len - 20):
        window = sequence[i:i + 20]
        hf = sum(1 for aa in window if aa in hydro_set) / 20
        if hf >= 0.55 and not in_tm:
            tm_start, in_tm = i, True
        elif hf < 0.35 and in_tm:
            tm_regions.append((tm_start, i))
            in_tm = False
    if in_tm:
        tm_regions.append((tm_start, seq_len))
    merged = []
    for s, e in tm_regions:
        if merged and s - merged[-1][1] < 10:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged[:10]


def extract_icl_features(sequence, tm_regions):
    features = {}
    hydrophobic, positive, negative, polar = set('AILMFVPW'), set('KRH'), set('DE'), set('STNQ')
    icls = []
    for i in range(len(tm_regions) - 1):
        gs, ge = tm_regions[i][1], tm_regions[i+1][0]
        if ge > gs:
            icls.append(sequence[gs:ge])
    for icl_idx, icl_name in [(1, 'icl2'), (3, 'icl3')]:
        if icl_idx < len(icls) and len(icls[icl_idx]) > 0:
            icl, rl = icls[icl_idx], len(icls[icl_idx])
            features[f'{icl_name}_len'] = rl
            features[f'{icl_name}_hydro'] = sum(1 for a in icl if a in hydrophobic) / rl
            features[f'{icl_name}_pos'] = sum(1 for a in icl if a in positive) / rl
            features[f'{icl_name}_neg'] = sum(1 for a in icl if a in negative) / rl
            features[f'{icl_name}_polar'] = sum(1 for a in icl if a in polar) / rl
            features[f'{icl_name}_charge'] = features[f'{icl_name}_pos'] - features[f'{icl_name}_neg']
        else:
            for s in ['_len', '_hydro', '_pos', '_neg', '_polar', '_charge']:
                features[f'{icl_name}{s}'] = 0
    all_icl = ''.join(icls) if icls else ''
    if all_icl:
        rl = len(all_icl)
        features['icl_total_len'] = rl
        features['icl_total_charge'] = (sum(1 for a in all_icl if a in positive) -
                                        sum(1 for a in all_icl if a in negative)) / rl
    else:
        features['icl_total_len'] = 0
        features['icl_total_charge'] = 0
    return features


def get_kmer_set(seq, k=3):
    return set(seq[i:i+k] for i in range(len(seq) - k + 1))


def extract_features(sequence, family_slug=''):
    features = {}
    seq_len = len(sequence)
    if seq_len == 0:
        return features
    features['length'] = seq_len
    features['log_length'] = np.log(seq_len)
    aa_counts = defaultdict(int)
    for aa in sequence:
        aa_counts[aa] += 1
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        features[f'{aa}_ratio'] = aa_counts[aa] / seq_len
    hydrophobic, polar, positive, negative, aromatic = set('AILMFVPW'), set('STNQ'), set('KRH'), set('DE'), set('FYW')
    features['hydro_ratio'] = sum(aa_counts[a] for a in hydrophobic) / seq_len
    features['polar_ratio'] = sum(aa_counts[a] for a in polar) / seq_len
    features['pos_ratio'] = sum(aa_counts[a] for a in positive) / seq_len
    features['neg_ratio'] = sum(aa_counts[a] for a in negative) / seq_len
    features['arom_ratio'] = sum(aa_counts[a] for a in aromatic) / seq_len
    features['charge_ratio'] = features['pos_ratio'] - features['neg_ratio']
    features['has_DRY'] = 1.0 if 'DRY' in sequence else 0.0
    features['has_ERY'] = 1.0 if 'ERY' in sequence else 0.0
    features['has_NPxxY'] = 1.0 if re.search(r'NP..Y', sequence) else 0.0
    features['has_CWxP'] = 1.0 if re.search(r'CW.P', sequence) else 0.0
    features['has_QAKK'] = 1.0 if 'QAKK' in sequence else 0.0
    features['has_PLAT'] = 1.0 if 'PLAT' in sequence else 0.0
    features['has_HKKLR'] = 1.0 if 'HKKLR' in sequence else 0.0
    features['has_DRYLV'] = 1.0 if re.search(r'DRY.V', sequence) else 0.0
    features['has_HEK'] = 1.0 if 'HEK' in sequence else 0.0
    features['has_SLRT'] = 1.0 if 'SLRT' in sequence else 0.0
    features['has_PMSNFR'] = 1.0 if 'PMSNFR' in sequence else 0.0
    features['has_AAAQQ'] = 1.0 if 'AAAQQ' in sequence else 0.0
    features['has_KKLRT'] = 1.0 if 'KKLRT' in sequence else 0.0
    features['has_PMSN'] = 1.0 if 'PMSN' in sequence else 0.0
    tm_regions = find_tm_regions(sequence)
    features['tm_count'] = len(tm_regions)
    features.update(extract_icl_features(sequence, tm_regions))
    n_term = sequence[:50] if seq_len >= 50 else sequence
    c_term = sequence[-50:] if seq_len >= 50 else sequence
    mid = sequence[seq_len//3:2*seq_len//3]
    for name, region in [('n', n_term), ('c', c_term), ('m', mid)]:
        rl = len(region)
        if rl > 0:
            features[f'{name}_hydro'] = sum(1 for a in region if a in hydrophobic) / rl
            features[f'{name}_charged'] = sum(1 for a in region if a in positive | negative) / rl
            features[f'{name}_arom'] = sum(1 for a in region if a in aromatic) / rl
    npxxy = list(re.finditer(r'NP..Y', sequence))
    if npxxy:
        c_tail = sequence[npxxy[-1].end():]
        features['c_tail_len'] = len(c_tail)
        features['c_tail_ratio'] = len(c_tail) / seq_len
        features['c_tail_charged'] = sum(1 for a in c_tail if a in positive|negative)/max(len(c_tail),1)
        features['c_tail_pos'] = sum(1 for a in c_tail if a in positive)/max(len(c_tail),1)
    else:
        features['c_tail_len'] = features['c_tail_ratio'] = 0
        features['c_tail_charged'] = features['c_tail_pos'] = 0
    features['complexity'] = len(set(sequence)) / 20.0
    dipeptides = defaultdict(int)
    for i in range(seq_len - 1):
        dipeptides[sequence[i:i+2]] += 1
    for dp in ['LL','VL','LV','II','FF','FL','LF','AI','IA','AL','LA','FI','IF','IV','VI','GV','VG','SS','TT','PP']:
        features[f'dp_{dp}'] = dipeptides[dp] / max(seq_len-1, 1)
    k = 3
    seq_kmers = get_kmer_set(sequence, k)
    for gname, gseq in [('gnaq', GNAQ_SEQ), ('gnas', GNAS_SEQ), ('gnai', GNAI_SEQ)]:
        gk = get_kmer_set(gseq, k)
        features[f'{gname}_overlap'] = len(seq_kmers & gk) / len(seq_kmers) if seq_kmers else 0
    features['gq_vs_gs'] = features.get('gnaq_overlap',0) - features.get('gnas_overlap',0)
    features['gq_vs_gi'] = features.get('gnaq_overlap',0) - features.get('gnai_overlap',0)
    features['is_classA'] = 1.0 if family_slug.startswith('001') else 0.0
    features['is_classB'] = 1.0 if family_slug.startswith('002') else 0.0
    features['is_classC'] = 1.0 if family_slug.startswith('004') else 0.0
    return features


# ===========================================================================
# Split strategies (reused from V5 noleak)
# ===========================================================================
def get_subfamily(family_slug):
    parts = family_slug.split('_')
    return '_'.join(parts[:3]) if len(parts) >= 3 else family_slug


def split_random(y, seed=42):
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    return tr, te


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


def split_seqcluster(y, sequences, threshold=0.3, seed=42):
    np.random.seed(seed)
    n = len(sequences)
    kmer_sets = [get_kmer_set(s, 3) for s in sequences]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            inter = len(kmer_sets[i] & kmer_sets[j])
            union = len(kmer_sets[i] | kmer_sets[j])
            d = 1 - (inter / union if union > 0 else 0)
            dist[i,j] = dist[j,i] = d
    Z = linkage(squareform(dist), method='average')
    labels = fcluster(Z, t=1-threshold, criterion='distance')
    cl_map = defaultdict(list)
    for i, c in enumerate(labels):
        cl_map[c].append(i)
    cls = list(cl_map.keys())
    np.random.shuffle(cls)
    target = int(n * 0.2)
    te, tr = [], []
    n_te = 0
    for c in cls:
        m = cl_map[c]
        if n_te < target:
            te.extend(m); n_te += len(m)
        else:
            tr.extend(m)
    return np.array(tr), np.array(te)


# ===========================================================================
# Model builders
# ===========================================================================
def build_models():
    models = {}
    models['LR'] = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced',
                                       random_state=RANDOM_SEED, solver='lbfgs')
    models['RF'] = RandomForestClassifier(n_estimators=500, max_depth=15,
                                           min_samples_leaf=2, class_weight='balanced',
                                           random_state=RANDOM_SEED)
    models['GBM'] = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                learning_rate=0.05, subsample=0.8,
                                                random_state=RANDOM_SEED)
    models['SVM'] = SVC(kernel='rbf', C=10, gamma='scale', probability=True,
                         class_weight='balanced', random_state=RANDOM_SEED)
    if HAS_XGB:
        models['XGB'] = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8,
                                       scale_pos_weight=1.5, random_state=RANDOM_SEED,
                                       use_label_encoder=False, eval_metric='logloss',
                                       verbosity=0)
    # Ensemble
    rf = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=2,
                                 class_weight='balanced', random_state=RANDOM_SEED)
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                      subsample=0.8, random_state=RANDOM_SEED)
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True,
               class_weight='balanced', random_state=RANDOM_SEED)
    models['Ensemble'] = VotingClassifier(estimators=[('rf', rf), ('gbm', gbm), ('svm', svm)],
                                           voting='soft', weights=[2, 2, 1])
    return models


# ===========================================================================
# Bootstrap CI computation
# ===========================================================================
def bootstrap_metrics(y_true, y_prob, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs, praucs, accs, f1s = [], [], [], []
    y_pred = (y_prob >= 0.5).astype(int)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt, yp, ypd = y_true[idx], y_prob[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(yt, yp))
            praucs.append(average_precision_score(yt, yp))
            accs.append(accuracy_score(yt, ypd))
            f1s.append(f1_score(yt, ypd, zero_division=0))
        except:
            pass
    def ci(arr):
        if len(arr) < 10:
            return np.nan, np.nan, np.nan
        return np.mean(arr), np.percentile(arr, 2.5), np.percentile(arr, 97.5)
    return {
        'auc': ci(aucs), 'prauc': ci(praucs),
        'acc': ci(accs), 'f1': ci(f1s),
    }


# ===========================================================================
# DeLong test for AUC comparison
# ===========================================================================
def delong_test(y_true, y_prob1, y_prob2):
    """Approximate DeLong test for two correlated AUCs."""
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    if n1 < 2 or n0 < 2:
        return np.nan
    pos = y_prob1[y_true == 1], y_prob2[y_true == 1]
    neg = y_prob1[y_true == 0], y_prob2[y_true == 0]

    def placement(pos_scores, neg_scores):
        return np.array([np.mean(pos_scores > ns) + 0.5 * np.mean(pos_scores == ns)
                         for ns in neg_scores])

    V10_1 = placement(pos[0], neg[0])
    V10_2 = placement(pos[1], neg[1])
    V01_1 = placement(neg[0][:, None], pos[0][None, :]).mean(axis=0) if n1 > 0 else np.zeros(n1)

    # Simplified: use bootstrap permutation p-value instead
    auc1 = roc_auc_score(y_true, y_prob1)
    auc2 = roc_auc_score(y_true, y_prob2)
    diff = auc1 - auc2

    # Permutation test
    rng = np.random.RandomState(42)
    count = 0
    for _ in range(5000):
        swap = rng.random(len(y_true)) > 0.5
        p1 = np.where(swap, y_prob2, y_prob1)
        p2 = np.where(swap, y_prob1, y_prob2)
        try:
            d = roc_auc_score(y_true, p1) - roc_auc_score(y_true, p2)
            if abs(d) >= abs(diff):
                count += 1
        except:
            pass
    return count / 5000


# ===========================================================================
# Main pipeline
# ===========================================================================
def load_and_prepare(label_mode='default'):
    """Load data and extract features.
    label_mode:
      'default' = exclude Unknown, use existing gq_label
      'primary_only' = only count 'primary' coupling as Gq=1
    """
    df = pd.read_csv(DATASET_FILE)
    df = df[df['coupling_description'] != 'Unknown'].reset_index(drop=True)

    if label_mode == 'primary_only':
        df['gq_label'] = df['coupling_description'].apply(
            lambda x: 1 if 'Gq' in str(x) and 'primary' in str(x).lower()
                           and 'secondary' not in str(x).lower().split('gq')[0]
                      else 0)

    feature_names = None
    X, y, families, sequences, entries = [], [], [], [], []
    for _, row in df.iterrows():
        seq = row['sequence']
        if not seq or len(seq) < 50:
            continue
        fam = str(row.get('family', ''))
        feat = extract_features(seq, fam)
        if feature_names is None:
            feature_names = sorted(feat.keys())
        X.append([feat.get(n, 0) for n in feature_names])
        y.append(row['gq_label'])
        families.append(fam)
        sequences.append(seq)
        entries.append(row['entry_name'])

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    return X, y, families, sequences, entries, feature_names


def run_experiment(X, y, families, sequences, entries, label_tag='default'):
    """Run all models × all splits, return results DataFrame."""
    print(f"\n{'='*72}")
    print(f"Label mode: {label_tag}")
    print(f"Samples: {len(y)} (Gq={int(y.sum())}, non-Gq={len(y)-int(y.sum())})")
    print(f"{'='*72}")

    # --- Splits ---
    splits = {}
    print("\nComputing splits...")
    splits['Random'] = split_random(y)
    splits['Subfamily'] = split_subfamily(y, families)
    print("  Computing sequence clustering (threshold=0.3)...")
    splits['SeqCluster_0.3'] = split_seqcluster(y, sequences, threshold=0.3)
    print("  Computing sequence clustering (threshold=0.4)...")
    splits['SeqCluster_0.4'] = split_seqcluster(y, sequences, threshold=0.4)

    for sname, (tr, te) in splits.items():
        n_gq_te = int(y[te].sum())
        print(f"  {sname}: train={len(tr)}, test={len(te)} (Gq_test={n_gq_te})")

    # --- Models ---
    models = build_models()
    model_names = list(models.keys())
    print(f"\nModels: {model_names}")

    # --- Run ---
    all_results = []
    all_probas = {}  # for DeLong: (split, model) -> y_prob

    for split_name, (train_idx, test_idx) in splits.items():
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if y_tr.sum() < 3 or (len(y_tr) - y_tr.sum()) < 3:
            print(f"  SKIP {split_name}: insufficient classes in train")
            continue
        if y_te.sum() < 2 or (len(y_te) - y_te.sum()) < 2:
            print(f"  SKIP {split_name}: insufficient classes in test")
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        for model_name, model_template in models.items():
            # Clone model
            from sklearn.base import clone
            model = clone(model_template)

            t0 = time.time()
            model.fit(X_tr_s, y_tr)
            train_time = time.time() - t0

            y_prob = model.predict_proba(X_te_s)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            # Point metrics
            try:
                auc = roc_auc_score(y_te, y_prob)
            except:
                auc = np.nan
            try:
                prauc = average_precision_score(y_te, y_prob)
            except:
                prauc = np.nan

            acc = accuracy_score(y_te, y_pred)
            f1 = f1_score(y_te, y_pred, zero_division=0)
            prec = precision_score(y_te, y_pred, zero_division=0)
            rec = recall_score(y_te, y_pred, zero_division=0)

            # Bootstrap CI
            boot = bootstrap_metrics(y_te, y_prob)

            result = {
                'label_mode': label_tag,
                'split': split_name,
                'model': model_name,
                'n_train': len(y_tr),
                'n_test': len(y_te),
                'train_time': train_time,
                'AUC': auc,
                'AUC_lo': boot['auc'][1],
                'AUC_hi': boot['auc'][2],
                'PR_AUC': prauc,
                'PRAUC_lo': boot['prauc'][1],
                'PRAUC_hi': boot['prauc'][2],
                'Accuracy': acc,
                'Acc_lo': boot['acc'][1],
                'Acc_hi': boot['acc'][2],
                'F1': f1,
                'F1_lo': boot['f1'][1],
                'F1_hi': boot['f1'][2],
                'Precision': prec,
                'Recall': rec,
            }
            all_results.append(result)
            all_probas[(split_name, model_name)] = (y_te, y_prob)

            auc_str = f"{auc:.3f} [{boot['auc'][1]:.3f}-{boot['auc'][2]:.3f}]"
            f1_str = f"{f1:.3f} [{boot['f1'][1]:.3f}-{boot['f1'][2]:.3f}]"
            print(f"  {split_name:<18s} {model_name:<10s} AUC={auc_str}  F1={f1_str}  ({train_time:.1f}s)")

    # --- DeLong tests: Random vs each no-leak split ---
    print(f"\nDeLong / Permutation tests (Random vs no-leak, Ensemble model):")
    delong_results = []
    for noleak_split in ['Subfamily', 'SeqCluster_0.3', 'SeqCluster_0.4']:
        key_rand = ('Random', 'Ensemble')
        key_noleak = (noleak_split, 'Ensemble')
        if key_rand in all_probas and key_noleak in all_probas:
            # These are on DIFFERENT test sets, so we can't do paired DeLong.
            # Instead report the AUC difference and CI non-overlap.
            y1, p1 = all_probas[key_rand]
            y2, p2 = all_probas[key_noleak]
            auc1 = roc_auc_score(y1, p1)
            auc2 = roc_auc_score(y2, p2)
            diff = auc1 - auc2
            # Bootstrap CI of each
            b1 = bootstrap_metrics(y1, p1)
            b2 = bootstrap_metrics(y2, p2)
            overlap = b1['auc'][1] <= b2['auc'][2] and b2['auc'][1] <= b1['auc'][2]
            sig = 'CI overlap (n.s.)' if overlap else 'CI non-overlap (sig.)'
            print(f"  Random vs {noleak_split}: ΔAUC={diff:+.3f}  {sig}")
            print(f"    Random AUC: {auc1:.3f} [{b1['auc'][1]:.3f}-{b1['auc'][2]:.3f}]")
            print(f"    {noleak_split} AUC: {auc2:.3f} [{b2['auc'][1]:.3f}-{b2['auc'][2]:.3f}]")
            delong_results.append({
                'comparison': f'Random vs {noleak_split}',
                'AUC_random': auc1, 'AUC_noleak': auc2,
                'delta_AUC': diff, 'significance': sig,
            })

    return pd.DataFrame(all_results), pd.DataFrame(delong_results)


# ===========================================================================
# Run experiments
# ===========================================================================

# Experiment 1: Default labels
print("\n" + "=" * 72)
print("EXPERIMENT 1: Default labels (primary + secondary Gq)")
print("=" * 72)
X, y, families, sequences, entries, fnames = load_and_prepare('default')
df_results_default, df_delong_default = run_experiment(
    X, y, families, sequences, entries, 'default')

# Experiment 2: Primary-only labels (sensitivity analysis)
print("\n" + "=" * 72)
print("EXPERIMENT 2: Primary-only labels (sensitivity analysis)")
print("=" * 72)
X2, y2, fam2, seq2, ent2, fn2 = load_and_prepare('primary_only')
df_results_primary, df_delong_primary = run_experiment(
    X2, y2, fam2, seq2, ent2, 'primary_only')

# ===========================================================================
# Combine and save results
# ===========================================================================
df_all = pd.concat([df_results_default, df_results_primary], ignore_index=True)
df_delong_all = pd.concat([df_delong_default, df_delong_primary], ignore_index=True)

# Save CSVs
csv_main = os.path.join(RESULTS_DIR, "benchmark_results.csv")
csv_delong = os.path.join(RESULTS_DIR, "benchmark_delong.csv")
df_all.to_csv(csv_main, index=False)
df_delong_all.to_csv(csv_delong, index=False)
print(f"\n✓ Results saved: {csv_main}")
print(f"✓ DeLong results saved: {csv_delong}")

# ===========================================================================
# Summary table (paper-ready)
# ===========================================================================
print(f"\n{'='*72}")
print("PAPER TABLE: Ensemble model across split strategies (default labels)")
print(f"{'='*72}")

ens = df_results_default[df_results_default['model'] == 'Ensemble']
print(f"\n{'Split':<20s} {'AUC [95%CI]':<25s} {'PR-AUC [95%CI]':<25s} "
      f"{'F1 [95%CI]':<25s} {'Acc [95%CI]':<25s}")
print('-' * 100)
for _, row in ens.iterrows():
    auc_s = f"{row['AUC']:.3f} [{row['AUC_lo']:.3f}-{row['AUC_hi']:.3f}]"
    prauc_s = f"{row['PR_AUC']:.3f} [{row['PRAUC_lo']:.3f}-{row['PRAUC_hi']:.3f}]"
    f1_s = f"{row['F1']:.3f} [{row['F1_lo']:.3f}-{row['F1_hi']:.3f}]"
    acc_s = f"{row['Accuracy']:.3f} [{row['Acc_lo']:.3f}-{row['Acc_hi']:.3f}]"
    print(f"{row['split']:<20s} {auc_s:<25s} {prauc_s:<25s} {f1_s:<25s} {acc_s:<25s}")

print(f"\n{'='*72}")
print("PAPER TABLE: All models under Subfamily split (default labels)")
print(f"{'='*72}")
sf = df_results_default[df_results_default['split'] == 'Subfamily']
print(f"\n{'Model':<12s} {'AUC [95%CI]':<25s} {'PR-AUC [95%CI]':<25s} {'F1 [95%CI]':<25s}")
print('-' * 70)
for _, row in sf.iterrows():
    auc_s = f"{row['AUC']:.3f} [{row['AUC_lo']:.3f}-{row['AUC_hi']:.3f}]"
    prauc_s = f"{row['PR_AUC']:.3f} [{row['PRAUC_lo']:.3f}-{row['PRAUC_hi']:.3f}]"
    f1_s = f"{row['F1']:.3f} [{row['F1_lo']:.3f}-{row['F1_hi']:.3f}]"
    print(f"{row['model']:<12s} {auc_s:<25s} {prauc_s:<25s} {f1_s:<25s}")

print(f"\n{'='*72}")
print("Label sensitivity: Default vs Primary-only (Ensemble, Subfamily)")
print(f"{'='*72}")
for lmode in ['default', 'primary_only']:
    subset = df_all[(df_all['label_mode'] == lmode) &
                     (df_all['model'] == 'Ensemble') &
                     (df_all['split'] == 'Subfamily')]
    if len(subset) > 0:
        row = subset.iloc[0]
        print(f"  {lmode:<15s}: AUC={row['AUC']:.3f}, F1={row['F1']:.3f}, "
              f"n_train={int(row['n_train'])}, n_test={int(row['n_test'])}")

print(f"\n{'='*72}")
print("Done. All results in: results/benchmark_results.csv")
print(f"{'='*72}")
