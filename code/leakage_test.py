#!/usr/bin/env python3
"""
leakage_test.py — Quick leakage diagnostic tool for GPCR coupling predictions.

Usage:
  python leakage_test.py                         # Run with built-in dataset
  python leakage_test.py --predictions user.csv  # Test user-supplied predictions

Accepts a CSV with columns: entry_name, predicted_probability
Outputs leakage degradation report comparing random vs no-leak splits.

Part of the GPCR coupling benchmarking protocol (BiB submission).
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def load_labels():
    """Load ground-truth Gq labels and subfamily assignments."""
    csv_path = os.path.join(DATA_DIR, "gpcrdb_coupling_dataset.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    if 'subfamily' not in df.columns:
        if 'family' in df.columns:
            df['subfamily'] = df['family'].astype(str).apply(
                lambda x: '_'.join(x.split('_')[:3]) if '_' in x else x
            )
        else:
            print("ERROR: dataset must contain either 'subfamily' or 'family' column.")
            sys.exit(1)
    return df[['entry_name', 'gq_label', 'subfamily']].copy()


def evaluate_split(y_true, y_pred, split_name):
    """Compute AUC and bootstrap CI."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    if len(np.unique(y_true)) < 2:
        return {'split': split_name, 'auc': float('nan'), 'pr_auc': float('nan')}

    auc = roc_auc_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)

    # Bootstrap 95% CI
    rng = np.random.RandomState(42)
    boot_aucs = []
    for _ in range(1000):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))

    ci_lo = np.percentile(boot_aucs, 2.5) if boot_aucs else float('nan')
    ci_hi = np.percentile(boot_aucs, 97.5) if boot_aucs else float('nan')

    return {
        'split': split_name,
        'auc': auc,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'pr_auc': pr_auc,
        'n_samples': len(y_true),
    }


def run_leakage_test(predictions_df=None):
    """
    Run the leakage diagnostic.

    If predictions_df is None, trains a default RF model internally.
    Otherwise, uses the supplied predicted probabilities.
    """
    labels_df = load_labels()

    if predictions_df is not None:
        # Merge user predictions with labels
        merged = labels_df.merge(predictions_df, on='entry_name', how='inner')
        if len(merged) == 0:
            print("ERROR: No matching entry_names between predictions and labels.")
            sys.exit(1)
        print(f"Matched {len(merged)} receptors.")
        y_true = merged['gq_label'].values
        y_pred = merged['predicted_probability'].values
        subfamilies = merged['subfamily'].values
    else:
        # Train default model with BW features
        print("No predictions supplied — training default BW-site RF model...")
        npz_path = os.path.join(DATA_DIR, "bw_site_features.npz")
        csv_path = os.path.join(DATA_DIR, "bw_site_feature_matrix.csv")

        if os.path.exists(csv_path):
            df_feat = pd.read_csv(csv_path)
            feature_cols = [c for c in df_feat.columns if c not in ['entry_name', 'gq_label', 'subfamily']]
            X = df_feat[feature_cols].values
            y_true = df_feat['gq_label'].values
            subfamilies = df_feat['subfamily'].values
        elif os.path.exists(npz_path):
            data = np.load(npz_path, allow_pickle=True)
            X = data['X']
            y_true = data['y']
            subfamilies = labels_df['subfamily'].values[:len(y_true)]
        else:
            # Fresh-clone fallback: build BW features directly from source files.
            try:
                from run_reviewer_analyses import load_bw_features
                X, y_true, _entry_names, subfamilies = load_bw_features()
            except Exception as exc:
                print("ERROR: No feature data found and BW feature construction failed.")
                print(f"Details: {exc}")
                print("Hint: run 'python code/run_reviewer_analyses.py' first.")
                sys.exit(1)

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        rf = RandomForestClassifier(n_estimators=500, max_depth=15,
                                     class_weight='balanced', random_state=42)
        rf.fit(X_scaled, y_true)
        y_pred = rf.predict_proba(X_scaled)[:, 1]  # In-sample for random
        # For proper evaluation we need cross-validated predictions
        from sklearn.model_selection import cross_val_predict, StratifiedKFold, GroupKFold
        y_pred_random = cross_val_predict(rf, X_scaled, y_true,
                                           cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                           method='predict_proba')[:, 1]

        y_pred_subfamily = np.full_like(y_true, np.nan, dtype=float)
        gkf = GroupKFold(n_splits=min(5, len(np.unique(subfamilies))))
        for train_idx, test_idx in gkf.split(X_scaled, y_true, groups=subfamilies):
            rf_fold = RandomForestClassifier(n_estimators=500, max_depth=15,
                                              class_weight='balanced', random_state=42)
            rf_fold.fit(X_scaled[train_idx], y_true[train_idx])
            y_pred_subfamily[test_idx] = rf_fold.predict_proba(X_scaled[test_idx])[:, 1]

        y_pred = y_pred_random  # Default for display

    # ============================================================
    # Evaluate under different splits
    # ============================================================
    print("\n" + "=" * 60)
    print("LEAKAGE DIAGNOSTIC REPORT")
    print("=" * 60)

    results = []

    if predictions_df is not None:
        # Can only evaluate overall (user doesn't control split)
        res = evaluate_split(y_true, y_pred, "Overall (user predictions)")
        results.append(res)
    else:
        # Random split evaluation
        res_random = evaluate_split(y_true, y_pred_random, "Random split (leaky)")
        results.append(res_random)

        # Subfamily split evaluation
        valid = ~np.isnan(y_pred_subfamily)
        res_sub = evaluate_split(y_true[valid], y_pred_subfamily[valid], "Subfamily split (no-leak)")
        results.append(res_sub)

    # Print report
    print(f"\n{'Split':<35} {'AUC-ROC':<12} {'95% CI':<20} {'PR-AUC':<10} {'N':<6}")
    print("-" * 83)
    for r in results:
        ci = f"[{r.get('ci_lo', 0):.3f}-{r.get('ci_hi', 0):.3f}]"
        print(f"{r['split']:<35} {r['auc']:<12.3f} {ci:<20} {r.get('pr_auc', 0):<10.3f} {r.get('n_samples', 0):<6}")

    if len(results) >= 2 and not np.isnan(results[0]['auc']) and not np.isnan(results[1]['auc']):
        delta = results[0]['auc'] - results[1]['auc']
        print(f"\n⚠ Leakage effect: ΔAUC = {delta:+.3f}")
        if delta > 0.1:
            print("  → SIGNIFICANT leakage detected! Random split inflates performance.")
            print("  → Use subfamily or sequence-identity split for reliable evaluation.")
        elif delta > 0.05:
            print("  → MODERATE leakage detected. Consider stricter splitting.")
        else:
            print("  → Minimal leakage. Model may genuinely generalize.")

    print("\n" + "=" * 60)
    return results


def main():
    parser = argparse.ArgumentParser(description="GPCR Coupling Leakage Diagnostic")
    parser.add_argument("--predictions", type=str, default=None,
                        help="CSV with columns: entry_name, predicted_probability")
    args = parser.parse_args()

    if args.predictions:
        pred_df = pd.read_csv(args.predictions)
        required = {'entry_name', 'predicted_probability'}
        if not required.issubset(pred_df.columns):
            print(f"ERROR: CSV must have columns: {required}")
            sys.exit(1)
        run_leakage_test(pred_df)
    else:
        run_leakage_test()


if __name__ == "__main__":
    main()
