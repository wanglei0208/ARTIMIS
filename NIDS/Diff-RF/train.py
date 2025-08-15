import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
# This local import works because we run the script from its own directory
from model_diff_rf import DiFF_TreeEnsemble
import os
import json
import argparse
from tqdm import tqdm

def main(args):
    """Main function to train and evaluate a DiFF-RF model."""
    print("--- DiFF-RF Training and Evaluation ---")
    
    # --- Automatic Directory Path Generation ---
    if args.base_dir is None:
        hyperparam_tag = f"n{args.n_trees}_s{args.sample_size}"
        base_dir = os.path.join(args.output_root, 'Diff-RF', args.dataset_tag, hyperparam_tag)
        print(f"Output directory not specified, automatically setting to: {base_dir}")
    else:
        base_dir = args.base_dir
        print(f"Using specified output directory: {base_dir}")
    # -----------------------------------------

    os.makedirs(base_dir, exist_ok=True)

    print(f"Loading data from: {args.data_dir}")
    X_train_benign = np.load(os.path.join(args.data_dir, 'X_train_benign.npy'))
    X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'))
    print(f"Benign training data shape: {X_train_benign.shape}")
    print(f"Test data shape: {X_test.shape}")

    model = DiFF_TreeEnsemble(
        n_trees=args.n_trees,
        sample_size=args.sample_size,
    )
    
    # --- FIX: Handle the n_jobs = -1 case for multiprocessing.Pool ---
    n_jobs = args.n_jobs if args.n_jobs > 0 else None
    print(f"Training DiFF-RF model with {args.n_trees} trees and sample size {args.sample_size}...")
    # Pass the corrected n_jobs value to the fit method
    model.fit(X_train_benign, n_jobs=n_jobs)
    # -----------------------------------------------------------------

    print("Calculating anomaly scores for the test set...")
    _, _, scores = model.anomaly_score(X_test)

    print("Searching for the optimal F1-score threshold...")
    best_f1, best_thresh = -1.0, 0.0
    threshold_candidates = np.linspace(np.min(scores), np.max(scores), 100)
    for thresh in tqdm(threshold_candidates, desc="Threshold Search"):
        preds = (scores > thresh).astype(int).reshape(-1)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print("\nEvaluating model with the best threshold...")
    final_preds = (scores > best_thresh).astype(int).reshape(-1)
    
    final_metrics = {
        "best_threshold": float(best_thresh),
        "accuracy": accuracy_score(y_test, final_preds),
        "precision": precision_score(y_test, final_preds, zero_division=0),
        "recall": recall_score(y_test, final_preds, zero_division=0),
        "f1_score": f1_score(y_test, final_preds, zero_division=0),
        "auc": roc_auc_score(y_test, scores)
    }

    print(f"Best Threshold found: {final_metrics['best_threshold']:.4f}")
    for key, value in final_metrics.items():
        if key != "best_threshold":
            print(f"{key.replace('_', ' ').title()}: {value:.4f}")

    model_path = os.path.join(base_dir, "diff_rf_model.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    threshold_path = os.path.join(base_dir, "diff_rf_threshold.txt")
    with open(threshold_path, "w") as f:
        f.write(str(best_thresh))
    print(f"Threshold saved to: {threshold_path}")

    metrics_path = os.path.join(base_dir, "metrics_diff_rf.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")
    print("--- Script Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a DiFF-RF anomaly detection model.')
    parser.add_argument('--data_dir', type=str, default='../../data/2018/brute_force', help='Directory containing the .npy data files.')
    parser.add_argument('--base_dir', type=str, default=None, help='(Optional) Override output directory. Auto-generated if not set.')
    parser.add_argument('--output_root', type=str, default='../../checkpoints', help='Root directory for saving outputs.')
    parser.add_argument('--dataset_tag', type=str, default='2018/brute_force', help='Tag for the dataset version.')
    parser.add_argument('--n_trees', type=int, default=10, help='The number of trees in the forest.')
    parser.add_argument('--sample_size', type=int, default=512, help='The size of the random subsample of items to grow each tree.')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs to run for fitting. -1 means using all available processors.')
    
    args = parser.parse_args()
    main(args)