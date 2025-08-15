import numpy as np
import pandas as pd
import pickle
import json
import os
import time
import argparse
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# This local import works because we run the script from its own directory
from kitnet_model import KitNET as kit

def extract_rmse_vectors(X, kitnet_model):
    """Helper function to extract RMSE vectors from a dataset using a trained KitNET model."""
    print(f"Extracting RMSE vectors for {X.shape[0]} samples...")
    rmse_vectors = []
    for i in tqdm(range(X.shape[0]), desc="RMSE Extraction"):
        rmse_vec = kitnet_model.extract_rmse_vector(X[i])
        rmse_vectors.append(rmse_vec)
    return np.array(rmse_vectors)

def main(args):
    """Main function to train the full KitNET + RandomForest pipeline."""
    print("--- KitNET and RandomForest Training Pipeline ---")

    # --- Automatic Directory Path Generation ---
    if args.base_dir is None:
        hyperparam_tag = f"AE{args.max_autoencoder_size}_FM{args.fm_grace}_AD{args.ad_grace}"
        base_dir = os.path.join(args.output_root, 'kitnet', args.dataset_tag, hyperparam_tag)
        print(f"Output directory not specified, automatically setting to: {base_dir}")
    else:
        base_dir = args.base_dir
        print(f"Using specified output directory: {base_dir}")
    # -----------------------------------------

    os.makedirs(base_dir, exist_ok=True)

    # ====================================================================
    # PART 1: Train the KitNET Autoencoder Feature Extractor
    # ====================================================================
    print("\n--- Part 1: Training KitNET Feature Extractor ---")
    kitnet_data_path = os.path.join(args.data_dir, 'kitnet', 'X_train_b_test_bm.csv')
    print(f"Loading KitNET training data from: {kitnet_data_path}")
    X_kitnet_train = pd.read_csv(kitnet_data_path, header=None).values

    # Build KitNET instance
    kitnet_model = kit.KitNET(n=X_kitnet_train.shape[1],
                              max_autoencoder_size=args.max_autoencoder_size,
                              FM_grace_period=args.fm_grace,
                              AD_grace_period=args.ad_grace,
                              learning_rate=args.learning_rate,
                              hidden_ratio=args.hidden_ratio)
    
    print("Training KitNET model (this may take a while)...")
    start_time = time.time()
    for i in tqdm(range(X_kitnet_train.shape[0]), desc="KitNET Training"):
        kitnet_model.process(X_kitnet_train[i])
    end_time = time.time()
    print(f"KitNET training complete. Time elapsed: {end_time - start_time:.2f} seconds")

    # Save the trained KitNET model
    kitnet_model_path = os.path.join(base_dir, "kitnet_model.pkl")
    with open(kitnet_model_path, "wb") as f:
        pickle.dump(kitnet_model, f)
    print(f"Trained KitNET model saved to: {kitnet_model_path}")

    # ====================================================================
    # PART 2: Train the RandomForest Classifier on extracted features
    # ====================================================================
    print("\n--- Part 2: Training RandomForest Classifier ---")
    print(f"Loading standard train/test data from: {args.data_dir}")
    X_train = np.load(os.path.join(args.data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(args.data_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'))

    # Extract RMSE features using the newly trained KitNET model
    X_train_rmse = extract_rmse_vectors(X_train, kitnet_model)
    X_test_rmse = extract_rmse_vectors(X_test, kitnet_model)

    print("Training RandomForest classifier...")
    rf_classifier = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state, n_jobs=-1)
    rf_classifier.fit(X_train_rmse, y_train)

    # Save the trained RF model
    rf_model_path = os.path.join(base_dir, "rf_model.pkl")
    with open(rf_model_path, "wb") as f:
        pickle.dump(rf_classifier, f)
    print(f"Trained RandomForest model saved to: {rf_model_path}")

    # Evaluate the RF model
    print("Evaluating RandomForest classifier on the test set...")
    y_pred = rf_classifier.predict(X_test_rmse)
    y_prob = rf_classifier.predict_proba(X_test_rmse)[:, 1]

    report = classification_report(y_test, y_pred, digits=4)
    print("\n--- Classification Report ---")
    print(report)
    
    # Save the full report to a file
    report_path = os.path.join(base_dir, "rf_evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("=== RandomForest Evaluation Report on KitNET Features ===\n\n")
        f.write(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}\n")
        f.write(f"Precision: {precision_score(y_test, y_pred):.4f}\n")
        f.write(f"Recall   : {recall_score(y_test, y_pred):.4f}\n")
        f.write(f"F1 Score : {f1_score(y_test, y_pred):.4f}\n")
        f.write(f"AUC      : {roc_auc_score(y_test, y_prob):.4f}\n\n")
        f.write(report)
    print(f"Evaluation report saved to: {report_path}")
    print("--- Script Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the complete KitNET + RandomForest pipeline.')
    # Path and Directory Arguments
    parser.add_argument('--data_dir', type=str, default='../../data/2018/brute_force', help='Directory for the .npy data files.')
    parser.add_argument('--base_dir', type=str, default=None, help='(Optional) Override output directory. Auto-generated if not set.')
    parser.add_argument('--output_root', type=str, default='../../checkpoints', help='Root directory for saving outputs.')
    parser.add_argument('--dataset_tag', type=str, default='2018/brute_force', help='Tag for the dataset version.')
    # KitNET Hyperparameters
    parser.add_argument('--max_autoencoder_size', type=int, default=10, help='Maximum size of any autoencoder in the ensemble layer.')
    parser.add_argument('--fm_grace', type=int, default=5000, help='Feature mapping grace period.')
    parser.add_argument('--ad_grace', type=int, default=25000, help='Anomaly detector grace period.')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for autoencoders.')
    parser.add_argument('--hidden_ratio', type=float, default=0.75, help='Ratio of hidden to visible neurons.')
    # RandomForest Hyperparameters
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the RandomForest.')
    parser.add_argument('--random_state', type=int, default=42, help='Seed for reproducibility.')
    
    args = parser.parse_args()
    main(args)