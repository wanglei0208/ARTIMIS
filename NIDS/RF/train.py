import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import joblib
import os
from tqdm import tqdm
import argparse

def main(args):
    """Main function to train and evaluate a Random Forest model."""
    print("--- Random Forest Training and Evaluation ---")
    
    # --- Automatic Directory Path Generation ---
    if args.base_dir is None:
        # Create a descriptive name, e.g., ../../checkpoints/RF/2018/brute_force/n10_d8_sqrt
        hyperparam_tag = f"n{args.n_estimators}_d{args.max_depth}_{args.max_features}"
        base_dir = os.path.join(args.output_root, 'RF', args.dataset_tag, hyperparam_tag)
        print(f"Output directory not specified, automatically setting to: {base_dir}")
    else:
        base_dir = args.base_dir
        print(f"Using specified output directory: {base_dir}")
    # -----------------------------------------

    os.makedirs(base_dir, exist_ok=True)

    print(f"Loading data from: {args.data_dir}")
    X_train = np.load(os.path.join(args.data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(args.data_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'))

    print(f"Loading feature names from: {args.feature_name_csv}")
    df = pd.read_csv(args.feature_name_csv)
    feature_names = list(df.columns)
    if 'label' in feature_names:
        feature_names.remove('label')
    assert len(feature_names) == X_train.shape[1], "Mismatch between feature names count and data columns."

    metrics_log = []

    print(f"Training Random Forest, iterating up to {args.n_estimators} estimators...")
    final_model = None
    for i in tqdm(range(1, args.n_estimators + 1), desc="Training Progress"):
        rf = RandomForestClassifier(
            n_estimators=i,
            max_depth=args.max_depth,
            max_features=args.max_features,
            random_state=args.random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        final_model = rf # Keep updating to the latest model

    # Evaluate only the final model after the loop
    print("Evaluating the final model...")
    preds = final_model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    
    final_metrics = {
        'estimators': args.n_estimators,
        'train_accuracy': accuracy_score(y_train, final_model.predict(X_train)),
        'val_accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1_score': f1_score(y_test, preds, zero_division=0),
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0
    }

    # Save the final model
    model_path = os.path.join(base_dir, 'random_forest_model.joblib')
    joblib.dump(final_model, model_path)
    print(f"\nFinal model saved to: {model_path}")

    # Write final metrics to a text file
    metrics_path = os.path.join(base_dir, 'final_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f'Final Metrics for Random Forest ({args.n_estimators} trees):\n')
        for key, value in final_metrics.items():
            f.write(f'{key.replace("_", " ").title()}: {value:.4f}\n')
    print(f"Final metrics saved to: {metrics_path}")

    # Save feature importances
    importances_path = os.path.join(base_dir, 'feature_importance.csv')
    importances = final_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    importance_df.to_csv(importances_path, index=False)
    print(f"Feature importances saved to: {importances_path}")
    
    print("--- Script Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Random Forest model.')
    # Path and Directory Arguments
    parser.add_argument('--data_dir', type=str, default='../../data/2018/brute_force', help='Directory containing the .npy data files.')
    parser.add_argument('--feature_name_csv', type=str, default='../../data/2018/brute_force/bf.csv', help='CSV file to extract feature names from.')
    parser.add_argument('--base_dir', type=str, default=None, help='(Optional) Override output directory. Auto-generated if not set.')
    parser.add_argument('--output_root', type=str, default='../../checkpoints', help='Root directory for saving outputs.')
    parser.add_argument('--dataset_tag', type=str, default='2018/brute_force', help='Tag for the dataset version.')
    # Model Hyperparameters
    parser.add_argument('--n_estimators', type=int, default=10, help='The total number of trees to train.')
    parser.add_argument('--max_depth', type=int, default=8, help='The maximum depth of the trees.')
    parser.add_argument('--max_features', type=str, default='sqrt', help='The number of features to consider when looking for the best split.')
    parser.add_argument('--random_state', type=int, default=42, help='Seed for reproducibility.')

    args = parser.parse_args()
    main(args)