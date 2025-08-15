import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import os
import argparse

def main(args):
    """Main function to train and evaluate an XGBoost model."""
    print("--- XGBoost Training and Evaluation ---")

    # --- Automatic Directory Path Generation ---
    if args.base_dir is None:
        # Create a descriptive name, e.g., ../../checkpoints/XGBOOST/2018/brute_force/n15_d6_lr0.1
        hyperparam_tag = f"n{args.num_boost_round}_d{args.max_depth}_lr{args.learning_rate}"
        base_dir = os.path.join(args.output_root, 'XGBOOST', args.dataset_tag, hyperparam_tag)
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

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    params = {
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'verbosity': 0,
        'seed': args.seed
    }
    
    print(f"Training XGBoost model for {args.num_boost_round} rounds...")
    model = xgb.train(params, dtrain, num_boost_round=args.num_boost_round)

    # Save the final model
    model_path = os.path.join(base_dir, 'xgboost_model.json')
    model.save_model(model_path)
    print(f"\nFinal model saved to: {model_path}")

    # Evaluate the final model
    print("Evaluating the final model...")
    final_preds_prob = model.predict(dtest)
    final_preds = (final_preds_prob > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, final_preds).ravel()

    final_metrics = {
        'accuracy': accuracy_score(y_test, final_preds),
        'precision': precision_score(y_test, final_preds, zero_division=0),
        'recall': recall_score(y_test, final_preds, zero_division=0),
        'f1_score': f1_score(y_test, final_preds, zero_division=0),
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0
    }

    # Write final metrics to a text file
    metrics_path = os.path.join(base_dir, 'final_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f'Final Metrics for XGBoost ({args.num_boost_round} rounds):\n')
        for key, value in final_metrics.items():
            f.write(f'{key.replace("_", " ").title()}: {value:.4f}\n')
    print(f"Final metrics saved to: {metrics_path}")

    # Save feature importances
    importances_path = os.path.join(base_dir, 'feature_importance.csv')
    importance_dict = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame([
        {'Feature': feat, 'Importance': importance_dict.get(feat, 0)}
        for feat in feature_names
    ]).sort_values(by='Importance', ascending=False)
    importance_df.to_csv(importances_path, index=False)
    print(f"Feature importances saved to: {importances_path}")
    
    print("--- Script Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an XGBoost model.')
    # Path and Directory Arguments
    parser.add_argument('--data_dir', type=str, default='../../data/2018/brute_force', help='Directory containing the .npy data files.')
    parser.add_argument('--feature_name_csv', type=str, default='../../data/2018/brute_force/bf.csv', help='CSV file to extract feature names from.')
    parser.add_argument('--base_dir', type=str, default=None, help='(Optional) Override output directory. Auto-generated if not set.')
    parser.add_argument('--output_root', type=str, default='../../checkpoints', help='Root directory for saving outputs.')
    parser.add_argument('--dataset_tag', type=str, default='2018/brute_force', help='Tag for the dataset version.')
    # Model Hyperparameters
    parser.add_argument('--num_boost_round', type=int, default=15, help='Number of boosting rounds.')
    parser.add_argument('--max_depth', type=int, default=6, help='Maximum tree depth.')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Step size shrinkage.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

    args = parser.parse_args()
    main(args)