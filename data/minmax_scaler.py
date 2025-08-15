import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from joblib import dump
import os
import argparse

def main(args):
    """
    Main function to perform all data preprocessing steps:
    1. Scale data with MinMaxScaler.
    2. Split into standard train/test sets.
    3. Create a benign-only training set for anomaly detectors.
    4. Create a combined dataset for KitNET's streaming training approach.
    5. Create a small sample of malicious test data for attack generation.
    """
    
    # --- Create necessary directories ---
    os.makedirs(args.output_dir, exist_ok=True)
    kitnet_dir = os.path.join(args.output_dir, "kitnet")
    attack_sample_dir = os.path.join(args.output_dir, "attack")
    os.makedirs(kitnet_dir, exist_ok=True)
    os.makedirs(attack_sample_dir, exist_ok=True)

    # --- Step 1: Load and Scale the original CSV data ---
    print(f"Step 1: Loading and scaling data from: {args.input_csv}")
    data = pd.read_csv(args.input_csv)
    features = data.drop('label', axis=1)

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    scaler_path = os.path.join(args.output_dir, "bf.joblib")
    print(f"Saving scaler object to: {scaler_path}")
    dump(scaler, scaler_path)

    data['label'] = data['label'].apply(lambda x: 0 if x == 'Benign' else 1)
    label_counts = data['label'].value_counts()
    print("\nLabel distribution in original data:")
    print(f"  Benign (0): {label_counts.get(0, 0)}")
    print(f"  Attack (1): {label_counts.get(1, 0)}")

    # --- Step 2: Split data into training and testing sets ---
    print("\nStep 2: Splitting data into training and testing sets (80/20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, 
        data['label'].values, 
        test_size=0.2, 
        random_state=args.random_state,
        stratify=data['label'].values
    )

    print(f"Saving standard train/test splits to '{args.output_dir}'...")
    np.save(os.path.join(args.output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(args.output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(args.output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.output_dir, "y_test.npy"), y_test)
    print("Standard .npy files saved successfully.")

    # --- Step 3: Create benign-only training data for DiFF-RF ---
    print("\nStep 3: Generating benign-only training data for DiFF-RF...")
    X_train_benign = X_train[y_train == 0]
    benign_path = os.path.join(args.output_dir, "X_train_benign.npy")
    np.save(benign_path, X_train_benign)
    print(f"Saved {len(X_train_benign)} benign samples to: {benign_path}")

    # --- Step 4: Create combined dataset for KitNET ---
    print("\nStep 4: Generating combined dataset for KitNET...")
    X_benign_train = X_train[y_train == 0]
    y_benign_train = y_train[y_train == 0]
    X_benign_test = X_test[y_test == 0]
    y_benign_test = y_test[y_test == 0]
    X_malicious_test = X_test[y_test == 1]
    y_malicious_test = y_test[y_test == 1]

    X_combined = np.concatenate((X_benign_train, X_benign_test, X_malicious_test), axis=0)
    y_combined = np.concatenate((y_benign_train, y_benign_test, y_malicious_test), axis=0)

    pd.DataFrame(X_combined).to_csv(os.path.join(kitnet_dir, "X_train_b_test_bm.csv"), index=False, header=False)
    pd.DataFrame(y_combined).to_csv(os.path.join(kitnet_dir, "y_train_b_test_bm.csv"), index=False, header=False)
    print(f"Saved combined KitNET data with {len(X_combined)} samples to '{kitnet_dir}'.")

    # --- Step 5: Create a small sample of malicious test data for attacks ---
    print(f"\nStep 5: Sampling {args.attack_samples} malicious instances for attack generation...")
    X_test_malicious_all = X_test[y_test == 1]
    y_test_malicious_all = y_test[y_test == 1]

    if len(X_test_malicious_all) < args.attack_samples:
        print(f"[Warning] Requested {args.attack_samples} samples, but only {len(X_test_malicious_all)} malicious samples are available in the test set. Using all available samples.")
        args.attack_samples = len(X_test_malicious_all)

    np.random.seed(args.random_state)
    indices = np.random.choice(len(X_test_malicious_all), size=args.attack_samples, replace=False)
    X_malicious_sample = X_test_malicious_all[indices]
    y_malicious_sample = y_test_malicious_all[indices]

    x_mal_path = os.path.join(attack_sample_dir, f'X_malicious_test_{args.attack_samples}.npy')
    y_mal_path = os.path.join(attack_sample_dir, f'y_malicious_test_{args.attack_samples}.npy')
    
    np.save(x_mal_path, X_malicious_sample)
    np.save(y_mal_path, y_malicious_sample)
    print(f"Successfully saved {args.attack_samples} malicious samples for attacks.")
    print(f"  - Features: {x_mal_path}")
    print(f"  - Labels:   {y_mal_path}")

    print("\nAll data preprocessing and file generation steps are complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the complete data preprocessing pipeline.")
    parser.add_argument('--input_csv', type=str, default="data/2018/brute_force/bf.csv", 
                        help='Path to the input CSV file.')
    parser.add_argument('--output_dir', type=str, default="data/2018/brute_force", 
                        help='Base directory to save all output files.')
    parser.add_argument('--attack_samples', type=int, default=100, 
                        help='Number of malicious samples to extract for attack generation.')
    parser.add_argument('--random_state', type=int, default=42, 
                        help='Random seed for reproducibility of splits and sampling.')
    args = parser.parse_args()
    main(args)