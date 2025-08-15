import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# This local import works because we run the script from its own directory
from model_mampf import ProbabilityFeatureExtractor, EnhancedClassifier
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
import argparse
import joblib

def setup_logger(log_file):
    """Sets up a logger to save training progress to a file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_file, 'a', 'utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    return logger

def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on a given dataset and returns key metrics."""
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    if len(np.unique(all_labels)) > 1:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    else:
        fpr = 0
        
    return total_loss / len(dataloader), accuracy, precision, recall, fpr, f1

def main(args):
    """Main function to train the MaMPF pipeline."""
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Automatic Directory Path Generation ---
    if args.base_dir is None:
        hyperparam_tag = f"lr{args.lr}_bs{args.batch_size}"
        base_dir = os.path.join(args.output_root, 'mampf', args.dataset_tag, hyperparam_tag)
        print(f"Output directory not specified, automatically setting to: {base_dir}")
    else:
        base_dir = args.base_dir
        print(f"Using specified output directory: {base_dir}")
    # -----------------------------------------

    os.makedirs(base_dir, exist_ok=True)
    log_file = os.path.join(base_dir, 'training.log')
    model_path = os.path.join(base_dir, 'classifier_best.pth')
    extractor_path = os.path.join(base_dir, 'extractor.pkl')
    logger = setup_logger(log_file)
    logger.info("Starting MaMPF training...")
    logger.info(f"Arguments: {vars(args)}")

    print(f"Loading data from: {args.data_dir}")
    X_train = np.load(os.path.join(args.data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(args.data_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'))

    print("Training the ProbabilityFeatureExtractor...")
    extractor = ProbabilityFeatureExtractor()
    extractor.fit(X_train, y_train)
    joblib.dump(extractor, extractor_path)
    print(f"Feature extractor saved to: {extractor_path}")

    print("Transforming features for the classifier...")
    train_features = extractor.transform(X_train)
    test_features = extractor.transform(X_test)
    print(f"Original feature dimension: {X_train.shape[1]}")
    print(f"New feature dimension after MaMPF: {train_features.shape[1]}")

    train_loader = DataLoader(TensorDataset(torch.tensor(train_features, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(test_features, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), batch_size=args.batch_size)
    
    model = EnhancedClassifier(input_dim=train_features.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_f1_epoch = 0
    early_stop_counter = 0

    print(f"Training EnhancedClassifier for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

        train_loss, _, _, _, _, train_f1 = evaluate(model, train_loader, criterion, device)
        test_loss, test_acc, _, _, test_fpr, test_f1 = evaluate(model, test_loader, criterion, device)

        logger.info(f"Epoch {epoch} | Train F1: {train_f1:.4f} | Test F1: {test_f1:.4f} | Test Acc: {test_acc:.4f} | Test FPR: {test_fpr:.4f}")

        if test_f1 > best_f1:
            best_f1, best_f1_epoch = test_f1, epoch
            torch.save(model.state_dict(), model_path)
            logger.info(f"New best F1: {best_f1:.4f} at epoch {epoch}. Model saved.")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.early_stop_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break
    
    print(f"Training finished. Best F1 score was {best_f1:.4f} at epoch {best_f1_epoch}.")
    print(f"Best classifier model saved to: {model_path}")
    print("--- Script Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the MaMPF (Malicious and Benign Probability Features) model.')
    # Path and Directory Arguments
    parser.add_argument('--data_dir', type=str, default='../../data/2018/brute_force', help='Directory for the .npy data files.')
    parser.add_argument('--base_dir', type=str, default=None, help='(Optional) Override output directory. Auto-generated if not set.')
    parser.add_argument('--output_root', type=str, default='../../checkpoints', help='Root directory for saving outputs.')
    parser.add_argument('--dataset_tag', type=str, default='2018/brute_force', help='Tag for the dataset version.')
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--device', type=int, default=0, help='GPU device index.')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='Patience for early stopping.')
    
    args = parser.parse_args()
    main(args)