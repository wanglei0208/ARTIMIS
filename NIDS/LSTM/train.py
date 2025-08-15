import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from model import LSTMModel, LSTMWithScalar2Vec, BILSTMIWithScalar2Vec

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

def get_model(model_name, input_size, hidden_size, num_layers, output_size, embed_dim=32):
    """Selects and returns the specified model architecture."""
    if model_name == 'LSTM':
        return LSTMModel(input_size, hidden_size, num_layers, output_size)
    elif model_name == 'LSTM_Embed':
        return LSTMWithScalar2Vec(input_features=input_size, embed_dim=embed_dim,
                                  hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
    elif model_name == 'BILSTM_Embed':
        return BILSTMIWithScalar2Vec(input_features=input_size, embed_dim=embed_dim,
                                   hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
    else:
        raise ValueError(f"Model name '{model_name}' not recognized.")

def evaluate_model(model, dataloader, criterion, device):
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
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Auto-generate output directory if not specified
    if args.base_dir is None:
        base_dir = os.path.join(args.output_root, 'LSTM', args.dataset_tag, f"{args.model}_h{args.hidden_size}_n{args.num_layers}")
        print(f"Output directory not specified, automatically setting to: {base_dir}")
    else:
        base_dir = args.base_dir
        print(f"Using specified output directory: {base_dir}")

    os.makedirs(base_dir, exist_ok=True)
    model_dir = os.path.join(base_dir, 'models')
    log_file = os.path.join(base_dir, 'training.log')
    os.makedirs(model_dir, exist_ok=True)

    logger = setup_logger(log_file)
    logger.info("Starting training...")
    logger.info(f"Arguments: {vars(args)}")

    print(f"Loading data from: {args.data_dir}")
    X_train = np.load(os.path.join(args.data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(args.data_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'))

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), batch_size=args.batch_size, shuffle=False)

    model = get_model(args.model, input_size=X_train.shape[1], hidden_size=args.hidden_size, num_layers=args.num_layers, output_size=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_f1 = 0.0
    best_f1_epoch = 0
    early_stop_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', unit="batch")
        for data, labels in progress_bar:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

        _, train_accuracy, _, _, _, train_f1 = evaluate_model(model, train_loader, criterion, device)
        test_loss, test_accuracy, _, _, test_fpr, test_f1 = evaluate_model(model, test_loader, criterion, device)
        
        logger.info(f'Epoch [{epoch}/{args.epochs}], Train Acc: {train_accuracy:.4f}, Train F1: {train_f1:.4f}, Test Acc: {test_accuracy:.4f}, Test F1: {test_f1:.4f}, Test FPR: {test_fpr:.4f}')

        if test_f1 > best_f1:
            best_f1, best_f1_epoch = test_f1, epoch
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_f1_model.pth'))
            logger.info(f'New best F1 score: {best_f1:.4f} at epoch {best_f1_epoch}. Model saved.')
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= args.early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch}.")
            break

    torch.save(model.state_dict(), os.path.join(model_dir, 'final_model.pth'))
    logger.info(f"Final model saved. Best F1 was {best_f1:.4f} at epoch {best_f1_epoch}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSTM models for NIDS anomaly detection.')
    # Path and Directory Arguments
    parser.add_argument('--data_dir', type=str, default='../../data/2018/brute_force', help='Directory for the .npy data files.')
    parser.add_argument('--base_dir', type=str, default=None, help='(Optional) Override output directory. Auto-generated if not set.')
    parser.add_argument('--output_root', type=str, default='../../checkpoints', help='Root directory for saving outputs.')
    parser.add_argument('--dataset_tag', type=str, default='2018/brute_force', help='Tag for the dataset version.')
    # Model and Training Hyperparameters
    parser.add_argument('--model', type=str, default='LSTM', choices=['LSTM', 'LSTM_Embed','BILSTM_Embed'], help='Model architecture.')
    parser.add_argument('--hidden_size', type=int, default=120, help='Number of hidden units in LSTM.')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of LSTM layers.')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Optimizer learning rate.')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='Patience for early stopping.')
    args = parser.parse_args()
    main(args)