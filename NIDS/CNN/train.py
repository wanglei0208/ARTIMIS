import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from model import MLP, AlertNet, DeepNet, LeNet1D, AlexNet1D, VGG11_1D, VGG16_1D
from model import IdsNet
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

def setup_logger(log_file):
    """Sets up a logger to save training progress to a file."""
    # Ensure the directory for the log file exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    # Avoid adding handlers if they already exist to prevent duplicate logs
    if not logger.handlers:
        handler = logging.FileHandler(log_file, 'a', 'utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    return logger

def get_model(model_name, num_classes, input_dim):
    """Selects and returns the specified model architecture."""
    if model_name == 'LeNet':
        return LeNet1D(input_dim=input_dim, num_classes=num_classes)
    elif model_name == 'IdsNet':
        return IdsNet(input_dim=input_dim, num_classes=num_classes)
    elif model_name == 'MLP':
        return MLP(input_dim=input_dim, num_classes=num_classes)
    elif model_name == 'AlertNet':
        return AlertNet(input_dim=input_dim, num_classes=num_classes)
    elif model_name == 'DeepNet':
        return DeepNet(input_dim=input_dim, num_classes=num_classes)
    elif model_name == 'AlexNet':
        return AlexNet1D(input_dim=input_dim, num_classes=num_classes)
    elif model_name == 'VGG11':
        return VGG11_1D(num_classes)
    elif model_name == 'VGG16':
        return VGG16_1D(num_classes)
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
    
    # Ensure there are enough samples to unpack confusion matrix
    if len(np.unique(all_labels)) > 1:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    else: # Handle case with only one class in labels
        fpr = 0

    return total_loss / len(dataloader), accuracy, precision, recall, fpr, f1

def main(args):
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Automatic Directory Path Generation ---
    if args.base_dir is None:
        base_dir = os.path.join(args.output_root, 'CNN', args.dataset_tag, args.model)
        print(f"Output directory not specified, automatically setting to: {base_dir}")
    else:
        base_dir = args.base_dir
        print(f"Using specified output directory: {base_dir}")
    # -----------------------------------------

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
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    model = get_model(args.model, num_classes=2, input_dim=input_dim).to(device)
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

        train_loss, _, _, _, _, train_f1 = evaluate_model(model, train_loader, criterion, device)
        test_loss, test_acc, _, _, test_fpr, test_f1 = evaluate_model(model, test_loader, criterion, device)
        
        log_message = (
            f'Epoch [{epoch}/{args.epochs}], Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, '
            f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test FPR: {test_fpr:.4f}'
        )
        logger.info(log_message)

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_f1_epoch = epoch
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_f1_model.pth'))
            logger.info(f'New best F1 score: {best_f1:.4f} at epoch {best_f1_epoch}. Model saved.')
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= args.early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch} due to no improvement in F1 score for {args.early_stop_patience} consecutive epochs.")
            break
            
    torch.save(model.state_dict(), os.path.join(model_dir, 'final_model.pth'))
    logger.info(f"Final model saved. Best F1 score was {best_f1:.4f} at epoch {best_f1_epoch}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN model for NIDS anomaly detection.')
    
    # --- Path and Directory Arguments ---
    parser.add_argument('--model', type=str, default="AlexNet", choices=['MLP','LeNet','DeepNet','AlertNet','IdsNet','AlexNet'], help='(Required) Model architecture to train.')
    parser.add_argument('--data_dir', type=str, default='../../data/2018/brute_force', help='Directory containing the .npy data files, relative to the script location.')
    parser.add_argument('--base_dir', type=str, default=None, help='(Optional) Override the full output directory. If not set, it will be generated automatically based on other arguments.')
    parser.add_argument('--output_root', type=str, default='../../checkpoints', help='Root directory for saving all model outputs.')
    parser.add_argument('--dataset_tag', type=str, default='2018/brute_force', help='A tag for the dataset version, used in auto-generating the output path.')
    
    # --- Training Hyperparameters ---
    parser.add_argument('--device', type=int, default=0, help='GPU device to use.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer.')
    parser.add_argument('--early_stop_patience', type=int, default=5, help='Patience for early stopping.')
    
    args = parser.parse_args()
    main(args)