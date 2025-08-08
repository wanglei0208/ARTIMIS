# train_mampf.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from model_mampf import ProbabilityFeatureExtractor, EnhancedClassifier
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
import argparse

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, 'a', 'utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger

def evaluate(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
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
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    fpr = fp / (fp + tn)
    return total_loss / len(dataloader), accuracy, precision, recall, fpr, f1

def main(args):
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'train.log')
    model_path = os.path.join(args.output_dir, 'classifier_best.pth')
    extractor_path = os.path.join(args.output_dir, 'extractor.pkl')
    logger = setup_logger(log_file)

    X_train = np.load(args.train_x)
    y_train = np.load(args.train_y)
    X_test = np.load(args.test_x)
    y_test = np.load(args.test_y)

    import joblib
    extractor = ProbabilityFeatureExtractor()
    extractor.fit(X_train, y_train)
    joblib.dump(extractor, extractor_path)

    # 拼接原始特征和生成特征
    train_features = extractor.transform(X_train)
    test_features = extractor.transform(X_test)

    X_train_tensor = torch.tensor(train_features, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(test_features, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=args.batch_size)
    print(f"[DEBUG] train_features.shape: {train_features.shape}")
    print(f"[DEBUG] train.shape: {X_train.shape}")
    model = EnhancedClassifier(input_dim=train_features.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    early_stop_counter = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        train_metrics = evaluate(model, train_loader, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

        logger.info(f"Epoch {epoch} | Train F1: {train_metrics[-1]:.4f} | Test F1: {test_metrics[-1]:.4f} | Acc: {test_metrics[1]:.4f} | FPR: {test_metrics[4]:.4f}")

        if test_metrics[-1] > best_f1:
            best_f1 = test_metrics[-1]
            torch.save(model.state_dict(), model_path)
            logger.info(f"New best F1: {best_f1:.4f} at epoch {epoch}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            logger.info(f"No improvement in F1. Early stop counter: {early_stop_counter}/{args.early_stop}")
            if early_stop_counter >= args.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_x', type=str, default='/raid/wl_raid/ARTIMIS/data/2018/dos1/X_train.npy', help='Path to training features')
    parser.add_argument('--train_y', type=str, default='/raid/wl_raid/ARTIMIS/data/2018/dos1/y_train.npy', help='Path to training labels')
    parser.add_argument('--test_x', type=str, default='/raid/wl_raid/ARTIMIS/data/2018/dos1/X_test.npy', help='Path to test features')
    parser.add_argument('--test_y', type=str, default='/raid/wl_raid/ARTIMIS/data/2018/dos1/y_test.npy', help='Path to test labels')
    parser.add_argument('--output_dir', type=str, default='./2018/dos1', help='Directory to save logs and models')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=int, default=0, help='GPU device index')
    parser.add_argument('--early_stop', type=int, default=10, help='Early stopping patience')
    args = parser.parse_args()
    main(args)
