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
from model import LSTMModel, LSTMWithScalar2Vec,BILSTMIWithScalar2Vec  # 导入新模型

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, 'a', 'utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger

def get_model(model_name, input_size, hidden_size, num_layers, output_size, embed_dim=32):
    if model_name == 'LSTM':
        return LSTMModel(input_size, hidden_size, num_layers, output_size)

    elif model_name == 'LSTM_Embed':
        return LSTMWithScalar2Vec(input_features=input_size, embed_dim=embed_dim,
                                  hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
    elif model_name == 'BILSTM_Embed':
        return LSTMWithScalar2Vec(input_features=input_size, embed_dim=embed_dim,
                                  hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
    else:
        raise ValueError("Model name not recognized.")

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
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

    if not os.path.exists(args.base_dir):
        os.makedirs(args.base_dir)
    model_dir = os.path.join(args.base_dir, 'models')
    log_file = os.path.join(args.base_dir, 'training.log')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    logger = setup_logger(log_file)
    logger.info("Starting training...")

    # 加载数据
    X_train = np.load('ARTIMIS/data/2018/brute_force/X_train.npy')
    y_train = np.load('ARTIMIS/data/2018/brute_force/y_train.npy')
    X_test = np.load('ARTIMIS/data/2018/brute_force/X_test.npy')
    y_test = np.load('ARTIMIS/data/2018/brute_force/y_test.npy')
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = get_model(args.model, input_size=60, hidden_size=120, num_layers=3, output_size=2).to(device)
    #model = get_model(args.model, input_size=60, hidden_size=120, num_layers=1, output_size=2, embed_dim=32).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_accuracy = 0.0
    best_f1 = 0.0
    early_stop_counter = 0
    early_stop_patience = 10
    for epoch in range(1, args.epochs + 1):
        model.train()
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for data, labels in progress_bar:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

        train_loss, train_accuracy, train_precision, train_recall, train_fpr, train_f1 = evaluate_model(model, train_loader, criterion, device)
        test_loss, test_accuracy, test_precision, test_recall, test_fpr, test_f1 = evaluate_model(model, test_loader, criterion, device)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_accuracy_epoch = epoch
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_accuracy_model.pth'))
            logger.info(f'New best accuracy: {best_accuracy:.4f} at epoch {best_accuracy_epoch}')

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_f1_epoch = epoch
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_f1_model.pth'))
            logger.info(f'New best F1 score: {best_f1:.4f} at epoch {best_f1_epoch}')
            early_stop_counter = 0  # reset counter
        else:
            early_stop_counter += 1

        logger.info(f'Epoch [{epoch}/{args.epochs}], '
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train FPR: {train_fpr:.4f}, Train F1: {train_f1:.4f}, '
                    f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test FPR: {test_fpr:.4f}, Test F1: {test_f1:.4f}, '
                    f'Best Acc: {best_accuracy:.4f} (epoch {best_accuracy_epoch}), Best F1: {best_f1:.4f} (epoch {best_f1_epoch})')

        # if epoch % 25 == 0:
        #     torch.save(model.state_dict(), os.path.join(model_dir, f'model_epoch_{epoch}.pth'))
        if early_stop_counter >= early_stop_patience:
            logger.info(f"Early stopping triggered at epoch {epoch}. No improvement in accuracy for {early_stop_patience} consecutive epochs.")
            break

    torch.save(model.state_dict(), os.path.join(model_dir, 'final_model.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSTMModel for NIDS anomaly detection')
    #parser.add_argument('--saved_dir', type=str, default='/home/wl/NIDS-master/data/all_scaler', help='Saved scaled data path')
    parser.add_argument('--base_dir', type=str, default='ARTIMIS/LSTM/2018/brute_force/h120n3', help='Base directory for saving models and logs')
    parser.add_argument('--device', type=int, default=1, help='GPU device to use')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    #parser.add_argument('--model', type=str, default='LSTM', choices=['LSTM'], help='Model to use for training')
    parser.add_argument('--model', type=str, default='LSTM', choices=['LSTM', 'LSTM_Embed','BILSTM_Embed'],
                        help='Model to use for training')
    args = parser.parse_args()

    main(args)
