import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from model import MLP,AlertNet,DeepNet,LeNet1D,AlexNet1D,FSNetForStatFeatures,FSNetForStatFeatures0,VGG11_1D,VGG16_1D  # 模型存储在model.py中
from model import IdsNet
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, 'a', 'utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger

#定义选择模型函数
def get_model(model_name, num_classes, input_dim):
    if model_name == 'LeNet':
        return LeNet1D(input_dim=input_dim,num_classes=num_classes)
    elif model_name == 'IdsNet':
        return IdsNet(input_dim=input_dim, num_classes=num_classes)
    elif model_name == 'MLP':
        return MLP(input_dim=input_dim,num_classes=num_classes)
    elif model_name == 'AlertNet':
        return AlertNet(input_dim=input_dim,num_classes=num_classes)
    elif model_name == 'DeepNet':
        return DeepNet(input_dim=input_dim,num_classes=num_classes)
    elif model_name == 'AlexNet':
        return AlexNet1D(input_dim=input_dim,num_classes=num_classes)
    elif model_name == 'FSNet':
        return FSNetForStatFeatures0(input_dim=input_dim,num_classes=num_classes)
    elif model_name == 'VGG11':
        return VGG11_1D(num_classes)
    elif model_name == 'VGG16':
        return VGG16_1D(num_classes)
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
            outputs = model(data.to(device))
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

    # 创建base_dir及子目录
    if not os.path.exists(args.base_dir):
        os.makedirs(args.base_dir)
    model_dir = os.path.join(args.base_dir, 'models')
    log_file = os.path.join(args.base_dir, 'training.log')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    logger = setup_logger(log_file)
    logger.info("Starting training...")

    # load data
    X_train = np.load('./data/2018/brute_force/X_train.npy')
    y_train = np.load('./data/2018/brute_force/y_train.npy')
    X_test = np.load('./data/2018/brute_force/X_test.npy')
    y_test = np.load('./data/2018/brute_force/y_test.npy')
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the model
    input_dim = X_train.shape[1]
    print(input_dim)
    model = get_model(args.model, num_classes=2, input_dim=input_dim).to(device)
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
            outputs = model(data.to(device))
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

        # if epoch % 50 == 0:
        #     torch.save(model.state_dict(), os.path.join(model_dir, f'model_epoch_{epoch}.pth'))
        if early_stop_counter >= early_stop_patience:
            logger.info(f"Early stopping triggered at epoch {epoch}. No improvement in accuracy for {early_stop_patience} consecutive epochs.")
            break
    torch.save(model.state_dict(), os.path.join(model_dir, 'final_model.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MLP for NIDS anomaly detection')
    parser.add_argument('--base_dir', type=str,  default='ARTIMIS/CNN/2018/brute_force/AlexNet', help='Base directory for saving models and logs')
    parser.add_argument('--device', type=int, default=0, help='GPU device to use')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--model', type=str,  default='AlexNet', choices=['MLP','LeNet','DeepNet','AlertNet','IdsNet','FSNet','AlexNet'], help='Model to use for training')
    args = parser.parse_args()
    main(args)
