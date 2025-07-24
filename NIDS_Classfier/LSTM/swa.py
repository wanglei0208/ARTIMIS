import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from train import get_model  # 模型加载函数
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--use_swa',action='store_true', help='是否启用 SWA 平均模型')
args = parser.parse_args()

model_name = 'LSTM'
model_ckpt = 'ARTIMIS/LSTM/2018/brute_force/h120n3/models/best_f1_model.pth'
save_path = 'ARTIMIS/LSTM/2018/brute_force/h120n3/models/epoch25/bayes_finetuned_model.pth'
data_dir = 'ARTIMIS/data/2018/brute_force/'
device = torch.device('cuda:0')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
sigma = 0.05
lambda_eps = 0.01
gamma = 1e-3
finetune_epochs = 25
batch_size = 64

X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

model = get_model(model_name, input_size=60, hidden_size=120, num_layers=3, output_size=2).to(device)
model.load_state_dict(torch.load(model_ckpt))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

if args.use_swa:
    swa_model = AveragedModel(model)
    swa_start = 5
    swa_every = 1

for epoch in range(finetune_epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)

        outputs = model(x)
        loss = criterion(outputs, y)
        grad_list = list(torch.autograd.grad(loss, model.parameters(), create_graph=True))

        delta_w_star = [lambda_eps * g / (g.norm() + 1e-6) for g in grad_list]

        original_data_list = [param.data.clone() for param in model.parameters()]
        for i, param in enumerate(model.parameters()):
            param.data += gamma * delta_w_star[i].detach()

        outputs_perturbed = model(x)
        loss_perturbed = criterion(outputs_perturbed, y)
        grad_perturbed = torch.autograd.grad(loss_perturbed, model.parameters())

        for i, param in enumerate(model.parameters()):
            param.data = original_data_list[i]
            hessian_approx = (grad_perturbed[i] - grad_list[i]) / gamma
            grad_list[i] = grad_list[i] + hessian_approx

        with torch.no_grad():
            for i, param in enumerate(model.parameters()):
                if param.grad is not None:
                    param.grad.zero_()
                if param.shape == grad_list[i].shape:
                    param.grad = grad_list[i].detach().clone()
                else:
                    print(f"Shape mismatch: param[{i}].shape = {param.shape}, grad.shape = {grad_list[i].shape}")
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1}/{finetune_epochs} complete")

    if args.use_swa and epoch >= swa_start and ((epoch - swa_start) % swa_every == 0):
        swa_model.update_parameters(model)

if args.use_swa:
    torch.save(swa_model.module.state_dict(), save_path)
    print(f"[SWA] 模型已保存至: {save_path}")
else:
    torch.save(model.state_dict(), save_path)
    print(f"[Vanilla] 模型已保存至: {save_path}")

def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            data = data.view(data.size(0), -1)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, prec, rec, f1

final_model = swa_model.module if args.use_swa else model
acc, prec, rec, f1 = evaluate_model(final_model, test_loader)
print(f"Test Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

with open(os.path.join(os.path.dirname(save_path), 'bayes_finetune_test_metrics.txt'), 'w') as f:
    f.write(f"Test Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}\n")
