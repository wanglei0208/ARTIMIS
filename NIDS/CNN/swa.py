import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
# This local import works because we run the script from its own directory
from train import get_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
from tqdm import tqdm

def evaluate_model(model, dataloader, device):
    """Evaluates the model on a given dataset and returns key metrics."""
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
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return acc, prec, rec, f1

def main(args):
    """Main function to run SWA fine-tuning."""
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Ensure the save directory exists
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Load data
    print(f"Loading data from {args.data_dir}...")
    X_train = np.load(os.path.join(args.data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(args.data_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'))

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), batch_size=args.batch_size, shuffle=False)

    print(f"Loading pre-trained model from: {args.model_ckpt}")
    model = get_model(args.model_name, num_classes=2, input_dim=X_train.shape[1]).to(device)
    model.load_state_dict(torch.load(args.model_ckpt, map_location=device))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    swa_model = AveragedModel(model) if args.use_swa else None

    print(f"Starting fine-tuning for {args.finetune_epochs} epochs...")
    for epoch in range(args.finetune_epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.finetune_epochs}', unit="batch")
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)

            outputs = model(x)
            loss = criterion(outputs, y)
            
            # The complex gradient manipulation part from the original script
            grad_list = list(torch.autograd.grad(loss, model.parameters(), create_graph=True))
            delta_w_star = [args.lambda_eps * g / (g.norm() + 1e-6) for g in grad_list]
            original_data_list = [param.data.clone() for param in model.parameters()]
            
            for i, param in enumerate(model.parameters()):
                param.data += args.gamma * delta_w_star[i].detach()

            outputs_perturbed = model(x)
            loss_perturbed = criterion(outputs_perturbed, y)
            grad_perturbed = torch.autograd.grad(loss_perturbed, model.parameters())

            for i, param in enumerate(model.parameters()):
                param.data = original_data_list[i]
                hessian_approx = (grad_perturbed[i] - grad_list[i]) / args.gamma
                grad_list[i] += hessian_approx

            optimizer.zero_grad()
            with torch.no_grad():
                for i, param in enumerate(model.parameters()):
                    if param.grad is not None:
                        param.grad.zero_()
                    if param.shape == grad_list[i].shape:
                        param.grad = grad_list[i].detach().clone()
            optimizer.step()
        
        if args.use_swa and epoch >= args.swa_start:
            swa_model.update_parameters(model)

    # Save the final model
    final_model = model
    if args.use_swa:
        torch.save(swa_model.module.state_dict(), args.save_path)
        print(f"[SWA] Model saved to: {args.save_path}")
        final_model = swa_model.module
    else:
        torch.save(model.state_dict(), args.save_path)
        print(f"[Vanilla] Model saved to: {args.save_path}")

    # Evaluate the final model and save metrics
    print("Evaluating final model on the test set...")
    acc, prec, rec, f1 = evaluate_model(final_model, test_loader, device)
    print(f"Test Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    metrics_path = os.path.join(os.path.dirname(args.save_path), 'finetune_test_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"SWA Used: {args.use_swa}\n")
        f.write(f"Test Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}\n")
    print(f"Test metrics saved to: {metrics_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune a model with optional Stochastic Weight Averaging (SWA).")
    
    # --- Path and Model Arguments (Defaults updated to AlexNet) ---
    parser.add_argument('--model_name', type=str, default='AlexNet', help='Name of the model architecture to use.')
    parser.add_argument('--model_ckpt', type=str, default='../../checkpoints/CNN/2018/brute_force/AlexNet/models/best_f1_model.pth', help='Path to the pre-trained model checkpoint.')
    parser.add_argument('--save_path', type=str, default='../../checkpoints/CNN/2018/brute_force/AlexNet/swa_model.pth', help='Path to save the fine-tuned model.')
    parser.add_argument('--data_dir', type=str, default='../../data/2018/brute_force', help='Directory of the .npy data files.')
    
    # --- Fine-tuning Hyperparameters ---
    parser.add_argument('--device', type=int, default=0, help='GPU device ID to use.')
    parser.add_argument('--finetune_epochs', type=int, default=15, help='Number of epochs for fine-tuning.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--lambda_eps', type=float, default=0.01, help='Hyperparameter for gradient calculation.')
    parser.add_argument('--gamma', type=float, default=1e-3, help='Hyperparameter for gradient calculation.')
    
    # --- SWA Specific Arguments ---
    parser.add_argument('--use_swa', action='store_true', help='Enable this flag to use Stochastic Weight Averaging.')
    parser.add_argument('--swa_start', type=int, default=5, help='Epoch to start updating the SWA model.')
    
    args = parser.parse_args()
    main(args)