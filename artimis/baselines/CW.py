import torch
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import os
from utils.get_dataset import get_dataset
from utils.get_models import get_models
from art.attacks.evasion import CarliniL2Method
from art.estimators.classification import PyTorchClassifier
from artimis import load_scaler, apply_constraints

def get_args():
    parser = argparse.ArgumentParser(description='C&W Black-box Attack with Final Statistics')
    parser.add_argument('--root_path', type=str, default='ARTIMIS/artimis')
    parser.add_argument('--dataset', type=str, default='cicids2017')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-worker', type=int, default=4)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--confidence', type=float, default=0)
    parser.add_argument('--max-iter', type=int, default=10)
    parser.add_argument('--clip-min', type=float, default=0.0)
    parser.add_argument('--clip-max', type=float, default=1.0)
    parser.add_argument('--binary-search-steps', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--initial-const', type=float, default=0.001)
    parser.add_argument('--target-models', nargs='+', default=['RF2','XGBOOST2','MAMPF','FSNET','KITNET','AlexNet','AlertNet','DeepNet','IdsNet'])
    parser.add_argument('--surrogate-model', type=str, default='MLP')
    parser.add_argument('--scaler-path', type=str, default='data/2018/dos1/dos1.joblib')
    return parser.parse_args()

def build_cw_attack(surrogate_model, device, args, input_dim):
    class SurrogateWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x)

    wrapped_model = SurrogateWrapper(surrogate_model).to(device)
    bb_model = PyTorchClassifier(
        model=wrapped_model,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(input_dim,),
        nb_classes=2,
        clip_values=(args.clip_min, args.clip_max)
    )
    attack = CarliniL2Method(
        bb_model,
        confidence=args.confidence,
        max_iter=args.max_iter,
        binary_search_steps=args.binary_search_steps,
        learning_rate=args.learning_rate,
        initial_const=args.initial_const,
        verbose=True
    )
    return attack

def main(args):
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    dataloader = get_dataset(args)
    models, metrics = get_models(args, device)
    surrogate_model = models[args.surrogate_model]['model'].to(device)
    target_model_names = args.target_models
    scaler = load_scaler(args.scaler_path)

    for batch_idx, (data, label, _) in enumerate(tqdm(dataloader)):
        data, label = data.to(device), label.to(device)
        attack = build_cw_attack(surrogate_model, device, args, data.shape[1])
        adv_samples = attack.generate(x=data.cpu().numpy())
        adv_samples = torch.tensor(adv_samples, dtype=torch.float32, device=device)
        adv_samples = apply_constraints(adv_samples, scaler)

        for model_name in target_model_names:
            model_info = models[model_name]
            model, model_type = model_info['model'], model_info['type']

            if model_type == 'sklearn':
                pred_clean = torch.tensor(model.predict(data.cpu().numpy())).long().to(device)
                pred_adv = torch.tensor(model.predict(adv_samples.cpu().numpy())).long().to(device)
            elif model_type == 'function':
                pred_clean = torch.tensor(model(data.cpu().numpy())).long().to(device)
                pred_adv = torch.tensor(model(adv_samples.cpu().numpy())).long().to(device)
            elif model_type == 'xgboost':
                import xgboost as xgb
                csv_path = 'data/2017/botnet/bot.csv'
                df = pd.read_csv(csv_path, nrows=1)
                feature_names = list(df.columns)
                feature_names.remove('label')
                dmatrix_clean = xgb.DMatrix(data.cpu().numpy(), feature_names=feature_names)
                dmatrix_adv = xgb.DMatrix(adv_samples.cpu().numpy(), feature_names=feature_names)
                pred_clean = torch.tensor((model.predict(dmatrix_clean) > 0.5).astype(int)).long().to(device)
                pred_adv = torch.tensor((model.predict(dmatrix_adv) > 0.5).astype(int)).long().to(device)
            else:
                model.eval()
                with torch.no_grad():
                    out_clean = model(data)
                    out_adv = model(adv_samples)
                pred_clean = out_clean.argmax(dim=1)
                pred_adv = out_adv.argmax(dim=1)

            correct_clean = (pred_clean == label).sum().item()
            correct_adv = (pred_adv == label).sum().item()
            metrics[model_name].update(correct_clean, correct_adv, label.size(0))

    print(f"🎯 Surrogate model: {args.surrogate_model}")
    print('-' * 73)
    print('|\tTarget model name\t|\tNat. Acc. (%)\t|\tAdv. Acc. (%)\t|\tASR. (%)\t|')
    for model_name in target_model_names:
        m = metrics[model_name]
        print(f"|\t{model_name.ljust(17)}\t"
              f"|\t{round(m.clean_acc * 100, 2):<13}\t"
              f"|\t{round(m.adv_acc * 100, 2):<13}\t"
              f"|\t{round(m.attack_rate * 100, 2):<8}\t|")
    print('-' * 73)

if __name__ == '__main__':
    args = get_args()
    main(args)
