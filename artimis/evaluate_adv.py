import os
import torch
import argparse
import numpy as np
import pandas as pd
from utils.get_models import get_models
from utils.AverageMeter import AccuracyMeter
import xgboost as xgb

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate adversarial success rate')
    parser.add_argument('--orig_path', type=str, default="csv")
    parser.add_argument('--adv_path', type=str, default="csv")
    parser.add_argument('--label_path', type=str, default="", help='Path to corresponding labels')
    parser.add_argument('--root_path', type=str, default='', help='Root directory for models')
    parser.add_argument('--target-models', nargs='+', default=['XGBOOST-BB','RF-BB','MLP1','LeNet','LSTM','MAMPF','FSNET','KITNET','AlexNet','AlertNet','DeepNet','IdsNet'],
                        help='Target model names to evaluate')
    parser.add_argument('--gpu-id', type=int, default=0)
    return parser.parse_args()

def main(args):
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    models, metrics = get_models(args, device)
    orig = np.loadtxt(args.orig_path, delimiter=',')
    adv = np.loadtxt(args.adv_path, delimiter=',')
    labels = np.load(args.label_path).astype(int)
    labels = torch.tensor(labels).long().to(device)

    orig = torch.tensor(orig, dtype=torch.float32).to(device)
    adv = torch.tensor(adv, dtype=torch.float32).to(device)

    assert orig.shape == adv.shape, "The shapes of the original sample and the adversarial sample are inconsistent"
    assert orig.shape[0] == labels.shape[0], "The number of samples does not match the number of labels"

    print(f"✅ load {orig.shape[0]} sample")

    for name in args.target_models:
        model_info = models[name]
        model, model_type = model_info['model'], model_info['type']
        metric = metrics[name]

        if model_type == 'sklearn':
            pred_clean = torch.tensor(model.predict(orig.cpu().numpy())).long().to(device)
            pred_adv = torch.tensor(model.predict(adv.cpu().numpy())).long().to(device)

        elif model_type == 'function':
            pred_clean = torch.tensor(model(orig.cpu().numpy())).long().to(device)
            pred_adv = torch.tensor(model(adv.cpu().numpy())).long().to(device)

        elif model_type == 'xgboost':
            csv_path = 'csv'
            df = pd.read_csv(csv_path, nrows=1)
            feature_names = list(df.columns)
            feature_names.remove('label')

            dmatrix_clean = xgb.DMatrix(orig.cpu().numpy(), feature_names=feature_names)
            dmatrix_adv = xgb.DMatrix(adv.cpu().numpy(), feature_names=feature_names)

            prob_clean = model.predict(dmatrix_clean)
            prob_adv = model.predict(dmatrix_adv)
            pred_clean = torch.tensor((prob_clean > 0.5).astype(int)).long().to(device)
            pred_adv = torch.tensor((prob_adv > 0.5).astype(int)).long().to(device)

        else:
            model.eval()
            with torch.no_grad():
                out_clean = model(orig)
                out_adv = model(adv)

            if model_type == 'logit':
                pred_clean = out_clean.argmax(dim=1)
                pred_adv = out_adv.argmax(dim=1)
            elif model_type == 'sigmoid':
                pred_clean = (out_clean > 0.5).long().squeeze()
                pred_adv = (out_adv > 0.5).long().squeeze()

        correct_clean = (pred_clean == labels).sum().item()
        correct_adv = (pred_adv == labels).sum().item()
        metric.update(correct_clean, correct_adv, len(labels))

    print("-" * 73)
    print('|\tModel Name\t|\tNat. Acc. (%)\t|\tAdv. Acc. (%)\t|\tASR. (%)\t|')
    for name in args.target_models:
        m = metrics[name]
        print(f"|\t{name.ljust(15)}\t"
              f"|\t{round(m.clean_acc * 100, 2):<13}\t"
              f"|\t{round(m.adv_acc * 100, 2):<13}\t"
              f"|\t{round(m.attack_rate * 100, 2):<8}\t|")
    print("-" * 73)

if __name__ == '__main__':
    args = get_args()
    main(args)
