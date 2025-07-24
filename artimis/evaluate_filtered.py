import os
import torch
import argparse
import numpy as np
import pandas as pd
from utils.get_models import get_models
import xgboost as xgb

def get_args():
    parser = argparse.ArgumentParser(description='Filtered evaluation of adversarial success rate')
    parser.add_argument('--orig_path', type=str, default="csv")
    parser.add_argument('--adv_path', type=str, default="csv")
    parser.add_argument('--label_path', type=str, default="npy")
    parser.add_argument('--root_path', type=str, default="ARTIMIS/artimis")
    parser.add_argument('--target-models', nargs='+', default=['XGBOOST2','DIFFRF','MAMPF','FSNET','KITNET','AlexNet','AlertNet','DeepNet','IdsNet'])
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--txt-path', type=str, default="txt", help='Path to save CSV results')
    return parser.parse_args()

def center(text, width):
    return str(text).center(width)

def main(args):
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    models, _ = get_models(args, device)

    orig = np.loadtxt(args.orig_path, delimiter=',')
    adv = np.loadtxt(args.adv_path, delimiter=',')
    labels = np.load(args.label_path).astype(int)
    labels = torch.tensor(labels).long().to(device)

    orig = torch.tensor(orig, dtype=torch.float32).to(device)
    adv = torch.tensor(adv, dtype=torch.float32).to(device)

    assert orig.shape == adv.shape and orig.shape[0] == labels.shape[0]
    print(f"✅ load sample: {orig.shape[0]}")

    lines = []
    line_sep = "-" * 153
    header = (
        f"| {center('Model Name', 16)} | {center('Total Acc (%)', 14)} | "
        f"{center('Adv Acc (All) (%)', 17)} | {center('ASR (All) (%)', 15)} | "
        f"{center('Filtered Num', 13)} | {center('Adv Acc (Filtered) (%)', 23)} | {center('ASR (Filtered) (%)', 19)} |"
    )

    print(line_sep)
    print(header)
    print(line_sep)
    lines.extend([line_sep, header, line_sep])

    for name in args.target_models:
        model_info = models[name]
        model, model_type = model_info['model'], model_info['type']

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

            d_clean = xgb.DMatrix(orig.cpu().numpy(), feature_names=feature_names)
            d_adv = xgb.DMatrix(adv.cpu().numpy(), feature_names=feature_names)

            pred_clean = torch.tensor((model.predict(d_clean) > 0.5).astype(int)).long().to(device)
            pred_adv = torch.tensor((model.predict(d_adv) > 0.5).astype(int)).long().to(device)

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

        total = labels.shape[0]
        correct_clean = (pred_clean == labels).sum().item()
        total_acc = correct_clean / total

        correct_adv_total = (pred_adv == labels).sum().item()
        adv_acc_all = correct_adv_total / total
        asr_all = 1.0 - adv_acc_all

        correct_mask = (pred_clean == labels)
        filtered_num = correct_mask.sum().item()

        if filtered_num == 0:
            line = (
                f"| {center(name, 16)} | {center(f'{total_acc * 100:.2f}%', 14)} | {center(f'{adv_acc_all * 100:.2f}%', 17)} | "
                f"{center(f'{asr_all * 100:.2f}%', 15)} | {center('0', 13)} | {center('-', 23)} | {center('-', 19)} |"
            )
            print(line)
            lines.append(line)
            continue

        pred_adv_filtered = pred_adv[correct_mask]
        labels_filtered = labels[correct_mask]
        correct_adv_filtered = (pred_adv_filtered == labels_filtered).sum().item()

        adv_acc_filtered = correct_adv_filtered / filtered_num
        asr_filtered = 1.0 - adv_acc_filtered

        line = (
            f"| {center(name, 16)} | {center(f'{total_acc * 100:.2f}%', 14)} | {center(f'{adv_acc_all * 100:.2f}%', 17)} | "
            f"{center(f'{asr_all * 100:.2f}%', 15)} | {center(filtered_num, 13)} | {center(f'{adv_acc_filtered * 100:.2f}%', 23)} | "
            f"{center(f'{asr_filtered * 100:.2f}%', 19)} |"
        )
        print(line)
        lines.append(line)

    print(line_sep)
    lines.append(line_sep)

    txt_path = args.txt_path
    with open(txt_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')

if __name__ == '__main__':
    args = get_args()
    main(args)

