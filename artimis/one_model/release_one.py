import torch
from tqdm import tqdm
import argparse
import os
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.get_dataset import get_dataset
from utils.get_models import get_models
from attack_one_methodl import MI_FGSM_single
import xgboost as xgb
def get_args():
    parser = argparse.ArgumentParser(description='Standard MI-FGSM Attack (Single Surrogate)')
    parser.add_argument('--root_path', type=str, default='/raid/wl_raid/NIDS-minmax/artimis')
    parser.add_argument('--dataset', type=str, default='cicids2017')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-worker', type=int, default=4)
    parser.add_argument('--gpu-id', type=int, default=0)

    # attack params
    parser.add_argument('--eps', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--momentum', type=float, default=1.0)

    #parser.add_argument('--target-models', nargs='+', default=['LR'])
    parser.add_argument('--target-models', nargs='+', default=['LR','DeepNet','AlertNet', 'LeNet','LSTM','RF','XGBOOST'])
    parser.add_argument('--surrogate-model', type=str, default='DeepNet')
    return parser.parse_args()

def main(args):
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    dataloader = get_dataset(args)
    models, metrics = get_models(args, device)

    surrogate_model_info = models[args.surrogate_model]
    surrogate_model = surrogate_model_info['model']
    surrogate_type = surrogate_model_info['type']
    target_models = args.target_models

    print(f"🎯 Surrogate model: {args.surrogate_model}")
    print(f"🎯 Target models: {target_models}")

    orig_all, adv_all, delta_all = [], [], []

    for batch_idx, (data, label, _) in enumerate(tqdm(dataloader)):
        data, label = data.to(device), label.to(device)
        adv_samples = MI_FGSM_single(
            surrogate_model, surrogate_type, data, label, args=args, num_iter=args.iters
        )

        orig_all.append(data.cpu())
        adv_all.append(adv_samples.cpu())
        delta_all.append((adv_samples - data).cpu())

        for model_name in target_models:
            model_info = models[model_name]
            model, model_type = model_info['model'], model_info['type']
            #print(f'🎯 Evaluating: {model_name} ({model_type})')

            if model_type == 'sklearn':
                # RF / DecisionTree
                pred_clean = torch.tensor(model.predict(data.cpu().numpy())).long().to(device)
                pred_adv = torch.tensor(model.predict(adv_samples.cpu().numpy())).long().to(device)

            elif model_type == 'xgboost':
                # XGBoost Booster 输出为概率，需要 > 0.5 二分类判断
                # 加载特征名（与训练时一致）
                csv_path = '/raid/wl_raid/NIDS-master/data/brute_force/bc_data.csv'
                df = pd.read_csv(csv_path, nrows=1)
                feature_names = list(df.columns)
                feature_names.remove('Attack Type')  # 移除标签列
                # 创建 DMatrix，并附带特征名
                dmatrix_clean = xgb.DMatrix(data.cpu().numpy(), feature_names=feature_names)
                dmatrix_adv = xgb.DMatrix(adv_samples.cpu().numpy(), feature_names=feature_names)
                # 使用训练模型进行预测（输出概率）
                prob_clean = model.predict(dmatrix_clean)
                prob_adv = model.predict(dmatrix_adv)
                # 二分类判断 > 0.5
                pred_clean = torch.tensor((prob_clean > 0.5).astype(int)).long().to(device)
                pred_adv = torch.tensor((prob_adv > 0.5).astype(int)).long().to(device)

            else:
                # PyTorch 模型
                model.eval()
                with torch.no_grad():
                    out_clean = model(data)
                    out_adv = model(adv_samples)

                if model_type == 'logit':
                    pred_clean = out_clean.argmax(dim=1)
                    pred_adv = out_adv.argmax(dim=1)
                elif model_type == 'sigmoid':
                    pred_clean = (out_clean > 0.5).long().squeeze()
                    pred_adv = (out_adv > 0.5).long().squeeze()
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")

            # 统计评估指标
            correct_clean = (pred_clean == label).sum().item()
            correct_adv = (pred_adv == label).sum().item()
            metrics[model_name].update(correct_clean, correct_adv, label.size(0))

    # 拼接目标模型名组合
    target_model_str = '_'.join(args.target_models)
    result_dir = os.path.join(args.root_path, 'one_results', f'{args.surrogate_model}_to_{target_model_str}')
    os.makedirs(result_dir, exist_ok=True)

    np.savetxt(os.path.join(result_dir, 'orig_all.csv'), torch.cat(orig_all).numpy(), delimiter=",")
    np.savetxt(os.path.join(result_dir, 'adv_all.csv'), torch.cat(adv_all).numpy(), delimiter=",")
    np.savetxt(os.path.join(result_dir, 'delta_all.csv'), torch.cat(delta_all).numpy(), delimiter=",")
    save_path = os.path.join(result_dir, "final_metrics.txt")
    # 创建文件对象并写入 header
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Surrogate model: {args.surrogate_model}\n")
        f.write("-" * 73 + "\n")
        f.write("|\tModel name\t|\tNat. Acc. (%)\t|\tAdv. Acc. (%)\t|\tASR. (%)\t|\n")
        for model_name in args.target_models:
            m = metrics[model_name]
            line = (f"|\t{model_name.ljust(10)}\t"
                    f"|\t{round(m.clean_acc * 100, 2):<13}\t"
                    f"|\t{round(m.adv_acc * 100, 2):<13}\t"
                    f"|\t{round(m.attack_rate * 100, 2):<8}\t|\n")
            f.write(line)
        f.write("-" * 73 + "\n")
    print(f"✅ 已保存最终指标至 {save_path}")

    print(f"🎯 Surrogate model: {args.surrogate_model}")
    print('-' * 73)
    print('|\tModel name\t|\tNat. Acc. (%)\t|\tAdv. Acc. (%)\t|\tASR. (%)\t|')
    for model_name in target_models:
        m = metrics[model_name]
        print(f"|\t{model_name.ljust(10)}\t"
              f"|\t{round(m.clean_acc * 100, 2):<13}\t"
              f"|\t{round(m.adv_acc * 100, 2):<13}\t"
              f"|\t{round(m.attack_rate * 100, 2):<8}\t|")
    print('-' * 73)

if __name__ == '__main__':
    args = get_args()
    main(args)
