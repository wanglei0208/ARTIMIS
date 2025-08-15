import torch
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import sys, os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.get_dataset import get_dataset
from utils.get_models import get_models
from artimis import META_RE_ARTIMIS
import xgboost as xgb

def get_args():
    parser = argparse.ArgumentParser(description='artimis on NIDS')
    #parser.add_argument('--root_path', type=str, default='artimis',help='attack_root_path')
    parser.add_argument('--dataset', type=str, default='cicids2017',help='dataset name')
    parser.add_argument('--batch-size', type=int, default=1,help='batch size for attack')
    parser.add_argument('--num-worker', type=int, default=4)
    parser.add_argument('--attack_method', type=str, default='Meta_Re_artimis')
    parser.add_argument('--gpu-id', type=int, default=0,help='gpu_id')
    parser.add_argument('--tree', type=str, default='tree')
    # attack parameters
    parser.add_argument('--eps', type=float, default=0.05)
    parser.add_argument('--walpha', type=float, default=0.005)
    parser.add_argument('--witers', type=int, default=5)
    parser.add_argument('--miters', type=int, default=2)
    parser.add_argument('--momentum', type=float, default=1.0,help='momentum value for MI-FGSM')
    parser.add_argument('--meta-iters', type=int, default=10, help='outer meta task iterations I')
    parser.add_argument('--meta-alpha', type=float, default=0.005, help='outer step size for meta gradient update')
    parser.add_argument('--target-models', nargs='+', default=['RF-BB','XGBOOST-BB','AlexNet','AlertNet','DeepNet','IdsNet'])
    parser.add_argument('--surrogate-models', nargs='+', default=['MLP1', 'LeNet', 'LSTM','RF1','XGBOOST1'])
    #parser.add_argument('--surrogate-models', nargs='+', default=['RF1','RF2','RF3'])
    #parser.add_argument('--surrogate-models', nargs='+', default=['RF1','RF2','XGBOOST1','XGBOOST2'])
    #parser.add_argument('--surrogate-models', nargs='+', default=['MLP1', 'MLP2','MLP3'])
    #parser.add_argument('--surrogate-models', nargs='+', default=['MLP1','LeNet','LSTM'])
    args = parser.parse_args()
    return args


def main(args):
    attack_map = {
        'Meta_Re_artimis': META_RE_ARTIMIS
    }

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    result_dir = "results/closed_set"
    os.makedirs(result_dir, exist_ok=True)
    
    dataloader = get_dataset(args)
    models, metrics = get_models(args, device)

    all_model_names = list(models.keys())

   
    if args.target_models is None:
        target_model_names = all_model_names
    else:
        target_model_names = args.target_models

    
    if args.surrogate_models is None:
        surrogate_model_names = [m for m in all_model_names if m not in target_model_names]
    else:
        surrogate_model_names = args.surrogate_models

    
    print(f"\n🎯 target_models: {target_model_names}")
    print(f"🎭 surrogate_models: {surrogate_model_names}\n")

    
    surrogate_models = [
        {
            'model': models[name]['model'],
            'type': models[name]['type']  
        }
        for name in surrogate_model_names
    ]
    
    orig_all = []
    adv_all = []
    delta_all = []
    all_weight_logs = []  # shape: [batch, iter, model]
    
    target_model_str = '_'.join(args.target_models)
    surrogate_model_str = '_'.join(args.surrogate_models)
    for batch_idx, (data, label, _) in enumerate(tqdm(dataloader)):
        data, label = data.to(device), label.to(device)
        batch_size = label.size(0)

        # print("[Before attack] samples.shape:", data.shape)
        # print("[Before attack] samples.dtype:", data.dtype)
        # print("[Before attack] samples content:", data)

        
        adv_samples, weight_history = attack_map[args.attack_method](
            surrogate_models, data, label, result_dir,args=args,return_weights=True)
        all_weight_logs.append(weight_history)
        orig_all.append(data.detach().cpu())
        adv_all.append(adv_samples.detach().cpu())
        delta_all.append((adv_samples - data).detach().cpu())

        
        adv_samples = adv_samples.to(dtype=torch.float32)  
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
                csv_path = '../data/2018/brute_force/bf.csv'
                df = pd.read_csv(csv_path, nrows=1)
                feature_names = list(df.columns)
                feature_names.remove('label') 
               
                dmatrix_clean = xgb.DMatrix(data.cpu().numpy(), feature_names=feature_names)
                dmatrix_adv = xgb.DMatrix(adv_samples.cpu().numpy(), feature_names=feature_names)
               
                prob_clean = model.predict(dmatrix_clean)
                prob_adv = model.predict(dmatrix_adv)
                
                pred_clean = torch.tensor((prob_clean > 0.5).astype(int)).long().to(device)
                pred_adv = torch.tensor((prob_adv > 0.5).astype(int)).long().to(device)

            else:
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

            correct_clean = (pred_clean == label).sum().item()
            correct_adv = (pred_adv == label).sum().item()
            metrics[model_name].update(correct_clean, correct_adv, batch_size)


    
    np.savetxt(os.path.join(result_dir, 'orig_all.csv'), torch.cat(orig_all).numpy(), delimiter=",")
    np.savetxt(os.path.join(result_dir, 'adv_all.csv'), torch.cat(adv_all).numpy(), delimiter=",")
    np.savetxt(os.path.join(result_dir, 'delta_all.csv'), torch.cat(delta_all).numpy(), delimiter=",")
    print(f"✅ All adversarial samples have been saved to {result_dir}")


    save_path = os.path.join(result_dir, "final_metrics.txt")
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Surrogate model: {args.surrogate_models}\n")
        f.write("-" * 73 + "\n")
        f.write("|\tTarget model name\t|\tNat. Acc. (%)\t|\tAdv. Acc. (%)\t|\tASR. (%)\t|\n")
        for model_name in target_model_names:
            m = metrics[model_name]
            line = (f"|\t{model_name.ljust(17)}\t"
                    f"|\t{round(m.clean_acc * 100, 2):<13}\t"
                    f"|\t{round(m.adv_acc * 100, 2):<13}\t"
                    f"|\t{round(m.attack_rate * 100, 2):<8}\t|\n")
            f.write(line)
        f.write("-" * 73 + "\n")
    print(f"✅ The final metric has been saved to {save_path}")

    print(f"🎯 Surrogate model: {args.surrogate_models}")
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
