import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from tqdm import trange, tqdm
import pandas as pd
import os
import numpy as np
import json
from json import JSONEncoder
import joblib
import copy
from sklearn.preprocessing import MinMaxScaler
from tree import leaf_tuple_attack
import xgboost as xgb
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import random
def _compute_model_gradient(model_info, x_tensor, labels, feature_names=None):
    model, mtype = model_info['model'], model_info['type']
    if mtype in ('sklearn',):
        return fd_grad_tree(model, x_tensor, labels, mtype='sklearn')
    elif mtype == 'xgboost':
        return fd_grad_tree(model, x_tensor, labels, mtype='xgboost',
                            feature_names=feature_names)
    else:
        model.train()
        logits = model(x_tensor)
        loss   = F.cross_entropy(logits, labels)
        grad   = torch.autograd.grad(loss, x_tensor, retain_graph=True)[0]
        return grad

def _model_logits(model_info, x_tensor, feature_names=None):
    model, mtype = model_info['model'], model_info['type']

    # --------- sklearn (RF/DT) -------------
    if mtype in ('sklearn',):
        X_np = x_tensor.detach().cpu().numpy()
        prob = model.predict_proba(X_np)  # (N, 2) 已满足
        prob = torch.from_numpy(prob).to(x_tensor.device)

    # --------- xgboost -------------
    elif mtype == 'xgboost':
        X_np = x_tensor.detach().cpu().numpy()
        dmat = xgb.DMatrix(X_np, feature_names=feature_names)
        prob_pos = model.predict(dmat)  # (N,)  or  (1,)
        prob_pos = torch.from_numpy(prob_pos).to(x_tensor.device).unsqueeze(1)
        prob = torch.cat([1 - prob_pos, prob_pos], dim=1)  # (N,2)

    # --------- PyTorch NN -------------
    else:
        return model(x_tensor)  # logits already

    return torch.log(prob + 1e-12)


def load_scaler(joblib_path: str):
    scaler = joblib.load(joblib_path)
    return scaler

def recompute_rates_from_scaler(x_tensor: torch.Tensor, scaler: MinMaxScaler) -> torch.Tensor:
    x_numpy = x_tensor.detach().cpu().numpy()
    x_raw = scaler.inverse_transform(x_numpy)

    duration = np.clip(x_raw[:, 1], 1e-6, None)
    fwd_packets = x_raw[:, 2]
    bwd_packets = x_raw[:, 3]
    fwd_bytes = x_raw[:, 4]
    bwd_bytes = x_raw[:, 5]

    x_raw[:, 22] = (fwd_bytes + bwd_bytes) / duration
    x_raw[:, 23] = (fwd_packets + bwd_packets) / duration
    x_raw[:, 24] = fwd_packets / duration
    x_raw[:, 25] = bwd_packets / duration
    x_raw[:, 26] = bwd_packets / np.clip(fwd_packets, 1e-6, None)

    df_tmp = pd.read_csv('../data/2018/brute_force/bf.csv', nrows=1)
    if 'label' in df_tmp.columns:
        df_tmp = df_tmp.drop(columns=['label'])
    feature_names = df_tmp.columns.tolist()

    x_df = pd.DataFrame(x_raw, columns=feature_names)

    x_rescaled = scaler.transform(x_df)

    x_rescaled = np.clip(x_rescaled, 0.0, 1.0)
    return torch.tensor(x_rescaled, dtype=torch.float32, device=x_tensor.device)

def apply_constraints(x_adv, scaler):
    x = x_adv.clone()
    def enforce_triplet(x, max_idx, mean_idx, min_idx):
        max_val = torch.max(torch.stack([x[:, max_idx], x[:, mean_idx], x[:, min_idx]]), dim=0).values
        min_val = torch.min(torch.stack([x[:, max_idx], x[:, mean_idx], x[:, min_idx]]), dim=0).values
        mean_val = ((x[:, max_idx] + x[:, mean_idx] + x[:, min_idx]) / 3).clamp(min=min_val, max=max_val)
        x[:, max_idx] = torch.max(torch.stack([x[:, max_idx], mean_val, min_val]), dim=0).values
        x[:, min_idx] = torch.min(torch.stack([x[:, min_idx], mean_val, max_val]), dim=0).values
        x[:, mean_idx] = mean_val
        return x

    triplet_groups = [
        (6, 8, 7),  # fwd_payload
        (10, 12, 11),  # bwd_payload
        (15, 16, 14),  # payload
        (29, 27, 30),  # packet_IAT
        (34, 32, 35),  # fwd_packets_IAT
        (39, 37, 40),  # bwd_packets_IAT
        (47, 45, 48),  # active
        (51, 49, 52),  # idle
    ]
    for max_idx, mean_idx, min_idx in triplet_groups:
        x = enforce_triplet(x, max_idx, mean_idx, min_idx)
    x = recompute_rates_from_scaler(x, scaler)
    return x

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):  # 如果需要处理PyTorch张量
            return obj.detach().cpu().numpy().tolist()
        return super().default(obj)
def clip_by_tensor(t, t_min, t_max):
    return torch.clamp(t, min=t_min, max=t_max)

class Weight_Selection(nn.Module):
    def __init__(self, weight_len) -> None:
        super(Weight_Selection, self).__init__()
        self.weight = nn.Parameter(torch.ones([weight_len]))

    def forward(self, x, index):
        return self.weight[index] * x
def normalize_grad(grad):
    return grad / (torch.mean(torch.abs(grad), dim=1, keepdim=True) + 1e-8)

def inverse_logit(p):
    return np.log(p / (1 - p + 1e-8))
def prob_to_logits(probs, eps=1e-8):
    probs = np.clip(probs, eps, 1 - eps)
    logits = np.log(probs)
    return logits
def zoo_estimate_gradient_rf(model: RandomForestClassifier,
                             x: torch.Tensor,
                             epsilon: float = 1e-4) -> torch.Tensor:
    device = x.device
    x_np = x.detach().cpu().numpy()
    batch_size, n_features = x_np.shape

    eye = np.eye(n_features)
    x_perturbed = np.vstack([
        x_np + epsilon * eye,
        x_np - epsilon * eye
    ]).reshape(-1, n_features)
    # Batch Prediction (Leveraging RF's Parallel Capabilities)
    prob = model.predict_proba(x_perturbed)[:, 1]

    prob = prob.reshape(2, n_features, batch_size)
    grad = (prob[0] - prob[1]) / (2 * epsilon)  # [n_features, batch_size]

    return torch.tensor(grad.T, device=device)

import xgboost as xgb
def zoo_estimate_gradient_xgb(model: xgb.Booster,
                              x: torch.Tensor,
                              epsilon: float = 1e-4,
                              feature_names: list = None) -> torch.Tensor:
    device = x.device
    x_np = x.detach().cpu().numpy()
    batch_size, n_features = x_np.shape

    eye = np.eye(n_features)
    x_perturbed = np.vstack([
        x_np + epsilon * eye,
        x_np - epsilon * eye
    ]).reshape(-1, n_features)

    dmatrix = xgb.DMatrix(x_perturbed, feature_names=feature_names)

    prob = model.predict(dmatrix)
    prob = prob.reshape(2, n_features, batch_size)

    grad = (prob[0] - prob[1]) / (2 * epsilon)
    return torch.tensor(grad.T, device=device)

def fd_grad_tree(model,
                 x_tensor: torch.Tensor,
                 label: torch.Tensor,
                 mtype: str,
                 feature_names=None,
                 h: float = 1e-3) -> torch.Tensor:
    device = x_tensor.device
    x_np = x_tensor.detach().cpu().numpy()
    n_feat = x_np.shape[1]
    grad   = np.zeros_like(x_np)

    def _loss(arr):
        if mtype == 'sklearn':
            prob = model.predict_proba(arr)[0]            # (2,)
        else:  # xgboost
            dmat = xgb.DMatrix(arr, feature_names=feature_names)
            p1 = model.predict(dmat)[0]
            prob = np.array([1 - p1, p1])

        lbl = label.item()
        return -np.log(prob[lbl] + 1e-12)

    base_loss = _loss(x_np)

    for j in range(n_feat):
        x_plus, x_minus = x_np.copy(), x_np.copy()
        x_plus[0, j]  += h
        x_minus[0, j] -= h
        grad[0, j] = (_loss(x_plus) - _loss(x_minus)) / (2 * h)

    return torch.tensor(grad, dtype=torch.float32, device=device)


def Reweight(surrogate_models, samples, labels, result_dir, args, weight_selection, optimizer, scaler,
             return_weights=False, model_names=None):

    samples = samples.to(torch.float32)
    eps = args.eps
    alpha = args.walpha
    beta = alpha
    momentum = args.momentum
    num_iter = args.witers
    sample_min = clip_by_tensor(samples - eps, 0.0, 1.0)
    sample_max = clip_by_tensor(samples + eps, 0.0, 1.0)

    times = args.miters
    m = len(surrogate_models)
    sample_times = m * times
    # weight_selection = Weight_Selection(m).to(samples.device)
    # optimizer = torch.optim.SGD(weight_selection.parameters(), lr=2e-2, weight_decay=2e-3)
    grad = 0
    weight_evolution = []  # Used to record the weight of each round
    logit_records = []

    if model_names is None:
        model_names = getattr(args, 'surrogate_models', [f'model_{i}' for i in range(len(surrogate_models))])

    if len(model_names) != len(surrogate_models):
        print(f"[Warning] Number of model names({len(model_names)})does not match the number of models({len(surrogate_models)}).Use the default name")
        model_names = [f'model_{i}' for i in range(len(surrogate_models))]

    xgb_feature_names = None
    if any(info.get('type') == 'xgboost' for info in surrogate_models):
        csv_path = '../data/2018/brute_force/bf.csv'
        df = pd.read_csv(csv_path, nrows=1)
        xgb_feature_names = list(df.columns)
        xgb_feature_names.remove('label')

    for outer_iter in trange(num_iter, desc="🔥 Outer Attack Iterations", ncols=100):
        x_inner = samples.clone().detach()
        x_before = samples.clone()
        noise_inner_all = torch.zeros([sample_times, *samples.shape]).to(samples.device)
        grad_inner = torch.zeros_like(samples)
        options = []

        for _ in range(int(sample_times / m)):
            options_single = list(range(m))
            np.random.shuffle(options_single)
            options.extend(options_single)

        for j in range(sample_times):
            option = options[j]
            model_info = surrogate_models[option]
            model = model_info['model']
            model_type = model_info['type']

            if model_type == 'sklearn':
                x_rf = x_inner.clone().detach()
                noise_im_inner = zoo_estimate_gradient_rf(model, x_rf)
                noise_im_inner = weight_selection(noise_im_inner, option)
                # M = 5
                # sigma_input = 0.05
                # grads = []
                # for _ in range(M):
                #     x_rf = x_inner.clone().detach()
                #     noise = torch.randn_like(x_rf) * sigma_input
                #     x_noisy = torch.clamp(x_rf + noise, 0.0, 1.0)
                #     grad_est = zoo_estimate_gradient_rf(model, x_noisy)
                #     grads.append(grad_est)
                # avg_grad = torch.stack(grads, dim=0).mean(dim=0)
                # noise_im_inner = weight_selection(avg_grad, option)

            elif model_type == 'xgboost':
                x_xgb = x_inner.clone().detach()
                noise_im_inner = zoo_estimate_gradient_xgb(model, x_xgb,
                                                           feature_names=xgb_feature_names)
                noise_im_inner = weight_selection(noise_im_inner, option)

                # M = 5
                # sigma_input = 0.05
                # grads = []
                # for _ in range(M):
                #     x_xgb = x_inner.clone().detach()
                #     noise = torch.randn_like(x_xgb) * sigma_input
                #     x_noisy = torch.clamp(x_xgb + noise, 0.0, 1.0)
                #     grad_est = zoo_estimate_gradient_xgb(model, x_noisy, feature_names=xgb_feature_names)
                #     grads.append(grad_est)
                # avg_grad = torch.stack(grads, dim=0).mean(dim=0)
                # noise_im_inner = weight_selection(avg_grad, option)

            else:
                # model.train()
                # x_1 = x_inner.clone().detach().requires_grad_(True)
                # out_logits = model(x_1)
                # out = weight_selection(out_logits, option)
                # loss = F.cross_entropy(out, labels)
                # noise_im_inner = torch.autograd.grad(loss, x_1)[0]
                # x_1.requires_grad = False

                # === using Bayesian sampling ===
                M = 1
                sigma = 0
                grad_sum = torch.zeros_like(x_inner)
                for _ in range(M):
                    x_1 = x_inner.clone().detach().requires_grad_(True)

                    sampled_model = copy.deepcopy(model)
                    for p in sampled_model.parameters():
                        if p.requires_grad:
                            p.data += torch.randn_like(p) * sigma
                    for module in sampled_model.modules():
                        if isinstance(module, torch.nn.LSTM):
                            module.flatten_parameters()
                    sampled_model.train()
                    out_logits = sampled_model(x_1)
                    out = weight_selection(out_logits, option)
                    loss = F.cross_entropy(out, labels)
                    grad = torch.autograd.grad(loss, x_1)[0]
                    grad_sum += grad
                noise_im_inner = grad_sum / M

            no_perturb_mask = torch.ones_like(noise_im_inner)
            no_perturb_mask[:, 53:60] = 0
            noise_im_inner = noise_im_inner * no_perturb_mask
            noise_normed = normalize_grad(noise_im_inner)
            grad_inner += noise_normed
            x_inner = x_inner + beta * torch.sign(grad_inner)
            x_inner = clip_by_tensor(x_inner, sample_min, sample_max)
            x_inner = apply_constraints(x_inner, scaler)
            noise_inner_all[j] = grad_inner.clone()

        # Initialize group_logits outside the inner loop
        group_logits = torch.zeros(x_inner.size(0), 2).to(samples.device)
        logit_row = {"iteration": outer_iter}
        # Compute group_logits after inner loop
        for k, (model_info, model_name) in enumerate(zip(surrogate_models, model_names)):
            model = model_info['model']
            model_type = model_info['type']
            if model_type == 'sklearn':
                x_inner_cpu = x_inner.detach().cpu().numpy()
                logits = prob_to_logits(model.predict_proba(x_inner_cpu))
                logits = torch.tensor(logits, device=samples.device)

            elif model_type == 'xgboost':
                x_inner_cpu = x_inner.detach().cpu().numpy()
                dmatrix = xgb.DMatrix(x_inner_cpu, feature_names=xgb_feature_names)
                prob = model.predict(dmatrix)
                logits = prob_to_logits(np.column_stack([1 - prob, prob]))
                logits = torch.tensor(logits, device=samples.device)

            else:
                model.train()
                logits = model(x_inner).to(samples.device)
                # individual_logits[model_name] = logits.detach().cpu().numpy().tolist()
                # group_logits += weight_selection(logits, k)

            logit_row[model_name] = logits.mean(dim=0).tolist()
            group_logits += weight_selection(logits, k)
        logit_records.append(logit_row)
        group_logits /= m
        loss_raw = F.cross_entropy(group_logits, labels)
        loss_val = torch.clamp(loss_raw, min=1e-6)
        outer_loss = -torch.log(loss_val)

        # Zero gradients before backward pass
        optimizer.zero_grad()
        outer_loss.backward()
        optimizer.step()

        noise_1 = noise_inner_all[-1].clone()
        no_perturb_mask = torch.ones_like(noise_1)
        no_perturb_mask[:, 53:60] = 0
        noise_1 = noise_1 * no_perturb_mask
        noise = normalize_grad(noise_1)
        grad = noise + momentum * grad
        samples = x_before + alpha * torch.sign(grad)
        samples = clip_by_tensor(samples, sample_min, sample_max)
        samples = apply_constraints(samples, scaler)
        weight_evolution.append(weight_selection.weight.detach().cpu().clone())

    # save logit
    logit_df = pd.DataFrame(logit_records)
    logit_path = os.path.join(result_dir, 'meta_smer_logit_records.csv')
    if os.path.exists(logit_path):
        logit_df.to_csv(logit_path, mode='a', index=False, header=True)
    else:
        logit_df.to_csv(logit_path, index=False)
    samples = samples.detach()
    if return_weights:
        return samples, torch.stack(weight_evolution), weight_selection, optimizer
    else:
        return samples

def META_RE_ARTIMIS(surrogate_models, samples, labels, result_dir, args, return_weights=False,sample_id=None):
    scaler_path = "../data/2018/brute_force/bf.joblib"
    scaler = load_scaler(scaler_path)

    I = args.meta_iters
    alpha_outer = args.meta_alpha
    eps = args.eps

    x_meta = samples.clone().detach().to(torch.float32)
    x_meta_min = clip_by_tensor(x_meta - eps, 0.0, 1.0)
    x_meta_max = clip_by_tensor(x_meta + eps, 0.0, 1.0)

    all_meta_weights = []
    all_logs = []
    metric_records = []

    m = len(surrogate_models)
    weight_selection = Weight_Selection(m).to(samples.device)
    optimizer = torch.optim.SGD(weight_selection.parameters(), lr=1e-3, weight_decay=2e-3)

    full_model_names = getattr(args, 'surrogate_models', [f'model_{i}' for i in range(m)])

    xgb_feature_names = None
    if any(info.get('type') == 'xgboost' for info in surrogate_models):
        csv_path = '../data/2018/brute_force/bf.csv'
        df = pd.read_csv(csv_path, nrows=1)
        xgb_feature_names = list(df.columns)
        xgb_feature_names.remove('label')

    for meta_iter in range(I):
        test_index = random.randint(0, len(surrogate_models) - 1)
        meta_test_model_info = surrogate_models[test_index]
        meta_test_model_name = full_model_names[test_index]

        meta_train_models = surrogate_models[:test_index] + surrogate_models[test_index + 1:]
        meta_train_model_names = full_model_names[:test_index] + full_model_names[test_index + 1:]
        adv_inner, weight_history, weight_selection, optimizer = Reweight(
            surrogate_models=meta_train_models,
            samples=x_meta,
            labels=labels,
            result_dir=result_dir,
            args=args,
            weight_selection=weight_selection,
            optimizer=optimizer,
            return_weights=True,
            scaler=scaler,
            model_names=meta_train_model_names
        )
        all_meta_weights.append(weight_history.detach().cpu())

        log_rows = []
        num_iter = weight_history.shape[0]
        full_weights = weight_selection.weight.detach().cpu().tolist()

        for iter_idx in range(num_iter):
            row = {
                'meta_task': meta_iter,
                'iteration': iter_idx,
                'meta_test_model': meta_test_model_name
            }
            weights_at_iter = weight_history[iter_idx].tolist()
            for model_idx, model_name in enumerate(full_model_names):
                row[model_name] = weights_at_iter[model_idx]
            log_rows.append(row)
        all_logs.extend(log_rows)

        x_test = adv_inner.clone().detach().requires_grad_(True)
        model = meta_test_model_info['model']
        model_type = meta_test_model_info['type']

        x_metrics = adv_inner.detach().clone().requires_grad_(True)
        per_grads = []
        per_losses = []

        for info in surrogate_models:
            # if info['type'] in ('sklearn', 'xgboost'):
            #     continue
            # ① logits / loss
            logits_i = _model_logits(info, x_metrics, feature_names=xgb_feature_names)
            loss_i = F.cross_entropy(logits_i, labels)
            per_losses.append(loss_i)

            # ② grad
            grad_i = _compute_model_gradient(info, x_metrics, labels, feature_names=xgb_feature_names)
            per_grads.append(grad_i)

        # —— Ensemble Loss ——
        ens_loss = torch.stack(per_losses).mean().item()

        # —— Ensemble Grad ——
        grad_ens = torch.stack(per_grads, dim=0).mean(dim=0)
        # grad_ens_norm = F.normalize(grad_ens.flatten(), dim=0, eps=1e-8)
        # grad_meta_test_norm = F.normalize(grad_meta_test.flatten(), dim=0, eps=1e-8)
        # —— PCS（Grad 同向性）——
        pcs_vals = []
        for i in range(len(per_grads)):
            for j in range(i + 1, len(per_grads)):
                pcs_vals.append(
                    F.cosine_similarity(
                        per_grads[i].flatten(), per_grads[j].flatten(), dim=0, eps=1e-8
                    )
                )
        pcs = torch.stack(pcs_vals).mean().item() if pcs_vals else 0.0

        if model_type == 'sklearn':
            grad_meta_test = fd_grad_tree(model, x_test, labels,
                                          mtype='sklearn')
        elif model_type == 'xgboost':
            grad_meta_test = fd_grad_tree(model, x_test, labels,
                                          mtype='xgboost',
                                          feature_names=xgb_feature_names)
            #grad_meta_test = zoo_estimate_gradient_xgb(model, x_test, feature_names=xgb_feature_names)
        else:
            model.train()
            logits = model(x_test)
            loss = F.cross_entropy(logits, labels)
            grad_meta_test = torch.autograd.grad(loss, x_test)[0]
        ###  <<< NEW — 计算 GA，并写入记录列表 >>>
        ga_val = F.cosine_similarity(
            grad_ens.flatten(), grad_meta_test.flatten(), dim=0, eps=1e-8
        ).item()
        metric_records.append({
            'sample_id': sample_id,
            'meta_task': meta_iter,
            'ensemble_loss': ens_loss,
            'grad_align': ga_val,
            'pcs': pcs,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        no_perturb_mask = torch.ones_like(grad_meta_test)
        no_perturb_mask[:, 53:60] = 0
        grad_meta_test = grad_meta_test * no_perturb_mask
        noise_normed = normalize_grad(grad_meta_test)
        perturb = alpha_outer * torch.sign(noise_normed)
        x_meta = x_meta + perturb
        x_meta = clip_by_tensor(x_meta, x_meta_min, x_meta_max)
        x_meta = apply_constraints(x_meta, scaler)
        # After each meta-task is completed, record the current global weight
        final_row = {
            'meta_task': meta_iter,
            'iteration': 'final',
            'meta_test_model': meta_test_model_name
        }
        for model_idx, model_name in enumerate(full_model_names):
            final_row[model_name] = full_weights[model_idx]
        all_logs.append(final_row)
    x_meta = x_meta.detach()

    df_metric = pd.DataFrame(metric_records)
    metric_path = os.path.join(result_dir, 'meta_smer_metrics.csv')
    if os.path.exists(metric_path):
        df_metric.to_csv(metric_path, mode='a', index=False, header=False)
    else:
        df_metric.to_csv(metric_path, index=False)
    print(f"✅ Metric log saved to: {metric_path}")

    if return_weights:
        df_logs = pd.DataFrame(all_logs)
        log_path = os.path.join(result_dir, 'meta_smer_weight_logs.csv')
        if os.path.exists(log_path):
            # 加一行空行用于视觉分隔（可选）
            with open(log_path, 'a') as f:
                f.write('\n')
            df_logs.to_csv(log_path, mode='a', index=False, header=False)
        else:
            df_logs.to_csv(log_path, index=False, header=True)
        print(f"✅ The current meta-task log is appended and saved to: {log_path}")
        return x_meta, torch.stack(all_meta_weights)
    else:
        return x_meta


