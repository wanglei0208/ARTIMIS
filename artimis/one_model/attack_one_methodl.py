import torch
import torch.nn.functional as F
import numpy as np

def clip_by_tensor(t, t_min, t_max):
    return torch.clamp(t, min=t_min, max=t_max)

def normalize_grad(grad):
    return grad / (torch.mean(torch.abs(grad), dim=1, keepdim=True) + 1e-8)

def MI_FGSM_single(surrogate_model, model_type,samples, labels, args, num_iter=10):
    eps = args.eps
    alpha = args.alpha
    momentum = args.momentum

    sample_min = clip_by_tensor(samples - eps, 0.0, 1.0)
    sample_max = clip_by_tensor(samples + eps, 0.0, 1.0)

    print("[Debug] samples dtype:", samples.dtype)
    print("[Debug] samples shape:", samples.shape)

    grad = 0
    for i in range(num_iter):
        x_adv = samples.clone().detach().requires_grad_()

        output = surrogate_model(x_adv)

        # 🔁 根据模型类型选择不同的 loss
        if model_type == 'sigmoid':  # LR、MLP二分类输出概率
            label = labels.float()
            if label.ndim == 1:
                label = label.unsqueeze(1)
            loss = F.binary_cross_entropy(output, label)

        else:  # 默认 logit 多分类模型
            loss = F.cross_entropy(output, labels)
        grad_cur = torch.autograd.grad(loss, x_adv)[0]
        grad_cur = normalize_grad(grad_cur)

        grad = grad * momentum + grad_cur
        adv_x = samples + alpha * torch.sign(grad)
        adv_x = clip_by_tensor(adv_x, sample_min, sample_max)
        samples = adv_x.detach()

    return samples
