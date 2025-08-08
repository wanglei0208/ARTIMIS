##采样所有恶意流量
# import numpy as np
# import os
#
# # 设置路径
# feature_path = '/raid/wl_raid/NIDS-minmax/data/brute_force/X_test.npy'
# label_path = '/raid/wl_raid/NIDS-minmax/data/brute_force/y_test.npy'
# save_dir = '/raid/wl_raid/NIDS-minmax/data/brute_force/attack'
#
# # 加载数据
# X_test = np.load(feature_path)
# y_test = np.load(label_path)
#
# # 筛选恶意流量（标签 == 1）
# malicious_mask = (y_test == 1)
# X_malicious = X_test[malicious_mask]
# y_malicious = y_test[malicious_mask]
#
# # 保存结果
# np.save(os.path.join(save_dir, 'X_malicious_test.npy'), X_malicious)
# np.save(os.path.join(save_dir, 'y_malicious_test.npy'), y_malicious)
#
# print(f'✅ 已保存 {X_malicious.shape[0]} 条恶意流量样本：')
# print(f'- 特征路径: {os.path.join(save_dir, "X_malicious_test.npy")}')
# print(f'- 标签路径: {os.path.join(save_dir, "y_malicious_test.npy")}')

import numpy as np
import os

# 设置路径
feature_path = '/raid/wl_raid/ARTIMIS/data/2018/brute_force/X_test.npy'
label_path = '/raid/wl_raid/ARTIMIS/data/2018/brute_force/y_test.npy'
save_dir = '/raid/wl_raid/ARTIMIS/data/2018/brute_force/attack'

# 加载数据
X_test = np.load(feature_path)
y_test = np.load(label_path)

# 筛选恶意流量（标签 == 1）
malicious_mask = (y_test == 1)
X_malicious_all = X_test[malicious_mask]
y_malicious_all = y_test[malicious_mask]

# ✅ 随机采样 500 条恶意样本（不放回）
np.random.seed(42)  # 固定随机种子，确保可复现
indices = np.random.choice(len(X_malicious_all), size=20, replace=False)
X_malicious = X_malicious_all[indices]
y_malicious = y_malicious_all[indices]

# 保存结果
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, 'X_malicious_test_20.npy'), X_malicious)
np.save(os.path.join(save_dir, 'y_malicious_test_20.npy'), y_malicious)

print(f'✅ 已成功保存 500 条恶意流量样本：')
print(f'- 特征路径: {os.path.join(save_dir, "X_malicious_test_1.npy")}')
print(f'- 标签路径: {os.path.join(save_dir, "y_malicious_test_1.npy")}')
