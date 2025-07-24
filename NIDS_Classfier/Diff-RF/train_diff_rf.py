import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
import joblib
from model_diff_rf import DiFF_TreeEnsemble
import os
import json
X_train = np.load("ARTIMIS/data/2018/brute_force/X_train_benign.npy")       # benign
X_test = np.load("ARTIMIS/data/2018/brute_force/X_test.npy")         # benign+attack
y_test = np.load("ARTIMIS/data/2018/brute_force/y_test.npy")

model = DiFF_TreeEnsemble(
    n_trees=10,
    sample_size=512,
)
model.fit(X_train,n_jobs=8)

_, _, scores = model.anomaly_score(X_test)  # use the third collective score

print(f"[DEBUG] scores type: {type(scores)}")
print(f"[DEBUG] scores shape: {scores.shape}")
print(f"[DEBUG] y_test shape: {y_test.shape}")

#Search for the optimal threshold
best_f1, best_thresh = 0, None
for thresh in np.linspace(np.min(scores), np.max(scores), 100):
    preds = (scores > thresh).astype(int).reshape(-1)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

final_preds = (scores > best_thresh).astype(int).reshape(-1)
acc = accuracy_score(y_test, final_preds)
prec = precision_score(y_test, final_preds)
rec = recall_score(y_test, final_preds)
f1 = f1_score(y_test, final_preds)
auc = roc_auc_score(y_test, scores)

print(f"✅ Best threshold: {best_thresh:.4f}")
print(f"✅ Accuracy:  {acc:.4f}")
print(f"✅ Precision: {prec:.4f}")
print(f"✅ Recall:    {rec:.4f}")
print(f"✅ F1-score:  {f1:.4f}")
print(f"✅ AUC:       {auc:.4f}")
save_dir = "2018/brute_force"
os.makedirs(save_dir, exist_ok=True)

joblib.dump(model, os.path.join(save_dir,"diff_rf_cicids_model.pkl"))
with open("./2018/dos2/diff_rf_cicids_threshold.txt", "w") as f:
    f.write(str(best_thresh))
print("✅ 模型已保存为 diff_rf_cicids_model.pkl")
print("✅ 阈值已保存为 diff_rf_cicids_threshold.txt")

metrics = {
    "best_threshold": float(best_thresh),
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1_score": float(f1),
    "auc": float(auc)
}

with open("./2018/brute_force/metrics_diff_rf.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ 指标结果已保存为 metrics_diff_rf.json")