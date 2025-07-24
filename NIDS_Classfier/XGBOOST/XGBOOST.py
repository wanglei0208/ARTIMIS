import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import os
import pandas as pd

X_train = np.load('ARTIMIS/data/2018/brute_force/X_train.npy')
y_train = np.load('ARTIMIS/data/2018/brute_force/y_train.npy')
X_test = np.load('ARTIMIS/data/2018/brute_force/X_test.npy')
y_test = np.load('ARTIMIS/data/2018/brute_force/y_test.npy')

csv_path = 'ARTIMIS/data/2018/brute_force2/bf2.csv'
df = pd.read_csv(csv_path)
feature_names = list(df.columns)
if 'label' in feature_names:
    feature_names.remove('label')
assert len(feature_names) == X_train.shape[1], "The number of feature names does not match the number of X_train columns"

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'verbosity': 0,
    'seed':20
}
num_boost_round = 15

model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

result_dir = './2018/brute_force/xg_n15_depth6'
os.makedirs(result_dir, exist_ok=True)
model.save_model(os.path.join(result_dir, 'xgboost4.json'))

importance_dict = model.get_score(importance_type='gain')
importance_df = pd.DataFrame([
    {'Feature': feat, 'Importance': importance_dict.get(feat, 0)}
    for feat in feature_names
]).sort_values(by='Importance', ascending=False)

importance_df.to_csv(os.path.join(result_dir, 'feature_importance.csv'), index=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'][:20][::-1], importance_df['Importance'][:20][::-1])
plt.xlabel('Gain')
plt.title('Top 20 Feature Importances (XGBoost)')
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'feature_importance_top20.png'))
plt.show()

final_preds = (model.predict(dtest) > 0.5).astype(int)
final_f1 = f1_score(y_test, final_preds)
final_acc = accuracy_score(y_test, final_preds)
final_precision = precision_score(y_test, final_preds)
final_recall = recall_score(y_test, final_preds)
tn, fp, fn, tp = confusion_matrix(y_test, final_preds).ravel()
final_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # 防止除以零

with open(os.path.join(result_dir, 'final_metrics.txt'), 'w') as f:
    f.write(f'Final Metrics for XGBoost (5 trees):\n')
    f.write(f'Final Accuracy: {final_acc:.4f}\n')
    f.write(f'Final Precision: {final_precision:.4f}\n')
    f.write(f'Final Recall: {final_recall:.4f}\n')
    f.write(f'Final F1 Score: {final_f1:.4f}\n')
    f.write(f'Final False Positive Rate (FPR): {final_fpr:.4f}\n')

f1_scores = []
precision_scores = []
recall_scores = []
fpr_scores = []
for n in range(1, num_boost_round + 1):
    y_pred_prob = model.predict(dtest, iteration_range=(0, n))
    y_pred = (y_pred_prob > 0.5).astype(int)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    f1_scores.append(f1)
    precision_scores.append(precision)
    recall_scores.append(recall)
    fpr_scores.append(fpr)

# ====== 绘图 ======
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_boost_round + 1), f1_scores, marker='o', label='F1 Score')
plt.plot(range(1, num_boost_round + 1), precision_scores, marker='o', label='Precision')
plt.plot(range(1, num_boost_round + 1), recall_scores, marker='o', label='Recall')
plt.plot(range(1, num_boost_round + 1), fpr_scores, marker='o', label='FPR')
plt.xlabel('Number of Trees')
plt.ylabel('Score')
plt.title('Metrics vs Number of Trees (XGBoost)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'metrics_vs_trees.png'))
plt.show()
