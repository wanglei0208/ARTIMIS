import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import json
import joblib
import os
from tqdm import tqdm

# load data
X_train = np.load('./data/2018/brute_force/X_train.npy')
y_train = np.load('./data/2018/brute_force/y_train.npy')
X_test = np.load('./data/2018/brute_force/X_test.npy')
y_test = np.load('./data/2018/brute_force/y_test.npy')

# Extract feature names
csv_path = './data/2018/brute_force/bf.csv'
df = pd.read_csv(csv_path)
feature_names = list(df.columns)
if 'label' in feature_names:
    feature_names.remove('label')
assert len(feature_names) == X_train.shape[1], "Number of feature names does not match the number of columns in X_train"

# Set parameters
n_estimators = 8
max_depth = 6
max_features = 'sqrt'

# Create a directory to save the results
result_dir = 'ARTEMIS/RF/2018/brute_force/n8_depth6_sqrt'
os.makedirs(result_dir, exist_ok=True)

train_accuracies = []
val_accuracies = []
precision_list = []
recall_list = []
fpr_list = []

# Gradually increase the number of trees, train the model, and record performance
for i in tqdm(range(1, n_estimators + 1), desc="Training Random Forest"):
    rf = RandomForestClassifier(n_estimators=i, max_depth=max_depth,
                                max_features=max_features, random_state=20, n_jobs=-1)
    rf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, rf.predict(X_train))
    val_acc = accuracy_score(y_test, rf.predict(X_test))
    preds = rf.predict(X_test)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # 防止除以零

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    precision_list.append(precision)
    recall_list.append(recall)
    fpr_list.append(fpr)

    tqdm.write(f"Estimator: {i}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}, "
               f"Precision: {precision:.4f}, Recall: {recall:.4f}, FPR: {fpr:.4f}")

final_model = rf
joblib.dump(final_model, os.path.join(result_dir, 'random_forest_model.joblib'))
final_preds = final_model.predict(X_test)
final_f1 = f1_score(y_test, final_preds)
final_acc = accuracy_score(y_test, final_preds)
final_precision = precision_score(y_test, final_preds)
final_recall = recall_score(y_test, final_preds)
tn, fp, fn, tp = confusion_matrix(y_test, final_preds).ravel()
final_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

# Write all metrics to a single txt file
with open(os.path.join(result_dir, 'final_metrics.txt'), 'w') as f:
    f.write(f'Final Metrics for Random Forest (50 trees):\n')
    f.write(f'Final Accuracy: {final_acc:.4f}\n')
    f.write(f'Final Precision: {final_precision:.4f}\n')
    f.write(f'Final Recall: {final_recall:.4f}\n')
    f.write(f'Final F1 Score: {final_f1:.4f}\n')
    f.write(f'Final False Positive Rate (FPR): {final_fpr:.4f}\n')

# Save feature importances (from the final model)
importances = final_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

importance_df.to_csv(os.path.join(result_dir, 'feature_importance.csv'), index=False)

# Visualize the accuracy curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_estimators + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, n_estimators + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy vs. Number of Trees')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'accuracy_vs_estimators.png'))
plt.show()

# Visualize Feature Importance
plt.figure(figsize=(12, 6))
plt.barh(importance_df['Feature'][:20][::-1], importance_df['Importance'][:20][::-1])
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances (Random Forest)')
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'feature_importance_top20.png'))
plt.show()
