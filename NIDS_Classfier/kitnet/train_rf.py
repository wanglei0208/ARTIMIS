import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ========== 文件路径 ==========
KITNET_MODEL_PATH = "ARTIMIS/kitnet/result/2018/brute_force/kitnet_model.pkl"       # The trained KitNET model
X_TRAIN_PATH = "ARTIMIS/data/2018/brute_force/X_train.npy"
X_TEST_PATH = "ARTIMIS/data/2018/brute_force/X_test.npy"
Y_TRAIN_PATH = "ARTIMIS/data/2018/brute_force/y_train.npy"
Y_TEST_PATH = "ARTIMIS/data/2018/brute_force/y_test.npy"
RF_MODEL_PATH = "ARTIMIS/kitnet/result/2018/brute_force/rf_model.pkl"
RF_REPORT_PATH = "ARTIMIS/kitnet/result/2018/brute_force/rf_evaluation_report.txt"

# ========== Loading data... ==========
print("Loading data...")
X_train = np.load(X_TRAIN_PATH)
X_test = np.load(X_TEST_PATH)
y_train = np.load(Y_TRAIN_PATH)
y_test = np.load(Y_TEST_PATH)

# ========== Loading KitNET model ==========
print("Loading KitNET model...")
with open(KITNET_MODEL_PATH, "rb") as f:
    K = pickle.load(f)

# ========== Extract the RMSE vector ==========
def extract_rmse_vectors(X, kitnet_model):
    rmse_vectors = []
    for i in range(X.shape[0]):
        rmse_vec = kitnet_model.extract_rmse_vector(X[i])
        rmse_vectors.append(rmse_vec)
    return np.array(rmse_vectors)

print("Extract the RMSE vector...")
X_train_rmse = extract_rmse_vectors(X_train, K)
X_test_rmse = extract_rmse_vectors(X_test, K)

# ========== train Random Forest ==========
print("train Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_rmse, y_train)

# ========== save RF model ==========
with open(RF_MODEL_PATH, "wb") as f:
    pickle.dump(rf, f)
print(f"The RF model has been saved to {RF_MODEL_PATH}")

y_pred = rf.predict(X_test_rmse)
y_prob = rf.predict_proba(X_test_rmse)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

report = classification_report(y_test, y_pred, digits=4)
with open(RF_REPORT_PATH, "w") as f:
    f.write("=== Random Forest Evaluation Report ===\n")
    f.write(f"Accuracy : {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall   : {rec:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")
    f.write(f"AUC      : {auc:.4f}\n\n")
    f.write(report)

print(f"评估完成，报告已保存至 {RF_REPORT_PATH}")
