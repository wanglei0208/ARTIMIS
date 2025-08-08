from kitnet_model import KitNET as kit
import numpy as np
import pandas as pd
import time
import os
import pickle
import json

print("Reading Sample dataset...")
X = pd.read_csv("ARTIMIS/data/2018/brute_force/kitnet/X_train_b_test_bm.csv", header=None).values #an m-by-n dataset with m observations
timestamps = pd.read_csv("ARTIMIS/data/2018/brute_force/kitnet/timestamps.csv",header=None).values
labels = pd.read_csv("ARTIMIS/data/2018/brute_force/kitnet/y_train_b_test_bm.csv", header=None).values.flatten()
# with open("/raid/wl_raid/NIDS-minmax/kitnet/feature_map_custom.json") as f:
#     feature_map = json.load(f)
# KitNET params:
maxAE = 10 #maximum size for any autoencoder in the ensemble layer
FMgrace = 5000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = 25000 #the number of instances used to train the anomaly detector (ensemble itself)
hidden_ratio = 0.75
learning_rate = 0.1
corruption_level = 0
#scaler_name="minmax"

save_dir = f"result/2018/brute_force"
os.makedirs(save_dir, exist_ok=True)  # 自动创建多级目录

# Build KitNET
K = kit.KitNET(n=X.shape[1],
               max_autoencoder_size=maxAE,
               FM_grace_period=FMgrace,
               AD_grace_period=ADgrace,
               feature_map=None,
               learning_rate=learning_rate,
               hidden_ratio=hidden_ratio,
               corruption_level=corruption_level)
RMSEs = np.zeros(X.shape[0]) # a place to save the scores

print("Running KitNET:")
start = time.time()
# Here we process (train/execute) each individual observation.
# In this way, X is essentially a stream, and each observation is discarded after performing process() method.

rmse_vectors = []
for i in range(X.shape[0]):
    if i % 1000 == 0:
        print(i)
    score = K.process(X[i])
    RMSEs[i] = score
    if i >= (FMgrace + ADgrace):
        rmse_vectors.append(K.extract_rmse_vector(X[i]))
stop = time.time()
print("Complete. Time elapsed: "+ str(stop - start))
rmse_vectors = np.array(rmse_vectors)
rmse_vec_path = os.path.join(save_dir, "rmse_vectors.npy")
np.save(rmse_vec_path, rmse_vectors)
print(f"The RMSE vector has been saved: {rmse_vec_path}")

# 2. save KitNET model
model_path = os.path.join(save_dir, "kitnet_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(K, f)

# Here we demonstrate how one can fit the RMSE scores to a log-normal distribution (useful for finding/setting a cutoff threshold \phi)
from scipy.stats import norm
benignSample = np.log(RMSEs[FMgrace+ADgrace+1:39944])
logProbs = norm.logsf(np.log(RMSEs), np.mean(benignSample), np.std(benignSample))

# save log(RMSE)
mu = float(np.mean(benignSample))
sigma = float(np.std(benignSample))
dist_path = os.path.join(save_dir, "rmse_log_distribution.json")
with open(dist_path, "w") as f:
    json.dump({"mu": mu, "sigma": sigma}, f)
print(f"The model and distributed parameters have been saved to:{save_dir}")


