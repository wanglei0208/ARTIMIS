# model_mampf.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.naive_bayes import GaussianNB
import joblib
import numpy as np

class ProbabilityFeatureExtractor:
    def __init__(self):
        self.model_benign = GaussianNB()
        self.model_attack = GaussianNB()

    def fit(self, X, y):
        benign_X = X[y == 0]
        attack_X = X[y == 1]
        #print(f"[DEBUG] benign_X.shape: {benign_X.shape}")
        #print(f"[DEBUG] attack_X.shape: {attack_X.shape}")
        self.model_benign.fit(benign_X, np.zeros((benign_X.shape[0],)))
        self.model_attack.fit(attack_X, np.ones((attack_X.shape[0],)))

    def transform(self, X):
        def get_safe_proba(model, X):
            proba = model.predict_proba(X)
            if proba.shape[1] == 1:
                return proba[:, 0]
            else:
                return proba[:, 1]
        proba_benign = get_safe_proba(self.model_benign, X)
        proba_attack = get_safe_proba(self.model_attack, X)
        n = X.shape[1]
        proba_benign = np.power(proba_benign, 1 / n)
        proba_attack = np.power(proba_attack, 1 / n)
        mampf_features = np.stack([proba_benign, proba_attack], axis=1)
        return np.concatenate([X, mampf_features], axis=1)


class EnhancedClassifier(nn.Module):
    def __init__(self, input_dim=56, hidden1=64, hidden2=32):
        super().__init__()
        print(f"[INFO] Initialized EnhancedClassifier with input_dim={input_dim}, hidden1={hidden1}, hidden2={hidden2}")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 2)
        )
    def forward(self, x):
        return self.net(x)