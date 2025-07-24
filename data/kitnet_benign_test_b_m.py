import numpy as np
import pandas as pd

X_train = np.load('ARTIMIS/data/2018/brute_force/X_train.npy')
y_train = np.load('ARTIMIS/data/2018/brute_force/y_train.npy')
X_test = np.load('ARTIMIS/data/2018/brute_force/X_test.npy')
y_test = np.load('ARTIMIS/data/2018/brute_force/y_test.npy')

X_benign_train = X_train[y_train == 0]
y_benign_train = y_train[y_train == 0]

X_benign_test = X_test[y_test == 0]
y_benign_test = y_test[y_test == 0]

X_malicious_test = X_test[y_test == 1]
y_malicious_test = y_test[y_test == 1]

X_combined = np.concatenate((X_benign_train, X_benign_test, X_malicious_test), axis=0)
y_combined = np.concatenate((y_benign_train, y_benign_test, y_malicious_test), axis=0)

df_X_combined = pd.DataFrame(X_combined)
df_y_combined = pd.DataFrame(y_combined)

df_X_combined.to_csv('ARTIMIS/data/2018/brute_force/kitnet/X_train_b_test_bm.csv', index=False, header=False)
df_y_combined.to_csv('ARTIMIS/data/2018/brute_force/kitnet/y_train_b_test_bm.csv', index=False, header=False)

print(f'The number of data samples after splicing:{len(X_combined)}')
