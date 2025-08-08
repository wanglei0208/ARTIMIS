import numpy as np
X_train_bc = np.load('ARTIMIS/data/2018/ddos1/X_train.npy')
y_train_bc = np.load('ARTIMIS/data/2018/ddos1/y_train.npy')
X_benign = X_train_bc[y_train_bc == 0]
np.save('/raid/wl_raid/ARTIMIS/data/2018/ddos1/X_train_benign.npy', X_benign)

