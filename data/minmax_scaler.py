from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump

file_path = "ARTIMIS/data/2018/brute_force/bf.csv"
data = pd.read_csv(file_path)
features = data.drop('label', axis=1)

# Normalize using MinMaxScaler
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# save scaler，for the convenience of subsequent inverse_transform
dump(scaler, 'ARTIMIS/data/2018/brute_force/bf.joblib')
data['label'] = data['label'].apply(lambda x: 0 if x == 'Benign' else 1)
label_counts = data['label'].value_counts()
print(f"(Benign)：{label_counts[0]}")
print(f"(attack)：{label_counts[1]}")

x = scaled_features
y = data['label']
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(x, y, test_size=0.2, random_state=0,stratify=y)

np.save('ARTIMIS/data/2018/brute_force/X_train.npy', X_train_bc)
np.save('ARTIMIS/data/2018/brute_force/X_test.npy', X_test_bc)
np.save('ARTIMIS/data/2018/brute_force/y_train.npy', y_train_bc)
np.save('ARTIMIS/data/2018/brute_force/y_test.npy', y_test_bc)

