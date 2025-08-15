import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class NIDSDataset(Dataset):
    def __init__(self, feature_path, label_path):
        self.features = np.load(feature_path).astype(np.float32)
        self.labels = np.load(label_path).astype(np.int64)

        assert len(self.features) == len(self.labels), "Feature and label size mismatch!"

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])
        y = torch.tensor(self.labels[idx])
        return x, y, 0

def get_dataset(args):
    if args.dataset == 'cicids2017':
        setattr(args, 'num_classes', 2)
        feature_path = '../data/2018/brute_force/attack/X_malicious_test_100.npy'
        label_path = '../data/2018/brute_force/attack/y_malicious_test_100.npy'

        dataset = NIDSDataset(feature_path, label_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_worker, pin_memory=True)
        return dataloader
    else:
        raise NotImplementedError("Dataset not supported.")
