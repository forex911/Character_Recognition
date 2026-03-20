import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.config import BATCH_SIZE

class SelfDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for idx, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            class_dir = os.path.join(root_dir, letter)
            if not os.path.exists(class_dir):
                continue

            for file in os.listdir(class_dir):
                path = os.path.join(class_dir, file)
                self.samples.append((path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.load(path)[0]  # already preprocessed
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return img, label


def get_dataloaders():
    dataset = SelfDataset("data/self_data")
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
