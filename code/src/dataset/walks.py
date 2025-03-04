import pandas as pd
from torch.utils.data import Dataset

class WalksDataset(Dataset):
    def __init__(self, file_path, transform=None, target_transform=None):
        self.walks = pd.read_parquet(file_path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.walks)
    
    def __getitem__(self, idx):
        walk = self.walks.iloc[idx, :]
        if self.transform:
            walk = self.transform(walk)
        if self.target_transform:
            walk = self.target_transform(walk)

        return walk
