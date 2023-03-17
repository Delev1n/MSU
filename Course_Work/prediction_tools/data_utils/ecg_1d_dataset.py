from torch.utils.data import Dataset
import numpy as np
import torch


class Ecg1dDataset(Dataset):

    def __init__(self, df, targets):
        self.df = df
        self.targets = targets

    def __getitem__(self, index):

        data = np.load(self.df.iloc[index]['fpath'])['arr_0']
        target = np.array(self.targets[index]).reshape(1)

        return torch.tensor(data, dtype=torch.float), torch.tensor(target, dtype=torch.float)

    def __len__(self):
        return len(self.df)
