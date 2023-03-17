from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch


class Ecg2dDataset(Dataset):

    def __init__(self, df, targets, base, size):
        self.df = df
        self.targets = targets
        self.base = base
        self.size = size

    def __getitem__(self, index):

        data = (np.array(Image.open("{}/combined/{}".format(self.base, self.df.iloc[index]['file_name'])).convert(
            'RGB')) / 255).reshape(3, self.size[0], self.size[1])
        target = np.array(self.targets[index]).reshape(1)

        return torch.tensor(data, dtype=torch.float), torch.tensor(target, dtype=torch.float)

    def __len__(self):
        return len(self.df)
