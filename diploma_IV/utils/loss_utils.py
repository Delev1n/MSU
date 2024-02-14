import numpy as np
import torch


def get_loss(df, device):
    target_array = np.array(df["AFIB"].tolist())
    zeros_count = np.sum(target_array == 0, axis=0)
    ones_count = np.sum(target_array == 1, axis=0)
    pos_weight = torch.tensor([(zeros_count / ones_count).tolist()]).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(
        size_average=True, reduce=True, reduction="mean", pos_weight=pos_weight
    )

    return criterion
