import hydra
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from utils.data_utils import get_dataloader, create_df
from utils.model_utils import get_model
import torch


@hydra.main(version_base=None, config_path="../", config_name="config")
def train(cfg: DictConfig):

    df = create_df(pd.read_csv(cfg.train_df_path), cfg)

    skf = StratifiedKFold(n_splits=cfg.training_params.folds)

    device = get_default_device()

    for fold_num, (train_index, val_index) in enumerate(
            skf.split(
                df, df["target"].apply(lambda x: "".join(str(it) for it in x))
            )
    ):
        print("Fold number:", fold_num)
        train_split_df, valid_split_df = df.iloc[train_index], df.iloc[val_index]

        pos_weight = len(train_split_df[[target == [0] for target in train_split_df.target]]) / len(
            train_split_df[[target == [1] for target in train_split_df.target]])

        train_loader = get_dataloader(train_split_df, cfg)
        valid_loader = get_dataloader(valid_split_df, cfg)

        model = get_model(cfg)
        to_device(model, device)

        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(device)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training_params.lr)

    return 0


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


if __name__ == "__main__":
    train()
