import hydra
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from data_utils.dataset import get_dataloader, create_df


@hydra.main(version_base=None, config_path="../", config_name="config")
def train(cfg: DictConfig):

    df = create_df(pd.read_csv(cfg.train_df_path), cfg)

    skf = StratifiedKFold(n_splits=cfg.training_params.folds)
    folds = cfg.training_params.folds

    for fold_num, (train_index, val_index) in enumerate(
            skf.split(
                df, df["target"].apply(lambda x: "".join(str(it) for it in x))
            )
    ):
        print("Fold number:", fold_num)
        train_split_df, valid_split_df = df.iloc[train_index], df.iloc[val_index]
        train_loader = get_dataloader(train_split_df, cfg)
        valid_loader = get_dataloader(valid_split_df, cfg)

    return 0


if __name__ == "__main__":
    train()
