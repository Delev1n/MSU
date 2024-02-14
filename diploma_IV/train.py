import hydra
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from ecglib.data import EcgDataset
from torch.utils.data import DataLoader
import numpy as np
import torch

from utils.data_utils import add_clients_columns, edit_file_paths
from trainer import BaseTrainer


@hydra.main(version_base=None, config_path="utils/", config_name="config")
def train(cfg: DictConfig):
    df = pd.read_csv(cfg.dataset_path)
    df = edit_file_paths(cfg, df)

    if cfg.mode == "federated_train":

        df = add_clients_columns(cfg, df)

    elif cfg.mode == "train":

        trainer = BaseTrainer(cfg)
        X_train, X_valid, y_train, y_valid = train_test_split(
            df.drop(columns=["AFIB"], axis=1),
            df["AFIB"],
            test_size=0.2,
            random_state=cfg.random_state,
            shuffle=True,
            stratify=df["AFIB"],
        )

        train_dataset = EcgDataset(
            X_train.reset_index(drop=True),
            y_train.reset_index(drop=True),
            frequency=500,
            leads=list(range(12)),
            data_type="npz",
            ecg_length=10,
            norm_type="z_norm",
            classes=1,
            cut_range=[0, 0],
            augmentation=None,
        )
        valid_dataset = EcgDataset(
            X_valid.reset_index(drop=True),
            y_valid.reset_index(drop=True),
            frequency=500,
            leads=list(range(12)),
            data_type="npz",
            ecg_length=10,
            norm_type="z_norm",
            classes=1,
            cut_range=[0, 0],
            augmentation=None,
        )

        trainer.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=True,
        )
        trainer.valid_loader = DataLoader(
            valid_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=True,
        )

        if cfg.device == "cuda":
            trainer.device = "{}:{}".format(cfg.device, cfg.device_ids[0])

        target_array = np.array(df["AFIB"].tolist())
        zeros_count = np.sum(target_array == 0, axis=0)
        ones_count = np.sum(target_array == 1, axis=0)
        pos_weight = torch.tensor([(zeros_count / ones_count).tolist()]).to(
            trainer.device
        )
        trainer.criterion = torch.nn.BCEWithLogitsLoss(
            size_average=True, reduce=True, reduction="mean", pos_weight=pos_weight
        )

        trainer.train()


if __name__ == "__main__":
    train()
