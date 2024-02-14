import hydra
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from utils.data_utils import (
    add_clients_columns,
    edit_file_paths,
    get_loader,
    change_labels_of_clients,
)
from utils.loss_utils import get_loss
from trainer import BaseTrainer
from federated_trainer import FederatedTrainer


@hydra.main(version_base=None, config_path="utils/", config_name="config")
def train(cfg: DictConfig):
    df = pd.read_csv(cfg.dataset_path)
    df = edit_file_paths(cfg, df)

    if cfg.mode == "federated_train":

        df = add_clients_columns(cfg, df)
        federated_trainer = FederatedTrainer(cfg)
        if cfg.device == "cuda":
            federated_trainer.device = "{}:{}".format(cfg.device, cfg.device_ids[0])
        if federated_trainer.attack:
            np.random.seed(cfg.random_state)
            federated_trainer.attacking_clients = np.random.choice(
                range(cfg.federated_params.clients_num),
                size=int(
                    cfg.federated_params.clients_num
                    * cfg.federated_params.amount_of_attackers
                ),
                replace=False,
            )
            print(f"Attacking client indeces: {federated_trainer.attacking_clients}")
            df = change_labels_of_clients(
                df,
                federated_trainer.attacking_clients,
                cfg.federated_params.percent_of_changed_labels,
            )
            print("Succesfully flipped the labels")

        federated_trainer.train_df, federated_trainer.valid_df = train_test_split(
            df,
            test_size=0.2,
            random_state=cfg.random_state,
            stratify=df["clients"],
        )

        if "FLTrust" in cfg.federated_params.method:
            fltrust_df = pd.read_csv(cfg.federated_params.server_dataset)
            fltrust_df = edit_file_paths(cfg, fltrust_df)
            federated_trainer.fltrust_train_df, federated_trainer.fltrust_valid_df = (
                train_test_split(
                    fltrust_df,
                    test_size=0.2,
                    random_state=cfg.random_state,
                    shuffle=True,
                    stratify=fltrust_df["AFIB"],
                )
            )
            print("Initialized FLTrust dataset")

        test_df = pd.read_csv(cfg.federated_params.test_dataset)
        test_df = edit_file_paths(cfg, test_df)
        federated_trainer.test_df = test_df
        federated_trainer.criterion = get_loss(df, federated_trainer.device)
        federated_trainer.federated_training()

    elif cfg.mode == "train":

        trainer = BaseTrainer(cfg)
        train_df, valid_df = train_test_split(
            df,
            test_size=0.2,
            random_state=cfg.random_state,
            shuffle=True,
            stratify=df["AFIB"],
        )

        trainer.train_loader = get_loader(cfg, train_df)
        trainer.valid_loader = get_loader(cfg, valid_df)

        if cfg.device == "cuda":
            trainer.device = "{}:{}".format(cfg.device, cfg.device_ids[0])

        trainer.criterion = get_loss(df, trainer.device)

        trainer.train()


if __name__ == "__main__":
    train()
