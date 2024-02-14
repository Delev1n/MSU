import numpy as np
from ecglib.data import EcgDataset
from torch.utils.data import DataLoader
import pandas as pd


def add_clients_columns(cfg, df):
    clients = np.full((len(df)), cfg.federated_params.clients_num)
    healthy_indices = df.index[df["AFIB"] == 0].tolist()
    afib_indices = df.index[df["AFIB"] == 1].tolist()

    healthy_divided = [
        healthy_indices[
            i : i + int(len(healthy_indices) / cfg.federated_params.clients_num)
        ]
        for i in range(
            0,
            len(healthy_indices),
            int(len(healthy_indices) / cfg.federated_params.clients_num),
        )
    ]
    afib_divided = [
        afib_indices[i : i + int(len(afib_indices) / cfg.federated_params.clients_num)]
        for i in range(
            0,
            len(afib_indices),
            int(len(afib_indices) / cfg.federated_params.clients_num),
        )
    ]
    for client in range(cfg.federated_params.clients_num):
        clients[healthy_divided[client]] = client
        clients[afib_divided[client]] = client

    clients[clients == cfg.federated_params.clients_num] = (
        cfg.federated_params.clients_num - 1
    )
    df["clients"] = clients
    return df


def edit_file_paths(cfg, df):

    df["fpath"] = df["fpath"].apply(
        lambda x: f'{"/".join(cfg.dataset_path.split("/")[:-1])}/{x}'
    )
    return df


def get_loader(cfg, df):
    df = df.reset_index(drop=True)

    dataset = EcgDataset(
        df.drop(columns=["AFIB"], axis=1),
        df["AFIB"],
        frequency=500,
        leads=list(range(12)),
        data_type="npz",
        ecg_length=10,
        norm_type="z_norm",
        classes=1,
        cut_range=[0, 0],
        augmentation=None,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.training_params.batch_size,
        shuffle=True,
        num_workers=cfg.training_params.num_workers,
        drop_last=False,
    )

    return loader


def change_labels_of_clients(df, attacking_clients, p):

    for num in attacking_clients:
        client_targets = df[df["clients"] == num]["AFIB"]
        labels = np.array(client_targets.tolist())
        labels.flat[
            np.random.choice(
                np.prod(labels.shape), int(p * np.prod(labels.shape)), replace=False
            )
        ] -= 1
        target_column = pd.Series(
            np.abs(labels).tolist(), name="AFIB", index=client_targets.index
        )
        df.update(target_column)
    return df
