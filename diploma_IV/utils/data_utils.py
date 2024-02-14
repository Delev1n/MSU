import numpy as np
import os


def add_clients_columns(cfg, df):
    clients = np.full((1000), cfg.clients_num)
    healthy_indices = df.index[df["AFIB"] == 0].tolist()
    afib_indices = df.index[df["AFIB"] == 1].tolist()

    healthy_divided = [
        healthy_indices[i : i + int(len(healthy_indices) / cfg.clients_num)]
        for i in range(
            0, len(healthy_indices), int(len(healthy_indices) / cfg.clients_num)
        )
    ]
    afib_divided = [
        afib_indices[i : i + int(len(afib_indices) / cfg.clients_num)]
        for i in range(0, len(afib_indices), int(len(afib_indices) / cfg.clients_num))
    ]
    for client in range(cfg.clients_num):
        clients[healthy_divided[client]] = client
        clients[afib_divided[client]] = client

    clients[clients == cfg.clients_num] = cfg.clients_num - 1
    df["clients"] = clients
    return df


def edit_file_paths(cfg, df):

    df["fpath"] = df["fpath"].apply(
        lambda x: f'{"/".join(cfg.dataset_path.split("/")[:-1])}/{x}'
    )
    return df
