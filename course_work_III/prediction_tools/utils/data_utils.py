import pandas as pd
from .datasets.ecg_2d_dataset import Ecg2dDataset
from .datasets.ecg_1d_dataset import Ecg1dDataset
from sklearn.preprocessing import MultiLabelBinarizer
import ast
from torch.utils.data import DataLoader


def create_df(cfg):

    assert cfg.dataset in ["ptbxl"]

    ecg_info = pd.read_csv(cfg.df_path.ptbxl.train_directory)
    ecg_info.reset_index(drop=True, inplace=True)
    one_hot = make_onehot(ecg_info, cfg.task_params.pathology_names)

    if cfg.task_params.merge_map:
        one_hot = merge_columns(df=one_hot, merge_map=cfg.task_params.merge_map)

    ecg_info["target"] = one_hot.values.tolist()

    return ecg_info


def get_dataloader(ecg_info: pd.DataFrame, cfg):

    ecg_target = ecg_info["target"].to_list()

    if cfg.ecg_record_params.input_type == "2d":
        ecg_dataset = Ecg2dDataset(
            ecg_info, ecg_target, cfg.ecg_record_params.base, cfg.ecg_record_params.size
        )
    else:
        ecg_dataset = Ecg1dDataset(
            ecg_info, ecg_target
        )
    ecg_loader = DataLoader(
        ecg_dataset,
        batch_size=cfg.training_params.batch_size,
        shuffle=True,
        drop_last=True
    )

    return ecg_loader


def make_onehot(ecg_df, pathology_names):

    mlb = MultiLabelBinarizer()
    one_hot = pd.DataFrame(
        mlb.fit_transform(ecg_df["scp_codes"].apply(ast.literal_eval)),
        columns=mlb.classes_,
    )
    drop_cols = set(one_hot.columns) - set(pathology_names)
    one_hot.drop(columns=drop_cols, inplace=True)

    return one_hot


def merge_columns(df, merge_map):

    for k, v in merge_map.items():
        tmp = df[v].apply(any, axis=1).astype(int)
        df.drop(columns=v, inplace=True)
        df[k] = tmp

    return df


