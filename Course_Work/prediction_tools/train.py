import hydra
from omegaconf import DictConfig
import pandas as pd
from data_utils.dataset import get_dataset


@hydra.main(version_base=None, config_path="../", config_name="config")
def train(cfg: DictConfig):

    df = pd.read_csv(cfg.train_df_path)
    ecg_dataset = get_dataset(df, cfg)
    print(ecg_dataset[0])

    return 0


if __name__ == "__main__":
    train()
