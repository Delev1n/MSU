import hydra
from omegaconf import DictConfig
import torch
import pandas as pd
from tqdm import tqdm

from utils.model_utils import get_model
from utils.data_utils import get_loader, edit_file_paths
from utils.metric_utils import calculate_metrics


@hydra.main(version_base=None, config_path="utils/", config_name="config")
def predict(cfg: DictConfig):

    state_dict = torch.load(cfg.predict.checkpoint_path)

    test_df = pd.read_csv(cfg.dataset_path)
    test_df = edit_file_paths(cfg, test_df)
    test_loader = get_loader(cfg, test_df)

    device = "cpu"
    if cfg.device == "cuda":
        device = "{}:{}".format(cfg.device, cfg.device_ids[0])

    model = get_model(cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    fin_targets = []
    fin_outputs = []

    with torch.no_grad():
        for _, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            _, (input, targets) = batch

            inp = input[0].to(device)
            targets = targets.reshape((len(inp), 1)).to(device)

            outputs = model(inp)

            fin_targets.extend(targets.tolist())
            fin_outputs.extend(outputs.tolist())

    calculate_metrics(fin_targets, fin_outputs)


if __name__ == "__main__":
    predict()
