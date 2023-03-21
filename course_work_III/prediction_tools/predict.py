import hydra
from omegaconf import DictConfig
from utils.model_utils import get_model
from utils.data_utils import create_df
from utils.datasets.ecg_2d_dataset import Ecg2dDataset
from utils.datasets.ecg_1d_dataset import Ecg1dDataset
from utils.device_utils import *
from utils.metrics_utils import metrics_report
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


@hydra.main(version_base=None, config_path=".", config_name="config")
def predict(cfg: DictConfig):

    model_info = torch.load(cfg.predict.checkpoint_path, map_location=torch.device("cpu"))
    state_dict = model_info["model"]
    metrics = model_info["metrics"]
    model_config = model_info["config_file"]

    device = get_default_device()

    print("Read test data...")

    ecg_info = create_df(cfg, "test")

    pathology_names = cfg.task_params.pathology_names
    if cfg.task_params.merge_map:
        pathology_names = cfg.task_params.merge_map.keys()

    test_target = ecg_info.target

    if cfg.ecg_record_params.input_type == "2d":
        test_dataset = Ecg2dDataset(
            ecg_info, test_target, cfg.ecg_record_params.base, cfg.ecg_record_params.size
        )
    else:
        test_dataset = Ecg1dDataset(
            ecg_info, test_target
        )

    model = get_model(cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    threshold = metrics["threshold"]
    data_loader = DataLoader(test_dataset, batch_size=model_config.training_params.batch_size)

    results = {}
    raw_preds, probs, true_labels, paths = ([], [], [], [])

    print("Model inference on input dataset:")
    for batch in tqdm(data_loader, total=len(data_loader)):

        inputs, targets = batch

        inputs = inputs.to(device)
        targets = targets.to(device)

        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        raw, prob = outputs, torch.nn.Sigmoid()(outputs)

        raw_preds.extend(raw.tolist())
        probs.extend(prob.tolist())
        true_labels.extend(targets.tolist())

    results["raw_preds"] = raw_preds
    results["true_labels"] = true_labels

    print("Calculation of predictions with threshold list: {}".format(threshold))
    results[str(threshold)] = (np.array(probs) > threshold).astype(float).tolist()

    print("=============METRICS REPORT=============")
    metrics_dict = {}
    print("Metrics with `threshold = {}`".format(threshold))
    metrics, conf_matrix = metrics_report(results["true_labels"], results[str(threshold)], pathology_names)
    sub_dict = {"metrics": metrics, "confusion_matrix": conf_matrix}
    metrics_dict[str(threshold)] = sub_dict


if __name__ == "__main__":
    predict()
