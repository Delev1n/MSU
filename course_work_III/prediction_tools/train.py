import hydra
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from utils.data_utils import get_dataloader, create_df
from utils.model_utils import get_model
from utils.metrics_utils import select_best_validation_threshold, metrics_report
import torch
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../", config_name="config")
def train(cfg: DictConfig):

    df = create_df(cfg)

    skf = StratifiedKFold(n_splits=cfg.training_params.folds)

    device = get_default_device()

    pathology_names = cfg.task_params.pathology_names
    if cfg.task_params.merge_map:
        pathology_names = cfg.task_params.merge_map.keys()

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.8,
            patience=3,
            verbose=True,
        )
        epochs_no_improve = 0
        early_stopping_patience = 10
        best_metric = 1000

        for ep in range(cfg.training_params.epochs):
            print("Epoch number:", ep)

            sum_loss = 0
            model.train()

            for bi, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                inputs, targets = batch

                inputs = inputs.to(device)
                targets = targets.to(device)

                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                sum_loss += loss.detach().item()

                optimizer.step()

            train_loss = sum_loss / len(train_loader)

            model.eval()
            sum_loss = 0
            fin_targets = []
            fin_outputs = []
            sigmoid = torch.nn.Sigmoid()

            with torch.no_grad():
                for bi, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                    inputs, targets = batch

                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs = model(inputs)

                    loss = criterion(outputs, targets)
                    sum_loss += loss.detach().item()

                    fin_targets.extend(targets.tolist())
                    fin_outputs.extend(outputs.tolist())

            fin_outputs = sigmoid(torch.as_tensor(fin_outputs))
            best_threshold = select_best_validation_threshold(fin_targets, fin_outputs)
            results = (fin_outputs > best_threshold).float()
            metrics, confusion_matrix = metrics_report(fin_targets, results.tolist(), pathology_names)
            val_loss = sum_loss / len(valid_loader)

            scheduler.step(val_loss)
            if val_loss >= best_metric:
                epochs_no_improve += 1
            else:
                best_metric = val_loss
                epochs_no_improve = 0

            if epochs_no_improve == 0:
                print("Model saved!")
                model_info = {
                    "model": model.state_dict(),
                    "config_file": cfg,
                    "metrics": {
                        "threshold": best_threshold,
                        "conf_mat": confusion_matrix,
                        "val_loss": val_loss,
                        "metrics_table": metrics,
                    },
                }
                composed_name = "_".join(cfg.model)
                checkpoint_path = f"{cfg.single_run_dir}/12_leads_experiment_{cfg.dataset}_{composed_name}_" \
                                  f"{cfg.task_params.pathology_names[0]}{fold_num}.pt"
                torch.save(model_info, checkpoint_path)

            print(f"\ntrain_loss:{train_loss}\n \
                    \nval_loss: {val_loss}\n \
                    \nval_metrics:\n{metrics}")
            print(confusion_matrix)
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping")
                break

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
