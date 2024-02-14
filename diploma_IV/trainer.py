from utils.model_utils import get_model
from utils.metric_utils import select_best_validation_threshold, metrics_report
import torch
from tqdm import tqdm


class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_loader = None
        self.valid_loader = None
        self.model = get_model(cfg)
        self.criterion = None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.best_loss = 1000
        self.epochs = cfg.epochs
        self.epochs_no_improve = 0
        self.prediction_threshold = 0.5
        self.early_stopping_patience = cfg.early_stopping_patience
        self.device = "cpu"

    def train_fn(self):
        self.model.train()

        for _, batch in tqdm(
            enumerate(self.train_loader), total=len(self.train_loader)
        ):
            _, (input, targets) = batch

            inp = input[0].to(self.device)
            targets = targets.reshape((self.cfg.batch_size, 1)).to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inp)

            loss = self.criterion(outputs, targets)

            loss.backward()

            self.optimizer.step()

    def eval_fn(self):
        self.model.eval()
        val_loss = 0
        fin_targets = []
        fin_outputs = []

        with torch.no_grad():
            for _, batch in tqdm(
                enumerate(self.valid_loader), total=len(self.valid_loader)
            ):
                _, (input, targets) = batch

                inp = input[0].to(self.device)
                targets = targets.reshape((self.cfg.batch_size, 1)).to(self.device)

                outputs = self.model(inp)

                loss = self.criterion(outputs, targets)
                val_loss += loss.detach().item()

                fin_targets.extend(targets.tolist())
                fin_outputs.extend(outputs.tolist())

        return val_loss, fin_targets, fin_outputs

    def calculate_metrics(self, fin_targets, fin_outputs):
        sigmoid = torch.nn.Sigmoid()

        fin_outputs = sigmoid(torch.as_tensor(fin_outputs))
        self.prediction_threshold = select_best_validation_threshold(
            fin_targets, fin_outputs
        )
        results = (fin_outputs > self.prediction_threshold).float()
        metrics_report(fin_targets, results.tolist(), ["AFIB"])

    def train(self):

        for ep in range(self.epochs):
            print("Epoch number:", ep)

            self.train_fn()
            val_loss, fin_targets, fin_outputs = self.eval_fn()
            self.calculate_metrics(fin_targets, fin_outputs)

            if val_loss >= self.best_loss:
                self.epochs_no_improve += 1
            else:
                self.epochs_no_improve = 0
                self.best_loss = val_loss

            if self.epochs_no_improve >= self.early_stopping_patience:
                print("Early stopping")
                break
            elif self.epochs_no_improve == 0:
                torch.save(
                    self.model.state_dict(),
                    f"{self.cfg.single_run_dir}/centrelized_training.pt",
                )
