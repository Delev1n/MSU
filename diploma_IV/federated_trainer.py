import torch
import copy
from tqdm import tqdm
from collections import OrderedDict
from torch.nn.functional import relu

from trainer import BaseTrainer
from utils.model_utils import get_model
from utils.data_utils import get_loader
from utils.metric_utils import select_best_validation_threshold


class FederatedTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.federated_method = cfg.federated_params.method
        self.train_df = None
        self.valid_df = None
        self.test_df = None
        self.number_com_rounds = cfg.federated_params.communication_rounds
        self.global_model = get_model(cfg)
        self.lr = cfg.training_params.learning_rate
        self.optimizer = torch.optim.Adam(self.global_model.parameters(), lr=self.lr)
        self.attack = cfg.federated_params.attack
        self.states = []
        self.attacking_clients = []
        if self.federated_method == "FedProx":
            self.current_com_round = 0
        if "FLTrust" in self.federated_method:
            self.fltrust_train_df = None
            self.fltrust_valid_df = None
            if self.federated_method == "FLTrust_new":
                self.fltrust_threshold = None
                self.sigmoid = torch.nn.Sigmoid()

    def init_client_dalaloaders(self, client_idx):
        client_train_df = self.train_df[self.train_df["clients"] == client_idx]
        client_valid_df = self.valid_df[self.valid_df["clients"] == client_idx]
        print(client_train_df["AFIB"].value_counts())
        self.train_loader = get_loader(
            self.cfg,
            client_train_df,
        )
        self.valid_loader = get_loader(
            self.cfg,
            client_valid_df,
        )

    def federated_training(self):
        for com_round in range(self.number_com_rounds):
            print(f"Communication round {com_round}")

            updated_list_of_trained_model_parameters = []

            for client_idx in range(self.cfg.federated_params.clients_num):

                print(f"Client number: {client_idx}")
                self.attack = client_idx in self.attacking_clients
                print(f"Client attacking: {self.attack}")
                self.init_client_dalaloaders(client_idx)
                self.model = copy.deepcopy(self.global_model)
                client_model_weights = self.federated_training_round()
                updated_list_of_trained_model_parameters.append(client_model_weights)

            if "FLTrust" in self.federated_method:
                print("\nStarted training server side model\n")

                self.train_loader = get_loader(
                    self.cfg,
                    self.fltrust_train_df,
                )
                self.valid_loader = get_loader(
                    self.cfg,
                    self.fltrust_valid_df,
                )

                self.model = copy.deepcopy(self.global_model)
                server_model_weights = self.federated_training_round()

                if self.federated_method == "FLTrust_new":
                    self.valid_loader = get_loader(
                        self.cfg,
                        self.test_df,
                    )
                    _, fin_targets, fin_outputs = self.eval_fn()
                    fin_outputs = self.sigmoid(torch.as_tensor(fin_outputs))
                    self.fltrust_threshold = select_best_validation_threshold(
                        fin_targets,
                        fin_outputs,
                    )

                updated_list_of_trained_model_parameters.append(server_model_weights)
                print("\nTrained server side model\n")

            self.update_global_model(
                updated_list_of_trained_model_parameters, com_round
            )

    def federated_training_round(self):

        for ep in range(self.cfg.federated_params.round_epochs):
            print("Epoch number:", ep)
            self.train_fn()
            _, fin_targets, fin_outputs = self.eval_fn()
            self.calculate_metrics(fin_targets, fin_outputs)

        best_client_model_weights = OrderedDict()
        for key, weights in self.states[0].items():
            best_client_model_weights[key] = 0
        for state in self.states:
            for key, weights in state.items():
                best_client_model_weights[key] = (
                    best_client_model_weights[key] + weights
                )
        self.states = []

        return best_client_model_weights

    def train_fn(self):

        old_state = copy.deepcopy(self.model)
        self.model.train()

        for _, batch in tqdm(
            enumerate(self.train_loader), total=len(self.train_loader)
        ):
            _, (input, targets) = batch

            inp = input[0].to(self.device)
            targets = targets.reshape((len(inp), 1)).to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inp)

            loss = self.criterion(outputs, targets)

            loss.backward()

            self.optimizer.step()

        self.model.eval()
        state = OrderedDict()
        for (key, weights1), (_, weights2) in zip(
            self.model.state_dict().items(), old_state.state_dict().items()
        ):
            state[key] = weights1 - weights2
        self.states.append(state)

    def update_global_model(self, updated_list_of_trained_model_parameters, com_round):
        aggregated_weights = self.global_model.state_dict()

        if self.federated_method == "FLTrust":
            updated_list_of_trained_model_parameters = self.fltrust_update(
                updated_list_of_trained_model_parameters
            )
        elif self.federated_method == "FLTrust_new":
            updated_list_of_trained_model_parameters = self.fltrust_new_update(
                updated_list_of_trained_model_parameters
            )

        for i in range(self.cfg.federated_params.clients_num):
            client_train_df = self.train_df[self.train_df["clients"] == i]
            client_weight = len(client_train_df) / len(self.train_df)
            for key, weights in updated_list_of_trained_model_parameters[i].items():
                aggregated_weights[key] = (
                    aggregated_weights[key] + weights * client_weight
                )

        self.global_model = get_model(self.cfg)
        self.global_model.load_state_dict(aggregated_weights)

        self.valid_loader = get_loader(
            self.cfg,
            self.test_df,
        )
        self.model = copy.deepcopy(self.global_model)
        _, fin_targets, fin_outputs = self.eval_fn()
        self.calculate_metrics(fin_targets, fin_outputs)

        checkpoint_path = f"{self.cfg.single_run_dir}/federated_learning_{com_round}.pt"
        torch.save(aggregated_weights, checkpoint_path)

    def fltrust_update(self, updated_list_of_trained_model_parameters):
        client_directions = []
        trust_scores = []
        server_direction = torch.cat(
            [
                x.flatten()
                for x in list(updated_list_of_trained_model_parameters[-1].values())
            ]
        )
        for i in range(self.cfg.federated_params.clients_num):
            client_directions.append(
                torch.cat(
                    [
                        x.flatten()
                        for x in list(
                            updated_list_of_trained_model_parameters[i].values()
                        )
                    ]
                )
            )
            trust_scores.append(
                relu(
                    torch.dot(server_direction, client_directions[i])
                    / (torch.norm(server_direction) * torch.norm(client_directions[i]))
                )
            )
            print(f"Client {i} trust score: {trust_scores[i]}")

        for i in range(self.cfg.federated_params.clients_num):
            for key, weights in updated_list_of_trained_model_parameters[i].items():
                updated_list_of_trained_model_parameters[i][key] = (
                    (1 / torch.stack(trust_scores, dim=0).sum(dim=0))
                    * trust_scores[i]
                    * (torch.norm(server_direction) / torch.norm(client_directions[i]))
                    * weights
                )
        return updated_list_of_trained_model_parameters

    def fltrust_new_update(self, updated_list_of_trained_model_parameters):
        signal = self.test_df[self.test_df["AFIB"] == 0].sample()
        self.valid_loader = get_loader(
            self.cfg,
            signal,
        )
        _, _, fin_outputs = self.eval_fn()
        print(torch.as_tensor(fin_outputs), torch.as_tensor(fin_outputs).shape)
        server_result = self.sigmoid(torch.as_tensor(fin_outputs))[0][0]
        trust_scores = []
        for i in range(self.cfg.federated_params.clients_num):

            tmp_weights = self.global_model.state_dict()
            for key, weights in updated_list_of_trained_model_parameters[i].items():
                tmp_weights[key] = tmp_weights[key] + weights

            self.model = get_model(self.cfg)
            self.model.load_state_dict(tmp_weights)
            _, _, fin_outputs = self.eval_fn()
            client_result = self.sigmoid(torch.as_tensor(fin_outputs))[0][0]
            trust_scores.append(
                self.count_trust_score_for_new_fltrust(server_result, client_result)
            )
            print(f"Client {i} trust score: {trust_scores[i]}")

        for i in range(self.cfg.federated_params.clients_num):
            for key, weights in updated_list_of_trained_model_parameters[i].items():
                updated_list_of_trained_model_parameters[i][key] = (
                    (1 / sum(trust_scores)) * trust_scores[i] * weights
                )

        return updated_list_of_trained_model_parameters

    def count_trust_score_for_new_fltrust(self, server_result, client_result):
        if (client_result - self.fltrust_threshold) * (
            server_result - self.fltrust_threshold
        ) < 0:
            return 0
        else:
            return 2 * (
                1 - self.sigmoid(torch.exp(10 * abs(client_result - server_result)) - 1)
            )
