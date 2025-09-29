import pickle
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from scipy import special
from sklearn import linear_model, preprocessing
import optuna
import jsonargparse

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange

from tqdm import tqdm
from .data_collator import DataCollator
from common_ai.generator import MyGenerator
from common_ai.initializer import MyInitializer
from common_ai.optimizer import MyOptimizer
from common_ai.train import MyTrain


class DeepHF(nn.Module):
    def __init__(
        self,
        ext1_up: int,
        ext1_down: int,
        ext2_up: int,
        ext2_down: int,
        em_drop: float,
        fc_drop: float,
        em_dim: int,
        rnn_units: int,
        fc_num_hidden_layers: int,
        fc_num_units: int,
        fc_activation: Literal["elu", "relu", "tanh", "sigmoid", "hard_sigmoid"],
    ) -> None:
        """DeepHF arguments.

        Args:
            ext1_up: upstream limit of the resection of the upstream end.
            ext1_down: downstream limit of the templated insertion of the upstream end.
            ext2_up: upstream limit of the templated insertion of the downstream end.
            ext2_down: downstream limit of the resection of the downstream end.
            em_drop: dropout probability of embedding.
            fc_drop: dropout probability of fully connected layer.
            em_dim: embedding dimension.
            rnn_units: BiLSTM output dimension.
            fc_num_hidden_layers: number of output fully connected layers.
            fc_num_units: hidden dimension of output fully connected layers.
            fc_activation: activation function of output fully connected layers.
        """
        super().__init__()
        self.ext1_up = ext1_up
        self.ext1_down = ext1_down
        self.ext2_up = ext2_up
        self.ext2_down = ext2_down

        self.data_collator = DataCollator(
            ext1_up=ext1_up,
            ext1_down=ext1_down,
            ext2_up=ext2_up,
            ext2_down=ext2_down,
        )

        self.embedding = nn.Embedding(
            num_embeddings=6,
            embedding_dim=em_dim,
        )

        self.dropout1d = nn.Dropout1d(p=em_drop)

        # According to https://stackoverflow.com/questions/56915567/keras-vs-pytorch-lstm-different-results, the output of pytorch lstm need activation to resemble the lstm of keras. The default activation is tanh.
        self.lstm = nn.LSTM(
            input_size=em_dim,
            hidden_size=rnn_units,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_ac = nn.Sequential(
            nn.Tanh(),
            Rearrange("b l e -> b (l e)"),
        )

        if fc_activation == "elu":
            self.fc_activation = nn.ELU()
        elif fc_activation == "relu":
            self.fc_activation = nn.ReLU()
        elif fc_activation == "tanh":
            self.fc_activation = nn.Tanh()
        elif fc_activation == "sigmoid":
            self.fc_activation = nn.Sigmoid()
        else:
            assert (
                fc_activation == "hard_sigmoid"
            ), f"unknown fc_activation {fc_activation}"
            self.fc_activation = nn.Hardsigmoid()

        # 22 is "S" + sgRNA21mer
        self.fc1 = nn.Sequential(
            nn.Linear(
                22 * rnn_units * 2 + 11,
                fc_num_units,
            ),
            self.fc_activation,
            nn.Dropout(fc_drop),
        )
        self.fcs = nn.Sequential(
            *sum(
                [
                    [
                        nn.Linear(fc_num_units, fc_num_units),
                        self.fc_activation,
                        nn.Dropout(fc_drop),
                    ]
                    for _ in range(1, fc_num_hidden_layers)
                ],
                [],
            )
        )

        out_dim = (ext1_up + ext1_down + 1) * (ext2_up + ext2_down + 1)
        self.mix_output = nn.Linear(fc_num_units, out_dim)

    def forward(
        self, input: dict, label: Optional[dict], my_generator: Optional[MyGenerator]
    ) -> dict:
        X = self.embedding(input["X"].to(self.device))
        X = self.dropout1d(X)
        X, _ = self.lstm(X)
        X = self.lstm_ac(X)
        X = self.fc1(
            torch.cat(
                [
                    X,
                    input["biological_input"].to(self.device),
                ],
                dim=1,
            )
        )
        X = self.fcs(X)
        logit = self.mix_output(X)
        if label is not None:
            observation = torch.stack(
                [
                    ob[
                        c2 - self.ext2_up : c2 + self.ext2_down + 1,
                        c1 - self.ext1_up : c1 + self.ext1_down + 1,
                    ]
                    for ob, c1, c2 in zip(
                        label["observation"], label["cut1"], label["cut2"]
                    )
                ]
            ).to(self.device)
            # negative log likelihood
            loss, loss_num = self.loss_fun(
                logit,
                observation,
            )
            return {
                "logit": logit,
                "loss": loss,
                "loss_num": loss_num,
            }
        return {"logit": logit}

    def loss_fun(self, logit: torch.Tensor, observation: torch.Tensor) -> float:
        loss = -einsum(
            F.log_softmax(logit, dim=1),
            rearrange(observation, "b r2 r1 -> b (r2 r1)"),
            "b f, b f ->",
        )
        loss_num = einsum(observation, "b r2 r1 ->")
        return loss, loss_num

    def eval_output(
        self, examples: list[dict], batch: dict, my_generator: MyGenerator
    ) -> pd.DataFrame:
        result = self(input=batch["input"], label=None, my_generator=None)

        probas = F.softmax(result["logit"], dim=1).cpu().numpy()
        batch_size = probas.shape[0]
        ref1_dim = self.ext1_up + self.ext1_down + 1
        ref2_dim = self.ext2_up + self.ext2_down + 1
        df = pd.DataFrame(
            {
                "sample_idx": repeat(
                    np.arange(batch_size),
                    "b -> (b r2 r1)",
                    r1=ref1_dim,
                    r2=ref2_dim,
                ),
                "proba": probas.flatten(),
                "rpos1": repeat(
                    np.arange(-self.ext1_up, self.ext1_down + 1),
                    "r1 -> (b r2 r1)",
                    b=batch_size,
                    r2=ref2_dim,
                ),
                "rpos2": repeat(
                    np.arange(-self.ext2_up, self.ext2_down + 1),
                    "r2 -> (b r2 r1)",
                    b=batch_size,
                    r1=ref1_dim,
                ),
            }
        )
        return df

    @classmethod
    def my_hpo(cls, trial: optuna.Trial) -> tuple[jsonargparse.Namespace, dict]:
        hparam_dict = {
            "em_drop": trial.suggest_float("em_drop", 0.0, 0.2),
            "fc_drop": trial.suggest_float("fc_drop", 0.0, 0.4),
            "em_dim": trial.suggest_int("em_dim", 33, 55),
            "rnn_units": trial.suggest_int("rnn_units", 50, 70),
            "fc_num_hidden_layers": trial.suggest_int("fc_num_hidden_layers", 2, 5),
            "fc_num_units": trial.suggest_int("fc_num_units", 220, 420),
            "fc_activation": trial.suggest_categorical(
                "fc_activation",
                choices=["elu", "relu", "tanh", "sigmoid", "hard_sigmoid"],
            ),
        }
        cfg = jsonargparse.Namespace()
        cfg.init_args = jsonargparse.Namespace(
            ext1_up=25,
            ext1_down=6,
            ext2_up=6,
            ext2_down=25,
            **hparam_dict,
        )

        return cfg, hparam_dict


class MLP(nn.Module):
    def __init__(
        self,
        ext1_up: int,
        ext1_down: int,
        ext2_up: int,
        ext2_down: int,
        fc_drop: float,
        fc_num_hidden_layers: int,
        fc_num_units: int,
        fc_activation: Literal["elu", "relu", "tanh", "sigmoid", "hard_sigmoid"],
    ) -> None:
        """MLP arguments.

        Args:
            ext1_up: upstream limit of the resection of the upstream end.
            ext1_down: downstream limit of the templated insertion of the upstream end.
            ext2_up: upstream limit of the templated insertion of the downstream end.
            ext2_down: downstream limit of the resection of the downstream end.
            fc_drop: dropout probability of fully connected layer.
            fc_num_hidden_layers: number of output fully connected layers.
            fc_num_units: hidden dimension of output fully connected layers.
            fc_activation: activation function of output fully connected layers.
        """
        super().__init__()
        self.ext1_up = ext1_up
        self.ext1_down = ext1_down
        self.ext2_up = ext2_up
        self.ext2_down = ext2_down

        self.data_collator = DataCollator(
            ext1_up=ext1_up,
            ext1_down=ext1_down,
            ext2_up=ext2_up,
            ext2_down=ext2_down,
        )

        if fc_activation == "elu":
            self.fc_activation = nn.ELU()
        elif fc_activation == "relu":
            self.fc_activation = nn.ReLU()
        elif fc_activation == "tanh":
            self.fc_activation = nn.Tanh()
        elif fc_activation == "sigmoid":
            self.fc_activation = nn.Sigmoid()
        else:
            assert (
                fc_activation == "hard_sigmoid"
            ), f"unknown fc_activation {fc_activation}"
            self.fc_activation = nn.Hardsigmoid()

        # 22 is "S" + sgRNA21mer, 6 is PSACGT
        self.fc1 = nn.Sequential(
            nn.Linear(
                22 * 6 + 11,
                fc_num_units,
            ),
            self.fc_activation,
            nn.Dropout(fc_drop),
        )
        self.fcs = nn.Sequential(
            *sum(
                [
                    [
                        nn.Linear(fc_num_units, fc_num_units),
                        self.fc_activation,
                        nn.Dropout(fc_drop),
                    ]
                    for _ in range(1, fc_num_hidden_layers)
                ],
                [],
            )
        )

        out_dim = (ext1_up + ext1_down + 1) * (ext2_up + ext2_down + 1)
        self.mix_output = nn.Linear(fc_num_units, out_dim)

    def forward(
        self, input: dict, label: Optional[dict], my_generator: Optional[MyGenerator]
    ) -> dict:
        X = self.fc1(
            torch.cat(
                [
                    rearrange(
                        F.one_hot(input["X"].to(self.device), num_classes=6),
                        "b s h -> b (s h)",
                    ),
                    input["biological_input"].to(self.device),
                ],
                dim=1,
            )
        )
        X = self.fcs(X)
        logit = self.mix_output(X)
        if label is not None:
            observation = torch.stack(
                [
                    ob[
                        c2 - self.ext2_up : c2 + self.ext2_down + 1,
                        c1 - self.ext1_up : c1 + self.ext1_down + 1,
                    ]
                    for ob, c1, c2 in zip(
                        label["observation"], label["cut1"], label["cut2"]
                    )
                ]
            ).to(self.device)
            # negative log likelihood
            loss, loss_num = self.loss_fun(
                logit,
                observation,
            )
            return {
                "logit": logit,
                "loss": loss,
                "loss_num": loss_num,
            }
        return {"logit": logit}

    def loss_fun(self, logit: torch.Tensor, observation: torch.Tensor) -> float:
        loss = -einsum(
            F.log_softmax(logit, dim=1),
            rearrange(observation, "b r2 r1 -> b (r2 r1)"),
            "b f, b f ->",
        )
        loss_num = einsum(observation, "b r2 r1 ->")
        return loss, loss_num

    def eval_output(
        self, examples: list[dict], batch: dict, my_generator: MyGenerator
    ) -> pd.DataFrame:
        result = self(input=batch["input"], label=None, my_generator=None)

        probas = F.softmax(result["logit"], dim=1).cpu().numpy()
        batch_size = probas.shape[0]
        ref1_dim = self.ext1_up + self.ext1_down + 1
        ref2_dim = self.ext2_up + self.ext2_down + 1
        df = pd.DataFrame(
            {
                "sample_idx": repeat(
                    np.arange(batch_size),
                    "b -> (b r2 r1)",
                    r1=ref1_dim,
                    r2=ref2_dim,
                ),
                "proba": probas.flatten(),
                "rpos1": repeat(
                    np.arange(-self.ext1_up, self.ext1_down + 1),
                    "r1 -> (b r2 r1)",
                    b=batch_size,
                    r2=ref2_dim,
                ),
                "rpos2": repeat(
                    np.arange(-self.ext2_up, self.ext2_down + 1),
                    "r2 -> (b r2 r1)",
                    b=batch_size,
                    r1=ref1_dim,
                ),
            }
        )
        return df

    @classmethod
    def my_hpo(cls, trial: optuna.Trial) -> tuple[jsonargparse.Namespace, dict]:
        hparam_dict = {
            "fc_drop": trial.suggest_float("fc_drop", 0.0, 0.2),
            "fc_num_hidden_layers": trial.suggest_int("fc_num_hidden_layers", 3, 5),
            "fc_num_units": trial.suggest_int("fc_num_units", 300, 500),
            "fc_activation": trial.suggest_categorical(
                "fc_activation",
                choices=["elu", "relu", "tanh", "sigmoid", "hard_sigmoid"],
            ),
        }
        cfg = jsonargparse.Namespace()
        cfg.init_args = jsonargparse.Namespace(
            ext1_up=25,
            ext1_down=6,
            ext2_up=6,
            ext2_down=25,
            **hparam_dict,
        )

        return cfg, hparam_dict


class CNN(nn.Module):
    def __init__(
        self,
        ext1_up: int,
        ext1_down: int,
        ext2_up: int,
        ext2_down: int,
        em_drop: float,
        fc_drop: float,
        em_dim: int,
        fc_num_hidden_layers: int,
        fc_num_units: int,
        fc_activation: Literal["elu", "relu", "tanh", "sigmoid", "hard_sigmoid"],
        kernel_sizes: list[int],
        feature_maps: list[int],
    ) -> None:
        """CNN arguments.

        Args:
            ext1_up: upstream limit of the resection of the upstream end.
            ext1_down: downstream limit of the templated insertion of the upstream end.
            ext2_up: upstream limit of the templated insertion of the downstream end.
            ext2_down: downstream limit of the resection of the downstream end.
            em_drop: dropout probability of embedding.
            fc_drop: dropout probability of fully connected layer.
            em_dim: embedding dimension.
            fc_num_hidden_layers: number of output fully connected layers.
            fc_num_units: hidden dimension of output fully connected layers.
            fc_activation: activation function of output fully connected layers.
            kernel_sizes: kernel sizes for CNN.
            feature_maps: channel sizes of CNN.
        """
        super().__init__()
        self.ext1_up = ext1_up
        self.ext1_down = ext1_down
        self.ext2_up = ext2_up
        self.ext2_down = ext2_down

        self.data_collator = DataCollator(
            ext1_up=ext1_up,
            ext1_down=ext1_down,
            ext2_up=ext2_up,
            ext2_down=ext2_down,
        )

        self.embedding = nn.Embedding(
            num_embeddings=6,
            embedding_dim=em_dim,
        )

        self.dropout1d = nn.Dropout1d(p=em_drop)

        if fc_activation == "elu":
            self.fc_activation = nn.ELU()
        elif fc_activation == "relu":
            self.fc_activation = nn.ReLU()
        elif fc_activation == "tanh":
            self.fc_activation = nn.Tanh()
        elif fc_activation == "sigmoid":
            self.fc_activation = nn.Sigmoid()
        else:
            assert (
                fc_activation == "hard_sigmoid"
            ), f"unknown fc_activation {fc_activation}"
            self.fc_activation = nn.Hardsigmoid()

        self.cnns = nn.ModuleList(
            [
                nn.Sequential(
                    Rearrange("b l c -> b c l"),
                    nn.Conv1d(
                        in_channels=em_dim,
                        out_channels=feature_map,
                        kernel_size=kernel_size,
                        stride=1,
                        padding="same",
                    ),
                    self.fc_activation,
                    nn.MaxPool1d(kernel_size=22),
                    Rearrange("b c 1 -> b c"),
                )
                for kernel_size, feature_map in zip(kernel_sizes, feature_maps)
            ]
        )

        self.fc1 = nn.Sequential(
            nn.Linear(
                sum(feature_maps) + 11,
                fc_num_units,
            ),
            self.fc_activation,
            nn.Dropout(fc_drop),
        )
        self.fcs = nn.Sequential(
            *sum(
                [
                    [
                        nn.Linear(fc_num_units, fc_num_units),
                        self.fc_activation,
                        nn.Dropout(fc_drop),
                    ]
                    for _ in range(1, fc_num_hidden_layers)
                ],
                [],
            )
        )

        out_dim = (ext1_up + ext1_down + 1) * (ext2_up + ext2_down + 1)
        self.mix_output = nn.Linear(fc_num_units, out_dim)

    def forward(
        self, input: dict, label: Optional[dict], my_generator: Optional[MyGenerator]
    ) -> dict:
        X = self.embedding(input["X"].to(self.device))
        X = self.dropout1d(X)
        X = torch.cat(
            [cnn(X) for cnn in self.cnns],
            dim=1,
        )
        X = self.fc1(
            torch.cat(
                [
                    X,
                    input["biological_input"].to(self.device),
                ],
                dim=1,
            )
        )
        X = self.fcs(X)
        logit = self.mix_output(X)
        if label is not None:
            observation = torch.stack(
                [
                    ob[
                        c2 - self.ext2_up : c2 + self.ext2_down + 1,
                        c1 - self.ext1_up : c1 + self.ext1_down + 1,
                    ]
                    for ob, c1, c2 in zip(
                        label["observation"], label["cut1"], label["cut2"]
                    )
                ]
            ).to(self.device)
            # negative log likelihood
            loss, loss_num = self.loss_fun(
                logit,
                observation,
            )
            return {
                "logit": logit,
                "loss": loss,
                "loss_num": loss_num,
            }
        return {"logit": logit}

    def loss_fun(self, logit: torch.Tensor, observation: torch.Tensor) -> float:
        loss = -einsum(
            F.log_softmax(logit, dim=1),
            rearrange(observation, "b r2 r1 -> b (r2 r1)"),
            "b f, b f ->",
        )
        loss_num = einsum(observation, "b r2 r1 ->")
        return loss, loss_num

    def eval_output(
        self, examples: list[dict], batch: dict, my_generator: MyGenerator
    ) -> pd.DataFrame:
        result = self(input=batch["input"], label=None, my_generator=None)

        probas = F.softmax(result["logit"], dim=1).cpu().numpy()
        batch_size = probas.shape[0]
        ref1_dim = self.ext1_up + self.ext1_down + 1
        ref2_dim = self.ext2_up + self.ext2_down + 1
        df = pd.DataFrame(
            {
                "sample_idx": repeat(
                    np.arange(batch_size),
                    "b -> (b r2 r1)",
                    r1=ref1_dim,
                    r2=ref2_dim,
                ),
                "proba": probas.flatten(),
                "rpos1": repeat(
                    np.arange(-self.ext1_up, self.ext1_down + 1),
                    "r1 -> (b r2 r1)",
                    b=batch_size,
                    r2=ref2_dim,
                ),
                "rpos2": repeat(
                    np.arange(-self.ext2_up, self.ext2_down + 1),
                    "r2 -> (b r2 r1)",
                    b=batch_size,
                    r1=ref1_dim,
                ),
            }
        )
        return df

    @classmethod
    def my_hpo(cls, trial: optuna.Trial) -> tuple[jsonargparse.Namespace, dict]:
        hparam_dict = {
            "em_drop": trial.suggest_float("em_drop", 0.0, 0.2),
            "fc_drop": trial.suggest_float("fc_drop", 0.0, 0.2),
            "em_dim": trial.suggest_int("em_dim", 26, 46),
            "fc_num_hidden_layers": trial.suggest_int("fc_num_hidden_layers", 2, 4),
            "fc_num_units": trial.suggest_int("fc_num_units", 300, 500),
            "fc_activation": trial.suggest_categorical(
                "fc_activation",
                choices=["elu", "relu", "tanh", "sigmoid", "hard_sigmoid"],
            ),
            "feature_maps": trial.suggest_categorical(
                "feature_maps",
                choices=[
                    [
                        20,
                        20,
                        20,
                        20,
                        20,
                        20,
                        20,
                        20,
                        20,
                        20,
                        20,
                        20,
                        20,
                        20,
                    ],
                    [
                        20,
                        20,
                        20,
                        20,
                        20,
                        20,
                        20,
                        40,
                        40,
                        40,
                        40,
                        40,
                        40,
                        40,
                    ],
                    [
                        20,
                        20,
                        20,
                        20,
                        40,
                        40,
                        40,
                        40,
                        80,
                        80,
                        80,
                        80,
                        80,
                        80,
                    ],
                    [
                        40,
                        40,
                        40,
                        40,
                        40,
                        40,
                        40,
                        40,
                        80,
                        80,
                        80,
                        80,
                        80,
                        80,
                    ],
                ],
            ),
        }
        cfg = jsonargparse.Namespace()
        cfg.init_args = jsonargparse.Namespace(
            ext1_up=25,
            ext1_down=6,
            ext2_up=6,
            ext2_down=25,
            kernel_sizes=[1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13],
            **hparam_dict,
        )
        hparam_dict["feature_maps"] = ":".join(
            [str(feature_map) for feature_map in hparam_dict["feature_maps"]]
        )

        return cfg, hparam_dict


class XGBoost:
    def __init__(
        self,
        ext1_up: int,
        ext1_down: int,
        ext2_up: int,
        ext2_down: int,
        eta: float,
        max_depth: int,
        subsample: float,
        reg_lambda: float,
        num_boost_round: int,
    ) -> None:
        """XGBoost arguments.

        Args:
            ext1_up: upstream limit of the resection of the upstream end.
            ext1_down: downstream limit of the templated insertion of the upstream end.
            ext2_up: upstream limit of the templated insertion of the downstream end.
            ext2_down: downstream limit of the resection of the downstream end.
            eta: Shrink of step size after each round.
            max_depth: maximum depth of a tree.
            subsample: subsample ratio of the training instances.
            reg_lambda: L2 regularization term on weights.
            num_boost_round: Number of trees generated in single epochs.
        """
        super().__init__()
        self.ext1_up = ext1_up
        self.ext1_down = ext1_down
        self.ext2_up = ext2_up
        self.ext2_down = ext2_down
        self.eta = eta
        self.max_depth = max_depth
        self.subsample = subsample
        self.reg_lambda = reg_lambda
        self.num_boost_round = num_boost_round

        self.data_collator = DataCollator(
            ext1_up=ext1_up,
            ext1_down=ext1_down,
            ext2_up=ext2_up,
            ext2_down=ext2_down,
        )
        self.booster = None

    def my_initialize_model(
        self, my_initializer: MyInitializer, my_generator: MyGenerator
    ):
        pass

    def eval_output(
        self, examples: list[dict], batch: dict, my_generator: MyGenerator
    ) -> pd.DataFrame:
        X_value = self._get_feature(
            input=batch["input"],
            label=None,
        )
        # Do not use inplace_predict because of WARNING of device.
        probas = self.booster.predict(
            data=xgb.DMatrix(
                data=X_value,
                feature_types=["c"] * 22 + ["q"] * 11,
                enable_categorical=True,
            )
        )
        batch_size = probas.shape[0]
        ref1_dim = self.ext1_up + self.ext1_down + 1
        ref2_dim = self.ext2_up + self.ext2_down + 1
        df = pd.DataFrame(
            {
                "sample_idx": repeat(
                    np.arange(batch_size),
                    "b -> (b r2 r1)",
                    r1=ref1_dim,
                    r2=ref2_dim,
                ),
                "proba": probas.flatten(),
                "rpos1": repeat(
                    np.arange(-self.ext1_up, self.ext1_down + 1),
                    "r1 -> (b r2 r1)",
                    b=batch_size,
                    r2=ref2_dim,
                ),
                "rpos2": repeat(
                    np.arange(-self.ext2_up, self.ext2_down + 1),
                    "r2 -> (b r2 r1)",
                    b=batch_size,
                    r1=ref1_dim,
                ),
            }
        )
        return df

    def state_dict(self) -> dict:
        return {"booster": torch.frombuffer(self.booster.save_raw(), dtype=torch.uint8)}

    def load_state_dict(self, state_dict: dict) -> None:
        self.booster = xgb.Booster(
            model_file=bytearray(state_dict["booster"].numpy().tobytes())
        )

    def my_train_epoch(
        self,
        my_train: MyTrain,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader,
        my_generator: MyGenerator,
        my_optimizer: MyOptimizer,
    ):
        if not hasattr(self, "Xy_train") or not hasattr(self, "train_loss_num"):
            X_train, y_train, w_train = [], [], []
            for examples in tqdm(train_dataloader):
                batch = self.data_collator(
                    examples, output_label=True, my_generator=my_generator
                )
                X_value, y_value, w_value = self._get_feature(
                    input=batch["input"], label=batch["label"]
                )
                X_train.append(X_value)
                y_train.append(y_value)
                w_train.append(w_value)
            X_train = np.concatenate(X_train)
            y_train = np.concatenate(y_train)
            w_train = np.concatenate(w_train)

            self.Xy_train = xgb.QuantileDMatrix(
                data=X_train,
                label=y_train,
                weight=w_train,
                feature_types=["c"] * 22 + ["q"] * 11,
                enable_categorical=True,
            )
            self.train_loss_num = w_train.sum().item()

        num_class = (self.ext1_up + self.ext1_down + 1) * (
            self.ext2_up + self.ext2_down + 1
        )
        evals_result = {}
        self.booster = xgb.train(
            params={
                "device": self.device,
                "eta": self.eta,
                "max_depth": self.max_depth,
                "subsample": self.subsample,
                "reg_lambda": self.reg_lambda,
                "objective": "multi:softprob",
                "num_class": num_class,
                "seed": my_generator.seed,
            },
            dtrain=self.Xy_train,
            num_boost_round=self.num_boost_round,
            # put Xy_eval at the last in evals because early stopping use the last dataset in evals by default
            evals=[(self.Xy_train, "train")],
            evals_result=evals_result,
            xgb_model=self.booster,
        )

        return (
            evals_result["train"]["mlogloss"][0] * self.train_loss_num,
            self.train_loss_num,
            float("nan"),
        )

    def my_eval_epoch(
        self,
        my_train: MyTrain,
        eval_dataloader: torch.utils.data.DataLoader,
        my_generator: MyGenerator,
        metrics: dict,
    ):
        if not hasattr(self, "Xy_eval") or not hasattr(self, "eval_loss_num"):
            X_eval, y_eval, w_eval = [], [], []
            for examples in tqdm(eval_dataloader):
                batch = self.data_collator(
                    examples, output_label=True, my_generator=my_generator
                )
                X_value, y_value, w_value = self._get_feature(
                    input=batch["input"], label=batch["label"]
                )
                X_eval.append(X_value)
                y_eval.append(y_value)
                w_eval.append(w_value)
            X_eval = np.concatenate(X_eval)
            y_eval = np.concatenate(y_eval)
            w_eval = np.concatenate(w_eval)

            # Use QuantileDMatrix for evaluation and test is not recommanded because it needs train data as ref, which defeats the purpose of saving memory. See https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.QuantileDMatrix and https://www.kaggle.com/code/cdeotte/xgboost-using-original-data-cv-0-976?scriptVersionId=257750413&cellId=24
            self.Xy_eval = xgb.DMatrix(
                data=X_eval,
                label=y_eval,
                weight=w_eval,
                feature_types=["c"] * 22 + ["q"] * 11,
                enable_categorical=True,
            )
            self.eval_loss_num = w_eval.sum().item()

        eval_loss = (
            float(self.booster.eval(self.Xy_eval).split(":")[1]) * self.eval_loss_num
        )
        for examples in tqdm(eval_dataloader):
            batch = self.data_collator(
                examples, output_label=True, my_generator=my_generator
            )
            df = self.eval_output(examples, batch, my_generator)
            for metric_name, metric_fun in metrics.items():
                metric_fun.step(
                    df=df,
                    examples=examples,
                    batch=batch,
                )

        metric_loss_dict = {}
        for metric_name, metric_fun in metrics.items():
            metric_loss_dict[metric_name] = metric_fun.epoch()

        return eval_loss, self.eval_loss_num, metric_loss_dict

    def _get_feature(
        self,
        input: dict,
        label: Optional[dict],
    ) -> tuple[np.ndarray]:
        X_value = np.concatenate(
            (
                input["X"].cpu().numpy(),
                input["biological_input"].cpu().numpy(),
            ),
            axis=1,
        )

        if label is not None:
            observation = (
                rearrange(
                    torch.stack(
                        [
                            ob[
                                c2 - self.ext2_up : c2 + self.ext2_down + 1,
                                c1 - self.ext1_up : c1 + self.ext1_down + 1,
                            ]
                            for ob, c1, c2 in zip(
                                label["observation"], label["cut1"], label["cut2"]
                            )
                        ]
                    ),
                    "b r2 r1 -> b (r2 r1)",
                )
                .cpu()
                .numpy()
            )
            sample_indices, y_value = observation.nonzero()
            w_value = observation[sample_indices, y_value]
            X_value = X_value[sample_indices]
            return X_value, y_value, w_value

        return X_value

    @classmethod
    def my_hpo(cls, trial: optuna.Trial) -> tuple[jsonargparse.Namespace, dict]:
        hparam_dict = {
            "eta": trial.suggest_float("eta", 0.05, 0.2),
            "max_depth": trial.suggest_int("max_depath", 4, 6),
            "subsample": trial.suggest_float("subsample", 0.8, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 400.0, 1000.0),
            "num_boost_round": trial.suggest_int("num_boost_round", 10, 20),
        }
        cfg = jsonargparse.Namespace()
        cfg.init_args = jsonargparse.Namespace(
            ext1_up=25,
            ext1_down=6,
            ext2_up=6,
            ext2_down=25,
            **hparam_dict,
        )
        return cfg, hparam_dict


class SGDClassifier:
    def __init__(
        self,
        ext1_up: int,
        ext1_down: int,
        ext2_up: int,
        ext2_down: int,
        penalty: Optional[Literal["l2", "l1", "elasticnet"]],
        alpha: float,
        l1_ratio: float,
    ) -> None:
        """SGDClassifier arguments.

        Args:
            ext1_up: upstream limit of the resection of the upstream end.
            ext1_down: downstream limit of the templated insertion of the upstream end.
            ext2_up: upstream limit of the templated insertion of the downstream end.
            ext2_down: downstream limit of the resection of the downstream end.
            penalty: regularization type among l2, l1, l2/l1 (elasticnet), None.
            alpha: constant that multiplies the penalty term, controlling regularization strength.
            l1_ratio: ratio of l1 regularization, only relevant for elasticnet.
        """
        super().__init__()
        self.ext1_up = ext1_up
        self.ext1_down = ext1_down
        self.ext2_up = ext2_up
        self.ext2_down = ext2_down

        self.data_collator = DataCollator(
            ext1_up=ext1_up,
            ext1_down=ext1_down,
            ext2_up=ext2_up,
            ext2_down=ext2_down,
        )
        self.sgd_classifier = linear_model.SGDClassifier(
            loss="log_loss",
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            n_jobs=-1,
        )
        self.classes = np.arange((ext1_up + ext1_down + 1) * (ext2_up + ext2_down + 1))

    def my_initialize_model(
        self, my_initializer: MyInitializer, my_generator: MyGenerator
    ):
        pass

    def eval_output(
        self, examples: list[dict], batch: dict, my_generator: MyGenerator
    ) -> pd.DataFrame:
        probas = preprocessing.normalize(
            np.maximum(
                special.expit(
                    self.sgd_classifier.decision_function(
                        X=self._get_feature(
                            input=batch["input"],
                            label=None,
                        )
                    ),
                ),
                1e-10,
            ),
            norm="l1",
            axis=1,
        )
        batch_size = probas.shape[0]
        ref1_dim = self.ext1_up + self.ext1_down + 1
        ref2_dim = self.ext2_up + self.ext2_down + 1
        df = pd.DataFrame(
            {
                "sample_idx": repeat(
                    np.arange(batch_size),
                    "b -> (b r2 r1)",
                    r1=ref1_dim,
                    r2=ref2_dim,
                ),
                "proba": probas.flatten(),
                "rpos1": repeat(
                    np.arange(-self.ext1_up, self.ext1_down + 1),
                    "r1 -> (b r2 r1)",
                    b=batch_size,
                    r2=ref2_dim,
                ),
                "rpos2": repeat(
                    np.arange(-self.ext2_up, self.ext2_down + 1),
                    "r2 -> (b r2 r1)",
                    b=batch_size,
                    r1=ref1_dim,
                ),
            }
        )
        return df

    def state_dict(self) -> dict:
        return {
            "sgd_classifier": torch.frombuffer(
                bytearray(pickle.dumps(self.sgd_classifier)), dtype=torch.uint8
            )
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.sgd_classifier = pickle.loads(
            state_dict["sgd_classifier"].numpy().tobytes()
        )

    def my_train_epoch(
        self,
        my_train: MyTrain,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader,
        my_generator: MyGenerator,
        my_optimizer: MyOptimizer,
    ) -> None:
        train_loss, train_loss_num = 0.0, 0.0
        for examples in tqdm(train_dataloader):
            batch = self.data_collator(
                examples, output_label=True, my_generator=my_generator
            )
            X_value, y_value, w_value = self._get_feature(
                input=batch["input"], label=batch["label"]
            )
            self.sgd_classifier.partial_fit(
                X=X_value,
                y=y_value,
                classes=self.classes,
                sample_weight=w_value,
            )
            train_loss += (
                (
                    np.ma.log(
                        preprocessing.normalize(
                            np.maximum(
                                special.expit(
                                    self.sgd_classifier.decision_function(X=X_value)
                                ),
                                1e-10,
                            ),
                            norm="l1",
                            axis=1,
                        )
                    ).filled(-1000)[np.arange(y_value.shape[0]), y_value]
                    * w_value
                )
                .sum()
                .item()
            )
            train_loss_num += w_value.sum().item()

        return train_loss, train_loss_num, float("nan")

    def my_eval_epoch(
        self,
        my_train: MyTrain,
        eval_dataloader: torch.utils.data.DataLoader,
        my_generator: MyGenerator,
        metrics: dict,
    ):
        eval_loss, eval_loss_num = 0.0, 0.0
        for examples in tqdm(eval_dataloader):
            batch = self.data_collator(
                examples, output_label=True, my_generator=my_generator
            )
            X_value, y_value, w_value = self._get_feature(
                input=batch["input"], label=batch["label"]
            )
            eval_loss += (
                (
                    np.ma.log(
                        preprocessing.normalize(
                            np.maximum(
                                special.expit(
                                    self.sgd_classifier.decision_function(X=X_value)
                                ),
                                1e-10,
                            ),
                            norm="l1",
                            axis=1,
                        )
                    ).filled(-1000)[np.arange(y_value.shape[0]), y_value]
                    * w_value
                )
                .sum()
                .item()
            )
            eval_loss_num += w_value.sum().item()
            df = self.eval_output(examples, batch, my_generator)
            for metric_name, metric_fun in metrics.items():
                metric_fun.step(
                    df=df,
                    examples=examples,
                    batch=batch,
                )

        metric_loss_dict = {}
        for metric_name, metric_fun in metrics.items():
            metric_loss_dict[metric_name] = metric_fun.epoch()

        return eval_loss, eval_loss_num, metric_loss_dict

    def _get_feature(
        self,
        input: dict,
        label: Optional[dict],
    ) -> tuple[np.ndarray]:
        X_value = np.concatenate(
            (
                rearrange(
                    F.one_hot(input["X"], num_classes=6),
                    "b s h -> b (s h)",
                )
                .cpu()
                .numpy(),
                input["biological_input"].cpu().numpy(),
            ),
            axis=1,
        )

        if label is not None:
            observation = (
                rearrange(
                    torch.stack(
                        [
                            ob[
                                c2 - self.ext2_up : c2 + self.ext2_down + 1,
                                c1 - self.ext1_up : c1 + self.ext1_down + 1,
                            ]
                            for ob, c1, c2 in zip(
                                label["observation"], label["cut1"], label["cut2"]
                            )
                        ]
                    ),
                    "b r2 r1 -> b (r2 r1)",
                )
                .cpu()
                .numpy()
            )
            sample_indices, y_value = observation.nonzero()
            w_value = observation[sample_indices, y_value]
            X_value = X_value[sample_indices]
            return X_value, y_value, w_value

        return X_value

    @classmethod
    def my_hpo(cls, trial: optuna.Trial) -> tuple[jsonargparse.Namespace, dict]:
        hparam_dict = {
            "penalty": trial.suggest_categorical(
                "penalty",
                choices=["l2", "l1", "elasticnet", None],
            ),
            "alpha": trial.suggest_float("alpha", 0.00005, 0.0002),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.075, 0.3),
        }
        cfg = jsonargparse.Namespace()
        cfg.init_args = jsonargparse.Namespace(
            ext1_up=25,
            ext1_down=6,
            ext2_up=6,
            ext2_down=25,
            **hparam_dict,
        )

        return cfg, hparam_dict
