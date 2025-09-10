import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Literal, Optional
import os
import pathlib
from scipy.special import softmax
import pickle
import itertools
from tqdm import tqdm

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops.layers.torch import Rearrange
from einops import rearrange, einsum, repeat
from einops.layers.torch import Rearrange
from transformers import PreTrainedModel, PretrainedConfig

import xgboost as xgb

xgb.config_context(use_rmm=True)

from sklearn import linear_model
import jsonargparse
import logging
import datasets
from .data_collator import DataCollator
from common_ai.utils import MyGenerator


class DeepHFConfig(PretrainedConfig):
    model_type = "DeepHF"

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
        **kwargs,
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
        self.ext1_up = ext1_up
        self.ext1_down = ext1_down
        self.ext2_up = ext2_up
        self.ext2_down = ext2_down
        self.em_drop = em_drop
        self.fc_drop = fc_drop
        self.em_dim = em_dim
        self.rnn_units = rnn_units
        self.fc_num_hidden_layers = fc_num_hidden_layers
        self.fc_num_units = fc_num_units
        self.fc_activation = fc_activation
        super().__init__(**kwargs)


class DeepHFModel(PreTrainedModel):
    config_class = DeepHFConfig

    def __init__(self, config: DeepHFConfig) -> None:
        super().__init__(config)
        self.data_collator = DataCollator(
            ext1_up=config.ext1_up,
            ext1_down=config.ext1_down,
            ext2_up=config.ext2_up,
            ext2_down=config.ext2_down,
        )

        self.embedding = nn.Embedding(
            num_embeddings=6,
            embedding_dim=config.em_dim,
        )

        self.dropout1d = nn.Dropout1d(p=config.em_drop)

        # According to https://stackoverflow.com/questions/56915567/keras-vs-pytorch-lstm-different-results, the output of pytorch lstm need activation to resemble the lstm of keras. The default activation is tanh.
        self.lstm = nn.LSTM(
            input_size=config.em_dim,
            hidden_size=config.rnn_units,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_ac = nn.Sequential(
            nn.Tanh(),
            Rearrange("b l e -> b (l e)"),
        )

        if config.fc_activation == "elu":
            self.fc_activation = nn.ELU()
        elif config.fc_activation == "relu":
            self.fc_activation = nn.ReLU()
        elif config.fc_activation == "tanh":
            self.fc_activation = nn.Tanh()
        elif config.fc_activation == "sigmoid":
            self.fc_activation = nn.Sigmoid()
        else:
            assert (
                config.fc_activation == "hard_sigmoid"
            ), f"unknown fc_activation {config.fc_activation}"
            self.fc_activation = nn.Hardsigmoid()

        # 22 is "S" + sgRNA21mer
        self.fc1 = nn.Sequential(
            nn.Linear(
                22 * config.rnn_units * 2 + 11,
                config.fc_num_units,
            ),
            self.fc_activation,
            nn.Dropout(config.fc_drop),
        )
        self.fcs = nn.Sequential(
            *sum(
                [
                    [
                        nn.Linear(config.fc_num_units, config.fc_num_units),
                        self.fc_activation,
                        nn.Dropout(config.fc_drop),
                    ]
                    for _ in range(1, config.fc_num_hidden_layers)
                ],
                [],
            )
        )

        out_dim = (config.ext1_up + config.ext1_down + 1) * (
            config.ext2_up + config.ext2_down + 1
        )
        self.mix_output = nn.Linear(config.fc_num_units, out_dim)

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
                        c2 - self.config.ext2_up : c2 + self.config.ext2_down + 1,
                        c1 - self.config.ext1_up : c1 + self.config.ext1_down + 1,
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

    def eval_output(self, examples: list[dict], batch: dict) -> pd.DataFrame:
        result = self(input=batch["input"], label=None, my_generator=None)

        probas = F.softmax(result["logit"], dim=1).cpu().numpy()
        batch_size = probas.shape[0]
        ref1_dim = self.config.ext1_up + self.config.ext1_down + 1
        ref2_dim = self.config.ext2_up + self.config.ext2_down + 1
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
                    np.arange(-self.config.ext1_up, self.config.ext1_down + 1),
                    "r1 -> (b r2 r1)",
                    b=batch_size,
                    r2=ref2_dim,
                ),
                "rpos2": repeat(
                    np.arange(-self.config.ext2_up, self.config.ext2_down + 1),
                    "r2 -> (b r2 r1)",
                    b=batch_size,
                    r1=ref1_dim,
                ),
            }
        )
        return df


class MLPConfig(PretrainedConfig):
    model_type = "MLP"

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
        **kwargs,
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
        self.ext1_up = ext1_up
        self.ext1_down = ext1_down
        self.ext2_up = ext2_up
        self.ext2_down = ext2_down
        self.fc_drop = fc_drop
        self.fc_num_hidden_layers = fc_num_hidden_layers
        self.fc_num_units = fc_num_units
        self.fc_activation = fc_activation
        super().__init__(**kwargs)


class MLPModel(PreTrainedModel):
    config_class = MLPConfig

    def __init__(self, config: MLPConfig) -> None:
        super().__init__(config)
        self.data_collator = DataCollator(
            ext1_up=config.ext1_up,
            ext1_down=config.ext1_down,
            ext2_up=config.ext2_up,
            ext2_down=config.ext2_down,
        )

        if config.fc_activation == "elu":
            self.fc_activation = nn.ELU()
        elif config.fc_activation == "relu":
            self.fc_activation = nn.ReLU()
        elif config.fc_activation == "tanh":
            self.fc_activation = nn.Tanh()
        elif config.fc_activation == "sigmoid":
            self.fc_activation = nn.Sigmoid()
        else:
            assert (
                config.fc_activation == "hard_sigmoid"
            ), f"unknown fc_activation {config.fc_activation}"
            self.fc_activation = nn.Hardsigmoid()

        # 22 is "S" + sgRNA21mer, 6 is PSACGT
        self.fc1 = nn.Sequential(
            nn.Linear(
                22 * 6 + 11,
                config.fc_num_units,
            ),
            self.fc_activation,
            nn.Dropout(config.fc_drop),
        )
        self.fcs = nn.Sequential(
            *sum(
                [
                    [
                        nn.Linear(config.fc_num_units, config.fc_num_units),
                        self.fc_activation,
                        nn.Dropout(config.fc_drop),
                    ]
                    for _ in range(1, config.fc_num_hidden_layers)
                ],
                [],
            )
        )

        out_dim = (config.ext1_up + config.ext1_down + 1) * (
            config.ext2_up + config.ext2_down + 1
        )
        self.mix_output = nn.Linear(config.fc_num_units, out_dim)

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
                        c2 - self.config.ext2_up : c2 + self.config.ext2_down + 1,
                        c1 - self.config.ext1_up : c1 + self.config.ext1_down + 1,
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

    def eval_output(self, examples: list[dict], batch: dict) -> pd.DataFrame:
        result = self(input=batch["input"], label=None, my_generator=None)

        probas = F.softmax(result["logit"], dim=1).cpu().numpy()
        batch_size = probas.shape[0]
        ref1_dim = self.config.ext1_up + self.config.ext1_down + 1
        ref2_dim = self.config.ext2_up + self.config.ext2_down + 1
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
                    np.arange(-self.config.ext1_up, self.config.ext1_down + 1),
                    "r1 -> (b r2 r1)",
                    b=batch_size,
                    r2=ref2_dim,
                ),
                "rpos2": repeat(
                    np.arange(-self.config.ext2_up, self.config.ext2_down + 1),
                    "r2 -> (b r2 r1)",
                    b=batch_size,
                    r1=ref1_dim,
                ),
            }
        )
        return df


class CNNConfig(PretrainedConfig):
    model_type = "CNN"

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
        **kwargs,
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
        self.ext1_up = ext1_up
        self.ext1_down = ext1_down
        self.ext2_up = ext2_up
        self.ext2_down = ext2_down
        self.em_drop = em_drop
        self.fc_drop = fc_drop
        self.em_dim = em_dim
        self.fc_num_hidden_layers = fc_num_hidden_layers
        self.fc_num_units = fc_num_units
        self.fc_activation = fc_activation
        self.kernel_sizes = kernel_sizes
        self.feature_maps = feature_maps
        super().__init__(**kwargs)


class CNNModel(PreTrainedModel):
    config_class = CNNConfig

    def __init__(self, config: CNNConfig) -> None:
        super().__init__(config)
        self.data_collator = DataCollator(
            ext1_up=config.ext1_up,
            ext1_down=config.ext1_down,
            ext2_up=config.ext2_up,
            ext2_down=config.ext2_down,
        )

        self.embedding = nn.Embedding(
            num_embeddings=6,
            embedding_dim=config.em_dim,
        )

        self.dropout1d = nn.Dropout1d(p=config.em_drop)

        if config.fc_activation == "elu":
            self.fc_activation = nn.ELU()
        elif config.fc_activation == "relu":
            self.fc_activation = nn.ReLU()
        elif config.fc_activation == "tanh":
            self.fc_activation = nn.Tanh()
        elif config.fc_activation == "sigmoid":
            self.fc_activation = nn.Sigmoid()
        else:
            assert (
                config.fc_activation == "hard_sigmoid"
            ), f"unknown fc_activation {config.fc_activation}"
            self.fc_activation = nn.Hardsigmoid()

        self.cnns = nn.ModuleList(
            [
                nn.Sequential(
                    Rearrange("b l c -> b c l"),
                    nn.Conv1d(
                        in_channels=config.em_dim,
                        out_channels=feature_map,
                        kernel_size=kernel_size,
                        stride=1,
                        padding="same",
                    ),
                    self.fc_activation,
                    nn.MaxPool1d(kernel_size=22),
                    Rearrange("b c 1 -> b c"),
                )
                for kernel_size, feature_map in zip(
                    config.kernel_sizes, config.feature_maps
                )
            ]
        )

        self.fc1 = nn.Sequential(
            nn.Linear(
                sum(config.feature_maps) + 11,
                config.fc_num_units,
            ),
            self.fc_activation,
            nn.Dropout(config.fc_drop),
        )
        self.fcs = nn.Sequential(
            *sum(
                [
                    [
                        nn.Linear(config.fc_num_units, config.fc_num_units),
                        self.fc_activation,
                        nn.Dropout(config.fc_drop),
                    ]
                    for _ in range(1, config.fc_num_hidden_layers)
                ],
                [],
            )
        )

        out_dim = (config.ext1_up + config.ext1_down + 1) * (
            config.ext2_up + config.ext2_down + 1
        )
        self.mix_output = nn.Linear(config.fc_num_units, out_dim)

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
                        c2 - self.config.ext2_up : c2 + self.config.ext2_down + 1,
                        c1 - self.config.ext1_up : c1 + self.config.ext1_down + 1,
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

    def eval_output(self, examples: list[dict], batch: dict) -> pd.DataFrame:
        result = self(input=batch["input"], label=None, my_generator=None)

        probas = F.softmax(result["logit"], dim=1).cpu().numpy()
        batch_size = probas.shape[0]
        ref1_dim = self.config.ext1_up + self.config.ext1_down + 1
        ref2_dim = self.config.ext2_up + self.config.ext2_down + 1
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
                    np.arange(-self.config.ext1_up, self.config.ext1_down + 1),
                    "r1 -> (b r2 r1)",
                    b=batch_size,
                    r2=ref2_dim,
                ),
                "rpos2": repeat(
                    np.arange(-self.config.ext2_up, self.config.ext2_down + 1),
                    "r2 -> (b r2 r1)",
                    b=batch_size,
                    r1=ref1_dim,
                ),
            }
        )
        return df


class XGBoostConfig(PretrainedConfig):
    model_type = "XGBoost"

    def __init__(
        self,
        ext1_up: int,
        ext1_down: int,
        ext2_up: int,
        ext2_down: int,
        booster: Literal["gbtree", "dart"],
        device: Literal["gpu", "cpu"],
        learning_rate: float,
        subsample: float,
        colsample_bytree: float,
        max_depth: int,
        reg_lambda: float,
        num_boost_round: int,
        early_stopping_rounds: int,
        **kwargs,
    ) -> None:
        """XGBoost arguments.

        Args:
            ext1_up: upstream limit of the resection of the upstream end.
            ext1_down: downstream limit of the templated insertion of the upstream end.
            ext2_up: upstream limit of the templated insertion of the downstream end.
            ext2_down: downstream limit of the resection of the downstream end.
            booster: booster to use. gbtree and dart use tree based models. gblinear uses linear functions and does not support categorial feature.
            device: device to use, cpu or gpu.
            learning_rate: learning rate.
            subsample: subsample ratio of the training instances.
            colsample_bytree: subsample ratio of columns when constructing each tree.
            max_depth: maximum depth of a tree.
            reg_lambda: L2 regularization term on weights.
            num_boost_round: XGBoost iteration numbers.
            early_stopping_rounds: Early stopping rounds of XGBoost.
        """
        self.ext1_up = ext1_up
        self.ext1_down = ext1_down
        self.ext2_up = ext2_up
        self.ext2_down = ext2_down
        self.booster = booster
        self.device = device
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        super().__init__(**kwargs)


class XGBoostModel(PreTrainedModel):
    config_class = XGBoostConfig

    def __init__(self, config: XGBoostConfig) -> None:
        super().__init__(config)
        self.data_collator = DataCollator(
            ext1_up=config.ext1_up,
            ext1_down=config.ext1_down,
            ext2_up=config.ext2_up,
            ext2_down=config.ext2_down,
        )

    def eval_output(
        self,
        examples: list[dict],
        batch: dict,
    ) -> pd.DataFrame:
        X_value = self._get_feature(
            input=batch["input"],
            label=None,
        )
        probas = self.booster.predict(
            xgb.DMatrix(
                data=X_value,
                feature_types=["c"] * 22 + ["q"] * 11,
                enable_categorical=True,
            )
        )
        batch_size = probas.shape[0]
        ref1_dim = self.config.ext1_up + self.config.ext1_down + 1
        ref2_dim = self.config.ext2_up + self.config.ext2_down + 1
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
                    np.arange(-self.config.ext1_up, self.config.ext1_down + 1),
                    "r1 -> (b r2 r1)",
                    b=batch_size,
                    r2=ref2_dim,
                ),
                "rpos2": repeat(
                    np.arange(-self.config.ext2_up, self.config.ext2_down + 1),
                    "r2 -> (b r2 r1)",
                    b=batch_size,
                    r1=ref1_dim,
                ),
            }
        )
        return df

    def my_train_model(
        self,
        dataset: datasets.Dataset,
        batch_size: int,
        train_parser: jsonargparse.ArgumentParser,
        cfg: jsonargparse.Namespace,
        model_path: os.PathLike,
        logger: logging.Logger,
    ) -> None:
        model_path = pathlib.Path(os.fspath(model_path))
        logger.info("train XGBoost")
        self._train_XGBoost(
            train_dataloader=torch.utils.data.DataLoader(
                dataset=dataset["train"],
                batch_size=batch_size,
                collate_fn=lambda examples: examples,
            ),
            eval_dataloader=torch.utils.data.DataLoader(
                dataset=dataset["validation"],
                batch_size=batch_size,
                collate_fn=lambda examples: examples,
            ),
        )
        logger.info("save XGBoost")
        self._save_XGBoost(model_path)
        train_parser.save(
            cfg=cfg,
            path=model_path / "train.yaml",
            overwrite=True,
        )

    def my_load_model(self, model_path: os.PathLike) -> None:
        model_path = pathlib.Path(os.fspath(model_path))
        self.booster = xgb.Booster(model_file=model_path / "XGBoost.ubj")

    def _save_XGBoost(self, model_path: os.PathLike) -> None:
        model_path = pathlib.Path(os.fspath(model_path))
        os.makedirs(model_path, exist_ok=True)
        self.booster.save_model(fname=model_path / "XGBoost.ubj")

    def _train_XGBoost(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader,
    ) -> None:
        X_train, y_train, w_train = [], [], []
        for examples in tqdm(train_dataloader):
            batch = self.data_collator(examples, output_label=True)
            X_value, y_value, w_value = self._get_feature(
                input=batch["input"], label=batch["label"]
            )
            X_train.append(X_value)
            y_train.append(y_value)
            w_train.append(w_value)
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        w_train = np.concatenate(w_train)

        X_eval, y_eval, w_eval = [], [], []
        for examples in tqdm(eval_dataloader):
            batch = self.data_collator(examples, output_label=True)
            X_value, y_value, w_value = self._get_feature(
                input=batch["input"], label=batch["label"]
            )
            X_eval.append(X_value)
            y_eval.append(y_value)
            w_eval.append(w_value)
        X_eval = np.concatenate(X_eval)
        y_eval = np.concatenate(y_eval)
        w_eval = np.concatenate(w_eval)

        Xy_train = xgb.DMatrix(
            data=X_train,
            label=y_train,
            weight=w_train,
            feature_types=["c"] * 22 + ["q"] * 11,
            enable_categorical=True,
        )
        Xy_eval = xgb.DMatrix(
            data=X_eval,
            label=y_eval,
            weight=w_eval,
            feature_types=["c"] * 22 + ["q"] * 11,
            enable_categorical=True,
        )

        num_class = (self.config.ext1_up + self.config.ext1_down + 1) * (
            self.config.ext2_up + self.config.ext2_down + 1
        )
        self.booster = xgb.train(
            params={
                "booster": self.config.booster,
                "device": self.config.device,
                "learning_rate": self.config.learning_rate,
                "subsample": self.config.subsample,
                "colsample_bytree": self.config.colsample_bytree,
                "max_depth": self.config.max_depth,
                "reg_lambda": self.config.reg_lambda,
                "objective": "multi:softprob",
                "num_class": num_class,
                "seed": 63036,
            },
            dtrain=Xy_train,
            num_boost_round=self.config.num_boost_round,
            evals=[
                (Xy_train, "train"),
                (Xy_eval, "eval"),
            ],
            early_stopping_rounds=self.config.early_stopping_rounds,
        )

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
                                c2
                                - self.config.ext2_up : c2
                                + self.config.ext2_down
                                + 1,
                                c1
                                - self.config.ext1_up : c1
                                + self.config.ext1_down
                                + 1,
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


class RidgeConfig(PretrainedConfig):
    model_type = "Ridge"

    def __init__(
        self,
        ext1_up: int,
        ext1_down: int,
        ext2_up: int,
        ext2_down: int,
        alpha: float,
        **kwargs,
    ) -> None:
        """Ridge arguments.

        Args:
            ext1_up: upstream limit of the resection of the upstream end.
            ext1_down: downstream limit of the templated insertion of the upstream end.
            ext2_up: upstream limit of the templated insertion of the downstream end.
            ext2_down: downstream limit of the resection of the downstream end.
            alpha: constant that multiplies the L2 term, controlling regularization strength.
        """
        self.ext1_up = ext1_up
        self.ext1_down = ext1_down
        self.ext2_up = ext2_up
        self.ext2_down = ext2_down
        self.alpha = alpha
        super().__init__(**kwargs)


class RidgeModel(PreTrainedModel):
    config_class = RidgeConfig

    def __init__(self, config: RidgeConfig) -> None:
        super().__init__(config)
        self.data_collator = DataCollator(
            ext1_up=config.ext1_up,
            ext1_down=config.ext1_down,
            ext2_up=config.ext2_up,
            ext2_down=config.ext2_down,
        )

    def eval_output(
        self,
        examples: list[dict],
        batch: dict,
    ) -> pd.DataFrame:
        X_value = self._get_feature(
            input=batch["input"],
            label=None,
        )
        logits = self.ridge.decision_function(X_value)
        probas = softmax(logits, axis=1)
        batch_size = probas.shape[0]
        ref1_dim = self.config.ext1_up + self.config.ext1_down + 1
        ref2_dim = self.config.ext2_up + self.config.ext2_down + 1
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
                    np.arange(-self.config.ext1_up, self.config.ext1_down + 1),
                    "r1 -> (b r2 r1)",
                    b=batch_size,
                    r2=ref2_dim,
                ),
                "rpos2": repeat(
                    np.arange(-self.config.ext2_up, self.config.ext2_down + 1),
                    "r2 -> (b r2 r1)",
                    b=batch_size,
                    r1=ref1_dim,
                ),
            }
        )
        return df

    def my_train_model(
        self,
        dataset: datasets.Dataset,
        batch_size: int,
        train_parser: jsonargparse.ArgumentParser,
        cfg: jsonargparse.Namespace,
        model_path: os.PathLike,
        logger: logging.Logger,
    ) -> None:
        model_path = pathlib.Path(os.fspath(model_path))
        logger.info("train Ridge")
        self._train_Ridge(
            train_dataloader=torch.utils.data.DataLoader(
                dataset=dataset["train"],
                batch_size=batch_size,
                collate_fn=lambda examples: examples,
            ),
            eval_dataloader=torch.utils.data.DataLoader(
                dataset=dataset["validation"],
                batch_size=batch_size,
                collate_fn=lambda examples: examples,
            ),
        )
        logger.info("save Ridge")
        self._save_Ridge(model_path)
        train_parser.save(
            cfg=cfg,
            path=model_path / "train.yaml",
            overwrite=True,
        )

    def my_load_model(self, model_path: os.PathLike) -> None:
        model_path = pathlib.Path(os.fspath(model_path))
        with open(model_path / "ridge.pkl", "rb") as fd:
            self.ridge = pickle.load(fd)

    def _save_Ridge(self, model_path: os.PathLike) -> None:
        model_path = pathlib.Path(os.fspath(model_path))
        os.makedirs(model_path, exist_ok=True)
        with open(model_path / "ridge.pkl", "wb") as fd:
            pickle.dump(self.ridge, fd)

    def _train_Ridge(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader,
    ) -> None:
        X_train, y_train, w_train = [], [], []
        for examples in tqdm(itertools.chain(train_dataloader, eval_dataloader)):
            batch = self.data_collator(examples, output_label=True)
            X_value, y_value, w_value = self._get_feature(
                input=batch["input"], label=batch["label"]
            )
            X_train.append(X_value)
            y_train.append(y_value)
            w_train.append(w_value)
        # Append all 1024 classes with zero weights to force ridge's internal classes_ equals 1024.
        X_train.append(X_train[-1][[-1] * 1024, :])
        y_train.append(np.arange(1024))
        w_train.append(np.zeros(1024))
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        w_train = np.concatenate(w_train)

        self.ridge = linear_model.RidgeClassifier(
            alpha=self.config.alpha,
            fit_intercept=True,
            copy_X=True,
            max_iter=None,
            solver="auto",
        )
        self.ridge.fit(X=X_train, y=y_train, sample_weight=w_train)

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
                                c2
                                - self.config.ext2_up : c2
                                + self.config.ext2_down
                                + 1,
                                c1
                                - self.config.ext1_up : c1
                                + self.config.ext1_down
                                + 1,
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
