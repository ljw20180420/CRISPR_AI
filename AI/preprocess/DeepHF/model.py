import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Literal, Optional

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops.layers.torch import Rearrange
from einops import rearrange, einsum, repeat
from ..model import BaseModel, BaseConfig


class DeepHFConfig(BaseConfig):
    model_type = "DeepHF"

    def __init__(
        self,
        ext1_up: int,
        ext1_down: int,
        ext2_up: int,
        ext2_down: int,
        seq_length: int,
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
            seq_length: input sequence length.
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
        self.seq_length = seq_length
        self.em_drop = em_drop
        self.fc_drop = fc_drop
        self.em_dim = em_dim
        self.rnn_units = rnn_units
        self.fc_num_hidden_layers = fc_num_hidden_layers
        self.fc_num_units = fc_num_units
        self.fc_activation = fc_activation
        super().__init__(**kwargs)


class DeepHFModel(BaseModel):
    config_class = DeepHFConfig

    def __init__(self, config) -> None:
        super().__init__(config)

        self.embedding = nn.Embedding(
            num_embeddings=6,
            embedding_dim=self.config.em_dim,
        )

        self.dropout1d = nn.Dropout1d(p=self.config.em_drop)

        # According to https://stackoverflow.com/questions/56915567/keras-vs-pytorch-lstm-different-results, the output of pytorch lstm need activation to resemble the lstm of keras. The default activation is tanh.
        self.lstm = nn.LSTM(
            input_size=self.config.em_dim,
            hidden_size=self.config.rnn_units,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_ac = nn.Sequential(
            nn.Tanh(),
            Rearrange("b l e -> b (l e)"),
        )

        if self.config.fc_activation == "elu":
            self.fc_activation = nn.ELU()
        elif self.config.fc_activation == "relu":
            self.fc_activation = nn.ReLU()
        elif self.config.fc_activation == "tanh":
            self.fc_activation = nn.Tanh()
        elif self.config.fc_activation == "sigmoid":
            self.fc_activation = nn.Sigmoid()
        else:
            assert (
                self.config.fc_activation == "hard_sigmoid"
            ), f"unknown fc_activation {self.config.fc_activation}"
            self.fc_activation = nn.Hardsigmoid()

        self.fc1 = nn.Sequential(
            nn.Linear(
                self.config.seq_length * self.config.rnn_units * 2 + 11,
                self.config.fc_num_units,
            ),
            self.fc_activation,
            nn.Dropout(self.config.fc_drop),
        )
        self.fcs = nn.Sequential(
            *sum(
                [
                    [
                        nn.Linear(self.config.fc_num_units, self.config.fc_num_units),
                        self.fc_activation,
                        nn.Dropout(self.config.fc_drop),
                    ]
                    for _ in range(1, self.config.fc_num_hidden_layers)
                ],
                [],
            )
        )

        out_dim = (self.config.ext1_up + self.config.ext1_down + 1) * (
            self.config.ext2_up + self.config.ext2_down + 1
        )
        self.mix_output = nn.Linear(self.config.fc_num_units, out_dim)

        self._initialize_model_layer_weights()

    def forward(self, input: dict, label: Optional[dict] = None) -> dict:
        X = self.embedding(input["X"])
        X = self.dropout1d(X)
        X, _ = self.lstm(X)
        X = self.lstm_ac(X)
        X = self.fc1(
            torch.cat(
                [
                    X,
                    input["biological_input"],
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
            )
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
        result = self(input=batch["input"])

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
