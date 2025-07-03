from transformers import PretrainedConfig, PreTrainedModel
import torch.nn as nn
import torch
import torch.nn.functional as F

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops.layers.torch import Rearrange
from einops import rearrange


class DeepHFConfig(PretrainedConfig):
    model_type = "DeepHF"
    label_names = ["observation"]

    def __init__(
        self,
        seq_length: int,
        em_drop: float,
        fc_drop: float,
        initializer: str,
        em_dim: int,
        rnn_units: int,
        fc_num_hidden_layers: int,
        fc_num_units: int,
        fc_activation: str,
        ext1_up: int,
        ext1_down: int,
        ext2_up: int,
        ext2_down: int,
        seed: int,  # random seed for intialization
        **kwargs,
    ) -> None:
        self.seq_length = seq_length
        self.em_drop = em_drop
        self.fc_drop = fc_drop
        self.initializer = initializer
        self.em_dim = em_dim
        self.rnn_units = rnn_units
        self.fc_num_hidden_layers = fc_num_hidden_layers
        self.fc_num_units = fc_num_units
        self.fc_activation = fc_activation
        self.ext1_up = ext1_up
        self.ext1_down = ext1_down
        self.ext2_up = ext2_up
        self.ext2_down = ext2_down
        self.seed = seed
        super().__init__(**kwargs)


class DeepHFModel(PreTrainedModel):
    config_class = DeepHFConfig

    def __init__(self, config) -> None:
        super().__init__(config)
        # In more recent versions of PyTorch, you no longer need to explicitly register_parameter, it's enough to set a member of your nn.Module with nn.Parameter to "notify" pytorch that this variable should be treated as a trainable parameter (https://stackoverflow.com/questions/59234238/how-to-add-parameters-in-module-class-in-pytorch-custom-model).
        self.generator = torch.Generator().manual_seed(config.seed)

        self.embedding = nn.Embedding(
            num_embeddings=6,
            embedding_dim=self.config.em_dim,
        )

        self.dropout1d = nn.Dropout1d(p=self.config.em_drop)

        # According to https://stackoverflow.com/questions/56915567/keras-vs-pytorch-lstm-different-results, the output of pytorch lstm need activation to resemble the lstm of keras. The default activation is tanh.
        self.lstm = nn.Sequential(
            [
                nn.LSTM(
                    input_size=self.config.em_dim,
                    hidden_size=self.config.rnn_units,
                    num_layers=1,
                    bias=True,
                    batch_first=True,
                    bidirectional=True,
                ),
                nn.Tanh(),
                Rearrange("b l e -> b (l e)"),
            ]
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
            [
                nn.Linear(
                    self.config.seq_length * self.config.rnn_units + 11,
                    self.config.fc_num_units,
                ),
                self.fc_activation,
                nn.Dropout(self.config.fc_drop),
            ]
        )
        self.fcs = nn.Sequential(
            sum(
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

        self.initialize_weights()

    def initialize_weights(self):
        if self.config.initializer == "lecun_uniform":
            init_func = (
                lambda weight, generator=self.generator: nn.init.kaiming_uniform_(
                    weight, nonlinearity="linear", generator=generator
                )
            )
        elif self.config.initializer == "normal":
            init_func = lambda weight, generator=self.generator: nn.init.normal_(
                weight, generator=generator
            )
        elif self.config.initializer == "he_normal":
            init_func = nn.init.kaiming_normal_
        elif self.config.initializer == "he_uniform":
            init_func = nn.init.kaiming_uniform_

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, X, biological_input, observation=None) -> torch.Tensor:
        X = self.embedding(X)
        X = self.dropout1d(X)
        X, _ = self.lstm(X)
        X = self.fc1(
            torch.cat(
                [
                    X,
                    biological_input,
                ],
                dim=1,
            )
        )
        X = self.fcs(X)
        logit = self.mix_output(X)
        if observation is not None:
            return {
                "logit": logit,
                # negative log likelihood
                "loss": -(
                    rearrange(observation, "b r2 r1 -> b (r2 r1)")
                    * F.log_softmax(logit, dim=1)
                ).sum(),
            }
        return {"logit": logit}
