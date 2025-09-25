from transformers import (
    RoFormerConfig,
    RoFormerModel,
)
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Optional
import optuna
import jsonargparse

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import einsum, repeat, rearrange

from .data_collator import DataCollator
from common_ai.generator import MyGenerator


class CRIformerModel(nn.Module):
    model_type = "CRIformer"

    def __init__(
        self,
        ext1_up: int,
        ext1_down: int,
        ext2_up: int,
        ext2_down: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
    ) -> None:
        """CRIformer parameters.

        Args:
            ext1_up: upstream limit of the resection of the upstream end.
            ext1_down: downstream limit of the templated insertion of the upstream end.
            ext2_up: upstream limit of the templated insertion of the downstream end.
            ext2_down: downstream limit of the resection of the downstream end.
            hidden_size: model embedding dimension.
            num_hidden_layers: number of EncoderLayer.
            num_attention_heads: number of attention heads.
            intermediate_size: feedForward intermediate dimension size.
            hidden_dropout_prob: the dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: the dropout ratio for the attention probabilities.
        """
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )
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
        self.model = RoFormerModel(
            RoFormerConfig(
                vocab_size=4,  # ACGT
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                max_position_embeddings=2
                ** int(np.ceil(np.log2(ext1_up + ext1_down + ext2_up + ext2_down + 2))),
            )
        )
        self.mlp = nn.Linear(
            in_features=hidden_size,
            out_features=(ext1_up + ext1_down + 1) * (ext2_up + ext2_down + 1),
        )

    def forward(
        self, input: dict, label: Optional[dict], my_generator: Optional[MyGenerator]
    ) -> dict:
        # refcode: batch_size X (ext1_up + ext1_down + ext2_up + ext2_down)
        # model(refcode): batch_size X (ext1_up + ext1_down + ext2_up + ext2_down) X hidden_size
        # model(refcode)[:, -1, :]: arbitrary choose the last position to predict the logits
        batch_size = input["refcode"].shape[0]
        logit = self.mlp(
            self.model(
                input_ids=input["refcode"].to(self.device),
                attention_mask=torch.ones(
                    batch_size,
                    self.ext1_up + self.ext1_down + self.ext2_up + self.ext2_down,
                    dtype=torch.int64,
                    device=self.device,
                ),
            ).last_hidden_state[:, -1, :]
        )
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
    def my_model_hpo(cls, trial: optuna.Trial) -> tuple[jsonargparse.Namespace, dict]:
        hparam_dict = {
            "hidden_size": trial.suggest_int("hidden_size", 128, 512, step=128),
            "num_hidden_layers": trial.suggest_int("num_hidden_layers", 2, 4),
            # num_attention_heads must devide hidden_size
            "num_attention_heads": trial.suggest_categorical(
                "num_attention_heads", choices=[2, 4, 8]
            ),
            "intermediate_size": trial.suggest_int(
                "intermediate_size", 512, 2048, step=512
            ),
            "hidden_dropout_prob": trial.suggest_float("hidden_dropout_prob", 0.0, 0.1),
            "attention_probs_dropout_prob": trial.suggest_float(
                "attention_probs_dropout_prob", 0.0, 0.1
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
