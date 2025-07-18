from transformers import PreTrainedModel, RoFormerConfig, RoFormerModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Optional

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import einsum, repeat, rearrange


class CRIformerConfig(RoFormerConfig):
    model_type = "CRIformer"
    label_names = ["label"]

    def __init__(
        self,
        ext1_up: Optional[int] = None,
        ext1_down: Optional[int] = None,
        ext2_up: Optional[int] = None,
        ext2_down: Optional[int] = None,
        hidden_size: Optional[int] = None,
        num_hidden_layers: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_dropout_prob: Optional[float] = None,
        attention_probs_dropout_prob: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
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
        self.ext1_up = ext1_up
        self.ext1_down = ext1_down
        self.ext2_up = ext2_up
        self.ext2_down = ext2_down
        self.seed = seed
        if ext1_up and ext1_down and ext2_up and ext2_down:
            # The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 1536).
            max_position_embeddings = 2 ** int(
                np.ceil(np.log2(ext1_up + ext1_down + ext2_up + ext2_down + 2))
            )
        else:
            max_position_embeddings = 1
        # vocab_size and max_position_embeddings are not in the signature of CRIformerConfig, so they will be in kwargs when loading the model. Set them in kwargs.
        kwargs["vocab_size"] = 4  # ACGT
        kwargs["max_position_embeddings"] = max_position_embeddings
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            **kwargs,
        )


class CRIformerModel(PreTrainedModel):
    config_class = CRIformerConfig

    def __init__(self, config: CRIformerConfig) -> None:
        super().__init__(config)
        self.generator = torch.Generator().manual_seed(config.seed)
        self.model = RoFormerModel(config)
        self.mlp = nn.Linear(
            in_features=config.hidden_size,
            out_features=(config.ext1_up + config.ext1_down + 1)
            * (config.ext2_up + config.ext2_down + 1),
        )
        self._initialize_model_layer_weights()

    # huggingface use the name initialize_weights, use another name here.
    def _initialize_model_layer_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1, generator=self.generator)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=1, generator=self.generator)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=1, generator=self.generator)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input: dict, label: Optional[dict] = None) -> dict:
        # refcode: batch_size X (ext1_up + ext1_down + ext2_up + ext2_down)
        # model(refcode): batch_size X (ext1_up + ext1_down + ext2_up + ext2_down) X hidden_size
        # model(refcode)[:, -1, :]: arbitrary choose the last position to predict the logits
        batch_size = input["refcode"].shape[0]
        logit = self.mlp(
            self.model(
                input_ids=input["refcode"].to(self.device),
                attention_mask=torch.ones(
                    batch_size,
                    self.config.ext1_up
                    + self.config.ext1_down
                    + self.config.ext2_up
                    + self.config.ext2_down,
                    dtype=torch.int64,
                    device=self.model.device,
                ),
            ).last_hidden_state[:, -1, :]
        )
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
            return {
                "logit": logit,
                "loss": self.loss_fun(
                    logit,
                    observation,
                ),
            }
        return {"logit": logit}

    def loss_fun(self, logit: torch.Tensor, observation: torch.Tensor) -> float:
        return -einsum(
            F.log_softmax(logit, dim=1),
            rearrange(observation, "b r2 r1 -> b (r2 r1)"),
            "b f, b f ->",
        )

    def eval_output(self, batch: dict) -> pd.DataFrame:
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
