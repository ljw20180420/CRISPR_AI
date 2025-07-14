from transformers import PreTrainedModel, RoFormerConfig, RoFormerModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class CRIformerConfig(RoFormerConfig):
    model_type = "CRISPR_transformer"
    label_names = ["observation"]

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
        super().__init__(
            vocab_size=4,  # ACGT
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=2
            ** int(
                np.ceil(np.log2(ext1_up + ext1_down + ext2_up + ext2_down + 2))
            ),  # The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 1536).
            **kwargs
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
        self.initialize_weights()

    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1, generator=self.generator)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, refcode: torch.Tensor, observation: torch.Tensor = None) -> dict:
        # refcode: batch_size X (ext1_up + ext1_down + ext2_up + ext2_down)
        # model(refcode): batch_size X (ext1_up + ext1_down + ext2_up + ext2_down) X hidden_size
        # model(refcode)[:, -1, :]: arbitrary choose the last position to predict the logits
        batch_size = refcode.shape[0]
        logit = self.mlp(
            self.model(
                input_ids=refcode,
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
        ).view(
            batch_size,
            self.config.ext2_up + self.config.ext2_down + 1,
            self.config.ext1_up + self.config.ext1_down + 1,
        )
        if observation is not None:
            return {
                "logit": logit,
                "loss": -(
                    observation.flatten(start_dim=1)
                    * F.log_softmax(logit.flatten(start_dim=1), dim=1)
                ).sum(),
            }
        return {"logit": logit}
