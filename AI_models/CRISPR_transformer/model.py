from transformers import PreTrainedModel, RoFormerConfig, RoFormerModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class CRISPRTransformerConfig(RoFormerConfig):
    model_type = "CRISPR_transformer"
    label_names = ["observation"]

    def __init__(
        self,
        vocab_size = 4, # ACGT
        hidden_size = 256, # model embedding dimension
        num_hidden_layers = 3, # number of EncoderLayer
        num_attention_heads = 4, # number of attention heads
        intermediate_size = 1024, # FeedForward intermediate dimension size
        hidden_dropout_prob = 0.1, # The dropout probability for all fully connected layers in the embeddings, encoder, and pooler
        attention_probs_dropout_prob = 0.1, # The dropout ratio for the attention probabilities
        max_position_embeddings = 256, # The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 1536).
        ref1len = 127, # length of reference 1
        ref2len = 127, # length of reference 2
        seed = 63036, # random seed for intialization
        **kwargs
    ):
        self.ref1len = ref1len
        self.ref2len = ref2len
        self.seed = seed
        super().__init__(
            vocab_size = vocab_size,
            hidden_size = hidden_size,
            num_hidden_layers = num_hidden_layers,
            num_attention_heads = num_attention_heads,
            intermediate_size = intermediate_size,
            hidden_dropout_prob = hidden_dropout_prob,
            attention_probs_dropout_prob = attention_probs_dropout_prob,
            max_position_embeddings = max_position_embeddings,
            **kwargs
        )

class CRISPRTransformerModel(PreTrainedModel):
    config_class = CRISPRTransformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.generator = torch.Generator().manual_seed(config.seed)
        self.model = RoFormerModel(config)
        self.mlp = nn.Linear(
            in_features=config.hidden_size,
            out_features=(config.ref1len + 1) * (config.ref2len + 1)
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1, generator=self.generator)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, refcode: torch.Tensor, observation: torch.Tensor=None):
        # refcode (batch_size X sequence_length)
        # model(refcode) (batch_size X sequence_length X hidden_size)
        # model(refcode)[:, -1, :] arbitrary choose the last position to predict the logits
        batch_size = refcode.shape[0]
        logit = self.mlp(
            self.model(
                input_ids=refcode,
                attention_mask=torch.ones(
                    batch_size,
                    self.config.ref1len + self.config.ref2len,
                    dtype=torch.int64,
                    device=self.model.device
                )
            ).last_hidden_state[:, -1, :]
        ).view(batch_size, self.config.ref2len + 1, self.config.ref1len + 1)
        if observation is not None:
            return {
                "logit": logit,
                "loss": - (
                    observation.flatten(start_dim=1) *
                    F.log_softmax(logit.flatten(start_dim=1), dim=1)
                ).sum()
            }
        return {"logit": logit}
