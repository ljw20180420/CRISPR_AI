from transformers import PreTrainedModel, RoFormerConfig, RoFormerModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class CRISPRTransformerConfig(RoFormerConfig):
    model_type = "CRISPR transformer"

    def __init__(
        self,
        vocab_size = 4, # ACGT
        hidden_size = 512, # model embedding dimension
        num_hidden_layers = 6, # number of EncoderLayer
        num_attention_heads = 8, # number of attention heads
        intermediate_size = 2048, # FeedForward intermediate dimension size
        hidden_dropout_prob = 0.1, # The dropout probability for all fully connected layers in the embeddings, encoder, and pooler
        attention_probs_dropout_prob = 0.1, # The dropout ratio for the attention probabilities
        ref1len = 127, # length of reference 1
        ref2len = 127, # length of reference 2
        **kwargs
    ):
        super().__init__(
            vocab_size = vocab_size,
            hidden_size = hidden_size,
            num_hidden_layers = num_hidden_layers,
            num_attention_heads = num_attention_heads,
            intermediate_size = intermediate_size,
            hidden_dropout_prob = hidden_dropout_prob,
            attention_probs_dropout_prob = attention_probs_dropout_prob,
            max_position_embeddings = int(2 ** np.ceil(np.log2(ref1len + ref2len))), # The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 1536).
            **kwargs
        )
        self.output_size = (ref1len + 1) * (ref2len + 1)

class CRISPRTransformerModel(PreTrainedModel):
    config_class = CRISPRTransformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = RoFormerModel(config)
        self.mlp = nn.Linear(in_features=config.hidden_size, out_features=config.output_size)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1, generator=self.generator)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, refcode: torch.tensor, observation: torch.tensor=None):
        # refcode (batch_size X sequence_length)
        # model(refcode) (batch_size X sequence_length X hidden_size)
        # model(refcode)[:, -1, :] arbitrary choose the last position to predict the logits
        logit = self.mlp(self.model(refcode)[:, -1, :]).view(refcode.shape[0], self.config.ref2len + 1, self.config.ref1len + 1)
        if observation is not None:
            return {
                "logit": logit,
                "loss": (
                    observation.flatten(start_dim=1) *
                    F.log_softmax(logit.flatten(start_dim=1), dim=1)
                ).sum()
            }
        return {"logit": logit}
