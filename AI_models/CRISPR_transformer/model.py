from transformers import PreTrainedModel, RoFormerConfig, RoFormerModel
import torch.nn as nn
import torch
import torch.nn.functional as F

class CRISPRTransformerConfig(RoFormerConfig):
    model_type = "CRISPR transformer"

    def __init__(
        self,
        vocab_size = 4, # ACGT
        hidden_size = 512, # model dimension
        num_hidden_layers = 6, # number of EncoderLayer
        num_attention_heads = 8,
        intermediate_size = 2048, # FeedForward intermediate dimension size
        hidden_dropout_prob = 0.1, # The dropout probability for all fully connected layers in the embeddings, encoder, and pooler
        attention_probs_dropout_prob = 0.1, # The dropout ratio for the attention probabilities
        max_position_embeddings = 256, # The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 1536).
        output_size = 128 * 128, # size of the output logits, which equals (ref1len + 1) * (ref2len + 1)
        **kwargs,
    ):
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
        self.output_size = output_size

class CRISPRTransformerModel(PreTrainedModel):
    config_class = CRISPRTransformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = RoFormerModel(config)
        self.mlp = nn.Linear(in_features=config.hidden_size, out_features=config.output_size)

    def forward(self, refcode: torch.tensor, observation: torch.tensor=None):
        # input_ids (batch_size X sequence_length)
        # model(input_ids) (batch_size X sequence_length X hidden_size)
        # model(input_ids)[:, -1, :] arbitrary choose the last position to predict the logits
        logit = self.mlp(self.model(refcode)[:, -1, :]).sum(dim=1)
        if observation is not None:
            return {
                "logit": logit,
                "loss": (
                    F.normalize(observation.flatten(start_dim=1), p=1, dim=1) *
                    logit
                ).sum()
            }
        return {"logit": logit}
