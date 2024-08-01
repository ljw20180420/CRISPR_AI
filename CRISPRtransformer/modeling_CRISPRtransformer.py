from transformers import PreTrainedModel, RoFormerModel
from .configuration_CRISPRtransformer import CRISPRtransformerConfig
import torch.nn as nn
import torch
import torch.nn.functional as F

class CRISPRtransformerModel(PreTrainedModel):
    config_class = CRISPRtransformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = RoFormerModel(config)
        self.mlp = nn.Linear(in_features=config.hidden_size, out_features=config.output_size)

    def forward(self, input_ids: torch.tensor, observations: torch.tensor=None):
        logits = self.mlp(self.model(input_ids)).sum(dim=-2)
        if observations is not None:
            distrbutions = F.normalize(observations, p=1, dim=1)
            return {
                "logits": logits,
                "loss": (distrbutions * logits).sum()
            }
        return {"logits": logits}
