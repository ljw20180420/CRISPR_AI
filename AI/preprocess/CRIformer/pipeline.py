import torch
from diffusers import DiffusionPipeline


class CRIformerPipeline(DiffusionPipeline):
    def __init__(self, CRISPR_transformer_model):
        super().__init__()

        self.register_modules(CRISPR_transformer_model=CRISPR_transformer_model)

    @torch.no_grad()
    def __call__(self, batch):
        return {
            "logit": self.CRISPR_transformer_model(
                batch["refcode"].to(self.CRISPR_transformer_model.device)
            )["logit"]
        }
