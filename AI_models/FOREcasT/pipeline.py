import torch
from diffusers import DiffusionPipeline
import torch.nn.functional as F


class FOREcasTPipeline(DiffusionPipeline):
    def __init__(self, FOREcasT_model) -> None:
        super().__init__()

        self.register_modules(FOREcasT_model=FOREcasT_model)
        self.lefts, self.rights, self.inss, _, _, _, _, _ = (
            self.FOREcasT_model.pre_calculated_features
        )

    @torch.no_grad()
    def __call__(self, batch: dict) -> dict:
        assert batch["feature"].shape[1] == len(
            self.lefts
        ), "the possible mutation number of the input feature does not fit the pipeline"
        return {
            "proba": F.softmax(
                self.FOREcasT_model(batch["feature"].to(self.FOREcasT_model.device))[
                    "logit"
                ],
                dim=-1,
            ),
            "left": self.lefts,
            "right": self.rights,
            "ins_seq": self.inss,
        }
