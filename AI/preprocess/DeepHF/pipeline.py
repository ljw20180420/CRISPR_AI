import torch
from diffusers import DiffusionPipeline
import torch.nn.functional as F

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import repeat


class DeepHFPipeline(DiffusionPipeline):
    def __init__(self, DeepHF_model) -> None:
        super().__init__()

        self.register_modules(DeepHF_model=DeepHF_model)
        self.rpos2s = repeat(
            torch.arange(
                -self.DeepHF_model.config.ext2_up,
                self.DeepHF_model.config.ext2_down + 1,
            ),
            "r2 -> r2 r1",
            r1=self.DeepHF_model.config.ext1_up
            + self.DeepHF_model.config.ext1_down
            + 1,
        ).flatten()

        self.rpos1s = repeat(
            torch.arange(
                -self.DeepHF_model.config.ext1_up,
                self.DeepHF_model.config.ext1_down + 1,
            ),
            "r1 -> r2 r1",
            r2=self.DeepHF_model.config.ext2_up
            + self.DeepHF_model.config.ext2_down
            + 1,
        ).flatten()

    @torch.no_grad()
    def __call__(self, batch: dict) -> dict:
        if "observation" in batch:
            result = self.DeepHF_model(
                X=batch["X"].to(self.DeepHF_model.device),
                biologial_input=batch["biological_input"].to(self.DeepHF_model.device),
                observation=batch["observation"].to(self.DeepHF_model.device),
            )
        else:
            result = self.DeepHF_model(
                X=batch["X"].to(self.DeepHF_model.device),
                biologial_input=batch["biological_input"].to(self.DeepHF_model.device),
            )
        probas = F.softmax(
            result["logit"],
            dim=-1,
        )

        if "observation" in batch:
            return {
                "proba": probas,
                "rpos1": self.rpos1s,
                "rpos2": self.rpos2s,
                "loss": result["loss"],
                "sample_num": batch["observation"].sum(),
            }
        return {
            "proba": probas,
            "rpos1": self.rpos1s,
            "rpos2": self.rpos2s,
        }
