import numpy as np
import pandas as pd
import torch
from diffusers import DiffusionPipeline
from transformers import PreTrainedModel
import torch.nn.functional as F

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import repeat
from .load_data import DataCollator


class CRIformerPipeline(DiffusionPipeline):
    # __init__ input name should be the same as the register module name
    def __init__(self, core_model: PreTrainedModel) -> None:
        super().__init__()

        self.register_modules(core_model=core_model)
        self.data_collator = DataCollator(
            ext1_up=core_model.config.ext1_up,
            ext1_down=core_model.config.ext1_down,
            ext2_up=core_model.config.ext2_up,
            ext2_down=core_model.config.ext2_down,
            output_label=True,
        )
        self.rpos2s = repeat(
            np.arange(
                -self.core_model.config.ext2_up,
                self.core_model.config.ext2_down + 1,
            ),
            "r2 -> r2 r1",
            r1=self.core_model.config.ext1_up + self.core_model.config.ext1_down + 1,
        ).flatten()

        self.rpos1s = repeat(
            np.arange(
                -self.core_model.config.ext1_up,
                self.core_model.config.ext1_down + 1,
            ),
            "r1 -> r2 r1",
            r2=self.core_model.config.ext2_up + self.core_model.config.ext2_down + 1,
        ).flatten()

    @torch.no_grad()
    def __call__(self, examples: list[dict], output_label: bool) -> dict:
        self.data_collator.output_label = output_label
        batch = self.data_collator(examples)
        if output_label:
            result = self.core_model(
                refcode=batch["refcode"].to(self.core_model.device),
                observation=batch["observation"].to(self.core_model.device),
            )
        else:
            result = self.core_model(
                refcode=batch["refcode"].to(self.core_model.device),
            )

        probas = F.softmax(result["logit"], dim=-1).cpu().numpy()
        batch_size, feature_size = probas.shape
        df = pd.DataFrame(
            {
                "sample_idx": repeat(
                    np.arange(batch_size), "b -> (b f)", f=feature_size
                ),
                "proba": probas.flatten(),
                "rpos1": repeat(self.rpos1s, "r1 -> (b r1)", b=batch_size),
                "rpos2": repeat(self.rpos2s, "r2 -> (b r2)", b=batch_size),
            }
        )

        if output_label:
            return df, result["loss"], batch["observation"].sum()
        return df

    @torch.no_grad()
    def inference(self, examples: list) -> dict:
        self.data_collator.output_label = False
        return self.__call__(
            examples=self.data_collator.inference(examples),
            output_label=False,
        )
