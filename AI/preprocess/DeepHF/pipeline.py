import torch
from diffusers import DiffusionPipeline
from transformers import PreTrainedModel
from typing import Optional, Callable

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import repeat
from .load_data import DataCollator


class DeepHFPipeline(DiffusionPipeline):
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

    @torch.no_grad()
    def __call__(
        self, examples: list[dict], output_label: bool, metric: Optional[Callable]
    ) -> dict:
        self.data_collator.output_label = output_label
        batch = self.data_collator(examples)
        df = self.core_model.eval_output(batch)

        if output_label:
            assert metric is not None, "not metric given"
            loss, loss_num = metric(df=df, batch=batch)
            return df, loss, loss_num
        return df
