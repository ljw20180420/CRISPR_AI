from diffusers import DiffusionPipeline
import torch
from transformers import PreTrainedModel
import pandas as pd
from typing import Optional, Callable
from .load_data import DataCollator


class inDelphiPipeline(DiffusionPipeline):
    def __init__(
        self,
        core_model: PreTrainedModel,
        auxilary_model: PreTrainedModel,
    ) -> None:
        super().__init__()

        self.register_modules(core_model=core_model)
        self.register_modules(auxilary_model=auxilary_model)
        self.data_collator = DataCollator(
            DELLEN_LIMIT=core_model.config.DELLEN_LIMIT,
            output_label=True,
        )

    @torch.no_grad()
    def __call__(
        self, examples: list[dict], output_label: bool, metric: Optional[Callable]
    ) -> pd.DataFrame:
        self.data_collator.output_label = output_label
        batch = self.data_collator(examples)
        df = self.core_model.eval_output(
            examples,
            batch,
            self.auxilary_model,
        )

        if output_label:
            assert metric is not None, "not metric given"
            loss, loss_num = metric(df=df, batch=batch)
            return df, loss, loss_num
        return df
