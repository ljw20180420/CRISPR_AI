from diffusers import DiffusionPipeline
import torch
from transformers import PreTrainedModel
from .load_data import DataCollator


class inDelphiPipeline(DiffusionPipeline):
    def __init__(
        self,
        core_model: PreTrainedModel,
        insert_model,
    ) -> None:
        super().__init__()

        self.register_modules(core_model=core_model)
        self.register_modules(insert_model=insert_model)
        self.data_collator = DataCollator(
            DELLEN_LIMIT=core_model.config.DELLEN_LIMIT,
            output_label=True,
        )

    @torch.no_grad()
    def __call__(self, examples: list[dict], output_label: bool):
        self.data_collator.output_label = output_label
        batch = self.data_collator(examples)
        if output_label:
            result = self.core_model(
                batch["mh_input"].to(self.core_model.device),
                batch["mh_del_len"].to(self.core_model.device),
                batch["genotype_count"].to(self.core_model.device),
                batch["total_del_len_count"].to(self.core_model.device),
            )
        else:
            result = self.core_model(
                batch["mh_input"].to(self.core_model.device),
                batch["mh_del_len"].to(self.core_model.device),
            )
        insert_batch = self.data_collator.insert_call(examples)
        insert_probabilities, insert_1bps = self.insert_model(
            result["total_del_len_weight"],
            insert_batch["onebp_feature"],
            insert_batch["m654"],
            use_m654=False,
        )
        return {
            "mh_weight": [
                mh_weights[
                    i, batch["mh_del_len"][i] < self.inDelphi_model.config.DELLEN_LIMIT
                ]
                for i in range(len(batch["mh_del_len"]))
            ],
            "mhless_weight": mhless_weights,
            "total_del_len_weight": total_del_len_weights,
            "pre_insert_probability": pre_insert_probabilities,
            "pre_insert_1bp": pre_insert_1bps,
        }
