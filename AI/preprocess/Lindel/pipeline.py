import torch
from diffusers import DiffusionPipeline
from transformers import PreTrainedModel
import pandas as pd
import torch.nn.functional as F
import numpy as np

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import repeat
from .load_data import DataCollator


class LindelPipeline(DiffusionPipeline):
    def __init__(self, core_model: PreTrainedModel) -> None:
        super().__init__()

        self.register_modules(core_model=core_model)
        self.data_collator = DataCollator(
            dlen=core_model.config.dlen,
            mh_len=core_model.config.mh_len,
            output_label=True,
        )
        self.inss = [
            "A",
            "C",
            "G",
            "T",
            "AA",
            "AC",
            "AG",
            "AT",
            "CA",
            "CC",
            "CG",
            "CT",
            "GA",
            "GC",
            "GG",
            "GT",
            "TA",
            "TC",
            "TG",
            "TT",
        ]

    @torch.no_grad()
    def __call__(self, examples: list[dict], output_label: bool) -> pd.DataFrame:
        self.data_collator.output_label = output_label
        batch = self.data_collator(examples)
        for key, value in batch["input"].items():
            batch["input"][key] = value.to(self.core_model.device)
        if output_label:
            for key, value in batch["count"].items():
                batch["count"][key] = value.to(self.core_model.device)
        if output_label:
            result = self.core_model(batch["input"], batch["count"])
        else:
            result = self.core_model(batch["input"])

        batch_size = len(examples)
        left_shift = np.zeros((batch_size, 20), dtype=int)
        right_shift = np.zeros((batch_size, 20), dtype=int)
        ins_shift = np.full((batch_size, 20), "", dtype="U2")
        for i, example in enumerate(examples):
            ref = (
                example["ref1"][: example["cut1"]] + example["ref2"][example["cut2"] :]
            )
            cut = example["cut1"]
            for j, ins in enumerate(self.inss):
                if ref[cut] == ins[0]:
                    if len(ins) == 2 and ref[cut + 1] == ins[1]:
                        left_shift[i, j] = 2
                    else:
                        left_shift[i, j] = 1
                if ref[cut - 1] == ins[-1]:
                    if len(ins) == 2 and ref[cut - 2] == ins[-2]:
                        right_shift[i, j] = 2
                    else:
                        right_shift[i, j] = 1
                ins_shift[i, j] = ins[left_shift[i, j] : (len(ins) - right_shift[i, j])]

        ins_lens = np.array([len(ins) for ins in self.inss])
        left_shift = np.minimum(ins_lens - right_shift, left_shift)

        indel_probas = F.softmax(result["logit_indel"], dim=1).cpu().numpy()
        ins_probas = F.softmax(result["logit_ins"], dim=1).cpu().numpy()
        del_probas = F.softmax(result["logit_del"], dim=1).cpu().numpy()
        long_ins_proba = ins_probas[:, -1] * indel_probas[:, 1]
        indel_probas[:, 1] = indel_probas[:, 1] - long_ins_proba
        indel_probas[:, 0] = indel_probas[:, 0] / (1 - long_ins_proba)
        indel_probas[:, 1] = indel_probas[:, 1] / (1 - long_ins_proba)
        ins_probas = ins_probas[:, :-1]

        probas = np.concatenate(
            [
                del_probas * indel_probas[:, [0]],
                ins_probas * indel_probas[:, [1]],
            ],
            axis=1,
        )
        rpos1s = np.concatenate(
            [
                repeat(self.data_collator.lefts, "f -> b f", b=batch_size),
                left_shift,
            ],
            axis=-1,
        )
        rpos2s = np.concatenate(
            [
                repeat(self.data_collator.rights, "f -> b f", b=batch_size),
                -right_shift,
            ],
            axis=-1,
        )
        random_ins = np.concatenate(
            [
                np.full((batch_size, len(self.data_collator.lefts)), "", dtype="U2"),
                ins_shift,
            ],
            axis=-1,
        )
        df = pd.DataFrame(
            {
                "sample_idx": repeat(
                    np.arange(batch_size),
                    "b -> (b f)",
                    f=len(self.data_collator.lefts) + 20,
                ),
                "proba": probas.flatten(),
                "rpos1": rpos1s.flatten(),
                "rpos2": rpos2s.flatten(),
                "random_ins": random_ins.flatten(),
            }
        )
        if output_label:
            return df, result["loss"], batch_size
        return df

    @torch.no_grad()
    def inference(self, examples: list) -> dict:
        self.data_collator.output_label = False
        return self.__call__(
            examples=self.data_collator.inference(examples),
            output_label=False,
        )
