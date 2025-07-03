import pandas as pd
import numpy as np
import torch
from diffusers import DiffusionPipeline
import torch.nn.functional as F

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import repeat


class FOREcasTPipeline(DiffusionPipeline):
    def __init__(self, model) -> None:
        super().__init__()

        self.register_modules(model=model)
        self.lefts, self.rights, self.inss, _, _, _, _, _ = (
            self.model.pre_calculated_features
        )

    @torch.no_grad()
    def __call__(self, batch: dict) -> dict:
        assert batch["feature"].shape[1] == len(
            self.lefts
        ), "the possible mutation number of the input feature does not fit the pipeline"
        if "count" in batch:
            result = self.model(
                batch["ref"],
                batch["cut"],
                batch["feature"].to(self.model.device),
                batch["count"].to(self.model.device),
            )
        else:
            result = self.model(
                batch["ref"],
                batch["cut"],
                batch["feature"].to(self.model.device),
            )
        probas = F.softmax(result["logit"], dim=-1).cpu().numpy()

        batch_size, feature_size = probas.shape
        left_shift, right_shift, ins_shift = (
            np.zeros((batch_size, 20), dtype=int),
            np.zeros((batch_size, 20), dtype=int),
            np.full((batch_size, 20), "", dtype="U2"),
        )
        for i, (ref, cut) in enumerate(zip(batch["ref"], batch["cut"])):
            for j, ins in enumerate(self.inss[-20:]):
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

        ins_lens = np.array([len(ins) for ins in self.inss[-20:]])
        left_shift = np.minimum(ins_lens - right_shift, left_shift)
        rpos1s = np.concatenate(
            [
                repeat(self.lefts[:-20], "f -> b f", b=batch_size),
                left_shift,
            ],
            axis=-1,
        )
        rpos2s = np.concatenate(
            [
                repeat(self.rights[:-20], "f -> b f", b=batch_size),
                -right_shift,
            ],
            axis=-1,
        )
        random_ins = np.concatenate(
            [
                np.full((batch_size, feature_size - 20), "", dtype="U2"),
                ins_shift,
            ],
            axis=-1,
        )
        df = pd.DataFrame(
            {
                "sample_idx": repeat(
                    np.arange(batch_size), "b -> (b f)", f=feature_size
                ),
                "proba": probas.flatten(),
                "rpos1": rpos1s.flatten(),
                "rpos2": rpos2s.flatten(),
                "random_ins": random_ins.flatten(),
            }
        )
        if "count" in batch:
            return (
                df,
                result["loss"],
            )
        return df
