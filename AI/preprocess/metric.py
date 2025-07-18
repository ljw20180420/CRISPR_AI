import numpy as np
import pandas as pd

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import einsum, rearrange


class NonWildTypeCrossEntropy:
    def __init__(
        self,
        metric_ext1_up: int,
        metric_ext1_down: int,
        metric_ext2_up: int,
        metric_ext2_down: int,
    ) -> None:
        self.metric_ext1_up = metric_ext1_up
        self.metric_ext1_down = metric_ext1_down
        self.metric_ext2_up = metric_ext2_up
        self.metric_ext2_down = metric_ext2_down

    def __call__(
        self,
        df: pd.DataFrame,
        batch: dict,
    ) -> tuple:
        observation = batch["label"]["observation"].cpu().numpy()
        cut1 = batch["label"]["cut1"]
        cut2 = batch["label"]["cut2"]
        observation[:, cut2, cut1] = 0
        observation = np.stack(
            [
                ob[
                    c2 - self.metric_ext2_up : c2 + self.metric_ext2_down + 1,
                    c1 - self.metric_ext1_up : c1 + self.metric_ext1_down + 1,
                ]
                for ob, c1, c2 in zip(observation, cut1, cut2)
            ]
        )
        # filter wild type
        df = df.query("rpos1 != 0 | rpos2 !=0")
        # filter out of range data
        df = df.query(
            "rpos1 <= @self.metric_ext1_down & rpos1 >= -@self.metric_ext1_up & rpos2 <= @self.metric_ext2_down & rpos2 >= -@self.metric_ext2_up"
        )
        # assume df contains not duplicates
        probas = np.zeros(observation.shape)
        probas[
            df["sample_idx"],
            df["rpos2"] + self.metric_ext2_up,
            df["rpos1"] + self.metric_ext1_up,
        ] = df["proba"]
        probas = probas / rearrange(
            np.maximum(einsum(probas, "b r2 r1 -> b"), np.finfo(np.float64).tiny),
            "b -> b 1 1",
        )
        loss = -einsum(
            np.log(probas).clip(-1000, 0), observation, "b r2 r1, b r2 r1 ->"
        )
        loss_num = einsum(observation, "b r2 r1 ->")

        return loss, loss_num
