import numpy as np
import pandas as pd

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import einsum, rearrange


class NonWildTypeCrossEntropy:
    def __init__(
        self,
        ext1_up: int,
        ext1_down: int,
        ext2_up: int,
        ext2_down: int,
    ) -> None:
        """NonWildTypeCrossEntropy arguments.

        Args:
            ext1_up: upstream limit of the resection of the upstream end.
            ext1_down: downstream limit of the templated insertion of the upstream end.
            ext2_up: upstream limit of the templated insertion of the downstream end.
            ext2_down: downstream limit of the resection of the downstream end.
        """
        self.ext1_up = ext1_up
        self.ext1_down = ext1_down
        self.ext2_up = ext2_up
        self.ext2_down = ext2_down

    def __call__(
        self,
        df: pd.DataFrame,
        observation: np.ndarray,
        cut1: np.ndarray,
        cut2: np.ndarray,
    ) -> tuple:
        batch_size = observation.shape[0]
        observation[np.arange(batch_size), cut2, cut1] = 0
        observation = np.stack(
            [
                ob[
                    c2 - self.ext2_up : c2 + self.ext2_down + 1,
                    c1 - self.ext1_up : c1 + self.ext1_down + 1,
                ]
                for ob, c1, c2 in zip(observation, cut1, cut2)
            ]
        )
        # filter wild type
        df = df.query("rpos1 != 0 | rpos2 !=0")
        # filter out of range data
        df = df.query(
            "rpos1 <= @self.ext1_down & rpos1 >= -@self.ext1_up & rpos2 <= @self.ext2_down & rpos2 >= -@self.ext2_up"
        )
        # assume df contains not duplicates
        probas = np.zeros(observation.shape)
        probas[
            df["sample_idx"],
            df["rpos2"] + self.ext2_up,
            df["rpos1"] + self.ext1_up,
        ] = df["proba"]
        probas = probas / rearrange(
            np.maximum(einsum(probas, "b r2 r1 -> b"), np.finfo(np.float64).tiny),
            "b -> b 1 1",
        )
        loss = -einsum(
            np.ma.log(probas).filled(-1000), observation, "b r2 r1, b r2 r1 -> b"
        )
        loss_num = einsum(observation, "b r2 r1 -> b")

        return loss, loss_num
