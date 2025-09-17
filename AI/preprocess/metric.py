import numpy as np
import pandas as pd
from docstring_inheritance import NumpyDocstringInheritanceMeta

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import einsum, rearrange


class CrossEntropyBase(metaclass=NumpyDocstringInheritanceMeta):
    def __init__(
        self,
        ext1_up: int,
        ext1_down: int,
        ext2_up: int,
        ext2_down: int,
    ) -> None:
        """CrossEntropyBase arguments.

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
        self.accum_loss = 0.0
        self.accum_loss_num = 0.0
        self.mask = np.ones(
            (self.ext2_up + self.ext2_down + 1, self.ext1_up + self.ext1_down + 1)
        )
        self.fill = -1000
        self.include_fill = True

    def step(self, df: pd.DataFrame, examples: list, batch: dict) -> None:
        observation = batch["label"]["observation"].cpu().numpy()
        cut1 = np.array([example["cut1"] for example in examples])
        cut2 = np.array([example["cut2"] for example in examples])
        observation = np.stack(
            [
                ob[
                    c2 - self.ext2_up : c2 + self.ext2_down + 1,
                    c1 - self.ext1_up : c1 + self.ext1_down + 1,
                ]
                for ob, c1, c2 in zip(observation, cut1, cut2)
            ]
        )
        observation = observation * self.mask
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
        probas = probas * self.mask
        probas = probas / rearrange(
            np.maximum(einsum(probas, "b r2 r1 -> b"), np.finfo(np.float64).tiny),
            "b -> b 1 1",
        )
        log_probas = np.ma.log(probas).filled(self.fill)
        if self.include_fill:
            mask_probas = np.ones(log_probas.shape)
        else:
            mask_probas = log_probas != self.fill
        self.accum_loss += -einsum(
            mask_probas, log_probas, observation, "b r2 r1, b r2 r1, b r2 r1 -> "
        )
        self.accum_loss_num += einsum(mask_probas, observation, "b r2 r1, b r2 r1 -> ")

    def epoch(self) -> float:
        mean_loss = (
            np.inf
            if self.accum_loss_num == 0.0
            else (self.accum_loss / self.accum_loss_num).item()
        )
        self.accum_loss = 0.0
        self.accum_loss_num = 0.0
        return mean_loss


class CrossEntropy(CrossEntropyBase):
    def __init__(
        self, ext1_up: int, ext1_down: int, ext2_up: int, ext2_down: int
    ) -> None:
        super().__init__(ext1_up, ext1_down, ext2_up, ext2_down)


class NonZeroCrossEntropy(CrossEntropyBase):
    def __init__(
        self, ext1_up: int, ext1_down: int, ext2_up: int, ext2_down: int
    ) -> None:
        super().__init__(ext1_up, ext1_down, ext2_up, ext2_down)
        self.include_fill = False


class NonWildTypeCrossEntropy(CrossEntropyBase):
    def __init__(
        self, ext1_up: int, ext1_down: int, ext2_up: int, ext2_down: int
    ) -> None:
        super().__init__(ext1_up, ext1_down, ext2_up, ext2_down)
        self.mask[self.ext2_up, self.ext1_up] = 0


class NonZeroNonWildTypeCrossEntropy(CrossEntropyBase):
    def __init__(
        self, ext1_up: int, ext1_down: int, ext2_up: int, ext2_down: int
    ) -> None:
        super().__init__(ext1_up, ext1_down, ext2_up, ext2_down)
        self.mask[self.ext2_up, self.ext1_up] = 0
        self.include_fill = False


class GreatestCommonCrossEntropy(CrossEntropyBase):
    def __init__(
        self,
        ext1_up: int,
        ext1_down: int,
        ext2_up: int,
        ext2_down: int,
    ) -> None:
        super().__init__(ext1_up, ext1_down, ext2_up, ext2_down)
        FOREcasT_min_del_size = 0
        inDelphi_min_del_size = 1
        Lindel_min_del_size = 1
        min_del_size = max(
            FOREcasT_min_del_size, inDelphi_min_del_size, Lindel_min_del_size
        )
        FOREcasT_max_del_size = 30
        inDelphi_max_del_size = 59
        Lindel_max_del_size = 59
        max_del_size = min(
            FOREcasT_max_del_size, inDelphi_max_del_size, Lindel_max_del_size
        )
        FOREcasT_ext = 0
        inDelphi_ext = 0
        Lindel_ext = 2
        ext = min(FOREcasT_ext, inDelphi_ext, Lindel_ext)
        FOREcasT_max_ins_size = 2
        inDelphi_max_ins_size = 1
        Lindel_max_ins_size = 2
        max_ins_size = min(
            FOREcasT_max_ins_size, inDelphi_max_ins_size, Lindel_max_ins_size
        )
        lefts = np.concatenate(
            [
                np.arange(-DEL_SIZE - ext, ext + 1)
                for DEL_SIZE in range(min_del_size, max_del_size + 1)
            ]
            + [np.arange(1, max_ins_size + 1)]
        )
        rights = np.concatenate(
            [
                np.arange(-ext, DEL_SIZE + ext + 1)
                for DEL_SIZE in range(min_del_size, max_del_size + 1)
            ]
            + [np.zeros(max_ins_size, dtype=int)]
        )
        in_range = (
            (lefts >= -self.ext1_up)
            & (lefts <= self.ext1_down)
            & (rights >= -self.ext2_up)
            & (rights <= self.ext2_down)
        )
        lefts = lefts[in_range]
        rights = rights[in_range]
        self.mask = np.zeros(
            (self.ext2_up + self.ext2_down + 1, self.ext1_up + self.ext1_down + 1)
        )
        self.mask[rights + self.ext2_up, lefts + self.ext1_up] = 1.0
