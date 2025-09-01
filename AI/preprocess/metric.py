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

    def epoch(self) -> float:
        mean_loss = (
            np.inf
            if self.accum_loss_num == 0.0
            else self.accum_loss / self.accum_loss_num
        )
        self.accum_loss = 0.0
        self.accum_loss_num = 0.0
        return mean_loss.item()


class CrossEntropy(CrossEntropyBase):
    def __init__(
        self, ext1_up: int, ext1_down: int, ext2_up: int, ext2_down: int
    ) -> None:
        super().__init__(ext1_up, ext1_down, ext2_up, ext2_down)

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
        self.accum_loss += -einsum(
            np.ma.log(probas).filled(-1000), observation, "b r2 r1, b r2 r1 -> "
        )
        self.accum_loss_num += einsum(observation, "b r2 r1 -> ")


class NonZeroCrossEntropy(CrossEntropyBase):
    def __init__(
        self, ext1_up: int, ext1_down: int, ext2_up: int, ext2_down: int
    ) -> None:
        super().__init__(ext1_up, ext1_down, ext2_up, ext2_down)

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
        # Fill 0 to mask 0 probabilities from loss.
        self.accum_loss += -einsum(
            np.ma.log(probas).filled(0), observation, "b r2 r1, b r2 r1 -> "
        )
        # Only nonzero probabbilities observation are counted.
        self.accum_loss_num += einsum(probas > 0, observation, "b r2 r1, b r2 r1 -> ")


class NonWildTypeCrossEntropy(CrossEntropyBase):
    def __init__(
        self, ext1_up: int, ext1_down: int, ext2_up: int, ext2_down: int
    ) -> None:
        super().__init__(ext1_up, ext1_down, ext2_up, ext2_down)

    def step(self, df: pd.DataFrame, examples: list, batch: dict) -> None:
        observation = batch["label"]["observation"].cpu().numpy()
        cut1 = np.array([example["cut1"] for example in examples])
        cut2 = np.array([example["cut2"] for example in examples])
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
        self.accum_loss += -einsum(
            np.ma.log(probas).filled(-1000), observation, "b r2 r1, b r2 r1 -> "
        )
        self.accum_loss_num += einsum(observation, "b r2 r1 -> ")


class NonZeroNonWildTypeCrossEntropy(CrossEntropyBase):
    def __init__(
        self, ext1_up: int, ext1_down: int, ext2_up: int, ext2_down: int
    ) -> None:
        super().__init__(ext1_up, ext1_down, ext2_up, ext2_down)

    def step(self, df: pd.DataFrame, examples: list, batch: dict) -> None:
        observation = batch["label"]["observation"].cpu().numpy()
        cut1 = np.array([example["cut1"] for example in examples])
        cut2 = np.array([example["cut2"] for example in examples])
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
        # Fill 0 to mask 0 probabilities from loss.
        self.accum_loss += -einsum(
            np.ma.log(probas).filled(0), observation, "b r2 r1, b r2 r1 -> "
        )
        # Only nonzero probabbilities observation are counted.
        self.accum_loss_num += einsum(probas > 0, observation, "b r2 r1, b r2 r1 -> ")
