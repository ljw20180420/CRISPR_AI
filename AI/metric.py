import jsonargparse
import numpy as np
import optuna
import pandas as pd
from common_ai.metric import MyMetricAbstract
from einops import einsum, rearrange
from scipy.stats import entropy

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum


def get_observation(
    examples, batch: dict, ext1_up: int, ext1_down: int, ext2_up: int, ext2_down: int
) -> np.ndarray:

    observation = batch["label"]["observation"].cpu().numpy()
    cut1 = np.array([example["cut1"] for example in examples])
    cut2 = np.array([example["cut2"] for example in examples])
    observation = np.stack(
        [
            ob[
                c2 - ext2_up : c2 + ext2_down + 1,
                c1 - ext1_up : c1 + ext1_down + 1,
            ]
            for ob, c1, c2 in zip(observation, cut1, cut2)
        ]
    )

    return observation


def get_proba(
    df: pd.DataFrame,
    ext1_up: int,
    ext1_down: int,
    ext2_up: int,
    ext2_down: int,
    batch_size: int,
) -> np.ndarray:
    # filter out of range data
    df = df.query(
        "rpos1 <= @ext1_down & rpos1 >= -@ext1_up & rpos2 <= @ext2_down & rpos2 >= -@ext2_up"
    )
    # assume df contains no duplicates
    probas = np.zeros((batch_size, ext2_up + ext2_down + 1, ext1_up + ext1_down + 1))
    probas[
        df["sample_idx"],
        df["rpos2"] + ext2_up,
        df["rpos1"] + ext1_up,
    ] = df["proba"]

    return probas


def step(
    df: pd.DataFrame,
    examples: list,
    batch: dict,
    ext1_up: int,
    ext1_down: int,
    ext2_up: int,
    ext2_down: int,
    mask: np.ndarray,
    fill: float,
    include_fill: bool,
) -> tuple:
    observation = get_observation(
        examples, batch, ext1_up, ext1_down, ext2_up, ext2_down
    )
    observation = observation * mask

    probas = get_proba(
        df, ext1_up, ext1_down, ext2_up, ext2_down, batch_size=observation.shape[0]
    )
    probas = probas * mask
    probas = probas / rearrange(
        np.maximum(einsum(probas, "b r2 r1 -> b"), np.finfo(np.float64).tiny),
        "b -> b 1 1",
    )
    log_probas = np.ma.log(probas).filled(fill)
    if include_fill:
        mask_probas = np.ones(log_probas.shape)
    else:
        mask_probas = log_probas != fill
    loss = -einsum(
        mask_probas, log_probas, observation, "b r2 r1, b r2 r1, b r2 r1 -> "
    )
    loss_num = einsum(mask_probas, observation, "b r2 r1, b r2 r1 -> ")

    return loss, loss_num


def extract_mask_and_normalize(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = rearrange(mask, "r2 r1 -> (r2 r1)").astype(bool)
    array = rearrange(array, "b r2 r1 -> b (r2 r1)")[:, mask]
    array = array / np.maximum(
        array.sum(axis=-1, keepdims=True), np.finfo(np.float64).tiny
    )

    return array


class CrossEntropy(MyMetricAbstract):
    def __init__(
        self, ext1_up: int, ext1_down: int, ext2_up: int, ext2_down: int
    ) -> None:
        """CrossEntropy arguments.

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
        self.mask = np.ones((ext2_up + ext2_down + 1, ext1_up + ext1_down + 1))

    def step(self, df: pd.DataFrame, examples: list, batch: dict) -> None:
        loss, loss_num = step(
            df,
            examples,
            batch,
            self.ext1_up,
            self.ext1_down,
            self.ext2_up,
            self.ext2_down,
            self.mask,
            fill=-1000.0,
            include_fill=True,
        )
        self.accum_loss += loss
        self.accum_loss_num += loss_num

    def epoch(self) -> float:
        mean_loss = (
            np.inf
            if self.accum_loss_num == 0.0
            else (self.accum_loss / self.accum_loss_num).item()
        )
        self.accum_loss = 0.0
        self.accum_loss_num = 0.0
        return mean_loss

    @classmethod
    def hpo(cls, trial: optuna.Trial, cfg: jsonargparse.Namespace) -> None:
        pass


class NonZeroCrossEntropy(MyMetricAbstract):
    def __init__(
        self, ext1_up: int, ext1_down: int, ext2_up: int, ext2_down: int
    ) -> None:
        """NonZeroCrossEntropy arguments.

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
        self.mask = np.ones((ext2_up + ext2_down + 1, ext1_up + ext1_down + 1))

    def step(self, df: pd.DataFrame, examples: list, batch: dict) -> None:
        loss, loss_num = step(
            df,
            examples,
            batch,
            self.ext1_up,
            self.ext1_down,
            self.ext2_up,
            self.ext2_down,
            self.mask,
            fill=-1000.0,
            include_fill=False,
        )
        self.accum_loss += loss
        self.accum_loss_num += loss_num

    def epoch(self) -> float:
        mean_loss = (
            np.inf
            if self.accum_loss_num == 0.0
            else (self.accum_loss / self.accum_loss_num).item()
        )
        self.accum_loss = 0.0
        self.accum_loss_num = 0.0
        return mean_loss

    @classmethod
    def hpo(cls, trial: optuna.Trial, cfg: jsonargparse.Namespace) -> None:
        pass


class NonWildTypeCrossEntropy(MyMetricAbstract):
    def __init__(
        self, ext1_up: int, ext1_down: int, ext2_up: int, ext2_down: int
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
        self.accum_loss = 0.0
        self.accum_loss_num = 0.0
        self.mask = np.ones((ext2_up + ext2_down + 1, ext1_up + ext1_down + 1))
        self.mask[ext2_up, ext1_up] = 0

    def step(self, df: pd.DataFrame, examples: list, batch: dict) -> None:
        loss, loss_num = step(
            df,
            examples,
            batch,
            self.ext1_up,
            self.ext1_down,
            self.ext2_up,
            self.ext2_down,
            self.mask,
            fill=-1000.0,
            include_fill=True,
        )
        self.accum_loss += loss
        self.accum_loss_num += loss_num

    def epoch(self) -> float:
        mean_loss = (
            np.inf
            if self.accum_loss_num == 0.0
            else (self.accum_loss / self.accum_loss_num).item()
        )
        self.accum_loss = 0.0
        self.accum_loss_num = 0.0
        return mean_loss

    @classmethod
    def hpo(cls, trial: optuna.Trial, cfg: jsonargparse.Namespace) -> None:
        pass


class NonZeroNonWildTypeCrossEntropy(MyMetricAbstract):
    def __init__(
        self, ext1_up: int, ext1_down: int, ext2_up: int, ext2_down: int
    ) -> None:
        """NonZeroNonWildTypeCrossEntropy arguments.

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
        self.mask = np.ones((ext2_up + ext2_down + 1, ext1_up + ext1_down + 1))
        self.mask[ext2_up, ext1_up] = 0

    def step(self, df: pd.DataFrame, examples: list, batch: dict) -> None:
        loss, loss_num = step(
            df,
            examples,
            batch,
            self.ext1_up,
            self.ext1_down,
            self.ext2_up,
            self.ext2_down,
            self.mask,
            fill=-1000.0,
            include_fill=False,
        )
        self.accum_loss += loss
        self.accum_loss_num += loss_num

    def epoch(self) -> float:
        mean_loss = (
            np.inf
            if self.accum_loss_num == 0.0
            else (self.accum_loss / self.accum_loss_num).item()
        )
        self.accum_loss = 0.0
        self.accum_loss_num = 0.0
        return mean_loss

    @classmethod
    def hpo(cls, trial: optuna.Trial, cfg: jsonargparse.Namespace) -> None:
        pass


class GreatestCommonCrossEntropy(MyMetricAbstract):
    def __init__(
        self,
        ext1_up: int,
        ext1_down: int,
        ext2_up: int,
        ext2_down: int,
        max_del_size: int,
    ) -> None:
        """GreatestCommonCrossEntropy arguments.

        Args:
            ext1_up: upstream limit of the resection of the upstream end.
            ext1_down: downstream limit of the templated insertion of the upstream end.
            ext2_up: upstream limit of the templated insertion of the downstream end.
            ext2_down: downstream limit of the resection of the downstream end.
            max_del_size: maximal deletion size.
        """
        self.ext1_up = ext1_up
        self.ext1_down = ext1_down
        self.ext2_up = ext2_up
        self.ext2_down = ext2_down
        self.accum_loss = 0.0
        self.accum_loss_num = 0.0

        FOREcasT_min_del_size = 0
        inDelphi_min_del_size = 1
        Lindel_min_del_size = 1
        min_del_size = max(
            FOREcasT_min_del_size, inDelphi_min_del_size, Lindel_min_del_size
        )

        # max_del_size = min(FOREcasT_max_del_size, inDelphi_max_del_size, Lindel_max_del_size)

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
            (lefts >= -ext1_up)
            & (lefts <= ext1_down)
            & (rights >= -ext2_up)
            & (rights <= ext2_down)
        )
        lefts = lefts[in_range]
        rights = rights[in_range]
        self.mask = np.zeros((ext2_up + ext2_down + 1, ext1_up + ext1_down + 1))
        self.mask[rights + ext2_up, lefts + ext1_up] = 1.0

    def step(self, df: pd.DataFrame, examples: list, batch: dict) -> None:
        loss, loss_num = step(
            df,
            examples,
            batch,
            self.ext1_up,
            self.ext1_down,
            self.ext2_up,
            self.ext2_down,
            self.mask,
            fill=-1000.0,
            include_fill=True,
        )
        self.accum_loss += loss
        self.accum_loss_num += loss_num

    def epoch(self) -> float:
        mean_loss = (
            np.inf
            if self.accum_loss_num == 0.0
            else (self.accum_loss / self.accum_loss_num).item()
        )
        self.accum_loss = 0.0
        self.accum_loss_num = 0.0
        return mean_loss

    @classmethod
    def hpo(cls, trial: optuna.Trial, cfg: jsonargparse.Namespace) -> None:
        pass


class Likelihood(GreatestCommonCrossEntropy):
    def epoch(self) -> float:
        return -super().epoch()


class Pearson(GreatestCommonCrossEntropy):
    def step(self, df: pd.DataFrame, examples: list, batch: dict) -> None:
        observation = extract_mask_and_normalize(
            array=get_observation(
                examples,
                batch,
                self.ext1_up,
                self.ext1_down,
                self.ext2_up,
                self.ext2_down,
            ),
            mask=self.mask,
        )

        probas = extract_mask_and_normalize(
            array=get_proba(
                df,
                self.ext1_up,
                self.ext1_down,
                self.ext2_up,
                self.ext2_down,
                batch_size=observation.shape[0],
            ),
            mask=self.mask,
        )

        for ob, pb in zip(observation, probas):
            if ob.std() == 0 or pb.std() == 0:
                continue
            self.accum_loss += np.corrcoef(ob, pb)[0, 1]
            self.accum_loss_num += 1


class MSE(GreatestCommonCrossEntropy):
    def step(self, df: pd.DataFrame, examples: list, batch: dict) -> None:
        observation = extract_mask_and_normalize(
            array=get_observation(
                examples,
                batch,
                self.ext1_up,
                self.ext1_down,
                self.ext2_up,
                self.ext2_down,
            ),
            mask=self.mask,
        )

        probas = extract_mask_and_normalize(
            array=get_proba(
                df,
                self.ext1_up,
                self.ext1_down,
                self.ext2_up,
                self.ext2_down,
                batch_size=observation.shape[0],
            ),
            mask=self.mask,
        )

        for ob, pb in zip(observation, probas):
            if ob.sum() == 0:
                continue
            self.accum_loss += np.mean((ob - pb) ** 2)
            self.accum_loss_num += 1


class SymKL(GreatestCommonCrossEntropy):
    def step(self, df: pd.DataFrame, examples: list, batch: dict) -> None:
        observation = extract_mask_and_normalize(
            array=get_observation(
                examples,
                batch,
                self.ext1_up,
                self.ext1_down,
                self.ext2_up,
                self.ext2_down,
            ),
            mask=self.mask,
        )

        probas = extract_mask_and_normalize(
            array=get_proba(
                df,
                self.ext1_up,
                self.ext1_down,
                self.ext2_up,
                self.ext2_down,
                batch_size=observation.shape[0],
            ),
            mask=self.mask,
        )

        for ob, pb in zip(observation, probas):
            if ob.sum() == 0:
                continue
            self.accum_loss += (entropy(pk=ob, qk=pb) + entropy(pk=pb, qk=ob)) / 2.0
            self.accum_loss_num += 1
