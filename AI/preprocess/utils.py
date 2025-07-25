import re
import os
import pathlib
import torch
import numpy as np
import logging
import sys
from typing import Literal, Optional

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import repeat, rearrange


def target_to_epoch(checkpoints_path: os.PathLike, target: str) -> int:
    """
    Infer the epoch from either the last checkpoint or the loweset metric (including loss).
    """
    checkpoints_path = pathlib.Path(os.fspath(checkpoints_path))
    if not os.path.exists(checkpoints_path):
        return -1
    check_epochs = [
        check_epoch
        for check_epoch in os.listdir(checkpoints_path)
        if re.search(r"^checkpoint-(\d+)$", check_epoch)
    ]
    if len(check_epochs) == 0:
        return -1
    if target == "resume":
        return max(
            [
                int(re.search(r"^checkpoint-(\d+)$", check_epoch).group(1))
                for check_epoch in check_epochs
            ]
        )

    metric_value_min = np.inf
    for check_epoch in os.listdir(checkpoints_path):
        with open(checkpoints_path / check_epoch / "meta_data.json", "r") as fd:
            meta_data = json.load(fd)
        if target == "loss":
            metric_value = (
                meta_data["performance"]["eval"]["loss"]
                / meta_data["performance"]["eval"]["loss_num"]
            )
        else:
            metric_value = (
                meta_data["performance"]["eval"][target]["loss"]
                / meta_data["performance"]["eval"][target]["loss_num"]
            )
        if metric_value < metric_value_min:
            metric_value_min = metric_value
            epoch = int(check_epoch.split("-")[1])

    return epoch


class MyGenerator:
    def __init__(self, seed: int) -> None:
        """Generator arguments.

        Args:
            seed: Random seed.
        """
        self.seed = seed
        self.np_rng = np.random.default_rng(self.seed)
        self.torch_c_rng = torch.Generator(device="cpu").manual_seed(self.seed)
        self.torch_g_rng = torch.Generator(device="cuda").manual_seed(self.seed)

    def get_torch_generator_by_device(
        self, device: str | torch.device
    ) -> torch.Generator:
        if device == "cpu" or device == torch.device("cpu"):
            return self.torch_c_rng
        return self.torch_g_rng

    def state_dict(self) -> dict:
        return {
            "np_rng": self.np_rng.bit_generator.state,
            "torch_c_rng": self.torch_c_rng.get_state(),
            "torch_g_rng": self.torch_g_rng.get_state(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.np_rng.bit_generator.state = state_dict["np_rng"]
        self.torch_c_rng.set_state(state_dict["torch_c_rng"])
        self.torch_g_rng.set_state(state_dict["torch_g_rng"])


def get_logger(
    log_level: Literal[
        "CRITICAL", "FATAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"
    ],
) -> None:
    """Logger arguments.

    Args:
        log_level: The level of logging.
    """
    logger = logging.getLogger("logger")
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(log_level)
    logger.addHandler(handler)
    return logger


class MicroHomologyTool:
    def __init__(self) -> None:
        pass

    def reinitialize(self, ref1: str, ref2: str) -> None:
        if (
            hasattr(self, "ref1len")
            and self.ref1len == len(ref1)
            and hasattr(self, "ref2len")
            and self.ref2len == len(ref2)
        ):
            return
        self.ref1len = len(ref1)
        self.ref2len = len(ref2)
        # diag_indices example for ref2len = 3 and ref1len = 2:
        # 6 9 11   row_indices 0 0 0   col_indices 0 1 2
        # 3 7 10               1 1 1               0 1 2
        # 1 4 8                2 2 2               0 1 2
        # 0 2 5                3 3 3               0 1 2
        # diag_indices = np.ravel_multi_index(
        #     multi_index=(
        #         tensor([3, 2, 3, 1, 2, 3, 0, 1, 2, 0, 1, 0]),
        #         tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 1, 2, 2])
        #     ),
        #     dims=(4, 3),
        # )
        row_indices = repeat(
            np.arange(self.ref2len + 1), "r2 -> r2 r1", r1=self.ref1len + 1
        )
        col_indices = repeat(
            np.arange(self.ref1len + 1), "r1 -> r2 r1", r2=self.ref2len + 1
        )
        self.diag_indices = np.ravel_multi_index(
            multi_index=(
                # row index
                np.concatenate(
                    [
                        row_indices.diagonal(offset)
                        for offset in range(-self.ref2len, self.ref1len + 1)
                    ]
                ),
                # col index
                np.concatenate(
                    [
                        col_indices.diagonal(offset)
                        for offset in range(-self.ref2len, self.ref1len + 1)
                    ]
                ),
            ),
            dims=(self.ref2len + 1, self.ref1len + 1),
        )

    def get_mh(
        self, ref1: str, ref2: str, cut1: int, cut2: int, ext1: int, ext2: int
    ) -> tuple[np.ndarray]:
        assert cut1 + ext1 <= len(ref1) and ext2 <= cut2, "extend too much"
        self.reinitialize(ref1, ref2)
        mh_matrix = np.pad(
            (
                rearrange(
                    np.frombuffer(ref1[: cut1 + ext1].encode(), dtype=np.int8),
                    "r1 -> 1 r1",
                )
                == rearrange(
                    np.frombuffer(ref2[cut2 - ext2 :].encode(), dtype=np.int8),
                    "r2 -> r2 1",
                )
            ).astype(int),
            pad_width=((cut2 - ext2, 1), (0, len(ref1) - cut1 - ext1 + 1)),
        )
        rep_num = np.diff(
            np.concatenate(
                (
                    np.array([-1], dtype=int),
                    np.where(np.diff(mh_matrix.flatten()[self.diag_indices]))[0],
                    np.array([(len(ref1) + 1) * (len(ref2) + 1) - 1], dtype=int),
                )
            )
        )
        rep_val = rep_num.copy()
        rep_val[0::2] = 0
        rep_num[1::2] = rep_num[1::2] + 1
        rep_num[2::2] = rep_num[2::2] - 1
        mh_matrix = mh_matrix.flatten()
        mh_matrix[self.diag_indices] = np.repeat(rep_val, rep_num)
        cum_rep_num = rep_num.cumsum()
        mh_idx_align_ref1 = self.diag_indices[cum_rep_num[1::2] - 1]
        mh_idx_align_ref2 = self.diag_indices[cum_rep_num[0:-1:2]]
        mh_rep_num = rep_num[1::2]
        return mh_matrix, mh_idx_align_ref1, mh_idx_align_ref2, mh_rep_num

    def correct_observation(
        self, observations: np.ndarray, mh_matrix: np.ndarray, mh_rep_num: np.ndarray
    ) -> np.ndarray:
        mh_mask = (mh_matrix > 0)[self.diag_indices]
        for i, observation in enumerate(observations):
            observation = observation.flatten()
            counts = np.zeros(len(mh_rep_num), dtype=int)
            np.add.at(
                counts,
                np.repeat(np.arange(len(mh_rep_num)), mh_rep_num),
                observation[self.diag_indices][mh_mask],
            )
            observation[self.diag_indices[mh_mask]] = np.repeat(counts, mh_rep_num)
            observations[i] = observation.reshape(self.ref2len + 1, self.ref1len + 1)

        return observations

    def get_observation(
        self,
        example: dict,
        mh_matrix: np.ndarray,
        mh_rep_num: np.ndarray,
        lefts: Optional[np.ndarray],
        rights: Optional[np.ndarray],
    ) -> tuple[np.ndarray]:
        mh_idx = mh_matrix.nonzero()
        mh_val = mh_matrix[mh_idx]
        # construct observations
        observations = np.zeros(
            (example["random_insert_uplimit"] + 2)
            * (len(example["ref2"]) + 1)
            * (len(example["ref1"]) + 1),
            dtype=np.float32,
        )
        observations[example["ob_idx"]] = np.array(example["ob_val"], dtype=np.float32)
        observations = observations.reshape(
            example["random_insert_uplimit"] + 2,
            len(example["ref2"]) + 1,
            len(example["ref1"]) + 1,
        )
        # correct observations
        observations = self.correct_observation(observations, mh_matrix, mh_rep_num)
        # cumulate observations for all random insertion size
        observation = observations.sum(axis=0)
        # output triangle
        if lefts is not None and rights is not None:
            all_counts = observation[
                rights + example["cut2"],
                lefts + example["cut1"],
            ]
        # distribute count to all positions in single micro-homology diagonal
        observation = observation.flatten()
        observation[mh_idx] = observation[mh_idx] / (mh_val + 1)
        observation = observation.reshape(
            len(example["ref2"]) + 1, len(example["ref1"]) + 1
        )
        if lefts is not None and rights is not None:
            return observation, all_counts
        return observation
