import re
import os
import pathlib
import torch
import numpy as np
import logging
import sys
import json
from typing import Literal, Optional

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import repeat, rearrange

from common_ai.utils import SeqTokenizer


class GetInsertionCount:
    def __init__(self, alphabet: str, insert_uplimit: int) -> None:
        self.insert_uplimit = insert_uplimit
        self.alphabet = alphabet
        self.base_cutoff = len(self.alphabet) ** np.arange(insert_uplimit + 1)
        self.base_cutoff[0] = 0
        self.base_cutoff = np.cumsum(self.base_cutoff)
        # ACGT to 0123
        self.DNA_tokenizer = SeqTokenizer("ACGT")

    def DNA_to_idx(self, DNA: str) -> int:
        vec = self.DNA_tokenizer(DNA) * (len("ACGT") ** np.arange(len(DNA) - 1, -1, -1))
        return self.base_cutoff[len(DNA) - 1] + vec.sum()

    def extract_insert_idx(
        self,
        ref1: str,
        ref2: str,
        cut1: int,
        cut2: int,
        ref1_end: int,
        ref2_start: int,
        random_insert: str,
    ) -> int:
        # base_cutoff[-1] = base + base^2 + base^3 + ... + base^insert_uplimit
        if ref1_end < cut1 or ref2_start > cut2:  # deletion or indel
            return -1
        insert = ref1[cut1:ref1_end] + random_insert + ref2[ref2_start:cut2]
        if not insert:  # wildtype
            return -1
        if insert.find("N") != -1:  # ambitious insertion
            return -1
        if len(insert) > self.insert_uplimit:  # insertion length beyond insert_uplimit
            return self.base_cutoff[-1]
        return self.DNA_to_idx(insert)

    def __call__(self, ref1: str, ref2: str, cut: dict) -> tuple[np.ndarray, int]:
        cut1, cut2, authors = cut["cut1"], cut["cut2"], cut["authors"]
        insert_counts = np.zeros(self.base_cutoff[-1])
        insert_count_long = 0
        for author in authors:
            for file in author["files"]:
                for ref1_end, ref2_start, random_insert, count in zip(
                    file["ref1_end"],
                    file["ref2_start"],
                    file["random_insert"],
                    file["count"],
                ):
                    idx = self.extract_insert_idx(
                        ref1, ref2, cut1, cut2, ref1_end, ref2_start, random_insert
                    )
                    if idx == -1:
                        continue
                    if idx < self.base_cutoff[-1]:
                        insert_counts[idx] += count
                    else:
                        insert_count_long += count

        return insert_counts, insert_count_long


class GetObservation:
    def __init__(self, random_insert_uplimit: int) -> None:
        self.random_insert_uplimit = random_insert_uplimit

    def __call__(self, ref1: str, ref2: str, authors: list[dict]) -> np.ndarray:
        observations = np.zeros(
            [self.random_insert_uplimit + 2, len(ref2) + 1, len(ref1) + 1],
            dtype=float,
        )
        for author in authors:
            for file in author["files"]:
                observations[
                    np.array(
                        [len(random_insert) for random_insert in file["random_insert"]]
                    ).clip(0, self.random_insert_uplimit + 1),
                    np.array(file["ref2_start"]),
                    np.array(file["ref1_end"]),
                ] += np.array(file["count"])

        return observations


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
