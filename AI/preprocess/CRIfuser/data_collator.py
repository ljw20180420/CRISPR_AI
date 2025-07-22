import torch
import numpy as np

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import repeat, rearrange
from ..utils import MicroHomologyTool
from ...dataset.utils import SeqTokenizer


class DataCollator:
    preprocess = "CRIfuser"

    def __init__(
        self,
        ext1_up: int,
        ext1_down: int,
        ext2_up: int,
        ext2_down: int,
        max_micro_homology: int,
    ):
        self.ext1_up = ext1_up
        self.ext1_down = ext1_down
        self.ext2_up = ext2_up
        self.ext2_down = ext2_down
        self.max_micro_homology = max_micro_homology
        self.seq_tokenizer = SeqTokenizer("ACGT")
        self.micro_homology_tool = MicroHomologyTool()

    @torch.no_grad()
    def __call__(self, examples: list[dict], output_label: bool) -> dict:
        conditions = []
        if output_label:
            observation_list = []
        for example in examples:
            ref = (
                example["ref1"][: example["cut1"]] + example["ref2"][example["cut2"] :]
            )
            cut = example["cut1"]
            self._assert_reference_length_and_cut(ref, cut)
            mh_matrix, _, _, mh_rep_num = self.micro_homology_tool.get_mh(
                example["ref1"],
                example["ref2"],
                example["cut1"],
                example["cut2"],
                ext1=0,
                ext2=0,
            )
            if output_label:
                observation = self.micro_homology_tool.get_observation(
                    example, mh_matrix, mh_rep_num
                )
                observation_list.append(observation)

            mh_matrix = (
                rearrange(
                    mh_matrix,
                    "(r2 r1) -> r2 r1",
                    r1=len(example["ref1"]) + 1,
                    r2=len(example["ref2"]) + 1,
                )[
                    example["cut2"]
                    - self.ext2_up : example["cut2"]
                    + self.ext2_down
                    + 1,
                    example["cut1"]
                    - self.ext1_up : example["cut1"]
                    + self.ext1_down
                    + 1,
                ].clip(
                    0, self.max_micro_homology
                )
                / self.max_micro_homology
            )
            one_hot_cut = torch.zeros(
                self.ext2_up + self.ext2_down + 1, self.ext1_up + self.ext1_down + 1
            )
            one_hot_cut[self.ext2_up, self.ext1_up] = 1
            one_hot_ref1 = repeat(
                np.eye(4)[
                    self.seq_tokenizer(
                        example["ref1"][
                            example["cut1"]
                            - self.ext1_up : example["cut1"]
                            + self.ext1_down
                            + 1
                        ]
                    )
                ],
                "r1 h -> h r2 r1",
                r2=self.ext2_up + self.ext2_down + 1,
            )
            one_hot_ref2 = repeat(
                np.eye(4)[
                    self.seq_tokenizer(
                        example["ref2"][
                            example["cut2"]
                            - self.ext2_up : example["cut2"]
                            + self.ext2_down
                            + 1
                        ]
                    )
                ],
                "r2 h -> h r2 r1",
                r1=self.ext1_up + self.ext1_down + 1,
            )
            conditions.append(
                np.concatenate(
                    (np.stack((mh_matrix, one_hot_cut)), one_hot_ref1, one_hot_ref2),
                    axis=0,
                    dtype=np.float32,
                )
            )
        if output_label:
            return {
                "input": {
                    "condition": torch.from_numpy(np.stack(conditions)),
                },
                "label": {
                    "observation": torch.from_numpy(np.stack(observation_list)),
                },
            }
        return {
            "input": {
                "condition": torch.from_numpy(np.stack(conditions)),
            },
        }

    def _assert_reference_length_and_cut(self, ref: str, cut: int) -> None:
        # len(ref) - cut > self.ext?_down instead of >= because one_hot_ref? needs an additional bp.
        assert (
            cut >= self.ext1_up
            and cut >= self.ext2_up
            and len(ref) - cut > self.ext1_down
            and len(ref) - cut > self.ext2_down
        ), f"reference is too short to support extensions, ext1_up: {self.ext1_up}, ext1_down: {self.ext1_down}, ext2_up: {self.ext2_up}, ext2_down: {self.ext2_down}"
