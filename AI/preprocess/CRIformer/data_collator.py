import torch
import numpy as np
from ..utils import MicroHomologyTool
from ...dataset.utils import SeqTokenizer


class DataCollator:
    preprocess = "CRIformer"

    def __init__(
        self,
        ext1_up: int,
        ext1_down: int,
        ext2_up: int,
        ext2_down: int,
    ) -> None:
        self.ext1_up = ext1_up
        self.ext1_down = ext1_down
        self.ext2_up = ext2_up
        self.ext2_down = ext2_down
        self.seq_tokenizer = SeqTokenizer("ACGT")
        self.micro_homology_tool = MicroHomologyTool()

    def __call__(self, examples: list[dict], output_label: bool) -> dict:
        refcodes = []
        if output_label:
            cut1s, cut2s, observation_list = [], [], []
        for example in examples:
            ref = (
                example["ref1"][: example["cut1"]] + example["ref2"][example["cut2"] :]
            )
            cut = example["cut1"]
            self._assert_reference_length_and_cut(ref, cut)
            ref_input = (
                example["ref1"][
                    example["cut1"] - self.ext1_up : example["cut1"] + self.ext1_down
                ]
                + example["ref2"][
                    example["cut2"] - self.ext2_up : example["cut2"] + self.ext2_down
                ]
            )
            refcodes.append(self.seq_tokenizer(ref_input))
            if output_label:
                cut1s.append(example["cut1"])
                cut2s.append(example["cut2"])
                mh_matrix, _, _, mh_rep_num = self.micro_homology_tool.get_mh(
                    example["ref1"],
                    example["ref2"],
                    example["cut1"],
                    example["cut2"],
                    ext1=0,
                    ext2=0,
                )
                observation = self.micro_homology_tool.get_observation(
                    example, mh_matrix, mh_rep_num, lefts=None, rights=None
                )
                observation_list.append(observation)

        if output_label:
            return {
                "input": {
                    "refcode": torch.from_numpy(np.stack(refcodes)),
                },
                "label": {
                    "cut1": np.array(cut1s),
                    "cut2": np.array(cut2s),
                    "observation": torch.from_numpy(np.stack(observation_list)),
                },
            }
        return {
            "input": {
                "refcode": torch.from_numpy(np.stack(refcodes)),
            }
        }

    def _assert_reference_length_and_cut(self, ref: str, cut: int) -> None:
        assert (
            cut >= self.ext1_up
            and cut >= self.ext2_up
            and len(ref) - cut >= self.ext1_down
            and len(ref) - cut >= self.ext2_down
        ), f"reference is too short to support extensions, ext1_up: {self.ext1_up}, ext1_down: {self.ext1_down}, ext2_up: {self.ext2_up}, ext2_down: {self.ext2_down}"
