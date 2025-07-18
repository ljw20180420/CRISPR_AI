import torch
import numpy as np
from ..utils import GetMH, SeqTokenizer


class DataCollator:
    def __init__(
        self,
        ext1_up: int,
        ext1_down: int,
        ext2_up: int,
        ext2_down: int,
        output_label: bool,
    ) -> None:
        self.ext1_up = ext1_up
        self.ext1_down = ext1_down
        self.ext2_up = ext2_up
        self.ext2_down = ext2_down
        self.seq_tokenizer = SeqTokenizer("ACGT")
        self.get_mh = GetMH()
        self.output_label = output_label

    def __call__(self, examples: list[dict]) -> dict:
        refcodes = []
        if self.output_label:
            observation_list, cut1s, cut2s = [], [], []
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
            if self.output_label:
                mh_matrix, _, _, mh_rep_num = self.get_mh(
                    example["ref1"],
                    example["ref2"],
                    example["cut1"],
                    example["cut2"],
                    ext1=0,
                    ext2=0,
                )
                observation = self.get_mh.get_observation(
                    example, mh_matrix, mh_rep_num
                )
                observation_list.append(observation)
                cut1s.append(example["cut1"])
                cut2s.append(example["cut2"])

        if self.output_label:
            return {
                "input": {
                    "refcode": torch.from_numpy(np.stack(refcodes)),
                },
                "label": {
                    "observation": torch.from_numpy(np.stack(observation_list)),
                    "cut1": np.array(cut1s),
                    "cut2": np.array(cut2s),
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
