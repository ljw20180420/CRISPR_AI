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
        self.get_mh = None
        self.output_label = output_label

    def __call__(self, examples: list[dict]) -> dict:
        refcodes = []
        if self.output_label:
            observation_list = []
        for example in examples:
            ref = (
                example["ref1"][: example["cut1"]] + example["ref2"][example["cut2"] :]
            )
            cut = example["cut1"]
            self.assert_reference_length_and_cut(ref, cut)
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
                if (
                    not self.get_mh
                    or self.get_mh.ref1len != len(example["ref1"])
                    or self.get_mh.ref2len != len(example["ref2"])
                ):
                    self.get_mh = GetMH(
                        ref1len=len(example["ref1"]),
                        ref2len=len(example["ref2"]),
                    )
                mh_matrix, _, _, mh_rep_num = self.get_mh(
                    example["ref1"],
                    example["ref2"],
                    example["cut1"],
                    example["cut2"],
                    ext1=0,
                    ext2=0,
                )
                mh_idx = mh_matrix.nonzero()
                mh_val = mh_matrix[mh_idx]
                # construct observations
                observations = np.zeros(
                    (example["random_insert_uplimit"] + 2)
                    * (len(example["ref2"]) + 1)
                    * (len(example["ref1"]) + 1),
                    dtype=np.float32,
                )
                observations[example["ob_idx"]] = np.array(
                    example["ob_val"], dtype=np.float32
                )
                observations = observations.reshape(
                    example["random_insert_uplimit"] + 2,
                    len(example["ref2"]) + 1,
                    len(example["ref1"]) + 1,
                )
                # correct observations
                observations = self.get_mh.correct_observation(
                    observations, mh_matrix, mh_rep_num
                )
                # cumulate observations for all random insertion size
                observation = observations.sum(axis=0).flatten()
                # distribute count to all positions in single micro-homology diagonal
                observation[mh_idx] = observation[mh_idx] / (mh_val + 1)
                observation = observation.reshape(
                    len(example["ref2"]) + 1, len(example["ref1"]) + 1
                )
                # take the observation region based on model extension limits
                observation = observation[
                    example["cut2"]
                    - self.ext2_up : example["cut2"]
                    + self.ext2_down
                    + 1,
                    example["cut1"]
                    - self.ext1_up : example["cut1"]
                    + self.ext1_down
                    + 1,
                ]
                observation_list.append(torch.from_numpy(observation))

        if self.output_label:
            return {
                "refcode": torch.from_numpy(np.stack(refcodes)),
                "observation": torch.stack(observation_list),
            }
        return {
            "refcode": torch.from_numpy(np.stack(refcodes)),
        }

    def assert_reference_length_and_cut(self, ref: str, cut: int) -> None:
        assert (
            cut >= self.ext1_up
            and cut >= self.ext2_up
            and len(ref) - cut >= self.ext1_down
            and len(ref) - cut >= self.ext2_down
        ), f"reference is too short to support extensions, ext1_up: {self.ext1_up}, ext1_down: {self.ext1_down}, ext2_up: {self.ext2_up}, ext2_down: {self.ext2_down}"

    def inference(self, examples: list[dict]) -> dict:
        assert not self.output_label, "inference cannot output count"
        for example in examples:
            ref, cut = example.pop("ref"), example.pop("cut")
            self.assert_reference_length_and_cut(ref, cut)
            example["ref1"] = example["ref2"] = ref
            example["cut1"] = example["cut2"] = cut
        return examples
