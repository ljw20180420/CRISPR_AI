import torch
import torch.nn.functional as F
import numpy as np
from ..utils import MicroHomologyTool
from common_ai.utils import SeqTokenizer
from common_ai.generator import MyGenerator


class DataCollator:
    def __init__(self, max_del_size: int, mh_len: int) -> None:
        self.max_del_size = max_del_size
        self.mh_len = mh_len
        self.lefts = np.concatenate(
            [np.arange(-dl - 2, 3) for dl in range(1, self.max_del_size + 1)]
        )
        self.rights = np.concatenate(
            [np.arange(-2, dl + 3) for dl in range(1, self.max_del_size + 1)]
        )
        self.del_lens = self.rights - self.lefts
        self.seq_tokenzier = SeqTokenizer("ACGT")
        self.micro_homology_tool = MicroHomologyTool()

    def __call__(
        self, examples: list[dict], output_label: bool, my_generator: MyGenerator
    ) -> dict:
        input_indels, input_inss, input_dels = [], [], []
        if output_label:
            count_indels, count_inss, count_dels, observation_list = (
                [],
                [],
                [],
                [],
            )
        for example in examples:
            ref = (
                example["ref1"][: example["cut1"]] + example["ref2"][example["cut2"] :]
            )
            cut = example["cut1"]
            self._assert_reference_length_and_cut(ref, cut)
            mh_matrix, mh_idx_align_ref1, _, mh_rep_num = (
                self.micro_homology_tool.get_mh(
                    example["ref1"],
                    example["ref2"],
                    example["cut1"],
                    example["cut2"],
                    ext1=0,
                    ext2=0,
                )
            )
            all_mh_lens = mh_matrix.reshape(
                len(example["ref2"]) + 1,
                len(example["ref1"]) + 1,
            )[
                self.rights + example["cut2"],
                self.lefts + example["cut1"],
            ]
            # construct mh_idx_align_ref1_2D
            mh_idx_align_ref1_2D = np.full(
                (len(example["ref2"]) + 1) * (len(example["ref1"]) + 1), False
            )
            mh_idx_align_ref1_2D[mh_idx_align_ref1] = True
            del_end_mask = mh_idx_align_ref1_2D.reshape(
                len(example["ref2"]) + 1,
                len(example["ref1"]) + 1,
            )[
                self.rights + example["cut2"],
                self.lefts + example["cut1"],
            ]
            # mh_lens
            mh_lens = np.concatenate(
                [
                    all_mh_lens[del_end_mask],
                    all_mh_lens[all_mh_lens == 0],
                ],
                axis=0,
            )
            # del_lens
            del_lens = np.concatenate(
                [
                    self.del_lens[del_end_mask],
                    self.del_lens[all_mh_lens == 0],
                ],
                axis=0,
            )
            # dstarts
            dstarts = np.concatenate(
                [
                    self.lefts[del_end_mask],
                    self.lefts[all_mh_lens == 0],
                ],
                axis=0,
            )
            # input_indels
            input_indel = self._onehot_encoder(
                example["ref1"][example["cut1"] - 17 : example["cut1"] + 3]
            )
            input_indels.append(input_indel)
            # input_dels
            input_dels.append(
                np.append(self._get_feature(dstarts, del_lens, mh_lens), input_indel)
            )
            # input_inss
            input_inss.append(
                self._onehot_encoder(
                    example["ref1"][example["cut1"] - 3 : example["cut1"] + 3]
                )
            )
            if output_label:
                observation, all_counts = self.micro_homology_tool.get_observation(
                    example, mh_matrix, mh_rep_num, lefts=self.lefts, rights=self.rights
                )
                observation_list.append(observation)
                # count_dels
                count_del = all_counts.copy()
                count_del[~del_end_mask & all_mh_lens > 0] = 0
                count_dels.append(count_del)
                # count_inss
                count_ins = np.append(
                    np.array(example["insert_count"], dtype=np.float32),
                    example["insert_count_long"],
                )
                count_inss.append(count_ins)
                # count_indels
                count_indels.append([count_del.sum().item(), count_ins.sum().item()])

        input_indels = torch.from_numpy(np.stack(input_indels))
        input_dels = torch.from_numpy(np.stack(input_dels))
        input_inss = torch.from_numpy(np.stack(input_inss))
        if output_label:
            count_indels = torch.tensor(count_indels)
            count_dels = torch.from_numpy(np.stack(count_dels))
            count_inss = torch.from_numpy(np.stack(count_inss))

        if output_label:
            return {
                "input": {
                    "input_indel": input_indels,
                    "input_del": input_dels,
                    "input_ins": input_inss,
                },
                "label": {
                    "count_indel": count_indels,
                    "count_del": count_dels,
                    "count_ins": count_inss,
                    "observation": torch.from_numpy(np.stack(observation_list)),
                },
            }
        return {
            "input": {
                "input_indel": input_indels,
                "input_del": input_dels,
                "input_ins": input_inss,
            },
        }

    def _assert_reference_length_and_cut(self, ref: str, cut: int) -> None:
        assert (
            cut >= self.max_del_size + 2 and len(ref) - cut >= self.max_del_size + 2
        ), f"ref is too short to contain deletion length {self.max_del_size} up to 2bp away from the cleavage site {cut}"

    def _onehot_encoder(self, guide: str) -> torch.Tensor:
        guideVal = torch.from_numpy(self.seq_tokenzier(guide))
        return torch.cat(
            [
                F.one_hot(guideVal, num_classes=4).flatten(),
                F.one_hot(guideVal[:-1] + guideVal[1:] * 4, num_classes=16).flatten(),
            ]
        ).to(torch.float32)

    def _get_feature(
        self, dstarts: np.ndarray, del_lens: np.ndarray, mh_lens: np.ndarray
    ) -> np.ndarray:
        features = (
            len(self.lefts) * np.minimum(mh_lens, self.mh_len)
            + (self.max_del_size + 5 + del_lens + 1 + 5)
            * (self.max_del_size - del_lens)
            // 2
            + dstarts
            + del_lens
            + 2
        )
        one_hot = np.zeros((self.mh_len + 1) * len(self.lefts), dtype=np.float32)
        one_hot[features] = 1.0
        return one_hot
