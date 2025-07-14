import torch
import torch.nn.functional as F
import numpy as np
from ..utils import GetMH, SeqTokenizer


class DataCollator:
    def __init__(self, dlen: int, mh_len: int, output_label: bool) -> None:
        self.dlen = dlen
        self.mh_len = mh_len
        self.lefts = np.concatenate(
            [np.arange(-dl - 2, 3) for dl in range(1, self.dlen)]
        )
        self.rights = np.concatenate(
            [np.arange(-2, dl + 3) for dl in range(1, self.dlen)]
        )
        self.del_lens = self.rights - self.lefts
        self.seq_tokenzier = SeqTokenizer("ACGT")
        self.get_mh = None
        self.output_label = output_label

    def __call__(self, examples: list[dict]) -> dict:
        input_indels, input_inss, input_dels = [], [], []
        if self.output_label:
            count_indels, count_inss, count_dels = [], [], []
        for example in examples:
            ref = (
                example["ref1"][: example["cut1"]] + example["ref2"][example["cut2"] :]
            )
            cut = example["cut1"]
            self.assert_reference_length_and_cut(ref, cut)
            if (
                not self.get_mh
                or self.get_mh.ref1len != len(example["ref1"])
                or self.get_mh.ref2len != len(example["ref2"])
            ):
                self.get_mh = GetMH(
                    ref1len=len(example["ref1"]),
                    ref2len=len(example["ref2"]),
                )
            (
                mh_matrix,
                mh_rep_num,
                all_mh_lens,
                del_end_mask,
                mh_lens,
                del_lens,
                dstarts,
            ) = self.get_delete_info(example)
            # input_indels
            input_indel = self.onehot_encoder(
                example["ref1"][example["cut1"] - 17 : example["cut1"] + 3]
            )
            input_indels.append(input_indel)
            # input_dels
            input_dels.append(
                np.append(self.get_feature(dstarts, del_lens, mh_lens), input_indel)
            )
            # input_inss
            input_inss.append(
                self.onehot_encoder(
                    example["ref1"][example["cut1"] - 3 : example["cut1"] + 3]
                )
            )
            if self.output_label:
                # count_inss
                count_ins = np.append(
                    np.array(example["insert_count"], dtype=np.float32),
                    example["insert_count_long"],
                )
                count_inss.append(count_ins)
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
                observation = observations.sum(axis=0)
                # count_dels
                count_del = observation[
                    self.rights + example["cut2"],
                    self.lefts + example["cut1"],
                ]
                count_del[~del_end_mask & all_mh_lens > 0] = 0
                count_dels.append(count_del)
                # count_indels
                count_indels.append([count_del.sum().item(), count_ins.sum().item()])

        input_indels = torch.from_numpy(np.stack(input_indels))
        input_dels = torch.from_numpy(np.stack(input_dels))
        input_inss = torch.from_numpy(np.stack(input_inss))
        if self.output_label:
            count_indels = torch.tensor(count_indels)
            count_dels = torch.from_numpy(np.stack(count_dels))
            count_inss = torch.from_numpy(np.stack(count_inss))

        if self.output_label:
            return {
                "input": {
                    "input_indel": input_indels,
                    "input_del": input_dels,
                    "input_ins": input_inss,
                },
                "count": {
                    "count_indel": count_indels,
                    "count_del": count_dels,
                    "count_ins": count_inss,
                },
            }
        return {
            "input": {
                "input_indel": input_indels,
                "input_del": input_dels,
                "input_ins": input_inss,
            },
        }

    def get_delete_info(self, example: dict) -> tuple:
        mh_matrix, mh_idx_align_ref1, _, mh_rep_num = self.get_mh(
            example["ref1"],
            example["ref2"],
            example["cut1"],
            example["cut2"],
            ext1=0,
            ext2=0,
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

        return (
            mh_matrix,
            mh_rep_num,
            all_mh_lens,
            del_end_mask,
            mh_lens,
            del_lens,
            dstarts,
        )

    def assert_reference_length_and_cut(self, ref: str, cut: int) -> None:
        assert (
            cut >= self.dlen - 1 + 2 and len(ref) - cut >= self.dlen - 1 + 2
        ), f"ref is too short to contain deletion length {self.dlen - 1} up to 2bp away from the cleavage site {cut}"

    def onehot_encoder(self, guide: str) -> torch.Tensor:
        guideVal = torch.from_numpy(self.seq_tokenzier(guide))
        return torch.cat(
            [
                F.one_hot(guideVal, num_classes=4).flatten(),
                F.one_hot(guideVal[:-1] + guideVal[1:] * 4, num_classes=16).flatten(),
            ]
        ).to(torch.float32)

    def get_feature(
        self, dstarts: np.ndarray, del_lens: np.ndarray, mh_lens: np.ndarray
    ) -> np.ndarray:
        features = (
            len(self.lefts) * np.minimum(mh_lens, self.mh_len)
            + (self.dlen - 1 + 5 + del_lens + 1 + 5) * (self.dlen - del_lens - 1) // 2
            + dstarts
            + del_lens
            + 2
        )
        one_hot = np.zeros((self.mh_len + 1) * len(self.lefts), dtype=np.float32)
        one_hot[features] = 1.0
        return one_hot
