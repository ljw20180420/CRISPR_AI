import torch
import numpy as np
from ..utils import MicroHomologyTool
from ...dataset.utils import SeqTokenizer


class DataCollator:
    preprocess = "inDelphi"

    def __init__(self, DELLEN_LIMIT: int) -> None:
        self.DELLEN_LIMIT = DELLEN_LIMIT
        self.lefts = np.concatenate(
            [
                np.arange(-DEL_SIZE, 1)
                for DEL_SIZE in range(self.DELLEN_LIMIT - 1, 0, -1)
            ]
        )
        self.rights = np.concatenate(
            [
                np.arange(0, DEL_SIZE + 1)
                for DEL_SIZE in range(self.DELLEN_LIMIT - 1, 0, -1)
            ]
        )
        self.del_lens = self.rights - self.lefts
        self.seq_tokenizer = SeqTokenizer("ACGT")
        self.epsilon = 1e-6
        self.micro_homology_tool = MicroHomologyTool()

    def __call__(self, examples: list[dict], output_label: bool) -> dict:
        mh_inputs, mh_del_lens, onebp_features, m654s, rightests = [], [], [], [], []
        if output_label:
            (
                genotype_counts,
                total_del_len_counts,
                insert_1bps,
                insert_probabilities,
                observation_list,
            ) = (
                [],
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
            # mh_mh_lens
            mh_mh_len = all_mh_lens[del_end_mask]
            # rightests
            rightests.append(self.rights[del_end_mask])
            # mh_gc_frac
            mh_gc_fracs = np.array(
                [
                    (ref[rt - mml : rt].count("G") + ref[rt - mml : rt].count("C"))
                    / mml
                    for mml, rt in zip(mh_mh_len, rightests[-1] + cut)
                ]
            )
            # mh_input
            mh_inputs.append(
                np.stack(
                    [mh_mh_len, mh_gc_fracs],
                    axis=1,
                )
            )
            # mh_del_lens
            mh_del_lens.append(self.del_lens[del_end_mask])
            # onebp_features
            onebp_features.append(
                np.eye(len("ACGT"))[
                    self.seq_tokenizer(ref[cut - 1 : cut + 1])
                ].flatten()
            )
            # m654s
            m654s.append(
                int(
                    "".join(str(i) for i in self.seq_tokenizer(ref[cut - 3 : cut])),
                    base=4,
                )
            )
            if output_label:
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
                observations = self.micro_homology_tool.correct_observation(
                    observations, mh_matrix, mh_rep_num
                )
                # cumulate observations for all random insertion size
                observation = observations.sum(axis=0)
                # mh_counts
                all_counts = observation[
                    self.rights + example["cut2"],
                    self.lefts + example["cut1"],
                ]
                mh_counts = all_counts[del_end_mask]
                # distribute count to all positions in single micro-homology diagonal
                observation[mh_idx] = observation[mh_idx] / (mh_val + 1)
                observation = observation.reshape(
                    len(example["ref2"]) + 1, len(example["ref1"]) + 1
                )
                observation_list.append(observation)
                # del_count
                del_count = observations[
                    self.rights + example["cut2"],
                    self.lefts + example["cut1"],
                ].sum()
                # insert_1bps
                insert_1bps.append(example["insert_count"][:4])
                # insert_probabilities
                insert_1bp_count = sum(example["insert_count"][:4])
                insert_probabilities.append(
                    insert_1bp_count / (insert_1bp_count + del_count + self.epsilon)
                )
                # mhless_counts
                mhless_counts = np.zeros(self.DELLEN_LIMIT - 1)
                np.add.at(
                    mhless_counts,
                    self.del_lens[all_mh_lens == 0] - 1,
                    all_counts[all_mh_lens == 0],
                )
                # genotype_counts
                genotype_counts.append((mh_counts, mhless_counts))
                # total_del_len_counts
                total_del_len_count = mhless_counts.copy()
                np.add.at(total_del_len_count, mh_del_lens[-1] - 1, mh_counts)
                total_del_len_counts.append(total_del_len_count)

        max_mh_genotype = max(mh_input.shape[0] for mh_input in mh_inputs)
        mh_inputs = torch.stack(
            [
                torch.from_numpy(
                    np.concatenate(
                        [
                            mh_input,
                            np.zeros(
                                (max_mh_genotype - mh_input.shape[0], mh_input.shape[1])
                            ),
                        ],
                        axis=0,
                        dtype=np.float32,
                    )
                )
                for mh_input in mh_inputs
            ]
        )
        mh_del_lens = torch.stack(
            [
                torch.from_numpy(
                    np.append(
                        mh_del_len,
                        np.full(max_mh_genotype - len(mh_del_len), self.DELLEN_LIMIT),
                    )
                )
                for mh_del_len in mh_del_lens
            ]
        )
        onebp_features = np.stack(onebp_features)
        m654s = np.array(m654s)
        if output_label:
            genotype_counts = torch.stack(
                [
                    torch.from_numpy(
                        np.concatenate(
                            [
                                mh_counts,
                                np.zeros(max_mh_genotype - len(mh_counts)),
                                mhless_counts,
                            ],
                            axis=0,
                        )
                    )
                    for mh_counts, mhless_counts in genotype_counts
                ]
            )
            total_del_len_counts = torch.stack(
                [
                    torch.from_numpy(total_del_len_count)
                    for total_del_len_count in total_del_len_counts
                ]
            )
            insert_probabilities = np.array(insert_probabilities)
            insert_1bps = np.array(insert_1bps)
        if output_label:
            return {
                "input": {
                    "mh_input": mh_inputs,
                    "mh_del_len": mh_del_lens,
                    "onebp_feature": onebp_features,
                    "m654": m654s,
                    "rightest": rightests,
                },
                "label": {
                    "genotype_count": genotype_counts,
                    "total_del_len_count": total_del_len_counts,
                    "insert_probability": insert_probabilities,
                    "insert_1bp": insert_1bps,
                    "observation": torch.from_numpy(np.stack(observation_list)),
                },
            }
        return {
            "input": {
                "mh_input": mh_inputs,
                "mh_del_len": mh_del_lens,
                "onebp_feature": onebp_features,
                "m654": m654s,
                "rightest": rightests,
            },
        }

    def _assert_reference_length_and_cut(self, ref: str, cut: int) -> None:
        assert (
            cut >= self.DELLEN_LIMIT - 1 and len(ref) - cut >= self.DELLEN_LIMIT - 1
        ), f"reference is too short to contain DELLEN_LIMIT {self.DELLEN_LIMIT}"
