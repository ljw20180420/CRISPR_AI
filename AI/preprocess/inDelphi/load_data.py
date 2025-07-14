import torch
import numpy as np
from ..utils import GetMH, SeqTokenizer


class DataCollator:
    def __init__(self, DELLEN_LIMIT: int, output_label: bool) -> None:
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
        self.get_mh = None
        self.output_label = output_label

    def gc_content(self, DNA: str) -> float:
        return (DNA.count("G") + DNA.count("C")) / len(DNA)

    def kmer2int(self, kmer):
        return int(
            "".join(str(i) for i in self.seq_tokenizer(kmer)),
            base=4,
        )

    def __call__(self, examples: list[dict]) -> dict:
        mh_inputs, mh_del_lens = [], []
        if self.output_label:
            genotype_counts, total_del_len_counts = [], []
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
            # mh_mh_lens
            mh_mh_lens = all_mh_lens[del_end_mask]
            # mh_gc_frac
            mh_gc_fracs = np.array(
                [
                    self.gc_content(ref[rightest - mh_mh_len : rightest])
                    for mh_mh_len, rightest in zip(
                        mh_mh_lens, self.rights[del_end_mask] + cut
                    )
                ]
            )
            # mh_input
            mh_inputs.append(
                np.stack(
                    [mh_mh_lens, mh_gc_fracs],
                    axis=1,
                )
            )
            # mh_del_lens
            mh_del_lens.append(self.del_lens[del_end_mask])
            if self.output_label:
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
                # mh_counts
                all_counts = observation[
                    self.rights + example["cut2"],
                    self.lefts + example["cut1"],
                ]
                mh_counts = all_counts[del_end_mask]
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
        if self.output_label:
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
        if self.output_label:
            return {
                "mh_input": mh_inputs,
                "mh_del_len": mh_del_lens,
                "genotype_count": genotype_counts,
                "total_del_len_count": total_del_len_counts,
            }
        return {
            "mh_input": mh_inputs,
            "mh_del_len": mh_del_lens,
        }

    def insert_call(self, examples: list[dict]) -> dict:
        onebp_features, m654s = [], []
        if self.output_label:
            insert_1bps, insert_probabilities = [], []
        for example in examples:
            ref = (
                example["ref1"][: example["cut1"]] + example["ref2"][example["cut2"] :]
            )
            cut = example["cut1"]
            self.assert_reference_length_and_cut(ref, cut)
            # onebp_features
            onebp_features.append(
                np.eye(len("ACGT"))[
                    self.seq_tokenizer(ref[cut - 1 : cut + 1])
                ].flatten()
            )
            # m654s
            m654s.append(self.kmer2int(ref[cut - 3 : cut]))
            if self.output_label:
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
                # cumulate observations for all random insertion size
                observation = observations.sum(axis=0)
                # del_count
                del_count = observation[
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
        onebp_features = np.stack(onebp_features)
        m654s = np.array(m654s)
        if self.output_label:
            insert_probabilities = np.array(insert_probabilities)
            insert_1bps = np.array(insert_1bps)

        if self.output_label:
            return {
                "onebp_feature": onebp_features,
                "m654": m654s,
                "insert_probability": insert_probabilities,
                "insert_1bp": insert_1bps,
            }
        return {
            "onebp_feature": onebp_features,
            "m654": m654s,
        }

    # the right end of the rightest correction of mh deletion
    def get_auxilaries(self, examples: list[dict]) -> tuple:
        rightests, mh_mh_lens, mh_del_lens = [], [], []
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

            mh_matrix, mh_idx_align_ref1, _, _ = self.get_mh(
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

            rightests.append(self.rights[del_end_mask])
            mh_mh_lens.append(all_mh_lens[del_end_mask])
            mh_del_lens.append(self.del_lens[del_end_mask])
        return rightests, mh_mh_lens, mh_del_lens

    def assert_reference_length_and_cut(self, ref: str, cut: int) -> None:
        assert (
            cut >= self.DELLEN_LIMIT - 1 and len(ref) - cut >= self.DELLEN_LIMIT - 1
        ), f"reference is too short to contain DELLEN_LIMIT {self.DELLEN_LIMIT}"

    def inference(self, examples: list[dict]) -> dict:
        assert not self.output_label, "inference cannot output count"
        for example in examples:
            ref, cut = example.pop("ref"), example.pop("cut")
            self.assert_reference_length_and_cut(ref, cut)
            example["ref1"] = example["ref2"] = ref
            example["cut1"] = example["cut2"] = cut
        return examples
