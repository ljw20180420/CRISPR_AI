import torch
import numpy as np


class DataCollator:
    def __init__(self, DELLEN_LIMIT, output_label: bool) -> None:
        self.DELLEN_LIMIT = DELLEN_LIMIT
        self.output_label = output_label
        self.lefts = np.concatenate(
            [
                np.arange(-DEL_SIZE, 1)
                for DEL_SIZE in range(self.DELLEN_LIMIT - 1, -1, 0)
            ]
        )
        self.rights = np.concatenate(
            [
                np.arange(0, DEL_SIZE + 1)
                for DEL_SIZE in range(self.DELLEN_LIMIT - 1, -1, 0)
            ]
        )
        self.del_lens = self.rights - self.lefts

    def __call__(self, examples: list[dict]) -> dict:
        max_mh_genotype = max(
            [len(example["mh_idx_align_ref1"]) for example in examples]
        )
        for example in examples:
            if self.output_label:
                # construct observations
                observations = torch.zeros(
                    (example["random_insert_uplimit"] + 2)
                    * (len(example["ref2"]) + 1)
                    * (len(example["ref1"]) + 1),
                    dtype=torch.float32,
                )
                observations[example["ob_idx"]] = torch.tensor(
                    example["ob_val"], dtype=torch.float32
                )
                # cumulate observations for all random insertion size
                observation = (
                    observations.reshape(
                        example["random_insert_uplimit"] + 2,
                        len(example["ref2"]) + 1,
                        len(example["ref1"]) + 1,
                    )
                    .sum(axis=0)
                    .flatten()
                )
                # construct mh_len_2D
                mh_len_2D = torch.zeros(
                    (len(example["ref2"]) + 1) * (len(example["ref1"]) + 1),
                    dtype=torch.float32,
                )
                mh_len_2D[example["mh_idx"]] = torch.tensor(example["mh_val"])
                # construct mh_idx_align_ref1_2D
                mh_idx_align_ref1_2D = torch.full(
                    (len(example["ref2"]) + 1) * (len(example["ref1"]) + 1), False
                )
                mh_idx_align_ref1_2D[example["mh_idx_align_ref1"]] = True
                # mh_counts
                del_end_mask = mh_idx_align_ref1_2D[
                    self.rights + example["cut2"],
                    self.lefts + example["cut1"],
                ]
                all_counts = observations[
                    self.rights + example["cut2"],
                    self.lefts + example["cut1"],
                ]
                mh_counts = all_counts[del_end_mask]
                # mhless_counts
                all_mh_lens = mh_len_2D[
                    self.rights + example["cut2"],
                    self.lefts + example["cut1"],
                ]
                mhless_counts = all_counts[all_mh_lens == 0]
                # mh_del_lens
                mh_del_lens = self.del_lens[del_end_mask]
                # mh_mh_lens
                mh_mh_lens = all_mh_lens[del_end_mask]

            ref = (
                example["ref1"][: example["cut1"]] + example["ref2"][example["cut2"] :]
            )
            cut = example["cut1"]


def kmer2int(kmer):
    return int(
        "".join(
            [
                str(i)
                for i in (np.frombuffer(kmer.encode(), dtype=np.int8) % 5).clip(max=3)
            ]
        ),
        base=4,
    )


outputs_train_deletion = [
    "mh_input",
    "mh_del_len",
    "genotype_count",
    "total_del_len_count",
]
outputs_train_insertion = [
    "mh_input",
    "mh_del_len",
    "onebp_feature",
    "m654",
    "insert_probability",
    "insert_1bp",
]
outputs_test = [
    "mh_input",
    "mh_del_len",
    "onebp_feature",
    "m654",
    "total_del_len_count",
    "insert_probability",
]
outputs_inference = ["mh_input", "mh_del_len", "onebp_feature", "m654"]


@torch.no_grad()
def data_collector(examples, DELLEN_LIMIT, outputs, epsilon=1e-6):
    max_mh_genotype = max([len(example["mh_mh_len"]) for example in examples])
    results = dict()
    if "mh_input" in outputs:
        results["mh_input"] = torch.tensor(
            [
                [
                    [mh_mh_len, mh_gc_frac]
                    for mh_mh_len, mh_gc_frac in zip(
                        example["mh_mh_len"], example["mh_gc_frac"]
                    )
                ]
                + [[0.0, 0.0]] * (max_mh_genotype - len(example["mh_mh_len"]))
                for example in examples
            ],
            dtype=torch.float32,
        )
    if "mh_del_len" in outputs:
        results["mh_del_len"] = torch.tensor(
            [
                example["mh_del_len"]
                + [DELLEN_LIMIT] * (max_mh_genotype - len(example["mh_del_len"]))
                for example in examples
            ],
            dtype=torch.int64,
        )
    if "genotype_count" in outputs:
        results["genotype_count"] = torch.tensor(
            [
                example["mh_count"]
                + [0] * (max_mh_genotype - len(example["mh_count"]))
                + example["mhless_count"]
                for example in examples
            ],
            dtype=torch.float32,
        )
    if "total_del_len_count" in outputs:
        results["total_del_len_count"] = torch.stack(
            [
                torch.tensor(example["mhless_count"], dtype=torch.float32).scatter_add(
                    dim=0,
                    index=torch.tensor(example["mh_del_len"], dtype=torch.int64) - 1,
                    src=torch.tensor(example["mh_count"], dtype=torch.float32),
                )
                for example in examples
            ]
        )
    if "onebp_feature" in outputs:
        results["onebp_feature"] = torch.stack(
            [
                torch.eye(4)[
                    torch.from_numpy(
                        (
                            np.frombuffer(
                                example["ref"][
                                    example["cut"] - 1 : example["cut"] + 1
                                ].encode(),
                                dtype=np.int8,
                            )
                            % 5
                        )
                        .clip(max=3)
                        .astype(np.int64)
                    )
                ].flatten()
                for example in examples
            ]
        )
    if "m654" in outputs:
        results["m654"] = torch.tensor(
            [
                kmer2int(example["ref"][example["cut"] - 3 : example["cut"]])
                for example in examples
            ],
            dtype=torch.int64,
        )
    if "insert_probability" in outputs:
        results["insert_probability"] = [
            sum(example["insert_1bp"])
            / (
                sum(example["insert_1bp"])
                + sum(example["mh_count"])
                + sum(example["mhless_count"])
                + epsilon
            )
            for example in examples
        ]
    if "insert_1bp" in outputs:
        results["insert_1bp"] = torch.tensor(
            [example["insert_1bp"] for example in examples], dtype=torch.float32
        )
    return results
