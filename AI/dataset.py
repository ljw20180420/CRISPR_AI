import os
import datasets
from typing import Literal, Callable
import re
import numpy as np

from common_ai.utils import split_train_valid_test, SeqTokenizer


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


def filter_refs(
    examples: dict,
    ref_filter: Callable,
    cut_filter: Callable,
    author_filter: Callable,
    file_filter: Callable,
):
    ref1s, ref2s, cutss = [], [], []
    for ref1, ref2, cuts in zip(examples["ref1"], examples["ref2"], examples["cuts"]):
        if ref_filter is None or ref_filter(ref1, ref2):
            if (
                cut_filter is not None
                or author_filter is not None
                or file_filter is not None
            ):
                cuts = filter_cuts(
                    cuts, ref1, ref2, cut_filter, author_filter, file_filter
                )
            if cuts:
                ref1s.append(ref1)
                ref2s.append(ref2)
                cutss.append(cuts)
    return {"ref1": ref1s, "ref2": ref2s, "cuts": cutss}


def filter_cuts(
    cuts: list,
    ref1: str,
    ref2: str,
    cut_filter: Callable,
    author_filter: Callable,
    file_filter: Callable,
):
    new_cuts = []
    for cut in cuts:
        if cut_filter is None or cut_filter(cut["cut1"], cut["cut2"], ref1, ref2):
            if author_filter is not None or file_filter is not None:
                cut["authors"] = filter_authors(
                    cut["authors"],
                    ref1,
                    ref2,
                    cut["cut1"],
                    cut["cut2"],
                    author_filter,
                    file_filter,
                )
            if cut["authors"]:
                new_cuts.append(cut)
    return new_cuts


def filter_authors(
    authors: list,
    ref1: str,
    ref2: str,
    cut1: int,
    cut2: int,
    author_filter: Callable,
    file_filter: Callable,
):
    new_authors = []
    for author in authors:
        if author_filter is None or author_filter(
            author["author"], ref1, ref2, cut1, cut2
        ):
            if file_filter is not None:
                author["files"] = filter_files(
                    author["files"],
                    ref1,
                    ref2,
                    cut1,
                    cut2,
                    author["author"],
                    file_filter,
                )
            if author["files"]:
                new_authors.append(author)
    return new_authors


def filter_files(
    files: list,
    ref1: str,
    ref2: str,
    cut1: int,
    cut2: int,
    author: str,
    file_filter: Callable,
):
    return [
        file
        for file in files
        if file_filter(file["file"], ref1, ref2, cut1, cut2, author)
    ]


def determine_scaffold(cut: dict) -> str:
    for author in cut["authors"]:
        if author["author"] == "SX":
            for file in author["files"]:
                if re.search("^(A2-|A7-|D2-)", file["file"]):
                    return "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGCTTTTTTG"
                elif re.search("^(X-|x-|B2-|36t-)", file["file"]) or re.search(
                    "^(i10t-|i83-)", file["file"]
                ):
                    return "GTTTCAGAGCTATGCTGGAAACAGCATAGCAAGTTGAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGCTTTTTTG"
                break
        break

    raise Exception("Unable to determine scaffold")


# trans_funcs
def trans_func(
    examples: dict,
    random_insert_uplimit: int,
    get_observation: GetObservation,
    get_insertion_count: GetInsertionCount,
):
    (
        ref1s,
        ref2s,
        cut1s,
        cut2s,
        scaffolds,
        ob_idxs,
        ob_vals,
        insert_countss,
        insert_count_longs,
    ) = ([], [], [], [], [], [], [], [], [])
    for ref1, ref2, cuts in zip(examples["ref1"], examples["ref2"], examples["cuts"]):
        for cut in cuts:
            # ref
            ref1s.append(ref1)
            ref2s.append(ref2)
            # cut
            cut1s.append(cut["cut1"])
            cut2s.append(cut["cut2"])
            # scaffold
            scaffolds.append(determine_scaffold(cut))
            # observe
            observations = get_observation(ref1, ref2, cut["authors"]).flatten()
            (ob_idx,) = observations.nonzero()
            ob_idxs.append(ob_idx)
            ob_vals.append(observations[ob_idx])
            # insert
            insert_counts, insert_count_long = get_insertion_count(ref1, ref2, cut)
            insert_countss.append(insert_counts)
            insert_count_longs.append(insert_count_long)
    return {
        "ref1": ref1s,
        "ref2": ref2s,
        "cut1": cut1s,
        "cut2": cut2s,
        "scaffold": scaffolds,
        "random_insert_uplimit": [random_insert_uplimit] * len(examples["ref1"]),
        "ob_idx": ob_idxs,
        "ob_val": ob_vals,
        "insert_count": insert_countss,
        "insert_count_long": insert_count_longs,
    }


def get_dataset(
    data_file: os.PathLike,
    name: Literal["SX_spcas9", "SX_spymac", "SX_ispymac"],
    test_ratio: float,
    validation_ratio: float,
    random_insert_uplimit: int,
    insert_uplimit: int,
    seed: int,
    **kwargs,
):
    """Parameters of dataset.

    Args:
        data_file: File path.
        name: Data name. Generally correpond to Cas protein name.
        test_ratio: Proportion for test samples.
        validation_ratio: Proportion for validation samples.
        random_insert_uplimit: The maximal discriminated length of random insertion.
        insert_uplimit: The maximal insertion length to count.
        seed: random seed.
    """
    ds = datasets.load_dataset(
        "json",
        data_files=data_file,
        features=datasets.Features(
            {
                "ref1": datasets.Value("string"),
                "ref2": datasets.Value("string"),
                "cuts": [
                    datasets.Features(
                        {
                            "cut1": datasets.Value("int64"),
                            "cut2": datasets.Value("int64"),
                            "authors": [
                                datasets.Features(
                                    {
                                        "author": datasets.Value("string"),
                                        "files": [
                                            datasets.Features(
                                                {
                                                    "file": datasets.Value("string"),
                                                    "ref1_end": datasets.Sequence(
                                                        datasets.Value("int64")
                                                    ),
                                                    "ref2_start": datasets.Sequence(
                                                        datasets.Value("int64")
                                                    ),
                                                    "random_insert": datasets.Sequence(
                                                        datasets.Value("string")
                                                    ),
                                                    "count": datasets.Sequence(
                                                        datasets.Value("int64")
                                                    ),
                                                }
                                            )
                                        ],
                                    }
                                )
                            ],
                        }
                    )
                ],
            }
        ),
    )

    if name == "SX_spcas9":
        filters = {
            "ref_filter": None,
            "cut_filter": None,
            "author_filter": lambda author, ref1, ref2, cut1, cut2: author == "SX",
            "file_filter": lambda file, ref1, ref2, cut1, cut2, author: bool(
                re.search("^(A2-|A7-|D2-)", file)
            ),
        }
    elif name == "SX_spymac":
        filters = {
            "ref_filter": None,
            "cut_filter": None,
            "author_filter": lambda author, ref1, ref2, cut1, cut2: author == "SX",
            "file_filter": lambda file, ref1, ref2, cut1, cut2, author: bool(
                re.search("^(X-|x-|B2-|36t-)", file)
            ),
        }
    else:
        assert (
            name == "SX_ispymac"
        ), "name can only be SX_spcas9, SX_spymac or SX_ispymac"
        filters = {
            "ref_filter": None,
            "cut_filter": None,
            "author_filter": lambda author, ref1, ref2, cut1, cut2: author == "SX",
            "file_filter": lambda file, ref1, ref2, cut1, cut2, author: bool(
                re.search("^(i10t-|i83-)", file)
            ),
        }

    ds = ds.map(
        lambda examples, filters=filters: filter_refs(examples, **filters),
        batched=True,
    )

    ds = ds.map(
        lambda examples, random_insert_uplimit=random_insert_uplimit, get_observation=GetObservation(
            random_insert_uplimit
        ), get_insertion_count=GetInsertionCount(
            "ACGT", insert_uplimit
        ): trans_func(
            examples, random_insert_uplimit, get_observation, get_insertion_count
        ),
        batched=True,
        remove_columns=["cuts"],
    )

    ds = split_train_valid_test(ds, validation_ratio, test_ratio, seed)

    return ds
