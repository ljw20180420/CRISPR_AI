#!/usr/bin/env python

import numpy as np


class SeqTokenizer:
    def __init__(self, alphabet: str) -> None:
        self.ascii_code = np.frombuffer(alphabet.encode(), dtype=np.int8)
        self.int2idx = np.empty(self.ascii_code.max() + 1, dtype=int)
        for i, c in enumerate(self.ascii_code):
            self.int2idx[c] = i

    def __call__(self, seq: str) -> np.ndarray:
        return self.int2idx[np.frombuffer(seq.encode(), dtype=np.int8)]


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
