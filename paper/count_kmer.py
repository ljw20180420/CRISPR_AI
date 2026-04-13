#!/usr/bin/env python

import os
import pathlib

import httpx
import numpy as np
import pandas as pd
import py2bit

os.chdir(pathlib.Path(__file__).resolve().parent.parent)


def download_t2t_genome() -> None:
    os.makedirs("paper/count_kmer")
    with httpx.stream(
        "GET", "https://hgdownload.gi.ucsc.edu/goldenPath/hs1/bigZips/hs1.2bit"
    ) as response:
        with open("paper/count_kmer/hs1.2bit", "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)


def count_kmer(k: int) -> None:
    with py2bit.open("paper/count_kmer/hs1.2bit", True) as tb:
        dfs = []
        for chrom, size in tb.chroms().items():
            seq = np.char.decode(
                np.frombuffer(tb.sequence(chrom).upper().encode(), dtype="S1")
            )
            df = pd.DataFrame()
            for i in range(k):
                df = df.assign(**{f"pos{i}": seq[i : len(seq) - k + i + 1]})
            df = (
                df.value_counts()
                .reset_index()
                .assign(
                    **{
                        "chrom": chrom,
                        f"{k}mer": lambda df, k=k: sum(
                            [df[f"pos{i}"] for i in range(1, k)], start=df[f"pos0"]
                        ),
                    }
                )
                .drop(columns=[f"pos{i}" for i in range(k)])
            )
            dfs.append(df)

        pd.concat(dfs).groupby(["chrom", f"{k}mer"]).agg(
            count=pd.NamedAgg(column="count", aggfunc="sum")
        ).reset_index().to_csv(f"paper/count_kmer/{k}mer.csv", index=False)


if __name__ == "__main__":
    count_kmer(k=2)
