#!/usr/bin/env python

import os
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, pathlib.Path(__file__).parent.parent.as_posix())
from datasets import concatenate_datasets

from AI.dataset import MyDataset
from AI.preprocess.utils import MicroHomologyTool


def correct_micro_homology(row: pd.Series) -> pd.Series:
    ref1 = row["ref1"]
    ref2 = row["ref2"]
    cut1 = row["cut1"]
    ref1_end = row["ref1_end"]
    ref2_start = row["ref2_start"]
    while ref1_end < cut1 and ref2_start < len(ref2):
        if ref1[ref1_end] != ref2[ref2_start]:
            break
        ref1_end += 1
        ref2_start += 1
    while ref1_end > cut1 and ref2_start > 0:
        if ref1[ref1_end - 1] != ref2[ref2_start - 1]:
            break
        ref1_end -= 1
        ref2_start -= 1

    return pd.Series(
        {
            "ref1_end": ref1_end,
            "ref2_start": ref2_start,
        }
    )


def search_stagger_examples(
    data_file: os.PathLike, data_name: str, minimal_count: int
) -> tuple:
    ds = MyDataset(
        data_file,
        data_name,
        test_ratio=0.05,
        validation_ratio=0.05,
        seed=63036,
        random_insert_uplimit=0,
        insert_uplimit=2,
    )()
    df = concatenate_datasets([ds["train"], ds["validation"], ds["test"]]).to_pandas()

    df = (
        df[["ref1", "ref2", "cut1", "cut2", "ob_idx", "ob_val"]]
        .explode(["ob_idx", "ob_val"])
        .reset_index(drop=True)
        .assign(
            ref1_end=lambda df: df["ob_idx"] % (df["ref1"].str.len() + 1),
            ref2_start=lambda df: (df["ob_idx"] // (df["ref1"].str.len() + 1))
            % (df["ref2"].str.len() + 1),
            ob_val=lambda df: df["ob_val"].astype(int),
        )
        .drop(columns=["ob_idx"])
    )
    total = df["ob_val"].sum()

    df = df.query("ref2_start < cut2 and ob_val >= @minimal_count").reset_index(
        drop=True
    )

    df_cmh = df.apply(correct_micro_homology, axis=1)
    df["ref1_end"] = df_cmh["ref1_end"]
    df["ref2_start"] = df_cmh["ref2_start"]

    df = (
        df.query("ref1_end == cut1 and ref2_start < cut2 - 2")
        .sort_values("ob_val", ascending=False)
        .reset_index(drop=True)
        .head(n=1000)
    )

    df.to_csv(f"paper/figure/stagger_samples_{data_name}.csv", index=False)
    ref1 = df.loc[0, "ref1"]
    ref2 = df.loc[0, "ref2"]
    cut1 = df.loc[0, "cut1"]
    cut2 = df.loc[0, "cut2"]
    count = df.loc[0, "ob_val"]
    stagger = df.loc[0, "cut2"] - df.loc[0, "ref2_start"]

    return total, ref1[cut1 - 25 : cut1] + ref2[cut2 : cut2 + 25], count, stagger


def draw_model_archtecture_mmej(
    total: int, ref: str, count: int, stagger: int, data_name: str
):
    ref1 = ref[:31]
    ref2 = ref[19:]
    micro_homology_tool = MicroHomologyTool()
    micro_homology_tool.reinitialize(ref1=ref1, ref2=ref2)
    mh_matrix, _, _, _ = micro_homology_tool.get_mh(
        ref1=ref1, ref2=ref2, cut1=25, cut2=6, ext1=0, ext2=0
    )
    mh_matrix = mh_matrix.reshape(len(ref2) + 1, len(ref1) + 1)
    for shift in range(-len(ref2) + 1, len(ref1)):
        row = -shift if shift < 0 else 0
        col = row + shift
        while row <= len(ref2) and col <= len(ref1):
            if mh_matrix[row, col] > 0:
                mh_len = mh_matrix[row, col]
                row += mh_len
                col += mh_len
                mh_matrix[row, col] = 0
            row += 1
            col += 1

    mh_matrix = mh_matrix[:-1, :-1]
    min_mh = 2
    mh_matrix[mh_matrix < min_mh] = 0

    fig, ax = plt.subplots()
    ax.imshow(
        mh_matrix,
        cmap=LinearSegmentedColormap(
            name="white_red",
            segmentdata={
                "red": [(0.0, 1.0, 1.0), (1.0, 1.0, 1.0)],
                "green": [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
                "blue": [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
            },
            N=256,
        ),
    )
    ax.scatter(24.5, 5.5, c="black", marker="x", clip_on=False)
    ax.scatter(
        np.random.choice(np.arange(-0.5, 31), 1),
        np.random.choice(np.arange(-0.5, 31), 1),
        c="black",
        marker="o",
        clip_on=False,
    )

    ax.set_xticks(
        ticks=np.arange(31),
        labels=list(ref1),
        fontdict={"family": "monospace", "color": "blue", "size": 10},
    )
    ax.set_yticks(
        ticks=np.arange(31),
        labels=list(ref2),
        fontdict={"family": "monospace", "color": "blue", "size": 10},
    )

    ax.spines["bottom"].set_color("green")
    ax.spines["top"].set_color("green")
    ax.spines["left"].set_color("green")
    ax.spines["right"].set_color("green")
    ax.tick_params(axis="both", left=False, bottom=False)
    ax.set_xticks(np.arange(-0.5, 31, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 31, 1), minor=True)
    ax.grid(which="minor", color="green", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.savefig(
        f"paper/figure/model_archtecture_mmej_{data_name}_{count}_{stagger}_{total}.pdf"
    )


# Swith to non-gui backend (https://stackoverflow.com/questions/52839758/matplotlib-and-runtimeerror-main-thread-is-not-in-main-loop).
plt.switch_backend("agg")
# Editable axis in illustrator (https://stackoverflow.com/questions/54101529/how-can-i-export-a-matplotlib-figure-as-a-vector-graphic-with-editable-text-fiel)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# for data_name in ["SX_spcas9", "SX_spymac", "SX_ispymac"]:
#     total, ref, count, stagger = search_stagger_examples(
#         data_file="AI/dataset/dataset.json.gz", data_name=data_name, minimal_count=1000
#     )
#     draw_model_archtecture_mmej(total, ref, count, stagger, data_name)

sx_ref = "CCTGAAAGATACACCTTGTAGTCCTCCGTAAGGTAGAGCAGGCCCAGGTA"
draw_model_archtecture_mmej(0, sx_ref, 0, 0, "unknown")
