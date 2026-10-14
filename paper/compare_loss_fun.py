#!/usr/bin/env python

import os
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from tbparse import SummaryReader


def get_loss_fun_result(
    data_names: list[str],
    loss_funs: list[str],
    metrics: list[str],
    output_dir: os.PathLike,
) -> pd.DataFrame:
    test_dict = {
        "data_name": [],
        "loss_fun": [],
        "metric": [],
        "value": [],
        "best_epoch": [],
    }

    for data_name in data_names:
        for metric in metrics:
            for loss_fun in loss_funs:
                tb_path = (
                    output_dir
                    / "CRIfuser"
                    / "CRIfuser"
                    / data_name
                    / loss_fun
                    / "test"
                    / metric
                )
                df = SummaryReader(tb_path.as_posix()).scalars
                best_epoch, _, value = (
                    df.loc[df["tag"] == f"test/{metric}"].to_numpy().flatten()
                )
                test_dict["data_name"].append(data_name)
                test_dict["loss_fun"].append(loss_fun)
                test_dict["metric"].append(metric)
                test_dict["value"].append(value)
                test_dict["best_epoch"].append(best_epoch)

    loss_df = pd.DataFrame(test_dict)

    # save
    os.makedirs("paper/compare_loss_fun", exist_ok=True)
    loss_df.to_csv("paper/compare_loss_fun/result.csv", index=False)

    return loss_df


def draw_loss_fun_result(
    loss_df: pd.DataFrame,
    metrics: list[str],
    loss_funs: list[str],
    data_names: list[str],
) -> None:
    fontsize = 30
    for data_name in data_names:
        for metric in metrics:
            ax = (
                loss_df.query(
                    "metric == @metric and loss_fun in @loss_funs and data_name == @data_name"
                )
                .sort_values("value")
                .rename(columns={"value": metric})
                .set_index(
                    keys=[
                        "loss_fun",
                    ]
                )
                .plot.bar(y=metric, figsize=(20, 10))
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=10, ha="right")
            ax.set_xlabel(xlabel="loss_fun", fontsize=fontsize)
            ax.set_ylabel(ylabel=metric, fontsize=fontsize)
            ax.tick_params(axis="both", labelsize=fontsize)
            ax.get_legend().set_visible(False)
            ax.get_figure().tight_layout()
            ax.get_figure().savefig(f"paper/compare_loss_fun/{data_name}_{metric}.pdf")
            plt.close("all")


# Swith to non-gui backend (https://stackoverflow.com/questions/52839758/matplotlib-and-runtimeerror-main-thread-is-not-in-main-loop).
plt.switch_backend("agg")
# Editable axis in illustrator (https://stackoverflow.com/questions/54101529/how-can-i-export-a-matplotlib-figure-as-a-vector-graphic-with-editable-text-fiel)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

loss_df = get_loss_fun_result(
    data_names=["SX_spcas9", "SX_spymac", "SX_ispymac"],
    loss_funs=[
        "double_sample_negative_ELBO",
        "forward_negative_ELBO",
        "importance_sample_negative_ELBO",
        "non_sample_CE",
        "reverse_negative_ELBO",
        "sample_CE",
    ],
    metrics=[
        "GreatestCommonCrossEntropy",
    ],
    output_dir=pathlib.Path("/home/ljw/sdc1/CRISPR_results/formal/CRIfuser/logs"),
)

draw_loss_fun_result(
    loss_df,
    metrics=["GreatestCommonCrossEntropy"],
    loss_funs=[
        "double_sample_negative_ELBO",
        "forward_negative_ELBO",
        "importance_sample_negative_ELBO",
        "non_sample_CE",
        "reverse_negative_ELBO",
        "sample_CE",
    ],
    data_names=["SX_spcas9", "SX_spymac", "SX_ispymac"],
)
