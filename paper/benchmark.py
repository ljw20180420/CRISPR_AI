#!/usr/bin/env python

import os
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from tbparse import SummaryReader


def get_benchmark(
    preprocess_model_cls_pairs: list[tuple[str, str]],
    data_names: list[str],
    metrics: list[str],
    output_dir: os.PathLike,
) -> pd.DataFrame:
    test_dict = {
        "preprocess": [],
        "model_cls": [],
        "data_name": [],
        "metric": [],
        "value": [],
        "best_epoch": [],
    }

    for preprocess, model_cls in preprocess_model_cls_pairs:
        for data_name in data_names:
            for metric in metrics:
                tb_path = (
                    output_dir
                    / preprocess
                    / model_cls
                    / data_name
                    / "default"
                    / "test"
                    / metric
                )
                df = SummaryReader(tb_path.as_posix()).scalars
                best_epoch, _, value = (
                    df.loc[df["tag"] == f"test/{metric}"].to_numpy().flatten()
                )
                test_dict["preprocess"].append(preprocess)
                test_dict["model_cls"].append(model_cls)
                test_dict["data_name"].append(data_name)
                test_dict["metric"].append(metric)
                test_dict["value"].append(value)
                test_dict["best_epoch"].append(best_epoch)

    bench_df = pd.DataFrame(test_dict)

    # save
    os.makedirs("paper/benchmark", exist_ok=True)
    bench_df.to_csv("paper/benchmark/default.csv", index=False)

    return bench_df


def save_latex(
    bench_df: pd.DataFrame,
    metrics: list[str],
    models: list[str],
    data_names: list[str],
    filename: str,
) -> None:
    for metric in metrics:
        bench_df.query(
            "metric == @metric and model_cls in @models and data_name in @data_names"
        ).drop(columns=["preprocess", "metric", "best_epoch"]).assign(
            data_name=lambda df: df["data_name"]
            .str.removeprefix("SX_")
            .str.replace("spcas9", "spycas9")
        ).rename(
            columns={
                "model_cls": "model",
                "data_name": "cas protein",
                "value": metric,
            }
        ).to_latex(
            f"paper/benchmark/{metric}.{filename}", index=False, escape=True
        )


def draw_benchmark(
    bench_df: pd.DataFrame,
    metrics: list[str],
    models: list[str],
    data_names: list[str],
    filename: str,
) -> None:
    for data_name in data_names:
        for metric in metrics:
            ax = (
                bench_df.query("data_name == @data_name and metric == @metric")
                .sort_values("value")
                .rename(columns={"value": metric})
                .set_index(
                    keys=[
                        "model_cls",
                    ]
                )
                .plot.bar(y=metric, figsize=(20, 10))
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=10, ha="right")
            ax.get_figure().savefig(f"paper/benchmark/{data_name}_{metric}_{filename}")


# Swith to non-gui backend (https://stackoverflow.com/questions/52839758/matplotlib-and-runtimeerror-main-thread-is-not-in-main-loop).
plt.switch_backend("agg")
# Editable axis in illustrator (https://stackoverflow.com/questions/54101529/how-can-i-export-a-matplotlib-figure-as-a-vector-graphic-with-editable-text-fiel)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


bench_df = get_benchmark(
    preprocess_model_cls_pairs=[
        ("CRIformer", "CRIformer"),
        ("CRIfuser", "CRIfuser"),
        ("DeepHF", "DeepHF"),
        ("DeepHF", "MLP"),
        ("DeepHF", "CNN"),
        ("DeepHF", "XGBoost"),
        ("DeepHF", "SGDClassifier"),
        ("FOREcasT", "FOREcasT"),
        ("inDelphi", "inDelphi"),
        ("Lindel", "Lindel"),
    ],
    data_names=["SX_spcas9", "SX_spymac", "SX_ispymac"],
    metrics=[
        "CrossEntropy",
        "GreatestCommonCrossEntropy",
        "NonWildTypeCrossEntropy",
        "NonZeroCrossEntropy",
        "NonZeroNonWildTypeCrossEntropy",
    ],
    output_dir=pathlib.Path("/home/ljw/sdc1/CRISPR_results/formal/default/logs"),
)

save_latex(
    bench_df,
    metrics=["GreatestCommonCrossEntropy"],
    models=[
        "CRIformer",
        "CRIfuser",
        "DeepHF",
        "MLP",
        "CNN",
        "XGBoost",
        "SGDClassifier",
    ],
    data_names=["SX_spcas9", "SX_spymac", "SX_ispymac"],
    filename="select.tex",
)

save_latex(
    bench_df,
    metrics=["GreatestCommonCrossEntropy"],
    models=[
        "CRIfuser",
        "FOREcasT",
        "inDelphi",
        "Lindel",
    ],
    data_names=["SX_spcas9", "SX_spymac", "SX_ispymac"],
    filename="bench.tex",
)


draw_benchmark(
    bench_df,
    metrics=["GreatestCommonCrossEntropy"],
    models=[
        "CRIformer",
        "CRIfuser",
        "DeepHF",
        "MLP",
        "CNN",
        "XGBoost",
        "SGDClassifier",
    ],
    data_names=["SX_spcas9", "SX_spymac", "SX_ispymac"],
    filename="select.pdf",
)

draw_benchmark(
    bench_df,
    metrics=["GreatestCommonCrossEntropy"],
    models=[
        "CRIfuser",
        "FOREcasT",
        "inDelphi",
        "Lindel",
    ],
    data_names=["SX_spcas9", "SX_spymac", "SX_ispymac"],
    filename="bench.pdf",
)
