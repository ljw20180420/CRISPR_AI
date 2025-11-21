#!/usr/bin/env python

import os
import pandas as pd
import pathlib
from tbparse import SummaryReader

output_dir = pathlib.Path("/home/ljw/sdc1/CRISPR_results/formal/default/logs")
preprocess_model_cls_pairs = [
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
]
data_names = ["SX_spcas9", "SX_spymac", "SX_ispymac"]
metrics = [
    "CrossEntropy",
    "GreatestCommonCrossEntropy",
    "NonWildTypeCrossEntropy",
    "NonZeroCrossEntropy",
    "NonZeroNonWildTypeCrossEntropy",
]

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


test_df = pd.DataFrame(test_dict)

# save
os.makedirs("paper/benchmark", exist_ok=True)
test_df.to_csv("paper/benchmark/default.csv", index=False)
test_df.query("metric == 'GreatestCommonCrossEntropy'").drop(
    columns=["metric", "best_epoch"]
).assign(
    data_name=lambda df: df["data_name"]
    .str.removeprefix("SX_")
    .str.replace("spcas9", "spycas9")
).rename(
    columns={"model_cls": "model", "data_name": "cas protein", "value": "cross entropy"}
).to_latex(
    "paper/benchmark/default.tex", index=False, escape=True
)

for data_name in data_names:
    for metric in metrics:
        ax = (
            test_df.query("data_name == @data_name and metric == @metric")
            .sort_values("value")
            .set_index(
                keys=[
                    "preprocess",
                    "model_cls",
                    "data_name",
                ]
            )
            .plot.bar(y="value", figsize=(20, 10))
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=10, ha="right")
        ax.get_figure().savefig(f"paper/benchmark/default_{data_name}_{metric}.pdf")
