#!/usr/bin/env python

import os
import pandas as pd
import pathlib
import matplotlib.pyplot as plt

output_dir = pathlib.Path("/home/ljw/sdc1/CRISPR_results")
preprocess_model_type_pairs = [
    ("CRIformer", "CRIformer"),
    ("CRIfuser", "CRIfuser"),
    ("DeepHF", "DeepHF"),
    ("DeepHF", "MLP"),
    ("DeepHF", "CNN"),
    ("DeepHF", "XGBoost"),
    # ("DeepHF", "Ridge"),
    ("FOREcasT", "FOREcasT"),
    ("inDelphi", "inDelphi"),
    ("Lindel", "Lindel"),
]
preprocesses, model_types, data_names, metrics = [], [], [], []
for preprocess, model_type in preprocess_model_type_pairs:
    for data_name in ["SX_spcas9", "SX_spymac", "SX_ispymac"]:
        model_path = output_dir / preprocess / model_type / data_name / "default"
        df = pd.read_csv(model_path / "test_result.csv")
        metrics.append(
            df["NonWildTypeCrossEntropy_loss"].sum()
            / df["NonWildTypeCrossEntropy_loss_num"].sum()
        )

        preprocesses.append(preprocess)
        model_types.append(model_type)
        data_names.append(data_name)

df = pd.DataFrame(
    {
        "preprocess": preprocesses,
        "model_type": model_types,
        "data_name": data_names,
        "metric": metrics,
    }
)

# save
os.makedirs("benchmark", exist_ok=True)
df.to_csv("benchmark/default.csv", index=False)
ax = df.set_index(keys=["preprocess", "model_type", "data_name"]).plot.bar(
    figsize=(20, 10)
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=10, ha="right")
ax.get_figure().savefig("benchmark/default.pdf")
