#!/usr/bin/env python

import os
import pathlib

import pandas as pd
from plotnine import (
    aes,
    element_text,
    geom_raster,
    ggplot,
    scale_fill_gradient,
    scale_y_continuous,
    theme,
)

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
shap_targets = ["large_indel", "mmej", "small_indel", "unilateral"]

shap_dfs = []
for preprocess, model_cls in preprocess_model_cls_pairs:
    for data_name in data_names:
        for shap_target in shap_targets:
            h5file = (
                pathlib.Path("/home/ljw/sdc1/CRISPR_results/formal/default/logs")
                / preprocess
                / model_cls
                / data_name
                / "default"
                / "explain"
                / shap_target
                / "explanation.h5"
            )

            with pd.HDFStore(h5file) as store:
                shap_dfs.append(
                    store["values"]
                    .copy()
                    .assign(
                        base_value=store["base_values"].item(),
                        preprocess=preprocess,
                        model_cls=model_cls,
                        data_name=data_name,
                        shap_target=shap_target,
                    )
                )


shap_df = (
    pd.concat(shap_dfs)
    .groupby(["preprocess", "model_cls", "data_name", "shap_target"])
    .apply(abs)
    .groupby(["preprocess", "model_cls", "data_name", "shap_target"])
    .mean()
    .reset_index()
)

os.makedirs("paper/summary_shap", exist_ok=True)
for data_name in data_names:
    for shap_target in shap_targets:
        data = (
            shap_df.query("data_name == @data_name and shap_target == @shap_target")
            .reset_index(drop=True)
            .reset_index(names="y")
        )

        breaks = data["y"].to_list()
        labels = data["model_cls"].to_list()

        data = data.melt(
            id_vars=["model_cls", "y"],
            value_vars=[f"pos{i}" for i in range(94)],
            var_name="pos",
            value_name="value",
        ).assign(pos=lambda df: df["pos"].str.replace("pos", "").astype(int))

        (
            ggplot(data=data, mapping=aes(x="pos", y="y", fill="value"))
            + geom_raster()
            + scale_y_continuous(
                breaks=breaks,
                labels=labels,
            )
            + scale_fill_gradient(low="#FFFFFF", high="#FF0000")
            + theme(axis_text_x=element_text(angle=90, vjust=0.5, hjust=1))
        ).save(pathlib.Path("paper/summary_shap") / f"{data_name}_{shap_target}.pdf")
