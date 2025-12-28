#!/usr/bin/env python

import os
import pathlib
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import umap
from plotnine import (
    aes,
    element_text,
    geom_point,
    geom_raster,
    ggplot,
    scale_color_manual,
    scale_fill_gradient,
    scale_fill_manual,
    scale_y_continuous,
    theme,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def collect_shap(
    preprocess_model_cls_pairs: list[tuple[str, str]],
    data_names: list[str],
    shap_targets: list[str],
) -> pd.DataFrame:
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

    return pd.concat(shap_dfs).reset_index(drop=True)


def shap_heatmap(
    shap_df: pd.DataFrame, data_names: list[str], shap_targets: list[str]
) -> None:
    abs_mean_df = (
        shap_df.groupby(["preprocess", "model_cls", "data_name", "shap_target"])
        .apply(abs)
        .groupby(["preprocess", "model_cls", "data_name", "shap_target"])
        .mean()
        .reset_index()
    )

    os.makedirs("paper/summary_shap", exist_ok=True)
    for data_name in data_names:
        for shap_target in shap_targets:
            data = (
                abs_mean_df.query(
                    "data_name == @data_name and shap_target == @shap_target"
                )
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
            ).save(
                pathlib.Path("paper/summary_shap") / f"{data_name}_{shap_target}.pdf"
            )


def shap_reducer(
    shap_df: pd.DataFrame,
    data_names: list[str],
    shap_targets: list[str],
    method: Literal["pca", "umap"],
) -> None:
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2)

    os.makedirs(f"paper/summary_shap/{method}", exist_ok=True)
    for data_name in data_names:
        for shap_target in shap_targets:
            shap_df_sub = shap_df.query(
                "data_name == @data_name and shap_target == @shap_target"
            )
            data = shap_df_sub[[f"pos{i}" for i in range(94)]].to_numpy()
            scaled_data = StandardScaler().fit_transform(data)
            embeddings = reducer.fit_transform(scaled_data)
            shap_df_sub = shap_df_sub.assign(
                **{
                    f"{method}_x": embeddings[:, 0],
                    f"{method}_y": embeddings[:, 1],
                }
            )

            (
                ggplot(
                    shap_df_sub,
                    aes(
                        x=f"{method}_x",
                        y=f"{method}_y",
                        color="model_cls",
                        fill="model_cls",
                    ),
                )
                + geom_point(alpha=0.1, size=0.2)
                + scale_color_manual(
                    values={
                        "CRIfuser": "#FF0000",
                        "CRIformer": "#FF0000",
                        "DeepHF": "#00FF00",
                        "CNN": "#00FF00",
                        "MLP": "#00FF00",
                        "inDelphi": "#00FF00",
                        "FOREcasT": "#00FF00",
                        "Lindel": "#00FF00",
                        "XGBoost": "#00FF00",
                        "SGDClassifier": "#00FF00",
                    }
                )
                + scale_fill_manual(
                    values={
                        "CRIfuser": "#FF0000",
                        "CRIformer": "#FF0000",
                        "DeepHF": "#00FF00",
                        "CNN": "#00FF00",
                        "MLP": "#00FF00",
                        "inDelphi": "#00FF00",
                        "FOREcasT": "#00FF00",
                        "Lindel": "#00FF00",
                        "XGBoost": "#00FF00",
                        "SGDClassifier": "#00FF00",
                    }
                )
            ).save(
                pathlib.Path(f"paper/summary_shap/{method}")
                / f"{data_name}_{shap_target}_{method}.pdf"
            )


# Swith to non-gui backend (https://stackoverflow.com/questions/52839758/matplotlib-and-runtimeerror-main-thread-is-not-in-main-loop).
plt.switch_backend("agg")

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

shap_df = collect_shap(preprocess_model_cls_pairs, data_names, shap_targets)
# shap_heatmap(shap_df, data_names, shap_targets)
# shap_reducer(shap_df, data_names, shap_targets, method="pca")
shap_reducer(shap_df, data_names, shap_targets, method="umap")
