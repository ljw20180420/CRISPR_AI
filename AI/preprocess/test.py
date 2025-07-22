import os
import logging
import pathlib
import shutil
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
import datasets
import torch
from torch.utils.data import DataLoader
import importlib
from .metric import NonWildTypeCrossEntropy


@torch.no_grad()
def test(
    preprocess: str,
    model_name: str,
    data_name: str,
    test_ratio: float,
    validation_ratio: float,
    random_insert_uplimit: int,
    insert_uplimit: int,
    owner: str,
    trial_name: str,
    metrics: dict,
    output_dir: pathlib.Path,
    batch_size: int,
    seed: int,
    device: str,
    logger: logging.Logger,
) -> None:
    logger.info("load models")
    model_module = importlib.import_module(f"preprocess.{preprocess}.model")
    models = {}
    if hasattr(model_module, f"{model_name}Model"):
        logger.info("load core model")
        models["core_model"] = getattr(
            model_module, f"{model_name}Model"
        ).from_pretrained(
            output_dir / preprocess / model_name / data_name / trial_name / "core_model"
        )
        assert (
            model_name == models["core_model"].config.model_type
        ), "model name is not consistent"
        models["core_model"].__module__ = "model"
    if hasattr(model_module, f"{model_name}Auxilary"):
        logger.info("load auxilary model")
        models["auxilary_model"] = getattr(
            model_module, f"{model_name}Auxilary"
        ).from_pretrained(
            output_dir
            / preprocess
            / model_name
            / data_name
            / trial_name
            / "auxilary_model"
        )
        models["auxilary_model"].load_auxilary(
            model_pickle_file=output_dir
            / preprocess
            / model_name
            / data_name
            / trial_name
            / "auxilary_model"
            / "auxilary.pkl"
        )
        models["auxilary_model"].__module__ = "model"

    logger.info("setup pipeline")
    pipeline_module = importlib.import_module(f"preprocess.{preprocess}.pipeline")
    pipe = getattr(pipeline_module, f"{model_name}Pipeline")(**models)
    if hasattr(pipe, "core_model"):
        pipe.core_model.to(device)

    logger.info("load test data")
    dl = DataLoader(
        dataset=load_dataset(
            path=f"{owner}/CRISPR_data",
            name=data_name,
            split=datasets.Split.TEST,
            trust_remote_code=True,
            test_ratio=test_ratio,
            validation_ratio=validation_ratio,
            seed=seed,
            random_insert_uplimit=random_insert_uplimit,
            insert_uplimit=insert_uplimit,
        ),
        batch_size=batch_size,
        collate_fn=lambda examples: examples,
    )

    logger.info("test pipeline")
    dfs, accum_sample_idx = [], 0
    total_loss = {metric: 0.0 for metric in metrics.keys()}
    total_loss_num = {metric: 0 for metric in metrics.keys()}
    for examples in tqdm(dl):
        current_batch_size = len(examples)
        df, loss, loss_num = pipe(
            examples, output_label=True, metric=non_wild_type_cross_entropy
        )
        df["sample_idx"] = df["sample_idx"] + accum_sample_idx
        accum_sample_idx += current_batch_size
        dfs.append(df)
        total_loss += loss
        total_loss_num += loss_num
    with open(
        output_dir
        / preprocess
        / model_name
        / data_name
        / trial_name
        / "test_metric.json",
        "w",
    ) as fd:
        fd.write(f"{total_loss / total_loss_num}\n")
    pd.concat(dfs).to_csv(
        f"preprocess/{preprocess}/pipeline/{model_name}/{data_name}/test_result.csv",
        index=False,
    )

    logger.info("save pipeline")
    pipe.save_pretrained(
        save_directory=f"preprocess/{preprocess}/pipeline/{model_name}/{data_name}"
    )

    shutil.copyfile(
        f"preprocess/{preprocess}/pipeline.py",
        f"preprocess/{preprocess}/pipeline/{model_name}/{data_name}/pipeline.py",
    )
    # Merge utils.py into load_data.py. Also suppress the import of utils.py in load_data.py.
    with open(
        f"preprocess/{preprocess}/pipeline/{model_name}/{data_name}/load_data.py",
        "w",
    ) as wd:
        with open(f"preprocess/utils.py", "r") as rd:
            wd.write(rd.read())
        wd.write("\n")
        with open(f"preprocess/{preprocess}/load_data.py", "r") as rd:
            for line in rd:
                if line.startswith("from ..utils import "):
                    continue
                wd.write(line)

    for component in ["core_model", "auxilary_model"]:
        if hasattr(pipe, component):
            shutil.copyfile(
                f"preprocess/{preprocess}/model.py",
                f"preprocess/{preprocess}/pipeline/{model_name}/{data_name}/{component}/model.py",
            )
            if component == "auxilary_model":
                shutil.copyfile(
                    output_dir
                    / preprocess
                    / model_name
                    / data_name
                    / "auxilary_model"
                    / "auxilary.pkl",
                    f"preprocess/{preprocess}/pipeline/{model_name}/{data_name}/{component}/auxilary.pkl",
                )
