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


@torch.no_grad()
def test(
    preprocess: str,
    model_name: str,
    data_name: str,
    test_ratio: float,
    validation_ratio: float,
    ref1len: int,
    ref2len: int,
    random_insert_uplimit: int,
    insert_uplimit: int,
    owner: str,
    output_dir: pathlib.Path,
    batch_size: int,
    seed: int,
    device: str,
    logger: logging.Logger,
) -> None:
    logger.info("load model")
    model_module = importlib.import_module(
        f"..{preprocess}.model", package="preprocess.common"
    )
    model = getattr(model_module, f"{model_name}Model").from_pretrained(
        output_dir / preprocess / model_name / data_name
    )
    assert model_name == model.config.model_type, "model name is not consistent"
    model.__module__ = "model"

    logger.info("setup pipeline")
    pipeline_module = importlib.import_module(
        f"..{preprocess}.pipeline", package="preprocess.common"
    )
    pipe = getattr(pipeline_module, f"{model_name}Pipeline")(model)
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
            ref1len=ref1len,
            ref2len=ref2len,
            random_insert_uplimit=random_insert_uplimit,
            insert_uplimit=insert_uplimit,
        ),
        batch_size=batch_size,
        collate_fn=lambda examples: examples,
    )

    logger.info("test pipeline")
    os.makedirs(
        f"preprocess/{preprocess}/pipeline/{pipe.core_model.config.model_type}/{data_name}",
        exist_ok=True,
    )
    dfs, total_loss, total_loss_num, accum_sample_idx = [], 0, 0, 0
    for examples in tqdm(dl):
        current_batch_size = len(examples)
        df, loss, loss_num = pipe(examples, output_label=True)
        df["sample_idx"] = df["sample_idx"] + accum_sample_idx
        accum_sample_idx += current_batch_size
        dfs.append(df)
        total_loss += loss
        total_loss_num += loss_num
    with open(
        f"preprocess/{preprocess}/pipeline/{pipe.core_model.config.model_type}/{data_name}/mean_test_loss.txt",
        "w",
    ) as fd:
        fd.write(f"{total_loss / total_loss_num}\n")
    pd.concat(dfs).to_csv(
        f"preprocess/{preprocess}/pipeline/{pipe.core_model.config.model_type}/{data_name}/test_result.csv",
        index=False,
    )

    logger.info("save pipeline")
    pipe.save_pretrained(
        save_directory=f"preprocess/{preprocess}/pipeline/{pipe.core_model.config.model_type}/{data_name}"
    )

    for file in ["pipeline.py", "load_data.py"]:
        shutil.copyfile(
            f"preprocess/{preprocess}/{file}",
            f"preprocess/{preprocess}/pipeline/{pipe.core_model.config.model_type}/{data_name}/{file}",
        )

    for component in pipe.components.keys():
        file_stem = pipe.config[component][0].split(".")[-1]
        shutil.copyfile(
            output_dir
            / preprocess
            / pipe.core_model.config.model_type
            / data_name
            / f"{file_stem}.py",
            f"preprocess/{preprocess}/pipeline/{pipe.core_model.config.model_type}/{data_name}/{component}/{file_stem}.py",
        )
