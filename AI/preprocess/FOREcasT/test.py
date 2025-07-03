#!/usr/bin/env python

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
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from .load_data import data_collator


@torch.no_grad()
def test(
    model_name: str,
    data_name: str,
    test_ratio: float,
    validation_ratio: float,
    ref1len: int,
    ref2len: int,
    random_insert_uplimit: int,
    insert_uplimit: int,
    owner: str,
    batch_size: int,
    output_dir: pathlib.Path,
    device: str,
    seed: int,
    logger: logging.Logger,
) -> None:
    logger.info("load model")
    if model_name == "FOREcasT":
        from .model import FOREcasTModel

        model = FOREcasTModel.from_pretrained(
            output_dir / "FOREcasT" / model_name / data_name
        )
        # remove parent module name
        model.__module__ = model.__module__.split(".")[-1]

    logger.info("setup pipeline")
    if model_name == "FOREcasT":
        from .pipeline import FOREcasTPipeline

        pipe = FOREcasTPipeline(model)
    pipe.model.to(device)

    logger.info("load test data")
    ds = load_dataset(
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
    )
    test_dataloader = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        collate_fn=lambda examples, pre_calculated_features=model.pre_calculated_features, output_count=True: data_collator(
            examples, pre_calculated_features, output_count
        ),
    )

    logger.info("test pipeline")
    dfs, total_loss, total_batch_num, accum_sample_idx = [], 0, 0, 0
    for batch in tqdm(test_dataloader):
        current_batch_size = len(batch["ref"])
        df, loss = pipe(batch)
        df["sample_idx"] = df["sample_idx"] + accum_sample_idx
        accum_sample_idx += current_batch_size
        dfs.append(df)
        total_loss += loss
        total_batch_num += 1
    os.makedirs(f"FOREcasT/pipeline/{model_name}/{data_name}", exist_ok=True)
    with open(
        f"FOREcasT/pipeline/{model_name}/{data_name}/mean_test_loss.txt", "w"
    ) as fd:
        fd.write(f"{total_loss / total_batch_num}\n")
    pd.concat(dfs).to_csv(
        f"FOREcasT/pipeline/{model_name}/{data_name}/test_result.csv", index=False
    )

    logger.info("save pipeline")
    pipe.save_pretrained(save_directory=f"FOREcasT/pipeline/{model_name}/{data_name}")

    def ignore_func(src, names):
        return [
            name
            for name in names
            if name.startswith(f"{PREFIX_CHECKPOINT_DIR}-") or name.startswith("_")
        ]

    shutil.copyfile(
        "FOREcasT/pipeline.py", f"FOREcasT/pipeline/FOREcasT/{data_name}/pipeline.py"
    )

    for component in pipe.components.keys():
        shutil.copytree(
            output_dir / "FOREcasT" / model_name / data_name,
            f"FOREcasT/pipeline/{model_name}/{data_name}/{component}",
            ignore=ignore_func,
            dirs_exist_ok=True,
        )
