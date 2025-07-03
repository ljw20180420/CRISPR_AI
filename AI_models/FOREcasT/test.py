#!/usr/bin/env python

import logging
import pathlib
import shutil
from tqdm import tqdm
from datasets import load_dataset
import datasets
import torch
from torch.utils.data import DataLoader
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from .model import FOREcasTModel
from .pipeline import FOREcasTPipeline
from .load_data import data_collator


@torch.no_grad()
def test(
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
    FOREcasT_model = FOREcasTModel.from_pretrained(
        output_dir / FOREcasTModel.config_class.model_type / data_name
    )
    # remove parent module name
    FOREcasT_model.__module__ = FOREcasT_model.__module__.split(".")[-1]

    logger.info("setup pipeline")
    pipe = FOREcasTPipeline(FOREcasT_model)
    pipe.FOREcasT_model.to(device)

    logger.info("load test data")
    ds = load_dataset(
        path="%s/CRISPR_data" % owner,
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
        collate_fn=lambda examples, pre_calculated_features=FOREcasT_model.pre_calculated_features, output_count=True: data_collator(
            examples, pre_calculated_features, output_count
        ),
    )

    logger.info("test pipeline")
    for batch in tqdm(test_dataloader):
        output = pipe(batch)

    logger.info("save pipeline")
    pipe.save_pretrained(save_directory="FOREcasT/pipeline")

    def ignore_func(src, names):
        return [
            name
            for name in names
            if name.startswith(f"{PREFIX_CHECKPOINT_DIR}-") or name.startswith("_")
        ]

    shutil.copyfile("FOREcasT/pipeline.py", "FOREcasT/pipeline/pipeline.py")

    for component in pipe.components.keys():
        shutil.copytree(
            output_dir / FOREcasTModel.config_class.model_type / data_name,
            "FOREcasT/pipeline/%s" % component,
            ignore=ignore_func,
            dirs_exist_ok=True,
        )
