#!/usr/bin/env python

import os
import pathlib
import sys

os.chdir(pathlib.Path(__file__).resolve().parent.parent)
sys.path.insert(0, os.getcwd())

import datasets
from common_ai.config import get_train_parser
from common_ai.test import MyTest

from AI.dataset import MyDataset


def draw_reverse_diffusion(
    output_dir: os.PathLike,
    run_type: str,
    run_name: str,
    preprocess: str,
    model_cls: str,
    data_name: str,
    trial_name: str,
    data_idx: int,
    sample_num: int,
) -> None:
    _, _, _, model, my_generator = MyTest(
        checkpoints_path=output_dir
        / run_type
        / run_name
        / "checkpoints"
        / preprocess
        / model_cls
        / data_name
        / trial_name,
        logs_path=output_dir
        / run_type
        / run_name
        / "logs"
        / preprocess
        / model_cls
        / data_name
        / trial_name,
        target="GreatestCommonCrossEntropy",
        maximize_target=False,
        overwrite={},
    ).load_model(get_train_parser())

    ds = MyDataset(
        data_file="AI/dataset/test.json.gz",
        name=data_name,
        test_ratio=0.05,
        validation_ratio=0.05,
        seed=63036,
        random_insert_uplimit=0,
        insert_uplimit=2,
    )()
    ds = datasets.concatenate_datasets([ds["train"], ds["validation"], ds["test"]])

    batch = model.data_collator(
        examples=[ds[data_idx]], output_label=True, my_generator=my_generator
    )

    condition = batch["input"]["condition"][0]
    cut1 = batch["label"]["cut1"][0]
    cut2 = batch["label"]["cut2"][0]
    perfect_ob = batch["label"]["observation"][0][
        cut2 - model.ext2_up : cut2 + model.ext2_down + 1,
        cut1 - model.ext1_up : cut1 + model.ext1_down + 1,
    ]

    perfect_path = model.reverse_diffusion(
        condition=condition,
        sample_num=sample_num,
        perfect_ob=perfect_ob,
    )
    model.draw_reverse_diffusion(
        perfect_path,
        filestem=f"paper/reverse_diffusion/perfect_{data_name}",
        interval=120,
        pad=5,
    )
    predict_path = model.reverse_diffusion(
        condition=condition,
        sample_num=sample_num,
        perfect_ob=None,
    )
    model.draw_reverse_diffusion(
        predict_path,
        filestem=f"paper/reverse_diffusion/predict_{data_name}",
        interval=120,
        pad=5,
    )


os.makedirs("paper/reverse_diffusion", exist_ok=True)
sample_num = 1000
for data_idx, data_name in zip([1, 6, 6], ["SX_spcas9", "SX_spymac", "SX_ispymac"]):
    draw_reverse_diffusion(
        output_dir=pathlib.Path(os.environ["OUTPUT_DIR"]),
        run_type="formal",
        run_name="default",
        preprocess="CRIfuser",
        model_cls="CRIfuser",
        data_name=data_name,
        trial_name="default",
        data_idx=data_idx,
        sample_num=sample_num,
    )
