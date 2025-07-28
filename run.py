#!/usr/bin/env python

import os
import pathlib
from AI.preprocess.config import get_config
from AI.preprocess.train import MyTrain
from AI.preprocess.test import MyTest

# change directory to the current script
os.chdir(pathlib.Path(__file__).parent)

# parse arguments
cfg = get_config().parse_args()

if cfg.subcommand == "train":
    MyTrain(**cfg.train.train.as_dict())(cfg.train)

elif cfg.subcommand == "test":
    MyTest(**cfg.test.test.as_dict())()

elif meta_data["subcommand"] == "upload":
    from AI.preprocess.upload import upload

    upload(
        preprocess=args.preprocess,
        model=model_name,
        data_name=args.dataset.data_name,
        owner=args.dataset.owner,
        logger=get_logger(args.log_level),
    )

elif meta_data["subcommand"] == "inference":
    from preprocess.inference import inference

    inference(
        preprocess=args.preprocess,
        model_name=model_name,
        inference_data=args.inference_data,
        inference_output=args.inference_output,
        data_name=args.dataset.data_name,
        owner=args.dataset.owner,
        batch_size=args.batch_size,
        device=args.device,
        logger=get_logger(args.log_level),
    )

elif meta_data["subcommand"] == "app":
    from preprocess.app import app

    app(
        preprocess=args.preprocess,
        model_name=model_name,
        data_name=args.dataset.data_name,
        owner=args.dataset.owner,
        device=args.device,
    )

elif meta_data["subcommand"] == "space":
    from preprocess.space import space

    space(
        preprocess=args.preprocess,
        model_name=model_name,
        data_name=args.dataset.data_name,
        owner=args.dataset.owner,
        device="cpu",
    )
