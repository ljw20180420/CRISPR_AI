#!/usr/bin/env python

import os
import pathlib
from AI.preprocess.config import get_config
from AI.preprocess.train import MyTrain
from AI.preprocess.test import MyTest

# change directory to the current script
os.chdir(pathlib.Path(__file__).parent)

# parse arguments
parser, train_parser, test_parser = get_config()
cfg = parser.parse_args()
if cfg.subcommand == "train":
    for performance in MyTrain(**cfg.train.train.as_dict())(
        train_parser=train_parser,
        cfg=cfg.train,
    ):
        pass

elif cfg.subcommand == "test":
    MyTest(**cfg.test.test.as_dict())(train_parser=train_parser)

elif cfg.subcommand == "upload":
    from AI.preprocess.upload import upload

    upload(
        preprocess=args.preprocess,
        model=model_name,
        data_name=args.dataset.data_name,
        owner=args.dataset.owner,
        logger=get_logger(args.log_level),
    )

elif cfg.subcommand == "inference":
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

elif cfg.subcommand == "app":
    from preprocess.app import app

    app(
        preprocess=args.preprocess,
        model_name=model_name,
        data_name=args.dataset.data_name,
        owner=args.dataset.owner,
        device=args.device,
    )

elif cfg.subcommand == "space":
    from preprocess.space import space

    space(
        preprocess=args.preprocess,
        model_name=model_name,
        data_name=args.dataset.data_name,
        owner=args.dataset.owner,
        device="cpu",
    )
