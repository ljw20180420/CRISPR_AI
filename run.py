#!/usr/bin/env python

import os
import pathlib
from AI.preprocess.config import get_config
from AI.preprocess.train import MyTrain
from AI.preprocess.test import MyTest

# change directory to the current script
os.chdir(pathlib.Path(__file__).parent)

# parse arguments
meta_data = get_config()

if meta_data["command"] == "train":
    my_train = MyTrain(**meta_data["train"])
    my_train(meta_data)

elif meta_data["command"] == "test":
    my_test = MyTest(**meta_data["test"])
    my_test()

elif meta_data["command"] == "upload":
    from AI.preprocess.upload import upload

    upload(
        preprocess=args.preprocess,
        model=model_name,
        data_name=args.dataset.data_name,
        owner=args.dataset.owner,
        logger=get_logger(args.log_level),
    )

elif meta_data["command"] == "inference":
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

elif meta_data["command"] == "app":
    from preprocess.app import app

    app(
        preprocess=args.preprocess,
        model_name=model_name,
        data_name=args.dataset.data_name,
        owner=args.dataset.owner,
        device=args.device,
    )

elif meta_data["command"] == "space":
    from preprocess.space import space

    space(
        preprocess=args.preprocess,
        model_name=model_name,
        data_name=args.dataset.data_name,
        owner=args.dataset.owner,
        device="cpu",
    )
