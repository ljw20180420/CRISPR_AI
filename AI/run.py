#!/usr/bin/env python

import os
import pathlib
import importlib
import inspect

# change directory to the current script
os.chdir(pathlib.Path(__file__).parent)

from preprocess.config import get_config, get_logger

args = get_config()
if not args.preprocess:
    raise ReferenceError("No preprocess")
model_name = args[args.preprocess].model_name
if not model_name:
    raise ReferenceError("No model name")

if args.train:
    from preprocess.train import train

    model_parameters = {
        param: value
        for param, value in getattr(
            getattr(
                args,
                args.preprocess,
            ),
            model_name,
        ).items()
        if param not in ["config", "__default_config__"]
    }
    model_parameters["seed"] = args.seed

    signature = inspect.signature(
        getattr(
            importlib.import_module(f"preprocess.{args.preprocess}.load_data"),
            "DataCollator",
        ).__init__
    )
    data_collator_parameters = {
        param: (model_parameters[param] if param != "output_label" else True)
        for param in signature.parameters.keys()
        if param not in ["self", "args", "kwargs"]
    }

    train(
        preprocess=args.preprocess,
        model_name=model_name,
        model_parameters=model_parameters,
        data_collator_parameters=data_collator_parameters,
        data_name=args.dataset.data_name,
        test_ratio=args.dataset.test_ratio,
        validation_ratio=args.dataset.validation_ratio,
        random_insert_uplimit=args.dataset.random_insert_uplimit,
        insert_uplimit=args.dataset.insert_uplimit,
        owner=args.dataset.owner,
        optimizer=args.optimizer.optimizer,
        learning_rate=args.optimizer.learning_rate,
        scheduler=args.scheduler.scheduler,
        num_epochs=args.scheduler.num_epochs,
        warmup_ratio=args.scheduler.warmup_ratio,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        logger=get_logger(args.log_level),
    )

elif args.test:
    from preprocess.test import test

    test(
        preprocess=args.preprocess,
        model_name=model_name,
        data_name=args.dataset.data_name,
        test_ratio=args.dataset.test_ratio,
        validation_ratio=args.dataset.validation_ratio,
        random_insert_uplimit=args.dataset.random_insert_uplimit,
        insert_uplimit=args.dataset.insert_uplimit,
        owner=args.dataset.owner,
        metric_ext1_up=args.metric.metric_ext1_up,
        metric_ext1_down=args.metric.metric_ext1_down,
        metric_ext2_up=args.metric.metric_ext2_up,
        metric_ext2_down=args.metric.metric_ext2_down,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        logger=get_logger(args.log_level),
    )

elif args.upload:
    from preprocess.upload import upload

    upload(
        preprocess=args.preprocess,
        model=model_name,
        data_name=args.dataset.data_name,
        owner=args.dataset.owner,
        logger=get_logger(args.log_level),
    )

elif args.inference:
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

elif args.app:
    from preprocess.app import app

    app(
        preprocess=args.preprocess,
        model_name=model_name,
        data_name=args.dataset.data_name,
        owner=args.dataset.owner,
        device=args.device,
    )

elif args.space:
    from preprocess.space import space

    space(
        preprocess=args.preprocess,
        model_name=model_name,
        data_name=args.dataset.data_name,
        owner=args.dataset.owner,
        device="cpu",
    )
