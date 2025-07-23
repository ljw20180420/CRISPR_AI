#!/usr/bin/env python

import os
import pathlib
import importlib
import inspect

# change directory to the current script
os.chdir(pathlib.Path(__file__).parent)

# parse arguments
from AI.preprocess.config import get_config

args = get_config()

breakpoint()

## generator
from preprocess.generator import MyGenerator

my_generator = MyGenerator(seed=args.generator.seed)

## dataset
from preprocess.dataset import MyDataset

my_dataset = MyDataset(
    name=args.dataset.name,
    test_ratio=args.dataset.test_ratio,
    validation_ratio=args.dataset.validation_ratio,
    random_insert_uplimit=args.dataset.random_insert_uplimit,
    insert_uplimit=args.dataset.insert_uplimit,
    owner=args.dataset.owner,
)(my_generator=my_generator)

## metric
metric_module = importlib.import_module(f"preprocess.metric")
metrics = {
    metric: getattr(metric_module, metric)(**dict(params.items()))
    for metric, params in vars(args.metric).items()
}


# commands
if args.train:
    ## model
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
    model_module = importlib.import_module(f"preprocess.{args.preprocess}.model")
    getattr(model_module, f"{model_name}Config").register_for_auto_class()
    config = getattr(model_module, f"{model_name}Config")(
        **model_parameters,
    )
    getattr(model_module, f"{model_name}Model").register_for_auto_class()
    model = getattr(model_module, f"{model_name}Model")(config).to(args.device)
    assert model_name == model.config.model_type, "model name is not consistent"
    ## optimizer
    from preprocess.optimizer import MyOptimizer

    my_optimizer = MyOptimizer(
        name=args.optimizer.optimizer,
        learning_rate=args.optimizer.learning_rate,
        weight_decay=args.optimizer.weight_decay,
    )(model)

    ## scheduler
    from preprocess.lr_scheduler import MyLRScheduler

    my_lr_scheduler = MyLRScheduler(
        name=args.lr_scheduler.name, warmup_ratio=args.lr_scheduler.warmup_ratio
    )(
        dataset=my_dataset.dataset,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        optimizer=my_optimizer.optimizer,
    )

    ## data collator
    DataCollator = getattr(
        importlib.import_module(f"preprocess.{args.preprocess}.load_data"),
        "DataCollator",
    )
    signature = inspect.signature(DataCollator.__init__)
    data_collator_parameters = {
        param: model_parameters[param]
        for param in signature.parameters.keys()
        if param != "self"
    }
    data_collator = DataCollator(**data_collator_parameters)
    assert args.preprocess == data_collator.preprocess, "preprocess is not consistent"

    from preprocess.train import train

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
