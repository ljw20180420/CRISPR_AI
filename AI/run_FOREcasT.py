#!/usr/bin/env python

import os
import pathlib

# change directory to the current script
os.chdir(pathlib.Path(__file__).parent)

from preprocess.FOREcasT.config import get_config, get_logger

args = get_config(
    [
        "preprocess/FOREcasT/config_default.ini",
        "preprocess/FOREcasT/config_custom.ini",
    ]
)

if args.command == "train":
    from preprocess.common.train import train

    if args.model_name == "FOREcasT":
        model_parameters = {
            "max_del_size": args.max_del_size,
            "reg_const": args.reg_const,
            "i1_reg_const": args.i1_reg_const,
        }
        data_collator_parameters = {
            "max_del_size": args.max_del_size,
            "output_count": True,
        }
    train(
        preprocess="FOREcasT",
        model_name=args.model_name,
        model_parameters=model_parameters,
        data_collator_parameters=data_collator_parameters,
        data_name=args.data_name,
        test_ratio=args.test_ratio,
        validation_ratio=args.validation_ratio,
        ref1len=args.ref1len,
        ref2len=args.ref2len,
        random_insert_uplimit=args.random_insert_uplimit,
        insert_uplimit=args.insert_uplimit,
        owner=args.owner,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        scheduler=args.scheduler,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        logger=get_logger(args.log_level),
    )


elif args.command == "test":
    from preprocess.common.test import test

    test(
        preprocess="FOREcasT",
        model_name=args.model_name,
        data_name=args.data_name,
        test_ratio=args.test_ratio,
        validation_ratio=args.validation_ratio,
        ref1len=args.ref1len,
        ref2len=args.ref2len,
        random_insert_uplimit=args.random_insert_uplimit,
        insert_uplimit=args.insert_uplimit,
        owner=args.owner,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        logger=get_logger(args.log_level),
    )

elif args.command == "upload":
    from preprocess.common.upload import upload

    upload(
        preprocess="FOREcasT",
        model=args.model_name,
        data_name=args.data_name,
        owner=args.owner,
        logger=get_logger(args.log_level),
    )

elif args.command == "inference":
    from preprocess.common.inference import inference

    inference(
        preprocess="FOREcasT",
        model_name=args.model_name,
        data_name=args.data_name,
        inference_data=args.inference_data,
        inference_output=args.inference_output,
        owner=args.owner,
        batch_size=args.batch_size,
        device=args.device,
        logger=get_logger(args.log_level),
    )

elif args.command == "app":
    from preprocess.common.app import app

    app(
        preprocess="FOREcasT",
        model_name=args.model_name,
        data_name=args.data_name,
        owner=args.owner,
        device=args.device,
    )

elif args.command == "space":
    from preprocess.common.space import space

    space(
        preprocess="FOREcasT",
        model_name=args.model_name,
        data_name=args.data_name,
        owner=args.owner,
        device=args.device,
    )
