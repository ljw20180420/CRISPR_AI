#!/usr/bin/env python

import os
import pathlib

# change directory to the current script
os.chdir(pathlib.Path(__file__).parent)

from FOREcasT.config import get_config, get_logger

args = get_config(
    [
        "FOREcasT/config_default.ini",
        "FOREcasT/config_custom.ini",
    ]
)

if args.command == "train":
    from FOREcasT.train import train

    train(
        data_name=args.data_name,
        test_ratio=args.test_ratio,
        validation_ratio=args.validation_ratio,
        ref1len=args.ref1len,
        ref2len=args.ref2len,
        random_insert_uplimit=args.random_insert_uplimit,
        insert_uplimit=args.insert_uplimit,
        owner=args.owner,
        max_del_size=args.max_del_size,
        reg_const=args.reg_const,
        i1_reg_const=args.i1_reg_const,
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
    from FOREcasT.test import test

    test(
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
    from FOREcasT.upload import upload

    upload(
        data_name=args.data_name,
        owner=args.owner,
        logger=get_logger(args.log_level),
    )

elif args.command == "inference":
    from FOREcasT.inference import inference

    inference(
        data_name=args.data_name,
        inference_data=args.inference_data,
        inference_output=args.inference_output,
        ref1len=args.ref1len,
        ref2len=args.ref2len,
        owner=args.owner,
        batch_size=args.batch_size,
        device=args.device,
        logger=get_logger(args.log_level),
    )

elif args.command == "app":
    from FOREcasT.app import app

    app(
        data_name=args.data_name,
        ref1len=args.ref1len,
        ref2len=args.ref2len,
        owner=args.owner,
        device=args.device,
    )

elif args.command == "space":
    from FOREcasT.space import space

    space(
        data_name=args.data_name,
        ref1len=args.ref1len,
        ref2len=args.ref2len,
        owner=args.owner,
        device=args.device,
    )
