#!/usr/bin/env python

import os
import pathlib

# change directory to the current script
os.chdir(pathlib.Path(__file__).parent)

from FOREcasT.config import get_config, get_logger
from FOREcasT.load_data import pre_calculation
from FOREcasT.train import train

args = get_config(
    [
        "FOREcasT/config_default.ini",
        "FOREcasT/config_custom.ini",
    ]
)

if args.command == "train":
    train(
        data_name=args.data_name,
        test_ratio=args.test_ratio,
        validation_ratio=args.validation_ratio,
        ref1len=args.ref1len,
        ref2len=args.ref2len,
        random_insert_uplimit=args.random_insert_uplimit,
        insert_uplimit=args.insert_uplimit,
        owner=args.owner,
        reg_const=args.reg_const,
        i1_reg_const=args.i1_reg_const,
        batch_size=args.batch_size,
        pre_calculated_features=pre_calculation(args.MAX_DEL_SIZE),
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        scheduler=args.scheduler,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        output_dir=args.output_dir,
        seed=args.seed,
        logger=get_logger(args.log_level),
    )
