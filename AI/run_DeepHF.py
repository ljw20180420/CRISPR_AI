#!/usr/bin/env python

import os
import pathlib

# change directory to the current script
os.chdir(pathlib.Path(__file__).parent)

from preprocess.DeepHF.config import get_config, get_logger

args = get_config(
    [
        "preprocess/DeepHF/config_default.ini",
        "preprocess/DeepHF/config_custom.ini",
    ]
)

if args.command == "train":
    from preprocess.DeepHF.train import train

    if args.model_name == "DeepHF":
        model_paramters = {
            "seq_length": args.seq_length,
            "em_drop": args.em_drop,
            "fc_drop": args.fc_drop,
            "initializer": args.initializer,
            "em_dim": args.em_dim,
            "rnn_units": args.rnn_units,
            "fc_num_hidden_layers": args.fc_num_hidden_layers,
            "fc_num_units": args.fc_num_units,
            "fc_activation": args.fc_activation,
            "ext1_up": args.ext1_up,
            "ext1_down": args.ext1_down,
            "ext2_up": args.ext2_up,
            "ext2_down": args.ext2_down,
        }
        train(
            model_name=args.model_name,
            model_paramters=model_paramters,
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
    if args.model == "DeepHF":
        from DeepHF.test import test_DeepHF

        test_DeepHF(
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
