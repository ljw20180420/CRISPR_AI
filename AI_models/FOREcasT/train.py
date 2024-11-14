#!/usr/bin/env python

from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from .model import FOREcasTConfig, FOREcasTModel
from ..config import get_config, get_logger
from .load_data import data_collector

args = get_config(config_file="config_FOREcasT.ini")
logger = get_logger(args)

def train(data_name=args.data_name):
    logger.info("loading data")
    ds = load_dataset(
        path = f"{args.owner}/CRISPR_data",
        name = f"{data_name}_{FOREcasTConfig.model_type}",
        trust_remote_code = True,
        test_ratio = args.test_ratio,
        validation_ratio = args.validation_ratio,
        seed = args.seed,
        FOREcasT_MAX_DEL_SIZE = args.FOREcasT_MAX_DEL_SIZE
    )

    logger.info("initialize model")
    FOREcasTConfig.register_for_auto_class()
    FOREcasTModel.register_for_auto_class()
    FOREcasT_model = FOREcasTModel(FOREcasTConfig(
        reg_const = args.FOREcasT_reg_const,
        i1_reg_const = args.FOREcasT_i1_reg_const,
        seed = args.seed
    ))

    logger.info("train model")
    training_args = TrainingArguments(
        output_dir = args.output_dir / FOREcasTConfig.model_type / f"{data_name}_{FOREcasTConfig.model_type}",
        seed = args.seed,
        logging_strategy = "epoch",
        eval_strategy = "epoch",
        save_strategy = "epoch",
        load_best_model_at_end = True,
        remove_unused_columns = False,
        label_names = FOREcasTConfig.label_names
    )
    training_args.set_dataloader(
        train_batch_size = args.batch_size,
        eval_batch_size = args.batch_size
    )
    training_args.set_optimizer(
        name = args.optimizer,
        learning_rate = args.learning_rate
    )
    training_args.set_lr_scheduler(
        name = args.scheduler,
        num_epochs = args.num_epochs,
        warmup_ratio = args.warmup_ratio
    )
    trainer = Trainer(
        model = FOREcasT_model,
        args = training_args,
        train_dataset = ds["train"],
        eval_dataset = ds["validation"],
        data_collator = data_collector
    )
    try:
        trainer.train(resume_from_checkpoint = True)
    except ValueError:
        trainer.train()

    logger.info("save model")
    trainer.save_model()
    trainer.create_model_card()
