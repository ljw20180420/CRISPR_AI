#!/usr/bin/env python

from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from .model import FOREcasTConfig, FOREcasTModel
from ..config import args, logger
from .load_data import data_collector

def train():
    logger.info("loading data")
    ds = load_dataset(
        path = args.data_path,
        name = f"{args.data_name}_{FOREcasTConfig.model_type}",
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
        output_dir = args.output_dir / FOREcasTConfig.model_type / f"{args.data_name}_{FOREcasTConfig.model_type}",
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

    logger.info("push model")
    trainer.save_model()
    trainer.create_model_card()