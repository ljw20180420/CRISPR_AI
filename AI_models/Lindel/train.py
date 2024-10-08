#!/usr/bin/env python

from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from .model import LindelConfig, LindelModel
from ..config import args, logger
from .load_data import data_collector

def train():
    logger.info("loading data")
    ds = load_dataset(
        path = args.data_path,
        name = f"{args.data_name}_{LindelConfig.model_type}",
        trust_remote_code = True,
        test_ratio = args.test_ratio,
        validation_ratio = args.validation_ratio,
        seed = args.seed,
        Lindel_dlen = args.Lindel_dlen,
        Lindel_mh_len = args.Lindel_mh_len
    )

    logger.info("initialize model")
    LindelConfig.register_for_auto_class()
    LindelModel.register_for_auto_class()
    Lindel_models = {
        model: LindelModel(LindelConfig(
            dlen=args.Lindel_dlen,
            mh_len=args.Lindel_mh_len,
            model=model,
            reg_mode=args.Lindel_reg_mode,
            reg_const=args.Lindel_reg_const,
            seed=args.seed
        ))
        for model in ["indel", "ins", "del"]
    }

    logger.info("train model")
    trainers = dict()
    for model in ["indel", "ins", "del"]:
        training_args = TrainingArguments(
            output_dir = args.output_dir / LindelConfig.model_type / f"{args.data_name}_{LindelConfig.model_type}_{model}",
            seed = args.seed,
            logging_strategy = "epoch",
            eval_strategy = "epoch",
            save_strategy = "epoch",
            load_best_model_at_end = True,
            remove_unused_columns = False,
            label_names = LindelConfig.label_names
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
        trainers[model] = Trainer(
            model = Lindel_models[model],
            args = training_args,
            train_dataset = ds["train"],
            eval_dataset = ds["validation"],
            # bind model to a local variable of lambda function to avoid access the non-local model when evaluate lambda function
            data_collator = lambda examples, model=model: data_collector(examples, Lindel_mh_len=args.Lindel_mh_len, model=model)
        )

    for model in ["indel", "ins", "del"]:
        try:
            trainers[model].train(resume_from_checkpoint = True)
        except ValueError:
            trainers[model].train()

    logger.info("push model")
    for model in ["indel", "ins", "del"]:
        trainers[model].save_model()
        trainers[model].create_model_card()
