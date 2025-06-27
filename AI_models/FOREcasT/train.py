#!/usr/bin/env python

import pathlib
import logging
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from .model import FOREcasTConfig, FOREcasTModel
from .load_data import data_collator


def train(
    data_name: str,
    test_ratio: float,
    validation_ratio: float,
    ref1len: int,
    ref2len: int,
    random_insert_uplimit: int,
    insert_uplimit: int,
    owner: str,
    reg_const: float,
    i1_reg_const: float,
    batch_size: int,
    pre_calculated_features: tuple,
    optimizer: str,
    learning_rate: float,
    scheduler: str,
    num_epochs: float,
    warmup_ratio: float,
    output_dir: pathlib.Path,
    seed: int,
    logger: logging.Logger,
) -> None:
    logger.info("loading data")
    ds = load_dataset(
        path="%s/CRISPR_data" % owner,
        name=data_name,
        trust_remote_code=True,
        test_ratio=test_ratio,
        validation_ratio=validation_ratio,
        seed=seed,
        ref1len=ref1len,
        ref2len=ref2len,
        random_insert_uplimit=random_insert_uplimit,
        insert_uplimit=insert_uplimit,
    )

    logger.info("initialize model")
    FOREcasTConfig.register_for_auto_class()
    FOREcasTModel.register_for_auto_class()
    FOREcasT_model = FOREcasTModel(
        FOREcasTConfig(
            reg_const=reg_const,
            i1_reg_const=i1_reg_const,
            seed=seed,
        )
    )

    logger.info("train model")
    training_args = TrainingArguments(
        output_dir=output_dir / FOREcasTConfig.model_type / data_name,
        seed=seed,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=False,
        label_names=FOREcasTConfig.label_names,
    )
    training_args.set_dataloader(
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
    )
    training_args.set_optimizer(
        name=optimizer,
        learning_rate=learning_rate,
    )
    training_args.set_lr_scheduler(
        name=scheduler,
        num_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
    )
    trainer = Trainer(
        model=FOREcasT_model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=lambda examples, pre_calculated_features=pre_calculated_features: data_collator(
            examples, pre_calculated_features
        ),
    )
    try:
        trainer.train(resume_from_checkpoint=True)
    except ValueError:
        trainer.train()

    logger.info("save model")
    trainer.save_model()
    trainer.create_model_card()
