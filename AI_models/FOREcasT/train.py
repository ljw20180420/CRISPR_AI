#!/usr/bin/env python

import pathlib
import logging
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from .model import FOREcasTConfig, FOREcasTModel
from .load_data import data_collator


def train_FOREcasT(
    data_name: str,
    test_ratio: float,
    validation_ratio: float,
    ref1len: int,
    ref2len: int,
    random_insert_uplimit: int,
    insert_uplimit: int,
    owner: str,
    max_del_size: int,
    reg_const: float,
    i1_reg_const: float,
    batch_size: int,
    optimizer: str,
    learning_rate: float,
    scheduler: str,
    num_epochs: float,
    warmup_ratio: float,
    output_dir: pathlib.Path,
    device: str,
    seed: int,
    logger: logging.Logger,
) -> None:
    logger.info("loading data")
    ds = load_dataset(
        path=f"{owner}/CRISPR_data",
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
            max_del_size=max_del_size,
            reg_const=reg_const,
            i1_reg_const=i1_reg_const,
            seed=seed,
        )
    )

    logger.info("train model")
    training_args = TrainingArguments(
        output_dir=output_dir / "FOREcasT/FOREcasT" / data_name,
        seed=seed,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        use_cpu=True if device == "cpu" else False,
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
        data_collator=lambda examples, pre_calculated_features=FOREcasT_model.pre_calculated_features, output_count=True: data_collator(
            examples, pre_calculated_features, output_count
        ),
    )
    try:
        trainer.train(resume_from_checkpoint=True)
    except ValueError:
        trainer.train()

    logger.info("save model")
    trainer.save_model()
    trainer.create_model_card()
