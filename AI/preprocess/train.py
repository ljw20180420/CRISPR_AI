import pathlib
import logging
from datasets import load_dataset
import importlib
from transformers import Trainer, TrainingArguments


def train(
    preprocess: str,
    model_name: str,
    model_parameters: dict,
    data_collator_parameters: dict,
    data_name: str,
    test_ratio: float,
    validation_ratio: float,
    random_insert_uplimit: int,
    insert_uplimit: int,
    owner: str,
    optimizer: str,
    learning_rate: float,
    scheduler: str,
    num_epochs: float,
    warmup_ratio: float,
    output_dir: pathlib.Path,
    batch_size: int,
    seed: int,
    device: str,
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
        random_insert_uplimit=random_insert_uplimit,
        insert_uplimit=insert_uplimit,
    )

    logger.info("construct data collator")
    data_collator = importlib.import_module(
        f"preprocess.{preprocess}.load_data",
    ).DataCollator(**data_collator_parameters)

    logger.info("initialize model")
    model_module = importlib.import_module(f"preprocess.{preprocess}.model")
    getattr(model_module, f"{model_name}Model").register_for_auto_class()
    getattr(model_module, f"{model_name}Config").register_for_auto_class()
    model = getattr(model_module, f"{model_name}Model")(
        getattr(model_module, f"{model_name}Config")(
            **model_parameters,
        )
    )
    assert model_name == model.config.model_type, "model name is not consistent"

    logger.info("train model")
    training_args = TrainingArguments(
        output_dir=output_dir / preprocess / model.config.model_type / data_name,
        seed=seed,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        use_cpu=True if device == "cpu" else False,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        label_names=model.config.label_names,
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
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
    )
    try:
        trainer.train(resume_from_checkpoint=True)
    except ValueError:
        trainer.train()

    logger.info("save model")
    trainer.save_model()
    trainer.create_model_card()
