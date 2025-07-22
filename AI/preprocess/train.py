import pathlib
import logging
from datasets import load_dataset
import importlib
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)


class SingularGradTrainerCallback(TrainerCallback):
    def on_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        for p in model.parameters():
            if p.grad is not None and (p.grad.isnan().any() or p.grad.isinf().any()):
                control.should_training_stop = True
                return


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
    trial_name: str,
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

    logger.info("train model")
    model_module = importlib.import_module(f"preprocess.{preprocess}.model")
    getattr(model_module, f"{model_name}Config").register_for_auto_class()
    config = getattr(model_module, f"{model_name}Config")(
        **model_parameters,
    )
    if hasattr(model_module, f"{model_name}Model"):
        logger.info("train core model")
        getattr(model_module, f"{model_name}Model").register_for_auto_class()
        model = getattr(model_module, f"{model_name}Model")(config)
        assert model_name == model.config.model_type, "model name is not consistent"

        logger.info("set training arguments")
        training_args = TrainingArguments(
            output_dir=output_dir
            / preprocess
            / model.config.model_type
            / data_name
            / trial_name
            / "core_model",
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
            callbacks=[SingularGradTrainerCallback],
        )
        try:
            trainer.train(resume_from_checkpoint=True)
        except ValueError:
            trainer.train()

        logger.info("save model")
        trainer.save_model()
        trainer.create_model_card()

    if hasattr(model_module, f"{model_name}Auxilary"):
        auxilary = getattr(model_module, f"{model_name}Auxilary")(config)
        auxilary.train_auxilary(
            preprocess=preprocess,
            model_name=model_name,
            data_collator=data_collator,  # pass data collator to inDelphi_insert.train so that model.py does not depend on load_data.py, thereby does not depend on utils.py
            data_name=data_name,
            ds=ds,
            trial_name=trial_name,
            output_dir=output_dir,
            batch_size=batch_size,
            device=device,
            logger=logger,
        )
