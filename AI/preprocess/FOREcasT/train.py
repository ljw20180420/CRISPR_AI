import pathlib
import logging


def train(
    model_name: str,
    model_parameters: dict,
    data_name: str,
    test_ratio: float,
    validation_ratio: float,
    ref1len: int,
    ref2len: int,
    random_insert_uplimit: int,
    insert_uplimit: int,
    owner: str,
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
    logger.info("initialize model")
    if model_name == "FOREcasT":
        from .model import FOREcasTConfig, FOREcasTModel

        FOREcasTConfig.register_for_auto_class()
        FOREcasTModel.register_for_auto_class()
        model = FOREcasTModel(
            FOREcasTConfig(
                **model_parameters,
                seed=seed,
            )
        )
    assert model_name == model.config.model_type, "model name is not consistent"

    logger.info("construct data collator")
    from .load_data import data_collator

    data_collator_bind = lambda examples, pre_calculated_features=model.pre_calculated_features: data_collator(
        examples, pre_calculated_features, output_count=True
    )

    logger.info("train model")
    from ..common.train import train

    train(
        preprocess="FOREcasT",
        model=model,
        data_collator=data_collator_bind,
        data_name=data_name,
        test_ratio=test_ratio,
        validation_ratio=validation_ratio,
        ref1len=ref1len,
        ref2len=ref2len,
        random_insert_uplimit=random_insert_uplimit,
        insert_uplimit=insert_uplimit,
        owner=owner,
        batch_size=batch_size,
        optimizer=optimizer,
        learning_rate=learning_rate,
        scheduler=scheduler,
        num_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        output_dir=output_dir,
        device=device,
        seed=seed,
        logger=logger,
    )
