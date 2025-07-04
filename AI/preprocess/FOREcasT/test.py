import logging
import pathlib
import torch


@torch.no_grad()
def test(
    model_name: str,
    data_name: str,
    test_ratio: float,
    validation_ratio: float,
    ref1len: int,
    ref2len: int,
    random_insert_uplimit: int,
    insert_uplimit: int,
    owner: str,
    batch_size: int,
    output_dir: pathlib.Path,
    device: str,
    seed: int,
    logger: logging.Logger,
) -> None:
    logger.info("load model")
    if model_name == "FOREcasT":
        from .model import FOREcasTModel

        model = FOREcasTModel.from_pretrained(
            output_dir / "FOREcasT" / model_name / data_name
        )
    assert model_name == model.config.model_type, "model name is not consistent"
    model.__module__ = "model"

    logger.info("setup pipeline")
    if model_name == "FOREcasT":
        from .pipeline import FOREcasTPipeline

        pipe = FOREcasTPipeline(model)

    logger.info("process common test")
    from ..common.test import test

    test(
        preprocess="FOREcasT",
        pipe=pipe,
        data_name=data_name,
        test_ratio=test_ratio,
        validation_ratio=validation_ratio,
        ref1len=ref1len,
        ref2len=ref2len,
        random_insert_uplimit=random_insert_uplimit,
        insert_uplimit=insert_uplimit,
        owner=owner,
        batch_size=batch_size,
        output_dir=output_dir,
        device=device,
        seed=seed,
        logger=logger,
    )
