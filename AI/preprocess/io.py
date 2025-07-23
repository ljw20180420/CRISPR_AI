import numpy as np
import inspect
from transformers import PreTrainedModel
import os
import pathlib
import json
import torch
import datasets
import importlib
import re
from .train import MyTrain
from .generator import MyGenerator
from .dataset import MyDataset
from .initializer import MyInitializer
from .optimizer import MyOptimizer
from .lr_scheduler import MyLRScheduler
from .utils import MyLogger


def config_dict(obj: object, skip: list[str] = []) -> dict:
    return {
        param: getattr(obj, param)
        for param in inspect.signature(obj.__init__).parameters.keys()
        if param != "self" and param not in skip
    }


def save_checkpoint(
    save_path: os.PathLike,
    performance: dict,
    my_generator: MyGenerator,
    metrics: dict,
    preprocess: str,
    model_type: str,
    model: PreTrainedModel,
    my_initializer: MyInitializer,
    my_optimizer: MyOptimizer,
    my_lr_scheduler: MyLRScheduler,
    my_logger: MyLogger,
) -> None:
    save_path = pathlib.Path(os.fspath(save_path))
    os.makedirs(save_path, exist_ok=True)
    meta_data = {
        "performance": performance,
        "generator": config_dict(my_generator),
        "metric": {
            metric_name: config_dict(metric_cls) for metric_name, metric_cls in metrics
        },
        "preprocess": preprocess,
        "model_type": model_type,
        "model": config_dict(
            model.config,
            skip=["**kwargs"],
        ),
        "initializer": config_dict(my_initializer),
        "optimizer": config_dict(
            my_optimizer,
            skip=["model"],
        ),
        "lr_scheduler": config_dict(
            my_lr_scheduler,
            skip=["my_optimizer"],
        ),
        "my_logger": config_dict(my_logger),
    }

    with open(save_path / "meta_data.json", "w") as fd:
        json.dump(meta_data, fd, indent=4)

    check_point = {
        "generator": my_generator.state_dict(),
        "model": model.state_dict(),
        "optimizer": my_optimizer.optimizer.state_dict(),
        "scheduler": my_lr_scheduler.lr_scheduler.state_dict(),
    }

    torch.save(
        check_point,
        save_path / "checkpoint.pt",
    )


def target_to_epoch(checkpoints_path: os.PathLike, target: str) -> int:
    """
    Infer the epoch from either the last checkpoint or the loweset metric (including loss).
    """
    checkpoints_path = pathlib.Path(os.fspath(checkpoints_path))
    check_epochs = [
        check_epoch
        for check_epoch in os.listdir(checkpoints_path)
        if re.search(r"^checkpoint-(\d+)$", check_epoch)
    ]
    assert len(check_epochs) > 0, "no checkpoint found"
    if target == "resume":
        return max(
            [
                int(re.search(r"^checkpoint-(\d+)$", check_epoch).group(1))
                for check_epoch in check_epochs
            ]
        )

    metric_value_min = np.inf
    for check_epoch in os.listdir(checkpoints_path):
        with open(checkpoints_path / check_epoch / "meta_data.json", "r") as fd:
            meta_data = json.load(fd)
        if target == "loss":
            metric_value = (
                meta_data["performance"]["eval"]["loss"]
                / meta_data["performance"]["eval"]["loss_num"]
            )
        else:
            metric_value = (
                meta_data["performance"]["eval"][target]["loss"]
                / meta_data["performance"]["eval"][target]["loss_num"]
            )
        if metric_value < metric_value_min:
            metric_value_min = metric_value
            epoch = int(check_epoch.split("-")[1])

    return epoch


def load_checkpoint(load_path: os.PathLike) -> tuple:
    load_path = pathlib.Path(os.fspath(load_path))

    # Load meta data and initialize components
    with open(load_path / "meta_data.json", "r") as fd:
        meta_data = json.load(fd)

    my_generator = MyGenerator(**meta_data["generator"])
    metrics = {
        metric_name: getattr(importlib.import_module(".metric"), metric_name)(
            **metric_params
        )
        for metric_name, metric_params in meta_data["metric"]
    }
    model_module = importlib.import_module(
        f'preprocess.{meta_data["preprocess"]}.model'
    )
    model = getattr(model_module, f'{meta_data["model_type"]}Model')(
        getattr(model_module, f'{meta_data["model_type"]}Config')(**meta_data["model"])
    )
    my_initializer = MyInitializer(**meta_data["initializer"])
    my_optimizer = MyOptimizer(
        **meta_data["optimizer"],
        model=model,
    )
    my_lr_scheduler = MyLRScheduler(
        **meta_data["lr_scheduler"],
        my_optimizer=my_optimizer,
    )
    my_logger = MyLogger(**meta_data["logger"])

    # Load checkpoint.
    checkpoint = torch.load(load_path / "checkpoint.pt")
    my_generator.load_state_dict(checkpoint["generator"])
    model.load_state_dict(checkpoint["model"])
    my_optimizer.optimizer.load_state_dict(checkpoint["optimizer"])
    my_lr_scheduler.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    return (
        my_generator,
        metrics,
        model,
        my_initializer,
        my_optimizer,
        my_lr_scheduler,
        my_logger,
    )


def save_my_dataset(
    save_path: os.PathLike,
    my_dataset: MyDataset,
) -> None:
    save_path = pathlib.Path(os.fspath(save_path))
    with open(save_path / "dataset.json", "w") as fd:
        json.dump(config_dict(my_dataset, skip=["my_generator"]), fd, indent=4)

    my_dataset.dataset.save_to_disk(dataset_dict_path=save_path / "dataset")


def load_my_dataset(
    load_path: os.PathLike,
) -> MyDataset:
    load_path = pathlib.Path(os.fspath(load_path))
    with open(load_path / "dataset.json", "r") as fd:
        my_dataset_params = json.load(
            config_dict(my_dataset, skip=["my_generator"]), fd, indent=4
        )
    my_dataset = MyDataset(**my_dataset_params, my_generator=None)
    my_dataset.dataset = datasets.load_from_disk(load_path / "dataset")

    return my_dataset
