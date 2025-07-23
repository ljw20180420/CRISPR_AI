import numpy as np
import logging
import inspect
import sys
from typing import Literal
from transformers import PreTrainedModel
import os
import pathlib
import json
import torch
import datasets
import importlib
from .train import MyTrain
from .generator import MyGenerator
from .dataset import MyDataset
from .initializer import MyInitializer
from .optimizer import MyOptimizer
from .lr_scheduler import MyLRScheduler

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import repeat, rearrange


class MyLogger:
    def __init__(
        self,
        log_level: Literal[
            "CRITICAL", "FATAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"
        ],
    ) -> None:
        """Logger arguments.

        Args:
            log_level: The level of logging.
        """
        self.log_level = log_level
        self.logger = logging.getLogger("logger")
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(log_level)
        self.logger.addHandler(handler)


def config_dict(obj: object, skip: list[str] = []) -> dict:
    return {
        param: getattr(obj, param)
        for param in inspect.signature(obj.__init__).parameters.keys()
        if param != "self" and param not in skip
    }


def save_checkpoint(
    save_path: os.PathLike,
    my_train: MyTrain,
    performance: dict,
    my_generator: MyGenerator,
    metrics: dict,
    model: PreTrainedModel,
    my_initializer: MyInitializer,
    my_optimizer: MyOptimizer,
    my_lr_scheduler: MyLRScheduler,
    my_logger: MyLogger,
) -> None:
    save_path = pathlib.Path(os.fspath(save_path))
    os.makedirs(save_path, exist_ok=True)
    meta_data = {
        "train": config_dict(my_train),
        "performance": performance,
        "generator": config_dict(my_generator),
        "metric": {
            metric_name: config_dict(metric_cls) for metric_name, metric_cls in metrics
        },
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
    Infer the epoch from with the loweset metric (including loss).
    """
    checkpoints_path = pathlib.Path(os.fspath(checkpoints_path))
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

    my_train = MyTrain(**meta_data["train"])
    my_generator = MyGenerator(**meta_data["generator"])
    metrics = {
        metric_name: getattr(importlib.import_module(".metric"), metric_name)(
            **metric_params
        )
        for metric_name, metric_params in meta_data["metric"]
    }
    model_module = importlib.import_module(f"preprocess.{my_train.preprocess}.model")
    model = getattr(model_module, f"{my_train.model_type}Model")(
        getattr(model_module, f"{my_train.model_type}Config")(**meta_data["model"])
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
):
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


class MicroHomologyTool:
    def __init__(self) -> None:
        pass

    def reinitialize(self, ref1: str, ref2: str) -> None:
        if (
            hasattr(self, "ref1len")
            and self.ref1len == len(ref1)
            and hasattr(self, "ref2len")
            and self.ref2len == len(ref2)
        ):
            return
        self.ref1len = len(ref1)
        self.ref2len = len(ref2)
        # diag_indices example for ref2len = 3 and ref1len = 2:
        # 6 9 11   row_indices 0 0 0   col_indices 0 1 2
        # 3 7 10               1 1 1               0 1 2
        # 1 4 8                2 2 2               0 1 2
        # 0 2 5                3 3 3               0 1 2
        # diag_indices = np.ravel_multi_index(
        #     multi_index=(
        #         tensor([3, 2, 3, 1, 2, 3, 0, 1, 2, 0, 1, 0]),
        #         tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 1, 2, 2])
        #     ),
        #     dims=(4, 3),
        # )
        row_indices = repeat(
            np.arange(self.ref2len + 1), "r2 -> r2 r1", r1=self.ref1len + 1
        )
        col_indices = repeat(
            np.arange(self.ref1len + 1), "r1 -> r2 r1", r2=self.ref2len + 1
        )
        self.diag_indices = np.ravel_multi_index(
            multi_index=(
                # row index
                np.concatenate(
                    [
                        row_indices.diagonal(offset)
                        for offset in range(-self.ref2len, self.ref1len + 1)
                    ]
                ),
                # col index
                np.concatenate(
                    [
                        col_indices.diagonal(offset)
                        for offset in range(-self.ref2len, self.ref1len + 1)
                    ]
                ),
            ),
            dims=(self.ref2len + 1, self.ref1len + 1),
        )

    def get_mh(
        self, ref1: str, ref2: str, cut1: int, cut2: int, ext1: int, ext2: int
    ) -> tuple[np.ndarray]:
        assert cut1 + ext1 <= len(ref1) and ext2 <= cut2, "extend too much"
        self.reinitialize(ref1, ref2)
        mh_matrix = np.pad(
            (
                rearrange(
                    np.frombuffer(ref1[: cut1 + ext1].encode(), dtype=np.int8),
                    "r1 -> 1 r1",
                )
                == rearrange(
                    np.frombuffer(ref2[cut2 - ext2 :].encode(), dtype=np.int8),
                    "r2 -> r2 1",
                )
            ).astype(int),
            pad_width=((cut2 - ext2, 1), (0, len(ref1) - cut1 - ext1 + 1)),
        )
        rep_num = np.diff(
            np.concatenate(
                (
                    np.array([-1], dtype=int),
                    np.where(np.diff(mh_matrix.flatten()[self.diag_indices]))[0],
                    np.array([(len(ref1) + 1) * (len(ref2) + 1) - 1], dtype=int),
                )
            )
        )
        rep_val = rep_num.copy()
        rep_val[0::2] = 0
        rep_num[1::2] = rep_num[1::2] + 1
        rep_num[2::2] = rep_num[2::2] - 1
        mh_matrix = mh_matrix.flatten()
        mh_matrix[self.diag_indices] = np.repeat(rep_val, rep_num)
        cum_rep_num = rep_num.cumsum()
        mh_idx_align_ref1 = self.diag_indices[cum_rep_num[1::2] - 1]
        mh_idx_align_ref2 = self.diag_indices[cum_rep_num[0:-1:2]]
        mh_rep_num = rep_num[1::2]
        return mh_matrix, mh_idx_align_ref1, mh_idx_align_ref2, mh_rep_num

    def correct_observation(
        self, observations: np.ndarray, mh_matrix: np.ndarray, mh_rep_num: np.ndarray
    ) -> np.ndarray:
        mh_mask = (mh_matrix > 0)[self.diag_indices]
        for i, observation in enumerate(observations):
            observation = observation.flatten()
            counts = np.zeros(len(mh_rep_num), dtype=int)
            np.add.at(
                counts,
                np.repeat(np.arange(len(mh_rep_num)), mh_rep_num),
                observation[self.diag_indices][mh_mask],
            )
            observation[self.diag_indices[mh_mask]] = np.repeat(counts, mh_rep_num)
            observations[i] = observation.reshape(self.ref2len + 1, self.ref1len + 1)

        return observations

    def get_observation(
        self, example: dict, mh_matrix: np.ndarray, mh_rep_num: np.ndarray
    ) -> np.ndarray:
        mh_idx = mh_matrix.nonzero()
        mh_val = mh_matrix[mh_idx]
        # construct observations
        observations = np.zeros(
            (example["random_insert_uplimit"] + 2)
            * (len(example["ref2"]) + 1)
            * (len(example["ref1"]) + 1),
            dtype=np.float32,
        )
        observations[example["ob_idx"]] = np.array(example["ob_val"], dtype=np.float32)
        observations = observations.reshape(
            example["random_insert_uplimit"] + 2,
            len(example["ref2"]) + 1,
            len(example["ref1"]) + 1,
        )
        # correct observations
        observations = self.correct_observation(observations, mh_matrix, mh_rep_num)
        # cumulate observations for all random insertion size
        observation = observations.sum(axis=0).flatten()
        # distribute count to all positions in single micro-homology diagonal
        observation[mh_idx] = observation[mh_idx] / (mh_val + 1)
        observation = observation.reshape(
            len(example["ref2"]) + 1, len(example["ref1"]) + 1
        )
        return observation
