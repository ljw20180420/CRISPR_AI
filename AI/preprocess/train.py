import torch
from torch import nn
import numpy as np
import os
import pathlib
from torch.utils.data import DataLoader
import json
from typing import Literal, Callable
import datasets
import inspect
from transformers.trainer_pt_utils import get_parameter_names
from tqdm import tqdm
from .model import get_model
from .dataset import get_dataset
from .metric import get_metrics
from .utils import get_logger, MyGenerator, target_to_epoch


class MyTrain:
    def __init__(
        self,
        output_dir: os.PathLike,
        trial_name: str,
        batch_size: int,
        num_epochs: int,
        clip_value: float,
        accumulate_steps: int,
        device: Literal["cpu", "cuda"],
        resume_from_checkpoint: bool,
    ):
        """Train arguments.

        Args:
            output_dir: Output directory.
            trial_name: name of the training trial
            batch_size: Batch size.
            num_epochs: Total number of training epochs to perform.
            clip_value: clip the norm of gradients.
            accumulate_steps: Accumulate gradients for these steps before update parameters.
            device: Device.
            resume_from_checkpoint: Resume from checkpoint.
        """
        self.output_dir = pathlib.Path(os.fspath(output_dir))
        self.trial_name = trial_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.clip_value = clip_value
        self.accumulate_steps = accumulate_steps
        self.device = device
        self.resume_from_checkpoint = resume_from_checkpoint

    def get_initializer(
        self,
        name: Literal[
            "uniform_",
            "normal_",
            "xavier_uniform_",
            "xavier_normal_",
            "kaiming_uniform_",
            "kaiming_normal_",
            "trunc_normal_",
        ],
    ) -> Callable:
        """Initializer arguments.

        Args:
            name: Name of the intialization method for model weights.
        """
        generator = self.my_generator.get_torch_generator_by_device(self.device)
        if name == "uniform_":
            return lambda tensor, generator=generator: nn.init.uniform_(
                tensor=tensor, a=-1.0, b=1.0, generator=generator
            )
        return lambda tensor, generator=generator: getattr(nn.init, name)(
            tensor=tensor, generator=generator
        )

    def get_optimizer(
        self,
        name: Literal[
            "Adadelta",
            "Adafactor",
            "Adagrad",
            "Adam",
            "AdamW",
            "SparseAdam",
            "Adamax",
            "ASGD",
            "LBFGS",
            "NAdam",
            "RAdam",
            "RMSprop",
            "Rprop",
            "SGD",
        ],
        learning_rate: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        """Parameters of optimizer.

        Args:
            name: Name of optimizer.
            learning_rate: Learn rate of the optimizer.
            weight_decay: The l2 regularization coefficient.
        """
        decay_parameters = get_parameter_names(
            model=self.model,
            forbidden_layer_types=[nn.LayerNorm],
            forbidden_layer_names=[
                r"bias",
                r"layernorm",
                r"rmsnorm",
                r"(?:^|\.)norm(?:$|\.)",
                r"_norm(?:$|\.)",
            ],
        )
        params = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        return getattr(torch.optim, name)(
            params=params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def get_lr_scheduler(
        self,
        name: Literal[
            "ConstantLR",
            "CosineAnnealingWarmRestarts",
            "ReduceLROnPlateau",
            "LRScheduler",
        ],
        warmup_epochs: int,
        period_epochs: int,
    ) -> torch.optim.lr_scheduler.SequentialLR:
        """Parameters for learning rate scheduler.

        Args:
            name: The scheduler type to use.
            warmup_epochs: Epochs used for a linear warmup from 0.1 to 1.0 factor of initial learning rate.
            period_epochs: The period to reset the learning rate for period scheduler.
        """
        warm_up_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        if name == "ConstantLR":
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer=self.optimizer,
                schedulers=[
                    warm_up_scheduler,
                    torch.optim.lr_scheduler.ConstantLR(
                        optimizer=self.optimizer,
                        factor=1,
                        total_iters=0,
                    ),
                ],
                milestones=[warmup_epochs],
            )
        if name == "CosineAnnealingWarmRestarts":
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer=self.optimizer,
                schedulers=[
                    warm_up_scheduler,
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer=self.optimizer,
                        T_0=period_epochs,
                        eta_min=self.optimizer.param_groups[0]["initial_lr"] * 0.1,
                    ),
                ],
                milestones=[warmup_epochs],
            )

        return torch.optim.lr_scheduler.SequentialLR(
            optimizer=self.optimizer,
            schedulers=[
                warm_up_scheduler,
                getattr(torch.optim.lr_scheduler, name)(optimizer=self.optimizer),
            ],
            milestones=[warmup_epochs],
        )

    def __call__(
        self,
        meta_data: dict,
    ) -> None:
        logger = get_logger(**meta_data["logger"])

        logger.info("initialize components")
        model_path = (
            self.output_dir
            / meta_data["model"]["preprocess"]
            / meta_data["model"]["model_type"]
            / meta_data["dataset"]["name"]
            / self.trial_name
        )
        last_epoch = -1
        if self.resume_from_checkpoint:
            last_epoch = target_to_epoch(
                checkpoints_path=model_path / "checkpoints", target="resume"
            )
            if last_epoch >= 0:
                with open(
                    model_path
                    / "checkpoints"
                    / f"checkpoint-{last_epoch}"
                    / "meta_data.json",
                    "r",
                ) as fd:
                    meta_data = json.load(fd)

        self.my_generator = MyGenerator(**meta_data["generator"])
        self.metrics = get_metrics(**meta_data["metric"], meta_data=meta_data)
        self.model = get_model(**meta_data["model"], meta_data=meta_data)
        self.initializer = self.get_initializer(**meta_data["initializer"])
        self.optimizer = self.get_optimizer(**meta_data["optimizer"])
        self.lr_scheduler = self.get_lr_scheduler(**meta_data["lr_scheduler"])

        if self.resume_from_checkpoint and last_epoch >= 0:
            checkpoint = torch.load(
                model_path
                / "checkpoints"
                / f"checkpoint-{last_epoch}"
                / "checkpoint.pt",
                weights_only=False,
            )
            self.my_generator.load_state_dict(checkpoint["generator"])
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.dataset = datasets.load_from_disk(dataset_path=model_path / "datasets")
        else:
            self.dataset = get_dataset(
                **meta_data["dataset"], my_generator=self.my_generator
            )

        assert (
            meta_data["model"]["preprocess"] == self.model.data_collator.preprocess
            and meta_data["model"]["model_type"] == self.model.config.model_type
        ), "preprocess or model type is inconsistent"

        train_dataloader = DataLoader(
            dataset=self.dataset["train"],
            batch_size=self.batch_size,
            collate_fn=lambda examples: examples,
            shuffle=True,
            generator=self.my_generator.torch_c_rng,
        )
        eval_dataloader = DataLoader(
            dataset=self.dataset["validation"],
            batch_size=self.batch_size,
            collate_fn=lambda examples: examples,
        )

        logger.info("initialize model weights")
        self.model = self.model.to(self.device)
        for m in self.model.modules():
            # linear layers
            if isinstance(m, nn.Linear) or isinstance(m, nn.Bilinear):
                self.initializer(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # (transposed) convolution layers
            if (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv3d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.ConvTranspose2d)
                or isinstance(m, nn.ConvTranspose3d)
            ):
                self.initializer(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        logger.info("enter train loop")
        for epoch in tqdm(range(last_epoch + 1, self.num_epochs)):
            logger.info("train model")
            self.model.train()
            self.model.zero_grad()  # optimizer.zero_grad() is different when multiple models share a common optimizer
            train_loss, train_loss_num, grad_norm = 0.0, 0.0, 0.0
            for step, examples in tqdm(enumerate(train_dataloader)):
                batch = self.model.data_collator(examples, output_label=True)
                if "my_generator" in inspect.signature(self.model.forward).parameters:
                    result = self.model(
                        input=batch["input"],
                        label=batch["label"],
                        my_generator=self.my_generator,
                    )
                else:
                    result = self.model(
                        input=batch["input"],
                        label=batch["label"],
                    )

                result["loss"].backward()
                if (step + 1) % self.accumulate_steps == 0 or step == len(
                    train_dataloader
                ):
                    grad_norm += nn.utils.clip_grad_norm_(
                        parameters=self.model.parameters(), max_norm=self.clip_value
                    ).item()
                    self.optimizer.step()
                    self.model.zero_grad()
                train_loss += result["loss"].item()
                train_loss_num += result["loss_num"].item()

            self.lr_scheduler.step()

            if hasattr(self.model, "train_scikit_learn"):
                logger.info("train scikit_learn")
                self.model.train_scikit_learn(train_dataloader)

            logger.info("eval model")
            with torch.no_grad():
                self.model.eval()
                eval_loss, eval_loss_num = 0.0, 0.0
                metric_loss_dict = {
                    metric_name: {"loss": 0.0, "loss_num": 0.0}
                    for metric_name in self.metrics.keys()
                }
                for examples in tqdm(eval_dataloader):
                    batch = self.model.data_collator(examples, output_label=True)
                    if (
                        "my_generator"
                        in inspect.signature(self.model.forward).parameters
                    ):
                        result = self.model(
                            input=batch["input"],
                            label=batch["label"],
                            my_generator=self.my_generator,
                        )
                    else:
                        result = self.model(
                            input=batch["input"],
                            label=batch["label"],
                        )
                    eval_loss += result["loss"].item()
                    eval_loss_num += result["loss_num"].item()
                    df = self.model.eval_output(examples, batch)
                    observations = batch["label"]["observation"].cpu().numpy()
                    cut1s = np.array([example["cut1"] for example in examples])
                    cut2s = np.array([example["cut2"] for example in examples])
                    for metric_name, metric_fun in self.metrics.items():
                        metric_loss, metric_loss_num = metric_fun(
                            df=df,
                            observation=observations,
                            cut1=cut1s,
                            cut2=cut2s,
                        )
                        metric_loss_dict[metric_name][
                            "loss"
                        ] += metric_loss.sum().item()
                        metric_loss_dict[metric_name][
                            "loss_num"
                        ] += metric_loss_num.sum().item()

            logger.info("save model")
            meta_data["performance"] = {
                "train": {
                    "loss": train_loss,
                    "loss_num": train_loss_num,
                    "grad_num": grad_norm,
                },
                "eval": {
                    "loss": eval_loss,
                    "loss_num": eval_loss_num,
                    **metric_loss_dict,
                },
            }
            print(
                {
                    "train_loss": train_loss / train_loss_num,
                    "eval_loss": eval_loss / eval_loss_num,
                }
            )
            # __default_config__ has value type Path, which is not JSON serializable.
            if "__default_config__" in meta_data:
                meta_data.pop("__default_config__")
            os.makedirs(
                model_path / "checkpoints" / f"checkpoint-{epoch}", exist_ok=True
            )
            with open(
                model_path / "checkpoints" / f"checkpoint-{epoch}" / "meta_data.json",
                "w",
            ) as fd:
                json.dump(meta_data, fd, indent=4)

            torch.save(
                obj={
                    "generator": self.my_generator.state_dict(),
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                },
                f=model_path / "checkpoints" / f"checkpoint-{epoch}" / "checkpoint.pt",
            )
            self.dataset.save_to_disk(dataset_dict_path=model_path / "datasets")
