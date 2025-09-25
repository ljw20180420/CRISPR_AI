#!/usr/bin/env python

import importlib
import os
import pathlib
from typing import Literal

import jsonargparse
import optuna
from torch import nn
import yaml
from torch.utils.tensorboard import SummaryWriter
from tbparse import SummaryReader
from common_ai.test import MyTest
from common_ai.train import MyTrain
from common_ai.utils import get_latest_event_file

from AI.preprocess.config import get_config
from AI.preprocess.dataset import get_dataset

# change directory to the current script
os.chdir(pathlib.Path(__file__).parent)


class Objective:
    def __init__(
        self,
        output_dir: os.PathLike,
        preprocess: str,
        model_type: str,
        data_file: os.PathLike,
        data_name: str,
        batch_size: int,
        num_epochs: int,
        study_name: str,
        target: str,
    ) -> None:
        self.output_dir = pathlib.Path(os.fspath(output_dir))
        self.preprocess = preprocess
        self.model_type = model_type
        self.data_file = os.fspath(data_file)
        self.data_name = data_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.study_name = study_name
        self.target = target
        self.checkpoints_path = (
            self.output_dir
            / "checkpoints"
            / self.preprocess
            / self.model_type
            / self.data_name
            / self.study_name
        )
        self.logs_path = (
            self.output_dir
            / "logs"
            / self.preprocess
            / self.model_type
            / self.data_name
            / self.study_name
        )

    def __call__(self, trial: optuna.Trial):
        os.makedirs(self.checkpoints_path / f"trial-{trial._trial_id}", exist_ok=True)
        with open(
            self.checkpoints_path / f"trial-{trial._trial_id}" / "train.yaml", "w"
        ) as fd:
            yaml.dump(self.config_train(trial).as_dict(), fd)
        with open(
            self.checkpoints_path / f"trial-{trial._trial_id}" / "test.yaml", "w"
        ) as fd:
            yaml.dump(self.config_test(trial).as_dict(), fd)
        with open(
            self.checkpoints_path
            / f"trial-{trial._trial_id}"
            / f"{self.model_type}.yaml",
            "w",
        ) as fd:
            model_module = importlib.import_module(
                f"AI.preprocess.{self.preprocess}.model"
            )
            cfg, hparam_dict = getattr(
                model_module, f"{self.model_type}Model"
            ).my_model_hpo(trial)
            cfg.class_path = (
                f"AI.preprocess.{self.preprocess}.model.{self.model_type}Model"
            )
            yaml.dump(cfg.as_dict(), fd)

        _, train_parser, test_parser = get_config()

        # train
        train_cfg = train_parser.parse_path(
            self.checkpoints_path / f"trial-{trial._trial_id}" / "train.yaml"
        )
        dataset = get_dataset(**train_cfg.dataset.as_dict())
        for epoch, logdir in MyTrain(**train_cfg.train.as_dict())(
            train_parser=train_parser,
            cfg=train_cfg,
            dataset=dataset,
        ):
            latest_event_file = get_latest_event_file(logdir)
            df = SummaryReader(latest_event_file.as_posix(), pivot=True).scalars
            trial.report(
                value=df.loc[df["step"] == epoch, f"eval/{self.target}"].item(),
                step=epoch,
            )
            if trial.should_prune():
                break

        # test
        test_cfg = test_parser.parse_path(
            self.checkpoints_path / f"trial-{trial._trial_id}" / "test.yaml"
        )
        my_test = MyTest(**test_cfg.test.as_dict())
        best_train_cfg = my_test.get_best_cfg(train_parser)
        dataset = get_dataset(**best_train_cfg.dataset.as_dict())
        epoch, logdir = my_test(cfg=best_train_cfg, dataset=dataset)
        latest_event_file = get_latest_event_file(logdir)
        df = SummaryReader(latest_event_file.as_posix(), pivot=True).scalars
        target_metric_val = df.loc[df["step"] == epoch, f"test/{my_test.target}"].item()
        tensorboard_writer = SummaryWriter(self.logs_path / "hpo")
        full_hparam_dict = {
            "preprocess": self.preprocess,
            "model_type": self.model_type,
        }
        full_hparam_dict.update(hparam_dict)
        tensorboard_writer.add_hparams(
            hparam_dict=full_hparam_dict,
            metric_dict={f"test/{my_test.target}": target_metric_val},
        )
        tensorboard_writer.close()

        return target_metric_val

    def config_test(self, trial: optuna.Trial) -> jsonargparse.Namespace:
        cfg = jsonargparse.Namespace()
        cfg.test = jsonargparse.Namespace(
            checkpoints_path=(
                self.checkpoints_path / f"trial-{trial._trial_id}"
            ).as_posix(),
            logs_path=(self.logs_path / f"trial-{trial._trial_id}").as_posix(),
            target=self.target,
            batch_size=100,
            device="cuda",
        )
        return cfg

    def config_train(self, trial: optuna.Trial) -> jsonargparse.Namespace:
        cfg = jsonargparse.Namespace()
        cfg.train = jsonargparse.Namespace(
            output_dir=self.output_dir.as_posix(),
            trial_name=f"{self.study_name}/trial-{trial._trial_id}",
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            last_epoch=-1,
            clip_value=1.0,
            accumulate_steps=1,
            device="cuda",
            evaluation_only=False,
        )
        model_module = importlib.import_module(f"AI.preprocess.{self.preprocess}.model")
        if issubclass(getattr(model_module, f"{self.model_type}Model"), nn.Module):
            cfg.initializer = jsonargparse.Namespace(
                name=trial.suggest_categorical(
                    "initializer.name",
                    choices=[
                        "uniform_",
                        "normal_",
                        "xavier_uniform_",
                        "xavier_normal_",
                        "kaiming_uniform_",
                        "kaiming_normal_",
                        "trunc_normal_",
                    ],
                ),
            )
            cfg.optimizer = jsonargparse.Namespace(
                name=trial.suggest_categorical(
                    "optimizer.name",
                    choices=[
                        "Adadelta",
                        "Adafactor",
                        "Adagrad",
                        "Adam",
                        "AdamW",
                        "Adamax",
                        "ASGD",
                        "NAdam",
                        "RAdam",
                        "RMSprop",
                        "SGD",
                    ],
                ),
                learning_rate=trial.suggest_float(
                    "optimizer.learning_rate", 1e-5, 1e-2, log=True
                ),
                weight_decay=0.0,
            )
            cfg.lr_scheduler = jsonargparse.Namespace(
                name=trial.suggest_categorical(
                    "lr_scheduler.name",
                    choices=[
                        "CosineAnnealingWarmRestarts",
                        "ConstantLR",
                        "ReduceLROnPlateau",
                    ],
                ),
                warmup_epochs=self.num_epochs // 10,
                period_epochs=self.num_epochs - self.num_epochs // 10,
            )
        else:
            # initializer, optimizer, lr_scheduler are not used. Assign dummy values to them.
            cfg.initializer = jsonargparse.Namespace(
                name="kaiming_uniform_",
            )
            cfg.optimizer = jsonargparse.Namespace(
                name="AdamW",
                learning_rate=0.0001,
                weight_decay=0.0,
            )
            cfg.lr_scheduler = jsonargparse.Namespace(
                name="CosineAnnealingWarmRestarts",
                warmup_epochs=3,
                period_epochs=30,
            )

        cfg.dataset = jsonargparse.Namespace(
            data_file=self.data_file,
            name=self.data_name,
            test_ratio=0.05,
            validation_ratio=0.05,
            random_insert_uplimit=0,
            insert_uplimit=2,
            seed=63036,
        )
        cfg.generator = jsonargparse.Namespace(
            seed=63036,
        )
        cfg.logger = jsonargparse.Namespace(
            log_level="WARNING",
        )
        cfg.early_stopping = jsonargparse.Namespace(
            patience=None,
            delta=0.0,
        )
        cfg.metric = [
            jsonargparse.Namespace(
                class_path=class_path,
                init_args=jsonargparse.Namespace(
                    ext1_up=25,
                    ext1_down=6,
                    ext2_up=6,
                    ext2_down=25,
                ),
            )
            for class_path in [
                "AI.preprocess.metric.CrossEntropy",
                "AI.preprocess.metric.NonZeroCrossEntropy",
                "AI.preprocess.metric.NonWildTypeCrossEntropy",
                "AI.preprocess.metric.NonZeroNonWildTypeCrossEntropy",
                "AI.preprocess.metric.GreatestCommonCrossEntropy",
            ]
        ]
        cfg.model = f"{self.model_type}.yaml"
        return cfg


def main(
    output_dir: os.PathLike,
    preprocess: str,
    model_type: str,
    data_file: os.PathLike,
    data_name: str,
    batch_size: int,
    num_epochs: int,
    target: str,
    sampler: Literal[
        "GridSampler",
        "RandomSampler",
        "TPESampler",
        "CmaEsSampler",
        "GPSampler",
        "PartialFixedSampler",
        "NSGAIISampler",
        "QMCSampler",
    ],
    pruner: Literal[
        "MedianPruner",
        "NopPruner",
        "PatientPruner",
        "PercentilePruner",
        "SuccessiveHalvingPruner",
        "HyperbandPruner",
        "ThresholdPruner",
        "WilcoxonPruner",
    ],
    study_name: str,
    n_trials: int,
    load_if_exists: bool,
) -> None:
    """Arguments of optuna.

    Args:
        output_dir: Output directory.
        preprocess: Preprocess method for data.
        model_type: Type of model.
        data_file: Data file.
        data_name: Dataset name.
        batch_size: Batch size.
        num_epochs: Number of epochs.
        target: Metric used to select hyperparameters.
        sampler: Sampler continually narrows down the search space using the records of suggested parameter values and evaluated objective values.
        pruner: Pruner to stop unpromising trials at the early stages.
        study_name: The name of the study.
        n_trials: The total number of trials in the study.
        load_if_exists: Flag to control the behavior to handle a conflict of study names. In the case where a study named study_name already exists in the storage.
    """
    # if model_type == "DeepHF":
    #     torch.backends.cudnn.enabled = False
    output_dir = pathlib.Path(os.fspath(output_dir))
    objective = Objective(
        output_dir=output_dir,
        preprocess=preprocess,
        model_type=model_type,
        data_file=data_file,
        data_name=data_name,
        batch_size=batch_size,
        num_epochs=num_epochs,
        study_name=study_name,
        target=target,
    )
    os.makedirs(objective.logs_path, exist_ok=True)
    study = optuna.create_study(
        storage=optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(
                (objective.logs_path / "optuna_journal_storage.log").as_posix()
            ),
        ),
        sampler=getattr(importlib.import_module("optuna.samplers"), sampler)(),
        pruner=getattr(importlib.import_module("optuna.pruners"), pruner)(),
        study_name=study_name,
        load_if_exists=load_if_exists,
    )
    study.optimize(
        func=objective,
        n_trials=n_trials,
    )


if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)
