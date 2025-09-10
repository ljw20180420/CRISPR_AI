#!/usr/bin/env python

import torch
import os
import pathlib
import optuna
from typing import Literal
import importlib
from optuna.samplers import (
    GridSampler,
    RandomSampler,
    TPESampler,
    CmaEsSampler,
    GPSampler,
    PartialFixedSampler,
    NSGAIISampler,
    QMCSampler,
)
from optuna.pruners import (
    MedianPruner,
    NopPruner,
    PatientPruner,
    PercentilePruner,
    SuccessiveHalvingPruner,
    HyperbandPruner,
    ThresholdPruner,
    WilcoxonPruner,
)
import jsonargparse
import numpy as np
import pandas as pd
import yaml
from AI.preprocess.config import get_config
from AI.preprocess.dataset import get_dataset
from common_ai.train import MyTrain
from common_ai.test import MyTest

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
        target_metric: str,
    ) -> None:
        self.output_dir = pathlib.Path(os.fspath(output_dir))
        self.preprocess = preprocess
        self.model_type = model_type
        self.data_file = os.fspath(data_file)
        self.data_name = data_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.study_name = study_name
        self.target_metric = target_metric

    def __call__(self, trial: optuna.Trial):
        model_path = (
            self.output_dir
            / self.preprocess
            / self.model_type
            / self.data_name
            / self.study_name
            / f"trial-{trial._trial_id}"
        )
        os.makedirs(model_path, exist_ok=True)
        with open(model_path / "train.yaml", "w") as fd:
            yaml.dump(self.config_train(trial).as_dict(), fd)
        with open(model_path / f"{self.model_type}.yaml", "w") as fd:
            yaml.dump(self.config_model(trial).as_dict(), fd)
        with open(model_path / "test.yaml", "w") as fd:
            yaml.dump(self.config_test(trial).as_dict(), fd)

        _, train_parser, test_parser = get_config()

        # train
        train_cfg = train_parser.parse_path(model_path / "train.yaml")
        dataset = get_dataset(**train_cfg.dataset.as_dict())
        for epoch, performance in enumerate(
            MyTrain(**train_cfg.train.as_dict())(
                train_parser=train_parser,
                cfg=train_cfg,
                dataset=dataset,
            )
        ):
            if performance is not None:
                trial.report(
                    value=performance["eval"][self.target_metric],
                    step=epoch,
                )
                if trial.should_prune():
                    break

        # test
        test_cfg = test_parser.parse_path(model_path / "test.yaml")
        my_test = MyTest(**test_cfg.test.as_dict())
        best_train_cfg = my_test.get_best_cfg(train_parser)
        dataset = get_dataset(**best_train_cfg.dataset.as_dict())
        my_test(cfg=best_train_cfg, dataset=dataset)

        df = pd.read_csv(model_path / "test_result.csv")
        return df.loc[df["name"] == self.target_metric, "loss"].item()

    def config_test(self, trial: optuna.Trial) -> jsonargparse.Namespace:
        cfg = jsonargparse.Namespace()
        cfg.test = jsonargparse.Namespace(
            model_path=(
                self.output_dir
                / self.preprocess
                / self.model_type
                / self.data_name
                / self.study_name
                / f"trial-{trial._trial_id}"
            ).as_posix(),
            target=self.target_metric,
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
        )
        model_module = importlib.import_module(f"AI.preprocess.{self.preprocess}.model")
        if not hasattr(
            getattr(model_module, f"{self.model_type}Model"), "my_train_model"
        ):
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
            ]
        ]
        cfg.model = f"{self.model_type}.yaml"
        return cfg

    def config_model(
        self,
        trial: optuna.Trial,
    ) -> jsonargparse.Namespace:
        cfg = jsonargparse.Namespace()
        cfg.class_path = (
            f"AI.preprocess.{self.preprocess}.model.{self.model_type}Config"
        )

        if self.preprocess == "CRIformer":
            if self.model_type == "CRIformer":
                cfg.init_args = jsonargparse.Namespace(
                    ext1_up=25,
                    ext1_down=6,
                    ext2_up=6,
                    ext2_down=25,
                    hidden_size=trial.suggest_int(
                        "CRIformer.CRIformer.hidden_size", 128, 512, step=128
                    ),
                    num_hidden_layers=trial.suggest_int(
                        "CRIformer.CRIformer.num_hidden_layers", 2, 4
                    ),
                    # num_attention_heads must devide hidden_size
                    num_attention_heads=trial.suggest_categorical(
                        "CRIformer.CRIformer.num_attention_heads", choices=[2, 4, 8]
                    ),
                    intermediate_size=trial.suggest_int(
                        "CRIformer.CRIformer.intermediate_size", 512, 2048, step=512
                    ),
                    hidden_dropout_prob=trial.suggest_float(
                        "CRIformer.CRIformer.hidden_dropout_prob", 0.0, 0.1
                    ),
                    attention_probs_dropout_prob=trial.suggest_float(
                        "CRIformer.CRIformer.attention_probs_dropout_prob", 0.0, 0.1
                    ),
                )
        elif self.preprocess == "CRIfuser":
            if self.model_type == "CRIfuser":
                half_layer_num = trial.suggest_int(
                    "CRIfuser.CRIfuser.unet_channels", 2, 4
                )
                cfg.init_args = jsonargparse.Namespace(
                    ext1_up=25,
                    ext1_down=6,
                    ext2_up=6,
                    ext2_down=25,
                    max_micro_homology=7,
                    loss_weights={
                        trial.suggest_categorical(
                            "CRIfuser.CRIfuser.loss_weights",
                            choices=[
                                "double_sample_negative_ELBO",
                                "importance_sample_negative_ELBO",
                                "forward_negative_ELBO",
                                "reverse_negative_ELBO",
                                "sample_CE",
                                "non_sample_CE",
                            ],
                        ): 1.0,
                    },
                    unet_channels=(
                        32 * np.r_[1:half_layer_num, half_layer_num:0:-1]
                    ).tolist(),
                    noise_scheduler=trial.suggest_categorical(
                        "CRIfuser.CRIfuser.noise_scheduler",
                        choices=["linear", "cosine", "exp", "uniform"],
                    ),
                    noise_timesteps=trial.suggest_int(
                        "CRIfuser.CRIfuser.noise_timesteps", 10, 30
                    ),
                    cosine_factor=0.008,
                    exp_scale=5.0,
                    exp_base=5.0,
                    uniform_scale=1.0,
                )
        elif self.preprocess == "DeepHF":
            if self.model_type == "DeepHF":
                cfg.init_args = jsonargparse.Namespace(
                    ext1_up=25,
                    ext1_down=6,
                    ext2_up=6,
                    ext2_down=25,
                    em_drop=trial.suggest_float("DeepHF.DeepHF.em_drop", 0.0, 0.2),
                    fc_drop=trial.suggest_float("DeepHF.DeepHF.fc_drop", 0.0, 0.4),
                    em_dim=trial.suggest_int("DeepHF.DeepHF.em_dim", 33, 55),
                    rnn_units=trial.suggest_int("DeepHF.DeepHF.rnn_units", 50, 70),
                    fc_num_hidden_layers=trial.suggest_int(
                        "DeepHF.DeepHF.fc_num_hidden_layers", 2, 5
                    ),
                    fc_num_units=trial.suggest_int(
                        "DeepHF.DeepHF.fc_num_units", 220, 420
                    ),
                    fc_activation=trial.suggest_categorical(
                        "DeepHF.DeepHF.fc_activation",
                        choices=["elu", "relu", "tanh", "sigmoid", "hard_sigmoid"],
                    ),
                )
            elif self.model_type == "CNN":
                cfg.init_args = jsonargparse.Namespace(
                    ext1_up=25,
                    ext1_down=6,
                    ext2_up=6,
                    ext2_down=25,
                    em_drop=trial.suggest_float("CNN.CNN.em_drop", 0.0, 0.2),
                    fc_drop=trial.suggest_float("CNN.CNN.fc_drop", 0.0, 0.2),
                    em_dim=trial.suggest_int("CNN.CNN.em_dim", 26, 46),
                    fc_num_hidden_layers=trial.suggest_int(
                        "CNN.CNN.fc_num_hidden_layers", 2, 4
                    ),
                    fc_num_units=trial.suggest_int("CNN.CNN.fc_num_units", 300, 500),
                    fc_activation=trial.suggest_categorical(
                        "CNN.CNN.fc_activation",
                        choices=["elu", "relu", "tanh", "sigmoid", "hard_sigmoid"],
                    ),
                    kernel_sizes=[1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13],
                    feature_maps=trial.suggest_categorical(
                        "CNN.CNN.feature_maps",
                        choices=[
                            [
                                20,
                                20,
                                20,
                                20,
                                20,
                                20,
                                20,
                                20,
                                20,
                                20,
                                20,
                                20,
                                20,
                                20,
                            ],
                            [
                                20,
                                20,
                                20,
                                20,
                                20,
                                20,
                                20,
                                40,
                                40,
                                40,
                                40,
                                40,
                                40,
                                40,
                            ],
                            [
                                20,
                                20,
                                20,
                                20,
                                40,
                                40,
                                40,
                                40,
                                80,
                                80,
                                80,
                                80,
                                80,
                                80,
                            ],
                            [
                                40,
                                40,
                                40,
                                40,
                                40,
                                40,
                                40,
                                40,
                                80,
                                80,
                                80,
                                80,
                                80,
                                80,
                            ],
                        ],
                    ),
                )
            elif self.model_type == "MLP":
                cfg.init_args = jsonargparse.Namespace(
                    ext1_up=25,
                    ext1_down=6,
                    ext2_up=6,
                    ext2_down=25,
                    fc_drop=trial.suggest_float("MLP.MLP.fc_drop", 0.0, 0.2),
                    fc_num_hidden_layers=trial.suggest_int(
                        "MLP.MLP.fc_num_hidden_layers", 3, 5
                    ),
                    fc_num_units=trial.suggest_int("MLP.MLP.fc_num_units", 300, 500),
                    fc_activation=trial.suggest_categorical(
                        "MLP.MLP.fc_activation",
                        choices=["elu", "relu", "tanh", "sigmoid", "hard_sigmoid"],
                    ),
                )
            elif self.model_type == "XGBoost":
                cfg.init_args = jsonargparse.Namespace(
                    ext1_up=25,
                    ext1_down=6,
                    ext2_up=6,
                    ext2_down=25,
                    booster=trial.suggest_categorical(
                        "XGBoost.XGBoost.booster",
                        choices=["gbtree", "dart"],
                    ),
                    device="gpu",
                    eta=trial.suggest_float("XGBoost.XGBoost.eta", 0.2, 0.8),
                    max_depth=trial.suggest_int("XGBoost.XGBoost.max_depath", 4, 6),
                    subsample=trial.suggest_float(
                        "XGBoost.XGBoost.subsample", 0.5, 1.0
                    ),
                    reg_lambda=trial.suggest_float(
                        "XGBoost.XGBoost.reg_lambda", 0.5, 2.0
                    ),
                    num_boost_round=trial.suggest_int(
                        "XGBoost.XGBoost.num_boost_round", 50, 200
                    ),
                    early_stopping_rounds=trial.suggest_int(
                        "XGBoost.XGBoost.early_stopping_rounds", 5, 20
                    ),
                )
            elif self.model_type == "Ridge":
                cfg.init_args = jsonargparse.Namespace(
                    ext1_up=25,
                    ext1_down=6,
                    ext2_up=6,
                    ext2_down=25,
                    alpha=trial.suggest_float("Ridge.Ridge.alpha", 50.0, 200.0),
                )
        elif self.preprocess == "FOREcasT":
            if self.model_type == "FOREcasT":
                cfg.init_args = jsonargparse.Namespace(
                    max_del_size=trial.suggest_int(
                        "FOREcasT.FOREcasT.max_del_size", 20, 40
                    ),
                    reg_const=trial.suggest_float(
                        "FOREcasT.FOREcasT.reg_const", 0.0, 0.02
                    ),
                    i1_reg_const=trial.suggest_float(
                        "FOREcasT.FOREcasT.i1_reg_const", 0.0, 0.02
                    ),
                )
        elif self.preprocess == "inDelphi":
            if self.model_type == "inDelphi":
                cfg.init_args = jsonargparse.Namespace(
                    DELLEN_LIMIT=trial.suggest_int(
                        "inDelphi.inDelphi.DELLEN_LIMIT", 40, 80
                    ),
                    mid_dim=trial.suggest_int("inDelphi.inDelphi.mid_dim", 16, 64),
                )
        elif self.preprocess == "Lindel":
            if self.model_type == "Lindel":
                cfg.init_args = jsonargparse.Namespace(
                    dlen=trial.suggest_int("Lindel.Lindel.dlen", 20, 40),
                    mh_len=trial.suggest_int("Lindel.Lindel.mh_len", 3, 5),
                    reg_mode=trial.suggest_categorical(
                        "Lindel.Lindel.reg_mode", choices=["l2", "l1"]
                    ),
                    reg_const=trial.suggest_float("Lindel.Lindel.reg_const", 0.0, 0.02),
                )

        return cfg


def main(
    output_dir: os.PathLike,
    preprocess: str,
    model_type: str,
    data_file: os.PathLike,
    data_name: str,
    batch_size: int,
    num_epochs: int,
    target_metric: str,
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
        target_metric: Metric used to select hyperparameters.
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
        target_metric=target_metric,
    )
    study_path = output_dir / preprocess / model_type / data_name / study_name
    os.makedirs(study_path, exist_ok=True)
    study = optuna.create_study(
        storage=optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(
                (study_path / "optuna_journal_storage.log").as_posix()
            ),
        ),
        sampler=eval(sampler)(),
        pruner=eval(pruner)(),
        study_name=study_name,
        load_if_exists=load_if_exists,
    )
    study.optimize(
        func=objective,
        n_trials=n_trials,
    )


if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)
