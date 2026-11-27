import os
import pathlib


def get_checkpoints_path(model: str, data_name: str) -> os.PathLike:
    return (
        pathlib.Path(os.environ["HOME"])
        / "sdc1"
        / "CRISPR_results"
        / "formal"
        / "default"
        / "checkpoints"
        / model
        / model
        / data_name
        / "default"
    )


def get_logs_path(model: str, dataname: str) -> os.PathLike:
    return (
        pathlib.Path(os.environ["HOME"])
        / "sdc1"
        / "CRISPR_results"
        / "formal"
        / "default"
        / "logs"
        / model
        / model
        / dataname
        / "default"
    )


def predict() -> dict:
    return {
        "models": ["CRIfuser", "inDelphi", "FOREcasT", "Lindel"],
        "data_name": "SX_spcas9",
        "target": "GreatestCommonCrossEntropy",
        "maximize_target": False,
    }


def cas12() -> dict:
    return {
        "output_dir": pathlib.Path(os.environ["HOME"])
        / "sdb1"
        / "CRISPR_AI"
        / "revise"
        / "cas12",
        "SRP": "SRP075607",
        "primer_file": "https://static-content.springer.com/esm/art%3A10.1038%2Fnbt.3620/MediaObjects/41587_2016_BFnbt3620_MOESM39_ESM.xlsx",
        "extend": 150,
        "short_extend": 30,
        "2bit": pathlib.Path(os.environ["HOME"])
        / "sdb1"
        / "ucsc"
        / "hubs"
        / "myHub"
        / "lhg19"
        / "lhg19.2bit",
        "black_list_sites": [
            "DNMT1_site_3",
            "DNMT1_site_7",
            "DNMT1_site_3_d1",
            "DNMT1_site_7_d0",
        ],
        "cas_proteins": ["AsCpf1", "LbCpf1"],
        "predict": predict(),
        "inference": {
            "ext1_up": 25,
            "ext1_down": 6,
            "ext2_up": 6,
            "ext2_down": 25,
            "max_del_size": 44,
        },
        "train": {
            "trial_name": "revise",
            "batch_size": 100,
            "num_epochs": 33,
            "last_epoch": -1,
            "clip_value": 1.0,
            "accumulate_steps": 1,
            "device": "cuda",
            "generator": {"seed": 63036},
            "initializer": {"name": "kaiming_uniform_"},
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 0.0001,
                "weight_decay": 0.0,
            },
            "lr_scheduler": {
                "name": "CosineAnnealingWarmRestarts",
                "warmup_epochs": 3,
                "period_epochs": 30,
            },
            "metrics": {
                "GreatestCommonCrossEntropy": {
                    "ext1_up": 25,
                    "ext1_down": 6,
                    "ext2_up": 6,
                    "ext2_down": 25,
                    "max_del_size": 44,
                },
                "MSE": {
                    "ext1_up": 25,
                    "ext1_down": 6,
                    "ext2_up": 6,
                    "ext2_down": 25,
                    "max_del_size": 44,
                },
            },
            "datasets": {
                "AsCpf1": {
                    "data_file": "XXXXXXXXXXXXXXXXXX",
                    "name": "AsCpf1",
                    "test_ratio": 0.00,
                    "validation_ratio": 0.2,
                    "seed": 63036,
                    "random_insert_uplimit": 0,
                    "insert_uplimit": 2,
                },
                "LbCpf1": {
                    "data_file": "XXXXXXXXXXXXXXXXXX",
                    "name": "LbCpf1",
                    "test_ratio": 0.00,
                    "validation_ratio": 0.2,
                    "seed": 63036,
                    "random_insert_uplimit": 0,
                    "insert_uplimit": 2,
                },
            },
        },
        "CRIfuser": {
            "ext1_up": 25,
            "ext1_down": 6,
            "ext2_up": 6,
            "ext2_down": 25,
            "max_micro_homology": 7,
            "loss_weights": {
                "importance_sample_negative_ELBO": 1.0,
                "non_sample_CE": 1.0,
            },
            "unet_channels": [
                32,
                64,
                96,
                64,
                32,
            ],
            "noise_scheduler": "linear",
            "noise_timesteps": 20,
            "cosine_factor": 0.008,
            "exp_scale": 5.0,
            "exp_base": 5.0,
            "uniform_scale": 1.0,
            "eval_output_step": 1,
            "eval_output_batch_size": 1,
            "autocast": False,
        },
        "inDelphi": {
            "max_del_size": 44,
            "mid_dim": 16,
        },
        "FOREcasT": {
            "max_del_size": 44,
            "reg_const": 0.01,
            "i1_reg_const": 0.01,
        },
        "Lindel": {
            "max_del_size": 44,
            "mh_len": 4,
            "reg_mode": "l2",
            "reg_const": 0.01,
        },
    }
