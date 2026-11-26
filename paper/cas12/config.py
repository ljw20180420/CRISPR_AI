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
    }
