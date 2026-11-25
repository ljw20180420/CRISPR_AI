import os
import pathlib


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
        "2bit": pathlib.Path(os.environ["HOME"])
        / "sdb1"
        / "ucsc"
        / "hubs"
        / "myHub"
        / "lhg19"
        / "lhg19.2bit",
    }
