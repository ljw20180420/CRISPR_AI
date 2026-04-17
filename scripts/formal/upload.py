#!/usr/bin/env python

import os
import pathlib
import shutil
import sys
import tempfile

from huggingface_hub import (
    create_repo,
    delete_repo,
    upload_folder,
    upload_large_folder,
    whoami,
)

# change to the dir to the project
os.chdir(pathlib.Path(__file__).resolve().parent.parent.parent)


def upload_first_time(username: str, preprocess: str, model_cls: str, data_name: str):
    repo_id = f"{username}/{preprocess}_{model_cls}_{data_name}"
    delete_repo(repo_id=repo_id, repo_type="model", missing_ok=True)
    create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

    with tempfile.TemporaryDirectory() as td:
        tmpdir = pathlib.Path(td)
        for comp in ["checkpoints", "logs"]:
            shutil.copytree(
                output_dir / comp / preprocess / model_cls / data_name / "default",
                tmpdir / comp,
            )
        upload_large_folder(
            repo_id=repo_id,
            folder_path=tmpdir,
            repo_type="model",
        )


def upload(username: str, preprocess: str, model_cls: str, data_name: str):
    repo_id = f"{username}/{preprocess}_{model_cls}_{data_name}"

    for comp in ["checkpoints", "logs"]:
        upload_folder(
            repo_id=repo_id,
            folder_path=output_dir
            / comp
            / preprocess
            / model_cls
            / data_name
            / "default",
            path_in_repo=comp,
            ignore_patterns="*.pdf",
            delete_patterns="*",
        )


username = whoami()["name"]
if "OUTPUT_DIR" in os.environ:
    output_dir = pathlib.Path(os.environ["OUTPUT_DIR"])
else:
    output_dir = pathlib.Path(os.environ["HOME"]) / "CRISPR_results"
output_dir = output_dir / "formal" / "default"

for data_name in ["SX_spcas9", "SX_spymac", "SX_ispymac"]:
    for preprocess, model_cls in [
        ("CRIformer", "CRIformer"),
        ("inDelphi", "inDelphi"),
        ("Lindel", "Lindel"),
        ("DeepHF", "DeepHF"),
        ("DeepHF", "CNN"),
        ("DeepHF", "MLP"),
        ("DeepHF", "XGBoost"),
        ("DeepHF", "SGDClassifier"),
        ("CRIfuser", "CRIfuser"),
        ("FOREcasT", "FOREcasT"),
    ]:
        success = False
        while not success:
            try:
                if "first" in sys.argv[1:]:
                    upload_first_time(username, preprocess, model_cls, data_name)
                else:
                    upload()

                success = True
            except Exception as e:
                print(e)
