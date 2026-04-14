#!/usr/bin/env python

import os
import pathlib

from huggingface_hub import upload_folder, whoami

# change to the dir to the project
os.chdir(pathlib.Path(__file__).resolve().parent.parent.parent)

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
                # upload checkpoints
                upload_folder(
                    repo_id=f"{username}/{preprocess}_{model_cls}_{data_name}",
                    folder_path=output_dir
                    / "checkpoints"
                    / preprocess
                    / model_cls
                    / data_name
                    / "default",
                    path_in_repo="checkpoints",
                    delete_patterns="*",
                )

                # upload logs
                upload_folder(
                    repo_id=f"{username}/{preprocess}_{model_cls}_{data_name}",
                    folder_path=output_dir
                    / "logs"
                    / preprocess
                    / model_cls
                    / data_name
                    / "default",
                    path_in_repo="logs",
                    ignore_patterns="*.pdf",
                    delete_patterns="*",
                )

                success = True
            except Exception as e:
                print(e)
