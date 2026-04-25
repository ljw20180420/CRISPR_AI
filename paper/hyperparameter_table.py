#!/usr/bin/env python

import os
import pathlib

import pandas as pd
import yaml

os.chdir(pathlib.Path(__file__).resolve().parent.parent)

pre_dir = pathlib.Path("AI/preprocess")
with pd.ExcelWriter("paper/hyperparameters.xlsx") as ew:
    for preprocess in os.listdir(pre_dir):
        if not os.path.isdir(pre_dir / preprocess):
            continue
        for model in os.listdir(pre_dir / preprocess):
            if not model.endswith(".yaml"):
                continue
            with open(pre_dir / preprocess / model, "r") as fd:
                hparams = yaml.safe_load(fd)["init_args"]

            model_cls = model.removesuffix(".yaml")
            df = pd.DataFrame({model_cls: hparams})
            df.to_excel(ew, sheet_name=model_cls)
