#!/usr/bin/env python

import os
import pathlib

import yaml

os.chdir(pathlib.Path(__file__).resolve().parent.parent)

hparams = {}
pre_dir = pathlib.Path("AI/preprocess")
for preprocess in os.listdir(pre_dir):
    if not os.path.isdir(pre_dir / preprocess):
        continue
    for model in os.listdir(pre_dir / preprocess):
        if not model.endswith(".yaml"):
            continue
        with open(pre_dir / preprocess / model, "r") as fd:
            hparams[model.removesuffix(".yaml")] = yaml.safe_load(fd)["init_args"]

with open("paper/hyperparameters.yaml", "w") as fd:
    yaml.safe_dump(data=hparams, stream=fd)
