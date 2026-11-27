#!/usr/bin/env python

import os
import pathlib
import sys

os.chdir(os.fspath(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, os.getcwd())

from cas12 import align, config, download, predict, train

if __name__ == "__main__":
    cfg = config.cas12()

    # download.download_metadata(cfg)
    # download.download_SRR(cfg)
    # align.get_references(cfg)
    # align.split_fastq_to_target_sites(cfg)
    # align.collect_rearr_input(cfg)
    # align.align_by_rearr(cfg)

    for model_name in cfg["train"]["models"]:
        for data_name in cfg["train"]["datasets"]:
            train.train_model(cfg, model_name="CRIfuser", data_name="AsCpf1")

    # predict.predict_by_AI_models(cfg)
    # predict.get_metrics(cfg)
