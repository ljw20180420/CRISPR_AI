#!/usr/bin/env python

import os
import pathlib
import sys

# change directory to the project
os.chdir(pathlib.Path(__file__).resolve().parent.parent)
# add project to search path
sys.path.insert(0, os.getcwd())

import pandas as pd
from common_ai.config import get_config
from common_ai.test import MyTest

from AI.inference import MyInference

# parse arguments
(
    parser,
    train_parser,
    test_parser,
    infer_parser,
    explain_parser,
    app_parser,
    hta_parser,
    hpo_parser,
) = get_config()

ref = "GGCATTATCTGGCTTACCTGAATCGTCCCGGGAATTTTCTTCGGTGAGCAT"
infer_df = pd.DataFrame(
    {
        "ref": [ref],
        "cut": [25],
        "scaffold": ["spcas9"],  # dummy, not used by CRIfuser
    }
)


my_inference = MyInference(
    ext1_up=25, ext1_down=6, ext2_up=6, ext2_down=25, max_del_size=0
)

(
    _,
    train_cfg,
    my_inference.logger,
    my_inference.model,
    my_inference.my_generator,
) = MyTest(
    checkpoints_path="/home/ljw/sdc1/CRISPR_results/formal/default/checkpoints/CRIfuser/CRIfuser/SX_spcas9/default",
    logs_path="/home/ljw/sdc1/CRISPR_results/formal/default/logs/CRIfuser/CRIfuser/SX_spcas9/default",
    target="GreatestCommonCrossEntropy",
    maximize_target=False,
    overwrite={"train.device": "cpu"},
).load_model(
    train_parser
)
my_inference.batch_size = train_cfg.train.batch_size

infer_out = my_inference(infer_df, test_cfg=None, train_parser=None)
