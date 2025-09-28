#!/usr/bin/env python

import os
import pathlib
from AI.config import get_config
from common_ai.train import MyTrain
from common_ai.test import MyTest

# change directory to the current script
os.chdir(pathlib.Path(__file__).parent)

# parse arguments
parser, train_parser, test_parser = get_config()
cfg = parser.parse_args()

if cfg.subcommand == "train":
    for epoch, logdir in MyTrain(**cfg.train.train.as_dict())(train_parser, cfg.train):
        pass

elif cfg.subcommand == "test":
    epoch, logdir = my_test = MyTest(**cfg.test.test.as_dict())(train_parser)
