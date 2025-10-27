#!/usr/bin/env python

import os
import pathlib
from common_ai.config import get_config, get_train_parser
from common_ai.train import MyTrain
from common_ai.test import MyTest
from common_ai.hta import MyHta
from common_ai.hpo import MyHpo

# change directory to the current script
os.chdir(pathlib.Path(__file__).parent)

# parse arguments
parser, train_parser, test_parser, hta_parser, hpo_parser = get_config()
cfg = parser.parse_args()

if cfg.subcommand == "train":
    for epoch in MyTrain(**cfg.train.train.as_dict())(train_parser):
        pass

elif cfg.subcommand == "test":
    epoch = MyTest(**cfg.test.as_dict())(train_parser)

elif cfg.subcommand == "hta":
    MyHta(**cfg.hta.as_dict())()

elif cfg.subcommand == "hpo":
    MyHpo(**cfg.hpo.hpo.as_dict())(hpo_parser, get_train_parser)
