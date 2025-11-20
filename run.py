#!/usr/bin/env python

import os
import pathlib
import pandas as pd
import numpy as np
from common_ai.config import get_config, get_train_parser
from common_ai.train import MyTrain
from common_ai.test import MyTest
from AI.inference import MyInference
from AI.shap import MyShap
from AI.gradio_fn import MyGradioFn
from common_ai.hta import MyHta
from common_ai.hpo import MyHpo

# change directory to the current script
os.chdir(pathlib.Path(__file__).parent)

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
cfg = parser.parse_args()

if cfg.subcommand == "train":
    for epoch in MyTrain(**cfg.train.train.as_dict())(train_parser):
        pass

elif cfg.subcommand == "test":
    epoch = MyTest(**cfg.test.as_dict())(train_parser)

elif cfg.subcommand == "infer":
    MyInference(**cfg.infer.inference.init_args.as_dict())(
        infer_df=pd.read_csv(cfg.infer.input),
        test_cfg=cfg.infer.test,
        train_parser=train_parser,
    ).to_csv(cfg.infer.output, index=False)

elif cfg.subcommand == "explain":
    my_shap = MyShap(**cfg.explain.shap.init_args.as_dict())
    explanation = my_shap(explain_parser, train_parser)
    my_shap.visualize(
        explain_parser,
        train_parser,
        explanation,
        local_idxs=[0],
    )
    explanation.data = np.array(["A", "C", "G", "T"])[explanation.data]
    my_shap.text_plot(
        explanation=explanation,
        local_idxs=[0],
        logs_path=pathlib.Path(os.fspath(cfg.explain.test.logs_path)),
    )

elif cfg.subcommand == "app":
    MyGradioFn(cfg.app, train_parser).launch()

elif cfg.subcommand == "hta":
    MyHta(**cfg.hta.as_dict())()

elif cfg.subcommand == "hpo":
    MyHpo(**cfg.hpo.hpo.as_dict())(hpo_parser, get_train_parser)
