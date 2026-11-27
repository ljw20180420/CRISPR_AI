import os

import pandas as pd
from common_ai.config import get_train_parser
from jsonargparse import Namespace

from AI.inference import MyInference


def inference(
    infer_df: pd.DataFrame, checkpoints_path: os.PathLike, logs_path: os.PathLike
) -> pd.DataFrame:
    """
    infer_df: dataframe with columns ref, cut, scaffold (spycas9, spymac, ispymac)
    """

    # These values depend on training setting. Do not change them.
    infer_out = MyInference(
        ext1_up=25,
        ext1_down=6,
        ext2_up=6,
        ext2_down=25,
        max_del_size=44,
    )(
        infer_df,
        test_cfg=Namespace(
            checkpoints_path=checkpoints_path,
            logs_path=logs_path,
            target="GreatestCommonCrossEntropy",
            maximize_target=False,
            overwrite={},
        ),
        train_parser=get_train_parser(),
    )

    return infer_out
