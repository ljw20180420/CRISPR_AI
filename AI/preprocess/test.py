import numpy as np
import logging
import pathlib
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from typing import Callable
from .dataset import MyDataset


@torch.no_grad()
def test(
    data_collator: Callable,
    model: PreTrainedModel,
    my_dataset: MyDataset,
    metrics: dict,
    trial_name: str,
    output_dir: pathlib.Path,
    batch_size: int,
    logger: logging.Logger,
) -> None:
    logger.info("load test data")
    dl = DataLoader(
        dataset=my_dataset.dataset["test"],
        batch_size=batch_size,
        collate_fn=lambda examples: examples,
    )

    logger.info("test model")
    dfs, accum_sample_idx = [], 0
    for examples in tqdm(dl):
        current_batch_size = len(examples)
        batch = data_collator(examples, output_label=True)
        df = model.eval_output(examples, batch)
        observations = batch["label"]["observation"].cpu().numpy()
        cut1s = np.array([example["cut1"] for example in examples])
        cut2s = np.array([example["cut2"] for example in examples])
        for metric_name, metric_fun in metrics.items():
            loss, loss_num = metric_fun(
                df=df,
                observation=observations,
                cut1=cut1s,
                cut2=cut2s,
            )
            df[f"{metric_name}_loss"] = loss
            df[f"{metric_name}_loss_num"] = loss_num
        df["sample_idx"] = df["sample_idx"] + accum_sample_idx
        accum_sample_idx += current_batch_size
        dfs.append(df)

    pd.concat(dfs).to_csv(
        output_dir
        / data_collator.preprocess
        / model.config.model_type
        / my_dataset.name
        / trial_name
        / "test_result.csv",
        index=False,
    )
