import numpy as np
import logging
import os
import pathlib
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Literal
from .io import target_to_epoch, load_checkpoint, load_my_dataset


class MyTest:
    def __init__(
        self,
        model_path: os.PathLike,
        batch_size: int,
        device: Literal["cpu", "cuda"],
    ) -> None:
        """Test arguments.

        Args:
            model_path: Path to the model.
            batch_size: Batch size.
            device: Device.
        """
        self.model_path = pathlib.Path(os.fspath(model_path))
        self.batch_size = batch_size
        self.device = device

    @torch.no_grad()
    def __call__(self) -> None:
        epoch = target_to_epoch(
            self.model_path / "checkpoints", target="NonWildTypeCrossEntropy"
        )
        (
            _,
            metrics,
            model,
            _,
            _,
            _,
            my_logger,
        ) = load_checkpoint(self.model_path / "checkpoints" / f"checkpoint-{epoch}")
        my_dataset = load_my_dataset(self.model_path)

        my_logger.info("load test data")
        dl = DataLoader(
            dataset=my_dataset.dataset["test"],
            batch_size=self.batch_size,
            collate_fn=lambda examples: examples,
        )

        my_logger.info("test model")
        dfs, accum_sample_idx = [], 0
        for examples in tqdm(dl):
            current_batch_size = len(examples)
            batch = model.data_collator(examples, output_label=True)
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

        my_logger.info("output results")
        pd.concat(dfs).to_csv(
            self.model_path / "test_result.csv",
            index=False,
        )
