import numpy as np
import os
import pathlib
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Literal
import json
import datasets
from .metric import get_metrics
from .model import get_model
from .utils import get_logger, target_to_epoch


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
        best_epoch = target_to_epoch(
            self.model_path / "checkpoints", target="NonWildTypeCrossEntropy"
        )
        with open(
            self.model_path
            / "checkpoints"
            / f"checkpoint-{best_epoch}"
            / "meta_data.json",
            "r",
        ) as fd:
            meta_data = json.load(fd)
        logger = get_logger(**meta_data["logger"])

        logger.info("load metric")
        metrics = get_metrics(**meta_data["metric"], meta_data=meta_data)

        logger.info("load model")
        model = get_model(**meta_data["model"], meta_data=meta_data)
        checkpoint = torch.load(
            self.model_path
            / "checkpoints"
            / f"checkpoint-{best_epoch}"
            / "checkpoint.pt",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        model.eval()

        logger.info("load test data")
        dataset = datasets.load_from_disk(self.model_path / "datasets")
        dl = DataLoader(
            dataset=dataset["test"],
            batch_size=self.batch_size,
            collate_fn=lambda examples: examples,
        )

        logger.info("test model")
        metric_dfs, accum_sample_idx = [], 0
        for examples in tqdm(dl):
            current_batch_size = len(examples)
            batch = model.data_collator(examples, output_label=True)
            df = model.eval_output(examples, batch)
            observations = batch["label"]["observation"].cpu().numpy()
            cut1s = np.array([example["cut1"] for example in examples])
            cut2s = np.array([example["cut2"] for example in examples])
            metric_df = pd.DataFrame({"sample_idx": np.arange(current_batch_size)})
            for metric_name, metric_fun in metrics.items():
                metric_loss, metric_loss_num = metric_fun(
                    df=df,
                    observation=observations,
                    cut1=cut1s,
                    cut2=cut2s,
                )
                metric_df[f"{metric_name}_loss"] = metric_loss
                metric_df[f"{metric_name}_loss_num"] = metric_loss_num
            metric_df["sample_idx"] = metric_df["sample_idx"] + accum_sample_idx
            accum_sample_idx += current_batch_size
            metric_dfs.append(metric_df)

        logger.info("output results")
        pd.concat(metric_dfs).to_csv(
            self.model_path / "test_result.csv",
            index=False,
        )
