import pandas as pd
import datasets
import torch
import jsonargparse
from common_ai.test import MyTest
from common_ai.inference import MyInferenceAbstract


class MyInference(MyInferenceAbstract):

    def __init__(
        self,
        ext1_up: int,
        ext1_down: int,
        ext2_up: int,
        ext2_down: int,
        max_del_size: int,
        **kwargs,
    ):
        """Inference arguments.

        Args:
            ext1_up: upstream limit of the resection of the upstream end.
            ext1_down: downstream limit of the templated insertion of the upstream end.
            ext2_up: upstream limit of the templated insertion of the downstream end.
            ext2_down: downstream limit of the resection of the downstream end.
            max_del_size: maximal deletion size.
        """
        # Lindel allow deletion to cross cut site for 2bps. FOREcasT needs ±3bps around deletion boundaries.
        self.ext_up = max(ext1_up, ext2_up, max_del_size + max(2, 3))
        self.ext_down = max(ext1_down, ext2_down, max_del_size + max(2, 3))
        self.scaffolds = {
            "spcas9": "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGCTTTTTTG",
            "spymac": "GTTTCAGAGCTATGCTGGAAACAGCATAGCAAGTTGAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGCTTTTTTG",
        }

    @torch.no_grad()
    def __call__(
        self,
        infer_df: pd.DataFrame,
        test_cfg: jsonargparse.Namespace,
        train_parser: jsonargparse.ArgumentParser,
    ) -> pd.DataFrame:
        # load model for the first call
        if (
            not hasattr(self, "logger")
            or not hasattr(self, "model")
            or not hasattr(self, "my_generator")
            or not hasattr(self, "batch_size")
        ):
            _, train_cfg, self.logger, self.model, self.my_generator = MyTest(
                **test_cfg.as_dict()
            ).load_model(train_parser)
            self.batch_size = train_cfg.train.batch_size

        self.logger.info("validate and prepare dataset")
        infer_df = infer_df.copy()
        infer_df["scaffold"] = infer_df["scaffold"].map(self.scaffolds)
        for ref, cut in zip(infer_df["ref"], infer_df["cut"]):
            assert (
                cut >= self.ext_up
            ), f"sequence upstream to cut should be at least {self.ext_up} bps"
            assert (
                len(ref) - cut >= self.ext_down
            ), f"sequence downstream to cut should be at least {self.ext_down} bps"
        infer_df = infer_df.rename(columns={"ref": "ref1", "cut": "cut1"})
        infer_df["ref2"] = infer_df["ref1"]
        infer_df["cut2"] = infer_df["cut1"]

        inference_dataloader = torch.utils.data.DataLoader(
            dataset=datasets.Dataset.from_pandas(infer_df),
            batch_size=self.batch_size,
            collate_fn=lambda examples: examples,
        )

        self.logger.info("inference")
        dfs, accum_sample_idx = [], 0
        for examples in inference_dataloader:
            batch = self.model.data_collator(
                examples, output_label=False, my_generator=self.my_generator
            )
            df = self.model.eval_output(examples, batch, self.my_generator)
            df["sample_idx"] += accum_sample_idx
            accum_sample_idx += len(examples)
            dfs.append(df)

        return pd.concat(dfs)
