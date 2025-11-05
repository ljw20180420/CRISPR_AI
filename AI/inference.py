import pandas as pd
import datasets
import torch
import jsonargparse
import tqdm
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
        # Lindel allow deletion to cross cut site for 2bps
        self.ext_up = max(ext1_up, ext2_up, max_del_size + 2)
        self.ext_down = max(ext1_down, ext2_down, max_del_size + 2)
        self.scaffolds = {
            "spcas9": "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGCTTTTTTG",
            "spymac": "GTTTCAGAGCTATGCTGGAAACAGCATAGCAAGTTGAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGCTTTTTTG",
        }

    def load_model(
        self, cfg: jsonargparse.Namespace, train_parser: jsonargparse.ArgumentParser
    ) -> MyInferenceAbstract:
        _, self.cfg, self.logger, self.model, self.my_generator = MyTest(
            **cfg.as_dict()
        ).load_model(train_parser)

        return self

    @torch.no_grad()
    def __call__(self, df_infer: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("validate and prepare dataset")
        df_infer = df_infer.copy()
        df_infer["scaffold"] = df_infer["scaffold"].map(self.scaffolds)
        for ref, cut in zip(df_infer["ref"], df_infer["cut"]):
            assert (
                cut >= self.ext_up
            ), f"sequence upstream to cut should be at least {self.ext_up} bps"
            assert (
                len(ref) - cut >= self.ext_down
            ), f"sequence downstream to cut should be at least {self.ext_down} bps"
        df_infer = df_infer.rename(columns={"ref": "ref1", "cut": "cut1"})
        df_infer["ref2"] = df_infer["ref1"]
        df_infer["cut2"] = df_infer["cut1"]

        inference_dataloader = torch.utils.data.DataLoader(
            dataset=datasets.Dataset.from_pandas(df_infer),
            batch_size=self.cfg.train.batch_size,
            collate_fn=lambda examples: examples,
        )

        self.logger.info("inference")
        accum_sample_idx = 0
        for examples in tqdm.tqdm(inference_dataloader):
            batch = self.model.data_collator(
                examples, output_label=False, my_generator=self.my_generator
            )
            df = self.model.eval_output(examples, batch, self.my_generator)
            df["sample_idx"] += accum_sample_idx
            accum_sample_idx += len(examples)

            yield df
