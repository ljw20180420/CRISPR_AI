import torch
import logging
import pathlib
from datasets import load_dataset, Features, Value
from torch.utils.data import DataLoader
from diffusers import DiffusionPipeline
from tqdm import tqdm
import pandas as pd


@torch.no_grad()
def inference(
    preprocess: str,
    model_name: str,
    inference_data: pathlib.Path,
    inference_output: pathlib.Path,
    data_name: str,
    owner: str,
    batch_size: int,
    device: str,
    logger: logging.Logger,
) -> None:
    logger.info("load inference data")
    dl = DataLoader(
        dataset=load_dataset(
            "json",
            data_files=inference_data.as_posix(),
            features=Features({"ref": Value("string"), "cut": Value("int64")}),
        )["train"],
        batch_size=batch_size,
        collate_fn=lambda examples: examples,
    )

    logger.info("load pipeline")
    pipe = DiffusionPipeline.from_pretrained(
        f"{owner}/{preprocess}_{model_name}_{data_name}",
        trust_remote_code=True,
        custom_pipeline=f"{owner}/{preprocess}_{model_name}_{data_name}",
    )
    if hasattr(pipe, "core_model"):
        pipe.core_model.to(device)
    if hasattr(pipe, "auxilary_model"):
        pipe.auxilary_model.load_auxilary(
            f"{owner}/{preprocess}_{model_name}_{data_name}/auxilary_model/auxilary.pkl"
        )

    dfs, accum_sample_idx = [], 0
    for examples in tqdm(dl):
        if preprocess == "DeepHF":
            # Use the common scaffold for spcas9.
            for example in examples:
                example["scaffold"] = (
                    "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGCTTTTTTG"
                )
        current_batch_size = len(examples)
        for example in examples:
            ref, cut = example.pop("ref"), example.pop("cut")
            example["ref1"] = example["ref2"] = ref
            example["cut1"] = example["cut2"] = cut
        df = pipe(
            examples=examples,
            output_label=False,
            metric=None,
        )
        df["sample_idx"] = df["sample_idx"] + accum_sample_idx
        accum_sample_idx += current_batch_size
        dfs.append(df)

    pd.concat(dfs).to_csv(inference_output, index=False)
