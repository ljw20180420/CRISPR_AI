import torch
import logging
import pathlib
from datasets import load_dataset, Features, Value
from torch.utils.data import DataLoader
from diffusers import DiffusionPipeline
from tqdm import tqdm
import pandas as pd
from .model import FOREcasTConfig
from .load_data import data_collator


@torch.no_grad()
def data_collator_inference(
    examples: list[dict],
    pre_calculated_features: tuple,
    ref1len: int,
    ref2len: int,
    max_del_size: int,
) -> dict:
    for example in examples:
        ref, cut = example.pop("ref"), example.pop("cut")
        assert (
            len(ref) >= ref1len and len(ref) >= ref2len
        ), f"ref of length {len(ref)} is too short, please decrease ref1len={ref1len} and/or ref2len={ref2len} in inference arguments"
        assert (
            cut <= ref1len and len(ref) - cut <= ref2len
        ), f"ref1len={ref1len} and/or ref2len={ref2len} is too short, please increase them to cover cut site {cut}"
        assert (
            cut >= max_del_size
        ), f"ref upstream to cut ({cut}) is less than max_del_size ({max_del_size}), extend ref to upstream"
        assert (
            len(ref) - cut >= max_del_size
        ), f"ref downstream to cut ({len(ref) - cut}) is less than max_del_size ({max_del_size}), extend ref to downstream"
        example["ref1"] = example["ref2"] = ref
        example["cut1"] = example["cut2"] = cut
    return data_collator(examples, pre_calculated_features, False)


@torch.no_grad()
def inference(
    data_name: str,
    inference_data: pathlib.Path,
    inference_output: pathlib.Path,
    ref1len: int,
    ref2len: int,
    owner: str,
    batch_size: int,
    device: str,
    logger: logging.Logger,
) -> None:
    logger.info("load inference data")
    ds = load_dataset(
        "json",
        data_files=inference_data,
        features=Features({"ref": Value("string"), "cut": Value("int64")}),
    )["train"]

    logger.info("load pipeline")
    pipe = DiffusionPipeline.from_pretrained(
        "%s/%s_%s" % (owner, data_name, FOREcasTConfig.model_type),
        trust_remote_code=True,
        custom_pipeline="%s/%s_%s" % (owner, data_name, FOREcasTConfig.model_type),
    )
    pipe.FOREcasT_model.to(device)

    inference_dataloader = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        collate_fn=lambda examples, pre_calculated_features=pipe.FOREcasT_model.pre_calculated_features, ref1len=ref1len, ref2len=ref2len, max_del_size=pipe.FOREcasT_model.max_del_size: data_collator_inference(
            examples, pre_calculated_features, ref1len, ref2len, max_del_size
        ),
    )

    dfs = []
    for batch in tqdm(inference_dataloader):
        dfs.append(pd.DataFrame(pipe(batch)))

    pd.concat(dfs).to_csv(inference_output, index=False)
