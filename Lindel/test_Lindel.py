#!/usr/bin/env python

from modeling_Lindel import LindelConfig, LindelModel
from pipeline_Lindel import LindelPipeline
from datasets import load_dataset
import datasets
from torch.utils.data import DataLoader
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from load_data import data_collector
import sys
import os
sys.path.append(os.getcwd())
from config import args, logger

logger.info("load model")
Lindel_models = {
    f"{model}_model": LindelModel.from_pretrained(args.output_dir / LindelConfig.model_type / f"{args.data_name}_{LindelConfig.model_type}_{model}")
    for model in ["indel", "ins", "del"]
}

logger.info("setup pipeline")
pipe = LindelPipeline(**Lindel_models)

logger.info("load test data")
ds = load_dataset(
    path=args.data_path,
    name=f"{args.data_name}_{LindelConfig.model_type}",
    split = datasets.Split.TEST,
    trust_remote_code = True,
    test_ratio = args.test_ratio,
    validation_ratio = args.validation_ratio,
    seed = args.seed,
    Lindel_dlen = args.Lindel_dlen,
    Lindel_mh_len = args.Lindel_mh_len
)
test_dataloader = DataLoader(
    dataset=ds,
    batch_size=args.batch_size,
    collate_fn=lambda examples: data_collector(examples, Lindel_mh_len=args.Lindel_mh_len)
)

logger.info("test pipeline")
for batch in test_dataloader:
    result = pipe(
        input_indel = batch["input_indel"],
        input_ins = batch["input_ins"],
        input_del = batch["input_del"]
    )
    break

logger.info("push to hub")
pipe.push_to_hub(f"ljw20180420/{args.data_name}_{LindelConfig.model_type}")
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    repo_id=f"ljw20180420/{args.data_name}_{LindelConfig.model_type}",
    path_or_fileobj="Lindel/pipeline_Lindel.py",
    path_in_repo="pipeline_Lindel.py"
)
for model in ["indel", "ins", "del"]:
    api.upload_folder(
        repo_id=f"ljw20180420/{args.data_name}_{LindelConfig.model_type}",
        folder_path=args.output_dir / LindelConfig.model_type / f"{args.data_name}_{LindelConfig.model_type}_{model}",
        path_in_repo=f"{model}_model",
        ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"]
    )