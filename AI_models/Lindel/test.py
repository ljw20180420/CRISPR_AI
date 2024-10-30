#!/usr/bin/env python

from datasets import load_dataset
import datasets
from torch.utils.data import DataLoader
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from .model import LindelConfig, LindelModel
from .pipeline import LindelPipeline
from .load_data import data_collector, outputs_test
from ..config import args, logger

def test(owner="ljw20180420", data_name="SX_spcas9"):
    logger.info("load model")
    Lindel_models = {
        f"{model}_model": LindelModel.from_pretrained(args.output_dir / LindelConfig.model_type / f"{data_name}_{LindelConfig.model_type}_{model}")
        for model in ["indel", "ins", "del"]
    }
    # remove parent module name
    for model in Lindel_models.values():
        model.__module__ = model.__module__.split(".")[-1]

    logger.info("setup pipeline")
    pipe = LindelPipeline(**Lindel_models)
    pipe.indel_model.to(args.device)
    pipe.ins_model.to(args.device)
    pipe.del_model.to(args.device)

    logger.info("load test data")
    ds = load_dataset(
        path=f"{owner}/CRISPR_data",
        name=f"{data_name}_{LindelConfig.model_type}",
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
        collate_fn=lambda examples: data_collector(examples, args.Lindel_dlen, args.Lindel_mh_len, outputs_test)
    )

    logger.info("test pipeline")
    for batch in test_dataloader:
        output = pipe(batch)

    logger.info("push to hub")
    pipe.push_to_hub(f"{owner}/{data_name}_{LindelConfig.model_type}")
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        repo_id=f"{owner}/{data_name}_{LindelConfig.model_type}",
        path_or_fileobj="AI_models/Lindel/pipeline.py",
        path_in_repo="pipeline.py"
    )
    for model in ["indel", "ins", "del"]:
        api.upload_folder(
            repo_id=f"{owner}/{data_name}_{LindelConfig.model_type}",
            folder_path=args.output_dir / LindelConfig.model_type / f"{data_name}_{LindelConfig.model_type}_{model}",
            path_in_repo=f"{model}_model",
            ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"]
        )
