#!/usr/bin/env python

from datasets import load_dataset
import datasets
from torch.utils.data import DataLoader
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from .model import FOREcasTConfig, FOREcasTModel
from .pipeline import FOREcasTPipeline
from .load_data import data_collector
from ..config import get_config, get_logger

args = get_config(config_file="config_FOREcasT.ini")
logger = get_logger(args)

def test(data_name=args.data_name):
    logger.info("load model")
    FOREcasT_model = FOREcasTModel.from_pretrained(args.output_dir / FOREcasTConfig.model_type / f"{data_name}_{FOREcasTConfig.model_type}")
    # remove parent module name
    FOREcasT_model.__module__ = FOREcasT_model.__module__.split(".")[-1]

    logger.info("setup pipeline")
    pipe = FOREcasTPipeline(FOREcasT_model, args.FOREcasT_MAX_DEL_SIZE)
    pipe.FOREcasT_model.to(args.device)

    logger.info("load test data")
    ds = load_dataset(
        path=f"{args.owner}/CRISPR_data",
        name=f"{data_name}_{FOREcasTConfig.model_type}",
        split = datasets.Split.TEST,
        trust_remote_code = True,
        test_ratio = args.test_ratio,
        validation_ratio = args.validation_ratio,
        seed = args.seed,
        FOREcasT_MAX_DEL_SIZE = args.FOREcasT_MAX_DEL_SIZE
    )
    test_dataloader = DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        collate_fn=data_collector
    )

    logger.info("test pipeline")
    for batch in test_dataloader:
        output = pipe(batch)

    logger.info("push to hub")
    pipe.push_to_hub(f"{args.owner}/{data_name}_{FOREcasTConfig.model_type}")
    from huggingface_hub import HfApi
    api = HfApi()
    while True:
        try:
            api.upload_file(
                repo_id=f"{args.owner}/{data_name}_{FOREcasTConfig.model_type}",
                path_or_fileobj="AI_models/FOREcasT/pipeline.py",
                path_in_repo="pipeline.py"
            )
            api.upload_folder(
                repo_id=f"{args.owner}/{data_name}_{FOREcasTConfig.model_type}",
                folder_path=args.output_dir / FOREcasTConfig.model_type / f"{data_name}_{FOREcasTConfig.model_type}",
                path_in_repo="FOREcasT_model",
                ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"]
            )
            break
        except Exception as err:
            print(err)
            print("retry")    
