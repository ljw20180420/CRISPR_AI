#!/usr/bin/env python

from datasets import load_dataset
import datasets
from torch.utils.data import DataLoader
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import pickle
from .pipeline import inDelphiPipeline
from .model import inDelphiConfig, inDelphiModel
from ..config import args, logger
from .load_data import data_collector, outputs_test

def test():
    logger.info("load model")
    inDelphi_model = inDelphiModel.from_pretrained(args.output_dir / inDelphiConfig.model_type / f"{args.data_name}_{inDelphiConfig.model_type}")
    # remove parent module name
    inDelphi_model.__module__ = inDelphi_model.__module__.split(".")[-1]

    logger.info("setup pipeline")
    with open(args.output_dir / inDelphiConfig.model_type / f"{args.data_name}_{inDelphiConfig.model_type}" / "insertion_model.pkl", "rb") as fd:
        onebp_features, insert_probabilities, m654 = pickle.load(fd)
    pipe = inDelphiPipeline(inDelphi_model, onebp_features, insert_probabilities, m654)
    pipe.inDelphi_model.to(args.device)

    logger.info("load test data")
    ds = load_dataset(
        path = args.data_path,
        name = f"{args.data_name}_{inDelphiConfig.model_type}",
        split = datasets.Split.TEST,
        trust_remote_code = True,
        test_ratio = args.test_ratio,
        validation_ratio = args.validation_ratio,
        seed = args.seed,
        DELLEN_LIMIT = inDelphi_model.DELLEN_LIMIT
    )
    test_dataloader = DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        collate_fn=lambda examples: data_collector(examples, args.DELLEN_LIMIT, outputs_test)
    )

    logger.info("test pipeline")
    for batch in test_dataloader:
        output = pipe(batch)

    logger.info("push to hub")
    pipe.push_to_hub(f"ljw20180420/{args.data_name}_{inDelphiConfig.model_type}")
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        repo_id=f"ljw20180420/{args.data_name}_{inDelphiConfig.model_type}",
        path_or_fileobj="AI_models/inDelphi/pipeline.py",
        path_in_repo="pipeline.py"
    )
    api.upload_folder(
        repo_id=f"ljw20180420/{args.data_name}_{inDelphiConfig.model_type}",
        folder_path=args.output_dir / inDelphiConfig.model_type / f"{args.data_name}_{inDelphiConfig.model_type}",
        path_in_repo="inDelphi_model",
        ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"]
    )
