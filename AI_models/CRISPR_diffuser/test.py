#!/usr/bin/env python

from requests.exceptions import ConnectionError
from datasets import load_dataset
import datasets
from torch.utils.data import DataLoader
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from .model import CRISPRDiffuserConfig, CRISPRDiffuserModel
from .pipeline import CRISPRDiffuserPipeline
from .load_data import data_collector, outputs_test
from ..config import get_config, get_logger
from .scheduler import scheduler

args = get_config(config_file="config_CRISPR_diffuser.ini")
logger = get_logger(args)

def test(data_name=args.data_name):
    logger.info("load scheduler")
    noise_scheduler = scheduler(
        noise_scheduler=args.noise_scheduler,
        noise_timesteps=args.noise_timesteps,
        cosine_factor=args.cosine_factor,
        exp_scale=args.exp_scale,
        exp_base=args.exp_base,
        uniform_scale=args.uniform_scale
    )
    # remove parent module name
    noise_scheduler.__module__ = noise_scheduler.__module__.split(".")[-1]
    
    logger.info("load model")
    CRISPR_diffuser_model = CRISPRDiffuserModel.from_pretrained(args.output_dir / CRISPRDiffuserConfig.model_type / f"{data_name}_{CRISPRDiffuserConfig.model_type}")
    # remove parent module name
    CRISPR_diffuser_model.__module__ = CRISPR_diffuser_model.__module__.split(".")[-1]

    logger.info("setup pipeline")
    pipe = CRISPRDiffuserPipeline(
        unet=CRISPR_diffuser_model,
        scheduler=noise_scheduler
    )
    pipe.unet.to(args.device)

    logger.info("load test data")
    ds = load_dataset(
        path=f"{args.owner}/CRISPR_data",
        name=f"{data_name}_{CRISPRDiffuserConfig.model_type}",
        split = datasets.Split.TEST,
        trust_remote_code = True,
        test_ratio = args.test_ratio,
        validation_ratio = args.validation_ratio,
        seed = args.seed
    )
    test_dataloader = DataLoader(
        dataset=ds,
        batch_size=1,
        collate_fn=lambda examples: data_collector(
            examples,
            noise_scheduler,
            pipe.stationary_sampler1,
            pipe.stationary_sampler2,
            outputs_test
        )
    )

    logger.info("test pipeline")
    for batch in test_dataloader:
        x1ts, x2ts, ts = pipe(batch, batch_size=args.batch_size, record_path=True)
        break

    while True:
        try:
            logger.info("push to hub")
            pipe.push_to_hub(f"{args.owner}/{data_name}_{CRISPRDiffuserConfig.model_type}")
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                repo_id=f"{args.owner}/{data_name}_{CRISPRDiffuserConfig.model_type}",
                path_or_fileobj="AI_models/CRISPR_diffuser/pipeline.py",
                path_in_repo="pipeline.py"
            )
            api.upload_folder(
                repo_id=f"{args.owner}/{data_name}_{CRISPRDiffuserConfig.model_type}",
                folder_path=args.output_dir / CRISPRDiffuserConfig.model_type / f"{data_name}_{CRISPRDiffuserConfig.model_type}",
                path_in_repo="unet",
                ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"]
            )
            api.upload_file(
                repo_id=f"{args.owner}/{data_name}_{CRISPRDiffuserConfig.model_type}",
                path_or_fileobj="AI_models/CRISPR_diffuser/scheduler.py",
                path_in_repo=f"scheduler/scheduler.py"
            )
            break
        except ConnectionError as err:
            print(err)
            print("retry")
