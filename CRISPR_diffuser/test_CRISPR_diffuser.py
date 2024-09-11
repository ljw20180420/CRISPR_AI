#!/usr/bin/env python

from modeling_CRISPR_diffuser import CRISPRDiffuserConfig, CRISPRDiffuserModel
from pipeline_CRISPR_diffuser import CRISPRDiffuserPipeline
from datasets import load_dataset
import datasets
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import pathlib
from load_data import data_collector
import sys
import os
sys.path.append(os.getcwd())
from config import args, logger

logger.info("load scheduler")
scheduler_file = pathlib.Path("CRISPR_diffuser")
if args.noise_scheduler == "linear":
    from linear_scheduler_CRISPR_diffuser import CRISPRDiffuserLinearScheduler
    noise_scheduler = CRISPRDiffuserLinearScheduler(
        num_train_timesteps=args.noise_timesteps
    )
    scheduler_file = scheduler_file / "linear_scheduler_CRISPR_diffuser.py"
elif args.noise_scheduler == "cosine":
    from cosine_scheduler_CRISPR_diffuser import CRISPRDiffuserCosineScheduler
    noise_scheduler = CRISPRDiffuserCosineScheduler(
        num_train_timesteps=args.noise_timesteps,
        cosine_factor = args.cosine_factor
    )
    scheduler_file = scheduler_file / "cosine_scheduler_CRISPR_diffuser.py"
elif args.noise_scheduler == "exp":
    from exp_scheduler_CRISPR_diffuser import CRISPRDiffuserExpScheduler
    noise_scheduler = CRISPRDiffuserExpScheduler(
        num_train_timesteps=args.noise_timesteps,
        exp_scale = args.exp_scale,
        exp_base = args.exp_base
    )
    scheduler_file = scheduler_file / "exp_scheduler_CRISPR_diffuser.py"
elif args.noise_scheduler == "uniform":
    from uniform_scheduler_CRISPR_diffuser import CRISPRDiffuserUniformScheduler
    noise_scheduler = CRISPRDiffuserUniformScheduler(
        num_train_timesteps=args.noise_timesteps,
        uniform_scale = args.uniform_scale
    )
    scheduler_file = scheduler_file / "uniform_scheduler_CRISPR_diffuser.py"
logger.info("load model")
CRISPR_diffuser_model = CRISPRDiffuserModel.from_pretrained(args.output_dir / CRISPRDiffuserConfig.model_type / f"{args.data_name}_{CRISPRDiffuserConfig.model_type}")

logger.info("setup pipeline")
pipe = CRISPRDiffuserPipeline(
    unet=CRISPR_diffuser_model,
    scheduler=noise_scheduler
)

logger.info("load test data")
ds = load_dataset(
    path=args.data_path,
    name=f"{args.data_name}_{CRISPRDiffuserConfig.model_type}",
    split = datasets.Split.TEST,
    trust_remote_code = True,
    test_ratio = args.test_ratio,
    validation_ratio = args.validation_ratio,
    seed = args.seed
)
stationary_sampler1 = Categorical(probs=CRISPR_diffuser_model.stationary_sampler1_probs)
stationary_sampler2 = Categorical(probs=CRISPR_diffuser_model.stationary_sampler2_probs)
test_dataloader = DataLoader(
    dataset=ds,
    batch_size=1,
    collate_fn=lambda examples: data_collector(
        examples,
        noise_scheduler,
        stationary_sampler1,
        stationary_sampler2
    )
)

logger.info("test pipeline")
for batch in test_dataloader:
    x1ts, x2ts, ts = pipe(batch["condition"], batch_size=args.batch_size, record_path=True)
    break

logger.info("push to hub")
pipe.push_to_hub(f"ljw20180420/{args.data_name}_{CRISPRDiffuserConfig.model_type}")
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    repo_id=f"ljw20180420/{args.data_name}_{CRISPRDiffuserConfig.model_type}",
    path_or_fileobj="CRISPR_diffuser/pipeline_CRISPR_diffuser.py",
    path_in_repo="pipeline_CRISPR_diffuser.py"
)
api.upload_folder(
    repo_id=f"ljw20180420/{args.data_name}_{CRISPRDiffuserConfig.model_type}",
    folder_path=args.output_dir / CRISPRDiffuserConfig.model_type / f"{args.data_name}_{CRISPRDiffuserConfig.model_type}",
    path_in_repo="unet",
    ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"]
)
api.upload_file(
    repo_id=f"ljw20180420/{args.data_name}_{CRISPRDiffuserConfig.model_type}",
    path_or_fileobj=scheduler_file,
    path_in_repo=f"scheduler/{scheduler_file.name}"
)
