#!/usr/bin/env python

from CRISPR_diffuser.unet.configuration_CRISPR_diffuser import CRISPRDiffuserConfig
from CRISPR_diffuser.unet.modeling_CRISPR_diffuser import CRISPRDiffuserModel
from datasets import load_dataset
import sys
import os
sys.path.append(os.getcwd())
from config import args, logger

logger.info("load scheduler")
if args.noise_scheduler == "linear":
    from CRISPR_diffuser.scheduler.linear_scheduler_CRISPR_diffuser import CRISPRDiffuserLinearScheduler
    noise_scheduler = CRISPRDiffuserLinearScheduler(
        num_train_timesteps=args.noise_timesteps,
        device=args.device
    )
elif args.noise_scheduler == "cosine":
    from CRISPR_diffuser.scheduler.cosine_scheduler_CRISPR_diffuser import CRISPRDiffuserCosineScheduler
    noise_scheduler = CRISPRDiffuserCosineScheduler(
        num_train_timesteps=args.noise_timesteps,
        cosine_factor = args.cosine_factor,
        device=args.device
    )
elif args.noise_scheduler == "exp":
    from CRISPR_diffuser.scheduler.exp_scheduler_CRISPR_diffuser import CRISPRDiffuserExpScheduler
    noise_scheduler = CRISPRDiffuserExpScheduler(
        num_train_timesteps=args.noise_timesteps,
        exp_scale = args.exp_scale,
        exp_base = args.exp_base,
        device=args.device
    )
elif args.noise_scheduler == "uniform":
    from CRISPR_diffuser.scheduler.uniform_scheduler_CRISPR_diffuser import CRISPRDiffuserUniformScheduler
    noise_scheduler = CRISPRDiffuserUniformScheduler(
        num_train_timesteps=args.noise_timesteps,
        uniform_scale = args.uniform_scale,
        device=args.device
    )

logger.info("initialize model")
CRISPRDiffuserConfig.register_for_auto_class()
CRISPRDiffuserModel.register_for_auto_class()
CRISPR_diffuser_model = CRISPRDiffuserModel(CRISPRDiffuserConfig(
    channels = [13] + args.unet_channels + [1],
    MCMC_corrector_factor = args.MCMC_corrector_factor,
    seed=args.seed
))

logger.info("loading data")
ds = load_dataset(
    path = args.data_path,
    name = f"{args.data_name}_CRISPR_diffuser",
    trust_remote_code = True,
    test_ratio = args.test_ratio,
    validation_ratio = args.validation_ratio,
    seed = args.seed
)