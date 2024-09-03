#!/usr/bin/env python

from CRISPR_diffuser.configuration_CRISPR_diffuser import CRISPRDiffuserConfig
from CRISPR_diffuser.modeling_CRISPR_diffuser import CRISPRDiffuserModel
from datasets import load_dataset
import sys
import os
sys.path.append(os.getcwd())
from config import args, logger

logger.info("initialize model")
CRISPRDiffuserConfig.register_for_auto_class()
CRISPRDiffuserModel.register_for_auto_class()
CRISPR_diffuser_model = CRISPRDiffuserModel(CRISPRDiffuserConfig(
    channels = [13] + args.unet_channels + [1]
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