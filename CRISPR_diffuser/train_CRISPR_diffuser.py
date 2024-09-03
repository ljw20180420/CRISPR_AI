#!/usr/bin/env python

from CRISPR_diffuser.configuration_CRISPR_diffuser import CRISPRDiffuserConfig
from CRISPR_diffuser.modeling_CRISPR_diffuser import CRISPRDiffuserModel
import sys
import os
sys.path.append(os.getcwd())
from config import args, logger

logger("initialize model")
CRISPR_diffuser_model = CRISPRDiffuserModel(CRISPRDiffuserConfig(seed=args.seed))