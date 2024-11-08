#!/usr/bin/env python

import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from AI_models.CRISPR_diffuser.model import CRISPRDiffuserConfig, CRISPRDiffuserModel
from AI_models.config import args, logger
from AI_models.CRISPR_diffuser.load_data import data_collector, outputs_train
from AI_models.CRISPR_diffuser.scheduler import scheduler
from AI_models.proxy import proxy
import matplotlib.pyplot as plt
import numpy as np

proxy(url="socks5h://127.0.0.1:1080")

logger.info("load scheduler")
noise_scheduler = scheduler()

logger.info("initialize model")
CRISPR_diffuser_model = CRISPRDiffuserModel(CRISPRDiffuserConfig(
    channels = [11] + args.unet_channels + [1],
    MCMC_corrector_factor = args.MCMC_corrector_factor,
    seed=args.seed
))

logger.info("loading data")
ds = load_dataset(
    path = f"{args.owner}/CRISPR_data",
    name = f"SX_spcas9_{CRISPRDiffuserConfig.model_type}",
    trust_remote_code = True,
    test_ratio = args.test_ratio,
    validation_ratio = args.validation_ratio,
    seed = args.seed
)


# train_counts = [sum(example['ob_val']) for example in ds[datasets.Split.TRAIN]]
# validation_counts = [sum(example['ob_val']) for example in ds[datasets.Split.VALIDATION]]
# gm = geometric_mean(train_counts + validation_counts)
breakpoint()
plt.boxplot(np.array(train_counts + validation_counts) / gm)
plt.show()


# stationary_sampler1 = Categorical(probs=CRISPR_diffuser_model.stationary_sampler1_probs.to("cpu"))
# stationary_sampler2 = Categorical(probs=CRISPR_diffuser_model.stationary_sampler2_probs.to("cpu"))
# train_dataloader = DataLoader(
#     dataset=ds[datasets.Split.TRAIN],
#     batch_size=args.batch_size,
#     collate_fn=lambda examples: data_collector(
#         examples,
#         noise_scheduler,
#         stationary_sampler1,
#         stationary_sampler2,
#         outputs_train
#     )
# )

# for batch in train_dataloader:
#     CRISPR_diffuser_model.forward(batch["x1t_x2t_t"], batch["condition"], batch["observation"])
#     break
