#!/usr/bin/env python

from modeling_CRISPR_diffuser import CRISPRDiffuserConfig, CRISPRDiffuserModel
from datasets import load_dataset
import sys
import os
sys.path.append(os.getcwd())
from AI_models.config import args, logger
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from torch.distributions import Categorical
import torch
import torch.nn.functional as F
from transformers.trainer_utils import EvalPrediction
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from load_data import data_collector

class CRISPRDiffuserTrainerCallback(TrainerCallback):
    def on_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if model.loss and (torch.isnan(model.loss) or torch.isinf(model.loss)):
            control.should_training_stop = True
            return
        for p in model.parameters():
            if p.grad is not None and (p.grad.isnan().any() or p.grad.isinf().any()):
                control.should_training_stop = True
                return

logger.info("load scheduler")
if args.noise_scheduler == "linear":
    from CRISPR_diffuser.linear_scheduler_CRISPR_diffuser import CRISPRDiffuserLinearScheduler
    noise_scheduler = CRISPRDiffuserLinearScheduler(
        num_train_timesteps = args.noise_timesteps
    )
elif args.noise_scheduler == "cosine":
    from CRISPR_diffuser.cosine_scheduler_CRISPR_diffuser import CRISPRDiffuserCosineScheduler
    noise_scheduler = CRISPRDiffuserCosineScheduler(
        num_train_timesteps = args.noise_timesteps,
        cosine_factor = args.cosine_factor
    )
elif args.noise_scheduler == "exp":
    from CRISPR_diffuser.exp_scheduler_CRISPR_diffuser import CRISPRDiffuserExpScheduler
    noise_scheduler = CRISPRDiffuserExpScheduler(
        num_train_timesteps = args.noise_timesteps,
        exp_scale = args.exp_scale,
        exp_base = args.exp_base
    )
elif args.noise_scheduler == "uniform":
    from CRISPR_diffuser.uniform_scheduler_CRISPR_diffuser import CRISPRDiffuserUniformScheduler
    noise_scheduler = CRISPRDiffuserUniformScheduler(
        num_train_timesteps = args.noise_timesteps,
        uniform_scale = args.uniform_scale
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
    name = f"{args.data_name}_{CRISPRDiffuserConfig.model_type}",
    trust_remote_code = True,
    test_ratio = args.test_ratio,
    validation_ratio = args.validation_ratio,
    seed = args.seed
)

logger.info("set trainer arguments")
training_args = TrainingArguments(
    output_dir = args.output_dir / CRISPRDiffuserConfig.model_type / f"{args.data_name}_{CRISPRDiffuserConfig.model_type}",
    seed = args.seed,
    logging_strategy = "epoch",
    eval_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end = True,
    remove_unused_columns = False,
    label_names = CRISPRDiffuserConfig.label_names,
    include_inputs_for_metrics = True
)
training_args.set_dataloader(
    train_batch_size = args.batch_size,
    eval_batch_size = args.batch_size
)
training_args.set_optimizer(
    name = args.optimizer,
    learning_rate = args.learning_rate
)
training_args.set_lr_scheduler(
    name = args.scheduler,
    num_epochs = args.num_epochs,
    warmup_ratio = args.warmup_ratio
)

logger.info("prepare metrics")
def compute_metrics(eval_prediction: EvalPrediction) -> dict:
    random_idx = compute_metrics.rng.integers(len(eval_prediction.predictions))
    compute_metrics.global_step += compute_metrics.epoch_step_nums
    compute_metrics.tb_writer.add_image(
        "p_theta_0",
        F.softmax(
            torch.from_numpy(eval_prediction.predictions[random_idx]).flatten(),
            dim = 0
        ).view(1, compute_metrics.ref2len + 1, -1) ** args.display_scale_factor,
        global_step = compute_metrics.global_step
    )
    compute_metrics.tb_writer.add_image(
        "normalized observation",
        F.normalize(
            torch.from_numpy(eval_prediction.label_ids[random_idx]).flatten(),
            p = 1.0, dim = 0
        ).view(1, compute_metrics.ref2len + 1, -1) ** args.display_scale_factor,
        global_step = compute_metrics.global_step
    )
    return {
        "random_idx": random_idx,
        "x1t": eval_prediction.inputs["x1t"][random_idx],
        "x2t": eval_prediction.inputs["x2t"][random_idx],
        "t": eval_prediction.inputs["t"][random_idx]
    }
compute_metrics.global_step = 0
compute_metrics.rng = np.random.default_rng(args.seed)
compute_metrics.tb_writer = SummaryWriter(training_args.logging_dir)
compute_metrics.epoch_step_nums = int(np.ceil(len(ds["train"]) / args.batch_size))
compute_metrics.ref2len = len(CRISPR_diffuser_model.stationary_sampler2_probs) - 1

logger.info("train model")
stationary_sampler1 = Categorical(probs=CRISPR_diffuser_model.stationary_sampler1_probs.to("cpu"))
stationary_sampler2 = Categorical(probs=CRISPR_diffuser_model.stationary_sampler2_probs.to("cpu"))
trainer = Trainer(
    model = CRISPR_diffuser_model,
    args = training_args,
    train_dataset = ds["train"],
    eval_dataset = ds["validation"],
    data_collator = lambda examples: data_collector(
        examples,
        noise_scheduler,
        stationary_sampler1,
        stationary_sampler2
    ),
    compute_metrics = compute_metrics,
    callbacks = [CRISPRDiffuserTrainerCallback]
)
try:
    trainer.train(resume_from_checkpoint = True)
except ValueError:
    trainer.train()

logger.info("save model")
trainer.save_model()
trainer.create_model_card()
