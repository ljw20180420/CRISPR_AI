#!/usr/bin/env python

from CRISPR_diffuser.unet.configuration_CRISPR_diffuser import CRISPRDiffuserConfig
from CRISPR_diffuser.unet.modeling_CRISPR_diffuser import CRISPRDiffuserModel
from datasets import load_dataset
import sys
import os
sys.path.append(os.getcwd())
from config import args, logger
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from torch.distributions import Categorical
import torch
import torch.nn.functional as F
from transformers.trainer_utils import EvalPrediction
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from load_data import data_collector

class CRISPRDiffuserTrainerCallback(TrainerCallback):
    def on_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        assert not model.loss or not torch.isnan(model.loss) and not torch.isinf(model.loss), "loss is nan or inf"
        for p in model.parameters():
            assert not p.grad.isnan().any() and not p.grad.isinf().any(), "gradient is nan or inf"

logger.info("load scheduler")
if args.noise_scheduler == "linear":
    from CRISPR_diffuser.scheduler.linear_scheduler_CRISPR_diffuser import CRISPRDiffuserLinearScheduler
    noise_scheduler = CRISPRDiffuserLinearScheduler(
        num_train_timesteps=args.noise_timesteps
    )
elif args.noise_scheduler == "cosine":
    from CRISPR_diffuser.scheduler.cosine_scheduler_CRISPR_diffuser import CRISPRDiffuserCosineScheduler
    noise_scheduler = CRISPRDiffuserCosineScheduler(
        num_train_timesteps=args.noise_timesteps,
        cosine_factor = args.cosine_factor
    )
elif args.noise_scheduler == "exp":
    from CRISPR_diffuser.scheduler.exp_scheduler_CRISPR_diffuser import CRISPRDiffuserExpScheduler
    noise_scheduler = CRISPRDiffuserExpScheduler(
        num_train_timesteps=args.noise_timesteps,
        exp_scale = args.exp_scale,
        exp_base = args.exp_base
    )
elif args.noise_scheduler == "uniform":
    from CRISPR_diffuser.scheduler.uniform_scheduler_CRISPR_diffuser import CRISPRDiffuserUniformScheduler
    noise_scheduler = CRISPRDiffuserUniformScheduler(
        num_train_timesteps=args.noise_timesteps,
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
    name = f"{args.data_name}_CRISPR_diffuser",
    trust_remote_code = True,
    test_ratio = args.test_ratio,
    validation_ratio = args.validation_ratio,
    seed = args.seed
)

logger.info("train model")
training_args = TrainingArguments(
    output_dir = args.output_dir / CRISPRDiffuserConfig.model_type / f"{args.data_name}_CRISPR_diffuser",
    seed = args.seed,
    logging_strategy = "epoch",
    eval_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end = True,
    remove_unused_columns = False,
    label_names = CRISPR_diffuser_model.config_class.label_names,
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
def compute_metrics(eval_prediction: EvalPrediction, rng: np.random.Generator, tb_writer: SummaryWriter, epoch_step_nums: int, ref2len: int) -> dict:
    random_idx = rng.integers(len(eval_prediction.predictions))
    compute_metrics.global_step += epoch_step_nums
    tb_writer.add_image(
        "p_theta_0",
        F.softmax(
            torch.from_numpy(eval_prediction.predictions[random_idx]).flatten(),
            dim = 0
        ).view(1, ref2len + 1, -1) ** args.display_scale_factor,
        global_step = compute_metrics.global_step
    )
    tb_writer.add_image(
        "normalized observation",
        F.normalize(
            torch.from_numpy(eval_prediction.label_ids[random_idx]).flatten(),
            p = 1.0, dim = 0
        ).view(1, ref2len + 1, -1) ** args.display_scale_factor,
        global_step = compute_metrics.global_step
    )
    return {
        "random_idx": random_idx,
        "x1t": eval_prediction.inputs["x1t"][random_idx],
        "x2t": eval_prediction.inputs["x2t"][random_idx],
        "t": eval_prediction.inputs["t"][random_idx]
    }
compute_metrics.global_step = 0
trainer = Trainer(
    model = CRISPR_diffuser_model,
    args = training_args,
    train_dataset = ds["train"],
    eval_dataset = ds["validation"],
    data_collator = lambda examples: data_collector(
        examples,
        noise_scheduler,
        Categorical(probs=CRISPR_diffuser_model.stationary_sampler1_probs.to("cpu")),
        Categorical(probs=CRISPR_diffuser_model.stationary_sampler2_probs.to("cpu"))
    ),
    compute_metrics=lambda eval_prediction: compute_metrics(
        eval_prediction,
        np.random.default_rng(args.seed),
        SummaryWriter(training_args.logging_dir),
        int(np.ceil(len(ds["train"]) / args.batch_size)),
        len(CRISPR_diffuser_model.stationary_sampler2_probs) - 1
    ),
    callbacks=[CRISPRDiffuserTrainerCallback]
)

try:
    trainer.train(resume_from_checkpoint = True)
    logger.info("push model")
    trainer.push_to_hub()
except AssertionError:
    logger.info("push model")
    trainer.push_to_hub()
except ValueError:
    try:
        trainer.train()
        logger.info("push model")
        trainer.push_to_hub()
    except AssertionError:
        logger.info("push model")
        trainer.push_to_hub()
    