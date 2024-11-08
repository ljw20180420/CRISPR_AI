#!/usr/bin/env python

from datasets import load_dataset
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from torch.distributions import Categorical
import torch
import torch.nn.functional as F
from transformers.trainer_utils import EvalPrediction
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from statistics import geometric_mean
from .model import CRISPRDiffuserConfig, CRISPRDiffuserModel
from ..config import get_config, get_logger
from .load_data import data_collector, outputs_train
from .scheduler import scheduler

args = get_config(config_file="config_CRISPR_diffuser.ini")
logger = get_logger(args)

class CRISPRDiffuserTrainerCallback(TrainerCallback):
    def on_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if model.loss and (torch.isnan(model.loss) or torch.isinf(model.loss)):
            control.should_training_stop = True
            return
        for p in model.parameters():
            if p.grad is not None and (p.grad.isnan().any() or p.grad.isinf().any()):
                control.should_training_stop = True
                return

def train(data_name=args.data_name):
    logger.info("loading data")
    ds = load_dataset(
        path = f"{args.owner}/CRISPR_data",
        name = f"{data_name}_{CRISPRDiffuserConfig.model_type}",
        trust_remote_code = True,
        test_ratio = args.test_ratio,
        validation_ratio = args.validation_ratio,
        seed = args.seed
    )

    logger.info("estimate count_normalize")
    train_counts = ds['train'].map(
        lambda examples: {'count': [sum(example) for example in examples]},
        batched=True,
        input_columns=['ob_val'],
        remove_columns=['ref1', 'ref2', 'cut1', 'cut2', 'mh_ref1', 'mh_ref2', 'mh_val', 'ob_ref1', 'ob_ref2', 'ob_val']
    )
    validation_counts = ds['validation'].map(
        lambda examples: {'count': [sum(example) for example in examples]},
        batched=True,
        input_columns=['ob_val'],
        remove_columns=['ref1', 'ref2', 'cut1', 'cut2', 'mh_ref1', 'mh_ref2', 'mh_val', 'ob_ref1', 'ob_ref2', 'ob_val']
    )
    # count_normalize = geometric_mean(train_counts['count'] + validation_counts['count'])
    count_normalize = max(train_counts['count'] + validation_counts['count'])
        
    logger.info("load scheduler")
    noise_scheduler = scheduler(
        noise_scheduler=args.noise_scheduler,
        noise_timesteps=args.noise_timesteps,
        cosine_factor=args.cosine_factor,
        exp_scale=args.exp_scale,
        exp_base=args.exp_base,
        uniform_scale=args.uniform_scale
    )

    logger.info("initialize model")
    CRISPRDiffuserConfig.register_for_auto_class()
    CRISPRDiffuserModel.register_for_auto_class()
    CRISPR_diffuser_model = CRISPRDiffuserModel(CRISPRDiffuserConfig(
        count_normalize = count_normalize,
        channels = [11] + args.unet_channels + [1],
        MCMC_corrector_factor = args.MCMC_corrector_factor,
        seed = args.seed
    ))

    logger.info("set trainer arguments")
    training_args = TrainingArguments(
        output_dir = args.output_dir / CRISPRDiffuserConfig.model_type / f"{data_name}_{CRISPRDiffuserConfig.model_type}",
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
            stationary_sampler2,
            outputs_train
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
