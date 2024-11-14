#!/usr/bin/env python

from datasets import load_dataset
from torch.utils.data import DataLoader
import datasets
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
import numpy as np
import pickle
from .model import CRISPRTransformerConfig, CRISPRTransformerModel
from ..config import get_config, get_logger
from .load_data import data_collector, outputs_train

args = get_config(config_file="config_CRISPR_transformer.ini")
logger = get_logger(args)

def train(data_name=args.data_name):
    logger.info("loading data")
    ds = load_dataset(
        path = f"{args.owner}/CRISPR_data",
        name = f"{data_name}_{CRISPRTransformerConfig.model_type}",
        trust_remote_code = True,
        test_ratio = args.test_ratio,
        validation_ratio = args.validation_ratio,
        seed = args.seed
    )

    logger.info("initialize model")
    CRISPRTransformerConfig.register_for_auto_class()
    CRISPRTransformerModel.register_for_auto_class()
    CRISPR_transformer_model = CRISPRTransformerModel(
        CRISPRTransformerConfig(
            hidden_size = args.hidden_size, # model embedding dimension
            num_hidden_layers = args.num_hidden_layers, # number of EncoderLayer
            num_attention_heads = args.num_attention_heads, # number of attention heads
            intermediate_size = args.intermediate_size, # FeedForward intermediate dimension size
            hidden_dropout_prob = args.hidden_dropout_prob, # The dropout probability for all fully connected layers in the embeddings, encoder, and pooler
            attention_probs_dropout_prob = args.attention_probs_dropout_prob # The dropout ratio for the attention probabilities
        )
    )

    logger.info("train model")
    training_args = TrainingArguments(
        output_dir = args.output_dir / CRISPRTransformerConfig.model_type / f"{data_name}_{CRISPRTransformerConfig.model_type}",
        seed = args.seed,
        logging_strategy = "epoch",
        eval_strategy = "epoch",
        save_strategy = "epoch",
        load_best_model_at_end = True,
        remove_unused_columns = False,
        label_names = CRISPRTransformerConfig.label_names
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
    trainer = Trainer(
        model = CRISPR_transformer_model,
        args = training_args,
        train_dataset = ds["train"],
        eval_dataset = ds["validation"],
        data_collator = lambda examples: data_collector(examples, outputs_train)
    )
    try:
        trainer.train(resume_from_checkpoint = True)
    except ValueError:
        trainer.train()

    logger.info("save model")
    trainer.save_model()
    trainer.create_model_card()
