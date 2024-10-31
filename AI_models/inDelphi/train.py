#!/usr/bin/env python

from datasets import load_dataset
from torch.utils.data import DataLoader
import datasets
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
import numpy as np
import pickle
from .model import inDelphiConfig, inDelphiModel
from ..config import args, logger
from .load_data import data_collector, outputs_train_deletion, outputs_train_insertion

def train_deletion(data_name=args.data_name):
    logger.info("loading data")
    ds = load_dataset(
        path = f"{args.owner}/CRISPR_data",
        name = f"{data_name}_{inDelphiConfig.model_type}",
        trust_remote_code = True,
        test_ratio = args.test_ratio,
        validation_ratio = args.validation_ratio,
        seed = args.seed,
        DELLEN_LIMIT=args.DELLEN_LIMIT
    )

    logger.info("initialize model")
    inDelphiConfig.register_for_auto_class()
    inDelphiModel.register_for_auto_class()
    inDelphi_model = inDelphiModel(inDelphiConfig(DELLEN_LIMIT=args.DELLEN_LIMIT, seed=args.seed, train_size=len(ds[datasets.Split.TRAIN])))

    logger.info("train model")
    training_args = TrainingArguments(
        output_dir = args.output_dir / inDelphiConfig.model_type / f"{data_name}_{inDelphiConfig.model_type}",
        seed = args.seed,
        logging_strategy = "epoch",
        eval_strategy = "epoch",
        save_strategy = "epoch",
        load_best_model_at_end = True,
        remove_unused_columns = False,
        label_names = inDelphiConfig.label_names
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
        model = inDelphi_model,
        args = training_args,
        train_dataset = ds["train"],
        eval_dataset = ds["validation"],
        data_collator = lambda examples: data_collector(examples, args.DELLEN_LIMIT, outputs_train_deletion)
    )
    try:
        trainer.train(resume_from_checkpoint = True)
    except ValueError:
        trainer.train()

    logger.info("push model")
    trainer.save_model()
    trainer.create_model_card()

def train_insertion(data_name=args.data_name):
    logger.info("loading model")
    inDelphi_model = inDelphiModel.from_pretrained(args.output_dir / inDelphiConfig.model_type / f"{data_name}_{inDelphiConfig.model_type}").to(args.device)

    logger.info("loading data")
    ds = load_dataset(
        path = f"{args.owner}/CRISPR_data",
        name = f"{data_name}_{inDelphiConfig.model_type}",
        split = datasets.Split.TRAIN,
        trust_remote_code = True,
        test_ratio = args.test_ratio,
        validation_ratio = args.validation_ratio,
        seed = args.seed,
        DELLEN_LIMIT = inDelphi_model.DELLEN_LIMIT
    )
    train_dataloader = DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        collate_fn=lambda examples: data_collector(examples, args.DELLEN_LIMIT, outputs_train_insertion)
    )

    logger.info("get mean and std of model inputs, get 1bp insertion distribution in classes based on -6, -5, -4 nucletides")
    with torch.no_grad():
        onebp_features = []
        insert_probabilities = []
        m654 = torch.zeros(4 ** 3, 4)
        for batch in train_dataloader:
            _, _, total_del_len_weights = inDelphi_model(batch["mh_input"].to(args.device), batch["mh_del_len"].to(args.device)).values()
            log_total_weights = total_del_len_weights.sum(dim=1, keepdim=True).log()
            precisions = 1 - torch.distributions.Categorical(total_del_len_weights[:,:28]).entropy() / torch.log(torch.tensor(28))
            onebp_features.extend(
                torch.cat([
                    batch["onebp_feature"],
                    precisions[:, None].cpu(),
                    log_total_weights.cpu()
                ], dim=1).tolist()
            )
            m654.scatter_add_(dim=0, index=batch["m654"][:, None].expand(-1, 4), src=batch["insert_1bp"])
            insert_probabilities.extend(batch["insert_probability"])
        onebp_features = np.array(onebp_features)
        insert_probabilities = np.array(insert_probabilities)
        m654 = m654.cpu().numpy()

    logger.info("save")
    with open(args.output_dir / inDelphiConfig.model_type / f"{data_name}_{inDelphiConfig.model_type}" / "insertion_model.pkl", "wb") as fd:
        pickle.dump(
            [
                onebp_features,
                insert_probabilities,
                m654
            ],
            fd
        )
