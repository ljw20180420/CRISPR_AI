import argparse
import pathlib
import os
import torch
import logging
import sys

parser = argparse.ArgumentParser(description="arguments for CRISPR DL models")
parser.add_argument("--output_dir", type=pathlib.Path, default=f'''{os.environ["HOME"]}/sdc1/CRISPR_results''', help="output directory")
parser.add_argument("--seed", type=int, default=63036, help="random seed")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="random seed")
parser.add_argument("--log", type=str, default="WARNING", choices=['CRITICAL', 'FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'], help="set logging level")

parser_dataset = parser.add_argument_group(title="dataset", description="parameters for loading and split dataset")
parser_dataset.add_argument("--data_path", type=str, default="ljw20180420/CRISPR_data", help="data path")
parser_dataset.add_argument("--data_name", type=str, default="SX_spcas9", choices=["SX_spcas9", "SX_spymac", "SX_ispymac"], help="data name")
parser_dataset.add_argument("--test_ratio", type=float, default=0.05, help="proportion for test samples")
parser_dataset.add_argument("--validation_ratio", type=float, default=0.05, help="proportion for validation samples")

parser_dataloader = parser.add_argument_group(title="data loader", description="parameters for data loader")
parser_dataloader.add_argument("--batch_size", type=int, default=100, help="batch size")

parser_optimizer = parser.add_argument_group(title="optimizer", description="parameters for optimizer")
parser_optimizer.add_argument("--optimizer", type=str, default="adamw_torch", choices=["adamw_hf", "adamw_torch", "adamw_torch_fused", "adamw_apex_fused", "adamw_anyprecision", "adafactor"], help="name of optimizer")
parser_optimizer.add_argument("--learning_rate", type=float, default=1e-3, help="learn rate of the optimizer")

parser_scheduler = parser.add_argument_group(title="scheduler", description="parameters for learning rate scheduler")
parser_scheduler.add_argument("--scheduler", type=str, default="linear", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "inverse_sqrt", "reduce_lr_on_plateau", "cosine_with_min_lr", "warmup_stable_decay"], help="The scheduler type to use.")
parser_scheduler.add_argument("--num_epochs", type=float, default=30.0, help="Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training).")
parser_scheduler.add_argument("--warmup_ratio", type=float, default=0.05, help="Ratio of total training steps used for a linear warmup from 0 to learning_rate")

args = parser.parse_args()

logger = logging.getLogger("logger")
logger.addHandler(logging.StreamHandler(stream=sys.stdout).setLevel(args.log))
