import argparse
from logging import getLogger
import pathlib
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = getLogger("Lindel")

parser = argparse.ArgumentParser(description="arguments for Lindel https://doi.org/10.1093/nar/gkz487")
parser.add_argument("--model_name", type=str, default="Lindel", help="the name of model")

parser_dataset = parser.add_argument_group(title="dataset", description="parameters for loading and split dataset")
parser_dataset.add_argument("--data_file", type=pathlib.Path, default=pathlib.Path(os.environ['HOME']) / "sdc1/SX/spcas9/spcas9.json", help="data file in json format")
parser_dataset.add_argument("--seed", type=int, default=63036, help="random seed")
parser_dataset.add_argument("--test_valid_ratio", type=float, default=0.05, help="proportion for test and valid samples")
parser_dataset.add_argument("--batch_size", type=int, default=100, help="batch size")
parser_dataset.add_argument("--not_correct_micro_homology", action=argparse.BooleanOptionalAction, help="do not divide count by length for positions in micro-homology")

parser_train = parser.add_argument_group(title="train", description="parameters for training")
parser_train.add_argument("--epoch_num", type=int, default=100, help="number of epochs")
parser_train.add_argument("--reg_const", type=float, default=0.01, help="regularization coefficient")
parser_train.add_argument("--reg_mode", type=str, default="l2", choices=["l2", "l1"], help="regularization method")
parser_train.add_argument("--target_model", type=str, default="del", choices=["indel", 'ins', "del"], help="the target model to train")

parser_optimizer = parser.add_argument_group(title="optimizer", description="parameters for optimizer")
# choices=["Adadelta", "Adagrad", "Adam", "AdamW", "SparseAdam", "Adamax", "ASGD", "SGD", "RAdam", "Rprop", "RMSprop", "NAdam", "LBFGS"]
parser_optimizer.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW"], help="name of optimizer")
parser_optimizer.add_argument("--learning_rate", type=float, default=1e-3, help="learn rate of the optimizer")
parser_optimizer.add_argument("--adam_betas", nargs=2, type=float, default=[0.9, 0.999], help="coefficients used for computing running averages of gradient and its square for adam optimizer")
parser_optimizer.add_argument("--adam_eps", type=float, default=1e-8, help="term added to the denominator to improve numerical stability of adam optimizer")
parser_optimizer.add_argument("--adam_weight_decay", type=float, default=1e-2, help="weight decay coefficient of adam optimizer")
parser_optimizer.add_argument("--adam_amsgrad", action=argparse.BooleanOptionalAction, help="whether to use the AMSGrad variant of adam (https://openreview.net/forum?id=ryQu7f-RZ)")

parser_learn_scheduler = parser.add_argument_group(title="learn_scheduler", description="parameters for learn scheduler")
parser_learn_scheduler.add_argument("--learn_scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "piecewise_constant"], help="name of learn scheduler")
parser_learn_scheduler.add_argument("--step_rules", type=str, default="1:10,0.1:20,0.01:30,0.005", help='The rules for the learning rate. Only used for piecewise_constant scheduler, ex: rule_steps="1:10,0.1:20,0.01:30,0.005" it means that the learning rate multiple 1 for the first 10 steps, mutiple 0.1 for the next 20 steps, multiple 0.01 for the next 30 steps and multiple 0.005 for the other steps.')
parser_learn_scheduler.add_argument("--num_warmup_steps", type=int, default=30, help="The number of warmup steps to do. This is not required by all schedulers")
parser_learn_scheduler.add_argument("--num_cycles", type=int, default=1, help="The number of hard restarts used in cosine_with_restarts scheduler")
parser_learn_scheduler.add_argument("--power", type=int, default=1, help="Power factor used in polynomial scheduler")

parser_accelerator = parser.add_argument_group(title="accelerator", description="accelerator settings")
parser_accelerator.add_argument("--mixed_precision", type=str, default="no", choices=["fp16", "no"], help="use mixed precision (fp16) or not (no)")
parser_accelerator.add_argument("--gradient_accumulation_steps", type=int, default=1, help="The number of steps that should pass before gradients are accumulated. A number > 1 should be combined with Accelerator.accumulate. If not passed, will default to the value in the environment variable ACCELERATE_GRADIENT_ACCUMULATION_STEPS. Can also be configured through a GradientAccumulationPlugin.")
parser_accelerator.add_argument("--max_norm", type=float, default=1, help="Max norm of gradients. Prevent gradient from exploding.")

parser_save = parser.add_argument_group(title="save", description="Parameters for saving.")
parser_save.add_argument("--push_to_hub", action=argparse.BooleanOptionalAction, help="Push output to huggingface hub.")
parser_save.add_argument("--save_model_epochs", type=int, default=1, help="Save model every this epoch.")

args = parser.parse_args()