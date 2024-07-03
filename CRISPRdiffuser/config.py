import argparse
import pathlib
import torch
from logging import getLogger
import os
import json

def list_huggingface_diffuser_models():
    return ['UNet2DConditionModel']

def list_all_diffuser_models():
    return list_huggingface_diffuser_models()

def list_huggingface_diffuser_schedulers():
    return ['DPMSolverMultistepSchedule']

def list_all_diffuser_schedulers():
    return list_huggingface_diffuser_schedulers()

def list_huggingface_learn_schedulers():
    return ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "piecewise_constant"]

def list_all_learn_schedulers():
    return list_huggingface_learn_schedulers()

def list_torch_optimizers():
    return ["AdamW"]
    # return ["Adadelta", "Adagrad", "Adam", "AdamW", "SparseAdam", "Adamax", "ASGD", "SGD", "RAdam", "Rprop", "RMSprop", "NAdam", "LBFGS"]

def list_all_optimizers():
    return list_torch_optimizers()

def get_diagonal_indices():
    # given offset in range(-reflen, reflen + 1)
    # row_idx_start = max(-offset, 0)
    # row_idx_end = min(reflen-offset, reflen)
    # col_idx_start = max(offset, 0)
    # col_idx_end = min(offset+reflen, reflen)
    # OUTPUT: (
    #   tensor((reflen+1)*(reflen+1),),
    #   tensor((reflen+1)*(reflen+1),)
    # )
    global reflen
    return (
        torch.cat([torch.arange(max(-offset, 0), min(reflen-offset, reflen)+1) for offset in range(-reflen, reflen+1)]),
        torch.cat([torch.arange(max(offset, 0), min(offset+reflen, reflen)+1) for offset in range(-reflen, reflen+1)])
    )

logger = getLogger(__name__)

parser = argparse.ArgumentParser(description="use cascaded diffuser (https://doi.org/10.48550/arXiv.2106.15282) to predict CRIPSR/Cas9 editing")

parser_dataset = parser.add_argument_group(title="dataset", description="parameters for loading and split dataset")
parser_dataset.add_argument("--data_file", type=pathlib.Path, default=pathlib.Path(os.environ["HOME"]) / "sdc1" / "SX" / "dataset.json", help="file after preprocess in json format")
parser_dataset.add_argument("--kmer_size", type=int, default=1, help="size of kmer")
parser_dataset.add_argument("--alphabet", type=str, default="ACGTN", help="alphabet of sequence")
parser_dataset.add_argument("--lr_warmup_steps", type=int, default=500, help="warm up steps for learning scheduler")
parser_dataset.add_argument("--seed", type=int, default=63036, help="random seed")
parser_dataset.add_argument("--test_valid_ratio", type=float, default=0.1, help="proportion for test and valid samples")

parser_diffuser_model = parser.add_argument_group(title="diffuser model", description="parameters for diffuser model")
parser_diffuser_model.add_argument("--diffuser_model", type=str, default="UNet2DConditionModel", choices=list_all_diffuser_models(), help="name of diffuser model")

parser_diffuser_scheduler = parser.add_argument_group(title="diffuser scheduler", description="parameters for diffuser scheduler")
parser_diffuser_scheduler.add_argument("--diffuser_scheduler", type=str, default="DPMSolverMultistepSchedule", choices=list_huggingface_diffuser_schedulers(), help="scheduler used for diffuser model")
parser_diffuser_scheduler.add_argument("--num_train_timesteps", type=int, default=1000, help="number of diffuser scheduler time steps")

parser_learn_scheduler = parser.add_argument_group(title="learn_scheduler", description="parameters for learn scheduler")
parser_learn_scheduler.add_argument("--learn_scheduler", type=str, default="cosine", choices=list_all_learn_schedulers(), help="name of learn scheduler")
parser_learn_scheduler.add_argument("--step_rules", type=str, default="1:10,0.1:20,0.01:30,0.005", help='The rules for the learning rate. Only used for piecewise_constant scheduler, ex: rule_steps="1:10,0.1:20,0.01:30,0.005" it means that the learning rate multiple 1 for the first 10 steps, mutiple 0.1 for the next 20 steps, multiple 0.01 for the next 30 steps and multiple 0.005 for the other steps.')
parser_learn_scheduler.add_argument("--num_warmup_steps", type=int, default=500, help="The number of warmup steps to do. This is not required by all schedulers")
parser_learn_scheduler.add_argument("--num_epochs", type=int, default=5, help="number of epochs")
parser_learn_scheduler.add_argument("--num_cycles", type=int, default=1, help="The number of hard restarts used in cosine_with_restarts scheduler")
parser_learn_scheduler.add_argument("--power", type=int, default=1, help="Power factor used in polynomial scheduler")

parser_optimizer = parser.add_argument_group(title="optimizer", description="parameters for optimizer")
parser_optimizer.add_argument("--optimizer", type=str, default="AdamW", choices=list_all_optimizers(), help="name of optimizer")
parser_optimizer.add_argument("--learning_rate", type=float, default=0.001, help="learn rate of the optimizer")
parser_optimizer.add_argument("--adam_betas", type=lambda x: tuple(map(float, x.split(","))), default=(0.9, 0.999), help="coefficients used for computing running averages of gradient and its square for adam optimizer")
parser_optimizer.add_argument("--adam_eps", type=float, default=1e-8, help="term added to the denominator to improve numerical stability of adam optimizer")
parser_optimizer.add_argument("--adam_weight_decay", type=float, default=1e-2, help="weight decay coefficient of adam optimizer")
parser_optimizer.add_argument("--adam_amsgrad", type=bool, default=False, help="whether to use the AMSGrad variant of adam (https://openreview.net/forum?id=ryQu7f-RZ)")

parser_accelerator = parser.add_argument_group(title="accelerator", description="parameters for acceleraor")
parser_accelerator.add_argument("--mixed_precision", type=str, default="fp16", choices=["fp16", "no"], help="use mixed precision (fp16) or not (no)")
parser_accelerator.add_argument("--gradient_accumulation_steps", type=int, default=1, help="The number of steps that should pass before gradients are accumulated. A number > 1 should be combined with Accelerator.accumulate. If not passed, will default to the value in the environment variable ACCELERATE_GRADIENT_ACCUMULATION_STEPS. Can also be configured through a GradientAccumulationPlugin.")
parser_accelerator.add_argument("--max_norm", type=float, default=1.0, help="Max norm of gradients. Prevent gradient from exploding.")

parser_logging = parser.add_argument_group(title="logging", description="Parameters for logging.")
parser_logging.add_argument("--push_to_hub", type=bool, default=False, help="Push output to huggingface hub.")
parser_logging.add_argument("--save_image_epochs", type=int, default=1000, help="Generate sample images every this epoch.")
parser_logging.add_argument("--save_model_epochs", type=int, default=1000, help="Save model every this epoch.")

args = parser.parse_args()


alphacode = torch.frombuffer(args.alphabet.encode(), dtype=torch.uint8)
json_line1 = json.loads(args.data_file.open("r").readline())
reflen = len(json_line1['ref1'])
try:
    assert reflen == len(json_line1['ref2']), "reference 1 and reference 2 have different lengths"
except Exception as err:
    logger.exception(str(err))
    raise
diag_indices = get_diagonal_indices()