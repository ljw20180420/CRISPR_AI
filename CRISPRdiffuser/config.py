import argparse
import pathlib
import torch
from logging import getLogger
import os
import json

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
    # given offset in range(-ref2len, ref1len + 1)
    # row_idx_start = max(-offset, 0)
    # row_idx_end = min(ref1len-offset, ref2len)
    # col_idx_start = max(offset, 0)
    # col_idx_end = min(offset+ref2len, ref1len)
    # OUTPUT: (
    #   tensor((ref2len+1)*(ref1len+1),),
    #   tensor((ref2len+1)*(ref1len+1),)
    # )
    return (
        torch.cat([torch.arange(max(-offset, 0), min(ref1len-offset, ref2len)+1) for offset in range(-ref2len, ref1len+1)]),
        torch.cat([torch.arange(max(offset, 0), min(offset+ref2len, ref1len)+1) for offset in range(-ref2len, ref1len+1)])
    )

logger = getLogger(__name__)

parser = argparse.ArgumentParser(description="use cascaded diffuser (https://doi.org/10.48550/arXiv.2106.15282) to predict CRIPSR/Cas9 editing")

parser_dataset = parser.add_argument_group(title="dataset", description="parameters for loading and split dataset")
parser_dataset.add_argument("--data_file", type=pathlib.Path, default=pathlib.Path(os.environ["HOME"]) / "sdc1" / "SX" / "dataset.json", help="file after preprocess in json format")
parser_dataset.add_argument("--kmer_size", type=int, default=1, help="size of kmer")
parser_dataset.add_argument("--alphabet", type=str, default="ACGTN", help="alphabet of sequence")
parser_dataset.add_argument("--seed", type=int, default=63036, help="random seed")
parser_dataset.add_argument("--test_valid_ratio", type=float, default=0.01, help="proportion for test and valid samples")
parser_dataset.add_argument("--batch_size", type=int, default=80, help="batch size for DataLoader")
parser_dataset.add_argument("--cross_reference", type=bool, default=False, help="use kmer from both reference to compose cross factors")
parser_dataset.add_argument("--max_micro_homology", type=int, default=7, help="Clip micro-homology strength to (0, max_micro_homology).")
parser_dataset.add_argument("--importance_sampling_factor", type=float, default=1., help="Factor for importance sampling from data.")

parser_train = parser.add_argument_group(title="train", description="parameters for training")
parser_train.add_argument("--epoch_num", type=int, default=50, help="number of epochs")
parser_train.add_argument("--batch_num", type=int, default=1e100, help="upbound for the number of batch in each epoch, mainly used in test")
parser_train.add_argument("--MCMC_corrector_factor", type=float, default=0.001, help="weight of the MCMC corrector term")

parser_model = parser.add_argument_group(title="model", description="parameters for model")
parser_model.add_argument("--forward_stationary_distribution", type=str, default="uniform", choices=["uniform"], help="stationary distribution of the forward diffusion process")
parser_model.add_argument("--unet_channels", nargs='+', type=int, default=[32, 64, 96, 64, 32], help="the output channels of Unet")

parser_noise_scheduler = parser.add_argument_group(title="noise scheduler", description="parameters for noise scheduler")
parser_noise_scheduler.add_argument("--noise_scheduler", type=str, default="exp", choices=["linear", "cosine", "exp"], help="noise scheduler used for diffuser model")
parser_noise_scheduler.add_argument("--noise_timesteps", type=int, default=20, help="number of noise scheduler time steps")
parser_noise_scheduler.add_argument("--cosine_factor", type=float, default=0.008, help="parameter control cosine noise scheduler")
parser_noise_scheduler.add_argument("--exp_scale", type=float, default=5., help="scale factor of exponential noise scheduler")
parser_noise_scheduler.add_argument("--exp_base", type=float, default=5., help="base parameter of exponential noise scheduler")

parser_learn_scheduler = parser.add_argument_group(title="learn_scheduler", description="parameters for learn scheduler")
parser_learn_scheduler.add_argument("--learn_scheduler", type=str, default="cosine", choices=list_all_learn_schedulers(), help="name of learn scheduler")
parser_learn_scheduler.add_argument("--step_rules", type=str, default="1:10,0.1:20,0.01:30,0.005", help='The rules for the learning rate. Only used for piecewise_constant scheduler, ex: rule_steps="1:10,0.1:20,0.01:30,0.005" it means that the learning rate multiple 1 for the first 10 steps, mutiple 0.1 for the next 20 steps, multiple 0.01 for the next 30 steps and multiple 0.005 for the other steps.')
parser_learn_scheduler.add_argument("--num_warmup_steps", type=int, default=100, help="The number of warmup steps to do. This is not required by all schedulers")
parser_learn_scheduler.add_argument("--num_cycles", type=int, default=1, help="The number of hard restarts used in cosine_with_restarts scheduler")
parser_learn_scheduler.add_argument("--power", type=int, default=1, help="Power factor used in polynomial scheduler")

parser_optimizer = parser.add_argument_group(title="optimizer", description="parameters for optimizer")
parser_optimizer.add_argument("--optimizer", type=str, default="AdamW", choices=list_all_optimizers(), help="name of optimizer")
parser_optimizer.add_argument("--learning_rate", type=float, default=1e-4, help="learn rate of the optimizer")
parser_optimizer.add_argument("--adam_betas", nargs=2, type=float, default=[0.9, 0.999], help="coefficients used for computing running averages of gradient and its square for adam optimizer")
parser_optimizer.add_argument("--adam_eps", type=float, default=1e-8, help="term added to the denominator to improve numerical stability of adam optimizer")
parser_optimizer.add_argument("--adam_weight_decay", type=float, default=1e-2, help="weight decay coefficient of adam optimizer")
parser_optimizer.add_argument("--adam_amsgrad", type=bool, default=False, help="whether to use the AMSGrad variant of adam (https://openreview.net/forum?id=ryQu7f-RZ)")

parser_accelerator = parser.add_argument_group(title="accelerator", description="parameters for acceleraor")
parser_accelerator.add_argument("--mixed_precision", type=str, default="fp16", choices=["fp16", "no"], help="use mixed precision (fp16) or not (no)")
parser_accelerator.add_argument("--gradient_accumulation_steps", type=int, default=1, help="The number of steps that should pass before gradients are accumulated. A number > 1 should be combined with Accelerator.accumulate. If not passed, will default to the value in the environment variable ACCELERATE_GRADIENT_ACCUMULATION_STEPS. Can also be configured through a GradientAccumulationPlugin.")
parser_accelerator.add_argument("--max_norm", type=float, default=1, help="Max norm of gradients. Prevent gradient from exploding.")

parser_save = parser.add_argument_group(title="save", description="Parameters for saving.")
parser_save.add_argument("--push_to_hub", type=bool, default=False, help="Push output to huggingface hub.")
parser_save.add_argument("--valid_epoch_num", type=int, default=10, help="Epoch number to estimate q0 in validation")
parser_save.add_argument("--valid_batch_num", type=int, default=1e100, help="upbound for the number of batch in each validation epoch, mainly used in test")
parser_save.add_argument("--valid_batch_size", type=int, default=80, help="batch size for validation")
parser_save.add_argument("--save_model_epochs", type=int, default=1, help="Save model every this epoch.")
parser_save.add_argument("--save_image_epochs", type=int, default=1, help="Save image every this epoch.")
parser_save.add_argument("--save_image_valid_epochs", type=int, default=1, help="Save image every this validation epoch.")
parser_save.add_argument("--save_image_valid_batchs", type=int, default=1, help="Save image every this validation batchs.")
parser_save.add_argument("--save_image_valid_in_batchs", type=int, default=80, help="Save image every this record in a validation batch.")
parser_save.add_argument("--display_scale_facter", type=float, default=0.1, help="scale factor for probability display")

args = parser.parse_args()

alphacode = torch.frombuffer(args.alphabet.encode(), dtype=torch.uint8)
json_line1 = json.loads(args.data_file.open("r").readline())
ref1len, ref2len = len(json_line1['ref1']), len(json_line1['ref2'])
diag_indices = get_diagonal_indices()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
