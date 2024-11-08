import configargparse
import pathlib
import torch
import logging
import sys

def get_config(config_file=None):
    if config_file:
        parser = configargparse.ArgumentParser(
            description="arguments for CRISPR DL models",
            default_config_files=[config_file]
        )
    else:
        parser.add_argument('--config', required=True, is_config_file=True, help='config file path')
    parser.add_argument("--output_dir", type=pathlib.Path, default="./CRISPR_results", help="output directory")
    parser.add_argument("--seed", type=int, default=63036, help="random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device")
    parser.add_argument("--log", type=str, default="WARNING", choices=['CRITICAL', 'FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'], help="set logging level")

    parser_dataset = parser.add_argument_group(title="dataset", description="parameters for loading and split dataset")
    parser_dataset.add_argument("--owner", type=str, default="ljw20180420", help="huggingface user name")
    parser_dataset.add_argument("--data_name", type=str, default="SX_spcas9", choices=["SX_spcas9", "SX_spymac", "SX_ispymac"])
    parser_dataset.add_argument("--test_ratio", type=float, default=0.05, help="proportion for test samples")
    parser_dataset.add_argument("--validation_ratio", type=float, default=0.05, help="proportion for validation samples")

    parser_dataloader = parser.add_argument_group(title="data loader", description="parameters for data loader")
    parser_dataloader.add_argument("--batch_size", type=int, default=100, help="batch size")

    parser_optimizer = parser.add_argument_group(title="optimizer", description="parameters for optimizer")
    parser_optimizer.add_argument("--optimizer", type=str, default="adamw_torch", choices=["adamw_hf", "adamw_torch", "adamw_torch_fused", "adamw_apex_fused", "adamw_anyprecision", "adafactor"], help="name of optimizer")
    parser_optimizer.add_argument("--learning_rate", type=float, default=0.001, help="learn rate of the optimizer")

    parser_scheduler = parser.add_argument_group(title="scheduler", description="parameters for learning rate scheduler")
    parser_scheduler.add_argument("--scheduler", type=str, default="linear", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "inverse_sqrt", "reduce_lr_on_plateau", "cosine_with_min_lr", "warmup_stable_decay"], help="The scheduler type to use.")
    parser_scheduler.add_argument("--num_epochs", type=float, default=30.0, help="Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training).")
    parser_scheduler.add_argument("--warmup_ratio", type=float, default=0.05, help="Ratio of total training steps used for a linear warmup from 0 to learning_rate")

    parser_CRISPR_diffuser = parser.add_argument_group(title="CRISPR diffuser", description="parameters for CRISPR diffuser")
    parser_CRISPR_diffuser.add_argument("--max_micro_homology", type=int, default=7, help="Clip micro-homology strength to (0, max_micro_homology).")
    parser_CRISPR_diffuser.add_argument("--MCMC_corrector_factor", nargs='+', type=float, default=[1., 0., 0.001], help="weight of the MCMC corrector term")
    parser_CRISPR_diffuser.add_argument("--unet_channels", nargs='+', type=int, default=[32, 64, 96, 64, 32], help="the output channels of Unet")
    parser_CRISPR_diffuser.add_argument("--noise_scheduler", type=str, default="exp", choices=["linear", "cosine", "exp", "uniform"], help="noise scheduler used for diffuser model")
    parser_CRISPR_diffuser.add_argument("--noise_timesteps", type=int, default=20, help="number of noise scheduler time steps")
    parser_CRISPR_diffuser.add_argument("--cosine_factor", type=float, default=0.008, help="parameter control cosine noise scheduler")
    parser_CRISPR_diffuser.add_argument("--exp_scale", type=float, default=5.0, help="scale factor of exponential noise scheduler")
    parser_CRISPR_diffuser.add_argument("--exp_base", type=float, default=5.0, help="base parameter of exponential noise scheduler")
    parser_CRISPR_diffuser.add_argument("--uniform_scale", type=float, default=1.0, help="scale parameter for uniform scheduler")
    parser_CRISPR_diffuser.add_argument("--display_scale_factor", type=float, default=0.1, help="exponential scale of the distribution image")

    parser_inDelphi = parser.add_argument_group(title="inDelphi", description="parameters for inDelphi")
    parser_inDelphi.add_argument("--DELLEN_LIMIT", type=int, default=60, help="deletion length upper limit of inDelphi model")

    parser_Lindel = parser.add_argument_group(title="Lindel", description="parameters for Lindel")
    parser_Lindel.add_argument("--Lindel_dlen", type=int, default=30, help="the upper limit of deletion length (strictly less than dlen)")
    parser_Lindel.add_argument("--Lindel_mh_len", type=int, default=4, help="the upper limit of micro-homology length")
    parser_Lindel.add_argument("--Lindel_reg_const", type=float, default=0.01, help="regularization coefficient")
    parser_Lindel.add_argument("--Lindel_reg_mode", type=str, default="l2", choices=["l2", "l1"], help="regularization method")

    parser_FOREcasT = parser.add_argument_group(title="FOREcasT", description="parameters for FOREcasT")
    parser_FOREcasT.add_argument("--FOREcasT_MAX_DEL_SIZE", type=int, default=30, help="max deletion size")
    parser_FOREcasT.add_argument("--FOREcasT_reg_const", type=float, default=0.01, help="regularization coefficient for deletion")
    parser_FOREcasT.add_argument("--FOREcasT_i1_reg_const", type=float, default=0.01, help="regularization coefficient for insertion")

    parser_inference = parser.add_argument_group(title="inference", description="parameters for inference")
    parser_inference.add_argument("--ref1len", type=int, default=127, help="length of reference 1")
    parser_inference.add_argument("--ref2len", type=int, default=127, help="length of reference 2")

    return parser.parse_args()

def get_logger(args):
    logger = logging.getLogger("logger")
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(args.log)
    logger.addHandler(handler)
    return logger
