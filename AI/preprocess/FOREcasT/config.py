import configargparse
import pathlib
import logging
import sys


def get_config(config_files):
    """
    config_files: Files contain hyper-parameters. The later config files will override the former ones.
    For example, if  config_files=['config_default.ini', 'config_custom.ini'], then settings in config_custom.ini will override settings in config_default.ini. A good practice is to put default settings in config_default.ini (do not modify config_default.ini), and then override default behaviors in config_custom.ini.
    """
    parser = configargparse.ArgumentParser(
        description="arguments of FOREcasT",
        default_config_files=config_files,
        auto_env_var_prefix="CRISPR_AI_",
        config_file_parser_class=configargparse.ConfigparserConfigFileParser,
    )

    # command paramters
    parser_command = parser.add_argument_group(
        title="command",
        description="Command parameters.",
    )
    parser_command.add_argument(
        "--command",
        type=str,
        required=True,
        choices=["train", "test", "upload", "inference", "app", "space"],
        help="What to do. The order is: train -> test -> upload -> inference -> app -> space.",
    )
    parser_command.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["FOREcasT"],
        help="The name of the model to train.",
    )

    # common parameters
    parser_common = parser.add_argument_group(
        title="common",
        description="Common parameters.",
    )
    parser_common.add_argument(
        "--output_dir",
        type=pathlib.Path,
        required=True,
        help="Output directory.",
    )
    parser_common.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed.",
    )
    parser_common.add_argument(
        "--device",
        type=str,
        required=True,
        choices=["cuda", "cpu"],
        help="Device.",
    )
    parser_common.add_argument(
        "--log_level",
        type=str,
        required=True,
        choices=["CRITICAL", "FATAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
        help="Logging level.",
    )

    # dataset parameters
    parser_dataset = parser.add_argument_group(
        title="dataset",
        description="Parameters of dataset.",
    )
    parser_dataset.add_argument(
        "--data_name",
        type=str,
        required=True,
        choices=["SX_spcas9", "SX_spymac", "SX_ispymac"],
        help="Data name for training. Generally correpond to Cas protein name.",
    )
    parser_dataset.add_argument(
        "--test_ratio",
        type=float,
        required=True,
        help="Proportion for test samples.",
    )
    parser_dataset.add_argument(
        "--validation_ratio",
        type=float,
        required=True,
        help="Proportion for validation samples.",
    )
    parser_dataset.add_argument(
        "--ref1len",
        type=int,
        required=True,
        help="Length of reference 1.",
    )
    parser_dataset.add_argument(
        "--ref2len",
        type=int,
        required=True,
        help="Length of reference 2.",
    )
    parser_dataset.add_argument(
        "--random_insert_uplimit",
        type=int,
        required=True,
        help="The maximal discriminated length of random insertion.",
    )
    parser_dataset.add_argument(
        "--insert_uplimit",
        type=int,
        required=True,
        help="The maximal insertion length to count.",
    )
    parser_dataset.add_argument(
        "--owner",
        type=str,
        required=True,
        help="Huggingface user name.",
    )
    parser_dataset.add_argument(
        "--inference_data",
        type=pathlib.Path,
        required=True,
        help="The data file of inference.",
    )
    parser_dataset.add_argument(
        "--inference_output",
        type=pathlib.Path,
        required=True,
        help="The output file of inference.",
    )

    parser_dataloader = parser.add_argument_group(
        title="data loader",
        description="Parameters of data loader",
    )
    parser_dataloader.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size.",
    )

    parser_optimizer = parser.add_argument_group(
        title="optimizer",
        description="Parameters of optimizer.",
    )
    parser_optimizer.add_argument(
        "--optimizer",
        type=str,
        required=True,
        choices=[
            "adamw_torch",
            "adamw_torch_fused",
            "adafactor",
        ],
        help="Name of optimizer.",
    )
    parser_optimizer.add_argument(
        "--learning_rate",
        type=float,
        required=True,
        help="Learn rate of the optimizer.",
    )

    parser_scheduler = parser.add_argument_group(
        title="scheduler",
        description="Parameters for learning rate scheduler",
    )
    parser_scheduler.add_argument(
        "--scheduler",
        type=str,
        required=True,
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
            "reduce_lr_on_plateau",
            "cosine_with_min_lr",
            "warmup_stable_decay",
        ],
        help="The scheduler type to use.",
    )
    parser_scheduler.add_argument(
        "--num_epochs",
        type=float,
        required=True,
        help="Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training).",
    )
    parser_scheduler.add_argument(
        "--warmup_ratio",
        type=float,
        required=True,
        help="Ratio of total training steps used for a linear warmup from 0 to learning_rate.",
    )

    parser_FOREcasT = parser.add_argument_group(
        title="FOREcasT",
        description="Parameters for FOREcasT.",
    )
    parser_FOREcasT.add_argument(
        "--max_del_size",
        type=int,
        required=True,
        help="Maximal deletion size.",
    )
    parser_FOREcasT.add_argument(
        "--reg_const",
        type=float,
        required=True,
        help="Regularization coefficient for deletion.",
    )
    parser_FOREcasT.add_argument(
        "--i1_reg_const",
        type=float,
        required=True,
        help="Regularization coefficient for insertion.",
    )

    return parser.parse_args()


def get_logger(log_level):
    logger = logging.getLogger("logger")
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(log_level)
    logger.addHandler(handler)
    return logger
