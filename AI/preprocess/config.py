import jsonargparse
import pathlib
import logging
import sys
import importlib
from dataclasses import dataclass
from typing import Literal
import importlib
import inspect


@dataclass
class Common:
    """Common parameters.

    Args:
        output_dir: Output directory.
        batch_size: Batch size.
        seed: Random seed.
        device: Device.
        log_level: Logging level.
        inference_data: The data file of inference.
        inference_output: The output file of inference.
    """

    output_dir: pathlib.Path
    batch_size: int
    seed: int
    device: Literal["cpu", "cuda"]
    log_level: Literal[
        "CRITICAL", "FATAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"
    ]
    inference_data: pathlib.Path
    inference_output: pathlib.Path


@dataclass
class Dataset:
    """Parameters of dataset.

    Args:
        data_name: Data name for training. Generally correpond to Cas protein name.
        test_ratio: Proportion for test samples.
        validation_ratio: Proportion for validation samples.
        random_insert_uplimit: The maximal discriminated length of random insertion.
        insert_uplimit: The maximal insertion length to count.
        owner: huggingface user name.
    """

    data_name: Literal["SX_spcas9", "SX_spymac", "SX_ispymac"]
    test_ratio: float
    validation_ratio: float
    random_insert_uplimit: int
    insert_uplimit: int
    owner: str


@dataclass
class Optimizer:
    """Parameters of optimizer.

    Args:
        optimizer: Name of optimizer.
        learning_rate: Learn rate of the optimizer.
    """

    optimizer: Literal["adamw_torch", "adamw_torch", "adamw_torch_fused", "adafactor"]
    learning_rate: float


@dataclass
class Scheduler:
    """Parameters for learning rate scheduler.

    Args:
        scheduler: The scheduler type to use.
        num_epochs: Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training).
        warmup_ratio: Ratio of total training steps used for a linear warmup from 0 to learning_rate.
    """

    scheduler: Literal[
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
    ]
    num_epochs: float
    warmup_ratio: float


preprocess_to_model = {
    "inDelphi": ["inDelphi"],
    "FOREcasT": ["FOREcasT"],
    "Lindel": ["Lindel"],
    "DeepHF": ["DeepHF"],
    "CRIformer": ["CRIformer"],
    "CRIfuser": ["CRIfuser"],
}


def get_config() -> jsonargparse.Namespace:
    parser = jsonargparse.ArgumentParser(
        description="Arguments of AI models.",
        default_config_files=["preprocess/config.yaml"],
        env_prefix="CRISPR_AI",
        # default_env=True,
    )

    parser.add_argument("--config", action="config")

    # Add commands. The run order is: train -> test -> upload -> inference -> app -> space
    parser_command = parser.add_mutually_exclusive_group(
        required=True,
    )
    parser_command.add_argument(
        "--train",
        action="store_true",
        help="Train the model.",
    )
    parser_command.add_argument(
        "--test",
        action="store_true",
        help="Test the model.",
    )
    parser_command.add_argument(
        "--upload",
        action="store_true",
        help="Upload the model.",
    )
    parser_command.add_argument(
        "--inference",
        action="store_true",
        help="Inference by the model.",
    )
    parser_command.add_argument(
        "--app",
        action="store_true",
        help="Start model app.",
    )
    parser_command.add_argument(
        "--space",
        action="store_true",
        help="Deploy the model app as huggingface space.",
    )

    # Add global arguments.
    parser.add_class_arguments(theclass=Common)
    parser.add_argument("--dataset", type=Dataset)
    parser.add_argument("--optimizer", type=Optimizer)
    parser.add_argument("--scheduler", type=Scheduler)

    parser_subcommands = parser.add_subcommands(required=False, dest="preprocess")
    for preprocess, model_names in preprocess_to_model.items():
        preprocess_command = jsonargparse.ArgumentParser(
            description=f"preprocess {preprocess}.",
        )
        parser_subcommands.add_subcommand(preprocess, preprocess_command)
        preprocess_subcommands = preprocess_command.add_subcommands(
            required=False, dest="model_name"
        )
        for model_name in model_names:
            model_command = jsonargparse.ArgumentParser(
                description=f"model {model_name} of preprocess {preprocess}.",
                default_config_files=[
                    f"preprocess/{preprocess}/configs/{model_name}.yaml"
                ],
            )
            preprocess_subcommands.add_subcommand(model_name, model_command)
            model_command.add_argument("--config", action="config")
            # Construct dynamic function from model config class
            theclass = getattr(
                importlib.import_module(f"preprocess.{preprocess}.model"),
                f"{model_name}Config",
            )

            signature = inspect.Signature(
                parameters=[
                    sig_value.replace(
                        annotation=sig_value.annotation.__reduce__()[1][1][0],
                        default=inspect._empty,
                    )
                    for sig_key, sig_value in inspect.signature(
                        theclass.__init__
                    ).parameters.items()
                    if sig_key not in ["self", "args", "kwargs"]
                ]
            )

            def dynamic_func(*args, **kwargs):
                pass

            dynamic_func.__signature__ = signature
            dynamic_func.__doc__ = theclass.__init__.__doc__
            model_command.add_function_arguments(
                function=dynamic_func,
                skip={"seed"},
            )

    return parser.parse_args()


def get_logger(log_level):
    logger = logging.getLogger("logger")
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(log_level)
    logger.addHandler(handler)
    return logger
